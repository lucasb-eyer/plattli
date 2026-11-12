import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
from plattli._indices import _segments_to_array
from plattli._ranges import _ranges_encode
from plattli.optimize_indices import optimize_indices


def _create(root, zipped=False, hot=False):
    storage = root / "plattli"
    storage.mkdir()
    segments = [{"start": 20 * i, "stop": 20 * i + 10, "step": 1} for i in range(33)]
    sparse = [{"start": 20 * i, "stop": 20 * i + 2, "step": 1} for i in range(33)]
    manifest = {
        "nested/loss": {"indices": segments, "dtype": "f32", "monotonic": "inc"},
        "dense": {"indices": "indices", "dtype": "f32"},
        "sparse": {"indices": sparse, "dtype": "f32"},
        "coalesce": {"indices": "ranges", "dtype": "f32"},
        "run_rows": 330,
        "when_exported": "2026-11-10T01:02:03Z",
    }
    (storage / "nested").mkdir()
    expected = {
        "nested/loss": _segments_to_array(segments),
        "dense": np.arange(8, dtype=np.uint32),
        "sparse": _segments_to_array(sparse),
        "coalesce": np.arange(8, dtype=np.uint32),
    }
    for name, steps in expected.items():
        steps.astype(np.float32).tofile(storage / f"{name}.f32")
    expected["dense"].tofile(storage / "dense.indices")
    (storage / "coalesce.ranges").write_bytes(_ranges_encode([
        {"start": 0, "stop": 4, "step": 1}, {"start": 4, "stop": 8, "step": 1},
    ]))
    (storage / "plattli.json").write_text(json.dumps(manifest), encoding="utf-8")
    (storage / "config.json").write_text('{"config": true}', encoding="utf-8")
    (storage / "notes.txt").write_bytes(b"preserve ancillary bytes")
    if hot:
        (storage / "hot.jsonl").write_bytes(b'{"step": 1000, "dense": 1000}\n')
    if zipped:
        with zipfile.ZipFile(root / "metrics.plattli", "w", compression=zipfile.ZIP_STORED) as archive:
            for path in storage.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(storage))
        shutil.rmtree(storage)
    return expected, manifest


def _contents(root, zipped):
    if zipped:
        with zipfile.ZipFile(root / "metrics.plattli") as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    return {path.relative_to(root / "plattli").as_posix(): path.read_bytes()
            for path in (root / "plattli").rglob("*") if path.is_file()}


class TestOptimizeIndices(unittest.TestCase):
    def test_live_sequence_heads_require_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            with plattli.DirectWriter(tmp, write_threads=0) as writer:
                writer.write(value=7)
            before = {path.name: path.read_bytes() for path in (Path(tmp) / "plattli").iterdir() if path.is_file()}
            with self.assertRaisesRegex(RuntimeError, "live value sequences"):
                optimize_indices(tmp)
            after = {path.name: path.read_bytes() for path in (Path(tmp) / "plattli").iterdir() if path.is_file()}
            self.assertEqual(before, after)

    def test_directory_and_archive_preserve_values_and_metadata(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                expected, original = _create(root, zipped)
                before = _contents(root, zipped)
                self.assertEqual(optimize_indices(root), 4)
                after = _contents(root, zipped)
                manifest = json.loads(after["plattli.json"])
                self.assertEqual(manifest["run_rows"], original["run_rows"])
                self.assertEqual(manifest["when_exported"], original["when_exported"])
                self.assertEqual(manifest["nested/loss"]["monotonic"], "inc")
                self.assertEqual(manifest["nested/loss"]["indices"], "ranges")
                self.assertEqual(manifest["sparse"]["indices"], "indices")
                self.assertIsInstance(manifest["dense"]["indices"], list)
                self.assertIsInstance(manifest["coalesce"]["indices"], list)
                self.assertNotIn("dense.indices", after)
                self.assertNotIn("coalesce.ranges", after)
                for name, data in before.items():
                    if name.endswith(".f32") or name in ("config.json", "notes.txt"):
                        self.assertEqual(after[name], data)
                with plattli.Reader(root) as reader:
                    for name, steps in expected.items():
                        np.testing.assert_array_equal(reader.metric_indices(name), steps)
                        np.testing.assert_array_equal(reader.metric_values(name), steps.astype(np.float32))
                self.assertEqual(optimize_indices(root), 0)
                self.assertEqual(_contents(root, zipped), after)
                if zipped:
                    with zipfile.ZipFile(root / "metrics.plattli") as archive:
                        self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))

    def test_directory_manifest_failure_keeps_old_indices_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected, _ = _create(root)
            before = (root / "plattli/plattli.json").read_bytes()
            with mock.patch("plattli.optimize_indices._replace_text_checked", side_effect=OSError("manifest failure")):
                with self.assertRaisesRegex(OSError, "manifest failure"):
                    optimize_indices(root)
            self.assertEqual((root / "plattli/plattli.json").read_bytes(), before)
            self.assertTrue((root / "plattli/dense.indices").is_file())
            self.assertTrue((root / "plattli/coalesce.ranges").is_file())
            with plattli.Reader(root) as reader:
                for name, steps in expected.items():
                    np.testing.assert_array_equal(reader.metric_indices(name), steps)

    def test_directory_retirement_follows_manifest_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected, _ = _create(root)
            unlink = Path.unlink

            def fail_retirement(path, *args, **kwargs):
                if path == root / "plattli/dense.indices":
                    raise OSError("retirement failure")
                return unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", fail_retirement), self.assertRaisesRegex(OSError, "retirement failure"):
                optimize_indices(root)
            self.assertIsInstance(json.loads((root / "plattli/plattli.json").read_text())["dense"]["indices"], list)
            with plattli.Reader(root) as reader:
                for name, steps in expected.items():
                    np.testing.assert_array_equal(reader.metric_indices(name), steps)

    def test_archive_failures_preserve_original_bytes(self):
        for failure in ("staging", "install"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _create(root, zipped=True)
                archive = root / "metrics.plattli"
                before = archive.read_bytes()
                if failure == "install":
                    patch = mock.patch.object(Path, "replace", side_effect=OSError("install failure"))
                else:
                    patch = mock.patch("plattli.optimize_indices.shutil.copyfileobj", side_effect=OSError("staging failure"))
                with patch, self.assertRaisesRegex(OSError, failure + " failure"):
                    optimize_indices(root)
                self.assertEqual(archive.read_bytes(), before)
                self.assertFalse((root / "metrics.plattli.tmp").exists())

    def test_hot_logs_are_rejected_without_changes(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _create(root, zipped=zipped, hot=True)
                before = _contents(root, zipped)
                with self.assertRaisesRegex(RuntimeError, "hot logs"):
                    optimize_indices(root)
                self.assertEqual(_contents(root, zipped), before)

    def test_missing_authoritative_files_fail_without_changes(self):
        for zipped in (False, True):
            for missing in ("nested/loss.f32", "dense.indices", "coalesce.ranges"):
                with self.subTest(zipped=zipped, missing=missing), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    _create(root, zipped=zipped)
                    if zipped:
                        contents = _contents(root, True)
                        del contents[missing]
                        with zipfile.ZipFile(root / "metrics.plattli", "w", compression=zipfile.ZIP_STORED) as archive:
                            for name, data in contents.items():
                                archive.writestr(name, data)
                        before = (root / "metrics.plattli").read_bytes()
                    else:
                        (root / "plattli" / missing).unlink()
                        before = _contents(root, False)
                    with self.assertRaises(FileNotFoundError):
                        optimize_indices(root)
                    self.assertEqual((root / "metrics.plattli").read_bytes() if zipped else _contents(root, False), before)

    def test_module_entry_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create(root)
            result = subprocess.run(
                [sys.executable, "-m", "plattli.optimize_indices", str(root)],
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), "Optimized indices for 4 metrics.")


if __name__ == "__main__":
    unittest.main()
