import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

import plattli
from plattli.writer import _zip_path_for_root


class TestReader(unittest.TestCase):
    def test_reader_piecewise_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            steps = [2, 1000, 2000, 3000, 4000]
            for step in steps:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=False)

            self.assertFalse((plattli_root / "loss.indices").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), steps)
                self.assertTrue(np.allclose(r.metric_values("loss"), np.asarray(steps, dtype=np.float32)))

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            steps = [2, 1000, 2000, 3000, 4000]
            for step in steps:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=True)

            self.assertTrue(_zip_path_for_root(run_root).exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), steps)
                self.assertTrue(np.allclose(r.metric_values("loss"), np.asarray(steps, dtype=np.float32)))

    def test_reader_rows_hot_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=100)
            w.write(loss=1.0, acc=0.5)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0, acc=0.7)
            w.end_step()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)
                self.assertEqual(r.rows("acc"), 2)

    def test_run_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            self.assertTrue(plattli.is_run(run_root))
            self.assertTrue(plattli.is_run_dir(run_root))
            self.assertEqual(plattli.resolve_run_dir(run_root), plattli_root.resolve())
            self.assertTrue(plattli.is_run(plattli_root))
            self.assertTrue(plattli.is_run_dir(plattli_root))
            self.assertEqual(plattli.resolve_run_dir(plattli_root), plattli_root.resolve())

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "ziprun"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=True)

            zip_path = run_root / "metrics.plattli"
            self.assertTrue(zip_path.exists())
            self.assertTrue(plattli.is_run(run_root))
            self.assertFalse(plattli.is_run_dir(run_root))
            self.assertIsNone(plattli.resolve_run_dir(run_root))
            self.assertTrue(plattli.is_run(zip_path))
            self.assertFalse(plattli.is_run_dir(zip_path))
            self.assertIsNone(plattli.resolve_run_dir(zip_path))

    def test_reader_rows_columnar_plus_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            if w._compact_future:
                w._compact_future.result()
            w.write(loss=3.0)
            w.end_step()
            if w._compact_future:
                w._compact_future.result()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)

    def test_reader_tolerates_partial_numeric_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(b"\x01\x02")
            with (plattli_root / "loss.f32").open("ab") as fh:
                fh.write(b"\x03\x04")

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1])
                self.assertTrue(np.allclose(r.metric_values("loss"),
                                            np.asarray([1.0, 2.0], dtype=np.float32)))
                self.assertEqual(r.rows("loss"), 2)

    def test_reader_tolerates_mismatch_and_jsonl_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0, text="a")
            w.end_step()
            w.write(loss=2.0, text="b")
            w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(np.asarray([2], dtype=np.uint32).tobytes())
            with (plattli_root / "text.jsonl").open("ab") as fh:
                fh.write(b"\"c")

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1])
                self.assertTrue(np.allclose(r.metric_values("loss"),
                                            np.asarray([1.0, 2.0], dtype=np.float32)))
                self.assertEqual(r.metric_values("text").tolist(), ["a", "b"])
                self.assertEqual(r.rows("text"), 2)

    def test_reader_missing_files_fail_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0, note="ok")
            w.end_step()
            w.finish(optimize=False, zip=False)

            (plattli_root / "loss.f32").unlink()
            (plattli_root / "note.indices").unlink()

            with plattli.Reader(run_root) as r:
                with self.assertRaises(FileNotFoundError):
                    r.metric_values("loss")
                with self.assertRaises(FileNotFoundError):
                    r.metric_indices("note")

    def test_reader_rows_mismatch_indices_uses_last_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(3):
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(np.asarray([100], dtype=np.uint32).tobytes())
            (plattli_root / "hot.jsonl").write_text(
                json.dumps({"step": 3, "loss": 9.0}) + "\n",
                encoding="utf-8",
            )

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 4)

    def test_reader_zip_indices_mismatch_and_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(3):
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(np.asarray([3], dtype=np.uint32).tobytes())

            zip_path = run_root / "metrics.plattli"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for path in sorted(plattli_root.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(plattli_root)
                    if path.is_symlink():
                        zf.writestr(rel.as_posix(), path.read_bytes())
                    else:
                        zf.write(path, rel)

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1, 2])
                self.assertEqual(r.rows("loss"), 3)
                idx, value = r.metric("loss", idx=1)
                self.assertEqual(int(idx), 1)
                self.assertEqual(float(value), 1.0)

    def test_reader_segment_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(5):
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=False)

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertIsInstance(manifest["loss"]["indices"], list)

            value_path = plattli_root / "loss.f32"
            with value_path.open("r+b") as fh:
                fh.truncate(3 * np.dtype(np.float32).itemsize)

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1, 2])

    def test_reader_columnar_and_hot_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            w.write(loss=1.0, text="a")
            w.end_step()
            w.write(loss=2.0, text="b")
            w.end_step()
            if w._compact_future:
                w._compact_future.result()
            w.write(loss=3.0, text="c", hot_only=7)
            w.end_step()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_values("loss").tolist(), [1.0, 2.0, 3.0])
                self.assertEqual(r.metric_values("text").tolist(), ["a", "b", "c"])
                self.assertEqual(r.metric_values("hot_only").tolist(), [7])
                with self.assertRaises(KeyError):
                    r.metric_values("missing")
            w.finish(optimize=False, zip=False)

    def test_reader_approx_max_rows_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            if w._compact_future:
                w._compact_future.result()
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.approx_max_rows(), 2)
                self.assertEqual(r.approx_max_rows(faster=False), 3)
            w.finish(optimize=False, zip=False)

    def test_reader_approx_max_rows_indices_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(3):
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.approx_max_rows(), 3)

            plattli_root = run_root / "plattli"
            zip_path = run_root / "metrics.plattli"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for path in sorted(plattli_root.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(plattli_root)
                    if path.is_symlink():
                        zf.writestr(rel.as_posix(), path.read_bytes())
                    else:
                        zf.write(path, rel)
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.approx_max_rows(), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
