import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli


def _create(root, zipped, name="retired", extras=None):
    writer = plattli.BulkWriter(root, config={"original": True})
    for value in (100, 200):
        writer.write({name: value})
        writer.end_step()
    writer.finish(zip=zipped)
    for filename, payload in (extras or {}).items():
        if zipped:
            with zipfile.ZipFile(root / "metrics.plattli", "a") as archive:
                archive.writestr(filename, payload)
        else:
            path = root / "plattli" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in root.rglob("*")}


def _contents(root, zipped):
    if zipped:
        with zipfile.ZipFile(root / "metrics.plattli") as archive:
            return {name: archive.read(name) for name in archive.namelist()}
    return {path.relative_to(root / "plattli").as_posix(): path.read_bytes()
            for path in (root / "plattli").rglob("*") if path.is_file()}


class TestBulkOverwrite(unittest.TestCase):
    def test_overwrite_retires_live_sequence_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with plattli.DirectWriter(root, write_threads=0) as previous:
                previous.write(old=7)
            (root / "plattli/notes.values").write_text("keep")
            writer = plattli.BulkWriter(root, overwrite=True)
            writer.write(current=8)
            writer.finish(zip=False)
            self.assertFalse((root / "plattli/old.values").exists())
            self.assertEqual((root / "plattli/notes.values").read_text(), "keep")

    def test_format_transition_matrix_preserves_extras_and_retires_old_metrics(self):
        extras = {"notes.txt": b"notes", "notes.jsonl": b"json lines", "notes.u8": b"ancillary bytes",
                  "nested/report.txt": b"nested", "scratch.txt.tmp": b"unfinished note"}
        for before_zip in (False, True):
            for after_zip in (False, True):
                with self.subTest(before_zip=before_zip, after_zip=after_zip), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    _create(root, before_zip, extras=extras)
                    writer = plattli.BulkWriter(root, overwrite=True, config={"new": True})
                    writer.write(current=np.float64(1.123456789))
                    writer.end_step()
                    writer.finish(zip=after_zip)
                    self.assertEqual((root / "metrics.plattli").exists(), after_zip)
                    self.assertEqual((root / "plattli").exists(), not after_zip)
                    contents = _contents(root, after_zip)
                    self.assertNotIn("retired.u8", contents)
                    self.assertNotIn("retired.indices", contents)
                    for name, payload in extras.items():
                        self.assertEqual(contents[name], payload)
                    with plattli.Reader(root) as reader:
                        self.assertEqual(reader.metrics(), ["current"])
                        self.assertEqual(reader.metric_values("current").tolist(), [1.123456789])
                        self.assertEqual(reader.config(), {"new": True})
                    resumed = plattli.DirectWriter(root, step=1, write_threads=0, allow_resume_finalized=True)
                    resumed.finish(optimize=False, zip=False)

    def test_self_source_config_is_snapshotted_and_external_source_stays_linked(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _create(root, False)
                writer = plattli.BulkWriter(root, overwrite=True, config="plattli/config.json")
                writer.write(current=1)
                writer.finish(zip=zipped)
                self.assertEqual(json.loads(_contents(root, zipped)["config.json"]), {"original": True})
                if not zipped:
                    self.assertFalse((root / "plattli/config.json").is_symlink())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create(root, False)
            external = root / "external.json"
            external.write_text('{"original":true}')
            writer = plattli.BulkWriter(root, overwrite=True, config="external.json")
            writer.finish(zip=False)
            self.assertTrue((root / "plattli/config.json").is_symlink())
            external.write_text('{"changed":true}')
            self.assertEqual(json.loads((root / "plattli/config.json").read_text()), {"changed": True})

    def test_both_previous_outputs_preserve_identical_extras_and_reject_conflicts(self):
        for conflicting in (False, True):
            for zipped in (False, True):
                with self.subTest(conflicting=conflicting, zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "run"
                    _create(root, True, "zip_old", {"common.txt": b"same", "zip.txt": b"zip"})
                    other = Path(tmp) / "other"
                    _create(other, False, "dir_old", {"common.txt": b"diff" if conflicting else b"same",
                                                       "dir.txt": b"directory"})
                    shutil.copytree(other / "plattli", root / "plattli")
                    before = _snapshot(root)
                    writer = plattli.BulkWriter(root, overwrite=True)
                    writer.write(current=1)
                    if conflicting:
                        with self.assertRaisesRegex(ValueError, "conflicting ancillary"):
                            writer.finish(zip=zipped)
                        self.assertEqual(_snapshot(root), before)
                    else:
                        writer.finish(zip=zipped)
                        contents = _contents(root, zipped)
                        self.assertEqual(contents["common.txt"], b"same")
                        self.assertEqual(contents["zip.txt"], b"zip")
                        self.assertEqual(contents["dir.txt"], b"directory")
                        self.assertNotIn("zip_old.u8", contents)
                        self.assertNotIn("dir_old.u8", contents)
                        self.assertEqual((root / "metrics.plattli").exists(), zipped)
                        self.assertEqual((root / "plattli").exists(), not zipped)

    def test_unequal_ancillary_sizes_reject_without_opening_contents(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "run"
                _create(root, True, extras={"common.txt": b"short"})
                other = Path(tmp) / "other"
                _create(other, False, extras={"common.txt": b"longer"})
                shutil.copytree(other / "plattli", root / "plattli")
                before = _snapshot(root)
                writer = plattli.BulkWriter(root, overwrite=True)
                writer.write(current=7)
                with mock.patch("plattli.bulk_writer._open_ancillary", side_effect=AssertionError("opened ancillary contents")), \
                        self.assertRaisesRegex(ValueError, "conflicting ancillary"):
                    writer.finish(zip=zipped)
                self.assertEqual(_snapshot(root), before)

    def test_staging_and_install_failures_preserve_previous_output(self):
        for before_zip in (False, True):
            for after_zip in (False, True):
                for failure in ("write", "install"):
                    with self.subTest(before_zip=before_zip, after_zip=after_zip, failure=failure), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        _create(root, before_zip, extras={"notes.txt": b"keep"})
                        before = _snapshot(root)
                        writer = plattli.BulkWriter(root, overwrite=True)
                        writer.write(current=7)
                        replace = Path.replace
                        write_bytes = Path.write_bytes
                        writestr = zipfile.ZipFile.writestr

                        def fail_replace(path, target):
                            if path.parent.name.startswith(".plattli-") and path.name in ("plattli", "metrics.plattli"):
                                raise OSError("injected install failure")
                            return replace(path, target)

                        def fail_file_write(path, payload):
                            if path.name == "plattli.json":
                                raise OSError("injected write failure")
                            return write_bytes(path, payload)

                        def fail_zip_write(archive, name, payload, *args, **kwargs):
                            if name == "plattli.json":
                                raise OSError("injected write failure")
                            return writestr(archive, name, payload, *args, **kwargs)

                        patch = mock.patch.object(Path, "replace", fail_replace) if failure == "install" else \
                            mock.patch.object(zipfile.ZipFile, "writestr", fail_zip_write) if after_zip else \
                            mock.patch.object(Path, "write_bytes", fail_file_write)
                        with patch, self.assertRaisesRegex(OSError, "injected"):
                            writer.finish(zip=after_zip)
                        self.assertEqual(_snapshot(root), before)
                        self.assertFalse(writer._finished)
                        with plattli.Reader(root) as reader:
                            self.assertEqual(reader.metric_values("retired").tolist(), [100, 200])
                        writer.finish(zip=after_zip)
                        with plattli.Reader(root) as reader:
                            self.assertEqual(reader.metric_values("current").tolist(), [7])

    def test_failed_restore_retains_previous_directory_backup(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _create(root, False, extras={"notes.txt": b"keep"})
                old_files = _contents(root, False)
                writer = plattli.BulkWriter(root, overwrite=True)
                writer.write(current=7)
                replace = Path.replace

                def fail_install_and_restore(path, target):
                    if path.parent.name.startswith(".plattli-") and path.name in ("plattli", "metrics.plattli", "previous-plattli"):
                        raise OSError("injected install/restore failure")
                    return replace(path, target)

                with mock.patch.object(Path, "replace", fail_install_and_restore), self.assertRaisesRegex(RuntimeError, "retained files"):
                    writer.finish(zip=zipped)
                workspaces = list(root.glob(".plattli-*"))
                self.assertEqual(len(workspaces), 1)
                backup = workspaces[0] / "previous-plattli"
                self.assertTrue(backup.is_dir())
                for name, payload in old_files.items():
                    self.assertEqual((backup / name).read_bytes(), payload)
                backup.replace(root / "plattli")
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.metric_values("retired").tolist(), [100, 200])

    def test_directory_retirement_failure_preserves_previous_output(self):
        for had_archive in (False, True):
            with self.subTest(had_archive=had_archive), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "run"
                _create(root, False, extras={"notes.txt": b"keep"})
                if had_archive:
                    other = Path(tmp) / "other"
                    _create(other, True, "archive_old")
                    shutil.copyfile(other / "metrics.plattli", root / "metrics.plattli")
                before = _snapshot(root)
                writer = plattli.BulkWriter(root, overwrite=True)
                writer.write(current=7)
                replace = Path.replace

                def fail_retirement(path, target):
                    if path == root / "plattli":
                        raise OSError("injected directory retirement failure")
                    return replace(path, target)

                with mock.patch.object(Path, "replace", fail_retirement), self.assertRaisesRegex(OSError, "retirement"):
                    writer.finish(zip=True)
                self.assertEqual(_snapshot(root), before)
                self.assertFalse(writer._finished)
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.metric_values("archive_old" if had_archive else "retired").tolist(), [100, 200])
                if not had_archive:
                    resumed = plattli.DirectWriter(root, step=2, write_threads=0)
                    resumed.finish(optimize=False, zip=False)

    def test_zip_install_failure_restores_both_previous_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _create(root, False, extras={"notes.txt": b"keep"})
            other = Path(tmp) / "other"
            _create(other, True, "archive_old")
            shutil.copyfile(other / "metrics.plattli", root / "metrics.plattli")
            before = _snapshot(root)
            writer = plattli.BulkWriter(root, overwrite=True)
            writer.write(current=7)
            replace = Path.replace

            def fail_install(path, target):
                if path.parent.name.startswith(".plattli-") and path.name == "metrics.plattli":
                    raise OSError("injected install failure")
                return replace(path, target)

            with mock.patch.object(Path, "replace", fail_install), self.assertRaisesRegex(OSError, "install"):
                writer.finish(zip=True)
            self.assertEqual(_snapshot(root), before)

    def test_interrupt_after_zip_publication_keeps_new_archive_and_old_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create(root, False, extras={"notes.txt": b"keep"})
            old_files = _contents(root, False)
            writer = plattli.BulkWriter(root, overwrite=True)
            writer.write(current=7)
            replace = Path.replace

            def interrupt_after_publication(path, target):
                result = replace(path, target)
                if path.parent.name.startswith(".plattli-") and path.name == "metrics.plattli":
                    raise OSError("interrupted after ZIP publication")
                return result

            with mock.patch.object(Path, "replace", interrupt_after_publication), self.assertRaisesRegex(OSError, "publication"):
                writer.finish(zip=True)
            self.assertFalse((root / "plattli").exists())
            with plattli.Reader(root) as reader:
                self.assertEqual(reader.metric_values("current").tolist(), [7])
            workspaces = list(root.glob(".plattli-*"))
            self.assertEqual(len(workspaces), 1)
            backup = workspaces[0] / "previous-plattli"
            for name, payload in old_files.items():
                self.assertEqual((backup / name).read_bytes(), payload)

    def test_interrupt_after_directory_publication_keeps_new_output_and_old_backup(self):
        for had_directory in (False, True):
            with self.subTest(had_directory=had_directory), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                old_files = None
                if had_directory:
                    _create(root, False, extras={"notes.txt": b"keep"})
                    old_files = _contents(root, False)
                writer = plattli.BulkWriter(root, overwrite=True)
                writer.write(current=7)
                replace = Path.replace

                def interrupt_after_publication(path, target):
                    result = replace(path, target)
                    if path.parent.name.startswith(".plattli-") and path.name == "plattli":
                        raise KeyboardInterrupt("interrupted after directory publication")
                    return result

                with mock.patch.object(Path, "replace", interrupt_after_publication), \
                        self.assertRaisesRegex(KeyboardInterrupt, "publication"):
                    writer.finish(zip=False)
                self.assertFalse(writer._finished)
                self.assertFalse((root / "metrics.plattli").exists())
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.metrics(), ["current"])
                    self.assertEqual(reader.metric_values("current").tolist(), [7])
                workspaces = list(root.glob(".plattli-*"))
                self.assertEqual(len(workspaces), int(had_directory))
                if had_directory:
                    backup = workspaces[0] / "previous-plattli"
                    for name, payload in old_files.items():
                        self.assertEqual((backup / name).read_bytes(), payload)
                resumed = plattli.DirectWriter(root, step=1, write_threads=0)
                resumed.write(current=8)
                resumed.finish(optimize=False, zip=False)
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.metric_values("current").tolist(), [7, 8])

    def test_interrupt_after_directory_install_restores_output_when_old_archive_remains(self):
        for had_directory in (False, True):
            with self.subTest(had_directory=had_directory), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "run"
                _create(root, True)
                if had_directory:
                    other = Path(tmp) / "other"
                    _create(other, False, "dir_old", {"notes.txt": b"keep"})
                    shutil.copytree(other / "plattli", root / "plattli")
                before = _snapshot(root)
                writer = plattli.BulkWriter(root, overwrite=True)
                writer.write(current=7)
                replace = Path.replace

                def interrupt_after_install(path, target):
                    result = replace(path, target)
                    if path.parent.name.startswith(".plattli-") and path.name == "plattli":
                        raise KeyboardInterrupt("interrupted after directory install")
                    return result

                with mock.patch.object(Path, "replace", interrupt_after_install), \
                        self.assertRaisesRegex(KeyboardInterrupt, "install"):
                    writer.finish(zip=False)
                self.assertEqual(_snapshot(root), before)
                self.assertFalse(writer._finished)
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.metric_values("retired").tolist(), [100, 200])

    def test_archive_retirement_failure_restores_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create(root, True)
            before = _snapshot(root)
            writer = plattli.BulkWriter(root, overwrite=True)
            writer.write(current=7)
            unlink = Path.unlink

            def fail_retirement(path, *args, **kwargs):
                if path == root / "metrics.plattli":
                    raise OSError("injected retirement failure")
                return unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", fail_retirement), self.assertRaisesRegex(OSError, "retirement"):
                writer.finish(zip=False)
            self.assertEqual(_snapshot(root), before)

    def test_retirement_interrupt_after_unlink_keeps_new_authoritative_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create(root, True)
            writer = plattli.BulkWriter(root, overwrite=True)
            writer.write(current=7)
            unlink = Path.unlink

            def interrupt_after_unlink(path, *args, **kwargs):
                result = unlink(path, *args, **kwargs)
                if path == root / "metrics.plattli":
                    raise OSError("interrupted after retirement")
                return result

            with mock.patch.object(Path, "unlink", interrupt_after_unlink), self.assertRaises(OSError):
                writer.finish(zip=False)
            with plattli.Reader(root) as reader:
                self.assertEqual(reader.metric_values("current").tolist(), [7])

    def test_unsafe_archive_extras_and_path_conflicts_fail_before_publication(self):
        for extra, metric in (("../escape.txt", "current"), ("nested", "nested/current"),
                              ("current.indices", "current")):
            for zipped in (False, True):
                with self.subTest(extra=extra, zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "run"
                    _create(root, True, extras={extra: b"keep"})
                    before = _snapshot(root)
                    writer = plattli.BulkWriter(root, overwrite=True)
                    writer.write({metric: 7})
                    with self.assertRaises((RuntimeError, ValueError)):
                        writer.finish(zip=zipped)
                    self.assertEqual(_snapshot(root), before)
                    self.assertFalse((root / "escape.txt").exists())
