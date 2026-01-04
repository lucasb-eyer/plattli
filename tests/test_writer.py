import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from plattli import PlattliWriter


class TestPlattliWriter(unittest.TestCase):
    def test_basic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliWriter(root)
            w.write(loss=1.2)
            w.write(note="ok")
            w.end_step()
            w.write(loss=1.3)
            w.end_step()
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(root / "loss.indices", dtype=np.uint32)
            note_vals = json.loads((root / "note.json").read_text(encoding="utf-8"))
            note_idx = np.fromfile(root / "note.indices", dtype=np.uint32)
            manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))

            self.assertTrue(np.allclose(loss_vals, [1.2, 1.3]))
            self.assertEqual(loss_idx.tolist(), [0, 1])
            self.assertEqual(note_vals, ["ok"])
            self.assertEqual(note_idx.tolist(), [0])
            self.assertEqual(manifest["loss"]["dtype"], "f32")
            self.assertEqual(manifest["loss"]["indices"], "indices")
            self.assertNotIn("rows", manifest["loss"])
            self.assertEqual(manifest["note"]["dtype"], "json")
            self.assertNotIn("rows", manifest["note"])
            self.assertIn("when_exported", manifest)

    def test_dtype_cast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliWriter(root, write_threads=0)
            w.write(loss=np.float64(1.5))
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(root / "loss.f64", dtype=np.float64)
            self.assertEqual(loss_vals.tolist(), [1.5])
            self.assertEqual((root / "loss.f64").stat().st_size, 8)

    def test_dtype_cast_existing_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliWriter(root, write_threads=0)
            w.write(loss=np.float32(1.0))
            w.end_step()
            w.write(loss=np.float64(1.5))
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            self.assertTrue(np.allclose(loss_vals, [1.0, 1.5]))
            self.assertEqual((root / "loss.f32").stat().st_size, 8)

    def test_duplicate_metric_same_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliWriter(root, write_threads=0)
            w.write(loss=1.0)
            with self.assertRaises(RuntimeError):
                w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(root / "loss.indices", dtype=np.uint32)
            self.assertTrue(np.allclose(loss_vals, [1.0, 3.0]))
            self.assertEqual(loss_idx.tolist(), [0, 1])

    def test_bool_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliWriter(root, write_threads=0)
            w.write(flag=True)
            w.end_step()
            w.write(flag=False)
            w.finish(optimize=False, zip=False)

            flag_vals = json.loads((root / "flag.json").read_text(encoding="utf-8"))
            self.assertEqual(flag_vals, [True, False])

    def test_background_error_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliWriter(root, write_threads=1)
            try:
                w.step = 0x1_0000_0000
                w.write(loss=1.0)
                with self.assertRaises(ValueError) as ctx:
                    w.end_step()
                notes = getattr(ctx.exception, "__notes__", [])
                self.assertTrue(any("Background write failed" in note for note in notes))
            finally:
                if w._executor:
                    w._executor.shutdown(wait=True)

    def test_resume_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliWriter(root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            w = PlattliWriter(root, step=1, write_threads=0)
            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(root / "loss.indices", dtype=np.uint32)
            self.assertTrue(np.allclose(loss_vals, [1.0]))
            self.assertEqual(loss_idx.tolist(), [0])

            w.write(loss=9.0)
            w.end_step()
            w.finish(optimize=False, zip=False)
            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(root / "loss.indices", dtype=np.uint32)
            self.assertTrue(np.allclose(loss_vals, [1.0, 9.0]))
            self.assertEqual(loss_idx.tolist(), [0, 1])

    def test_optimize_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            w = PlattliWriter(root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=True, zip=False)

            manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["loss"]["indices"], {"start": 0, "stop": 3, "step": 1})
            self.assertFalse((root / "loss.indices").exists())
            self.assertEqual(manifest["run_rows"], 3)

    def test_optimize_zip_and_tighten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            w = PlattliWriter(root, write_threads=0)
            w.write(loss=1)
            w.end_step()
            w.write(loss=2)
            w.end_step()
            w.finish(optimize=True, zip=True)

            zip_path = Path(f"{root}.zip")
            self.assertTrue(zip_path.exists())
            self.assertFalse(root.exists())
            with zipfile.ZipFile(zip_path) as zf:
                self.assertIn("plattli.json", zf.namelist())
                manifest = json.loads(zf.read("plattli.json"))
                self.assertEqual(manifest["loss"]["dtype"], "u8")
                self.assertIn("loss.u8", zf.namelist())
                self.assertNotIn("loss.i64", zf.namelist())
                self.assertEqual(manifest["run_rows"], 2)

    def test_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            w = PlattliWriter(root, config={"seed": 7})
            w.write(loss=1.0)
            w.end_step()
            w.set_config({"seed": 9, "note": "ok"})
            w.finish(optimize=False, zip=False)

            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config, {"seed": 9, "note": "ok"})

    def test_resume_from_compacted_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            w = PlattliWriter(root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=True, zip=False)

            self.assertFalse((root / "loss.indices").exists())

            w = PlattliWriter(root, step=2, write_threads=0)
            loss_idx = np.fromfile(root / "loss.indices", dtype=np.uint32)
            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))

            self.assertEqual(loss_idx.tolist(), [0, 1])
            self.assertTrue(np.allclose(loss_vals, [1.0, 2.0]))
            self.assertEqual(manifest["loss"]["indices"], "indices")


if __name__ == "__main__":
    unittest.main()
