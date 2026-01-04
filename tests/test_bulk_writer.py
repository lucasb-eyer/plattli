import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from plattli import PlattliBulkWriter


class TestPlattliBulkWriter(unittest.TestCase):
    def test_negative_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(AssertionError):
                PlattliBulkWriter(root, step=-1)

    def test_basic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            w = PlattliBulkWriter(root)
            w.write(loss=1.2, note="ok", meta={"a": 1})
            w.end_step()
            w.write(loss=1.3)
            w.end_step()
            w.finish(zip=False)

            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            note_vals = json.loads((root / "note.json").read_text(encoding="utf-8"))
            meta_vals = json.loads((root / "meta.json").read_text(encoding="utf-8"))
            note_idx = np.fromfile(root / "note.indices", dtype=np.uint32)
            manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))

            self.assertTrue(np.allclose(loss_vals, [1.2, 1.3]))
            self.assertFalse((root / "loss.indices").exists())
            self.assertEqual(note_vals, ["ok"])
            self.assertEqual(meta_vals, [{"a": 1}])
            self.assertEqual(note_idx.tolist(), [0])
            self.assertEqual(manifest["loss"]["dtype"], "f32")
            self.assertEqual(manifest["loss"]["indices"], {"start": 0, "stop": 2, "step": 1})
            self.assertEqual(manifest["note"]["dtype"], "json")
            self.assertEqual(manifest["run_rows"], 2)

    def test_duplicate_metric_same_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = PlattliBulkWriter(root)
            w.write(loss=1.0)
            with self.assertRaises(RuntimeError):
                w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(root / "loss.indices", dtype=np.uint32)
            manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))
            self.assertTrue(np.allclose(loss_vals, [1.0, 3.0]))
            self.assertEqual(loss_idx.tolist(), [0, 1])
            self.assertEqual(manifest["loss"]["indices"], "indices")

    def test_optimize_zip_and_tighten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            w = PlattliBulkWriter(root)
            w.write(loss=1, delta=-1, note="ok")
            w.end_step()
            w.write(loss=2, delta=-2)
            w.end_step()
            w.finish(zip=True)

            zip_path = Path(f"{root}.zip")
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path) as zf:
                manifest = json.loads(zf.read("plattli.json"))
                self.assertEqual(manifest["loss"]["dtype"], "u8")
                self.assertEqual(manifest["delta"]["dtype"], "i8")
                self.assertEqual(manifest["note"]["dtype"], "json")
                self.assertEqual(manifest["loss"]["indices"], {"start": 0, "stop": 2, "step": 1})
                self.assertEqual(manifest["run_rows"], 2)
                self.assertIn("loss.u8", zf.namelist())
                self.assertNotIn("loss.indices", zf.namelist())
                self.assertIn("note.indices", zf.namelist())

    def test_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            w = PlattliBulkWriter(root, config={"seed": 7})
            w.write(loss=1.0)
            w.end_step()
            w.set_config({"seed": 9, "note": "ok"})
            w.finish(zip=False)

            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config, {"seed": 9, "note": "ok"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
