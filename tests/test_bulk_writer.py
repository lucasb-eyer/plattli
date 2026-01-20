import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

import plattli
from plattli.writer import _zip_path_for_root


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestPlattliBulkWriter(unittest.TestCase):
    def test_negative_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            with self.assertRaises(AssertionError):
                plattli.PlattliBulkWriter(run_root, step=-1)

    def test_basic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.PlattliBulkWriter(run_root)
            w.write(loss=1.2, note="ok", meta={"a": 1})
            w.end_step()
            w.write(loss=1.3)
            w.end_step()
            w.finish(zip=False)

            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            note_vals = _read_jsonl(plattli_root / "note.jsonl")
            meta_vals = _read_jsonl(plattli_root / "meta.jsonl")
            note_idx = np.fromfile(plattli_root / "note.indices", dtype=np.uint32)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))

            self.assertTrue(np.allclose(loss_vals, [1.2, 1.3]))
            self.assertFalse((plattli_root / "loss.indices").exists())
            self.assertEqual(note_vals, ["ok"])
            self.assertEqual(meta_vals, [{"a": 1}])
            self.assertEqual(note_idx.tolist(), [0])
            self.assertEqual(manifest["loss"]["dtype"], "f32")
            self.assertEqual(manifest["loss"]["indices"], [{"start": 0, "stop": 2, "step": 1}])
            self.assertEqual(manifest["note"]["dtype"], "jsonl")
            self.assertEqual(manifest["run_rows"], 2)

    def test_duplicate_metric_same_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = plattli.PlattliBulkWriter(run_root)
            w.write(loss=1.0)
            with self.assertRaises(RuntimeError):
                w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(plattli_root / "loss.indices", dtype=np.uint32)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertTrue(np.allclose(loss_vals, [1.0, 3.0]))
            self.assertEqual(loss_idx.tolist(), [0, 1])
            self.assertEqual(manifest["loss"]["indices"], "indices")

    def test_optimize_zip_and_tighten(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.PlattliBulkWriter(run_root)
            w.write(loss=1, delta=-1, note="ok")
            w.end_step()
            w.write(loss=2, delta=-2)
            w.end_step()
            w.finish(zip=True)

            zip_path = _zip_path_for_root(run_root)
            self.assertTrue(zip_path.exists())
            with zipfile.ZipFile(zip_path) as zf:
                manifest = json.loads(zf.read("plattli.json"))
                self.assertEqual(manifest["loss"]["dtype"], "u8")
                self.assertEqual(manifest["delta"]["dtype"], "i8")
                self.assertEqual(manifest["note"]["dtype"], "jsonl")
                self.assertEqual(manifest["loss"]["indices"], [{"start": 0, "stop": 2, "step": 1}])
                self.assertEqual(manifest["run_rows"], 2)
                self.assertIn("loss.u8", zf.namelist())
                self.assertNotIn("loss.indices", zf.namelist())
                self.assertIn("note.indices", zf.namelist())
                self.assertIn("note.jsonl", zf.namelist())

    def test_config_symlink_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            target = Path(tmp) / "config_source.json"
            target.write_text(json.dumps({"seed": 7}), encoding="utf-8")
            w = plattli.PlattliBulkWriter(run_root, config=str(target))
            w.write(loss=1.0)
            w.end_step()
            w.finish(zip=True)

            zip_path = _zip_path_for_root(run_root)
            with zipfile.ZipFile(zip_path) as zf:
                config = json.loads(zf.read("config.json"))
            self.assertEqual(config, {"seed": 7})

    def test_config_auto_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "config.json").write_text(json.dumps({"seed": 3}), encoding="utf-8")
            w = plattli.PlattliBulkWriter(run_root)
            w.write(loss=1.0)
            w.end_step()
            w.finish(zip=False)

            self.assertTrue((plattli_root / "config.json").is_symlink())
            config = json.loads((plattli_root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config, {"seed": 3})

    def test_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.PlattliBulkWriter(run_root, config={"seed": 7})
            w.write(loss=1.0)
            w.end_step()
            w.set_config({"seed": 9, "note": "ok"})
            w.finish(zip=False)

            config = json.loads((plattli_root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config, {"seed": 9, "note": "ok"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
