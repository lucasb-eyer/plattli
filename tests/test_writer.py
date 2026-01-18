import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from plattli import CompactingWriter, DirectWriter, Reader
from plattli.writer import _find_arange_params, _zip_path_for_root


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestDirectWriter(unittest.TestCase):
    def test_negative_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            with self.assertRaises(AssertionError):
                DirectWriter(run_root, step=-1)

    def test_basic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root)
            w.write(loss=1.2)
            w.write(note="ok")
            w.end_step()
            w.write(loss=1.3)
            w.end_step()
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(plattli_root / "loss.indices", dtype=np.uint32)
            note_vals = _read_jsonl(plattli_root / "note.jsonl")
            note_idx = np.fromfile(plattli_root / "note.indices", dtype=np.uint32)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))

            self.assertTrue(np.allclose(loss_vals, [1.2, 1.3]))
            self.assertEqual(loss_idx.tolist(), [0, 1])
            self.assertEqual(note_vals, ["ok"])
            self.assertEqual(note_idx.tolist(), [0])
            self.assertEqual(manifest["loss"]["dtype"], "f32")
            self.assertEqual(manifest["loss"]["indices"], "indices")
            self.assertNotIn("rows", manifest["loss"])
            self.assertEqual(manifest["note"]["dtype"], "jsonl")
            self.assertNotIn("rows", manifest["note"])
            self.assertIn("when_exported", manifest)

    def test_dtype_cast(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=np.float64(1.5))
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(plattli_root / "loss.f64", dtype=np.float64)
            self.assertEqual(loss_vals.tolist(), [1.5])
            self.assertEqual((plattli_root / "loss.f64").stat().st_size, 8)

    def test_dtype_cast_existing_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=np.float32(1.0))
            w.end_step()
            w.write(loss=np.float64(1.5))
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            self.assertTrue(np.allclose(loss_vals, [1.0, 1.5]))
            self.assertEqual((plattli_root / "loss.f32").stat().st_size, 8)

    def test_duplicate_metric_same_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            with self.assertRaises(RuntimeError):
                w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(plattli_root / "loss.indices", dtype=np.uint32)
            self.assertTrue(np.allclose(loss_vals, [1.0, 3.0]))
            self.assertEqual(loss_idx.tolist(), [0, 1])

    def test_bool_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(flag=True)
            w.end_step()
            w.write(flag=False)
            w.finish(optimize=False, zip=False)

            flag_vals = _read_jsonl(plattli_root / "flag.jsonl")
            self.assertEqual(flag_vals, [True, False])

    def test_background_error_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            w = DirectWriter(run_root, write_threads=1)
            try:
                w.step = 0x1_0000_0000
                with self.assertRaises(ValueError):
                    w.write(loss=1.0)
            finally:
                if w._executor:
                    w._executor.shutdown(wait=True)

    def test_resume_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0, note="a")
            w.end_step()
            w.write(loss=2.0, note="b")
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            w = DirectWriter(run_root, step=1, write_threads=0)
            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(plattli_root / "loss.indices", dtype=np.uint32)
            note_vals = _read_jsonl(plattli_root / "note.jsonl")
            note_idx = np.fromfile(plattli_root / "note.indices", dtype=np.uint32)
            self.assertTrue(np.allclose(loss_vals, [1.0]))
            self.assertEqual(loss_idx.tolist(), [0])
            self.assertEqual(note_vals, ["a"])
            self.assertEqual(note_idx.tolist(), [0])

            w.write(loss=9.0)
            w.end_step()
            w.finish(optimize=False, zip=False)
            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            loss_idx = np.fromfile(plattli_root / "loss.indices", dtype=np.uint32)
            self.assertTrue(np.allclose(loss_vals, [1.0, 9.0]))
            self.assertEqual(loss_idx.tolist(), [0, 1])

    def test_hot_resume_after_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = CompactingWriter(run_root, hotsize=5)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            self.assertTrue((plattli_root / "hot.jsonl").exists())
            w = None

            w = CompactingWriter(run_root, step=2, hotsize=5)
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            self.assertTrue(np.allclose(loss_vals, [1.0, 2.0, 3.0]))
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["loss"]["indices"], [{"start": 0, "stop": 3, "step": 1}])
            self.assertFalse((plattli_root / "loss.indices").exists())
            self.assertFalse((plattli_root / "hot.jsonl").exists())

    def test_compacting_indices_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = CompactingWriter(run_root, hotsize=100)
            steps = [0, 10, 11, 25, 26, 40, 41, 60, 61, 85]
            for step in steps:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=False, zip=False)

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["loss"]["indices"], "indices")
            idx = np.fromfile(plattli_root / "loss.indices", dtype=np.uint32)
            self.assertEqual(idx.tolist(), steps)

    def test_optimize_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=True, zip=False)

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["loss"]["indices"], [{"start": 0, "stop": 3, "step": 1}])
            self.assertFalse((plattli_root / "loss.indices").exists())
            self.assertEqual(manifest["run_rows"], 3)

    def test_optimize_piecewise_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.step = 2
            w.write(loss=1.0)
            w.end_step()
            w.step = 1000
            w.write(loss=2.0)
            w.end_step()
            w.step = 2000
            w.write(loss=3.0)
            w.end_step()
            w.step = 3000
            w.write(loss=4.0)
            w.end_step()
            w.step = 4000
            w.write(loss=5.0)
            w.end_step()
            w.finish(optimize=True, zip=False)

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["loss"]["indices"],
                [{"start": 2, "stop": 1998, "step": 998}, {"start": 2000, "stop": 5000, "step": 1000}],
            )
            self.assertFalse((plattli_root / "loss.indices").exists())

    def test_optimize_zip_and_tighten(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1, delta=-1, note="ok")
            w.end_step()
            w.write(loss=2, delta=-2)
            w.end_step()
            w.finish(optimize=True, zip=True)

            zip_path = _zip_path_for_root(run_root)
            self.assertTrue(zip_path.exists())
            self.assertTrue(run_root.exists())
            self.assertFalse(plattli_root.exists())
            with zipfile.ZipFile(zip_path) as zf:
                self.assertIn("plattli.json", zf.namelist())
                manifest = json.loads(zf.read("plattli.json"))
                self.assertEqual(manifest["loss"]["dtype"], "u8")
                self.assertEqual(manifest["delta"]["dtype"], "i8")
                self.assertEqual(manifest["note"]["dtype"], "jsonl")
                self.assertIn("loss.u8", zf.namelist())
                self.assertNotIn("loss.i64", zf.namelist())
                self.assertIn("delta.i8", zf.namelist())
                self.assertNotIn("delta.i64", zf.namelist())
                self.assertIn("note.jsonl", zf.namelist())
                self.assertEqual(manifest["run_rows"], 2)

    def test_config_symlink_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            target = Path(tmp) / "config_source.json"
            target.write_text(json.dumps({"seed": 7}), encoding="utf-8")
            w = DirectWriter(run_root, write_threads=0, config=str(target))
            self.assertTrue((plattli_root / "config.json").is_symlink())
            w.write(loss=1.0)
            w.finish(optimize=False, zip=True)
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
            w = DirectWriter(run_root, write_threads=0)
            self.assertTrue((plattli_root / "config.json").is_symlink())
            w.write(loss=1.0)
            w.finish(optimize=False, zip=False)
            config = json.loads((plattli_root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config, {"seed": 3})

    def test_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, config={"seed": 7})
            w.write(loss=1.0)
            w.end_step()
            w.set_config({"seed": 9, "note": "ok"})
            w.finish(optimize=False, zip=False)

            config = json.loads((plattli_root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config, {"seed": 9, "note": "ok"})

    def test_resume_from_compacted_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=True, zip=False)

            self.assertFalse((plattli_root / "loss.indices").exists())

            w = DirectWriter(run_root, step=2, write_threads=0)
            self.assertTrue((plattli_root / "loss.indices").exists())
            loss_vals = np.fromfile(plattli_root / "loss.f32", dtype=np.float32)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))

            self.assertTrue(np.allclose(loss_vals, [1.0, 2.0]))
            self.assertEqual(manifest["loss"]["indices"], "indices")

            w.write(loss=9.0)
            w.end_step()
            w.finish(optimize=False, zip=False)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["loss"]["indices"], "indices")

    def test_resume_from_piecewise_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.step = 2
            w.write(loss=1.0)
            w.end_step()
            w.step = 1000
            w.write(loss=2.0)
            w.end_step()
            w.step = 2000
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=True, zip=False)

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertIsInstance(manifest["loss"]["indices"], list)
            self.assertFalse((plattli_root / "loss.indices").exists())

            w = DirectWriter(run_root, step=1000, write_threads=0)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["loss"]["indices"], "indices")
            self.assertTrue((plattli_root / "loss.indices").exists())

            w.write(loss=6.0)
            w.end_step()
            w.finish(optimize=False, zip=False)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["loss"]["indices"], "indices")

    def test_reject_non_scalar_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            with self.assertRaises(ValueError):
                w.write(loss=[1, 2])

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            w = DirectWriter(run_root, write_threads=0)
            with self.assertRaises(ValueError):
                w.write(loss=np.array([1, 2]))

    def test_unsupported_array_dtype_falls_back_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(note=np.array("x", dtype="U"))
            w.end_step()
            w.finish(optimize=False, zip=False)

            note_vals = _read_jsonl(plattli_root / "note.jsonl")
            self.assertEqual(note_vals, ["x"])

    def test_find_arange_params_none(self):
        self.assertIsNone(_find_arange_params(np.array([0], dtype=np.uint32)))
        self.assertIsNone(_find_arange_params(np.array([0, 1, 3], dtype=np.uint32)))


class TestReader(unittest.TestCase):
    def test_reader_piecewise_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            steps = [2, 1000, 2000, 3000, 4000]
            for step in steps:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=False)

            self.assertFalse((plattli_root / "loss.indices").exists())
            with Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), steps)
                self.assertTrue(np.allclose(r.metric_values("loss"), np.asarray(steps, dtype=np.float32)))

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = DirectWriter(run_root, write_threads=0)
            steps = [2, 1000, 2000, 3000, 4000]
            for step in steps:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=True)

            self.assertTrue(_zip_path_for_root(run_root).exists())
            with Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), steps)
                self.assertTrue(np.allclose(r.metric_values("loss"), np.asarray(steps, dtype=np.float32)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
