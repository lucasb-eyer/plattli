import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
import plattli.bulk_writer as bulk_module
from plattli.writer import _zip_path_for_root


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestBulkWriter(unittest.TestCase):
    def test_fixed_width_appends_defer_sequence_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = plattli.BulkWriter(tmp)
            with mock.patch.object(bulk_module, "_find_step_sequence", side_effect=AssertionError("per-value affine inference")), \
                    mock.patch.object(bulk_module, "_find_value_sequence", side_effect=AssertionError("per-value sequence inference")):
                for row in range(20):
                    writer.write(constant=7, floating=0.5, unsigned=np.uint64(2**63 + 5),
                                 arithmetic=row * 3, irregular=row * row, huge=10**23 + row * row)
                    writer.end_step()
            writer.finish(zip=False)
            with plattli.Reader(tmp) as reader:
                for name in ("constant", "floating", "unsigned", "arithmetic"):
                    self.assertIn("values", reader.manifest()[name])
                self.assertNotIn("values", reader.manifest()["irregular"])
                self.assertEqual(reader.metric_values("irregular").tolist(), [row * row for row in range(20)])

    def test_big_integer_promotion_defers_sequence_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = plattli.BulkWriter(tmp)
            writer.write(counter=7)
            writer.end_step()
            with mock.patch.object(bulk_module, "_find_value_sequence", wraps=bulk_module._find_value_sequence) as infer:
                for row in range(1, 5):
                    writer.write(counter=7 + row * 10**20)
                    writer.end_step()
                self.assertEqual(infer.call_count, 0)
            writer.finish(zip=False)
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values("counter").tolist(), [7 + row * 10**20 for row in range(5)])

    def test_irregular_unsigned_columns_accept_large_python_integers(self):
        for kind in ("bulk", "direct", "cold", "hot"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = plattli.BulkWriter(tmp) if kind == "bulk" else \
                    plattli.DirectWriter(tmp, write_threads=0) if kind == "direct" else \
                    plattli.CompactingWriter(tmp, hotsize=2 if kind == "cold" else 100)
                expected = [2**63 + offset for offset in (100, 104, 110, 123)] + [2**64 - 1]
                for row, value in enumerate(expected):
                    writer.write(counter=np.uint64(value) if row < 3 else value)
                    writer.end_step()
                for value in (2**64, -1):
                    writer.write(counter=value)
                    writer.end_step()
                    expected.append(value)
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.manifest()["counter"]["dtype"], "i128")
                    self.assertNotIn("values", reader.manifest()["counter"])
                    self.assertEqual(reader.metric_values("counter", dtype="storage").tolist(), expected)

    def test_numeric_dtypes_match_streaming_writers(self):
        for optimize in (False, True):
            for zipped in (False, True):
                with self.subTest(optimize=optimize, zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                    snapshots = []
                    for cls in (plattli.BulkWriter, plattli.DirectWriter, plattli.CompactingWriter):
                        path = Path(tmp) / cls.__name__
                        kwargs = {"write_threads": 0} if cls is plattli.DirectWriter else {"hotsize": 100} if cls is plattli.CompactingWriter else {}
                        w = cls(path, **kwargs)
                        w.write(narrow=np.uint8(100), widening=np.uint8(200), loss=1.123456789,
                                precise=np.float64(1.123456789), flag=True)
                        w.end_step()
                        w.write(narrow=np.uint8(200), widening=300, loss=np.float64(2.123456789),
                                precise=2.123456789, flag=False)
                        w.end_step()
                        w.finish(optimize=optimize, zip=zipped)
                        with plattli.Reader(path) as r:
                            snapshots.append({name: (r.manifest()[name]["dtype"], r.metric_values(name).tolist())
                                              for name in r.metrics()})
                            self.assertEqual(r.manifest()["loss"]["dtype"], "f64")
                            self.assertEqual(r.manifest()["precise"]["dtype"], "f64")
                            self.assertEqual(r.manifest()["flag"]["dtype"], "jsonl")
                            self.assertEqual(r.metric_values("loss").tolist(),
                                             [float(np.float32(1.123456789)), 2.123456789])
                            self.assertEqual(r.metric_values("precise").tolist(), [1.123456789, 2.123456789])
                    self.assertEqual(snapshots[0], snapshots[1])
                    self.assertEqual(snapshots[0], snapshots[2])

    def test_full_unsigned_range_stays_exact(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize), tempfile.TemporaryDirectory() as tmp:
                w = plattli.BulkWriter(tmp)
                w.write(counter=np.uint64(2**64 - 1))
                w.end_step()
                w.write(counter=7)
                w.end_step()
                w.write(counter=-1)
                w.finish(optimize=optimize, zip=False)
                with plattli.Reader(tmp) as r:
                    self.assertEqual(r.metric_values("counter", dtype="storage").tolist(), [2**64 - 1, 7, -1])

    def test_mixed_numeric_families_raise_before_mutating(self):
        for first, invalid in ((1, 1.5), (1.0, "not numeric")):
            with self.subTest(first=first), tempfile.TemporaryDirectory() as tmp:
                w = plattli.BulkWriter(tmp)
                w.write(value=first)
                w.end_step()
                with self.assertRaises((TypeError, ValueError)):
                    w.write(value=invalid)
                w.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as r:
                    self.assertEqual(r.metric_values("value").tolist(), [first])

    def test_unsigned_bounds_follow_appends_and_rewinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = plattli.BulkWriter(tmp)
            for value in (np.uint64(1), np.uint64(2**64 - 1), 2, 3):
                w.write(counter=value)
                w.end_step()
            self.assertEqual(w._columns["counter"].integer_bounds(), (1, 2**64 - 1))
            # The converter rewinds buffers in-place before continuing a run.
            column = w._columns["counter"]
            del column.i[1:]
            del column.v[1:]
            w.step = 1
            w.write(counter=-1)
            w.finish(optimize=False, zip=False)
            with plattli.Reader(tmp) as r:
                self.assertEqual(r.metric_values("counter").tolist(), [1, -1])
                self.assertEqual(r.manifest()["counter"]["dtype"], "i64")

    def test_unsigned_bounds_invalidate_after_same_dtype_rewind_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = plattli.BulkWriter(tmp)
            for value in (np.uint64(1), np.uint64(2**64 - 1), 2, 3):
                writer.write(counter=value)
                writer.end_step()
            column = writer._columns["counter"]
            del column.i[1:]
            del column.v[1:]
            writer.step = 1
            # Matching dtypes do not query bounds; refill to the old cached length.
            for value in (7, 8, 9):
                writer.write(counter=np.uint64(value))
                writer.end_step()
            writer.write(counter=-1)
            writer.finish(optimize=False, zip=False)
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values("counter").tolist(), [1, 7, 8, 9, -1])

    def test_negative_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            with self.assertRaises(ValueError):
                plattli.BulkWriter(run_root, step=-1)

    def test_basic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.BulkWriter(run_root)
            w.write(loss=1.2, note="ok", meta={"a": 1})
            w.end_step()
            w.write(loss=1.3)
            w.end_step()
            w.finish(zip=False)

            with plattli.Reader(run_root) as reader:
                loss_vals = reader.metric_values("loss")
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
            w = plattli.BulkWriter(run_root)
            w.write(loss=1.0)
            with self.assertRaises(RuntimeError):
                w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            with plattli.Reader(run_root) as reader:
                loss_vals = reader.metric_values("loss")
            loss_idx = np.fromfile(plattli_root / "loss.indices", dtype=np.uint32)
            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertTrue(np.allclose(loss_vals, [1.0, 3.0]))
            self.assertEqual(loss_idx.tolist(), [0, 1])
            self.assertEqual(manifest["loss"]["indices"], "indices")

    def test_optimize_zip_and_tighten(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.BulkWriter(run_root)
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
                self.assertNotIn("loss.u8", zf.namelist())
                self.assertEqual(manifest["loss"]["values"], {"start": 1, "step": 1, "count": 2})
                self.assertNotIn("loss.indices", zf.namelist())
                self.assertIn("note.indices", zf.namelist())
                self.assertIn("note.jsonl", zf.namelist())

    def test_refuses_overwrite_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.BulkWriter(run_root)
            w.write(loss=1.0)
            w.end_step()
            w.finish(zip=True)

            with self.assertRaises(RuntimeError):
                plattli.BulkWriter(run_root)
            w = plattli.BulkWriter(run_root, overwrite=True)
            w.write(loss=2.0)
            w.end_step()
            w.finish(optimize=False, zip=True)
            with plattli.Reader(run_root) as reader:
                self.assertEqual(reader.metric_values("loss").tolist(), [2.0])

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            (run_root / "plattli").mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                plattli.BulkWriter(run_root)

    def test_non_scalar_array_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = plattli.BulkWriter(Path(tmp) / "run")
            with self.assertRaises(ValueError):
                w.write(loss=np.asarray([1.0, 2.0]))

    def test_metrics_is_a_valid_keyword_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.BulkWriter(run_root)
            w.write(metrics=1.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_values("metrics").tolist(), [1.0])

    def test_list_values_stay_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.BulkWriter(run_root)
            w.write(even=[1.0, 2.0], ragged=[1])
            w.end_step()
            w.write(even=[3.0, 4.0], ragged=[2, 3])
            w.end_step()
            w.finish(zip=False)  # optimize=True must not flatten lists into a numeric column.

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["even"]["dtype"], "jsonl")
            self.assertEqual(manifest["ragged"]["dtype"], "jsonl")
            self.assertEqual(_read_jsonl(plattli_root / "even.jsonl"), [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(_read_jsonl(plattli_root / "ragged.jsonl"), [[1], [2, 3]])

    def test_monotonic_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.BulkWriter(run_root)
            w.write(loss=1.0, delta=3.0, broken=1.0, flat=2.0, note="a")
            w.end_step()
            w.write(loss=2.0, delta=2.0, broken=2.0, flat=2.0, note="b")
            w.end_step()
            w.write(loss=2.0, delta=1.0, broken=1.0, flat=2.0)
            w.end_step()
            w.finish(zip=False)

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["loss"]["monotonic"], "inc")
            self.assertEqual(manifest["delta"]["monotonic"], "dec")
            self.assertNotIn("monotonic", manifest["broken"])
            self.assertEqual(manifest["flat"]["monotonic"], "inc")
            self.assertNotIn("monotonic", manifest["note"])

    def test_config_symlink_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            target = Path(tmp) / "config_source.json"
            target.write_text(json.dumps({"seed": 7}), encoding="utf-8")
            w = plattli.BulkWriter(run_root, config=str(target))
            w.write(loss=1.0)
            w.end_step()
            w.finish(zip=True)

            zip_path = _zip_path_for_root(run_root)
            with zipfile.ZipFile(zip_path) as zf:
                config = json.loads(zf.read("config.json"))
            self.assertEqual(config, {"seed": 7})

    def test_finish_merges_config_overrides_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            source = Path(tmp) / "config.json"
            source.write_text(json.dumps({"seed": 7, "model": "small"}), encoding="utf-8")
            w = plattli.BulkWriter(run_root, config=str(source))
            w.write(loss=1.0)
            w.end_step()
            w.finish(config_overrides={"seed": 9, "note": "final"})

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.config(), {"seed": 9, "model": "small", "note": "final"})
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), {"seed": 7, "model": "small"})

    def test_config_auto_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "config.json").write_text(json.dumps({"seed": 3}), encoding="utf-8")
            w = plattli.BulkWriter(run_root)
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
            w = plattli.BulkWriter(run_root, config={"seed": 7})
            w.write(loss=1.0)
            w.end_step()
            w.set_config({"seed": 9, "note": "ok"})
            w.finish(zip=False)

            config = json.loads((plattli_root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config, {"seed": 9, "note": "ok"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
