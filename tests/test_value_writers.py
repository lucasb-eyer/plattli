import copy
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
import plattli.writer as writer_module


def _writer(kind, root, **kwargs):
    if kind == "bulk":
        return plattli.BulkWriter(root, **kwargs)
    if kind == "direct":
        return plattli.DirectWriter(root, write_threads=0, **kwargs)
    return plattli.CompactingWriter(root, hotsize=2 if kind == "cold" else 100, **kwargs)


class TestValueWriters(unittest.TestCase):
    def test_killed_live_promotion_can_resume(self):
        for kind in ("direct", "cold", "hot"):
            for dtype, value in (("u8", np.uint8(7)), ("f32", np.float32(-0.0)), ("i64", 7)):
                with self.subTest(kind=kind, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                    result = subprocess.run([sys.executable, "-c", """
import os
import sys
import numpy as np
import plattli
import plattli.writer as writer_module

writer = plattli.DirectWriter(sys.argv[1], write_threads=0)
value = {'u8': np.uint8(7), 'f32': np.float32(-0.0), 'i64': 7}[sys.argv[2]]
writer.write(value=value)
writer.end_step()
writer_module._write_manifest = lambda *args, **kwargs: os._exit(42)
writer.write(value={'u8': 7, 'f32': np.float64(-0.0), 'i64': 10**20}[sys.argv[2]])
""", tmp, dtype], capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 42, result.stderr)
                    root = Path(tmp) / "plattli"
                    self.assertEqual(json.loads((root / "plattli.json").read_text())["value"]["dtype"], dtype)
                    self.assertNotEqual(json.loads((root / "value.values").read_text())["dtype"], dtype)
                    with _writer(kind, tmp, step=1) as resumed:
                        self.assertEqual(resumed._manifest["value"]["dtype"], dtype)
                        with plattli.Reader(tmp) as reader:
                            self.assertEqual(reader.metric_values("value").tolist(), [value])
                        resumed.write(value=value)
                        resumed.end_step()
                        resumed.finish(optimize=False, zip=False)
                    with plattli.Reader(tmp) as reader:
                        self.assertEqual(reader.metric_indices("value").tolist(), [0, 1])
                        self.assertEqual(reader.metric_values("value").tolist(), [value] * 2)
                        if dtype == "f32":
                            self.assertTrue(np.signbit(reader.metric_values("value")).all())

    def test_failed_live_promotion_restores_sequence_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer("direct", tmp)
            for _ in range(3):
                writer.write(value=np.uint8(7))
                writer.end_step()
            path = Path(tmp) / "plattli/value.values"
            original = path.read_bytes()
            with mock.patch.object(writer_module, "_write_manifest", side_effect=OSError("publication failed")):
                with self.assertRaises(OSError):
                    writer.write(value=7)
            self.assertEqual(path.read_bytes(), original)
            resumed = _writer("direct", tmp, step=3)
            resumed.write(value=np.uint8(7))
            resumed.finish(optimize=False, zip=False)
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values("value").tolist(), [7] * 4)

    def test_resume_other_inline_metrics_remain_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = _writer("direct", tmp)
            for _ in range(3):
                writer.write(a=7, b=8)
                writer.end_step()
            writer.finish(optimize=False, zip=False)
            resumed = _writer("direct", tmp, step=3)
            resumed.write(a=7)
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values("a").tolist(), [7] * 4)
                self.assertEqual(reader.metric_values("b").tolist(), [8] * 3)
            resumed.finish(optimize=False, zip=False)

    def test_failed_batch_does_not_change_sequence_basis(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = _writer(kind, tmp)
                for step in (0, 100, 200):
                    writer.step = step
                    writer.write(counter=10**20 + step, other=1)
                    writer.end_step()
                writer.step = 251
                with self.assertRaises(TypeError):
                    writer.write(counter=10**20 + 251, other="invalid")
                writer.write(counter=10**20 + 300, other=1)
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.metric_values("counter").tolist(), [10**20 + step for step in (0, 100, 200, 300)])

    def test_constants_and_sequences_need_no_value_file(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            for zipped in (False, True):
                for optimize in (False, True):
                    with self.subTest(kind=kind, zipped=zipped, optimize=optimize), tempfile.TemporaryDirectory() as tmp:
                        writer = _writer(kind, tmp)
                        for row, step in enumerate((0, 2, 7, 8, 30)):
                            writer.step = step
                            writer.write(constant=42, increasing=3 + row * 7,
                                         decreasing=10 - row * 3, fraction=np.float32(0.5))
                            writer.end_step()
                        writer.finish(optimize=optimize, zip=zipped)
                        with plattli.Reader(tmp) as reader:
                            for name, start, stride in (("constant", 42, 0), ("increasing", 3, 7),
                                                        ("decreasing", 10, -3), ("fraction", 0.5, 0)):
                                spec = reader.manifest()[name]
                                self.assertEqual(spec["values"], {"start": start, "step": stride, "count": 5})
                                self.assertEqual(reader.metric_indices(name).tolist(), [0, 2, 7, 8, 30])
                                self.assertEqual(reader.metric_values(name).tolist(), [start + row * stride for row in range(5)])
                                if zipped:
                                    with zipfile.ZipFile(Path(tmp) / "metrics.plattli") as archive:
                                        self.assertNotIn(f"{name}.{spec['dtype']}", archive.namelist())
                                else:
                                    self.assertFalse((Path(tmp) / "plattli" / f"{name}.{spec['dtype']}").exists())

    def test_float_changes_use_raw_storage_at_the_second_measurement(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            for optimize in (False, True):
                for dtype, values in ((np.float32, [0.5, 0.75]),
                                      (np.float64, [1.25, 1.5]),
                                      (np.float32, [-0.0, 0.0])):
                    with self.subTest(kind=kind, optimize=optimize, dtype=dtype, values=values), tempfile.TemporaryDirectory() as tmp:
                        writer = _writer(kind, tmp)
                        for value in values:
                            writer.write(value=dtype(value))
                            writer.end_step()
                        if kind == "direct":
                            self.assertNotIn("values", writer._manifest["value"])
                        writer.finish(optimize=optimize, zip=False)
                        with plattli.Reader(tmp) as reader:
                            spec = reader.manifest()["value"]
                            self.assertNotIn("values", spec)
                            self.assertEqual(reader.metric_values("value", dtype="storage").tobytes(),
                                             np.asarray(values, dtype=dtype).tobytes())
                        self.assertFalse((Path(tmp) / "plattli/value.values").exists())

    def test_huge_integer_sequences_are_exact(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            for start, stride in ((10**20, 10**6), (-10**30, -10**8), (10**38, 0)):
                with self.subTest(kind=kind, start=start, stride=stride), tempfile.TemporaryDirectory() as tmp:
                    writer = _writer(kind, tmp)
                    expected = [start + row * stride for row in range(6)]
                    for value in expected:
                        writer.write(counter=value)
                        writer.end_step()
                    writer.finish(optimize=False, zip=False)
                    with plattli.Reader(tmp) as reader:
                        self.assertEqual(reader.manifest()["counter"]["dtype"], "i128")
                        self.assertEqual(reader.manifest()["counter"]["values"],
                                         {"start": start, "step": stride, "count": 6})
                        for dtype in ("standard", "storage", object):
                            values = reader.metric_values("counter", dtype=dtype)
                            self.assertEqual(values.dtype, np.dtype(object))
                            self.assertEqual(values.tolist(), expected)
                            self.assertTrue(all(type(value) is int for value in values))
                        self.assertEqual(reader.metric("counter", idx=-1)[1], expected[-1])
                    self.assertFalse((Path(tmp) / "plattli/counter.i128").exists())

    def test_streaming_sequences_are_readable_before_finish(self):
        for kind in ("direct", "cold", "hot"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                with _writer(kind, tmp) as writer:
                    for row in range(6):
                        writer.write(counter=10**20 + row * 10**6, constant=0.5)
                        writer.end_step()
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.metric_values("counter").tolist(),
                                     [10**20 + row * 10**6 for row in range(6)])
                    self.assertEqual(reader.metric_values("constant").tolist(), [0.5] * 6)
                writer.finish(optimize=False, zip=False)

    def test_irregular_fixed_width_values_fall_back_to_raw(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = _writer(kind, tmp)
                for value in (4, 4, 5, 9, 9, 9):
                    writer.write(value=value)
                    writer.end_step()
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertNotIn("values", reader.manifest()["value"])
                    self.assertEqual(reader.metric_values("value").tolist(), [4, 4, 5, 9, 9, 9])
                    dtype = reader.manifest()["value"]["dtype"]
                self.assertTrue((Path(tmp) / "plattli" / f"value.{dtype}").exists())

    def test_i128_overflow_rejects_whole_batch(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = _writer(kind, tmp)
                for row in range(3):
                    writer.write(counter=10**20 + row * 10**6, other=row)
                    writer.end_step()
                if kind == "cold":
                    writer._wait_for_compaction()

                def snapshot():
                    if kind == "bulk":
                        state = {name: (column.dtype, list(column.i), list(column.v))
                                 for name, column in writer._columns.items()}
                    else:
                        state = (writer._manifest, getattr(writer, "_current_row", {}),
                                 getattr(writer, "_hot_rows", []))
                    files = {str(path.relative_to(tmp)): path.read_bytes()
                             for path in Path(tmp).rglob("*") if path.is_file()}
                    return copy.deepcopy((state, writer._step_metrics)), files

                before = snapshot()
                with self.assertRaises((ValueError, OverflowError)):
                    writer.write(other=99, counter=2**127, new=5)
                self.assertEqual(snapshot(), before)
                writer.write(counter=10**20 + 3 * 10**6, other=3)
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertNotIn("new", reader.metrics())
                    self.assertEqual(reader.metric_values("other").tolist(), [0, 1, 2, 3])

    def test_small_to_huge_integer_promotion_preserves_sequence(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            for expected in ([7, 10**20 + 7, 2 * 10**20 + 7],
                             [2**63 - 4, 2**63 - 2, 2**63, 2**63 + 2]):
                with self.subTest(kind=kind, expected=expected), tempfile.TemporaryDirectory() as tmp:
                    writer = _writer(kind, tmp)
                    for value in expected:
                        writer.write(counter=value)
                        writer.end_step()
                    writer.finish(optimize=False, zip=False)
                    with plattli.Reader(tmp) as reader:
                        self.assertEqual(reader.manifest()["counter"]["dtype"], "i128")
                        self.assertEqual(reader.metric_values("counter").tolist(), expected)

    def test_huge_integer_promotes_irregular_history(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = _writer(kind, tmp)
                for value in (0, 1, 3):
                    writer.write(counter=value)
                    writer.end_step()
                writer.write(counter=10**20 + 2)
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.metric_values("counter").tolist(), [0, 1, 3, 10**20 + 2])

    def test_resume_sequences_and_rewind_to_zero(self):
        for kind in ("direct", "cold", "hot"):
            for zipped in (False, True):
                for cutoff in (0, 3, 6):
                    with self.subTest(kind=kind, zipped=zipped, cutoff=cutoff), tempfile.TemporaryDirectory() as tmp:
                        writer = _writer(kind, tmp)
                        for row in range(6):
                            writer.write(counter=10**20 + row * 11, constant=0.5)
                            writer.end_step()
                        writer.finish(zip=zipped)
                        writer = _writer(kind, tmp, step=cutoff, allow_resume_finalized=zipped)
                        for row in range(cutoff, cutoff + 3):
                            writer.write(counter=(-10**25 + row * 7) if cutoff == 0 else 10**20 + row * 11,
                                         constant=1.5 if cutoff == 0 else 0.5)
                            writer.end_step()
                        writer.finish(optimize=False, zip=False)
                        with plattli.Reader(tmp) as reader:
                            expected = ([-10**25 + row * 7 for row in range(3)] if cutoff == 0 else
                                        [10**20 + row * 11 for row in range(cutoff + 3)])
                            self.assertEqual(reader.metric_values("counter").tolist(), expected)
                            self.assertEqual(reader.metric_values("constant").tolist(),
                                             [1.5 if cutoff == 0 else 0.5] * len(expected))

    def test_compacting_resume_recovers_huge_hot_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with plattli.CompactingWriter(tmp, hotsize=100) as writer:
                for row in range(5):
                    writer.write(counter=10**20 + row * 17)
                    writer.end_step()
            with plattli.CompactingWriter(tmp, step=4, hotsize=2) as resumed:
                resumed.write(counter=10**20 + 4 * 17)
                resumed.end_step()
                resumed.write(counter=10**20 + 5 * 17)
                resumed.finish(optimize=False, zip=False)
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values("counter").tolist(),
                                 [10**20 + row * 17 for row in range(6)])

    def test_constant_negative_zero_preserves_sign(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = _writer(kind, tmp)
                for _ in range(4):
                    writer.write(zero=-0.0)
                    writer.end_step()
                writer.finish(zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertTrue(np.signbit(reader.metric_values("zero")).all())

    def test_direct_resume_after_failed_sequence_head_update(self):
        for start in (10, 10**20):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as tmp:
                writer = _writer("direct", tmp)
                writer.write(value=start)
                writer.end_step()
                with mock.patch.object(writer_module, "_replace_text_checked", side_effect=OSError("interrupted sequence head")):
                    with self.assertRaises(OSError):
                        writer.write(value=start + 1)
                resumed = _writer("direct", tmp, step=2)
                resumed.write(value=start + 2)
                resumed.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.metric_indices("value").tolist(), [0, 2])
                    self.assertEqual(reader.metric_values("value").tolist(), [start, start + 2])
                    self.assertEqual(reader.rows("value"), 2)

    def test_failed_raw_to_huge_sequence_promotion_keeps_raw_values(self):
        for kind in ("direct", "cold", "hot"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                storage = Path(tmp) / "plattli"
                storage.mkdir()
                (storage / "plattli.json").write_text(json.dumps({"counter": {"indices": "indices", "dtype": "i64"}}))
                np.asarray([7], dtype=np.int64).tofile(storage / "counter.i64")
                np.asarray([0], dtype=np.uint32).tofile(storage / "counter.indices")
                writer = _writer(kind, tmp, step=1)
                with mock.patch.object(writer_module, "_replace_text_checked", side_effect=OSError("interrupted manifest")):
                    with self.assertRaises(OSError):
                        writer.write(counter=10**20 + 7)
                spec = json.loads((storage / "plattli.json").read_text())["counter"]
                self.assertEqual(spec["dtype"], "i64")
                self.assertNotIn("values", spec)
                self.assertEqual(np.fromfile(storage / "counter.i64", dtype=np.int64).tolist(), [7])
                resumed = _writer(kind, tmp, step=1)
                resumed.write(counter=10**20 + 7)
                resumed.end_step()
                resumed.write(counter=2 * 10**20 + 7)
                resumed.finish(optimize=False, zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.metric_values("counter").tolist(), [7, 10**20 + 7, 2 * 10**20 + 7])
                self.assertFalse((storage / "counter.i64").exists())

    def test_float_widening_preserves_rounded_raw_history(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = _writer(kind, tmp)
                expected = [float(np.float32(value)) for value in (0.1, 0.2, 0.3)] + [0.4]
                for value in (np.float32(0.1), np.float32(0.2), np.float32(0.3), np.float64(0.4)):
                    writer.write(value=value)
                    writer.end_step()
                writer.finish(zip=False)
                with plattli.Reader(tmp) as reader:
                    self.assertEqual(reader.manifest()["value"]["dtype"], "f64")
                    self.assertNotIn("values", reader.manifest()["value"])
                    self.assertEqual(reader.metric_values("value").tolist(), expected)
                if kind != "bulk":
                    resumed = _writer(kind, tmp, step=2)
                    resumed.write(value=np.float64(0.5))
                    resumed.finish(zip=False)
                    with plattli.Reader(tmp) as reader:
                        self.assertEqual(reader.metric_values("value").tolist(), expected[:2] + [0.5])

    def test_float_constant_widening_keeps_compression_and_zero_sign(self):
        for kind in ("direct", "cold", "hot", "bulk"):
            for value in (0.5, -0.0):
                with self.subTest(kind=kind, value=value), tempfile.TemporaryDirectory() as tmp:
                    writer = _writer(kind, tmp)
                    for dtype in (np.float32, np.float32, np.float32, np.float64):
                        writer.write(value=dtype(value))
                        writer.end_step()
                    writer.finish(zip=False)
                    with plattli.Reader(tmp) as reader:
                        spec = reader.manifest()["value"]
                        self.assertEqual(spec["dtype"], "f64")
                        self.assertEqual(spec["values"], {"start": value, "step": 0.0, "count": 4})
                        self.assertEqual(reader.metric_values("value").tobytes(), np.full(4, value, dtype=np.float64).tobytes())
                    self.assertFalse((Path(tmp) / "plattli/value.f64").exists())

    def test_optimized_resume_can_break_a_fixed_width_sequence(self):
        for kind in ("direct", "cold", "hot"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                writer = _writer(kind, tmp)
                for value in (100, 200, 300):
                    writer.write(value=value)
                    writer.end_step()
                writer.finish()
                resumed = _writer(kind, tmp, step=2, allow_resume_finalized=True)
                resumed.write(value=-1)
                resumed.end_step()
                resumed.finish(zip=False)
                with plattli.Reader(tmp) as reader:
                    spec = reader.manifest()["value"]
                    self.assertNotIn("values", spec)
                    self.assertEqual(reader.metric_values("value").tolist(), [100, 200, -1])
                    self.assertTrue((Path(tmp) / "plattli" / f"value.{spec['dtype']}").exists())

    def test_fixed_width_sequences_handle_large_intermediates(self):
        sequences = [
            [np.int64(-2**63), np.int64(2**63 - 1)],
            [np.int64(2**63 - 1), np.int64(-2**63)],
            [np.int64(-2**63), np.int64(-1), np.int64(2**63 - 2)],
            [np.uint64(0), np.uint64(2**63 - 1), np.uint64(2**64 - 2)],
            [np.uint64(2**64 - 1), np.uint64(0)],
        ]
        for kind in ("direct", "cold", "hot", "bulk"):
            for values in sequences:
                with self.subTest(kind=kind, values=values), tempfile.TemporaryDirectory() as tmp:
                    writer = _writer(kind, tmp)
                    for value in values:
                        writer.write(value=value)
                        writer.end_step()
                    writer.finish(zip=False)
                    with plattli.Reader(tmp) as reader:
                        self.assertIn("values", reader.manifest()["value"])
                        self.assertEqual(reader.metric_values("value", dtype="storage").tolist(),
                                         [int(value) for value in values])
                        self.assertEqual(int(reader.metric("value", idx=-1, dtype="storage")[1]), int(values[-1]))

    def test_value_optimization_failure_keeps_raw_encoding_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "plattli"
            storage.mkdir()
            manifest = {
                "raw": {"dtype": "f32", "indices": "indices"},
                "inline": {"dtype": "i64", "indices": [{"start": 0, "stop": 4, "step": 1}],
                           "values": {"start": 1, "step": 2, "count": 4}},
            }
            (storage / "plattli.json").write_text(json.dumps(manifest))
            np.asarray([0.5] * 4, dtype=np.float32).tofile(storage / "raw.f32")
            np.arange(4, dtype=np.uint32).tofile(storage / "raw.indices")
            before = copy.deepcopy(manifest)
            with mock.patch.object(writer_module, "_write_manifest", side_effect=OSError("interrupted manifest")):
                with self.assertRaises(OSError):
                    writer_module._optimize_values(storage, manifest)
            self.assertEqual(manifest, before)
            self.assertTrue((storage / "raw.f32").exists())
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values("raw").tolist(), [0.5] * 4)
                self.assertNotIn("values", reader.manifest()["raw"])
            writer_module._optimize_values(storage, manifest)
            self.assertFalse((storage / "raw.f32").exists())
            with plattli.Reader(tmp) as reader:
                self.assertEqual(reader.metric_values("raw").tolist(), [0.5] * 4)
                self.assertEqual(reader.metric_values("inline").tolist(), [1, 3, 5, 7])
