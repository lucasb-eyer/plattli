import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
from plattli.writer import DTYPE_TO_NUMPY


class TestReaderDtypes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_run(self, columns, *, zipped=False, hot=None, monotonic=None):
        root = Path(self.tmp.name) / str(len(list(Path(self.tmp.name).iterdir())))
        root.mkdir()
        manifest = {}
        for name, (dtype, steps, values) in columns.items():
            manifest[name] = {"dtype": dtype, "indices": "indices"}
            if monotonic and name in monotonic:
                manifest[name]["monotonic"] = monotonic[name]
            np.asarray(steps, dtype=np.uint32).tofile(root / f"{name}.indices")
            if dtype == "jsonl":
                (root / f"{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
            else:
                np.asarray(values, dtype=DTYPE_TO_NUMPY[dtype]).tofile(root / f"{name}.{dtype}")
        (root / "plattli.json").write_text(json.dumps(manifest))
        if hot is not None:
            (root / "hot.jsonl").write_text("".join(json.dumps(row) + "\n" for row in hot))
        if zipped:
            archive = root.with_suffix(".plattli")
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
                for path in root.iterdir():
                    zf.write(path, path.name)
            return archive
        return root

    def test_standard_and_storage_across_all_numeric_storage_types(self):
        for zipped in (False, True):
            for storage, numpy_type in DTYPE_TO_NUMPY.items():
                with self.subTest(zipped=zipped, storage=storage):
                    path = self.make_run({"value": (storage, [0, 1], [100, 120])}, zipped=zipped)
                    standard = np.int64 if np.dtype(numpy_type).kind in "iu" else numpy_type
                    with plattli.Reader(path) as r:
                        for dtype, expected in (("standard", standard), ("storage", numpy_type), (np.float64, np.float64)):
                            values = r.metric_values("value", dtype=dtype)
                            self.assertEqual(values.dtype, expected)
                            self.assertEqual(values.tolist(), [100, 120])
                            steps, values = r.metric("value", dtype=dtype)
                            self.assertEqual(steps.dtype, np.uint32)
                            self.assertEqual(values.dtype, expected)
                            step, value = r.metric("value", idx=-1, dtype=dtype)
                            self.assertIsInstance(step, np.uint32)
                            self.assertIsInstance(value, expected)
                            self.assertEqual(value, 120)
                            steps, columns = r.table(["value"], dtype=dtype)
                            self.assertEqual(steps.dtype, np.uint32)
                            self.assertEqual(columns["value"].dtype, expected)
                        self.assertEqual(r.metric_indices("value").dtype, np.uint32)
                        self.assertEqual(r.manifest()["value"]["dtype"], storage)
                        if standard == np.int64:
                            self.assertEqual((r.metric_values("value") * 2).tolist(), [200, 240])
                            self.assertEqual((r.metric_values("value") - 300).tolist(), [-200, -180])

    def test_every_selector_converts_selected_values_and_empty_arrays(self):
        for zipped in (False, True):
            for direction in (None, "inc"):
                path = self.make_run(
                    {"value": ("u8", [0, 2, 4], [1, 2, 3])}, zipped=zipped,
                    monotonic={"value": direction} if direction else None,
                )
                with plattli.Reader(path) as r:
                    for selector, expected in (({}, [1, 2, 3]), ({"start": 2, "stop": 2}, [2]),
                                               ({"istart": 1, "istop": 2}, [2]),
                                               ({"vstart": 2, "vstop": 2}, [2]),
                                               ({"istart": 0, "istop": 0}, []),
                                               ({"start": 100}, []), ({"vstart": 100}, [])):
                        with self.subTest(zipped=zipped, direction=direction, selector=selector):
                            for dtype, expected_dtype in (("standard", np.int64), ("storage", np.uint8),
                                                          (np.dtype("float64"), np.float64)):
                                values = r.metric_values("value", dtype=dtype, **selector)
                                self.assertEqual(values.tolist(), expected)
                                self.assertEqual(values.dtype, expected_dtype)
                                steps, values = r.metric("value", dtype=dtype, **selector)
                                self.assertEqual(steps.dtype, np.uint32)
                                self.assertEqual(values.tolist(), expected)
                                self.assertEqual(values.dtype, expected_dtype)
                                steps, columns = r.table(["value"], on="value", dtype=dtype, **selector)
                                self.assertEqual(steps.dtype, np.uint32)
                                self.assertEqual(columns["value"].tolist(), expected)
                                self.assertEqual(columns["value"].dtype, expected_dtype)

    def test_uint64_overflow_checks_happen_after_all_selection(self):
        for zipped in (False, True):
            for direction in (None, "inc"):
                path = self.make_run({
                    "value": ("u64", [0, 1, 2], [1, 2**63 - 1, 2**63]),
                    "sparse": ("u8", [0, 1], [10, 20]),
                }, zipped=zipped, monotonic={"value": direction} if direction else None)
                with plattli.Reader(path) as r:
                    for read in (lambda: r.metric_values("value"), lambda: r.metric("value"),
                                 lambda: r.metric("value", idx=-1), lambda: r.table(["value"])):
                        with self.assertRaisesRegex(OverflowError, "do not fit int64.*value"):
                            read()
                    for selector in ({"stop": 1}, {"istop": 2}, {"vstop": 2**63 - 1}):
                        self.assertEqual(r.metric_values("value", **selector).tolist(), [1, 2**63 - 1])
                        self.assertEqual(r.metric("value", **selector)[1].tolist(), [1, 2**63 - 1])
                    for idx in (slice(0, 2), [0, 1], np.asarray([True, True, False])):
                        self.assertEqual(r.metric("value", idx=idx)[1].tolist(), [1, 2**63 - 1])
                    self.assertIsInstance(r.metric("value", idx=0)[1], np.int64)
                    self.assertIsInstance(r.metric("value", idx=0, istop=2)[1], np.int64)
                    for kwargs in ({}, {"on": "value"}, {"on": "sparse"}, {"istop": 2}):
                        steps, columns = r.table(["value", "sparse"], **kwargs)
                        self.assertEqual(steps.tolist(), [0, 1])
                        self.assertEqual(columns["value"].tolist(), [1, 2**63 - 1])
                    self.assertEqual(r.table(["value"], istop=2)[1]["value"].tolist(), [1, 2**63 - 1])
                    self.assertEqual(r.table(["sparse"], on="value")[1]["sparse"].tolist(), [10, 20])

    def test_full_uint64_explicit_reads_bypass_standard_conversion(self):
        path = self.make_run({"value": ("u64", [0, 1], [2**63, 2**64 - 1])})
        with plattli.Reader(path) as r:
            for dtype in ("storage", np.uint64, object):
                values = r.metric_values("value", dtype=dtype)
                self.assertEqual(values.tolist(), [2**63, 2**64 - 1])
                self.assertEqual(int(r.metric("value", idx=-1, dtype=dtype)[1]), 2**64 - 1)
                self.assertEqual(r.table(["value"], dtype={"value": dtype})[1]["value"].tolist(), values.tolist())
                if dtype is object:
                    self.assertIsInstance(values[0], int)
            self.assertEqual(r.metric_values("value", istop=0).dtype, np.int64)

    def test_explicit_integer_conversion_checks_boundaries_and_fractional_values(self):
        path = self.make_run({
            "signed": ("i64", [0, 1, 2], [-1, 127, 128]),
            "unsigned": ("u64", [0, 1, 2], [2**63 - 1, 2**63, 2**64 - 1]),
            "float": ("f64", list(range(8)), [1, 1.5, np.nan, np.inf, -np.inf, 2**63, 2**64, -2**63]),
        })
        with plattli.Reader(path) as r:
            for name, dtype in (("signed", np.int8), ("signed", np.uint8), ("unsigned", np.int64)):
                with self.subTest(name=name, dtype=dtype), self.assertRaises(OverflowError):
                    r.metric_values(name, dtype=dtype)
            self.assertEqual(r.metric_values("signed", istart=1, istop=2, dtype=np.int8).tolist(), [127])
            self.assertEqual(r.metric_values("unsigned", istop=1, dtype=np.int64).tolist(), [2**63 - 1])
            self.assertEqual(r.metric("float", idx=0, dtype=np.int64)[1], 1)
            for idx in (1, 2, 3, 4):
                with self.subTest(idx=idx), self.assertRaises(ValueError):
                    r.metric("float", idx=idx, dtype=np.int64)
            for idx, dtype in ((5, np.int64), (6, np.uint64), (7, np.uint64)):
                with self.subTest(idx=idx, dtype=dtype), self.assertRaises(OverflowError):
                    r.metric("float", idx=idx, dtype=dtype)
            self.assertEqual(r.metric("float", idx=5, dtype=np.uint64)[1], 2**63)
            self.assertEqual(r.metric("float", idx=7, dtype=np.int64)[1], -2**63)
            self.assertEqual(r.metric("float", idx=1, dtype=np.float32)[1], np.float32(1.5))

    def test_table_per_metric_overrides_and_alignment_precede_explicit_cast(self):
        path = self.make_run({
            "a": ("u16", [0, 1, 2], [1, 256, 3]),
            "b": ("u8", [0, 2], [4, 5]),
            "float": ("f32", [0, 1, 2], [1, 2, 3]),
        })
        with plattli.Reader(path) as r:
            steps, columns = r.table(["a", "b", "float"], dtype={"a": np.uint8, "float": np.float64})
            self.assertEqual(steps.tolist(), [0, 2])
            self.assertEqual(steps.dtype, np.uint32)
            self.assertEqual(columns["a"].dtype, np.uint8)
            self.assertEqual(columns["a"].tolist(), [1, 3])
            self.assertEqual(columns["b"].dtype, np.int64)
            self.assertEqual(columns["float"].dtype, np.float64)
            for values in r.table(["a", "b"], dtype=np.float32)[1].values():
                self.assertEqual(values.dtype, np.float32)
            self.assertEqual(r.table(["a"], istop=0, dtype={"a": np.uint8})[1]["a"].dtype, np.uint8)
            with self.assertRaisesRegex(ValueError, "unrequested table metrics"):
                r.table(["a"], dtype={"typo": np.uint8})

    def test_table_materializes_generator_names_before_mapping_and_retries(self):
        from plattli.reader import _ManifestStale

        path = self.make_run({"a": ("u8", [0, 1], [1, 2]), "b": ("u8", [0, 1], [3, 4])})
        with plattli.Reader(path) as reader:
            original = reader.metric
            calls = []

            def interrupted_metric(name, *args, **kwargs):
                calls.append(name)
                if calls == ["a", "b"]:
                    raise _ManifestStale(name)
                return original(name, *args, **kwargs)

            with mock.patch.object(reader, "metric", side_effect=interrupted_metric):
                steps, columns = reader.table((name for name in ("a", "b")), dtype={"a": np.uint8})
            self.assertEqual(calls, ["a", "b", "a", "b"])
            self.assertEqual(steps.tolist(), [0, 1])
            self.assertEqual(list(columns), ["a", "b"])
            self.assertEqual(columns["a"].dtype, np.uint8)
            self.assertEqual(columns["b"].dtype, np.int64)
            for dtype in ("standard", {}, {"a": np.uint8}):
                for names in ([], (name for name in ())):
                    with self.assertRaisesRegex(ValueError, "needs at least one metric name"):
                        reader.table(names, dtype=dtype)

    def test_table_restarts_all_columns_after_promotion_and_preserves_instrumentation(self):
        path = Path(self.tmp.name) / "table-promotion"
        writer = plattli.DirectWriter(path, write_threads=0)
        writer.write(a=0.5, b=0.5)
        writer.end_step()

        class InstrumentedReader(plattli.Reader):
            def metric(self, name, *args, **kwargs):
                self.calls.append(name)
                steps, values = super().metric(name, *args, **kwargs)
                self.source_bytes += steps.nbytes + values.nbytes
                if self.calls == ["a"]:
                    writer.write(a=np.float64(1.123456789), b=np.float64(2.123456789))
                    writer.end_step()
                return steps, values

        with InstrumentedReader(path) as reader:
            reader.calls = []
            reader.source_bytes = 0
            steps, columns = reader.table((name for name in ("a", "b")))
            self.assertEqual(reader.calls, ["a", "b", "a", "b"])
            self.assertEqual(reader.source_bytes, 56)
            self.assertEqual(steps.tolist(), [0, 1])
            self.assertEqual(columns["a"].dtype, np.float64)
            self.assertEqual(columns["b"].dtype, np.float64)
            self.assertEqual(columns["a"].tolist(), [0.5, 1.123456789])
            self.assertEqual(columns["b"].tolist(), [0.5, 2.123456789])
        writer.finish(optimize=False, zip=False)

    def test_failed_table_restores_retry_ownership(self):
        from plattli.reader import _ManifestStale

        path = self.make_run({"value": ("u8", [0], [1])})
        with plattli.Reader(path) as reader:
            with self.assertRaises(KeyError):
                reader.table(["value", "missing"])
            original = reader._metric_values_full
            attempts = 0

            def changed_once(name):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise _ManifestStale(name)
                return original(name)

            with mock.patch.object(reader, "_metric_values_full", side_effect=changed_once):
                self.assertEqual(reader.metric_values("value").tolist(), [1])
            self.assertEqual(attempts, 2)

    def test_hot_only_and_mixed_columns_follow_same_policy(self):
        path = self.make_run({
            "a": ("u8", [0], [100]), "hot_only": ("u8", [], []),
            "f32": ("f32", [0], [1.5]), "f64": ("f64", [0], [1.5]),
        }, hot=[{"step": 1, "a": 200, "hot_only": 7, "f32": 2.5, "f64": 2.5}])
        with plattli.Reader(path) as r:
            self.assertEqual(r.metric_values("a").tolist(), [100, 200])
            for name in ("a", "hot_only"):
                self.assertEqual(r.metric_values(name).dtype, np.int64)
                self.assertEqual(r.metric_values(name, istart=-1).dtype, np.int64)
                self.assertEqual(r.metric_values(name, dtype="storage").dtype, np.uint8)
                self.assertIsInstance(r.metric(name, idx=-1)[1], np.int64)
            self.assertEqual(r.metric_values("f32").dtype, np.float32)
            self.assertEqual(r.metric_values("f64").dtype, np.float64)
            self.assertEqual(r.table(["a", "hot_only"])[1]["a"].tolist(), [200])
        with plattli.Reader(path, exclude_hot=True) as r:
            self.assertEqual(r.metric_values("a").tolist(), [100])
            self.assertEqual(r.metric_values("hot_only").dtype, np.int64)
            self.assertEqual(r.metric_values("hot_only", dtype="storage").dtype, np.uint8)

    def test_json_and_steps_are_unchanged(self):
        path = self.make_run({"value": ("jsonl", [2**32 - 2, 2**32 - 1], ["a", {"b": 2}])})
        with plattli.Reader(path) as r:
            for dtype in ("standard", "storage", object):
                steps, values = r.metric("value", dtype=dtype)
                self.assertEqual(steps.dtype, np.uint32)
                self.assertEqual(steps.tolist(), [2**32 - 2, 2**32 - 1])
                self.assertEqual(values.dtype, object)
                self.assertEqual(values.tolist(), ["a", {"b": 2}])
                self.assertEqual(r.metric("value", idx=-1, dtype=dtype)[1], {"b": 2})
                self.assertEqual(r.metric("value", idx=1, istop=2, dtype=dtype)[1], {"b": 2})
            with self.assertRaises(TypeError):
                r.metric_values("value", dtype=np.int64)

    def test_invalid_dtype_rejected_even_for_empty_selection(self):
        path = self.make_run({"value": ("u8", [0], [1])})
        with plattli.Reader(path) as r:
            for dtype in ("invalid", None):
                for read in (lambda: r.metric_values("value", istop=0, dtype=dtype),
                             lambda: r.metric("value", istop=0, dtype=dtype),
                             lambda: r.table(["value"], istop=0, dtype=dtype)):
                    with self.assertRaises(TypeError):
                        read()

    def test_scalar_conversion_keeps_bounded_io_and_no_conversion_reuses_arrays(self):
        for zipped in (False, True):
            path = self.make_run({"value": ("u8", list(range(100)), list(range(100)))}, zipped=zipped)
            with plattli.Reader(path) as r:
                with mock.patch.object(r, "_metric_full", side_effect=AssertionError("full read")), \
                     mock.patch.object(r, "_read_value_slice", wraps=r._read_value_slice) as read:
                    value = r.metric("value", idx=50)[1]
                    self.assertIsInstance(value, np.int64)
                    self.assertEqual(value, 50)
                    self.assertEqual(read.call_args.args[-2:], (50, 1))
                raw = np.asarray([1, 2], dtype=np.float32)
                self.assertIs(r._convert_values("value", raw, "standard"), raw)
                self.assertIs(r._convert_values("value", raw, "storage"), raw)
                self.assertIs(r._convert_values("value", raw, np.float32), raw)

    def test_checked_same_width_integer_conversion_reuses_storage(self):
        reader = object.__new__(plattli.Reader)
        for source, target in ((np.uint64, np.int64), (np.int64, np.uint64),
                               (np.uint8, np.int8), (np.int16, np.uint16)):
            values = np.asarray([0, 100], dtype=source)
            result = reader._convert_values("value", values, target)
            self.assertEqual(result.dtype, target)
            self.assertEqual(result.tolist(), [0, 100])
            self.assertTrue(np.shares_memory(result, values))
        values = np.asarray([0, 2**63 - 1], dtype=np.uint64)
        result = reader._convert_values("value", values, "standard")
        self.assertTrue(np.shares_memory(result, values))
        self.assertEqual(result.tolist(), values.tolist())
        for source, target in ((">u8", "<i8"), ("<u8", ">i8"), (">u8", ">i8")):
            values = np.asarray([0, 123456789], dtype=source)
            result = reader._convert_values("value", values, np.dtype(target))
            self.assertEqual(result.dtype, np.dtype(target))
            self.assertEqual(result.tolist(), [0, 123456789])
            standard = reader._convert_values("value", values, "standard")
            self.assertEqual(standard.dtype, np.dtype(np.int64))
            self.assertEqual(standard.tolist(), [0, 123456789])

    def test_live_promotions_refresh_obsolete_dtype_without_losing_cold_or_hot_values(self):
        for kind in ("direct", "cold", "hot"):
            with self.subTest(kind=kind):
                path = Path(self.tmp.name) / kind
                writer = (plattli.DirectWriter(path, write_threads=0) if kind == "direct"
                          else plattli.CompactingWriter(path, hotsize=1 if kind == "cold" else 100))
                for value in (100, 200):
                    writer.write(value=np.uint8(value), precise=float(value))
                    writer.end_step()
                if kind == "cold":
                    writer._wait_for_compaction()
                with plattli.Reader(path) as reader:
                    reader.manifest()
                    if kind != "hot":
                        self.assertEqual(reader.metric_values("value").tolist(), [100, 200])
                    writer.write(value=300, precise=np.float64(1.123456789012345))
                    writer.end_step()
                    self.assertEqual(reader.metric_values("value").tolist(), [100, 200, 300])
                    self.assertEqual(reader.metric_values("value", dtype="storage").dtype, np.int64)
                    precise = reader.metric_values("precise")
                    self.assertEqual(precise.dtype, np.float64)
                    self.assertEqual(precise[-1], 1.123456789012345)
                writer.finish(optimize=False, zip=False)

    def test_promotion_between_manifest_and_hot_snapshot_preserves_float_precision(self):
        path = Path(self.tmp.name) / "hot-race"
        writer = plattli.CompactingWriter(path, hotsize=100)
        writer.write(value=1.0)
        writer.end_step()
        with plattli.Reader(path) as reader:
            original = reader._read_live_jsonl
            promoted = False

            def promote_before_hot_read(path):
                nonlocal promoted
                if not promoted and path.name == "hot.jsonl":
                    promoted = True
                    writer.write(value=np.float64(1.123456789012345))
                    writer.end_step()
                return original(path)

            with mock.patch.object(reader, "_read_live_jsonl", side_effect=promote_before_hot_read):
                values = reader.metric_values("value")
            self.assertTrue(promoted)
            self.assertEqual(values.dtype, np.float64)
            self.assertEqual(values.tolist(), [1.0, 1.123456789012345])
        writer.finish(optimize=False, zip=False)

    def test_retired_hot_rows_restart_empty_index_reads(self):
        reads = {
            "values": lambda r: r.metric_values("value").tolist(),
            "metric": lambda r: r.metric("value")[1].tolist(),
            "scalar": lambda r: int(r.metric("value", idx=-1)[1]),
            "range": lambda r: r.metric_values("value", istart=1, istop=2).tolist(),
            "value_range": lambda r: r.metric_values("value", vstart=150).tolist(),
            "indices": lambda r: r.metric_indices("value").tolist(),
            "rows": lambda r: r.rows("value"),
            "table": lambda r: r.table(["value"])[1]["value"].tolist(),
        }
        for fresh_race in (False, True):
            for promote in (False, True):
                for name, read in reads.items():
                    with self.subTest(fresh_race=fresh_race, promote=promote, read=name):
                        path = Path(self.tmp.name) / f"retired-{fresh_race}-{promote}-{name}"
                        writer = plattli.CompactingWriter(path, hotsize=100)
                        writer.write(value=np.uint8(100))
                        writer.end_step()
                        last = 300 if promote else 200

                        def finalize():
                            writer.write(value=last if promote else np.uint8(last))
                            writer.end_step()
                            writer.finish(optimize=False, zip=False)

                        with plattli.Reader(path) as reader:
                            if fresh_race:
                                original = reader._read_live_jsonl
                                finalized = False

                                def finalize_before_hot_read(path):
                                    nonlocal finalized
                                    if not finalized:
                                        finalized = True
                                        finalize()
                                    return original(path)

                                with mock.patch.object(reader, "_read_live_jsonl", side_effect=finalize_before_hot_read):
                                    result = read(reader)
                            else:
                                self.assertEqual(reader.manifest()["value"]["indices"], [])
                                finalize()
                                result = read(reader)
                            expected = {"scalar": last, "range": [last], "value_range": [last],
                                        "indices": [0, 1], "rows": 2}.get(name, [100, last])
                            self.assertEqual(result, expected)

    def test_retired_metric_is_detected_when_other_hot_metrics_remain(self):
        for promote in (False, True):
            with self.subTest(promote=promote):
                path = Path(self.tmp.name) / f"other-hot-{promote}"
                writer = plattli.CompactingWriter(path, hotsize=1)
                for step in range(3):
                    writer.write(anchor=step)
                    if step == 2:
                        writer.write(value=np.uint8(100))
                    writer.end_step()
                    writer._wait_for_compaction()
                with plattli.Reader(path) as reader:
                    self.assertEqual(reader.manifest()["value"]["indices"], [])
                    self.assertNotEqual(reader.manifest()["anchor"]["indices"], [])
                    last = 300 if promote else 200
                    writer.write(value=last if promote else np.uint8(last), anchor=3)
                    writer.end_step()
                    writer._wait_for_compaction()
                    writer.write(anchor=4)
                    writer.end_step()
                    writer._wait_for_compaction()
                    hot_rows = (path / "plattli/hot.jsonl").read_text()
                    self.assertIn("anchor", hot_rows)
                    self.assertNotIn("value", hot_rows)
                    steps, values = reader.metric("value")
                    self.assertEqual(steps.tolist(), [2, 3])
                    self.assertEqual(values.tolist(), [100, last])
                writer.finish(optimize=False, zip=False)

    def test_compaction_landing_after_columnar_snapshot_keeps_every_row(self):
        reads = {
            "values": ("_columnar_values", lambda r: r.metric_values("value").tolist()),
            "metric": ("_columnar_values", lambda r: r.metric("value")[1].tolist()),
            "indices": ("_columnar_indices", lambda r: r.metric_indices("value").tolist()),
            "rows": ("_columnar_count_and_last_step", lambda r: r.rows("value")),
            "position": ("_metric_count", lambda r: r.metric_values("value", istart=0, istop=3).tolist()),
            "step": ("_metric_count", lambda r: r.metric_values("value", start=0, stop=2).tolist()),
            "value": ("_metric_count", lambda r: r.metric_values("value", vstart=100, vstop=300).tolist()),
            "table": ("_columnar_values", lambda r: r.table(["value"])[1]["value"].tolist()),
        }
        for name, (method, read) in reads.items():
            with self.subTest(read=name):
                path = self.make_run({"value": ("u16", [0], [100])},
                                     hot=[{"step": 2, "value": 300}], monotonic={"value": "inc"})
                compacting = path / "hot.compacting.jsonl"
                compacting.write_text(json.dumps({"step": 1, "value": 200}) + "\n")
                with plattli.Reader(path) as reader:
                    original = getattr(reader, method)
                    landed = False

                    def land_compaction(*args, **kwargs):
                        nonlocal landed
                        result = original(*args, **kwargs)
                        if not landed:
                            landed = True
                            with (path / "value.u16").open("ab") as fh:
                                np.asarray([200], dtype=np.uint16).tofile(fh)
                            with (path / "value.indices").open("ab") as fh:
                                np.asarray([1], dtype=np.uint32).tofile(fh)
                            compacting.unlink()
                        return result

                    with mock.patch.object(reader, method, side_effect=land_compaction):
                        result = read(reader)
                    self.assertTrue(landed)
                    self.assertEqual(result, [0, 1, 2] if name == "indices" else 3 if name == "rows" else [100, 200, 300])

    def test_empty_index_recovery_preserves_explicit_refresh_and_exclude_hot(self):
        path = Path(self.tmp.name) / "refresh-contract"
        writer = plattli.CompactingWriter(path, hotsize=100)
        writer.write(value=np.uint8(100))
        writer.end_step()
        with plattli.Reader(path) as reader, plattli.Reader(path, exclude_hot=True) as cold:
            self.assertEqual(reader.metric_values("value").tolist(), [100])
            self.assertEqual(cold.metric_values("value").tolist(), [])
            writer.write(value=np.uint8(200))
            writer.end_step()
            self.assertEqual(reader.metric_values("value").tolist(), [100])
            reader.refresh()
            self.assertEqual(reader.metric_values("value").tolist(), [100, 200])
            writer.finish(optimize=False, zip=False)
            with mock.patch.object(cold, "_read_live_jsonl", side_effect=AssertionError("hot probe")), \
                 mock.patch.object(cold, "_check_storage_metadata", side_effect=AssertionError("metadata probe")):
                self.assertEqual(cold.metric_values("value").tolist(), [])
                cold.refresh()
                self.assertEqual(cold.metric_values("value").tolist(), [100, 200])

    def test_immutable_zip_does_not_probe_empty_index_metadata(self):
        path = self.make_run({"value": ("u8", [], [])}, zipped=True)
        with plattli.Reader(path) as reader:
            reader.manifest()
            with mock.patch.object(reader, "_check_storage_metadata", side_effect=AssertionError("metadata probe")):
                self.assertEqual(reader.metric_values("value").tolist(), [])
                self.assertEqual(reader.metric_indices("value").tolist(), [])

    def test_cold_file_promotion_races_restart_original_selection(self):
        reads = {
            "values": lambda r: r.metric_values("value"),
            "metric": lambda r: r.metric("value")[1],
            "scalar": lambda r: r.metric("value", idx=1)[1],
            "range": lambda r: r.metric_values("value", istart=1, istop=2),
            "monotonic": lambda r: r.metric_values("value", vstart=150, vstop=250),
            "table": lambda r: r.table(["value"], start=1, stop=1)[1]["value"],
        }
        for operation in ("stat", "open"):
            for name, read in reads.items():
                with self.subTest(operation=operation, read=name):
                    path = Path(self.tmp.name) / f"{operation}-{name}"
                    writer = plattli.DirectWriter(path, write_threads=0)
                    for value in (100, 200):
                        writer.write(value=np.uint8(value))
                        writer.end_step()
                    old_path = path / "plattli/value.u8"
                    original = getattr(Path, operation)
                    promoted = False

                    def promote_before_access(path, *args, **kwargs):
                        nonlocal promoted
                        if path == old_path and not promoted:
                            promoted = True
                            writer.write(value=300)
                            writer.end_step()
                        return original(path, *args, **kwargs)

                    with plattli.Reader(path) as reader:
                        reader.manifest()
                        with mock.patch.object(Path, operation, promote_before_access):
                            if name in ("scalar", "range", "monotonic", "table"):
                                with mock.patch.object(reader, "_metric_full", side_effect=AssertionError("full read")):
                                    values = read(reader)
                            else:
                                values = read(reader)
                        self.assertTrue(promoted)
                        self.assertEqual(values.dtype, np.int64)
                        self.assertEqual(np.asarray(values).tolist(), 200 if name == "scalar" else
                                         [100, 200, 300] if name in ("values", "metric") else [200])
                    writer.finish(optimize=False, zip=False)

    def test_unchanged_hot_snapshot_checks_manifest_once_and_cold_reads_do_not(self):
        path = self.make_run({"value": ("u8", [0], [100])}, hot=[{"step": 1, "value": 200}])
        with plattli.Reader(path) as reader:
            reader.manifest()
            with mock.patch.object(reader, "_read_text", wraps=reader._read_text) as read:
                self.assertEqual(reader.metric_values("value").tolist(), [100, 200])
                self.assertEqual(reader.metric_values("value").tolist(), [100, 200])
                self.assertEqual(read.call_count, 1)
        (path / "hot.jsonl").unlink()
        with plattli.Reader(path) as reader:
            reader.manifest()
            with mock.patch.object(reader, "_read_text", side_effect=AssertionError("unexpected manifest read")):
                self.assertEqual(reader.metric_values("value").tolist(), [100])
                self.assertEqual(reader.metric("value", idx=0)[1], 100)

    def test_repeated_representation_changes_fail_after_bounded_retries(self):
        from plattli.reader import _ManifestStale

        path = self.make_run({"value": ("u8", [0], [100])})
        with plattli.Reader(path) as reader:
            with mock.patch.object(reader, "_metric_values_full", side_effect=_ManifestStale("value")) as read:
                with self.assertRaisesRegex(RuntimeError, "storage metadata kept changing"):
                    reader.metric_values("value")
                self.assertEqual(read.call_count, 3)

    def test_exhausted_retries_invalidate_partial_hot_snapshot_before_next_read(self):
        path = Path(self.tmp.name) / "exhausted-retries"
        writer = plattli.CompactingWriter(path, hotsize=100)
        writer.write(value=np.uint8(100))
        writer.end_step()
        with plattli.Reader(path) as reader:
            original = reader._read_live_jsonl
            promotions = iter((np.int16(300), np.int32(70000), np.int64(2**40)))

            def promote_before_hot_read(path):
                if path.name == "hot.jsonl":
                    value = next(promotions, None)
                    if value is not None:
                        writer.write(value=value)
                        writer.end_step()
                return original(path)

            with mock.patch.object(reader, "_read_live_jsonl", side_effect=promote_before_hot_read):
                with self.assertRaisesRegex(RuntimeError, "storage metadata kept changing"):
                    reader.metric_values("value")
                values = reader.metric_values("value")
            self.assertEqual(values.dtype, np.int64)
            self.assertEqual(values.tolist(), [100, 300, 70000, 2**40])
        writer.finish(optimize=False, zip=False)


if __name__ == "__main__":
    unittest.main()
