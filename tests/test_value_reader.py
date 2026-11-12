import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli


class TestValueReader(unittest.TestCase):
    def make_run(self, root, *, zipped=False, indices="inline", dtype="int", start=10**20, step=10**6, count=6):
        root.mkdir()
        if indices == "inline":
            index_spec = [{"start": 0, "stop": count * 2, "step": 2}] if count else []
        else:
            index_spec = indices
            if indices == "indices":
                np.arange(0, count * 2, 2, dtype=np.uint32).tofile(root / "value.indices")
            elif count:
                (root / "value.ranges").write_bytes(struct.pack("<IIQ", 0, 2, count))
            else:
                (root / "value.ranges").write_bytes(b"")
        (root / "plattli.json").write_text(json.dumps({
            "value": {"dtype": dtype, "indices": index_spec,
                      "values": {"start": start, "step": step, "count": count}},
        }))
        if not zipped:
            return root
        archive = root.with_suffix(".plattli")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
            for path in root.iterdir():
                zf.write(path, path.name)
        return archive

    def test_exact_big_integer_sequences_with_all_index_encodings(self):
        for zipped in (False, True):
            for indices in ("inline", "indices", "ranges"):
                with self.subTest(zipped=zipped, indices=indices), tempfile.TemporaryDirectory() as tmp:
                    path = self.make_run(Path(tmp) / "run", zipped=zipped, indices=indices)
                    expected = [10**20 + i * 10**6 for i in range(6)]
                    with plattli.Reader(path) as reader:
                        self.assertEqual(reader.rows("value"), 6)
                        self.assertEqual(reader.metric("value")[0].tolist(), [0, 2, 4, 6, 8, 10])
                        for dtype in ("standard", "storage", object):
                            values = reader.metric_values("value", dtype=dtype)
                            self.assertEqual(values.dtype, object)
                            self.assertEqual(values.tolist(), expected)
                            self.assertTrue(all(type(value) is int for value in values))
                            self.assertEqual(reader.metric("value", idx=-1, dtype=dtype), (10, expected[-1]))
                        for selector, positions in (({"start": 3, "stop": 8}, [2, 3, 4]),
                                                    ({"istart": -3, "istop": -1}, [3, 4]),
                                                    ({"vstart": expected[1] + 1, "vstop": expected[4]}, [2, 3, 4]),
                                                    ({"vstart": expected[-1] + 1}, [])):
                            with self.subTest(selector=selector):
                                steps, values = reader.metric("value", **selector)
                                self.assertEqual(steps.tolist(), [2 * i for i in positions])
                                self.assertEqual(values.tolist(), [expected[i] for i in positions])
                                self.assertEqual(reader.metric_indices("value", **selector).tolist(), steps.tolist())
                                self.assertEqual(reader.metric_values("value", **selector).tolist(), values.tolist())
                        with self.assertRaises(OverflowError):
                            reader.metric_values("value", dtype=np.int64)
                        self.assertEqual(reader.metric_values("value", istop=0, dtype=np.int64).dtype, np.int64)

    def test_constant_and_descending_values_and_conversions(self):
        for zipped in (False, True):
            for dtype, start, step in (("f32", -0.0, 0.0), ("i16", 100, -3), ("int", 10**20, -(10**20))):
                with self.subTest(zipped=zipped, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                    path = self.make_run(Path(tmp) / "run", zipped=zipped, dtype=dtype, start=start, step=step, count=3)
                    with plattli.Reader(path) as reader:
                        expected = [start + i * step for i in range(3)]
                        values = reader.metric_values("value")
                        self.assertEqual(values.tolist(), expected)
                        if dtype == "f32":
                            self.assertEqual(values.dtype, np.float32)
                            self.assertTrue(np.signbit(values).all())
                            self.assertEqual(reader.metric_values("value", vstart=1).tolist(), [])
                            self.assertEqual(reader.metric_values("value", vstop=0).tolist(), expected)
                        elif dtype == "i16":
                            self.assertEqual(values.dtype, np.int64)
                            self.assertEqual(reader.metric_values("value", dtype="storage").dtype, np.int16)
                            self.assertEqual(reader.metric_values("value", vstart=95, vstop=98).tolist(), [97])
                        else:
                            self.assertEqual(reader.metric("value", idx=1, dtype=np.int64), (2, 0))
                            self.assertEqual(reader.metric_values("value", vstart=-1, vstop=1).tolist(), [0])

    def test_single_row_and_value_selection_do_not_expand_whole_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "plattli.json").write_text(json.dumps({
                "value": {"dtype": "int", "indices": [{"start": 0, "step": 1}],
                          "values": {"start": 10**20, "step": 1, "count": 10**9}},
            }))
            with plattli.Reader(root) as reader:
                import plattli.reader as reader_module
                original = reader_module._sequence_array

                def small_slice(spec, offset=0, count=None):
                    self.assertIsNotNone(count)
                    self.assertLessEqual(count, 3)
                    return original(spec, offset=offset, count=count)

                with mock.patch.object(reader_module, "_sequence_array", side_effect=small_slice):
                    self.assertEqual(reader.metric("value", idx=-1), (10**9 - 1, 10**20 + 10**9 - 1))
                    self.assertEqual(reader.metric_values("value", vstart=10**20 + 17, vstop=10**20 + 19).tolist(),
                                     [10**20 + 17, 10**20 + 18, 10**20 + 19])

    def test_hot_rows_and_table_alignment_remain_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp) / "run", count=3)
            manifest = json.loads((root / "plattli.json").read_text())
            manifest["other"] = {"dtype": "u8", "indices": [{"start": 2, "stop": 10, "step": 2}],
                                 "values": {"start": 5, "step": 0, "count": 4}}
            (root / "plattli.json").write_text(json.dumps(manifest))
            (root / "hot.jsonl").write_text(json.dumps({"step": 6, "value": 10**20 + 3 * 10**6}) + "\n")
            with plattli.Reader(root) as reader:
                steps, values = reader.metric("value")
                self.assertEqual(steps.tolist(), [0, 2, 4, 6])
                self.assertEqual(values.tolist(), [10**20 + i * 10**6 for i in range(4)])
                self.assertEqual(reader.rows("value"), 4)
                steps, columns = reader.table(["value", "other"], on="value", vstart=10**20 + 10**6)
                self.assertEqual(steps.tolist(), [2, 4, 6])
                self.assertEqual(columns["value"].tolist(), [10**20 + i * 10**6 for i in range(1, 4)])
                self.assertEqual(columns["other"].tolist(), [5, 5, 5])

    def test_retired_raw_values_retry_inline_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp) / "run", dtype="i64", start=7, step=0, count=3)
            manifest_path = root / "plattli.json"
            manifest = json.loads(manifest_path.read_text())
            sequence = manifest["value"].pop("values")
            np.asarray([7, 7, 7], dtype=np.int64).tofile(root / "value.i64")
            manifest_path.write_text(json.dumps(manifest))
            with plattli.Reader(root) as reader:
                reader.manifest()
                manifest["value"]["values"] = sequence
                manifest_path.write_text(json.dumps(manifest))
                (root / "value.i64").unlink()
                self.assertEqual(reader.metric_values("value").tolist(), [7, 7, 7])

    def test_refresh_picks_up_inline_encoding_switch_and_rewind(self):
        for change in ("raw", "rewind"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = self.make_run(Path(tmp) / "run", dtype="i64", start=7, step=0, count=3)
                manifest_path = root / "plattli.json"
                with plattli.Reader(root) as reader:
                    reader.manifest()
                    manifest = json.loads(manifest_path.read_text())
                    if change == "raw":
                        manifest["value"].pop("values")
                        manifest["value"]["indices"][0]["stop"] = 8
                        expected = [7, 7, 7, 8]
                        np.asarray(expected, dtype=np.int64).tofile(root / "value.i64")
                    else:
                        manifest["value"]["values"]["count"] = 2
                        manifest["value"]["indices"][0]["stop"] = 4
                        expected = [7, 7]
                    manifest_path.write_text(json.dumps(manifest))
                    self.assertEqual(reader.metric_values("value").tolist(), [7, 7, 7])
                    reader.refresh()
                    self.assertEqual(reader.metric_values("value").tolist(), expected)
                    self.assertEqual(reader.metric_indices("value").tolist(), list(range(0, len(expected) * 2, 2)))

    def test_replaced_constant_preserves_zero_sign_with_cached_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp) / "run", dtype="f32", start=-0.0, step=0, count=3)
            with plattli.Reader(root) as reader:
                self.assertTrue(np.signbit(reader.metric_values("value")).all())
                path = root / "plattli.json"
                manifest = json.loads(path.read_text())
                manifest["value"]["values"]["start"] = 0.0
                path.write_text(json.dumps(manifest))
                reader.refresh()
                self.assertFalse(np.signbit(reader.metric_values("value")).any())

    def test_closed_inline_metrics_do_not_probe_manifest_per_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {}
            for index in range(600):
                name = f"metric{index}"
                manifest[name] = {"dtype": "f32", "indices": [{"start": 0, "stop": 10, "step": 1}]}
                if index % 2:
                    manifest[name]["values"] = {"start": 7.0, "step": 0.0, "count": 10}
                else:
                    np.full(10, 7, dtype=np.float32).tofile(root / f"{name}.f32")
            (root / "plattli.json").write_text(json.dumps(manifest))
            original = Path.open

            def open_without_manifest(path, *args, **kwargs):
                self.assertNotEqual(path.name, "plattli.json")
                return original(path, *args, **kwargs)

            with plattli.Reader(root, exclude_hot=True) as reader:
                reader.manifest()
                with mock.patch.object(Path, "open", open_without_manifest), \
                        mock.patch("plattli.reader.json.loads", side_effect=AssertionError("reparsed manifest")), \
                        mock.patch("plattli.reader.json.load", side_effect=AssertionError("reparsed manifest")):
                    for name in manifest:
                        self.assertEqual(reader.metric_values(name).tolist(), [7] * 10)

    def test_live_sequence_sidecar_reads_growth_without_manifest_updates(self):
        for dtype, start, step in (("f32", 0.25, 0.0), ("int", 10**20, 10**6)):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                root = self.make_run(Path(tmp) / "run", indices="indices", dtype=dtype, start=start, step=step, count=3)
                path = root / "plattli.json"
                manifest = json.loads(path.read_text())
                sequence = manifest["value"]["values"]
                sequence.pop("count")
                (root / "value.values").write_text(json.dumps({"dtype": dtype, "values": sequence}))
                manifest["value"]["values"] = "sequence"
                path.write_text(json.dumps(manifest))
                with plattli.Reader(root, exclude_hot=True) as reader:
                    reader.manifest()
                    self.assertEqual(reader.metric_values("value").tolist(), [start + i * step for i in range(3)])
                    with (root / "value.indices").open("ab") as fh:
                        np.asarray([6, 8], dtype=np.uint32).tofile(fh)
                    with mock.patch.object(reader, "_latest_inline_manifest", side_effect=AssertionError("probed manifest")):
                        self.assertEqual(reader.metric_values("value").tolist(), [start + i * step for i in range(5)])
                        self.assertEqual(reader.metric("value", idx=-1), (8, start + 4 * step))

    def test_retired_live_sequence_sidecar_retries_raw_or_inline(self):
        for encoding in ("raw", "inline"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as tmp:
                root = self.make_run(Path(tmp) / "run", indices="indices", dtype="i64", start=7, step=0, count=3)
                path = root / "plattli.json"
                manifest = json.loads(path.read_text())
                sequence = manifest["value"].pop("values")
                (root / "value.values").write_text(json.dumps({"dtype": "i64", "values": sequence}))
                manifest["value"]["values"] = "sequence"
                path.write_text(json.dumps(manifest))
                with plattli.Reader(root, exclude_hot=True) as reader:
                    self.assertEqual(reader.metric_values("value").tolist(), [7, 7, 7])
                    if encoding == "raw":
                        manifest["value"].pop("values")
                        expected = [7, 7, 8]
                        np.asarray(expected, dtype=np.int64).tofile(root / "value.i64")
                    else:
                        manifest["value"]["values"] = {**sequence, "start": 8}
                        expected = [8, 8, 8]
                    path.write_text(json.dumps(manifest))
                    (root / "value.values").unlink()
                    self.assertEqual(reader.metric_values("value").tolist(), expected)

    def test_step_affine_values_follow_irregular_logging_steps(self):
        for indices in ("inline", "ranges", "indices"):
            for stride in (10**6, -(10**6)):
                with self.subTest(indices=indices, stride=stride), tempfile.TemporaryDirectory() as tmp:
                    root = self.make_run(Path(tmp) / "run", indices=indices, count=3)
                    path = root / "plattli.json"
                    manifest = json.loads(path.read_text())
                    manifest["value"]["values"] = {"start": 10**20, "step": stride, "count": 3, "on": "step"}
                    path.write_text(json.dumps(manifest))
                    expected = [10**20 + index * stride for index in (0, 2, 4)]
                    with plattli.Reader(root) as reader:
                        self.assertEqual(reader.metric_values("value").tolist(), expected)
                        self.assertEqual(reader.metric("value", idx=-1), (4, expected[-1]))
                        self.assertEqual(reader.metric_values("value", vstart=expected[1], vstop=expected[1]).tolist(), [expected[1]])
                        self.assertEqual(reader.metric_values("value", istart=1).tolist(), expected[1:])

    def test_live_sidecar_count_is_captured_before_reading_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp) / "run", indices="indices", dtype="i64", start=7, step=1, count=3)
            path = root / "plattli.json"
            manifest = json.loads(path.read_text())
            (root / "value.values").write_text(json.dumps({"dtype": "i64", "values": {"start": 7, "step": 1}}))
            manifest["value"]["values"] = "sequence"
            path.write_text(json.dumps(manifest))
            with plattli.Reader(root, exclude_hot=True) as reader:
                original = reader._read_text

                def append_before_sidecar_read(name):
                    if name == "value.values":
                        with (root / "value.indices").open("ab") as fh:
                            np.asarray([6], dtype=np.uint32).tofile(fh)
                    return original(name)

                with mock.patch.object(reader, "_read_text", side_effect=append_before_sidecar_read):
                    self.assertEqual(reader.metric_values("value").tolist(), [7, 8, 9])

    def test_sidecar_dtype_is_authoritative_for_table_integer_casts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_run(Path(tmp) / "run", indices="indices", dtype="i64", start=7, step=1, count=2)
            path = root / "plattli.json"
            manifest = json.loads(path.read_text())
            manifest["value"]["values"] = "sequence"
            path.write_text(json.dumps(manifest))
            (root / "value.values").write_text(json.dumps({"dtype": "int", "values": {"start": 7, "step": 10**20}}))
            with plattli.Reader(root, exclude_hot=True) as reader:
                self.assertEqual(reader.table(["value"], istop=1, dtype=np.int64)[1]["value"].tolist(), [7])
                with self.assertRaises(OverflowError):
                    reader.table(["value"], dtype=np.int64)
