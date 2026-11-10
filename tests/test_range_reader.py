import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli


def _fixture(root, zipped=False):
    root.mkdir()
    (root / "plattli.json").write_text(json.dumps({
        "loss": {"indices": "ranges", "dtype": "f32", "monotonic": "inc"},
        "note": {"indices": "ranges", "dtype": "jsonl"},
        "wall": {"indices": [{"start": 0, "stop": 15, "step": 1}], "dtype": "f64", "monotonic": "inc"},
        "sparse": {"indices": "indices", "dtype": "i16"},
        "run_rows": 15,
    }), encoding="utf-8")
    data = struct.pack("<IIQIIQ", 0, 2, 3, 10, 2, 6)
    (root / "loss.ranges").write_bytes(data)
    (root / "note.ranges").write_bytes(data)
    np.arange(6, dtype=np.float32).tofile(root / "loss.f32")
    np.arange(15, dtype=np.float64).tofile(root / "wall.f64")
    np.asarray([2, 10, 14], dtype=np.uint32).tofile(root / "sparse.indices")
    np.asarray([20, 100, 140], dtype=np.int16).tofile(root / "sparse.i16")
    (root / "note.jsonl").write_text("".join(json.dumps(f"n{i}") + "\n" for i in range(6)), encoding="utf-8")
    if not zipped:
        return root
    archive = root.with_suffix(".plattli")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in root.iterdir():
            zf.write(path, path.name)
    return archive


class TestRangeReader(unittest.TestCase):
    def test_ranges_read_and_select_in_directories_and_archives(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                with plattli.Reader(_fixture(Path(tmp) / "run", zipped)) as reader:
                    self.assertEqual(reader.rows("loss"), 6)
                    self.assertEqual(reader.approx_max_rows(), 15)
                    self.assertEqual(reader.metric_indices("loss").tolist(), [0, 2, 4, 10, 12, 14])
                    self.assertEqual(reader.metric_values("note").tolist(), [f"n{i}" for i in range(6)])
                    self.assertEqual(reader.metric("loss", idx=-1), (14, 5.0))
                    self.assertEqual(reader.metric("loss", idx=3), (10, 3.0))
                    for selector, indices, values in (
                        ({"start": 3, "stop": 12}, [4, 10, 12], [2, 3, 4]),
                        ({"start": 3.5, "stop": 11}, [4, 10], [2, 3]),
                        ({"start": 10}, [10, 12, 14], [3, 4, 5]),
                        ({"stop": 4}, [0, 2, 4], [0, 1, 2]),
                        ({"istart": 1, "istop": -1}, [2, 4, 10, 12], [1, 2, 3, 4]),
                        ({"istart": -2}, [12, 14], [4, 5]),
                        ({"vstart": 2, "vstop": 4}, [4, 10, 12], [2, 3, 4]),
                        ({"start": 5, "stop": 9}, [], []),
                        ({"start": 20, "stop": 1}, [], []),
                        ({"istart": 10}, [], []),
                    ):
                        with self.subTest(selector=selector):
                            selected, actual = reader.metric("loss", **selector)
                            self.assertEqual(selected.tolist(), indices)
                            self.assertEqual(actual.tolist(), values)
                            self.assertEqual(reader.metric_indices("loss", **selector).tolist(), indices)
                            self.assertEqual(reader.metric_values("loss", **selector).tolist(), values)
                            self.assertEqual(selected.dtype, np.uint32)
                            self.assertEqual(actual.dtype, np.float32)

    def test_table_aligns_ranges_inline_and_explicit(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                with plattli.Reader(_fixture(Path(tmp) / "run", zipped)) as reader:
                    steps, columns = reader.table(["wall", "loss", "sparse", "note"])
                    self.assertEqual(steps.tolist(), [2, 10, 14])
                    self.assertEqual(columns["loss"].tolist(), [1, 3, 5])
                    self.assertEqual(columns["sparse"].tolist(), [20, 100, 140])
                    self.assertEqual(columns["sparse"].dtype, np.int64)
                    self.assertEqual(columns["note"].tolist(), ["n1", "n3", "n5"])
                    steps, columns = reader.table(["loss", "sparse"], on="wall", vstart=2, vstop=12, dtype={"sparse": "storage"})
                    self.assertEqual(steps.tolist(), [2, 10])
                    self.assertEqual(columns["sparse"].dtype, np.int16)
                    steps, columns = reader.table(["wall", "note"], on="loss", istart=1, istop=4)
                    self.assertEqual(steps.tolist(), [2, 4, 10])
                    self.assertEqual(columns["note"].tolist(), ["n1", "n2", "n3"])

    def test_listing_does_not_read_range_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            with plattli.Reader(_fixture(Path(tmp) / "run")) as reader:
                original_open = Path.open

                def open_metadata(path, *args, **kwargs):
                    if path.suffix == ".ranges":
                        raise AssertionError("listing opened a range sidecar")
                    return original_open(path, *args, **kwargs)

                with mock.patch.object(Path, "open", open_metadata):
                    self.assertEqual(reader.metrics(), ["loss", "note", "sparse", "wall"])
                    self.assertEqual(reader.manifest()["loss"]["indices"], "ranges")
                    self.assertEqual(reader.approx_max_rows(), 15)

    def test_open_tail_and_hot_rows_use_the_same_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
            (root / "loss.ranges").write_bytes(struct.pack("<IIQII", 0, 2, 3, 10, 2))
            np.arange(4, dtype=np.float32).tofile(root / "loss.f32")
            (root / "hot.jsonl").write_text("".join(json.dumps({"step": step, "loss": value}) + "\n" for step, value in ((10, 3.0), (12, 4.0), (14, 5.0))), encoding="utf-8")
            with plattli.Reader(root) as reader:
                self.assertEqual(reader.rows("loss"), 6)
                steps, values = reader.metric("loss")
                self.assertEqual(steps.tolist(), [0, 2, 4, 10, 12, 14])
                self.assertEqual(values.tolist(), [0, 1, 2, 3, 4, 5])
                steps, columns = reader.table(["loss"], on="loss", istart=-3)
                self.assertEqual(steps.tolist(), [10, 12, 14])
                self.assertEqual(columns["loss"].tolist(), [3, 4, 5])
            with plattli.Reader(root, exclude_hot=True) as reader:
                self.assertEqual(reader.rows("loss"), 4)
                self.assertEqual(reader.metric_indices("loss").tolist(), [0, 2, 4, 10])

    def test_every_partial_index_append_keeps_published_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
            np.arange(3, dtype=np.float32).tofile(root / "loss.f32")
            old = struct.pack("<II", 0, 1)
            addition = struct.pack("<QIIQII", 5, 10, 1, 7, 20, 1)
            for length in range(len(addition) + 1):
                with self.subTest(length=length):
                    (root / "loss.ranges").write_bytes(old + addition[:length])
                    with plattli.Reader(root) as reader:
                        self.assertEqual(reader.metric_indices("loss").tolist(), [0, 1, 2])
                        self.assertEqual(reader.metric_values("loss").tolist(), [0, 1, 2])
                        self.assertEqual(reader.rows("loss"), 3)

    def test_closed_ranges_ahead_of_values_are_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
            (root / "loss.ranges").write_bytes(struct.pack("<IIQIIQ", 0, 2, 3, 10, 2, 6))
            for count in (0, 1, 2, 3, 4, 5):
                np.arange(count, dtype=np.float32).tofile(root / "loss.f32")
                with self.subTest(count=count), plattli.Reader(root) as reader:
                    self.assertEqual(reader.rows("loss"), count)
                    self.assertEqual(reader.metric_indices("loss").tolist(), [0, 2, 4, 10, 12, 14][:count])

    def test_live_read_captures_values_before_opening_range_file(self):
        for operation in ("metric", "metric_indices", "metric_values", "rows"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
                (root / "loss.ranges").write_bytes(struct.pack("<II", 0, 1))
                np.arange(3, dtype=np.float32).tofile(root / "loss.f32")
                with plattli.Reader(root) as reader:
                    original_count = reader._values_count
                    captured = False

                    def capture(name, spec):
                        nonlocal captured
                        captured = True
                        return original_count(name, spec)

                    original_open = Path.open

                    def ordered_open(path, *args, **kwargs):
                        if path.suffix == ".ranges":
                            self.assertTrue(captured, "range metadata preceded the value count")
                        return original_open(path, *args, **kwargs)

                    with mock.patch.object(reader, "_values_count", capture), mock.patch.object(Path, "open", ordered_open):
                        getattr(reader, operation)("loss")

    def test_open_tail_is_fresh_after_a_new_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
            path = root / "loss.ranges"
            path.write_bytes(struct.pack("<II", 0, 1))
            np.arange(3, dtype=np.float32).tofile(root / "loss.f32")
            with plattli.Reader(root) as reader:
                self.assertEqual(reader.metric_indices("loss").tolist(), [0, 1, 2])
                with path.open("ab") as fh:
                    fh.write(struct.pack("<QII", 3, 10, 1))
                np.arange(5, dtype=np.float32).tofile(root / "loss.f32")
                self.assertEqual(reader.metric_indices("loss").tolist(), [0, 1, 2, 10, 11])
                reader.refresh()
                self.assertEqual(reader.rows("loss"), 5)

    def test_compaction_during_range_read_keeps_captured_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
            path = root / "loss.ranges"
            path.write_bytes(struct.pack("<II", 0, 1))
            np.arange(3, dtype=np.float32).tofile(root / "loss.f32")
            with plattli.Reader(root, exclude_hot=True) as reader:
                original_read = reader._read_bytes
                compacted = False

                def compact_after_read(name):
                    nonlocal compacted
                    data = original_read(name)
                    if name == "loss.ranges" and not compacted:
                        compacted = True
                        with path.open("ab") as fh:
                            fh.write(struct.pack("<QII", 3, 10, 1))
                        np.arange(5, dtype=np.float32).tofile(root / "loss.f32")
                    return data

                with mock.patch.object(reader, "_read_bytes", compact_after_read):
                    steps, values = reader.metric("loss")
                self.assertTrue(compacted)
                self.assertEqual(steps.tolist(), [0, 1, 2])
                self.assertEqual(values.tolist(), [0, 1, 2])
                self.assertEqual(reader.metric_indices("loss").tolist(), [0, 1, 2, 10, 11])

    def test_approximate_count_respects_probe_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({name: {"dtype": "f32", "indices": "ranges"} for name in ("a", "b", "c")}), encoding="utf-8")
            for name, count in (("a", 3), ("b", 5), ("c", 7)):
                (root / f"{name}.ranges").write_bytes(struct.pack("<II", 0, 1))
                np.arange(count, dtype=np.float32).tofile(root / f"{name}.f32")
            with plattli.Reader(root) as reader:
                with mock.patch.object(reader, "_probe_metric_count", wraps=reader._probe_metric_count) as probe:
                    self.assertEqual(reader.approx_max_rows(nprobes=0), 0)
                    probe.assert_not_called()
                    self.assertIn(reader.approx_max_rows(nprobes=1), (3, 5, 7))
                    self.assertEqual(probe.call_count, 1)
                self.assertEqual(reader.approx_max_rows(nprobes=3), 7)

    def test_cached_inline_open_tail_does_not_extrapolate_after_a_gap(self):
        for representation in ("inline", "ranges"):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = root / "plattli.json"
                manifest.write_text(json.dumps({"loss": {"dtype": "f32", "indices": [{"start": 0, "step": 1}]}}), encoding="utf-8")
                np.arange(3, dtype=np.float32).tofile(root / "loss.f32")
                with plattli.Reader(root, exclude_hot=True) as reader:
                    reader.manifest()
                    indices = [{"start": 0, "stop": 3, "step": 1}, {"start": 10, "step": 1}]
                    if representation == "ranges":
                        indices = "ranges"
                        (root / "loss.ranges").write_bytes(struct.pack("<IIQII", 0, 1, 3, 10, 1))
                    manifest.write_text(json.dumps({"loss": {"dtype": "f32", "indices": indices}}), encoding="utf-8")
                    np.arange(4, dtype=np.float32).tofile(root / "loss.f32")
                    steps, values = reader.metric("loss")
                    self.assertEqual(steps.tolist(), [0, 1, 2, 10])
                    self.assertEqual(values.tolist(), [0, 1, 2, 3])
                    self.assertEqual(reader.rows("loss"), 4)

    def test_inline_open_promotion_during_value_count_retries(self):
        for representation in ("inline", "ranges"):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = root / "plattli.json"
                manifest.write_text(json.dumps({"loss": {"dtype": "f32", "indices": [{"start": 0, "step": 1}]}}), encoding="utf-8")
                np.arange(3, dtype=np.float32).tofile(root / "loss.f32")
                with plattli.Reader(root, exclude_hot=True) as reader:
                    original_count = reader._values_count
                    promoted = False

                    def promote_before_count(name, spec):
                        nonlocal promoted
                        if not promoted:
                            promoted = True
                            indices = [{"start": 0, "stop": 3, "step": 1}, {"start": 10, "step": 1}]
                            if representation == "ranges":
                                indices = "ranges"
                                (root / "loss.ranges").write_bytes(struct.pack("<IIQII", 0, 1, 3, 10, 1))
                            manifest.write_text(json.dumps({"loss": {"dtype": "f32", "indices": indices}}), encoding="utf-8")
                            np.arange(4, dtype=np.float32).tofile(root / "loss.f32")
                        return original_count(name, spec)

                    with mock.patch.object(reader, "_values_count", promote_before_count):
                        steps, values = reader.metric("loss")
                    self.assertTrue(promoted)
                    self.assertEqual(steps.tolist(), [0, 1, 2, 10])
                    self.assertEqual(values.tolist(), [0, 1, 2, 3])

    def test_many_inline_open_columns_share_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f"metric{i}" for i in range(64)]
            (root / "plattli.json").write_text(json.dumps({name: {"dtype": "f32", "indices": [{"start": 0, "step": 1}]} for name in names}), encoding="utf-8")
            for name in names:
                np.arange(10, dtype=np.float32).tofile(root / f"{name}.f32")
            original_loads = json.loads
            original_load = json.load
            with plattli.Reader(root, exclude_hot=True) as reader:
                with mock.patch("plattli.reader.json.loads", wraps=original_loads) as loads, mock.patch("plattli.reader.json.load", wraps=original_load) as load:
                    steps, columns = reader.table(names)
                self.assertGreater(loads.call_count + load.call_count, 0)
                self.assertLessEqual(loads.call_count + load.call_count, 3)
                self.assertEqual(steps.tolist(), list(range(10)))
                self.assertEqual(list(columns), names)
                for values in columns.values():
                    self.assertEqual(values.tolist(), list(range(10)))

    def test_missing_retired_ranges_retries_new_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _fixture(root)
            with plattli.Reader(root) as reader:
                reader.manifest()
                manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))
                manifest["loss"]["indices"] = "indices"
                (root / "plattli.json").write_text(json.dumps(manifest), encoding="utf-8")
                np.asarray([0, 2, 4, 10, 12, 14], dtype=np.uint32).tofile(root / "loss.indices")
                (root / "loss.ranges").unlink()
                self.assertEqual(reader.metric_indices("loss").tolist(), [0, 2, 4, 10, 12, 14])

    def test_explicit_retirement_after_count_retries_range_encoding(self):
        for operation in ("metric", "metric_indices"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "run"
                _fixture(root)
                with plattli.Reader(root, exclude_hot=True) as reader:
                    original_count = reader._indices_count_and_last
                    switched = False

                    def retire_after_count(name, spec):
                        nonlocal switched
                        result = original_count(name, spec)
                        if name == "sparse" and spec == "indices" and not switched:
                            switched = True
                            (root / "sparse.ranges").write_bytes(struct.pack("<IIQIIQ", 2, 8, 2, 14, 1, 3))
                            manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))
                            manifest["sparse"]["indices"] = "ranges"
                            (root / "plattli.json").write_text(json.dumps(manifest), encoding="utf-8")
                            (root / "sparse.indices").unlink()
                        return result

                    with mock.patch.object(reader, "_indices_count_and_last", retire_after_count):
                        result = getattr(reader, operation)("sparse")
                    self.assertTrue(switched)
                    if operation == "metric":
                        steps, values = result
                        self.assertEqual(values.tolist(), [20, 100, 140])
                    else:
                        steps = result
                    self.assertEqual(steps.tolist(), [2, 10, 14])

    def test_last_uint32_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
            (root / "loss.ranges").write_bytes(struct.pack("<IIQ", 2**32 - 2, 1, 2))
            np.asarray([10, 20], dtype=np.float32).tofile(root / "loss.f32")
            with plattli.Reader(root) as reader:
                self.assertEqual(reader.metric_indices("loss").tolist(), [2**32 - 2, 2**32 - 1])
                self.assertEqual(reader.metric("loss", idx=-1), (2**32 - 1, 20))
                self.assertEqual(reader.metric_values("loss", start=2**32 - 1).tolist(), [20])

    def test_table_retries_all_columns_after_range_retirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _fixture(root)
            with plattli.Reader(root) as reader:
                reader.manifest()
                manifest = json.loads((root / "plattli.json").read_text(encoding="utf-8"))
                manifest["loss"]["indices"] = "indices"
                (root / "plattli.json").write_text(json.dumps(manifest), encoding="utf-8")
                np.asarray([0, 2, 4, 10, 12, 14], dtype=np.uint32).tofile(root / "loss.indices")
                (root / "loss.ranges").unlink()
                with mock.patch.object(reader, "metric", wraps=reader.metric) as metric:
                    steps, columns = reader.table(["wall", "loss"])
                self.assertEqual([call.args[0] for call in metric.call_args_list], ["wall", "loss", "wall", "loss"])
                self.assertEqual(steps.tolist(), [0, 2, 4, 10, 12, 14])
                self.assertEqual(columns["wall"].tolist(), [0, 2, 4, 10, 12, 14])
                self.assertEqual(columns["loss"].tolist(), [0, 1, 2, 3, 4, 5])

    def test_large_range_selection_reads_one_value_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
            (root / "loss.ranges").write_bytes(b"".join(struct.pack("<IIQ", 51 * i, 1, 50 * (i + 1)) for i in range(52_000)))
            np.arange(52_000 * 50, dtype=np.float32).tofile(root / "loss.f32")
            with plattli.Reader(root) as reader:
                with mock.patch.object(reader, "_metric_full", side_effect=AssertionError("expanded full metric")), mock.patch.object(reader, "_read_value_slice", wraps=reader._read_value_slice) as reads:
                    steps, values = reader.metric("loss", start=51 * 40_000 + 45, stop=51 * 40_001 + 2)
                self.assertEqual(steps.tolist(), list(range(51 * 40_000 + 45, 51 * 40_000 + 50)) + list(range(51 * 40_001, 51 * 40_001 + 3)))
                self.assertEqual(values.tolist(), list(range(50 * 40_000 + 45, 50 * 40_001 + 3)))
                self.assertEqual(reads.call_count, 1)

    def test_random_ranges_match_expanded_indices(self):
        random = np.random.default_rng(42)
        for case in range(20):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "plattli.json").write_text(json.dumps({"loss": {"dtype": "f32", "indices": "ranges"}}), encoding="utf-8")
                pieces = []
                payload = bytearray()
                start = 2**32 - 100_000 if case % 2 else 0
                end = 0
                for _ in range(int(random.integers(1, 30))):
                    count = int(random.integers(1, 30))
                    stride = int(random.choice([1, 2, 7]))
                    piece = start + np.arange(count, dtype=np.uint32) * stride
                    pieces.append(piece)
                    end += count
                    payload.extend(struct.pack("<IIQ", start, stride, end))
                    start = int(piece[-1]) + int(random.integers(1, 9))
                steps = np.concatenate(pieces)
                values = np.arange(len(steps), dtype=np.float32)
                values.tofile(root / "loss.f32")
                if case % 3:
                    del payload[-8:]  # Exercise the virtual finite record for an open head.
                (root / "loss.ranges").write_bytes(payload)
                with plattli.Reader(root) as reader:
                    actual_steps, actual_values = reader.metric("loss")
                    np.testing.assert_array_equal(actual_steps, steps)
                    np.testing.assert_array_equal(actual_values, values)
                    for first, last in ((None, None), (len(steps) // 3, len(steps) * 2 // 3), (-5, None), (None, -2), (-len(steps) - 5, len(steps) + 7), (4, 4), (len(steps) + 3, None), (7, -1000)):
                        actual_steps, actual_values = reader.metric("loss", istart=first, istop=last)
                        np.testing.assert_array_equal(actual_steps, steps[first:last])
                        np.testing.assert_array_equal(actual_values, values[first:last])
                    for first, last in ((int(steps[0]) - 0.5, int(steps[-1]) + 0.5), (int(steps[len(steps) // 4]) + 0.1, int(steps[len(steps) * 3 // 4]) - 0.1), (int(steps[-1]) + 1, None), (None, int(steps[0]) - 1)):
                        keep = np.ones(len(steps), dtype=bool)
                        if first is not None:
                            keep &= steps >= first
                        if last is not None:
                            keep &= steps <= last
                        actual_steps, actual_values = reader.metric("loss", start=first, stop=last)
                        np.testing.assert_array_equal(actual_steps, steps[keep])
                        np.testing.assert_array_equal(actual_values, values[keep])

    def test_position_cuts_at_uint32_row_capacity(self):
        reader = object.__new__(plattli.Reader)
        for records, bounds, expected in (
            ([(0, 1, 2**32)], (2**32 - 2, 2**32), [2**32 - 2, 2**32 - 1]),
            ([(0, 2**32 - 1, 2)], (1, 2), [2**32 - 1]),
            ([(0, 1, 2**32 - 1), (2**32 - 1, 1, 2**32)], (2**32 - 2, 2**32), [2**32 - 2, 2**32 - 1]),
        ):
            with self.subTest(records=records):
                actual = reader._range_indices_chunks(np.asarray(records, dtype=[("start", "<u4"), ("step", "<u4"), ("end_row", "<u8")]), [bounds])
                self.assertEqual(actual.tolist(), expected)
                self.assertEqual(actual.dtype, np.uint32)


if __name__ == "__main__":
    unittest.main()
