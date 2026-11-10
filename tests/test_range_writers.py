import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
import plattli.writer as writer_module
from plattli._ranges import _ranges_parse, _ranges_encode
from plattli._indices import _piecewise_segments, _segments_truncate, _segments_to_array


def _fixture(root):
    storage = root / "plattli"
    storage.mkdir()
    segments = [{"start": i * 100, "stop": i * 100 + 50, "step": 1} for i in range(33)]
    steps = [step for segment in segments for step in range(segment["start"], segment["stop"])]
    (storage / "plattli.json").write_text(json.dumps({
        "loss": {"indices": segments, "dtype": "f32", "monotonic": "inc"},
    }), encoding="utf-8")
    np.asarray(steps, dtype=np.float32).tofile(storage / "loss.f32")
    return steps


class TestRangeWriters(unittest.TestCase):
    def test_truncate_segment_overhang(self):
        steps = np.asarray([3, 40, 45, 50, 55, 60, 65, 70], dtype=np.uint32)
        segments = _piecewise_segments(steps)
        for cutoff in range(80):
            with self.subTest(cutoff=cutoff):
                kept, count = _segments_truncate(segments, cutoff)
                np.testing.assert_array_equal(_segments_to_array(kept), steps[steps < cutoff])
                self.assertEqual(count, sum(steps < cutoff))

    def test_resume_preserves_steps_in_segment_overhang(self):
        steps = np.asarray([3, 40, 45, 50, 55, 60, 65, 70], dtype=np.uint32)
        for representation in ("inline", "ranges", "indices"):
            for kind in ("direct", "compact"):
                with self.subTest(representation=representation, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    storage = root / "plattli"
                    storage.mkdir()
                    segments = _piecewise_segments(steps)
                    (storage / "plattli.json").write_text(json.dumps({
                        "loss": {"indices": segments if representation == "inline" else representation, "dtype": "f32"},
                        "run_rows": len(steps),
                    }))
                    steps.astype(np.float32).tofile(storage / "loss.f32")
                    if representation == "ranges":
                        (storage / "loss.ranges").write_bytes(_ranges_encode(segments))
                    elif representation == "indices":
                        steps.tofile(storage / "loss.indices")
                    writer = plattli.DirectWriter(root, step=75, write_threads=0, allow_resume_finalized=True) if kind == "direct" else plattli.CompactingWriter(root, step=75, hotsize=50, allow_resume_finalized=True)
                    writer.finish(optimize=False, zip=False)
                    with plattli.Reader(root) as reader:
                        np.testing.assert_array_equal(reader.metric_indices("loss"), steps)
                        np.testing.assert_array_equal(reader.metric_values("loss"), steps)

    def test_resume_open_tail_continues_twenty_rows_without_index_writes(self):
        for dtype in ("f32", "jsonl"):
            for recover_hot in (False, True):
                with self.subTest(dtype=dtype, recover_hot=recover_hot), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    steps = _fixture(root)
                    storage = root / "plattli"
                    manifest_path = storage / "plattli.json"
                    manifest = json.loads(manifest_path.read_text())
                    data = _ranges_encode(manifest["loss"]["indices"], open_tail=True)
                    manifest["loss"].update(indices="ranges", dtype=dtype)
                    manifest_path.write_text(json.dumps(manifest))
                    (storage / "loss.ranges").write_bytes(data)
                    if dtype == "jsonl":
                        (storage / "loss.f32").unlink()
                        (storage / "loss.jsonl").write_text("".join(f"{step}\n" for step in steps))
                    added = list(range(steps[-1] + 1, steps[-1] + 21))
                    if recover_hot:
                        (storage / "hot.compacting.jsonl").write_text("".join(
                            json.dumps({"step": step, "loss": float(step)}) + "\n" for step in added))
                    original_open = Path.open
                    writes = []

                    def track(path, mode="r", *args, **kwargs):
                        if path.name in ("loss.ranges", "loss.ranges.tmp") and any(flag in mode for flag in "wa+"):
                            writes.append(path.name)
                        return original_open(path, mode, *args, **kwargs)

                    with mock.patch.object(Path, "open", track):
                        writer = plattli.CompactingWriter(root, step=added[-1] + 1 if recover_hot else added[0], hotsize=20)
                        if not recover_hot:
                            for step in added:
                                writer.write(loss=float(step))
                                writer.end_step()
                        writer.write(flush=True)
                        writer._compact_future.result()
                    try:
                        self.assertEqual(writer._manifest["loss"]["indices"], "ranges")
                        self.assertEqual(writes, [])
                        self.assertEqual((storage / "loss.ranges").read_bytes(), data)
                        with plattli.Reader(root) as reader:
                            np.testing.assert_array_equal(reader.metric_indices("loss"), steps + added)
                            np.testing.assert_array_equal(reader.metric_values("loss"), steps + added)
                    finally:
                        writer.finish(optimize=False, zip=False)

    def test_resume_skips_unchanged_ranges_but_truncates_extra_values(self):
        for dtype in ("f32", "jsonl"):
            for extra in (False, True):
                with self.subTest(dtype=dtype, extra=extra), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    manifest = {"loss": {"indices": "ranges", "dtype": dtype}}
                    (root / "plattli.json").write_text(json.dumps(manifest))
                    data = _ranges_encode([{"start": 0, "stop": 3, "step": 1}], open_tail=not extra)
                    (root / "loss.ranges").write_bytes(data)
                    path = root / f"loss.{dtype}"
                    if dtype == "f32":
                        np.arange(3 + extra, dtype=np.float32).tofile(path)
                    else:
                        path.write_text("".join(f"{i}\n" for i in range(3 + extra)))
                    original_open = Path.open
                    writes = []

                    def track(path, mode="r", *args, **kwargs):
                        if path.name.startswith("loss.") and any(flag in mode for flag in "wa+"):
                            writes.append(path.name)
                        return original_open(path, mode, *args, **kwargs)

                    with mock.patch.object(Path, "open", track):
                        writer_module._truncate_to_step(root, manifest, 100, allow_missing=False)
                    self.assertEqual(writes, [path.name] if extra else [])
                    self.assertEqual((root / "loss.ranges").read_bytes(), data)
                    with plattli.Reader(root) as reader:
                        self.assertEqual(reader.metric_values("loss").tolist(), [0, 1, 2])
                    self.assertEqual(path.stat().st_size, 12 if dtype == "f32" else 6)

    def test_rewind_closes_range_tail_against_stale_reader_count(self):
        for recover_hot in (False, True):
            with self.subTest(recover_hot=recover_hot), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                storage = root / "plattli"
                storage.mkdir()
                steps = list(range(0, 202, 2)) + list(range(400, 598, 2))
                (storage / "plattli.json").write_text(json.dumps({"loss": {"indices": "ranges", "dtype": "f32"}}))
                (storage / "loss.ranges").write_bytes(_ranges_encode([
                    {"start": 0, "stop": 202, "step": 2},
                    {"start": 400, "stop": 598, "step": 2},
                ], open_tail=True))
                np.asarray(steps, dtype=np.float32).tofile(storage / "loss.f32")
                if recover_hot:
                    # Resume cuts loss to 101 rows at step 301. Recovery then cuts
                    # at 250, retaining those same rows: it must not reopen the tail.
                    (storage / "hot.jsonl").write_text('{"step":250,"other":1.0}\n')
                with plattli.Reader(root, exclude_hot=True) as reader:
                    read_bytes = reader._read_bytes
                    read_values = reader._read_value_slice
                    state = {}

                    def rewind_before_ranges(name):
                        if name == "loss.ranges" and not state:
                            state["writer"] = plattli.CompactingWriter(root, step=301, hotsize=50)
                            records, tail = _ranges_parse((storage / "loss.ranges").read_bytes())
                            self.assertIsNone(tail)
                            self.assertEqual(records["end_row"].tolist(), [101])
                        return read_bytes(name)

                    def append_before_values(*args):
                        if "appended" not in state:
                            state["appended"] = True
                            writer = state["writer"]
                            for step in range(301, 501):
                                writer.step = step
                                writer.write(loss=float(step))
                                writer.end_step()
                            writer.write(flush=True)
                            writer._compact_future.result()
                        return read_values(*args)

                    # Count 200 values before rewind; read the retained range before
                    # a different cadence replaces the discarded values.
                    try:
                        with mock.patch.object(reader, "_read_bytes", rewind_before_ranges), \
                                mock.patch.object(reader, "_read_value_slice", append_before_values):
                            indices, values = reader.metric("loss")
                        np.testing.assert_array_equal(indices, steps[:101])
                        np.testing.assert_array_equal(values, steps[:101])
                    finally:
                        if "writer" in state:
                            state["writer"].finish(optimize=False, zip=False)
                with plattli.Reader(root) as reader:
                    expected = steps[:101] + list(range(301, 501))
                    np.testing.assert_array_equal(reader.metric_indices("loss"), expected)
                    np.testing.assert_array_equal(reader.metric_values("loss"), expected)
                    if recover_hot:
                        self.assertEqual(reader.metric("other", idx=0), (250, 1.0))

    def test_resume_ranges_before_first_row_leaves_empty_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            storage = root / "plattli"
            manifest_path = storage / "plattli.json"
            manifest = json.loads(manifest_path.read_text())
            (storage / "loss.ranges").write_bytes(_ranges_encode(manifest["loss"]["indices"], open_tail=True))
            manifest["loss"]["indices"] = "ranges"
            manifest_path.write_text(json.dumps(manifest))
            writer = plattli.CompactingWriter(root, step=0, hotsize=20)
            self.assertEqual((storage / "loss.ranges").read_bytes(), b"")
            self.assertEqual((storage / "loss.f32").stat().st_size, 0)
            writer.write(loss=123.0)
            writer.end_step()
            writer.finish(optimize=False, zip=False)
            with plattli.Reader(root) as reader:
                self.assertEqual(reader.metric_indices("loss").tolist(), [0])
                self.assertEqual(reader.metric_values("loss").tolist(), [123.0])

    def test_finish_index_plan_publishes_once_and_skips_unchanged_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segments = [{"start": i * 20, "stop": i * 20 + 10, "step": 1} for i in range(33)]
            steps = _segments_to_array(segments)
            manifest = {name: {"indices": "indices", "dtype": "f32"} for name in ("a", "b")}
            for name in manifest:
                steps.tofile(root / f"{name}.indices")
                steps.astype(np.float32).tofile(root / f"{name}.f32")
            (root / "plattli.json").write_text(json.dumps(manifest))
            publish = writer_module._write_manifest

            def inspect_publication(path, planned):
                for name in manifest:
                    self.assertTrue((root / f"{name}.ranges").exists())
                    self.assertTrue((root / f"{name}.indices").exists())
                return publish(path, planned)

            with mock.patch.object(writer_module, "_write_manifest", side_effect=inspect_publication) as write:
                writer_module._optimize_indices(root, manifest)
            self.assertEqual(write.call_count, 1)
            self.assertTrue(all(spec["indices"] == "ranges" for spec in manifest.values()))
            with mock.patch.object(writer_module, "_replace_bytes", side_effect=AssertionError("unchanged sidecar")), mock.patch.object(writer_module, "_write_manifest", side_effect=AssertionError("unchanged manifest")):
                writer_module._optimize_indices(root, manifest)
            with plattli.Reader(root) as reader:
                for name in manifest:
                    np.testing.assert_array_equal(reader.metric_indices(name), steps)

    def test_live_policy_rejects_invalid_option_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "live_indices"):
                plattli.CompactingWriter(root, hotsize=50, live_indices="ranges")
            self.assertFalse(root.exists())

    def test_live_policy_keeps_short_runs_explicit_then_optimizes_at_finish(self):
        for policy in ("auto", "explicit"):
            for length in (5, 32):
                with self.subTest(policy=policy, length=length), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    steps = [i * (length + 1) + j for i in range(40) for j in range(length)]
                    writer = plattli.CompactingWriter(root, hotsize=len(steps), live_indices=policy)
                    for step in steps:
                        writer.step = step
                        writer.write(loss=float(step))
                        writer.end_step()
                    writer.write(flush=True)
                    writer._compact_future.result()
                    self.assertEqual(writer._manifest["loss"]["indices"], "ranges" if policy == "auto" and length == 32 else "indices")
                    with plattli.Reader(root) as reader:
                        np.testing.assert_array_equal(reader.metric_indices("loss"), steps)
                    writer.finish(zip=False)
                    with plattli.Reader(root) as reader:
                        self.assertEqual(reader.manifest()["loss"]["indices"], "ranges")
                        np.testing.assert_array_equal(reader.metric_indices("loss"), steps)
                        np.testing.assert_array_equal(reader.metric_values("loss"), steps)

    def test_explicit_live_policy_converts_existing_inline_and_ranges(self):
        for representation in ("inline", "ranges"):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                steps = _fixture(root)
                if representation == "ranges":
                    writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
                    writer.finish(optimize=False, zip=False)
                writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50, live_indices="explicit")
                self.assertEqual(writer._manifest["loss"]["indices"], "indices")
                writer.write(loss=float(writer.step))
                writer.end_step()
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(root) as reader:
                    self.assertEqual(reader.manifest()["loss"]["indices"], "indices")
                    np.testing.assert_array_equal(reader.metric_indices("loss"), steps + [steps[-1] + 1])
                    np.testing.assert_array_equal(reader.metric_values("loss"), steps + [steps[-1] + 1])

    def test_recent_gaps_trigger_fallback_despite_long_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = _fixture(root)
            writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
            # Only 10 new ranges: history still averages over 32 samples per range.
            added = [steps[-1] + 2 + i + i // 5 for i in range(50)]
            self.assertGreater((len(steps) + len(added)) / 43, 32)
            writer._compact_rows([{"step": step, "loss": float(step)} for step in added])
            self.assertEqual(writer._manifest["loss"]["indices"], "indices")
            with plattli.Reader(root) as reader:
                np.testing.assert_array_equal(reader.metric_indices("loss"), steps + added)
                np.testing.assert_array_equal(reader.metric_values("loss"), steps + added)
            writer.finish(optimize=False, zip=False)

    def test_bulk_and_direct_finalize_to_ranges(self):
        for kind in ("bulk", "direct"):
            for zipped in (False, True):
                with self.subTest(kind=kind, zipped=zipped), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    writer = plattli.BulkWriter(root) if kind == "bulk" else plattli.DirectWriter(root, write_threads=0)
                    steps = [20 * segment + offset for segment in range(33) for offset in range(10)]
                    for step in steps:
                        writer.step = step
                        writer.write(loss=float(step))
                        writer.end_step()
                    writer.finish(zip=zipped)
                    with plattli.Reader(root) as reader:
                        self.assertEqual(reader.manifest()["loss"]["indices"], "ranges")
                        np.testing.assert_array_equal(reader.metric_indices("loss"), steps)
                        np.testing.assert_array_equal(reader.metric_values("loss"), steps)

    def test_continuing_open_range_does_not_rewrite_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = _fixture(root)
            writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
            path = root / "plattli/loss.ranges"
            before = path.read_bytes()
            manifest = (root / "plattli/plattli.json").read_bytes()
            original_open = Path.open

            def forbid_range_write(path, mode="r", *args, **kwargs):
                if path.suffix == ".ranges" and any(flag in mode for flag in "wa+"):
                    raise AssertionError("continuing cadence wrote range metadata")
                return original_open(path, mode, *args, **kwargs)

            new_steps = list(range(steps[-1] + 1, steps[-1] + 51))
            with mock.patch.object(Path, "open", forbid_range_write):
                writer._compact_rows([{"step": step, "loss": float(step)} for step in new_steps])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual((root / "plattli/plattli.json").read_bytes(), manifest)
            with plattli.Reader(root) as reader:
                np.testing.assert_array_equal(reader.metric_indices("loss"), steps + new_steps)
            writer.finish(optimize=False, zip=False)
            self.assertIsNone(_ranges_parse(path.read_bytes())[1])

    def test_gap_publishes_ranges_before_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = _fixture(root)
            writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
            path = root / "plattli/loss.ranges"
            before = path.read_bytes()
            new_steps = list(range(steps[-1] + 10, steps[-1] + 60))
            append = writer_module._append_numeric
            def inspect_before_values(path, values, dtype):
                self.assertTrue(sync.called)
                data = (root / "plattli/loss.ranges").read_bytes()
                self.assertTrue(data.startswith(before))
                records, tail = _ranges_parse(data)
                self.assertEqual(int(records[-1]["end_row"]), len(steps))
                self.assertEqual(tail, (new_steps[0], 1))
                with plattli.Reader(root) as reader:
                    np.testing.assert_array_equal(reader.metric_indices("loss"), steps)
                return append(path, values, dtype)

            with (
                mock.patch.object(writer_module, "_append_numeric", inspect_before_values),
                mock.patch.object(writer_module.os, "fsync", wraps=writer_module.os.fsync) as sync,
            ):
                writer._compact_rows([{"step": step, "loss": float(step)} for step in new_steps])
            writer.finish(optimize=False, zip=False)
            with plattli.Reader(root) as reader:
                np.testing.assert_array_equal(reader.metric_indices("loss"), steps + new_steps)

    def test_interrupted_range_compaction_replays_hot_rows_once(self):
        for partial in (0, 17):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                steps = _fixture(root)
                new_steps = list(range(steps[-1] + 10, steps[-1] + 60))
                writer = plattli.CompactingWriter(root, step=new_steps[0], hotsize=50)
                append = writer_module._append_numeric

                def interrupt(path, values, dtype):
                    if partial:
                        append(path, values[:partial], dtype)
                    raise OSError("interrupted values")

                with mock.patch.object(writer_module, "_append_numeric", interrupt):
                    for step in new_steps:
                        writer.step = step
                        writer.write(loss=float(step))
                        writer.end_step()
                    writer.write({}, flush=True)
                    with self.assertRaisesRegex(OSError, "interrupted values"):
                        writer._compact_future.result()
                writer._close_hot_file()
                writer._compact_executor.shutdown(wait=True)
                writer._compact_executor = None
                self.assertTrue((root / "plattli/hot.compacting.jsonl").exists())
                self.assertEqual((root / "plattli/loss.f32").stat().st_size // 4, len(steps) + partial)

                writer = plattli.CompactingWriter(root, step=new_steps[-1] + 1, hotsize=50)
                with plattli.Reader(root) as reader:
                    np.testing.assert_array_equal(reader.metric_indices("loss"), steps + new_steps)
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(root) as reader:
                    np.testing.assert_array_equal(reader.metric_indices("loss"), steps + new_steps)
                    np.testing.assert_array_equal(reader.metric_values("loss"), steps + new_steps)

    def test_finalization_cleans_orphan_range_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = _fixture(root)
            writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
            (root / "plattli/orphan.ranges").write_bytes(b"orphan")
            (root / "plattli/orphan.f32").write_bytes(b"orphan")
            (root / "plattli/standalone.ranges").write_bytes(b"ancillary")
            writer.finish(optimize=False, zip=False)
            self.assertFalse((root / "plattli/orphan.ranges").exists())
            self.assertFalse((root / "plattli/orphan.f32").exists())
            self.assertEqual((root / "plattli/standalone.ranges").read_bytes(), b"ancillary")

    def test_resume_inside_range_for_both_streaming_writers(self):
        for kind in ("compact", "direct"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                steps = _fixture(root)
                writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
                writer.finish(optimize=False, zip=True)
                cut = steps[173]
                writer = plattli.CompactingWriter(root, step=cut, hotsize=50, allow_resume_finalized=True) if kind == "compact" else \
                    plattli.DirectWriter(root, step=cut, write_threads=0, allow_resume_finalized=True)
                writer.write(loss=float(cut + 1000))
                writer.end_step()
                writer.finish(optimize=False, zip=False)
                with plattli.Reader(root) as reader:
                    np.testing.assert_array_equal(reader.metric_indices("loss"), steps[:173] + [cut])
                    np.testing.assert_array_equal(reader.metric_values("loss"), steps[:173] + [cut + 1000])
                    # The rewind closes the tail, so this one-row batch starts a new range
                    # and falls back to explicit indices.
                    self.assertEqual(reader.manifest()["loss"]["indices"], "indices")

    def test_live_fallback_retires_ranges_and_retries_cached_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = _fixture(root)
            writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
            with plattli.Reader(root, exclude_hot=True) as reader:
                np.testing.assert_array_equal(reader.metric_indices("loss"), steps)
                point = steps[-1]
                new_steps = []
                for i in range(1000):
                    point += 2 + i % 2
                    new_steps.append(point)
                writer._compact_rows([{"step": step, "loss": float(step)} for step in new_steps])
                self.assertEqual(writer._manifest["loss"]["indices"], "indices")
                self.assertFalse((root / "plattli/loss.ranges").exists())
                np.testing.assert_array_equal(reader.metric_indices("loss"), steps + new_steps)
            writer.finish(optimize=False, zip=False)

    def test_failed_manifest_publication_preserves_explicit_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = plattli.DirectWriter(root, write_threads=0)
            steps = [20 * segment + offset for segment in range(33) for offset in range(10)]
            for step in steps:
                writer.step = step
                writer.write(loss=float(step))
                writer.end_step()
            with mock.patch.object(writer_module, "_write_manifest", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    writer_module._optimize_indices(writer.root, writer._manifest)
            self.assertEqual(writer._manifest["loss"]["indices"], "indices")
            self.assertTrue((writer.root / "loss.indices").exists())
            with plattli.Reader(root) as reader:
                np.testing.assert_array_equal(reader.metric_indices("loss"), steps)
            writer.finish(optimize=False, zip=False)
            self.assertFalse((writer.root / "loss.ranges").exists())

    def test_bulk_overwrite_retires_ranges_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = _fixture(root)
            writer = plattli.CompactingWriter(root, step=steps[-1] + 1, hotsize=50)
            writer.finish(optimize=False, zip=False)
            (root / "plattli/standalone.ranges").write_bytes(b"ancillary")
            writer = plattli.BulkWriter(root, overwrite=True)
            writer.write(current=1)
            writer.end_step()
            writer.finish(zip=False)
            self.assertFalse((root / "plattli/loss.ranges").exists())
            self.assertFalse((root / "plattli/loss.f32").exists())
            self.assertEqual((root / "plattli/standalone.ranges").read_bytes(), b"ancillary")
