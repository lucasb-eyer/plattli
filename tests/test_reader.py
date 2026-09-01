import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import plattli
from plattli.writer import _zip_path_for_root


class TestReader(unittest.TestCase):
    def test_reader_accepts_trusted_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.set_config({"trusted": True})
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            with mock.patch("plattli.reader._resolve_plattli", side_effect=AssertionError("unexpected discovery")):
                with plattli.Reader(run_root / "plattli", kind="dir") as r:
                    self.assertEqual(r.config(), {"trusted": True})
                    self.assertEqual(r.approx_max_rows(), 1)

    def test_reader_accepts_trusted_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.set_config({"trusted": True})
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=True)

            with mock.patch("plattli.reader._resolve_plattli", side_effect=AssertionError("unexpected discovery")):
                with plattli.Reader(run_root / "metrics.plattli", kind="zip") as r:
                    self.assertEqual(r.config(), {"trusted": True})
                    self.assertEqual(r.approx_max_rows(), 1)

    def test_reader_rejects_invalid_trusted_kind(self):
        with self.assertRaisesRegex(ValueError, "invalid reader kind"):
            plattli.Reader("ignored", kind="other")

    def test_reader_open_tail_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            plattli_root.mkdir(parents=True)
            (plattli_root / "plattli.json").write_text(
                json.dumps({"loss": {"indices": [{"start": 0, "step": 2}], "dtype": "f32"}}),
                encoding="utf-8",
            )
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tofile(plattli_root / "loss.f32")

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 2, 4])
                self.assertEqual(r.metric_indices("loss", start=1, stop=3).tolist(), [2])
                self.assertTrue(np.allclose(r.metric_values("loss", start=1, stop=3), [2.0]))
                self.assertEqual(r.approx_max_rows(), 3)

    def test_reader_open_tail_truncates_closed_segments_to_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            plattli_root.mkdir(parents=True)
            (plattli_root / "plattli.json").write_text(
                json.dumps({"loss": {"indices": [
                    {"start": 0, "stop": 3, "step": 1},
                    {"start": 10, "stop": 14, "step": 2},
                    {"start": 20, "step": 1},
                ], "dtype": "f32"}}),
                encoding="utf-8",
            )
            np.asarray([0.0, 1.0, 2.0, 10.0], dtype=np.float32).tofile(plattli_root / "loss.f32")

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1, 2, 10])
                self.assertEqual(r.rows("loss"), 4)
                self.assertEqual(r.approx_max_rows(), 4)

    def test_reader_piecewise_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            steps = [2, 1000, 2000, 3000, 4000]
            for step in steps:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=False)

            self.assertFalse((plattli_root / "loss.indices").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), steps)
                self.assertTrue(np.allclose(r.metric_values("loss"), np.asarray(steps, dtype=np.float32)))

    def test_reader_metric_position_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(10):
                w.step = step * 10
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=False)

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss", istart=2, istop=5).tolist(), [20, 30, 40])
                self.assertTrue(np.allclose(r.metric_values("loss", istart=2, istop=5), [2, 3, 4]))
                idx, values = r.metric("loss", istart=2, istop=5)
                self.assertEqual(idx.tolist(), [20, 30, 40])
                self.assertTrue(np.allclose(values, [2, 3, 4]))

    def test_reader_metric_step_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in [0, 2, 4, 20, 22, 24]:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=False)

            with plattli.Reader(run_root) as r:
                idx, values = r.metric("loss", start=3, stop=22)
                self.assertEqual(idx.tolist(), [4, 20, 22])
                self.assertTrue(np.allclose(values, [4, 20, 22]))

    def test_reader_metric_value_range_uses_monotonic_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step, value in enumerate([10.0, 11.0, 12.0, 13.0, 14.0]):
                w.step = step
                w.write(walltime=value)
                w.end_step()
            w.finish(optimize=False, zip=False)

            with plattli.Reader(run_root) as r:
                idx, values = r.metric("walltime", vstart=11.0, vstop=13.0)
                self.assertEqual(idx.tolist(), [1, 2, 3])
                self.assertTrue(np.allclose(values, [11, 12, 13]))

    def test_reader_metric_value_range_descending(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step, value in enumerate([5.0, 4.0, 3.0, 2.0, 1.0]):
                w.step = step
                w.write(delta=value)
                w.end_step()
            w.finish(optimize=False, zip=False)

            with plattli.Reader(run_root) as r:
                idx, values = r.metric("delta", vstart=2.0, vstop=4.0)
                self.assertEqual(idx.tolist(), [1, 2, 3])
                self.assertTrue(np.allclose(values, [4, 3, 2]))

    def test_reader_metric_value_range_falls_back_when_not_monotonic(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step, value in enumerate([0.0, 10.0, 2.0, 3.0, 30.0]):
                w.step = step
                w.write(walltime=value)
                w.end_step()
            w.finish(optimize=False, zip=False)

            with plattli.Reader(run_root) as r:
                idx, values = r.metric("walltime", vstart=2.0, vstop=10.0)
                self.assertEqual(idx.tolist(), [1, 2, 3])
                self.assertTrue(np.allclose(values, [10, 2, 3]))

    def test_reader_metric_range_selectors_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            with plattli.Reader(run_root) as r:
                with self.assertRaises(ValueError):
                    r.metric("loss", start=0, istart=0)

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            steps = [2, 1000, 2000, 3000, 4000]
            for step in steps:
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=True)

            self.assertTrue(_zip_path_for_root(run_root).exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), steps)
                self.assertTrue(np.allclose(r.metric_values("loss"), np.asarray(steps, dtype=np.float32)))

    def test_zip_slice_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for i in range(100):
                w.write(loss=float(i))
                w.end_step()
            w.finish(optimize=False, zip=True)  # Keep .indices files so zip seek-slicing is exercised.

            with plattli.Reader(run_root) as r:
                idx, values = r.metric("loss", istart=40, istop=45)
                self.assertEqual(idx.tolist(), [40, 41, 42, 43, 44])
                self.assertTrue(np.allclose(values, [40, 41, 42, 43, 44]))
                idx, values = r.metric("loss", start=90, stop=94)
                self.assertEqual(idx.tolist(), [90, 91, 92, 93, 94])
                self.assertTrue(np.allclose(values, [90, 91, 92, 93, 94]))
                step, value = r.metric("loss", idx=-1)
                self.assertEqual(step, 99)
                self.assertEqual(value, np.float32(99.0))

    def test_table_aligns_mixed_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for i in range(12):
                metrics = {"loss": float(i), "wall": 2.0 * i}
                if i % 3 == 0:
                    metrics["acc"] = i / 12
                w.write(**metrics)
                w.end_step()
            w.finish(optimize=False, zip=False)

            with plattli.Reader(run_root) as r:
                steps, cols = r.table(["loss", "acc"])
                self.assertEqual(steps.tolist(), [0, 3, 6, 9])
                self.assertTrue(np.allclose(cols["loss"], [0, 3, 6, 9]))
                self.assertTrue(np.allclose(cols["acc"], [0, 0.25, 0.5, 0.75]))

                steps, cols = r.table(["loss", "acc"], start=3, stop=8)
                self.assertEqual(steps.tolist(), [3, 6])

                steps, cols = r.table(["loss", "acc"], istart=-2)
                self.assertEqual(steps.tolist(), [6, 9])

                # Selectors on a metric `on` column: select by wall value, others follow.
                steps, cols = r.table(["wall", "loss"], on="wall", vstart=4.0, vstop=10.0)
                self.assertEqual(steps.tolist(), [2, 3, 4, 5])
                self.assertTrue(np.allclose(cols["wall"], [4, 6, 8, 10]))
                self.assertTrue(np.allclose(cols["loss"], [2, 3, 4, 5]))

                steps, cols = r.table(["acc"], on="wall", vstart=4.0, vstop=19.0)
                self.assertEqual(steps.tolist(), [3, 6, 9])

                steps, cols = r.table(["loss", "acc"], start=100, stop=200)
                self.assertEqual(steps.tolist(), [])
                self.assertEqual(len(cols["loss"]), 0)
                self.assertEqual(cols["loss"].dtype, np.float32)

                with self.assertRaises(ValueError):
                    r.table(["loss"], vstart=1.0)
                with self.assertRaises(ValueError):
                    r.table([])
                with self.assertRaises(KeyError):
                    r.table(["loss", "nope"])

    def test_table_skips_alignment_for_equal_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for i in range(6):
                w.write(loss=10.0 + i, wall=100.0 + i)
                w.end_step()
            w.finish(optimize=True, zip=False)

            # These are expensive on large columns. Equal index arrays can use their
            # values directly, so guard the fast path without a timing-based test.
            intersect1d = np.intersect1d
            searchsorted = np.searchsorted
            with (
                mock.patch("plattli.reader.np.intersect1d", wraps=intersect1d) as intersect,
                mock.patch("plattli.reader.np.searchsorted", wraps=searchsorted) as search,
                plattli.Reader(run_root) as r,
            ):
                steps, cols = r.table(["loss"])
                self.assertEqual(steps.tolist(), [0, 1, 2, 3, 4, 5])
                self.assertTrue(np.allclose(cols["loss"], [10, 11, 12, 13, 14, 15]))

                steps, cols = r.table(["loss", "wall"])
                self.assertEqual(steps.tolist(), [0, 1, 2, 3, 4, 5])
                self.assertTrue(np.allclose(cols["wall"], [100, 101, 102, 103, 104, 105]))

                # Position selectors apply after the join, including negative bounds.
                steps, cols = r.table(["loss", "wall"], istart=1, istop=-1)
                self.assertEqual(steps.tolist(), [1, 2, 3, 4])
                self.assertTrue(np.allclose(cols["loss"], [11, 12, 13, 14]))
                self.assertTrue(np.allclose(cols["wall"], [101, 102, 103, 104]))

                # The `on` column may be requested or used only as the row selector.
                steps, cols = r.table(["wall", "loss"], on="wall", istart=1, istop=4)
                self.assertEqual(steps.tolist(), [1, 2, 3])
                self.assertTrue(np.allclose(cols["wall"], [101, 102, 103]))
                self.assertTrue(np.allclose(cols["loss"], [11, 12, 13]))

                steps, cols = r.table(["loss"], on="wall", istart=1, istop=4)
                self.assertEqual(steps.tolist(), [1, 2, 3])
                self.assertEqual(list(cols), ["loss"])
                self.assertTrue(np.allclose(cols["loss"], [11, 12, 13]))

                steps, cols = r.table(["loss"], start=100, stop=200)
                self.assertEqual(steps.tolist(), [])
                self.assertEqual(cols["loss"].tolist(), [])
                self.assertEqual(steps.dtype, np.uint32)
                self.assertEqual(cols["loss"].dtype, np.float32)

                intersect.assert_not_called()
                search.assert_not_called()

    def test_table_equal_cadence_trims_racing_values(self):
        # A live writer may append a value after metric() has read its indices.
        # The alignment gather historically ignored that not-yet-indexed tail.
        indices = np.asarray([0, 1], dtype=np.uint32)
        r = object.__new__(plattli.Reader)
        r._run_name = "mock-live-run"
        r.metric = mock.Mock(side_effect=[
            (indices, np.asarray([10.0, 11.0, 12.0], dtype=np.float32)),
            (indices.copy(), np.asarray([20.0, 21.0, 22.0], dtype=np.float32)),
        ])

        steps, cols = r.table(["loss", "acc"])

        self.assertEqual(steps.tolist(), [0, 1])
        self.assertEqual(cols["loss"].tolist(), [10.0, 11.0])
        self.assertEqual(cols["acc"].tolist(), [20.0, 21.0])
        self.assertEqual(len(steps), len(cols["loss"]))
        self.assertEqual(len(steps), len(cols["acc"]))

    def test_table_aligns_equal_length_different_indices(self):
        r = object.__new__(plattli.Reader)
        r._run_name = "mock-run"
        r.metric = mock.Mock(side_effect=[
            (np.asarray([0, 1, 3], dtype=np.uint32), np.asarray([10.0, 11.0, 13.0], dtype=np.float32)),
            (np.asarray([0, 2, 3], dtype=np.uint32), np.asarray([20.0, 22.0, 23.0], dtype=np.float32)),
        ])

        steps, cols = r.table(["loss", "acc"])

        self.assertEqual(steps.tolist(), [0, 3])
        self.assertEqual(cols["loss"].tolist(), [10.0, 13.0])
        self.assertEqual(cols["acc"].tolist(), [20.0, 23.0])

    def test_table_live_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            for i in range(7):
                metrics = {"loss": float(i)}
                if i % 2 == 0:
                    metrics["acc"] = float(i)
                w.write(metrics)
                w.end_step()
                if w._compact_future:
                    w._compact_future.result()

            with plattli.Reader(run_root) as r:  # Mixed columnar + hot.
                steps, cols = r.table(["loss", "acc"])
                self.assertEqual(steps.tolist(), [0, 2, 4, 6])
                self.assertTrue(np.allclose(cols["acc"], [0, 2, 4, 6]))

            w.finish(zip=True)
            with plattli.Reader(run_root) as r:
                steps, cols = r.table(["loss", "acc"], start=2, stop=5)
                self.assertEqual(steps.tolist(), [2, 4])
                self.assertTrue(np.allclose(cols["loss"], [2, 4]))

    def test_refresh_picks_up_live_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.CompactingWriter(run_root, hotsize=100)
            w.write(loss=1.0)
            w.end_step()

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metrics(), ["loss"])
                self.assertEqual(r.rows("loss"), 1)

                w.write(loss=2.0, acc=0.5)
                w.end_step()

                # Cached metadata is stale until refresh.
                self.assertEqual(r.metrics(), ["loss"])
                self.assertEqual(r.rows("loss"), 1)

                r.refresh()
                self.assertEqual(r.metrics(), ["acc", "loss"])
                self.assertEqual(r.rows("loss"), 2)
                self.assertEqual(r.metric_values("loss").tolist(), [1.0, 2.0])
                step, value = r.metric("acc", idx=-1)
                self.assertEqual((step, value), (1, 0.5))

    def test_tail_reads_fast_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            for zip in (False, True):
                run_root = Path(tmp) / f"run{zip}"
                w = plattli.DirectWriter(run_root, write_threads=0)
                for i in range(100):
                    w.write(loss=float(i), note=f"n{i}")
                    w.end_step()
                w.finish(optimize=False, zip=zip)

                with plattli.Reader(run_root) as r:
                    with mock.patch.object(r, "_metric_full", side_effect=AssertionError("fell back to full read")):
                        step, value = r.metric("loss", idx=-1)
                        self.assertEqual((step, value), (99, np.float32(99.0)))
                        step, value = r.metric("loss", idx=5)
                        self.assertEqual((step, value), (5, np.float32(5.0)))
                        step, value = r.metric("note", idx=-2)
                        self.assertEqual((step, value), (98, "n98"))
                        self.assertEqual(r.metric_values("loss", istart=-3).tolist(), [97.0, 98.0, 99.0])
                        self.assertEqual(r.metric_indices("loss", istart=-5, istop=-3).tolist(), [95, 96])
                        with self.assertRaises(IndexError):
                            r.metric("loss", idx=100)
                        with self.assertRaises(IndexError):
                            r.metric("loss", idx=-101)

    def test_zip_reads_use_owned_4k_buffered_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for i in range(100):
                w.write(loss=float(i))
                w.end_step()
            w.finish(optimize=False, zip=True)

            zip_path = run_root / "metrics.plattli"
            original_open = Path.open
            opened = []

            def tracked_open(path, *args, **kwargs):
                fh = original_open(path, *args, **kwargs)
                if path == zip_path:
                    opened.append((args, kwargs, fh))
                return fh

            with mock.patch.object(Path, "open", tracked_open):
                with plattli.Reader(run_root) as r:
                    fh = r._zip_fh
                    self.assertIs(r._zip.fp, fh)
                    self.assertEqual(r.metric("loss", idx=-1), (99, np.float32(99.0)))
                    idx, values = r.metric("loss", istart=20, istop=25)
                    self.assertEqual(idx.tolist(), [20, 21, 22, 23, 24])
                    self.assertEqual(values.tolist(), [20.0, 21.0, 22.0, 23.0, 24.0])
                    self.assertEqual(r.metric_values("loss").tolist(), [float(i) for i in range(100)])

            self.assertEqual([(args, kwargs) for args, kwargs, _ in opened], [
                (("rb",), {"buffering": 4096}),
            ])
            self.assertTrue(fh.closed)

    def test_zip_constructor_failures_close_buffered_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            invalid_zip = Path(tmp) / "invalid.plattli"
            invalid_zip.write_bytes(b"not a zip")
            missing_manifest = Path(tmp) / "missing-manifest.plattli"
            with zipfile.ZipFile(missing_manifest, "w") as zf:
                zf.writestr("other.json", "{}")

            original_open = Path.open
            opened = []

            def tracked_open(path, *args, **kwargs):
                fh = original_open(path, *args, **kwargs)
                if path in (invalid_zip, missing_manifest):
                    opened.append((args, kwargs, fh))
                return fh

            with mock.patch.object(Path, "open", tracked_open):
                with self.assertRaises(FileNotFoundError):
                    plattli.Reader(invalid_zip)
                with self.assertRaises(FileNotFoundError):
                    plattli.Reader(missing_manifest)

            self.assertEqual([(args, kwargs) for args, kwargs, _ in opened], [
                (("rb",), {"buffering": 4096}),
                (("rb",), {"buffering": 4096}),
            ])
            self.assertTrue(all(fh.closed for _, _, fh in opened))

    def test_repeated_zip_reader_construction_does_not_leak_handles(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=True)

            handles = []
            for _ in range(100):
                with plattli.Reader(run_root) as r:
                    handles.append(r._zip_fh)
                    self.assertFalse(r._zip_fh.closed)
            self.assertTrue(all(fh.closed for fh in handles))

    def test_tail_reads_with_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            for i in range(7):
                w.write(loss=float(i))
                w.end_step()
                if w._compact_future:
                    w._compact_future.result()

            with plattli.Reader(run_root) as r:
                step, value = r.metric("loss", idx=-1)
                self.assertEqual((step, value), (6, 6.0))
                self.assertEqual(r.metric_values("loss", istart=-2).tolist(), [5.0, 6.0])

    def test_selectors_with_hot_use_fast_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            for i in range(7):
                w.write(loss=float(i))
                w.end_step()
                if w._compact_future:
                    w._compact_future.result()
            # Some rows are compacted to columnar, the rest still live in the hot log.
            self.assertTrue((plattli_root / "hot.jsonl").exists())
            self.assertTrue((plattli_root / "loss.f32").exists())

            with plattli.Reader(run_root) as r:
                with mock.patch.object(r, "_metric_full", side_effect=AssertionError("fell back to full read")):
                    idx, values = r.metric("loss", start=2, stop=5)
                    self.assertEqual(idx.tolist(), [2, 3, 4, 5])
                    self.assertTrue(np.allclose(values, [2, 3, 4, 5]))
                    idx, values = r.metric("loss", istart=1, istop=6)
                    self.assertEqual(idx.tolist(), [1, 2, 3, 4, 5])
                    self.assertTrue(np.allclose(values, [1, 2, 3, 4, 5]))
                    idx, values = r.metric("loss", vstart=3.0, vstop=5.0)  # loss is monotonic inc
                    self.assertEqual(idx.tolist(), [3, 4, 5])
                    self.assertEqual(r.metric_indices("loss", start=4).tolist(), [4, 5, 6])
                    self.assertEqual(r.metric_values("loss", istart=5).tolist(), [5.0, 6.0])
                    idx, values = r.metric("loss", istart=3, istop=4)
                    self.assertEqual((idx.tolist(), values.tolist()), ([3], [3.0]))
                    idx, values = r.metric("loss", start=10, stop=20)
                    self.assertEqual(idx.tolist(), [])
                    self.assertEqual(values.tolist(), [])

    def test_reader_rows_hot_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=100)
            w.write(loss=1.0, acc=0.5)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0, acc=0.7)
            w.end_step()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)
                self.assertEqual(r.rows("acc"), 2)

    def test_run_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            self.assertTrue(plattli.is_run(run_root))
            self.assertTrue(plattli.is_run_dir(run_root))
            self.assertEqual(plattli.resolve_run_dir(run_root), plattli_root.resolve())
            self.assertTrue(plattli.is_run(plattli_root))
            self.assertTrue(plattli.is_run_dir(plattli_root))
            self.assertEqual(plattli.resolve_run_dir(plattli_root), plattli_root.resolve())

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "ziprun"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=True)

            zip_path = run_root / "metrics.plattli"
            self.assertTrue(zip_path.exists())
            self.assertTrue(plattli.is_run(run_root))
            self.assertFalse(plattli.is_run_dir(run_root))
            self.assertIsNone(plattli.resolve_run_dir(run_root))
            self.assertTrue(plattli.is_run(zip_path))
            self.assertFalse(plattli.is_run_dir(zip_path))
            self.assertIsNone(plattli.resolve_run_dir(zip_path))

    def test_reader_rows_columnar_plus_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            if w._compact_future:
                w._compact_future.result()
            w.write(loss=3.0)
            w.end_step()
            if w._compact_future:
                w._compact_future.result()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)

    def test_reader_tolerates_partial_numeric_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(b"\x01\x02")
            with (plattli_root / "loss.f32").open("ab") as fh:
                fh.write(b"\x03\x04")

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1])
                self.assertTrue(np.allclose(r.metric_values("loss"),
                                            np.asarray([1.0, 2.0], dtype=np.float32)))
                self.assertEqual(r.rows("loss"), 2)

    def test_reader_tolerates_mismatch_and_jsonl_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0, text="a")
            w.end_step()
            w.write(loss=2.0, text="b")
            w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(np.asarray([2], dtype=np.uint32).tobytes())
            with (plattli_root / "text.jsonl").open("ab") as fh:
                fh.write(b"\"c")

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1])
                self.assertTrue(np.allclose(r.metric_values("loss"),
                                            np.asarray([1.0, 2.0], dtype=np.float32)))
                self.assertEqual(r.metric_values("text").tolist(), ["a", "b"])
                self.assertEqual(r.rows("text"), 2)

    def test_reader_tolerates_hot_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=100)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()

            with (plattli_root / "hot.jsonl").open("ab") as fh:
                fh.write(b"{\"step\":2,\"loss\":")

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1])
                self.assertTrue(np.allclose(r.metric_values("loss"),
                                            np.asarray([1.0, 2.0], dtype=np.float32)))
                self.assertEqual(r.rows("loss"), 2)

    def test_reader_tolerates_transient_hot_unlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            transient = plattli_root / "hot.compacting.jsonl"
            transient.write_text(json.dumps({"step": 0, "loss": 1.0}) + "\n", encoding="utf-8")
            original_read_bytes = Path.read_bytes

            def unlink_before_read(path):
                if path == transient:
                    transient.unlink()
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", unlink_before_read):
                with plattli.Reader(run_root) as r:
                    indices, values = r.metric("loss")
                    self.assertEqual(indices.tolist(), [0])
                    self.assertTrue(np.allclose(values, [1.0]))

    def test_reader_missing_files_fail_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0, note="ok")
            w.end_step()
            w.finish(optimize=False, zip=False)

            (plattli_root / "loss.f32").unlink()
            (plattli_root / "note.indices").unlink()

            with plattli.Reader(run_root) as r:
                with self.assertRaises(FileNotFoundError):
                    r.metric_values("loss")
                with self.assertRaises(FileNotFoundError):
                    r.metric_indices("note")

    def test_reader_rows_mismatch_indices_uses_last_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(3):
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(np.asarray([100], dtype=np.uint32).tobytes())
            (plattli_root / "hot.jsonl").write_text(
                json.dumps({"step": 3, "loss": 9.0}) + "\n",
                encoding="utf-8",
            )

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 4)

    def test_reader_zip_indices_mismatch_and_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(3):
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(np.asarray([3], dtype=np.uint32).tobytes())

            zip_path = run_root / "metrics.plattli"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for path in sorted(plattli_root.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(plattli_root)
                    if path.is_symlink():
                        zf.writestr(rel.as_posix(), path.read_bytes())
                    else:
                        zf.write(path, rel)

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1, 2])
                self.assertEqual(r.rows("loss"), 3)
                idx, value = r.metric("loss", idx=1)
                self.assertEqual(int(idx), 1)
                self.assertEqual(float(value), 1.0)

    def test_reader_segment_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(5):
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            w.finish(optimize=True, zip=False)

            manifest = json.loads((plattli_root / "plattli.json").read_text(encoding="utf-8"))
            self.assertIsInstance(manifest["loss"]["indices"], list)

            value_path = plattli_root / "loss.f32"
            with value_path.open("r+b") as fh:
                fh.truncate(3 * np.dtype(np.float32).itemsize)

            with plattli.Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1, 2])

    def test_reader_columnar_and_hot_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            w.write(loss=1.0, text="a")
            w.end_step()
            w.write(loss=2.0, text="b")
            w.end_step()
            if w._compact_future:
                w._compact_future.result()
            w.write(loss=3.0, text="c", hot_only=7)
            w.end_step()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.metric_values("loss").tolist(), [1.0, 2.0, 3.0])
                self.assertEqual(r.metric_values("text").tolist(), ["a", "b", "c"])
                self.assertEqual(r.metric_values("hot_only").tolist(), [7])
                with self.assertRaises(KeyError):
                    r.metric_values("missing")
            w.finish(optimize=False, zip=False)

    def test_reader_approx_max_rows_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.CompactingWriter(run_root, hotsize=2)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0)
            w.end_step()
            if w._compact_future:
                w._compact_future.result()
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.approx_max_rows(), 2)
            w.finish(optimize=False, zip=False)

    def test_reader_approx_max_rows_stratifies_probes_by_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            plattli_root = Path(tmp) / "run" / "plattli"
            plattli_root.mkdir(parents=True)
            manifest = {}
            for i in range(2400):
                manifest[f"slow_{i}"] = {"indices": {"start": 0, "step": 10}, "dtype": "f32"}
            for i in range(90):
                manifest[f"frequent_{i}"] = {"indices": {"start": 0, "step": 2}, "dtype": "f32"}
            for i in range(10):
                manifest[f"dense_{i}"] = {"indices": {"start": 0, "step": 1}, "dtype": "f32"}
            for i in range(3):
                manifest[f"explicit_{i}"] = {"indices": "indices", "dtype": "f32"}
            (plattli_root / "plattli.json").write_text(json.dumps(manifest), encoding="utf-8")

            probed = []

            def probe(name, spec):
                if spec["indices"] == "indices":
                    cadence = "indices"
                else:
                    cadence = spec["indices"]["step"]
                probed.append(cadence)
                return {"indices": 20, 1: 1000, 2: 500, 10: 100}[cadence]

            with plattli.Reader(plattli_root) as r:
                with mock.patch.object(r, "_probe_metric_count", side_effect=probe):
                    self.assertEqual(r.approx_max_rows(nprobes=12), 1000)

            self.assertEqual(len(probed), 12)
            self.assertEqual({cadence: probed.count(cadence) for cadence in set(probed)}, {
                "indices": 3,
                1: 3,
                2: 3,
                10: 3,
            })

    def test_reader_approx_max_rows_probes_all_candidates_within_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            plattli_root = Path(tmp) / "run" / "plattli"
            plattli_root.mkdir(parents=True)
            (plattli_root / "plattli.json").write_text(json.dumps({
                "closed": {"indices": {"start": 0, "stop": 7, "step": 1}, "dtype": "f32"},
                "explicit": {"indices": "indices", "dtype": "f32"},
                "open": {"indices": {"start": 0, "step": 2}, "dtype": "f32"},
                "open_jsonl": {"indices": {"start": 0, "step": 1}, "dtype": "jsonl"},
            }), encoding="utf-8")

            with plattli.Reader(plattli_root) as r:
                with mock.patch.object(r, "_probe_metric_count", side_effect=lambda name, spec: {
                    "explicit": 10,
                    "open": 12,
                }[name]) as probe:
                    self.assertEqual(r.approx_max_rows(nprobes=2), 12)
                    self.assertEqual({call.args[0] for call in probe.call_args_list}, {"explicit", "open"})
                    probe.reset_mock()
                    self.assertEqual(r.approx_max_rows(nprobes=3), 12)
                    self.assertEqual({call.args[0] for call in probe.call_args_list}, {"explicit", "open"})
                    probe.reset_mock()
                    self.assertEqual(r.approx_max_rows(nprobes=0), 7)
                    probe.assert_not_called()

    def test_reader_approx_max_rows_prefers_history_within_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            plattli_root = Path(tmp) / "run" / "plattli"
            plattli_root.mkdir(parents=True)
            (plattli_root / "plattli.json").write_text(json.dumps({
                "new": {"indices": {"start": 100, "step": 1}, "dtype": "f32"},
                "old": {"indices": [
                    {"start": 0, "stop": 100, "step": 1},
                    {"start": 100, "step": 1},
                ], "dtype": "f32"},
            }), encoding="utf-8")

            with plattli.Reader(plattli_root) as r:
                with mock.patch.object(r, "_probe_metric_count", return_value=101) as probe:
                    self.assertEqual(r.approx_max_rows(nprobes=1), 101)
                    self.assertEqual(probe.call_args.args[0], "old")

    def test_reader_approx_max_rows_validates_probe_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            plattli_root = Path(tmp) / "run" / "plattli"
            plattli_root.mkdir(parents=True)
            (plattli_root / "plattli.json").write_text("{}", encoding="utf-8")

            with plattli.Reader(plattli_root) as r:
                with self.assertRaises(TypeError):
                    r.approx_max_rows(nprobes=True)
                with self.assertRaises(TypeError):
                    r.approx_max_rows(nprobes=1.5)
                with self.assertRaises(ValueError):
                    r.approx_max_rows(nprobes=-1)

    def test_reader_approx_max_rows_indices_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            w = plattli.DirectWriter(run_root, write_threads=0)
            for step in range(3):
                w.step = step
                w.write(loss=float(step))
                w.end_step()
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.approx_max_rows(), 3)

            plattli_root = run_root / "plattli"
            zip_path = run_root / "metrics.plattli"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for path in sorted(plattli_root.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(plattli_root)
                    if path.is_symlink():
                        zf.writestr(rel.as_posix(), path.read_bytes())
                    else:
                        zf.write(path, rel)
            with plattli.Reader(run_root) as r:
                self.assertEqual(r.approx_max_rows(), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
