import tempfile
import unittest
from pathlib import Path

import numpy as np

from plattli import CompactingWriter, DirectWriter, Reader
from plattli.writer import _zip_path_for_root


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

    def test_reader_rows_hot_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = CompactingWriter(run_root, hotsize=100)
            w.write(loss=1.0, acc=0.5)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.write(loss=3.0, acc=0.7)
            w.end_step()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)
                self.assertEqual(r.rows("acc"), 2)

    def test_reader_rows_columnar_plus_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = CompactingWriter(run_root, hotsize=2)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            if w._compact_future:
                w._compact_future.result()
            w.write(loss=3.0)
            w.end_step()

            self.assertTrue((plattli_root / "hot.jsonl").exists())
            with Reader(run_root) as r:
                self.assertEqual(r.rows("loss"), 3)

    def test_reader_tolerates_partial_numeric_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0)
            w.end_step()
            w.write(loss=2.0)
            w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(b"\x01\x02")
            with (plattli_root / "loss.f32").open("ab") as fh:
                fh.write(b"\x03\x04")

            with Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1])
                self.assertTrue(np.allclose(r.metric_values("loss"),
                                            np.asarray([1.0, 2.0], dtype=np.float32)))
                self.assertEqual(r.rows("loss"), 2)

    def test_reader_tolerates_mismatch_and_jsonl_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            plattli_root = run_root / "plattli"
            w = DirectWriter(run_root, write_threads=0)
            w.write(loss=1.0, text="a")
            w.end_step()
            w.write(loss=2.0, text="b")
            w.end_step()
            w.finish(optimize=False, zip=False)

            with (plattli_root / "loss.indices").open("ab") as fh:
                fh.write(np.asarray([2], dtype=np.uint32).tobytes())
            with (plattli_root / "text.jsonl").open("ab") as fh:
                fh.write(b"\"c")

            with Reader(run_root) as r:
                self.assertEqual(r.metric_indices("loss").tolist(), [0, 1])
                self.assertTrue(np.allclose(r.metric_values("loss"),
                                            np.asarray([1.0, 2.0], dtype=np.float32)))
                self.assertEqual(r.metric_values("text").tolist(), ["a", "b"])
                self.assertEqual(r.rows("text"), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
