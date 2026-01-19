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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
