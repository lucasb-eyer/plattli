import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from plattli._indices import _segments_to_array
from plattli.jsonl2plattli import convert_run
from plattli.writer import DTYPE_TO_NUMPY


def _read_indices(zf, name, indices_spec):
    if indices_spec == "indices":
        return np.frombuffer(zf.read(f"{name}.indices"), dtype=np.uint32)
    return _segments_to_array(indices_spec)


def _read_jsonl(zf, name):
    payload = zf.read(f"{name}.jsonl").decode("utf-8")
    return [json.loads(line) for line in payload.splitlines()]


class TestJsonl2Plattli(unittest.TestCase):
    def test_step_column_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            metrics_path = run_dir / "metrics.jsonl"
            lines = [
                {"step": 2, "loss": 1.0},
                {"step": 5, "loss": 2.0, "note": "ok"},
            ]
            metrics_path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

            outpath, nrows, ncols, manifest = convert_run(run_dir, run_dir, False)

            self.assertEqual(nrows, 2)
            self.assertEqual(ncols, 2)
            self.assertNotIn("step", manifest)
            with zipfile.ZipFile(outpath) as zf:
                loss_idx = _read_indices(zf, "loss", manifest["loss"]["indices"])
                self.assertEqual(loss_idx.tolist(), [2, 5])
                loss_dtype = manifest["loss"]["dtype"]
                loss_vals = np.frombuffer(zf.read(f"loss.{loss_dtype}"), dtype=DTYPE_TO_NUMPY[loss_dtype])
                self.assertTrue(np.allclose(loss_vals, [1.0, 2.0]))

                note_idx = _read_indices(zf, "note", manifest["note"]["indices"])
                self.assertEqual(note_idx.tolist(), [5])
                self.assertEqual(_read_jsonl(zf, "note"), ["ok"])
