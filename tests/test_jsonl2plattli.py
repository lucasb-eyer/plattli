import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from plattli._indices import _segments_to_array
from plattli.jsonl2plattli import convert_run, main
from plattli.writer import DTYPE_TO_NUMPY


def _read_indices(zf, name, indices_spec):
    if indices_spec == "indices":
        return np.frombuffer(zf.read(f"{name}.indices"), dtype=np.uint32)
    return _segments_to_array(indices_spec)


def _read_jsonl(zf, name):
    payload = zf.read(f"{name}.jsonl").decode("utf-8")
    return [json.loads(line) for line in payload.splitlines()]


class _FakePool:
    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, jobs):
        for job in jobs:
            yield func(job)


def _run_cli(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with mock.patch.object(sys, "argv", ["jsonl2plattli", *[str(arg) for arg in args]]):
            with mock.patch("plattli.jsonl2plattli.Pool", _FakePool):
                main()
    return buf.getvalue()


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

    def test_allow_rewinds_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            metrics_path = run_dir / "metrics.jsonl"
            lines = [
                {"step": 1, "loss": 1.0},
                {"step": 2, "loss": 2.0},
                {"step": 3, "loss": 3.0},
                {"step": 4, "loss": 4.0},
                {"step": 5, "loss": 5.0},
                {"step": 2, "loss": 20.0},
                {"step": 3, "loss": 20.0},
            ]
            metrics_path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

            outpath, _, _, manifest = convert_run(run_dir, run_dir, False, allow_rewinds=True)

            with zipfile.ZipFile(outpath) as zf:
                loss_idx = _read_indices(zf, "loss", manifest["loss"]["indices"])
                self.assertEqual(loss_idx.tolist(), [1, 2, 3])
                loss_dtype = manifest["loss"]["dtype"]
                loss_vals = np.frombuffer(zf.read(f"loss.{loss_dtype}"), dtype=DTYPE_TO_NUMPY[loss_dtype])
                self.assertTrue(np.allclose(loss_vals, [1.0, 20.0, 20.0]))

    def test_step_rewind_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            metrics_path = run_dir / "metrics.jsonl"
            lines = [
                {"step": 1, "loss": 1.0},
                {"step": 1, "loss": 2.0},
            ]
            metrics_path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                convert_run(run_dir, run_dir, False)
            self.assertIn("--allow-rewinds", str(ctx.exception))

    def test_cli_no_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = _run_cli([tmp])

        self.assertIn("No runs found. Nothing to export.", output)

    def test_cli_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run_dir = root / "run1"
            run_dir.mkdir(parents=True)
            metrics_path = run_dir / "metrics.jsonl"
            metrics_path.write_text(
                "".join(json.dumps(line) + "\n" for line in [{"loss": 1.0}, {"loss": 2.0}]),
                encoding="utf-8",
            )

            output = _run_cli([root, "--workers", "1"])

            self.assertTrue((run_dir / "metrics.plattli").exists())

        self.assertIn("Found 1 runs", output)
        self.assertIn("done. exported 1 runs in-place", output)

    def test_cli_outdir_skipcols_deep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "exp"
            run_dir = root / "nested" / "run1"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({"lr": 1e-3}), encoding="utf-8")
            metrics_path = run_dir / "metrics.jsonl"
            lines = [
                {"step": 0, "loss": 1.0, "pnorm": 0.5},
                {"step": 1, "loss": 2.0, "pnorm": 0.25},
            ]
            metrics_path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

            outdir = Path(tmp) / "out"
            output = _run_cli([root, "--deep", "--outdir", outdir, "--skipcols", "DEFAULT", "--workers", "1"])

            dest = outdir / root.name / run_dir.relative_to(root)
            outpath = dest / f"{dest.name}.plattli"
            self.assertTrue(outpath.exists())
            with zipfile.ZipFile(outpath) as zf:
                manifest = json.loads(zf.read("plattli.json"))

            self.assertNotIn("pnorm", manifest)

        self.assertIn("done. exported 1 runs into", output)
