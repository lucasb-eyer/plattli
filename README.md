# Plättli

[![PyPI - Version](https://img.shields.io/pypi/v/plattli?logo=python&logoColor=white&color=green)](https://pypi.org/project/plattli/)
[![Tests](https://github.com/lucasb-eyer/plattli/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lucasb-eyer/plattli/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lucasb-eyer/plattli/branch/main/graph/badge.svg)](https://codecov.io/gh/lucasb-eyer/plattli)
[![PyPI - License](https://img.shields.io/pypi/l/plattli)](https://github.com/lucasb-eyer/plattli?tab=MIT-1-ov-file#readme)

Readers and writers for the Plättli metric format.
There is a fundamental issue in metric logging: reads are columnar (metrics), writes are rows (steps).
Plättli solves this by making the format on disk columnar (like parquet) with an optional row-wise "hot log" (like jsonl) for recent writes.

It consists of one file per metric (raw homogeneous array or jsonl),
plus a metrics manifest (`plattli.json`) that describes dtype and indices,
a `config.json` with info about the run, and an optional `hot.jsonl` during live logging.

At some point I will take the time to write more details about it,
but essentially it combines the best of parquet and jsonl while keeping everything very simple.

## Install

```bash
pip install plattli
```

Requires Python 3.11+ (tested on 3.11-3.14).

## CLI

A tool to convert jsonl (a common adhoc format) to plattli is provided, see

```bash
jsonl2plattli --help
```

By default it writes in-place as `<run_dir>/metrics.plattli`.
With `--outdir`, it writes `<run_name>.plattli` into the output tree.

## API

```python
from plattli import CompactingWriter, DirectWriter

w = CompactingWriter("/experiments/123456", hotsize=200, config={"lr": 3e-4, "depth": 32})
w.write(loss=1.2)  # First write creates new metric, auto-guesses dtype (float32 here)
w.write(note="ok")  # strings work too. Writes are non-blocking.
w.end_step()  # Increments step by one. Flushes hot log.

w.write(loss=1.3)  # Next write appends
# Not every metric needs to be written every step.
w.write(accuracy=0.73)
w.end_step()

# Data is written ASAP, so almost nothing is lost on crash/preemption.
del w

# If we specify a start step and destination exists,
# existing metrics will be truncated to that and we continue from there.
w = CompactingWriter("/experiments/123456", step=1, hotsize=200, config={"lr": 3e-4, "depth": 32})
w.write(loss=1.1)

# You can also write json, btw (stored as jsonl).
w.write(prediction={"qid": "42096", "answer": "Yes"})

# When finishing cleanly, we can hindsight-optimize the data for faster consumption.
# This writes /experiments/123456/metrics.plattli and removes /experiments/123456/plattli.
w.finish()

# For fast local disks, write directly to columnar files:
d = DirectWriter("/experiments/123456", config={"lr": 3e-4, "depth": 32})
d.write(loss=1.2)
d.end_step()
d.finish()
```

Note: this library is meant to be called from a single thread.
`DirectWriter` uses threads internally to be non-blocking, and `CompactingWriter` compacts in the background.
Calling `end_step` from a different thread would lead to silently inconsistent data.

### DirectWriter(outdir, step=0, write_threads=16, config="config.json", allow_resume_finalized=False)
- Prepares the writer to write under `outdir/plattli`, creating the dir and writing the config there.
- If `outdir/plattli/plattli.json` already exists, all metric files are truncated to `step` so you
  can resume a run and overwrite later data safely.
- If `outdir/metrics.plattli` exists, the constructor refuses to proceed unless
  `allow_resume_finalized=True`, which unzips into `outdir/plattli` and removes the zip.
- `write_threads=0` disables background writes.
- `config` is a dict written to `config.json`, or a string path (resolved relative to `outdir`)
  to symlink `config.json` to (default: `"config.json"`).
- If the target path does not exist, an empty config is written; pass `None` to force an empty config.

### CompactingWriter(outdir, step=0, hotsize, config="config.json", allow_resume_finalized=False)
- Hot mode: writes rows to `hot.jsonl` and compacts them into columnar files in the background.
- `hotsize` must be > 0 and sets the compaction batch size: once the hot log reaches `hotsize` completed steps, the oldest `hotsize` rows are compacted at once.
- `config` follows the same rules as `DirectWriter`.
- `allow_resume_finalized` follows the same rules as `DirectWriter`.

### DirectWriter.write(**metrics)
- Appends each metric at the current step.
- Auto-dtype rules:
  - array-like scalars -> use their dtype if supported
  - bool -> `jsonl`
  - float -> `f32`
  - int -> `i64`
  - everything else -> `jsonl`
- Force a dtype by casting the value (for example: `write(dim=np.float32(128))`).
- Only scalar values are supported (including 0-d array-likes).
- Only standard dtypes are supported for now: no bf16, nvfp4, fp8; no complex/composite.

### CompactingWriter.write(metrics=None, flush=False, **metrics)
- Appends each metric at the current step (pass a dict or kwargs).
- `flush=True` forces a `hot.jsonl` rewrite without advancing the step (use `write(flush=True)` to flush only).
- Uses the same auto-dtype rules and scalar restrictions as `DirectWriter.write`.

### end_step()
- Increments step counter by one.
- `DirectWriter` waits for all previous step writes to finish and checks for errors.
- `CompactingWriter` flushes the hot row for the current step.

### set_config(config)
- Replaces `config.json` with the provided json-dumpable config.

### finish(optimize=True, zip=True)
- `DirectWriter` flushes writes; `CompactingWriter` compacts any remaining hot rows and removes `hot.jsonl`.
- Updates `plattli.json`.
- If `optimize=True`:
  - Tightens numeric dtypes (floats -> `f32`, ints -> smallest fitting int/uint).
  - Converts monotonically spaced indices into `{start, stop, step}` and removes the `.indices` file.
  - Writes `run_rows` (max rows across metrics) into the manifest.
- If `zip=True`, zips the run folder to `<outdir>/metrics.plattli` (stored, not compressed).
- When zipping, `outdir/plattli` is removed after the zip is written.

### Reader(path)
```python
from plattli import Reader

with Reader("/experiments/123456") as r:
    print(r.metrics())
    print(r.rows("loss"), r.approx_max_rows(), r.when_exported())
    steps, values = r.metric("loss")
    step, value = r.metric("loss", idx=-1)
```

- Prefers `metrics.plattli` if present, otherwise reads the `plattli/` directory.
- Keeps zip files open until `close()` (use a `with` block or call `close()` manually).
- List all available metric names with `metrics()`.
- Read a metric with one of `metric(name, idx=None) -> (indices, values)`, `metric_indices(name)`, `metric_values(name)`, which return numpy arrays.
- Some useful metadata: `config()` returns the attached config dict; `when_exported()` is a timestamp, `rows(name)` is the exact row count (not last step!) in the given metric,
  but because `rows(name)` can be a bit expensive for in-progress runs, `approx_max_rows(faster=True)` is a fast likely-correct estimate of the row count of the most-frequent metric.
- While the data format is simple, the reader code is a bit more complex because it tolerates corrupt tails, such that it's fine to read plattli's while they are being written.

### Helpers
- `plattli.is_run(path)` -> whether the `path` is a plattli run (a correct folder structure, or a `metrics.plattli` zipfile).
- `plattli.is_run_dir(path)` -> whether the folder `path` contains plattli metrics (be it as subfolder or zipped).
- `plattli.resolve_run_dir(path)` -> resolved directory that contains `plattli.json` (returns either `path` or `path/plattli`), or `None`.

## Data format

Each run directory contains a `plattli/` folder, while the `.plattli` archive contains the same files at the top level:

```
run_dir/
  plattli/
    config.json
    plattli.json
    <metric>.indices
    <metric>.<dtype>   # or <metric>.jsonl
    hot.jsonl           # present during live logging if hotsize is enabled
  metrics.plattli
```

### Manifest (`plattli.json`)
JSON object keyed by metric name, plus metadata keys like `run_rows` and `when_exported`:

```
{
  "loss": {"indices": "indices", "dtype": "f32"},
  "note": {"indices": "indices", "dtype": "jsonl"},
  "run_rows": 1234,
  "when_exported": "2026-01-03T12:34:56Z"
}
```

Fields:
- `indices`: `"indices"`, a list of `{start, stop, step}` segments (canonical), or a single `{start, stop, step}` (legacy).
- `dtype`: one of `f{32,64}`, `{i,u}{8,16,32,64}`, or `jsonl`.
- `run_rows`: optional max rows across all metrics (written on `finish` only).
- `when_exported`: timestamp updated on manifest writes.

### Indices (`<metric>.indices`)
Raw little-endian uint32 array. Each entry is the step value for that metric
write. If `optimize=True` during `finish()`, the file may be removed and
replaced by a list of `{start, stop, step}` segments (canonical) or a single
`{start, stop, step}` (legacy) in the manifest.

### Config (`config.json`)
Arbitrary JSON object (dict), written when a config is provided.

### Values (`<metric>.<dtype>`)
Raw little-endian typed array. One scalar is appended per write call.

### JSONL values (`<metric>.jsonl`)
One JSON value per line:

```
{"event":"start"}
{"event":"done"}
```

### Metric names and subfolders
Metric names are used as file paths. A slash creates subfolders:
`detail/thing0` -> `detail/thing0.f32`.
The metric name `step` is reserved.
