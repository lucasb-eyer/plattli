# Plättli

[![Tests](https://github.com/lucasb-eyer/plattli/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lucasb-eyer/plattli/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/lucasb-eyer/plattli/branch/main/graph/badge.svg)](https://codecov.io/gh/lucasb-eyer/plattli)

Minimal streaming writer for the Plättli metric format.
The format is very simple, and allows for efficient appending, reading and slicing.
It consists of one file per metric, which is just a raw homogeneous array,
plus a metrics manifest (`plattli.json`) that describes dtype and indices,
and a `config.json` with info about the run.

## Install

```bash
pip install plattli
```

Requires Python 3.11+.

## CLI

A tool to convert jsonl (a common adhoc format) to plattli is provided, see

```bash
jsonl2plattli --help
```

## API

```python
from plattli import PlattliWriter

w = PlattliWriter("/experiments/123456", config={"lr": 3e-4, "depth": 32})
w.write(loss=1.2)  # First write creates new metric, auto-guesses dtype (float32 here)
w.write(note="ok")  # strings work too. Writes are non-blocking.
w.end_step()  # Increments step by one. Makes sure previous writes are flushed.

w.write(loss=1.3)  # Next write appends
# Not every metric needs to be written every step.
w.write(accuracy=0.73)
w.end_step()

# Data is written ASAP, so almost nothing is lost on crash/preemption.
del w

# If we specify a start step and destination exists,
# existing metrics will be truncated to that and we continue from there.
w = PlattliWriter("/experiments/123456", step=1, config={"lr": 3e-4, "depth": 32})
w.write(loss=1.1)

# You can also write json, btw.
w.write(prediction={"qid": "42096", "answer": "Yes"})

# When finishing cleanly, we can hindsight-optimize the data for faster consumption
w.finish()
```

Note: this library is meant to be called from a single thread.
`write` uses threads internally to be non-blocking as it's meant to be used on the critical path,
but calling `end_step` from a different thread would lead to silently inconsistent data.

### PlattliWriter(outdir, step=0, write_threads=16, config=None)
- Prepares the writer to write to outdir, creating the dir and writing the config there.
- If `plattli.json` already exists, all metric files are truncated to `step` so you
  can resume a run and overwrite later data safely.
- `write_threads=0` disables background writes.
- `config` is a dict written to `config.json` (empty dict by default).

### write(**metrics)
- Appends each metric at the current step.
- Auto-dtype rules:
  - array-like scalars -> use their dtype if supported
  - bool -> `json`
  - float -> `f32`
  - int -> `i64`
  - everything else -> `json`
- Force a dtype by casting the value (for example: `write(dim=np.float32(128))`).
- Only scalar values are supported (including 0-d array-likes).
- Only standard dtypes are supported for now: no bf16, nvfp4, fp8; no complex/composite.

### end_step()
- Increments step counter by one.
- Waits for all previous step writes to finish and checks for errors.
- This could also be made non-blocking with a bit more effort, but let's first keep things simple.

### set_config(config)
- Replaces `config.json` with the provided json-dumpable config.

### finish(optimize=True, zip=True)
- Flushes writes and updates `plattli.json`.
- If `optimize=True`:
  - Tightens numeric dtypes (floats -> `f32`, ints -> smallest fitting int/uint).
  - Converts monotonically spaced indices into `{start, stop, step}` and removes the `.indices` file.
  - Writes `run_rows` (max rows across metrics) into the manifest.
- If `zip=True`, zips the run folder to `<outdir>.zip` (stored, not compressed).
- When zipping, the original run folder is removed after the zip is written.

## Data format

Each run directory or zip contains:

```
run_dir/
  config.json
  plattli.json
  <metric>.indices
  <metric>.<dtype>   # or <metric>.json
```

### Manifest (`plattli.json`)
JSON object keyed by metric name, plus metadata keys like `run_rows` and `when_exported`:

```
{
  "loss": {"indices": "indices", "dtype": "f32"},
  "note": {"indices": "indices", "dtype": "json"},
  "run_rows": 1234,
  "when_exported": "2026-01-03T12:34:56Z"
}
```

Fields:
- `indices`: `"indices"` or `{start, stop, step}`.
- `dtype`: one of `f{32,64}`, `{i,u}{8,16,32,64}`, or `json`.
- `run_rows`: optional max rows across all metrics (written on `finish` only).
- `when_exported`: timestamp updated on manifest writes.

### Indices (`<metric>.indices`)
Raw little-endian uint32 array. Each entry is the step value for that metric
write. If `optimize=True` during `finish()`, the file may be removed and
replaced by `{start, stop, step}` in the manifest.

### Config (`config.json`)
Arbitrary JSON object (dict), written when a config is provided.

### Values (`<metric>.<dtype>`)
Raw little-endian typed array. One scalar is appended per write call.

### JSON values (`<metric>.json`)
JSON array of values, still valid JSON, but written with newlines:

```
[
{"event":"start"},
{"event":"done"}
]
```

### Metric names and subfolders
Metric names are used as file paths. A slash creates subfolders:
`detail/thing0` -> `detail/thing0.f32`.
