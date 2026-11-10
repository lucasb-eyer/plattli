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

`DirectWriter` and `CompactingWriter` also work as context managers: `__exit__` flushes pending work, it does not `finish()` the run, so it stays resumable.

### DirectWriter(outdir, step=0, write_threads=16, config="config.json", allow_resume_finalized=False)
- Prepares the writer to write under `outdir/plattli`, creating the dir and writing the config there.
- If `outdir/plattli/plattli.json` already exists, all metric files are truncated to `step` so you
  can resume a run and overwrite later data safely.
- If `outdir/metrics.plattli` exists, the constructor refuses to proceed unless
  `allow_resume_finalized=True`, which validates the archive paths, unzips into
  `outdir/plattli`, and removes the zip.
- `write_threads=0` disables background writes.
- `config` is a dict written to `config.json`, or a string path (resolved relative to `outdir`)
  to symlink `config.json` to (default: `"config.json"`).
- If the target path does not exist, an empty config is written. Passing `None`
  preserves an existing config or writes an empty config for a new run.
- Constructor options and in-memory configuration are validated before extracting
  a finalized run or truncating existing data for resume. Config paths that would
  become directories during extraction are rejected before changing the archive.
- Recovery removes alternate representations of known metrics and recognized
  writer temporary files. Finalization also removes orphan value/index pairs.
  Unrelated extra files are preserved.

### CompactingWriter(outdir, step=0, *, hotsize, config="config.json", allow_resume_finalized=False, live_indices="auto")
- Hot mode: writes rows to `hot.jsonl` and compacts them into columnar files in the background.
- `hotsize` must be > 0 and is the compaction trigger: once the hot log holds `hotsize` completed steps, all completed rows are compacted in one background batch.
- Backpressure: writes normally never block on compaction; if the filesystem cannot keep up with the write rate, completed rows accumulate in memory (and in the hot log, so nothing is lost on crash) and batches get bigger. Once the backlog reaches 10x `hotsize`, `end_step` blocks until the in-flight batch lands, so memory stays bounded and logging degrades to filesystem speed instead of exhausting RAM.
- `config` follows the same rules as `DirectWriter`.
- `allow_resume_finalized` follows the same rules as `DirectWriter`.
- `live_indices="auto"` keeps small range lists inline. Larger lists use `.ranges`
  only when they average at least 32 samples per range; otherwise they use explicit
  `.indices`. Existing range files also fall back to explicit indices when a batch
  has fewer than 32 samples per new range it adds. A batch continuing the existing
  cadence adds no range records. Once explicit, a metric stays explicit while live.
- `live_indices="explicit"` stores compacted steps in `.indices`, including after
  resume. Use it to avoid per-metric range-file syncs during gap-heavy compaction.
- `live_indices` controls live storage only. `finish(optimize=True)` may still
  convert explicit indices to inline ranges or `.ranges` based on size. Use
  `finish(optimize=False)` to retain explicit indices.

### BulkWriter(outdir, step=0, config="config.json", overwrite=False)
- Buffers a complete run in memory and writes the columnar export on `finish()`.
- Uses the same dtype inference, checked conversions, and promotion rules as the streaming writers.
- Refuses to replace an existing `outdir/plattli` directory or `outdir/metrics.plattli`
  unless `overwrite=True`.
- Replacements are written to a staging location before publication. The previous
  output is retired, including when switching between directory and ZIP output;
  unrelated extra files are preserved. Conflicting extra files cause an error
  before publication.
- A config sourced from the output being replaced is copied before that output is retired.

### DirectWriter.write(...)
- Appends each metric at the current step (pass at most one dict or keyword metrics; the dict form is needed for slash-named metrics like `detail/thing0`).
- Auto-dtype rules:
  - array-like scalars -> use their dtype if supported
  - bool -> `jsonl`
  - float -> `f32`
  - int -> `i64`
  - explicit numpy types (eg `np.float64`) are taken as-is.
  - everything else -> `jsonl`
- Select an initial storage dtype by casting the value (for example: `write(dim=np.float32(128))`).
- Each new measurement can widen its metric's storage dtype, but never narrow it:
  `np.uint8` followed by a Python int promotes to `i64`, and `f32` followed by
  `np.float64` promotes to `f64`. Repeated `np.uint8` values stay `u8`, including
  after resuming. Python floats continue to mean `f32`; they do not narrow an
  existing `f64` metric. An integer can join a float metric if it is exactly
  representable at the current width, or after promotion to `f64`; otherwise it
  raises. Float values cannot join an integer metric.
- Integer writes are checked before casting; overflow never silently wraps.
  Signed/unsigned mixtures remain exact integers. A negative value cannot coexist
  with values above the `i64` maximum in a supported fixed-width integer column.
- Finite float values that would overflow to infinity raise. Explicit NaN and
  infinity are accepted. Promotion to `f64` cannot recover precision already lost
  when earlier values were stored as `f32`.
- Opening or resuming a writer does not widen columns. A required promotion happens
  on the next measurement, preserving retained data and pending hot rows. Promotion
  rewrites the existing column, with time and I/O proportional to its size. In
  particular, after `finish(optimize=True)` narrows an integer column, the first
  plain Python integer written on resume widens it back to `i64` and rewrites it.
- Existing `jsonl` metrics accept heterogeneous JSON-serializable values;
  numeric metrics reject strings and other nonnumeric values.
- All three writers validate every metric in a `write()` call before changing
  values or column dtypes. Invalid input leaves the batch unapplied. This does not
  make a batch transactional against filesystem failures.
- JSON values are captured during `write()`. Later changes to a supplied list or
  dictionary, including nested values, do not change the logged measurement.
- NumPy arrays and objects implementing `__array__` must be 0-d. Plain Python lists
  and dictionaries can be stored as JSON.
- Only standard dtypes are supported for now: no bf16, nvfp4, fp8; no complex/composite.
- Steps must be integers in the `uint32` range; booleans and non-integers raise
  `TypeError`, and out-of-range steps raise `ValueError`, including under `python -O`.
  NumPy integer steps are normalized to Python integers so
  incrementing cannot wrap. Constructor checks precede any resume or overwrite
  operations. The last valid step can still be ended and finalized.
  TODO: automatically promote step storage and returned indices to `uint64` when
  this range is exceeded.

### CompactingWriter.write(..., flush=False)
- Appends each metric at the current step (pass at most one dict or keyword metrics).
- `flush=True` forces a `hot.jsonl` rewrite without advancing the step (use `write(flush=True)` to flush only).
- Uses the same auto-dtype rules and scalar restrictions as `DirectWriter.write`.
- Pending rows are checked for valid steps before publication, including explicit
  flushes, `end_step()`, `finish()`, and context-manager exit.

### end_step()
- Increments step counter by one.
- `DirectWriter` waits for all previous step writes to finish and checks for errors.
- `CompactingWriter` flushes the hot row for the current step.

### set_config(config)
- Streaming writers atomically replace `config.json` with the provided json-dumpable
  config. Replacing a linked config leaves the original link target unchanged.
- `BulkWriter` buffers the config until `finish()`.

### finish(optimize=True, zip=True, config_overrides=None)
- `DirectWriter` flushes writes; `CompactingWriter` compacts any remaining hot rows and removes `hot.jsonl`.
- Updates `plattli.json`.
- If `optimize=True`:
  - Tightens numeric dtypes (floats -> keep original float width, ints -> smallest fitting int/uint).
  - Stores regular indices as inline ranges or `.ranges` files, whichever fits; irregular indices stay explicit.
  - Writes `run_rows` (max rows across metrics) into the manifest.
- If `zip=True`, zips the run folder to `<outdir>/metrics.plattli` (stored, not compressed).
- When zipping, `outdir/plattli` is removed after the zip is written.
- Shallow-merges `config_overrides` into the finalized `config.json`; override values win.
- Does not modify the original config dict or linked config file.

### Reader(path, kind=None, exclude_hot=False)
```python
from plattli import Reader

with Reader("/experiments/123456") as r:
    print(r.metrics())
    print(r.rows("loss"), r.approx_max_rows(), r.when_exported())
    steps, values = r.metric("loss")
    step, value = r.metric("loss", idx=-1)
```

Callers that already know the exact storage path can pass `kind="dir"` for a `plattli/` directory or `kind="zip"` for a `.plattli` archive. This bypasses filesystem discovery, so the path and kind must be trusted.

Each `Reader` is intended for one thread. Use separate instances or an external
lock when reading from multiple threads.

Pass `exclude_hot=True` to read only already-compacted columnar data. This can
substantially reduce I/O and parsing work when a live run's metric/column count
makes `hot.jsonl` very wide, and exact recency is less important than read
speed. The default includes the hot tail.

- Prefers `metrics.plattli` if present, otherwise reads the `plattli/` directory.
- Keeps zip files open until `close()` (use a `with` block or call `close()` manually).
- List all available metric names with `metrics()`.
- Read a metric with one of `metric(name, idx=None) -> (indices, values)`, `metric_indices(name)`, `metric_values(name)`, which return numpy arrays.
- Metric value reads default to `dtype="standard"`: integers become checked `int64`,
  while `f32` and `f64` retain their width. JSON values keep their object dtype.
  Step indices remain `uint32`, independently of the value dtype.
- Pass keyword-only `dtype="storage"` to return the stored value dtype, or an actual
  NumPy dtype such as `dtype=np.float64` to request a conversion. These overrides
  apply to `metric`, `metric_values`, and `table`; `metric_indices` always returns
  `uint32`. `manifest()[name]["dtype"]` describes storage, not the default return type.
- Conversions happen after selection and alignment, including empty and single-row
  reads. Scalar reads return the matching NumPy scalar. Integer conversions reject
  overflow and fractional truncation; explicit float narrowing uses NumPy rounding.
- Stored `u64` values above the `int64` maximum raise on standard reads. Select
  `dtype=np.uint64`, `dtype=object`, or `dtype="storage"` explicitly for these values.
- Returned arrays may be read-only. Use `.copy()` when you need to modify them.
- An integer `idx` (like `idx=-1` for the latest value) reads just that one row, without loading the whole column.
- Some useful metadata: `config()` returns the attached config dict; `when_exported()` is a timestamp, and `rows(name)` is the exact row count (not last step!) in the given metric.
- `approx_max_rows(nprobes=12)` cheaply estimates the row count of the most-frequent metric. Closed index specs and finalized `run_rows` metadata need no probes; otherwise it checks at most `nprobes` explicit-index or open numeric columns, distributed across their index cadences. The result becomes exact for stable columnar data when the budget covers every candidate, but may otherwise underestimate; hot rows and open JSONL columns are excluded.
- While the data format is simple, the reader code is a bit more complex because it tolerates corrupt tails, such that it's fine to read plattli's while they are being written.
- Metadata (manifest, config, hot rows, row counts, jsonl values) is cached on first use, while raw data files are read fresh on every call. On a long-lived `Reader` of a live run, call `refresh()` to drop the caches and pick up new metrics and hot rows. Zip readers are immutable snapshots.
- If a live storage promotion invalidates a read's cached dtype, or a read encounters
  a new hot metric missing from its cached manifest, the reader refreshes and retries
  that read. It also retries when a fresh hot snapshot reveals that a previously
  hot-only metric has moved into columnar storage. For `table()`, this restarts the
  whole table. This does not guarantee an atomic snapshot of concurrent writes or
  replace `refresh()` for ordinary appended data and guaranteed visibility of new metrics.

### Aligned reads: table()

`table(names, on="step", **selectors)` reads several metrics step-aligned, keeping only
steps present in every requested metric (inner join on steps). It returns
`(steps, {name: values})` with all arrays of equal length.

Selectors (see below) select rows of the `on` column: with the default `on="step"` they
apply to the aligned table itself, while `on="some_metric"` selects that metric's rows —
including by value via `vstart`/`vstop` — and the other columns follow. Columns other
than `on` are only read within the selected step window, so zoomed reads stay cheap even
when metrics were logged at different cadences.

```python
import numpy as np

with Reader("/experiments/123456") as r:
    steps, cols = r.table(["loss", "accuracy"])                    # aligned full read
    steps, cols = r.table(["walltime", "loss"], on="walltime", vstart=10.0, vstop=20.0)
    x, y = cols["walltime"], cols["loss"]

    # Override one metric's dtype; all other columns use "standard".
    steps, cols = r.table(["loss", "tokens"], dtype={"loss": np.float64})
```

`table(..., dtype=...)` accepts either one dtype for every requested metric or a
mapping from metric names to dtypes. It never changes the dtype of the returned steps.
`names` can be any nonempty iterable, including a generator.

### Advanced API topics

#### Range selectors
Range selectors can be passed to any metric read:
- `start`/`stop` read this range of **step values**. `stop` is inclusive, like label slicing with pandas `.loc`.
- `vstart`/`vstop` read this range of **metric values**. `vstop` is inclusive. Mostly useful for monotonic metrics.
- `istart`/`istop` read this range of **physical row positions**. `istop` is exclusive and negative positions count from the end, like Python slices (`istart=-100` reads the last 100 rows).

These cannot be mixed.

```python
from plattli import Reader

with Reader("/experiments/123456") as r:
    zoomed_loss_steps, zoomed_loss_values = r.metric("loss", start=100, stop=200)
    x_steps, x_values = r.metric("walltime", vstart=10.0, vstop=20.0)
```

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
    <metric>.indices  # or <metric>.ranges; omitted for inline ranges
    <metric>.<dtype>   # or <metric>.jsonl
    hot.jsonl           # present during live logging if hotsize is enabled
    hot.compacting.jsonl  # transient: rows being compacted right now; unlinked when done
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
- `indices`: `"indices"`, `"ranges"`, a list of `{start, stop, step}` segments, or a single segment (legacy). A final inline segment may omit `stop`; readers derive its length from the values.
- `dtype`: storage type, one of `f{32,64}`, `{i,u}{8,16,32,64}`, or `jsonl`.
  Integer compaction changes this field without changing standard reader output.
- `monotonic`: optional `"inc"` or `"dec"` for numeric metrics whose stored values are monotonic; flat-only metrics use `"inc"`.
- `run_rows`: optional max rows across all metrics (written on `finish` only).
- `when_exported`: timestamp updated on manifest writes.

### Indices (`<metric>.indices`)
Raw little-endian uint32 array, one step per measurement. Small range lists
stay inline (at most 32 segments). With `finish(optimize=True)`, larger lists
use `.ranges` when smaller than explicit indices. Live `CompactingWriter`
selection uses the stricter 32-sample rule described above, unless
`live_indices="explicit"`. `DirectWriter` uses explicit indices while live.

### Ranges (`<metric>.ranges`)
Little-endian `(start: uint32, step: uint32, end_row: uint64)` records (16 bytes).
`end_row` is the cumulative sample count: `(0, 2, 3), (10, 2, 5)` represents
steps `[0, 2, 4, 10, 12]`. A final live record may omit `end_row` (8 bytes);
its length comes from the values. Closing it only appends `end_row`, so
continuing the same cadence needs no index writes. `finish()` closes open tails.

Convert existing files with `python -m plattli.optimize_indices PATH`.

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
Names must be non-empty relative paths. Absolute paths, backslashes, NULs, empty or
`.`/`..` path components, and non-string names are rejected. The following names are
reserved: `step`, `run_rows`, `when_exported`, `hot`, and `hot.compacting`.
