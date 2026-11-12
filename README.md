# Plättli

[![PyPI - Version](https://img.shields.io/pypi/v/plattli?logo=python&logoColor=white&color=green)](https://pypi.org/project/plattli/)
[![Tests](https://github.com/lucasb-eyer/plattli/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lucasb-eyer/plattli/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lucasb-eyer/plattli/branch/main/graph/badge.svg)](https://codecov.io/gh/lucasb-eyer/plattli)
[![PyPI - License](https://img.shields.io/pypi/l/plattli)](https://github.com/lucasb-eyer/plattli?tab=MIT-1-ov-file#readme)

Readers and writers for the Plättli metric format.
There is a fundamental issue in metric logging: reads are columnar (metrics), writes are rows (steps).
Plättli solves this by making the format on disk columnar (like parquet) with an optional row-wise "hot log" (like jsonl) for recent writes.

It consists of per-metric values (raw homogeneous arrays, jsonl, or inline constants and integer sequences),
plus a metrics manifest (`plattli.json`) that describes dtype and indices,
a `config.json` with info about the run, and an optional `hot.jsonl` during live logging.

Essentially, it combines the best of parquet and jsonl while things simple and optimizing for common ML-related patterns under the hood.

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
- Writes under `outdir/plattli`, creating the directory and writing the config there.
- If a run already exists there, all metrics are truncated to `step` so you
  can resume a run and overwrite later data safely.
- If `outdir/metrics.plattli` exists (a finished run), the constructor refuses to proceed unless
  `allow_resume_finalized=True`, which unzips it into `outdir/plattli` and removes the zip.
- `write_threads=0` disables background writes.
- `config` is a dict written to `config.json`, or a string path (resolved relative to `outdir`)
  to symlink `config.json` to (default: `"config.json"`).
- If the target path does not exist, an empty config is written. Passing `None`
  preserves an existing config or writes an empty config for a new run.

### CompactingWriter(outdir, step=0, *, hotsize, config="config.json", allow_resume_finalized=False, live_indices="auto")
- Hot mode: writes rows to `hot.jsonl` and compacts them into columnar files in the background.
- `hotsize` must be > 0 and is the compaction trigger: once the hot log holds `hotsize` completed steps, all completed rows are compacted in one background batch.
- Backpressure: writes normally never block on compaction. If the filesystem cannot keep up, completed rows accumulate in memory (and in the hot log, so nothing is lost on crash). Once the backlog reaches 10x `hotsize`, `end_step` blocks until the in-flight batch lands, so memory stays bounded.
- `config` and `allow_resume_finalized` follow the same rules as `DirectWriter`.
- `live_indices="auto"` (default) lets the writer choose a compact encoding for each metric's steps.
  `live_indices="explicit"` always stores plain step arrays while live; use it to avoid
  per-metric range-file syncs during gap-heavy compaction. See [Indices](#indices-metricindices).

### BulkWriter(outdir, step=0, config="config.json", overwrite=False)
- Buffers a complete run in memory and writes the columnar export on `finish()`.
- Uses the same dtype inference, checked conversions, and promotion rules as the streaming writers.
- Refuses to replace an existing `outdir/plattli` directory or `outdir/metrics.plattli`
  unless `overwrite=True`.

### DirectWriter.write(...)
- Appends each metric at the current step (pass at most one dict or keyword metrics; the dict form is needed for slash-named metrics like `detail/thing0`).
- Auto-dtype rules:
  - array-like scalars -> use their dtype if supported
  - bool -> `jsonl`
  - float -> `f32`
  - int -> `i64`; Python ints outside the signed 64-bit range -> exact `int`, with restrictions (see [Big integers](#big-integers))
  - explicit numpy types (eg `np.float64`) are taken as-is.
  - everything else -> `jsonl`
- Select an initial storage dtype by casting the value (for example: `write(dim=np.float32(128))`).
- Later values can widen a metric's dtype, but never narrow it. Conversions that would
  lose or wrap a value raise instead (see [Dtypes and promotion](#dtypes-and-promotion)).
- Values must be scalars: NumPy arrays and objects implementing `__array__` must be 0-d.
  Plain Python lists and dictionaries are stored as JSON.
- Existing `jsonl` metrics accept heterogeneous JSON-serializable values;
  numeric metrics reject strings and other nonnumeric values.
- Every metric in a `write()` call is validated before anything is applied;
  invalid input leaves the whole call unapplied.
- Constant and evenly spaced metrics are stored compactly without any configuration,
  so logging a constant every step is cheap.
- Only standard dtypes are supported for now: no bf16, nvfp4, fp8; no complex/composite.
- Steps must be integers in the `uint32` range.

### CompactingWriter.write(..., flush=False)
- Appends each metric at the current step (pass at most one dict or keyword metrics).
- `flush=True` forces a `hot.jsonl` rewrite without advancing the step (use `write(flush=True)` to flush only).
- Uses the same auto-dtype rules and scalar restrictions as `DirectWriter.write`.

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
  - Re-encodes steps and values in their most compact form (see [Data format](#data-format)).
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

- Prefers `metrics.plattli` if present, otherwise reads the `plattli/` directory.
  Pass `kind="dir"` or `kind="zip"` when `path` already is the exact `plattli/` directory
  or `.plattli` archive; this skips discovery, so the path and kind must be trusted.
- `exclude_hot=True` reads only already-compacted data. This is faster on live runs
  with very many metrics, at the cost of missing the most recent steps.
- Keeps zip files open until `close()` (use a `with` block or call `close()` manually).
- Each `Reader` is intended for one thread. Use separate instances or an external
  lock when reading from multiple threads.
- It is fine to read a run while it is being written. On a long-lived `Reader` of a
  live run, call `refresh()` to pick up new data (see [Reading live runs](#reading-live-runs)).
- List all available metric names with `metrics()`.
- Read a metric with one of `metric(name, idx=None) -> (indices, values)`, `metric_indices(name)`, `metric_values(name)`, which return numpy arrays.
- An integer `idx` (like `idx=-1` for the latest value) reads just that one row, without loading the whole column.
- Values default to `dtype="standard"`: integers are returned as `int64`, floats keep
  their width, JSON values are object arrays. Steps are always `uint32`.
- Pass keyword-only `dtype="storage"` to get the stored dtype, or a NumPy dtype such
  as `dtype=np.float64` to convert. This applies to `metric`, `metric_values`, and `table`.
  See [Dtypes and promotion](#dtypes-and-promotion) for the conversion rules.
- Returned arrays may be read-only. Use `.copy()` when you need to modify them.
- Some useful metadata: `config()` returns the attached config dict; `when_exported()` is a timestamp, and `rows(name)` is the exact row count (not last step!) in the given metric.
- `approx_max_rows(nprobes=12)` cheaply estimates the row count of the most-frequent metric. It may underestimate and excludes hot rows.

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

## Behavior details

The exact rules behind the API above. Skip this unless you hit an edge case.

### Dtypes and promotion
Writing:
- Each new measurement can widen its metric's storage dtype, but never narrow it:
  `np.uint8` followed by a Python int promotes to `i64`, and `f32` followed by
  `np.float64` promotes to `f64`. Repeated `np.uint8` values stay `u8`, including
  after resuming. Python floats continue to mean `f32`; they do not narrow an
  existing `f64` metric.
- An integer can join a float metric if it is exactly representable at the current
  width, or after promotion to `f64`; otherwise it raises. Float values cannot join
  an integer metric.
- Integer writes are checked before casting; overflow never silently wraps.
  Signed/unsigned mixtures remain exact integers. A negative value cannot coexist
  with values above the `i64` maximum in a fixed-width integer column.
- Finite float values that would overflow to infinity raise. Explicit NaN and
  infinity are accepted. Promotion to `f64` cannot recover precision already lost
  when earlier values were stored as `f32`.
- Opening or resuming a writer does not widen columns. A required promotion happens
  on the next measurement, preserving retained data and pending hot rows. Promotion
  rewrites raw columns, with time and I/O proportional to their size; inline
  values usually need only a metadata update. In particular, after
  `finish(optimize=True)` narrows an integer column, the first plain Python integer
  written on resume widens it back to `i64`.

Reading:
- `dtype="standard"` returns integers as checked `int64`, while `f32` and `f64`
  retain their width. `manifest()[name]["dtype"]` describes storage, not the
  default return type. `metric_indices` always returns `uint32`.
- Conversions happen after selection and alignment, including empty and single-row
  reads. Scalar reads return the matching NumPy scalar. Integer conversions reject
  overflow and fractional truncation; explicit float narrowing uses NumPy rounding.
- Stored `u64` values above the `int64` maximum raise on standard reads. Select
  `dtype=np.uint64`, `dtype=object`, or `dtype="storage"` explicitly for these values.

### Big integers
- Python integers outside the signed 64-bit range are stored exactly with the `int`
  dtype. Use `10**20` for an exact Python integer; `1e20` is a float.
- `int` metrics must be constant or grow by a fixed amount, either per measurement
  or per logging step (`base + step * increment`). For example, logging
  `10**20 + step * 10**6` every N steps plus once at the final step works.
- An off-pattern value raises `ValueError` before the write is applied.
  Irregular big integers are not yet supported.
- An existing integer metric can promote to `int` only when its retained values and
  the new value together form such a sequence.
- Reads return object arrays of Python integers with both `dtype="standard"` and
  `dtype="storage"`. Explicit fixed-width integer casts check bounds after selection.
  Explicit float casts may lose precision.

### Write validation
- All three writers validate every metric in a `write()` call before changing
  values or column dtypes. Invalid input leaves the batch unapplied. This does not
  make a batch transactional against filesystem failures.
- JSON values are captured during `write()`. Later changes to a supplied list or
  dictionary, including nested values, do not change the logged measurement.
- Booleans and non-integer steps raise `TypeError`, and out-of-range steps raise
  `ValueError`, including under `python -O`. NumPy integer steps are normalized to
  Python integers so incrementing cannot wrap. The last valid step can still be
  ended and finalized.
  TODO: automatically promote step storage and returned indices to `uint64` when
  this range is exceeded.
- `CompactingWriter` checks pending rows for valid steps before publication,
  including explicit flushes, `end_step()`, `finish()`, and context-manager exit.

### Resume and recovery
- Constructor options and in-memory configuration are validated before extracting
  a finalized run or truncating existing data for resume. Config paths that would
  become directories during extraction are rejected before changing the archive.
- `allow_resume_finalized=True` validates the archive paths before unzipping.
- Recovery removes alternate representations of known metrics and recognized
  writer temporary files. Finalization also removes orphan value/index pairs.
  Unrelated extra files are preserved.
- `BulkWriter` replacements are written to a staging location before publication.
  The previous output is retired, including when switching between directory and
  ZIP output; unrelated extra files are preserved. Conflicting extra files cause
  an error before publication. A config sourced from the output being replaced is
  copied before that output is retired.

### Reading live runs
- While the data format is simple, the reader code is a bit more complex because it
  tolerates corrupt tails, such that it's fine to read plattli's while they are being written.
- Metadata (manifest, config, hot rows, row counts, jsonl values, inline values) is
  cached on first use, while raw data files are read fresh on every call. On a
  long-lived `Reader` of a live run, call `refresh()` to drop the caches and pick up
  new metrics and hot rows. Zip readers are immutable snapshots.
- If a live storage promotion invalidates a read's cached dtype, or a read encounters
  a new hot metric missing from its cached manifest, the reader refreshes and retries
  that read. It also retries when a fresh hot snapshot reveals that a previously
  hot-only metric has moved into columnar storage. For `table()`, this restarts the
  whole table. This does not guarantee an atomic snapshot of concurrent writes or
  replace `refresh()` for ordinary appended data and guaranteed visibility of new metrics.
- `exclude_hot=True` can substantially reduce I/O and parsing work when a live run's
  metric count makes `hot.jsonl` very wide, and exact recency is less important
  than read speed.
- `approx_max_rows(nprobes=12)`: closed index specs and finalized `run_rows`
  metadata need no probes; otherwise it checks at most `nprobes` explicit-index or
  open numeric columns, distributed across their index cadences. The result becomes
  exact for stable columnar data when the budget covers every candidate, but may
  otherwise underestimate; hot rows and open JSONL columns are excluded.

## Data format

Each run directory contains a `plattli/` folder, while the `.plattli` archive contains the same files at the top level:

```
run_dir/
  plattli/
    config.json
    plattli.json
    <metric>.indices  # or <metric>.ranges; omitted for inline ranges
    <metric>.<dtype>   # or <metric>.jsonl; omitted for inline values
    <metric>.values    # small sequence head during DirectWriter live logging
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
- `dtype`: storage type, one of `f{32,64}`, `{i,u}{8,16,32,64}`, `int`, or `jsonl`.
  Integer compaction changes this field without changing standard reader output.
- `values`: optional inline values replacing the value file, see
  [Inline values](#inline-values-values-in-the-manifest).
- `monotonic`: optional `"inc"` or `"dec"` for numeric metrics whose stored values are monotonic; flat-only metrics use `"inc"`.
- `run_rows`: optional max rows across all metrics (written on `finish` only).
- `when_exported`: timestamp updated on manifest writes.

### Indices (`<metric>.indices`)
Raw little-endian uint32 array, one step per measurement.

Regular steps are stored more compactly, as inline segments in the manifest or as a
`.ranges` file. Writers choose as follows:
- Small range lists stay inline (at most 32 segments).
- With `finish(optimize=True)`, larger lists use `.ranges` when smaller than
  explicit indices; irregular indices stay explicit. Use `finish(optimize=False)`
  to retain the live encoding.
- Live `CompactingWriter` with `live_indices="auto"` uses `.ranges` for larger lists
  only when they average at least 32 samples per range; otherwise explicit
  `.indices`. Existing range files also fall back to explicit indices when a batch
  has fewer than 32 samples per new range it adds. A batch continuing the existing
  cadence adds no range records. Once explicit, a metric stays explicit while live.
- Live `CompactingWriter` with `live_indices="explicit"` stores compacted steps in
  `.indices`, including after resume.
- `DirectWriter` uses explicit indices while live.

### Ranges (`<metric>.ranges`)
Little-endian `(start: uint32, step: uint32, end_row: uint64)` records (16 bytes).
`end_row` is the cumulative sample count: `(0, 2, 3), (10, 2, 5)` represents
steps `[0, 2, 4, 10, 12]`. A final live record may omit `end_row` (8 bytes);
its length comes from the values. Closing it only appends `end_row`, so
continuing the same cadence needs no index writes. `finish()` closes open tails.

Convert existing files with `python -m plattli.optimize_indices PATH`.

### Values (`<metric>.<dtype>`)
Raw little-endian typed array. One scalar is appended per write call.
Omitted when the manifest contains `values`.

### Inline values (`values` in the manifest)
A `{start, step, count}` arithmetic sequence that replaces the value file:
- The value at physical row `i` is `start + i * step`. `step: 0` denotes a constant;
  decreasing sequences use a negative step.
- Integer sequences may also include `"on": "step"`: the value is then
  `start + logging_step * step`, with `start` the intercept at logging step zero.
- Floats require a zero step and repeat the stored `start` exactly, including its
  zero sign.
- The explicit `count` also bounds live open index ranges.
- The `int` dtype exists only in this form and uses arbitrary-size JSON integer
  components; read the manifest with a parser that preserves integer precision.

For example:

```json
{
  "capacity": {
    "indices": [{"start": 0, "stop": 1000, "step": 10}],
    "dtype": "int",
    "values": {"start": 100000000000000000000, "step": 1000000, "count": 100}
  }
}
```

This represents 100 exact integer values sampled at steps `0, 10, ..., 990`,
without either a value file or an index file.

When writers use it:
- Streaming numeric metrics start inline and stay inline while the pattern holds,
  also with `finish(optimize=False)`. Integer metrics detect both the row-based and
  the step-based form.
- When the pattern breaks, fixed-width metrics switch to a raw array (`int` metrics
  raise instead). For floats this is the first value with different stored bits,
  including a change between positive and negative zero. Nonfinite floats always
  use raw arrays.
- `finish(optimize=True)` converts raw constants and integer sequences to inline values.
- `BulkWriter` inspects fixed-width values once at finish.

### Live value sequences (`<metric>.values`)
During live `DirectWriter` logging the manifest holds `"values": "sequence"`, which
points to a small JSON head such as
`{"dtype": "int", "values": {"start": 100000000000000000000, "step": 1000000}}`.
The head omits `count`; its length comes from the metric's explicit `.indices`
file. Readers capture the index count before reading the head, and read both fresh
on every call. Continuing the
same sequence writes only the index, without rewriting either the head or the
run manifest. Integer coefficient changes replace the head. Before switching to a raw
array the writer closes the old head with a finite `count`, publishes the new
manifest, then removes the head. Finalization replaces live heads with finite
inline sequences.

### JSONL values (`<metric>.jsonl`)
One JSON value per line:

```
{"event":"start"}
{"event":"done"}
```

### Config (`config.json`)
Arbitrary JSON object (dict), written when a config is provided.

### Metric names and subfolders
Metric names are used as file paths. A slash creates subfolders:
`detail/thing0` -> `detail/thing0.f32`.
Names must be non-empty relative paths. Absolute paths, backslashes, NULs, empty or
`.`/`..` path components, and non-string names are rejected. The following names are
reserved: `step`, `run_rows`, `when_exported`, `hot`, and `hot.compacting`.
