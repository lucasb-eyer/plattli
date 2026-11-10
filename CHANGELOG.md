# Changelog

All notable changes to this project are documented in this file.

## Unreleased
- Fixed resume dropping later segments when a preceding segment's canonical stop
  extends past the next segment's start.
- Fixed reader gaps during hot-log rotation while preserving active-row precedence.

## 0.19.0 - 2026-09-22
- Breaking API change: metric reads now default to signed `int64` for integers,
  independently of compact storage. Float widths and `uint32` steps are unchanged.
  Per-read `dtype` overrides select storage types or explicit NumPy types;
  `table` also accepts per-column overrides.
- Writers lazily widen storage when new measurements request wider types, including
  after resume. Matching explicit narrow types stay narrow across preemption.
  Integer conversions reject overflow, and signed/unsigned promotion stays exact.
- BulkWriter now follows the streaming writers' dtype rules, preserves explicit
  `float64`, and rejects float inputs into integer metrics instead of silently casting.
- Integer inputs into float metrics must be exactly representable at the current
  width or after promotion to `float64`. Finite float overflow raises; explicit NaN
  and infinity remain supported.
- All writers validate complete input batches before changing metric data or dtypes.
  Step-range validation remains active under `python -O` and precedes resume or
  overwrite operations.
- Aligned table reads retry the entire table on observed storage dtype changes and
  preserve generator inputs across validation and retries.
- Constructor option and configuration validation precedes resume side effects;
  compacting writers validate pending steps on every publication path.
- Steps reject non-integers and booleans; NumPy integer steps cannot wrap on increment.
- Bulk overwrites stage replacement output, retire the previous representation, and
  preserve unrelated files and configs sourced from the previous output.
- Finalized resume validates config paths against archive directories before extraction.
  Config replacement is atomic and leaves external config targets unchanged.
- JSON measurements are snapshotted during `write()`, so caller mutations cannot
  change buffered or background writes.
- Recovery and finalization clean obsolete metric representations left by interrupted
  publication, preventing discarded rows from leaking into finalized archives.
- Readers detect hot-only metrics moving into columnar storage instead of returning
  stale empty results. JSONL conversion drops emptied columns on rewind so discarded
  rows no longer determine their dtype.

## 0.11.0 - 2026-07-11
- Breaking API change: `PlattliBulkWriter` is now `BulkWriter`. Update imports and
  constructor calls; no compatibility alias is provided.
- `CompactingWriter` now requires `hotsize` as a keyword argument.
- Unified `write()` support across writers: pass one metrics dict, keyword metrics,
  or both (without duplicate names).
- Added aligned multi-column reads with `Reader.table()` and live-run cache reset
  with `Reader.refresh()`.
- Improved crash-safe resume and background compaction, including finalized-archive
  path validation and protection against accidental `BulkWriter` overwrites.
- Metric names must be safe relative paths and may not use reserved metadata or hot-log
  names (`step`, `run_rows`, `when_exported`, `hot`, or `hot.compacting`).

## 0.1.0 - 2026-01-03
- Initial release extracted from flattlibrettli.
- PlattliWriter API and jsonl2plattli CLI.
