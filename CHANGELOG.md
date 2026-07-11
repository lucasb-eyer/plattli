# Changelog

All notable changes to this project are documented in this file.

## 0.11.0 - 2026-07-11
- Breaking API change: `PlattliBulkWriter` is now `BulkWriter`. Update imports and
  constructor calls; no compatibility alias is provided.
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
