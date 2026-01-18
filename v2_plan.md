# Plattli v2 plan (hot + jsonl)

## Goal
Make live writes fast on Lustre without losing column-friendly reads. Keep the existing columnar layout for readers, but write live data through a small, readable hot log and compact it in the background.

## Observations from the current code
- Writes are per-metric appends to two files (values + indices). On Lustre this is a worst-case pattern: many small opens, metadata round-trips, and tiny writes.
- JSON metrics are stored as an in-place JSON array and require tail scanning and rewrites per append.
- Reads are already columnar and efficient (one large sequential read per metric).

## Proposed data path
- **Hot (live) path**: write rows to a `hot.jsonl` file (one JSON object per line, includes `step`).
- **Cold (columnar) path**: keep the existing per-metric files (`<metric>.<dtype>` + `<metric>.indices`) for reads and archival.
- **Background compaction**: when `hot.jsonl` exceeds `hotsize` steps, compact the oldest hot rows into columnar files in a single background worker.

## Why this works on Lustre
- Live writes are a single file rewrite per flush/end_step (sequential and low metadata cost).
- Columnar reads remain fast and unchanged.
- Compaction cost is amortized and no longer on the critical path.

## Design choices
- **No v1 support**: switch JSON metrics to newline-delimited JSON (`jsonl`) everywhere; drop JSON array support.
- **Hot file format**: `{"step": N, "metric1": v1, ...}` per line in `hot.jsonl`.
- **Writer API**:
  - `DirectWriter(...)` => direct columnar writes (no hot file).
  - `CompactingWriter(..., hotsize=...)` => hot mode + background compaction.
  - `CompactingWriter.write(**metrics, flush=False)`; `write({}, flush=True)` flushes without incrementing.
- **Crash recovery**: on init, if `hot.jsonl` exists, load it into memory. If the last hot row matches the current `step`, treat it as the current in-progress row.
- **Reader merge**: read columnar first, then merge hot rows that have `step > last_columnar_step` for that metric.

## Incremental changes
1) Update README to describe hot mode and jsonl storage.
2) Writer updates:
   - `jsonl` dtype, jsonl file writes, remove JSON array logic.
   - Add hot buffering, `hot.jsonl` persistence, background compaction.
   - Add crash recovery from `hot.jsonl`.
3) Reader updates:
   - Support `jsonl` values.
   - Merge `hot.jsonl` with columnar results.
4) Bulk writer updates to emit jsonl.
5) Tests:
   - Update existing JSON tests to jsonl.
   - Add test: crash during hot, resume, verify flushed data survives.

## Notes
- Background compaction is a single thread; it appends in large batches to reduce Lustre overhead.
- `finish()` compacts remaining hot rows, removes `hot.jsonl`, then zips for archival.
