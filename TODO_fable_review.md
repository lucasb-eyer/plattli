# Plättli review (2026-07-05, Fable)

Scope: full read of `writer.py`, `reader.py`, `_indices.py`, `bulk_writer.py`, `jsonl2plattli.py`.
All 74 tests pass. Claims below were verified with probes/benchmarks where marked.

Overall: the format design is good (torn-tail tolerance, segment-compressed indices,
resume-heals-duplicates via min-hot-step truncation, atomic manifest replace).
`CompactingWriter` delivers on the fast-user-thread goal (~13µs/step measured).
Main gaps: `DirectWriter`'s thread pool actively hurts, the reader has 4-5x read
amplification, and the headline read use case (step-aligned sets of columns) has no API.

## Bugs / correctness

- [x] **Dead crash-recovery branch in `_load_hot_rows`** (writer.py:710-732, verified).
  `if row_step >= self.step: continue` made the `row_step == self.step` branch and all
  the `_current_row`/`_step_metrics` restoration below it unreachable. Decision: keep the
  simple no-special-case semantics — resuming at step N nukes N and everything beyond,
  writing starts with a fresh step N. Dead code deleted, v2_plan.md amended.
- [x] **In-process duplicate rows after a compaction error** (writer.py:742-750, 836-886).
  `_compact_rows` appends metric-by-metric; on partial failure `_drain_errors` raises and
  clears `_compact_steps` but rows stay in `_hot_rows`. If the caller catches and keeps
  writing, the next compaction re-appends already-written values. Fixed by poisoning the
  writer: after any background write/compaction error, all further `write`/`end_step`/
  `finish` calls raise loudly; recovery = recreate the writer, which heals partial
  compaction on disk via the existing `_truncate_to_step(min_hot_step)` resume path.
  Applied to both `CompactingWriter` and `DirectWriter` (same partial-append hazard).
- [ ] **Manifest mutations happen outside `_hot_lock`** (writer.py:628, 633 vs 876-882).
  `write()` inserts specs / flips `monotonic` unlocked while the compaction thread
  serializes the same dict under the lock. Stress test could not trigger a crash — safe
  today only because GIL + C json encoder make `{**manifest}`/`json.dumps` atomic. Breaks
  under free-threaded Python; the lock protects the file write, not the data.
- [ ] **Non-atomic jsonl truncation on resume** (writer.py:153). `_truncate_to_step`
  rewrites jsonl with plain `write_text`; crash mid-resume tears the file. Manifest
  correctly uses `_replace_text_checked`; jsonl should too.
- [x] **`PlattliBulkWriter` never validates scalars** (bulk_writer.py:121). A list value
  becomes a flattened 2-D array whose length no longer matches the indices — corrupt
  output, no error. Fixed: non-scalar array-likes now raise at `write()` (matching
  `DirectWriter`), and the tighten path in `finish()` only fires for 1-d numeric columns,
  so list values (even/ragged) flow to jsonl like they do in `DirectWriter`.

## Write-path performance

Benchmark (2000 steps x 8 metrics, local disk, per-step `write`+`end_step` latency):

| Writer                              | p50    | p99    | max    |
|-------------------------------------|--------|--------|--------|
| DirectWriter, 16 threads (default)  | 1803µs | 2423µs | 5210µs |
| DirectWriter, write_threads=0       |  134µs |  291µs |  711µs |
| CompactingWriter, hotsize=200       | 12.9µs | 22.8µs |  880µs |

- [ ] **Default DirectWriter thread pool is 13x slower than synchronous** (verified).
  One executor task per metric per step, each opening+closing two files
  (`_write_entry`, writer.py:514), then `end_step` blocks on all futures (writer.py:480)
  — full submission overhead and you still wait. Also contradicts the README's
  "writes are non-blocking". Better shape: one dedicated writer thread with an ordered
  queue (ordering without blocking `end_step`) + cached open fds per metric (same trick
  as the hot log, commit 75c5f5e).
- [ ] **New metrics / monotonic flips rewrite the whole manifest with 2 fsyncs on the
  user thread** (writer.py:477, 637-639). Quadratic in metric count: a 3000-metric probe
  did not finish in 2 minutes. Consider deferring manifest writes to the background thread.
- [ ] **Post-compaction hot rewrite runs on the user thread** with fsyncs via
  `_replace_text_checked` (writer.py:759-763) — the 880µs max spike above, every
  `hotsize` steps; milliseconds on Lustre. Move to the compaction thread.
- [ ] **`_refresh_monotonic_metadata` iterates every stored value in a Python loop on
  resume** (writer.py:332). Vectorize with `np.diff` sign checks; a 1M-row x 20-metric
  resume is seconds of pure Python otherwise.
- [ ] **jsonl metrics compact quadratically**: `_compact_rows` calls
  `_stored_values_count`, which json-parses the entire file, once per batch
  (writer.py:860 -> 224-233). Track the count in memory.
- [ ] **No backpressure**: if compaction falls behind, `_hot_rows` grows unboundedly and
  each hot rewrite gets bigger.
- [ ] Minor: `_compact_batch_locked` scans + sorts all hot rows every `end_step`.

## Read-path performance

- [ ] **One `Reader.metric()` call reads the indices member 5x and values 4x**
  (zip case, all full reads; dir case opens indices 5x — measured).
  `_metric_indices_full` and `_metric_values_full` each independently re-derive counts
  (`_indices_count_and_last` + `_values_count`, which in the zip case fully read the
  member just to count), and `_metric_values_full` re-reads the whole indices file just
  for `last_step` even when no hot file exists (reader.py:764-771). Compute counts once
  per call; skip the last-step read when `kind != "dir"`.
- [ ] **Zip slice reads aren't slices**: `_read_indices_slice`/`_read_value_slice` read
  the whole member then slice (reader.py:215, 246). Members are ZIP_STORED; `zf.open()`
  supports cheap seek — slices could be O(slice) like the dir path.
- [ ] **jsonl slices re-parse the whole file per chunk**: `_read_value_slice` calls
  `_columnar_values(...)[offset:offset+count]` (reader.py:240) and `_values_count`
  parses it all again. Cache the parsed list per metric on the Reader.
- [ ] **Any `hot.jsonl` disables the optimized selector path entirely**
  (`_selector_chunks`, reader.py:351) — live runs, exactly the dashboard-polling case,
  always fall back to full-read-and-mask. Chunk the columnar part, filter only the hot tail.
- [ ] **No tail fast path**: `metric("loss", idx=-1)` reads the whole column, and
  `_position_slice` rejects negative `istart`/`istop` (reader.py:146), so "last N rows"
  needs a `rows()` round-trip first. Tail reads are the most common dashboard query;
  support negative positions and resolve them from the counts already computed.
- [ ] **Inconsistent caching for live runs**: manifest + hot columns are cached forever,
  data files read fresh — a long-lived Reader sees new columnar rows but a stale hot tail
  and metric list. Document Readers as one-shot or add `refresh()`.
- [ ] Minor: `_is_plattli_zip` opens the zip (central directory read), then `Reader.__init__`
  opens it again.

## API design

- [ ] **The stated core read use case has no API**: "read whole sets of columns
  step-aligned on one column" currently means N x `r.metric(name)` + manual alignment in
  every consumer. Add e.g. `r.table(["loss", "acc"], on="loss", start=..., stop=...)`
  returning steps + aligned value arrays; it can also share count/index work across
  columns (fixes the amplification for the multi-column case too).
- [ ] **`write()` signatures diverge**: `CompactingWriter.write(metrics=None, flush=False,
  **kw)` vs `DirectWriter.write(**kw)` only. Slash-named metrics (`detail/thing0`,
  advertised in the README) reach DirectWriter only via `**{...}` unpacking. Give
  DirectWriter the positional-dict form.
- [ ] **Post-`finish()` misuse raises `TypeError: 'NoneType' object is not callable`**
  (writer.py:509) instead of a real error; `finish()` on a run with zero writes silently
  does nothing, leaving a dir with config but no `plattli.json` that `is_run()` rejects.
  Both deserve loud, clear failures.
- [ ] **`PlattliBulkWriter`** is the odd name out next to `DirectWriter`/`CompactingWriter`
  and silently overwrites an existing `metrics.plattli`.
- [ ] Writers support `del w` but not `with` — add a context manager (`__exit__` ->
  flush, not finish); matches the crash-safe philosophy.
- [ ] Naming nits: `hotsize=None`-then-raise instead of a required arg;
  `approx_max_rows(faster=True)` where `faster=False` secretly means "also scan hot";
  `when_exported` bumps on every live manifest write so it doesn't mean "exported";
  `w.step` is writable and jsonl2plattli assigns it, but step jumps are undocumented and
  unguarded on the other writers.

## Suggested order

1. Dead-code/resume decision (semantics call for the owner).
2. Reader amplification + tail fast path (big win, small diff).
3. DirectWriter threading rework (big win, medium diff).
4. Everything else opportunistically.
