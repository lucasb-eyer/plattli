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
- [x] **Manifest mutations happen outside `_hot_lock`** (writer.py:628, 633 vs 876-882).
  `write()` inserted specs / flipped `monotonic` unlocked while the compaction thread
  serialized the same dict under the lock; the compaction thread also mutated segment
  lists in place outside the lock. Reproduced on free-threaded 3.14t (RuntimeError:
  dictionary changed size during iteration, with widened iteration windows). Fixed:
  `write()` mutates the manifest under `_hot_lock`, and `_compact_rows` snapshots
  dtype/spec under the lock and mutates a copy of the segments, swapping it in under the
  lock. Verified: stress clean on 3.14t, full suite passes on GIL and free-threaded builds.
- [x] **Non-atomic jsonl truncation on resume** (writer.py:153). `_truncate_to_step`
  rewrote jsonl with plain `write_text`; crash mid-resume could tear the file. Fixed:
  jsonl now finds the byte offset of the first dropped line and truncates in place with
  `ftruncate`, exactly like the numeric branch (atomic, no rewrite).
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

- [x] **One `Reader.metric()` call reads the indices member 5x and values 4x**
  (zip case, all full reads; dir case opens indices 5x — measured). Fixed: zip counts
  now come from `getinfo().file_size`, last-step peeks are 4-byte seek-reads, and
  `_metric_values_full` skips the hot-merge last-step work entirely when no hot file
  exists. Now: values read 1x, indices 1x + two 4-byte peeks. Measured on 2M rows:
  zip full read 33.6ms -> 11.7ms, dir full 2.9ms -> 1.8ms.
- [x] **Zip slice reads aren't slices**: `_read_indices_slice`/`_read_value_slice` read
  the whole member then sliced. Fixed via `_read_zip_slice` (seek + exact-size read on
  ZIP_STORED members, O(1) seek verified). Measured: zip position-slice of 100 rows out
  of 2M went 12.2ms -> 0.05ms; step-slice 17.0ms -> 4.8ms (rest is the searchsorted
  indices read, inherent for `.indices`-file metrics).
- [x] **jsonl slices re-parse the whole file per chunk**: `_read_value_slice` called
  `_columnar_values(...)[offset:offset+count]` and `_values_count` parsed it all again.
  Fixed: parsed jsonl is cached per metric for the Reader's lifetime (same semantics as
  the manifest/hot caches), and slice reads slice the cached list before building the
  object array. Measured on 200k jsonl rows: 100-row slice 461ms -> 0.03ms, idx=-1
  474ms -> 0.02ms, rows() 228ms -> 0.01ms.
- [x] **Any `hot.jsonl` disables the optimized selector path entirely**
  (`_selector_chunks`, reader.py:351) — live runs, exactly the dashboard-polling case,
  always fell back to full-read-and-mask. Fixed: `_metric_select` chunks the columnar
  part as before and filters only the in-memory hot tail (mask for step/value selectors,
  boundary split for position), then concatenates. Falls back to full-read only for
  value selectors on non-monotonic metrics, as before. Measured on 2M rows + hot tail:
  position slice 9.1ms -> 0.06ms, step slice 10.3ms -> 2.6ms, value slice 10.1ms -> 0.16ms.
- [x] **No tail fast path**: `metric("loss", idx=-1)` read the whole column, and
  `_position_slice` rejected negative `istart`/`istop`. Fixed: negative positions now
  resolve Python-slice-style against the total count (columnar + hot), and an integer
  `idx` with no other selector seek-reads just that one row. Measured on 2M rows:
  idx=-1 and last-100 reads went ~12ms -> 0.03-0.07ms on zip, dir, and live+hot runs.
- [x] **Inconsistent caching for live runs**: manifest + hot columns are cached forever,
  data files read fresh — a long-lived Reader saw new columnar rows but a stale hot tail
  and metric list. Fixed: added `Reader.refresh()` which drops all cached metadata
  (manifest, config, hot rows, row counts, parsed jsonl); documented the caching policy
  in the README. Zip readers are immutable snapshots.
- [x] Minor: `_is_plattli_zip` opened the zip (central directory read) via
  `zipfile.is_zipfile` + `ZipFile`, then `Reader.__init__` opened it again — three scans.
  Fixed: `_resolve_plattli` returns the already-open ZipFile and the Reader adopts it.

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
