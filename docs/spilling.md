# Spilling and memory-aware execution

How `order_by(...)` and `group_by(...).agg(...)` avoid growing an
unbounded in-memory buffer, and the byte-offset optimization that lets
`CSVDataSource` read a large file without re-parsing it once per
partition. See `docs/execution-model.md` for the
DAG/Stage/Task/Worker/Scheduler picture this fits into, `docs/shuffle.md`
for what happens at the shuffle boundary around each of these operators,
and `docs/benchmarks.md`'s "Spilling: what does it cost?" and "Data
skew" sections for the real, measured numbers referenced throughout.

## Why spilling is needed at all

`SortExec` and `HashAggregateExec` both need to see every row in a
partition before any row's place in the result is fully known: a sort
buffer holds every row until it can all be ordered; a hash-aggregate
table holds one entry per distinct key until every row contributing to
that key has been folded in. Without spilling, both simply grow that
buffer in memory for as long as the partition's input lasts, correct for
any partition small enough to fit in memory, and silently unbounded
(eventually an `OSError`/`MemoryError` from the OS, not a controlled
failure) for one that is not. Spilling gives each operator a way to
notice its buffer has grown past a configured threshold and move part of
it to local disk, bounding memory at the cost of some real, measured
slowdown (see "What spilling costs," below) whenever it actually
triggers.

## `spill_threshold_bytes`

`MemoryConfig.spill_threshold_bytes` (`config/config.py`) is a computed
property, `int(execution_limit_mb * 1024 * 1024 * spill_threshold)`, the
same "derived from other fields, not itself independently settable"
pattern `EngineConfig.num_workers` already uses for `master`. The default
(`execution_limit_mb=4096`, `spill_threshold=0.8`) is roughly 3.3 GB, in
practice never crossed by the query sizes this project's own test suite
and benchmarks use, matching the pre-Milestone-9 behavior exactly unless
a caller deliberately configures a smaller `MemoryConfig`.

`api/dataframe.py` reads `session.config.memory.spill_threshold_bytes`
and passes it into `physical/planner.py`'s `plan_physical()` as a plain
`int`, which threads it into every `HashAggregateExec`/`SortExec` node it
builds (the same threading pattern already used for
`shuffle_partitions`). `physical/` still never imports `config/`: a node
built directly in a test, without passing `spill_threshold_bytes`,
defaults to `NEVER_SPILL` (`physical/plan.py`, `2**62`), so every
pre-Milestone-9 test and call site keeps its exact old behavior with no
code changes.

Both operators estimate buffer size with the same heuristic already used
elsewhere in this codebase (`output_bytes`/`optimizer/statistics.py`'s
`compute_statistics`): `sys.getsizeof` summed over a record's values (or
a group's key tuple plus aggregate state, for `HashAggregateExec`).
This is a rough, Python-object-overhead-inclusive estimate, not an exact
or on-disk byte count, and it is shallow, not recursive: a value holding
a large nested container (a hypothetical `collect_list` aggregate, say)
would be under-counted. Documented at each call site, not hidden.

## Sort: external merge sort

`_execute_sort_partition` (`physical/operators.py`) buffers `(seq,
record)` pairs (`seq` from a per-partition `itertools.count()`, explained
below) and their running estimated byte size. When that size crosses
`spill_threshold_bytes`, the buffer is sorted (`_sort_rows`, the existing
repeated-stable-sort-passes technique, last key first, unchanged since
Milestone 5) and written to a spill file (`physical/spill.py`, pickled
records) as one already-sorted run, then cleared. If the partition never
crosses the threshold, this is byte-for-byte the pre-Milestone-9
behavior: sort the one in-memory list, return it eagerly. If it does
spill at least once, the final remaining buffer is sorted as one more
run, and every run (every spill file plus the final in-memory one) is
merged lazily with `heapq.merge()`, streamed out through the returned
`Partition`'s `records_fn` rather than materialized, and the spill
directory is removed in a `finally` block that runs even if the
generator is closed early (`GeneratorExit`).

### The tie-breaking bug

An early version of the merge used only each row's sort-key values as
`heapq.merge()`'s comparison key. A test built specifically to construct
many full ties (every sort column equal across many rows) on data forced
to spill in multiple rounds found that spilled output did not always
match non-spilling output: the two disagreed on which of two fully-tied
rows came first. `heapq.merge()` breaks a tie between equal-keyed
elements from *different* input runs by each run's position in the
merge, not by anything about the records, so two tied rows split across
different spill chunks could come out in a different relative order than
a single, non-spilling stable sort of the identical data would produce.
This is exactly the kind of bug the build spec's correctness-first
framing warns about: an internal, invisible-to-the-plan performance knob
(whether spilling happened to trigger) silently changing an observable
query result.

The fix: every record is tagged with a strictly increasing `seq` the
moment it is first read from the child operator, carried through
buffering and spilling as `(seq, record)`, and `seq` is appended as the
final element of `_composite_sort_key()`'s returned tuple, the key
`heapq.merge()` actually compares by. This reproduces, explicitly, what a
single stable sort already gives for free: original arrival order as the
tie-breaker after every real sort key is exhausted. `tests/unit/
test_sort_physical_plan.py`'s `test_spilling_produces_same_result_as_
non_spilling_on_random_data` is the regression test, kept in the suite
rather than removed once the fix landed: the buggy version "passed every
test that did not specifically construct enough full ties to expose it,"
so only a test built to construct that condition on purpose guards
against it recurring.

## HashAggregate: grace-hash spilling

`_execute_hash_aggregate_partition` (`physical/operators.py`) tracks its
`groups` dict's estimated size incrementally (subtracting a key's old
estimate before replacing its state, adding the new one) as rows are
folded in via `update()` (map-side/partial) or `merge()` (reduce-
side/final), the same two functions grouping already used without
spilling. When the estimate crosses `spill_threshold_bytes`, the
*entire* table is
partitioned by key hash (`shuffle/partitioner.py`'s `HashPartitioner`,
reused for consistency with the rest of the codebase's stable,
cross-process hashing) into `_AGGREGATE_SPILL_BUCKETS` (32, a fixed
fan-out, not derived from any config) buckets, each non-empty bucket
written to its own spill file, and the table cleared. A key spilled once
and seen again later simply restarts from `initialize()`/incoming state,
as if it were new; the two partial results for that key are reconciled
later, at merge time, via `AggregateFunction.merge()`, the same function
that already reconciles map-side states from different source partitions
after a real shuffle.

If a partition never crosses the threshold, this is byte-for-byte the
pre-Milestone-9 behavior. If it does spill, the final, still-in-memory
remainder is *not* itself spilled (saving one round-trip); instead the
returned `Partition`'s `records_fn` processes one of the 32 buckets at a
time: seed a small dict from the remainder's keys that hash to that
bucket, fold in every spill file written for that bucket across every
round (via `merge()`), finalize, and yield, before moving to the next
bucket. This bounds memory during the merge phase too, to one bucket's
distinct-key set at a time, not the whole partition's; a smaller design
that spilled once and merged everything in a single pass would not have
that property.

### What this does not protect against

Bucket count bounds memory, not time: if one bucket happens to hold a
disproportionate share of distinct keys (skew, whether from the source
data's own key distribution or an unlucky hash), that bucket's merge
step still takes proportionally longer, and nothing here sub-partitions
an oversized bucket further. `benchmarks/skew.py` measures this kind of
imbalance (for the reduce-side shuffle partitioning generally, not this
specific bucket mechanism, but the mechanism is the same underlying
issue: `HashPartitioner` routes a whole key's data to one place, so a
key holding an outsized share of rows becomes an outsized share of work
somewhere); it is a measurement, explicitly, not a fix, per this
milestone's own scope decision.

### What spilling costs

`benchmarks/spilling.py` (`docs/benchmarks.md`'s "Spilling: what does it
cost?") measured, on this development machine, spilling as 1.83x slower
than staying fully in memory for a 2,000,000-row sort, and 3.16x slower
for a 2,000,000-row, 500,000-distinct-key group-by. The aggregate case
is proportionally more expensive because a spill resets the whole table,
not just the excess (see above): a key can be written to and read back
from disk more than once over one partition's lifetime, where a sort's
rows are each written at most once. This is the direct, expected cost of
favoring a simpler, still-correct, still memory-bounded-during-merge
design over one that would minimize total spill I/O at the cost of more
bookkeeping (e.g. an eviction policy that only spills the excess); see
`docs/architecture.md`'s Key spilling and CSV byte-offset design
decisions for why that alternative was not taken.

## CSV byte-offset seeking

A different kind of memory-aware change, not related to spilling itself:
previously, every `CSVDataSource` partition's `records_fn` ran a
`csv.reader` from the top
of the file and threw away every row before its own assigned range
(`itertools.islice`), meaning the file's data section was effectively
parsed once per partition, `num_partitions` times per query. `read()`
now also records one byte offset per partition
(`_locate_partition_offsets`), computed via two extra, cheap
`readline()`-only (no CSV tokenizing) full-file passes, so each
partition's factory seeks straight to its own first row instead.

The obstacle, found empirically before writing any implementation code:
`csv.reader(f)` disables `f.tell()` for the rest of that file object's
life (`OSError: telling position disabled by next() call`), so neither
the offset-recording pass nor a partition's own read can iterate a
`csv.reader` directly over the file the way the pre-Milestone-9 code did.
`f.readline()` has no such restriction and round-trips correctly with
`f.seek()`, so both now parse one already-read line at a time via
`next(csv.reader([line]))`. The accepted cost: a quoted CSV field
containing a literal embedded newline is split across two `readline()`
calls, which either misparses the row or raises `ValueError`
(`_coerce_row`'s `zip(..., strict=True)` sees a line with fewer fields
than the header), where the old top-of-file `csv.reader(f)` approach
handled it correctly. This codebase's CSV reader was never a full RFC
4180 implementation (see `storage/csv.py`'s `_try_parse` and its lack of
custom delimiter/quoting support); this is one more, now-documented, gap
in that same spirit, and `tests/unit/test_csv_byte_offset.py`'s
`test_embedded_newline_in_quoted_field_is_a_known_limitation` pins down
exactly what happens rather than leaving it as a silent surprise.

## What this is not

No cost-based or adaptive threshold, `spill_threshold_bytes` is a fixed
number a query either crosses or does not, not adjusted based on
observed memory pressure or available system memory at runtime. No
spilling for `HashJoinExec`'s build-side hash table (only `SortExec` and
`HashAggregateExec` spill, per this milestone's own scoped decision).
No sub-partitioning of an oversized grace-hash bucket (skew is measured,
not mitigated, see above). No exact byte accounting, both operators'
size estimates are the same `sys.getsizeof`-based heuristic already used
elsewhere in this codebase, not a true measurement of process memory
use. No compression of spill files (pickled records, same uncompressed
format shuffle blocks and checkpoints already use).
