# Shuffle

How `group_by(...).agg(...)`, `join(...)`, and `order_by(...)` move data
between partitions. See `docs/execution-model.md` for
narrow vs wide dependencies and how a wide dependency becomes a stage
boundary in general; this document is specifically about what happens at
that boundary, and how the three operators above differ in what they
shuffle and why.

## Why a shuffle is needed at all

`group_by("country")` needs every row for a given country in one place
before it can produce that country's final count/sum/average. Rows for
"US" can start out scattered across every source partition. A shuffle is
the mechanism that gets them all into the same place: every source
partition's rows are hash-partitioned by the group key and written out;
every target partition then reads back only the blocks that belong to it,
which together contain every row for the keys assigned to that target
partition, from every source.

## Partial aggregation before the shuffle

Shuffling every raw row would mean `group_by(country).sum(revenue)` moves
as much data across the shuffle as the original table. Instead, each
source partition first runs a local, partial aggregate (map-side): rows
are grouped *within that one partition* and combined via each aggregate
function's `update()` step (`expressions/aggregate.py`), producing at
most one row per distinct key seen in that partition, carrying opaque
partial state (`Avg`'s is a `(sum, count)` tuple, for example) instead of
a finished value. Only these partial rows are shuffled. The reduce side
then re-groups by the same key and combines the incoming partial states
with `merge()`, and only at the very end calls `finalize()` to produce
the value the user sees. This is the same node type
(`HashAggregateExec`) doing both passes, `is_partial` selects `update()`
(map side) vs `merge()` (reduce side); see physical/plan.py and
physical/operators.py.

## Partitioning

`shuffle/partitioner.py`'s `HashPartitioner` decides which target
partition a group key belongs to: `stable_hash(key) % num_partitions`.
The hash is deliberately not Python's builtin `hash()`: CPython
randomizes `hash()` for `str` per process (`PYTHONHASHSEED`) unless
disabled, which would mean two different worker processes computing the
target partition for the same string key could disagree, silently
splitting one group's rows across two target partitions. A stable hash
(`hashlib.md5` over `repr(key)`) is used instead, verified in
`tests/unit/test_partitioner.py` by literally spawning a second Python
process and checking it computes the same answer.

`RangePartitioner` is what `order_by()` uses: see "Sort: range
partitioning" below for how its boundaries are chosen and why negation,
not a different partitioner, is what makes a descending sort work.

## Join: two shuffles in, one join stage out

`left.join(right, on="id")` (the default, no `broadcast=True` hint) is a
shuffle hash join: both `left` and `right` get their own `ExchangeExec`,
hash-partitioned by `on`'s columns, into the *same* `shuffle_partitions`
target count. Because `HashPartitioner` is a pure function of the key
value, not of which side produced it, a row from `left` and a row from
`right` with equal join keys are guaranteed to land in the same target
partition, from two different upstream stages. `execution/stages.py`'s
`build_stages()` turns this into three stages: two shuffle-write stages
(one per side) and one join stage, whose `HashJoinExec` reads from both
via two `ShuffleReadExec` leaves (see "Reading from more than one prior
stage" below). The per-partition join itself (build a hash table on
`left`'s rows, probe with `right`'s, `physical/operators.py`) needs no
data movement of its own; both inputs already arrived shuffled.

`left.join(right, on="id", broadcast=True)` skips the shuffle for `left`
entirely. Only `right` gets an `ExchangeExec`, with `num_partitions=1`
and `is_broadcast=True`: every row goes to the single target partition
regardless of key (`HashPartitioner(1)` sends everything to partition 0
anyway, so no special partitioner is needed for this, just the
`num_partitions=1` choice). `execution/scheduler.py` is what actually
makes this a *broadcast*: when building tasks for a stage that reads a
broadcast exchange, every task requests target partition 0, the same
blocks, regardless of its own `partition_id` (see
`shuffle/manager.py`'s `ShuffleManager.blocks_for()`). `left` is left
completely unshuffled, so the join stage runs with `left`'s original
partition count, not 1.

Choosing broadcast is an explicit hint, not automatic: see
`logical/nodes.py`'s `Join` docstring for why (no persistent catalog, no
reliable byte-size estimate without scanning data).

## Reading from more than one prior stage

A `HashJoinExec`-rooted stage's plan has two `ShuffleReadExec` leaves
(`physical/plan.py`'s `leaves()` walks both). `execution/tasks.py`'s
`Task.shuffle_blocks` is therefore keyed by source `stage_id`
(`dict[int, list[ShuffleBlockMeta]]`), not a single flat list: each
`ShuffleReadExec` looks up only its own stage's entry
(`physical/operators.py`'s `_execute_shuffle_read_partition`).
`execution/scheduler.py` builds this dict per task by finding every
read leaf in the stage's plan and resolving each one's blocks
independently, honoring `is_broadcast` per leaf.

## Sort: range partitioning

`order_by("age")` needs every row globally ordered by `age`, not just
grouped by it: rows are hash-partitioned by *equality* for a join or a
group-by, they need to be range-partitioned by *order* for a sort, so
that partition 0 holds the smallest keys, partition 1 the next range up,
and so on, and simply reading partitions back in order (which the
scheduler always does) produces a fully sorted result. This is what
`shuffle/partitioner.py`'s `RangePartitioner` does: `bisect_right` on a
list of `num_partitions - 1` boundary values.

Two things this needs that a hash-partitioned shuffle does not:

* **Where to put the boundaries.** `RangePartitioner` needs them handed
  in, computed from the actual data; there is no distributed sampling
  stage to compute them from a sample without touching data before the
  main shuffle runs. `physical/planner.py`'s `_sort_range_boundaries()`
  gets them the direct way: it eagerly executes the child plan
  (`physical/operators.py`'s whole-Dataset `execute()`, so only a
  Scan/Filter/Project child chain works, not one ending in `Aggregate`
  or `Join`) and computes the sort key's exact min/max with
  `optimizer/statistics.py`'s `compute_statistics()`, then splits that
  range into `shuffle_partitions` equal-*width* buckets. This is a real,
  deliberate exception to "a plan is built without touching data," flagged
  loudly in both code and `docs/query-planning.md`, not hidden. Equal-width
  (not equal-row-count) bucketing also means skewed data can still produce
  uneven partition sizes; a real sampling-based partitioner would draw
  boundaries from the data's actual distribution instead.
* **Non-numeric and single-partition fallback.** Equal-width bucketing
  needs subtraction and division, which do not mean anything for a string
  (or other non-numeric) sort key. Sorting by such a column, or requesting
  only one shuffle partition, falls back to `range_boundaries=None`
  (`shuffle/writer.py`'s `HashPartitioner(1)` then handles it, since with
  one target partition hash vs range partitioning cannot differ): still
  fully correct, since there is nothing to be out of range relative to
  with only one partition, just not parallel.
* **Descending order.** `RangePartitioner` always assigns *ascending*
  target partitions, and the scheduler always merges partitions back in
  id order; for a descending sort that combination would put the
  smallest-keyed partition (itself sorted descending internally) first, a
  locally correct but globally wrong result. `_sort_range_boundaries()`
  fixes this by negating the partitioning key (`-value`, via a synthetic
  `Multiply(primary_key, Literal(-1))` expression) and the boundaries
  computed from the negated range, rather than teaching `RangePartitioner`
  or the scheduler anything about sort direction. Verified in
  `tests/unit/test_sort_physical_plan.py` and with a real-multiprocessing
  regression test in `tests/integration/test_sort_e2e.py`, this exact bug
  (descending sort producing a blockwise-ascending, not globally
  descending, result) was caught and fixed during development.

The multi-key case (`order_by("age", "name")`) only partitions by the
*first* key: the local sort (both before and after the shuffle,
`physical/plan.py`'s `SortExec`, used identically for the pre-shuffle
local sort and the final post-shuffle sort, matching `HashAggregateExec`'s
partial/final reuse pattern) still fully respects every key and its own
direction, via repeated stable sorts, last key first. Nulls always sort
last, regardless of ascending/descending, a documented simplification
rather than per-column `NULLS FIRST`/`NULLS LAST` placement.

## On-disk block format

```
<shuffle root>/
  stage_<stage_id>/
    partition_<target_partition>/
      block_<source_task_id>.pkl
      block_<source_task_id>.pkl   (one per source task that wrote here)
```

Each block file is a sequence of pickled `Record`s, written back-to-back
(`pickle.dump` in a loop; read back with `pickle.load` in a loop until
`EOFError`), not newline-delimited JSON. JSON would silently turn a
tuple (e.g. `Avg`'s partial state) into a list on the way back out, and
cannot represent `NaN` by default; pickle preserves exact Python types
across the round trip. Each block's metadata
(`shuffle/writer.py`'s `ShuffleBlockMeta`) records `stage_id`,
`source_task_id`, `target_partition`, `path`, `record_count`,
`byte_length`, and an MD5 `checksum` of the block's bytes, computed
incrementally as it is written. A block's checksum is verified against
its bytes on read by default (`shuffle/reader.py`); a mismatch raises
`ShuffleChecksumError` rather than silently returning corrupted rows.
A missing block file (the file itself is gone, e.g. deleted, rather than
present but corrupted) raises the standard library's `FileNotFoundError`
at the same point. `physical/operators.py`'s `_execute_shuffle_read_partition`
catches both and re-raises them as one `MissingShuffleDataError`
(`shuffle/reader.py`), naming which stage and target partition could not
be read: see `docs/execution-model.md`'s "Lineage-based recomputation"
for what the scheduler does with that, recompute the stage that produced
the missing data, not just retry the read that failed.

`storage/checkpoint.py`'s checkpoint files (`DataFrame.checkpoint()`)
reuse this exact same back-to-back-pickle format, one
file per partition, for the same reason: it is a durable, exact,
type-preserving way to persist a sequence of Records and read them back,
and that need is identical whether the records are one target
partition's shuffle output or a whole checkpointed partition. Checkpoint
files are not shuffle blocks, though: they carry no `ShuffleBlockMeta`,
no checksum, and are not registered with a `ShuffleManager` or cleaned up
at the end of a query, see `docs/execution-model.md`'s "Checkpointing"
for why.

Writing streams one record at a time to whichever target file it belongs
to (an open file handle and a running checksum per target partition is
the only per-target state kept in memory while writing); this is what
"do not hold the entire shuffle dataset in RAM" means in practice.
Reading a block loads that one block's bytes into memory once (to verify
its checksum against exactly the bytes that were written), then
deserializes from that buffer: memory is bounded by one block's size, not
by the whole target partition's or the whole dataset's size. This is not
a fully streaming reader for a single very large block; a production
system would checksum framed chunks instead of whole blocks to stream a
multi-GB block, unneeded complexity for what this design needs to
demonstrate.

## Driver-side bookkeeping

`shuffle/manager.py`'s `ShuffleManager` owns the shuffle's scratch
directory (a fresh `tempfile.mkdtemp()` per query, removed in a
`finally` block once the query finishes) and, once a shuffle-write
stage's tasks report their blocks, answers "which blocks does target
partition P of stage S need to read." This bookkeeping lives only in the
driver process's memory: a worker process does not share memory with the
driver, so a reduce task is handed the exact, already-filtered block
list(s) it needs as plain data on its `Task` (`execution/tasks.py`'s
`shuffle_blocks: dict[stage_id, list[ShuffleBlockMeta]]` field, one entry
per upstream stage it reads from), not a reference to a live
`ShuffleManager`.

`register_blocks(stage_id, blocks)` *overwrites* whatever was already
registered for that `stage_id`, rather than accumulating; a stage is
still only ever registered once from its own normal run (with every one
of that stage's tasks' blocks combined into a single call). The reason
it needs to be overwrite, not append, is lineage-based recomputation
(`docs/execution-model.md`): recomputing a stage whose
blocks were found missing calls `register_blocks` a second time for the
same `stage_id`, with the fresh blocks it just produced, and those must
fully replace the stale metadata (pointing at files that are gone or
corrupt), not sit alongside it and risk a later reader picking the stale
entry again.

## How this fits into stages and tasks

`execution/stages.py`'s `build_stages()` rewrites the physical planner's
`ExchangeExec` marker into two pieces: a `ShuffleWriteExec` ending the
upstream stage, and a `ShuffleReadExec` starting the downstream stage.
`execution/scheduler.py`'s `LocalScheduler.run_plan()` runs stages in
order (a downstream stage cannot start until the upstream one has fully
finished writing, that is what the wide dependency means) and, between
them, registers the write stage's reported blocks into the
`ShuffleManager` before building the read stage's tasks. See
`docs/execution-model.md` for the Task/Worker/Scheduler picture this
plugs into.

## What this is not

No compression (`ExecutionConfig.shuffle_compression` exists but is not
read by anything yet). No shuffle across machines: every block is a
local file under one machine's temp directory, real multiprocessing on
one machine, not a distributed shuffle service (see
`docs/distributed-readiness.md` for exactly what the block format
already gets right for a future fetch-over-network read path, and what
it does not). (The partial-aggregate hash table itself, upstream of the
shuffle write, does spill to disk under memory pressure, see
`docs/spilling.md`, a different mechanism from anything on this page.)
