# Shuffle

How `group_by(...).agg(...)` moves data between partitions, as of
Milestone 4. See `docs/execution-model.md` for narrow vs wide
dependencies and how a wide dependency becomes a stage boundary in
general; this document is specifically about what happens at that
boundary.

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

`RangePartitioner` also exists (the build spec asks for it explicitly)
but has no consumer yet: it is the right partitioner for a total-order
sort, which is Milestone 5's `Sort`, not this milestone's `group_by`.

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
multi-GB block, unneeded complexity for what this milestone needs to
demonstrate.

## Driver-side bookkeeping

`shuffle/manager.py`'s `ShuffleManager` owns the shuffle's scratch
directory (a fresh `tempfile.mkdtemp()` per query, removed in a
`finally` block once the query finishes) and, once a shuffle-write
stage's tasks report their blocks, answers "which blocks does target
partition P of stage S need to read." This bookkeeping lives only in the
driver process's memory: a worker process does not share memory with the
driver, so a reduce task is handed the exact, already-filtered block list
it needs as plain data on its `Task` (`execution/tasks.py`'s
`shuffle_blocks` field), not a reference to a live `ShuffleManager`.

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
read by anything yet). No spilling an in-progress partial-aggregate hash
table to disk under memory pressure (Milestone 9). No shuffle across
machines: every block is a local file under one machine's temp
directory, real multiprocessing on one machine, not a distributed
shuffle service.
