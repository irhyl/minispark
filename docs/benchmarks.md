# Benchmarks

Every number on this page was produced by actually running the scripts
in `benchmarks/` on this development machine (see "Environment," below),
right before this document was written. None of these are a controlled,
isolated benchmark rig: no dedicated hardware, no repeated-and-averaged
trials, no other processes quiesced. Where a run was repeated and the
numbers moved meaningfully between runs, that variance is reported too,
not hidden behind a single cherry-picked number. Per this project's own
rule (`CLAUDE.md`, the build spec's "Non-negotiables"): never fabricate a
benchmark value, and if something cannot be measured here, say
"NOT RUN (hardware limitation)" rather than invent a number. Everything
below *was* run; nothing here is invented.

## Environment

```
Python 3.13.14, Windows-11-10.0.26100-SP0
16 logical CPUs, 16.9 GB RAM
```

This is a Windows development machine, not a dedicated Linux benchmark
host. Two facts about that matter for interpreting the numbers below:

* `local[N]` uses `concurrent.futures.ProcessPoolExecutor`, and Windows
  can only start new worker processes via `spawn` (re-importing the
  whole `minispark` package fresh in every worker), never `fork`. `fork`
  (Linux/macOS's default) is far cheaper, copying the parent process's
  already-imported state instead of re-running every import. Multi-
  process overhead measured here is therefore a *worse* case than the
  same code would show on Linux, not a property of MiniSpark's design.
* Shuffle blocks and Parquet files live on this machine's local disk
  (whatever that is in this environment, not a benchmarked SSD/NVMe
  spec); no attempt was made to characterize or control disk I/O speed.

## `local[1]` vs `local[N]`: does more workers mean faster?

`benchmarks/scaling.py`: a shuffle-heavy `group_by(key).agg(sum(...))`
over synthetic in-memory data (200 distinct keys, 8 source partitions),
run once sequentially (`local[1]`) and once across `N = os.cpu_count()
= 16` real worker processes (`local[16]`).

```
      rows |   local[1] |   local[N]
------------------------------------
     20000 |     0.180s |     1.025s
    200000 |     0.440s |     1.374s
   2000000 |     2.937s |     9.012s
```

**`local[1]` was faster than `local[16]` at every size tested here,**
including 2,000,000 rows. This is a real, measured result, not a
mistake in the script: at these data sizes, the per-task overhead this
architecture pays (pickling a whole `PhysicalPlan` plus `Task` to every
worker via `spawn`, writing/reading real shuffle blocks to local disk
per task, spawning worker processes from scratch on Windows) costs more
than the compute this workload actually needs, which is very little
(summing a few million integers). Only 8 source partitions were used
(`num_partitions=8` in the script), which also caps how much of the 16
workers could ever be used concurrently for the map stage regardless.

Repeating the 20,000/200,000-row cases in an earlier run of the same
script produced meaningfully different numbers (`local[1]`: 1.153s and
1.485s; `local[16]`: 1.004s and 1.512s, both runs shown in this
repository's commit history for `benchmarks/scaling.py`), a nearly
6x difference for `local[1]` at 20,000 rows between the two runs of the
literal same code on the literal same machine. This is exactly the kind
of variance an uncontrolled environment produces (background load,
process-spawn scheduling jitter, disk cache state), and is reported here
rather than silently discarded: it means every number on this page
should be read as "roughly this order of magnitude, on this machine,
this run," not a precise, reproducible measurement.

**What this does and does not show.** It does not show that `local[N]`
is pointless: Milestone 3's own tests (`tests/integration/
test_scheduler_multiprocessing.py`) already prove real OS processes are
genuinely used, and a CPU-bound workload with enough data per partition
to amortize task overhead would very plausibly look different. It does
show, honestly, that "more workers" is not a free win at these sizes on
this machine, and that measuring before claiming a speedup (this
project's own stated rule) is not a formality.

## CSV vs Parquet: does real column pruning and predicate pushdown help?

`benchmarks/csv_vs_parquet.py`: 200,000 rows, 10 columns (`id`, `value`,
and 8 unused padding columns), written as both CSV and Parquet (40 row
groups of 5,000 rows each, `id` monotonically increasing so each row
group covers a disjoint range). Query: `filter(id >= 190000).select(
"id", "value")`, run under `local[1]`.

```
    source |  wall time |  input_records (pre-filter, from metadata)
--------------------------------------------------------------------
       csv |     2.401s |                                     200000
   parquet |     0.546s |                                     200000
```

**Parquet was about 4.4x faster** (2.401s -> 0.546s) for the identical
query and identical result (10,000 matching rows, both sources). This
reflects two real effects together, not separable from this measurement
alone: real column pruning (2 of 10 columns ever decoded) and real
row-group-level predicate pushdown (row groups whose own min/max `id`
statistics prove they cannot contain a match are never read).

`input_records` is identical for both sources here (200,000), which
looks like it contradicts row-group pruning; it does not.
`ParquetDataSource`'s `PartitionMetadata.row_count` (`execution/
metrics.py`'s `total_input_records`, ultimately) is deliberately
pre-filter, cheap footer metadata (which row groups were *assigned* to
a partition), not a count of rows actually decoded after the pushed
filter ran; see `storage/parquet.py`'s and `execution/metrics.py`'s own
docstrings for this documented imprecision. The real effect of pushdown
is only visible in wall time here, not in this particular metric, a
useful reminder that a metric measuring one thing (assigned rows) is not
a substitute for measuring the thing actually being claimed (bytes
decoded).

## Broadcast join vs shuffle hash join

`benchmarks/join_strategy.py`: a 200,000-row "fact" table joined against
a 500-row "dimension" table on a shared key, once as the default
shuffle hash join (both sides exchanged) and once with `broadcast=True`
(only the 500-row side is shuffled, as a single broadcast partition),
under `local[4]`.

```
    strategy |  wall time |   rows out
--------------------------------------
     shuffle |     2.482s |     200000
   broadcast |     1.280s |     200000
```

**Broadcast was about 1.9x faster** (2.482s -> 1.280s), consistent with
what the design predicts: the shuffle hash join pays the cost of
writing and reading shuffle blocks for *both* the 200,000-row fact table
and the 500-row dimension table, while the broadcast join only pays that
cost for the 500-row side, leaving the fact table completely unshuffled.

## Data skew: does one dominant key hurt the reduce stage?

`benchmarks/skew.py` (measurement-only, not a fix, see the script's own
docstring): `group_by(key).agg(count(*))` over 4,000,000
rows, 200 distinct keys, 8 shuffle partitions, run twice, once with rows
spread evenly across all 200 keys ("balanced") and once with 85% of all
rows sharing a single key ("skewed"). `HashPartitioner` sends every row
for a given key to the same reduce task, so the skewed run forces one of
the 8 reduce tasks to merge far more partial state than the other seven.
Run under both `local[1]` (sequential, no `ProcessPoolExecutor` spawn
cost to confound the measurement) and `local[16]`, reading each run's
reduce-stage `StageMetrics` directly (`wall_clock_seconds` and
`total_execution_time_seconds`, not a manually-timed `collect()` call,
which would span both the map and reduce stages together).

```
             run |  reduce wall |  reduce task-sum | reduce tasks | mean task time
------------------------------------------------------------------------------------
local[1]/balanced |       0.093s |           0.092s |            8 |         0.012s
 local[1]/skewed |       0.165s |           0.164s |            8 |         0.020s
local[16]/balanced |       0.568s |           0.129s |            8 |         0.016s
local[16]/skewed |       0.571s |           0.187s |            8 |         0.023s
```

**Under `local[1]`, reduce-stage wall clock tracked the skew almost
exactly** (0.093s -> 0.165s, a 1.77x increase, matching mean task time's
1.77x increase): with no parallelism to hide behind, wall clock *is* the
task-time sum, so the one overloaded reduce task's extra merge work shows
up directly in how long the whole stage takes.

**Under `local[16]`, reduce-stage wall clock barely moved** (0.568s ->
0.571s, effectively flat) even though mean task time still grew 1.45x
(0.016s -> 0.023s): consistent with the "`local[1]` vs `local[N]`"
section above, per-stage `ProcessPoolExecutor` spawn cost on this
Windows/spawn machine (roughly half a second here) is large enough at
this task size to swamp the real, underlying compute-time skew, which is
only visible once that overhead is removed, i.e. under `local[1]`. This
is not a flaw in the skew experiment; it is the same measured overhead
this page already documents, now shown to also mask a *different* real
effect (skew) when the two are combined at a small-enough task size.

**Why the multiplier is modest (1.77x, not close to the 0.85 / 0.15 ≈
5.7x row-count skew).** The reduce stage does not process raw rows: by
the time data reaches it, Milestone 4's map-side partial aggregation has
already collapsed each source partition's rows into one partial count
per (partition, key) pair, per `physical/operators.py`'s
`_execute_hash_aggregate_partition`. What the skewed reduce task
actually does more of is merge more partial states and finalize more
result rows for its one dominant key, not iterate 5.7x more raw rows;
the row-count skew is real, but partial aggregation absorbs most of its
cost before the reduce side ever sees it. This is expected, not a
measurement error, and is itself worth knowing: it means partial
aggregation (already implemented since Milestone 4) is doing real,
unplanned-for-this-benchmark work reducing skew's impact, on top of its
original purpose of shrinking shuffle volume.

## Spilling: what does it cost?

`benchmarks/spilling.py`: the same `order_by(value)` (2,000,000 rows) and
`group_by(key).agg(count(*))` (2,000,000 rows, 500,000 distinct keys)
queries, each run twice under `local[1]` (isolating spilling's own cost
from the `ProcessPoolExecutor` overhead measured elsewhere on this page):
once with the default `MemoryConfig` (`spill_threshold_bytes` ~= 3.4 GB,
effectively never crossed at this data size, so this is the pre-
Milestone-9 in-memory-only behavior) and once with `spill_threshold_bytes`
forced down to ~5.2 MB, small enough to force many real spill rounds
(`physical/operators.py`'s external-merge-sort for `order_by`, grace-hash
spill/merge for `group_by`).

```
                       query |   no spill |   spilling |  slowdown
------------------------------------------------------------------
     order_by (2000000 rows) |    22.542s |    41.145s |     1.83x
      group_by (500000 keys) |    17.777s |    56.113s |     3.16x
```

**Spilling was slower in both cases, as expected: 1.83x for sort, 3.16x
for group-by.** This is the correct, honest tradeoff spilling makes, not
a regression: without it, either query would grow its in-memory buffer
(a sort buffer, a hash-aggregate table) without bound and eventually
exhaust available memory on data large enough; spilling trades some of
that speed for a bounded memory footprint, at the cost of real disk I/O
(writing and reading pickled spill files) that a purely in-memory run
never pays. The two workloads pay this cost differently:

* **Sort's external-merge-sort** (1.83x) writes each accumulated buffer
  as one already-sorted run per spill, then does one final
  `heapq.merge()` pass across every run: the total amount of data
  written and read is proportional to the row count once, regardless of
  how many spill rounds that gets split into.
* **Grace-hash aggregate spilling was proportionally more expensive**
  (3.16x) because, per `_execute_hash_aggregate_partition`'s own
  docstring, a spill clears the *entire* in-memory `groups` table, not
  just the excess: a key spilled once and seen again later restarts from
  `initialize()`, so the same key's state can be written to disk and
  re-merged multiple times over the course of one partition, not written
  once like a sort's rows are. This is the direct cost of the design
  choice documented there (favoring bounded memory during the merge
  phase too, one bucket at a time, over minimizing total spill I/O), not
  an accident.

## Reproducing these numbers

```bash
python -m benchmarks.scaling
python -m benchmarks.csv_vs_parquet   # needs: pip install -e ".[columnar]"
python -m benchmarks.join_strategy
python -m benchmarks.skew
python -m benchmarks.spilling
```

Run from the repository root (not `python benchmarks/<name>.py`
directly): each script imports shared helpers from `benchmarks._common`,
which only resolves when `benchmarks` itself is importable as a package,
exactly what `-m` gives (see `benchmarks/__init__.py`'s docstring).

## What this is not

Not a substitute for profiling a specific, real workload: these three
scripts each isolate one design decision (worker count, storage format,
join strategy) on synthetic data shaped to make that one effect visible,
not a representative "typical query." Not a comparison against Apache
Spark, DuckDB, or pandas: the build spec explicitly scopes benchmarking
to measuring *this* engine's own behavior, not claiming parity or
superiority against a real distributed engine (`docs/architecture.md`'s
opening line: "not a Spark replacement"). Not run on multiple machines
or operating systems: every number above reflects this one Windows
development machine, and, per the "local[1] vs local[N]" section above,
would very plausibly look different on Linux (`fork` instead of
`spawn`) or under sustained, larger-than-memory workloads neither
tested here.
