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

## Reproducing these numbers

```bash
python -m benchmarks.scaling
python -m benchmarks.csv_vs_parquet   # needs: pip install -e ".[columnar]"
python -m benchmarks.join_strategy
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
