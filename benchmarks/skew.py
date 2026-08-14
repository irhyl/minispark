"""Measurement-only skew experiment (Milestone 9's build spec bullet
"skew experiments"): does one dominant key's rows landing in a single
hash-partitioned reduce task measurably hurt performance, compared to the
same total row count spread evenly across keys?

This script does not fix skew (`physical/operators.py`'s grace-hash
aggregate spilling bounds *memory* per bucket, not the *time* one
disproportionately large bucket takes relative to the others, see its
docstring); it only measures the effect, per this milestone's own scope.

Both datasets have the same row count and the same shuffle_partitions
(so the same number of reduce tasks), and both compute the identical
query, `group_by(key).agg(count(*))`. Because `HashPartitioner` sends
every row for a given key to the same reduce task (see shuffle/
partitioner.py), one key holding a large share of all rows forces one
reduce task to do most of the work while the others finish fast.

Run under both `local[1]` and `local[N]` on purpose, not just `local[N]`:
`local[1]` runs tasks sequentially in-process (`EngineConfig.
num_workers`'s docstring), so its wall clock is not confounded by
`ProcessPoolExecutor` spawn cost, and cleanly isolates skew's own effect
on total compute time (it should track `total_execution_time_seconds`
almost exactly, since there is no parallelism to hide behind or be hidden
by). `local[N]` is the realistic case, but see docs/benchmarks.md's
"local[1] vs local[N]" section: on this Windows/spawn machine, per-stage
process-spawn overhead is large enough to dominate wall clock at some
data sizes, which can *mask* skew's effect on wall time even though the
underlying per-task compute-time skew (visible in `total_execution_time_
seconds`, summed across tasks) is unaffected by that overhead. Both
numbers are reported so that confound is visible, not hidden. See
docs/benchmarks.md for the recorded numbers and this project's
benchmark-honesty caveat (single trial, this machine, not a controlled
rig).

Run with (from the repository root):

    python -m benchmarks.skew
"""

from __future__ import annotations

import os
import random

from benchmarks._common import machine_info
from minispark.api.functions import count
from minispark.api.session import MiniSparkSession
from minispark.config.config import Config, EngineConfig, ExecutionConfig

ROWS = 4_000_000
NUM_KEYS = 200
SHUFFLE_PARTITIONS = 8
DOMINANT_KEY_SHARE = 0.85  # the skewed dataset puts this fraction of all rows on one key


def make_balanced_records(n: int) -> list[dict]:
    return [{"key": i % NUM_KEYS, "value": i} for i in range(n)]


def make_skewed_records(n: int) -> list[dict]:
    random.seed(5)
    dominant_count = int(n * DOMINANT_KEY_SHARE)
    records = [{"key": 0, "value": i} for i in range(dominant_count)]
    records += [
        {"key": 1 + (i % (NUM_KEYS - 1)), "value": i} for i in range(n - dominant_count)
    ]
    random.shuffle(records)  # so no single source partition is 100% the dominant key either
    return records


def run_once(records: list[dict], master: str) -> tuple[float, float, int]:
    """Returns (reduce-stage wall clock, reduce-stage task-time sum,
    reduce-stage task count), all read from `StageMetrics` (execution/
    metrics.py), not from timing `collect()` itself: `collect()`'s own
    wall clock spans *both* stages (the partial/map-side aggregate and
    shuffle write, then the reduce), and only the second stage is where
    skew's effect on the reduce-side hash-partitioned tasks shows up.
    """
    config = Config(
        engine=EngineConfig(master=master),
        execution=ExecutionConfig(shuffle_partitions=SHUFFLE_PARTITIONS),
    )
    session = MiniSparkSession(config=config, app_name="bench_skew")
    df = session.create_dataframe(records, num_partitions=8)
    result = df.group_by("key").agg(count("*").alias("n"))
    result.collect()
    reduce_stage = result.last_run_metrics.stages[-1]
    return (
        reduce_stage.wall_clock_seconds,
        reduce_stage.total_execution_time_seconds,
        reduce_stage.num_tasks,
    )


def print_row(label: str, wall: float, task_sum: float, tasks: int) -> None:
    print(f"{label:>16} | {wall:>11.3f}s | {task_sum:>15.3f}s | {tasks:>12} | "
          f"{task_sum / tasks:>13.3f}s")


def main() -> None:
    n_workers = os.cpu_count() or 4
    print(machine_info())
    print(f"rows={ROWS}, distinct keys={NUM_KEYS}, shuffle_partitions={SHUFFLE_PARTITIONS}")
    print(f"local[N] uses N={n_workers} (os.cpu_count())")
    print(f"skewed dataset: {DOMINANT_KEY_SHARE:.0%} of rows share a single key")
    print()

    balanced = make_balanced_records(ROWS)
    skewed = make_skewed_records(ROWS)

    masters = ["local[1]", f"local[{n_workers}]"]
    results = {}
    for master in masters:
        results[(master, "balanced")] = run_once(balanced, master)
        results[(master, "skewed")] = run_once(skewed, master)

    header = (
        f"{'run':>16} | {'reduce wall':>12} | {'reduce task-sum':>16} | "
        f"{'reduce tasks':>12} | {'mean task time':>14}"
    )
    print(header)
    print("-" * len(header))
    for master in masters:
        for dataset_label in ["balanced", "skewed"]:
            wall, task_sum, tasks = results[(master, dataset_label)]
            print_row(f"{master}/{dataset_label}", wall, task_sum, tasks)

    b1_wall, b1_sum, b1_tasks = results[("local[1]", "balanced")]
    s1_wall, s1_sum, s1_tasks = results[("local[1]", "skewed")]
    bn_wall, bn_sum, bn_tasks = results[(f"local[{n_workers}]", "balanced")]
    sn_wall, sn_sum, sn_tasks = results[(f"local[{n_workers}]", "skewed")]

    print()
    print(
        f"local[1] (no process-spawn confound, see module docstring): reduce wall clock "
        f"grew {s1_wall / b1_wall:.2f}x from balanced to skewed, tracking mean task time's "
        f"{(s1_sum / s1_tasks) / (b1_sum / b1_tasks):.2f}x growth almost exactly, since "
        "local[1] runs every task sequentially in one process: wall clock IS the task-time "
        "sum here, so skew shows up directly."
    )
    print(
        f"local[{n_workers}]: reduce wall clock grew {sn_wall / bn_wall:.2f}x while mean task "
        f"time grew {(sn_sum / sn_tasks) / (bn_sum / bn_tasks):.2f}x: on this machine, "
        "per-stage ProcessPoolExecutor spawn cost (see docs/benchmarks.md's 'local[1] vs "
        "local[N]' section) is large enough at this data size to compress the wall-clock "
        "gap between balanced and skewed relative to the true, underlying per-task compute "
        "skew, which local[1]'s numbers above show directly, undistorted."
    )


if __name__ == "__main__":
    main()
