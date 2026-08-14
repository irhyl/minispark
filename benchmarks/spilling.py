"""Benchmark: what does external-merge-sort / grace-hash-aggregate
spilling (Milestone 9) actually cost in wall-clock time, compared to
letting the same query run entirely in memory?

Runs the same `order_by(...)` and `group_by(...).agg(...)` queries twice
each: once with `spill_threshold_bytes` effectively infinite (the
pre-Milestone-9 in-memory-only behavior, `NEVER_SPILL`) and once with it
set low enough to force real spill files to be written to and read back
from local disk. `local[1]` throughout, deliberately: this isolates
spilling's own cost from `ProcessPoolExecutor` spawn overhead (see
docs/benchmarks.md's "local[1] vs local[N]" section), which is a
separate, already-measured effect this script is not trying to
re-demonstrate. See docs/benchmarks.md for the recorded numbers and this
project's benchmark-honesty caveat (single trial, this machine, not a
controlled rig).

Run with (from the repository root):

    python -m benchmarks.spilling
"""

from __future__ import annotations

import random

from benchmarks._common import machine_info, timed
from minispark.api.functions import count
from minispark.api.session import MiniSparkSession
from minispark.config.config import Config, EngineConfig, MemoryConfig

SORT_ROWS = 2_000_000
AGG_ROWS = 2_000_000
AGG_DISTINCT_KEYS = 500_000

# execution_limit_mb=1, spill_threshold=5.0 -> spill_threshold_bytes ~=
# 5 * 1024 * 1024 = 5,242,880 (see MemoryConfig.spill_threshold_bytes).
# Small enough, relative to each workload's full in-memory working set
# below, to force many real spill rounds rather than just one or two.
SPILLING_MEMORY = MemoryConfig(execution_limit_mb=1, spill_threshold=5.0)
NO_SPILL_MEMORY = MemoryConfig()  # default: effectively never spills at this data size


def make_session(memory: MemoryConfig) -> MiniSparkSession:
    config = Config(engine=EngineConfig(master="local[1]"), memory=memory)
    return MiniSparkSession(config=config, app_name="bench_spilling")


def run_sort(memory: MemoryConfig, records: list[dict]) -> float:
    session = make_session(memory)
    df = session.create_dataframe(records, num_partitions=1)
    with timed() as t:
        df.order_by("value").collect()
    return t["seconds"]


def run_aggregate(memory: MemoryConfig, records: list[dict]) -> float:
    session = make_session(memory)
    df = session.create_dataframe(records, num_partitions=1)
    with timed() as t:
        df.group_by("key").agg(count("*").alias("n")).collect()
    return t["seconds"]


def main() -> None:
    print(machine_info())
    print(f"spilling spill_threshold_bytes ~= {SPILLING_MEMORY.spill_threshold_bytes:,}")
    print(f"non-spilling spill_threshold_bytes ~= {NO_SPILL_MEMORY.spill_threshold_bytes:,}")
    print("local[1] throughout: isolates spilling's cost from ProcessPoolExecutor overhead")
    print()

    random.seed(17)
    sort_records = [{"value": random.randint(0, SORT_ROWS)} for _ in range(SORT_ROWS)]
    t_sort_no_spill = run_sort(NO_SPILL_MEMORY, sort_records)
    t_sort_spilling = run_sort(SPILLING_MEMORY, sort_records)

    random.seed(19)
    agg_records = [
        {"key": random.randint(0, AGG_DISTINCT_KEYS - 1)} for _ in range(AGG_ROWS)
    ]
    t_agg_no_spill = run_aggregate(NO_SPILL_MEMORY, agg_records)
    t_agg_spilling = run_aggregate(SPILLING_MEMORY, agg_records)

    header = f"{'query':>28} | {'no spill':>10} | {'spilling':>10} | {'slowdown':>9}"
    print(header)
    print("-" * len(header))
    print(
        f"{'order_by (' + str(SORT_ROWS) + ' rows)':>28} | {t_sort_no_spill:>9.3f}s | "
        f"{t_sort_spilling:>9.3f}s | {t_sort_spilling / t_sort_no_spill:>8.2f}x"
    )
    print(
        f"{'group_by (' + str(AGG_DISTINCT_KEYS) + ' keys)':>28} | {t_agg_no_spill:>9.3f}s | "
        f"{t_agg_spilling:>9.3f}s | {t_agg_spilling / t_agg_no_spill:>8.2f}x"
    )


if __name__ == "__main__":
    main()
