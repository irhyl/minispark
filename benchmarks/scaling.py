"""Benchmark: does local[N] with N > 1 actually finish a shuffle-heavy
query faster than local[1]?

Runs `group_by(key).agg(sum(...))` over synthetic in-memory data at a
couple of sizes, once under `local[1]` (sequential) and once under
`local[N]` (a real ProcessPoolExecutor, N = this machine's logical CPU
count), and reports wall-clock time for each. See docs/benchmarks.md for
the actual numbers recorded from running this script, and this file's
own honesty caveat: single-trial, not a controlled benchmark rig.

Run with (from the repository root):

    python -m benchmarks.scaling
"""

from __future__ import annotations

import os

from benchmarks._common import machine_info, timed
from minispark.api.functions import sum as ssum
from minispark.api.session import MiniSparkSession

ROW_COUNTS = [20_000, 200_000, 2_000_000]
NUM_KEYS = 200


def make_records(n: int) -> list[dict]:
    return [{"key": i % NUM_KEYS, "value": i} for i in range(n)]


def run_once(records: list[dict], master: str) -> float:
    session = MiniSparkSession.builder.master(master).app_name("bench_scaling").get_or_create()
    df = session.create_dataframe(records, num_partitions=8)
    with timed() as t:
        df.group_by("key").agg(ssum("value").alias("total")).collect()
    return t["seconds"]


def main() -> None:
    n_workers = os.cpu_count() or 4
    print(machine_info())
    print(f"local[N] uses N={n_workers} (os.cpu_count())")
    print()
    print(f"{'rows':>10} | {'local[1]':>10} | {'local[N]':>10}")
    print("-" * 36)
    for n in ROW_COUNTS:
        records = make_records(n)
        t1 = run_once(records, "local[1]")
        tn = run_once(records, f"local[{n_workers}]")
        print(f"{n:>10} | {t1:>9.3f}s | {tn:>9.3f}s")


if __name__ == "__main__":
    main()
