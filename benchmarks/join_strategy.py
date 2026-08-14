"""Benchmark: does broadcast=True actually avoid shuffling the large side
of a join, and is it faster than the default shuffle hash join for a
large-fact/small-dimension shape?

Joins a synthetic "fact" table against a much smaller "dimension" table
on a shared key, once as the default shuffle hash join (both sides
exchanged) and once with `broadcast=True` (only the dimension table is
exchanged, as a single broadcast partition; the fact table is left
unshuffled). Reports wall-clock time for each. See docs/benchmarks.md
for the recorded numbers and this project's benchmark-honesty caveat
(single trial, this machine, not a controlled rig).

Run with (from the repository root):

    python -m benchmarks.join_strategy
"""

from __future__ import annotations

from benchmarks._common import machine_info, timed
from minispark.api.session import MiniSparkSession

FACT_ROWS = 200_000
NUM_KEYS = 500


def main() -> None:
    print(machine_info())
    print(f"fact rows={FACT_ROWS}, dimension rows={NUM_KEYS}")
    print()

    session = MiniSparkSession.builder.master("local[4]").app_name("bench_join").get_or_create()
    fact = session.create_dataframe(
        [{"key": i % NUM_KEYS, "amount": i} for i in range(FACT_ROWS)], num_partitions=8
    )
    dimension = session.create_dataframe(
        [{"key": k, "label": f"label-{k}"} for k in range(NUM_KEYS)], num_partitions=2
    )

    with timed() as t_shuffle:
        shuffle_rows = fact.join(dimension, on="key", broadcast=False).count()

    with timed() as t_broadcast:
        broadcast_rows = fact.join(dimension, on="key", broadcast=True).count()

    assert shuffle_rows == broadcast_rows == FACT_ROWS

    print(f"{'strategy':>12} | {'wall time':>10} | {'rows out':>10}")
    print("-" * 38)
    print(f"{'shuffle':>12} | {t_shuffle['seconds']:>9.3f}s | {shuffle_rows:>10}")
    print(f"{'broadcast':>12} | {t_broadcast['seconds']:>9.3f}s | {broadcast_rows:>10}")


if __name__ == "__main__":
    main()
