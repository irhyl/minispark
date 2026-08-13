"""Milestone 5 end-to-end example: join() and order_by() with real shuffles.

Joins `data/users.csv` against a small in-memory region lookup table on
"country", then sorts the result by age. `local[3]` means this genuinely
runs across real OS worker processes; watch for the
`StageStarted`/`ShuffleCompleted`/`StageCompleted` log lines. Compare
`explain()` (the raw logical plan) against `explain(optimized=True)`
(which shows the shuffle-hash-join and local-sort/range-shuffle/final-sort
shapes the physical planner actually built). Run with:

    python examples/joins.py
"""

from __future__ import annotations

from pathlib import Path

from minispark.api.session import MiniSparkSession

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "users.csv"

REGIONS = [
    {"country": "US", "region": "Americas"},
    {"country": "CA", "region": "Americas"},
    {"country": "UK", "region": "Europe"},
    {"country": "DE", "region": "Europe"},
]


def main() -> None:
    session = (
        MiniSparkSession.builder.master("local[3]").app_name("joins_example")
        .get_or_create()
    )

    users = session.read.csv(str(DATA_PATH))
    regions = session.create_dataframe(REGIONS, num_partitions=2)

    result = users.join(regions, on="country").order_by("region", "age")

    print("Logical plan (unoptimized):")
    result.explain()
    print()

    print("Logical plan (analyzed / optimized / physical / stages):")
    result.explain(optimized=True)
    print()

    print("Result:")
    result.show()


if __name__ == "__main__":
    main()
