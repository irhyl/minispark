"""Milestone 4 end-to-end example: group_by().agg() with a real shuffle.

`local[3]` means this genuinely splits into two stages (see
docs/execution-model.md and docs/shuffle.md): a partial aggregate per
source partition, a disk-backed shuffle hash-partitioned by "country",
and a final aggregate merging every partition's partial results per
country, all across real OS worker processes. Watch for the
`StageStarted`/`ShuffleCompleted`/`StageCompleted` log lines this
produces, and compare `explain()` (the raw logical plan) against
`explain(optimized=True)` (which shows the partial/exchange/final shape
the optimizer and physical planner actually chose). Run with:

    python examples/aggregations.py
"""

from __future__ import annotations

from pathlib import Path

from minispark.api.functions import avg, col, count
from minispark.api.session import MiniSparkSession

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "users.csv"


def main() -> None:
    session = (
        MiniSparkSession.builder.master("local[3]").app_name("aggregations_example")
        .get_or_create()
    )

    df = session.read.csv(str(DATA_PATH))

    result = (
        df.filter(col("age") >= 18)
        .group_by("country")
        .agg(count("*").alias("adults"), avg("age").alias("avg_age"))
    )

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
