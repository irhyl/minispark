"""Milestone 1/2 end-to-end example.

Reads a small CSV, selects then filters, and shows the result. select()
before filter() and a foldable arithmetic expression are chosen on purpose
here (rather than the equivalent filter().select()) so that
`explain(optimized=True)` has something to show: constant folding collapses
`10 + 8` to `18`, predicate pushdown moves the filter below the select, and
projection pruning drops "country" (selected nowhere, filtered on nowhere)
right after the scan. Run with:

    python examples/basic_dataframe.py
"""

from __future__ import annotations

from pathlib import Path

from minispark.api.functions import col, lit
from minispark.api.session import MiniSparkSession

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "users.csv"


def main() -> None:
    session = (
        MiniSparkSession.builder.master("local[4]").app_name("basic_dataframe_example")
        .get_or_create()
    )

    df = session.read.csv(str(DATA_PATH))

    result = df.select("name", "age").filter(col("age") > (lit(10) + lit(8)))

    print("Logical plan (unoptimized):")
    result.explain()
    print()

    print("Logical plan (analyzed / optimized / physical):")
    result.explain(optimized=True)
    print()

    print("Result:")
    result.show()
    print()

    print(f"Row count: {result.count()}")


if __name__ == "__main__":
    main()
