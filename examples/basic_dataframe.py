"""Milestone 1 end-to-end example.

Reads a small CSV, filters, projects, and shows the result — the exact
`filter().select().collect()`/`show()` flow required by the Milestone 1
scope in the build spec. Run with:

    python examples/basic_dataframe.py
"""

from __future__ import annotations

from pathlib import Path

from minispark.api.functions import col
from minispark.api.session import MiniSparkSession

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "users.csv"


def main() -> None:
    session = (
        MiniSparkSession.builder.master("local[4]").app_name("basic_dataframe_example")
        .get_or_create()
    )

    df = session.read.csv(str(DATA_PATH))

    result = df.filter(col("age") > 18).select("name", "age", "country")

    print("Logical plan:")
    result.explain()
    print()

    print("Result:")
    result.show()
    print()

    print(f"Row count: {result.count()}")


if __name__ == "__main__":
    main()
