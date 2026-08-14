"""Milestone 8 end-to-end example: session.sql().

Registers two in-memory DataFrames as temp views and runs a SQL query
that joins, groups, filters with HAVING, and orders, exactly the same
capability a chained DataFrame API call already has, just spelled as
SQL text. `explain(optimized=True)` on the result shows this went
through the identical analyzer/optimizer/physical-planner path any
DataFrame does; there is no separate SQL execution engine. Run with:

    python examples/sql.py
"""

from __future__ import annotations

from minispark.api.session import MiniSparkSession

USERS = [
    {"name": "alice", "age": 30, "country": "US"},
    {"name": "bob", "age": 17, "country": "CA"},
    {"name": "carol", "age": 45, "country": "US"},
    {"name": "dave", "age": 19, "country": "UK"},
    {"name": "erin", "age": 15, "country": "UK"},
]

REGIONS = [
    {"country": "US", "region": "Americas"},
    {"country": "CA", "region": "Americas"},
    {"country": "UK", "region": "Europe"},
]


def main() -> None:
    session = (
        MiniSparkSession.builder.master("local[2]").app_name("sql_example").get_or_create()
    )

    users = session.create_dataframe(USERS, num_partitions=3)
    regions = session.create_dataframe(REGIONS, num_partitions=2)
    session.create_or_replace_temp_view("users", users)
    session.create_or_replace_temp_view("regions", regions)

    result = session.sql(
        """
        SELECT regions.region, COUNT(*) AS adults, AVG(users.age) AS avg_age
        FROM users
        JOIN regions ON users.country = regions.country
        WHERE users.age >= 18
        GROUP BY regions.region
        HAVING COUNT(*) >= 1
        ORDER BY region
        """
    )
    # ORDER BY sorts by the group key (a string), not a numeric aggregate
    # output, on purpose: order_by()'s range-partition boundaries need to
    # eagerly execute their child plan (see physical/planner.py's
    # _sort_range_boundaries()), which only supports a Scan/Filter/
    # Project chain, not one ending in Aggregate or Join, an existing
    # Milestone 5 scope limit, not something Milestone 8 changes. A
    # string sort key skips that eager-execution path entirely (falls
    # back to a single, non-parallel partition, still fully correct),
    # so this example demonstrates GROUP BY/HAVING/ORDER BY together
    # without hitting that unrelated limitation.

    print("Physical plan (identical machinery to the DataFrame API):")
    result.explain(optimized=True)
    print()

    print("Result:")
    result.show()

    print()
    print("Metrics from this run:")
    print(result.last_run_metrics.summary())


if __name__ == "__main__":
    main()
