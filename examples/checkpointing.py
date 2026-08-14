"""Milestone 6 end-to-end example: DataFrame.checkpoint().

Builds a small aggregate, checkpoints it (running it now and writing the
result to local disk under a temp directory), then continues querying
the checkpointed DataFrame. `explain()` on the checkpointed DataFrame
shows a bare Scan over the checkpoint directory, not the group_by/agg
plan that produced it: the whole point of checkpointing is that nothing
downstream needs to re-run that computation to get this data back. Run
with:

    python examples/checkpointing.py

Lineage-based recomputation (the other half of Milestone 6, recovering a
shuffle stage whose output was lost after it was already written) has no
"normal usage" shape to demonstrate this way, since it only ever fires
after an injected failure; see tests/integration/test_lineage_recovery_e2e.py
for a real, end-to-end demonstration of that instead.
"""

from __future__ import annotations

from minispark.api.functions import col
from minispark.api.functions import sum as ssum
from minispark.api.session import MiniSparkSession

SALES = [
    {"country": "US", "revenue": 100},
    {"country": "US", "revenue": 50},
    {"country": "CA", "revenue": 30},
    {"country": "UK", "revenue": 70},
]


def main() -> None:
    session = (
        MiniSparkSession.builder.master("local[2]").app_name("checkpoint_example")
        .get_or_create()
    )

    totals = session.create_dataframe(SALES, num_partitions=3).group_by("country").agg(
        ssum("revenue").alias("total")
    )

    print("Plan before checkpoint (group_by/agg, would re-run from SALES if recomputed):")
    totals.explain(optimized=True)
    print()

    checkpointed = totals.checkpoint()

    print("Plan after checkpoint (a bare Scan over the checkpoint directory):")
    checkpointed.explain()
    print()

    print("Continuing to query the checkpointed DataFrame:")
    checkpointed.filter(col("total") >= 50).show()


if __name__ == "__main__":
    main()
