import pickle

from minispark.api.functions import avg, col, count
from minispark.api.functions import sum as ssum
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.logical.nodes import Aggregate, Scan
from minispark.physical.operators import execute_partition
from minispark.physical.plan import ExchangeExec, HashAggregateExec, ScanExec
from minispark.physical.planner import plan_physical
from minispark.storage.memory import MemoryDataSource


def make_scan(rows):
    schema = Schema([Field("country", STRING), Field("revenue", INT)])
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=len(rows)))
    return Scan(Dataset(schema, [partition]), "sales")


def test_plan_physical_translates_aggregate_to_partial_exchange_final():
    scan = make_scan([{"country": "US", "revenue": 10}])
    agg = Aggregate(scan, [col("country")], [count("*").alias("n")])
    physical = plan_physical(agg, shuffle_partitions=3)

    assert isinstance(physical, HashAggregateExec)
    assert physical.is_partial is False
    exchange = physical.child
    assert isinstance(exchange, ExchangeExec)
    assert exchange.num_partitions == 3
    partial = exchange.child
    assert isinstance(partial, HashAggregateExec)
    assert partial.is_partial is True
    assert isinstance(partial.child, ScanExec)


def test_final_schema_matches_logical_schema():
    scan = make_scan([{"country": "US", "revenue": 10}])
    agg = Aggregate(scan, [col("country")], [ssum("revenue").alias("total")])
    physical = plan_physical(agg)
    assert physical.schema.field_names() == agg.schema.field_names()


def test_partial_execution_groups_within_one_partition():
    rows = [
        {"country": "US", "revenue": 10},
        {"country": "US", "revenue": 20},
        {"country": "CA", "revenue": 5},
    ]
    scan = make_scan(rows)
    agg = Aggregate(scan, [col("country")], [count("*").alias("n"), ssum("revenue").alias("total")])
    physical = plan_physical(agg, shuffle_partitions=1)
    partial_exec = physical.child.child

    partial_partition = execute_partition(partial_exec, 0)
    partial_rows = {r["country"]: r for r in partial_partition.to_list()}
    assert partial_rows["US"]["__agg_state_0"] == 2
    assert partial_rows["US"]["__agg_state_1"] == 30
    assert partial_rows["CA"]["__agg_state_0"] == 1
    assert partial_rows["CA"]["__agg_state_1"] == 5


def test_final_execution_merges_partial_states_into_output_names():
    """Simulates the reduce side by feeding a partial-aggregate ScanExec
    directly into the final HashAggregateExec, without a real shuffle in
    between (execution/scheduler.py covers the real shuffle path)."""
    partial_schema = Schema(
        [Field("country", STRING), Field("__agg_state_0", STRING), Field("__agg_state_1", STRING)]
    )
    partial_rows = [
        {"country": "US", "__agg_state_0": 2, "__agg_state_1": 30},
        {"country": "US", "__agg_state_0": 1, "__agg_state_1": 5},  # a second source partition
        {"country": "CA", "__agg_state_0": 1, "__agg_state_1": 5},
    ]
    fake_shuffle_read = ScanExec(
        Dataset(
            partial_schema,
            [Partition(0, partial_schema, lambda: iter(partial_rows), PartitionMetadata())],
        ),
        "fake",
    )
    scan = make_scan([{"country": "US", "revenue": 1}])
    agg = Aggregate(scan, [col("country")], [count("*").alias("n"), ssum("revenue").alias("total")])
    physical = plan_physical(agg, shuffle_partitions=1)
    final_exec = HashAggregateExec(
        fake_shuffle_read, physical.group_by, physical.aggregates, physical.schema, is_partial=False
    )

    result = {r["country"]: r for r in execute_partition(final_exec, 0).to_list()}
    assert result["US"]["n"] == 3
    assert result["US"]["total"] == 35
    assert result["CA"]["n"] == 1
    assert result["CA"]["total"] == 5


def test_physical_plan_with_aggregate_is_picklable():
    # A real MemoryDataSource, not make_scan()'s raw-lambda Partition: only
    # a genuine data source builds picklable records_fn (see storage/
    # memory.py's _make_records_fn docstring). make_scan() is fine for
    # every other test in this file, which never pickles its plan.
    dataset = MemoryDataSource([{"country": "US", "revenue": 10}], num_partitions=1).read()
    scan = Scan(dataset, "sales")
    agg = Aggregate(scan, [col("country")], [avg("revenue").alias("avg_rev")])
    physical = plan_physical(agg)
    restored = pickle.loads(pickle.dumps(physical))
    rows = execute_partition(restored.child.child, 0).to_list()
    assert rows == [{"country": "US", "__agg_state_0": (10, 1)}]
