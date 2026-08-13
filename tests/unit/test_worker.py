from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.execution.tasks import Task, TaskState
from minispark.execution.worker import execute_task
from minispark.logical.nodes import Filter, Project, Scan
from minispark.physical.planner import plan_physical


def make_plan():
    schema = Schema([Field("name", STRING), Field("age", INT)])
    rows = [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": 17},
        {"name": "carol", "age": 45},
    ]
    partition = Partition(0, schema, lambda: iter(rows), PartitionMetadata(row_count=len(rows)))
    scan = Scan(Dataset(schema, [partition]), "users")
    logical = Project(Filter(scan, col("age") > 18), [col("name")])
    return plan_physical(logical)


def test_execute_task_success_returns_rows_and_metrics():
    plan = make_plan()
    task = Task(task_id=0, stage_id=0, partition_id=0, plan=plan)
    result = execute_task(task)
    assert result.state is TaskState.SUCCESS
    assert result.error is None
    assert sorted(result.rows, key=lambda r: r["name"]) == [{"name": "alice"}, {"name": "carol"}]
    assert result.metrics.output_records == 2
    assert result.metrics.input_records == 3
    assert result.metrics.execution_time_seconds >= 0
    assert result.metrics.shuffle_bytes == 0


def test_execute_task_failure_returns_failed_result_not_an_exception():
    plan = make_plan()
    # partition_id 5 does not exist on a single-partition Dataset: this
    # should surface as a FAILED TaskResult, not propagate as an exception,
    # exactly as it would if the failure happened inside a worker process.
    task = Task(task_id=0, stage_id=0, partition_id=5, plan=plan)
    result = execute_task(task)
    assert result.state is TaskState.FAILED
    assert result.rows == []
    assert result.error is not None
    assert "IndexError" in result.error
