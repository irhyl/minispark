"""Unit tests for physical/planner.py's scan-pushdown pre-pass
(_pushdown_scan_reads): does the physical planner correctly detect a
Project/Filter chain sitting on a Scan with a `source`, and re-read that
source with the right columns/filter hints?

Uses a lightweight, hand-written `RecordingSource` (no pyarrow needed):
it satisfies `logical.nodes.ScanSource` purely structurally, records
every `read(columns=, filter=)` call it receives, and actually projects
rows so the resulting physical plan can be executed and checked for
correctness end to end, not just inspected for which arguments were
passed. Real Parquet-specific pushdown behavior (row-group skipping) is
covered separately in tests/unit/test_parquet_source.py; this file is
about the planner's own detection/threading logic, independent of which
DataSource is on the other end.
"""

from __future__ import annotations

from minispark.api.functions import col
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.expressions.column import Column
from minispark.logical.nodes import Aggregate, Filter, Project, Scan
from minispark.physical.operators import execute as execute_physical
from minispark.physical.planner import plan_physical

FULL_SCHEMA = Schema([Field("name", STRING), Field("age", INT), Field("country", STRING)])
ROWS = [
    {"name": "alice", "age": 30, "country": "US"},
    {"name": "bob", "age": 17, "country": "CA"},
    {"name": "carol", "age": 45, "country": "US"},
]


class RecordingSource:
    """Structurally a `logical.nodes.ScanSource`: records every call and
    actually narrows/filters rows, so the physical plan built from it
    remains genuinely executable and checkable, not just inspectable."""

    def __init__(self):
        self.calls: list[tuple[list[str] | None, object]] = []

    def read(self, columns=None, filter=None):
        self.calls.append((columns, filter))
        schema = FULL_SCHEMA.select(columns) if columns is not None else FULL_SCHEMA
        rows = ROWS
        if filter is not None:
            rows = [r for r in rows if filter.evaluate(r)]
        if columns is not None:
            rows = [{name: r[name] for name in columns} for r in rows]
        partition = Partition(0, schema, lambda rows=rows: iter(rows), PartitionMetadata())
        return Dataset(schema, [partition])


def make_scan(source: RecordingSource) -> Scan:
    dataset = source.read()
    return Scan(dataset, "recording", source=source)


def test_project_directly_on_scan_pushes_columns():
    source = RecordingSource()
    logical = Project(make_scan(source), [Column("name")])
    plan_physical(logical)
    assert source.calls[-1] == (["name"], None)


def test_filter_directly_on_scan_pushes_the_condition():
    source = RecordingSource()
    logical = Filter(make_scan(source), col("age") > 18)
    plan_physical(logical)
    columns, filt = source.calls[-1]
    assert columns is None  # no Project restricted anything
    assert filt is not None


def test_filter_then_project_pushes_the_union_of_both():
    source = RecordingSource()
    logical = Project(Filter(make_scan(source), col("age") > 18), [Column("name")])
    plan_physical(logical)
    columns, filt = source.calls[-1]
    assert columns is not None and set(columns) == {"name", "age"}
    assert filt is not None


def test_pushdown_result_still_executes_correctly():
    source = RecordingSource()
    logical = Project(Filter(make_scan(source), col("age") > 18), [Column("name")])
    physical = plan_physical(logical)
    rows = list(execute_physical(physical).iter_records())
    assert sorted(r["name"] for r in rows) == ["alice", "carol"]


def test_filter_and_project_are_still_present_in_the_physical_plan():
    """Pushdown never elides the row-level operators: they stay in the
    plan regardless of whether the source below them actually honored
    the hint, see storage/datasource.py's DataSource.read() docstring."""
    from minispark.physical.plan import FilterExec, ProjectExec

    source = RecordingSource()
    logical = Project(Filter(make_scan(source), col("age") > 18), [Column("name")])
    physical = plan_physical(logical)
    assert isinstance(physical, ProjectExec)
    assert isinstance(physical.child, FilterExec)


def test_scan_with_no_source_is_left_untouched():
    partition = Partition(0, FULL_SCHEMA, lambda: iter(ROWS), PartitionMetadata())
    dataset = Dataset(FULL_SCHEMA, [partition])
    logical = Project(Scan(dataset, "no_source"), [Column("name")])
    physical = plan_physical(logical)
    rows = list(execute_physical(physical).iter_records())
    assert sorted(r["name"] for r in rows) == ["alice", "bob", "carol"]


def test_filter_above_a_computed_project_does_not_push_past_it():
    """A Filter whose condition depends on a computed Project's output
    (a different namespace) must not have that condition pushed further
    down past the Project to the Scan, which only has the *original*
    column names."""
    source = RecordingSource()
    computed = Project(make_scan(source), [(col("age") + col("age")).alias("double_age")])
    logical = Filter(computed, col("double_age") > 40)
    physical = plan_physical(logical)
    rows = list(execute_physical(physical).iter_records())
    assert sorted(r["double_age"] for r in rows) == [60, 90]
    # The Project's own pushdown call (for its referenced column "age")
    # must not have carried the outer Filter's condition down with it.
    columns, filt = source.calls[-1]
    assert filt is None


def test_pushdown_does_not_cross_an_aggregate_boundary():
    source = RecordingSource()
    logical = Aggregate(Filter(make_scan(source), col("age") > 0), [Column("country")], [])
    plan_physical(logical)
    # The Filter directly on Scan (inside the Aggregate's child) still
    # gets its own pushdown; nothing above the Aggregate leaks into it.
    columns, filt = source.calls[-1]
    assert filt is not None
