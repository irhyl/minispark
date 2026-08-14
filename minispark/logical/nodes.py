"""Logical plan nodes: Scan, Filter, Project, Aggregate, Join, Sort.

Limit, Union, Repartition, Distinct are listed in the target architecture
but are added alongside the milestone that gives each one meaning, rather
than stubbed out empty now.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from minispark.core.dataset import Dataset
from minispark.core.schema import Field, Schema
from minispark.core.types import STRING, DataType
from minispark.expressions.aggregate import AggregateFunction
from minispark.expressions.base import Alias, Expression
from minispark.expressions.column import Column


class ScanSource(Protocol):
    """Structural stand-in for `storage.datasource.DataSource`.

    `logical/` deliberately never imports `storage/` (see `Scan`'s
    docstring): the logical-plan layer only knows the data model
    (`Dataset`) a source produces, not the storage layer's I/O code. A
    `Protocol` lets `Scan` type-check and call `.read(columns=, filter=)`
    on whatever `source` it was given without importing the concrete
    `DataSource` class that satisfies it; any object with a matching
    `read` method (every `DataSource` subclass already has one) works,
    purely structurally, no inheritance or registration needed.
    """

    def read(self, columns: list[str] | None = None, filter: Expression | None = None) -> Dataset:
        ...


class LogicalPlan(ABC):
    """Base class for all logical plan nodes."""

    @property
    @abstractmethod
    def schema(self) -> Schema:
        """The output schema this node produces, without executing anything."""

    @property
    def children(self) -> list[LogicalPlan]:
        return []

    @property
    def node_label(self) -> str:
        """One-line description of this node, used by the explain() printer."""
        return type(self).__name__


class Scan(LogicalPlan):
    """Reads an already-materialized Dataset (see minispark.storage).

    The Dataset itself is lazy per-partition (see core/partition.py), so
    "already materialized" only means schema + partition boundaries are
    known — row data is still pulled on demand during execution.

    `source`, when given, is the `ScanSource` (structurally, a
    `storage.datasource.DataSource`) that produced `dataset`, kept around
    so `physical/planner.py` can re-invoke `.read(columns=, filter=)` with
    pushdown hints computed by the optimizer, discarding this Scan's
    original, unpruned `dataset` in favor of a freshly, more narrowly read
    one (see physical/planner.py's scan-pushdown pass). `None` (the
    default, and the only option before Milestone 7) means this Scan
    cannot be pushed into further: nothing re-reads it, it is used exactly
    as built. A hand-built `Scan(dataset, name)` in a test, with no
    `source`, still works exactly as it always has.
    """

    def __init__(self, dataset: Dataset, source_name: str, source: ScanSource | None = None):
        self.dataset = dataset
        self.source_name = source_name
        self.source = source

    @property
    def schema(self) -> Schema:
        return self.dataset.schema

    @property
    def node_label(self) -> str:
        cols = ", ".join(self.dataset.schema.field_names())
        return f"Scan[{self.source_name}] ({cols})"


class Filter(LogicalPlan):
    def __init__(self, child: LogicalPlan, condition: Expression):
        self.child = child
        self.condition = condition

    @property
    def schema(self) -> Schema:
        return self.child.schema

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        return f"Filter[{self.condition!r}]"


class Project(LogicalPlan):
    def __init__(self, child: LogicalPlan, columns: list[Expression]):
        self.child = child
        self.columns = columns

    @property
    def schema(self) -> Schema:
        fields = [_output_field(expr, self.child.schema) for expr in self.columns]
        return Schema(fields)

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        cols = ", ".join(output_name(c) for c in self.columns)
        return f"Project[{cols}]"


class Aggregate(LogicalPlan):
    """Groups `child`'s rows by `group_by`, computing `aggregates` per group.

    `group_by` entries must be plain `Column` expressions (grouping by a
    computed expression is not supported): the group key is evaluated per
    row as the shuffle-partitioning key (see physical/planner.py and
    shuffle/partitioner.py), so it needs to be something meaningful to
    hash, not an arbitrary expression tree. `aggregates` holds
    `AggregateFunction` expressions, optionally wrapped in `Alias` for
    output naming (e.g. `count("*").alias("users")`).
    """

    def __init__(
        self,
        child: LogicalPlan,
        group_by: list[Expression],
        aggregates: list[Expression],
    ):
        self.child = child
        self.group_by = group_by
        self.aggregates = aggregates

    @property
    def schema(self) -> Schema:
        group_fields = [_output_field(g, self.child.schema) for g in self.group_by]
        agg_fields = [_aggregate_output_field(a, self.child.schema) for a in self.aggregates]
        return Schema(group_fields + agg_fields)

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        group_cols = ", ".join(output_name(g) for g in self.group_by)
        agg_cols = ", ".join(output_name(a) for a in self.aggregates)
        return f"Aggregate[groupBy=({group_cols}), aggregates=({agg_cols})]"


class Join(LogicalPlan):
    """Inner equi-join of `left` and `right` on the column names in `on`.

    Scope, deliberately: `on` is a list of column names present (with the
    same name) on *both* sides, matching the common-case `df.join(other,
    on="col")` form; differently-named join keys (`left_on`/`right_on`)
    are not supported. `how` only accepts `"inner"`: left/right/full outer
    and semi/anti joins are not implemented. Neither limitation is an
    oversight; both are the scope this milestone's node set covers,
    documented so a later milestone extending `Join` knows exactly what
    it is extending.

    `broadcast` is an explicit hint (`DataFrame.join(..., broadcast=True)`),
    not an automatic decision: MiniSpark has no persistent table catalog
    and no reliable byte-size estimate to threshold against without
    scanning data (see optimizer/statistics.py's documented caveats), so
    automatic broadcast-vs-shuffle selection is not implemented. See
    physical/planner.py for how this hint changes the physical plan.
    """

    def __init__(
        self,
        left: LogicalPlan,
        right: LogicalPlan,
        on: list[str],
        how: str = "inner",
        broadcast: bool = False,
    ):
        self.left = left
        self.right = right
        self.on = on
        self.how = how
        self.broadcast = broadcast

    @property
    def schema(self) -> Schema:
        left_fields = list(self.left.schema)
        on_set = set(self.on)
        right_fields = [f for f in self.right.schema if f.name not in on_set]
        return Schema(left_fields + right_fields)

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.left, self.right]

    @property
    def node_label(self) -> str:
        on_cols = ", ".join(self.on)
        tag = " broadcast" if self.broadcast else ""
        return f"Join[{self.how}, on=({on_cols})]{tag}"


class Sort(LogicalPlan):
    """Globally sorts `child`'s rows by `sort_exprs`.

    Each entry in `sort_exprs` must be a plain `Column` (enforced by the
    analyzer), for the same reason `Aggregate.group_by` requires it: the
    sort key is evaluated per row both for local sorting and as the
    range-partitioning key for the shuffle that produces a global order
    (see physical/planner.py, shuffle/partitioner.py's RangePartitioner),
    so it needs to be something meaningful to partition by, not an
    arbitrary expression tree. `ascending[i]` controls the direction of
    `sort_exprs[i]`.
    """

    def __init__(self, child: LogicalPlan, sort_exprs: list[Expression], ascending: list[bool]):
        if len(sort_exprs) != len(ascending):
            raise ValueError("sort_exprs and ascending must be the same length")
        self.child = child
        self.sort_exprs = sort_exprs
        self.ascending = ascending

    @property
    def schema(self) -> Schema:
        return self.child.schema

    @property
    def children(self) -> list[LogicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        cols = ", ".join(
            f"{output_name(e)} {'ASC' if asc else 'DESC'}"
            for e, asc in zip(self.sort_exprs, self.ascending, strict=True)
        )
        return f"Sort[{cols}]"


def output_name(expr: Expression) -> str:
    if isinstance(expr, Alias):
        return expr.name
    if isinstance(expr, Column):
        return expr.name
    return repr(expr)


def _output_field(expr: Expression, child_schema: Schema) -> Field:
    name = output_name(expr)
    inner = expr.child if isinstance(expr, Alias) else expr
    if isinstance(inner, Column) and child_schema.has_field(inner.name):
        source_field = child_schema.get_field(inner.name)
        return Field(name, source_field.data_type, source_field.nullable)
    # Computed expressions (arithmetic, etc.) have no static type inference.
    # The Milestone 2 analyzer validates that referenced columns exist, but
    # does not yet infer result types for arithmetic; default to
    # STRING/nullable so schema propagation never crashes. Real type
    # inference is not implemented.
    return Field(name, STRING, nullable=True)


def _infer_child_type(expr: Expression, schema: Schema) -> DataType:
    """The DataType `expr` would produce, if it is a plain Column lookup.

    Same fallback as `_output_field`: a computed (non-Column) expression
    defaults to STRING rather than attempting real type inference.
    """
    if isinstance(expr, Column) and schema.has_field(expr.name):
        return schema.get_field(expr.name).data_type
    return STRING


def _aggregate_output_field(expr: Expression, child_schema: Schema) -> Field:
    name = output_name(expr)
    inner = expr.child if isinstance(expr, Alias) else expr
    if not isinstance(inner, AggregateFunction):
        raise ValueError(f"Expected an aggregate expression, got {expr!r}")
    child_type = _infer_child_type(inner.child, child_schema) if inner.child is not None else STRING
    return Field(name, inner.result_type(child_type), nullable=True)
