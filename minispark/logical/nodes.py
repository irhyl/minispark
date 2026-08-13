"""Logical plan nodes: Scan, Filter, Project.

Only these three exist in Milestone 1 (per the build prompt's explicit
scope). Aggregate, Join, Sort, Limit, Union, Repartition, Distinct are
listed in the target architecture but are added alongside the milestones
that give them meaning (Aggregate in Milestone 4, Join/Sort in Milestone
5, etc.) rather than stubbed out empty now.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from minispark.core.dataset import Dataset
from minispark.core.schema import Field, Schema
from minispark.core.types import STRING
from minispark.expressions.base import Alias, Expression
from minispark.expressions.column import Column


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
    """

    def __init__(self, dataset: Dataset, source_name: str):
        self.dataset = dataset
        self.source_name = source_name

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
        cols = ", ".join(_output_name(c) for c in self.columns)
        return f"Project[{cols}]"


def _output_name(expr: Expression) -> str:
    if isinstance(expr, Alias):
        return expr.name
    if isinstance(expr, Column):
        return expr.name
    return repr(expr)


def _output_field(expr: Expression, child_schema: Schema) -> Field:
    name = _output_name(expr)
    inner = expr.child if isinstance(expr, Alias) else expr
    if isinstance(inner, Column) and child_schema.has_field(inner.name):
        source_field = child_schema.get_field(inner.name)
        return Field(name, source_field.data_type, source_field.nullable)
    # Computed expressions (arithmetic, etc.) have no static type inference
    # in Milestone 1 — no analyzer exists yet to derive one. Default to
    # STRING/nullable so schema propagation never crashes; Milestone 2's
    # analyzer is expected to replace this with real type inference.
    return Field(name, STRING, nullable=True)
