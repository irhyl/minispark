"""Physical plan nodes: ScanExec, FilterExec, ProjectExec.

Structurally mirror logical/nodes.py's Scan/Filter/Project on purpose: a
physical node is "a logical node plus a chosen execution strategy," and
right now there is exactly one strategy per node type, so the shape does
not yet diverge. Each node carries a `schema` (needed by DataFrame.schema
after execute() and computed once by the planner, not recomputed here) and
a `node_label` / `children` pair so `explain_string()` (logical/plan.py)
can render a physical plan the same way it renders a logical one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from minispark.core.dataset import Dataset
from minispark.core.schema import Schema
from minispark.expressions.base import Expression
from minispark.logical.nodes import output_name


class PhysicalPlan(ABC):
    @property
    @abstractmethod
    def schema(self) -> Schema: ...

    @property
    def children(self) -> list[PhysicalPlan]:
        return []

    @property
    def node_label(self) -> str:
        return type(self).__name__


class ScanExec(PhysicalPlan):
    """Reads an already-materialized Dataset. See logical/nodes.py's Scan."""

    def __init__(self, dataset: Dataset, source_name: str):
        self.dataset = dataset
        self.source_name = source_name

    @property
    def schema(self) -> Schema:
        return self.dataset.schema

    @property
    def node_label(self) -> str:
        cols = ", ".join(self.dataset.schema.field_names())
        return f"ScanExec[{self.source_name}] ({cols})"


class FilterExec(PhysicalPlan):
    def __init__(self, child: PhysicalPlan, condition: Expression):
        self.child = child
        self.condition = condition

    @property
    def schema(self) -> Schema:
        return self.child.schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        return f"FilterExec[{self.condition!r}]"


class ProjectExec(PhysicalPlan):
    def __init__(self, child: PhysicalPlan, columns: list[Expression], schema: Schema):
        self.child = child
        self.columns = columns
        self._schema = schema

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        cols = ", ".join(output_name(c) for c in self.columns)
        return f"ProjectExec[{cols}]"
