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


class HashAggregateExec(PhysicalPlan):
    """Groups rows by `group_by` and applies `aggregates`.

    The same node type is used for both the partial (pre-shuffle,
    map-side) and final (post-shuffle, reduce-side) aggregation passes:
    the grouping logic is identical either way (see
    physical/operators.py); only the *input* differs (raw rows for
    partial, upstream partial states for final) and whether
    AggregateFunction.update() or .merge() combines rows into a group.
    `is_partial` selects between them. `schema` is supplied by the
    planner (physical/planner.py), not recomputed here, exactly like
    ProjectExec: a partial aggregate's schema (group columns plus opaque
    internal state columns) and a final aggregate's schema (group columns
    plus named, typed aggregate outputs) are different enough that
    deriving both here would duplicate what the planner already knows.
    """

    def __init__(
        self,
        child: PhysicalPlan,
        group_by: list[Expression],
        aggregates: list[Expression],
        schema: Schema,
        is_partial: bool,
    ):
        self.child = child
        self.group_by = group_by
        self.aggregates = aggregates
        self._schema = schema
        self.is_partial = is_partial

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        kind = "partial" if self.is_partial else "final"
        group_cols = ", ".join(output_name(g) for g in self.group_by)
        agg_cols = ", ".join(output_name(a) for a in self.aggregates)
        return f"HashAggregateExec[{kind}](groupBy=({group_cols}), aggregates=({agg_cols}))"


class ExchangeExec(PhysicalPlan):
    """Marks a shuffle boundary, as produced by the physical planner.

    Never executed directly: `execution/stages.py`'s `build_stages()`
    rewrites every `ExchangeExec` into a `ShuffleWriteExec` (ending the
    upstream stage) and a `ShuffleReadExec` (starting the downstream
    stage) before a plan ever reaches a Task. A bare `ExchangeExec`
    reaching `physical/operators.py` means stage splitting was skipped or
    is broken, not something a Task legitimately holds.
    """

    def __init__(
        self, child: PhysicalPlan, num_partitions: int, partition_exprs: list[Expression]
    ):
        self.child = child
        self.num_partitions = num_partitions
        self.partition_exprs = partition_exprs

    @property
    def schema(self) -> Schema:
        return self.child.schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        keys = ", ".join(output_name(e) for e in self.partition_exprs)
        return f"Exchange[hash({keys}), {self.num_partitions} partitions]"


class ShuffleWriteExec(PhysicalPlan):
    """Terminal node of a stage whose task output must be hash-partitioned
    and written to shuffle storage (`shuffle/writer.py`) instead of
    returned as this query's rows. Built by `execution/stages.py`'s
    `build_stages()` from an `ExchangeExec`'s position in the plan, never
    constructed by `physical/planner.py` directly.
    """

    def __init__(
        self, child: PhysicalPlan, num_partitions: int, partition_exprs: list[Expression]
    ):
        self.child = child
        self.num_partitions = num_partitions
        self.partition_exprs = partition_exprs

    @property
    def schema(self) -> Schema:
        return self.child.schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        keys = ", ".join(output_name(e) for e in self.partition_exprs)
        return f"ShuffleWriteExec[hash({keys}), {self.num_partitions} partitions]"


class ShuffleReadExec(PhysicalPlan):
    """Leaf node of a stage that reads a prior stage's shuffled output for
    exactly one target partition, in place of a Scan reading from a
    DataSource. Built by `execution/stages.py`'s `build_stages()`.
    """

    def __init__(self, from_stage_id: int, schema: Schema):
        self.from_stage_id = from_stage_id
        self._schema = schema

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def node_label(self) -> str:
        return f"ShuffleReadExec[stage {self.from_stage_id}]"
