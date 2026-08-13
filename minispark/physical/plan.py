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

    `is_broadcast=True` marks a broadcast exchange (`DataFrame.join(...,
    broadcast=True)`, see physical/planner.py): every row goes to a single
    target partition (`num_partitions` is 1) regardless of
    `partition_exprs`, and every consuming task reads that same partition
    in full, rather than each task reading only its own partition_id's
    slice. See execution/scheduler.py for where that distinction is
    actually applied (it is a task-building decision, not something the
    physical operators themselves need to know about).

    `range_boundaries`, when not `None`, marks a *range* exchange
    (`order_by(...)`, see physical/planner.py): `partition_exprs` must be
    exactly one expression (the primary sort key), and rows are assigned
    to a target partition by where their key falls among these
    `num_partitions - 1` boundary values (`shuffle/partitioner.py`'s
    `RangePartitioner`), not by hashing. `None` (the default) means a hash
    exchange (`shuffle/partitioner.py`'s `HashPartitioner`), used by
    `group_by`/`join`.
    """

    def __init__(
        self,
        child: PhysicalPlan,
        num_partitions: int,
        partition_exprs: list[Expression],
        is_broadcast: bool = False,
        range_boundaries: list | None = None,
    ):
        self.child = child
        self.num_partitions = num_partitions
        self.partition_exprs = partition_exprs
        self.is_broadcast = is_broadcast
        self.range_boundaries = range_boundaries

    @property
    def schema(self) -> Schema:
        return self.child.schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        if self.is_broadcast:
            return "Exchange[broadcast]"
        keys = ", ".join(output_name(e) for e in self.partition_exprs)
        kind = "range" if self.range_boundaries is not None else "hash"
        return f"Exchange[{kind}({keys}), {self.num_partitions} partitions]"


class ShuffleWriteExec(PhysicalPlan):
    """Terminal node of a stage whose task output must be partitioned and
    written to shuffle storage (`shuffle/writer.py`) instead of returned
    as this query's rows. Built by `execution/stages.py`'s
    `build_stages()` from an `ExchangeExec`'s position in the plan, never
    constructed by `physical/planner.py` directly. `range_boundaries` is
    copied from that `ExchangeExec`; see its docstring.
    """

    def __init__(
        self,
        child: PhysicalPlan,
        num_partitions: int,
        partition_exprs: list[Expression],
        range_boundaries: list | None = None,
    ):
        self.child = child
        self.num_partitions = num_partitions
        self.partition_exprs = partition_exprs
        self.range_boundaries = range_boundaries

    @property
    def schema(self) -> Schema:
        return self.child.schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        keys = ", ".join(output_name(e) for e in self.partition_exprs)
        kind = "range" if self.range_boundaries is not None else "hash"
        return f"ShuffleWriteExec[{kind}({keys}), {self.num_partitions} partitions]"


class ShuffleReadExec(PhysicalPlan):
    """Leaf node of a stage that reads a prior stage's shuffled output, in
    place of a Scan reading from a DataSource. Built by
    `execution/stages.py`'s `build_stages()`.

    For a normal (non-broadcast) read, each task reads only its own
    partition_id's blocks. For a broadcast read (`is_broadcast=True`,
    copied from the `ExchangeExec` this was rewritten from), every task in
    the consuming stage reads the *same* single target partition (0) in
    full, regardless of its own partition_id: see execution/scheduler.py's
    task-building code, which is where this flag is actually consulted.
    """

    def __init__(self, from_stage_id: int, schema: Schema, is_broadcast: bool = False):
        self.from_stage_id = from_stage_id
        self._schema = schema
        self.is_broadcast = is_broadcast

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def node_label(self) -> str:
        tag = " broadcast" if self.is_broadcast else ""
        return f"ShuffleReadExec[stage {self.from_stage_id}]{tag}"


class HashJoinExec(PhysicalPlan):
    """Inner equi-join: build a hash table on `left`'s rows keyed by
    `left_keys`, probe it with `right`'s rows keyed by `right_keys`.

    This one node type covers both join strategies the build spec asks
    for; "broadcast join" versus "shuffle hash join" is entirely a
    decision the physical planner makes about *how `left`/`right` arrive*
    (see physical/planner.py): a shuffle hash join wraps both sides in an
    `ExchangeExec`; a broadcast join wraps only the smaller side (in a
    broadcast `ExchangeExec`) and leaves the other side exactly as it was,
    unshuffled. The per-partition build-and-probe logic
    (physical/operators.py) does not need to know, or care, which case it
    is in: by the time it runs, both `left` and `right` already produce
    exactly the rows this partition's join needs to see.
    """

    def __init__(
        self,
        left: PhysicalPlan,
        right: PhysicalPlan,
        left_keys: list[Expression],
        right_keys: list[Expression],
        on: list[str],
        schema: Schema,
    ):
        self.left = left
        self.right = right
        self.left_keys = left_keys
        self.right_keys = right_keys
        self.on = on
        self._schema = schema

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.left, self.right]

    @property
    def node_label(self) -> str:
        on_cols = ", ".join(self.on)
        return f"HashJoinExec[inner, on=({on_cols})]"


class SortExec(PhysicalPlan):
    """Sorts the rows of exactly one partition by `sort_exprs`/`ascending`.

    Used twice in a full `order_by()` plan (see physical/planner.py), and
    is the same node type both times: once as a *local* pre-sort (each
    source partition sorted independently, before the range-partitioning
    shuffle), and once as the *final* sort (each post-shuffle target
    partition sorted independently; since a `RangePartitioner` assigns
    entire contiguous key ranges to each target partition, sorting every
    target partition and reading them out in partition order gives a
    globally sorted result). The per-partition sorting logic
    (physical/operators.py) does not need to know which case it is in.
    """

    def __init__(
        self,
        child: PhysicalPlan,
        sort_exprs: list[Expression],
        ascending: list[bool],
        schema: Schema,
    ):
        self.child = child
        self.sort_exprs = sort_exprs
        self.ascending = ascending
        self._schema = schema

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def children(self) -> list[PhysicalPlan]:
        return [self.child]

    @property
    def node_label(self) -> str:
        cols = ", ".join(
            f"{output_name(e)} {'ASC' if asc else 'DESC'}"
            for e, asc in zip(self.sort_exprs, self.ascending, strict=True)
        )
        return f"SortExec[{cols}]"


def leaves(plan: PhysicalPlan) -> list[PhysicalPlan]:
    """Every leaf (a node with no children) reachable from `plan`.

    Generic over `children`, so it works whether `plan` has one child
    (most nodes), two (`HashJoinExec`), or none (`ScanExec`/
    `ShuffleReadExec` themselves). Used by execution/worker.py and
    execution/scheduler.py to find every `ShuffleReadExec` a stage's plan
    depends on, a HashJoinExec-rooted stage can have more than one.
    """
    if not plan.children:
        return [plan]
    return [leaf for child in plan.children for leaf in leaves(child)]
