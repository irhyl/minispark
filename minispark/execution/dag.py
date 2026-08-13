"""DAG: the dependency graph a physical plan forms.

Two dependency classes, per the build spec:

  * Narrow: a child partition depends on exactly one parent partition
    (map, filter, project). No data needs to move between partitions.
  * Wide: a child partition depends on data from every parent partition
    (group by, join, sort). Producing it needs a shuffle: every parent
    partition must finish and be written out before any wide-dependency
    task can start.

`ScanExec`, `FilterExec`, `ProjectExec`, and `HashAggregateExec` are all
narrow: even `HashAggregateExec` only groups rows *within* the one
partition it is given (see physical/operators.py), it does not itself
move data between partitions. `ExchangeExec` is the one wide node: it is
exactly the marker the physical planner leaves at a shuffle boundary
(Aggregate's partial-aggregate-then-shuffle-then-final-aggregate shape,
see physical/planner.py). `ShuffleWriteExec`/`ShuffleReadExec` (the
post-stage-split rewrite of an `ExchangeExec`, see stages.py) are narrow
again: by the time either exists, the wide dependency they came from has
already been resolved into two separate stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from minispark.physical.plan import (
    ExchangeExec,
    FilterExec,
    HashAggregateExec,
    PhysicalPlan,
    ProjectExec,
    ScanExec,
    ShuffleReadExec,
    ShuffleWriteExec,
)

_NARROW_NODE_TYPES = (
    ScanExec,
    FilterExec,
    ProjectExec,
    HashAggregateExec,
    ShuffleWriteExec,
    ShuffleReadExec,
)
_WIDE_NODE_TYPES = (ExchangeExec,)


class DependencyKind(Enum):
    NARROW = auto()
    WIDE = auto()


def dependency_kind(plan: PhysicalPlan) -> DependencyKind:
    """The dependency `plan` has on its child(ren)."""
    if isinstance(plan, _NARROW_NODE_TYPES):
        return DependencyKind.NARROW
    if isinstance(plan, _WIDE_NODE_TYPES):
        return DependencyKind.WIDE
    raise NotImplementedError(
        f"No dependency classification for physical node {type(plan).__name__}"
    )


@dataclass
class DAGNode:
    plan: PhysicalPlan
    dependency: DependencyKind
    children: list[DAGNode]


def build_dag(plan: PhysicalPlan) -> DAGNode:
    return DAGNode(
        plan=plan,
        dependency=dependency_kind(plan),
        children=[build_dag(child) for child in plan.children],
    )


def has_wide_dependency(node: DAGNode) -> bool:
    if node.dependency is DependencyKind.WIDE:
        return True
    return any(has_wide_dependency(child) for child in node.children)
