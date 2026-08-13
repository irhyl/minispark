"""DAG: the dependency graph a physical plan forms.

Two dependency classes, per the build spec:

  * Narrow: a child partition depends on exactly one parent partition
    (map, filter, project). No data needs to move between partitions.
  * Wide: a child partition depends on data from every parent partition
    (group by, join, sort). Producing it needs a shuffle: every parent
    partition must finish and be written out before any wide-dependency
    task can start.

`ScanExec`, `FilterExec`, `ProjectExec`, `HashAggregateExec`,
`HashJoinExec`, and `SortExec` are all narrow: even `SortExec` only sorts
the rows *within* the one partition it is given (see
physical/operators.py); the ordering it produces is only locally correct
until the shuffle around it has moved rows into the right target
partitions. `ExchangeExec` is the one wide node: it is exactly the marker
the physical planner leaves at a shuffle boundary (Aggregate's
partial-aggregate-then-shuffle-then-final-aggregate shape, a
HashJoinExec's shuffled or broadcast side, or order_by()'s local-sort-
then-range-shuffle-then-final-sort shape, see physical/planner.py).
`ShuffleWriteExec`/`ShuffleReadExec` (the post-stage-split rewrite of an
`ExchangeExec`, see stages.py) are narrow again: by the time either
exists, the wide dependency they came from has already been resolved
into two separate stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from minispark.physical.plan import (
    ExchangeExec,
    FilterExec,
    HashAggregateExec,
    HashJoinExec,
    PhysicalPlan,
    ProjectExec,
    ScanExec,
    ShuffleReadExec,
    ShuffleWriteExec,
    SortExec,
)

_NARROW_NODE_TYPES = (
    ScanExec,
    FilterExec,
    ProjectExec,
    HashAggregateExec,
    HashJoinExec,
    SortExec,
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
