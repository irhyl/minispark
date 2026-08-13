"""DAG: the dependency graph a physical plan forms.

Two dependency classes, per the build spec:

  * Narrow: a child partition depends on exactly one parent partition
    (map, filter, project). No data needs to move between partitions.
  * Wide: a child partition depends on data from every parent partition
    (group by, join, sort). Producing it needs a shuffle: every parent
    partition must finish and be written out before any wide-dependency
    task can start.

`ScanExec`, `FilterExec`, and `ProjectExec` (the only physical nodes that
exist as of Milestone 3) are all narrow. This module still builds a real
DAGNode tree with a per-node classification, not a hardcoded "everything
is narrow" shortcut, so Milestone 4's Aggregate (and Milestone 5's Join,
Sort) only need to add a case to `dependency_kind()`; `stages.py`'s stage
splitting already reads this classification generically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from minispark.physical.plan import FilterExec, PhysicalPlan, ProjectExec, ScanExec

_NARROW_NODE_TYPES = (ScanExec, FilterExec, ProjectExec)


class DependencyKind(Enum):
    NARROW = auto()
    WIDE = auto()


def dependency_kind(plan: PhysicalPlan) -> DependencyKind:
    """The dependency `plan` has on its child(ren)."""
    if isinstance(plan, _NARROW_NODE_TYPES):
        return DependencyKind.NARROW
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
