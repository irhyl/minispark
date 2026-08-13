"""Stage: a maximal run of narrow-dependency physical operators.

A stage boundary sits at every wide dependency (a shuffle): operators
after the boundary cannot start until every partition on the producing
side has finished. Since no physical node is wide yet (see dag.py), every
plan today produces exactly one Stage holding the whole plan; there is no
shuffle boundary to cut at. `build_stages()` still routes through
`build_dag()` and checks for wide dependencies explicitly, rather than
hardcoding "one stage," so the case that matters (a real cut point) has
somewhere to go once Milestone 4 adds Aggregate: splitting the DAG at
each wide dependency into upstream/downstream stages, where a downstream
stage's tasks read the upstream stage's materialized shuffle output
instead of a live PhysicalPlan.
"""

from __future__ import annotations

from dataclasses import dataclass

from minispark.execution.dag import build_dag, has_wide_dependency
from minispark.physical.plan import PhysicalPlan, ScanExec


@dataclass
class Stage:
    stage_id: int
    plan: PhysicalPlan
    num_partitions: int


def build_stages(plan: PhysicalPlan) -> list[Stage]:
    """Split `plan` into stages, in upstream-to-downstream execution order.

    Raises NotImplementedError if `plan` contains a wide dependency: that
    would require an actual shuffle-boundary split, which does not exist
    until Milestone 4. Every plan buildable today (Scan/Filter/Project
    only) is all-narrow, so this always succeeds with exactly one stage.
    """
    dag = build_dag(plan)
    if has_wide_dependency(dag):
        raise NotImplementedError(
            "Stage splitting at a wide dependency is not implemented yet "
            "(no physical node is wide until Milestone 4's Aggregate)."
        )
    return [Stage(stage_id=0, plan=plan, num_partitions=_leaf_num_partitions(plan))]


def _leaf_num_partitions(plan: PhysicalPlan) -> int:
    if isinstance(plan, ScanExec):
        return plan.dataset.num_partitions()
    if plan.children:
        return _leaf_num_partitions(plan.children[0])
    raise NotImplementedError(
        f"Cannot determine partition count: {type(plan).__name__} has no Scan leaf"
    )
