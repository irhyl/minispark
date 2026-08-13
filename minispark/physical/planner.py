"""Physical planner: translates an (analyzed, optimized) LogicalPlan into a PhysicalPlan.

One physical node per logical node, in the same shape, because Scan,
Filter, and Project each have exactly one execution strategy today. This
function is the seam where that stops being true: once Aggregate/Join
logical nodes exist, `plan_physical` becomes the place that picks
HashAggregate vs SortAggregate or HashJoin vs BroadcastJoin, instead of a
1:1 structural copy.
"""

from __future__ import annotations

from minispark.logical.nodes import Filter, LogicalPlan, Project, Scan
from minispark.physical.plan import FilterExec, PhysicalPlan, ProjectExec, ScanExec


def plan_physical(logical_plan: LogicalPlan) -> PhysicalPlan:
    if isinstance(logical_plan, Scan):
        return ScanExec(logical_plan.dataset, logical_plan.source_name)
    if isinstance(logical_plan, Filter):
        return FilterExec(plan_physical(logical_plan.child), logical_plan.condition)
    if isinstance(logical_plan, Project):
        return ProjectExec(
            plan_physical(logical_plan.child), logical_plan.columns, logical_plan.schema
        )
    raise NotImplementedError(
        f"No physical strategy implemented for logical node {type(logical_plan).__name__}"
    )
