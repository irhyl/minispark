"""Stage: a maximal run of narrow-dependency physical operators.

A stage boundary sits at every wide dependency (an `ExchangeExec`, see
dag.py): operators after the boundary cannot start until every partition
on the producing side has finished and written its shuffle output.
`build_stages()` walks the physical plan and, at each `ExchangeExec`,
rewrites it into two pieces that belong to two different stages:

  * a `ShuffleWriteExec` ending the upstream stage (its task output must
    be hash-partitioned and written to shuffle storage instead of
    returned as query rows), and
  * a `ShuffleReadExec` starting the downstream stage (its task input
    comes from reading a prior stage's shuffle blocks instead of a Scan
    or a live parent partition).

A plan with no `ExchangeExec` (Scan/Filter/Project only, Milestone 1-3's
whole vocabulary) still produces exactly one stage, unchanged from before.
A plan with one `ExchangeExec` (`group_by().agg()`, Milestone 4) produces
two. A plan with more than one (e.g. two chained aggregations) produces
more than two: the splitting below is not special-cased to "at most one
shuffle," it walks the whole tree.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class Stage:
    stage_id: int
    plan: PhysicalPlan
    num_partitions: int


def build_stages(plan: PhysicalPlan) -> list[Stage]:
    """Split `plan` into stages, in upstream-to-downstream execution order."""
    stages: list[Stage] = []
    final_fragment, final_partitions = _split(plan, stages)
    stages.append(Stage(stage_id=len(stages), plan=final_fragment, num_partitions=final_partitions))
    return stages


def _split(plan: PhysicalPlan, stages: list[Stage]) -> tuple[PhysicalPlan, int]:
    """Rewrite `plan`, appending completed upstream stages to `stages`
    (mutated in place) at each ExchangeExec found. Returns the plan
    fragment for the current, not-yet-closed stage, and how many
    partitions that fragment has.
    """
    if isinstance(plan, ScanExec):
        return plan, plan.dataset.num_partitions()

    if isinstance(plan, ExchangeExec):
        child_fragment, child_partitions = _split(plan.child, stages)
        write_plan = ShuffleWriteExec(child_fragment, plan.num_partitions, plan.partition_exprs)
        stages.append(Stage(stage_id=len(stages), plan=write_plan, num_partitions=child_partitions))
        read_plan = ShuffleReadExec(from_stage_id=len(stages) - 1, schema=plan.schema)
        return read_plan, plan.num_partitions

    if len(plan.children) == 1:
        original_child = plan.children[0]
        child_fragment, child_partitions = _split(original_child, stages)
        # Only rebuild if something below actually changed (an Exchange
        # was found and rewritten). A plan with no shuffle boundary below
        # this node should come back out identical to how it went in, not
        # a value-equal but freshly allocated copy of every node above the
        # (nonexistent) split point.
        if child_fragment is original_child:
            return plan, child_partitions
        return _with_child(plan, child_fragment), child_partitions

    raise NotImplementedError(
        f"Stage splitting has no rule for physical node {type(plan).__name__}"
    )


def _with_child(plan: PhysicalPlan, new_child: PhysicalPlan) -> PhysicalPlan:
    """Rebuild `plan` with `new_child` in place of its original (single) child."""
    if isinstance(plan, FilterExec):
        return FilterExec(new_child, plan.condition)
    if isinstance(plan, ProjectExec):
        return ProjectExec(new_child, plan.columns, plan.schema)
    if isinstance(plan, HashAggregateExec):
        return HashAggregateExec(
            new_child, plan.group_by, plan.aggregates, plan.schema, plan.is_partial
        )
    raise NotImplementedError(
        f"Cannot rebuild physical node {type(plan).__name__} with a new child"
    )
