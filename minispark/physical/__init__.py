"""Physical plan: *how* a query runs, as opposed to the logical plan's *what*.

Milestone 2 scope: physical/planner.py translates an optimized LogicalPlan
into a PhysicalPlan tree (plan.py) that physical/operators.py executes.
Today the translation is 1:1, one physical node per logical node, because
there is only one implementation strategy available for Scan, Filter, and
Project. This package exists now, ahead of that being interesting, so the
seam is in place before Milestone 4/5 need it: HashAggregate versus
SortAggregate, BroadcastJoin versus HashJoin are strategy choices the
physical planner will make once Aggregate/Join logical nodes exist.
`execution/executor.py` (Milestone 1's naive interpreter of a *logical*
plan) is retained as the correctness oracle physical execution is tested
against, not as something DataFrame calls anymore.
"""
