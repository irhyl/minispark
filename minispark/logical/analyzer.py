"""The analyzer: validates a logical plan before it is optimized or executed.

Milestone 1 had no analyzer, so a bad column name surfaced late, as a plain
Python KeyError, only once collect()/show()/count() actually evaluated a
Column against a row (see the note in expressions/column.py). This module
closes that gap for the plan shapes that exist so far (Scan, Filter,
Project): every Column referenced anywhere in the plan is checked against
its child's schema up front, so `df.select("does_not_exist")` fails with a
clear AnalysisException as soon as an action runs, before any row is read.

What this analyzer does not do yet: type checking beyond "does this column
exist" (arithmetic expressions still default to STRING in
logical/nodes.py's `_output_field`, there is no type-inference engine), and
join/aggregate validation (those nodes do not exist until Milestone 4/5).
Both are explicitly out of scope here, not oversights.
"""

from __future__ import annotations

from minispark.core.schema import Schema
from minispark.expressions.base import Expression
from minispark.expressions.column import Column
from minispark.logical.nodes import Filter, LogicalPlan, Project, Scan, output_name


class AnalysisException(Exception):
    """Raised when a logical plan references something that does not exist."""


def analyze(plan: LogicalPlan) -> LogicalPlan:
    """Validate `plan` and every plan below it. Returns `plan` unchanged.

    Raises AnalysisException on the first problem found. Validation is
    plan-shaped, not a rewrite: analyze() never changes the tree it is
    given, it only decides whether to raise.
    """
    for child in plan.children:
        analyze(child)

    if isinstance(plan, Scan):
        pass  # a Scan's schema is ground truth; nothing to validate against.
    elif isinstance(plan, Filter):
        _check_columns_exist(
            plan.condition, plan.child.schema, context=f"filter({plan.condition!r})"
        )
    elif isinstance(plan, Project):
        _analyze_project(plan)
    else:
        raise NotImplementedError(f"Analyzer has no rule for logical node {type(plan).__name__}")

    return plan


def _analyze_project(plan: Project) -> None:
    seen_names: set[str] = set()
    for expr in plan.columns:
        _check_columns_exist(expr, plan.child.schema, context=f"select({expr!r})")
        name = output_name(expr)
        if name in seen_names:
            raise AnalysisException(
                f"Duplicate output column '{name}' in select(); "
                "give one of them a different alias()"
            )
        seen_names.add(name)


def _check_columns_exist(expr: Expression, schema: Schema, context: str) -> None:
    for column in referenced_columns(expr):
        if not schema.has_field(column):
            raise AnalysisException(
                f"Column '{column}' not found in {context}. "
                f"Available columns: {schema.field_names()}"
            )


def referenced_columns(expr: Expression) -> set[str]:
    """Every column name referenced anywhere in `expr`'s tree.

    Walks Expression.children generically rather than pattern-matching on
    every expression subtype, so new expression types (added in later
    milestones) are covered automatically as long as they implement
    `children` correctly.
    """
    names: set[str] = set()
    if isinstance(expr, Column):
        names.add(expr.name)
    for child in expr.children:
        names |= referenced_columns(child)
    return names
