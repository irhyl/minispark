"""The analyzer: validates a logical plan before it is optimized or executed.

Milestone 1 had no analyzer, so a bad column name surfaced late, as a plain
Python KeyError, only once collect()/show()/count() actually evaluated a
Column against a row (see the note in expressions/column.py). This module
closes that gap for the plan shapes that exist so far (Scan, Filter,
Project, Aggregate): every Column referenced anywhere in the plan is
checked against its child's schema up front, so `df.select("does_not_exist")`
fails with a clear AnalysisException as soon as an action runs, before any
row is read.

What this analyzer does not do yet: type checking beyond "does this column
exist" (arithmetic expressions still default to STRING in
logical/nodes.py's `_output_field`, there is no type-inference engine), and
join validation (Join does not exist until Milestone 5).
"""

from __future__ import annotations

from minispark.core.schema import Schema
from minispark.expressions.aggregate import AggregateFunction
from minispark.expressions.base import Alias, Expression
from minispark.expressions.column import Column
from minispark.logical.nodes import Aggregate, Filter, LogicalPlan, Project, Scan, output_name


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
    elif isinstance(plan, Aggregate):
        _analyze_aggregate(plan)
    else:
        raise NotImplementedError(f"Analyzer has no rule for logical node {type(plan).__name__}")

    return plan


def _analyze_project(plan: Project) -> None:
    seen_names: set[str] = set()
    for expr in plan.columns:
        _check_columns_exist(expr, plan.child.schema, context=f"select({expr!r})")
        _reject_duplicate_name(output_name(expr), seen_names, context="select()")


def _analyze_aggregate(plan: Aggregate) -> None:
    seen_names: set[str] = set()
    for group_expr in plan.group_by:
        if not isinstance(group_expr, Column):
            raise AnalysisException(
                f"group_by() only accepts column names, got {group_expr!r}"
            )
        _check_columns_exist(group_expr, plan.child.schema, context=f"group_by({group_expr!r})")
        _reject_duplicate_name(output_name(group_expr), seen_names, context="group_by()/agg()")

    for agg_expr in plan.aggregates:
        inner = agg_expr.child if isinstance(agg_expr, Alias) else agg_expr
        if not isinstance(inner, AggregateFunction):
            raise AnalysisException(
                f"agg() only accepts aggregate expressions (count/sum/avg/min/max), "
                f"got {agg_expr!r}"
            )
        if inner.child is not None:
            _check_columns_exist(inner.child, plan.child.schema, context=f"agg({agg_expr!r})")
        _reject_duplicate_name(output_name(agg_expr), seen_names, context="group_by()/agg()")


def _reject_duplicate_name(name: str, seen_names: set[str], context: str) -> None:
    if name in seen_names:
        raise AnalysisException(
            f"Duplicate output column '{name}' in {context}; give one of them "
            "a different alias()"
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
