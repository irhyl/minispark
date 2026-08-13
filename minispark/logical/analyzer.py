"""The analyzer: validates a logical plan before it is optimized or executed.

Milestone 1 had no analyzer, so a bad column name surfaced late, as a plain
Python KeyError, only once collect()/show()/count() actually evaluated a
Column against a row (see the note in expressions/column.py). This module
closes that gap for every plan shape that exists: every Column referenced
anywhere in the plan is checked against its child's schema up front, so
`df.select("does_not_exist")` fails with a clear AnalysisException as soon
as an action runs, before any row is read.

What this analyzer does not do yet: type checking beyond "does this column
exist" (arithmetic expressions still default to STRING in
logical/nodes.py's `_output_field`, there is no type-inference engine).
"""

from __future__ import annotations

from minispark.core.schema import Schema
from minispark.expressions.aggregate import AggregateFunction
from minispark.expressions.base import Alias, Expression
from minispark.expressions.column import Column
from minispark.logical.nodes import (
    Aggregate,
    Filter,
    Join,
    LogicalPlan,
    Project,
    Scan,
    Sort,
    output_name,
)


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
    elif isinstance(plan, Join):
        _analyze_join(plan)
    elif isinstance(plan, Sort):
        _analyze_sort(plan)
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


_SUPPORTED_JOIN_TYPES = {"inner"}


def _analyze_join(plan: Join) -> None:
    if plan.how not in _SUPPORTED_JOIN_TYPES:
        raise AnalysisException(
            f"Unsupported join type {plan.how!r}; only {sorted(_SUPPORTED_JOIN_TYPES)} "
            "are implemented (left/right/full outer and semi/anti joins are not)."
        )
    if not plan.on:
        raise AnalysisException("join() requires at least one column in on=")

    left_schema = plan.left.schema
    right_schema = plan.right.schema
    for name in plan.on:
        if not left_schema.has_field(name):
            raise AnalysisException(
                f"Join column '{name}' not found on the left side. "
                f"Available columns: {left_schema.field_names()}"
            )
        if not right_schema.has_field(name):
            raise AnalysisException(
                f"Join column '{name}' not found on the right side. "
                f"Available columns: {right_schema.field_names()}"
            )

    on_set = set(plan.on)
    colliding = (set(left_schema.field_names()) & set(right_schema.field_names())) - on_set
    if colliding:
        raise AnalysisException(
            f"Column(s) {sorted(colliding)} exist on both sides of the join and are not "
            "in on=; rename or select() them on one side before joining "
            "(differently-named join keys via left_on/right_on are not supported)."
        )


def _analyze_sort(plan: Sort) -> None:
    for expr in plan.sort_exprs:
        if not isinstance(expr, Column):
            raise AnalysisException(f"order_by() only accepts column names, got {expr!r}")
        _check_columns_exist(expr, plan.child.schema, context=f"order_by({expr!r})")


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
