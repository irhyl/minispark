"""Optimizer rules.

Each rule is a pure function from LogicalPlan to LogicalPlan: it never
mutates its input, it returns a new (possibly identical) tree. `Rule` is a
thin ABC rather than a bare function so `Optimizer` can carry a `name` for
logging/debugging and so later milestones can add rules with constructor
state (e.g. a rule that needs access to `optimizer/statistics.py`) without
changing the calling convention.

Every rule here rewrites plan *shape* using only information already on the
plan (schemas, expression trees). None of them consult table statistics;
that is deliberate for Milestone 2. statistics.py exists for later
milestones (join strategy selection) to use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeGuard

from minispark.expressions.base import Alias, Expression
from minispark.expressions.binary import And, BinaryExpression, Or
from minispark.expressions.column import Column
from minispark.expressions.literal import Literal
from minispark.expressions.predicates import IsNotNull, IsNull, Not
from minispark.logical.analyzer import referenced_columns
from minispark.logical.nodes import Aggregate, Filter, Join, LogicalPlan, Project, Scan, Sort


class Rule(ABC):
    name: str = "Rule"

    @abstractmethod
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        """Return a rewritten plan. May return `plan` itself if nothing changed."""


def map_expressions(plan: LogicalPlan, fn: Callable[[Expression], Expression]) -> LogicalPlan:
    """Rebuild `plan`, applying `fn` to every expression it holds (Filter's
    condition, Project's columns, Aggregate's group_by and aggregates).

    Isinstance dispatch over the node types that exist, matching the style
    already used in execution/executor.py. A generic visitor abstraction
    is not worth it yet for six node types; revisit if the branch count
    keeps growing. `fn` is not applied inside an AggregateFunction's own
    child expression (e.g. `sum(age + 0)`'s `age + 0`): neither `_fold`
    nor `_simplify_bool` below has a case for AggregateFunction, so it is
    returned unchanged rather than recursed into; that leaves a foldable
    expression un-folded inside an aggregate, a missed optimization, not a
    correctness problem.
    """
    if isinstance(plan, Scan):
        return plan
    if isinstance(plan, Filter):
        return Filter(map_expressions(plan.child, fn), fn(plan.condition))
    if isinstance(plan, Project):
        return Project(map_expressions(plan.child, fn), [fn(c) for c in plan.columns])
    if isinstance(plan, Aggregate):
        return Aggregate(
            map_expressions(plan.child, fn),
            [fn(g) for g in plan.group_by],
            [fn(a) for a in plan.aggregates],
        )
    if isinstance(plan, Join):
        return Join(
            map_expressions(plan.left, fn),
            map_expressions(plan.right, fn),
            plan.on,
            plan.how,
            plan.broadcast,
        )
    if isinstance(plan, Sort):
        return Sort(
            map_expressions(plan.child, fn), [fn(e) for e in plan.sort_exprs], plan.ascending
        )
    raise NotImplementedError(f"map_expressions has no rule for logical node {type(plan).__name__}")


# ---- constant folding ------------------------------------------------------


def _fold(expr: Expression) -> Expression:
    if isinstance(expr, BinaryExpression):
        left = _fold(expr.left)
        right = _fold(expr.right)
        if isinstance(left, Literal) and isinstance(right, Literal):
            return Literal(type(expr).op(left.value, right.value))
        return type(expr)(left, right)
    if isinstance(expr, Alias):
        return Alias(_fold(expr.child), expr.name)
    if isinstance(expr, Not):
        child = _fold(expr.child)
        if isinstance(child, Literal):
            return Literal(not bool(child.value))
        return Not(child)
    if isinstance(expr, (IsNull, IsNotNull)):
        child = _fold(expr.child)
        if isinstance(child, Literal):
            is_null = child.value is None
            return Literal(is_null if isinstance(expr, IsNull) else not is_null)
        return type(expr)(child)
    return expr


class ConstantFolding(Rule):
    """Evaluate sub-expressions made entirely of literals at plan time.

    `age > 10 + 10` becomes `age > 20`: `10 + 10` has no Column reference,
    so it can be evaluated once here instead of once per row at execution.
    """

    name = "ConstantFolding"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        return map_expressions(plan, _fold)


# ---- filter simplification --------------------------------------------------


def _simplify_bool(expr: Expression) -> Expression:
    if isinstance(expr, And):
        left = _simplify_bool(expr.left)
        right = _simplify_bool(expr.right)
        if isinstance(left, Literal):
            return right if left.value else Literal(False)
        if isinstance(right, Literal):
            return left if right.value else Literal(False)
        return And(left, right)
    if isinstance(expr, Or):
        left = _simplify_bool(expr.left)
        right = _simplify_bool(expr.right)
        if isinstance(left, Literal):
            return Literal(True) if left.value else right
        if isinstance(right, Literal):
            return Literal(True) if right.value else left
        return Or(left, right)
    if isinstance(expr, Not):
        child = _simplify_bool(expr.child)
        if isinstance(child, Not):
            return child.child
        return Not(child)
    if isinstance(expr, BinaryExpression):
        return type(expr)(_simplify_bool(expr.left), _simplify_bool(expr.right))
    if isinstance(expr, Alias):
        return Alias(_simplify_bool(expr.child), expr.name)
    return expr


class FilterSimplification(Rule):
    """Simplify boolean expressions: `x AND True` -> `x`, `Not(Not(x))` -> `x`, etc.

    Runs after ConstantFolding so conditions like `x > 10 AND (1 == 1)`
    have already had `1 == 1` folded to `Literal(True)` by the time this
    rule sees them. Only simplifies the expression tree in place; it does
    not remove a Filter node even when its condition folds down to
    `Literal(True)` or `Literal(False)` (that would need a dedicated
    "always true / always false" plan rewrite, which is not implemented
    here to avoid introducing a new plan node just for this case).
    """

    name = "FilterSimplification"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        return map_expressions(plan, _simplify_bool)


# ---- predicate pushdown -----------------------------------------------------


class PredicatePushdown(Rule):
    """Push a Filter below a Project when the filter does not need the
    Project, and below a Join into whichever side it exclusively needs.

    `df.select("name", "age").filter(col("age") > 18)` builds
    `Filter(Project(Scan, [name, age]), age > 18)`. Filtering after
    projecting means every row is projected before most of them get
    thrown away. This rule swaps the two when every column the filter
    needs is already present on the Project's child, producing
    `Project(Filter(Scan, age > 18), [name, age])` instead: rows are
    dropped before the (comparatively cheap, here) projection work runs.

    The Join case is the textbook predicate-pushdown example:
    `Filter(Join(left, right, on), cond)` becomes
    `Join(Filter(left, cond), right, on)` when `cond` only references
    left's columns (or the symmetric case for right). This is only valid
    because `Join.how` is always `"inner"` (see logical/nodes.py's `Join`
    docstring): pushing a filter into one side of an outer join can drop
    rows the join's null-padding semantics would otherwise have kept, so
    this rule would need to check `how` before doing this if outer joins
    existed.
    """

    name = "PredicatePushdown"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        return _pushdown(plan)


def _pushdown(plan: LogicalPlan) -> LogicalPlan:
    if isinstance(plan, Scan):
        return plan
    if isinstance(plan, Project):
        return Project(_pushdown(plan.child), plan.columns)
    if isinstance(plan, Aggregate):
        # A Filter above an Aggregate filters *aggregated* results (a
        # HAVING clause, semantically); it cannot be pushed through the
        # Aggregate into pre-aggregation rows without changing what it
        # means. The isinstance(new_child, Project) check below already
        # never matches an Aggregate, so no swap happens here regardless;
        # this branch only makes _pushdown able to recurse past Aggregate
        # at all.
        return Aggregate(_pushdown(plan.child), plan.group_by, plan.aggregates)
    if isinstance(plan, Join):
        return Join(
            _pushdown(plan.left), _pushdown(plan.right), plan.on, plan.how, plan.broadcast
        )
    if isinstance(plan, Sort):
        return Sort(_pushdown(plan.child), plan.sort_exprs, plan.ascending)
    if isinstance(plan, Filter):
        new_child = _pushdown(plan.child)
        if isinstance(new_child, Project):
            needed = referenced_columns(plan.condition)
            available = set(new_child.child.schema.field_names())
            if needed <= available:
                return Project(Filter(new_child.child, plan.condition), new_child.columns)
        if isinstance(new_child, Join):
            needed = referenced_columns(plan.condition)
            left_fields = set(new_child.left.schema.field_names())
            right_fields = set(new_child.right.schema.field_names())
            if needed <= left_fields:
                return Join(
                    Filter(new_child.left, plan.condition),
                    new_child.right,
                    new_child.on,
                    new_child.how,
                    new_child.broadcast,
                )
            if needed <= right_fields:
                return Join(
                    new_child.left,
                    Filter(new_child.right, plan.condition),
                    new_child.on,
                    new_child.how,
                    new_child.broadcast,
                )
        return Filter(new_child, plan.condition)
    raise NotImplementedError(
        f"PredicatePushdown has no rule for logical node {type(plan).__name__}"
    )


# ---- projection pruning -----------------------------------------------------


class ProjectionPruning(Rule):
    """Insert a minimal Project directly above Scan, keeping only referenced columns.

    Computes, top-down, the set of columns actually needed by everything
    above a given point in the plan (final Project output plus every
    Filter condition in between) and, if the Scan's schema is wider than
    that, wraps it in a Column-only Project.

    Honesty check on what this buys: CSVDataSource.read() (storage/csv.py)
    parses every column of every row regardless of what is requested. This
    rule reduces the width of the Record dicts flowing through Filter and
    Project below it (less per-row work, less memory per row in transit)
    but does not reduce bytes read from disk. True source-level pruning
    (skip parsing unrequested CSV columns, or Parquet column-level reads)
    is out of scope until the physical/storage layers can honor a
    "requested columns" hint, which is not implemented yet.
    """

    name = "ProjectionPruning"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        return _prune(plan, needed=None)


def _prune(plan: LogicalPlan, needed: set[str] | None) -> LogicalPlan:
    if isinstance(plan, Scan):
        schema_cols = plan.schema.field_names()
        if needed is not None and set(schema_cols) - needed:
            keep = [c for c in schema_cols if c in needed]
            if keep and len(keep) < len(schema_cols):
                return Project(plan, [Column(c) for c in keep])
        return plan
    if isinstance(plan, Filter):
        # `needed=None` means "no Project above has restricted the output
        # columns yet", i.e. every column reaching this point is still
        # needed. A Filter does not narrow that on its own (it does not
        # change which columns exist, only which rows survive) -- it only
        # *adds* its own condition's columns to whatever was already
        # required. Folding `needed=None` down to just `cond_cols` here
        # would wrongly drop every column the filter's own condition does
        # not reference, even when nothing above ever asked for that.
        cond_cols = referenced_columns(plan.condition)
        child_needed = None if needed is None else (needed | cond_cols)
        return Filter(_prune(plan.child, child_needed), plan.condition)
    if isinstance(plan, Project):
        proj_cols: set[str] = set()
        for expr in plan.columns:
            proj_cols |= referenced_columns(expr)
        return Project(_prune(plan.child, proj_cols), plan.columns)
    if isinstance(plan, Aggregate):
        # Like Project, an Aggregate fully determines what it needs from
        # its child (group_by columns plus every aggregate's own child
        # column): nothing above an Aggregate can reach a column that
        # is not one of its own outputs, so `needed` from above does not
        # apply to the child side and is replaced, not merged.
        agg_cols: set[str] = set()
        for g in plan.group_by:
            agg_cols |= referenced_columns(g)
        for a in plan.aggregates:
            agg_cols |= referenced_columns(a)
        return Aggregate(_prune(plan.child, agg_cols), plan.group_by, plan.aggregates)
    if isinstance(plan, Join):
        # Also a replace, like Aggregate: a Join's own output only ever
        # contains left's and right's columns (minus right's `on`
        # duplicates), so `needed` from above is split by which side each
        # name belongs to, plus the `on` columns, which the join itself
        # needs from both sides regardless of what is needed above it.
        on_set = set(plan.on)
        left_fields = set(plan.left.schema.field_names())
        right_fields = set(plan.right.schema.field_names())
        if needed is None:
            left_needed, right_needed = left_fields, right_fields
        else:
            left_needed = (needed & left_fields) | on_set
            right_needed = (needed & right_fields) | on_set
        return Join(
            _prune(plan.left, left_needed),
            _prune(plan.right, right_needed),
            plan.on,
            plan.how,
            plan.broadcast,
        )
    if isinstance(plan, Sort):
        # Like Filter: Sort does not change which columns exist, only
        # their order, so it adds its own sort-key columns to whatever
        # was already required rather than replacing it.
        sort_cols: set[str] = set()
        for e in plan.sort_exprs:
            sort_cols |= referenced_columns(e)
        child_needed = None if needed is None else (needed | sort_cols)
        return Sort(_prune(plan.child, child_needed), plan.sort_exprs, plan.ascending)
    raise NotImplementedError(
        f"ProjectionPruning has no rule for logical node {type(plan).__name__}"
    )


# ---- redundant projection elimination ---------------------------------------


def _is_plain_column_projection(node: LogicalPlan) -> TypeGuard[Project]:
    """True if `node` is a Project whose columns are all plain Column refs.

    Typed as a TypeGuard (not just `-> bool`) so `if _is_plain_column_
    projection(new_child):` narrows `new_child` to `Project` for the type
    checker, letting `new_child.child` below type-check without a cast.
    """
    return isinstance(node, Project) and all(isinstance(c, Column) for c in node.columns)


class RedundantProjectionElimination(Rule):
    """Remove Project nodes that do no real work.

    Two cases:

    1. Identity projection: a Project whose columns are plain Column
       references, in the same order as its child's schema, is a no-op
       and is replaced by its child directly.
    2. Collapsing a plain-Column Project directly below another Project:
       if the child only renames-nothing/selects a subset of columns
       (no Alias, no computed expression), the parent Project's
       expressions can reference the grandchild's schema directly, since
       the values passed through are unchanged. This is the common case
       ProjectionPruning produces (a plain-Column pruning Project inserted
       next to the user's own select()).

    General Project-of-Project fusion (folding an Alias or a computed
    expression from the inner Project into the outer one, e.g. rewriting
    references to a computed column) is not implemented: it needs
    expression substitution, which is more machinery than today's node
    set justifies.
    """

    name = "RedundantProjectionElimination"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        return _eliminate(plan)


def _eliminate(plan: LogicalPlan) -> LogicalPlan:
    if isinstance(plan, Scan):
        return plan
    if isinstance(plan, Filter):
        return Filter(_eliminate(plan.child), plan.condition)
    if isinstance(plan, Aggregate):
        return Aggregate(_eliminate(plan.child), plan.group_by, plan.aggregates)
    if isinstance(plan, Join):
        return Join(
            _eliminate(plan.left), _eliminate(plan.right), plan.on, plan.how, plan.broadcast
        )
    if isinstance(plan, Sort):
        return Sort(_eliminate(plan.child), plan.sort_exprs, plan.ascending)
    if isinstance(plan, Project):
        new_child = _eliminate(plan.child)
        if _is_plain_column_projection(plan):
            column_names = [c.name for c in plan.columns if isinstance(c, Column)]
            if column_names == new_child.schema.field_names():
                return new_child
        if _is_plain_column_projection(new_child):
            return Project(new_child.child, plan.columns)
        return Project(new_child, plan.columns)
    raise NotImplementedError(
        f"RedundantProjectionElimination has no rule for logical node {type(plan).__name__}"
    )
