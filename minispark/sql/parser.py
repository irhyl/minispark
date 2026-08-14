"""SQL parser: turns a token stream into a LogicalPlan, reusing exactly
the same nodes (`logical.nodes`) and expressions (`expressions.*`) the
DataFrame API builds. Per the build spec's rule that there must not be a
separate SQL execution engine, this is a translator, not a second
interpreter: `session.sql(...)`'s output goes through the exact same
analyzer/optimizer/physical-planner/scheduler path any DataFrame does
(see api/session.py's `sql()`).

Supported grammar (deliberately no more than what the DataFrame API can
already express; a SQL front-end is a translator, not a capability
expansion, see docs/sql.md):

    select_stmt   := SELECT select_list FROM table_ref [join_clause]
                     [WHERE expr] [GROUP BY expr_list] [HAVING expr]
                     [ORDER BY order_item_list]
    select_list   := '*' | select_item (',' select_item)*
    select_item   := expr [AS IDENTIFIER]
    table_ref     := IDENTIFIER
    join_clause   := JOIN IDENTIFIER ON qualified_name '=' qualified_name
    expr_list     := expr (',' expr)*
    order_item    := expr [ASC | DESC]

    expr          := or_expr
    or_expr       := and_expr (OR and_expr)*
    and_expr      := not_expr (AND not_expr)*
    not_expr      := NOT not_expr | null_check
    null_check    := comparison [IS [NOT] NULL]
    comparison    := additive [(= | != | <> | < | <= | > | >=) additive]
    additive      := multiplicative (('+' | '-') multiplicative)*
    multiplicative:= unary (('*' | '/') unary)*
    unary         := '-' unary | primary
    primary       := NUMBER | STRING | TRUE | FALSE | NULL
                    | IDENTIFIER '(' ('*' | [expr (',' expr)*]) ')'
                    | qualified_name | '(' expr ')'
    qualified_name:= IDENTIFIER ['.' IDENTIFIER]

`qualified_name`'s table-qualifier (`orders.id`) is accepted for
readability but discarded, not used for disambiguation: MiniSpark's
expression tree resolves a `Column` by bare name only (see expressions/
column.py), the same as every `col("x")` built through the DataFrame
API. A query relying on `t1.x`/`t2.x` meaning two different columns
would already be rejected earlier, at `Schema.__init__`'s duplicate-
field-name check, when the join's merged schema is built; SQL does not
special-case or improve on that.
"""

from __future__ import annotations

from minispark.expressions.aggregate import AggregateFunction, Avg, Count, Max, Min, Sum
from minispark.expressions.base import Alias, Expression
from minispark.expressions.binary import (
    Add,
    And,
    BinaryExpression,
    Divide,
    Equal,
    GreaterEqual,
    GreaterThan,
    LessEqual,
    LessThan,
    Multiply,
    NotEqual,
    Or,
    Subtract,
)
from minispark.expressions.column import Column
from minispark.expressions.literal import Literal
from minispark.expressions.predicates import IsNotNull, IsNull, Not
from minispark.logical.nodes import Aggregate, Filter, Join, LogicalPlan, Project, Sort, output_name
from minispark.sql.tokenizer import SqlSyntaxError, Token, TokenType, tokenize

_AGGREGATE_FUNCTION_NAMES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
_COMPARISON_OPERATORS = {
    "=": Equal, "!=": NotEqual, "<>": NotEqual,
    "<": LessThan, "<=": LessEqual, ">": GreaterThan, ">=": GreaterEqual,
}


class SqlParseError(SqlSyntaxError):
    """A syntactically tokenizable but structurally invalid or
    unsupported SQL string: an unknown table name, a function that is
    not one of COUNT/SUM/AVG/MIN/MAX, a SELECT list item that is neither
    aggregated nor a GROUP BY column, a JOIN ... ON comparing two
    differently-named columns (MiniSpark's Join only supports same-named
    `on=` columns, see logical/nodes.py's Join docstring), or a plain
    token-stream syntax error (expected X, got Y)."""


def parse_sql(query: str, tables: dict[str, LogicalPlan]) -> LogicalPlan:
    """Parse `query` into a LogicalPlan. `tables` maps a table name (as
    it appears in a `FROM`/`JOIN` clause) to the LogicalPlan it refers
    to, already resolved by the caller (`api/session.py`'s `sql()`,
    which resolves names against `MiniSparkSession`'s registered temp
    views); this function itself never touches session state, so it
    stays testable with a plain dict.
    """
    return _Parser(tokenize(query), tables).parse_select_stmt()


class _Parser:
    def __init__(self, tokens: list[Token], tables: dict[str, LogicalPlan]):
        self._tokens = tokens
        self._pos = 0
        self._tables = tables

    # ---- token stream helpers ------------------------------------------

    @property
    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        if token.type is not TokenType.EOF:
            self._pos += 1
        return token

    def _check_keyword(self, keyword: str) -> bool:
        return self._current.type is TokenType.KEYWORD and self._current.value == keyword

    def _match_keyword(self, keyword: str) -> bool:
        if self._check_keyword(keyword):
            self._advance()
            return True
        return False

    def _expect_keyword(self, keyword: str) -> None:
        if not self._match_keyword(keyword):
            raise SqlParseError(
                f"Expected {keyword!r} at position {self._current.position}, "
                f"got {self._current.value!r}"
            )

    def _match_punctuation(self, value: str) -> bool:
        if self._current.type is TokenType.PUNCTUATION and self._current.value == value:
            self._advance()
            return True
        return False

    def _expect_punctuation(self, value: str) -> None:
        if not self._match_punctuation(value):
            raise SqlParseError(
                f"Expected {value!r} at position {self._current.position}, "
                f"got {self._current.value!r}"
            )

    def _expect_identifier(self) -> str:
        if self._current.type is not TokenType.IDENTIFIER:
            raise SqlParseError(
                f"Expected an identifier at position {self._current.position}, "
                f"got {self._current.value!r}"
            )
        return self._advance().value

    def _match_operator(self, value: str) -> bool:
        if self._current.type is TokenType.OPERATOR and self._current.value == value:
            self._advance()
            return True
        return False

    # ---- statement --------------------------------------------------------

    def parse_select_stmt(self) -> LogicalPlan:
        self._expect_keyword("SELECT")
        select_items = self._parse_select_list()
        self._expect_keyword("FROM")
        plan = self._resolve_table(self._expect_identifier())

        if self._match_keyword("JOIN"):
            plan = self._parse_join(plan)

        if self._match_keyword("WHERE"):
            plan = Filter(plan, self._parse_expr())

        group_by = self._parse_group_by() if self._match_keyword("GROUP") else []

        plan = self._apply_select(plan, select_items, group_by)

        if self._match_keyword("HAVING"):
            having_expr = self._parse_expr()
            if isinstance(plan, Aggregate):
                # HAVING's condition runs *after* Aggregate, against rows
                # that already hold finalized aggregate values keyed by
                # output name, e.g. plan.aggregates' own alias; an
                # AggregateFunction call written directly in HAVING
                # (`HAVING COUNT(*) >= 1`) must therefore resolve to a
                # Column reference to that output, not be re-evaluated as
                # a raw AggregateFunction (which has no per-row value and
                # would raise, see expressions/aggregate.py's evaluate()).
                having_expr = _substitute_aggregates_with_output_columns(
                    having_expr, plan.aggregates
                )
            plan = Filter(plan, having_expr)

        if self._match_keyword("ORDER"):
            plan = self._parse_order_by(plan)

        if self._current.type is not TokenType.EOF:
            raise SqlParseError(
                f"Unexpected {self._current.value!r} at position {self._current.position}"
            )
        return plan

    def _resolve_table(self, name: str) -> LogicalPlan:
        try:
            return self._tables[name]
        except KeyError:
            raise SqlParseError(
                f"Unknown table {name!r}. Register it first with "
                f"session.create_or_replace_temp_view({name!r}, df)."
            ) from None

    def _parse_join(self, left: LogicalPlan) -> LogicalPlan:
        right_name = self._expect_identifier()
        right = self._resolve_table(right_name)
        self._expect_keyword("ON")
        left_col = self._parse_qualified_name()
        self._expect_operator("=")
        right_col = self._parse_qualified_name()
        if left_col != right_col:
            raise SqlParseError(
                f"JOIN ... ON {left_col} = {right_col}: MiniSpark's Join only supports "
                "joining on a column with the same name on both sides (see "
                "logical/nodes.py's Join docstring), not differently-named keys."
            )
        return Join(left, right, on=[left_col], how="inner")

    def _expect_operator(self, value: str) -> None:
        if not self._match_operator(value):
            raise SqlParseError(
                f"Expected {value!r} at position {self._current.position}, "
                f"got {self._current.value!r}"
            )

    # ---- SELECT list / GROUP BY / aggregate assembly -----------------------

    def _parse_select_list(self) -> list[Expression] | None:
        """Returns None for `SELECT *`."""
        if self._match_operator("*"):
            return None
        items = [self._parse_select_item()]
        while self._match_punctuation(","):
            items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> Expression:
        expr = self._parse_expr()
        if self._match_keyword("AS"):
            return Alias(expr, self._expect_identifier())
        return expr

    def _parse_group_by(self) -> list[Expression]:
        self._expect_keyword("BY")
        items = [self._parse_expr()]
        while self._match_punctuation(","):
            items.append(self._parse_expr())
        return items

    def _apply_select(
        self, plan: LogicalPlan, select_items: list[Expression] | None, group_by: list[Expression]
    ) -> LogicalPlan:
        if select_items is None:
            if group_by:
                raise SqlParseError("SELECT * cannot be combined with GROUP BY")
            return plan

        aggregates = [item for item in select_items if _contains_aggregate(item)]
        plain_items = [item for item in select_items if not _contains_aggregate(item)]

        if not group_by and not aggregates:
            return Project(plan, select_items)

        if not group_by and aggregates and plain_items:
            raise SqlParseError(
                "SELECT list mixes an aggregate with a non-aggregated column and there "
                "is no GROUP BY; every non-aggregated column must appear in GROUP BY."
            )

        group_by_names = {
            name for name in (_plain_column_name(expr) for expr in group_by) if name is not None
        }
        for item in plain_items:
            name = _plain_column_name(item)
            if name is None or name not in group_by_names:
                raise SqlParseError(
                    f"SELECT list item {item!r} is neither aggregated nor a GROUP BY column"
                )
        return Aggregate(plan, group_by, aggregates)

    # ---- ORDER BY -----------------------------------------------------------

    def _parse_order_by(self, plan: LogicalPlan) -> LogicalPlan:
        self._expect_keyword("BY")
        exprs: list[Expression] = []
        ascending: list[bool] = []
        while True:
            exprs.append(self._parse_expr())
            if self._match_keyword("DESC"):
                ascending.append(False)
            else:
                self._match_keyword("ASC")
                ascending.append(True)
            if not self._match_punctuation(","):
                break
        return Sort(plan, exprs, ascending)

    # ---- expressions (precedence climbing) -----------------------------------

    def _parse_expr(self) -> Expression:
        return self._parse_or()

    def _parse_or(self) -> Expression:
        left = self._parse_and()
        while self._match_keyword("OR"):
            left = Or(left, self._parse_and())
        return left

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        while self._match_keyword("AND"):
            left = And(left, self._parse_not())
        return left

    def _parse_not(self) -> Expression:
        if self._match_keyword("NOT"):
            return Not(self._parse_not())
        return self._parse_null_check()

    def _parse_null_check(self) -> Expression:
        expr = self._parse_comparison()
        if self._match_keyword("IS"):
            if self._match_keyword("NOT"):
                self._expect_keyword("NULL")
                return IsNotNull(expr)
            self._expect_keyword("NULL")
            return IsNull(expr)
        return expr

    def _parse_comparison(self) -> Expression:
        left = self._parse_additive()
        is_comparison = (
            self._current.type is TokenType.OPERATOR
            and self._current.value in _COMPARISON_OPERATORS
        )
        if is_comparison:
            op_cls = _COMPARISON_OPERATORS[self._advance().value]
            return op_cls(left, self._parse_additive())
        return left

    def _parse_additive(self) -> Expression:
        left = self._parse_multiplicative()
        while self._current.type is TokenType.OPERATOR and self._current.value in ("+", "-"):
            op = self._advance().value
            right = self._parse_multiplicative()
            left = Add(left, right) if op == "+" else Subtract(left, right)
        return left

    def _parse_multiplicative(self) -> Expression:
        left = self._parse_unary()
        while self._current.type is TokenType.OPERATOR and self._current.value in ("*", "/"):
            op = self._advance().value
            right = self._parse_unary()
            left = Multiply(left, right) if op == "*" else Divide(left, right)
        return left

    def _parse_unary(self) -> Expression:
        if self._match_operator("-"):
            return Multiply(Literal(-1), self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        token = self._current
        if token.type is TokenType.NUMBER:
            self._advance()
            return Literal(float(token.value) if "." in token.value else int(token.value))
        if token.type is TokenType.STRING:
            self._advance()
            return Literal(token.value)
        if self._match_keyword("TRUE"):
            return Literal(True)
        if self._match_keyword("FALSE"):
            return Literal(False)
        if self._match_keyword("NULL"):
            return Literal(None)
        if self._match_punctuation("("):
            expr = self._parse_expr()
            self._expect_punctuation(")")
            return expr
        if token.type is TokenType.IDENTIFIER:
            name = self._advance().value
            if self._current.type is TokenType.PUNCTUATION and self._current.value == "(":
                return self._parse_function_call(name)
            if self._match_punctuation("."):
                return Column(self._expect_identifier())
            return Column(name)
        raise SqlParseError(f"Unexpected token {token.value!r} at position {token.position}")

    def _parse_qualified_name(self) -> str:
        name = self._expect_identifier()
        if self._match_punctuation("."):
            return self._expect_identifier()
        return name

    def _parse_function_call(self, name: str) -> Expression:
        self._expect_punctuation("(")
        upper = name.upper()
        if upper not in _AGGREGATE_FUNCTION_NAMES:
            raise SqlParseError(
                f"Unknown function {name!r}: only COUNT/SUM/AVG/MIN/MAX are supported, "
                "matching what the DataFrame API's aggregate functions cover "
                "(see api/functions.py)."
            )
        if upper == "COUNT" and self._match_operator("*"):
            self._expect_punctuation(")")
            return Count(None)
        arg = self._parse_expr()
        self._expect_punctuation(")")
        if upper == "COUNT":
            return Count(arg)
        if upper == "SUM":
            return Sum(arg)
        if upper == "AVG":
            return Avg(arg)
        if upper == "MIN":
            return Min(arg)
        return Max(arg)


def _contains_aggregate(expr: Expression) -> bool:
    if isinstance(expr, AggregateFunction):
        return True
    return any(_contains_aggregate(child) for child in expr.children)


def _plain_column_name(expr: Expression) -> str | None:
    inner = expr.child if isinstance(expr, Alias) else expr
    return inner.name if isinstance(inner, Column) else None


def _substitute_aggregates_with_output_columns(
    expr: Expression, aggregate_items: list[Expression]
) -> Expression:
    """Rebuild `expr`, replacing every `AggregateFunction` node with a
    `Column` reference to the matching SELECT-list aggregate's output
    name (`plan.aggregates`' own alias, or its `repr()` if unaliased,
    matching `logical/nodes.py`'s own `output_name()`). Matching is
    structural (`repr()` equality, not `==`: `Expression.__eq__` is
    overloaded to build an `Equal` expression node, not compare for
    equality, the same reason `optimizer/optimizer.py`'s fixed-point
    check compares `explain_string()` text instead of `==`).

    Raises `SqlParseError` if `expr` references an aggregate that is not
    present in the SELECT list: MiniSpark's HAVING only resolves against
    already-selected aggregates, it does not add a second, hidden
    aggregate the way some SQL engines allow (`HAVING AVG(age) > 10`
    when only `COUNT(*)` was selected), a deliberate scope limit, not an
    oversight.
    """
    if isinstance(expr, AggregateFunction):
        target = repr(expr)
        for item in aggregate_items:
            inner = item.child if isinstance(item, Alias) else item
            if repr(inner) == target:
                return Column(output_name(item))
        raise SqlParseError(
            f"HAVING references {expr!r}, which is not in the SELECT list. MiniSpark's "
            "HAVING only supports aggregates already selected; give it an alias in "
            "SELECT (e.g. `SELECT COUNT(*) AS n ... HAVING n >= 1`) and reference that."
        )
    if isinstance(expr, BinaryExpression):
        return type(expr)(
            _substitute_aggregates_with_output_columns(expr.left, aggregate_items),
            _substitute_aggregates_with_output_columns(expr.right, aggregate_items),
        )
    if isinstance(expr, Not):
        return Not(_substitute_aggregates_with_output_columns(expr.child, aggregate_items))
    if isinstance(expr, (IsNull, IsNotNull)):
        return type(expr)(
            _substitute_aggregates_with_output_columns(expr.child, aggregate_items)
        )
    return expr
