"""GroupedData: the result of `DataFrame.group_by()`, before `.agg()` turns
it back into a DataFrame wrapping an `Aggregate` logical plan node.

A separate, tiny class (rather than `group_by()` returning a `DataFrame`
directly) because "grouped, but not yet aggregated" has no sensible
`filter()`/`select()`/`collect()`: the build spec's own example chains
`group_by(...).agg(...)` as two distinct calls, and this type mirrors
that shape instead of smuggling a half-built `Aggregate` node into
`DataFrame`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minispark.expressions.base import Expression
from minispark.logical.nodes import Aggregate, LogicalPlan

if TYPE_CHECKING:
    from minispark.api.dataframe import DataFrame
    from minispark.api.session import MiniSparkSession


class GroupedData:
    def __init__(self, session: MiniSparkSession, plan: LogicalPlan, group_by: list[Expression]):
        self._session = session
        self._plan = plan
        self._group_by = group_by

    def agg(self, *aggregates: Expression) -> DataFrame:
        if not aggregates:
            raise ValueError("agg() requires at least one aggregate expression")
        # Imported here, not at module level: DataFrame.group_by() needs to
        # import GroupedData eagerly to construct one, so this file cannot
        # also import DataFrame eagerly without a circular import at
        # module-load time. By the time agg() actually runs, DataFrame is
        # already fully loaded (see api/dataframe.py's group_by()).
        from minispark.api.dataframe import DataFrame

        return DataFrame(self._session, Aggregate(self._plan, self._group_by, list(aggregates)))
