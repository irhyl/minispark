"""Pretty-printing for plan trees (used by DataFrame.explain()).

Works for both LogicalPlan and physical/plan.py's PhysicalPlan: both share
the same `node_label` / `children` shape, described here as `ExplainNode`
rather than importing LogicalPlan specifically, so physical/planner.py does
not need its own copy of this printer for `df.explain(optimized=True)`'s
"== Physical Plan ==" section.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class ExplainNode(Protocol):
    """`list` is invariant, so a `children` property typed `list[LogicalPlan]`
    would not structurally satisfy `children() -> list[ExplainNode]` even
    though LogicalPlan and PhysicalPlan both fit this shape. `Sequence` is
    declared covariant, which is what makes that structural match work."""

    @property
    def node_label(self) -> str: ...

    @property
    def children(self) -> Sequence[ExplainNode]: ...


def explain_string(plan: ExplainNode) -> str:
    lines: list[str] = []
    _render(plan, prefix="", lines=lines)
    return "\n".join(lines)


def _render(plan: ExplainNode, prefix: str, lines: list[str]) -> None:
    lines.append(f"{prefix}{plan.node_label}")
    for child in plan.children:
        _render(child, prefix=prefix + "  ", lines=lines)
