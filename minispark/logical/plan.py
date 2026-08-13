"""Pretty-printing for logical plan trees (used by DataFrame.explain())."""

from __future__ import annotations

from minispark.logical.nodes import LogicalPlan


def explain_string(plan: LogicalPlan) -> str:
    lines: list[str] = []
    _render(plan, prefix="", lines=lines)
    return "\n".join(lines)


def _render(plan: LogicalPlan, prefix: str, lines: list[str]) -> None:
    lines.append(f"{prefix}{plan.node_label}")
    for child in plan.children:
        _render(child, prefix=prefix + "  ", lines=lines)
