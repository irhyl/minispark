"""SQL front-end: parses a SQL string into the same LogicalPlan nodes
(`minispark.logical.nodes`) the DataFrame API builds, per the build
spec's rule that there must not be a separate SQL execution engine. See
`minispark.sql.parser`'s module docstring for the supported grammar and
`docs/sql.md` for the full picture (why a hand-written parser, exactly
what SQL subset is covered, and why).
"""

from __future__ import annotations
