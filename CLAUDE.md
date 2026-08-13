# CLAUDE.md

Operating rules for working in this repository. See `docs/build-spec.md` for
the full project spec, technology constraints, and milestone breakdown, and
`docs/architecture.md` for the current layering and design decisions.

## Commit conventions

- One commit per file. Do not bundle multiple files into a single commit,
  even within the same feature or milestone.
- Plain conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`,
  `chore:`, ...). Describe what changed and why in plain language.
- No mention of AI, LLM, or Claude anywhere in a commit: no
  `Co-Authored-By: Claude`, no "Generated with Claude Code" trailer, no
  reference to an assistant having written the change. Commits read as if
  written directly by the author.
- No em dashes in commit messages, code comments, or docs in this repo. Use
  a period, comma, or parentheses instead.
- Never use `--no-verify`, `--amend`, or force-push without being explicitly
  asked.

## Working style

- Follow the milestone order in `docs/build-spec.md`. Do not implement a
  later milestone's subsystem while an earlier one is still in progress.
- Every simplification, incomplete optimization, or hardware-limited
  benchmark must be documented inline (docstring/comment) or in
  `docs/architecture.md`, not left implicit.
- Never fabricate benchmark numbers. If something cannot be measured on this
  machine, record it as "NOT RUN (hardware limitation)".
