.PHONY: test lint format typecheck

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy minispark
