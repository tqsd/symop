.PHONY: lint format typecheck check

format:
	ruff format src

lint:
	ruff check src

typecheck:
	mypy src

contracts:
	lint-imports --config importlinter.ini

check: format lint contracts typecheck

