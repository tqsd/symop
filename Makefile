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

.PHONY: docs docs-html docs-live docs-clean

docs: docs-html

docs-html:
	$(MAKE) -C docs html

docs-live:
	$(MAKE) -C docs livehtml

docs-clean:
	$(MAKE) -C docs clean

.PHONY: coverage docs-coverage

test:
	python -m pytest

test-all:
	tox

coverage:
	python -m coverage run -m pytest
	python -m coverage html
	python -m coverage xml

docs-coverage: coverage
	rm -rf docs/source/_static/coverage
	mkdir -p docs/source/_static
	cp -r htmlcov docs/source/_static/coverage
	$(MAKE) -C docs html
