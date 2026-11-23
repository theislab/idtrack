# Shortcuts for development, run in the Poetry environment that `make install` makes (SETUP.md). `make` lists them.
# The checks of CI (every session on every supported Python) are nox's: `nox --list-sessions` (SETUP.md).
.PHONY: help install test test-slow coverage lint format docs servedocs dist clean
.DEFAULT_GOAL := help

# The Python files of the repository, as the pre-commit hooks of CI see them (notebooks are not formatted).
PYTHON_FILES := $(shell git ls-files '*.py' '*.pyi')

help: ## list the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "%-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## install idtrack (editable) with the development tools of poetry.lock
	poetry install

test: ## run the test suite (tests marked slow are skipped)
	poetry run pytest

test-slow: ## run only the tests marked slow (network and long runs)
	poetry run pytest --runslow -m slow

coverage: ## run the test suite with a coverage report (terminal and htmlcov/)
	poetry run pytest --cov=idtrack --cov-report=term --cov-report=html

lint: ## check formatting (black, isort) and lint (flake8 with its plugins)
	poetry run black --check $(PYTHON_FILES)
	poetry run isort --check-only $(PYTHON_FILES)
	poetry run flake8 $(PYTHON_FILES)

format: ## format the code (black, isort)
	poetry run black $(PYTHON_FILES)
	poetry run isort $(PYTHON_FILES)

docs: ## build the documentation into docs/_build/html, every warning an error
	poetry run $(MAKE) -C docs clean
	poetry run $(MAKE) -C docs html

servedocs: ## serve the documentation and rebuild it when a file changes
	poetry run sphinx-autobuild docs docs/_build/html

dist: ## build the sdist and the wheel into dist/
	poetry build

clean: ## remove build, test, coverage and Python artifacts
	rm -rf build/ dist/ .eggs/ htmlcov/ .pytest_cache/ .mypy_cache/ .coverage .coverage.* coverage.xml docs/_build/
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find . -name '*.py[co]' -delete
