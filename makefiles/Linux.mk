.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os
import sys
import webbrowser
from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re
import sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage .coverage.* coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 idtrack tests

test: ## run tests quickly with the default Python
	pytest

test-slow: ## run slow tests (network/integration tests)
	pytest -m slow

test-integration: ## run all integration tests
	pytest -m integration

test-all-with-slow: ## run all tests including slow tests
	pytest --ignore-glob='**/test_external_mappers.py' -m '' tests/

test-full: ## run complete test suite including slow and external-mappers
	pytest -m ''

test-coverage: ## run tests with coverage report
	pytest --cov=idtrack --cov-report=term --cov-report=html

test-all: ## run tests on every Python version with nox
	nox

coverage: ## check code coverage quickly with the default Python
	coverage run --source idtrack -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/idtrack.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ idtrack
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	poetry publish --build

dist: clean-build clean-pyc ## builds source and wheel package
	poetry build

install: clean-build clean-pyc ## install the package to the active Python's site-packages
	poetry install
