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
	if exist build rd /s /q build
	if exist dist rd /s /q dist
	if exist .eggs rd /s /q .eggs
	for /d /r . %%d in (*.egg-info) do @if exist "%%d" echo "%%d" && rd /s/q "%%d"
	del /q /s /f .\*.egg


clean-pyc: ## remove Python file artifacts
	del /s /f /q .\*.pyc
	del /s /f /q .\*.pyo
	del /s /f /q .\*~
	for /d /r . %%d in (*__pycache__) do @if exist "%%d" echo "%%d" && rd /s/q "%%d"

clean-test: ## remove test and coverage artifacts
	if exist .tox rd /s /q .tox
	if exist .coverage del /f /q .coverage
	if exist .coverage.* del /f /q .coverage.*
	if exist coverage.xml del /f /q coverage.xml
	if exist htmlcov rd /s /q htmlcov
	if exist .pytest_cache rd /s /q .pytest_cache

lint: ## check style with flake8
	flake8 idtrack tests

test: ## run tests quickly with the default Python
	pytest

test-slow: ## run slow tests (network/integration tests)
	pytest --runslow -m slow

test-integration: ## run all integration tests (including slow ones)
	pytest --runslow -m integration

test-all-with-slow: ## run all tests including slow tests
	pytest --runslow -m '' tests/

test-coverage: ## run tests with coverage report
	pytest --cov=idtrack --cov-report=term --cov-report=html

test-all: ## run tests on every Python version with nox
	nox

coverage: ## check code coverage quickly with the default Python
	coverage run --source idtrack -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov\index.html

docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs\_build\html\index.html

servedocs: ## serve the documentation and rebuild it when a file changes
	sphinx-autobuild docs docs\_build\html

dist: clean-build clean-pyc ## builds source and wheel package
	poetry build

install: clean-build clean-pyc ## install the package to the active Python's site-packages
	poetry install
