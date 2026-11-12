=================
Contributor Guide
=================

Thank you for your interest in contributing to IDTrack! This project is open-source
under the `BSD 3-Clause License`_ and welcomes contributions from the community.

.. contents::
    :local:
    :depth: 2
    :backlinks: none


Ways to Contribute
------------------

There are many ways to contribute to IDTrack, regardless of your experience level:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Contribution Type
      - Description
    * - **Report Bugs**
      - Help us identify issues and improve reliability
    * - **Request Features**
      - Suggest new functionality or enhancements
    * - **Improve Documentation**
      - Fix typos, clarify explanations, add examples
    * - **Submit Code**
      - Fix bugs or implement new features
    * - **Add Organism Support**
      - Declare a new species and configure its external databases (see `Adding Organism Support`_)
    * - **Answer Questions**
      - Help other users in discussions and issues


Quick Links
-----------

.. list-table::
    :widths: 30 70

    * - Source Code
      - `github.com/theislab/idtrack <https://github.com/theislab/idtrack>`_
    * - Documentation
      - `idtrack.readthedocs.io <https://idtrack.readthedocs.io/>`_
    * - Issue Tracker
      - `GitHub Issues <https://github.com/theislab/idtrack/issues>`_
    * - Pull Requests
      - `GitHub Pull Requests <https://github.com/theislab/idtrack/pulls>`_
    * - Discussions
      - `GitHub Discussions <https://github.com/theislab/idtrack/discussions>`_
    * - Code of Conduct
      - :doc:`code_of_conduct`


Reporting Bugs
--------------

Found a bug? Please report it on the `Issue Tracker`_.

A good bug report includes:

1. **Summary** - A clear, concise description of the problem
2. **Environment** - Python version, IDTrack version, operating system
3. **Steps to reproduce** - Minimal code example that triggers the bug
4. **Expected behavior** - What you expected to happen
5. **Actual behavior** - What actually happened (include error messages)

Bug Report Template
~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ## Environment
    - IDTrack version: 0.0.x (run `python -c "import idtrack; print(idtrack.__version__)"`)
    - Python version: 3.12.x (run `python --version`)
    - OS: Ubuntu 22.04 / macOS 14.x / Windows 11

    ## Description
    A clear, concise description of what the bug is.

    ## Steps to Reproduce
    ```python
    import idtrack

    # Minimal code that triggers the bug
    api = idtrack.API(local_repository="./cache")
    # ...
    ```

    ## Expected Behavior
    What you expected to happen.

    ## Actual Behavior
    What actually happened. Include the complete error traceback if applicable.

    ## Additional Context
    Any other context about the problem (screenshots, related issues, etc.).


Requesting Features
-------------------

Have an idea for a new feature? Open an issue on the `Issue Tracker`_ with:

1. **Use case** - Describe the problem you're trying to solve
2. **Proposed solution** - Your suggested approach
3. **Alternatives considered** - Other approaches you've thought about
4. **Impact** - Who would benefit from this feature

.. tip::

    It's a good idea to open an issue before starting work on a feature.
    This allows maintainers to discuss the approach and avoid duplicate effort.


Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

Before setting up your development environment, ensure you have:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Requirement
      - Details
    * - **Python**
      - 3.12 to 3.15
    * - **Git**
      - For version control
    * - **Pandoc**
      - Required for building documentation with Jupyter notebooks

Install Pandoc:

.. code-block:: console

    # macOS
    $ brew install pandoc

    # Ubuntu/Debian
    $ sudo apt-get install -y pandoc

    # Windows (using Chocolatey)
    $ choco install pandoc

    # Or download from https://pandoc.org/installing.html

Development Tools
~~~~~~~~~~~~~~~~~

IDTrack uses the following tools for development:

.. list-table::
    :header-rows: 1
    :widths: 20 40 40

    * - Tool
      - Purpose
      - Installation
    * - Poetry_
      - Dependency management
      - ``pip install poetry``
    * - Nox_
      - Test automation
      - ``pip install nox``
    * - nox-poetry_
      - Nox + Poetry integration
      - ``pip install nox-poetry``

Install all development tools:

.. code-block:: console

    $ pip install poetry poetry-plugin-export nox nox-poetry

    # Alternative: If you installed Poetry via pipx
    $ pipx install poetry
    $ pipx inject poetry poetry-plugin-export

Setting Up the Project
~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Fork and clone**

.. code-block:: console

    # Fork on GitHub, then clone your fork
    $ git clone https://github.com/YOUR_USERNAME/idtrack.git
    $ cd idtrack
    $ git remote add upstream https://github.com/theislab/idtrack.git

**Step 2: Install dependencies with Poetry**

.. code-block:: console

    $ make install

This installs IDTrack in development mode with all dependencies.

**Step 3: Verify your setup**

.. code-block:: console

    # Start an interactive Python session
    $ poetry run python

    # Or run the CLI
    $ poetry run idtrack --help

**Step 4: Install pre-commit hooks (recommended)**

.. code-block:: console

    $ nox --session=pre-commit -- install

This automatically runs code formatting and linting checks before each commit.

.. note::

    See :doc:`installation` for additional requirements like HDF5 on Apple Silicon.


Running Tests
-------------

IDTrack uses pytest_ for testing and Nox_ for test automation.

Run the Full Test Suite
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ nox

This runs all tests, linting, and type checking.

Run Specific Test Sessions
~~~~~~~~~~~~~~~~~~~~~~~~~~

List available sessions:

.. code-block:: console

    $ nox --list-sessions

Run the default test suite (everything except tests marked ``slow``; network-backed integration tests skip
themselves when Ensembl is not reachable):

.. code-block:: console

    $ nox --session=tests

Run tests with a specific Python version:

.. code-block:: console

    $ nox --session=tests --python=3.12

Run Tests Directly with pytest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For faster iteration during development:

.. code-block:: console

    # Run all tests (slow-marked tests are skipped by default)
    $ poetry run pytest

    # Include slow tests
    $ poetry run pytest --runslow

    # Run a specific test file
    $ poetry run pytest tests/unit/test_api.py

    # Run the HTTPS-dump integration tests (they need only HTTPS access to Ensembl)
    $ poetry run pytest --runslow tests/integration/test_ensembl_ftp_fallback.py

    # Run tests matching a pattern
    $ poetry run pytest -k "test_convert"

    # Run with verbose output
    $ poetry run pytest -v

    # Run with coverage report
    $ poetry run pytest --cov=idtrack --cov-report=xml --cov-report=term

    # Run with coverage report (including slow tests)
    $ poetry run pytest --runslow --cov=idtrack --cov-report=xml --cov-report=term

Test Organization
~~~~~~~~~~~~~~~~~

Tests are located in the ``tests/`` directory:

.. code-block:: text

    tests/
    ├── conftest.py               # Shared fixtures, markers and the per-organism real-graph settings
    ├── _real_graph_builder.py    # Builds the tiny real graphs of the end-to-end tests (MySQL or HTTPS)
    ├── test_main.py              # Command-line entry point
    ├── test_external_mappers.py  # External mapping backends (slow classes query live services)
    ├── unit/                     # Offline tests, one file per module (test_api.py, test_db.py, ...)
    └── integration/              # Network-backed tests against Ensembl (MySQL and HTTPS dumps, end to end)

Markers: ``slow`` (skipped unless ``--runslow`` or ``IDTRACK_SKIP_SLOW=0``), ``network``, ``integration`` and
``database``. The end-to-end tests run for every organism that ships a default externals YAML
(``IDTRACK_TEST_ORGANISMS=a,b`` selects others); their small real graphs are cached in
``IDTRACK_TEST_CACHE_DIR``. Where the Ensembl MySQL ports are closed, the graphs are built from the HTTPS dumps,
which happens only with ``--runslow``.


Building Documentation
----------------------

IDTrack uses Sphinx_ with nbsphinx_ to build documentation from reStructuredText
files and Jupyter notebooks.

Install Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ pip install -r docs/requirements.txt

Build the Documentation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ cd docs
    $ make html

The generated HTML files are in ``docs/_build/html/``. Open ``index.html`` in your browser.

Live Preview During Editing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For automatic rebuilds while editing:

.. code-block:: console

    $ pip install sphinx-autobuild
    $ cd docs
    $ sphinx-autobuild . _build/html

Then open http://localhost:8000 in your browser.

.. note::

    IDTrack itself must be installed for the API documentation to build correctly.
    Notebooks are not executed during builds (``nbsphinx_execute = "never"``);
    they are rendered with pre-existing outputs.

Documentation Guidelines
------------------------

When contributing to the documentation website:

- **Keep pages scannable**: start with a short summary, then use ``.. contents::`` and tables for structured material.
- **Prefer cross-links over duplication**: use ``:doc:`...``` for pages and ``:meth:`...``` / ``:class:`...``` for API items.
- **Tutorial notebooks are source of truth for narrative workflows**: notebooks live in ``docs/_notebooks`` and are
  rendered by *nbsphinx* without execution during builds.
- **Avoid huge notebook outputs**: keep outputs concise and deterministic so the rendered docs remain readable.


Submitting Changes
------------------

Workflow Overview
~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 10 30 60

    * - Step
      - Action
      - Details
    * - 1
      - Fork & Clone
      - Create your own copy of the repository
    * - 2
      - Branch
      - Create a feature branch from ``development``
    * - 3
      - Develop
      - Make your changes with tests
    * - 4
      - Test
      - Run tests and linting locally
    * - 5
      - Submit
      - Create a pull request

Step-by-Step Guide
~~~~~~~~~~~~~~~~~~

**Step 1: Fork and clone**

.. code-block:: console

    # Fork on GitHub, then clone your fork
    $ git clone https://github.com/YOUR_USERNAME/idtrack.git
    $ cd idtrack
    $ git remote add upstream https://github.com/theislab/idtrack.git

**Step 2: Create a feature branch**

.. code-block:: console

    $ git checkout development
    $ git pull upstream development
    $ git checkout -b feature/your-feature-name

Use descriptive branch names:

- ``feature/add-zebrafish-support``
- ``fix/hgnc-download-timeout``
- ``docs/improve-quickstart``

**Step 3: Make your changes**

- Write clear, focused commits
- Follow the existing code style
- Add tests for new functionality
- Update documentation if needed

**Step 4: Run tests and linting**

.. code-block:: console

    $ nox

All tests and checks must pass before submitting.

**Step 5: Push and create a pull request**

.. code-block:: console

    $ git push origin feature/your-feature-name

Then open a pull request on GitHub against the ``development`` branch.

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Your pull request should:

- **Pass all tests** - The Nox test suite must pass without errors
- **Include tests** - Add tests for bug fixes and new features
- **Update documentation** - If your changes affect user-facing behavior
- **Have a clear description** - Explain what the PR does and why
- **Reference issues** - Link to related issues (e.g., "Fixes #123")

Pull Request Template
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ## Summary
    Brief description of what this PR does.

    ## Changes
    - Added support for X
    - Fixed bug in Y
    - Updated documentation for Z

    ## Testing
    - [ ] Unit tests added/updated
    - [ ] All tests pass locally (`nox`)
    - [ ] Documentation builds without errors

    ## Related Issues
    Fixes #123


Code Style
----------

IDTrack follows these conventions:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Aspect
      - Convention
    * - **Formatting**
      - Black (automatically applied by pre-commit)
    * - **Linting**
      - Ruff
    * - **Type hints**
      - Use type annotations for public APIs
    * - **Docstrings**
      - NumPy style

Example Function
~~~~~~~~~~~~~~~~

.. code-block:: python

    def convert_identifier(
        self,
        identifier: str,
        to_release: int,
        explain: bool = False,
    ) -> ConversionResult:
        """Convert a biological identifier to a target release.

        Parameters
        ----------
        identifier
            The identifier to convert (e.g., gene symbol, Ensembl ID).
        to_release
            Target Ensembl release number.
        explain
            If True, include detailed mapping information.

        Returns
        -------
        ConversionResult
            The conversion result with mapped identifiers.

        Raises
        ------
        ValueError
            If the identifier format is not recognized.

        Examples
        --------
        >>> api.convert_identifier("TP53", to_release=112)
        ConversionResult(...)
        """
        ...


Adding Organism Support
-----------------------

An organism is declared in one place, the organism table ``DB.assembly_mysqlport_priority`` in
``idtrack/_db.py``; everything else (supported organisms, assembly node types, assembly priorities) is derived
from it. Rhesus and cynomolgus macaque were added this way. The recipe:

1. **Establish the assemblies.** List the organism's core databases in Ensembl's dump listings
   (``https://ftp.ensembl.org/pub/release-<n>/mysql/``): each is named
   ``<organism>_core_<release>_<code>[patch letter]``. Read each assembly's name from the ``meta`` table
   (``assembly.default``) and note the first and last release of each code. Check that every release has one
   assembly (a clean handoff) or note where two overlap (as human GRCh37 and GRCh38 do).

2. **Write the table entry.** One entry per assembly, keyed by an integer that is unique within the organism:

   .. code-block:: python

       "macaca_mulatta": {
           10: {"Name": "Mmul_10", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [98, None]},
           801: {"Name": "Mmul_8.0.1", "Ports": [3306, 5306], "Priority": 2, "ReleaseRange": [86, 97]},
           1: {"Name": "MMUL_1", "DbSuffix": "10", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 85]},
       },

   The key equals Ensembl's code where that is unambiguous. Where Ensembl reuses a code for two assemblies of
   the organism (rhesus ``_10`` is MMUL_1 up to 85 and Mmul_10 from 98), the earlier assembly gets its own key
   and names the shared code in ``DbSuffix``; their ``ReleaseRange`` values must not overlap. ``Ports`` are the
   public MySQL ports in preference order (3306 and 5306 serve release 48 onwards; 3337 is the GRCh37 archive
   and may be listed for human GRCh37 only, see ``DB.mysql_port_assembly``); ``Priority`` 1 is the preferred, newest
   assembly. Add the organism's symbol databases to ``DB.symbol_databases`` (e.g. ``("VGNC Symbol", "HGNC
   Symbol")``). The table is validated at import.

3. **Try it without editing the package.** ``idtrack.DB.register_organism(name, assemblies,
   symbol_databases)`` adds the same entry for the running Python session, validates it, and recomputes
   everything derived from the table.

4. **Make the externals file.** Generate the template over every release and form
   (``DatabaseManager.create_database_content()`` and ``ExternalDatabases.create_template_yaml()``), switch on the
   allowlisted databases with ``idtrack._default_externals.apply_allowlist`` (add the organism's allowlist to
   ``DEFAULT_EXTERNALS`` there and to tutorial 02), and ship ``<organism>_externals_template.yml`` and
   ``<organism>_externals_modified.yml`` in ``idtrack/default_config/``. See
   :doc:`_notebooks/02_prepare_new_external_yaml`. When Ensembl publishes a new release, extend the shipped files
   with ``ExternalDatabases.extend_modified_yaml`` instead of regenerating them.

5. **Teach the tests.** Add a row per assembly to ``EXPECTED_RANGES`` in
   ``tests/integration/test_ensembl_ftp_fallback.py`` and the organism's defaults (assembly, release window, seed
   genes) to ``REAL_GRAPH_DEFAULTS`` in ``tests/conftest.py``. The unit tests check the table, the allowlist and
   the shipped files; the live MySQL tests check ports and releases where MySQL is reachable.

6. **Document it.** Add the organism to the Supported Organisms table in :doc:`usage` and to the tutorials that
   list organisms.

7. **Validate and submit.** Build a small graph across each assembly handoff, convert known identifiers, and
   open a pull request with the table entry, the externals files, the tests and the docs.


Getting Help
------------

If you have questions or need help:

- **General questions**: Open a `GitHub Discussion <https://github.com/theislab/idtrack/discussions>`_
- **Bug reports and features**: Use the `Issue Tracker`_
- **Direct contact**: See :doc:`authors`

We appreciate all contributions, no matter how small. Thank you for helping make IDTrack better!


.. _BSD 3-Clause License: https://opensource.org/licenses/BSD-3-Clause
.. _Issue Tracker: https://github.com/theislab/idtrack/issues
.. _Poetry: https://python-poetry.org/
.. _Nox: https://nox.thea.codes/
.. _nox-poetry: https://nox-poetry.readthedocs.io/
.. _pytest: https://pytest.readthedocs.io/
.. _Sphinx: https://www.sphinx-doc.org/
.. _nbsphinx: https://nbsphinx.readthedocs.io/
