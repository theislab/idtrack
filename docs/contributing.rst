=================
Contributor Guide
=================

Thank you for your interest in contributing to idtrack! This project is open-source
under the `BSD 3-Clause License`_ and welcomes contributions from the community.

.. contents::
    :local:
    :depth: 2
    :backlinks: none


Ways to Contribute
------------------

There are many ways to contribute to idtrack, regardless of your experience level:

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
      - Declare a new species and configure its external databases (:ref:`adding-an-organism`)
    * - **Answer Questions**
      - Help other users in the issues


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
    * - Setting up, testing, releasing
      - `SETUP.md <https://github.com/theislab/idtrack/blob/main/SETUP.md>`_
    * - Code of Conduct
      - :doc:`code_of_conduct`


Reporting Bugs
--------------

Found a bug? Please report it on the `Issue Tracker`_.

A good bug report includes:

1. **Summary** - A clear, concise description of the problem
2. **Environment** - Python version, idtrack version, operating system
3. **Steps to reproduce** - Minimal code example that triggers the bug
4. **Expected behavior** - What you expected to happen
5. **Actual behavior** - What actually happened (include error messages)

Bug Report Template
~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ## Environment
    - idtrack version: 0.1.x (run `idtrack --version`)
    - Python version: 3.12.x (run `python --version`)
    - OS: Ubuntu 22.04 / macOS 14.x / Windows 11

    ## Description
    A clear, concise description of what the bug is.

    ## Steps to Reproduce
    ```python
    from idtrack import API

    # Minimal code that triggers the bug, with the graph you used
    api = API.load(API.graph_path("homo_sapiens", 115, "./idtrack_cache"))
    print(api.describe())  # what the graph is and what built it
    result = api.convert("...", explain=True)
    print(result.to_dict())
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


Development Setup, Tests and Documentation
------------------------------------------

`SETUP.md <https://github.com/theislab/idtrack/blob/main/SETUP.md>`_ at the root of the repository holds every
command, run as written: the development install (Poetry and nox, then ``make install``), the test suite
(``make test``; tests marked ``slow`` need ``--runslow`` and the network), lint (``make lint``), the documentation
(``make docs``, every warning an error; it needs `pandoc <https://pandoc.org/installing.html>`_), the checks of CI as
nox sessions (``nox --list-sessions``), and the release. ``make`` lists the shortcuts.

Test Organization
~~~~~~~~~~~~~~~~~

Tests are located in the ``tests/`` directory:

.. code-block:: text

    tests/
    ├── conftest.py               # Shared fixtures (among them the graphs of real organisms) and markers
    ├── _real_tables.py           # Serves real Ensembl tables, cut down to a few hundred genes, to DatabaseManager
    ├── _tools.py                 # Imports a tool of tools/ for its test (test_tools.py)
    ├── data/real/                # Per organism: the cut tables, the gene list, the expected answers
    ├── unit/                     # Offline tests, one file per module (test_api.py, test_main.py: the command line, ...)
    │                             # and test_real_*.py
    └── integration/              # Network-backed tests against Ensembl (MySQL and HTTPS dumps)

Markers: ``slow`` (skipped unless ``--runslow`` or ``IDTRACK_SKIP_SLOW=0``), ``network``, ``integration``,
``database`` and ``wrong_argument_type`` (a test that passes an argument of a wrong type on purpose; skipped in the
typeguard session, where typeguard refuses the argument first).

The tests that can catch a wrong mapping are ``tests/unit/test_real_*.py``. They need no network:
``tests/_real_tables.py`` serves real Ensembl tables of pig and human, cut down to a few hundred genes
(``tests/data/real/<organism>.tables.json.xz``, each under 600 kB: the repository's size check for added files; the
tables are not part of the package's distributions), to :py:class:`idtrack.build.DatabaseManager`, so the
graphs of both organisms are built by the production code (tables, history, graph file, engine, API) in the test
session. Conversions are compared with expected answers that were derived from Ensembl's own files
(``tests/data/real/<organism>.truth.jsonl``) and with the tables themselves, read by independent code; the package's
self-tests (:py:class:`idtrack.checks.GraphChecks`) run on the same graphs.

To write a table file again from a gene list (``tests/data/real/<organism>.genes.txt``), for example after adding
genes that a new test needs:

.. code-block:: console

    $ poetry run python tools/cut_tables.py --organism sus_scrofa --repository <local repository>

It reads the tables from the local repository and downloads what the repository lacks over HTTPS.
``tests/integration/test_real_tables_source.py`` (``--runslow``, network) fetches one release of each organism again
and compares it with the stored file. This tool and the others that write files of the repository (the shipped
externals YAML files, the consumer-contract expectations) are in ``tools/``; ``tools/README.md`` says what each
writes, from what, and how to check the result.


Documentation Guidelines
------------------------

When contributing to the documentation website:

- **Keep pages scannable**: start with a short summary, then use ``.. contents::`` and tables for structured material.
- **Prefer cross-links over duplication**: use ``:doc:`...``` for pages and ``:meth:`...``` / ``:class:`...``` for API items.
- **Tutorial notebooks are source of truth for narrative workflows**: notebooks live in ``docs/_notebooks`` and are
  rendered by *nbsphinx* without execution during builds.
- **Avoid huge notebook outputs**: keep outputs concise and deterministic so the rendered docs remain readable.
- **Outputs come from running the code**: after changing a notebook, run it in full (with the graphs of the pinned
  snapshot release the tutorials use, 115) so that every shown output is what the code prints; never edit an output
  by hand. The examples of the pages and the README show outputs of the same snapshot.
- **One place per concept**: a rule is explained once, on the page where a reader first needs it (:doc:`concepts`,
  :doc:`reading`, :doc:`limitations`), and linked from elsewhere; the table of time and memory is on
  :doc:`performance`, and other pages give a range and link to it.


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
      - Create a feature branch from ``main``
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

    $ git checkout main
    $ git pull upstream main
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

    $ make test
    $ make lint

The workflows of the pull request run every check of CI (the nox sessions on Python 3.12 to 3.15, the tests of the
built wheel and the documentation; SETUP.md says how to run each locally); all must pass.

**Step 5: Push and create a pull request**

.. code-block:: console

    $ git push origin feature/your-feature-name

Then open a pull request on GitHub against the ``main`` branch.

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Your pull request should:

- **Pass all checks** - The workflows of the pull request must pass
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
    - [ ] Tests and lint pass locally (`make test`, `make lint`)
    - [ ] Documentation builds without errors

    ## Related Issues
    Fixes #123


Code Style
----------

idtrack follows these conventions:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Aspect
      - Convention
    * - **Formatting**
      - Black, line length 120, and isort (applied by the pre-commit hooks)
    * - **Linting**
      - flake8 with flake8-docstrings, flake8-bugbear, flake8-bandit, flake8-builtins, flake8-rst-docstrings and
        pep8-naming (``.flake8``), pyupgrade ``--py312-plus``
    * - **Type hints**
      - Type annotations for public APIs; mypy (``nox --session=mypy``)
    * - **Docstrings**
      - Google style (``docstring-convention = google``): the arguments, the return value and the errors raised
        documented; cross-references with the ``:py:meth:``, ``:py:class:`` ... roles, which the documentation build
        checks (an ``idtrack`` name that resolves to nothing fails it)

The pre-commit session (``nox --session=pre-commit``) runs all of these.

Example Function
~~~~~~~~~~~~~~~~

A docstring in the style of the package (modelled on :py:meth:`idtrack.API.final_databases`):

.. code-block:: python

    def final_databases(self, to_release: int | None = None) -> list[str]:
        """Return the values accepted as ``final_database`` for a target release, sorted.

        These are the external databases of the genome assembly that serves *to_release* and the two Ensembl gene
        namespaces (versioned ``ensembl_gene`` and ``base_ensembl_gene``).

        Args:
            to_release: The target release; ``None`` means the graph's snapshot release.

        Returns:
            list[str]: The database names.
        """


Adding Organism Support
-----------------------

An organism is declared in one table (:py:data:`idtrack.organisms.ORGANISM_TABLE`) and ships its externals
configuration; the recipe, from Ensembl's dump listings to the tests, is :ref:`adding-an-organism` on
:doc:`organisms`.


Getting Help
------------

If you have questions or need help:

- **Questions, bug reports and features**: Use the `Issue Tracker`_
- **Direct contact**: See :doc:`authors`

We appreciate all contributions, no matter how small. Thank you for helping make idtrack better!


.. _BSD 3-Clause License: https://opensource.org/licenses/BSD-3-Clause
.. _Issue Tracker: https://github.com/theislab/idtrack/issues
