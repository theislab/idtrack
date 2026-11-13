.. highlight:: shell

============
Installation
============

This guide covers all installation methods for IDTrack. Choose the approach that best fits
your workflow.

.. contents::
    :local:
    :depth: 2
    :backlinks: none


System Requirements
-------------------

Before installing IDTrack, ensure your system meets these requirements:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Requirement
      - Details
    * - **Python**
      - 3.12 to 3.15
    * - **Operating Systems**
      - Linux, macOS (Intel and Apple Silicon), Windows
    * - **Memory**
      - Using a graph: 0.26 GB (cynomolgus) to 1.14 GB (human) after loading, plus your data. Building a graph:
        0.6 GB (cynomolgus) to 4.5 GB (human) at the peak. See :ref:`time-and-memory`.
    * - **Disk Space**
      - The tables a graph is built from stay in the local repository: 0.27 GB (cynomolgus), 0.65 GB (pig and
        rhesus), 2.1 GB (mouse), 5.3 GB (human) for a graph of snapshot release 115, plus the graph file itself
        (47 MB to 345 MB).
    * - **Network**
      - HTTPS access to ``ftp.ensembl.org`` and ``rest.ensembl.org`` to build a graph (and for
        :py:meth:`idtrack.API.resolve_organism` to learn the latest release); the Ensembl MySQL ports are optional
        (see :ref:`ensembl-access`). Loading a graph that is on disk needs no network.

.. note::

    **Apple Silicon users (M1/M2/M3/M4):** If ``h5py`` fails to install or import, install HDF5 first.
    See the :ref:`apple-silicon` section below.


Install with pip
----------------

IDTrack is published on PyPI (it is not on conda-forge or bioconda). Install it into a fresh environment:

.. code-block:: console

    $ python -m venv idtrack-env
    $ source idtrack-env/bin/activate      # Windows: idtrack-env\Scripts\activate
    $ pip install idtrack

or into a conda/mamba environment, with pip:

.. code-block:: console

    $ mamba create -n idtrack-env -c conda-forge python=3.13
    $ conda activate idtrack-env
    $ pip install idtrack

.. important::

    The release on PyPI, 0.0.5, is older than the version these pages describe: the changes listed under
    "Unreleased" in :doc:`changes` are not in it, and it reads neither the graphs nor the table cache of this version
    (nor this version its). Until the next release, install this version from GitHub:

    .. code-block:: console

        $ pip install "git+https://github.com/theislab/idtrack.git@main"

    ``idtrack.__version__`` does not tell the two apart before then (both say 0.0.5).


.. _apple-silicon:

Apple Silicon (M1/M2/M3/M4) Setup
---------------------------------

IDTrack stores its tables with ``h5py``, which needs HDF5. When pip finds no pre-built ``h5py`` wheel for your
Python and machine, it compiles ``h5py`` against an installed HDF5; installing HDF5 through conda-forge or Homebrew
first gives it one. This is a one-time setup.

**Step 1: Install HDF5**

Choose one of these methods:

.. code-block:: console

    # Method 1: Using Mamba (recommended)
    $ mamba install -c conda-forge hdf5

    # Method 2: Using conda
    $ conda install -c conda-forge hdf5

    # Method 3: Using Homebrew
    $ brew install hdf5

**Step 2: Install IDTrack**

.. code-block:: console

    $ pip install idtrack

**Verification:**

.. code-block:: console

    $ python -c "import h5py; print('HDF5 version:', h5py.version.hdf5_version)"

.. tip::

    If you still encounter HDF5 errors after installation, ensure your Python environment
    was created *after* installing HDF5. Creating a fresh environment often resolves issues.


Creating a Dedicated Environment (Recommended)
----------------------------------------------

Using a dedicated environment is a good practice for any Python project and makes it
easier to manage your installation. IDTrack runs on Python 3.12 to 3.15.

Using Mamba or conda
~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    # Create a new environment with Python 3.13
    $ mamba create -n idtrack-env -c conda-forge python=3.13    # or: conda create ...

    # Activate the environment
    $ conda activate idtrack-env

    # Install IDTrack (from PyPI, with pip)
    $ pip install idtrack

Using venv + pip
~~~~~~~~~~~~~~~~

.. code-block:: console

    # Create a new virtual environment
    $ python -m venv idtrack-env

    # Activate the environment
    # On Linux/macOS:
    $ source idtrack-env/bin/activate
    # On Windows:
    $ idtrack-env\Scripts\activate

    # Install IDTrack
    $ pip install idtrack


Development Installation
------------------------

For contributors or users who want the latest development features.

Prerequisites
~~~~~~~~~~~~~

Ensure you have these tools installed:

- `Git <https://git-scm.com/>`_ for version control
- `Poetry <https://python-poetry.org/>`_ for dependency management

.. code-block:: console

    $ pip install poetry poetry-plugin-export

Clone and Install
~~~~~~~~~~~~~~~~~

**Step 1: Clone the repository**

.. code-block:: console

    $ git clone https://github.com/theislab/idtrack.git
    $ cd idtrack

**Step 2: Install with Poetry**

.. code-block:: console

    $ make install

This installs IDTrack in development mode with all dependencies, including development
and testing tools.

**Step 3: Verify installation**

.. code-block:: console

    $ poetry run python -c "import idtrack; print(f'IDTrack {idtrack.__version__}')"

Alternative: Install from Tarball
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users who want the code of the ``main`` branch without cloning the repository:

.. code-block:: console

    $ curl -OJL https://github.com/theislab/idtrack/tarball/main
    $ tar -xzf theislab-idtrack-*.tar.gz
    $ cd theislab-idtrack-*
    $ pip install .


Verifying Your Installation
---------------------------

After installation, verify that IDTrack is working correctly by running this verification
script:

.. code-block:: python

    from pathlib import Path
    import idtrack

    # Pick a stable, writable cache directory (reused across runs)
    local_repo = Path("./idtrack_cache").resolve()
    local_repo.mkdir(parents=True, exist_ok=True)

    # Print version information
    print(f"IDTrack version: {idtrack.__version__}")

    # Test API initialization
    api = idtrack.API(local_repository=str(local_repo))
    print("API initialized successfully")

    # Test organism resolution
    organism, release = api.resolve_organism("human")
    print(f"Resolved organism: {organism}")
    print(f"Latest Ensembl release: {release}")

    print("\nInstallation verified successfully!")

**Expected output:**

.. code-block:: text

    IDTrack version: 0.0.x
    API initialized successfully
    Resolved organism: homo_sapiens
    Latest Ensembl release: 116

    Installation verified successfully!

If you see output similar to this, your installation is complete and working (the latest release is the one
Ensembl serves on the day you run it).


.. _ensembl-access:

Network Access: MySQL or HTTPS
------------------------------

IDTrack reads Ensembl's core databases in one of two ways: from the public MySQL service
(``ensembldb.ensembl.org``, ports 3306, 5306 and 3337) or from the MySQL dumps on
``https://ftp.ensembl.org/pub/``. Many institutional networks and HPC clusters block the MySQL ports;
HTTPS is enough. The access mode is chosen with the environment variable ``IDTRACK_ENSEMBL_ACCESS`` or
the attribute ``idtrack.DB.ensembl_access`` (which wins when set):

.. list-table::
    :header-rows: 1
    :widths: 15 85

    * - Mode
      - Behaviour
    * - ``auto`` (default)
      - Each MySQL port is probed once per Python process, with a short timeout. Releases on a port that
        answers are read over MySQL; everything else, and everything after a MySQL connection fails, is read
        from the HTTPS dumps.
    * - ``https``
      - MySQL is never contacted; every table is read from the HTTPS dumps.

.. code-block:: console

    $ export IDTRACK_ENSEMBL_ACCESS=https

.. code-block:: python

    import idtrack

    idtrack.DB.ensembl_access = "https"  # or "auto"

**What differs between the modes:** nothing in the tables: both read the same Ensembl tables, with the same rows,
values and column types. The dump reader decodes MySQL's escaping (tabs, newlines and backslashes inside fields,
``NULL``) so that each row holds what the MySQL server returns, and takes the column types from the table
definitions of the dump, so a text column of digit strings stays text. The HTTPS path downloads whole compressed
tables, so it is slower per table than a MySQL query (seconds to a minute per large table), but it needs no open port.
In ``auto`` mode a blocked port costs one short probe per process, not one per table, and a port whose connection
fails is not tried again for ten minutes (then after twice as long, up to a day; the tables come over HTTPS meanwhile,
with a warning; this costs only speed). A query or data error of the MySQL server is raised, not answered over
HTTPS.

If you route IDTrack through an SSH tunnel with :py:class:`idtrack.ConnectionBridge`, starting and stopping
the bridge forgets the probe results, so the ports are probed again through the new route.

**Slow networks, timeouts and retries.** Every request to Ensembl REST (organism names and the latest release)
waits ``idtrack.DB.connection_timeout`` seconds to connect (default 15) and ``idtrack.DB.reading_timeout``
seconds for the answer (default 60; it also applies to the HTTPS dumps), and is tried
``idtrack.DB.rest_attempts`` times (default 3) when it times out, the connection fails, or Ensembl answers 429
or 5xx. When every attempt fails, :py:class:`idtrack.EnsemblRestError` (a :py:class:`ConnectionError`) names the
URL, the timeouts and what to change; a download of the dumps that keeps failing raises
:py:class:`idtrack.EnsemblDownloadError` (also a :py:class:`ConnectionError`), and a release that cannot be listed
is an error, never read as "Ensembl has no such release". On a network where Ensembl answers slowly, raise the timeouts before the first call:

.. code-block:: python

    import idtrack

    idtrack.DB.reading_timeout = 120
    idtrack.DB.rest_attempts = 5

The network is needed only to download, and to learn which release is the latest: the formal name, the short name
and the assembly names of a supported organism (``"homo_sapiens"``, ``"Homo sapiens"``, ``"human"``, ``"GRCh38"``)
resolve without it, and a graph file on disk loads and converts without it (see :ref:`reusing-built-graphs`), as does
a harmonization whose graph is on disk. The local repository may be read-only: it then serves the tables and graphs
it holds, and a table that would have to be computed raises :py:class:`PermissionError` naming the folder. An HTTPS
proxy is honoured through ``HTTPS_PROXY``.


Optional Dependencies
---------------------

IDTrack has optional dependencies for advanced features:

External Mapping Backends
~~~~~~~~~~~~~~~~~~~~~~~~~

For cross-database identifier mapping using external services:

.. code-block:: console

    # Install the optional backends via Python extras (recommended)
    $ pip install "idtrack[external-mappers]"

    # Or install only the libraries you need
    $ pip install gget mygene pybiomart gprofiler-official

Ortholog Analysis
~~~~~~~~~~~~~~~~~

For cross-species ortholog analysis with sequence alignment:

.. code-block:: console

    $ pip install "idtrack[ortholog]"

Protein Embeddings
~~~~~~~~~~~~~~~~~~

:py:func:`idtrack._external_mappers._ortholog.align_ortholog_pair_with_features` can add protein-language-model embedding
features to an ortholog alignment. They need ``torch`` and ``transformers``; with them installed, the model named by
``embedding_model_name`` (``facebook/esm2_t12_35M_UR50D`` by default) is downloaded from Hugging Face on first use.
Without them, or with ``embedding_model_name=None``, those features are left empty (with ``verbose=True`` a warning says why) and the
other alignment features are computed.

.. code-block:: console

    $ pip install "idtrack[embeddings]"

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

Install all optional dependencies at once:

.. code-block:: console

    $ pip install "idtrack[all-external]"


Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**HDF5 / h5py import errors**

This usually indicates an HDF5 binary compatibility issue. Install ``h5py`` from conda-forge in a fresh environment,
then IDTrack with pip:

.. code-block:: console

    $ mamba create -n idtrack-env -c conda-forge python=3.13 h5py
    $ conda activate idtrack-env
    $ pip install idtrack

**SSL Certificate errors during download**

If IDTrack encounters SSL errors when downloading external databases:

.. code-block:: console

    # Update certificates
    $ pip install --upgrade certifi

    # On macOS, you may also need:
    $ /Applications/Python\ 3.x/Install\ Certificates.command

**Permission errors during installation**

.. code-block:: console

    # Option 1: Use --user flag with pip
    $ pip install --user idtrack

    # Option 2: Use a virtual environment (recommended)
    $ python -m venv idtrack-env
    $ source idtrack-env/bin/activate
    $ pip install idtrack


Getting Help
~~~~~~~~~~~~

If you encounter issues not covered here:

1. Check the `GitHub Issues <https://github.com/theislab/idtrack/issues>`_ for known problems and solutions
2. Search `existing discussions <https://github.com/theislab/idtrack/discussions>`_ for similar questions
3. Open a new issue with:

   - Your operating system and Python version
   - The complete error message
   - Steps to reproduce the issue

4. See the :doc:`_notebooks/01_installation_guide` tutorial for detailed environment setup


What's Next?
------------

After successful installation, continue with these resources:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Resource
      - Description
    * - :doc:`usage`
      - Minimal quickstart code (5-minute introduction)
    * - :doc:`_notebooks/01_installation_guide`
      - Detailed environment configuration tutorial
    * - :doc:`_notebooks/00_idtrack_overview`
      - Understand the mental model behind IDTrack
    * - :doc:`tutorials`
      - Complete learning path (Parts 0-7)


.. _pip: https://pip.pypa.io
.. _Poetry: https://python-poetry.org/
