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
      - 3.12 to 3.15 (3.15 needs HDF5 and a C compiler, see :ref:`python-315`)
    * - **Operating Systems**
      - Linux, macOS (Intel and Apple Silicon), Windows
    * - **Memory**
      - Using a graph: 0.3 GB (cynomolgus) to 1.5 GB (human) after loading, plus your data. Building a graph: 1.1 GB (cynomolgus) to 6.8 GB
        (human) at the peak. See :ref:`time-and-memory`.
    * - **Disk Space**
      - The tables a graph is built from stay in the local repository: 0.24 GB (cynomolgus) and at most 1.1 GB (pig),
        1.2 GB (rhesus), 3.3 GB (mouse) and 9.3 GB (human) for a graph of snapshot release 115, plus the graph file
        itself (about 23 MB to 156 MB; sizes vary a little from build to build).
    * - **Network**
      - HTTPS access to ``ftp.ensembl.org`` and ``rest.ensembl.org`` to build a graph (and for
        :py:meth:`idtrack.API.ensembl_organism` to learn the latest release); the Ensembl MySQL ports are optional
        (see :ref:`ensembl-access`). Loading a graph that is on disk needs no network.

.. note::

    **Python 3.15**, and **Apple Silicon users (M1/M2/M3/M4)** whose ``h5py`` fails to install or import: install
    HDF5 first, see :ref:`python-315` and :ref:`apple-silicon` below.


Install with pip
----------------

Install IDTrack from GitHub into a fresh environment (it is not on conda-forge or bioconda):

.. code-block:: console

    $ python3.12 -m venv idtrack-env    # any Python 3.12 to 3.15: python3 --version tells which python3 is
    $ source idtrack-env/bin/activate      # Windows: idtrack-env\Scripts\activate
    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.1"

``@v0.1.1`` installs this release; without it, pip installs the current ``main`` of the repository, which may be ahead
of these pages. Or into a conda/mamba environment, with pip:

.. code-block:: console

    $ mamba create -n idtrack-env -c conda-forge python=3.13
    $ conda activate idtrack-env
    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.1"

To harmonize ``.h5ad`` datasets (:py:class:`idtrack.harmonize.HarmonizeFeatures`), install the ``harmonize`` extra,
which brings anndata and scipy:

.. code-block:: console

    $ pip install "idtrack[harmonize] @ git+https://github.com/theislab/idtrack.git@v0.1.1"

.. important::

    The release on PyPI (``pip install idtrack``), 0.0.5, is an older, different version with another API: the
    changes listed in :doc:`changes` are not in it, and it reads neither the graphs nor the table cache of this version
    (nor this version its). Install from GitHub until this version is published on PyPI.
    ``idtrack.__version__`` tells them apart: this version reports 0.1.1 (``unknown`` when the code runs from a source
    folder that is not installed).


.. _python-315:

Python 3.15
-----------

h5py publishes no wheels for Python 3.15 yet (h5py 3.16.0, November 2026), so on Python 3.15 pip compiles h5py from
source. That needs a C compiler and the HDF5 library with its headers, installed before idtrack:

.. code-block:: console

    # Debian or Ubuntu
    $ sudo apt-get install build-essential libhdf5-dev

    # Fedora, RHEL
    $ sudo dnf install gcc hdf5-devel

    # macOS
    $ brew install hdf5

    # or, in a conda environment
    $ conda install -c conda-forge hdf5

If pip does not find HDF5, set ``HDF5_DIR`` to the folder that holds its ``include/`` and ``lib/``. The install then
takes about 6 minutes (compiling h5py) instead of about 2.5 (measured for the plain install on a Linux server).
Python 3.12 to 3.14 need none of this: their h5py wheels bundle HDF5.


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

    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.1"

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

    # Install IDTrack (from GitHub, with pip)
    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.1"

Using venv + pip
~~~~~~~~~~~~~~~~

.. code-block:: console

    # Create a new virtual environment (with a Python 3.12 to 3.15)
    $ python3.12 -m venv idtrack-env

    # Activate the environment
    # On Linux/macOS:
    $ source idtrack-env/bin/activate
    # On Windows:
    $ idtrack-env\Scripts\activate

    # Install IDTrack
    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.1"


Development Installation
------------------------

For contributors: clone the repository and install it with Poetry and nox as
`SETUP.md <https://github.com/theislab/idtrack/blob/main/SETUP.md>`_ describes (``make install``; the tests, the checks
and the release are there too). Without cloning, the code of the ``main`` branch installs with pip as above.


Verifying Your Installation
---------------------------

After installation, verify that IDTrack is working correctly by running this verification
script:

.. code-block:: python

    import idtrack
    import idtrack.organisms

    # Print version information
    print(f"IDTrack version: {idtrack.__version__}")

    # Offline: the organisms of the table
    print("Organisms:", ", ".join(idtrack.organisms.supported_organisms()))

    # Online (Ensembl REST): organism resolution and Ensembl's latest release
    organism, release = idtrack.API.ensembl_organism("human")
    print(f"Resolved organism: {organism}")
    print(f"Latest Ensembl release: {release}")

    print("\nInstallation verified successfully!")

**Expected output:**

.. code-block:: text

    IDTrack version: 0.1.1
    Organisms: homo_sapiens, macaca_fascicularis, macaca_mulatta, mus_musculus, sus_scrofa
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
with :py:func:`idtrack.settings.configure_ensembl` (``access=``, which wins when set):

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

    import idtrack.settings

    idtrack.settings.configure_ensembl(access="https")  # or "auto"

**What differs between the modes:** nothing in the tables: both read the same Ensembl tables, with the same rows,
values and column types. The dump reader decodes MySQL's escaping (tabs, newlines and backslashes inside fields,
``NULL``) so that each row holds what the MySQL server returns, and takes the column types from the table
definitions of the dump, so a text column of digit strings stays text. The HTTPS path downloads whole compressed
tables, so it is slower per table than a MySQL query (seconds to a minute per large table), but it needs no open port.
In ``auto`` mode a blocked port costs one short probe per process, not one per table, and a port whose connection
fails is not tried again for ten minutes (then after twice as long, up to a day; the tables come over HTTPS meanwhile,
with a warning; this costs only speed). A query or data error of the MySQL server is raised, not answered over
HTTPS.

**Slow networks, timeouts and retries.** Every request to Ensembl REST (organism names and the latest release)
waits ``connection_timeout`` seconds to connect (default 15) and ``reading_timeout`` seconds for the answer
(default 60; it also applies to the HTTPS dumps), and is tried ``rest_attempts`` times (default 3) when it times out,
the connection fails, or Ensembl answers 429 or 5xx (the fields of :py:class:`idtrack.settings.EnsemblSettings`, set
for the whole process with :py:func:`idtrack.settings.configure_ensembl`). When every attempt fails,
:py:class:`idtrack.errors.EnsemblRestError` (an :py:class:`idtrack.errors.EnsemblAccessError`, also a
:py:class:`ConnectionError`) names the URL, the timeouts and what to change; a download of the dumps that keeps
failing raises :py:class:`idtrack.errors.EnsemblDownloadError` (likewise), and a release that cannot be listed
is an error, never read as "Ensembl has no such release". On a network where Ensembl answers slowly, raise the timeouts before the first call:

.. code-block:: python

    import idtrack.settings

    idtrack.settings.configure_ensembl(reading_timeout=120, rest_attempts=5)

The network is needed only to download, and to learn which release is the latest: the formal name, the short name
and the assembly names of a supported organism (``"homo_sapiens"``, ``"Homo sapiens"``, ``"human"``, ``"GRCh38"``)
resolve without it, and a graph file on disk loads and converts without it (see :ref:`reusing-built-graphs`), as does
a harmonization whose graph is on disk. The local repository may be read-only: it then serves the tables and graphs
it holds, and a table that would have to be computed raises :py:class:`PermissionError` naming the folder. An HTTPS
proxy is honoured through ``HTTPS_PROXY``.


Optional Dependencies
---------------------

The ``harmonize`` extra brings anndata and scipy for :py:class:`idtrack.harmonize.HarmonizeFeatures` (harmonization
of ``.h5ad`` datasets); conversions, graph building and the pure core of harmonization
(:py:mod:`idtrack.harmonize`) do not need it. Without it, using ``HarmonizeFeatures`` raises an ImportError that names
the extra.

.. code-block:: console

    $ pip install "idtrack[harmonize] @ git+https://github.com/theislab/idtrack.git@v0.1.1"


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
    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.1"

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
    $ pip install --user "git+https://github.com/theislab/idtrack.git@v0.1.1"

    # Option 2: Use a virtual environment (recommended)
    $ python3.12 -m venv idtrack-env
    $ source idtrack-env/bin/activate
    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.1"


Getting Help
~~~~~~~~~~~~

If you encounter issues not covered here:

1. Check the `GitHub Issues <https://github.com/theislab/idtrack/issues>`_ for known problems and solutions
2. Open a new issue with:

   - Your operating system and Python version
   - The complete error message
   - Steps to reproduce the issue

3. See the :doc:`_notebooks/01_installation_guide` tutorial for detailed environment setup


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
