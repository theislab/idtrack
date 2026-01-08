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
      - 3.9 or later (3.11 recommended)
    * - **Operating Systems**
      - Linux, macOS (Intel and Apple Silicon), Windows
    * - **Memory**
      - Minimum 4 GB RAM (8 GB recommended for large graphs)
    * - **Disk Space**
      - 2-5 GB for cached databases (varies by organism)
    * - **Network**
      - Internet connection required for initial database downloads

.. note::

    **Apple Silicon users (M1/M2/M3/M4):** If you install via pip/venv, install HDF5 first (or use conda-forge).
    See the :ref:`apple-silicon` section below.


Quick Install with Mamba (Recommended)
--------------------------------------

We recommend `Mamba <https://mamba.readthedocs.io/>`_ for the fastest and most reliable
installation experience. Mamba is a drop-in replacement for conda with significantly
faster dependency resolution.

.. note::

    Environment activation is handled by conda’s shell integration, even if you use
    Mamba for installs. Use ``conda activate <env>`` after creating an environment.

**Install Mamba** (if you don't have it):

.. code-block:: console

    # Option 1: Install Miniforge (includes Mamba) - recommended
    # Download from https://github.com/conda-forge/miniforge

    # Option 2: Install Mamba into existing conda
    $ conda install -n base -c conda-forge mamba

**Install IDTrack with Mamba**:

.. code-block:: console

    # Option 1 (recommended): create an isolated environment in one command
    $ mamba create -n idtrack-env -c conda-forge python=3.11 idtrack
    $ conda activate idtrack-env

    # Option 2: install into your currently active conda environment
    $ mamba install -c conda-forge idtrack

.. tip::

    **Why Mamba?** Mamba resolves dependencies 10-100x faster than conda and handles
    complex scientific packages (HDF5, NumPy, SciPy) more reliably. If you encounter
    issues with other installation methods, try Mamba first.


Alternative Installation Methods
--------------------------------

.. tab-set::

    .. tab-item:: Using conda

        .. code-block:: console

            $ conda install -c conda-forge idtrack

    .. tab-item:: Using pip

        If you prefer pip or are working in a virtual environment:

        .. code-block:: console

            $ pip install idtrack

        .. warning::

            Pip installation may fail on some systems due to HDF5 compilation requirements.
            If you encounter errors, use Mamba/conda instead or install HDF5 first.


.. _apple-silicon:

Apple Silicon (M1/M2/M3/M4) Setup
---------------------------------

Apple Silicon Macs require HDF5 to be installed before IDTrack. This is because the
``h5py`` and ``pytables`` packages (used for efficient data caching) need pre-compiled
HDF5 binaries that match the ARM64 architecture. When installing via pip, these binaries
may not be available or may fail to compile. Installing HDF5 through Mamba/conda ensures
compatible ARM64 binaries are used. This is a one-time setup.

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

    $ mamba install -c conda-forge idtrack

**Verification:**

.. code-block:: console

    $ python -c "import h5py; print('HDF5 version:', h5py.version.hdf5_version)"

.. tip::

    If you still encounter HDF5 errors after installation, ensure your Python environment
    was created *after* installing HDF5. Creating a fresh environment often resolves issues.


Creating a Dedicated Environment (Recommended)
----------------------------------------------

Using a dedicated environment is a good practice for any Python project and makes it
easier to manage your installation.

Using Mamba (fastest)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    # Create a new environment with Python 3.11
    $ mamba create -n idtrack-env python=3.11

    # Activate the environment
    $ conda activate idtrack-env

    # Install IDTrack
    $ mamba install -c conda-forge idtrack

Using conda
~~~~~~~~~~~

.. code-block:: console

    # Create a new environment with Python 3.11
    $ conda create -n idtrack-env python=3.11

    # Activate the environment
    $ conda activate idtrack-env

    # Install IDTrack
    $ conda install -c conda-forge idtrack

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

For users who want a specific release without cloning the repository:

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
    Latest Ensembl release: 114

    Installation verified successfully!

If you see output similar to this, your installation is complete and working.


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

This usually indicates an HDF5 binary compatibility issue. The most reliable fix is to
use conda-forge (Mamba or conda):

.. code-block:: console

    # Create a fresh environment with Mamba
    $ mamba create -n idtrack-env python=3.11 idtrack -c conda-forge
    $ conda activate idtrack-env

If you must use pip, install HDF5 dependencies first:

.. code-block:: console

    $ mamba install -c conda-forge hdf5
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

**Slow dependency resolution with conda**

Switch to Mamba for significantly faster installation:

.. code-block:: console

    $ conda install -n base -c conda-forge mamba
    $ mamba install -c conda-forge idtrack


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
