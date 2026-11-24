.. highlight:: shell

============
Installation
============

idtrack runs on Python 3.12 to 3.15, on Linux, macOS and Windows, and installs with pip.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


Install with pip
----------------

In a fresh environment (a venv, or a conda environment):

.. code-block:: console

    $ python3.12 -m venv idtrack-env          # any Python 3.12 to 3.15
    $ source idtrack-env/bin/activate         # Windows: idtrack-env\Scripts\activate
    $ pip install idtrack

To harmonize ``.h5ad`` datasets (:doc:`harmonization`), add the ``harmonize`` extra, which brings anndata and scipy
(about 150 MB):

.. code-block:: console

    $ pip install "idtrack[harmonize]"

Conversions, graph building and the pure core of harmonization (:py:mod:`idtrack.harmonize`) do not need the extra;
without it, using :py:class:`idtrack.harmonize.HarmonizeFeatures` raises an ImportError that names it.

The same release from GitHub, by its tag (without ``@v0.1.2`` pip installs the current ``main``, which may be ahead
of these pages):

.. code-block:: console

    $ pip install "git+https://github.com/theislab/idtrack.git@v0.1.2"
    $ pip install "idtrack[harmonize] @ git+https://github.com/theislab/idtrack.git@v0.1.2"

In a conda or mamba environment, install Python from conda-forge and idtrack with pip (idtrack is not on
conda-forge or bioconda):

.. code-block:: console

    $ mamba create -n idtrack-env -c conda-forge python=3.13
    $ conda activate idtrack-env
    $ pip install idtrack

For contributors, the development install (a clone, Poetry and nox, ``make install``), the tests and the release are
in `SETUP.md <https://github.com/theislab/idtrack/blob/main/SETUP.md>`_.

.. note::

    Versions before 0.1.0 (0.0.5 and earlier) have another API, and neither they nor this version read each other's
    graphs and table caches. :doc:`changes` says what a user of 0.0.5 must do.


.. _python-315:

Python 3.15 and HDF5
--------------------

idtrack stores Ensembl's tables with h5py. Its wheels bundle HDF5 for Python 3.12 to 3.14 on Linux, macOS (Intel and
Apple Silicon) and Windows, so nothing else is needed there. h5py publishes no wheels for Python 3.15 yet (h5py
3.16.0, November 2026), so on 3.15 pip compiles h5py from source, which needs a C compiler and the HDF5 library with
its headers, installed before idtrack:

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
takes about 6 minutes (compiling h5py) instead of about 2.5 (measured for the plain install on a Linux server). The
same applies wherever pip finds no h5py wheel for a Python and machine.


Checking the install
--------------------

.. code-block:: python

    import idtrack
    import idtrack.organisms

    print(f"IDTrack version: {idtrack.__version__}")
    print("Organisms:", ", ".join(idtrack.organisms.supported_organisms()))       # offline

    organism, release = idtrack.API.ensembl_organism("human")                     # asks Ensembl REST
    print(f"Resolved organism: {organism}")
    print(f"Latest Ensembl release: {release}")

.. code-block:: text

    IDTrack version: 0.1.2
    Organisms: homo_sapiens, macaca_fascicularis, macaca_mulatta, mus_musculus, sus_scrofa
    Resolved organism: homo_sapiens
    Latest Ensembl release: 116

The latest release is the one Ensembl serves on the day you run it. ``idtrack.__version__`` reads the installed
package's metadata (``unknown`` when the code runs from a source folder that is not installed); ``idtrack --version``
prints the same.


.. _ensembl-access:

Network access: MySQL or HTTPS
------------------------------

Building a graph reads Ensembl's core databases in one of two ways: from the public MySQL service
(``ensembldb.ensembl.org``, ports 3306, 5306 and 3337) or from the MySQL dumps on ``https://ftp.ensembl.org/pub/``.
Many institutional networks and HPC clusters block the MySQL ports; HTTPS is enough. The tables are the same either
way: the dump reader decodes MySQL's escaping and takes the column types from the dumps' table definitions, so each
row holds what the MySQL server returns. The HTTPS path downloads whole compressed tables, so it is slower per table
(seconds to a minute per large table), but it needs no open port.

.. list-table::
    :header-rows: 1
    :widths: 15 85

    * - Mode
      - Behaviour
    * - ``auto`` (default)
      - Each MySQL port is probed once per Python process, with a short timeout. Releases on a port that answers are
        read over MySQL; everything else is read from the HTTPS dumps. A port whose connection fails is not tried
        again for ten minutes (then after twice as long, up to a day), with a warning; this costs only speed. A query
        or data error of the MySQL server is raised, not answered over HTTPS.
    * - ``https``
      - MySQL is never contacted; every table is read from the HTTPS dumps.

The mode is set with the environment variable ``IDTRACK_ENSEMBL_ACCESS`` or, from Python, with
:py:func:`idtrack.settings.configure_ensembl` (``access=``, which wins when set):

.. code-block:: console

    $ export IDTRACK_ENSEMBL_ACCESS=https

.. code-block:: python

    import idtrack.settings

    idtrack.settings.configure_ensembl(access="https")

**Timeouts and retries.** Every request to Ensembl REST (organism names and the latest release) waits
``connection_timeout`` seconds to connect (default 15) and ``reading_timeout`` seconds for the answer (default 60; it
also applies to the HTTPS dumps), and is tried ``rest_attempts`` times (default 3) when it times out, the connection
fails, or Ensembl answers 429 or 5xx (the fields of :py:class:`idtrack.settings.EnsemblSettings`). When every attempt
fails, :py:class:`idtrack.errors.EnsemblRestError` (an :py:class:`idtrack.errors.EnsemblAccessError`, also a
:py:class:`ConnectionError`) names the URL, the timeouts and what to change; a download of the dumps that keeps
failing raises :py:class:`idtrack.errors.EnsemblDownloadError`, and a release that cannot be listed is an error, never
read as "Ensembl has no such release". On a slow network, raise the timeouts before the first call:

.. code-block:: python

    idtrack.settings.configure_ensembl(reading_timeout=120, rest_attempts=5)

**What needs the network.** Only downloading Ensembl's tables (a build) and asking for the latest release
(:py:meth:`idtrack.API.ensembl_organism`). The names of a supported organism (``"homo_sapiens"``, ``"Homo sapiens"``,
``"human"``, ``"GRCh38"``) resolve without it, and a graph file on disk loads and converts without it, as does a
harmonization whose graph is on disk. The local repository may be read-only: it then serves the tables and graphs it
holds, and a table that would have to be computed raises :py:class:`PermissionError` naming the folder.


.. _environment-variables:

Environment variables
---------------------

idtrack reads one environment variable, ``IDTRACK_ENSEMBL_ACCESS`` (``auto`` or ``https``, above). The HTTP library
it uses, requests, honours ``HTTPS_PROXY`` (and ``REQUESTS_CA_BUNDLE`` for a certificate bundle) for the downloads.
The tutorials read ``IDTRACK_LOCAL_REPO`` for their local repository (default ``idtrack_cache``); that is a convention
of the tutorials, which pass it on to :py:meth:`idtrack.API.build`: idtrack itself takes the folder as an argument
and reads no variable for it.


Troubleshooting
---------------

**h5py fails to install or import.** On Python 3.15, or where pip finds no h5py wheel, install HDF5 first
(:ref:`python-315`). An h5py that imports but fails with an HDF5 version error was built against another HDF5 than
the one it finds: install h5py from conda-forge into a fresh environment, then idtrack with pip:

.. code-block:: console

    $ mamba create -n idtrack-env -c conda-forge python=3.13 h5py
    $ conda activate idtrack-env
    $ pip install idtrack

**SSL certificate errors while downloading Ensembl's tables or asking Ensembl REST.** Update the certificates
(``pip install --upgrade certifi``; on macOS with the python.org installer, run its ``Install Certificates.command``),
or point ``REQUESTS_CA_BUNDLE`` at your institution's bundle.

**Permission errors during installation.** Install into a virtual environment (above) rather than the system Python.

**A build cannot reach Ensembl.** Set ``IDTRACK_ENSEMBL_ACCESS=https`` where the MySQL ports are blocked; behind a
proxy, set ``HTTPS_PROXY``.

If something is not covered here, see the `GitHub issues <https://github.com/theislab/idtrack/issues>`_ or open one
with your operating system, Python and idtrack versions, the complete error message and the steps that reproduce it
(:doc:`contributing`). The tutorial :doc:`_notebooks/01_installation_guide` checks an environment step by step.

Next: :doc:`usage`.
