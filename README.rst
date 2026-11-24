**IDTrack**
===========

|PyPI| |PyPIDownloads| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov|

.. |PyPI| image:: https://img.shields.io/pypi/v/idtrack.svg
   :target: https://pypi.org/project/idtrack/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/idtrack
   :target: https://pypi.org/project/idtrack
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/theislab/idtrack
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/idtrack/latest.svg?label=Read%20the%20Docs
   :target: https://idtrack.readthedocs.io/
   :alt: Read the documentation at https://idtrack.readthedocs.io/
.. |Build| image:: https://github.com/theislab/idtrack/actions/workflows/build_package.yml/badge.svg?branch=main
   :target: https://github.com/theislab/idtrack/actions/workflows/build_package.yml
   :alt: Build Package Status
.. |Tests| image:: https://github.com/theislab/idtrack/actions/workflows/run_tests.yml/badge.svg?branch=main
   :target: https://github.com/theislab/idtrack/actions/workflows/run_tests.yml
   :alt: Tests status
.. |PyPIDownloads| image:: https://pepy.tech/badge/idtrack
   :target: https://pepy.tech/project/idtrack
   :alt: downloads
.. |Codecov| image:: https://codecov.io/gh/theislab/idtrack/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/theislab/idtrack
   :alt: Codecov

.. image:: https://raw.githubusercontent.com/theislab/idtrack/main/docs/_logo/logo.png
    :width: 350
    :alt: IDTrack logo

Cross-Temporal and Cross-Database Biological Identifier Mapping
--------------------------------------------------------------------

**IDTrack** converts gene identifiers (Ensembl gene, transcript and translation ids, symbols, and the ids of external
databases such as HGNC, NCBI Gene, RefSeq and UniProt) to the genes of one chosen Ensembl release, or to their ids in
a chosen database. It is for anyone with gene lists or matrices made with different Ensembl releases, genome
assemblies or databases. It follows each identifier through Ensembl's own history of retired, merged and split ids,
across assemblies; it says how each string was read and which evidence decided; it returns a tie instead of a guess
when the evidence cannot decide; and it states its own limits with how often they matter.

It supports human, mouse, pig, rhesus macaque and cynomolgus macaque, on Python 3.12 to 3.15.

.. code-block:: console

    $ pip install idtrack

.. code-block:: python

    import os

    from idtrack import API

    os.makedirs("./idtrack_cache", exist_ok=True)              # the local repository: tables and graph files
    api = API.build("homo_sapiens", 115, "./idtrack_cache")    # built once (hours for human), then loaded in seconds
    print(api.convert("TP53").targets())                                                    # ('ENSG00000141510.20',)
    print(api.convert("ENSG00000141510", from_release=100, final_database="HGNC Symbol").targets())   # ('TP53',)

The same from the command line:

.. code-block:: console

    $ idtrack build human 115 --repository idtrack_cache
    $ idtrack convert idtrack_cache/graph_homo_sapiens_assembly38_min48_max115.npz TP53 BRCA1

What it does
------------

* **Across releases**: an identifier of any release from the organism's first one in IDTrack to the graph's
  snapshot release converts to any other, through Ensembl's id history (versions, replacements, splits, merges, with
  Ensembl's mapping scores).
* **Across assemblies**: identifiers of earlier genome builds (human NCBI36 and GRCh37, mouse NCBIM37 and GRCm38, ...)
  convert to the build Ensembl maintains at the target release.
* **Across databases**: symbols and ids of external databases convert to Ensembl genes and back, read as each
  database wrote them at each release.
* **Reproducible**: a graph is fixed by its organism, snapshot release and selection of databases; the same graph
  gives the same answer in every run, and records what built it.
* **Transparent**: every result has a status and, where it has no target, a reason; ties are returned as ties; with
  ``explain=True`` the path behind the answer is data.
* **Fast once built**: a graph loads from disk in 1 to 6 seconds into 0.3 to 1.5 GB of memory, without network, and a
  conversion takes about 0.2 to 0.3 ms at the median.
* **Harmonization** of ``.h5ad`` datasets onto one release (``pip install "idtrack[harmonize]"``).

Documentation
-------------

https://idtrack.readthedocs.io/: the `Quickstart <https://idtrack.readthedocs.io/en/latest/usage.html>`_, `how IDTrack
answers <https://idtrack.readthedocs.io/en/latest/concepts.html>`_, its `limitations
<https://idtrack.readthedocs.io/en/latest/limitations.html>`_, the `command line
<https://idtrack.readthedocs.io/en/latest/cli.html>`_, the tutorials and the reference. Setting up, building and
checking graphs, testing and releasing: `SETUP.md`_.

.. _SETUP.md: https://github.com/theislab/idtrack/blob/main/SETUP.md
