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

Modern biology constantly mixes identifiers from different years, databases, and genome builds. The result is a familiar set of problems:
IDs disappear, symbols change, references disagree, and “the same gene” isn’t always represented the same way across datasets.

**IDTrack** is built for that reality. It provides a **time-aware, audit-friendly** way to translate and harmonize biological identifiers
across **Ensembl releases** and across **external namespaces** (HGNC, UniProt, RefSeq, Entrez, …), while keeping ambiguity explicit
instead of silently forcing a single answer.

What makes IDTrack different
----------------------------

* **Time-aware mapping**: treat Ensembl releases as a “time axis” and travel forward/backward through identifier history.
* **Assembly-aware mapping**: identifiers of earlier genome builds (e.g. GRCh37) convert to the build Ensembl maintains at the target release, and external databases are read for the assembly they belong to.
* **Snapshot boundary for reproducibility**: build a release-bounded graph snapshot; the same snapshot gives the same answers in every run.
* **Explicit external database opt-in**: choose which external namespaces participate via a small, editable YAML contract.
* **Transparency over coercion**: conversions are classified as **1→0** (no match), **1→1** (clean), or **1→n** (ambiguous); an answer shared by several equally supported genes is returned as a tie, and every result says how the identifier was read.
* **Fast once built**: a graph built once loads from disk in 1 to 6 seconds into 0.3 to 1.5 GB of memory, without network, and a conversion takes about 0.2 to 0.3 ms at the median (measured for the five supported organisms; see the Quickstart's Time and Memory).

Who is it for?
--------------

* Wet-lab researchers who need a step-by-step path from “my gene list is old” to “my analysis is reproducible”.
* Bioinformaticians who want release-pinned, auditable conversions in notebooks, pipelines, and integration workflows.
* Atlas builders / integrators who need to harmonize gene identifiers across many cohorts (different Ensembl releases, symbols, and external IDs), keep an explicit audit trail of what mapped/failed/was ambiguous, and ship a release-pinned, reproducible feature space for downstream integration and publication.

Common use cases
----------------

* **Dataset harmonization** before integration (single-cell, bulk, atlas-scale collections).
* **Legacy data rescue** (old Ensembl releases, mixed symbols/IDs, retired identifiers).
* **Publication-grade reproducibility** (pin a snapshot boundary + share the exact external configuration).
* **Cross-database interoperability** when collaborators use different identifier conventions.

Installation and a first conversion
-----------------------------------

IDTrack runs on Python 3.12 to 3.15 (on 3.15, install HDF5 and a C compiler first: h5py has no wheels for it yet; see
the installation page) and supports human, mouse, pig, rhesus macaque and cynomolgus macaque. Install it from GitHub
(with the ``harmonize`` extra for harmonizing ``.h5ad`` datasets, which brings anndata and scipy):

.. code-block:: console

    $ pip install "git+https://github.com/theislab/idtrack.git"
    $ pip install "idtrack[harmonize] @ git+https://github.com/theislab/idtrack.git"

The release on PyPI (``pip install idtrack``, version 0.0.5) is an older, different version with another API; use
GitHub until this version is published on PyPI.

.. code-block:: python

    import os

    from idtrack import API

    os.makedirs("./idtrack_cache", exist_ok=True)  # the local repository: downloaded tables and graph files
    # Built from Ensembl's tables the first time (hours for human), loaded from disk in seconds afterwards.
    api = API.build("homo_sapiens", 115, "./idtrack_cache")
    # A graph file on disk loads without any network: api = API.load(API.graph_path("homo_sapiens", 115, "./idtrack_cache"))

    result = api.convert("TP53")  # to the snapshot release
    print(result.targets())       # ('ENSG00000141510.20',)
    result = api.convert("ENSG00000141510", from_release=100, final_database="HGNC Symbol")
    print(result.targets())       # ('TP53',)

The same from the command line (``idtrack --help`` says more):

.. code-block:: console

    $ idtrack build human 115 --repository idtrack_cache
    $ idtrack convert idtrack_cache/graph_homo_sapiens_assembly38_min48_max115.npz TP53 BRCA1

Documentation and tutorials
---------------------------

The documentation includes a **full tutorial suite** designed to be the primary learning resource:

* Documentation: Documentation_
* Tutorials: start from the “Tutorials” section in the docs (Part 0 → Part 7).
* What this version does not do yet, with how often it matters: `Current Limitations`_.
* Setting up, building and checking graphs, testing and releasing: `SETUP.md`_.

.. _PyPI: https://pypi.org/
.. _pip: https://pip.pypa.io/
.. _Documentation: https://idtrack.readthedocs.io/
.. _Current Limitations: https://idtrack.readthedocs.io/en/latest/usage.html#current-limitations
.. _SETUP.md: https://github.com/theislab/idtrack/blob/main/SETUP.md
