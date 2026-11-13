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
* **Fast once built**: a graph built once loads from disk in 1 to 5 seconds into 0.26 to 1.14 GB of memory, without network, and a conversion takes 0.2 to 0.3 ms at the median (measured for the five supported organisms; see the Quickstart).

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

IDTrack runs on Python 3.12 to 3.15 and supports human, mouse, pig, rhesus macaque and cynomolgus macaque.

.. code-block:: console

    $ pip install idtrack
    $ pip install "git+https://github.com/theislab/idtrack.git@main"   # or: the code of the main branch

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")
    # Built from Ensembl's tables the first time (about 2 hours for human), loaded from disk in seconds afterwards.
    api.build_graph(organism_name="homo_sapiens", snapshot_release=115)

    result = api.convert_identifier("TP53")  # to the snapshot release
    print(result["target_id"])                # ['ENSG00000141510.20']
    result = api.convert_identifier("ENSG00000141510", from_release=100, final_database="HGNC Symbol")
    print(result["target_id"])                # ['TP53']

Start here
----------

* :doc:`tutorials` — the full tutorial suite (Part 0 → Part 7); primary learning resource
* :doc:`usage` — a minimal quickstart
* :doc:`installation` — installation, Ensembl access (MySQL or HTTPS), troubleshooting
* :doc:`changes` — what changed, and what a user of an earlier version must do
