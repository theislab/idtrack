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
* **Assembly-aware mapping**: harmonize identifiers across genome builds (e.g. GRCh37 ↔ GRCh38) and respect external databases that are assembly-scoped.
* **Snapshot boundary for reproducibility**: build a release-bounded graph snapshot so results are stable and repeatable.
* **Explicit external database opt-in**: choose which external namespaces participate via a small, editable YAML contract.
* **Transparency over coercion**: conversions are naturally classified as **1→0** (no match), **1→1** (clean), or **1→n** (ambiguous).
* **Scale-ready workflows**: caching and snapshot reuse make repeated conversions and multi-dataset harmonization practical.

Who is it for?
--------------

* Wet-lab researchers who need a reliable, step-by-step path from “my gene list is old” to “my analysis is reproducible”.
* Bioinformaticians who want release-pinned, auditable conversions in notebooks, pipelines, and integration workflows.
* Atlas builders / integrators who need to harmonize gene identifiers across many cohorts (different Ensembl releases, symbols, and external IDs), keep an explicit audit trail of what mapped/failed/was ambiguous, and ship a release-pinned, reproducible feature space for downstream integration and publication.

Common use cases
----------------

* **Dataset harmonization** before integration (single-cell, bulk, atlas-scale collections).
* **Legacy data rescue** (old Ensembl releases, mixed symbols/IDs, retired identifiers).
* **Publication-grade reproducibility** (pin a snapshot boundary + share the exact external configuration).
* **Cross-database interoperability** when collaborators use different identifier conventions.

Documentation and tutorials
---------------------------

The documentation includes a **full tutorial suite** designed to be the primary learning resource:

* Documentation: Documentation_
* Tutorials: start from the “Tutorials” section in the docs (Part 0 → Part 7).

.. _PyPI: https://pypi.org/
.. _pip: https://pip.pypa.io/
.. _Documentation: https://idtrack.readthedocs.io/
