================
External Mappers
================

External mappers are **optional helpers** for cross-database mapping and orthology-related utilities.
They are *not* the core “time-aware graph snapshot” engine (that lives in the backbone graph), but they can be useful for:

- fetching or aligning ortholog tables for cross-species workflows
- using external services to map between common identifier types when you need a convenience layer

.. note::

   Some external mapper backends require optional dependencies (or network access).
   The Part 6 tutorial shows how to check availability before you rely on them.

.. contents::
   :local:
   :depth: 3
   :backlinks: none


Package API
-----------

.. automodule:: idtrack._external_mappers
   :members:
   :private-members:
   :show-inheritance: True


Conversion helpers
------------------

.. automodule:: idtrack._external_mappers._convert
   :members:
   :private-members:
   :show-inheritance: True


Orthology helpers
-----------------

.. automodule:: idtrack._external_mappers._ortholog
   :members:
   :private-members:
   :show-inheritance: True


Backend implementations
-----------------------

.. automodule:: idtrack._external_mappers._backend_pybiomart
   :members:
   :private-members:
   :show-inheritance: True

.. automodule:: idtrack._external_mappers._backend_mygene
   :members:
   :private-members:
   :show-inheritance: True

.. automodule:: idtrack._external_mappers._backend_gprofiler
   :members:
   :private-members:
   :show-inheritance: True

.. automodule:: idtrack._external_mappers._backend_gget
   :members:
   :private-members:
   :show-inheritance: True


Shared utilities and constants
------------------------------

.. automodule:: idtrack._external_mappers._utils
   :members:
   :private-members:
   :show-inheritance: True

.. automodule:: idtrack._external_mappers._constants
   :members:
   :private-members:
   :show-inheritance: True


Ontology
--------

This section defines the vocabulary used throughout IDTrack’s tutorials and API reference.
The goal is to make results *interpretable* without requiring graph theory or database background.

.. glossary::

   Ensembl release
      A numbered snapshot of Ensembl reference data (e.g., release 114). In IDTrack, releases define the **time axis**.

   Snapshot release (snapshot boundary)
      The release you choose as the **upper time boundary** for a graph snapshot. It makes conversions reproducible: the same inputs and the same snapshot boundary should yield the same outputs.

   Backbone namespace
      The Ensembl “core” identifier spaces that carry historical relationships across releases (e.g., Ensembl gene IDs). Backbone edges enable time-travel mapping across releases.

   External namespace
      A non-Ensembl identifier system (HGNC, EntrezGene, UniProtKB, RefSeq, MGI, …). External edges connect backbone nodes to external identifiers when enabled by configuration.

   External YAML
      A user-editable configuration file that declares which external namespaces are allowed to participate in mapping for a given organism/snapshot. It is an explicit contract that improves reproducibility and reduces accidental ambiguity.

   Assembly
      The genome build context for an organism (e.g., human GRCh38 vs GRCh37). Assemblies can affect which releases are reachable and which identifiers are valid.

   Graph snapshot
      A precomputed mapping graph built for a specific organism and snapshot release (and often multiple assemblies). It is stored on disk under your local repository so it can be reused across sessions.

   Identifier drift
      The fact that identifiers change across releases (retirements, merges, splits, version changes) and differ across databases. Drift is the core reason conversions must be time-aware.

   1→0 / 1→1 / 1→n outcomes
      The three conversion outcome families: no mapping, unique mapping, or ambiguous mapping with multiple valid targets. IDTrack reports these explicitly rather than silently forcing a single answer.

   Strategy (best vs all)
      A conversion policy. ``strategy='best'`` returns a single preferred target; ``strategy='all'`` returns all plausible targets so you can handle ambiguity explicitly.

   Explainability payload
      Optional structured information returned alongside a conversion result that helps you audit *why* a mapping happened (paths, intermediate nodes, decisions).

   Hyperconnected node
      An identifier that connects to many other identifiers (common in some external namespaces). These can explode the search space; IDTrack uses safeguards so “promiscuous” nodes do not dominate results.

   Local repository (cache directory)
      A writable directory where IDTrack stores downloads, derived tables, and graph snapshots. You can set it via the ``IDTRACK_LOCAL_REPO`` environment variable.
