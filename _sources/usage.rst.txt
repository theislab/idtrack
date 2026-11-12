==========
Quickstart
==========

Get started with IDTrack in minutes. This page provides minimal, copy-pasteable code examples
to help you perform your first identifier conversions.

.. contents::
    :local:
    :depth: 2
    :backlinks: none

.. tip::

    For the complete learning path with detailed explanations, see :doc:`tutorials` (Parts 0-7).

.. note::

    IDTrack snapshots are both **release-aware** and **assembly-aware**. By default, a snapshot graph keeps the
    organism’s configured assemblies (e.g. human GRCh38 + GRCh37), so you can harmonize mixed-build inputs into one
    target space (your snapshot release + chosen output/primary assembly).


The Three-Step Workflow
-----------------------

IDTrack follows a simple three-step workflow for all identifier conversions:

.. list-table::
    :header-rows: 1
    :widths: 10 30 60

    * - Step
      - Action
      - Description
    * - 1
      - **Initialize**
      - Create an API instance with a local cache directory
    * - 2
      - **Build**
      - Build a graph snapshot for your target organism and snapshot release (multi-assembly by default)
    * - 3
      - **Convert**
      - Map identifiers using the precomputed graph

The first graph build downloads external databases and takes several minutes. Subsequent
runs load the cached graph in seconds.


Example 1: Convert a Human Gene Symbol
--------------------------------------

Convert a gene symbol to its Ensembl ID at the latest Ensembl release:

.. code-block:: python

    import os
    from pathlib import Path

    import idtrack

    # Step 1: Set up your local cache directory
    # This stores downloaded databases and built graphs for reuse
    local_repo = Path(os.environ.get("IDTRACK_LOCAL_REPO", "./idtrack_cache")).resolve()
    local_repo.mkdir(parents=True, exist_ok=True)

    # Step 2: Initialize the API
    api = idtrack.API(local_repository=str(local_repo))

    # Step 3: Resolve the organism and get the latest release
    organism, latest_release = api.resolve_organism("human")
    print(f"Using {organism} at Ensembl release {latest_release}")

    # Step 4: Build the graph (cached after first run)
    api.build_graph(
        organism_name=organism,
        snapshot_release=latest_release,
        calculate_caches=True
    )

    # Step 5: Convert an identifier
    result = api.convert_identifier("TP53", to_release=latest_release)
    target = result["target_id"][0] if result["target_id"] else None
    print(f"TP53 -> {target}")

**Expected output:**

.. code-block:: text

    Using homo_sapiens at Ensembl release 116
    TP53 -> ENSG00000141510

.. note::

    The first graph build downloads external databases and constructs the mapping graph.
    This can take several minutes depending on your internet connection.
    Subsequent runs load the pre-built graph from your local cache in seconds.


Example 2: Batch Conversion
---------------------------

Convert multiple identifiers at once for better performance:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")
    organism, release = api.resolve_organism("human")
    api.build_graph(organism_name=organism, snapshot_release=release, calculate_caches=True)

    # Convert a batch of gene symbols
    genes = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
    results = api.convert_identifier_multiple(genes, to_release=release, verbose=False)

    # Print results
    print("Gene Symbol -> Ensembl ID")
    print("-" * 30)
    for gene, result in zip(genes, results):
        target = result["target_id"][0] if result["target_id"] else None
        print(f"{gene:<10} -> {target}")

**Expected output:**

.. code-block:: text

    Gene Symbol -> Ensembl ID
    ------------------------------
    TP53       -> ENSG00000141510
    BRCA1      -> ENSG00000012048
    EGFR       -> ENSG00000146648
    KRAS       -> ENSG00000133703
    MYC        -> ENSG00000136997


Example 3: Working with Mouse
-----------------------------

IDTrack supports multiple organisms. Here's how to work with mouse:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")

    # Resolve mouse organism
    organism, release = api.resolve_organism("mouse")
    print(f"Using {organism} at Ensembl release {release}")

    # Build the mouse graph
    api.build_graph(organism_name=organism, snapshot_release=release, calculate_caches=True)

    # Convert mouse gene symbols
    mouse_genes = ["Trp53", "Brca1", "Egfr"]
    results = api.convert_identifier_multiple(mouse_genes, to_release=release, verbose=False)

    for gene, result in zip(mouse_genes, results):
        target = result["target_id"][0] if result["target_id"] else None
        print(f"{gene} -> {target}")


Example 4: Understanding Conversion Outcomes
--------------------------------------------

IDTrack is transparent about mapping ambiguity. Conversions can result in three outcomes:

.. list-table::
    :header-rows: 1
    :widths: 15 25 60

    * - Outcome
      - Meaning
      - Interpretation
    * - **1 -> 1**
      - Unique mapping
      - Ideal case: one identifier maps to exactly one target
    * - **1 -> 0**
      - No mapping found
      - Identifier may be retired, unknown, or from a different namespace
    * - **1 -> n**
      - Multiple mappings
      - Ambiguous: multiple valid targets exist (honestly reported)

By default, :meth:`idtrack.API.convert_identifier` uses ``strategy="best"`` (it returns a single best target).
To *surface ambiguity*, use ``strategy="all"`` and inspect ``result["target_id"]``.

Get detailed conversion information with the ``explain`` parameter:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")
    organism, release = api.resolve_organism("human")
    api.build_graph(organism_name=organism, snapshot_release=release, calculate_caches=True)

    # Get detailed conversion information
    result = api.convert_identifier(
        "TP53",
        to_release=release,
        strategy="best",
        explain=True  # Returns detailed mapping information
    )
    print("Targets:", result["target_id"])
    print("Matched to graph node:", result["graph_id"])
    print("No corresponding node:", result["no_corresponding"])
    print("No conversion possible:", result["no_conversion"])

The ``explain=True`` option provides details about the mapping path, intermediate nodes,
and confidence of the conversion.


Example 5: Cross-Release Conversion
-----------------------------------

Convert identifiers between different Ensembl releases to handle legacy data:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")
    organism, latest_release = api.resolve_organism("human")

    # Build graph for the latest release
    api.build_graph(
        organism_name=organism,
        snapshot_release=latest_release,
        calculate_caches=True
    )

    # Convert an identifier from an older release to the latest
    # This handles retired IDs, merged genes, and nomenclature changes
    old_ensembl_id = "ENSG00000141510"  # TP53

    result = api.convert_identifier(
        old_ensembl_id,
        from_release=100,      # Identifier came from Ensembl release 100
        to_release=latest_release
    )
    target = result["target_id"][0] if result["target_id"] else None
    print(f"Release 100 -> Release {latest_release}: {target}")

**Working with an older Ensembl release.** A graph covers every release from idtrack's floor (48) to its
``snapshot_release``, so the latest graph already converts to and from any older release
(``to_release=93``, ``from_release=93``); no older graph is needed for that. Build a graph whose snapshot is an
older release only to reproduce results as they were at that release (the graph then knows nothing after it):

.. code-block:: python

    api.build_graph(organism_name="homo_sapiens", snapshot_release=93)

The graph file is named after the window (``graph_homo_sapiens_min48_max93_narrow.pickle``), so graphs of
different snapshots live side by side in one local repository.


Environment Variable Configuration
----------------------------------

For convenience, set the ``IDTRACK_LOCAL_REPO`` environment variable to avoid specifying
the cache directory in every script.

.. tab-set::

    .. tab-item:: Bash/Zsh

        .. code-block:: bash

            # Add to your ~/.bashrc or ~/.zshrc
            export IDTRACK_LOCAL_REPO="$HOME/.idtrack"

    .. tab-item:: Fish

        .. code-block:: fish

            # Persist across sessions
            set -Ux IDTRACK_LOCAL_REPO $HOME/.idtrack

    .. tab-item:: Windows PowerShell

        .. code-block:: powershell

            # Set user environment variable
            [Environment]::SetEnvironmentVariable("IDTRACK_LOCAL_REPO", "$env:USERPROFILE\.idtrack", "User")

Then in your Python code:

.. code-block:: python

    import os
    import idtrack

    # Automatically uses IDTRACK_LOCAL_REPO if set
    local_repo = os.environ.get("IDTRACK_LOCAL_REPO", "./idtrack_cache")
    api = idtrack.API(local_repository=local_repo)


Supported Organisms
-------------------

IDTrack supports the following organisms. Each is declared once, in the organism table
``idtrack.DB.assembly_mysqlport_priority``, and ships a default external-database configuration
(``idtrack/default_config/<organism>_externals_modified.yml``).

.. list-table::
    :header-rows: 1
    :widths: 18 22 40 20

    * - Organism
      - Identifier
      - Assemblies (key: name, Ensembl releases)
      - Symbol databases
    * - Human
      - ``"homo_sapiens"``, ``"human"`` or ``9606``
      - 38: GRCh38, 76+; 37: GRCh37, 55+ (archive from 76); 36: NCBI36, 48-54
      - HGNC Symbol
    * - Mouse
      - ``"mus_musculus"``, ``"mouse"`` or ``10090``
      - 39: GRCm39, 103+; 38: GRCm38, 68-102; 37: NCBIM37, 48-67
      - MGI Symbol
    * - Pig
      - ``"sus_scrofa"`` or ``9823``
      - 111: Sscrofa11.1, 90+; 102: Sscrofa10.2, 67-89; 9: Sscrofa9, 56-66
      - VGNC Symbol, HGNC Symbol
    * - Rhesus macaque
      - ``"macaca_mulatta"`` or ``9544``
      - 10: Mmul_10, 98+; 801: Mmul_8.0.1, 86-97; 1: MMUL_1, 48-85
      - VGNC Symbol, HGNC Symbol
    * - Cynomolgus macaque
      - ``"macaca_fascicularis"`` or ``9541``
      - 6: Macaca_fascicularis_6.0, 103+; 5: Macaca_fascicularis_5.0, 91-102
      - VGNC Symbol, HGNC Symbol

The **key** of an assembly is what IDTrack uses everywhere (the ``genome_assembly`` argument, node types such as
``assembly_801_ensembl_gene``, cache files ``<organism>_assembly-<key>.h5`` and the assembly keys of the externals
YAML). It is Ensembl's code from the core database name ``<organism>_core_<release>_<code>`` wherever that is
unambiguous. Rhesus is the exception: Ensembl names both MMUL_1 (releases up to 85) and Mmul_10 (from 98)
``macaca_mulatta_core_<release>_10``, so Mmul_10 keeps key ``10`` and MMUL_1 has the key ``1``; IDTrack tells them
apart by release. Releases before 48 are not available in IDTrack. Except for human (GRCh37 and GRCh38 overlap),
every organism has one assembly per release, and IDTrack links the assemblies through Ensembl's own history.

Identifiers are matched case-insensitively. Ensembl species names (also written with spaces, e.g.
``"Macaca mulatta"``) and taxon IDs always resolve; a common name that several Ensembl species share raises an
error listing the candidates (``"pig"`` is also the common name of every pig breed in Ensembl).

The symbol databases are the default final database of :py:class:`idtrack.HarmonizeFeatures` (the first one) and
the ones to try for gene symbols. In current Ensembl, macaque genes carry VGNC symbols on the newest assemblies
and HGNC symbols on earlier ones, and cynomolgus 6.0 has few symbols of either kind.

.. tip::

    Want to add support for another organism? See "Adding Organism Support" in :doc:`contributing` for the
    recipe, and :doc:`_notebooks/02_prepare_new_external_yaml` for configuring its external databases.


Common Patterns
---------------

.. _reusing-built-graphs:

Reusing Built Graphs
~~~~~~~~~~~~~~~~~~~~

Once a graph is built, it's cached locally. Check if a graph exists before building:

.. code-block:: python

    api = idtrack.API(local_repository="./idtrack_cache")
    organism, release = api.resolve_organism("human")

    # The graph is automatically loaded from cache if it exists
    # Building is skipped if the cache is valid
    api.build_graph(
        organism_name=organism,
        snapshot_release=release,
        calculate_caches=True
    )

Loading a graph that is on disk needs no network: the first build saves, next to the tables, which releases the
Ensembl dumps hold (``ensembl_https_index_<organism>_assembly-<key>.json``), and later runs with the same or an
earlier ``snapshot_release`` read it from there (in ``https`` mode, see :ref:`ensembl-access`; ``auto`` mode
still probes the MySQL ports once). Only questions about the *latest* release need Ensembl: pass the formal name
and the release instead of calling :py:meth:`idtrack.API.resolve_organism`, which asks Ensembl REST for the
latest release:

.. code-block:: python

    api.build_graph(organism_name="homo_sapiens", snapshot_release=115)  # offline once built

Working with AnnData Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IDTrack integrates with AnnData for single-cell analysis workflows:

.. code-block:: python

    import anndata as ad

    import idtrack

    # Load your AnnData object
    adata = ad.read_h5ad("your_data.h5ad")

    # Use the harmonizer for batch feature conversion
    api = idtrack.API(local_repository="./idtrack_cache")
    organism, release = api.resolve_organism("human")
    api.build_graph(organism_name=organism, snapshot_release=release, calculate_caches=True)

    # See the harmonization tutorial for complete workflows
    # :doc:`_notebooks/05_tutorial_harmonization`

:py:meth:`idtrack.HarmonizeFeatures.unify_multiple_anndatas` merges the harmonised datasets without losing or
rewriting anything by default: ``obs`` keeps every column of any dataset (``obs_join="outer"``; a dataset without a
column has missing values there; ``obs_join="inner"`` keeps only the common columns, and ``obs_columns_to_keep``
selects among them), every ``var`` column is kept (one column where the datasets agree on each feature, else one per
dataset, ``<column>_<handle>``), layers, ``obsm``, ``obsp`` and ``uns`` entries that every dataset has are carried
over, and ``X`` keeps its dtype. ``raw`` is dropped, because its features are not harmonised; keep raw counts in a
layer.

For complete AnnData harmonization workflows, see :doc:`_notebooks/05_tutorial_harmonization`.


What's Next?
------------

Now that you've completed the quickstart, explore these resources for deeper understanding:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Resource
      - Description
    * - :doc:`_notebooks/00_idtrack_overview`
      - Understand the mental model (time axis, space axis, snapshot boundary)
    * - :doc:`_notebooks/01_installation_guide`
      - Detailed environment setup and configuration
    * - :doc:`_notebooks/03_initialization_graph`
      - Graph building, caching, and management
    * - :doc:`_notebooks/04_api_deep_dive_human`
      - Complete API reference with advanced examples
    * - :doc:`_notebooks/05_tutorial_harmonization`
      - Real-world dataset harmonization workflows
    * - :doc:`_notebooks/06_tutorial_humanization_mouse_pig_to_human`
      - Cross-species "humanization" workflows
    * - :doc:`tutorials`
      - Complete learning path (Parts 0-7)
