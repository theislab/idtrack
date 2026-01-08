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

    Using homo_sapiens at Ensembl release 114
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

IDTrack currently supports the following organisms:

.. list-table::
    :header-rows: 1
    :widths: 20 30 50

    * - Organism
      - Identifier
      - External Database Support
    * - Human
      - ``"human"`` or ``"homo_sapiens"``
      - Full Ensembl + HGNC, NCBI, UniProt
    * - Mouse
      - ``"mouse"`` or ``"mus_musculus"``
      - Full Ensembl + MGI, NCBI, UniProt
    * - Pig
      - ``"pig"`` or ``"sus_scrofa"``
      - Full Ensembl support

.. tip::

    Want to add support for another organism? See :doc:`_notebooks/prepare_new_external_yaml`
    for instructions on configuring external databases.


Common Patterns
---------------

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
    # :doc:`_notebooks/tutorial_harmonization`

For complete AnnData harmonization workflows, see :doc:`_notebooks/tutorial_harmonization`.


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
    * - :doc:`_notebooks/initialization_graph`
      - Graph building, caching, and management
    * - :doc:`_notebooks/api_deep_dive_human`
      - Complete API reference with advanced examples
    * - :doc:`_notebooks/tutorial_harmonization`
      - Real-world dataset harmonization workflows
    * - :doc:`_notebooks/tutorial_humanization_mouse_pig_to_human`
      - Cross-species "humanization" workflows
    * - :doc:`tutorials`
      - Complete learning path (Parts 0-7)
