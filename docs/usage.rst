Quickstart
==========

This page is a minimal, copy/pasteable quickstart. For the full learning path (Parts 0–7), see :doc:`tutorials`.

Minimal human conversion (Python)
---------------------------------

.. code-block:: python

    import os
    from pathlib import Path

    import idtrack

    # Choose a stable cache directory (graphs + downloads + YAML live here)
    local_repo = Path(os.environ.get("IDTRACK_LOCAL_REPO", "./idtrack_cache")).resolve()
    local_repo.mkdir(parents=True, exist_ok=True)

    api = idtrack.API(local_repository=str(local_repo))
    organism, latest_release = api.resolve_organism("human")

    # Build once, then reuse (loads from cache on subsequent runs)
    api.build_graph(organism_name=organism, snapshot_release=latest_release, calculate_caches=True)

    # Convert a symbol to Ensembl at the snapshot boundary
    api.convert_identifier("TP53", to_release=latest_release)

What to do next
---------------

- For installation + environment verification: :doc:`_notebooks/01_installation_guide`
- For external database configuration (YAML): :doc:`_notebooks/prepare_new_external_yaml`
- For graph builds (human/mouse/pig + cache management): :doc:`_notebooks/initialization_graph`
- For the full tutorial suite: :doc:`tutorials`
