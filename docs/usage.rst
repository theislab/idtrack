Tutorials
=========

These tutorials are written for end users (not developers) and are meant to be read **in order**.

How to read these notebooks:

1. Start at the top of the series and work downward.
2. Copy/paste code cells into your own notebook if you prefer (that’s a normal workflow).
3. Keep your `IDTRACK_LOCAL_REPO` stable within a project so your graphs and caches are reused.

.. list-table:: Quick map (by task)
    :header-rows: 1

    * - Task
      - Tutorial(s)
    * - Understand the mental model and workflow
      - :doc:`_notebooks/00_idtrack_overview`
    * - Prepare external YAMLs (human/mouse/pig)
      - :doc:`_notebooks/prepare_new_external_yaml`
    * - Build graph snapshots (human/mouse/pig)
      - :doc:`_notebooks/initialization_graph`
    * - Run sanity checks (human/mouse/pig)
      - :doc:`_notebooks/initialization_test`
    * - Convert IDs with the human API (deep dive)
      - :doc:`_notebooks/api_deep_dive_human`
    * - Harmonize multiple datasets (HLCA-style)
      - :doc:`_notebooks/tutorial_harmonization`, :doc:`_notebooks/tutorial_hlca_experiments`
    * - Humanize mouse/pig to human (advanced)
      - :doc:`_notebooks/tutorial_humanization_mouse_pig_to_human`

..
    .. click:: idtrack.__main__:main
       :prog: idtrack
       :nested: full

.. toctree::
    :maxdepth: 1
    :caption: 00–03 — Multi-Organism Setup (Human/Mouse/Pig)

    _notebooks/00_idtrack_overview
    _notebooks/prepare_new_external_yaml
    _notebooks/initialization_graph
    _notebooks/initialization_test

.. toctree::
    :maxdepth: 1
    :caption: 04–06 — Human Workflows (Deep Dive + HLCA)

    _notebooks/api_deep_dive_human
    _notebooks/tutorial_harmonization
    _notebooks/tutorial_hlca_experiments

.. toctree::
    :maxdepth: 1
    :caption: 07 — Cross-Species (Advanced)

    _notebooks/tutorial_humanization_mouse_pig_to_human
