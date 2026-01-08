=========
Tutorials
=========

These tutorials are the **primary learning resource** for IDTrack. They are written for wet-lab researchers (basic Python familiarity) and bioinformaticians (advanced patterns), and are meant to be read **in order**.

How to use the tutorials
------------------------

1. Read **Part 0** first (mental model).
2. Run **Part 1** once per machine/environment (installation + cache directory).
3. Run **Part 2–3** once per organism/configuration (YAML + graph snapshots).
4. Reuse the built snapshots for all downstream work (Parts 4–7).

.. note::

   Notebook pages are rendered as documentation and are **not executed** during the docs build.
   Download the notebooks and run them locally if you want to execute the code cells.

.. list-table:: Quick map (by task)
    :header-rows: 1

    * - Task
      - Tutorial(s)
    * - Understand the mental model and workflow
      - :doc:`_notebooks/00_idtrack_overview`
    * - Install + verify your environment
      - :doc:`_notebooks/01_installation_guide`
    * - Prepare external database configuration (YAML)
      - :doc:`_notebooks/prepare_new_external_yaml`
    * - Build graph snapshots and manage caches
      - :doc:`_notebooks/initialization_graph`
    * - Run sanity checks (optional)
      - :doc:`_notebooks/initialization_test`
    * - Convert identifiers with the human API (deep dive)
      - :doc:`_notebooks/api_deep_dive_human`
    * - Harmonize datasets (HLCA-style) + legacy rescue
      - :doc:`_notebooks/tutorial_harmonization`, :doc:`_notebooks/tutorial_hlca_experiments`
    * - Cross-species humanization (mouse/pig → human)
      - :doc:`_notebooks/tutorial_humanization_mouse_pig_to_human`
    * - Advanced topics (profiles, diagnostics, pipelines)
      - :doc:`_notebooks/08_advanced_topics`

.. toctree::
    :maxdepth: 1
    :caption: Part 0 — Conceptual Foundation 📚

    _notebooks/00_idtrack_overview

.. toctree::
    :maxdepth: 1
    :caption: Part 1 — Environment Setup & Installation 💻

    _notebooks/01_installation_guide

.. toctree::
    :maxdepth: 1
    :caption: Part 2 — External Database Configuration 💻

    _notebooks/prepare_new_external_yaml

.. toctree::
    :maxdepth: 1
    :caption: Part 3 — Graph Initialization & Management 💻

    _notebooks/initialization_graph
    _notebooks/initialization_test

.. toctree::
    :maxdepth: 1
    :caption: Part 4 — Core API Deep-Dive 💻

    _notebooks/api_deep_dive_human

.. toctree::
    :maxdepth: 1
    :caption: Part 5 — Real-World Experiments 🔬

    _notebooks/tutorial_harmonization
    _notebooks/tutorial_hlca_experiments

.. toctree::
    :maxdepth: 1
    :caption: Part 6 — Cross-Species Workflows 💻

    _notebooks/tutorial_humanization_mouse_pig_to_human

.. toctree::
    :maxdepth: 1
    :caption: Part 7 — Advanced Topics 🛠️

    _notebooks/08_advanced_topics

