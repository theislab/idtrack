=========
Tutorials
=========

.. toctree::
   :maxdepth: 1
   :hidden:

   _notebooks/00_idtrack_overview
   _notebooks/01_installation_guide
   _notebooks/prepare_new_external_yaml
   _notebooks/initialization_graph
   _notebooks/api_deep_dive_human
   _notebooks/tutorial_harmonization
   _notebooks/tutorial_humanization_mouse_pig_to_human
   _notebooks/08_advanced_topics
   Appendix <_notebooks/initialization_test>

These tutorials are the **primary learning resource** for IDTrack. They are written for wet-lab researchers (basic Python familiarity) and bioinformaticians (advanced patterns), and they cover both cross-release and cross-assembly identifier mapping.

.. rubric:: Learning path

.. list-table::
    :header-rows: 1
    :class: idtrack-tutorials-table

    * - Part
      - What you will learn
      - Start here
    * - Part 0
      - The mental model (time axis vs space axis, snapshot boundary, outcomes 1→0/1→1/1→n)
      - :doc:`_notebooks/00_idtrack_overview`
    * - Part 1
      - Installation and a clean, reusable local repository directory
      - :doc:`_notebooks/01_installation_guide`
    * - Part 2
      - External database configuration (YAML) for human, mouse, pig (and how to add a new organism)
      - :doc:`_notebooks/prepare_new_external_yaml`
    * - Part 3
      - Build snapshot graphs (multi-assembly by default), reuse caches, and manage rebuilds across organisms and output assemblies
      - :doc:`_notebooks/initialization_graph`
    * - Part 4
      - Core API usage (single + batch conversion, explainability, introspection)
      - :doc:`_notebooks/api_deep_dive_human`
    * - Part 5
      - Real dataset workflows (harmonization, integration hygiene, legacy rescue)
      - :doc:`_notebooks/tutorial_harmonization`
    * - Part 6
      - Cross-species “humanization” workflows and comparative-analysis preparation
      - :doc:`_notebooks/tutorial_humanization_mouse_pig_to_human`
    * - Part 7
      - Production patterns (profiles, automation, diagnostics, pipelines)
      - :doc:`_notebooks/08_advanced_topics`
    * - Appendix
      - Self-tests, sanity checks, and troubleshooting helpers
      - :doc:`Appendix <_notebooks/initialization_test>`
