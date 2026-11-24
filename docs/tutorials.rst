=========
Tutorials
=========

.. toctree::
   :maxdepth: 1
   :hidden:

   _notebooks/00_idtrack_overview
   _notebooks/01_installation_guide
   _notebooks/02_initialization_graph
   _notebooks/03_api_deep_dive_human
   _notebooks/04_tutorial_harmonization
   _notebooks/05_tutorial_humanization_mouse_pig_to_human
   _notebooks/06_prepare_new_external_yaml
   _notebooks/07_advanced_topics
   Appendix <_notebooks/99_initialization_test>

Jupyter notebooks, in the order a reader needs them: install, build or load a graph, convert, harmonize, then the
advanced topics. Every organism ships its externals configuration, so the configuration tutorial (Part 6) is needed
only to change which external databases a graph holds. Each page has a "Download notebook" link; the outputs shown
were made with this version on the graphs of Ensembl 115. The reference pages they link to (:doc:`concepts`,
:doc:`reading`, :doc:`limitations`, :doc:`performance`) hold the rules and numbers in full.

.. list-table::
    :header-rows: 1
    :class: idtrack-tutorials-table

    * - Part
      - What you will learn
      - Notebook
    * - Part 0
      - The mental model: releases and assemblies, the snapshot release, the outcomes of a conversion
      - :doc:`_notebooks/00_idtrack_overview`
    * - Part 1
      - Installation and a clean, reusable local repository
      - :doc:`_notebooks/01_installation_guide`
    * - Part 2
      - Building graphs for the five organisms, loading them, managing graph files
      - :doc:`_notebooks/02_initialization_graph`
    * - Part 3
      - The API in depth: single and batch conversion, results, explained paths, introspection
      - :doc:`_notebooks/03_api_deep_dive_human`
    * - Part 4
      - Harmonizing real datasets: integration hygiene, legacy rescue
      - :doc:`_notebooks/04_tutorial_harmonization`
    * - Part 5
      - Cross-species "humanization" with an orthology table
      - :doc:`_notebooks/05_tutorial_humanization_mouse_pig_to_human`
    * - Part 6
      - The externals configuration (YAML) of each organism, extending it to new Ensembl releases, adding an organism
      - :doc:`_notebooks/06_prepare_new_external_yaml`
    * - Part 7
      - Production patterns: profiles, automation, diagnostics, pipelines
      - :doc:`_notebooks/07_advanced_topics`
    * - Appendix
      - Self-tests and sanity checks of built graphs
      - :doc:`Appendix <_notebooks/99_initialization_test>`
