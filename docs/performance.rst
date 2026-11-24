.. _time-and-memory:

===============
Time and Memory
===============

What building, loading and converting cost, for the five organisms, snapshot release 115, the window from each
organism's first release and the shipped externals configurations, measured with this version on a Linux server.
Other pages give ranges and point here; this table is the one place that holds the numbers.

.. list-table::
    :header-rows: 1
    :widths: 28 14 14 14 14 16

    * -
      - Pig
      - Rhesus
      - Cynomolgus
      - Mouse
      - Human
    * - Build (tables on disk)
      - 13 min
      - 14 min
      - 8 min
      - 40 min
      - 2 h 35 min
    * - Peak memory of the build
      - 1.8 GB
      - 1.7 GB
      - 1.1 GB
      - 4.6 GB
      - 6.8 GB
    * - Tables on disk
      - at most 1.1 GB
      - at most 1.2 GB
      - 0.24 GB
      - at most 3.3 GB
      - at most 9.3 GB
    * - Graph file
      - 40.2 MB
      - 34.0 MB
      - 22.5 MB
      - 100.9 MB
      - 155.6 MB
    * - Import and load (file in the system's cache)
      - 1.2 s
      - 1.1 s
      - 1.8 s
      - 2.9 s
      - 5.4 s
    * - Memory after load
      - 0.45 GB
      - 0.42 GB
      - 0.28 GB
      - 0.95 GB
      - 1.48 GB
    * - Single conversions per second
      - 1,339
      - 692
      - 739
      - 313
      - 513
    * - Per conversion: median / 99th percentile
      - 0.28 / 4.4 ms
      - 0.22 / 3.1 ms
      - 0.21 / 1.3 ms
      - 0.28 / 7.6 ms
      - 0.26 / 2.9 ms
    * - Slowest conversion
      - 1.2 s
      - 7.3 s
      - 6.2 s
      - 9.7 s
      - 11.3 s

What each row measured:

* **Build**: one process, from Ensembl's tables already in the local repository. A first build also downloads the
  tables, which depends on the network: the cynomolgus graph took 12 to 14 minutes from nothing over HTTPS, the human
  graph about 2.6 hours plus the download of its tables. The table sizes other than cynomolgus's were measured on
  repositories that also held tables of earlier builds, so they are upper bounds. Sizes and times vary a little from
  build to build (the cynomolgus graph file has been 22.5 to 24.1 MB). Run a long build where it may run for hours (a
  batch job, ``nohup``); an interrupted build starts again from the tables already downloaded.
* **Import and load**: a new Python process, ``import idtrack`` and :py:meth:`idtrack.API.load` together, with the
  file in the operating system's cache. The first load of a file on a machine reads it from disk and takes longer
  (cynomolgus 9.6 s once on a network file system). Loading needs no network and no table.
* **Conversions**: one call of :py:meth:`idtrack.API.convert` per distinct identifier of a fixed, seeded sample of
  7,468 to 8,051 strings per organism (gene, transcript and translation ids with and without version, symbols and
  other external ids of every release of the window, converted to a release of the graph and partly to an external
  database), Python 3.13, 4 cores. The slowest calls are the cases of :ref:`known-costs`: a name displayed for
  thousands of genes (pig and the macaques), the first question about a copy of a gene (mouse and human), each of them
  avoidable as that section says. :py:meth:`idtrack.API.convert_many` converts each distinct string of a batch once.

Each conversion has a budget (:py:class:`idtrack.settings.SearchSettings`: 200,000 search states and 30 s by default;
``API.load(path, search=SearchSettings(...))`` sets other defaults for an API, and ``budget_states=`` /
``budget_seconds=`` of :py:meth:`idtrack.API.convert` bound one call); a search past it raises
:py:class:`idtrack.errors.SearchBudgetExceededError` (no conversion of the measurements came near it). Threads share a
loaded graph but do not convert in parallel; a server scales with processes (:ref:`known-costs`).

The harmonizer (:doc:`harmonization`) converts every feature of every dataset in its constructor: on three test
datasets of 50 cells that hold every gene of the organism, 35 s (cynomolgus) to 199 s (human), at a peak of 0.8 to
2.9 GB; merging them (:py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas`) 8 to 19 s.
