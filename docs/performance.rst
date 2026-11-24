.. _time-and-memory:

===============
Time and Memory
===============

What building, loading and converting cost, for the five organisms, snapshot release 115, the window from each
organism's first release and the shipped externals configurations.
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
      - 13 to 20 min
      - 14 to 23 min
      - 8 to 10 min
      - 40 to 60 min
      - 2 h 16 to 2 h 35
    * - Peak memory of the build
      - 1.8 to 2.6 GB
      - 1.7 to 2.4 GB
      - 1.1 to 1.7 GB
      - 4.6 to 5.5 GB
      - 6.8 to 9.6 GB
    * - Tables on disk
      - at most 1.1 GB
      - at most 1.3 GB
      - 0.24 GB
      - at most 3.2 GB
      - at most 9.1 GB
    * - Graph file
      - 40.2 MB
      - 34.0 MB
      - 22.5 MB
      - 100.9 MB
      - 155.6 MB
    * - Load of the graph file
      - 0.82 s
      - 0.68 s
      - 0.45 s
      - 2.26 s
      - 3.96 s
    * - Resident memory after the load
      - 0.43 GB
      - 0.39 GB
      - 0.25 GB
      - 0.95 GB
      - 1.39 GB
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

How each row was measured, with this version on a Linux server:

* **Build** and **peak memory**: two builds of each graph from Ensembl's tables already in the local repository, one
  process each, on two machines; the range is theirs (a build that also computes the derived tables of every release
  takes the longer time). A first build also downloads the tables, which depends on the network: the cynomolgus graph
  took 12 to 14 minutes from nothing over HTTPS, the human graph about 2.3 to 2.6 hours plus the download. Run a long
  build where it may run for hours (a batch job, ``nohup``); an interrupted build starts again from the tables already
  downloaded.
* **Tables on disk**: a cynomolgus repository built from nothing holds 0.24 GB; for the others, the table folders of
  those builds, which also hold tables of other selections of databases, so these are upper bounds.
* **Graph file**: the files of the graphs of 115. Sizes vary a little from build to build (the cynomolgus graph file
  has been 22.5 to 24.1 MB).
* **Load** and **memory**: :py:meth:`idtrack.API.load` in a new process with the file in the operating system's cache,
  the lower median of three runs, and the process's resident memory after it. ``import idtrack`` adds about 0.5 s (up
  to 16 s on a cold network file system); the first load of a file on a machine reads it from disk and takes longer.
  Loading needs no network and no table.
* **Conversions**: one call of :py:meth:`idtrack.API.convert` per distinct identifier of a fixed, seeded sample of
  7,468 to 8,051 strings per organism (gene, transcript and translation ids with and without version, symbols and
  other external ids of every release of the window, converted to a release of the graph and partly to an external
  database), Python 3.13, 4 cores, the lower median of three runs. The slowest calls are the cases of
  :ref:`known-costs`: a name displayed for thousands of genes (pig and the macaques), the first question about a copy
  of a gene (mouse and human), each of them avoidable as that section says. :py:meth:`idtrack.API.convert_many`
  converts each distinct string of a batch once.

Each conversion has a budget (:py:class:`idtrack.settings.SearchSettings`: 200,000 search states and 30 s by default;
``API.load(path, search=SearchSettings(...))`` sets other defaults for an API, and ``budget_states=`` /
``budget_seconds=`` of :py:meth:`idtrack.API.convert` bound one call); a search past it raises
:py:class:`idtrack.errors.SearchBudgetExceededError` (no conversion of the measurements came near it). Threads share a
loaded graph but do not convert in parallel; a server scales with processes (:ref:`known-costs`).

The harmonizer (:doc:`harmonization`) converts every feature of every dataset in its constructor: on three test
datasets of 50 cells that hold every gene of the organism, 35 s (cynomolgus) to 199 s (human), at a peak of 0.8 to
2.9 GB; merging them (:py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas`) 8 to 19 s.
