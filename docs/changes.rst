Changes
=======

What changed for users, newest first. For each version: what you must do, the answers that change, and what is new.
Version numbers follow ``pyproject.toml``; the package, its web app and its benchmark repository carry one version
number, and each release tags all three ``v<version>``.


0.1.2
-----

**What you must do:** nothing. Conversions, graphs and file formats are those of 0.1.0 and 0.1.1
(:py:data:`idtrack.settings.RULES_VERSION` is unchanged, so a harmonizer's result cache stays valid); graphs built
with 0.1.0 or 0.1.1 are read, and their answers are the same.

**Answers that change:** none.

**What changed:**

* **On PyPI.** ``pip install idtrack`` installs this version (``pip install "idtrack[harmonize]"`` with the
  harmonization extra); PyPI's earlier release, 0.0.5, had another API.
* **The documentation** is reorganized: a short :doc:`usage`; pages of their own for the :doc:`cli`, for
  :doc:`concepts` (the graph, history travel, bridges, ranking, ties, statuses and reasons, and a table that maps the
  statuses of a conversion, the bins of :py:meth:`idtrack.API.classify` and the harmonizer's statuses onto each
  other), :doc:`reading`, :doc:`limitations`, :doc:`organisms`, :doc:`databases`, :doc:`harmonization` and
  :doc:`performance`; the :doc:`reference/index` one page per module, after an overview of the public API; the
  tutorials in the order a reader needs them (the externals configuration after the API). The tutorials' numbers of
  time, size and memory are those of this version.
* **Messages and docstrings.** The texts of some messages, log lines and descriptions
  (:py:data:`idtrack.results.MATCH_PREFIXES`, :py:data:`idtrack.results.READING_REASONS`) say their rule in words
  where they cited a number of the maintainers' records; codes, statuses and reasons are unchanged.
* ``idtrack --help`` prints the version on a line of its own.


0.1.1
-----

**What you must do:** nothing. Conversions, graphs and file formats are those of 0.1.0: a graph built with 0.1.0 is
read by 0.1.1 and one built with 0.1.1 by 0.1.0 (the loader checks the graph format, the organism table and the
selection of external databases, not the package version; a graph's record names the version that built it,
``api.describe().package_version``).

**What changed:**

* **The command line works** (:doc:`cli`). ``idtrack build <organism> <release> --repository <folder>`` builds a graph
  (as :py:meth:`idtrack.API.build`, making the folder), ``idtrack verify <graph file>`` checks a graph file (its
  format and record, and its content against the digest it records; with ``--repository``, also the self-tests of
  :py:class:`idtrack.checks.GraphChecks` against its tables), ``idtrack info <graph file>`` shows what it holds
  (:py:meth:`idtrack.API.describe`, ``--json``), and ``idtrack convert <graph file> <identifiers>`` converts (one line
  per identifier, or ``--json``; ``--file`` reads a list). ``idtrack`` without a command prints its help. Errors of
  idtrack exit with status 1 and one line on standard error.
* **Setting up.** ``SETUP.md`` at the root of the repository: install, configure, build and check graphs, use, test,
  release; one ``Makefile`` with working targets (``make`` lists them).
* **Documentation.** The documentation builds with every warning an error and every cross-reference to an ``idtrack``
  name checked; it is published by Read the Docs. The installation page says what Python 3.15 needs (HDF5 and a C
  compiler for h5py).


0.1.0
-----

This version changes what conversions return where 0.0.5 returned wrong genes, how graphs are stored, the layout of
the package and its whole API. Graphs and table caches of 0.0.5 are not read.

What a user of 0.0.5 must do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Build your graphs again.** Graph files of 0.0.5 (``graph_<organism>_min<a>_max<b>_narrow.pickle``) and its table
  cache (``<organism>_assembly-<key>.h5``) are not read; :py:meth:`idtrack.API.build` builds the graph from Ensembl's
  tables (downloaded again) and writes the new files beside the old ones, which you can then delete. How long that
  takes and how much memory it needs: :doc:`performance` (0.0.5 needed up to 11 hours and 65 GB for human).
* **Use Python 3.12 to 3.15** (3.9 to 3.11 are no longer supported).
* **Bring your code to the new API.** The calls of 0.0.5 and their replacements:

  .. list-table::
      :header-rows: 1
      :widths: 45 55

      * - 0.0.5
        - 0.1.0
      * - ``API(local_repository=...)``, then ``api.build_graph(...)``
        - :py:meth:`idtrack.API.build` (organism, snapshot release, local repository) or :py:meth:`idtrack.API.load`
          (a graph file); ``API(graph)`` wraps a graph in memory
      * - ``api.convert_identifier(...)``, ``strategy=``
        - :py:meth:`idtrack.API.convert`, one search: ``result.view("best")`` and ``result.view("all")``
      * - ``convert_identifier_multiple``
        - :py:meth:`idtrack.API.convert_many`
      * - the flags ``no_corresponding``, ``no_conversion``, ``ambiguous`` of a result
        - ``result.status`` and ``result.reason`` (:ref:`concept-statuses`)
      * - ``the_path`` (untyped edge lists)
        - ``convert(..., explain=True)``: ``result.path`` (:ref:`concept-explained-path`)
      * - ``classify_multiple_conversion``, ``binned_conversion_summary``, ``print_binned_conversion``
        - :py:meth:`idtrack.API.classify`, :py:meth:`idtrack.API.classification_summary`
      * - ``graph_releases``, ``list_ensembl_releases``
        - :py:meth:`idtrack.API.releases`
      * - ``valid_final_databases``, ``list_external_databases``, ``list_external_databases_by_assembly``,
          ``external_database_forms``
        - :py:meth:`idtrack.API.final_databases`, :py:meth:`idtrack.API.external_databases`,
          :py:meth:`idtrack.API.databases_by_assembly`, :py:meth:`idtrack.API.database_forms`
      * - ``infer_identifier_source`` (modes ``ensembl_release``, ``assembly_ensembl_release``)
        - :py:meth:`idtrack.API.infer_source` (``release``, ``assembly_release``) and
          :py:meth:`idtrack.API.source_inference`
      * - ``resolve_organism`` (with the network), ``list_genome_assemblies``
        - :py:meth:`idtrack.API.ensembl_organism`; the assemblies are in :py:meth:`idtrack.API.describe`
      * - ``get_database_manager``, ``API.track``
        - :py:meth:`idtrack.API.database_manager`; the engine is ``idtrack.engine.Track(graph)``
      * - the class ``DB``
        - :py:mod:`idtrack.organisms` (the organism table), :py:mod:`idtrack.ids` (the vocabulary),
          :py:mod:`idtrack.settings`
      * - ``TrackTests``, ``build_graph(..., return_test=True)``
        - :py:class:`idtrack.checks.GraphChecks` (``GraphChecks(graph, database_manager)``), checks returning
          :py:class:`idtrack.checks.CheckResult`
      * - ``Track.search_budget_*``, ``Track.non_bridging_databases``, ``TheGraph.genes_at_once_threshold``,
          ``TheGraph.cache_sizes``
        - :py:class:`idtrack.settings.SearchSettings` and :py:class:`idtrack.settings.GraphSettings`
          (``API.load(path, search=..., graph_settings=...)``); network settings with
          :py:func:`idtrack.settings.configure_ensembl`

  Arguments have one name per concept in every call: ``snapshot_release`` (was ``ignore_after``,
  ``graph_last_ensembl_release``), ``first_release`` (``ignore_before``), ``to_release``
  (``target_ensembl_release``), ``release`` (``ensembl_release``, ``ens_rel``), ``organism`` (``organism_name``),
  ``identifier``/``identifiers`` (``from_id``, ``identifier_list``, ``dataset_ids``), ``final_database``
  (``to_database``), ``allow_bridges`` and ``bridge_limit`` (``go_external``, ``jump_limit``), ``assembly``
  (``genome_assembly``), ``forms`` (``form_list``), ``database_manager`` (``db_manager``), ``progress`` (``verbose``
  of the self-tests), ``verbosity`` (``verbose_level`` of the harmonizer), ``from_release_by_dataset`` and
  ``from_database_by_dataset`` (``source_releases``, ``source_databases``), ``features`` of
  ``unify_multiple_anndatas`` (``mode``).
* **Pin the snapshot release** in scripts (``API.build("homo_sapiens", 115, local_repository)``) rather than building
  at :py:meth:`idtrack.API.ensembl_organism`'s latest release, which changes with every Ensembl release.
* **Expect other answers where 0.0.5 was wrong** (below). Results of 0.0.5 that you keep cannot be reproduced with
  this version; record the version with your results.

Answers that change, and why
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where an identifier is read:

* **The starting release.** Without ``from_release`` an identifier is read at the target release where it exists
  there, else at the latest release at which it exists (:ref:`reading-starting-release`). 0.0.5 read it at the
  *oldest* period in which it existed, so that a current human gene id without version converted to another gene in
  2,557 of 78,899 cases (``NFYA`` to ``NFYB``), and 17 % of human NCBI gene ids and 1 to 2 % of symbols returned the
  wrong gene.
* **One release decides what a string means and where it is read**, and an identifier exists at a release only on the
  assembly that serves it; a meaning the string has at other releases only is reported in ``other_readings``, not
  answered (0.0.5: human ``RNR1`` to release 90 answered NR4A2, whose synonym it is, where it was MT-RNR1's symbol).
* **A** ``from_release`` **at which the graph does not hold the identifier** is no longer ignored silently: the API
  reads the identifier where the graph holds it and says so (``reason == "read_at_another_release"``,
  ``from_release_fallback``); the engine raises :py:class:`idtrack.errors.InactiveAtReleaseError`.
* **The string is read as typed and cleaned of what a person did not mean** (:ref:`reading-cleaning`): invisible
  characters, Unicode forms, quotes and trailing punctuation are removed and the match says so (0.0.5: ``ADORA3;``
  answered another gene); a missing value (``nan``, ``NA``, ...) is never a gene (0.0.5: ``nan`` answered SCN11A, and
  the harmonizer summed a ``nan`` column into it); the graph's bookkeeping (``synonym_id::TP53``,
  ``ENSG00000141510.Void``) is refused (0.0.5: TP53BP1 and TP53); nothing is cut off a string (0.0.5 cut a dash or
  underscore suffix: ``SNORD116_10`` became ``SNORD116``, 10,506 human ids resolved to another gene).
* **Mouse symbols in one letter case** (``AR``, ``CCK``) are read first as MGI symbols without letter case where that
  names one gene (0.0.5: the exact synonym of another gene won, 145 such symbols).
* **Names made unique by single-cell tools** (``SRF.1`` beside ``SRF`` in a batch) are copies of the name they repeat,
  and alone are not read into another gene's synonym (0.0.5: ``SRF.1`` answered SPATA6).
* **Versions.** A version of an Ensembl id that the graph does not hold is ``unknown_version`` with the versions there
  are (0.0.5 answered the gene of the id without version); a version newer than the graph is read as the latest. An
  accession's version is read where Ensembl attached it, and UniProt's entry versions, which count edits of the
  record, are not judged (:ref:`reading-accession-versions`). Transcript and translation ids without version are found (none
  was in 0.0.5).

How the answer is chosen:

* **Ties are ties, in a stated order.** Candidates equal in all evidence are all the best answer (``tie``), in
  ascending order of gene and target id; 0.0.5 gave them in an order that changed with Python's hash seed, so the
  first target could be another gene in the next run (:ref:`concept-ties`).
* **A target in a database is an id of the gene answered.** Where no gene of the best answer has an id in
  ``final_database``, the result is those genes (``no_target``), never a lower-ranked gene's id (0.0.5: pig ``PSME1``
  to ``VGNC Symbol`` answered ``DCAF11``, the symbol of another gene; 648 such lines over three releases per
  organism, and the harmonizer moved the feature onto that gene).
* **Ensembl's display name of a gene ranks before the number of databases** that name it (:ref:`concept-ranking`);
  genes that numbering schemes attach one number to tie rather than being ranked by how many databases attach it;
  Ensembl's score 0 is no score (0.0.5 counted it as 1, the best).
* **Copies of genes** on alternative loci, patches and LRG records are set apart (``alternative_locus_genes``) when a
  gene of the reference sequence is reached, and every id of an LRG record answers the gene it describes; a queried
  gene id that exists at the target release is its own answer; the X copy of a pseudoautosomal gene answers its name,
  the Y copy set apart (0.0.5: a tie).

How the history is followed:

* **Ensembl's id history is checked against the ids of each release** when a graph is built; a session recorded with
  wrong releases is placed where its ids change, and one that fits no pair of releases stops the build (0.0.5:
  1,123 pig gene ids converted to ids that do not exist at the target).
* **A step between the mitochondrial genome and a nuclear sequence is not followed** (rhesus ``MT-ND1`` of release 97
  answered an unnamed gene of chromosome 8; it now answers the mitochondrial ``ND1``).
* **A bridge is evidence of a lineage only where an id leaves one gene for another at the same place**
  (:ref:`concept-bridges`): never through a synonym, a family label or an id that several genes carry (0.0.5: a
  retired rhesus id answered three SNORD59 genes on other chromosomes through a shared label).
* **The search is bounded**: every state is visited once and a call has a budget (0.0.5 enumerated paths: some
  conversions ran for more than 40 minutes, and ``strategy="best"`` raised ``unexpected!2`` for 2.7 % of mouse gene
  ids).
* **A conversion to a release of an earlier assembly returns that assembly's ids** (0.0.5: none, e.g. 0 of 2,981
  mouse symbols at release 102), and ids are returned as their database wrote them at the target release.

What the graph holds of external databases (:doc:`databases`):

* **One database name means one kind of identifier**: the labels of a database whose labels and accessions Ensembl
  names alike take a name of their own (NCBI's symbols are ``NCBI gene (formerly Entrezgene)``, so converting to
  ``EntrezGene`` returns NCBI's gene numbers, not numbers and symbols as a tie).
* **A selected database is read under every name Ensembl filed it under** (RefSeq's accessions were ``RefSeq_dna`` up
  to release 63): 0.0.5 read only today's names, so identifiers of earlier releases were read at their latest release
  and carried back to another gene, or not found.
* **Every gene's display name at every release is a name of the gene**, whatever the selection (0.0.5: every pig name
  of releases 56 to 60 and every rhesus name of 48 to 60 was missing).
* **An Rfam label names only the gene whose display name it is** (``5S_rRNA`` named 531 human genes; Ensembl names
  10 with it), and an Rfam accession is never a bridge; miRBase names are on miRNA genes only, and an entry on genes
  at several loci is left out.

Lists and datasets:

* :py:meth:`idtrack.API.source_inference` (new) and :py:meth:`idtrack.API.infer_source` count each distinct identifier
  once, read as it was at each release (0.0.5 read every name as at the last release: 23 of 32 symbol tables of
  published single-cell studies were placed at 115), and answer the band of origins the list cannot tell apart.
* **The harmonizer** (:doc:`harmonization`): the final-database column never holds an Ensembl id (0.0.5: 58 % of pig
  rows converted to VGNC symbols had one there, unflagged); each dataset is converted on its own (0.0.5 deleted an id
  that was n-to-1 in one dataset from every dataset: 12 % of the counts of two pig datasets were lost); its result
  cache is keyed by the graph's content and :py:data:`idtrack.settings.RULES_VERSION`, not by the package version.

New
~~~

* **Typed results** (:py:class:`idtrack.results.ConversionResult`, frozen dataclasses with strict JSON ``to_dict()``
  and ``from_dict``): every candidate with its ranking record, ``status``, ``reasons``, ``match``, ``from_release``
  and whether it was inferred, ``other_readings``, ``alternative_locus_genes``, ``no_target_genes``,
  ``known_versions``, ``error``; ``decided_by`` names the criterion between the best answer and the runner-up.
* **The explained path** as data (``explain=True``): nodes, steps, every bridge considered with its rule, refused steps.
* **Batches** that never stop on one identifier's error, check every item first, convert each distinct identifier
  once, and accept any sequence of strings.
* ``from_database`` (read an identifier as an id of one database), :py:meth:`idtrack.API.genes_named` and
  :py:meth:`idtrack.API.names_of` (names per database and release), :py:meth:`idtrack.API.describe` (what a graph
  holds, offline), :py:meth:`idtrack.API.reason_text`.
* **One error hierarchy** (:py:mod:`idtrack.errors`): every error is an :py:class:`idtrack.errors.IdtrackError`; a
  caller's mistake is an :py:class:`idtrack.errors.InputError` naming the argument and the valid values (also a
  ``ValueError``, or a ``TypeError`` for a wrong type).
* **Graph files** are one ``.npz`` archive of arrays and JSON (no pickle: loading executes no code), named
  ``graph_<organism>_assembly<key>_min<first>_max<snapshot>.npz``, with one record of what the graph is and what built
  it, and a content digest; a graph is deterministic (built twice, the same digest). Loading needs no table and no
  network (0.0.5: 206 s and 35 GB for human). A file that holds another graph than the one asked for raises
  :py:class:`idtrack.errors.GraphMismatchError` (0.0.5 loaded it silently). The graph holds every gene version's place.
* **Tables** are stored one file per table, written atomically, named by what they depend on; a read-only repository
  serves what it holds.
* **Access to Ensembl** (:ref:`ensembl-access`): ``IDTRACK_ENSEMBL_ACCESS`` (``auto`` or ``https``), the same tables
  either way; retries with timeouts; a failed request is an error, never read as "absent" (0.0.5 dropped such releases
  silently); the HTTPS dumps are decoded as MySQL loads them (0.0.5 split or dropped rows with escaped tabs and
  newlines).
* **Organisms**: rhesus and cynomolgus macaque, beside human, mouse and pig, declared in one table
  (:py:data:`idtrack.organisms.ORGANISM_TABLE`; :doc:`organisms`) with short names (``"human"``, ``"pig"``, ...); the
  shipped externals configurations list Ensembl releases up to 116.
* **Harmonization**: ``merge="sum"`` or ``"flag"``, ``keep_unconverted``, ``from_release_by_dataset``,
  ``from_database_by_dataset``, ``api``; ``conversion_report``, ``feature_table``, ``output_plan``; every ``var`` and
  ``obs`` column kept by :py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas`, ``X`` keeping its
  dtype (`#85 <https://github.com/theislab/idtrack/issues/85>`_). Harmonization is the ``harmonize`` extra.
* **pandas 3** is supported, with pandas 2.3 (`#86 <https://github.com/theislab/idtrack/issues/86>`_); Ensembl REST
  errors name the URL and the timeouts (`#84 <https://github.com/theislab/idtrack/issues/84>`_); a graph of an older
  snapshot release is built with ``API.build(organism, <release>, local_repository)``
  (`#71 <https://github.com/theislab/idtrack/issues/71>`_).
* **Dependencies**: pandas 2.3, numpy 2.1, h5py 3.11, networkx 3.2, PyYAML 6.0.2, PyMySQL 1.1.1, tqdm 4.66.3,
  requests 2.33.0, urllib3 2.8.0 at least; anndata 0.13 and scipy 1.14 with the ``harmonize`` extra
  (``pyproject.toml`` gives the reason of each floor).
* The version is read from the installed package's metadata (``idtrack.__version__``); the package is marked
  Production/Stable.

Removed
~~~~~~~

* The external-database mappers (the wrappers of MyGene, g:Profiler, gget, BioMart and the ortholog alignment, with
  their extras ``external-mappers``, ``ortholog``, ``embeddings`` and ``all-external``), the connection bridge
  (``ConnectionBridge`` and PySocks: graphs build over Ensembl's HTTPS dumps where MySQL is blocked), and the folder
  ``reproducibility/`` (the manuscript's notebooks and scripts are at the tag ``manuscript-2026-03``).
* Of the API: ``build_graph``'s ``calculate_caches`` and ``assembly`` (a graph is built on the organism's assembly
  order), ``calculate_graph_caches``. Of the engine ``Track``: ``deprioritize_lrg_genes``, ``reduction``,
  ``remove_na``, ``score_of_the_queried_item``, ``version_info``, ``get_possible_paths``, ``calculate_node_scores``,
  ``from_graph``; ``Track`` is no longer importable from ``idtrack`` itself.
* Of ``TheGraph``: ``combined_edges``, ``combined_edges_genes``, ``combined_edges_assembly_specific_genes``,
  ``get_active_ranges_of_id``, ``node_trios``, ``lower_chars_graph``, ``node_name_alternatives`` (use
  :py:meth:`idtrack.graph.TheGraph.combined`, :py:meth:`idtrack.graph.TheGraph.active_ranges`,
  :py:meth:`idtrack.graph.TheGraph.trios`, :py:meth:`idtrack.graph.TheGraph.resolve_name`), ``is_latest``,
  ``is_frozen`` and the other members nothing but the tests used.
* Of ``DatabaseManager``: the version-info and relation-archive tables and their functions, ``tables_in_disk``,
  ``file_name``, ``available_tables_mysql``, ``get_release_date``, and the table cache functions of 0.0.5
  (``export_disk``, ``read_exported``, ``repack_hdf5``); ``GraphMaker.initialize_downloads`` and
  ``update_graph_with_the_new_release``.
* Of the harmonizer: ``get_idtrack_matchings_for_all_datasets``, ``dict_n_to_1``, ``unified_matching_dict``,
  ``multiple_ensembl_dict``, the ``removed_*``/``kept_*`` diagnostics, ``conversion_failed_identifiers``,
  ``debugging_variables``.
* The fields ``Database Index`` and ``Potential Synonymous`` of the externals YAML files (never read; files that
  still have them load as before).
