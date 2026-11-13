Changes
=======

This page lists the changes that affect users, newest first. Version numbers follow ``pyproject.toml``.


Unreleased
----------

This version changes what conversions return where 0.0.5 returned wrong genes, how graphs are stored, and the
arguments and results of the API. Graphs and table caches of 0.0.5 are not read.

**What a user of 0.0.5 must do**

* **Build your graphs again.** Graph files of 0.0.5 (``graph_<organism>_min<a>_max<b>_narrow.pickle``) and its table
  cache (``<organism>_assembly-<key>.h5``) are not read; :py:meth:`idtrack.API.build_graph` builds the graph from
  Ensembl's tables (downloaded again) and writes the new files beside the old ones, which you can then delete. A build
  takes from 6 minutes (cynomolgus) to 2 hours (human) and at most 4.5 GB of memory (previously up to 11 hours and
  65 GB for human); see "Time and Memory" in :doc:`usage`.
* Remove the arguments and attributes listed under "Removed" below from your code: ``calculate_caches`` and
  ``genome_assembly`` of ``API.build_graph``, ``API.calculate_graph_caches``, ``deprioritize_lrg_genes`` and three
  scoring arguments of ``Track.convert``, the diagnostic attributes of ``HarmonizeFeatures``.
* Use Python 3.12 to 3.15.
* Pin the snapshot release in scripts (``build_graph(..., snapshot_release=115)``) rather than building at
  :py:meth:`idtrack.API.resolve_organism`'s latest release, which changes with every Ensembl release.
* Expect other answers where 0.0.5 was wrong (below). Results of 0.0.5 that you keep cannot be reproduced with this
  version; record the version with your results.

**Conversions: what an identifier means**

* **The starting release.** With ``from_release`` the identifier is read at that release. Without it, it is read at
  the target release when it exists there, and otherwise at the latest release at which it exists. Previously an id
  without ``from_release`` was read at the *oldest* period in which it existed, so that, for example, a current human gene id
  without version converted to another gene in 2,557 of 78,899 cases (``NFYA`` to ``NFYB``), and 17 % of human NCBI
  gene ids and 1 to 2 % of symbols returned the wrong gene. The rule is the same for Ensembl ids, symbols and other
  external ids, and the result says which release was used (``from_release``, ``from_release_inferred``).
* **A** ``from_release`` **at which the graph does not hold the identifier.** :py:meth:`idtrack.Track.convert` raises
  :py:class:`idtrack.InactiveAtReleaseError`, which names the releases at which the identifier exists (previously the
  answer came silently, as if no release had been given). :py:meth:`idtrack.API.convert_identifier` reads the
  identifier where the graph holds it instead and flags the result (``from_release_fallback``, ``from_release_given``).
* **What a string means.** When a string can mean several nodes (a symbol that was the name of one gene and is a
  synonym of another today), the one that exists at the release asked about is taken (``from_release``, without it the
  snapshot release); an Ensembl stable id is never passed over for a synonym spelt the same. An identifier exists at a
  release only on the genome assembly that Ensembl maintained at that release: the names that human GRCh37, listed
  beside GRCh38 with its annotation of release 75, still carries do not exist today (3,277 of them are a synonym of
  a gene on GRCh38 today; read as the old name, 638 returned other genes than the synonym's). Then the string is
  matched
  exactly, then ignoring letter case, then taking ``-``, ``_`` and ``.`` as one character, then (only for ids of
  sequence-accession databases such as RefSeq and UniProt) without a version the graph does not hold; each step on
  names and then on synonyms. A name typed as a database wrote it at the release it is read at is an exact match of
  that name, also where the database writes it in another letter case today or two databases write it in two ways
  (at release 75 MGI wrote the mouse gene ``Bap1`` and UniProt wrote it ``BAP1``), and is not taken for another
  gene's synonym spelt the same.
  The new result field ``match`` says how it matched. The engine reads a string by the same
  rule (:py:meth:`idtrack.TheGraph.resolve_name`): ``Track.convert("X")`` no longer takes a string that happens to be
  the name of a node as that node; to convert a particular node whatever the release, give a ``from_release`` at which
  it exists. Previously a dash-or-underscore suffix was cut off any string (``SNORD116_10`` became ``SNORD116``:
  10,506 human ids resolved to another gene), the least-connected spelling of an id written in several cases was kept,
  and every combination of separators was tried (minutes for a string with 20 of them); the lookup now costs
  microseconds whatever the string. Surrounding white space is ignored.
* **Versions.** An Ensembl id with a version that the graph does not hold is not answered with another version's gene:
  the result has no target, and ``known_versions`` lists the versions that exist (the engine raises
  :py:class:`idtrack.UnknownVersionError`). A version higher than all the graph holds, of an id still active at the
  snapshot, may come from a newer release: it is read as the latest version and reported as
  ``match == "version_newer_than_graph"``. Previously any unknown version was answered with the gene of the id without
  version. Transcript and translation ids without version are now found (previously none was).
* **Ties.** ``strategy="best"`` returns one target, or, when several genes or targets are equal in all evidence, all
  of them, with ``tie`` true, ordered by the mapping score of each to the gene the identifier was read as and then by
  gene and target id. Previously several targets could come in an
  order that changed between runs (the hash seed), so ``target_id[0]`` could be another gene in the next run. Every list of
  a result is now in a defined order, and two runs give the same answer. A tie is judged among the genes that have an
  id in the requested database (the others are named in ``no_target_genes``).
* **Every answer, best first.** ``strategy="all"`` returns the genes of the best answer first, in the order in which
  ``strategy="best"`` returns them, then the others by their evidence: ``target_id[0]`` is the first target of the
  best answer with either strategy.
* **Copies on alternative loci, patches and LRG records** (human; mouse GRCm38) are marked in the graph from Ensembl's
  sequence regions. When an identifier leads to a gene of the reference sequence, the copies are left out of the
  answer and listed in the new result field ``alternative_locus_genes``; when it leads to copies only, they are the
  answer. :py:meth:`idtrack.TheGraph.is_non_reference` and :py:meth:`idtrack.TheGraph.non_reference_releases` tell
  where a gene is such a copy. (Previously LRG records were left out by their id prefix, without saying so.)
* **A gene id is its own answer.** A queried Ensembl gene id that exists at the target release is answered with
  itself: its own version comes first in the result, is never set apart as a copy, and is the answer of
  ``strategy="best"`` whatever else its history reaches; ``strategy="all"`` lists the other genes after it.
* **A database named for the identifier.** ``from_database`` of :py:meth:`idtrack.API.convert_identifier` (and of
  :py:meth:`idtrack.Track.convert`) reads the identifier as an id of that database: only the genes that database
  attaches it to are entered, and between them the evidence of every database still counts. A symbol that two
  databases give to two genes is a tie without it and that database's gene with it; a symbol that the named database
  gives to two genes, of which other databases confirm one, is that one.
* **The databases behind an identifier.** The result field ``from_databases`` names every database that attaches the
  identifier to its genes at the release it was read at, whether to a gene itself or to one of its transcripts or
  translations, and the support of a gene (``entry_support`` of :py:meth:`idtrack.Track.convert`) counts all of them.
* **History.** Ensembl's id history (``mapping_session``) is checked against the ids of each release when a graph is
  built; a session recorded with wrong releases is placed where its ids change (pig 106 to "104" is 106 to 107; human
  114 to "114" is 114 to 115), each correction is recorded in the graph file and logged, and a session that fits no
  pair of releases stops the build (:py:class:`idtrack.HistoryValidationError`). Previously 1,123 pig gene ids converted
  to ids that do not exist at the target. The human graph now has the id history before release 76, which the
  previous table cache could omit.
* **Search.** The search follows Ensembl's history first; only when that leads to no gene does it pass through an
  external id or transcript that two genes carried (at most two such bridges, the fewest possible, in the order of
  time, never through clone-based names or through ids that more than five genes carry at once). It is bounded: every
  state is visited once, and a call has a budget (:py:attr:`idtrack.Track.search_budget_states`,
  :py:attr:`idtrack.Track.search_budget_seconds`) past which it raises :py:class:`idtrack.SearchBudgetExceededError`.
  The previous search enumerated paths: some conversions ran for more than 40 minutes, and ``strategy="best"`` raised
  ``unexpected!2`` for 2.7 % of mouse gene ids.
* **Assemblies.** A conversion uses, at each release, the assembly Ensembl maintained at that release (human GRCh38
  from 76; the parallel GRCh37 is not walked, its ids are converted from). A conversion to a release of an earlier
  assembly returns that assembly's ids (previously none, e.g. 0 of 2,981 mouse symbols at release 102), and an id is
  returned as the database wrote it at the target release (``IFN-omega-4`` for UniProt, ``IFN-OMEGA-4`` for NCBI). A
  graph's external ids are named after the spelling of the latest release in which they exist (``PKD1``, not
  ``pkd1``).
* **Archive copies.** Ensembl's archive server (MySQL port 3337) carries older assemblies forward under later release
  numbers (rhesus MMUL_1, pig Sscrofa10.2 for releases 90 to 99, mouse GRCm38); these frozen copies are not part of any
  graph, so a graph built over MySQL equals one built over HTTPS. Port 3337 is read for human GRCh37 only.

**Results of the API**

* Results are plain data: ``json.dumps`` works on every result (previously NumPy booleans and tuple-keyed paths).
* New fields of :py:meth:`idtrack.API.convert_identifier`: ``match``, ``known_versions``, ``from_release`` (the
  release used), ``from_release_inferred``, ``from_release_given``, ``from_release_fallback``, ``from_databases``,
  ``to_release``, ``strategy``, ``ambiguous``, ``tie``, ``alternative_locus_genes``, ``no_target_genes`` (genes reached
  that have no id in the requested database while others have) and ``error``. ``last_node`` holds lists
  ``[gene, target]`` (previously tuples). With ``explain=True``, ``the_path`` is a list with one entry per pair of
  ``last_node``: the gene the identifier was read as, the number of bridges, the product of Ensembl's mapping scores and
  the edges walked.
* Every argument is checked before the search: the identifier (a non-empty string), the releases (integers of the
  graph: :py:meth:`idtrack.API.graph_releases`; previously releases the graph does not hold were answered), the final
  database (:py:meth:`idtrack.API.valid_final_databases`, case ignored; previously checked only when the identifier
  was found) and the strategy. An invalid argument raises :py:class:`TypeError` or :py:class:`ValueError` naming it.
* :py:meth:`idtrack.API.convert_identifier_multiple` checks all items first (positions named), converts each distinct
  identifier once, and records a search over its budget as ``error`` on that item instead of losing the batch.
* :py:meth:`idtrack.API.list_ensembl_releases` returns the graph's releases (pig 56 to 115; previously the primary
  assembly's releases only, 90 to 115); the ``list_*`` methods return new sorted lists.
* :py:meth:`idtrack.API.infer_identifier_source` returns a list: every origin tied for the highest count, or all
  origins with their counts; it raises :py:class:`ValueError` when no identifier is in the graph (previously
  ``IndexError``).
* :py:meth:`idtrack.API.classify_multiple_conversion` counts a query that converts to itself (without version, in
  another case) as unchanged; previously every query without version counted as "changed".
* :py:meth:`idtrack.API.configure_logger` configures the ``"idtrack"`` logger only (previously it called
  ``logging.basicConfig`` on the root logger); every module logs under ``idtrack.<module>``.
* ``import idtrack`` imports nothing heavy; ``idtrack.API`` does not import anndata or scipy.
* Errors a caller may want to catch are exported: :py:class:`idtrack.InactiveAtReleaseError`,
  :py:class:`idtrack.UnknownVersionError`, :py:class:`idtrack.SearchBudgetExceededError`,
  :py:class:`idtrack.GraphMismatchError`, :py:class:`idtrack.HistoryValidationError`,
  :py:class:`idtrack.EnsemblRestError`, :py:class:`idtrack.EnsemblDownloadError`.

**Graph files**

* A graph is one ``.npz`` file of arrays and JSON, named
  ``graph_<organism>_assembly<key>_min<first>_max<snapshot>_narrow.npz``; it holds no pickle, so loading it executes
  no code. It records what it is (format version, package version, organism, primary assembly, window, snapshot,
  forms, the selection of external databases that takes effect, the organism table, the tables it was built from, the
  history corrections, build date, content digest: ``graph.graph["manifest"]``). A file that holds another graph than
  the one asked for raises :py:class:`idtrack.GraphMismatchError` naming the differences and how to build the graph again (previously it was loaded silently, so
  a graph of another assembly or of another externals YAML answered for this one).
* :py:meth:`idtrack.API.build_graph` loads a graph file with no table, no :py:class:`idtrack.DatabaseManager` and no
  network (:py:meth:`idtrack.GraphMaker.load_graph`, :py:meth:`idtrack.Track.from_graph`). Loading takes 1 to 5 s and
  holds 0.26 to 1.14 GB (human: 4.3 s and 1.14 GB; previously 206 s and 35 GB, 50 GB with the caches); nothing is
  computed in advance.
* A graph is deterministic: built twice from the same tables, under different hash seeds, it has the same content
  digest (``graph.graph["manifest"]``).
* Tables are stored one file per table (``idtrack_tables/<organism>_assembly-<key>/ens<release>_<table>.h5``), each
  written to a temporary file and renamed, so an interrupted run leaves every stored table readable (previously one HDF5
  file per assembly, which a killed writer could leave unopenable). A table's file name and content record what it
  depends on (the release window of the id history, the databases the externals YAML enables): changing the YAML or
  the window no longer serves a stale table, and nothing needs to be deleted by hand. A read-only repository serves
  the tables it holds.

**Organisms**

* Supported: human, mouse, pig, rhesus macaque (``macaca_mulatta``; assemblies Mmul_10, Mmul_8.0.1 and MMUL_1, keys
  ``10``, ``801`` and ``1``) and cynomolgus macaque (``macaca_fascicularis``; Macaca_fascicularis_6.0 and 5.0, keys
  ``6`` and ``5``). Ensembl names the MMUL_1 and Mmul_10 core databases alike (``macaca_mulatta_core_<release>_10``);
  MMUL_1 therefore has the key ``1``. See the Supported Organisms table in :doc:`usage`.
* Organisms are declared in one table, ``DB.assembly_mysqlport_priority``; ``DB.register_organism`` adds or replaces
  one for the running session (see "Adding Organism Support" in :doc:`contributing`). Each organism declares short
  names (``"human"``, ``"mouse"``, ``"pig"``, ``"rhesus"``, ``"cynomolgus"``, ``DB.organism_aliases``) that resolve
  without the network and win over Ensembl's other species of that name (``"pig"`` is also the common name of
  every pig breed in Ensembl). Assembly names (``"GRCm38"``) resolve to their organism.
* Organisms whose Ensembl ids carry no versions are not supported: building a graph for one raises a
  :py:class:`ValueError` that says so.
* A graph's window starts at the organism's first release in the table (``DB.first_release``: 48 for human, mouse and
  rhesus, 56 for pig, 91 for cynomolgus), for :py:class:`idtrack.DatabaseManager` and the API alike (previously 48 in the
  API, the selected assembly's first release in the manager).
* ``API.build_graph`` builds on the organism's assembly order; the primary assembly is the preferred one that has the
  snapshot release.
* The shipped external-database files of all five organisms include Ensembl release 116.
  ``ExternalDatabases.extend_modified_yaml`` extends a file to new releases (keeping a ``.bak`` of the file it
  replaces), and a graph build warns once about releases its file does not cover. ``Include`` must be a YAML boolean;
  entries for assemblies or releases that the organism table does not give the organism are left out with a warning
  (an error in a shipped file).

**Harmonization** (:py:class:`idtrack.HarmonizeFeatures`)

* Each converted gene is one output feature, named by its Ensembl gene id at the target release; its name in the final
  database is a column (the first in sorted order, the others in ``<final database>_alternatives``). A gene without a
  name there keeps a missing value and status ``converted_no_final_id``: the final-database column never holds an
  Ensembl id (previously 58 % of pig rows converted to VGNC symbols had an Ensembl id there, unflagged).
* Nothing is dropped silently: every output feature has ``var["idtrack_status"]`` and ``var["idtrack_reason"]``
  (statuses in ``idtrack._harmonize_features.STATUSES``). A dataset is converted on its own: an id that is n-to-1 in
  one dataset is no longer deleted from every dataset (previously 12 % of the counts of two pig datasets were lost this way).
  A feature that is a gene id existing at the target release is converted to itself (see "A gene id is its own answer"),
  not flagged ambiguous between its own later version and other genes its history reaches.
* New arguments: ``merge="sum"`` adds up the input features of one gene (``"flag"``, the default, keeps the feature
  that names the gene and flags the others), ``keep_unconverted``, ``source_releases`` (the release each dataset was
  annotated with), ``source_databases`` (the database each dataset's feature names are ids of), ``api`` (an
  :py:class:`idtrack.API` whose graph is loaded).
* New methods: ``feature_table``, ``output_plan``, ``conversion_report``; the attributes ``matchings``,
  ``feature_names``, ``gene_names``. Removed: ``get_idtrack_matchings_for_all_datasets``, ``dict_n_to_1``,
  ``unified_matching_dict``, ``multiple_ensembl_dict``, the ``removed_*``/``kept_*`` diagnostics,
  ``conversion_failed_identifiers``, ``debugging_variables``.
* The result cache is one JSON file per dataset (``idtrack_result_<project>_<dataset>.json``), written atomically, with
  a header of everything it depends on (idtrack version, organism, final database, releases, source database, graph);
  anything else is recomputed. Caches of 0.0.5 are recomputed once.
* ``organism_name`` is read from the graphs in the local repository when not given; ``graph_last_ensembl_release``
  defaults to the newest local graph of the organism; ``final_database`` defaults to the organism's first symbol
  database (``DB.symbol_databases``: HGNC Symbol for human, MGI Symbol for mouse, VGNC Symbol for pig and the
  macaques). In 0.0.5 they defaulted to human and release 114.
* :py:meth:`idtrack.HarmonizeFeatures.unify_multiple_anndatas` loses no ``obs`` column by default (`#85
  <https://github.com/theislab/idtrack/issues/85>`_): ``obs_join="outer"`` keeps every column of any dataset, missing
  (as a missing value of the column's kind; integers and booleans become nullable) in the datasets without it;
  ``obs_join="inner"`` keeps the columns every dataset has. ``obs_columns_to_keep`` selects among the joined columns.
  Every input ``var`` column is kept (one column where the datasets agree on each feature, otherwise
  ``<column>_<handle>``). Layers, ``obsm``, ``obsp`` and shared ``uns`` entries are carried over, ``X`` keeps its dtype,
  ``raw`` is dropped with a warning. Values are no longer rewritten (no placeholder ``_PLACEHOLDER_NA_``; text columns
  become categorical). ``feature_harmonizer`` returns the whole dataset re-indexed by the harmonised features.
* Feature names are read from the files without loading them, checked (empty, not strings, repeated) before any
  conversion; ``mode``, ``obs_join`` and ``merge`` are validated before any work.

**Access to Ensembl**

* Access modes, set with the environment variable ``IDTRACK_ENSEMBL_ACCESS`` or ``DB.ensembl_access``: ``auto``
  (default: MySQL where its ports answer, the HTTPS dumps otherwise) and ``https`` (MySQL is never contacted). Both give
  the same tables, with the same values and column types. A MySQL port that fails is tried again after ten minutes;
  MySQL query errors are raised. See :ref:`ensembl-access`.
* A failed request is never read as "absent": a release that cannot be listed, or a table that cannot be downloaded
  after the retries, raises (:py:class:`idtrack.EnsemblDownloadError`); previously such releases were dropped silently, and one
  failed request could select another primary assembly.
* Ensembl REST requests (organism names, the latest release) are retried on timeouts, connection and SSL errors and
  the HTTP statuses 429 and 5xx (``DB.rest_attempts``, default 3), with the timeouts ``DB.connection_timeout`` (15 s)
  and ``DB.reading_timeout`` (60 s; also used by the HTTPS dumps). After the last attempt
  :py:class:`idtrack.EnsemblRestError` (a ``ConnectionError``) names the URL, the timeouts and how to raise them
  (`#84 <https://github.com/theislab/idtrack/issues/84>`_).
* :py:class:`idtrack.VerifyOrganism` asks Ensembl REST for the species list at most once per process and only when a
  look-up needs it; ``fetch_organism_and_latest_release`` takes no timeout arguments.
* Tables read from the HTTPS dumps are decoded as MySQL loads them (escaped tabs, newlines and backslashes), so such
  rows are no longer split or dropped. Building a pig graph from nothing takes 15 minutes (previously 73 minutes) and
  0.65 GB of disk (previously 6.9 GB).

**pandas 3** (`#86 <https://github.com/theislab/idtrack/issues/86>`_)

* idtrack works with pandas 2.3 and 3.x. 0.0.5 could not store a table with a text column under pandas 3, and
  ``DatabaseManager.create_ids`` then reported the misleading ``Table 'gene_stable_id' not found``. Only a schema answer
  (``idtrack._db.EnsemblSchemaError``: the release has no such table or column) leads to the older-schema fallbacks;
  any other error is raised as it is. Text comes back in the running pandas' default text dtype; tables written under
  one pandas version are read under the other.

**Dependencies**

* idtrack supports Python 3.12 to 3.15; Python 3.9, 3.10 and 3.11 are no longer supported (on Python 3.11 no anndata
  release writes ``.h5ad`` files with pandas 3).
* Minimum versions are raised to releases that install on Python 3.12 and pass the test suite together: pandas 2.3
  (and 3.x), anndata 0.13, numpy 2.1, scipy 1.14, h5py 3.11, networkx 3.2, PyYAML 6.0.2, PyMySQL 1.1.1,
  tqdm 4.66.3, PySocks 1.7.1; optional: biopython 1.81, torch 2.2, transformers 4.36. ``sparse`` is no longer a
  dependency. See ``pyproject.toml`` for the reason of each.
* idtrack requires ``requests>=2.33.0`` and ``urllib3>=2.8.0``, the first releases without the vulnerabilities
  CVE-2026-25645 (requests) and CVE-2026-44431, -44432, -97687, -97688 and -97689 (urllib3).

**Older snapshots** (`#71 <https://github.com/theislab/idtrack/issues/71>`_)

* ``GraphMaker`` explains, when a manager is not at the newest release of its window, how to build a graph whose
  snapshot is an older release (``API.build_graph(..., snapshot_release=<release>)``); see "Working with an older
  Ensembl release" in :doc:`usage`.

**Removed**

* ``API.build_graph``: the arguments ``calculate_caches`` (nothing is computed in advance) and ``genome_assembly``
  (a graph is built on the organism's assembly order; a graph with another primary assembly was never built
  correctly). ``API.calculate_graph_caches``.
* ``Track.convert``: ``deprioritize_lrg_genes`` (LRG records follow the rule for copies above), ``reduction``,
  ``remove_na``, ``score_of_the_queried_item`` (the score of a path is the product of Ensembl's mapping scores).
  ``Track.version_info``, the graph attribute ``version_info``; ``EmptyConversionMetricsError`` (nothing raises it).
* ``TheGraph``: ``combined_edges``, ``combined_edges_genes``, ``combined_edges_assembly_specific_genes``,
  ``get_active_ranges_of_id``, ``node_trios``, ``lower_chars_graph`` and ``node_name_alternatives`` (use
  :py:meth:`idtrack.TheGraph.combined`, :py:meth:`idtrack.TheGraph.active_ranges`, :py:meth:`idtrack.TheGraph.trios`,
  :py:meth:`idtrack.TheGraph.resolve_name`); the node attribute ``is_latest``; ``GraphMaker.get_graph``'s
  ``overwrite_even_if_exist`` (a rebuild always replaces the file).
* ``DatabaseManager``: ``check_version_info``, ``version_fix``, ``version_fix_incomplete``, ``version_uniformize``,
  ``create_version_info``, ``get_db("versioninfo")``, ``available_releases_no_save``, ``tables_in_disk``,
  ``file_name``, ``create_id_history_fixed``, ``create_relation_archive`` and ``get_db("relationarchive")``,
  ``available_tables_mysql``, ``get_release_date``; ``GraphMaker.initialize_downloads`` and
  ``GraphMaker.update_graph_with_the_new_release`` (they only raised ``NotImplementedError``).
* ``DB``: ``main_assembly`` (use ``DB.assemblies_by_priority(organism)[0]``), ``first_version``,
  ``hyperconnecting_threshold`` (the threshold is ``TheGraph.genes_at_once_threshold``), ``placeholder_na``,
  ``MISSING_VALUES``, ``conn_dict_str_ensembl_base``, ``nts_base_ensembl_reverse``.
* The table cache functions of ``idtrack._utils_hdf5`` that wrote one file per assembly (``export_disk``,
  ``read_exported``, ``repack_hdf5``).

**Other changes**

* ``Track`` raises a ``ValueError`` for a graph whose organism is not in the organism table (previously the
  assemblies were ordered by their numeric code); register the organism first.
* Without an explicit assembly, ``DatabaseManager`` picks the highest-priority assembly that has the release it is
  made for, inside ``ignore_before``/``ignore_after`` (e.g. a mouse snapshot at release 100 uses GRCm38).
* ``DatabaseManager.create_available_databases`` lists the core databases of the organism's configured assemblies
  within their release ranges, the same in both access modes; ``DatabaseManager.get_db("generegion")`` tells which
  genes of a release lie outside the reference sequence.
* ``idtrack._external_mappers.align_ortholog_pair_with_features`` computes its alignment features again (it returned
  ``{}`` for every ortholog, because ``gget.muscle`` writes the alignment to a file rather than returning it). The
  alignment of an ortholog is in the key ``alignment_aligned_fasta`` (aligned FASTA as MUSCLE writes it; formerly
  ``alignment_raw_clustal``).
* The self-tests of a graph (:py:class:`idtrack.TrackTests`, ``API.build_graph(..., return_test=True)``): every check
  (``is_...``) returns a :py:class:`idtrack._track_tests.CheckResult` with ``holds`` (also its truth value) and
  ``details`` (what was compared and every violation found), where it returned ``True``, ``False`` or a tuple. A
  check does not raise for what it finds: a table that cannot be read or a release without identifiers is a result
  that does not hold, with the error in its details. A database that the graph does not hold is refused before
  anything is converted, by the checks and by ``history_travel_testing``: a :py:class:`ValueError` that names the
  argument and lists the valid databases, as the API raises for a wrong argument (previously
  ``history_travel_testing`` returned a report in which every identifier was lost or failed, and the name of a
  form was the only name it refused). The checks compare assembly by assembly at every release an assembly has
  (previously they raised at the first release of an earlier assembly, or reported every graph with more than one
  assembly as inconsistent), report identifiers that only the graph has as well as those it lacks, and count a gene
  that has an id in a database and is not converted to it. ``random_dataset_source_generator`` returns ``None`` when
  nothing can be drawn (previously the tuple ``("", -1, -1)``), and
  ``history_travel_testing_random_arguments_generator`` raises :py:class:`ValueError` after 100 attempts.
* ``ConnectionBridge`` used as ``with ConnectionBridge(...)`` raises ``ConnectionError`` when its check fails; a bridge
  stays active until ``stop()`` or ``ConnectionBridge.stop_all()``, and restores the proxy settings it found.
* The notebooks under ``reproducibility/experiments/`` are records of the analysis published with the released version
  and are not kept running against later versions (see the README of that folder).
