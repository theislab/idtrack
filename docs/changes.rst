Changes
=======

This page lists the changes that affect users, newest first. Version numbers follow ``pyproject.toml``.


0.1.0
-----

This version changes what conversions return where 0.0.5 returned wrong genes, how graphs are stored, the layout of
the package and its whole API. Graphs and table caches of 0.0.5 are not read. Install it from GitHub until it is
published on PyPI (:doc:`installation`).

**The package and its API**

* **Public modules.** The package is laid out by responsibility, in layers that import only downwards:
  :py:mod:`idtrack.errors`, :py:mod:`idtrack.settings` and :py:mod:`idtrack.ids` (the graph's vocabulary and the rule of
  ``<id>.<version>``) at the bottom; :py:mod:`idtrack.organisms` (the organism table); :py:mod:`idtrack.graph` (the
  graph and its file, :py:mod:`idtrack.graph.files`); :py:mod:`idtrack.engine` (the search and its ranking);
  :py:mod:`idtrack.results` and :py:mod:`idtrack.api`; :py:mod:`idtrack.harmonize`; :py:mod:`idtrack.build` (Ensembl
  access, the table store, the externals configuration, the graph builder); :py:mod:`idtrack.checks` (the self-tests).
  No module name starts with an underscore any more where users read it, and ``import idtrack`` then
  ``idtrack.API.load(<graph file>)`` imports none of the build stack (pandas, h5py, PyMySQL, PyYAML).
* **Loading and building are separate calls.** :py:meth:`idtrack.API.load` loads a graph file by its path and never
  builds or uses the network; a missing or unreadable file is an :py:class:`idtrack.errors.GraphFileError` that names
  it. :py:meth:`idtrack.API.build` builds a graph from Ensembl's tables (or loads the file of the same graph from the
  local repository; ``rebuild=True`` replaces it). :py:meth:`idtrack.API.graph_path` says where the file of a graph is,
  :py:meth:`idtrack.API.database_manager` gives the manager of the tables. ``API(graph)`` wraps a graph in memory.
  ``API(local_repository=...)``, ``API.build_graph``, ``API.get_database_manager`` and the attribute ``API.track``
  are gone.
* **One conversion call, typed results.** :py:meth:`idtrack.API.convert` returns an
  :py:class:`idtrack.results.ConversionResult` (a frozen dataclass; ``to_dict()`` is strict JSON and
  ``ConversionResult.from_dict`` reads it back) with every candidate the search reached
  (:py:class:`idtrack.results.Candidate`), ranked, the best marked: ``result.view("best")`` and ``result.view("all")``
  are views of the one search, so the argument ``strategy`` and the second search it cost are gone. Each candidate
  carries its ranking record (final confidence, bridges, gap, the display name, support of the entry gene, score,
  being the query itself, steps, the annotation counts where compared), and ``result.decided_by`` names the criterion
  between the best answer and the runner-up. ``result.status`` (``converted``, ``tie``, ``no_target``, ``not_found``, ``unknown_version``,
  ``no_conversion``, ``error``) and ``result.reason`` (why a result has no candidate,
  :py:data:`idtrack.results.REASONS`) replace the flags ``no_corresponding``, ``no_conversion`` and ``ambiguous``.
  ``convert_identifier`` is gone.
* **The path as data.** With ``explain=True`` the result has :py:attr:`~idtrack.results.ConversionResult.path`, an
  :py:class:`idtrack.results.ExplainedPath`: every node (kind, label, databases, active releases, assemblies, the
  releases outside the reference sequence, a gene's places) and every step (kind, releases, Ensembl's score, splits and
  merges, databases and assembly of a cross-reference, the assemblies on both sides of a history step and where its
  later gene lies relative to the earlier one, the reason it was taken as a code of
  :py:data:`idtrack.results.STEP_REASONS`) once per conversion, and every bridge the search considered with its rule
  (:py:class:`idtrack.results.BridgeCheck`); candidates refer to their steps by index.
  Nodes, the genes and ids of the bridges, and databases are named as results name them (a synonym without the graph's
  prefix, its node of kind ``synonym``). The untyped edge lists of ``the_path`` are gone.
* **Batches.** :py:meth:`idtrack.API.convert_many` (was ``convert_identifier_multiple``) accepts any sequence of strings
  (a list, a NumPy array, a pandas Index; a single string is refused) and never stops on one identifier's error: the
  item has ``status == "error"`` and the error in ``result.error``. It shows no progress bar unless ``progress=True``.
  :py:meth:`idtrack.API.classify` (was ``classify_multiple_conversion``) bins the results of one view, and
  :py:meth:`idtrack.API.classification_summary` (was ``binned_conversion_summary``; ``print_binned_conversion`` is
  gone) writes the counts.
* **What a graph holds, and names.** :py:meth:`idtrack.API.describe` (organism, releases, assemblies with their names
  and the releases they serve, databases, the package version and commit that built the graph) needs no network.
  :py:meth:`idtrack.API.genes_named` (which genes a name names, per database and release) and
  :py:meth:`idtrack.API.names_of` (the names of a gene) are public, as :py:meth:`idtrack.graph.TheGraph.genes_named`
  and :py:meth:`idtrack.graph.TheGraph.names_of`. Renamed: ``graph_releases`` and ``list_ensembl_releases`` to
  :py:meth:`idtrack.API.releases`, ``valid_final_databases`` to :py:meth:`idtrack.API.final_databases`,
  ``list_external_databases`` to :py:meth:`idtrack.API.external_databases`, ``list_external_databases_by_assembly``
  to :py:meth:`idtrack.API.databases_by_assembly` (string keys, strict JSON), ``external_database_forms`` to
  :py:meth:`idtrack.API.database_forms`, ``infer_identifier_source`` to :py:meth:`idtrack.API.infer_source` (its
  modes ``release`` and ``assembly_release`` were ``ensembl_release`` and ``assembly_ensembl_release``),
  ``resolve_organism`` (with the network) to :py:meth:`idtrack.API.ensembl_organism`; ``list_genome_assemblies``
  is gone: the assemblies are in :py:meth:`idtrack.API.describe`.
* **Errors.** Every error idtrack raises is an :py:class:`idtrack.errors.IdtrackError`: input errors
  (:py:class:`idtrack.errors.InputError`, with the argument and the valid values: unknown organism, release, assembly,
  database, identifier, a missing value, an inactive release, an unknown version, a wrong type or value; each also a
  ``ValueError``, or a ``TypeError`` for a wrong type), graph-file errors (:py:class:`idtrack.errors.GraphFileError`,
  :py:class:`idtrack.errors.GraphMismatchError`), :py:class:`idtrack.errors.SearchBudgetExceededError`, build errors
  (:py:class:`idtrack.errors.BuildError`: Ensembl access, schema, table format, history, externals) and
  :py:class:`idtrack.errors.IntegrityError` (a bug or a damaged file, never the caller's input).
* **Settings.** The budget of a search and the rules of bridges are a :py:class:`idtrack.settings.SearchSettings` held
  by an API (``API.load(path, search=...)``), and a call can give its own budget (``budget_states``,
  ``budget_seconds``; the engine takes the call's settings, ``search``); the hyper-connectivity threshold and the cache
  bounds are a :py:class:`idtrack.settings.GraphSettings` of a loaded graph (``graph_settings`` wherever one is
  passed). How Ensembl is reached (access mode, timeouts,
  retries, hosts) is set once per process with :py:func:`idtrack.settings.configure_ensembl`. The class attributes
  ``Track.search_budget_states``, ``Track.search_budget_seconds``, ``Track.non_bridging_databases``,
  ``TheGraph.genes_at_once_threshold`` and ``TheGraph.cache_sizes`` are gone.
* **The organism table and the vocabulary.** The class ``DB`` is gone: the organism table and its functions are
  :py:mod:`idtrack.organisms` (:py:data:`~idtrack.organisms.ORGANISM_TABLE`,
  :py:data:`~idtrack.organisms.SYMBOL_DATABASES`, :py:data:`~idtrack.organisms.ALIASES`,
  :py:func:`~idtrack.organisms.register_organism`, :py:func:`~idtrack.organisms.first_release`,
  :py:func:`~idtrack.organisms.resolve_organism` without the network, ...), the node types and markers are
  :py:mod:`idtrack.ids`, the network settings :py:mod:`idtrack.settings`.
* **Self-tests.** :py:class:`idtrack.checks.GraphChecks` (``GraphChecks(graph, database_manager)``) replaces ``TrackTests``
  (and ``API.build_graph(..., return_test=True)``); it holds an engine rather than being one. The checks keep their
  names and return :py:class:`idtrack.checks.CheckResult`; ``how_many_corresponding_path_ensembl`` is gone, and
  ``history_travel_testing_random`` always returns its result.
* **One name per concept.** The same thing has the same name in every call: ``snapshot_release`` (was ``ignore_after``
  of ``DatabaseManager`` and ``graph_file_path``, ``graph_last_ensembl_release`` of ``HarmonizeFeatures``,
  ``TheGraph.last_release``), ``first_release`` (was ``ignore_before``), ``to_release`` (was
  ``target_ensembl_release``), ``release`` (was ``ensembl_release`` of ``DatabaseManager`` and ``ExternalDatabases``,
  ``ens_rel``), ``organism`` (was ``organism_name``), ``identifier``/``identifiers`` (was ``from_id``,
  ``identifier_list``, ``dataset_ids``), ``name`` for :py:meth:`idtrack.graph.TheGraph.genes_named` (was ``text``),
  ``final_database`` (was ``to_database``), ``serving_assembly`` for the assembly that serves a release (was
  ``TheGraph.backbone_assembly`` and the engine's ``primary_assembly``; "primary" names only the graph's own
  assembly), ``allow_bridges`` and ``bridge_limit`` (were ``go_external`` and ``jump_limit``), ``progress`` for a
  progress bar (was ``verbose`` of the self-tests) and ``verbosity`` (was ``verbose_level`` of ``HarmonizeFeatures``),
  ``assembly`` for the assembly an argument names (was ``genome_assembly`` of ``API.database_manager``,
  ``DatabaseManager``, ``ExternalDatabases`` and ``GraphMaker.load_graph``; the graph file keeps its stored
  attribute ``genome_assembly``) and ``TheGraph.available_assemblies`` (was ``available_genome_assemblies``),
  ``forms`` (was ``form_list``), ``database_manager`` (was ``db_manager``), ``strategy`` for the best answer or all
  candidates also in the self-tests (was ``prioritize_to_one_filter``), ``from_release``, ``from_database`` and
  ``to_release`` in :py:meth:`idtrack.graph.TheGraph.resolve_name` and in ``is_final_external_conversion_robust``
  (``final_database``, ``to_release``), ``from_release_by_dataset`` and ``from_database_by_dataset`` of
  ``HarmonizeFeatures`` (were ``source_releases`` and ``source_databases``), ``features`` of
  ``unify_multiple_anndatas`` (was ``mode``), ``external_databases`` of :py:class:`idtrack.results.GraphDescription`
  and :py:class:`idtrack.results.AssemblyDescription`, and ``graph_id`` of :py:class:`idtrack.engine.Conversion`
  (as in :py:class:`idtrack.results.ConversionResult`).
* **The engine.** ``Track(graph)`` is built from a graph (``Track.from_graph`` and the constructor that built a graph
  are gone); :py:meth:`idtrack.engine.Track.convert` returns an :py:class:`idtrack.engine.Conversion` with the ranked
  candidates. ``Track.get_possible_paths`` and ``Track.calculate_node_scores`` are gone. ``Track`` is no longer
  importable from ``idtrack`` itself. Each argument is checked by one rule with one message: the engine's checks
  (a release of the graph; a database of the graph at the target release, matched without letter case and
  surrounding white space), which :py:class:`idtrack.API` uses before it hands the checked values to the search; the
  API also reads a user's identifier (surrounding white space removed, an empty string refused), where the engine
  reads the string as it is given.
* **Harmonization is an extra.** ``pip install "idtrack[harmonize] @ git+https://github.com/theislab/idtrack.git"``
  brings anndata and scipy for :py:class:`idtrack.harmonize.HarmonizeFeatures`; without them that class raises an
  ImportError that names the extra. It is imported by its name (``from idtrack import HarmonizeFeatures``): it is not
  in ``__all__``, so that ``from idtrack import *`` works on the plain install. Its pure core,
  :py:func:`idtrack.harmonize.convert_features`, :py:func:`idtrack.harmonize.feature_table` and
  :py:func:`idtrack.harmonize.output_plan`, needs pandas only.
* **Removed:** the external-database mappers (``idtrack._external_mappers``: the wrappers of MyGene, g:Profiler, gget,
  BioMart and the ortholog alignment, with their extras ``external-mappers``, ``ortholog``, ``embeddings`` and
  ``all-external``), the connection bridge (``ConnectionBridge``, and the dependency PySocks: every graph builds over
  Ensembl's HTTPS dumps where MySQL is blocked), and the folder ``reproducibility/`` of the repository (the manuscript's
  notebooks and scripts stay reachable at the tag ``manuscript-2026-03``).
* **The version** is 0.1.0, no longer a beta (the classifier ``Development Status :: 5 - Production/Stable``),
  written in ``pyproject.toml`` only and read from the installed package's metadata
  (``idtrack.__version__``; ``unknown`` for a source folder that is not installed). A graph file records the version
  that built it and, where it was known, the git commit (``package_commit``; :py:meth:`idtrack.API.describe`).

**What a user of 0.0.5 must do**

* **Build your graphs again.** Graph files of 0.0.5 (``graph_<organism>_min<a>_max<b>_narrow.pickle``) and its table
  cache (``<organism>_assembly-<key>.h5``) are not read; :py:meth:`idtrack.API.build` builds the graph from
  Ensembl's tables (downloaded again) and writes the new files beside the old ones, which you can then delete. A build
  takes from 6 minutes (cynomolgus) to 2 hours (human) and at most 4.5 GB of memory (previously up to 11 hours and
  65 GB for human); see "Time and Memory" in :doc:`usage`.
* Bring your code to the API described under "The package and its API" above; the arguments and attributes of 0.0.5
  listed under "Removed" below are gone.
* Use Python 3.12 to 3.15.
* Pin the snapshot release in scripts (``API.build("homo_sapiens", 115, local_repository)``) rather than building at
  :py:meth:`idtrack.API.ensembl_organism`'s latest release, which changes with every Ensembl release.
* Expect other answers where 0.0.5 was wrong (below). Results of 0.0.5 that you keep cannot be reproduced with this
  version; record the version with your results.

**Conversions: what users type and paste**

* **The string is read before it is looked up, and the result says how** (:py:mod:`idtrack.reading`). Invisible and
  control characters (a byte-order mark, zero-width characters, bidirectional marks, soft hyphens) are removed,
  compatibility forms (full-width letters and digits) and Unicode's dashes read as plain characters (``match``
  prefixed ``unicode:``); quotation marks around an identifier and the punctuation that ends a sentence or an item of
  a list at its end (Unicode's Terminal_Punctuation) are removed (``punctuation:``): previously ``ADORA3;`` answered
  another gene (UniProt's ``A1BG;`` was read with its semicolon at release 78), and ``'Agt'`` in mouse another gene
  through quoted synonyms; GENCODE's ``<id>_PAR_Y`` of a gene or transcript is read as the id (``par_y:``),
  previously not found. Unversioned transcript and translation ids are found in any letter case. The values of
  Ensembl's tables are read the same way where the graph is made, so the graph holds UniProt's ``A1BG;`` as ``A1BG``
  (one reading on both sides).
* **A missing value is never a gene.** The strings pandas reads as missing by default (``nan``, ``NA``, ``N/A``,
  ``None``, ``null``, ``<NA>``, ``#N/A`` and the others of its list), pandas' ``NaT`` and UniProt's placeholder
  ``unknown`` (any letter case) are refused with
  :py:class:`idtrack.errors.MissingValueError` unless ``from_database`` names their database (previously ``nan``
  answered SCN11A's gene, and the harmonizer summed a ``nan`` column into it). In a batch, an empty string or a
  missing value is that item's error and the batch goes on (previously an empty string stopped the whole batch); so is
  a cell that holds no value as Python, NumPy and pandas hold it (``None``, a float NaN, ``pd.NA``, ``NaT``: a pandas
  column before a cast to text), in :py:meth:`idtrack.API.convert_many` and :py:meth:`idtrack.API.source_inference`
  (previously the whole batch was refused as holding items that are not strings; any other such item still is).
* **The graph's bookkeeping never reaches a user** and is no input. A synonym is written as it is in every result
  (``graph_id``, ``from_databases``, :py:class:`idtrack.results.NameMatch`, the nodes and steps of an explained path):
  ``match`` (``synonym:exact``), ``NameMatch.synonym`` and the node kind ``synonym`` say it is one. The synonyms a
  database records are read under its own name: ``from_database="HGNC Symbol"`` reads HGNC's synonyms where no HGNC
  name matches, and the
  ``synonym_id::`` databases are gone from :py:meth:`idtrack.API.external_databases`,
  :py:meth:`idtrack.API.final_databases` and :py:meth:`idtrack.API.describe` (a synonym is no target database).
  ``synonym_id::TP53`` and ``ENSG00000141510.Void`` are refused with :py:class:`idtrack.errors.InvalidArgumentError`
  (previously they answered TP53BP1 and TP53).
* **Bounds.** A string longer than 1,000 characters and an Ensembl version of more than 5 digits are refused with
  :py:class:`idtrack.errors.InvalidArgumentError` (previously a version of more than 4,300 digits raised Python's
  ``ValueError``).
* **Mouse symbols in one letter case** (``AR``, ``CCK``, ``nos2``) are read first as MGI symbols without letter case
  where that names one gene (previously the exact synonym of another gene won: 133 symbols in capitals and 12 in
  lower case answered another gene); the other reading is reported in the new field
  ``ConversionResult.other_readings``. The rule holds where the graph's symbols are written in mixed case
  (:py:meth:`idtrack.graph.TheGraph.symbol_case`).
* **Names made unique by single-cell tools** (``SRF.1``): in a batch, an item that is another item with ``-1``,
  ``.1`` or ``_1`` appended (and no exact name itself where its own conversion reads it) is that item's copy
  (``match`` prefixed ``copy:``); every other item of a batch is answered as :py:meth:`idtrack.API.convert` answers it
  alone (an item that is a name of another release only, such as pig ``SLA-10`` or rhesus ``mml-mir-129-2`` read where
  the graph holds it, was made a copy before: human 2,847 names of whole-release lists without a release, 828 read at
  their release). In any call such a string is not read as another gene's synonym by its separators or letter case
  (previously ``SRF.1`` answered SPATA6 through the synonym ``SRF-1``). The harmonizer says so in
  ``idtrack_reason``.
* **A separator reading of several names of one release is one answer** (``AY172335_1`` read where both
  ``AY172335-1`` and ``AY172335.1`` are names; previously the first in sort order was taken silently): the readings'
  candidates are ranked as the genes of one conversion, so readings whose best answers are equal in the evidence of
  their lineage tie and a reading whose answer needs a bridge does not tie with one that needs none. Readings that
  never exist at one release are not joined: the string is read at the latest release at which one of them exists
  (mouse ``AY172335_1`` is ``AY172335.1`` of release 60).
* **Numbers.** Genes that different numbering schemes attach one number to (an NCBI gene id and an old Havana number
  spelt alike) tie instead of being ranked by how many databases attach it; every candidate names the databases it
  was read through (``Candidate.entry_databases``) and whether a name authority gives it the identifier as its
  symbol (``Candidate.authority``; the authorities are the organism's symbol databases and NCBI's symbols, named as
  the graph names the labels of Ensembl's ``EntrezGene``: :py:attr:`idtrack.TheGraph.authority_databases`).
* **Ensembl's score 0 between two ids is no score**: such a step ranks a path below every path whose scores are
  measured (``Candidate.unscored_steps``); previously it counted as the best score, 1.
* **Pseudoautosomal genes and LRG records.** From release 110 the X and Y copies of a pseudoautosomal gene have ids of
  their own; a name both carry answers the X copy, the Y copy listed in ``alternative_locus_genes`` (previously a
  tie). A gene of a pseudoautosomal region of Y is read at the place of the region of X it repeats, so the Y copy is
  one gene with the X gene at its place (for the count of genes, the bridges and this rule: :py:mod:`idtrack.graph.genes`)
  and no copy of an X gene elsewhere. The regions are GRC's (:py:data:`idtrack.organisms.PSEUDOAUTOSOMAL_REGIONS`),
  equal to the ``PAR`` rows of Ensembl's assembly exceptions that a built graph holds. Every id of an LRG record answers the gene the record
  describes (``match`` prefixed ``lrg:``, the record listed apart; previously the record itself): the record's id
  (``LRG_321``), its transcript and translation ids (``LRG_321t1``, ``LRG_321p1``, ``LRG_321t1-1``) and the gene id of
  the GRCh37 releases (``LRG_7_g1``), which round 1 still answered with the record itself (human 4,145 ids); a record
  is the gene that Ensembl names by a record's id or places on a sequence region so named. Without ``from_release`` its
  starting release is reported as inferred, as for any identifier. In a batch an Ensembl transcript or translation id
  of its own (``LRG_321t1-1`` beside ``LRG_321t1``) is no copy of another item.
* **Where a list comes from** (:py:meth:`idtrack.API.source_inference`, new, and :py:meth:`idtrack.API.infer_source`):
  each distinct identifier counts once (previously a repeated identifier counted at every repetition), read as it was
  at each release, where it is a name of that release and not where it is only a synonym (previously every name was
  read as it is at the last release, so lists of names of human 75, 80, 105 and 110 and of mouse 75 and 110 were
  placed at later releases, and 23 of 32 symbol tables of published single-cell studies at 115), and the answer is the band of origins the list
  cannot tell apart, with the margin to the next; in the mode ``release`` a release counts on the assembly that serves
  it.
* **The harmonizer's result cache** is keyed by the graph's content digest and :py:data:`idtrack.settings.RULES_VERSION`
  (raised with every change of an answer), not by the package version (previously an upgrade served the old
  answers). Its reasons are the API's (:py:data:`idtrack.results.REASONS`); a missing value is
  ``unconverted_missing_value``.

**Conversions: what an identifier means**

* **The starting release.** With ``from_release`` the identifier is read at that release. Without it, it is read at
  the target release when it exists there, and otherwise at the latest release at which it exists. Previously an id
  without ``from_release`` was read at the *oldest* period in which it existed, so that, for example, a current human gene id
  without version converted to another gene in 2,557 of 78,899 cases (``NFYA`` to ``NFYB``), and 17 % of human NCBI
  gene ids and 1 to 2 % of symbols returned the wrong gene. The rule is the same for Ensembl ids, symbols and other
  external ids, and the result says which release was used (``from_release``, ``from_release_inferred``), also when
  the string leads to no gene at the target release (previously such a result reported no release).
* **A** ``from_release`` **at which the graph does not hold the identifier.** :py:meth:`idtrack.engine.Track.convert` raises
  :py:class:`idtrack.errors.InactiveAtReleaseError`, which names the releases at which the identifier exists (previously the
  answer came silently, as if no release had been given). :py:meth:`idtrack.API.convert` reads the
  identifier where the graph holds it instead and flags the result (``from_release_fallback``, ``from_release_given``);
  a result with targets read so has ``reason == "read_at_another_release"`` (its status stays ``converted`` or ``tie``:
  it answers what the identifier meant where it was read, which can be another gene), and
  :py:meth:`idtrack.API.classify` puts it in a bin of its own, ``read_at_another_release``.
* **What a string means.** When a string can mean several nodes (a symbol that was the name of one gene and is a
  synonym of another today), the one that exists at the release asked about is taken (``from_release``, without it
  the target release; where none exists there, the string is read at the latest release at which one of them exists,
  by the steps of that release), so that one release decides both the node and where it is read (previously, without
  ``from_release``, the node was chosen at the snapshot release and read at the target release: human ``RNR1`` to 90
  answered NR4A2, whose synonym it is, and with ``from_release=90`` MT-RNR1, whose NCBI symbol it is at 90; 2 human
  and 5 mouse of 2,000 sampled names); an Ensembl stable id is never passed over for a synonym spelt the same. A
  meaning the string has at other releases only is reported in ``other_readings`` (``reason="other_release"``,
  :py:attr:`idtrack.results.OtherReading.releases`), not answered: ``RNR1`` without a release is NR4A2's synonym, the
  symbol of MT-RNR1 at releases 86 to 93 and 95 reported. An identifier exists at a
  release only on the genome assembly that Ensembl maintained at that release: the names that human GRCh37, listed
  beside GRCh38 with its annotation of release 75, still carries do not exist today (3,277 of them are a synonym of
  a gene on GRCh38 today; read as the old name, 638 returned other genes than the synonym's). Then the string is
  matched
  exactly, then ignoring letter case, then taking ``-``, ``_`` and ``.`` as one character, then (only for a versioned
  accession) with the version written (below); each step on names and then on synonyms. A name typed as a database wrote it at the release it is read at is an exact match of
  that name, also where the database writes it in another letter case today or two databases write it in two ways
  (at release 75 MGI wrote the mouse gene ``Bap1`` and UniProt wrote it ``BAP1``), and is not taken for another
  gene's synonym spelt the same.
  The new result field ``match`` says how it matched. The engine reads a string by the same
  rule (:py:meth:`idtrack.graph.TheGraph.resolve_name`): ``Track.convert("X")`` no longer takes a string that happens to be
  the name of a node as that node; to convert a particular node whatever the release, give a ``from_release`` at which
  it exists. Previously a dash-or-underscore suffix was cut off any string (``SNORD116_10`` became ``SNORD116``:
  10,506 human ids resolved to another gene), the least-connected spelling of an id written in several cases was kept,
  and every combination of separators was tried (minutes for a string with 20 of them); the lookup now costs
  microseconds whatever the string. Surrounding white space is ignored.
* **Versions.** An Ensembl id with a version that the graph does not hold is not answered with another version's gene:
  the result has no target, and ``known_versions`` lists the versions that exist (the engine raises
  :py:class:`idtrack.errors.UnknownVersionError`). A version higher than all the graph holds, of an id still active at the
  snapshot, may come from a newer release: it is read as the latest version and reported as
  ``match == "version_newer_than_graph"``. Previously any unknown version was answered with the gene of the id without
  version. A version is written after ``.``: ``ENSG00000141510-5`` and ``ENSG00000141510_5`` are not found (a tool's
  suffix for a repeated name as often as a version). Transcript and translation ids without version are now found
  (previously none was).
* **Versions of sequence accessions** are read by the same rule. The graph keeps, per link and release, the version
  Ensembl's cross reference attached (``xref.version``; :py:meth:`idtrack.graph.TheGraph.accession_versions`), and which
  databases carry versions is read from it (RefSeq, UniProt, CCDS, the EMBL protein ids, Havana's ids), not from a list
  of database names (``idtrack.ids.ACCESSION_DATABASE_PREFIXES`` is gone). An accession written with a version
  (``XM_003129161.3``) is the accession as the links that attached that version carry it, read where they did (without
  ``from_release`` at the target release if they attached it there, else at the latest release at which they did); a
  version above every version attached is the latest (``match == "version_newer_than_graph"``); a version below the
  latest that no link attached is ``unknown_version`` with the versions there are. A version above every version is
  read as the latest only where the accession is attached at the graph's last release (decision 14's condition for
  Ensembl ids: only then can it be newer than the graph); for an accession Ensembl no longer attaches it is
  ``unknown_version`` (previously answered: of the accessions whose latest version is not attached at 115, pig
  77,755 and human 460,870, a typed latest version plus one was answered for 243 and 298 of 300 sampled). A string
  whose identifier is also, at the release read, an id of a database that writes no version (human ``2633.1``: a
  numeric Havana translation id with version 1 at releases 50 and 51, and ``2633``, NCBI's gene id of GBP1) is read
  by the database that has the identifier with that version at that release; where none has it there, the version
  reading is taken by the rule above and the other reading (the identifier as that database's id) is reported beside
  it (``other_readings``, ``reason="unversioned_database"``), never one reading silently (human 2,493 numbers: of 300
  sampled, ``<id>.1`` answered another gene than ``<id>`` in 298, unreported). Where a database's number is an
  edit counter of its records and no version of the sequence (UniProt's entry version: the version differs in 58 to
  99.7 % of the consecutive-release pairs of one link, against at most 14.6 % for RefSeq, CCDS, Havana and Vega,
  measured from the graph's own versions: :py:meth:`idtrack.graph.TheGraph.edit_counter_databases`), a typed version
  is not judged: the accession is read without it, with ``reason="version_not_judged"`` (decision 101). Previously the version was cut off
  and the accession read where it last was, whatever version was written: the answer differed from the one read where
  the version was attached for pig 2,678 of 39,991 old RefSeq versions, rhesus 1,829 of 53,273, cynomolgus 255 of
  28,288, mouse 1,169 of 104,926 and human 1,454 of 156,621, and a version the accession never had was answered with
  the gene (93 to 100 % of such lines of the benchmark's family C). In a batch, such an accession beside the
  accession without version is its version, not a copy (decision 82).
* **Ties.** The best answer is one target, or, when several genes or targets are equal in all evidence, all
  of them, with ``tie`` true, in ascending order of gene id and target id. Previously several targets could come in an
  order that changed between runs (the hash seed), so the first target could be another gene in the next run. Every list of
  a result is now in a defined order, and two runs give the same answer. A tie is judged among the genes that have an
  id in the requested database (the others are named in ``no_target_genes``).
* **A target in a database is an id of the gene answered.** With ``final_database`` the answer is first a gene: the
  genes are ranked as Ensembl genes, and the targets are the ids that the database attaches at the target release to
  the genes of that best answer; where none of them has one, the result is the genes themselves (status ``no_target``,
  ``reason="no_id_in_final_database"``), never the id of a gene of a lower rank. Previously the genes without an id
  left the ranking first, so a gene below them became the certain answer (pig ``PSME1`` to ``VGNC Symbol`` answered
  ``DCAF11``, the symbol of a gene that carries ``PSME1`` only as a UniProt gene name, status ``converted``: 648 lines
  over three releases per organism), and the harmonizer moved the feature onto that gene.
* **Every answer, best first.** Every candidate is ranked: the genes of the best answer first, then the others by
  their evidence, so the first candidate is the first target of the best answer.
* **Copies on alternative loci, patches and LRG records** (human; mouse GRCm38) are marked in the graph from Ensembl's
  sequence regions. When an identifier leads to a gene of the reference sequence, the copies are left out of the
  answer and listed in the new result field ``alternative_locus_genes``; when it leads to copies only, they are the
  answer. :py:meth:`idtrack.graph.TheGraph.is_non_reference` and :py:meth:`idtrack.graph.TheGraph.non_reference_releases` tell
  where a gene is such a copy. (Previously LRG records were left out by their id prefix, without saying so.)
* **A gene id is its own answer.** A queried Ensembl gene id that exists at the target release is answered with
  itself: its own version comes first in the result, is never set apart as a copy, and is the answer of
  the best answer whatever else its history reaches; the other candidates follow it.
* **A database named for the identifier.** ``from_database`` of :py:meth:`idtrack.API.convert` (and of
  :py:meth:`idtrack.engine.Track.convert`) reads the identifier as an id of that database: only the genes that database
  attaches it to are entered, and between them the evidence of every database still counts. A symbol that two
  databases give to two genes is a tie without it and that database's gene with it; a symbol that the named database
  gives to two genes, of which other databases confirm one, is that one.
* **The databases behind an identifier.** The result field ``from_databases`` names every database that attaches the
  identifier to its genes at the release it was read at, whether to a gene itself or to one of its transcripts or
  translations, and the support of a gene (``entry_support`` of :py:meth:`idtrack.engine.Track.convert`) counts all of them.
* **Ensembl's display name ranks first.** Among the genes that carry a name at the release it is read at, the gene
  whose display name it is there (``gene.display_xref_id``: Ensembl's own statement of the gene's name at that
  release, whatever database it came from) ranks before support: support counts databases, and a projection or an NCBI
  symbol that Ensembl attached to another gene's translation is one database like the gene's own. Two genes that both
  display the name are equal in it, and support decides between them as before. The display name holds for the string
  as typed (decision 102): where it has letters of both cases, only a display name its database spelt so counts (mouse
  ``Foxl1`` at 70 is the gene MGI displays so, not the gene UniProt displays as ``FOXL1``: mouse 46 lines and 23 ties,
  human 3, ``C2orf15``); a string in one letter case carries no evidence of case, and where the organism's symbols are
  written in mixed case the name authority's display name, read without case, ranks before another source's
  (``Candidate.display_rank``: mouse ``GAPDH`` at 70 is Gapdh, the gene UniProt displays as ``GAPDH`` reported after
  it). ``Candidate.display_name`` says it of each candidate's entry genes and ``decided_by`` is
  ``"display_name"`` where it decided (:py:data:`idtrack.results.CRITERIA`, :py:mod:`idtrack.engine.ranking`). The
  symbol of a name authority against another gene's synonym is decided before both, when the string is read
  (decision 94). Where the history merges several entry genes into one gene, the display name and the support of that
  gene are the best of all its entry genes, not those of the one path the search keeps (decision 106: pig ``C/EBP``
  from 78 to 115 is the tie of ENSSSCG00000002866 and ENSSSCG00000034207, which ENSSSCG00000028739, the gene that
  displays ``C/EBP``, merged into; before, the path kept from ENSSSCG00000024816 hid that display name).
* **History.** Ensembl's id history (``mapping_session``) is checked against the ids of each release when a graph is
  built; a session recorded with wrong releases is placed where its ids change (pig 106 to "104" is 106 to 107; human
  114 to "114" is 114 to 115), each correction is recorded in the graph file and logged, and a session that fits no
  pair of releases stops the build (:py:class:`idtrack.errors.HistoryValidationError`). Previously 1,123 pig gene ids converted
  to ids that do not exist at the target. The human graph now has the id history before release 76, which the
  previous table cache could omit.
* **A history step between the mitochondrial genome and a nuclear sequence is not followed.** At three assembly
  changes Ensembl gave the ids of mitochondrial genes to nuclear genes (rhesus 5, cynomolgus 5 and human 4 ids:
  rhesus ``convert("MT-ND1", from_release=97)`` answered an unnamed gene of chromosome 8 as ``converted``); such a step
  is no lineage, the id has no successor across it, and the explained path lists it in the new field
  ``ExplainedPath.refused`` with ``reason == "mitochondrial_genome"``. Which sequence is the mitochondrial genome is
  read from Ensembl's sequence regions (:py:func:`idtrack.graph.genes.place_kinds`); the step is refused where one
  version lies on it and the other on a chromosome or a scaffold, also where the two places cannot be compared. The
  versions on the two sides are two genes that share a stable id: bridges compare the places of each side, and a
  bridge between the two sides is refused (bridge rule ``mitochondrial_genome``), so rhesus ``MT-ND1`` of release 97
  answers the mitochondrial ``ND1`` of Mmul_10 through its name.
* **Search.** The search follows Ensembl's history first; only when that leads to no gene does it pass through an
  external id or transcript that two genes carried (at most two such bridges, the fewest possible, in the order of
  time, never through clone-based names or through ids that more than five genes carry at once). It is bounded: every
  state is visited once, and a call has a budget (:py:class:`idtrack.settings.SearchSettings`) past which it raises :py:class:`idtrack.errors.SearchBudgetExceededError`.
  The previous search enumerated paths: some conversions ran for more than 40 minutes, and ``strategy="best"`` raised
  ``unexpected!2`` for 2.7 % of mouse gene ids.
* **A bridge is evidence of a lineage only where an id leaves one gene for another at the same place.** A synonym is
  no bridge; where both genes lie on the reference sequence of one assembly, a bridge needs their places to overlap
  there (the graph holds every gene version's place; one rule of places for bridges and for counting genes,
  :py:mod:`idtrack.graph.genes`); where the places cannot be compared (an assembly change lies
  between the genes), a bridge needs an id that no release gave to both. A gene on a patch, a haplotype, an alternative
  locus or an LRG record (whose place is not compared with the reference sequence) bridges through the ids it carried
  together with the other gene at one release, as with the gene it is a copy of, and only where at each such release
  the genes that carry the id are one gene (an id of several genes, a family label or a shared alias, does not say
  which gene the copy stands for: ``copy_id_of_several_genes``; human ``ENST00000639379``, a gene of a chromosome-16
  patch, from 97 to 112 answered a SCARNA6 gene of chromosome 2 and now answers nothing), not through an id that came
  to another gene at another time. Previously an id whose history ends was answered through a shared alias, a duplicated symbol or a family
  label with a gene elsewhere in the genome (``convert("ENSMMUG00000036773", from_release=65, to_release=48)`` in
  rhesus answered three SNORD59 genes on other chromosomes; it now answers nothing). The explained path lists every
  bridge considered with the rule that took or refused it (``result.path.bridges``), each gene's places, and for each
  history step where the later gene lies relative to the earlier one (``places``).
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
* New fields of a result (:py:class:`idtrack.results.ConversionResult`): ``match``, ``known_versions``,
  ``from_release`` (the release used), ``from_release_inferred``, ``from_release_given``, ``from_release_fallback``,
  ``from_databases``, ``to_release``, ``tie``, ``alternative_locus_genes``, ``no_target_genes`` (genes reached that
  have no id in the requested database while others have), ``reasons`` (every reason that applies, in a stated order:
  why there is no target, that the identifier was read at another release than the one given, that a version was not
  judged; ``reason`` is the first of them, so a result read at another release whose version was not judged says
  both) and ``error``; the candidates and the explained path are described above.
* Every argument is checked before the search: the identifier (a non-empty string), the releases (integers of the
  graph: :py:meth:`idtrack.API.releases`; previously releases the graph does not hold were answered), the final
  database (:py:meth:`idtrack.API.final_databases`, case ignored; previously checked only when the identifier
  was found). An invalid argument raises an :py:class:`idtrack.errors.InputError` naming it.
* :py:meth:`idtrack.API.convert_many` checks all items first (positions named), converts each distinct
  identifier once, and records an error of one identifier on that item instead of losing the batch.
* :py:meth:`idtrack.API.releases` returns the graph's releases (pig 56 to 115; previously the primary
  assembly's releases only, 90 to 115); the methods that list databases and assemblies return new sorted lists.
* :py:meth:`idtrack.API.infer_source` returns a list: the band of origins that the list cannot tell apart from the
  best one (:py:meth:`idtrack.API.source_inference`, below), or all origins with their counts; it raises
  :py:class:`idtrack.errors.UnknownIdentifierError` when no identifier is in the graph (previously ``IndexError``).
* :py:meth:`idtrack.API.classify` counts a query that converts to itself (without version, in
  another case) as unchanged; previously every query without version counted as "changed". A result read at another
  release than the one given is in the bin ``read_at_another_release`` (not among the matches), and
  :py:meth:`idtrack.API.classification_summary` counts it.
* :py:meth:`idtrack.API.configure_logger` configures the ``"idtrack"`` logger only (previously it called
  ``logging.basicConfig`` on the root logger); every module logs under ``idtrack.<module>``.
* ``import idtrack`` imports nothing heavy; ``idtrack.API`` does not import anndata or scipy.
* Errors a caller may want to catch are exported (:py:mod:`idtrack.errors`, above).

**Graph files**

* A graph is one ``.npz`` file of arrays (each deflated where that makes it at least a tenth smaller, inflated in
  parallel threads when read) and JSON, named
  ``graph_<organism>_assembly<key>_min<first>_max<snapshot>.npz`` (``_sessions`` when its record lists Ensembl's
  mapping sessions in full, ``_extall`` when it holds every external database); it holds no pickle, so loading it
  executes no code. It holds one record of what it is and what built it, each fact once
  (:py:class:`idtrack.graph.GraphRecord`, ``graph.record``): organism, primary assembly, window, snapshot, releases
  with the assembly that serves each, forms, the external databases per assembly, the selection of external databases
  that takes effect and the release of the externals configuration whose settings each release took, the organism
  table, the package version and commit, the tables it was built from, the counts of the build, the history report,
  build date, and the digest of its content (which covers the spellings of the external ids and the places of the
  genes). A graph built from a source folder that is not an installed idtrack records no version but says so
  (``none: built from a source folder, not an installed idtrack``): its commit alone names its code, and a commit that
  a later squash or rebase replaced is on no published branch, so graphs to share are built with an installed release.
  The graph also holds the version every link of an accession attached at each release and the links that are a
  gene's display name at each release (both in the content digest).
  The graph also holds the places of the genes carried onto later assemblies by Ensembl's mapping of the assemblies
  (:py:meth:`idtrack.TheGraph.carried_places`), Ensembl's assembly exceptions of every release
  (:py:attr:`idtrack.TheGraph.assembly_exceptions`), the database each display name is filed under, and in its record
  the chain files it read (``assembly_mappings``) and the names of Ensembl's databases each selected database holds
  with their releases (``database_names``).
  The format version is 6; files of earlier versions are not read. A file that holds another graph than the
  one asked for raises :py:class:`idtrack.errors.GraphMismatchError` naming the differences and how to build the graph
  again (previously it was loaded silently, so a graph of another assembly or of another externals YAML answered for
  this one). ``GraphMaker``'s ``narrow`` and ``narrow_external`` are ``session_details`` and ``all_external`` (each the
  opposite, both ``False`` by default).
* A graph holds where every gene version lies: assembly, sequence region, start, end and strand, at the releases at
  which it lies there, and what the sequence region is (``Place.kind``: a chromosome, the mitochondrial genome or a
  scaffold, read from Ensembl's coordinate systems and region attributes; :py:meth:`idtrack.TheGraph.places`).
* A graph is built on the organism's default primary assembly of its snapshot release; a
  :py:class:`idtrack.build.GraphMaker` of a manager of another assembly is refused, and the graph has no node of one
  assembly only (no graph of the shipped organisms had one).
* :py:meth:`idtrack.API.load` loads a graph file with no table, no :py:class:`idtrack.build.DatabaseManager` and no
  network. Loading takes 1 to 5 s and
  holds 0.26 to 1.14 GB (human: 4.3 s and 1.14 GB; previously 206 s and 35 GB, 50 GB with the caches); nothing is
  computed in advance.
* A graph is deterministic: built twice from the same tables, under different hash seeds, it has the same content
  digest (``graph.record.content_digest``).
* Tables are stored one file per table (``idtrack_tables/<organism>_assembly-<key>/ens<release>_<table>.h5``), each
  written to a temporary file and renamed, so an interrupted run leaves every stored table readable (previously one HDF5
  file per assembly, which a killed writer could leave unopenable). A table's file name and content record what it
  depends on (the release window of the id history, the databases the externals YAML enables): changing the YAML or
  the window no longer serves a stale table, and nothing needs to be deleted by hand. A read-only repository serves
  the tables it holds.

**What the graph holds of the external databases**

* **One database name means one kind of identifier at every release.** Where Ensembl gives the label rows and the
  accession rows of a database one name, the labels take a name of their own, the same at every release: the labels of
  ``EntrezGene`` (gene symbols, named ``EntrezGene`` like the NCBI gene numbers up to release 87, ``NCBI gene`` at 88 to
  99) are ``NCBI gene (formerly Entrezgene)`` at every release, so that converting to ``EntrezGene`` returns the NCBI
  gene number and not the number and the symbol as a tie; the labels of ``RFAM``, ``miRBase``, ``CCDS``, ``WikiGene``,
  ``Reactome`` and a few others, whose display name is the name of their accessions, are ``<database> label``. Naming
  the labels changes the name of rows, never their number: where a database's display name is its own name at a
  release, a label that only repeats its accession stays one row with it (as Ensembl's naming gave it); elsewhere a
  label equal to its accession is kept under the labels' name (``Havana gene``, the labels of ``OTTG``).
  The one name of a database's labels is its display name at the newest release at which that name is not also the
  name of accessions, else ``<database> label`` (:py:func:`idtrack.build.database_manager.label_names_of_databases`,
  read from Ensembl's tables of the graph's releases). The names it replaces (``NCBI gene``,
  ``synonym_id::EntrezGene``, ``synonym_id::NCBI gene`` among others) are no databases of a graph any more: a
  conversion that names them (``final_database``, ``from_database``) raises
  :py:class:`idtrack.errors.UnknownDatabaseError`, whose message names the graph's name (the graph record's
  ``renamed_databases``); there is no alias.
* **A missing value is not an identifier.** Identifiers without a letter or a digit (``.``, ``-``) and synonyms that
  carry UniProt's evidence text (``BHMT {ECO:0000313|EMBL:...}``) are left out where the tables are made.
* **An RFAM family label names only the gene whose display name it is** (``gene.display_xref_id``): ``5S_rRNA`` named
  531 human genes, Ensembl names 10 with it. The accession (``RF00001``) stays on every gene of the family; since it
  names a family, not a gene, it is never a bridge between two genes
  (:py:data:`idtrack.settings.FAMILY_DATABASES`, part of :py:data:`idtrack.settings.NON_BRIDGING_DATABASES`; pig
  ENSSSCG00000020242.1 was carried through ``RF00049`` from chromosome 14 to a gene of chromosome 1).
* **Every display name of a release is a name of its gene at that release**, whatever database Ensembl filed it under
  and whatever the externals YAML selects: Ensembl gave the display names of pig and macaque genes before release 61
  from HGNC's projections (``HGNC_curated_gene``, ``HGNC_automatic_gene``), mouse names before 52 under MGI's database
  names of the time, and the names of genes without a better one from UniProt's entry names, RefSeq or Rfam, often
  through a transcript or a translation. Such a name was not in the graph (every pig name at 56 to 60, every rhesus
  name at 48 to 60): asked at its release it was read at its latest release instead and carried back, lost or
  another gene (pig ``SMURF1`` at 56). A display name that the gene has already from a selected database adds nothing;
  the others are labels of the gene under the name of their database's labels (``HGNC (curated)``, ``HGNC
  (automatic)``, ``UniProtKB/TrEMBL``), so that a projected name is told from the authority's own name; the
  definitions of the databases hold for them (a miRBase name that Ensembl displays on the host or antisense gene of
  the miRNA stays the miRNA's). The externals templates list the databases' own cross references only: a display
  name is held whatever the YAML selects, so it is no entry to select. NCBI's gene
  ids, their symbols and synonyms are selected on whatever form Ensembl attached them: to translations up to release
  62 (they were not in the graph before 63).
* **One gene is not one stable id.** Ensembl gives a gene a new stable id more often than the gene changes, so the
  rules that ask how many genes an identifier names, and the rule of bridges, read one definition
  (:py:mod:`idtrack.graph.genes`, :py:meth:`idtrack.TheGraph.count_genes`): the ids of one locus over the releases
  are one gene (an id that Ensembl's history carried to another id that follows it, no release at which both live;
  ids whose places overlap on one region and strand of one assembly's reference sequence), and a copy that lies only
  outside the reference sequence (a patch, a haplotype, an alternative locus, an LRG record) adds no gene. Two ids that
  live at one release are two genes unless their places overlap then, whatever history they share: a split of one id
  into two that live side by side is two genes (mouse ``mmu-mir-30f``: one id of GRCm38 split into two of GRCm39 on
  the two strands of the same bases was counted as one gene). Genes are counted where the graph tells them apart (at
  one release, on one assembly, and by the names of chromosomes across assemblies), and no more: holders on two
  assemblies without a history link and without a comparable place are not two genes (rhesus ``mml-mir-1232``,
  ``MI0006322``, on a gene of MMUL_1 and then on a gene of Mmul_10, was counted as two). The count is the size of a
  set of ids told pairwise apart, built in one order (a lower bound of the most): an id of the frame that shows the most
  loci is compared with the others too (two genes at one place at one release that lie apart at the next are two).
  Where Ensembl publishes a mapping of an earlier assembly onto a later one (decision 103: its chain files, human
  NCBI36, GRCh37 and GRCh38, mouse NCBIM37, GRCm38 and GRCm39, pig Sscrofa9 onto Sscrofa10.2), the graph carries every
  place of the earlier assembly onto the later ones at build (:py:meth:`idtrack.TheGraph.carried_places`; the files are
  downloaded once and kept in the local repository) and compares the carried place with the later gene's by overlap,
  as places on one assembly: a bridge across the assembly change that the mapping puts elsewhere is refused (mouse
  ``Gm5923``: ENSMUSG00000078334 of NCBIM37 to a gene of X of GRCm38 540 kb from where the mapping puts it). Where
  Ensembl publishes none (pig Sscrofa10.2 onto Sscrofa11.1, the macaques), places on two assemblies are compared by the
  sequences they lie on only: two genes that lie on chromosomes of different names, or one on the mitochondrial genome
  and the other on a chromosome, lie apart (``places_apart``), since the assemblies of one organism name its
  chromosomes alike from one to the next (by its karyotype); a bridge between them is refused (pig retired ids of
  Sscrofa9 were carried to genes of other chromosomes of Sscrofa11.1: 14 of 202 bridges, ``RPL28`` from chromosome 15
  to 6), and a history step between them is marked so (rhesus ``MT-ND1``, from the mitochondrial genome of Mmul_8.0.1
  to chromosome 8 of Mmul_10). A copy on a patch or a haplotype stands for the region of the reference sequence that
  Ensembl's assembly exception of its region gives (``assembly_exception``, read at every release,
  :py:attr:`idtrack.TheGraph.assembly_exceptions`): it and a gene that shares no base with that region lie apart (the
  mouse copy ENSMUSG00000108031 on a patch of Y 49.2 to 49.6 Mb was bridged through an NCBI id to a Y gene 29 Mb
  away); inside the region, and for an LRG record (no exception), its coordinates are not compared with the reference
  sequence's.
* **miRBase names are in the graph**, at the gene level, on genes of the biotype ``miRNA`` only (a protein-coding
  gene that hosts a miRNA does not take its name), whatever database Ensembl files them under (the names it derives
  from miRBase and displays at human and mouse 61 and 62 are filed under ``miRBase_gene_name``:
  :py:func:`idtrack.build.database_manager.rules_database`), and only where the graph does not show the entry's genes
  at two loci: an entry on genes at several loci, at once (copies of one precursor at several loci; mouse
  ``mmu-mir-1906-1``) or one after the other at places apart (mouse ``MI0000399``, moved to a gene at another locus),
  says nothing about which gene it is (:py:data:`idtrack.build.database_manager.ONE_GENE_ONLY`; the record counts the links left
  out and lists the entries, :py:attr:`idtrack.TheGraph.entries_of_several_genes`, which the self-checks against
  Ensembl's tables leave out too). An entry on a gene and its copies, or on the successive ids of one locus, stays
  (judged by stable id, these were left out: in human 247 entries of a gene and its copies and 500 of successive
  ids, of the 1,252 left out). A link read on a transcript counts for the gene that holds the transcript at the
  releases of the link (decision 108): the miRBase rows Ensembl attached to the transcripts of miRNA genes up to
  release 62 (read since decision 104) do not count for the host gene that Ensembl later also put such a transcript
  in (human ``MI0000467`` on ENST00000384838, of ENSG00000207570 and later of its host ENSG00000245526; counted for
  every gene the transcript ever belonged to, human entries such as hsa-mir-21, let-7b, mir-34a and mir-143 were
  left out). The pig and
  macaque miRNA genes whose display name is their miRBase name convert by it.
* **A label that named many genes over its releases is a family name**: a display name or symbol that named more genes
  than :py:attr:`idtrack.settings.GraphSettings.hyperconnective_threshold` over all its releases is never a bridge, as
  an id linked to more genes than that at one release was already (mouse ``SCARNA15``, the label of 9 genes over the
  releases and of at most 5 at once, carried a retired id to copies at other loci). The count over the releases reads
  the one definition of a gene: a symbol a gene shares with its copies on patches and haplotypes is no family name
  (counted by stable id, 636 human labels such as ``ACTB`` reached the threshold only through their copies). The count
  at once still counts stable ids: an id a gene shares with many copies at once stays no bridge, as Ensembl's history
  gives for such retired ids. The graph record names the label databases by Ensembl's database (``label_names``);
  :py:attr:`idtrack.TheGraph.label_families` holds the counts.
* **A spelling belongs to its link.** Where databases write an identifier in different letter cases, the graph keeps
  one node and the spelling of every link (identifier, Ensembl id, database, release): ``Tc`` (a synonym of one mouse
  gene) and ``tc`` (a synonym of Noto) are one node, and ``Tc`` typed exactly is the first gene's synonym only. Two
  spellings that one database gives one gene at one release are one value.
* **A release after the last one the externals configuration lists takes the settings of that last release**, and the
  graph's record says so: the next Ensembl release is no longer built without external databases.
* **A selected database is, at every release, every database of Ensembl whose links continue it** (decision 104,
  :py:mod:`idtrack.build.continuity`, :py:meth:`idtrack.DatabaseManager.database_filing`). Ensembl renamed and refiled
  databases over the releases: RefSeq's mRNA accessions were ``RefSeq_dna`` up to release 63 (``RefSeq_mRNA`` from 64;
  its ncRNA accessions, in the same ``RefSeq_dna``, ``RefSeq_ncRNA``), its predicted ones ``RefSeq_dna_predicted``,
  the HGNC projections of pig and macaque genes ``HGNC_curated_gene`` and ``HGNC_automatic_gene`` up to 60, NCBI's
  gene ids lay on translations up to 62, Rfam's and miRBase's accessions on transcripts. A selection by today's names
  missed every earlier one (``RefSeq_dna``: asked at their release, RefSeq accessions were read at their latest release
  and carried back to another gene, a version Ensembl wrote then was answered ``unknown_version``, predicted ones were
  not found). The rule reads it from Ensembl's links (identifier as written, and gene): where, between two releases
  next to each other, at least half of the links of a database move to a selected one (held by the one and not the
  other at the first release, by the other and not the one at the next), the database that hands them over is that
  one's earlier name, or its later name; its identifiers go by their form to the selected database whose identifier
  scheme the form is (``NM_`` to ``RefSeq_mRNA``, ``NR_`` to ``RefSeq_ncRNA``), the others to its main continuation.
  A selected database's name on another form is judged the same way, per run of releases, and can be that database
  only (NCBI's ids on translations up to 62, Rfam's and miRBase's accessions on transcripts up to 62; the human HGNC
  symbols on GRCh37's transcripts are not Vega's gene names). A database that only mirrors a
  selected one (``WikiGene``, NCBI's ids at the same releases) moves no link and is not taken; nor are Ensembl's own
  transcript names that it filed at rhesus 70 and 71 under Rfam, miRBase and UniProt on transcripts (they move to
  Ensembl's transcript-name databases; the 450 it filed under EntrezGene at 70 to 72 are held, as NCBI's ids are
  selected on every form), nor the human symbols Ensembl displayed on a few mouse genes. An
  identifier of a selected database that has the form of another selected database's identifiers (Havana's
  ``OTTMUSG``), on a link that database holds at some release, is that database's: MGI gave the Havana ids of 312
  mouse genes without a symbol as their symbols at releases 48 to 53, and their versions were not read. The record
  lists what each selected database holds (``database_names``). Of a seeded sample of the RefSeq lines filed under
  ``RefSeq_dna`` and ``RefSeq_dna_predicted`` (1,000 each of those the round-2 audit found wrong, lost and right),
  every one now answers the gene Ensembl's cross references of its release give it, read at that release (pig,
  rhesus, mouse and human: 3,000 of 3,000 each). A database that ends at a boundary (it holds no link at the next
  release) is also the earlier name of the selected database whose links there the links it hands over are at least
  half of, however few of its own move (decision 107): human ``RefSeq_dna_predicted`` hands 474 and 1,540 of its 6,466
  links to ``RefSeq_mRNA_predicted`` and ``RefSeq_ncRNA_predicted`` from GRCh37's release 63 to 64 (Ensembl drops
  the rest), 59 % and 67 % of their links at 64, so its predicted accessions of releases 48 to 63 are read at their
  release (773 of the sampled lines: 12 were answered before, 693 not at all). On the five organisms this adds only
  hand-overs at release 63 to 64 (``RefSeq_dna`` to ``RefSeq_ncRNA`` in rhesus, mouse and human,
  ``RefSeq_dna_predicted`` to ``RefSeq_ncRNA_predicted`` in rhesus, mouse and human and to
  ``RefSeq_mRNA_predicted`` in human), each to a database that begins at 64 and held none of the links before; only
  the human ``RefSeq_dna_predicted`` ones change a graph (the others' accessions are filed so already by their form). The form a database is attached to (:py:meth:`idtrack.API.database_forms`)
  is that of its own links: an identifier that two databases hold (a RefSeq accession that is a gene's display name
  under ``RefSeq_dna``'s labels and a transcript's link under ``RefSeq_mRNA``) gave each database the form of all its
  links.

**The history**

* A mapping session whose events fit two pairs of releases equally well, the recorded one not among them, stops the
  build with an error that names the session and the pairs (no such session exists in the shipped organisms; the rule
  that took the nearer pair is gone). An event whose old version did not exist at the session's old release, while its
  stable id did, is placed with the version the id had there and listed in the record (human ``ENSG00000233776``,
  split at 107 to 108: its event to ``CHD2`` names version 3, the id had version 5); it was left out before.

**Organisms**

* Supported: human, mouse, pig, rhesus macaque (``macaca_mulatta``; assemblies Mmul_10, Mmul_8.0.1 and MMUL_1, keys
  ``10``, ``801`` and ``1``) and cynomolgus macaque (``macaca_fascicularis``; Macaca_fascicularis_6.0 and 5.0, keys
  ``6`` and ``5``). Ensembl names the MMUL_1 and Mmul_10 core databases alike (``macaca_mulatta_core_<release>_10``);
  MMUL_1 therefore has the key ``1``. See the Supported Organisms table in :doc:`usage`.
* Organisms are declared in one table, :py:data:`idtrack.organisms.ORGANISM_TABLE`; ``idtrack.organisms.register_organism`` adds or replaces
  one for the running session (see "Adding Organism Support" in :doc:`contributing`). Each organism declares short
  names (``"human"``, ``"mouse"``, ``"pig"``, ``"rhesus"``, ``"cynomolgus"``, ``idtrack.organisms.ALIASES``) that resolve
  without the network and win over Ensembl's other species of that name (``"pig"`` is also the common name of
  every pig breed in Ensembl). Assembly names (``"GRCm38"``) resolve to their organism.
* Organisms whose Ensembl ids carry no versions are not supported: building a graph for one raises a
  :py:class:`ValueError` that says so.
* A graph's window starts at the organism's first release in the table (``idtrack.organisms.first_release``: 48 for human, mouse and
  rhesus, 56 for pig, 91 for cynomolgus), for :py:class:`idtrack.build.DatabaseManager` and the API alike (previously 48 in the
  API, the selected assembly's first release in the manager).
* :py:meth:`idtrack.API.build` builds on the organism's assembly order; the primary assembly is the preferred one that has the
  snapshot release.
* The shipped external-database files of all five organisms include Ensembl release 116.
  ``ExternalDatabases.extend_modified_yaml`` extends a file to new releases (keeping a ``.bak`` of the file it
  replaces), and a graph build warns once about releases its file does not cover. ``Include`` must be a YAML boolean;
  entries for assemblies or releases that the organism table does not give the organism are left out with a warning
  (an error in a shipped file).

**Harmonization** (:py:class:`idtrack.harmonize.HarmonizeFeatures`)

* Each converted gene is one output feature, named by its Ensembl gene id at the target release; its name in the final
  database is a column (the first in sorted order, the others in ``<final database>_alternatives``). A gene without a
  name there keeps a missing value and status ``converted_no_final_id``: the final-database column never holds an
  Ensembl id (previously 58 % of pig rows converted to VGNC symbols had an Ensembl id there, unflagged). A feature is
  the gene its name is (the best answer as Ensembl genes), whether or not that gene has a name in the final database:
  it is never moved onto another gene that has one.
* Nothing is dropped silently: every output feature has ``var["idtrack_status"]`` and ``var["idtrack_reason"]``
  (statuses in :py:data:`idtrack.harmonize.STATUSES`). A dataset is converted on its own: an id that is n-to-1 in
  one dataset is no longer deleted from every dataset (previously 12 % of the counts of two pig datasets were lost this way).
  A feature that is a gene id existing at the target release is converted to itself (see "A gene id is its own answer"),
  not flagged ambiguous between its own later version and other genes its history reaches.
* New arguments: ``merge="sum"`` adds up the input features of one gene (``"flag"``, the default, keeps the feature
  that names the gene and flags the others), ``keep_unconverted``, ``from_release_by_dataset`` (the release each dataset was
  annotated with), ``from_database_by_dataset`` (the database each dataset's feature names are ids of), ``api`` (an
  :py:class:`idtrack.API` whose graph is loaded).
* New methods: ``feature_table``, ``output_plan``, ``conversion_report``; the attributes ``matchings``,
  ``feature_names``, ``gene_names``. Removed: ``get_idtrack_matchings_for_all_datasets``, ``dict_n_to_1``,
  ``unified_matching_dict``, ``multiple_ensembl_dict``, the ``removed_*``/``kept_*`` diagnostics,
  ``conversion_failed_identifiers``, ``debugging_variables``.
* The result cache is one JSON file per dataset (``idtrack_result_<project>_<dataset>.json``), written atomically, with
  a header of everything it depends on (idtrack version, organism, final database, releases, source database, graph);
  anything else is recomputed. Caches of 0.0.5 are recomputed once.
* ``organism`` is read from the graphs in the local repository when not given; ``snapshot_release``
  defaults to the newest local graph of the organism; ``final_database`` defaults to the organism's first symbol
  database (``idtrack.organisms.SYMBOL_DATABASES``: HGNC Symbol for human, MGI Symbol for mouse, VGNC Symbol for pig and the
  macaques). In 0.0.5 they defaulted to human and release 114.
* :py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas` loses no ``obs`` column by default (`#85
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

* Access modes, set with the environment variable ``IDTRACK_ENSEMBL_ACCESS`` or ``idtrack.settings.configure_ensembl(access=...)``: ``auto``
  (default: MySQL where its ports answer, the HTTPS dumps otherwise) and ``https`` (MySQL is never contacted). Both give
  the same tables, with the same values and column types. A MySQL port that fails is tried again after ten minutes;
  MySQL query errors are raised. See :ref:`ensembl-access`.
* A failed request is never read as "absent": a release that cannot be listed, or a table that cannot be downloaded
  after the retries, raises (:py:class:`idtrack.errors.EnsemblDownloadError`); previously such releases were dropped silently, and one
  failed request could select another primary assembly.
* Ensembl REST requests (organism names, the latest release) are retried on timeouts, connection and SSL errors and
  the HTTP statuses 429 and 5xx (``rest_attempts``, default 3), with the timeouts ``connection_timeout`` (15 s)
  and ``reading_timeout`` (60 s; also used by the HTTPS dumps) of :py:class:`idtrack.settings.EnsemblSettings`. After the last attempt
  :py:class:`idtrack.errors.EnsemblRestError` (a ``ConnectionError``) names the URL, the timeouts and how to raise them
  (`#84 <https://github.com/theislab/idtrack/issues/84>`_).
* :py:class:`idtrack.build.VerifyOrganism` asks Ensembl REST for the species list at most once per process and only when a
  look-up needs it; ``fetch_organism_and_latest_release`` takes no timeout arguments.
* Tables read from the HTTPS dumps are decoded as MySQL loads them (escaped tabs, newlines and backslashes), so such
  rows are no longer split or dropped. Building a pig graph from nothing takes 15 minutes (previously 73 minutes) and
  0.65 GB of disk (previously 6.9 GB).

**pandas 3** (`#86 <https://github.com/theislab/idtrack/issues/86>`_)

* idtrack works with pandas 2.3 and 3.x. 0.0.5 could not store a table with a text column under pandas 3, and
  ``DatabaseManager.create_ids`` then reported the misleading ``Table 'gene_stable_id' not found``. Only a schema answer
  (:py:class:`idtrack.errors.EnsemblSchemaError`: the release has no such table or column) leads to the older-schema fallbacks;
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
  snapshot is an older release (``API.build(organism, <release>, local_repository)``); see "Working with an older
  Ensembl release" in :doc:`usage`.

**Removed**

* ``API.build_graph``: the arguments ``calculate_caches`` (nothing is computed in advance) and ``assembly``
  (a graph is built on the organism's assembly order; a graph with another primary assembly was never built
  correctly). ``API.calculate_graph_caches``.
* ``Track.convert``: ``deprioritize_lrg_genes`` (LRG records follow the rule for copies above), ``reduction``,
  ``remove_na``, ``score_of_the_queried_item`` (the score of a path is the product of Ensembl's mapping scores).
  ``Track.version_info``, the graph attribute ``version_info``; ``EmptyConversionMetricsError`` (nothing raises it).
* ``TheGraph``: ``combined_edges``, ``combined_edges_genes``, ``combined_edges_assembly_specific_genes``,
  ``get_active_ranges_of_id``, ``node_trios``, ``lower_chars_graph`` and ``node_name_alternatives`` (use
  :py:meth:`idtrack.graph.TheGraph.combined`, :py:meth:`idtrack.graph.TheGraph.active_ranges`, :py:meth:`idtrack.graph.TheGraph.trios`,
  :py:meth:`idtrack.graph.TheGraph.resolve_name`); the node attribute ``is_latest``; ``GraphMaker.get_graph``'s
  ``overwrite_even_if_exist`` (a rebuild always replaces the file).
* ``DatabaseManager``: ``check_version_info``, ``version_fix``, ``version_fix_incomplete``, ``version_uniformize``,
  ``create_version_info``, ``get_db("versioninfo")``, ``available_releases_no_save``, ``tables_in_disk``,
  ``file_name``, ``create_id_history_fixed``, ``create_relation_archive`` and ``get_db("relationarchive")``,
  ``available_tables_mysql``, ``get_release_date``; ``GraphMaker.initialize_downloads`` and
  ``GraphMaker.update_graph_with_the_new_release`` (they only raised ``NotImplementedError``).
* The class ``DB`` with all of its attributes (above), among them ``main_assembly`` (use
  ``idtrack.organisms.assemblies_by_priority(organism)[0]``) and ``hyperconnecting_threshold`` (the threshold is
  :py:attr:`idtrack.settings.GraphSettings.hyperconnective_threshold`).
* The table cache functions of 0.0.5 that wrote one file per assembly (``export_disk``,
  ``read_exported``, ``repack_hdf5``).
* Members nothing used but the tests: ``TheGraph.is_frozen``, ``get_two_nodes_coinciding_releases``,
  ``get_external_database_nodes``, ``calculate_caches(for_test=...)``; ``GraphMaker.construct_graph_form`` and
  ``GraphMaker.split_id`` (:py:func:`idtrack.ids.split`); ``DatabaseManager.node_name_maker`` and ``node_dict_maker``
  (:py:func:`idtrack.ids.join`), ``available_releases_versions`` (:py:attr:`idtrack.build.DatabaseManager.available_releases`),
  ``create_database_content(just_download=...)``, ``create_external_all(return_mode=...)``; ``read_table_params``;
  ``PortReachability.reset`` and ``known``. The fields ``Database Index`` and ``Potential Synonymous`` of the externals
  YAML files (never read; files that still have them load as before).

**Other changes**

* ``Track`` raises an :py:class:`idtrack.errors.UnknownOrganismError` for a graph whose organism is not in the organism table (previously the
  assemblies were ordered by their numeric code); register the organism first.
* Without an explicit assembly, ``DatabaseManager`` picks the highest-priority assembly that has the release it is
  made for, inside ``first_release``/``snapshot_release`` (e.g. a mouse snapshot at release 100 uses GRCm38).
* ``DatabaseManager.create_available_databases`` lists the core databases of the organism's configured assemblies
  within their release ranges, the same in both access modes; ``DatabaseManager.get_db("generegion")`` tells which
  genes of a release lie outside the reference sequence.
* The self-tests of a graph (:py:class:`idtrack.checks.GraphChecks`): every check
  (``is_...``) returns a :py:class:`idtrack.checks.CheckResult` with ``holds`` (also its truth value) and
  ``details`` (what was compared and every violation found), where it returned ``True``, ``False`` or a tuple. A
  check does not raise for what it finds: a table that cannot be read or a release without identifiers is a result
  that does not hold, with the error in its details. A database that the graph does not hold is refused before
  anything is converted, by the checks and by ``history_travel_testing``: an :py:class:`idtrack.errors.InputError` that names the
  argument and lists the valid databases, as the API raises for a wrong argument (previously
  ``history_travel_testing`` returned a report in which every identifier was lost or failed, and the name of a
  form was the only name it refused). The checks compare assembly by assembly at every release an assembly has
  (previously they raised at the first release of an earlier assembly, or reported every graph with more than one
  assembly as inconsistent), report identifiers that only the graph has as well as those it lacks, and count a gene
  that has an id in a database and is not converted to it. ``random_dataset_source_generator`` returns ``None`` when
  nothing can be drawn (previously the tuple ``("", -1, -1)``), and
  ``history_travel_testing_random_arguments_generator`` raises an :py:class:`idtrack.errors.InputError` after 100 attempts.
