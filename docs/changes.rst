Changes
=======

This page lists the changes that affect users, newest first. Version numbers follow ``pyproject.toml``.


Unreleased
----------

**New organisms**

* Rhesus macaque (``macaca_mulatta``; assemblies Mmul_10, Mmul_8.0.1 and MMUL_1, keys ``10``, ``801`` and ``1``)
  and cynomolgus macaque (``macaca_fascicularis``; Macaca_fascicularis_6.0 and 5.0, keys ``6`` and ``5``) are
  supported like human, mouse and pig. Ensembl names the MMUL_1 and Mmul_10 core databases alike
  (``macaca_mulatta_core_<release>_10``); MMUL_1 therefore has the key ``1``. See the Supported Organisms table in
  :doc:`usage`.
* Organisms are declared in one table, ``DB.assembly_mysqlport_priority``; ``DB.register_organism`` adds or
  replaces one for the running session (see "Adding Organism Support" in :doc:`contributing`).
* The shipped external-database files of human, mouse and pig now include Ensembl release 116.
  ``ExternalDatabases.extend_modified_yaml`` extends a file to new releases, and a graph build warns once about
  releases its file does not cover.

**Access to Ensembl**

* New access modes, set with the environment variable ``IDTRACK_ENSEMBL_ACCESS`` or ``DB.ensembl_access``:
  ``auto`` (default: MySQL where its ports answer, the HTTPS dumps otherwise) and ``https`` (MySQL is never
  contacted). Each MySQL port is probed once per process instead of before every table; a port whose connection
  fails is not tried again in that process. An invalid value raises a ``ValueError`` when a ``DatabaseManager`` is
  created. See :ref:`ensembl-access`.
* MySQL port 3337 (Ensembl's GRCh37 archive) is read for human GRCh37 only, the one parallel assembly Ensembl
  maintains there. The archive also carries older assemblies forward as frozen copies under its later release
  numbers (rhesus MMUL_1 as ``macaca_mulatta_core_79_10`` to ``_99_10``, pig Sscrofa10.2 for 90 to 99, mouse
  GRCm38); these are not annotation of those releases and are no longer part of any graph. An assembly's releases
  are those of the main servers and of the HTTPS dumps, so a graph built over MySQL equals one built over HTTPS.
  Mouse GRCm38 and pig Sscrofa10.2 no longer list 3337, and ``DB.register_organism`` rejects 3337 for any other
  assembly (``DB.mysql_port_assembly``). A pig graph that was built over MySQL with an earlier version had
  Sscrofa10.2 edges at releases 90 to 99 from such copies; a rebuilt graph has none.
* Tables read from the HTTPS dumps are now decoded as MySQL loads them (escaped tabs, newlines and backslashes in
  fields), so such rows are no longer split or dropped.

**Dependencies**

* idtrack supports Python 3.12 to 3.15; Python 3.9, 3.10 and 3.11 are no longer supported (see pandas 3 below).
* Minimum versions are raised to releases that install on Python 3.12 and pass the test suite together: pandas 2.3
  (and 3.x), anndata 0.13, numpy 2.1, scipy 1.14, h5py 3.11 (the last three as anndata 0.13 requires), networkx 3.2,
  sparse 0.15, PyYAML 6.0.2, PyMySQL 1.1.1, tqdm 4.66.3, PySocks 1.7.1; optional: biopython 1.81, torch 2.2,
  transformers 4.36. See ``pyproject.toml`` for the reason of each.
* idtrack requires ``requests>=2.33.0`` and ``urllib3>=2.8.0``, the first releases without the vulnerabilities
  CVE-2026-25645 (requests) and CVE-2026-44431, -44432, -97687, -97688 and -97689 (urllib3).

**pandas 3**

* idtrack works with pandas 3 (tested with 2.3 and 3.1 on Python 3.13). The table cache wrote no table with a text column
  under pandas 3 (``Unsupported dtype <StringDtype(...)>``), and ``DatabaseManager.create_ids`` then reported the
  misleading ``Table 'gene_stable_id' not found``. Text columns of every pandas text dtype are cached now and come
  back in the default text dtype of the running pandas with ``NaN`` for missing values, which is what pandas gives
  for freshly read text; cache files written under pandas 2 are read unchanged. Cache files are written with the
  default HDF5 format bounds, so files written with HDF5 2.x stay readable by HDF5 1.14.
* Only a schema answer (``idtrack._db.EnsemblSchemaError``: the release has no such table or column) leads to the
  older-schema fallbacks (``<form>_stable_id`` tables, legacy ``identity_xref`` columns); any other error, such as a
  failed download or cache write, is raised as it is.
* Supported: Python 3.12 to 3.15 with pandas 2.3 or later (including 3.x) and anndata 0.13 or later. On Python 3.11
  no anndata release supports pandas 3 (the last ones for 3.11 require ``pandas<3``; older ones cannot write ``str``
  columns to ``.h5ad``).

**Network**

* Ensembl REST requests (organism names, the latest release) are retried on timeouts, connection and SSL errors and
  the HTTP statuses 429 and 5xx (``DB.rest_attempts``, default 3), with the timeouts ``DB.connection_timeout``
  (now 15 s) and ``DB.reading_timeout`` (now 60 s; also used by the HTTPS dumps). After the last attempt
  ``EnsemblRestError`` (a ``ConnectionError``; formerly a ``TimeoutError`` for timeouts and raw ``requests``
  exceptions otherwise) names the URL, the timeouts and how to raise them. See :ref:`ensembl-access`.
* ``VerifyOrganism`` asks Ensembl REST for the species list at most once per process and only when a look-up needs
  it; the formal name of a supported organism (also written with spaces) resolves without the network.
  ``fetch_organism_and_latest_release`` takes no timeout arguments.
* ``DatabaseManager`` saves which releases the HTTPS dumps hold in the local repository
  (``ensembl_https_index_<organism>_assembly-<key>.json``) and reads it for a window that ends at a release the
  saved scan reached, so loading a graph that is on disk, and ``HarmonizeFeatures`` with it, need no network in
  ``https`` mode.

**Older snapshots**

* ``GraphMaker`` explains, when a manager is not at the newest release of its window, how to build a graph whose
  snapshot is an older release (``API.build_graph(..., snapshot_release=<release>)``); see "Working with an older
  Ensembl release" in :doc:`usage`.

**Harmonization**

* ``HarmonizeFeatures.unify_multiple_anndatas`` loses no ``obs`` column by default (formerly it kept only those
  named in ``obs_columns_to_keep``): the new ``obs_join="outer"`` keeps every column of any dataset, missing (as a
  missing value of the column's kind; integers and booleans become nullable) in the datasets without it;
  ``obs_join="inner"`` keeps the columns every dataset has. ``obs_columns_to_keep`` selects among the joined
  columns; a name that is not among them raises ``ValueError`` naming it. Every input ``var`` column is kept: as one
  column where the datasets agree on each feature, otherwise once per dataset (``<column>_<handle>``). Layers,
  ``obsm``, ``obsp`` and shared ``uns`` entries are carried over, and ``X`` keeps its dtype (formerly ``float64``).
  ``raw`` is dropped with a warning.
* Values of ``obs`` and ``var`` are no longer rewritten: the strings ``"NA"``, ``"none"``, ``"-"``, ``"unknown"``
  and others stay as they are, numbers stay numbers, and missing values stay missing instead of becoming the
  placeholder ``_PLACEHOLDER_NA_``. Text columns become categorical. ``intersection`` is computed from presence.
  The list of "missing-like" strings used for that rewriting, ``idtrack._db.MISSING_VALUES``, is removed.
* ``mode`` and ``obs_join`` are validated before any work; a ``handle_anndata_key`` that is already an ``obs``
  column, colliding cell names and a generated ``var`` column name that an input column already has raise
  ``ValueError``. ``create_intersection_column_values`` is removed.
* ``feature_harmonizer`` returns the whole dataset re-indexed by the harmonised features (layers, ``obsm``,
  ``obsp``, ``uns``, ``varm``, the input ``var`` columns), not only ``X`` and ``obs``.

**Changed behaviour and removed attributes**

* ``DB.main_assembly`` (a human-only default, ``38``) is removed. Use the organism's preferred assembly,
  ``DB.assemblies_by_priority(organism)[0]``.
* ``HarmonizeFeatures`` no longer defaults to human:

  * ``organism_name`` is read from the graphs in the IDTrack local repository when not given; if they name no or
    several organisms, it must be passed.
  * ``graph_last_ensembl_release`` defaults to the newest local graph of the organism (else Ensembl's latest
    release) instead of the fixed release 114, so conversions that relied on the default can differ.
  * ``final_database`` defaults to the organism's first symbol database (``DB.symbol_databases``: HGNC Symbol for
    human, MGI Symbol for mouse, VGNC Symbol for pig and the macaques).
  * Result caches are recomputed once: their file name now carries organism, final database, target release and
    graph release (``idtrack_result_<project>_<dataset>_<organism>_<database>_rel<r>_graph<g>.pickle``). The old
    name held none of these, so it could return a result made with another configuration.

* ``API.resolve_organism`` / ``VerifyOrganism`` match case-insensitively and accept taxon IDs as integers or
  strings; a taxon ID shared by strains or breeds resolves to the species (``9823`` → ``sus_scrofa``). A common
  name that several Ensembl species share, such as ``"pig"``, raises a ``ValueError`` that lists the candidates;
  use the species name or the taxon ID.
* ``VerifyOrganism.name_synonyms_dataframe`` has a new ``kind`` column (``formal_name``, ``taxon_id`` or
  ``other``); synonyms are stored as lower-case strings, and the separate integer rows for taxon IDs are gone.
* ``Track`` raises a ``ValueError`` for a graph whose organism is not in the organism table (previously the
  assemblies were ordered by their numeric code); register the organism first.
* Without an explicit assembly, ``DatabaseManager`` now picks the highest-priority assembly that has a release
  inside ``ignore_before``/``ignore_after`` (e.g. a mouse snapshot at release 100 uses GRCm38).
* ``DatabaseManager.create_available_databases`` lists the core databases of the organism's configured assemblies
  within their release ranges, the same in both access modes.
* Removed ``DatabaseManager.create_id_history_fixed`` (deprecated; ``create_id_history`` handles the same cases) and
  ``DatabaseManager.create_relation_archive`` with its ``get_db("relationarchive")`` table (it read the incomplete
  ``gene_archive`` table; ``create_relation_current`` per release is what the graph uses). Neither was used.
* Removed methods that only raised ``NotImplementedError``: ``GraphMaker.initialize_downloads``,
  ``GraphMaker.update_graph_with_the_new_release``, ``DatabaseManager.available_tables_mysql`` and
  ``DatabaseManager.get_release_date``.
* The notebooks under ``reproducibility/experiments/prev/`` are frozen history (they are not part of the
  experiment index in ``reproducibility/experiments/README.md``) and are not updated; for example,
  ``prev/initialization_test.ipynb`` still uses ``DB.main_assembly``.
