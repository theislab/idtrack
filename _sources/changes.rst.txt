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
* Tables read from the HTTPS dumps are now decoded as MySQL loads them (escaped tabs, newlines and backslashes in
  fields), so such rows are no longer split or dropped.

**Dependencies**

* On Python 3.10 and later, idtrack requires ``requests>=2.33.0`` and ``urllib3>=2.8.0``, the first releases
  without the vulnerabilities CVE-2026-25645 (requests) and CVE-2026-44431, -44432, -97687, -97688 and -97689
  (urllib3). These releases do not support Python 3.9, where the earlier floors (``requests>=2.32.4``,
  ``urllib3>=2.5.0``) stay.

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
* Removed methods that only raised ``NotImplementedError``: ``GraphMaker.initialize_downloads``,
  ``GraphMaker.update_graph_with_the_new_release``, ``DatabaseManager.available_tables_mysql`` and
  ``DatabaseManager.get_release_date``.
* The notebooks under ``reproducibility/experiments/prev/`` are frozen history (they are not part of the
  experiment index in ``reproducibility/experiments/README.md``) and are not updated; for example,
  ``prev/initialization_test.ipynb`` still uses ``DB.main_assembly``.
