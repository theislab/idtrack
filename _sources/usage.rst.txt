==========
Quickstart
==========

Get started with IDTrack. This page provides minimal, copy-pasteable code examples to help you perform your first
identifier conversions, and says what the answers mean.

.. contents::
    :local:
    :depth: 2
    :backlinks: none

.. tip::

    For the complete learning path with detailed explanations, see :doc:`tutorials` (Parts 0-7).

.. note::

    An IDTrack graph is **release-aware** and **assembly-aware**. A graph covers every Ensembl release from the
    organism's first release in IDTrack to its *snapshot release*, on the organism's assemblies in their order (human
    NCBI36, GRCh37 and GRCh38), so identifiers of any of these releases and assemblies convert to one target: a
    release of the graph, on the assembly that Ensembl maintained at that release.


The Three-Step Workflow
-----------------------

IDTrack follows a simple three-step workflow for all identifier conversions:

.. list-table::
    :header-rows: 1
    :widths: 10 30 60

    * - Step
      - Action
      - Description
    * - 1
      - **Initialize**
      - Create an API instance with a local repository (the folder of downloaded tables and graph files)
    * - 2
      - **Load or build**
      - Load the graph of an organism and a snapshot release from the local repository, or build it if it is not
        there yet
    * - 3
      - **Convert**
      - Map identifiers with the loaded graph

Building a graph is done once per organism and snapshot release: it downloads Ensembl's tables of every release and
takes from about 6 minutes (cynomolgus) to 2 hours (human); see :ref:`time-and-memory`. Loading a graph that is on
disk takes 1 to 5 seconds (up to 7 for the first read of the file), needs no network, and holds 0.26 GB (cynomolgus)
to 1.14 GB (human) of memory.


Example 1: Convert a Human Gene Symbol
--------------------------------------

Convert a gene symbol to its Ensembl gene id:

.. code-block:: python

    import os
    from pathlib import Path

    import idtrack

    # Step 1: Set up your local repository
    # This stores downloaded tables and built graphs for reuse
    local_repo = Path(os.environ.get("IDTRACK_LOCAL_REPO", "./idtrack_cache")).resolve()
    local_repo.mkdir(parents=True, exist_ok=True)

    # Step 2: Initialize the API
    api = idtrack.API(local_repository=str(local_repo))

    # Step 3: Load (or, the first time, build) the graph of a pinned snapshot release.
    # api.resolve_organism("human") gives ("homo_sapiens", <Ensembl's newest release>) if you want to look them up.
    SNAPSHOT_RELEASE = 115
    api.build_graph(organism_name="homo_sapiens", snapshot_release=SNAPSHOT_RELEASE)

    # Step 4: Convert an identifier (to the snapshot release, unless `to_release` says otherwise)
    result = api.convert_identifier("TP53")
    target = result["target_id"][0] if result["target_id"] else None
    print(f"TP53 -> {target}")

**Expected output:**

.. code-block:: text

    TP53 -> ENSG00000141510.20

The target is the Ensembl gene id **with its version** at the target release. For the id without version, ask for
``final_database="base_ensembl_gene"`` (``['ENSG00000141510']``); for a symbol, for example
``final_database="HGNC Symbol"``.

.. note::

    Pin the snapshot release in your scripts. :py:meth:`idtrack.API.resolve_organism` returns Ensembl's newest
    release, which changes every few months: a script that builds at "the latest release" builds a new graph (a new
    download) after each Ensembl release, and its answers can change with it. A pinned snapshot loads from disk with
    the same answers on every run.


Example 2: Batch Conversion
---------------------------

Convert several identifiers with the same arguments at once:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")
    api.build_graph(organism_name="homo_sapiens", snapshot_release=115)

    # Convert a batch of gene symbols
    genes = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
    results = api.convert_identifier_multiple(genes, verbose=False)

    # Print results
    print("Gene Symbol -> Ensembl ID")
    print("-" * 30)
    for gene, result in zip(genes, results):
        target = result["target_id"][0] if result["target_id"] else None
        print(f"{gene:<10} -> {target}")

**Expected output:**

.. code-block:: text

    Gene Symbol -> Ensembl ID
    ------------------------------
    TP53       -> ENSG00000141510.20
    BRCA1      -> ENSG00000012048.27
    EGFR       -> ENSG00000146648.22
    KRAS       -> ENSG00000133703.15
    MYC        -> ENSG00000136997.22

The arguments are checked once, before any conversion; each distinct identifier is converted once. A search that
exceeds its budget (:py:class:`idtrack.SearchBudgetExceededError`) does not stop the batch: that item gets
``"error": {"type": ..., "message": ...}`` and no target.


Example 3: Working with Mouse
-----------------------------

IDTrack supports several organisms (see `Supported Organisms`_). Here's how to work with mouse:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")

    # Load the mouse graph
    api.build_graph(organism_name="mus_musculus", snapshot_release=115)

    # Convert mouse gene symbols
    mouse_genes = ["Trp53", "Brca1", "Egfr"]
    results = api.convert_identifier_multiple(mouse_genes, verbose=False)

    for gene, result in zip(mouse_genes, results):
        target = result["target_id"][0] if result["target_id"] else None
        print(f"{gene} -> {target}")

**Expected output:**

.. code-block:: text

    Trp53 -> ENSMUSG00000059552.14
    Brca1 -> ENSMUSG00000017146.13
    Egfr -> ENSMUSG00000020122.17

One :py:class:`idtrack.API` holds one graph: :py:meth:`idtrack.API.build_graph` for another organism replaces the
graph of the object. Use one object per organism to keep several loaded.


Example 4: Understanding Conversion Outcomes
--------------------------------------------

IDTrack is transparent about mapping ambiguity. Conversions can result in three outcomes:

.. list-table::
    :header-rows: 1
    :widths: 15 25 60

    * - Outcome
      - Meaning
      - Interpretation
    * - **1 -> 1**
      - Unique mapping
      - One target: ``target_id`` has one element
    * - **1 -> 0**
      - No mapping found
      - ``no_corresponding``: the identifier is not in the graph (unknown, a version that does not exist, or of a
        database the graph does not include); ``no_conversion``: it is in the graph, but no gene at the target release
        descends from it (a retired id)
    * - **1 -> n**
      - Multiple mappings
      - ``ambiguous``: several targets (a split gene, a name that two genes carry)

:py:meth:`idtrack.API.convert_identifier` uses ``strategy="best"`` by default: it returns the best answer, which is
one target, or, when several genes or targets are equal in all the evidence the graph holds, all of them in a fixed
order (by the mapping score of each to the gene the identifier was read as, then by gene id and target id) with
``"tie": True``. Nothing is chosen by position, and the same call on the same graph gives the same answer in every
run. ``strategy="all"`` returns every target reached, best first: the genes of the best answer lead, in the order in
which ``"best"`` returns them, and the others follow by their evidence. So ``result["target_id"][0]`` is the first
target of the best answer with either strategy; check ``result["tie"]`` or ``result["ambiguous"]`` before you take it
as the only one.

A tie is judged among the genes that can be named in the database you ask for. One result has one final database:
genes that have no id in it leave the result (``result["no_target_genes"]`` names them), and what remains is a tie or
not. A symbol that two databases give to two genes is therefore a tie when it is converted to Ensembl gene ids, and
one answer when it is converted to a symbol database in which only one of the two genes has a symbol.

Genes on alternative loci (haplotypes) and assembly patches, and LRG records, are copies of genes of the reference
sequence with stable ids of their own (human; mouse GRCm38). When an identifier leads to a gene of the reference
sequence and to such copies, the copies are not part of the answer: they are listed in
``result["alternative_locus_genes"]``. When only copies are reached, they are the answer. A queried Ensembl gene id
that exists at the target release is its own answer: its own version comes first, is never set apart as a copy, and is
the answer of ``strategy="best"`` whatever else its history reaches (``strategy="all"`` lists the other genes after
it).

Get the path behind an answer with ``explain=True``:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")
    api.build_graph(organism_name="homo_sapiens", snapshot_release=115)

    result = api.convert_identifier("TP53", explain=True)
    print("Targets:", result["target_id"])
    print("Read as:", result["graph_id"], "| match:", result["match"], "| at release", result["from_release"])
    print("Tie:", result["tie"], "| copies on alternative loci:", result["alternative_locus_genes"])
    for step in result["the_path"]:
        print(step)

**Expected output:**

.. code-block:: text

    Targets: ['ENSG00000141510.20']
    Read as: TP53 | match: exact | at release 115
    Tie: False | copies on alternative loci: ['LRG_321.1']
    {'ensembl_gene': 'ENSG00000141510.20', 'target_id': 'ENSG00000141510.20', 'entry_gene': 'ENSG00000141510.20', 'external_jump': 0, 'external_gap': 0, 'score': 1.0, 'edges': [['TP53', 'ENSG00000141510.20', 0, 115]]}

``the_path`` has one entry per pair of ``last_node``: the gene the identifier was read as (``entry_gene``), the number
of bridges through external ids that the search needed (``external_jump``; 0 means Ensembl's history alone), the
product of Ensembl's mapping scores along the history (``score``) and the edges walked. Every result is plain data
(``json.dumps`` works on it); :py:meth:`idtrack.API.convert_identifier` lists every field.


Example 5: Cross-Release Conversion
-----------------------------------

Convert identifiers between different Ensembl releases to handle legacy data:

.. code-block:: python

    import idtrack

    api = idtrack.API(local_repository="./idtrack_cache")
    api.build_graph(organism_name="homo_sapiens", snapshot_release=115)

    # An identifier from a table made with Ensembl release 100, converted to the snapshot release.
    # Retired, merged and split ids are followed through Ensembl's history.
    result = api.convert_identifier("ENSG00000141510", from_release=100)
    print("Release 100 -> Release 115:", result["target_id"])

**Expected output:**

.. code-block:: text

    Release 100 -> Release 115: ['ENSG00000141510.20']

**Working with an older Ensembl release.** A graph covers every release from the organism's first release in
IDTrack to its ``snapshot_release`` (:py:meth:`idtrack.API.graph_releases`; human 48 to 115), so the graph of the
latest snapshot already converts to and from any older release (``to_release=93``, ``from_release=93``); no older
graph is needed for that. Build a graph whose snapshot is an older release only to reproduce results as they were at
that release (the graph then knows nothing after it):

.. code-block:: python

    api.build_graph(organism_name="homo_sapiens", snapshot_release=93)

The graph file is named after the organism, the primary assembly and the window
(``graph_homo_sapiens_assembly38_min48_max93_narrow.npz``), so graphs of different snapshots live side by side in
one local repository.


.. _reading-an-identifier:

How an Identifier Is Read
-------------------------

**The starting release.** With ``from_release``, the identifier is read at that release. Without it, it is read at
the target release if it exists there, and otherwise at the latest release at which it exists; the result says which
release was used (``from_release``) and whether it was chosen for you (``from_release_inferred``). The rule is the
same for Ensembl ids, symbols and the ids of other databases. A symbol given without ``from_release`` is therefore
read as what it names at the target release (or at its latest release), not as what it named years ago. If the graph
does not hold the identifier at a given ``from_release`` (a wrong release, or an id of a database the graph does not
include at that release), :py:meth:`idtrack.API.convert_identifier` reads it by the rule without ``from_release`` and
says so: ``from_release_fallback`` is true, ``from_release_given`` is the release you gave. (The engine,
:py:meth:`idtrack.Track.convert`, raises :py:class:`idtrack.InactiveAtReleaseError` instead, naming the releases at
which the identifier exists.)

**Matching the string.** Surrounding white space is ignored. When a string can mean several nodes of the graph (a
symbol that was the name of one gene and is a synonym of another today), the one that exists at the release asked
about (``from_release``, without it the snapshot release) is taken; an Ensembl stable id is never passed over for a
synonym spelt the same. An identifier exists at a release when a database has it there on the genome assembly that
Ensembl maintained at that release: human GRCh37 is still listed beside GRCh38 with its names of release 75, and a
name that only GRCh37 lists today does not exist today. Among those, the string is matched exactly, then ignoring
letter case, then taking ``-``, ``_`` and ``.`` as the same character (``HLA_A`` is ``HLA-A``), then, for an id of a
sequence-accession database (RefSeq, UniProt, CCDS), without a version the graph does not hold (``NM_000546.9`` is
``NM_000546``); each step is tried on names and then on synonyms before the next. "Exactly" means as a database wrote
the identifier at the release it is read at: databases change the letter case of a name over the releases, and two
databases can write one name in two ways (at release 75 MGI wrote the mouse gene ``Bap1`` and UniProt wrote it
``BAP1``), so each of these spellings is an exact match where it was written, and is not taken for another gene's
synonym spelt the same (``BAP1`` with ``from_release=75`` is that gene; today, when no database writes the name so,
``BAP1`` is exactly the synonym of another mouse gene). The
graph keeps one node per identifier, named as at its last release, which is what ``result["graph_id"]`` shows.
Nothing else is ever cut off an external id (``SLA.7`` is not ``SLA``). ``result["match"]`` says how the string
matched (``"exact"``, ``"case"``, ``"separator"``, ``"version"``, each also as ``"synonym:..."``),
``result["graph_id"]`` which node it was read as, and ``result["from_databases"]`` the databases that attach the
identifier to its genes at the release it was read at (to a gene itself or to one of its transcripts or
translations). With ``from_database="HGNC Symbol"`` the string is read as an id of that database: it stands only for
the genes that HGNC attaches it to, and between those the evidence of every database still counts (a symbol that the
named database gives to two genes, of which other databases confirm one, is that one).

**Versions.** An Ensembl gene id without version stands for its version at the starting release. A version the graph
does not hold is not answered with another version's gene: the result has no target, ``no_corresponding`` is true and
``known_versions`` lists the versions that exist. The one exception is a version higher than every version the graph
holds of an id that is still active at the snapshot release: it may come from a release newer than the graph, so it
is read as the latest version (``ENSG00000141510.25`` as ``ENSG00000141510.20``) and ``match`` is
``"version_newer_than_graph"``. Transcript and translation ids are read like gene ids (with or without version) and
converted to their gene.

**Arguments.** Every argument is checked before the search starts: the identifier must be a non-empty string, the
releases integers of the graph (:py:meth:`idtrack.API.graph_releases`), ``final_database`` one of
:py:meth:`idtrack.API.valid_final_databases` for the target release (case is ignored), ``from_database`` one of
:py:meth:`idtrack.API.list_external_databases`. An invalid argument raises :py:class:`TypeError` or
:py:class:`ValueError` naming it and the valid values, for every identifier alike.


Environment Variable Configuration
----------------------------------

For convenience, set the ``IDTRACK_LOCAL_REPO`` environment variable to avoid specifying
the local repository in every script.

.. tab-set::

    .. tab-item:: Bash/Zsh

        .. code-block:: bash

            # Add to your ~/.bashrc or ~/.zshrc
            export IDTRACK_LOCAL_REPO="$HOME/.idtrack"

    .. tab-item:: Fish

        .. code-block:: fish

            # Persist across sessions
            set -Ux IDTRACK_LOCAL_REPO $HOME/.idtrack

    .. tab-item:: Windows PowerShell

        .. code-block:: powershell

            # Set user environment variable
            [Environment]::SetEnvironmentVariable("IDTRACK_LOCAL_REPO", "$env:USERPROFILE\.idtrack", "User")

Then in your Python code:

.. code-block:: python

    import os
    import idtrack

    # IDTrack does not read the variable itself: pass it on
    local_repo = os.environ.get("IDTRACK_LOCAL_REPO", "./idtrack_cache")
    api = idtrack.API(local_repository=local_repo)


Supported Organisms
-------------------

IDTrack supports the following organisms. Each is declared once, in the organism table
``idtrack.DB.assembly_mysqlport_priority``, and ships a default external-database configuration
(``idtrack/default_config/<organism>_externals_modified.yml``).

.. list-table::
    :header-rows: 1
    :widths: 18 22 40 20

    * - Organism
      - Identifier
      - Assemblies (key: name, Ensembl releases)
      - Symbol databases
    * - Human
      - ``"homo_sapiens"``, ``"human"`` or ``9606``
      - 38: GRCh38, 76+; 37: GRCh37, 55+ (maintained beside GRCh38 from 76); 36: NCBI36, 48-54
      - HGNC Symbol
    * - Mouse
      - ``"mus_musculus"``, ``"mouse"`` or ``10090``
      - 39: GRCm39, 103+; 38: GRCm38, 68-102; 37: NCBIM37, 48-67
      - MGI Symbol
    * - Pig
      - ``"sus_scrofa"``, ``"pig"`` or ``9823``
      - 111: Sscrofa11.1, 90+; 102: Sscrofa10.2, 67-89; 9: Sscrofa9, 56-66
      - VGNC Symbol, HGNC Symbol
    * - Rhesus macaque
      - ``"macaca_mulatta"``, ``"rhesus"`` or ``9544``
      - 10: Mmul_10, 98+; 801: Mmul_8.0.1, 86-97; 1: MMUL_1, 48-85
      - VGNC Symbol, HGNC Symbol
    * - Cynomolgus macaque
      - ``"macaca_fascicularis"``, ``"cynomolgus"`` or ``9541``
      - 6: Macaca_fascicularis_6.0, 103+; 5: Macaca_fascicularis_5.0, 91-102
      - VGNC Symbol, HGNC Symbol

The **key** of an assembly is what IDTrack uses everywhere (the ``genome_assembly`` argument of
:py:class:`idtrack.DatabaseManager`, node types such as ``assembly_801_ensembl_gene``, the table folders
``idtrack_tables/<organism>_assembly-<key>/``, the graph file names and the assembly keys of the externals YAML). It is
Ensembl's code from the core database name ``<organism>_core_<release>_<code>`` wherever that is unambiguous. Rhesus is
the exception: Ensembl names both MMUL_1 (releases up to 85) and Mmul_10 (from 98)
``macaca_mulatta_core_<release>_10``, so Mmul_10 keeps key ``10`` and MMUL_1 has the key ``1``; IDTrack tells them
apart by release. Releases before 48 are not available in IDTrack. Except for human (GRCh37 and GRCh38 overlap),
every organism has one assembly per release, and IDTrack links the assemblies through Ensembl's own history. A graph
converts on the assembly that Ensembl maintained at each release (human: GRCh38 from 76); GRCh37 ids after release
75 are in the human graph as ids to convert from.

Organism names are matched case-insensitively. Ensembl species names (also written with spaces, e.g.
``"Macaca mulatta"``), taxon IDs, the assembly names of the table (``"GRCm38"``) and the short names in the table
above resolve; a short name declared for an organism wins over Ensembl's other species of that common name (``"pig"``
is ``sus_scrofa``, not one of the pig breeds that Ensembl also lists). Another common name that several Ensembl
species share raises an error listing the candidates.

The symbol databases are the default final database of :py:class:`idtrack.HarmonizeFeatures` (the first one) and
the ones to try for gene symbols. In current Ensembl, macaque genes carry VGNC symbols on the newest assemblies
and HGNC symbols on earlier ones, and cynomolgus 6.0 has few symbols of either kind.

IDTrack reads Ensembl ids with their versions: an organism whose Ensembl ids carry no versions is not supported (a
graph build refuses it with a :py:class:`ValueError` that says so). The five organisms above all have versions.

.. tip::

    Want to add support for another organism? See "Adding Organism Support" in :doc:`contributing` for the
    recipe, and :doc:`_notebooks/02_prepare_new_external_yaml` for configuring its external databases.


Common Patterns
---------------

.. _reusing-built-graphs:

Reusing Built Graphs
~~~~~~~~~~~~~~~~~~~~

Once a graph is built, its file is in the local repository, and :py:meth:`idtrack.API.build_graph` loads it from
there:

.. code-block:: python

    api = idtrack.API(local_repository="./idtrack_cache")

    # Loaded from the graph file if it exists; built (and saved) only if it does not
    api.build_graph(organism_name="homo_sapiens", snapshot_release=115)

Loading reads the graph file alone: no table, no :py:class:`idtrack.DatabaseManager` and no network connection is
needed, so a graph on disk also loads where Ensembl cannot be reached. The file is one ``.npz`` archive of arrays and
JSON (no pickle: loading it executes no code). Its name, ``graph_<organism>_assembly<key>_min<first>_max<snapshot>_narrow.npz``,
says the organism, the primary assembly and the window; the file itself also records the selection of external
databases it was built from, the organism table, the IDTrack graph format and the package version, and the history
corrections made while building it (``api.track.graph.graph["manifest"]``). When the file under the expected name holds
another graph than the one asked for (it was built under another externals YAML, another organism table or by
another version of the graph format), loading raises :py:class:`idtrack.GraphMismatchError`, which names the
differences: IDTrack never answers silently from another graph, and never starts a build of hours without being asked.
To build the graph again, move the file away and call :py:meth:`idtrack.API.build_graph`, or replace it with
``idtrack.GraphMaker(api.get_database_manager("homo_sapiens", 115)).get_graph(create_even_if_exist=True)``.

Graphs written by IDTrack 0.0.5 (``graph_<organism>_min<first>_max<snapshot>_narrow.pickle``) and its table cache
(``<organism>_assembly-<key>.h5``) are not read: build the graphs again with this version.

Working with AnnData Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`idtrack.HarmonizeFeatures` converts the features of several ``.h5ad`` datasets to one Ensembl release and
merges them:

.. code-block:: python

    import idtrack

    harmonizer = idtrack.HarmonizeFeatures(
        project_name="my_project",
        data_h5ad_dict={"study_a": "study_a.h5ad", "study_b": "study_b.h5ad"},
        project_local_repository="./harmonization",   # result cache, one JSON file per dataset
        idtrack_local_repository="./idtrack_cache",   # the graph
        target_ensembl_release=115,
        organism_name="homo_sapiens",
        graph_last_ensembl_release=115,
        final_database="HGNC Symbol",
    )
    print(harmonizer.conversion_report())                         # one row per input feature, with status and reason
    adata = harmonizer.unify_multiple_anndatas(mode="union")      # or mode="intersect"

The constructor converts every feature (the long part; the results are cached). The rules of the output:

* Each converted gene is one feature, named by its Ensembl gene id at the target release; its name in the final
  database is in a column named after the database (a gene without one has a missing value there and status
  ``"converted_no_final_id"``: the final-database column never holds an Ensembl id).
* Nothing is dropped silently: every output feature has ``var["idtrack_status"]`` (``"converted"``, ``"merged"``,
  ``"unconverted_ambiguous"``, ``"unconverted_not_in_graph"``, ...) and ``var["idtrack_reason"]``. Unconverted
  features keep their input names; ``keep_unconverted=False`` (of
  :py:meth:`idtrack.HarmonizeFeatures.unify_multiple_anndatas` and
  :py:meth:`idtrack.HarmonizeFeatures.feature_harmonizer`) leaves them out.
* Several input features of one dataset on one gene (``merge=`` of the same two methods): ``merge="sum"`` adds them
  up (right for counts); ``merge="flag"`` (default) keeps the one that names the gene itself as the gene's feature and flags the others
  (``"unconverted_merge_partner"``).
* Each dataset is converted on its own: a dataset is harmonised the same alone and together with others. A feature
  that is a gene id existing at the target release is that gene.
  ``source_releases={"study_a": 100}`` reads a dataset's features at the release it was annotated with;
  ``source_databases={"study_a": "HGNC Symbol"}`` reads them as ids of that database (a symbol that another database
  gives to another gene is then that database's gene, not a tie). Nothing is inferred on your behalf. Without it, a
  symbol that two databases give to two genes is ``"unconverted_ambiguous"`` when both genes can be named in the
  final database; when only one of them has a name there, the feature is converted to that gene (a tie is judged
  among the genes that can be named in the final database, see Example 4).
* ``api=`` passes an :py:class:`idtrack.API` whose graph is already loaded.

:py:meth:`idtrack.HarmonizeFeatures.unify_multiple_anndatas` merges the harmonised datasets without losing or
rewriting anything by default: ``obs`` keeps every column of any dataset (``obs_join="outer"``; a dataset without a
column has missing values there; ``obs_join="inner"`` keeps only the common columns, and ``obs_columns_to_keep``
selects among them), every ``var`` column is kept (one column where the datasets agree on each feature, else one per
dataset, ``<column>_<handle>``), layers, ``obsm``, ``obsp`` and ``uns`` entries that every dataset has are carried
over, and ``X`` keeps its dtype. ``raw`` is dropped, because its features are not harmonised; keep raw counts in a
layer.

For complete AnnData harmonization workflows, see :doc:`_notebooks/05_tutorial_harmonization`.


.. _time-and-memory:

Time and Memory
---------------

Measured with this version for the five organisms, snapshot release 115, the window from each organism's first
release, the shipped externals YAML files, on 2 cores of a Linux server, Python 3.13 (a graph built from nothing over
HTTPS, so the build time is mostly download time and depends on the network):

.. list-table::
    :header-rows: 1
    :widths: 25 15 15 15 15 15

    * -
      - Pig
      - Rhesus
      - Cynomolgus
      - Mouse
      - Human
    * - Build from nothing
      - 15 min
      - 14 min
      - 6 min
      - 36 min
      - 1 h 55 min
    * - Peak memory of the build
      - 1.1 GB
      - 1.0 GB
      - 0.6 GB
      - 2.9 GB
      - 4.5 GB
    * - Tables on disk
      - 0.65 GB
      - 0.66 GB
      - 0.27 GB
      - 2.1 GB
      - 5.3 GB
    * - Graph file
      - 85 MB
      - 69 MB
      - 47 MB
      - 228 MB
      - 345 MB
    * - Load (file in the system's cache)
      - 1.3 s
      - 1.2 s
      - 0.9 s
      - 2.8 s
      - 4.3 s
    * - Memory after load
      - 0.38 GB
      - 0.34 GB
      - 0.26 GB
      - 0.76 GB
      - 1.14 GB
    * - 5,000 mixed conversions
      - 7.4 s
      - 5.9 s
      - 5.2 s
      - 14.2 s
      - 17.2 s
    * - Per conversion: median / 99th percentile
      - 0.24 / 21 ms
      - 0.22 / 17 ms
      - 0.19 / 13 ms
      - 0.26 / 50 ms
      - 0.25 / 67 ms

The first load of a file on a machine reads it from disk and took up to 7 s. The conversions are a fixed, seeded
mix of gene ids, ids without version, symbols and other external ids of every release of the window, converted to a
release of the graph and partly to an external database. Each conversion has a budget
(:py:attr:`idtrack.Track.search_budget_states`, 200,000 search states, and
:py:attr:`idtrack.Track.search_budget_seconds`, 30 s); a search past it raises
:py:class:`idtrack.SearchBudgetExceededError` (no conversion of the measurements came near it). The graph is only read
by conversions, so several threads can share one :py:class:`idtrack.API`, but they do not convert in parallel.


What's Next?
------------

Now that you've completed the quickstart, explore these resources for deeper understanding:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Resource
      - Description
    * - :doc:`_notebooks/00_idtrack_overview`
      - Understand the mental model (time axis, space axis, snapshot boundary)
    * - :doc:`_notebooks/01_installation_guide`
      - Detailed environment setup and configuration
    * - :doc:`_notebooks/03_initialization_graph`
      - Graph building, loading, and management
    * - :doc:`_notebooks/04_api_deep_dive_human`
      - The API in detail, with advanced examples
    * - :doc:`_notebooks/05_tutorial_harmonization`
      - Real-world dataset harmonization workflows
    * - :doc:`_notebooks/06_tutorial_humanization_mouse_pig_to_human`
      - Cross-species "humanization" workflows
    * - :doc:`tutorials`
      - Complete learning path (Parts 0-7)
