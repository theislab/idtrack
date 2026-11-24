.. _databases:

==================
External Databases
==================

Which external databases a graph holds, why the names that :py:meth:`idtrack.API.external_databases` and
:py:meth:`idtrack.API.final_databases` list are not the names of the externals configuration, which name to pass as
``final_database`` or ``from_database``, and how to change the selection.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


.. _selected-databases:

The selection a graph is built with
-----------------------------------

Ensembl's tables attach the identifiers of dozens of external databases to its genes, transcripts and translations.
A graph holds those of the **selected databases**: the entries with ``Include: true`` of the organism's **externals
configuration**, a YAML file of ``(form, database, assembly)`` entries with the releases at which Ensembl has each.
Every organism ships one, ``idtrack/default_config/<organism>_externals_modified.yml``, written from the organism's
template (``<organism>_externals_template.yml``, every database Ensembl has for it) by the allowlists of
:py:data:`idtrack.build.default_externals.DEFAULT_EXTERNALS`. The shipped selections follow one set for every
organism, with the species' own symbol databases:

.. list-table::
    :header-rows: 1
    :widths: 16 84

    * - Form
      - Selected (where the organism's template has them)
    * - gene
      - the species' symbol databases (``HGNC Symbol``, ``MGI Symbol``, ``VGNC Symbol``), NCBI's gene ids and symbols
        (``EntrezGene``, ``NCBI gene (formerly Entrezgene)``), ``UniProtKB Gene Name``, ``RFAM`` and ``miRBase`` with
        their labels, Havana, Vega and clone-based names, and the synonyms the symbol databases record
    * - transcript
      - ``CCDS`` with its labels, Havana transcripts, RefSeq mRNA and ncRNA, curated and predicted
        (``RefSeq_mRNA``, ``RefSeq_ncRNA``, ``RefSeq_mRNA_predicted``, ``RefSeq_ncRNA_predicted``)
    * - translation
      - Havana translations, RefSeq peptides (``RefSeq_peptide``, ``RefSeq_peptide_predicted``), UniProt
        (``Uniprot/SWISSPROT``, ``Uniprot/SPTREMBL``)

Not selected in any shipped configuration: RNAcentral, the name authorities' own ids (``HGNC:11998``, ``MGI:98834``,
``VGNC:...``), UniProt's secondary accessions, entry names and isoforms, and most databases of other kinds
(ArrayExpress, Reactome, ...). An identifier of a database the graph does not hold is not found, never another gene
(:ref:`limitation-databases`). Whatever the selection, the graph holds every gene's **display name** at every release
(Ensembl's own name of the gene there, :ref:`concept-ranking`), under whatever database Ensembl filed it.

A graph records the selection it was built with (:py:class:`idtrack.graph.GraphRecord`), and a graph file under the
expected name that was built with another selection is refused by :py:meth:`idtrack.API.build` with
:py:class:`idtrack.errors.GraphMismatchError`, which names the difference: a graph never answers for another.


.. _database-filing:

How Ensembl's names are filed under a selected database
-------------------------------------------------------

A selected database is named by what it is today, but Ensembl renamed and refiled databases over the releases. The
graph reads each selected database, at every release, as every database of Ensembl whose links continue it
(:py:mod:`idtrack.build.continuity`; :py:meth:`idtrack.DatabaseManager.database_filing`):

* RefSeq's mRNA and ncRNA accessions were ``RefSeq_dna`` up to release 63 (``RefSeq_mRNA`` and ``RefSeq_ncRNA`` from
  64), the predicted ones ``RefSeq_dna_predicted``; each accession goes by its form (``NM_`` to ``RefSeq_mRNA``,
  ``NR_`` to ``RefSeq_ncRNA``);
* NCBI's gene ids lay on translations up to release 62, Rfam's and miRBase's accessions on transcripts up to 62;
* the HGNC symbols projected onto pig and macaque genes were ``HGNC_curated_gene`` and ``HGNC_automatic_gene`` up to
  release 60.

An Ensembl database continues a selected one where, between two adjacent releases, at least half of its links (an
identifier and its gene) move to it (:py:data:`idtrack.build.continuity.CONTINUITY_SHARE`): the renames and refilings
of the five organisms move 82 to 100 % of a database's links, while a database that Ensembl keeps beside a selected
one moves at most 38 % into it at one boundary. What each selected database holds is in the graph's record:

.. code-block:: python

    from idtrack import API

    api = API.load(API.graph_path("homo_sapiens", 115, "./idtrack_cache"))
    for name, held in api.graph.record.database_names.items():
        if name.startswith("RefSeq_mRNA"):
            print(name, held)

.. code-block:: text

    RefSeq_mRNA {'RefSeq_dna': ((48, 63), (65, 65), (67, 68), (70, 70), (76, 76)), 'RefSeq_mRNA': ((64, 115),)}
    RefSeq_mRNA_predicted {'RefSeq_dna_predicted': ((48, 63), (76, 76)), 'RefSeq_mRNA_predicted': ((64, 115),)}


.. _database-names:

The names a graph lists
-----------------------

Three things make the names of a graph differ from the names of its configuration:

1. **One name means one kind of identifier.** Where Ensembl gives the labels and the accessions of a database one
   name, the labels take a name of their own, the same at every release: NCBI's symbols are ``NCBI gene (formerly
   Entrezgene)`` (``EntrezGene`` is NCBI's gene numbers), the labels of ``RFAM``, ``miRBase`` and ``CCDS`` are
   ``RFAM label``, ``miRBase label`` and ``CCDS label``. The graph's record gives these names
   (``record.label_names``, and ``record.renamed_databases`` for names of earlier releases that are no databases of
   the graph: passing one raises :py:class:`idtrack.errors.UnknownDatabaseError`, whose message names the graph's
   name).
2. **Display names are held under the name of their database's labels.** A gene's display name that no selected
   database gives it is held as a label of the database Ensembl filed it under: ``HGNC (curated)`` and ``HGNC
   (automatic)`` (HGNC's projections), ``UniProtKB/Swiss-Prot`` and ``UniProtKB/TrEMBL`` (UniProt's entry names),
   ``RefSeq DNA``, ``RFAM gene name``, ``miRBase gene name``, ``IMGT/GENE-DB``. They are not in the configuration, since
   they are held whatever it selects.
3. **Synonyms are read under their database's name.** The configuration selects the synonyms a database records as
   ``synonym_id::<database>``; a graph has no database of that name: ``from_database="HGNC Symbol"`` covers HGNC's
   synonyms, and a synonym is never a target (:ref:`reading-match`).

:py:meth:`idtrack.API.external_databases` lists every external database of the graph, at any release and on any
assembly: the values ``from_database`` accepts. :py:meth:`idtrack.API.final_databases` lists, for one target release,
the databases of the assembly that serves it, and the two Ensembl gene namespaces (``ensembl_gene``, the id with its
version, and ``base_ensembl_gene``, without): the values ``final_database`` accepts there (letter case ignored).
:py:meth:`idtrack.API.databases_by_assembly` and :py:meth:`idtrack.API.database_forms` say on which assembly and form
each lies. For human at 115:

.. code-block:: python

    print(len(api.external_databases()), "external databases;", len(api.final_databases(115)), "final databases at 115")
    print(sorted(set(api.external_databases()) - set(api.final_databases(115))))

.. code-block:: text

    35 external databases; 27 final databases at 115
    ['HGNC (automatic)', 'HGNC (curated)', 'IMGT/GENE-DB', 'RFAM gene name', 'RefSeq DNA', 'RefSeq peptide', 'UniProtKB/SpliceVariant', 'UniProtKB/Swiss-Prot', 'UniProtKB/TrEMBL', 'miRBase gene name']

Which name to pass:

.. list-table::
    :header-rows: 1
    :widths: 36 64

    * - To get
      - ``final_database``
    * - The Ensembl gene id with version (default) / without
      - ``ensembl_gene`` / ``base_ensembl_gene``
    * - The symbol of the organism's name authority
      - ``HGNC Symbol`` (human, and the HGNC symbols of pig and the macaques), ``MGI Symbol`` (mouse), ``VGNC Symbol``
        (pig, macaques): :py:data:`idtrack.organisms.SYMBOL_DATABASES`
    * - NCBI's gene id / NCBI's symbol
      - ``EntrezGene`` / ``NCBI gene (formerly Entrezgene)``
    * - RefSeq accessions
      - ``RefSeq_mRNA``, ``RefSeq_ncRNA``, ``RefSeq_peptide``, and their ``_predicted`` forms (``XM_``, ``XR_``, ``XP_``)
    * - UniProt accessions
      - ``Uniprot/SWISSPROT`` (reviewed), ``Uniprot/SPTREMBL`` (unreviewed)
    * - UniProt's gene names
      - ``UniProtKB Gene Name``

A gene can have no id in the database asked for: the result is then ``no_target`` (:ref:`concept-statuses`).


.. _changing-the-selection:

Changing the selection
----------------------

A file ``<organism>_externals_modified.yml`` in the local repository replaces the shipped selection for every graph
built in that repository; a graph built so records it and is refused under another. A selection is changed by
setting ``Include`` of entries of the template, by hand or with
:py:func:`idtrack.build.default_externals.apply_allowlist`, and written with
:py:func:`idtrack.build.externals.write_externals_yaml`. :doc:`_notebooks/06_prepare_new_external_yaml` walks through
it for each organism; :doc:`_notebooks/07_advanced_topics` keeps several selections (profiles) in their own local
repositories. Selecting more databases makes the graph larger and its build longer; selecting fewer leaves out
identifiers that would otherwise be found, and with them evidence that ranks candidates (support).

A release after the last one a configuration lists takes that last release's settings, and the graph's record says
so; ``ExternalDatabases.extend_modified_yaml`` (:py:class:`idtrack.build.ExternalDatabases`) extends a file to new
releases. The shipped files list Ensembl releases up to 116.
