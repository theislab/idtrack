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
      - **Local repository**
      - Make the folder of downloaded tables and graph files
    * - 2
      - **Build or load**
      - :py:meth:`idtrack.API.build` builds the graph of an organism and a snapshot release (or loads its file from
        the local repository when it is there); :py:meth:`idtrack.API.load` loads a graph file and never builds
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

    # Step 2: Build (the first time) or load the graph of a pinned snapshot release.
    # idtrack.API.ensembl_organism("human") gives ("homo_sapiens", <Ensembl's newest release>) if you want to look
    # them up (it asks Ensembl).
    SNAPSHOT_RELEASE = 115
    api = idtrack.API.build("homo_sapiens", SNAPSHOT_RELEASE, str(local_repo))

    # Step 3: Convert an identifier (to the snapshot release, unless `to_release` says otherwise)
    result = api.convert("TP53")
    targets = result.targets()  # the best answer; result.targets("all") for every target reached
    print(f"TP53 -> {targets[0] if targets else None}")

**Expected output:**

.. code-block:: text

    TP53 -> ENSG00000141510.20

The target is the Ensembl gene id **with its version** at the target release. For the id without version, ask for
``final_database="base_ensembl_gene"`` (``['ENSG00000141510']``); for a symbol, for example
``final_database="HGNC Symbol"``.

.. note::

    Pin the snapshot release in your scripts. :py:meth:`idtrack.API.ensembl_organism` returns Ensembl's newest
    release, which changes every few months: a script that builds at "the latest release" builds a new graph (a new
    download) after each Ensembl release, and its answers can change with it. A pinned snapshot loads from disk with
    the same answers on every run.


Example 2: Batch Conversion
---------------------------

Convert several identifiers with the same arguments at once:

.. code-block:: python

    import idtrack

    api = idtrack.API.build("homo_sapiens", 115, "./idtrack_cache")

    # Convert a batch of gene symbols
    genes = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
    results = api.convert_many(genes)  # progress=True shows a progress bar

    # Print results
    print("Gene Symbol -> Ensembl ID")
    print("-" * 30)
    for gene, result in zip(genes, results):
        targets = result.targets()
        print(f"{gene:<10} -> {targets[0] if targets else None}")

**Expected output:**

.. code-block:: text

    Gene Symbol -> Ensembl ID
    ------------------------------
    TP53       -> ENSG00000141510.20
    BRCA1      -> ENSG00000012048.27
    EGFR       -> ENSG00000146648.22
    KRAS       -> ENSG00000133703.15
    MYC        -> ENSG00000136997.22

The arguments are checked once, before any conversion; each distinct identifier is converted once. An error of one
identifier (an empty string or a missing value such as ``nan``, an internal name, a search that exceeds its budget,
:py:class:`idtrack.errors.SearchBudgetExceededError`, or any other) does not stop the batch: that item has
``status == "error"``, no candidate, and the error in ``result.error`` (:py:class:`idtrack.results.ErrorInfo`: its
type, message, and whether it is the caller's). An item that is another item of the batch with a suffix of a
single-cell tool (``SRF.1`` beside ``SRF``) is that item's copy (see "How an Identifier Is Read").
:py:meth:`idtrack.API.classify` sorts a batch into the bins 1→0, 1→1 and 1→n, and
:py:meth:`idtrack.API.classification_summary` writes the counts.


Example 3: Working with Mouse
-----------------------------

IDTrack supports several organisms (see `Supported Organisms`_). Here's how to work with mouse:

.. code-block:: python

    import idtrack

    # Load the mouse graph
    api = idtrack.API.build("mus_musculus", 115, "./idtrack_cache")

    # Convert mouse gene symbols
    mouse_genes = ["Trp53", "Brca1", "Egfr"]
    results = api.convert_many(mouse_genes)

    for gene, result in zip(mouse_genes, results):
        targets = result.targets()
        print(f"{gene} -> {targets[0] if targets else None}")

**Expected output:**

.. code-block:: text

    Trp53 -> ENSMUSG00000059552.14
    Brca1 -> ENSMUSG00000017146.13
    Egfr -> ENSMUSG00000020122.17

One :py:class:`idtrack.API` holds one graph (one organism, one snapshot release). Use one object per organism to keep
several loaded.


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
      - One target: ``result.targets()`` has one element (``result.status == "converted"``)
    * - **1 -> 0**
      - No mapping found
      - No candidate: ``status`` is ``"not_found"`` (the identifier is not in the graph: unknown, or of a database the
        graph does not include), ``"unknown_version"`` (a version that does not exist) or ``"no_conversion"`` (it is in
        the graph, but no gene at the target release descends from it, e.g. a retired id); ``result.reason`` says why
        (:py:data:`idtrack.results.REASONS`)
    * - **1 -> n**
      - Multiple mappings
      - Several targets (a split gene, a name that two genes carry); ``result.tie`` when they are equal in all evidence

:py:meth:`idtrack.API.convert` returns one :py:class:`idtrack.results.ConversionResult` with every candidate the search
reached (:py:attr:`~idtrack.results.ConversionResult.candidates`), ranked, the best marked
(:py:attr:`idtrack.results.Candidate.best`): ``result.view("best")`` and ``result.view("all")`` are two views of the
one search, and ``result.targets("best")`` / ``result.targets("all")`` their target ids. The best answer is one
target, or, when several genes or targets are equal in the evidence that decides, all of them in ascending order of
gene id and target id, with ``result.tie`` true. Nothing is chosen by position, and the same call on the same graph gives the same answer in every
run. In ``"all"`` the genes of the best answer lead and the others follow by their evidence, the targets of one gene
together. Each candidate carries its ranking record (final confidence, bridges, bridge gap, whether the identifier is
the display name of an entry gene, the support on it, score, being the query itself, steps, and the annotation
counts where they were compared), and ``result.decided_by`` names the criterion that put the best answer before the
runner-up (:py:data:`idtrack.results.CRITERIA`). Among the genes that carry a name at the release it is read at, the
gene whose display name it is there (Ensembl's own statement of the gene's name, ``gene.display_xref_id``, whatever
database it came from: ``Candidate.display_name``) ranks before the gene that more databases name with it
(``Candidate.entry_support``): rhesus ``ATP6V0C`` at 69 is ENSMMUG00000019094, the gene Ensembl displays it for (from
HGNC), not ENSMMUG00000005679, which NCBI and UniProt name with it (``decided_by="display_name"``). Two genes that
both display it are equal in it, and support decides between them as before (rhesus ``RPL31`` at 89, displayed for
two genes). The display name holds for the string as you type it: where it has letters of both cases, only a display
name that its database spelt so counts (mouse ``Foxl1`` at 70 is the gene MGI displays so, not the gene UniProt
displays as ``FOXL1``); a string in one letter case says nothing of case, and where the organism's symbols are written
in mixed case (mouse) the name authority's display name, read without case, ranks before another source's (mouse
``GAPDH`` is Gapdh; the gene UniProt displays as ``GAPDH`` follows it: ``Candidate.display_rank``). A symbol of a name
authority against another gene's synonym is decided before both, when the string is
read (below). ``result.status`` is ``"converted"``, ``"tie"``, ``"no_target"``,
``"not_found"``, ``"unknown_version"``, ``"no_conversion"`` or ``"error"``; ``result.reasons`` lists every reason that
applies, in this order: why a result has no target in the database asked for, that the answer was read at another
release than the one you gave (``"read_at_another_release"``, below: :ref:`reading-an-identifier`), and that a version
written after an accession was not judged (``"version_not_judged"``); ``result.reason`` is the first of them.

With ``final_database`` the answer is first a gene: the genes are ranked as Ensembl genes, and a target is an id that
the database attaches at the target release to a gene of that best answer. Where no gene of the best answer has one,
the result is the genes themselves (``status="no_target"``, ``reason="no_id_in_final_database"``), never the id of a
gene of a lower rank: pig ``PSME1`` is ENSSSCG00000002006 (its display name), which has no VGNC symbol, so
``convert("PSME1", final_database="VGNC Symbol")`` answers that gene, not ``DCAF11``, the symbol of another gene that
UniProt names PSME1. A tie is judged among the genes that can be named in the database you ask for. One result has
one final database: genes of a tie that have no id in it leave the result (``result.no_target_genes`` names them), and
what remains is a tie or not. A symbol that two databases give to two genes is therefore a tie when it is converted to
Ensembl gene ids, and one answer when it is converted to a symbol database in which only one of the two genes has a
symbol.

Genes on alternative loci (haplotypes) and assembly patches, and LRG records, are copies of genes of the reference
sequence with stable ids of their own (human; mouse GRCm38). When an identifier leads to a gene of the reference
sequence and to such copies, the copies are not part of the answer: they are listed in
``result.alternative_locus_genes``. When only copies are reached, they are the answer. A queried Ensembl gene id
that exists at the target release is its own answer: its own version comes first, is never set apart as a copy, and is
the best answer whatever else its history reaches (``result.view("all")`` lists the other genes after it).

Get the path behind an answer with ``explain=True``:

.. code-block:: python

    import idtrack

    api = idtrack.API.build("homo_sapiens", 115, "./idtrack_cache")

    result = api.convert("TP53", explain=True)
    print("Targets:", list(result.targets()))
    print("Read as:", result.graph_id, "| match:", result.match, "| at release", result.from_release)
    print("Tie:", result.tie, "| copies on alternative loci:", list(result.alternative_locus_genes))
    best = result.view("best")[0]
    for index in best.path + best.final_path:
        step = result.path.steps[index]
        print(step.kind, step.source, "->", step.target, "| release", step.release, "|", ", ".join(step.databases))

**Expected output:**

.. code-block:: text

    Targets: ['ENSG00000141510.20']
    Read as: TP53 | match: exact | at release 115
    Tie: False | copies on alternative loci: ['LRG_321.1']
    entry TP53 -> ENSG00000141510.20 | release 115 | HGNC Symbol, NCBI gene (formerly Entrezgene), UniProtKB Gene Name

``result.path`` is an :py:class:`idtrack.results.ExplainedPath`: one graph per conversion, with every node once
(:py:class:`idtrack.results.PathNode`: its kind, label, databases, active releases, the assemblies that serve them,
the releases at which it lies outside the reference sequence, and for a gene where it lies: assembly, sequence region,
first and last base, strand and releases) and every step once (:py:class:`idtrack.results.PathStep`: its kind
(``entry``, ``history``, ``self_loop``, ``bridge_near``, ``bridge_far``, ``final``, ``identity``), the releases,
Ensembl's mapping score, the splits and merges of a history event, where the event's later gene lies relative to the
earlier one (``places``: ``places_apart`` where Ensembl's history moves a gene to another place (on one assembly, or
across an assembly change to a chromosome of another name), which the walk follows), the databases and the assembly
of a cross-reference, and why the step was taken); each candidate lists the indices of its steps (``path`` to the
gene, ``final_path`` from the gene to the target). One kind of Ensembl's history steps is not followed: a step between
a gene of the mitochondrial genome and a gene of a nuclear sequence is no lineage (at three assembly changes Ensembl
gave the ids of mitochondrial genes to nuclear genes: rhesus ``MT-ND1`` from release 97 led to an unnamed gene of
chromosome 8). The kinds of the two places decide (one on the mitochondrial genome, the other on a chromosome or a
scaffold), also across an assembly change where the places cannot be compared. The versions on the two sides of such
a step are two genes that share a stable id by Ensembl's reuse: their places are compared side by side, and no id makes
one the other (a bridge between the two sides is refused, ``"mitochondrial_genome"``), so the mitochondrial gene's name
leads to the mitochondrial gene of the new assembly. The id then has no successor across that step, and ``result.path.refused`` lists the step with the rule
that refused it (``reason == "mitochondrial_genome"``). Every result is a frozen
dataclass with ``to_dict()``, strict JSON (``json.dumps(result.to_dict(), allow_nan=False)`` works), and
``ConversionResult.from_dict`` reads it back.

**Bridges.** Where Ensembl's history leads to no gene at the target release, the search may pass from a gene to
another through an id they carried one after the other (a *bridge*; ``Candidate.bridges`` counts them). A bridge is
taken only where it is evidence that one gene became the other: never through a synonym (it names other genes too
often); where both genes lie on the reference sequence of one assembly, only when their places overlap there (same
sequence region and strand); where their places cannot be compared (an assembly change lies between them), only when
no release gave the id to both (an id two living genes carry at once names both, and the end of one says nothing
about the other). Across an assembly change for which Ensembl publishes a mapping of the two assemblies (human,
mouse, pig Sscrofa9 to Sscrofa10.2), the earlier place is carried onto the later assembly by it and the two places
must overlap, as on one assembly; where none is published, two genes that lie on chromosomes of different names (or
one on the mitochondrial genome, the other on a chromosome) lie apart, since the assemblies of one organism name its
chromosomes alike from one to the next (by its karyotype): such a bridge is refused (``places_apart``). An Rfam accession names a
family of RNA genes, not a gene, and is never a bridge (:py:data:`idtrack.settings.NON_BRIDGING_DATABASES`). A gene
on a patch or a haplotype stands for the reference region its assembly exception replaces (Ensembl's
``assembly_exception``): a gene outside that region lies apart from it. Inside it, and for an LRG record (which has no
exception), a copy bridges only through an id it carried together with the other gene at one release, as it
does with the gene it is a copy of (an id that came to another gene after the copy's end is no evidence), and only
where, at each such release, the genes that carry the id are one gene (:py:meth:`idtrack.graph.TheGraph.count_genes`):
a family label or an alias that genes elsewhere carry too does not say which gene a copy stands for
(``copy_id_of_several_genes``; see :ref:`current-limitations`). ``result.path.bridges`` lists every bridge
the search considered (:py:class:`idtrack.results.BridgeCheck`: both genes, where they lie relative to each other,
whether it was taken, and each id between them with its rule, :py:data:`idtrack.results.BRIDGE_RULES`), also when
nothing was found (``result.reason`` is then ``no_path_to_target``).


Example 5: Cross-Release Conversion
-----------------------------------

Convert identifiers between different Ensembl releases to handle legacy data:

.. code-block:: python

    import idtrack

    api = idtrack.API.build("homo_sapiens", 115, "./idtrack_cache")

    # An identifier from a table made with Ensembl release 100, converted to the snapshot release.
    # Retired, merged and split ids are followed through Ensembl's history.
    result = api.convert("ENSG00000141510", from_release=100)
    print("Release 100 -> Release 115:", list(result.targets()))

**Expected output:**

.. code-block:: text

    Release 100 -> Release 115: ['ENSG00000141510.20']

**Working with an older Ensembl release.** A graph covers every release from the organism's first release in
IDTrack to its ``snapshot_release`` (:py:meth:`idtrack.API.releases`; human 48 to 115), so the graph of the
latest snapshot already converts to and from any older release (``to_release=93``, ``from_release=93``); no older
graph is needed for that. Build a graph whose snapshot is an older release only to reproduce results as they were at
that release (the graph then knows nothing after it):

.. code-block:: python

    api_93 = idtrack.API.build("homo_sapiens", 93, "./idtrack_cache")

The graph file is named after the organism, the primary assembly and the window
(``graph_homo_sapiens_assembly38_min48_max93.npz``), so graphs of different snapshots live side by side in
one local repository.


.. _reading-an-identifier:

How an Identifier Is Read
-------------------------

**The starting release.** With ``from_release``, the identifier is read at that release. Without it, it is read at
the target release if it exists there, and otherwise at the latest release at which it exists; the result says which
release was used (``from_release``) and whether it was chosen for you (``from_release_inferred``). The rule is the
same for Ensembl ids, symbols and the ids of other databases. A symbol given without ``from_release`` is therefore
read as what it names at the target release (or at its latest release), not as what it named years ago. One release
decides both what a string means and where it is read: ``RNR1`` with ``to_release=90`` is read at 90, where it is
NCBI's symbol of MT-RNR1 (and a synonym of NR4A2), exactly as with ``from_release=90``. What a string meant at other
releases only is not its answer, but the result reports it: ``result.other_readings`` lists such a reading with
``reason="other_release"``, the gene it led to and the releases at which the string had that meaning (``RNR1`` without
a release is NR4A2's synonym, and MT-RNR1's symbol of releases 86 to 93 and 95 is reported); give ``from_release`` to read it
there. If the graph does not hold the
identifier at a given ``from_release`` (a wrong release, an id of a database the graph does not include at that
release, a name of an early release whose source the graph lacks), :py:meth:`idtrack.API.convert` reads it by the rule
without ``from_release`` and says so: ``from_release_fallback`` is true, ``from_release_given`` is the release you
gave, and ``result.reasons`` holds ``"read_at_another_release"`` (the ``reason`` of a result with targets, whose
status stays ``"converted"`` or ``"tie"``; after the reason a result without targets has). ``from_release`` is the release the string was read at, also in a result without targets. Such an answer is what the identifier meant where it was read, which can be another gene than the one it
meant at the release you gave: :py:meth:`idtrack.API.classify` puts it in a bin of its own
(``read_at_another_release``), and the harmonizer says so in ``idtrack_reason``. (The engine,
:py:meth:`idtrack.engine.Track.convert`, raises :py:class:`idtrack.errors.InactiveAtReleaseError` instead, naming the
releases at which the identifier exists.)

**What is read.** Before anything is looked up, the string is read as a person who pasted it meant it, and the result
says so (:py:func:`idtrack.reading.read_identifier`):

* surrounding white space is ignored;
* invisible and control characters are removed (a byte-order mark at the start of a file, zero-width spaces and
  joiners, bidirectional marks, soft hyphens, control characters), compatibility forms are read as their plain
  characters (NFKC: full-width ``ＴＰ５３`` is ``TP53``) and Unicode's dashes (its category of dash punctuation and
  the minus sign) as ``-`` (``HLA–A`` is ``HLA-A``); the match is then prefixed ``unicode:``;
* quotation marks around the identifier (Unicode's initial and final quotes, the ASCII apostrophe and quotation
  mark) and the punctuation that ends a sentence, a clause or an item of a list at its end (Unicode's
  Terminal_Punctuation: ``;``, ``,``, ``.``, ``:``, ``!``, ``?``) are removed (``'Agt'``, ``ADORA3;`` and ``CCDC39.``
  are ``Agt``, ``ADORA3`` and ``CCDC39``); the match is prefixed ``punctuation:``. A quotation mark at one end only
  stays (``HSP70B'`` is a name), as do other punctuation (``IGH@``) and brackets;
* GENCODE's ``<id>_PAR_Y`` (the Y copy of a gene or transcript of a pseudoautosomal region) is read as the id; the
  match is prefixed ``par_y:`` and the Y copy, where Ensembl gives it an id of its own (release 110 on), is listed in
  ``alternative_locus_genes``;
* a spelling of a missing value (the strings pandas reads as missing by default, such as ``nan``, ``NA``, ``N/A``,
  ``None``, ``null``, ``<NA>`` and ``#N/A``, and pandas' ``NaT`` and UniProt's placeholder ``unknown``, in any letter
  case: :py:data:`idtrack.reading.MISSING_VALUES`) is refused with
  :py:class:`idtrack.errors.MissingValueError`, unless ``from_database`` names the database whose identifier it is
  (``NA`` is a synonym of a few genes, ``NaN`` an old one of SCN11A): pandas writes ``nan`` for a missing cell of a
  column converted to text, and a missing value is never a gene. In a batch (:py:meth:`idtrack.API.convert_many`,
  :py:meth:`idtrack.API.source_inference`) a cell that holds no value as Python, NumPy and pandas hold it (``None``,
  a float NaN, ``pd.NA``, ``NaT``: what ``df[column]`` gives for an empty cell before a cast to text) is that item's
  missing value too, and the rest of the batch is answered; any other value that is not a string refuses the batch;
* the graph's bookkeeping is no identifier: a string that begins with ``synonym_id::`` or is an Ensembl id with the
  version ``Void`` or ``Retired`` is refused with :py:class:`idtrack.errors.InvalidArgumentError`;
* a string longer than any identifier (:py:data:`idtrack.reading.MAX_IDENTIFIER_LENGTH`, 1,000 characters) and an
  Ensembl version of more digits than Ensembl stores (:py:data:`idtrack.reading.MAX_VERSION_DIGITS`, 5) are refused
  with :py:class:`idtrack.errors.InvalidArgumentError`.

Forms that are not read stay not found (never another gene): database prefixes (``HGNC:11998``, ``UniProtKB:P04637``),
Greek letters (``IL-1β``), UniProt isoforms (``P04637-2``), NCBI gene ids written as numbers of a spreadsheet
(``7157.0``), and version forms other than ``.<version>`` (``ENSG00000141510-5``, ``ENSG00000141510_5``,
``ENSG00000141510.v5``: a ``-`` or ``_`` and digits after an Ensembl id is as often a tool's suffix for a repeated
name as a version, so it is no version).

**Matching the string.** When a string can mean several nodes of the graph (a
symbol that was the name of one gene and is a synonym of another today), the one that exists at the release asked
about (``from_release``; without it the target release) is taken; where none exists there, the string is read at the
latest release at which one of them does, by the steps of that release. An Ensembl stable id is never passed over for
a synonym spelt the same. An identifier exists at a release when a database has it there on the genome assembly that
Ensembl maintained at that release: human GRCh37 is still listed beside GRCh38 with its names of release 75, and a
name that only GRCh37 lists today does not exist today. Among those, the string is matched exactly, then ignoring
letter case, then taking ``-``, ``_`` and ``.`` as the same character (``HLA_A`` is ``HLA-A``), then, for a
versioned accession, with the version written (below: **Versions**); each step is tried on names and then on synonyms
before the next. "Exactly" means as a database wrote
the identifier at the release it is read at: databases change the letter case of a name over the releases, two
databases can write one name in two ways (at release 75 MGI wrote the mouse gene ``Bap1`` and UniProt wrote it
``BAP1``), and one database can give two genes one name in two cases (``Tc`` is a synonym of one mouse gene, ``tc``
one of Noto), so each of these spellings is an exact match where it was written, and is not taken for another gene's
synonym spelt the same (``BAP1`` with ``from_release=75`` is that gene exactly; today no database writes the name so,
and ``BAP1`` is exactly a synonym of another mouse gene, but as a string in one letter case it is first read as a
mouse symbol without letter case, below, and answers ``Bap1``). The graph keeps one node per identifier, named as at
its last
release, which is what ``result.graph_id`` shows, and the spelling of every link of it (the identifier, an Ensembl id,
a database, a release): a string matched exactly is followed only through the links spelt as it is typed (``Tc`` is
the first gene's synonym alone, ``tc`` Noto's), a string matched only by letter case through all of them (``TC`` is
both).
Nothing else is ever cut off an external id (``SLA.7`` is not ``SLA``). Ensembl's own ids are written in capitals
only, so an id in another letter case is found by case (``enst00000016171`` is ``ENST00000016171``).

Four rules decide between readings of one string, and the result reports every other reading it set aside or joined
(``result.other_readings``, :py:class:`idtrack.results.OtherReading`: the node, its match, why, and the genes it
would answer):

* **Symbols without letter case first, where symbols are written in mixed case.** MGI writes mouse symbols with a
  capital and then small letters (``Trp53``); the graph says so (:py:meth:`idtrack.graph.TheGraph.symbol_case`: most
  of the organism's symbols have a small letter). There, a string in one letter case (``AR``, ``CCK``, ``nos2``) that
  is not exactly a name of the release is read first as a symbol without letter case where that names exactly one
  gene: ``AR`` is ``Ar``, not the three
  genes whose synonym ``AR`` is (the exact synonym is reported among ``other_readings``,
  ``reason="symbol_without_case"``). With ``from_database`` naming another database the rule is off.
* **Names made unique by single-cell tools.** Tools append ``-1``, ``.1`` or ``_1`` to repeated feature names. In a
  batch (:py:meth:`idtrack.API.convert_many`, and so in the harmonizer), an item that is another item of the batch
  with such a suffix, and is itself no exact name where it is read alone (at ``from_release`` where the graph holds
  it there, else where :py:meth:`idtrack.API.convert` reads it), is that item's copy: its result is the item's, its
  match prefixed ``copy:`` (``SRF.1`` beside ``SRF``; ``IGHV3-30-2`` beside ``IGHV3-30`` stays a name of its own, and
  so does pig ``SLA-10``, a name of release 68, beside ``SLA``). Every other item of a batch is answered as a single
  conversion answers it. An Ensembl id with such a suffix (``ENSG00000141510-1`` beside ``ENSG00000141510``) is
  neither a copy nor a version: it stays not found, never another gene. A versioned accession written with a version
  (``NM_000546.5`` beside ``NM_000546``) is that version, read as it is alone, never a copy.
  In any call, such a string is not read, with its separators taken as one or without letter case, as a synonym of
  another gene than the one an authority's symbol written before the suffix names: ``SRF.1`` alone is not the synonym
  ``SRF-1`` of SPATA6 (the reading is reported, ``reason="make_unique_suffix"``, and ``SRF.1`` is not found). A name
  is still read so (``MIR138_2`` is ``MIR138-2``).
* **A separator reading of several names that exist at one release is one answer.** Where ``-``, ``_`` and ``.``
  read as one reach several names of the release read, their answers join, the second reading reported
  (``reason="same_step"``), and are ranked as the genes of one conversion: the readings whose best answers are equal
  in the evidence of their lineage (final confidence, bridges, their gap, support) tie, and a reading whose answer
  needs a bridge does not tie with one that needs none. Readings that never exist at one release are not joined:
  mouse ``AY172335-1`` is a name at releases 54 to 56 and ``AY172335.1`` at 57 to 60, so ``AY172335_1`` without a
  release is read at 60, as ``AY172335.1``, alone.
* **Two ids that share a number.** A string of digits is read in every database whose ids are numbers (NCBI gene,
  numeric Havana and RefSeq ids of early releases); genes that only other numbering schemes attach it to are not
  ranked below each other by how many databases attach it: they tie. Each candidate names the databases it was read
  through (``Candidate.entry_databases``). Written with a version, such a number is read by the database that has it
  with that version at the release read: human ``2633.1`` with ``from_release=51`` is the Havana translation number
  of releases 50 and 51 (ENSG00000116783). Where no database has it so at that release (``2633.1`` without a release),
  the version reading is taken where it exists, and the other reading, ``2633`` as an id of the database that writes
  no version (NCBI's gene id of GBP1), is reported beside it (``reason="unversioned_database"``).

``result.match`` says how the string matched: a step (``"exact"``, ``"case"``, ``"separator"``, ``"version"``,
``"version_newer_than_graph"``, ``"version_not_judged"``: :py:data:`idtrack.results.MATCH_STEPS`) after the prefixes that apply
(``"copy:"``, ``"unicode:"``, ``"punctuation:"``, ``"par_y:"``, ``"lrg:"``, ``"synonym:"``:
:py:data:`idtrack.results.MATCH_PREFIXES`), for example ``"punctuation:synonym:exact"``. ``result.graph_id`` is the
identifier it was read as (a synonym written as it is; the match says it is a synonym), and ``result.from_databases``
the databases that attach the identifier to its genes at the release it was read at (to a gene itself or to one of
its transcripts or translations; a synonym's database by its own name). With ``from_database="HGNC Symbol"`` the
string is read as an id of that database (its names; where none of its names matches at any release, the synonyms
it records): it stands only for the genes that HGNC attaches it to, and between those the evidence of every database still counts (a symbol that the named database
gives to two genes, of which other databases confirm one, is that one). The synonyms a database records are no
target: ``final_database`` is a database's ids.

**Copies and records.** Copies on alternative loci and patches are listed apart (``alternative_locus_genes``) beside a
gene of the reference sequence. So is the Y copy of a pseudoautosomal gene (a gene of a pseudoautosomal region of Y)
beside the gene at its place on X, the region of Y read on the region of X it repeats (from release 110 Ensembl gives
the Y copies ids of their own: ``PLCXD1`` is the X gene, the Y copy listed apart; the regions are GRC's,
:py:data:`idtrack.organisms.PSEUDOAUTOSOMAL_REGIONS`; one rule of genes and places, :py:mod:`idtrack.graph.genes`). An LRG record's id
(``LRG_321``) answers the gene the record describes (TP53), the match prefixed ``lrg:`` and the record listed apart;
so does every id of the record: its transcripts and translations (``LRG_321t1``, ``LRG_321p1``) and the record's gene
id of the GRCh37 releases (``LRG_7_g1``).

**Scores.** Ensembl writes the score 0 where it measured nothing between two ids; such a step is no score
(``PathStep.weight`` is ``None``), and a candidate whose path has one ranks below every candidate whose scores are
measured (``Candidate.unscored_steps``).

**Versions.** An Ensembl gene id without version stands for its version at the starting release. A version the graph
does not hold is not answered with another version's gene: the result has no candidate, ``status`` is
``"unknown_version"`` and ``known_versions`` lists the versions that exist. The one exception is a version higher than every version the graph
holds of an id that is still active at the snapshot release: it may come from a release newer than the graph, so it
is read as the latest version (``ENSG00000141510.25`` as ``ENSG00000141510.20``) and ``match`` is
``"version_newer_than_graph"``. Transcript and translation ids are read like gene ids (with or without version) and
converted to their gene.

**Versions of sequence accessions.** Ensembl's cross references give the version of the
sequence they attach (``xref.version``: RefSeq, UniProt, CCDS, the EMBL protein ids, Havana's ids; a symbol or an NCBI
gene id has none), and the graph keeps it per link and release (:py:meth:`idtrack.graph.TheGraph.accession_versions`):
which databases carry versions is Ensembl's statement, not a list of names. An accession's versions are not ids of
their own, as an Ensembl id's versions are, but what its links attached at each release, so its rule is said of those
links. An accession written with a version
(``match="version"``) is the accession as the links that attached that version carry it, read where they did: without
``from_release`` at the target release if they attached it there, else at the latest release at which they did (pig
``XM_003129161.3``, attached at 67 to 77 to ENSSSCG00000009034 and ENSSSCG00000026015, is read at 77 and answers
ENSSSCG00000026015, the one of the two that lives on; ``XM_003129161.4``, attached at 78 to 89 to ENSSSCG00000009034
only, which Ensembl retired at 90, has no conversion to 115); with a ``from_release`` at which no link attached it, by
that same rule (``from_release_fallback``). A version above every version the links attached is read as the latest
(``match="version_newer_than_graph"``) only where the accession is attached at the graph's last release: only then can
it be newer than Ensembl's import (pig ``XM_005662123.4``, an accession attached with version 1 at 78 to 82 and never
after, is ``unknown_version``). Any version that no link attached (below the latest, in a gap, or above all versions of
an accession Ensembl no longer attaches) is not answered (``status="unknown_version"``, ``known_versions`` the
versions there are). An accession without a version is read as every identifier is.

A version dates an accession only where it is a property of the sequence record that holds over time (decision 101).
UniProt's number in Ensembl's tables is UniProt's version of the entry (``Q9TSV4.120``), which changes with the entry's
annotation in nearly every release; a typed ``P04637.4`` means a version of the sequence, which Ensembl's records do
not hold. The graph tells the two kinds apart from its own versions
(:py:meth:`idtrack.graph.TheGraph.edit_counter_databases`): of the pairs of consecutive releases at which one link (an
accession, one version of an Ensembl id, a database) holds, the share in which the version differs; above one half
(:py:data:`idtrack.settings.VERSION_EDIT_COUNTER_SHARE`) the database's number counts the record's edits. For such a
database a typed version is not judged: the accession is read without it, and the result says so
(``match="version_not_judged"``, ``"version_not_judged"`` among ``result.reasons``), never ``unknown_version``. The shares on the graphs
of Ensembl 115 (per cent of the pairs; ``-``: the database has no versioned link in that graph):

.. list-table::
    :header-rows: 1
    :widths: 30 14 14 14 14 14

    * - Database
      - Pig
      - Rhesus
      - Cynomolgus
      - Mouse
      - Human
    * - Uniprot/SPTREMBL
      - 68.4
      - 63.3
      - 58.5
      - 91.1
      - 95.3
    * - Uniprot/SWISSPROT
      - 69.2
      - 67.2
      - 62.4
      - 91.6
      - 99.7
    * - RefSeq_mRNA, _ncRNA, _peptide
      - 0 to 0.1
      - 0 to 0.1
      - 0.0
      - 0.2 to 0.8
      - 0.2 to 1.8
    * - RefSeq predicted (mRNA, ncRNA, peptide)
      - 0 to 0.8
      - 0 to 0.2
      - 0.7 to 1.8
      - 0.1 to 3.0
      - 0.1 to 3.8
    * - CCDS
      - -
      - -
      - -
      - 0
      - 0
    * - Havana gene, transcript, translation
      - 0
      - -
      - -
      - 0 to 1.9
      - 1.1 to 14.6
    * - Vega gene
      - -
      - -
      - -
      - 0.6
      - 1.9

**Arguments.** Every argument is checked before the search starts: the identifier must be a string that is read as
above, the
releases integers of the graph (:py:meth:`idtrack.API.releases`), ``final_database`` one of
:py:meth:`idtrack.API.final_databases` for the target release (case is ignored), ``from_database`` one of
:py:meth:`idtrack.API.external_databases`. An invalid argument raises an :py:class:`idtrack.errors.InputError` that
names it (``error.argument``) and the valid values (``error.allowed``): :py:class:`idtrack.errors.ArgumentTypeError`
(also a ``TypeError``) for a value of the wrong type, :py:class:`idtrack.errors.MissingValueError` for an empty
identifier or a missing value, :py:class:`idtrack.errors.InvalidArgumentError` for an internal name or a string too
long to be an identifier, :py:class:`idtrack.errors.UnknownReleaseError` and :py:class:`idtrack.errors.UnknownDatabaseError` (also
``ValueError``) for a value the graph does not have. Every error idtrack raises is an
:py:class:`idtrack.errors.IdtrackError`; :py:mod:`idtrack.errors` describes the hierarchy.


Where a List Comes From
-----------------------

:py:meth:`idtrack.API.source_inference` says which release (or assembly, or database, by ``mode``) a list of
identifiers comes from, as an :py:class:`idtrack.results.SourceInference`. Each distinct identifier counts once for
every origin at which it is an identifier, read as name resolution reads it at that release: a name counts where it
is a name (a name renamed later counts as the name it was where it was one), never where it is only a synonym (a
synonym says the string was once a name or an alias, not that a table of that release wrote it). The answer is a
*band*: the origins the list cannot tell apart from
the best one (a one-sided sign test on the identifiers found at one and not the other, at
:py:data:`idtrack.api.SOURCE_SIGNIFICANCE`), with the margin to the best origin outside it. In the mode ``release``
a release counts on the assembly that serves it (a list of GRCh37's names of release 75 is release 75, not every later
release at which Ensembl still lists GRCh37). :py:meth:`idtrack.API.infer_source` gives the band as a list, or with
``report_only_winner=False`` every origin with its count.

.. code-block:: python

    names = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
    inference = api.source_inference(names, mode="release")
    print(inference.band[:3], inference.margin)


.. _current-limitations:

Current Limitations
-------------------

What this version does not do yet, with how often it matters, measured on the graphs of the five organisms built with
Ensembl 115 (every number is a count over Ensembl's own records of the releases asked about, not over a benchmark's
judgement):

* **Without a release, a string is what it means today** (or at the target release), by design: what it meant only
  at other releases is reported in ``result.other_readings`` (``reason="other_release"``), not answered. ``RNR1``
  asked without a release is NR4A2's synonym, as Ensembl's records of release 115 have it; with ``from_release=90``
  it is MT-RNR1, whose NCBI symbol it was at releases 86 to 93 and 95. Give the release a table was made with to read its
  names as they were then.
* **Across an assembly change, places are compared through Ensembl's own mapping of the two assemblies, and only
  where Ensembl publishes one.** Where it does (human NCBI36 to GRCh37 to GRCh38, mouse NCBIM37 to GRCm38 to GRCm39,
  pig Sscrofa9 to Sscrofa10.2), a gene's place on the earlier assembly is carried onto the later one and a bridge needs
  the two places to overlap, as on one assembly. Before, such bridges compared chromosome names only, and the mapping
  put the old gene elsewhere in 7.3 % of the mapped bridges of mouse (94 of 1,285), 1.0 % of human (44 of 4,382) and
  14.8 % of pig Sscrofa9 to Sscrofa10.2 (92 of 623), counted over every gene version whose history ends, converted to
  release 115. Requiring an overlap also refuses the bridges that the mapping puts within a few kilobases of the gene
  entered without overlapping it (mouse 5, human 17 such bridges: 4 mouse and 12 human answers lost, mostly one curated
  gene whose model moved a little). Where Ensembl publishes no mapping (pig Sscrofa10.2 to Sscrofa11.1, pig Sscrofa9
  to Sscrofa11.1 across both changes, rhesus MMUL_1 to Mmul_8.0.1 and Mmul_8.0.1 to Mmul_10, cynomolgus
  Macaca_fascicularis_5.0 to 6.0), two genes are told apart across the change only by the names of their chromosomes,
  and how often such a bridge lands on another gene of the same chromosome is not known (pig 2,927 bridges at
  Sscrofa10.2 to 11.1 and 870 at Sscrofa9 to 11.1, rhesus 1,013 and 1,241, cynomolgus 917, over the gene ids gone at
  each change; likely of pig's order). ``allow_bridges=False`` turns bridges off; ``result.path.bridges`` lists each with its rule.
* **Copies on patches and alternative loci are placed by Ensembl's assembly exceptions.** A copy's place is the
  region of the reference sequence that its patch or haplotype replaces at that release (Ensembl's
  ``assembly_exception``), and a gene outside that region lies apart from it (pig and the macaques have no assembly
  exceptions). Inside that region, and for an LRG record (which has none), a copy bridges only through an id that it
  and the other gene carried at one release and that names one gene at each such release
  (:ref:`bridges <reading-an-identifier>`), which does not tell two genes of one region apart. Of the copies whose
  history ends (840 human, 732 mouse), every bridge to a gene on another sequence is refused (human 10 of 10, mouse 4
  of 4), and of those to a gene more than 1 Mb away on the same sequence mouse 10 of 10 and human 7 of 23 (the other
  16 lie inside the region of their exception, which the rule above judges). The rule also refuses a copy's link to
  its own gene where a family shares the id.
* **A number written with a version that two numbering schemes share is read where its version exists.** Human
  numeric Havana translation ids of releases 50 and 51 carry the version 1 and 2,493 of them are also NCBI gene ids:
  ``2633.1`` without a release is read as the Havana id of release 51 (another gene than NCBI's ``2633``), with NCBI's
  reading reported beside it (``reason="unversioned_database"``); give the release a table was made with, or write the
  NCBI id without a suffix.
* **A gene's own continuation does not outrank a branch that left it while it lived on.** Where Ensembl maps an id to
  new ids at a session while the id itself continues, the candidate on a branch can have the larger product of
  Ensembl's scores and become the best answer (pig ``convert("ENSSSCG00000014532", from_release=56, to_release=98)``
  answers ENSSSCG00000042508, an offshoot, while the id's own successors rank fourth to sixth). Pig: 7 ids, 1,133
  conversions (748 of them without a tie), all at the Sscrofa9 to Sscrofa10.2 change; human: none; mouse and the
  macaques: no id is branched while it lives. ``result.view("all")`` lists the id's own successors.
* **An id that Ensembl retired and later brought back with an old version is one node with two lives.** Converting
  within the gap or across it can differ by direction (mouse ENSMUSG00000118483, version 1 at 97 to 102 and again from
  109; human ENSG00000288686): 1 mouse and 1 human id found, and 9 more ids of the two organisms with a version that
  falls back to another release.
* **At its own release, a symbol of the organism's name authority can answer another gene** where Ensembl displays
  the name for both genes, or for neither, and more databases give the other gene the name: Ensembl's display name
  ranks first, then the support of the evidence (the authority's symbol wins against another gene's synonym only). Of
  1,500 symbols per organism, each asked at a release where the organism's symbol database names a gene with it: pig
  5, rhesus 5, mouse 2, human 3 best answers include a gene the authority does not give the name (of them pig 2,
  rhesus 1, mouse 2 and human 2 ties with the authority's gene; on the same samples before the display name ranked
  first: pig 10, rhesus 11, mouse 7, human 8). Each candidate says
  whether an authority gave it the name (``Candidate.authority``) and whether it is the gene's display name
  (``Candidate.display_name``); in rhesus the authority's names are themselves projected from human.
* **A name in one letter case that a link of another gene spells exactly so answers that gene, not the gene the name
  authority displays with it in mixed case.** Where, at one release, Ensembl displays a name for one mouse gene as the
  authority writes it (``Foxl1``) and for another gene in capitals, the string in one letter case is an exact match of
  the links spelt so, and an exact match follows only those links (**Matching the string** above): the authority's gene is then
  not among the genes reached, and the rule that ranks the authority's display name first, read without letter case,
  ranks only the genes reached. Of the 743 spellings in one letter case of such pairs of mouse display names, each
  asked at its release, 424 answer the gene that carries the name exactly as typed, mostly names of snoRNA and scaRNA
  families at releases 63 to 71 (reading such strings without letter case at the exact step would undo the exact
  match for strings that do name a gene as typed, which is kept). Written as the authority writes it, the name is read
  as that gene's display name.
* **A gene that Ensembl retired without a successor can answer another gene than the one that holds its accessions
  today.** Where Ensembl retired a gene and filed its accession (an NCBI gene id, a RefSeq or UniProt accession, a
  miRBase accession) on another gene before, or without, giving the accession to the gene that holds it at the last
  release, the retired id, asked at its last release, travels by a bridge to the gene nearest in releases, not to the
  one the accession names now: human ``ENSG00000199004`` (hsa-mir-21, retired at 84 to 85) answers its host gene VMP1,
  which Ensembl gave NCBI's id 406991 of MIR21 at 85 to 87, while ``MI0000077`` and 406991 are held by
  ENSG00000284190, Ensembl's new miRNA gene from 88. Of the retired ids whose accession another gene holds at
  release 115 (asked at their last release, accessions of families such as Rfam not counted), this is the answer of
  human 784 of 3,279 (326 of them miRNA genes; the answered gene held the accession in the gap in 618), mouse 310 of
  1,911, pig 323 of 4,550, rhesus 118 of 3,579 and cynomolgus 26 of 1,182 (the others answer that gene, human 1,553,
  or nothing, human 942); nearly all are protein-coding outside human. Ask by the accession or the name at the target
  release (``convert("MI0000077")``), which answers the gene that holds it then; ``result.path.bridges`` shows the
  bridge taken.
* **A gene whose history splits, converted to an external database, answers no identifier where the first-ranked
  of its descendants has none, although another descendant has one.** The answer is first a gene, ranked as Ensembl
  genes, and the identifiers are those of the genes of that best answer; where none of them has one, the result is
  those genes (``no_target``), in ``view("all")`` too, and the other descendants are listed as genes only. Of the
  Ensembl gene ids whose history splits, asked at their last release to release 115 (several genes in
  ``view("all")``), this happens for at least one database for human 197 of 776 (532 id and database pairs), mouse
  439 of 2,593 (735), pig 364 of 3,304 (646; HGNC Symbol 197 of them), rhesus 195 of 1,549 (364) and cynomolgus 84
  of 873 (158); never with another gene's identifier. Convert to Ensembl genes first, then each gene of
  ``view("all")`` to the database (``convert(gene, final_database=...)``): on 100 sampled pairs per organism this
  gives the descendants' identifiers every time.
* **In mouse releases 80 and 81, RefSeq_mRNA also holds predicted and RefSeqGene accessions.** Ensembl attaches
  gene-level RefSeq accessions of every kind to mouse genes at these releases under ``RefSeq_dna``, which continues
  RefSeq_mRNA; the forms that never moved to another RefSeq database are filed under that main continuation, since
  RefSeq_mRNA has no identifier scheme of its own (``NM_`` is 89 % of its accessions over the releases). Converted to
  ``RefSeq_mRNA`` at 80 or 81, a gene can answer ``XM_`` (6,832 accessions at 80, 4,426 at 81), ``XR_`` (1,264 and
  1,148) and ``NG_`` (674 and 680) accessions beside its ``NM_`` ones; pig has 90 ``XR_`` accessions under
  RefSeq_mRNA at 63 to 66; human (76 included), rhesus and cynomolgus have none. The prefix says what an accession
  is (``NM_`` an mRNA, ``XM_`` and ``XR_`` predicted, ``NG_`` a RefSeqGene record); ask ``RefSeq_mRNA_predicted`` or
  ``RefSeq_ncRNA_predicted`` for the predicted ones.
* **The support of a name counts a database that writes it in another letter case.** Typed in mixed case, a mouse
  symbol is the display name only as its database spelt it, but the number of databases that attach it to a gene
  counts UniProt's gene name in capitals too. At release 62, where UniProt's names are read on translations, 63 MGI
  symbols name two genes or more, UniProt names some of them in capitals, and for 53 of them that support alone
  makes one gene the answer (``Adora3`` at 62 answers ENSMUSG00000000562, which MGI, NCBI and UniProt name, not the
  tie with ENSMUSG00000074344, which MGI and NCBI name). ``view("all")`` lists the other gene, and each candidate's
  ``entry_databases`` names the databases counted.


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
    api = idtrack.API.build("homo_sapiens", 115, local_repo)


Supported Organisms
-------------------

IDTrack supports the following organisms. Each is declared once, in the organism table
:py:data:`idtrack.organisms.ORGANISM_TABLE`, and ships a default external-database configuration
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

The **key** of an assembly is what IDTrack uses everywhere (the ``assembly`` argument of
:py:class:`idtrack.build.DatabaseManager`, labels such as ``assembly_801_ensembl_gene``, the table folders
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

The symbol databases (:py:data:`idtrack.organisms.SYMBOL_DATABASES`) are the default final database of
:py:class:`idtrack.harmonize.HarmonizeFeatures` (the first one) and the ones to try for gene symbols. In current Ensembl, macaque genes carry VGNC symbols on the newest assemblies
and HGNC symbols on earlier ones, and cynomolgus 6.0 has few symbols of either kind.

IDTrack reads Ensembl ids with their versions: an organism whose Ensembl ids carry no versions is not supported (a
graph build refuses it with an error that says so). The five organisms above all have versions.

.. tip::

    Want to add support for another organism? See "Adding Organism Support" in :doc:`contributing` for the
    recipe, and :doc:`_notebooks/02_prepare_new_external_yaml` for configuring its external databases.


Common Patterns
---------------

.. _reusing-built-graphs:

Reusing Built Graphs
~~~~~~~~~~~~~~~~~~~~

Once a graph is built, its file is in the local repository. :py:meth:`idtrack.API.build` loads it from there instead
of building it again, and :py:meth:`idtrack.API.load` loads a graph file by its path and never builds:

.. code-block:: python

    # Loaded from the graph file if it exists; built (and saved) only if it does not
    api = idtrack.API.build("homo_sapiens", 115, "./idtrack_cache")

    # The file alone: never a build, never the network (a missing file is an idtrack.errors.GraphFileError)
    path = idtrack.API.graph_path("homo_sapiens", 115, "./idtrack_cache")
    api = idtrack.API.load(path)
    print(api.describe().releases[-1])  # what the graph holds: organism, releases, assemblies, ...

Loading reads the graph file alone: no table, no :py:class:`idtrack.build.DatabaseManager` and no network connection
is needed (and none of the build stack, pandas, h5py, PyMySQL or PyYAML, is imported), so a graph on disk also loads
where Ensembl cannot be reached. The file is one ``.npz`` archive of arrays and JSON (no pickle: loading it executes no
code); each array is deflated where that makes it at least a tenth smaller (arrays of hashes are stored as they are),
and the arrays are inflated in parallel threads when the file is read. Its name, ``graph_<organism>_assembly<key>_min<first>_max<snapshot>.npz``,
says the organism, the primary assembly and the window; the file itself holds one record of what the graph is and what
built it (:py:class:`idtrack.graph.GraphRecord`, ``api.graph.record``; :py:meth:`idtrack.API.describe` gives the main
facts): the organism, the primary assembly, the window and the snapshot, the releases with the assembly that serves
each, the forms, the external databases per assembly, the selection of external databases and the organism table it
was built with (and, for each release, the release of the externals configuration whose settings it took: a release
after the last one the configuration lists takes the settings of that last one), the package version that built it
(for a graph built from a source folder that is not an installed idtrack, the record says so instead of a version:
``none: built from a source folder, not an installed idtrack``; such a graph is named by its git commit alone, where it
was known, and a commit that a later squash or rebase replaced is on no published branch: build graphs to share with
an installed release) and (where it was known) the git commit that built it, the counts of the build, the history report (the sessions Ensembl recorded at
other releases than its id tables show, the events placed with the version an id had at a release, the events left
out) and the digest of the content (nodes, edges, spellings, places of the genes). Besides the identifiers, the graph
holds where every gene version lies, on which assembly, sequence region, bases and strand
(:py:meth:`idtrack.TheGraph.places`). When the file under the expected name holds another graph than the one asked for (it was
built under another externals YAML, another organism table or by another version of the graph format),
:py:meth:`idtrack.API.build` raises :py:class:`idtrack.errors.GraphMismatchError`, which names the differences:
IDTrack never answers silently from another graph, and never starts a build of hours without being asked. To build the
graph again, call ``idtrack.API.build("homo_sapiens", 115, "./idtrack_cache", rebuild=True)``, which replaces the file.

Graphs written by IDTrack 0.0.5 (``graph_<organism>_min<first>_max<snapshot>_narrow.pickle``) and its table cache
(``<organism>_assembly-<key>.h5``) are not read: build the graphs again with this version.

Names per Database and Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Which genes does a name name, in which database, at which release? :py:meth:`idtrack.API.genes_named` answers it as
data, one :py:class:`idtrack.results.NameMatch` per name, gene and database, with the releases at which it holds (on
the assembly that serves each release, for genes of the reference sequence; ``fold_case=True`` ignores letter case);
:py:meth:`idtrack.API.names_of` gives the names the databases attach to a gene:

.. code-block:: python

    for match in api.genes_named("TP53"):
        print(match.database, match.synonym, match.gene, match.releases[0], "-", match.releases[-1])
    print([m.name for m in api.names_of("ENSG00000141510.20", 115) if m.database == "HGNC Symbol"])

A name a database recorded as a synonym of the gene has ``synonym`` true; the name and its database are written as
they are (the database that records the synonym, ``HGNC Symbol``).

Working with AnnData Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`idtrack.harmonize.HarmonizeFeatures` converts the features of several ``.h5ad`` datasets to one Ensembl
release and merges them. It needs the ``harmonize`` extra (anndata and scipy):
``pip install "idtrack[harmonize] @ git+https://github.com/theislab/idtrack.git"``. Its pure core
(:py:func:`idtrack.harmonize.convert_features`, :py:func:`idtrack.harmonize.feature_table`,
:py:func:`idtrack.harmonize.output_plan`) works on a list of names without files and needs pandas only.

.. code-block:: python

    from idtrack.harmonize import HarmonizeFeatures

    harmonizer = HarmonizeFeatures(
        project_name="my_project",
        data_h5ad_dict={"study_a": "study_a.h5ad", "study_b": "study_b.h5ad"},
        project_local_repository="./harmonization",   # result cache, one JSON file per dataset
        idtrack_local_repository="./idtrack_cache",   # the graph
        to_release=115,
        organism="homo_sapiens",
        snapshot_release=115,
        final_database="HGNC Symbol",
    )
    print(harmonizer.conversion_report())                         # one row per input feature, with status and reason
    adata = harmonizer.unify_multiple_anndatas(features="union")      # or features="intersect"

The constructor converts every feature (the long part; the results are cached). The rules of the output:

* Each converted gene is one feature, named by its Ensembl gene id at the target release; its name in the final
  database is in a column named after the database (a gene without one has a missing value there and status
  ``"converted_no_final_id"``: the final-database column never holds an Ensembl id).
* Nothing is dropped silently: every output feature has ``var["idtrack_status"]`` (``"converted"``, ``"merged"``,
  ``"unconverted_ambiguous"``, ``"unconverted_not_in_graph"``, ...) and ``var["idtrack_reason"]``. Unconverted
  features keep their input names; ``keep_unconverted=False`` (of
  :py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas` and
  :py:meth:`idtrack.harmonize.HarmonizeFeatures.feature_harmonizer`) leaves them out.
* Several input features of one dataset on one gene (``merge=`` of the same two methods): ``merge="sum"`` adds them
  up (right for counts); ``merge="flag"`` (default) keeps the one that names the gene itself as the gene's feature and flags the others
  (``"unconverted_merge_partner"``).
* Each dataset is converted on its own: a dataset is harmonised the same alone and together with others. A feature
  that is a gene id existing at the target release is that gene.
  ``from_release_by_dataset={"study_a": 100}`` reads a dataset's features at the release it was annotated with;
  ``from_database_by_dataset={"study_a": "HGNC Symbol"}`` reads them as ids of that database (a symbol that another database
  gives to another gene is then that database's gene, not a tie). Nothing is inferred on your behalf. Without it, a
  symbol that two databases give to two genes is ``"unconverted_ambiguous"`` when both genes can be named in the
  final database; when only one of them has a name there, the feature is converted to that gene (a tie is judged
  among the genes that can be named in the final database, see Example 4).
* ``api=`` passes an :py:class:`idtrack.API` whose graph is already loaded.

:py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas` merges the harmonised datasets without losing or
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
(:py:class:`idtrack.settings.SearchSettings`: 200,000 search states and 30 s by default; ``API.load(path,
search=SearchSettings(...))`` sets other defaults, and ``budget_states=`` / ``budget_seconds=`` of
:py:meth:`idtrack.API.convert` bound one call); a search past it raises
:py:class:`idtrack.errors.SearchBudgetExceededError` (no conversion of the measurements came near it). The graph is
only read by conversions, so several threads can share one :py:class:`idtrack.API`, but they do not convert in
parallel: a server scales with processes.


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
