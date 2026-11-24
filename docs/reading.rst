.. _reading:

=========================
How an Identifier Is Read
=========================

Before anything is looked up, the typed string is read as a person who pasted it meant it: where it is read (the
starting release), what is cleaned from it, how it is matched against the identifiers of the graph, and what is
never read. The result says each of these (``from_release``, ``from_release_inferred``, ``match``, ``graph_id``,
``other_readings``), so an answer can always be traced back to the string. What happens after the string is read is
on :doc:`concepts`.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


.. _reading-an-identifier:
.. _reading-starting-release:

The starting release
--------------------

With ``from_release``, the identifier is read at that release. Without it, it is read at
the target release if it exists there, and otherwise at the latest release at which it exists; the result says which
release was used (``from_release``) and whether it was chosen for you (``from_release_inferred``). The rule is the
same for Ensembl ids, symbols and the ids of other databases. A symbol given without ``from_release`` is therefore
read as what it names at the target release (or at its latest release), not as what it named years ago. One release
decides both what a string means and where it is read: ``RNR1`` with ``to_release=90`` is read at 90, where it is
NCBI's symbol of MT-RNR1 (and a synonym of NR4A2), exactly as with ``from_release=90``. What a string meant at other
releases only is not its answer, but the result reports it: ``result.other_readings`` lists such a reading with
``reason="other_release"``, the gene it led to and the releases at which the string had that meaning (``RNR1`` without
a release is NR4A2's synonym, and MT-RNR1's symbol of releases 86 to 93 and 95 is reported); give ``from_release`` to read it
there.

If the graph does not hold the identifier at a given ``from_release`` (a wrong release, an id of a database the graph does not include at that
release, a name of an early release whose source the graph lacks), :py:meth:`idtrack.API.convert` reads it by the rule
without ``from_release`` and says so: ``from_release_fallback`` is true, ``from_release_given`` is the release you
gave, and ``result.reasons`` holds ``"read_at_another_release"`` (the ``reason`` of a result with targets, whose
status stays ``"converted"`` or ``"tie"``; after the reason a result without targets has). ``from_release`` is the
release the string was read at, also in a result without targets. Such an answer is what the identifier meant where
it was read, which can be another gene than the one it meant at the release you gave: :py:meth:`idtrack.API.classify` puts it in a bin of its own
(``read_at_another_release``), and the harmonizer says so in ``idtrack_reason``. (The engine,
:py:meth:`idtrack.engine.Track.convert`, raises :py:class:`idtrack.errors.InactiveAtReleaseError` instead, naming the
releases at which the identifier exists.)


.. _reading-cleaning:

What is cleaned
---------------

What a person who pasted a string did not mean is removed or refused before the string is matched, and the match
says so (:py:func:`idtrack.reading.read_identifier`; the prefixes are :py:data:`idtrack.results.MATCH_PREFIXES`):

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


.. _reading-not-read:

What is never read
------------------

Forms that are not read stay not found (never another gene): database prefixes (``HGNC:11998``, ``UniProtKB:P04637``),
Greek letters (``IL-1β``), UniProt isoforms (``P04637-2``), NCBI gene ids written as numbers of a spreadsheet
(``7157.0``), and version forms other than ``.<version>`` (``ENSG00000141510-5``, ``ENSG00000141510_5``,
``ENSG00000141510.v5``: a ``-`` or ``_`` and digits after an Ensembl id is as often a tool's suffix for a repeated
name as a version, so it is no version).


.. _reading-matching:

Matching the string
-------------------

When a string can mean several nodes of the graph (a
symbol that was the name of one gene and is a synonym of another today), the one that exists at the release asked
about (``from_release``; without it the target release) is taken; where none exists there, the string is read at the
latest release at which one of them does, by the steps of that release. An Ensembl stable id is never passed over for
a synonym spelt the same. An identifier exists at a release when a database has it there on the genome assembly that
Ensembl maintained at that release: human GRCh37 is still listed beside GRCh38 with its names of release 75, and a
name that only GRCh37 lists today does not exist today. Among those, the string is matched exactly, then ignoring
letter case, then taking ``-``, ``_`` and ``.`` as the same character (``HLA_A`` is ``HLA-A``), then, for a
versioned accession, with the version written (:ref:`reading-accession-versions`); each step is tried on names and then on synonyms
before the next. "Exactly" means as a database wrote
the identifier at the release it is read at: databases change the letter case of a name over the releases, two
databases can write one name in two ways (at release 75 MGI wrote the mouse gene ``Bap1`` and UniProt wrote it
``BAP1``), and one database can give two genes one name in two cases (``Tc`` is a synonym of one mouse gene, ``tc``
one of Noto), so each of these spellings is an exact match where it was written, and is not taken for another gene's
synonym spelt the same (``BAP1`` with ``from_release=75`` is that gene exactly; today no database writes the name so,
and ``BAP1`` is exactly a synonym of another mouse gene, but as a string in one letter case it is first read as a
mouse symbol without letter case, :ref:`reading-rules`, and answers ``Bap1``). The graph keeps one node per
identifier, named as at its last release, which is what ``result.graph_id`` shows, and the spelling of every link of it (the identifier, an Ensembl id,
a database, a release): a string matched exactly is followed only through the links spelt as it is typed (``Tc`` is
the first gene's synonym alone, ``tc`` Noto's), a string matched only by letter case through all of them (``TC`` is
both). Nothing else is ever cut off an external id (``SLA.7`` is not ``SLA``). Ensembl's own ids are written in capitals
only, so an id in another letter case is found by case (``enst00000016171`` is ``ENST00000016171``).


.. _reading-rules:

Four rules between readings
---------------------------

Four rules decide between readings of one string, and the result reports every other reading it set aside or joined
(``result.other_readings``, :py:class:`idtrack.results.OtherReading`: the node, its match, why, and the genes it
would answer):

* **Symbols without letter case first, where symbols are written in mixed case.** MGI writes mouse symbols with a
  capital and then small letters (``Trp53``); the graph says so (:py:meth:`idtrack.graph.TheGraph.symbol_case`: most
  of the organism's symbols have a small letter). There, a string in one letter case (``AR``, ``CCK``, ``nos2``) that
  is not exactly a name of the release is read first as a symbol without letter case where that names exactly one
  gene: ``AR`` is ``Ar``, not the three genes whose synonym ``AR`` is (the exact synonym is reported among ``other_readings``,
  ``reason="symbol_without_case"``). With ``from_database`` naming another database the rule is off.
* **Names made unique by single-cell tools.** Tools append ``-1``, ``.1`` or ``_1`` to repeated feature names. In a
  batch (:py:meth:`idtrack.API.convert_many`, and so in the harmonizer), an item that is another item of the batch
  with such a suffix, and is itself no exact name where it is read alone (at ``from_release`` where the graph holds
  it there, else where :py:meth:`idtrack.API.convert` reads it), is that item's copy: its result is the item's, its
  match prefixed ``copy:`` (``SRF.1`` beside ``SRF``; ``IGHV3-30-2`` beside ``IGHV3-30`` stays a name of its own, and
  so does pig ``SLA-10``, a name of release 68, beside ``SLA``). Every other item of a batch is answered as a single
  conversion answers it. An Ensembl id with such a suffix (``ENSG00000141510-1`` beside ``ENSG00000141510``) is
  neither a copy nor a version: it stays not found, never another gene. A versioned accession written with a version
  (``NM_000546.5`` beside ``NM_000546``) is that version, read as it is alone, never a copy. In any call, such a string is not read, with its separators taken as one or without letter case, as a synonym of
  another gene than the one an authority's symbol written before the suffix names at the release asked about
  (:ref:`limitation-suffix`): ``SRF.1`` alone is not the synonym ``SRF-1`` of SPATA6 (the reading is reported,
  ``reason="make_unique_suffix"``, and ``SRF.1`` is not found). A name is still read so (``MIR138_2`` is
  ``MIR138-2``).
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


.. _reading-match:

How the reading is reported, and naming the database
-----------------------------------------------------

``result.match`` says how the string matched: a step (``"exact"``, ``"case"``, ``"separator"``, ``"version"``,
``"version_newer_than_graph"``, ``"version_not_judged"``: :py:data:`idtrack.results.MATCH_STEPS`) after the
prefixes that apply (``"copy:"``, ``"unicode:"``, ``"punctuation:"``, ``"par_y:"``, ``"lrg:"``, ``"synonym:"``:
:py:data:`idtrack.results.MATCH_PREFIXES`), for example ``"punctuation:synonym:exact"``. ``result.graph_id`` is the
identifier it was read as (a synonym written as it is; the match says it is a synonym), and ``result.from_databases``
the databases that attach the identifier to its genes at the release it was read at (to a gene itself or to one of
its transcripts or translations; a synonym's database by its own name). With ``from_database="HGNC Symbol"`` the
string is read as an id of that database (its names; where none of its names matches at any release, the synonyms
it records): it stands only for the genes that HGNC attaches it to, and between those the evidence of every database
still counts (a symbol that the named database gives to two genes, of which other databases confirm one, is that one). The synonyms a database records are no
target: ``final_database`` is a database's ids.


Copies and records
------------------

Copies on alternative loci and patches are listed apart (``alternative_locus_genes``) beside a gene of the
reference sequence (:ref:`concept-places`). So is the Y copy of a pseudoautosomal gene (a gene of a pseudoautosomal region of Y)
beside the gene at its place on X, the region of Y read on the region of X it repeats (from release 110 Ensembl gives
the Y copies ids of their own: ``PLCXD1`` is the X gene, the Y copy listed apart; the regions are GRC's,
:py:data:`idtrack.organisms.PSEUDOAUTOSOMAL_REGIONS`; one rule of genes and places, :py:mod:`idtrack.graph.genes`).
An LRG record's id (``LRG_321``) answers the gene the record describes (TP53), the match prefixed ``lrg:`` and the record listed apart;
so does every id of the record: its transcripts and translations (``LRG_321t1``, ``LRG_321p1``) and the record's gene
id of the GRCh37 releases (``LRG_7_g1``).


.. _reading-versions:

Versions of Ensembl ids
-----------------------

An Ensembl gene id without version stands for its version at the starting release. A version the graph
does not hold is not answered with another version's gene: the result has no candidate, ``status`` is
``"unknown_version"`` and ``known_versions`` lists the versions that exist. The one exception is a version higher than every version the
graph holds of an id that is still active at the snapshot release: it may come from a release newer than the graph, so it
is read as the latest version (``ENSG00000141510.25`` as ``ENSG00000141510.20``) and ``match`` is
``"version_newer_than_graph"``. Transcript and translation ids are read like gene ids (with or without version) and
converted to their gene.


.. _reading-accession-versions:

Versions of sequence accessions
-------------------------------

Ensembl's cross references give the version of the sequence they attach (``xref.version``: RefSeq, UniProt, CCDS, the EMBL protein ids, Havana's ids; a symbol or an NCBI
gene id has none), and the graph keeps it per link and release (:py:meth:`idtrack.graph.TheGraph.accession_versions`):
which databases carry versions is Ensembl's statement, not a list of names. An accession's versions are not ids of
their own, as an Ensembl id's versions are, but what its links attached at each release, so its rule is said of those
links. An accession written with a version (``match="version"``) is the accession as the links that attached that version carry it, read where they did: without
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

A version dates an accession only where it is a property of the sequence record that holds over time.
UniProt's number in Ensembl's tables is UniProt's version of the entry (``Q9TSV4.120``), which changes with the entry's
annotation in nearly every release; a typed ``P04637.4`` means a version of the sequence, which Ensembl's records do
not hold. The graph tells the two kinds apart from its own versions
(:py:meth:`idtrack.graph.TheGraph.edit_counter_databases`): of the pairs of consecutive releases at which one link (an
accession, one version of an Ensembl id, a database) holds, the share in which the version differs; above one half
(:py:data:`idtrack.settings.VERSION_EDIT_COUNTER_SHARE`) the database's number counts the record's edits. For such a
database a typed version is not judged: the accession is read without it, and the result says so
(``match="version_not_judged"``, ``"version_not_judged"`` among ``result.reasons``), never ``unknown_version``. The
shares on the graphs of Ensembl 115 (per cent of the pairs; ``-``: the database has no versioned link in that graph):

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


.. _reading-arguments:

Arguments
---------

Every argument is checked before the search starts: the identifier must be a string that is read as this page
says, the releases integers of the graph (:py:meth:`idtrack.API.releases`), ``final_database`` one of
:py:meth:`idtrack.API.final_databases` for the target release (case is ignored), ``from_database`` one of
:py:meth:`idtrack.API.external_databases`. An invalid argument raises an :py:class:`idtrack.errors.InputError` that
names it (``error.argument``) and the valid values (``error.allowed``): :py:class:`idtrack.errors.ArgumentTypeError`
(also a ``TypeError``) for a value of the wrong type, :py:class:`idtrack.errors.MissingValueError` for an empty
identifier or a missing value, :py:class:`idtrack.errors.InvalidArgumentError` for an internal name or a string too
long to be an identifier, :py:class:`idtrack.errors.UnknownReleaseError` and :py:class:`idtrack.errors.UnknownDatabaseError` (also
``ValueError``) for a value the graph does not have. Every error idtrack raises is an
:py:class:`idtrack.errors.IdtrackError`; :py:mod:`idtrack.errors` describes the hierarchy.


.. _source-inference:

Where a list comes from
-----------------------

:py:meth:`idtrack.API.source_inference` says which release (or assembly, or database, by ``mode``) a list of
identifiers comes from, as an :py:class:`idtrack.results.SourceInference`. Each distinct identifier counts once for
every origin at which it is an identifier, read as name resolution reads it at that release: a name counts where it
is a name (a name renamed later counts as the name it was where it was one), never where it is only a synonym (a
synonym says the string was once a name or an alias, not that a table of that release wrote it). The answer is a
*band*: the origins the list cannot tell apart from the best one (a one-sided sign test on the identifiers found at one and not the other, at
:py:data:`idtrack.api.SOURCE_SIGNIFICANCE`), with the margin to the best origin outside it. In the mode ``release``
a release counts on the assembly that serves it (a list of GRCh37's names of release 75 is release 75, not every later
release at which Ensembl still lists GRCh37). :py:meth:`idtrack.API.infer_source` gives the band as a list, or with
``report_only_winner=False`` every origin with its count.

.. code-block:: python

    from idtrack import API

    api = API.load(API.graph_path("homo_sapiens", 115, "./idtrack_cache"))
    current = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
    old = ["MLL", "MLL2", "MLL3", "C10orf2", "FAM5B", "C11orf30", "C1orf186", "C9orf3", "SEPT9", "GPR98", "TP53", "BRCA1"]
    for names in (current, old):
        inference = api.source_inference(names, mode="release")
        band = sorted(inference.band)
        print(f"best {inference.band[0]}; band {band[0]} to {band[-1]} ({len(band)} releases); margin {inference.margin}")

.. code-block:: text

    best 48; band 48 to 115 (68 releases); margin None
    best 50; band 48 to 87 (40 releases); margin 5

Five symbols that every release names cannot tell releases apart: every release is in the band, and ``margin`` is
``None`` (no origin lies outside it). Symbols that HGNC renamed (``MLL`` is KMT2A since 2013) count only at the
releases at which they were names, and the band shrinks to those; ``margin`` is the number of identifiers by which
the best release leads the best release outside the band. The longer the list, the narrower the band. The other
modes (:py:data:`idtrack.engine.SOURCE_MODES`) count by ``assembly``, by ``[assembly, release]``
(``assembly_release``, the default) and by ``[database, assembly, release]`` (``complete``).