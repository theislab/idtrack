.. _limitations:
.. _current-limitations:

=============================
Limitations and Known Costs
=============================

What this version does not do yet, with how often it matters, measured on the graphs of the five organisms built
with Ensembl 115, and what it costs in time and memory where that is not what a reader would expect. Each limitation
says how to see it in a result and what to do about it. The terms are those of :doc:`concepts` and :doc:`reading`.

A number is a count over Ensembl's own records of the releases asked about, unless the entry cites **the
benchmarks**: the benchmark repository `idtrack-reproducibility <https://github.com/theislab/idtrack-reproducibility>`_
judges this version's answers against Ensembl's files and the naming authorities, family by family, and its table of
known limitations (``results/limitations/limitations.tsv`` at commit ``72eeb8a``) counts each limitation in the
complete run of the families on the graphs of 115. Such a count is of the answers the benchmark judged wrong (each
line answered under the views ``best`` and ``all``), or, for the comparison with other tools, of the lines of its read
sample where idtrack is wrong (1,015 lines: pig 163, mouse 375, human 408, rhesus 52, cynomolgus 17;
:doc:`comparison`).

.. contents::
    :local:
    :depth: 2
    :backlinks: none


Limitations
-----------

Without a release, a string is what it means today
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Without a release, a string is what it means today** (or at the target release), by design: what it meant only
at other releases is reported in ``result.other_readings`` (``reason="other_release"``), not answered. ``RNR1``
asked without a release is NR4A2's synonym, as Ensembl's records of release 115 have it; with ``from_release=90``
it is MT-RNR1, whose NCBI symbol it was at releases 86 to 93 and 95. Give the release a table was made with to read its
names as they were then.

A string is read in every name database the graph holds, so without a release a symbol is read at the target release
wherever any of them gives it there, also where Ensembl's display name of the list's own release belonged to another
gene: in the benchmarks' marker lists asked without a release (CellMarker and PanglaoDB), mouse 49 and human 713
answers.

.. _limitation-assembly-mapping:

Places across an assembly change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Across an assembly change, places are compared through Ensembl's own mapping of the two assemblies, and only where
Ensembl publishes one.** Where it does (human NCBI36 to GRCh37 to GRCh38, mouse NCBIM37 to GRCm38 to GRCm39, pig
Sscrofa9 to Sscrofa10.2), a gene's place on the earlier assembly is carried onto the later one and a bridge needs the
two places to overlap, as on one assembly.

Compared by the names of their chromosomes alone, the mapping puts the old gene elsewhere in 7.3 % of such bridges of
mouse (94 of 1,285), 1.0 % of human (44 of 4,382) and 14.8 % of pig Sscrofa9 to Sscrofa10.2 (92 of 623), counted over
every gene version whose history ends, converted to release 115. Requiring an overlap also refuses the bridges that the
mapping puts within a few kilobases of the gene entered without overlapping it (mouse 5, human 17 such bridges: 4 mouse
and 12 human answers lost, mostly one curated gene whose model moved a little).

Where Ensembl publishes no mapping (pig Sscrofa10.2 to Sscrofa11.1, pig Sscrofa9 to Sscrofa11.1 across both changes,
rhesus MMUL_1 to Mmul_8.0.1 and Mmul_8.0.1 to Mmul_10, cynomolgus Macaca_fascicularis_5.0 to 6.0), two genes are told
apart across the change only by the names of their chromosomes, and how often such a bridge lands on another gene of the
same chromosome is not known (pig 2,927 bridges at Sscrofa10.2 to 11.1 and 870 at Sscrofa9 to 11.1, rhesus 1,013 and
1,241, cynomolgus 917, over the gene ids gone at each change). Where a mapping is published, the names of chromosomes
alone would have put 1.0 to 14.8 % of such bridges elsewhere (above); that is the only measured guide to the size.
``allow_bridges=False`` turns bridges off; ``result.path.bridges`` lists each with its rule.

Such a bridge can follow any identifier the old gene carried, also another gene's name: pig ``convert("AKR1C3",
from_release=72, to_release=115)`` reads AKR1C3 as ENSSSCG00000023446 (Sscrofa10.2, last at release 89, no successor)
and answers ENSSSCG00000062058, which carries AKR1C3 at 115, while ``view("all")`` adds ENSSSCG00000063387, reached
through AKR1C1, an NCBI symbol the old gene also carried at 89: both lie on chromosome 10, which is all that the
bridge can compare. In the benchmarks' comparison with other tools, 35 of the sampled wrong lines are names read at
an old release and carried so across an assembly change without a published mapping (pig 24, rhesus 10, mouse 1).

.. _limitation-copies:

Copies on patches and alternative loci
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Copies on patches and alternative loci are placed by Ensembl's assembly exceptions.** A copy's place is the region of
the reference sequence that its patch or haplotype replaces at that release (Ensembl's ``assembly_exception``), and a
gene outside that region lies apart from it (pig and the macaques have no assembly exceptions). Inside that region, and
for an LRG record (which has none), a copy bridges only through an id that it and the other gene carried at one release
and that names one gene at each such release (:ref:`concept-bridges`), which does not tell two genes of one region
apart.

Of the copies whose history ends (840 human, 732 mouse), every bridge to a gene on another sequence is refused (human 10
of 10, mouse 4 of 4), and of those to a gene more than 1 Mb away on the same sequence mouse 10 of 10 and human 7 of 23
(the other 16 lie inside the region of their exception, which the rule of this section judges). The rule also refuses a
copy's link to its own gene where a family shares the id.

.. _limitation-numbers:

Numbers that several numbering schemes share
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A string of digits is read in every database whose ids are numbers.** These are NCBI's gene ids, and the numeric
Havana ids and RefSeq rows of early releases (:ref:`reading-rules`). A number that is no NCBI gene id of the organism can
therefore answer the gene whose Havana number it was: human ``convert("1898")``, an NCBI gene id that NCBI
discontinued, answers five genes of the MHC haplotypes through a Havana transcript number of release 52, and
``convert("39152")``, a spreadsheet's serial number of a date that a symbol was turned into (MARCH11), answers
ENSG00000099991 through a Havana translation number of release 51. The benchmarks count mouse 149 and human 123 such
answers among NCBI gene ids of other organisms, and 6 human lines (date serials 4, discontinued NCBI ids 2) in the
comparison with other tools. Name the database of NCBI gene ids: ``convert("1898", from_database="NCBI gene (formerly
Entrezgene)")`` answers nothing. Keep symbols from being turned into dates before converting a table.

**A number written with a version that two numbering schemes share is read where its version exists.** Human
numeric Havana translation ids of releases 50 and 51 carry the version 1 and 2,493 of them are also NCBI gene ids:
``2633.1`` without a release is read as the Havana id of release 51 (another gene than NCBI's ``2633``), with NCBI's
reading reported beside it (``reason="unversioned_database"``); give the release a table was made with, or write the
NCBI id without a suffix.

A branch can outrank the gene's own continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A gene's own continuation does not outrank a branch that left it while it lived on.** Where Ensembl maps an id to
new ids at a session while the id itself continues, the candidate on a branch can have the larger product of
Ensembl's scores and become the best answer (pig ``convert("ENSSSCG00000014532", from_release=56, to_release=98)``
answers ENSSSCG00000042508, an offshoot, while the id's own successors rank fourth to sixth). Pig: 7 ids, 1,133
conversions (748 of them without a tie), all at the Sscrofa9 to Sscrofa10.2 change; human: none; mouse and the
macaques: no id is branched while it lives. ``result.view("all")`` lists the id's own successors.

.. _limitation-moves:

An id that Ensembl moves to another place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**An id that Ensembl's history carries to another place answers that id, and the result does not say that the gene
moved.** Where Ensembl keeps a stable id across a version event that puts the gene on another chromosome or strand,
idtrack follows the id as Ensembl does; only the explained path marks the step (``PathStep.places`` is
``"places_apart"``), and ``Candidate.score``, the product of Ensembl's scores along the path, shows how weak the event
is.

Ensembl moved human 609, mouse 210 and pig 34 stable gene ids to another chromosome within one assembly, and human 386,
mouse 113 and pig 226 to the other strand. Human ``convert("ENSG00000115934", from_release=61, to_release=115)`` answers
ENSG00000115934.11, a processed transcript on chromosome 12 from release 62, although at 61 the id was a protein-coding
gene on chromosome 19 (Ensembl's score of the event 0.046, ``Candidate.score`` 0.033). Where a move matters, convert
with ``explain=True`` and read the ``places`` of the history steps.

An id with two lives
~~~~~~~~~~~~~~~~~~~~

**An id that Ensembl retired and later brought back with an old version is one node with two lives.** Converted
into the gap between its lives, it answers by the life it starts from: human ENSG00000288686 (version 1 at 103,
version 2 at 104 to 107, absent at 108 and 109, version 1 again from 110) answers XG, ENSG00000124343.14, from its
first life (``convert("ENSG00000288686", from_release=107, to_release=108)``) and nothing from its second
(``from_release=110``). It is the one id found: of the human and mouse ids that Ensembl brought back after a gap or
whose version falls back to an earlier number (human 7, mouse 4), each converted between every two releases at
which it exists, both ways, and into its gaps from every release before and after them, every other id answers
alike in both directions.

An id that is a transcript at some releases and a translation at others
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**An Ensembl id that names a transcript at some releases and a translation at others is one node.** The graph keeps
one node per identifier: Ensembl 107 gives a translation of IL2RB the id ENST00000515292, the id of a transcript of
LINC02497 (ENSG00000251182) at releases 58 to 96, so ``convert("ENST00000515292", from_release=107,
to_release=107)`` answers ENSG00000251182.2, not IL2RB's ENSG00000100385.15. It is the one such id in the graphs of
the five organisms. Ask by IL2RB's transcript, ``ENST00000703410``, or its gene.

A symbol of the name authority can answer another gene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**At its own release, a symbol of the organism's name authority can answer another gene** where Ensembl displays the
name for both genes, or for neither, and more databases give the other gene the name: Ensembl's display name ranks
first, then the support of the evidence (the authority's symbol wins against another gene's synonym only). Of 1,500
symbols per organism, each asked at a release where the organism's symbol database names a gene with it: pig 5, rhesus
5, mouse 2, human 3 best answers include a gene the authority does not give the name (of them pig 2, rhesus 1, mouse 2
and human 2 ties with the authority's gene; were support ranked before the display name, it would be pig 10, rhesus 11,
mouse 7 and human 8).

Each candidate says whether an authority gave it the name (``Candidate.authority``) and whether it is the gene's display
name (``Candidate.display_name``); in rhesus the authority's names are themselves projected from human.

.. _limitation-other-sources:

A name that another source gives another gene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A string is read in every name source and database the graph holds at the release read, so the answer holds every
gene that any of them gives it.** Ensembl's display name ranks first, then the support (:ref:`concept-ranking`), and
the other genes are in ``view("all")``: human ``convert("BBS1", from_release=69, to_release=115)`` answers BBS1,
ENSG00000174483, and ``view("all")`` adds ENSG00000256349, a gene to which UniProt's gene name BBS1 is attached at
69. Where only copies on haplotypes or patches display the name, a gene of the reference sequence that any source
gives it is the answer and the copies are listed apart (:ref:`reading-copies`), also when that gene is another gene:
human ``convert("HLA-DRB3", from_release=70, to_release=70)`` answers a tie of HLA-DQB1 and HLA-DRB1, which UniProt's
gene names call HLA-DRB3 too, and lists the HLA-DRB3 genes of the MHC haplotypes apart. A string that is also a value
of another database (a symbol that is a clone name, a number that is an old Havana id) names the genes of both.

The benchmarks count these answers where the truth reads the display name or the authority's file: a name of another
name source, pig 24, mouse 476, human 2,720 and cynomolgus 8 answers (in the comparison with other tools 109 lines); a
string read as another gene's name, a synonym or at another release than the list's, pig 32, mouse 252 and human 406 (36
lines); a value of another database, pig 30, mouse 2,703 and human 7,499 (59 lines). Name the database
(``from_database="HGNC Symbol"`` answers the HLA-DRB3 genes above) and read each candidate's ``entry_databases`` and
``display_name``.

.. _limitation-spellings:

A name in one letter case that another gene's link spells so
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A name in one letter case that a link of another gene spells exactly so answers that gene, not the gene the name
authority displays with it in mixed case.** Where, at one release, Ensembl displays a name for one mouse gene as the
authority writes it (``Foxl1``) and for another gene in capitals, the string in one letter case is an exact match of the
links spelt so, and an exact match follows only those links (:ref:`reading-matching`): the authority's gene is then not
among the genes reached, and the rule that ranks the authority's display name first, read without letter case, ranks
only the genes reached.

Of the 743 spellings in one letter case of such pairs of mouse display names, each asked at its release, 424 answer the
gene that carries the name exactly as typed, mostly names of snoRNA and scaRNA families at releases 63 to 71 (reading
such strings without letter case at the exact step would undo the exact match for strings that do name a gene as typed,
which is kept). Written as the authority writes it, the name is read as that gene's display name.

.. _limitation-case:

Symbols typed in small letters where symbols are written in capitals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Where an organism's symbols are written in capitals, a symbol typed with small letters can answer another gene whose synonym is spelt so.** The rule that reads a string against the organism's letter case first as a symbol
without letter case (:ref:`reading-rules`) holds where symbols are written in mixed case
(mouse); in pig, the macaques and human an exact synonym of another gene comes first. Of the symbols of the name
authorities at release 115 that name one gene, each typed in small letters and with a capital first: pig 2 of 15,979
symbols (``convert("afp")`` answers ENSSSCG00000001233, whose NCBI synonym is ``afp``, not AFP, ENSSSCG00000008949;
``Has2`` answers ENSSSCG00000057079, not HAS2), rhesus (15,206 symbols), cynomolgus (748) and human (40,826) none.
Type the symbol as the authority writes it (``AFP``), or name its database:
``convert("afp", from_database="VGNC Symbol")`` is AFP. In the benchmarks' comparison with other tools, 78 of the
sampled wrong lines are strings matched only in another letter case (pig 39, rhesus 15, cynomolgus 15, mouse 5, human
4), mostly aliases as NCBI's files type them: read as another gene's synonym, or as a synonym of the gene the alias
names where the truth, which reads no other case, expects none (pig ``NGN3`` answers NEUROG3, ENSSSCG00000034488,
through its synonym ``Ngn3`` of release 98).

.. _limitation-bridges:

Answers that only a bridge reaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Where Ensembl's history gives a gene no successor, an answer is the gene a bridge reaches, which no record of Ensembl
states.** A bridge needs an identifier that the two genes carried one after the other and places that overlap (or,
across an assembly change that Ensembl publishes no mapping for, chromosomes of one name:
:ref:`limitation-assembly-mapping`), so its gene is the plausible successor at that place, not one that Ensembl's
history or the place proves.

Of the gene ids whose history ends without a successor, each asked at its last release to release 115, a bridge answers
human 8,679 of 34,385, mouse 4,017 of 11,375, pig 5,245 of 22,656, rhesus 3,492 of 21,890 and cynomolgus 961 of 8,483;
the others answer nothing (but 18 human ids: LRG records, which answer the gene they describe, and one id that Ensembl
brought back). Human ``convert("ENSG00000198477", from_release=48, to_release=76)`` (SUHW2 at 48, ZNF280B later, last at
release 75) answers ENSG00000275004.1, ZNF280B on GRCh38, through NCBI's id 140883, RefSeq's NM_080764 and the symbol
ZNF280B at the same place on chromosome 22. ``Candidate.bridges`` counts the bridges of an answer and
``result.path.bridges`` lists them; ``allow_bridges=False`` answers by Ensembl's history alone.

A name read at an old release whose genes' history ends is answered the same way: human ``convert("AC090377.15",
from_release=53, to_release=115)`` (a clone-based name of release 53) answers ENSG00000267497, which a bridge through
the HGNC symbol NFE2L3P1 reaches at the old gene's place on chromosome 18, and nothing with ``allow_bridges=False``. The
benchmarks count the answers that only a bridge reaches and that do not lie at the old gene's place, where their truth
follows Ensembl's history alone: pig 58,235, mouse 372,446, human 1,042,878, rhesus 6 and cynomolgus 14 answers over the
families of Ensembl ids, names, external identifiers, lists and datasets, and in the comparison with other tools 361
lines, with 74 more names read at an old release (at the old place 30, across an assembly change without a published
mapping 35, through a copy outside the reference sequence 8, and one RFAM accession read as such).

A retired gene whose accession another gene holds today
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A gene that Ensembl retired without a successor can answer another gene than the one that holds its accessions
today.** Where Ensembl retired a gene and filed its accession (an NCBI gene id, a RefSeq or UniProt accession, a miRBase
accession) on another gene before, or without, giving the accession to the gene that holds it at the last release, the
retired id, asked at its last release, travels by a bridge to the gene nearest in releases, not to the one the accession
names now: human ``ENSG00000199004`` (hsa-mir-21, retired at 84 to 85) answers its host gene VMP1, which Ensembl gave
NCBI's id 406991 of MIR21 at 85 to 87, while ``MI0000077`` and 406991 are held by ENSG00000284190, Ensembl's new miRNA
gene from 88.

Of the retired ids whose accession another gene holds at release 115 (asked at their last release, accessions of
families such as Rfam not counted), this is the answer of human 784 of 3,279 (326 of them miRNA genes; the answered gene
held the accession in the gap in 618), mouse 310 of 1,911, pig 323 of 4,550, rhesus 118 of 3,579 and cynomolgus 26 of
1,182 (the others answer that gene, human 1,553, or nothing, human 942); nearly all are protein-coding outside human.

Ask by the accession or the name at the target release (``convert("MI0000077")``), which answers the gene that holds it
then; ``result.path.bridges`` shows the bridge taken.

A split gene converted to an external database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A gene whose history splits, converted to an external database, answers no identifier where the first-ranked of its
descendants has none, although another descendant has one.** The answer is first a gene, ranked as Ensembl genes, and
the identifiers are those of the genes of that best answer; where none of them has one, the result is those genes
(``no_target``), in ``view("all")`` too, and the other descendants are listed as genes only.

Of the Ensembl gene ids whose history splits, asked at their last release to release 115 (several genes in
``view("all")``), this happens for at least one database for human 197 of 776 (532 id and database pairs), mouse 439 of
2,593 (735), pig 364 of 3,304 (646; HGNC Symbol 197 of them), rhesus 195 of 1,549 (364) and cynomolgus 84 of 873 (158);
never with another gene's identifier. Convert to Ensembl genes first, then each gene of ``view("all")`` to the database
(``convert(gene, final_database=...)``): on 100 sampled pairs per organism this gives the descendants' identifiers every
time.

.. _limitation-split-place:

Among the successors of a split, the best answer does not look at the old place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Where Ensembl's history splits a gene, the best answer is ranked by Ensembl's scores and the evidence of the
candidates, not by which successor lies where the old gene lay.** ``view("all")`` holds every successor, and with
``explain=True`` each candidate's path ends at a gene whose place the result gives (``PathNode.places``). The
benchmarks' gene lists (family D) count the best answers that lie away from the old gene model while another
successor holds it: pig 658, mouse 1,457 and human 69 answers; and where the old place holds one successor while the
answer holds others too, pig 1,498, mouse 2,156 and human 221. Human ``convert("ENSG00000272196", from_release=98,
to_release=115, explain=True)`` answers ENSG00000288825 (chromosome 1, 149,842,218 to 149,842,750, reverse strand;
``Candidate.score`` 0.972), while ENSG00000288859, the other successor (0.964), lies where the old gene lay
(149,851,061 to 149,851,652, forward strand); ``view("all")`` holds both. To map count matrices, keep the successor
that overlaps the old place.

RefSeq_mRNA in mouse releases 80 and 81
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**In mouse releases 80 and 81, RefSeq_mRNA also holds predicted and RefSeqGene accessions.** Ensembl attaches
gene-level RefSeq accessions of every kind to mouse genes at these releases under ``RefSeq_dna``, which continues
RefSeq_mRNA; the forms that never moved to another RefSeq database are filed under that main continuation, since
RefSeq_mRNA has no identifier scheme of its own (``NM_`` is 89 % of its accessions over the releases). Converted to
``RefSeq_mRNA`` at 80 or 81, a gene can answer ``XM_`` (6,832 accessions at 80, 4,426 at 81), ``XR_`` (1,264 and
1,148) and ``NG_`` (674 and 680) accessions beside its ``NM_`` ones; pig has 90 ``XR_`` accessions under
RefSeq_mRNA at 63 to 66; human (76 included), rhesus and cynomolgus have none. The prefix says what an accession
is (``NM_`` an mRNA, ``XM_`` and ``XR_`` predicted, ``NG_`` a RefSeqGene record); ask ``RefSeq_mRNA_predicted`` or
``RefSeq_ncRNA_predicted`` for the predicted ones.

The support of a name counts other letter cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The support of a name counts a database that writes it in another letter case.** Typed in mixed case, a mouse
symbol is the display name only as its database spelt it, but the number of databases that attach it to a gene
counts UniProt's gene name in capitals too. At release 62, where UniProt's names are read on translations, 63 MGI
symbols name two genes or more, UniProt names some of them in capitals, and for 53 of them that support alone
makes one gene the answer (``Adora3`` at 62 answers ENSMUSG00000000562, which MGI, NCBI and UniProt name, not the
tie with ENSMUSG00000074344, which MGI and NCBI name). ``view("all")`` lists the other gene, and each candidate's
``entry_databases`` names the databases counted.

.. _limitation-suffix:

Single-cell suffixes after a name that is no authority's symbol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A suffix of single-cell tools after a name that is no symbol of the name authority at the release asked about can be
read into another gene's synonym.** The rule that keeps ``SRF.1`` from SPATA6's synonym ``SRF-1`` (:ref:`reading-rules`)
needs the string before the suffix to be a symbol that a name authority gives at the release asked about (without a
release, the target release), also where the synonym is read at another release.

Cynomolgus ``CAV1`` is NCBI's symbol at releases 91 to 102 and HGNC's at 103 to 114, but only UniProt's name at 115, so
``convert("CAV1-1")`` is read at 98 as ``Cav1.1``, a synonym of CACNA1S, and answers ENSMFAG00000037274, while
``convert("CAV1-1", from_release=98)`` is not found. Of the names of release 115 written with ``-1``, ``.1`` and ``_1``,
each asked alone, such a string answers another gene's synonym in cynomolgus 35 of 61,284 strings, pig 4 of 59,994 (in
these 39 the string before the suffix is an authority's symbol where the synonym is read), human 10 of 131,985 and mouse
3 of 232,527 (the string before the suffix is no authority's symbol there: human ``FY.1`` is TBCK's synonym ``FY-1``);
rhesus none.

A string that is exactly a synonym or a name of another gene is that gene's, as everywhere. In a batch with the name it
repeats, as a single-cell matrix holds it, such a string is that name's copy (``convert_many(["CAV1", "CAV1-1"])``
answers CAV1 twice).

.. _limitation-lrg-gene-ids:

The GRCh37 gene ids of LRG records list their answer among the copies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**An LRG record's gene id of the GRCh37 releases answers the gene the record describes, but its result lists that
gene among the copies set apart, and not the record.** ``convert("LRG_7_g1")`` answers ENSG00000141837.24 (CACNA1A)
with ``alternative_locus_genes == ("ENSG00000141837.24",)``, while ``convert("LRG_7")`` and ``convert("LRG_7t1")``
answer the same gene with the record ``LRG_7.1`` listed apart (:ref:`reading-copies`). It concerns the 11 ids of the
form ``LRG_<n>_g<k>`` in the human graph of 115; the 342 transcript and 532 translation ids of those releases
(``LRG_<n>_t<k>``, ``LRG_<n>_p<k>``) list the record as the record's own ids do. The answer is right; code that
drops the genes of ``alternative_locus_genes`` from a result would drop it. Convert by the record's id (``LRG_7``)
or by the gene, or keep ``result.targets()`` whatever ``alternative_locus_genes`` holds.

Low-9 quotation marks
~~~~~~~~~~~~~~~~~~~~~

**Low-9 quotation marks are not read as quotes.** The quotation marks removed around an identifier are Unicode's
initial and final quotes and the ASCII ones (:ref:`reading-cleaning`); the low-9 marks that open a quotation in German
and other languages (``„`` and ``‚``) belong to Unicode's category of opening punctuation together with the brackets,
and no property of Unicode tells the two apart, so ``„TP53“`` and ``‚TP53‘`` are not found (never another gene).
Remove them before converting.

.. _limitation-databases:

Databases the graph does not hold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Identifiers of databases that the graph does not hold answer nothing.** A graph holds the databases that its
externals configuration selects (``idtrack/default_config/<organism>_externals_modified.yml``;
:py:meth:`idtrack.API.external_databases` lists them), and the shipped configurations do not select RNAcentral, the
name authorities' own ids (HGNC, MGI and VGNC ids), UniProt's secondary accessions and entry names (but the entry
names that early releases display as genes' names) or UniProt's isoforms. Such an identifier is not found, never
another gene, unless the same string is an identifier of a database the graph holds: human ``Q15086`` (a secondary
accession of TP53's P04637), ``URS0000761A87`` (RNAcentral's id of a MALAT1 transcript) and ``P53_HUMAN``, mouse
``MGI:98834`` (Trp53) and pig ``VGNC:100174`` are not found. Ask by the primary accession, the symbol or the
Ensembl id; to read another database of Ensembl, build a graph whose configuration selects it
(:doc:`databases`).

At mouse release 49 Ensembl's GTF names 21,182 genes by their MGI accession, so these names are not found at 49
either (``convert("MGI:2687328", from_release=49, to_release=49)``): in the benchmarks, 79,786 mouse answers to
names of release 49 (and 16 cynomolgus answers to names that are accessions of another database). Ask by the
gene's MGI symbol or its Ensembl id.

.. _limitation-mgi-havana:

MGI's symbols of mouse releases 48 to 54 that are Havana ids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**At mouse releases 48 to 54, a gene whose MGI symbol was its Havana id has no MGI symbol in a conversion.** MGI gave
genes without a symbol of their own their Havana gene id as their symbol, and Ensembl's records of those releases
file it as the gene's ``MGI Symbol`` label (``OTTMUSG00000000990`` of ENSMUSG00000018925 at release 52; MGI keeps it
as a withdrawn symbol of the gene, now Heatr9). The graph files every string of Havana's identifier scheme under
``Havana gene`` (:ref:`database-filing`), so ``convert("ENSMUSG00000018925", from_release=52, to_release=52,
final_database="MGI Symbol")`` answers the gene itself with ``status="no_target"``, while ``final_database="Havana
gene"`` answers ``OTTMUSG00000000990`` and, at release 115, ``final_database="MGI Symbol"`` answers ``Heatr9``. The
benchmarks ask 1,534 such conversions (374 genes, releases 48 to 54) and count 2,365 of their 3,068 answers wrong.
Convert to ``Havana gene`` at these releases, or to ``MGI Symbol`` at a later release.

.. _limitation-mirbase:

miRBase's names on miRNA genes only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The graph holds miRBase's names and accessions on genes of a miRNA biotype only, and leaves out an entry that Ensembl
gives to genes at two loci.** Such a name therefore answers the miRNA gene, or nothing, where Ensembl's records give it
to other genes too (:doc:`changes`, 0.1.0). At release 76 Ensembl displays the names of rhesus and pig miRNAs also for
their host genes: rhesus ``convert("mml-mir-126", from_release=76, to_release=76)`` answers the miRNA gene
ENSMMUG00000032649, not its host EGFL7 (ENSMMUG00000026981), whose display name it is there too. The benchmarks count 62
answers to a miRBase name that is not found (human 24, pig 14, rhesus 24) and, among the display names of miRBase and of
names of several genes at early releases, answers read at another release (pig 20, mouse 40, human 56, rhesus 28); in
the comparison with other tools, 17 lines (8 of them such host genes). Ask by the gene's Ensembl id or its symbol
(``MIR126``).

Synonyms that are fragments of UniProt's evidence text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A few synonyms are fragments of UniProt's evidence text.** The graph drops a synonym that carries the evidence
UniProt writes in a gene name (``{ECO:...}``), but where Ensembl's rows split such a text, the piece without its
opening brace stays a synonym of the gene: human 45 synonyms, mouse 20, pig and the macaques none
(``api.genes_named("GNECO:0000312|HGNC:HGNC:11125}")`` gives UniProt's synonym of SUMO2, ENSG00000188612, at
release 79). They name the gene they came with and are reached only by typing such a string;
:py:meth:`idtrack.API.names_of` lists them among a gene's synonyms.

.. _limitation-cross-references:

Ensembl's cross references against the databases' own files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**An identifier of another database answers the gene Ensembl's cross-references give it, also where the database's own
file says otherwise.** Ensembl's tables are the record idtrack reads; where NCBI, RefSeq or UniProt give an identifier
another gene, or none, in their own files, the answer is still Ensembl's gene. At release 115 the databases' own files
(NCBI's ``gene2ensembl`` and ``gene2refseq`` and UniProt's ``idmapping_selected``, as fetched in November 2026) give
another gene or none to these shares of the identifiers that Ensembl gives a gene: NCBI gene ids cynomolgus 52.8 %
(7,158 of 13,563; no gene 6,554), rhesus 12.6 % (2,695 of 21,446); RefSeq accessions cynomolgus 44.8 %, rhesus 39.9 %;
UniProt accessions cynomolgus 5.4 %, rhesus 14.4 %, pig 38.6 %.

NCBI's files give pig's ids to the genes of another Ensembl annotation of pig (``ENSSSCG00220...``), so none of them
names a gene of the reference assembly; human and mouse were not compared. Rhesus ``convert("100425060")`` answers
ENSMMUG00000002917, the gene Ensembl gives NCBI's id 100425060, while NCBI's ``gene2ensembl`` gives it
ENSMMUG00000049684. Where the authority's assignment matters, check the answer against the authority's own file.

The same holds for an NCBI gene id that NCBI discontinued or replaced and that Ensembl's records keep as a synonym of
a gene's NCBI or UniProt link: mouse ``convert("668009")`` answers ENSMUSG00000078896 (``match="synonym:exact"``, read
at release 87), while NCBI's gene history gives the id's successor to ENSMUSG00000078879 (3 mouse lines in the
benchmarks' comparison with other tools).


.. _known-costs:

Known costs
-----------

Measured with this version on the graphs of Ensembl 115, on a Linux server (the lower median of three runs, each in a
new process); :doc:`performance` gives the build, load, memory and conversion times of each organism.

The first conversion that meets a copy of a gene computes a table of every gene, once per process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where a bridge would join a copy on a patch, a haplotype or an LRG record to another gene through an identifier both
carry, idtrack asks whether the genes that carry it are one gene (:ref:`concept-bridges`), and the first such question
in a process computes the facts of every gene of the graph and keeps them: about 10 s and 0.8 GB in human, 9 s and 0.7
GB in mouse; later calls do not pay it. Pig and the macaques have no such copies, and none of their conversions computes
it. A server that loads the graph and then forks its workers computes it again in every worker that meets such a copy (a
server of the five organisms with four workers: up to 2.0 GB of private memory per worker, its slowest call 16 s).

Make one such conversion when the process starts, before the first request and before a server forks, so that the
workers inherit the table: ``api.convert("ENSG00000228326.1")`` in human, ``api.convert("ENSMUSP00000155749.1")`` in
mouse (each computes it; asked again, it answers in a few milliseconds).

A name that is the display name of very many genes takes seconds on every call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensembl displays the names of some RNA families for hundreds or thousands of genes, and ranking them by their
display names takes time that grows with the number of genes: cynomolgus ``Metazoa_SRP`` (3,274 genes, a tie) 6.2 s
and ``RF00026`` (1,212 genes) 0.7 s, rhesus ``Metazoa_SRP`` (3,419 genes) 7.3 s and ``U6`` (1,209 genes) 2.3 s, pig
``U6`` (876 genes) 1.2 s, each time it is asked: nothing is kept between calls. These names make the slowest calls of
pig and the macaques. Convert such a name once and reuse its result; in one :py:meth:`idtrack.API.convert_many` each
distinct string is converted once (pig ``U6`` three times in one batch takes the time of one call).

The harmonizer reads its own result cache slowly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructed again over a valid result cache, :py:class:`idtrack.harmonize.HarmonizeFeatures` takes about as long as
converting every feature again (0.7 to 1.4 times as long, by organism): on three test datasets of 50 cells that hold every gene of the organism (by id at
release 115, by id at an older release, by symbol), the first construction took pig 38 s and human 199 s, the
construction over the cache pig 52 s and human 130 s (the time goes into reading each stored result back into its
typed form). Keep the harmonizer object, or the merged AnnData written to a file, rather than constructing it again.

Threads share a graph but do not convert in parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A loaded graph is only read by conversions, so several threads can use one :py:class:`idtrack.API`, but Python's
interpreter lock lets one convert at a time: four threads took 1.2 to 2.5 times as long as one thread for the same
batch. To convert in parallel, use processes (each loads the graph: :doc:`performance` gives its memory).
