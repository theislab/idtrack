.. _concepts:

===================
How IDTrack Answers
===================

This page says what an answer of :py:meth:`idtrack.API.convert` means, how it was found, and when it can be wrong.
Each term is defined once, here, and the other pages link to it. How the typed string is read before any of this
happens is on its own page, :doc:`reading`; the cases in which the rules below give a wrong or a missing answer,
with how often they occur, are in :doc:`limitations`.

.. contents::
    :local:
    :depth: 2
    :backlinks: none


.. _concept-graph:

The graph
---------

A **graph** holds the identifier history of one organism, built once from Ensembl's own tables (:doc:`performance`
says how long that takes) and then loaded from its file. Its nodes are identifiers: Ensembl gene, transcript and
translation ids with their versions, and the ids and names of the **external databases** the graph holds (HGNC and
NCBI symbols, RefSeq and UniProt accessions, NCBI gene ids, ...: :doc:`databases`). Its edges are of two kinds:

* **history**: Ensembl's id history (its ``mapping_session`` and ``stable_id_event`` tables): an id that continues
  with a new version, is replaced, split or merged between two releases, with Ensembl's mapping score of the event;
* **cross references**: at a release, an Ensembl gene (or one of its transcripts or translations) carries an id of an
  external database. Every such edge holds for the releases at which Ensembl's tables give it.

Besides the identifiers, the graph holds where every gene version lies (assembly, sequence region, first and last
base, strand: :py:meth:`idtrack.TheGraph.places`), Ensembl's display name of every gene at every release, and the
version that each cross reference attached to an accession. Loading a graph needs no table and no network.

.. code-block:: text

    a sketch (ids shortened):

    release:        100                    105                    110                    115
    history:   ENSG_A.1 ──new version──▶ ENSG_A.2 ──────split──────▶ ENSG_B.1 ─────────▶ ENSG_B.1
                                                        └───split──────▶ ENSG_C.1 ─────────▶ ENSG_C.3
    cross refs:   │                         │                             │
               "OLD1" (HGNC Symbol)    "OLD1" (HGNC Symbol)          "NEW1" (HGNC Symbol)

    convert("OLD1", from_release=100) reads OLD1 at 100 (entry gene ENSG_A), follows the history to 115 and
    answers the genes that descend from it there (ENSG_B.1 and ENSG_C.3, ranked: below).

A graph is fixed by four things, and the same graph gives the same answer to the same call in every run:

* the **organism** (:doc:`organisms`);
* the **snapshot release**: the last Ensembl release the graph holds (``snapshot_release``: 115 in these pages). A
  graph knows nothing after it;
* the **window**: every release from the organism's first release in idtrack (48 for human, mouse and rhesus, 56 for
  pig, 91 for cynomolgus) to the snapshot release (:py:meth:`idtrack.API.releases`). A conversion goes from any
  release of the window to any release of the window, so a graph of release 115 also converts to release 93;
* the **externals configuration**: the YAML file that selects the external databases (:doc:`databases`). Every
  organism ships one, so a graph is built without writing any.

The graph file is named after the organism, the primary assembly and the window
(``graph_homo_sapiens_assembly38_min48_max115.npz``) and records what it is and what built it
(:py:meth:`idtrack.API.describe`); graphs of several organisms and snapshot releases live side by side in one
**local repository**, the folder of downloaded tables and graph files.


.. _concept-releases-assemblies:

Releases and assemblies
-----------------------

The **starting release** is the release an identifier is read at (``from_release``; how it is chosen when not given:
:ref:`reading-starting-release`). The **target release** is the release the answer is given at (``to_release``,
default the snapshot release). Between them, the conversion follows the identifier's history.

An organism's genome assemblies follow one another (human NCBI36, GRCh37, GRCh38; :doc:`organisms` lists each
organism's). At every release one assembly **serves** the release: the assembly Ensembl maintained then (human GRCh38
from release 76). An identifier exists at a release only on the assembly that serves it; human GRCh37, which Ensembl
still lists beside GRCh38 with its annotation of release 75, is in the human graph as identifiers to convert *from*,
not as a second present. An identifier of an earlier assembly converts to the gene of the assembly that serves the
target release, through Ensembl's history across the assembly change.


.. _concept-history:

History travel and Ensembl's scores
-----------------------------------

From the gene an identifier names at the starting release (its **entry gene**), the search follows Ensembl's history
forward or backward to the target release. Each history step is a **version event** of Ensembl:

.. list-table::
    :header-rows: 1
    :widths: 22 78

    * - Event (``PathStep.reason``)
      - What Ensembl recorded
    * - ``new_version``
      - The id continues with a new version (the gene's model changed).
    * - ``replaced``
      - The id was replaced by another id.
    * - ``split``, ``merge``, ``split_and_merge``
      - One id became several, several became one, or both at one release.
    * - ``self_loop``
      - Ensembl mapped the gene to itself again, with a score.

Each event carries Ensembl's **mapping score** (0 to 1: how much of the old gene the new one keeps). A
candidate's ``score`` is the product of the scores along its path. Ensembl writes 0 where it measured nothing between
two ids; such a step has no score (``PathStep.weight`` is ``None``) and a path with one ranks below every path whose
scores are all measured (``Candidate.unscored_steps``).

One kind of step is not followed: a step between a gene of the mitochondrial genome and a gene of a nuclear
sequence is no lineage (at three assembly changes Ensembl gave the ids of mitochondrial genes to nuclear genes:
rhesus ``MT-ND1`` of release 97 led to an unnamed gene of chromosome 8). The id then has no successor across that
step, and the explained path lists the step among ``result.path.refused`` with ``reason == "mitochondrial_genome"``;
the two sides of such a step are two genes, which no bridge joins either.


.. _concept-bridges:

Bridges and their checks
------------------------

Where Ensembl's history leads to no gene at the target release (a gene retired without a successor), the search may
pass from that gene to another through an identifier the two genes carried one after the other: a **bridge**
(``Candidate.bridges`` counts them; :py:attr:`idtrack.settings.SearchSettings.bridge_limit`, 2 by default, bounds them
per path; ``allow_bridges=False`` turns them off). No record of Ensembl states a bridge: it is idtrack's inference that
one gene became the other, so it is taken only where the identifier is evidence of that:

.. list-table::
    :header-rows: 1
    :widths: 28 72

    * - Check (``BridgeCheck.rule``)
      - A bridge is taken or refused because
    * - ``same_place`` (taken)
      - The two genes overlap on one assembly (same sequence region and strand), or, across an assembly change for
        which Ensembl publishes a mapping of the two assemblies (human, mouse, pig Sscrofa9 to Sscrofa10.2), the
        earlier place carried onto the later assembly overlaps the later gene.
    * - ``never_shared`` (taken)
      - The places cannot be compared (an assembly change without a published mapping), and no release gave the id
        to both genes: the id left one gene for the other.
    * - ``outside_reference`` (taken)
      - A copy on a patch, haplotype or LRG record carried the id together with the other gene at one release, and
        at every such release the genes that carried it are one gene.
    * - ``synonym`` (refused)
      - A synonym names other genes too often to stand for one.
    * - ``places_apart`` (refused)
      - The two genes do not overlap, or the published mapping puts the earlier gene elsewhere, or, across a change
        without a mapping, they lie on chromosomes of different names (an organism's assemblies name its chromosomes
        alike, by its karyotype).
    * - ``copy_id_of_several_genes``, ``copy_id_elsewhere`` (refused)
      - The id of a copy names several genes at that release (a family label, a shared alias), or came to the other
        gene only at another time.
    * - ``mitochondrial_genome`` (refused)
      - The two genes are the two sides of a refused mitochondrial step (above).
    * - ``shared_at_one_release`` (refused)
      - The graph holds no place of a gene (a graph made in memory) and both genes carried the id at one release.

Identifiers of databases that name a family or a clone, not a gene (Rfam's accessions, the clone-based names:
:py:data:`idtrack.settings.NON_BRIDGING_DATABASES`), and labels that named more genes over their releases than
:py:attr:`idtrack.settings.GraphSettings.hyperconnective_threshold`, are never bridges. ``result.path.bridges`` lists
every bridge the search considered, taken or refused, with its rule (:py:class:`idtrack.results.BridgeCheck`), also
when nothing was found. How often a bridge decides an answer, and why its answer is a plausible successor rather
than a proven one: :doc:`limitations`.


.. _concept-places:

Places, and copies of genes
---------------------------

A gene's **place** is its sequence region, bases and strand on an assembly at the releases it lies there
(:py:class:`idtrack.graph.Place`; ``Place.kind`` says whether the region is a chromosome, the mitochondrial genome or a
scaffold, read from Ensembl's sequence regions). Places decide bridges (above) and how many genes an identifier names
(:py:meth:`idtrack.TheGraph.count_genes`: the successive stable ids of one locus are one gene; two ids that live at
one release are two genes unless their places overlap; :py:mod:`idtrack.graph.genes`).

Some genes are **copies** of a gene of the reference sequence, with stable ids of their own: genes on alternative loci
(haplotypes) and assembly patches (human; mouse GRCm38), LRG records (human), and, from release 110, the Y copy of a
gene of a pseudoautosomal region. A copy on a patch or a haplotype stands for the region of the reference sequence
its assembly exception replaces (Ensembl's ``assembly_exception``). When an identifier leads to a gene of the
reference sequence and to copies, the copies are no part of the answer: they are listed in
``result.alternative_locus_genes``. When only copies are reached, they are the answer. An LRG record's ids answer the
gene the record describes (the match prefixed ``lrg:``, the record listed apart). A pseudoautosomal gene's name
answers its X copy, the Y copy listed apart (the regions are GRC's,
:py:data:`idtrack.organisms.PSEUDOAUTOSOMAL_REGIONS`).

A queried Ensembl gene id that exists at the target release is **its own answer**: its own version comes first, is
never set apart as a copy, and is the best answer whatever else its history reaches.


.. _concept-ranking:

How candidates are ranked
-------------------------

The search returns every **candidate** it reached (:py:class:`idtrack.results.Candidate`): a gene at the target
release and its **target** (the gene id itself, or its id in the database asked for, ``final_database``). The
candidates are ranked by the criteria of :py:data:`idtrack.results.CRITERIA`, in this order; the first criterion in
which two candidates differ decides between them, and ``result.decided_by`` names the one that put the best answer
before the runner-up.

.. list-table::
    :header-rows: 1
    :widths: 18 50 32

    * - Criterion
      - Compares
      - First is
    * - ``own_version``
      - Whether the gene is the queried gene id's own version at the target release.
      - The id's own version.
    * - ``final_confidence``
      - How the target is attached to the gene: 0 to the gene at the target release, 1 to a transcript or
        translation the gene had at another release; ``None`` when no gene has an id in ``final_database`` and the
        gene stands in for it (``no_target``, below).
      - 0, then 1.
    * - ``bridges``
      - The number of bridges on the path.
      - Fewer bridges (0: Ensembl's history alone).
    * - ``bridge_gap``
      - The releases between leaving a gene and entering the next, summed over the bridges.
      - The smaller gap.
    * - ``display_name``
      - Whether the identifier is Ensembl's display name of the entry gene at the starting release
        (``gene.display_xref_id``: Ensembl's own statement of the gene's name, whatever database it came from;
        ``Candidate.display_rank``). Letter case counts where the typed string has both cases.
      - The gene whose display name it is (rhesus ``ATP6V0C`` at 69 is ENSMMUG00000019094, which Ensembl displays
        so, not ENSMMUG00000005679, which NCBI and UniProt name with it).
    * - ``entry_support``
      - The number of databases that attach the identifier to the entry gene at the starting release.
      - More databases.
    * - ``score``
      - Between successors of one entry gene: steps without a score, then the product of Ensembl's scores.
      - Measured scores, then the higher product.
    * - ``same_as_query``
      - Whether the target is the identifier itself.
      - The identifier itself.
    * - ``annotation``
      - Between successors still equal: the gene's external ids, transcripts and translations at the target release.
      - The richer gene.

The first five after ``own_version`` compare the **lineage** (which entry gene the identifier stands for); ``score``,
``same_as_query`` and ``annotation`` compare successors of one entry gene. Where the history merges several entry
genes into one gene, the display name and support of that gene are the best of all its entry genes. Nothing is chosen
by position or by hash order.

With ``final_database`` the answer is first a gene: the genes are ranked as Ensembl genes, and a target is an id that
the database attaches at the target release to a gene of that best answer. Where no gene of the best answer has one,
the result is the genes themselves (``status == "no_target"``), never an id of a lower-ranked gene: pig ``PSME1`` is
ENSSSCG00000002006 (its display name), which has no VGNC symbol, so ``convert("PSME1", final_database="VGNC Symbol")``
answers that gene, not ``DCAF11``, the symbol of another gene that UniProt names PSME1.


.. _concept-ties:

The best answer and ties
------------------------

The **best answer** is the first-ranked candidate, or, when several genes are equal in every lineage criterion, all of
them: a **tie** (``result.tie`` is true, ``status == "tie"``), in ascending order of gene id and target id. idtrack
returns a tie rather than a guess: human ``SNORD116`` answers the three genes whose name it is at 115. A tie is judged
among the genes that can be named in the final database: genes of a tie without an id there leave the result
(``result.no_target_genes``), so a symbol that two databases give to two genes is a tie converted to Ensembl gene ids,
and one answer converted to a symbol database in which only one of the two genes has a symbol. Several targets of one
gene (two RefSeq accessions of one gene) are one answer, not a tie.

``result.view("best")`` and ``result.view("all")`` are two views of one search; ``result.targets("best")`` (the
default) and ``result.targets("all")`` their target ids. In ``"all"`` the genes of the best answer lead and the others
follow by rank, the targets of one gene together.


.. _concept-statuses:

Statuses and reasons
--------------------

Every result has one ``status``; a result without a target, or with a caveat, has one or more ``reasons``
(``result.reasons``, in this order: why there is no target, that the identifier was read at another release, that a
version was not judged; ``result.reason`` is the first). :py:meth:`idtrack.API.reason_text` gives a reason's sentence
(:py:data:`idtrack.results.REASONS`).

.. list-table::
    :header-rows: 1
    :widths: 17 23 35 25

    * - ``status``
      - ``reason``
      - Means
      - What to do
    * - ``converted``
      - none (or a caveat below)
      - One gene at the target release; its targets.
      -
    * - ``tie``
      - none (or a caveat below)
      - Several genes equal in all evidence.
      - Give ``from_release`` or ``from_database`` where you know them; else keep the tie.
    * - ``no_target``
      - ``no_id_in_final_database``
      - The genes reached have no id in ``final_database``; the targets are the genes themselves.
      - Use the gene ids, or another database.
    * - ``not_found``
      - ``not_in_graph``
      - The string is no identifier of this graph at any release (unknown, or of a database the graph does not
        hold).
      - Check the spelling and :doc:`reading`; see whether the database is held (:doc:`databases`).

    * - ``unknown_version``
      - ``unknown_version``
      - An Ensembl id or accession with a version the graph does not hold; ``result.known_versions`` lists those
        it does.
      - Correct the version, or convert the id without it.
    * - ``no_conversion``
      - ``not_an_id_of_database``
      - With ``from_database``: the string is an Ensembl id, not an id of the database named.
      - Drop ``from_database`` for Ensembl ids.
    * - ``no_conversion``
      - ``never_active``
      - An Ensembl id whose versions are never active on an assembly that serves a release of the graph.
      - Check the id and its assembly (:ref:`concept-releases-assemblies`).
    * - ``no_conversion``
      - ``no_gene_at_start``
      - At the release read, the identifier is attached to no gene.
      - Give the release your table was made with.
    * - ``no_conversion``
      - ``no_path_to_target``
      - Ensembl's history (and the bridges allowed) leads to no gene at the target release: a retired id.
      - ``result.path.bridges`` with ``explain=True`` shows what was considered.
    * - ``error``
      - ``error``
      - In a batch: this item raised (an empty string, a missing value, a search past its budget); ``result.error``
        has its type and message.
      - Correct the item; a single call raises the error instead.
    * - (any with targets)
      - ``read_at_another_release``
      - The graph does not hold the identifier at the ``from_release`` given; it was read where it does
        (``from_release`` says where; ``from_release_fallback`` is true), so the answer can be another gene than the
        one it meant at the release given.
      - Check the release you gave.
    * - (any)
      - ``version_not_judged``
      - The version written after an accession was not judged (its database's number counts edits of the record,
        not of the sequence: :ref:`reading-accession-versions`).
      - Nothing; the accession was read without it.

"Not found", "no conversion" and "no target" are three different answers: the string means nothing in the graph; it
means something that has no gene at the target release; it has a gene, but not an id in the database asked for.

The same outcome is counted in four places, with these words:

.. list-table::
    :header-rows: 1
    :widths: 22 24 30 24

    * - ``result.status``
      - :py:meth:`idtrack.API.classify` bin
      - Harmonizer ``conversion_report()["status"]``
      - Harmonizer ``var["idtrack_status"]``
    * - ``converted`` (one target)
      - ``matching_1_to_1``
      - ``converted``, or ``converted_no_final_id`` (no name in the final database)
      - ``converted``, ``converted_no_final_id``, ``merged``, ``unconverted_merge_partner``
    * - ``converted`` (several targets of one gene), ``tie``
      - ``matching_1_to_n``
      - ``converted`` (one gene), ``ambiguous`` (several genes)
      - ``converted``, ``unconverted_ambiguous``
    * - ``no_target``
      - ``alternative_target_1_to_1`` or ``alternative_target_1_to_n``
      - ``converted_no_final_id`` or ``ambiguous``
      - ``converted_no_final_id``, ``unconverted_ambiguous``
    * - ``not_found``, ``unknown_version``
      - ``matching_1_to_0``
      - ``not_in_graph``
      - ``unconverted_not_in_graph``
    * - ``no_conversion``
      - ``matching_1_to_0``
      - ``no_conversion``
      - ``unconverted_no_conversion``
    * - ``error``
      - ``matching_1_to_0``
      - ``missing_value`` or ``error``
      - ``unconverted_missing_value``, ``unconverted_error``
    * - any, with ``read_at_another_release``
      - ``read_at_another_release``
      - its status; ``reason`` names the release read
      - its status; ``idtrack_reason`` names the release read

The "1→0, 1→1, 1→n" of the tutorials are these bins. :py:meth:`idtrack.API.classification_summary` writes their
counts; in it, "Changed only" counts results whose target is not the identifier itself (a symbol converted to its
gene id is "changed"; a gene id without version converted to its own versioned id is not), "Alternative targets"
counts ``no_target`` results (genes standing in for a missing id of the final database), and "Rest" the 1→1 results
that are neither. :doc:`harmonization` explains the harmonizer's columns.


.. _concept-explained-path:

The explained path
------------------

``convert(..., explain=True)`` adds ``result.path``, an :py:class:`idtrack.results.ExplainedPath`: one graph per
conversion with every node once (:py:class:`idtrack.results.PathNode`: kind, label, databases, active releases, the
assemblies that serve them, and for a gene its places) and every step once (:py:class:`idtrack.results.PathStep`: its
kind, ``entry``, ``history``, ``self_loop``, ``bridge_near``, ``bridge_far``, ``final`` or ``identity``; the releases;
Ensembl's score; the splits and merges of a history event; where the later gene lies relative to the earlier one,
``places``; the databases and assembly of a cross reference; why it was taken, ``reason``). Each candidate lists the
indices of its steps (``path`` to the gene, ``final_path`` from the gene to the target). ``result.path.bridges``
lists every bridge considered, and ``result.path.refused`` the history steps refused.

.. code-block:: python

    from idtrack import API

    api = API.load(API.graph_path("homo_sapiens", 115, "./idtrack_cache"))
    result = api.convert("ENSG00000198477", from_release=48, to_release=76, explain=True)
    print(result.status, result.targets())
    best = result.view("best")[0]
    for index in best.path + best.final_path:
        step = result.path.steps[index]
        at = step.release if step.release is not None else f"{step.old_release} to {step.new_release}"
        print(step.kind, step.source, "->", step.target, "| release", at, "|", step.reason)

.. code-block:: text

    converted ('ENSG00000275004.1',)
    entry ENSG00000198477 -> ENSG00000198477.2 | release 48 | entry
    history ENSG00000198477.2 -> ENSG00000198477.3 | release 49 to 50 | new_version
    bridge_near ENSG00000198477.3 -> ZNF280B | release 75 | bridge_near
    bridge_far ZNF280B -> ENSG00000275004.1 | release 76 | bridge_far

Every result is a frozen dataclass with ``to_dict()``, strict JSON (``json.dumps(result.to_dict(),
allow_nan=False)`` works), and ``ConversionResult.from_dict`` reads it back.
