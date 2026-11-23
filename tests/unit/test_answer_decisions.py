"""What a conversion answers by decisions 56, 57, 79, 86, 94 to 97, on small graphs written by hand.

Each test names the plausible wrong version it fails under. The graphs are written with the writer of
:py:mod:`tests.unit.test_explained_path` (releases 1 to 9; assembly 37 serves 1 to 4, 38 serves 5 to 9).
"""

from __future__ import annotations

import math

import pytest

from idtrack import API
from idtrack.errors import InvalidArgumentError, UnknownDatabaseError
from idtrack.graph.store import _Places
from idtrack.graph.types import ReleaseSet
from idtrack.results import ConversionResult, SourceInference
from tests.unit.test_explained_path import _Writer

NAN = math.nan
ALL = range(1, 10)
EARLY, LATE = range(1, 5), range(5, 10)


def best(result: ConversionResult) -> list[str]:
    """The genes of the best answer."""
    return sorted({c.gene for c in result.best})


def mask(releases) -> int:
    """The bit mask of releases."""
    return sum(1 << r for r in releases)


@pytest.fixture(scope="module")
def api() -> API:
    """The genes of these tests in one graph (human)."""
    w = _Writer()
    # decision 57: a split whose two successors are reached by Ensembl's score 0 (stored as no score) and by 0.1
    w.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
    w.event("A.1", "B.1", 4, NAN).event("A.1", "C.1", 4, 0.1)
    # decision 86: one number in two numbering schemes (two genes), and in one scheme on two genes
    w.gene("E", [(1, 1, 9)]).gene("F", [(1, 1, 9)])
    w.xref("123", "E.1", "EntrezGene", ALL).xref("123", "E.1", "WikiGene", ALL)
    w.xref("123", "F.1", "Havana translation", ALL)
    w.gene("H", [(1, 1, 9)]).gene("K", [(1, 1, 9)])
    w.xref("456", "H.1", "EntrezGene", ALL).xref("456", "H.1", "WikiGene", ALL).xref("456", "K.1", "EntrezGene", ALL)
    # decisions 56 and 94: a symbol, a name of another database, a synonym
    w.gene("S", [(1, 1, 9)]).xref("SYMS", "S.1", "HGNC Symbol", ALL).xref("NAMEU", "S.1", "UniProtKB Gene Name", ALL)
    w.gene("Z", [(1, 1, 9)]).xref("synonym_id::OLDSYM", "Z.1", "synonym_id::HGNC Symbol", ALL)
    # decision 95: a pseudoautosomal name on the X gene and its Y copy (places set below)
    w.gene("PX", [(1, 1, 9)]).gene("PY", [(1, 5, 9)])
    w.xref("PARG", "PX.1", "HGNC Symbol", ALL).xref("PARG", "PY.1", "HGNC Symbol", LATE)
    # ... and a name of a gene in that region of Y and of a gene of X outside the region Y repeats
    w.gene("QX", [(1, 1, 9)]).gene("QY", [(1, 5, 9)])
    w.xref("PARQ", "QX.1", "HGNC Symbol", ALL).xref("PARQ", "QY.1", "HGNC Symbol", LATE)
    # ... and a name of a gene in that region of Y and of a gene in the region of X it repeats, at another place
    w.gene("WX", [(1, 1, 9)]).gene("WY", [(1, 5, 9)])
    w.xref("PARW", "WX.1", "HGNC Symbol", ALL).xref("PARW", "WY.1", "HGNC Symbol", LATE)
    # decision 93: GENCODE's `<id>_PAR_Y` of a gene of X's pseudoautosomal region whose Y copy has an id of its own
    w.gene("ENSG00000000093", [(1, 1, 9)]).gene("ENSG00000000094", [(1, 5, 9)])
    w.xref("PARX", "ENSG00000000093.1", "HGNC Symbol", ALL).xref("PARX", "ENSG00000000094.1", "HGNC Symbol", LATE)
    # decision 96: an LRG record that carries its gene's symbol (the record lies outside the reference sequence)
    w.gene("LRG_9", [(1, 1, 9)]).gene("R", [(1, 1, 9)])
    w.xref("SYMR", "R.1", "HGNC Symbol", ALL).xref("SYMR", "LRG_9.1", "HGNC Symbol", ALL)
    # ... with its transcripts (LRG_9t1-1 is an Ensembl transcript id of its own), and a record of the GRCh37 releases
    # whose gene has an id of its own on the region named by the record (LRG_8_g1 on LRG_8, places set below)
    w.transcript("LRG_9t1.1", "LRG_9.1", ALL).transcript("LRG_9t1-1.1", "LRG_9.1", ALL)
    w.gene("LRG_8_g1", [(1, 1, 4)]).gene("Q", [(1, 1, 9)])
    w.xref("SYMQ", "Q.1", "HGNC Symbol", ALL).xref("SYMQ", "LRG_8_g1.1", "HGNC Symbol", EARLY)
    # I151: a name two databases give PG1 (no HGNC symbol) and one database gives PG2 (HGNC symbol OTHERSYM)
    w.gene("PG1", [(1, 1, 9)]).gene("PG2", [(1, 1, 9)])
    w.xref("TWOGENES", "PG1.1", "NCBI gene (formerly Entrezgene)", ALL)
    w.xref("TWOGENES", "PG1.1", "UniProtKB Gene Name", ALL)
    w.xref("TWOGENES", "PG2.1", "UniProtKB Gene Name", ALL).xref("OTHERSYM", "PG2.1", "HGNC Symbol", ALL)
    # decision 79: six names of early genes, and eight names renamed at release 5 (synonyms after it)
    for i in range(6):
        w.gene(f"V{i}", [(1, 1, 4)]).xref(f"EARLY{i}", f"V{i}.1", "HGNC Symbol", EARLY)
    for i in range(8):
        w.gene(f"N{i}", [(1, 1, 9)]).xref(f"RENAMED{i}", f"N{i}.1", "HGNC Symbol", EARLY)
        w.xref(f"synonym_id::RENAMED{i}", f"N{i}.1", "synonym_id::HGNC Symbol", LATE)
    graph = w.graph
    store = graph._frozen()
    index = store.index
    store.places = _Places.of(
        {
            (index["PX.1"], 38, "X", 100_000, 200_000, 1): mask(LATE),
            (index["PY.1"], 38, "Y", 100_000, 200_000, 1): mask(LATE),
            (index["QX.1"], 38, "X", 50_000_000, 50_100_000, 1): mask(LATE),
            (index["QY.1"], 38, "Y", 300_000, 400_000, 1): mask(LATE),
            (index["WX.1"], 38, "X", 1_000_000, 1_100_000, 1): mask(LATE),
            (index["WY.1"], 38, "Y", 2_000_000, 2_100_000, 1): mask(LATE),
            (index["ENSG00000000093.1"], 38, "X", 500_000, 600_000, 1): mask(LATE),
            (index["ENSG00000000094.1"], 38, "Y", 500_000, 600_000, 1): mask(LATE),
            (index["LRG_8_g1.1"], 37, "LRG_8", 5_001, 25_000, 1): mask(EARLY),
        }
    )
    store.non_reference[index["LRG_9.1"]] = ReleaseSet.from_mask(mask(ALL))
    store.non_reference[index["LRG_8_g1.1"]] = ReleaseSet.from_mask(mask(EARLY))
    return API(graph)


# ---------------------------------------------------------------------------------------------- decision 57


def test_a_step_ensembl_scored_0_ranks_below_a_measured_score(api):
    """A splits into B (score 0: no score) and C (0.1): C is the best answer, B ranks after it.

    Fails if no score is read as the factor 1 (B would win with 1.0 against 0.1).
    """
    result = api.convert("A.1", to_release=9, explain=True)
    assert best(result) == ["C.1"] and result.decided_by == "score"
    by_gene = {c.gene: c for c in result.candidates}
    assert (by_gene["B.1"].unscored_steps, by_gene["C.1"].unscored_steps) == (1, 0)
    assert by_gene["C.1"].score == pytest.approx(0.1)
    weights = {s.target: s.weight for s in result.path.steps if s.kind == "history"}
    assert weights["B.1"] is None and weights["C.1"] == pytest.approx(0.1)


# ------------------------------------------------------------------------------------------ decisions 86, 94


def test_a_number_in_two_numbering_schemes_is_a_tie_with_its_databases_named(api):
    """``123`` is E's NCBI gene id (and WikiGene's) and F's Havana number: a tie, each with its databases.

    Fails if the count of databases decides between two numbering schemes (E would win by two to one).
    """
    result = api.convert("123")
    assert best(result) == ["E.1", "F.1"] and result.tie
    by_gene = {c.gene: c for c in result.candidates}
    assert by_gene["E.1"].entry_databases == ("EntrezGene", "WikiGene")
    assert by_gene["F.1"].entry_databases == ("Havana translation",)


def test_within_one_numbering_the_support_still_decides(api):
    """``456``: EntrezGene gives it to H and K, WikiGene to H too: H. Fails if every number became a tie."""
    result = api.convert("456")
    assert best(result) == ["H.1"] and result.decided_by == "entry_support"


def test_the_ranking_record_names_the_authority(api):
    """Decision 94: an HGNC symbol is read from an authority, a UniProt gene name is not. Fails without the record."""
    (symbol,) = api.convert("SYMS").best
    (name,) = api.convert("NAMEU").best
    assert (symbol.authority, symbol.entry_databases) == (True, ("HGNC Symbol",))
    assert (name.authority, name.entry_databases) == (False, ("UniProtKB Gene Name",))


# ---------------------------------------------------------------------------------------------- decision 56


def test_a_synonym_is_written_as_it_is(api):
    """No result names the graph's prefix ``synonym_id::``: the match, the record and the path say it is a synonym.

    Fails if any field keeps the graph's bookkeeping (the graph id, the databases, a path node or step).
    """
    result = api.convert("OLDSYM", explain=True)
    assert (result.graph_id, result.match, result.from_databases) == ("OLDSYM", "synonym:exact", ("HGNC Symbol",))
    assert best(result) == ["Z.1"]
    (node,) = (n for n in result.path.nodes if n.kind == "synonym")
    assert (node.id, node.label, node.databases) == ("OLDSYM", "OLDSYM", ("HGNC Symbol",))
    assert "synonym_id" not in str(result.to_dict())
    (named,) = api.genes_named("OLDSYM")
    assert (named.name, named.database, named.synonym) == ("OLDSYM", "HGNC Symbol", True)


def test_a_database_covers_its_synonyms_and_synonyms_are_no_target(api):
    """``from_database`` names a database for its ids and the synonyms it records; no list offers ``synonym_id::``.

    Fails if a synonym database is offered as a target or a source, or if a named database misses its synonyms.
    """
    assert best(api.convert("OLDSYM", from_database="HGNC Symbol")) == ["Z.1"]
    assert not any("synonym_id" in d for d in api.final_databases() + api.external_databases())
    description = api.describe()
    assert "HGNC Symbol" in description.synonym_databases
    assert not any("synonym_id" in d for d in description.external_databases)
    with pytest.raises(UnknownDatabaseError, match="no target"):
        api.convert("SYMS", final_database="synonym_id::HGNC Symbol")
    with pytest.raises(UnknownDatabaseError, match="own name"):
        api.convert("OLDSYM", from_database="synonym_id::HGNC Symbol")
    with pytest.raises(InvalidArgumentError):
        api.genes_named("synonym_id::OLDSYM")


# -------------------------------------------------------------------------------------------- decisions 95, 96


def test_a_pseudoautosomal_name_answers_the_x_copy_and_lists_the_y_copy_apart(api):
    """``PARG`` names the X gene and its Y copy at 5 to 9: the X gene, the Y copy apart. Fails if they tie."""
    result = api.convert("PARG", to_release=9)
    assert best(result) == ["PX.1"] and result.alternative_locus_genes == ("PY.1",)
    assert best(api.convert("PARG", to_release=3)) == ["PX.1"]  # before the Y copy had an id of its own


def test_a_par_y_id_answers_the_gene_and_lists_its_y_copy_apart(api):
    """``ENSG00000000093_PAR_Y`` is GENCODE's Y copy of the X gene: the X gene, its Y copy (an id of its own) apart.

    Fails if the form is not read (not found), or if the Y copy that shares the gene's symbol is not listed.
    """
    result = api.convert("ENSG00000000093_PAR_Y", to_release=9)
    assert (best(result), result.match) == (["ENSG00000000093.1"], "par_y:exact")
    assert result.alternative_locus_genes == ("ENSG00000000094.1",)


def test_a_gene_of_y_is_a_copy_only_beside_the_gene_of_the_region_it_repeats(api):
    """``PARQ`` names a gene of Y's pseudoautosomal region and a gene of X outside the region it repeats: a tie.

    Fails if a gene of the region of Y is taken for a copy beside any gene of X.
    """
    result = api.convert("PARQ", to_release=9)
    assert best(result) == ["QX.1", "QY.1"] and result.alternative_locus_genes == ()


def test_a_gene_of_y_is_a_copy_only_of_the_gene_at_its_place(api):
    """``PARW`` names a gene of Y's pseudoautosomal region and a gene of the region of X it repeats, at another place.

    The one rule of a gene reads the Y gene's place on X (:py:mod:`idtrack.graph.genes`): it lies elsewhere, so it is
    no copy of that gene, and the answer is a tie. Fails if a gene of the region of Y is taken for a copy beside any
    gene of the region of X (the rule before: WY would be listed apart).
    """
    result = api.convert("PARW", to_release=9)
    assert best(result) == ["WX.1", "WY.1"] and result.alternative_locus_genes == ()
    assert api.graph.count_genes(["PX", "PY"]) == 1 and api.graph.count_genes(["WX", "WY"]) == 2


def test_an_lrg_id_answers_the_gene_its_record_describes(api):
    """``LRG_9`` answers R (whose symbol the record carries), the record listed apart, the match ``lrg:``.

    Fails if an LRG id answers itself (as before decision 96).
    """
    for typed in ("LRG_9", "LRG_9.1"):
        result = api.convert(typed)
        assert best(result) == ["R.1"]
        assert result.match == "lrg:exact" and result.graph_id == typed
        assert "LRG_9.1" in result.alternative_locus_genes


def test_an_lrg_id_keeps_the_inference_of_its_starting_release(api):
    """Without ``from_release`` the LRG id's starting release is inferred, and the result says so; given, it is not.

    Fails if the described gene's conversion, which is made from an explicit release, lends its flag to the result
    (``from_release_inferred`` False although no release was given).
    """
    inferred = api.convert("LRG_9")
    assert inferred.from_release_inferred and inferred.from_release_given is None
    given = api.convert("LRG_9", from_release=9)
    assert not given.from_release_inferred and given.from_release_given == 9


def test_every_id_of_an_lrg_record_answers_as_the_record(api):
    """I153: a transcript id of the record, and the record's gene id of the GRCh37 releases, answer as the record id.

    ``LRG_9t1`` (a transcript of LRG_9) answers R with the record listed apart, as ``LRG_9`` does; ``LRG_8_g1`` (the
    gene of the record LRG_8, on the region LRG_8, at 1 to 4) answers Q. In a batch ``LRG_9t1-1``, a transcript id of
    its own, is answered as alone, not as a copy of ``LRG_9t1``. Fails if only the record id answers the gene it
    describes (the rule before: the transcript answered LRG_9.1 itself, LRG_8_g1 itself), or if the batch makes an
    Ensembl transcript id a copy.
    """
    record = api.convert("LRG_9")
    for typed in ("LRG_9t1", "LRG_9t1.1"):
        result = api.convert(typed)
        assert best(result) == best(record) == ["R.1"], typed
        assert result.match == "lrg:exact" and "LRG_9.1" in result.alternative_locus_genes
    old = api.convert("LRG_8_g1", from_release=3, to_release=3)
    assert best(old) == ["Q.1"] and old.match == "lrg:exact" and "LRG_8_g1.1" in old.alternative_locus_genes
    alone = api.convert("LRG_9t1-1")
    batch = api.convert_many(["LRG_9t1", "LRG_9t1-1"])
    assert batch[1].match == alone.match == "lrg:exact" and best(batch[1]) == best(alone)


def test_a_target_in_a_database_is_an_id_of_the_gene_answered(api):
    """I151: converted to a database, the answer is the gene answered as Ensembl genes, with its ids there or none.

    ``TWOGENES`` is PG1 by support (two databases against one). PG1 has no HGNC symbol: the result is PG1 itself
    (``no_target``), not OTHERSYM, the symbol of PG2, which the name reaches through one database. Fails if the
    targets are ranked among the genes that have an id only (the rule before: OTHERSYM, status ``converted``), so
    that a gene of a lower rank becomes the certain answer; and the harmonizer would move the feature onto PG2.
    """
    genes = api.convert("TWOGENES")
    assert best(genes) == ["PG1.1"] and genes.decided_by == "entry_support"
    named = api.convert("TWOGENES", final_database="HGNC Symbol")
    assert (named.status, named.no_target, named.reason) == ("no_target", True, "no_id_in_final_database")
    assert [c.target for c in named.best] == ["PG1.1"] and "OTHERSYM" not in named.targets("all")
    strict = api.engine.convert("TWOGENES", final_database="HGNC Symbol", return_ensembl_alternative=False)
    assert strict.candidates == () and strict.reason == "no_id_in_final_database"
    # the gene of a lower rank keeps its own id where it is asked for itself
    assert api.convert("PG2.1", final_database="HGNC Symbol").targets() == ("OTHERSYM",)


# --------------------------------------------------------------------------------------------- decisions 79, 97


def test_a_list_is_read_at_each_release_and_answered_as_a_band():
    """Six names of early genes and eight renamed at 5 (synonyms after it), listed as at release 3.

    Read at each release, the list is found whole at 1 to 4 and nowhere else (at 5 to 9 the renamed names are only
    synonyms, which place no list): the band is 1 to 4, with no origin outside it. Fails if every name is read at the
    last release, if a synonym counts at the releases where it is one (the margin would be six), or if one release is
    presented as certain.
    """
    w = _Writer()
    for i in range(6):
        w.gene(f"V{i}", [(1, 1, 4)]).xref(f"EARLY{i}", f"V{i}.1", "HGNC Symbol", EARLY)
    for i in range(8):
        w.gene(f"N{i}", [(1, 1, 9)]).xref(f"RENAMED{i}", f"N{i}.1", "HGNC Symbol", EARLY)
        w.xref(f"synonym_id::RENAMED{i}", f"N{i}.1", "synonym_id::HGNC Symbol", LATE)
    api = API(w.graph)
    names = [f"EARLY{i}" for i in range(6)] + [f"RENAMED{i}" for i in range(8)]
    inference = api.source_inference(names, mode="release")
    assert isinstance(inference, SourceInference)
    assert sorted(inference.band) == [1, 2, 3, 4] and inference.margin is None
    assert dict(inference.counts) == {1: 14, 2: 14, 3: 14, 4: 14}
    assert inference.identifiers == 14 and inference.not_found == 0
    assert sorted(api.infer_source(names, mode="release")) == [1, 2, 3, 4]
    # decision 97: each distinct identifier counts once; repeats and spellings of one identifier change nothing
    again = api.source_inference(names + names[:7] + [" EARLY0", "'EARLY1'"], mode="release")
    assert again.band == inference.band and again.counts == inference.counts
    assert SourceInference.from_dict(inference.to_dict()) == inference


def test_a_band_holds_the_releases_a_small_difference_cannot_tell_apart():
    """Three names found at 1 to 4 only, among many found throughout: the band is every release.

    Fails if the best count alone is the answer (a difference of three identifiers is no evidence at the 5 % level).
    """
    w = _Writer()
    for i in range(3):
        w.gene(f"V{i}", [(1, 1, 4)]).xref(f"EARLY{i}", f"V{i}.1", "HGNC Symbol", EARLY)
    for i in range(20):
        w.gene(f"G{i}", [(1, 1, 9)]).xref(f"ALWAYS{i}", f"G{i}.1", "HGNC Symbol", ALL)
    api = API(w.graph)
    inference = api.source_inference(
        [f"EARLY{i}" for i in range(3)] + [f"ALWAYS{i}" for i in range(20)], mode="release"
    )
    assert sorted(inference.band) == list(ALL) and inference.margin is None
