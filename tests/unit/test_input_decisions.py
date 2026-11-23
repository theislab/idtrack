"""What a conversion makes of what users type (decisions 52, 55, 56, 80 to 85, 87, 93), on small graphs written by hand.

Each test names the plausible wrong version it fails under. The graphs are written with the writer of
:py:mod:`tests.unit.test_explained_path` (releases 1 to 9; assembly 37 serves 1 to 4, 38 serves 5 to 9).
"""

from __future__ import annotations

import pytest

from idtrack import API, organisms
from idtrack.errors import InvalidArgumentError, MissingValueError
from idtrack.results import MATCH_PREFIXES, MATCH_STEPS, READING_REASONS, ConversionResult
from tests.unit.test_explained_path import _Writer

ALL = range(1, 10)


def graph(organism: str = "homo_sapiens") -> _Writer:
    """The genes of these tests, as a writer (the organism set on the graph)."""
    w = _Writer()
    w.graph.graph["organism"] = organism
    symbol = organisms.SYMBOL_DATABASES[organism][0]
    # decisions 55 and 80: a symbol of one gene, and the same string in capitals an exact synonym of another
    w.gene("G1", [(1, 1, 9)]).xref("Ar", "G1.1", symbol, ALL)
    w.gene("G2", [(1, 1, 9)]).xref("synonym_id::AR", "G2.1", "synonym_id::EntrezGene", ALL)
    # ... and a symbol that two genes carry (no single gene): the rule does not apply
    w.gene("G4", [(1, 1, 9)]).xref("Cck", "G4.1", symbol, ALL)
    w.gene("G5", [(1, 1, 9)]).xref("Cck", "G5.1", symbol, ALL)
    w.gene("G6", [(1, 1, 9)]).xref("synonym_id::CCK", "G6.1", "synonym_id::EntrezGene", ALL)
    # decision 82: a name, and that name with a make-unique suffix (another separator) a synonym of another gene
    w.gene("G7", [(1, 1, 9)]).xref("SRF", "G7.1", "HGNC Symbol", ALL)
    w.gene("G8", [(1, 1, 9)]).xref("synonym_id::SRF-1", "G8.1", "synonym_id::HGNC Symbol", ALL)
    # ... a numbered name of its own beside the name it extends, and a numbered name whose stem is no name
    w.gene("G9", [(1, 1, 9)]).xref("IGHV3-30-2", "G9.1", "HGNC Symbol", ALL)
    w.gene("G10", [(1, 1, 9)]).xref("IGHV3-30", "G10.1", "HGNC Symbol", ALL)
    w.gene("G11", [(1, 1, 9)]).xref("MIR486-2", "G11.1", "HGNC Symbol", ALL)
    # decision 87: two names equal up to separators, of two genes
    w.gene("G12", [(1, 1, 9)]).xref("AY1-1", "G12.1", "Clone_based_ensembl_gene", ALL)
    w.gene("G13", [(1, 1, 9)]).xref("AY1.1", "G13.1", "Clone_based_ensembl_gene", ALL)
    # decision 87 (ids): a transcript without a gene id node of its own
    w.gene("G14", [(1, 1, 9)]).transcript("ENST00000000014.3", "G14.1", ALL)
    # decision 52: a real synonym spelt like a missing value
    w.gene("G15", [(1, 1, 9)]).xref("synonym_id::NA", "G15.1", "synonym_id::HGNC Symbol", ALL)
    # decision 82: a symbol written in capitals, and a synonym of another gene that extends it in small letters
    w.gene("G16", [(1, 1, 9)]).xref("GAL", "G16.1", "HGNC Symbol", ALL)
    w.gene("G17", [(1, 1, 9)]).xref("synonym_id::gal-2", "G17.1", "synonym_id::EntrezGene", ALL)
    # decision 56: a name of one database and a synonym of another, spelt alike
    w.gene("G18", [(1, 1, 9)]).xref("DUAL", "G18.1", "UniProtKB Gene Name", ALL)
    w.gene("G19", [(1, 1, 9)]).xref("synonym_id::DUAL", "G19.1", "synonym_id::HGNC Symbol", ALL)
    # decision 82: a name, and that name with a make-unique suffix a synonym of another gene at old releases only
    w.gene("G20", [(1, 1, 9)]).xref("TEF", "G20.1", "HGNC Symbol", ALL)
    w.gene("G21", [(1, 1, 9)]).xref("synonym_id::TEF-1", "G21.1", "synonym_id::HGNC Symbol", range(1, 5))
    return w


@pytest.fixture(scope="module")
def human() -> API:
    """An API over the graph as human's."""
    return API(graph().graph)


@pytest.fixture(scope="module")
def mouse() -> API:
    """An API over the graph as mouse's (MGI symbols, written ``Trp53``)."""
    return API(graph("mus_musculus").graph)


def best(result: ConversionResult) -> list[str]:
    """The genes of the best answer."""
    return sorted({c.gene for c in result.best})


# --------------------------------------------------------------------------------------- decisions 55 and 80


@pytest.mark.parametrize("typed", ["AR", "ar"])
def test_mouse_reads_a_one_case_string_as_its_symbol_first(mouse, typed):
    """Mouse: ``AR`` and ``ar`` are the MGI symbol ``Ar`` (G1), the exact synonym of G2 reported beside it.

    Fails if the exact synonym comes first (decisions 28/38 alone), or if the synonym reading is not reported.
    """
    result = mouse.convert(typed)
    assert best(result) == ["G1.1"]
    assert (result.graph_id, result.match) == ("Ar", "case")
    if typed == "AR":
        (other,) = result.other_readings
        assert (other.match, other.reason, other.genes) == ("synonym:exact", "symbol_without_case", ("G2.1",))
        assert other.graph_id.endswith("AR")


def test_the_symbol_rule_needs_exactly_one_gene(mouse):
    """``CCK``: the symbol ``Cck`` names two genes, so the exact synonym of G6 stands. Fails without the count."""
    result = mouse.convert("CCK")
    assert best(result) == ["G6.1"]
    assert result.other_readings == ()


def test_the_symbol_rule_is_mouse_s(human):
    """Human has no rule of letter case: ``AR`` is the exact synonym (G2). Fails if the rule holds everywhere."""
    assert best(human.convert("AR")) == ["G2.1"]


def test_a_named_database_that_is_no_symbol_database_turns_the_rule_off(mouse):
    """``from_database`` names a database: only its ids count. Fails if the symbol rule ignores the database named."""
    result = mouse.convert("AR", from_database="EntrezGene")  # the synonyms EntrezGene records (decision 56)
    assert best(result) == ["G2.1"]


def test_a_string_in_the_convention_is_read_by_the_steps(mouse):
    """``Ar`` is exactly the symbol; no other reading. Fails if the rule fires on an exact name."""
    result = mouse.convert("Ar")
    assert (best(result), result.match, result.other_readings) == (["G1.1"], "exact", ())


def test_the_convention_of_letter_case_is_read_from_the_symbols(mouse, human):
    """Mouse's symbol database writes ``Ar`` and ``Cck``: mixed; human's eight symbols have two with a small letter.

    Fails if the convention is not the majority's (any small letter would make the human graph mixed).
    """
    assert mouse.engine.graph.symbol_case() == "mixed"
    assert human.engine.graph.symbol_case() == "capitals"


# --------------------------------------------------------------------------------------------- decision 82


@pytest.mark.parametrize("typed", ["SRF.1", "SRF_1"])
def test_a_make_unique_suffix_is_never_joined_into_another_gene_s_name(human, typed):
    """``SRF.1``: the separator reading ``SRF-1`` (G8's synonym) is set aside, reported, and nothing is answered.

    Fails if the separator step takes it (the wrong answer G8), or if the reading is dropped without a word.
    """
    result = human.convert(typed)
    assert result.status == "not_found" and best(result) == []
    (other,) = result.other_readings
    assert (other.match, other.reason, other.genes) == ("synonym:separator", "make_unique_suffix", ("G8.1",))


def test_the_suffixed_name_itself_is_read(human):
    """``SRF-1`` typed exactly is G8's synonym. Fails if the guard reaches the exact step."""
    assert best(human.convert("SRF-1")) == ["G8.1"]


def test_a_reading_taken_is_never_reported_as_set_aside(human):
    """``TEF-1`` is G21's synonym at releases 1 to 4 only: read exactly there, it is the answer, and nothing else.

    The steps of letter case and separators reach the same synonym at release 9, where the guard sets it aside. Fails
    if the readings set aside are not checked against the reading taken (the answer listed twice as set aside).
    """
    result = human.convert("TEF-1")
    assert (best(result), result.match, result.other_readings) == (["G21.1"], "synonym:exact", ())


def test_a_numbered_name_whose_stem_is_no_name_is_read_by_separators(human):
    """``MIR486.2``: ``MIR486`` names nothing, so ``MIR486-2`` (G11) is its separator reading.

    Fails if every suffix is refused.
    """
    result = human.convert("MIR486.2")
    assert (best(result), result.match) == (["G11.1"], "separator")


def test_the_guard_needs_a_symbol_written_exactly_before_the_suffix(human):
    """``gal.2``: ``gal`` is the symbol ``GAL`` only by letter case, so ``gal-2`` (G17's synonym) is its reading.

    Fails if the guard fires whatever reading finds the stem (a synonym typed with other separators is lost).
    """
    assert best(human.convert("gal.2")) == ["G17.1"]


def test_a_named_database_reads_the_synonyms_it_records(human):
    """``DUAL`` is UniProt's name of G18 and HGNC's synonym of G19: named as HGNC's, it is G19's synonym.

    Fails if the synonyms are not counted as ids of the database named (the name of G18 would be taken, and then
    nothing found through HGNC).
    """
    result = human.convert("DUAL", from_database="HGNC Symbol")
    assert (best(result), result.match, result.from_databases) == (["G19.1"], "synonym:exact", ("HGNC Symbol",))
    assert best(human.convert("DUAL")) == ["G18.1"]


def test_a_batch_reports_a_copy_of_another_item(human):
    """In a batch, ``SRF.1`` beside ``SRF`` is SRF's copy (G7, ``copy:exact``); ``IGHV3-30-2`` is a name of its own.

    Fails if a copy is read on its own (not found, or another gene), or if every numbered name becomes a copy.
    """
    results = human.convert_many(["SRF", "SRF.1", "IGHV3-30", "IGHV3-30-2", "SRF.1"])
    assert [best(r) for r in results] == [["G7.1"], ["G7.1"], ["G10.1"], ["G9.1"], ["G7.1"]]
    assert [r.match for r in results] == ["exact", "copy:exact", "exact", "exact", "copy:exact"]
    assert results[1].query_id == "SRF.1"
    alone = human.convert_many(["SRF.1"])[0]
    assert alone.status == "not_found"
    # an Ensembl id with a version beside the id is its version, never a copy
    versions = human.convert_many(["ENST00000000014", "ENST00000000014.3", "ENST00000000014.9"])
    assert [r.match for r in versions] == ["exact", "exact", "version_newer_than_graph"]


# ------------------------------------------------------------------------------------------ decision 87


def test_a_separator_reading_of_several_names_is_a_tie(human):
    """``AY1_1`` is ``AY1-1`` (G12) and ``AY1.1`` (G13) alike: a tie of both, the second reading named.

    Fails if one name is taken silently (the first in sort order), as before decision 87.
    """
    result = human.convert("AY1_1", explain=True)
    assert result.status == "tie" and result.tie
    assert best(result) == ["G12.1", "G13.1"]
    (other,) = result.other_readings
    assert (other.graph_id, other.reason, other.genes) == ("AY1.1", "same_step", ("G13.1",))
    steps = result.path.steps
    for candidate in result.candidates:
        assert steps[candidate.path[-1]].target == candidate.gene


def test_a_tie_of_readings_keeps_the_bridges_of_every_reading():
    """``BY1_1`` is ``BY1-1`` (H1) and ``BY1.1`` (H2); both genes end at 4, each id left its gene for a later one.

    The joined answer is a tie of H3 and H4, each reached by a bridge of its own reading; the explained path lists both
    bridges. Fails if the tie keeps only the first reading's explanation of bridges (H2 to H4 would be missing).
    """
    w = _Writer()
    w.graph.graph["organism"] = "homo_sapiens"
    w.gene("H1", [(1, 1, 4)]).xref("BY1-1", "H1.1", "Clone_based_ensembl_gene", range(1, 5))
    w.gene("H2", [(1, 1, 4)]).xref("BY1.1", "H2.1", "Clone_based_ensembl_gene", range(1, 5))
    w.gene("H3", [(1, 5, 9)]).gene("H4", [(1, 5, 9)])
    w.xref("X1", "H1.1", "EntrezGene", range(1, 5)).xref("X1", "H3.1", "EntrezGene", range(5, 10))
    w.xref("X2", "H2.1", "EntrezGene", range(1, 5)).xref("X2", "H4.1", "EntrezGene", range(5, 10))
    result = API(w.graph).convert("BY1_1", from_release=2, to_release=9, explain=True)
    assert result.tie and best(result) == ["H3.1", "H4.1"]
    assert [(c.source, c.target, c.taken) for c in result.path.bridges] == [
        ("H1.1", "H3.1", True),
        ("H2.1", "H4.1", True),
    ]


@pytest.mark.parametrize(("typed", "match"), [("ENST00000000014", "exact"), ("enst00000000014", "case")])
def test_a_transcript_id_without_version_is_found_in_any_case(human, typed, match):
    """Ensembl writes its ids in capitals: an unversioned transcript id in lower case is found by case.

    Fails if only the exact prefix is searched (not found, IN-5).
    """
    result = human.convert(typed)
    assert (best(result), result.match) == (["G14.1"], match)


# ------------------------------------------------------------------------------------ decisions 52, 56, 81 to 85, 93


@pytest.mark.parametrize("typed", ["NA", "nan", "N/A", "unknown", "NaT", "None.", "  null "])
def test_a_missing_value_is_refused_by_a_single_call(human, typed):
    """Refused before any lookup (``NA`` is G15's synonym all the same). Fails if it is looked up."""
    with pytest.raises(MissingValueError):
        human.convert(typed)


def test_a_missing_value_is_looked_up_in_a_named_database(human):
    """Decision 52: with its source database named, ``NA`` is G15's synonym. Fails without the exemption."""
    assert best(human.convert("NA", from_database="HGNC Symbol")) == ["G15.1"]


def test_a_batch_answers_every_item_and_refuses_the_missing_ones_alone(human):
    """Decision 81 with 46 (g): every refused item is that item's error, the other items are answered.

    A missing value, an empty string, an internal name and a string longer than any identifier are refused. Fails if
    the batch stops or if they convert.
    """
    results = human.convert_many(["SRF", "nan", "", "synonym_id::SRF-1", "A" * 5000, "SRF"])
    assert [r.status for r in results] == ["converted", "error", "error", "error", "error", "converted"]
    assert [r.error.type for r in results if r.error] == [
        "MissingValueError",
        "MissingValueError",
        "InvalidArgumentError",
        "InvalidArgumentError",
    ]
    assert all(r.error.input_error for r in results if r.error)


def test_internal_names_are_refused(human):
    """Decision 56: the synonym prefix and an Ensembl id's birth mark are no identifiers."""
    with pytest.raises(InvalidArgumentError):
        human.convert("synonym_id::SRF-1")


@pytest.mark.parametrize(
    ("typed", "match"),
    [
        ("﻿SRF", "unicode:exact"),
        ("SRF​", "unicode:exact"),
        ("ＳＲＦ", "unicode:exact"),
        ("'SRF'", "punctuation:exact"),
        ("SRF;", "punctuation:exact"),
        ("“SRF”", "punctuation:exact"),
        ("​'srf';", "unicode:punctuation:case"),
    ],
)
def test_what_a_reading_changed_is_named_in_the_match(human, typed, match):
    """Decisions 83 and 84: found, and the match says the string was read. Fails if a reading claims ``exact``."""
    result = human.convert(typed)
    assert (best(result), result.match) == (["G7.1"], match)
    assert result.query_id == typed


def test_every_match_kind_is_documented():
    """The vocabulary of the match: steps, prefixes, the reasons of other readings."""
    assert {"exact", "case", "separator", "version", "version_newer_than_graph", "version_not_judged"} == set(
        MATCH_STEPS
    )
    assert {"copy", "unicode", "punctuation", "par_y", "lrg", "synonym"} == set(MATCH_PREFIXES)
    assert {"symbol_without_case", "make_unique_suffix", "same_step", "other_release", "unversioned_database"} == set(
        READING_REASONS
    )
