"""Where a string is read and what a batch answers, on small graphs written by hand.

One release decides both which node a string means and where a conversion reads it; readings of one step are joined
only where they exist at one release and are ranked by the evidence of their lineage; an Ensembl version is written
after ``.`` only; an answer read at another release than the one given says so; a batch answers every item as a
single conversion does, but for the copies of decision 82; a missing value as a table holds it is one item's missing
value. Each test names the plausible wrong version it fails under. The graphs are written with the writer of
:py:mod:`tests.unit.test_explained_path` (releases 1 to 9; assembly 37 serves 1 to 4, 38 serves 5 to 9).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from idtrack import API
from idtrack.errors import ArgumentTypeError
from idtrack.results import REASONS, ConversionResult
from tests.unit.test_explained_path import _Writer

ALL = range(1, 10)


def best(result: ConversionResult) -> list[str]:
    """The genes of the best answer."""
    return sorted({c.gene for c in result.best})


@pytest.fixture(scope="module")
def api() -> API:
    """The genes of these tests in one graph (human)."""
    w = _Writer()
    # a symbol of A at 3 to 5 that is a synonym of B at every release (NCBI's RNR1 of MT-RNR1 and NR4A2's synonym)
    w.gene("A", [(1, 1, 9)]).gene("B", [(1, 1, 9)])
    w.xref("SWAP", "A.1", "HGNC Symbol", range(3, 6)).xref("synonym_id::SWAP", "B.1", "synonym_id::HGNC Symbol", ALL)
    # two names equal up to separators that never exist at one release (mouse AY172335-1 at 54-56, AY172335.1 at 57-60)
    w.gene("C", [(1, 1, 9)]).gene("D", [(1, 1, 9)])
    w.xref("CL-1", "C.1", "Clone_based_ensembl_gene", [2, 3]).xref("CL.1", "D.1", "Clone_based_ensembl_gene", [5, 6])
    # two names equal up to separators at one release: E's answer needs no bridge, F's one (F ends at 4; G took the
    # NCBI id that F carried, both on assembly 37 at one place, so a bridge leads from F to G)
    w.gene("E", [(1, 1, 9)]).gene("F", [(1, 1, 4)]).gene("G", [(1, 3, 9)])
    w.xref("TW-1", "E.1", "Clone_based_ensembl_gene", ALL).xref("TW.1", "F.1", "Clone_based_ensembl_gene", [1, 2, 3])
    w.xref("77", "F.1", "EntrezGene", [1, 2]).xref("77", "G.1", "EntrezGene", range(3, 10))
    # Ensembl ids with versions, and a gene id with versions 1 and 2
    w.gene("ENSG00000000101", [(1, 1, 4), (2, 5, 9)])
    # a name of H at 1 to 3 only (a name the graph holds at old releases), and H's later name
    w.gene("H", [(1, 1, 9)]).xref("OLDNAME", "H.1", "HGNC Symbol", [1, 2, 3]).xref("NEWNAME", "H.1", "HGNC Symbol", ALL)
    # decision 82 in a batch: a name, and a name of another gene that extends it with `-10`, at 2 to 3 only
    w.gene("J", [(1, 1, 9)]).gene("K", [(1, 1, 9)])
    w.xref("SL", "J.1", "HGNC Symbol", ALL).xref("SL-10", "K.1", "HGNC Symbol", [2, 3])
    return API(w.graph)


# ------------------------------------------------------------------------------------- one release decides (F2)


def test_without_from_release_the_string_is_read_at_the_target_release(api):
    """Without ``from_release`` the string is read at the target release, as with it.

    ``SWAP`` is A's symbol at 3 to 5 and B's synonym at every release: read at 4 it is A's symbol, given as
    ``from_release`` or as the target release alone; read at 9 (no symbol there) it is B's synonym.

    Fails if the node is chosen by existence at the last release of the graph and read at the target release (the
    synonym taken at 4), which answers one question two ways.
    """
    for kwargs in ({"from_release": 4}, {"to_release": 4}, {"from_release": 4, "to_release": 4}):
        result = api.convert("SWAP", **kwargs)
        assert best(result) == ["A.1"] and result.match == "exact" and result.from_release == 4, kwargs
    result = api.convert("SWAP", to_release=9)
    assert best(result) == ["B.1"] and result.match == "synonym:exact" and result.from_release == 9


def test_resolve_name_and_readings_ask_at_the_target_release(api):
    """The graph's own readings follow the same rule. Fails if ``to_release`` only steers the spelling."""
    assert api.graph.resolve_name("SWAP", to_release=4) == ("SWAP", "exact")
    assert api.graph.resolve_name("SWAP") == ("synonym_id::SWAP", "synonym:exact")
    found = api.graph.readings("SWAP", to_release=4)
    assert found.taken == (("SWAP", "exact"),) and found.here


# -------------------------------------------------------------------------------- readings of one release (F8)


def test_readings_that_never_exist_at_one_release_are_not_joined(api):
    """``CL_1`` reaches ``CL-1`` (2 to 3) and ``CL.1`` (5 to 6) by separators; neither exists at 9.

    The string is read at the latest release of a reading (6) as ``CL.1``, alone, ``CL-1`` reported as its reading of
    other releases; asked at 2 it is ``CL-1``. Fails if the separator step's matches are joined into a tie although
    they never exist at one release (C and D).
    """
    result = api.convert("CL_1")
    assert best(result) == ["D.1"] and result.graph_id == "CL.1" and result.from_release == 6 and not result.tie
    assert [(o.graph_id, o.reason, o.genes, o.releases) for o in result.other_readings] == [
        ("CL-1", "other_release", ("C.1",), (2, 3))
    ]
    found = api.graph.readings("CL_1")
    assert found.taken == (("CL.1", "separator"),) and not found.here
    early = api.convert("CL_1", from_release=2, to_release=9)
    assert best(early) == ["C.1"] and early.graph_id == "CL-1" and not early.tie


def test_joined_readings_are_ranked_by_the_evidence_of_their_lineage(api):
    """At 2 ``TW_1`` reads ``TW-1`` (E, history alone) and ``TW.1`` (F, which reaches G at 9 only by a bridge).

    E is the best answer, G a candidate after it, decided by the bridges. Fails if every reading's best answer is
    marked best (the tie E, G of the rule before), which ignores the evidence of each reading.
    """
    result = api.convert("TW_1", from_release=2, to_release=9)
    assert best(result) == ["E.1"] and not result.tie and result.decided_by == "bridges"
    assert [c.gene for c in result.candidates] == ["E.1", "G.1"]
    assert [(o.graph_id, o.reason, o.genes) for o in result.other_readings] == [("TW.1", "same_step", ("G.1",))]


# ----------------------------------------------------------------------------- an Ensembl version is `.<v>` (F9)


@pytest.mark.parametrize("typed", ["ENSG00000000101-1", "ENSG00000000101_2", "ENSG00000000101-9", "ensg00000000101_1"])
def test_a_dash_or_underscore_suffix_is_no_version(api, typed):
    """``-<digits>`` and ``_<digits>`` after an Ensembl id stay not found (never another version's answer).

    Fails if the separator step joins the suffix into the version (``-1`` read as version 1, ``-9`` as a version
    newer than the graph).
    """
    result = api.convert(typed)
    assert result.status == "not_found" and result.candidates == () and result.known_versions == ()


def test_a_version_after_a_dot_is_still_read(api):
    """``.1`` and ``.2`` are versions, ``.9`` a version newer than the graph. Fails if the rule above cuts ``.``."""
    assert api.convert("ENSG00000000101.1").graph_id == "ENSG00000000101.1"
    newer = api.convert("ENSG00000000101.9")
    assert newer.match == "version_newer_than_graph" and newer.graph_id == "ENSG00000000101.2"


# --------------------------------------------------------------------------- read at another release (F10)


def test_an_answer_read_at_another_release_says_so_and_is_binned_apart(api):
    """``OLDNAME`` is a name at 1 to 3; asked at 7, it is read at its latest release (3) and answered from there.

    The status stays ``converted``, the reason is ``read_at_another_release``, and :py:meth:`API.classify` puts the
    result in a bin of its own. Fails if the fallback is answered as an ordinary conversion (reason ``None``, binned
    with the matches), which a caller that checks the status cannot tell from an answer to its question.
    """
    result = api.convert("OLDNAME", from_release=7, to_release=9)
    assert result.status == "converted" and result.from_release_fallback and result.from_release == 3
    assert result.reason == "read_at_another_release" and API.reason_text(result) == REASONS[result.reason]
    plain = api.convert("NEWNAME", from_release=7, to_release=9)
    assert plain.reason is None and not plain.from_release_fallback
    bins = API.classify([result, plain])
    assert bins["read_at_another_release"] == [result] and bins["matching_1_to_1"] == [plain]
    assert sum(result in members for name, members in bins.items() if not name.startswith("changed")) == 2
    assert "Read at another release: 1" in API.classification_summary(bins)


# ------------------------------------------------------------------------------- batch against single (H2)


def test_every_item_of_a_batch_is_answered_as_alone_but_for_the_copies(api):
    """A batch of names, some of them at other releases, and a copy made unique by a single-cell tool.

    ``SL-10`` is a name of K at 2 to 3 only: asked at 7 its single conversion reads it at 3 as K's name, so beside
    ``SL`` it is no copy; ``SL-3`` is no name anywhere, so beside ``SL`` it is SL's copy (decision 82). Every other item
    is answered as :py:meth:`API.convert` answers it. Fails if the batch asks for an exact name only at the release
    given (``SL-10`` would be SL's copy, J, while alone it is K).
    """
    items = ["SL", "SL-10", "SL-3", "OLDNAME", "SWAP", "CL_1", "TW_1", "ENSG00000000101-1"]
    for kwargs in ({"from_release": 7, "to_release": 9}, {"to_release": 4}, {}, {"from_release": 2}):
        batch = api.convert_many(items, **kwargs)
        for item, result in zip(items, batch, strict=True):
            if item == "SL-3":
                assert result.match == "copy:exact" and best(result) == ["J.1"], kwargs
                continue
            assert result == api.convert(item, **kwargs), (item, kwargs)
    assert best(api.convert_many(["SL", "SL-10"], from_release=7)[1]) == ["K.1"]


# --------------------------------------------------------------------------------- missing values (F13, F12)


@pytest.mark.parametrize(
    "missing", [None, math.nan, np.nan, np.float32("nan"), pd.NA, pd.NaT, np.datetime64("NaT", "ns")]
)
def test_a_missing_value_as_a_table_holds_it_is_one_items_missing_value(api, missing):
    """A cell without a value (``None``, NaN, ``pd.NA``, ``NaT``) is that item's ``MissingValueError``.

    The rest of the batch is converted. Fails if such an item fails the whole batch as a non-string (the rule before).
    """
    results = api.convert_many(["NEWNAME", missing, "NEWNAME"])
    assert best(results[0]) == ["H.1"] and results[2] == results[0]
    assert results[1].status == "error" and results[1].error is not None
    assert results[1].error.type == "MissingValueError" and results[1].query_id == str(missing)
    results[1].to_dict()  # strict JSON: no NaN written


def test_a_missing_value_in_a_series_and_in_source_inference(api):
    """A pandas column with an empty cell converts, and a list with one is placed. Fails as above."""
    results = api.convert_many(pd.Series(["NEWNAME", np.nan]))
    assert best(results[0]) == ["H.1"] and results[1].error.type == "MissingValueError"
    inference = api.source_inference(["NEWNAME", None, "OLDNAME"], mode="release")
    assert inference.not_found == 1 and inference.identifiers == 2


@pytest.mark.parametrize("other", [3, 1.5, b"NEWNAME", object()])
def test_other_items_that_are_not_strings_still_fail_the_batch(api, other):
    """A number, bytes or an object is no missing value: the batch is refused, the position named.

    Fails if every item that is not a string is read as a missing value.
    """
    with pytest.raises(ArgumentTypeError, match=r"positions \[1\]"):
        api.convert_many(["NEWNAME", other])


# ----------------------------------------------------------------- a meaning at another time is reported (99)


def test_a_reading_that_exists_at_other_releases_only_is_reported_with_them():
    """A reading of other releases only is reported with its releases, never answered in place of today's.

    ``S`` is A's name at 1 to 3 and B's synonym at 4 to 9: without a release it is B's synonym, A's name reported
    with its releases; with ``from_release=2`` it is A.

    Fails if the reading of another time is chosen over the one of the release asked about, or is not reported (a
    caller could not see that the string named another gene before).
    """
    w = _Writer()
    w.gene("A", [(1, 1, 9)]).gene("B", [(1, 1, 9)])
    w.xref("S", "A.1", "HGNC Symbol", [1, 2, 3]).xref("synonym_id::S", "B.1", "synonym_id::HGNC Symbol", range(4, 10))
    api = API(w.graph)
    now = api.convert("S")
    assert best(now) == ["B.1"] and now.match == "synonym:exact" and now.from_release == 9
    assert [(o.graph_id, o.match, o.reason, o.genes, o.releases) for o in now.other_readings] == [
        ("S", "exact", "other_release", ("A.1",), (1, 2, 3))
    ]
    then = api.convert("S", from_release=2)
    assert best(then) == ["A.1"] and then.match == "exact" and then.other_readings == ()
    assert api.graph.readings("S").other_releases == (("S", "exact", (1, 2, 3)),)


def test_a_string_that_leads_to_no_gene_reports_the_release_it_was_read_at():
    """``GONE`` is a name of R at 2 to 3; R retires at 3 and nothing descends from it.

    Without a release, and with a release at which the graph does not hold it, the string is read at its latest
    release (3): the answer has no target, and the result reports where it was read, inferred. Fails if a result
    without a target drops the release it was read at (``from_release`` ``None``), which hides from a caller that the
    string was read at all, and where.
    """
    w = _Writer()
    w.gene("R", [(1, 1, 3)]).xref("GONE", "R.1", "HGNC Symbol", [2, 3])
    api = API(w.graph)
    for given in (None, 7):
        result = api.convert("GONE", from_release=given, to_release=9, allow_bridges=False)
        assert (result.status, result.reason, result.candidates) == ("no_conversion", "no_path_to_target", ())
        assert (result.from_release, result.from_release_inferred) == (3, True)
        assert result.from_release_given == given and result.from_release_fallback is (given is not None)
    asked = api.convert("GONE", from_release=2, to_release=9, allow_bridges=False)
    assert (asked.status, asked.from_release, asked.from_release_inferred) == ("no_conversion", 2, False)


def test_a_tie_is_given_in_ascending_order_of_gene_id():
    """``TIE`` is a name of Q itself and of N through its transcript: one database each, equal in the evidence.

    The two genes tie; N, reached in more steps, comes first, by its id. Fails if the steps that did not decide
    order the tie (Q first), an order a caller cannot state.
    """
    w = _Writer()
    w.gene("Q", [(1, 1, 9)]).gene("N", [(1, 1, 9)]).transcript("TN.1", "N.1", ALL)
    w.xref("TIE", "Q.1", "HGNC Symbol", ALL).xref("TIE", "TN.1", "HGNC Symbol", ALL)
    result = API(w.graph).convert("TIE")
    assert result.tie and result.status == "tie"
    assert list(result.targets("best")) == ["N.1", "Q.1"] == [c.gene for c in result.best]
    assert result.best[0].external_steps > result.best[1].external_steps  # the steps did not decide the order
