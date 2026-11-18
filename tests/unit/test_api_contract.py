"""The contract of :py:class:`idtrack.API` as the backend of a service: typed results, validation, no hidden state."""

import json
import logging
import subprocess  # noqa: S404
import sys

import pytest

from idtrack import errors
from idtrack.errors import InactiveAtReleaseError, SearchBudgetExceededError, UnknownVersionError
from idtrack.results import ConversionResult
from tests.unit._fake_engine import fake_api

ANSWERS = {
    # a symbol whose candidate genes mix a gene with an HGNC symbol and one without
    "MIXED": {"ENSG01.2": {"HGNC Symbol": ["SYM1"]}, "ENSG02.1": {}},
    # a gene split into two, both scored equally
    "ENSG09.1": {"ENSG09.2": {"HGNC Symbol": ["NINE"]}, "ENSG10.1": {"HGNC Symbol": ["TEN"]}},
    "QSER1": {"ENSG03.4": {"HGNC Symbol": ["QSER1"]}},
    "ENSG03": {"ENSG03.4": {"HGNC Symbol": ["QSER1"]}},
    "NOGENE": {},
    "ENSG01.2": {"ENSG01.2": {"HGNC Symbol": ["SYM1"]}},
    "ENSG02.1": {"ENSG02.1": {}},
}


def pairs(result: ConversionResult, strategy: str = "best") -> list[tuple[str, str]]:
    """The (gene, target) pairs of a view, in rank order."""
    return [(c.gene, c.target) for c in result.view(strategy)]


@pytest.fixture
def api(tmp_path):
    """An API on the fake engine with ANSWERS."""
    return fake_api(tmp_path, ANSWERS)


def test_candidates_with_mixed_fallbacks_are_the_genuine_targets(api):
    """api_harmonize-1: one candidate without an id in the database does not raise; it is listed apart."""
    r = api.convert("MIXED", final_database="HGNC Symbol")
    assert r.targets("all") == ("SYM1",)
    assert pairs(r, "all") == [("ENSG01.2", "SYM1")]
    assert r.final_database == "HGNC Symbol" and r.no_target is False and r.status == "converted"
    assert r.no_target_genes == ("ENSG02.1",)
    only_fallback = api.convert("ENSG02.1", final_database="HGNC Symbol")
    assert only_fallback.no_target is True and only_fallback.final_database == "ensembl_gene"
    assert only_fallback.targets("all") == ("ENSG02.1",) and only_fallback.status == "no_target"


def test_tied_best_targets_are_flagged_and_keep_the_order_of_the_engine(tmp_path):
    """api_harmonize-2, -3: several tied targets are an explicit tuple in the engine's order, never a set's."""
    a = fake_api(tmp_path, ANSWERS).convert("ENSG09.1", final_database="HGNC Symbol")
    assert a.tie is True and a.status == "tie" and a.targets() == ("NINE", "TEN")
    assert pairs(a) == [("ENSG09.2", "NINE"), ("ENSG10.1", "TEN")]
    assert [t for _, t in pairs(a)] == list(a.targets())
    # the engine's ranking (best first) is the order of the result, whatever the ids are
    ranking = {"ENSG09.1": {"ENSG10.1": {"HGNC Symbol": ["TEN"]}, "ENSG09.2": {"HGNC Symbol": ["NINE"]}}}
    ranked = fake_api(tmp_path, ranking).convert("ENSG09.1", final_database="HGNC Symbol")
    assert pairs(ranked, "all") == [("ENSG10.1", "TEN"), ("ENSG09.2", "NINE")] and ranked.targets("all") == (
        "TEN",
        "NINE",
    )
    # several targets ("ambiguous" in d4c2dda); a tie is a property of the best view of the one search (decision 46 d)
    assert len(ranked.targets("all")) > 1 and ranked.tie is (len(ranked.view("best")) > 1)
    one = fake_api(tmp_path, ANSWERS).convert("QSER1")
    assert len(one.targets()) == 1 and one.tie is False and one.status == "converted"


def test_result_says_how_the_identifier_was_read(api):
    """The match, the node, the starting release and the databases of the query are part of the result."""
    exact = api.convert("QSER1", final_database="HGNC Symbol")
    assert (exact.graph_id, exact.match) == ("QSER1", "exact")
    assert exact.from_release == 101 and exact.from_release_inferred is True and exact.from_databases == ()
    by_case = api.convert("qser1", from_release=100, from_database="HGNC Symbol")
    assert (by_case.graph_id, by_case.match) == ("QSER1", "case")
    assert by_case.from_release == 100 and by_case.from_release_inferred is False
    assert by_case.from_databases == ("HGNC Symbol",)
    unknown = api.convert("zzz")
    assert (unknown.graph_id, unknown.match, unknown.from_release) == (None, None, None)
    assert unknown.status == "not_found" and unknown.reason == "not_in_graph" and unknown.from_release_inferred is True
    assert api.reason_text(unknown) and api.reason_text(exact) is None
    # an id without version that has no node of its own is read as its version
    assert api.convert("ENSG01").status != "not_found"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"to_release": 999}, errors.UnknownReleaseError),
        ({"to_release": 10}, errors.UnknownReleaseError),
        pytest.param({"to_release": "101"}, errors.ArgumentTypeError, marks=pytest.mark.wrong_argument_type),
        pytest.param({"to_release": 101.5}, errors.ArgumentTypeError, marks=pytest.mark.wrong_argument_type),
        ({"to_release": True}, errors.ArgumentTypeError),
        ({"from_release": 999}, errors.UnknownReleaseError),
        ({"final_database": "nope"}, errors.UnknownDatabaseError),
        pytest.param({"explain": "no"}, errors.ArgumentTypeError, marks=pytest.mark.wrong_argument_type),
        ({"budget_states": 0}, errors.InvalidArgumentError),
    ],
)
def test_invalid_arguments_raise_before_any_search(api, kwargs, error):
    """api_harmonize-4, -13: arguments are checked at the boundary, for found and unknown ids alike."""
    for identifier in ("QSER1", "zzz_unknown"):
        with pytest.raises(error) as info:
            api.convert(identifier, **kwargs)
        assert isinstance(info.value, errors.InputError)
        if "release" in next(iter(kwargs)) and error is errors.UnknownReleaseError:
            assert "100 to 101" in str(info.value)
            assert info.value.allowed == (100, 101)
    assert api.engine.calls == []


def test_per_call_budget_reaches_the_engine(api):
    """A call's budget is the call's own (decision 46 g); the API's settings are unchanged."""
    api.convert("QSER1", budget_states=7, budget_seconds=2.5)
    used = api.engine.calls[-1]["search"]
    assert (used.budget_states, used.budget_seconds) == (7, 2.5)
    assert api.search.budget_states == 200_000
    api.convert("QSER1")
    assert api.engine.calls[-1]["search"] is api.search


@pytest.mark.wrong_argument_type
def test_identifier_that_is_not_a_string_is_an_error_naming_the_argument(api):
    """api_harmonize-13: a non-string identifier is an error that names the argument."""
    with pytest.raises(errors.ArgumentTypeError, match="`identifier` must be a string"):
        api.convert(None)
    assert api.engine.calls == []


def test_identifier_input_is_checked_and_stripped(api):
    """api_harmonize-13: white space is ignored, an empty string is a missing value naming the argument."""
    assert api.convert(" QSER1 ").targets() == ("ENSG03.4",)
    assert api.convert(" QSER1 ").query_id == " QSER1 "
    with pytest.raises(errors.MissingValueError, match="empty"):
        api.convert("  ")
    assert api.convert("QSER1", final_database="hgnc symbol").targets() == ("QSER1",)


def test_releases_are_those_of_the_graph(tmp_path):
    """api_harmonize-6: the releases come from the graph, not from the network state at start."""
    api = fake_api(tmp_path, ANSWERS, releases=[56, 57, 58, 90, 101])
    assert api.releases() == [56, 57, 58, 90, 101]
    assert api.convert("QSER1", to_release=56).to_release == 56


def test_batch_validates_first_converts_each_id_once_and_keeps_per_item_errors(tmp_path):
    """api_harmonize-7: a bad item fails before any work; an error of one identifier does not lose the batch."""
    inactive = InactiveAtReleaseError("ENSG03", 100, [[101, 101]])
    api = fake_api(tmp_path, ANSWERS, errors={("ENSG03", 100): inactive})
    with pytest.raises(errors.ArgumentTypeError, match=r"positions \[1\]"):
        api.convert_many(["QSER1", None, "NOGENE"])
    with pytest.raises(errors.MissingValueError, match=r"empty strings, at positions \[2\]"):
        api.convert_many(["QSER1", "NOGENE", ""])
    with pytest.raises(errors.ArgumentTypeError, match="sequence of strings"):
        api.convert_many("QSER1")
    assert api.engine.calls == []
    out = api.convert_many(["QSER1", "ENSG03", "QSER1", "zzz"], from_release=100)
    assert [r.query_id for r in out] == ["QSER1", "ENSG03", "QSER1", "zzz"]
    assert out[0] == out[2]
    assert out[3].status == "not_found"
    # an id that the graph does not hold at the release given is read where the graph holds it, and flagged
    assert [(c["identifier"], c["from_release"]) for c in api.engine.calls] == [
        ("QSER1", 100),
        ("ENSG03", 100),
        ("ENSG03", None),
    ]
    assert out[1].error is None and out[1].targets() == ("ENSG03.4",)
    assert out[1].from_release_fallback is True and out[1].from_release_given == 100
    assert out[1].from_release == 101 and out[1].from_release_inferred is True
    assert out[0].from_release_fallback is False and out[0].from_release == 100
    single = api.convert("ENSG03", from_release=100)
    assert single == out[1]


def test_batch_accepts_any_sequence_of_strings(api):
    """accuracy_track-11: a NumPy array or a pandas Index of strings is a batch; a string is not."""
    import numpy as np
    import pandas as pd

    expected = [r.targets() for r in api.convert_many(["QSER1", "ENSG03"])]
    assert [r.targets() for r in api.convert_many(np.array(["QSER1", "ENSG03"]))] == expected
    assert [r.targets() for r in api.convert_many(pd.Index(["QSER1", "ENSG03"]))] == expected


def test_every_error_of_one_identifier_is_kept_per_item(tmp_path):
    """web_setup-11: a search over its budget, and any other error of one identifier, are that item's answer."""
    over = SearchBudgetExceededError("QSER1", 10, 0.5, 10, 30.0)
    out = fake_api(tmp_path, ANSWERS, errors={"QSER1": over}).convert_many(["QSER1", "ENSG03"])
    assert out[0].error.type == "SearchBudgetExceededError" and "budget: 10 states" in out[0].error.message
    assert out[0].error.input_error is False and out[0].status == "error" and out[0].reason == "error"
    assert out[0].graph_id == "QSER1" and out[1].error is None and out[1].targets() == ("ENSG03.4",)
    for other, is_input in ((errors.UnknownIdentifierError("not a database"), True), (RuntimeError("broken"), False)):
        item = fake_api(tmp_path, ANSWERS, errors={"QSER1": other}).convert_many(["QSER1"])[0]
        assert item.error.type == type(other).__name__ and item.error.input_error is is_input
    with pytest.raises(SearchBudgetExceededError):
        fake_api(tmp_path, ANSWERS, errors={"QSER1": over}).convert("QSER1")


def test_unknown_version_is_answered_with_nothing_and_the_versions_that_exist(api):
    """An id with a version the graph does not hold is not converted; the result names the versions that exist."""
    result = api.convert("ENSG01.7")
    assert result.candidates == () and result.status == "unknown_version" and result.reason == "unknown_version"
    assert result.known_versions == ("ENSG01.2",) and result.graph_id is None and result.error is None
    assert api.engine.calls == []
    assert api.convert("ENSG01.2").known_versions == ()
    assert api.convert("zzz.7").known_versions == ()
    assert issubclass(UnknownVersionError, ValueError)


def test_results_are_strict_json_and_read_back_equal(api):
    """api_harmonize-8: every result survives a strict JSON round trip unchanged, with and without explain."""
    for identifier in ("MIXED", "ENSG09.1", "QSER1", "NOGENE", "zzz"):
        for explain in (False, True):
            r = api.convert(identifier, final_database="HGNC Symbol", explain=explain)
            data = json.loads(json.dumps(r.to_dict(), allow_nan=False))
            assert ConversionResult.from_dict(data) == r
            assert type(data["no_target"]) is bool
    r = api.convert("MIXED", final_database="HGNC Symbol", explain=True)
    first = r.candidates[0]
    assert (first.gene, first.target, first.entry_gene, first.bridges, first.score) == (
        "ENSG01.2",
        "SYM1",
        "ENSG01.2",
        0,
        1.0,
    )
    steps = [r.path.steps[i] for i in first.path + first.final_path]
    assert [(s.kind, s.source, s.target, s.release) for s in steps] == [
        ("entry", "MIXED", "ENSG01.2", 101),
        ("final", "ENSG01.2", "SYM1", 101),
    ]
    assert api.convert("MIXED").path is None


def test_configure_logger_leaves_the_root_logger_alone(api):
    """api_harmonize-9: only the package logger is configured, idempotently, with a changeable level."""
    root = logging.getLogger()
    before = (root.level, list(root.handlers))
    package = logging.getLogger("idtrack")
    old = (package.level, list(package.handlers))
    try:
        api.configure_logger(logging.ERROR)
        api.configure_logger("DEBUG")
        assert (root.level, list(root.handlers)) == before
        assert package.level == logging.DEBUG
        assert len(package.handlers) == len(old[1]) + 1
        assert logging.getLogger("idtrack.api").getEffectiveLevel() == logging.DEBUG
    finally:
        package.setLevel(old[0])
        package.handlers[:] = old[1]


def test_infer_source_names_every_tied_origin_and_rejects_nothing_to_infer(api):
    """api_harmonize-10: no IndexError; ties are all returned, sorted; a string is not a list (accuracy_track-11)."""
    with pytest.raises(errors.UnknownIdentifierError, match="None of the 0"):
        api.infer_source([])
    with pytest.raises(errors.UnknownIdentifierError, match="None of the 1"):
        api.infer_source(["zzz"])
    with pytest.raises(errors.InvalidArgumentError, match="is not a mode"):
        api.infer_source(["QSER1"], mode="best")
    with pytest.raises(errors.ArgumentTypeError, match="sequence of strings"):
        api.infer_source("QSER1")
    assert api.infer_source(["QSER1", "ENSG03"]) == [[38, 100], [38, 101]]
    assert api.infer_source(["QSER1"], mode="release", report_only_winner=False) == [[100, 1], [101, 1]]


def test_list_methods_return_copies(api):
    """api_harmonize-11: editing a returned value does not change the graph; assemblies are JSON keys."""
    by_assembly = api.databases_by_assembly()
    assert by_assembly == {"37": ["Old DB"], "38": ["HGNC Symbol"]}
    by_assembly["38"].remove("HGNC Symbol")
    api.external_databases().clear()
    api.releases().clear()
    api.database_forms().clear()
    assert api.graph.available_external_databases_assembly[38] == {"HGNC Symbol"}
    assert api.convert("QSER1", final_database="HGNC Symbol").targets() == ("QSER1",)
    assert api.releases() == [100, 101]
    assert api.final_databases() == ["HGNC Symbol", "base_ensembl_gene", "ensembl_gene"]


def test_classify_counts_a_query_that_maps_to_itself_as_unchanged(api):
    """api_harmonize-12: 'changed' compares with the resolved id, without version or case."""
    results = api.convert_many(["ENSG03", "qser1", "MIXED"])
    bins = api.classify(results)
    assert len(bins["matching_1_to_1"]) == 2  # ENSG03 -> ENSG03.4, qser1 -> ENSG03.4 (QSER1's gene)
    assert [r.query_id for r in bins["changed_only_1_to_1"]] == ["qser1"]  # a symbol becomes a gene id
    by_symbol = api.classify(api.convert_many(["qser1", "ENSG03"], final_database="HGNC Symbol"))
    assert [r.query_id for r in by_symbol["changed_only_1_to_1"]] == ["ENSG03"]
    assert len(bins["matching_1_to_n"]) == 1 and bins["changed_only_1_to_n"][0].query_id == "MIXED"
    assert "Total processed: 3" in api.classification_summary(bins)
    with pytest.raises(errors.InvalidArgumentError):
        api.classify(results, "first")


def test_import_idtrack_and_api_does_not_load_anndata():
    """api_harmonize-40: the conversion core imports without anndata and scipy."""
    code = "import sys, idtrack; idtrack.API; print(sorted(m for m in ('anndata', 'scipy') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert out.strip() == "[]"
    pytest.importorskip("anndata")
    code = "import sys, idtrack; idtrack.HarmonizeFeatures; print('anndata' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert out.strip() == "True"


def test_conversions_from_threads_equal_serial_ones(api):
    """api_harmonize-46: conversions only read the graph; threads give the serial answers."""
    from concurrent.futures import ThreadPoolExecutor

    identifiers = ["MIXED", "ENSG09.1", "QSER1", "ENSG03", "NOGENE", "zzz"] * 20
    serial = [api.convert(i) for i in identifiers]
    with ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(api.convert, identifiers))
    assert threaded == serial


def test_from_database_is_validated_and_passed_to_the_engine(api):
    """The database a query is meant as reaches the engine; an unknown one is an error before any search."""
    with pytest.raises(errors.UnknownDatabaseError, match="`from_database`='nope'"):
        api.convert("QSER1", from_database="nope")
    assert api.engine.calls == []
    api.convert("QSER1", from_database="hgnc symbol")
    assert api.engine.calls[-1]["from_database"] == "HGNC Symbol"
    api.convert("QSER1")
    assert api.engine.calls[-1]["from_database"] is None
