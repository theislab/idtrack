"""The contract of :py:class:`idtrack.API` as the backend of a service: plain results, validation, no hidden state."""

import json
import logging
import subprocess  # noqa: S404
import sys

import pytest

from idtrack._track import InactiveAtReleaseError, SearchBudgetExceededError, UnknownVersionError
from tests.unit._fake_engine import fake_api

ANSWERS = {
    # a symbol whose candidate genes mix a gene with an HGNC symbol and one without (strategy="all")
    "MIXED": {"ENSG01.2": {"HGNC Symbol": ["SYM1"]}, "ENSG02.1": {}},
    # a gene split into two, both scored equally
    "ENSG09.1": {"ENSG09.2": {"HGNC Symbol": ["NINE"]}, "ENSG10.1": {"HGNC Symbol": ["TEN"]}},
    "QSER1": {"ENSG03.4": {"HGNC Symbol": ["QSER1"]}},
    "ENSG03": {"ENSG03.4": {"HGNC Symbol": ["QSER1"]}},
    "NOGENE": {},
    "ENSG01.2": {"ENSG01.2": {"HGNC Symbol": ["SYM1"]}},
    "ENSG02.1": {"ENSG02.1": {}},
}


@pytest.fixture
def api(tmp_path):
    """An API on the fake engine with ANSWERS."""
    return fake_api(tmp_path, ANSWERS)


def test_strategy_all_with_mixed_fallbacks_returns_the_genuine_targets(api):
    """api_harmonize-1: one candidate without an id in the database no longer raises; it is listed apart."""
    r = api.convert_identifier("MIXED", final_database="HGNC Symbol", strategy="all")
    assert r["target_id"] == ["SYM1"]
    assert r["last_node"] == [["ENSG01.2", "SYM1"]]
    assert r["final_database"] == "HGNC Symbol" and r["no_target"] is False
    assert r["no_target_genes"] == ["ENSG02.1"]
    only_fallback = api.convert_identifier("ENSG02.1", final_database="HGNC Symbol", strategy="all")
    assert only_fallback["no_target"] is True and only_fallback["final_database"] == "ensembl_gene"
    assert only_fallback["target_id"] == ["ENSG02.1"]


def test_tied_best_targets_are_flagged_and_keep_the_order_of_the_engine(tmp_path):
    """api_harmonize-2, -3: several tied targets are an explicit list in the engine's order, never a set's."""
    a = fake_api(tmp_path, ANSWERS).convert_identifier("ENSG09.1", final_database="HGNC Symbol")
    assert a["ambiguous"] is True and a["tie"] is True and a["target_id"] == ["NINE", "TEN"]
    assert a["last_node"] == [["ENSG09.2", "NINE"], ["ENSG10.1", "TEN"]]
    assert [t for _, t in a["last_node"]] == a["target_id"]
    # the engine's ranking (best first) is the order of the result, whatever the ids are
    ranking = {"ENSG09.1": {"ENSG10.1": {"HGNC Symbol": ["TEN"]}, "ENSG09.2": {"HGNC Symbol": ["NINE"]}}}
    ranked = fake_api(tmp_path, ranking).convert_identifier("ENSG09.1", final_database="HGNC Symbol", strategy="all")
    assert ranked["last_node"] == [["ENSG10.1", "TEN"], ["ENSG09.2", "NINE"]] and ranked["target_id"] == ["TEN", "NINE"]
    assert ranked["ambiguous"] is True and ranked["tie"] is False
    one = fake_api(tmp_path, ANSWERS).convert_identifier("QSER1")
    assert one["ambiguous"] is False and one["tie"] is False


def test_result_says_how_the_identifier_was_read(api):
    """The match, the node, the starting release and the databases of the query are part of the result."""
    exact = api.convert_identifier("QSER1", final_database="HGNC Symbol")
    assert (exact["graph_id"], exact["match"]) == ("QSER1", "exact")
    assert exact["from_release"] == 101 and exact["from_release_inferred"] is True and exact["from_databases"] == []
    by_case = api.convert_identifier("qser1", from_release=100, from_database="HGNC Symbol")
    assert (by_case["graph_id"], by_case["match"]) == ("QSER1", "case")
    assert by_case["from_release"] == 100 and by_case["from_release_inferred"] is False
    assert by_case["from_databases"] == ["HGNC Symbol"]
    unknown = api.convert_identifier("zzz")
    assert (unknown["graph_id"], unknown["match"], unknown["from_release"]) == (None, None, None)
    assert unknown["no_corresponding"] is True and unknown["from_release_inferred"] is True
    # an id without version that has no node of its own is read as its version
    assert api.convert_identifier("ENSG01")["no_corresponding"] is False


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"to_release": 999}, ValueError),
        ({"to_release": 10}, ValueError),
        pytest.param({"to_release": "101"}, TypeError, marks=pytest.mark.wrong_argument_type),
        pytest.param({"to_release": 101.5}, TypeError, marks=pytest.mark.wrong_argument_type),
        ({"to_release": True}, TypeError),
        ({"from_release": 999}, ValueError),
        ({"final_database": "nope"}, ValueError),
        pytest.param({"strategy": "first"}, ValueError, marks=pytest.mark.wrong_argument_type),
    ],
)
def test_invalid_arguments_raise_before_any_search(api, kwargs, error):
    """api_harmonize-4, -13: releases and databases are checked at the boundary, for found and unknown ids alike."""
    for identifier in ("QSER1", "zzz_unknown"):
        with pytest.raises(error) as info:
            api.convert_identifier(identifier, **kwargs)
        if "release" in next(iter(kwargs)) and error is ValueError:
            assert "100 to 101" in str(info.value)
    assert api.track.calls == []


@pytest.mark.wrong_argument_type
def test_identifier_that_is_not_a_string_is_an_error_naming_the_argument(api):
    """api_harmonize-13: a non-string identifier is an error that names the argument."""
    with pytest.raises(TypeError, match="`identifier` must be a string"):
        api.convert_identifier(None)
    assert api.track.calls == []


def test_identifier_input_is_checked_and_stripped(api):
    """api_harmonize-13: white space is ignored, an empty string is an error naming the argument."""
    assert api.convert_identifier(" QSER1 ")["target_id"] == ["ENSG03.4"]
    assert api.convert_identifier(" QSER1 ")["query_id"] == " QSER1 "
    with pytest.raises(ValueError, match="empty"):
        api.convert_identifier("  ")
    assert api.convert_identifier("QSER1", final_database="hgnc symbol")["target_id"] == ["QSER1"]


def test_list_ensembl_releases_are_those_of_the_graph(tmp_path):
    """api_harmonize-6: the releases come from the graph, not from the network state at start."""
    api = fake_api(tmp_path, ANSWERS, releases=[56, 57, 58, 90, 101])
    assert api.list_ensembl_releases() == [56, 57, 58, 90, 101]
    assert api.convert_identifier("QSER1", to_release=56)["to_release"] == 56


def test_batch_validates_first_converts_each_id_once_and_keeps_per_item_errors(tmp_path):
    """api_harmonize-7: a bad item fails before any work; a per-query failure does not lose the batch."""
    inactive = InactiveAtReleaseError("ENSG03", 100, [[101, 101]])
    api = fake_api(tmp_path, ANSWERS, errors={("ENSG03", 100): inactive})
    with pytest.raises(TypeError, match=r"positions \[1\]"):
        api.convert_identifier_multiple(["QSER1", None, "NOGENE"], verbose=False)
    with pytest.raises(ValueError, match=r"empty strings, at positions \[2\]"):
        api.convert_identifier_multiple(["QSER1", "NOGENE", ""], verbose=False)
    assert api.track.calls == []
    out = api.convert_identifier_multiple(["QSER1", "ENSG03", "QSER1", "zzz"], from_release=100, verbose=False)
    assert [r["query_id"] for r in out] == ["QSER1", "ENSG03", "QSER1", "zzz"]
    assert out[0] == out[2] and out[0] is not out[2]
    assert out[3]["no_corresponding"] is True
    # an id that the graph does not hold at the release given is read where the graph holds it, and flagged
    assert [(c["from_id"], c["from_release"]) for c in api.track.calls] == [
        ("QSER1", 100),
        ("ENSG03", 100),
        ("ENSG03", None),
    ]
    assert out[1]["error"] is None and out[1]["target_id"] == ["ENSG03.4"]
    assert out[1]["from_release_fallback"] is True and out[1]["from_release_given"] == 100
    assert out[1]["from_release"] == 101 and out[1]["from_release_inferred"] is True
    assert out[0]["from_release_fallback"] is False and out[0]["from_release"] == 100
    single = api.convert_identifier("ENSG03", from_release=100)
    assert single == out[1]


def test_only_a_search_over_its_budget_is_kept_per_item(tmp_path):
    """A search over its budget is an item error; any other error of the engine stops the batch."""
    over = SearchBudgetExceededError("QSER1", 10, 0.5, 10, 30.0)
    out = fake_api(tmp_path, ANSWERS, errors={"QSER1": over}).convert_identifier_multiple(["QSER1"], verbose=False)
    assert out[0]["error"]["type"] == "SearchBudgetExceededError" and "budget: 10 states" in out[0]["error"]["message"]
    for other in (ValueError("not a database"), RuntimeError("broken")):
        with pytest.raises(type(other)):
            fake_api(tmp_path, ANSWERS, errors={"QSER1": other}).convert_identifier_multiple(["QSER1"], verbose=False)


def test_unknown_version_is_answered_with_nothing_and_the_versions_that_exist(api):
    """An id with a version the graph does not hold is not converted; the result names the versions that exist."""
    result = api.convert_identifier("ENSG01.7")
    assert result["target_id"] == [] and result["no_corresponding"] is True and result["no_conversion"] is False
    assert result["known_versions"] == ["ENSG01.2"] and result["graph_id"] is None and result["error"] is None
    assert api.track.calls == []
    assert api.convert_identifier("ENSG01.2")["known_versions"] == []
    assert api.convert_identifier("zzz.7")["known_versions"] == []
    assert issubclass(UnknownVersionError, ValueError)


def test_results_are_plain_data(api):
    """api_harmonize-8: every result survives a JSON round trip unchanged, with and without explain."""
    for identifier in ("MIXED", "ENSG09.1", "QSER1", "NOGENE", "zzz"):
        for explain in (False, True):
            r = api.convert_identifier(identifier, final_database="HGNC Symbol", strategy="all", explain=explain)
            assert json.loads(json.dumps(r)) == r
            assert type(r["no_target"]) is bool
    r = api.convert_identifier("MIXED", final_database="HGNC Symbol", strategy="all", explain=True)
    assert r["the_path"][0]["ensembl_gene"] == "ENSG01.2" and r["the_path"][0]["target_id"] == "SYM1"
    assert r["the_path"][0]["edges"] == [["MIXED", "ENSG01.2", 0, 101], ["ENSG01.2", "SYM1", 0, 101]]
    assert r["the_path"][0]["entry_gene"] == "ENSG01.2" and r["the_path"][0]["external_jump"] == 0
    assert r["the_path"][0]["score"] == 1.0


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
        assert logging.getLogger("idtrack._api").getEffectiveLevel() == logging.DEBUG
    finally:
        package.setLevel(old[0])
        package.handlers[:] = old[1]


def test_infer_identifier_source_names_every_tied_origin_and_rejects_nothing_to_infer(api):
    """api_harmonize-10: no IndexError; ties are all returned, sorted."""
    with pytest.raises(ValueError, match="None of the 0"):
        api.infer_identifier_source([])
    with pytest.raises(ValueError, match="None of the 1"):
        api.infer_identifier_source(["zzz"])
    with pytest.raises(ValueError, match="Unknown mode"):
        api.infer_identifier_source(["QSER1"], mode="best")
    assert api.infer_identifier_source(["QSER1", "ENSG03"]) == [[38, 100], [38, 101]]
    assert api.infer_identifier_source(["QSER1"], mode="ensembl_release", report_only_winner=False) == [
        [100, 1],
        [101, 1],
    ]


def test_list_methods_return_copies(api):
    """api_harmonize-11: editing a returned value does not change the graph."""
    by_assembly = api.list_external_databases_by_assembly()
    by_assembly[38].remove("HGNC Symbol")
    api.list_external_databases().clear()
    api.list_genome_assemblies().clear()
    api.external_database_forms().clear()
    assert api.track.graph.available_external_databases_assembly[38] == {"HGNC Symbol"}
    assert api.convert_identifier("QSER1", final_database="HGNC Symbol")["target_id"] == ["QSER1"]
    assert api.list_genome_assemblies() == [37, 38]


def test_classify_counts_a_query_that_maps_to_itself_as_unchanged(api):
    """api_harmonize-12: 'changed' compares with the resolved id, without version or case."""
    results = api.convert_identifier_multiple(["ENSG03", "qser1", "MIXED"], verbose=False)
    bins = api.classify_multiple_conversion(results)
    assert len(bins["matching_1_to_1"]) == 2  # ENSG03 -> ENSG03.4, qser1 -> ENSG03.4 (QSER1's gene)
    assert [r["query_id"] for r in bins["changed_only_1_to_1"]] == ["qser1"]  # a symbol becomes a gene id
    by_symbol = api.classify_multiple_conversion(
        api.convert_identifier_multiple(["qser1", "ENSG03"], final_database="HGNC Symbol", verbose=False)
    )
    assert [r["query_id"] for r in by_symbol["changed_only_1_to_1"]] == ["ENSG03"]
    assert len(bins["matching_1_to_n"]) == 1 and bins["changed_only_1_to_n"][0]["query_id"] == "MIXED"


def test_import_idtrack_and_api_does_not_load_anndata():
    """api_harmonize-40: the conversion core imports without anndata and scipy."""
    code = "import sys, idtrack; idtrack.API; print(sorted(m for m in ('anndata', 'scipy') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert out.strip() == "[]"
    code = "import sys, idtrack; idtrack.HarmonizeFeatures; print('anndata' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert out.strip() == "True"


def test_conversions_from_threads_equal_serial_ones(api):
    """api_harmonize-46: conversions only read the graph; threads give the serial answers."""
    from concurrent.futures import ThreadPoolExecutor

    ids = ["MIXED", "ENSG09.1", "QSER1", "ENSG03", "NOGENE", "zzz"] * 20
    serial = [api.convert_identifier(i, strategy="all") for i in ids]
    with ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda i: api.convert_identifier(i, strategy="all"), ids))
    assert threaded == serial


def test_from_database_is_validated_and_passed_to_the_engine(api):
    """The database a query is meant as reaches the engine; an unknown one is an error before any search."""
    with pytest.raises(ValueError, match="`from_database`='nope'"):
        api.convert_identifier("QSER1", from_database="nope")
    assert api.track.calls == []
    api.convert_identifier("QSER1", from_database="hgnc symbol")
    assert api.track.calls[-1]["from_database"] == "HGNC Symbol"
    api.convert_identifier("QSER1")
    assert api.track.calls[-1]["from_database"] is None
