#!/usr/bin/env python3
"""High-value unit tests for `idtrack._track_tests.TrackTests`.

The goal is to exercise the developer-facing test harness on a tiny synthetic
graph (2 releases) without requiring large MySQL downloads.
"""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._the_graph import ReleaseSet
from idtrack._track_tests import CheckResult, TrackTests


@dataclass(frozen=True)
class _SyntheticGraphMySQL:
    tables_by_release: dict[int, dict[str, pd.DataFrame]]
    available_releases: list[int]


def _synthetic_graph_mysql() -> _SyntheticGraphMySQL:
    gene_r100 = pd.DataFrame(
        {
            "gene_id": [1, 2],
            "stable_id": ["ENSG00000000001", "ENSG00000000002"],
            "version": [1, 1],
            "seq_region_id": [1, 1],
        }
    )
    gene_r101 = pd.DataFrame(
        {
            "gene_id": [1, 3],
            "stable_id": ["ENSG00000000001", "ENSG00000000003"],
            "version": [2, 1],
            "seq_region_id": [1, 1],
        }
    )

    stable_id_event = pd.DataFrame(
        {
            "mapping_session_id": [1, 1],
            "old_stable_id": ["ENSG00000000001", "ENSG00000000002"],
            "old_version": [1, 1],
            "new_stable_id": ["ENSG00000000001", "ENSG00000000003"],
            "new_version": [2, 1],
            "score": [0.8, 0.9],
            "type": ["gene", "gene"],
        }
    )
    mapping_session = pd.DataFrame(
        {
            "mapping_session_id": [1],
            "old_db_name": ["homo_sapiens_core_100_38"],
            "new_db_name": ["homo_sapiens_core_101_38"],
            "old_release": [100],
            "new_release": [101],
            "old_assembly": ["38"],
            "new_assembly": ["38"],
            "created": ["2020-01-01"],
        }
    )

    # every gene lies on sequence region 1, a chromosome of the reference sequence
    regions = {
        "attrib_type": pd.DataFrame({"attrib_type_id": [6, 16], "code": ["toplevel", "non_ref"]}),
        "seq_region_attrib": pd.DataFrame({"seq_region_id": [1], "attrib_type_id": [6]}),
    }
    tables_by_release = {
        100: {"gene": gene_r100, "stable_id_event": stable_id_event, "mapping_session": mapping_session, **regions},
        101: {"gene": gene_r101, "stable_id_event": stable_id_event, "mapping_session": mapping_session, **regions},
    }
    return _SyntheticGraphMySQL(tables_by_release=tables_by_release, available_releases=[100, 101])


@pytest.fixture
def synthetic_track_tests(tmp_path, monkeypatch) -> TrackTests:
    """Create a TrackTests instance backed by a tiny synthetic 2-release graph."""
    synthetic = _synthetic_graph_mysql()

    def _fake_core_index(*, organism: str, genome_assembly: int, **_):  # noqa: ARG001
        releases = list(synthetic.available_releases)
        db_for_release = {r: f"{organism}_core_{r}_{genome_assembly}" for r in releases}
        return {
            "organism": organism,
            "genome_assembly": genome_assembly,
            "ports": (3306,),
            "releases_by_port": {3306: set(releases)},
            "db_by_port_release": {3306: db_for_release.copy()},
            "releases": releases,
            "port_for_release": {r: 3306 for r in releases},
            "db_for_release": db_for_release,
            "releases_on_mysql": releases,
        }

    def _available_releases_versions(self: DatabaseManager, **kwargs) -> list[int]:
        return list(synthetic.available_releases)

    def _download_table(self: DatabaseManager, table_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
        df = synthetic.tables_by_release[int(self.ensembl_release)].get(table_key)
        if df is None:
            return pd.DataFrame()
        if usecols is None:
            return df.copy(deep=True)
        return df.loc[:, usecols].copy(deep=True)

    def _create_external_all(
        self: DatabaseManager, return_mode: str, narrow_external: bool = True, **kwargs  # noqa: ARG002
    ) -> pd.DataFrame:
        release = int(self.ensembl_release)
        graph_id = "ENSG00000000001.1" if release == 100 else "ENSG00000000001.2"
        return pd.DataFrame(
            [
                {
                    "release": release,
                    "graph_id": graph_id,
                    "id_db": "TP53",
                    "name_db": "HGNC",
                    "ensembl_identity": np.nan,
                    "xref_identity": np.nan,
                    "assembly": 38,
                },
            ]
        )

    monkeypatch.setattr(DatabaseManager, "available_releases_versions", _available_releases_versions)
    monkeypatch.setattr(DatabaseManager, "download_table", _download_table)
    monkeypatch.setattr(DatabaseManager, "create_external_all", _create_external_all)
    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: _fake_core_index(**kw)))

    dm = DatabaseManager(
        organism="homo_sapiens",
        form="gene",
        local_repository=str(tmp_path),
        ensembl_release=101,
        ignore_before=100,
        ignore_after=101,
        genome_assembly=38,
        store_raw_always=True,
    )

    # the harness is made as a user makes it: by its constructor, which builds the graph through the manager
    return TrackTests(dm, narrow=True, save_after_calculation=False, form_list=["gene"])


def test_track_tests_basic_invariants(synthetic_track_tests):
    """The invariants of a built graph hold for the synthetic graph, and each result says what was compared."""
    tests = synthetic_track_tests
    results = [
        tests.is_node_consistency_robust(verbose=False),
        tests.is_range_functions_robust(verbose=False),
        tests.is_base_is_range_correct(verbose=False),
        tests.is_combined_summary_complete(),
        tests.is_edge_with_same_nts_only_at_backbone_nodes(),
        tests.is_id_functions_consistent_ensembl(verbose=False),
        tests.is_id_functions_consistent_ensembl_2(verbose=False),
    ]
    for result in results:
        assert isinstance(result, CheckResult) and result.holds is True and bool(result) is True, result
        assert "error" not in result.details and result.details["compared"] > 0, result
    assert [result.check for result in results] == [
        "is_node_consistency_robust",
        "is_range_functions_robust",
        "is_base_is_range_correct",
        "is_combined_summary_complete",
        "is_edge_with_same_nts_only_at_backbone_nodes",
        "is_id_functions_consistent_ensembl",
        "is_id_functions_consistent_ensembl_2",
    ]
    # two releases on each of two assemblies (the world lists GRCh37 beside GRCh38 with the same genes); four
    # gene versions on the backbone, three gene ids without version
    assert results[5].details == {"compared": 4, "n_differing": 0, "differing": []}
    assert results[6].details["compared"] == 4 and results[1].details["compared"] == 3


def test_result_of_a_check_is_its_verdict_and_never_a_truthy_container():
    """A result is true exactly when the invariant holds, whatever its details hold."""
    holds = CheckResult("is_something", True, {"compared": 3})
    fails = CheckResult("is_something", False, {"compared": 3, "n_differing": 1, "differing": ["x"]})
    assert holds and not fails
    assert (bool(holds), bool(fails)) == (True, False)
    assert CheckResult("is_something", False).details == {}
    with pytest.raises(AttributeError):
        holds.holds = False  # a result cannot be changed


def test_check_that_cannot_be_completed_is_a_result_that_does_not_hold(synthetic_track_tests, monkeypatch):
    """The checks never raise: what stops a check is reported in its result, and the next check still runs."""

    def no_tables(self, *args, **kwargs):
        raise OSError("the tables cannot be read")

    with monkeypatch.context() as broken:
        broken.setattr(DatabaseManager, "get_db", no_tables)
        for check in (
            synthetic_track_tests.is_id_functions_consistent_ensembl,
            synthetic_track_tests.is_id_functions_consistent_external,
        ):
            result = check(verbose=False)
            assert result.holds is False and not result
            assert result.details["error"] == "OSError: the tables cannot be read"
            assert "no_tables" in result.details["traceback"]
        conversions = synthetic_track_tests.is_final_external_conversion_robust(
            database="HGNC", ens_rel=101, verbose=False
        )
        assert conversions.holds is False and conversions.details["error"].startswith("OSError")
    # a release or a fraction that cannot be used is reported the same way
    serve_external_table(monkeypatch, TP53_ROWS)
    for arguments in (
        {"database": "HGNC", "ens_rel": 99},
        {"database": "HGNC", "ens_rel": 101, "from_fraction": 2.0},
    ):
        result = synthetic_track_tests.is_final_external_conversion_robust(verbose=False, **arguments)
        assert result.holds is False and result.details["error"].startswith("ValueError"), arguments
    # a check of the graph alone is not touched by the tables
    assert synthetic_track_tests.is_combined_summary_complete().holds is True


def test_history_travel_testing_reports_the_conversions_of_the_synthetic_history(synthetic_track_tests):
    """The conversion harness on a history that is known: which ids it converted, to what, and which it lost.

    Release 100 has the genes 1 (version 1, named TP53) and 2; in release 101 gene 1 has version 2 (still TP53)
    and gene 2 became gene 3 (Ensembl's event, score 0.9).
    """

    def run(from_release, from_database, to_release, to_database):
        return synthetic_track_tests.history_travel_testing(
            from_release=from_release, from_assembly=38, from_database=from_database, to_release=to_release,
            to_database=to_database, go_external=False, prioritize_to_one_filter=True, convert_using_release=True,
            from_fraction=1.0, verbose=False,
        )  # fmt: skip

    genes = run(100, "ensembl_gene", 101, "ensembl_gene")
    assert genes["ids"]["from"] == ["ENSG00000000001.1", "ENSG00000000002.1"]
    assert genes["conversion"] == {
        "ENSG00000000001.1": ["ENSG00000000001.2"],
        "ENSG00000000002.1": ["ENSG00000000003.1"],
    }
    assert genes["lost_item"] == [] and genes["one_to_one_ids"] == genes["conversion"]
    assert genes["found_ids_not_accurate"] == {}  # every target is a gene of release 101

    symbol = run(100, "HGNC", 101, "ensembl_gene")
    assert symbol["conversion"] == {"TP53": ["ENSG00000000001.2"]}
    assert symbol["parameters"]["from_database"] == "HGNC"

    to_symbol = run(100, "ensembl_gene", 101, "HGNC")
    assert to_symbol["conversion"] == {"ENSG00000000001.1": ["TP53"]}
    assert to_symbol["lost_item"] == ["ENSG00000000002.1"]  # gene 3 has no symbol: nothing in the database

    report = synthetic_track_tests._format_history_travel_testing_report(to_symbol, include_header=True)
    assert any("History-Travel-Testing" in line for line in report)


def test_database_that_the_graph_does_not_hold_is_refused_before_anything_is_converted(
    synthetic_track_tests, monkeypatch
):
    """A database that cannot work raises and says what is valid; it never gives a report of lost identifiers."""
    tests = synthetic_track_tests
    asked = []
    monkeypatch.setattr(tests, "convert", lambda **arguments: asked.append(arguments), raising=False)
    monkeypatch.setattr(tests.graph, "get_id_list", lambda *arguments: asked.append(arguments) or [], raising=False)

    def run(from_database, to_database, from_assembly=38):
        return tests.history_travel_testing(
            from_release=100, from_assembly=from_assembly, from_database=from_database, to_release=101,
            to_database=to_database, go_external=False, prioritize_to_one_filter=True, convert_using_release=True,
            verbose=False,
        )  # fmt: skip

    sources = "valid: ['HGNC', 'assembly_38_ensembl_gene', 'base_ensembl_gene', 'ensembl_gene']"
    targets = "valid: ['HGNC', 'base_ensembl_gene', 'ensembl_gene']"
    for source in ("No Such Database", "hgnc", "gene", "ensembl_transcript", "assembly_37_ensembl_gene"):
        with pytest.raises(ValueError) as refused:  # the graph was built with genes only, and 37 is another assembly
            run(source, "ensembl_gene")
        assert f"`from_database`={source!r} is not a database of the loaded graph on genome assembly 38" in str(
            refused.value
        )
        assert str(refused.value).endswith(f"{sources}.")
    # a target is a database that a conversion can end in: the ids of one assembly and of other forms are none
    for target in ("No Such Database", "hgnc", "gene", "assembly_38_ensembl_gene", "ensembl_transcript"):
        with pytest.raises(ValueError) as refused:
            run("ensembl_gene", target)
        assert str(refused.value) == (
            f"`to_database`={target!r} is not a database of the loaded graph at Ensembl release 101; {targets}."
        )
    with pytest.raises(
        ValueError, match=r"`from_database`='HGNC' is not a database .* genome assembly 12 .*valid: \[\]"
    ):
        run("HGNC", "ensembl_gene", from_assembly=12)  # an assembly the graph does not have holds nothing
    with pytest.raises(ValueError, match="`database`='No Such Database' is not an external database .*valid: "):
        tests.is_final_external_conversion_robust(database="No Such Database", ens_rel=101, verbose=False)
    with pytest.raises(ValueError, match=r"`database`='ensembl_gene' is not an external database .*valid: \['HGNC'\]"):
        tests.is_final_external_conversion_robust(database="ensembl_gene", verbose=False)
    assert asked == []  # refused before an identifier was listed or converted


@pytest.mark.wrong_argument_type
def test_database_that_is_not_a_string_is_refused_by_the_self_tests(synthetic_track_tests):
    """A database of another type raises the error that names the argument, as the API's does."""
    arguments = dict(
        from_release=100, from_assembly=38, to_release=101, go_external=False, prioritize_to_one_filter=True,
        convert_using_release=True, verbose=False,
    )  # fmt: skip
    with pytest.raises(TypeError, match="`from_database` must be a string, got None"):
        synthetic_track_tests.history_travel_testing(from_database=None, to_database="ensembl_gene", **arguments)
    with pytest.raises(TypeError, match="`to_database` must be a string, got 5"):
        synthetic_track_tests.history_travel_testing(from_database="ensembl_gene", to_database=5, **arguments)
    with pytest.raises(TypeError, match="`database` must be a string, got 5"):
        synthetic_track_tests.is_final_external_conversion_robust(database=5, verbose=False)


def test_history_travel_testing_takes_the_node_types_of_ensembl_ids(synthetic_track_tests, monkeypatch):
    """Ensembl ids are named by their node type, as source and as target."""

    def run(from_database, to_database, **more):
        arguments = dict(
            from_release=100, from_assembly=38, to_release=101, go_external=False, prioritize_to_one_filter=True,
            convert_using_release=True, verbose=False,
        )  # fmt: skip
        return synthetic_track_tests.history_travel_testing(
            from_database=from_database, to_database=to_database, **{**arguments, **more}
        )

    for source in ("ensembl_gene", "assembly_38_ensembl_gene"):
        assert run(source, "ensembl_gene")["conversion"]["ENSG00000000001.1"] == ["ENSG00000000001.2"]
    assert run("base_ensembl_gene", "base_ensembl_gene")["conversion"] == {
        "ENSG00000000001": ["ENSG00000000001"],
        "ENSG00000000002": ["ENSG00000000003"],
    }
    with pytest.raises(ValueError, match="from_fraction must be in range"):
        run("ensembl_gene", "ensembl_gene", from_fraction=0.0)

    # a gene id that is lost although its id exists at the target release is counted as such
    monkeypatch.setattr(synthetic_track_tests, "convert", lambda **arguments: None, raising=False)
    lost = run("ensembl_gene", "ensembl_gene")
    assert lost["lost_item"] == ["ENSG00000000001.1", "ENSG00000000002.1"]
    assert lost["lost_item_but_the_same_id_exists"] == ["ENSG00000000001.1"]


def test_random_arguments_are_releases_and_databases_of_the_graph(synthetic_track_tests):
    """The random arguments of a history-travel run name a release and a database that exist."""
    graph = synthetic_track_tests.graph
    wrong = []
    drawn_releases = set()
    for seed in range(40):
        random.seed(seed)
        drawn = synthetic_track_tests.history_travel_testing_random_arguments_generator(
            True, [False, False, False, True]
        )
        drawn_releases.add((drawn["from_release"], drawn["to_release"]))
        if (
            drawn["to_release"] not in graph.graph["confident_for_release"]
            or drawn["to_release"] < drawn["from_release"]
            or drawn["to_database"] != "HGNC"
            or drawn["from_database"] != "ensembl_gene"
        ):
            wrong.append((seed, drawn))
    assert not wrong, wrong
    assert drawn_releases == {(100, 100), (100, 101), (101, 101)}  # a database with one release left is drawn too

    random.seed(0)
    metrics = synthetic_track_tests.history_travel_testing_random(
        1.0, False, False, False, True, verbose=False, strict_forward=True, return_result=True
    )
    assert metrics["parameters"]["to_release"] >= metrics["parameters"]["from_release"] and metrics["conversion"]
    assert synthetic_track_tests.history_travel_testing_random(1.0, verbose=False) is None


def test_draw_without_a_candidate_is_one_value_and_the_attempts_are_bounded(synthetic_track_tests, monkeypatch):
    """A draw without a candidate is ``None``, never a made-up release; a draw that cannot succeed ends with an error."""
    tests = synthetic_track_tests
    switches = dict(include_external=True, include_ensembl=False, for_final_database=True, only_backbone_tests=False)
    random.seed(3)
    assert tests.random_dataset_source_generator(38, **switches) in {("HGNC", 38, 100), ("HGNC", 38, 101)}
    assert tests.random_dataset_source_generator(38, release_lower_limit=101, **switches) == ("HGNC", 38, 101)
    assert tests.random_dataset_source_generator(38, release_lower_limit=102, **switches) is None
    assert tests.random_dataset_source_generator(37, **switches) is None  # an assembly without external ids
    with pytest.raises(ValueError, match="leave no database to draw from"):
        tests.random_dataset_source_generator(
            38, include_external=False, include_ensembl=False, for_final_database=True, only_backbone_tests=False
        )

    calls = []
    real = type(tests).random_dataset_source_generator

    def never_a_target(self, assembly, **arguments):
        calls.append(arguments["for_final_database"])
        return None if arguments["for_final_database"] else real(self, assembly, **arguments)

    monkeypatch.setattr(type(tests), "random_dataset_source_generator", never_a_target)
    with pytest.raises(ValueError, match="could be drawn in 100 attempts"):
        tests.history_travel_testing_random_arguments_generator(False, [True, True, True, True])
    assert calls.count(True) == 100  # bounded: one target draw per attempt


def test_versions_of_one_id_active_at_the_same_release_are_reported(synthetic_track_tests, monkeypatch):
    """Two versions of one gene id that are both active at a release violate the history; the check must say so."""
    graph = synthetic_track_tests.graph
    assert synthetic_track_tests.is_range_functions_robust(verbose=False).holds is True
    real = type(graph).active_ranges

    def overlapping(self, node):
        return [[100, 101]] if node == "ENSG00000000001.1" else real(self, node)  # version 2 is active from 101

    monkeypatch.setattr(type(graph), "active_ranges", overlapping)
    result = synthetic_track_tests.is_range_functions_robust(verbose=False)
    assert result.holds is False
    assert result.details["n_overlapping"] == 1
    (pair,) = result.details["overlapping"]
    assert pair["id"] == "ENSG00000000001" and pair["releases"] == [[101, 101]]
    assert sorted(pair["versions"]) == ["ENSG00000000001.1", "ENSG00000000001.2"]


def test_gene_version_whose_edges_and_history_disagree_is_reported(synthetic_track_tests, monkeypatch):
    """A gene version referred to at other releases than its history has is an inconsistency of the graph."""
    graph = synthetic_track_tests.graph
    real = type(graph).active_releases

    def one_release_more(self, node):
        return ReleaseSet.of([100, 101]) if node == "ENSG00000000002.1" else real(self, node)

    monkeypatch.setattr(type(graph), "active_releases", one_release_more)
    result = synthetic_track_tests.is_id_functions_consistent_ensembl_2(verbose=False)
    assert result.holds is False and result.details["n_differing"] == 1
    assert result.details["differing"] == [
        {"node": "ENSG00000000002.1", "edges": [[100, 100]], "history": [[100, 101]]}
    ]


def test_gene_ids_of_the_graph_are_compared_with_the_id_tables_both_ways(synthetic_track_tests, monkeypatch):
    """A gene of the tables that the graph lacks and a gene only the graph has are both reported."""
    real = DatabaseManager.id_ver_from_df

    def other_ids(self, frame):
        ids = real(self, frame)
        if int(self.ensembl_release) == 101 and int(self.genome_assembly) == 38:
            ids = [i for i in ids if i != "ENSG00000000003.1"] + ["ENSG00000000009.1"]
        return ids

    monkeypatch.setattr(DatabaseManager, "id_ver_from_df", other_ids)
    result = synthetic_track_tests.is_id_functions_consistent_ensembl(verbose=False)
    assert result.holds is False and result.details["compared"] == 4 and result.details["n_differing"] == 1
    assert result.details["differing"] == [
        {
            "assembly": 38,
            "release": 101,
            "n_missing": 1,
            "missing": ["ENSG00000000009.1"],
            "n_extra": 1,
            "extra": ["ENSG00000000003.1"],
        }
    ]


def serve_external_table(monkeypatch, rows):
    """Make the manager's table of external ids hold the given ``(release, graph id, external id, database)`` rows.

    The rows are of GRCh38 (the world's GRCh37, listed beside it with the same genes, has no external ids).
    """
    real = DatabaseManager.get_db

    def get_db(self, df_indicator, *args, **kwargs):
        if df_indicator != "external_relevant":
            return real(self, df_indicator, *args, **kwargs)
        release = int(self.ensembl_release)
        frame = pd.DataFrame(
            [
                {"release": release, "graph_id": graph_id, "id_db": external, "name_db": database}
                for at, graph_id, external, database in rows
                if at == release and int(self.genome_assembly) == 38
            ],
            columns=["release", "graph_id", "id_db", "name_db"],
        )
        return frame.assign(ensembl_identity=np.nan, xref_identity=np.nan)

    monkeypatch.setattr(DatabaseManager, "get_db", get_db)


TP53_ROWS = [(100, "ENSG00000000001.1", "TP53", "HGNC"), (101, "ENSG00000000001.2", "TP53", "HGNC")]


def test_external_ids_of_the_graph_are_compared_with_the_tables(synthetic_track_tests, monkeypatch):
    """The check passes when graph and tables agree and fails when the graph lacks an id of the tables."""
    serve_external_table(monkeypatch, TP53_ROWS)
    agree = synthetic_track_tests.is_id_functions_consistent_external(verbose=False)
    assert agree.holds is True and agree.details == {"compared": 2, "n_differing": 0, "differing": []}
    # one node per identifier whatever its letter case: the tables may write it otherwise
    serve_external_table(monkeypatch, [(release, gene, "Tp53", database) for release, gene, _, database in TP53_ROWS])
    assert synthetic_track_tests.is_id_functions_consistent_external(verbose=False).holds is True

    serve_external_table(monkeypatch, TP53_ROWS + [(101, "ENSG00000000003.1", "KRAS", "HGNC")])
    lacking = synthetic_track_tests.is_id_functions_consistent_external(verbose=False)
    assert lacking.holds is False
    assert lacking.details["differing"] == [
        {
            "database": "HGNC",
            "assembly": 38,
            "release": 101,
            "n_missing": 1,
            "missing": ["kras"],
            "n_extra": 0,
            "extra": [],
        }
    ]
    # a database of the tables that the graph does not have at all is a difference too
    serve_external_table(monkeypatch, TP53_ROWS + [(100, "ENSG00000000002.1", "Q1", "UniProt")])
    other = synthetic_track_tests.is_id_functions_consistent_external(verbose=False)
    assert other.holds is False and other.details["differing"][0]["database"] == "UniProt"


def test_external_id_that_only_the_graph_has_is_reported(synthetic_track_tests, monkeypatch):
    """An external id in the graph that Ensembl's tables do not hold is an inconsistency."""
    serve_external_table(monkeypatch, [])  # the tables hold no id of HGNC; the graph has TP53 at both releases
    result = synthetic_track_tests.is_id_functions_consistent_external(verbose=False)
    assert result.holds is False and result.details["n_differing"] == 2
    assert [(d["release"], d["missing"], d["extra"]) for d in result.details["differing"]] == [
        (100, [], ["tp53"]),
        (101, [], ["tp53"]),
    ]
    # an identifier of the tables that is an Ensembl id of the graph is no external identifier there
    serve_external_table(monkeypatch, TP53_ROWS + [(101, "ENSG00000000003.1", "ENSG00000000001", "HGNC")])
    monkeypatch.setitem(synthetic_track_tests.graph.graph, "misplaced_external_entry", {"ENSG00000000001"})
    assert synthetic_track_tests.is_id_functions_consistent_external(verbose=False).holds is True


def test_final_conversion_check_passes_when_the_conversions_equal_the_tables(synthetic_track_tests, monkeypatch):
    """The check of gene-to-database conversions converts the genes of the release and compares with the tables."""
    serve_external_table(monkeypatch, TP53_ROWS)
    result = synthetic_track_tests.is_final_external_conversion_robust(
        database="HGNC", ens_rel=101, verbose=False, convert_using_release=True
    )
    assert result.holds is True and bool(result) is True
    details = result.details
    assert (details["database"], details["assembly"], details["release"], details["compared"]) == ("HGNC", 38, 101, 2)
    assert details["different"] == [] and details["unexpected"] == [] and details["failed"] == []
    assert details["run"]["conversion"] == {"ENSG00000000001.2": ["TP53"]}
    assert details["lost"] == ["ENSG00000000003.1"]  # the gene without an id in the database
    assert details["lost_with_id"] == []

    serve_external_table(monkeypatch, [(101, "ENSG00000000001.2", "KRAS", "HGNC")])  # the tables say otherwise
    result = synthetic_track_tests.is_final_external_conversion_robust(
        database="HGNC", ens_rel=101, verbose=False, convert_using_release=True
    )
    assert result.holds is False and not result
    assert result.details["different"] == [{"id": "ENSG00000000001.2", "converted": ["TP53"], "expected": ["KRAS"]}]

    serve_external_table(monkeypatch, [])  # the tables give the gene no id, and it is converted to one
    result = synthetic_track_tests.is_final_external_conversion_robust(
        database="HGNC", ens_rel=101, verbose=False, convert_using_release=True
    )
    assert result.holds is False
    assert result.details["unexpected"] == [{"id": "ENSG00000000001.2", "converted": ["TP53"]}]


def test_final_conversion_check_fails_when_a_gene_with_an_id_is_lost(synthetic_track_tests, monkeypatch):
    """A gene that the tables give an id in the database and that the converter loses is a failure of the check."""
    serve_external_table(monkeypatch, TP53_ROWS)
    monkeypatch.setattr(synthetic_track_tests, "convert", lambda **arguments: None, raising=False)
    result = synthetic_track_tests.is_final_external_conversion_robust(
        database="HGNC", ens_rel=101, verbose=False, convert_using_release=True
    )
    assert result.details["run"]["conversion"] == {}
    assert result.details["lost_with_id"] == ["ENSG00000000001.2"]  # the converter lost the gene of TP53
    assert result.details["lost"] == ["ENSG00000000001.2", "ENSG00000000003.1"]
    assert result.holds is False and not result

    def crashes(**arguments):
        raise RuntimeError("the converter crashed")

    monkeypatch.setattr(synthetic_track_tests, "convert", crashes, raising=False)
    result = synthetic_track_tests.is_final_external_conversion_robust(
        database="HGNC", ens_rel=101, verbose=False, convert_using_release=True
    )
    assert result.holds is False and result.details["failed"] == ["ENSG00000000001.2", "ENSG00000000003.1"]


def test_final_conversion_check_draws_a_database_of_gene_ids_at_a_release_it_has(synthetic_track_tests, monkeypatch):
    """Without a database or a release the check draws one that exists; it never tests an Ensembl node type."""
    serve_external_table(monkeypatch, TP53_ROWS)
    drawn = set()
    for seed in range(12):
        random.seed(seed)
        result = synthetic_track_tests.is_final_external_conversion_robust(verbose=False, convert_using_release=True)
        assert result.holds is True, result.details
        drawn.add((result.details["database"], result.details["release"]))
    assert drawn == {("HGNC", 100), ("HGNC", 101)}
    only = synthetic_track_tests.is_final_external_conversion_robust(database="HGNC", verbose=False)
    assert only.holds is True and only.details["database"] == "HGNC"
    at_100 = synthetic_track_tests.is_final_external_conversion_robust(ens_rel=100, verbose=False)
    assert at_100.holds is True and at_100.details["release"] == 100
