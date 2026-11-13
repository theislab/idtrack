#!/usr/bin/env python3
"""Unit tests for `idtrack._harmonize_features`.

The real API and harmoniser run on tiny `.h5ad` files; only the engine (`Track.convert`) is replaced by the table of
`tests/unit/_fake_engine.py`, so the conversions are known and no graph or network is needed.
"""

import json
import os
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, identity

import idtrack
from idtrack._db import DB
from idtrack._harmonize_features import HarmonizeFeatures, _aggregate, feature_table, output_plan
from idtrack._track import InactiveAtReleaseError, SearchBudgetExceededError
from tests.unit._fake_engine import fake_api

HGNC = "HGNC Symbol"


def _q(gene: str, *names: str) -> dict:
    """A query answered by one gene with these HGNC names."""
    return {gene: {HGNC: list(names)} if names else {}}


QUERIES = {
    "TP53": _q("ENSG_TP53.1", "TP53"),
    "P53": _q("ENSG_TP53.1", "P53"),  # the engine prefers a target spelled like the query
    "GENEA": _q("ENSGA.1", "GENEA"),
    "ENSG00000000007": _q("ENSG00000000007.16", "GENE7"),
    "OLDID": _q("ENSG00000000007.16", "GENE7"),
    "AMB": {"ENSG01.1": {HGNC: ["X"]}, "ENSG02.1": {HGNC: ["Y"]}},
    "OLDNAME": _q("ENSG01.1", "X"),
    "ENSG_OLD_A": _q("ENSG_NEW.1", "NEW"),
    "ENSG_OLD_B": _q("ENSG_NEW.1", "NEW"),
    "FAM231C": _q("ENSG_F.2", "FAM231C"),
    "FAM231B": _q("ENSG_F.2", "FAM231B"),
    "ALIAS1": _q("GENE_A.1", "SYM"),
    "ALIAS2": _q("GENE_B.1", "SYM"),
    "ALIAS3": _q("GENE_B.1", "SYM"),
    "NT": _q("ENSG_NT.1"),
    "OLD": _q("ENSG_REAL.1", "AAA1"),
    "NOCONV": {},
    "ERR": _q("ENSG_ERR.1", "E"),
    # the genes of the old fixture
    "GENE_OK_1": _q("ENSG00000000001", "G1"),
    "GENE_OK_2": _q("ENSG00000000002", "G2"),
    "GENE_OK_3": _q("ENSG00000000003", "G3"),
    "TP53_ALIAS": _q("ENSG00000141510", "TP53"),
    "TP53B": _q("ENSG00000141510", "TP53"),
}
GENE_NAMES = {
    "ENSG_TP53.1": ["TP53"],
    "ENSGA.1": ["GENEA"],
    "ENSG00000000007.16": ["GENE7"],
    "ENSG01.1": ["X"],
    "ENSG02.1": ["Y"],
    "ENSG_NEW.1": ["NEW"],
    "ENSG_F.2": ["FAM231C", "FAM231B"],
    "GENE_A.1": ["SYM"],
    "GENE_B.1": ["SYM"],
    "ENSG_NT.1": [],
    "ENSG_REAL.1": ["AAA1"],
    "ENSG_ERR.1": ["E"],
    "ENSG00000000001": ["G1"],
    "ENSG00000000002": ["G2"],
    "ENSG00000000003": ["G3"],
    "ENSG00000141510": ["TP53"],
}
ANSWERS = {**QUERIES, **{g: _q(g, *n) for g, n in GENE_NAMES.items()}}


def _api(tmp_path, **kwargs):
    return fake_api(tmp_path / "idtrack_repo", ANSWERS, **kwargs)


def _write(tmp_path, name, var_names, values=None, dtype=np.float32, layer=True) -> str:
    x = np.asarray(values if values is not None else np.arange(2 * len(var_names)).reshape(2, -1) + 1, dtype=dtype)
    adata = ad.AnnData(
        X=csr_matrix(x),
        obs=pd.DataFrame({"cell_type": ["T", "B"][: x.shape[0]]}, index=[f"c{i}" for i in range(x.shape[0])]),
        var=pd.DataFrame(index=var_names),
    )
    if layer:
        adata.layers["counts"] = csr_matrix(x * 2)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name.replace('/', '_')}.h5ad"
    adata.write_h5ad(path)
    return str(path)


def _harmonizer(tmp_path, datasets, api=None, **kwargs) -> HarmonizeFeatures:
    paths = {
        n: _write(tmp_path, n, *v) if isinstance(v, tuple) else _write(tmp_path, n, v) for n, v in datasets.items()
    }
    return HarmonizeFeatures(
        project_name="unit",
        data_h5ad_dict=paths,
        project_local_repository=str(tmp_path / "project"),
        idtrack_local_repository=str(tmp_path / "idtrack_repo"),
        target_ensembl_release=kwargs.pop("target_ensembl_release", 101),
        final_database=kwargs.pop("final_database", HGNC),
        api=api if api is not None else _api(tmp_path),
        verbose_level=0,
        **kwargs,
    )


def _column(adata, feature, dataset=None):
    """Values of one output feature, for the cells of one dataset (or all)."""
    x = adata[:, feature].X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    if dataset is not None:
        x = x[(adata.obs["handle_anndata"] == dataset).to_numpy()]
    return x.ravel().tolist()


def test_a_dataset_is_harmonized_the_same_alone_and_with_others(tmp_path):
    """api_harmonize-16: n-to-1 in one dataset is resolved there and never removes the feature from another one."""
    joint = _harmonizer(tmp_path, {"d1": ["TP53", "P53", "GENEA"], "d2": ["P53", "GENEA"]})
    alone = _harmonizer(tmp_path / "alone", {"d2": ["P53", "GENEA"]})
    a, _, _ = joint.feature_harmonizer("d2")
    b, _, _ = alone.feature_harmonizer("d2")
    assert a.var_names.tolist() == b.var_names.tolist() == ["ENSG_TP53.1", "ENSGA.1"]
    assert (a.X != b.X).nnz == 0
    merged = joint.unify_multiple_anndatas()
    assert _column(merged, "ENSG_TP53.1", "d2") == [1.0, 3.0]  # d2's P53, measured
    assert _column(merged, "ENSG_TP53.1", "d1") == [1.0, 4.0]  # d1's TP53, the gene itself
    assert merged.var.loc["P53", "idtrack_status_d1"] == "unconverted_merge_partner"
    assert merged.var.loc["ENSG_TP53.1", "intersection"] == 1


def test_the_gene_itself_is_converted_when_another_id_merges_into_it(tmp_path):
    """api_harmonize-17: identity is decided on resolved ids (version and case ignored), not on raw strings."""
    h = _harmonizer(tmp_path, {"d": ["ENSG00000000007", "OLDID", "tp53", "P53"]})
    adata, _, _ = h.feature_harmonizer("d")
    assert adata.var_names.tolist()[:2] == ["ENSG00000000007.16", "OLDID"]
    assert adata.var.loc["ENSG00000000007.16", "Query ID"] == "ENSG00000000007"
    assert adata.var.loc["OLDID", "idtrack_status"] == "unconverted_merge_partner"
    assert adata.var.loc["OLDID", "matching_ensembl_gene"] == "ENSG00000000007.16"
    assert adata.var.loc["ENSG_TP53.1", "Query ID"] == "tp53"  # the symbol in another case is the gene itself
    assert adata.var.loc["P53", "idtrack_status"] == "unconverted_merge_partner"  # an alias is not


def test_an_ambiguous_query_does_not_block_a_clean_one(tmp_path):
    """api_harmonize-18: only queries with one gene take part in the n-to-1 grouping."""
    adata, _, _ = _harmonizer(tmp_path, {"d": ["AMB", "OLDNAME"]}).feature_harmonizer("d")
    assert adata.var_names.tolist() == ["AMB", "ENSG01.1"]
    assert adata.var.loc["ENSG01.1", "idtrack_status"] == "converted"
    assert adata.var.loc["AMB", "idtrack_status"] == "unconverted_ambiguous"
    assert adata.var.loc["AMB", "idtrack_reason"] == "maps to 2 genes: ENSG01.1;ENSG02.1"


def test_merged_features_are_summed_on_request_and_flagged_otherwise(tmp_path):
    """api_harmonize-19: with merge="sum" the gene's values are the sum of its input features, in X and layers."""
    values = np.array([[1, 2, 5], [3, 4, 7]])
    h = _harmonizer(tmp_path, {"d": (["ENSG_OLD_A", "ENSG_OLD_B", "GENEA"], values, np.int32)})
    summed, t0, t1 = h.feature_harmonizer("d", merge="sum")
    assert (t0, t1) == (3, 2)
    assert summed.var_names.tolist() == ["ENSG_NEW.1", "ENSGA.1"]
    assert summed.X.dtype == np.int32
    assert summed.X.toarray().tolist() == [[3, 5], [7, 7]]
    assert summed.layers["counts"].toarray().tolist() == [[6, 10], [14, 14]]
    assert summed.var.loc["ENSG_NEW.1", "idtrack_status"] == "merged"
    assert summed.var.loc["ENSG_NEW.1", "Query ID"] == "ENSG_OLD_A;ENSG_OLD_B"
    assert summed.var.loc["ENSG_NEW.1", HGNC] == "NEW"
    flagged, _, _ = h.feature_harmonizer("d")
    assert flagged.var_names.tolist() == ["ENSG_OLD_A", "ENSG_OLD_B", "ENSGA.1"]
    assert set(flagged.var["idtrack_status"][:2]) == {"unconverted_merge_partner"}
    assert flagged.var[HGNC][:2].isna().all()  # an unconverted row never carries a name in the final-database column
    plan = h.output_plan("d", merge="sum")
    dense = _aggregate(values.astype(np.float32), plan, 3)
    assert isinstance(dense, np.ndarray) and dense.dtype == np.float32 and dense.tolist() == [[3, 5], [7, 7]]
    assert h.unify_multiple_anndatas(merge="sum").X.toarray().tolist() == [[3, 5], [7, 7]]


def test_a_name_reached_through_two_genes_keeps_every_dataset(tmp_path):
    """api_harmonize-20: nothing is deleted for sharing a name; each gene is its own feature."""
    h = _harmonizer(
        tmp_path,
        {"d1": (["ALIAS1"], [[10]]), "d2": (["ALIAS2"], [[20]]), "d3": (["ALIAS3"], [[30]])},
    )
    merged = h.unify_multiple_anndatas()
    assert merged.var_names.tolist() == ["GENE_A.1", "GENE_B.1"]
    assert merged.X.toarray().tolist() == [[10, 0], [0, 20], [0, 30]]
    assert merged.var["converted_id"].tolist() == ["SYM", "SYM"]


def test_a_gene_with_two_names_gets_one_name_and_keeps_its_data(tmp_path):
    """api_harmonize-21: the name is the gene's (first in sorted order), whatever spelling a dataset used."""
    h = _harmonizer(tmp_path, {"d1": (["FAM231C"], [[1]]), "d2": (["FAM231B"], [[2]])})
    merged = h.unify_multiple_anndatas()
    assert merged.var_names.tolist() == ["ENSG_F.2"]
    assert merged.X.toarray().ravel().tolist() == [1, 2]
    assert merged.var.loc["ENSG_F.2", "converted_id"] == "FAM231B"
    assert merged.var.loc["ENSG_F.2", f"{HGNC}_alternatives"] == "FAM231C"
    assert merged.var.loc["ENSG_F.2", "Query ID_d1"] == "FAM231C"


def test_unconverted_ids_never_compete_with_converted_ones(tmp_path):
    """api_harmonize-22: a kept unconverted id spelled like another query's target removes nothing."""
    adata, _, _ = _harmonizer(tmp_path, {"d": ["AAA1", "OLD"]}).feature_harmonizer("d")
    assert adata.var_names.tolist() == ["AAA1", "ENSG_REAL.1"]
    assert adata.var.loc["ENSG_REAL.1", HGNC] == "AAA1"
    assert adata.var.loc["AAA1", "idtrack_status"] == "unconverted_not_in_graph"


def test_a_gene_without_final_id_is_flagged_and_its_name_is_missing(tmp_path):
    """api_harmonize-new-1: the final-database column never holds an Ensembl id; the status says why it is empty."""
    h = _harmonizer(tmp_path, {"d": ["NT", "GENEA"]})
    merged = h.unify_multiple_anndatas()
    assert merged.var_names.tolist() == ["ENSG_NT.1", "ENSGA.1"]
    assert pd.isna(merged.var.loc["ENSG_NT.1", "converted_id"])
    assert merged.var.loc["ENSG_NT.1", "idtrack_status_d"] == "converted_no_final_id"
    assert not merged.var["converted_id"].astype(str).str.startswith("ENS").any()


def test_every_feature_is_converted_or_flagged_with_a_reason(tmp_path):
    """api_harmonize-19, -new-2: no feature disappears; every unconverted one has a status and a reason."""
    names = ["TP53", "P53", "AMB", "NOCONV", "AAA1", "NT", "OLDNAME", "ERR"]
    over_budget = SearchBudgetExceededError("ERR", 200_001, 1.5, 200_000, 30.0)
    h = _harmonizer(tmp_path, {"d": names}, api=_api(tmp_path, errors={"ERR": over_budget}))
    adata, t0, t1 = h.feature_harmonizer("d")
    assert t0 == t1 == len(names)
    assert sorted(adata.var["Query ID"]) == sorted(names)
    unconverted = adata.var[adata.var["idtrack_status"].astype(str).str.startswith("unconverted")]
    assert unconverted["idtrack_reason"].notna().all() and (unconverted["idtrack_reason"] != "").all()
    assert adata.var.loc["ERR", "idtrack_reason"] == f"SearchBudgetExceededError: {over_budget}"
    assert adata.var.loc["NOCONV", "idtrack_status"] == "unconverted_no_conversion"
    dropped, _, kept = h.feature_harmonizer("d", keep_unconverted=False)
    assert set(dropped.var["idtrack_status"]) <= {"converted", "converted_no_final_id", "unconverted_merge_partner"}


def test_the_source_release_of_a_dataset_is_used(tmp_path):
    """api_harmonize-23: a declared release is passed as from_release; an id the graph lacks there is flagged."""
    api = _api(tmp_path, errors={("OLDNAME", 100): InactiveAtReleaseError("OLDNAME", 100, [[101, 101]])})
    h = _harmonizer(tmp_path, {"d": ["GENEA", "OLDNAME"], "e": ["OLDNAME"]}, api=api, source_releases={"d": 100})
    queries = {(c["from_id"], c["from_release"]) for c in api.track.calls}
    assert {("GENEA", 100), ("OLDNAME", 100), ("OLDNAME", None)} <= queries
    # the feature is converted from the release at which the graph holds it, and the row says so
    assert h.feature_table("d").loc["OLDNAME", "status"] == "converted"
    assert h.feature_table("d").loc["OLDNAME", "reason"] == "not in the graph at release 100; read at release 101"
    assert pd.isna(h.feature_table("d").loc["GENEA", "reason"])
    assert h.feature_table("e").loc["OLDNAME", "status"] == "converted"
    assert pd.isna(h.feature_table("e").loc["OLDNAME", "reason"])
    with pytest.raises(ValueError, match="not in `data_h5ad_dict`"):
        _harmonizer(tmp_path / "x", {"d": ["GENEA"]}, source_releases={"zz": 100})


def test_the_source_database_of_a_dataset_is_used(tmp_path):
    """Decision 33: a declared database is passed as from_database for that dataset only; nothing is inferred."""
    api = _api(tmp_path)
    h = _harmonizer(tmp_path, {"d": ["GENEA", "TP53"], "e": ["GENEA"]}, api=api, source_databases={"d": "hgnc symbol"})
    asked = {(c["from_id"], c["from_database"]) for c in api.track.calls}
    assert asked == {("GENEA", HGNC), ("TP53", HGNC), ("GENEA", None)}  # the API's spelling of the database
    assert h.source_databases == {"d": "hgnc symbol"}
    assert h.feature_table("d")["status"].tolist() == ["converted", "converted"]

    # the declared database is part of what a stored result depends on
    again = _api(tmp_path)
    _harmonizer(tmp_path, {"d": ["GENEA", "TP53"], "e": ["GENEA"]}, api=again, source_databases={"d": "hgnc symbol"})
    assert again.track.calls == []
    other = _api(tmp_path)
    _harmonizer(tmp_path, {"d": ["GENEA", "TP53"], "e": ["GENEA"]}, api=other)
    assert {(c["from_id"], c["from_database"]) for c in other.track.calls} == {("GENEA", None), ("TP53", None)}

    with pytest.raises(ValueError, match="`source_databases` names datasets that are not in `data_h5ad_dict`"):
        _harmonizer(tmp_path / "x", {"d": ["GENEA"]}, source_databases={"zz": HGNC})
    with pytest.raises(ValueError, match="must name one external database per dataset"):
        _harmonizer(tmp_path / "y", {"d": ["GENEA"]}, source_databases={"d": " "})
    with pytest.raises(ValueError, match="`from_database`='No Such Database' is not an external database"):
        _harmonizer(tmp_path / "z", {"d": ["GENEA"]}, source_databases={"d": "No Such Database"})


@pytest.mark.wrong_argument_type
def test_the_source_database_of_a_dataset_must_be_a_name(tmp_path):
    """A source database that is not a string is refused before any conversion."""
    api = _api(tmp_path)
    with pytest.raises(ValueError, match="must name one external database per dataset"):
        _harmonizer(tmp_path, {"d": ["GENEA"]}, api=api, source_databases={"d": 7})
    assert api.track.calls == []


def test_repeated_feature_names_fail_before_any_conversion(tmp_path):
    """api_harmonize-24: the constructor names the dataset and the repeated ids."""
    api = _api(tmp_path)
    path = tmp_path / "dup.h5ad"
    with pytest.warns(UserWarning, match="not unique"):
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), var=pd.DataFrame(index=["G1", "G1", "G2"]))
    adata.write_h5ad(path)
    with pytest.raises(ValueError, match=r"Dataset 'dup' has 1 repeated feature names.*\['G1'\]"):
        HarmonizeFeatures("p", {"dup": str(path)}, str(tmp_path / "p"), str(tmp_path), 101, api=api, verbose_level=0)
    assert api.track.calls == []


def test_arguments_are_checked_before_any_conversion(tmp_path):
    """api_harmonize-4, -34: releases and the dataset dict are validated first."""
    api = _api(tmp_path)
    with pytest.raises(ValueError, match="after the graph snapshot 101"):
        _harmonizer(tmp_path, {"d": ["GENEA"]}, api=api, target_ensembl_release=102)
    with pytest.raises(ValueError, match="valid releases are 100 to 101"):
        _harmonizer(tmp_path, {"d": ["GENEA"]}, api=api, target_ensembl_release=50)
    assert api.track.calls == []
    with pytest.raises(ValueError, match="non-empty dict"):
        HarmonizeFeatures("p", {}, str(tmp_path), str(tmp_path), 101, api=api)
    with pytest.raises(ValueError, match="do not exist"):
        HarmonizeFeatures("p", {"d": str(tmp_path / "none.h5ad")}, str(tmp_path), str(tmp_path), 101, api=api)


@pytest.mark.wrong_argument_type
def test_unknown_merge_mode_is_an_error_naming_the_argument(tmp_path):
    """api_harmonize-34: a merge mode other than "flag" and "sum" is refused."""
    with pytest.raises(ValueError, match="`merge` must be"):
        _harmonizer(tmp_path, {"d": ["GENEA"]}).unify_multiple_anndatas(merge="mean")


def test_result_cache_is_json_written_atomically_and_checked(tmp_path, monkeypatch):
    """api_harmonize-36: a damaged or stale cache is recomputed; names with '/' work; the folder is created."""
    datasets = {"lung/2021": ["GENEA", "TP53"]}
    first = _harmonizer(tmp_path, datasets)
    path = first.result_cache_path("lung/2021")
    assert os.path.dirname(path) == str(tmp_path / "project") and path.endswith(".json")
    stored = json.loads(pathlib.Path(path).read_text())
    assert stored["header"]["idtrack_version"] == idtrack.__version__
    assert [m["query_id"] for m in stored["matchings"]] == ["GENEA", "TP53"]
    api = _api(tmp_path)
    again = _harmonizer(tmp_path, datasets, api=api)
    assert api.track.calls == [] and again.matchings == first.matchings
    pathlib.Path(path).write_text(pathlib.Path(path).read_text()[:20])  # an interrupted writer of the old code
    api = _api(tmp_path)
    repaired = _harmonizer(tmp_path, datasets, api=api)
    assert len(api.track.calls) > 0 and repaired.matchings == first.matchings
    api = _api(tmp_path)
    api.track.graph.graph["confident_for_release"] = [99, 100, 101]  # another graph
    _harmonizer(tmp_path, datasets, api=api)
    assert len(api.track.calls) > 0
    assert not [f for f in os.listdir(tmp_path / "project") if f.startswith(".tmp_")]


def test_a_given_api_is_used_and_no_graph_is_kept(tmp_path, monkeypatch):
    """api_harmonize-37: with an API no graph is loaded; without one, nothing is loaded when all is cached."""
    (tmp_path / "idtrack_repo").mkdir()
    (tmp_path / "idtrack_repo" / "graph_homo_sapiens_assembly38_min48_max101_narrow.npz").write_bytes(b"graph")
    _harmonizer(tmp_path, {"d": ["GENEA"]})

    def _fail(*args, **kwargs):
        raise AssertionError("build_graph must not be called")

    monkeypatch.setattr(idtrack.API, "build_graph", _fail)
    h = HarmonizeFeatures(
        "unit",
        {"d": str(tmp_path / "d.h5ad")},
        str(tmp_path / "project"),
        str(tmp_path / "idtrack_repo"),
        101,
        final_database=HGNC,
        organism_name="homo_sapiens",
        graph_last_ensembl_release=101,
        verbose_level=0,
    )
    assert h.feature_table("d").loc["GENEA", "ensembl_gene"] == "ENSGA.1" and h._api is None


def test_reports_are_pure_and_repeatable(tmp_path):
    """api_harmonize-38: reading the results twice changes nothing."""
    h = _harmonizer(tmp_path, {"d1": ["TP53", "P53"], "d2": ["P53"]})
    before = h.conversion_report()
    copy = h.feature_table("d1")
    copy.iloc[0, 0] = "edited"
    pd.testing.assert_frame_equal(before, h.conversion_report())
    assert before["dataset"].tolist() == ["d1", "d1", "d2"]
    assert h.output_plan("d1").equals(h.output_plan("d1"))


def test_feature_names_are_read_without_loading_the_datasets(tmp_path, monkeypatch):
    """api_harmonize-33: the constructor reads only the var index."""
    paths = {"d": _write(tmp_path, "d", ["GENEA", "TP53"])}

    def _fail(*args, **kwargs):
        raise AssertionError("read_h5ad must not be called by the constructor")

    monkeypatch.setattr(ad, "read_h5ad", _fail)
    h = HarmonizeFeatures("p", paths, str(tmp_path / "p"), str(tmp_path), 101, api=_api(tmp_path), verbose_level=0)
    assert h.feature_names == {"d": ["GENEA", "TP53"]}


def test_the_package_never_forces_a_garbage_collection():
    """api_harmonize-30: no explicit gc.collect (about 10 s each with a graph loaded)."""
    source = pathlib.Path(idtrack.__file__).parent
    assert not [p.name for p in source.rglob("*.py") if "gc.collect(" in p.read_text()]


def test_the_harmonizer_leaves_logging_configuration_alone(tmp_path):
    """api_harmonize-9: constructing a harmonizer changes neither the root nor the package logger."""
    import logging

    root, package = logging.getLogger(), logging.getLogger("idtrack")
    before = (root.level, list(root.handlers), package.level, list(package.handlers))
    HarmonizeFeatures(
        "p", {"d": _write(tmp_path, "d", ["GENEA"])}, str(tmp_path / "p"), str(tmp_path), 101, api=_api(tmp_path)
    )
    assert (root.level, list(root.handlers), package.level, list(package.handlers)) == before


def test_pure_functions_classify_and_plan(tmp_path):
    """feature_table and output_plan are functions of one dataset's matchings only."""
    api = _api(tmp_path)
    names = ["ENSG_OLD_A", "ENSG_OLD_B", "AMB"]
    matchings = api.convert_identifier_multiple(names, final_database=HGNC, verbose=False)
    table = feature_table(names, matchings, {"ENSG_NEW.1": ["NEW"]})
    assert table["status"].tolist() == ["converted", "converted", "ambiguous"]
    plan = output_plan(table, "sum", keep_unconverted=False)
    assert plan.index.tolist() == ["ENSG_NEW.1"] and plan["sources"].tolist() == [[0, 1]]


# ---------------------------------------------------------------------------------------------------------------------
# The fixture of the earlier tests: obs, var, layers and the other AnnData elements (issue #85).


@pytest.fixture
def harmonizer(tmp_path) -> HarmonizeFeatures:
    """Two datasets of six features: shared, own, unconvertible ones, and an alias of TP53."""
    d1 = ["GENE_OK_1", "GENE_OK_2", "FAIL_ALL", "FAIL_ONLY_D1", "TP53", "TP53_ALIAS"]
    d2 = ["GENE_OK_1", "GENE_OK_3", "FAIL_ALL", "FAIL_ONLY_D2", "TP53", "TP53_ALIAS"]
    answers = {**ANSWERS, "TP53": _q("ENSG00000141510", "TP53")}
    api = fake_api(tmp_path / "idtrack_repo", answers)
    paths = {}
    for handle, var_names in (("d1", d1), ("d2", d2)):
        x = csr_matrix(np.arange(12, dtype=np.float32).reshape(2, 6))
        obs = {
            "cell_type": ["T", "nan"] if handle == "d1" else ["B", None],
            "age": ["10", "20"] if handle == "d1" else ["30", "40"],
        }
        paths[handle] = str(tmp_path / f"{handle}.h5ad")
        ad.AnnData(X=x, obs=pd.DataFrame(obs, index=["c0", "c1"]), var=pd.DataFrame(index=var_names)).write_h5ad(
            paths[handle]
        )
    return HarmonizeFeatures(
        project_name="unit",
        data_h5ad_dict=paths,
        project_local_repository=str(tmp_path / "project_repo"),
        idtrack_local_repository=str(tmp_path / "idtrack_repo"),
        target_ensembl_release=101,
        final_database=HGNC,
        api=api,
        verbose_level=0,
    )


UNION = {
    "ENSG00000000001",
    "ENSG00000000002",
    "ENSG00000000003",
    "ENSG00000141510",
    "FAIL_ALL",
    "FAIL_ONLY_D1",
    "FAIL_ONLY_D2",
    "TP53_ALIAS",
}


def test_unify_union_keeps_every_feature_with_its_status(harmonizer):
    """Union mode keeps converted and (flagged) unconverted features; intersection is by presence."""
    adata = harmonizer.unify_multiple_anndatas(
        mode="union", obs_columns_to_keep=["cell_type", "age"], numeric_obs_columns={"age"}, handle_anndata_key="handle"
    )
    assert adata.n_obs == 4 and adata.obs["age"].dtype.kind == "f"
    assert adata.obs["handle"].dtype.name == "category"
    assert set(adata.var.index) == UNION
    assert adata.var["intersection"].dtype.kind in {"i", "u"}
    assert set(adata.var.index[adata.var["intersection"] == 1]) == {
        "ENSG00000000001",
        "ENSG00000141510",
        "FAIL_ALL",
        "TP53_ALIAS",
    }
    assert adata.var.loc["FAIL_ALL", "idtrack_status_d1"] == "unconverted_not_in_graph"
    assert pd.isna(adata.var.loc["FAIL_ALL", "converted_id"])
    assert adata.var.loc["TP53_ALIAS", "idtrack_status_d2"] == "unconverted_merge_partner"
    assert adata.obs["cell_type"].astype(object).where(adata.obs["cell_type"].notna(), None).tolist() == [
        "T",
        "nan",
        "B",
        None,
    ]
    assert not any("PLACEHOLDER" in str(category) for category in adata.obs["cell_type"].cat.categories)


def test_unify_intersect_keeps_shared_features(harmonizer):
    """Intersect mode keeps only shared features."""
    adata = harmonizer.unify_multiple_anndatas(mode="intersect")
    assert set(adata.var.index) == {"ENSG00000000001", "ENSG00000141510", "FAIL_ALL", "TP53_ALIAS"}
    assert "intersection" not in adata.var.columns


@pytest.mark.parametrize(
    ("organism", "symbol_database"),
    [(organism, db) for organism, databases in sorted(DB.symbol_databases.items()) for db in databases],
)
def test_harmonizer_works_for_every_symbol_database(tmp_path, organism, symbol_database):
    """The harmonisation runs the same way for each organism's symbol databases."""
    answers = {"GENE": {"ENSX.1": {symbol_database: ["SYMB"]}}, "ENSX.1": {"ENSX.1": {symbol_database: ["SYMB"]}}}
    api = fake_api(tmp_path, answers, databases={symbol_database}, organism=organism)
    paths = {"d": _write(tmp_path, "d", ["GENE"])}
    h = HarmonizeFeatures(
        "p", paths, str(tmp_path / "p"), str(tmp_path), 101, symbol_database, api=api, verbose_level=0
    )
    assert h.organism_name == organism
    adata = h.unify_multiple_anndatas(mode="intersect")
    assert adata.var_names.tolist() == ["ENSX.1"] and adata.var["converted_id"].tolist() == ["SYMB"]


@pytest.mark.parametrize("organism", sorted(DB.symbol_databases))
def test_default_final_database_is_the_first_symbol_database(organism):
    """Without `final_database`, the organism's first symbol database is used (no human default)."""
    assert HarmonizeFeatures.default_final_database(organism) == DB.symbol_databases[organism][0]
    assert HarmonizeFeatures.default_final_database("macaca_mulatta") == "VGNC Symbol"
    assert HarmonizeFeatures.default_final_database("mus_musculus") == "MGI Symbol"


def _bare_harmonizer(idtrack_repo, project_repo=None) -> HarmonizeFeatures:
    h = HarmonizeFeatures.__new__(HarmonizeFeatures)
    h.idtrack_local_repository = str(idtrack_repo)
    h.project_local_repository = str(project_repo or idtrack_repo)
    return h


def test_organism_and_snapshot_resolved_from_graph_files(tmp_path):
    """Without `organism_name`/`graph_last_ensembl_release`, the graphs `API.build_graph` made are used."""
    for name in (
        "graph_macaca_mulatta_assembly10_min48_max114_narrow.npz",
        "graph_macaca_mulatta_assembly10_min48_max116_narrow.npz",
        # not graphs `build_graph` would load: another window, the earlier assembly as primary, gene ids only,
        # every external database, the full history, a graph file of an earlier idtrack
        "graph_macaca_mulatta_assembly10_min100_max117_narrow.npz",
        "graph_macaca_mulatta_assembly801_min48_max118_narrow.npz",
        "graph_macaca_mulatta_assembly10_min48_max119_narrow_formsgene.npz",
        "graph_macaca_mulatta_assembly10_min48_max120_narrow_extall.npz",
        "graph_macaca_mulatta_assembly10_min48_max121.npz",
        "graph_macaca_mulatta_min48_max122_narrow.pickle",
    ):
        (tmp_path / name).write_bytes(b"")
    h = _bare_harmonizer(tmp_path)
    h.organism_name = h._resolve_organism_name(None)
    assert h.organism_name == "macaca_mulatta"
    assert h._default_graph_last_release() == 116

    (tmp_path / "graph_macaca_mulatta_assembly801_min48_max90_narrow.npz").write_bytes(b"")  # Mmul_8.0.1 at 90
    assert _bare_harmonizer(tmp_path)._graph_snapshots() == {"macaca_mulatta": [90, 114, 116]}
    (tmp_path / "graph_homo_sapiens_assembly38_min48_max115_narrow.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="Pass `organism_name`.*several organisms.*homo_sapiens, macaca_mulatta"):
        _bare_harmonizer(tmp_path)._resolve_organism_name(None)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match='Pass `organism_name`.*holds no graph.*organism_name="homo_sapiens"'):
        _bare_harmonizer(empty)._resolve_organism_name(None)


def test_organism_without_symbol_database_needs_final_database():
    """Organism without symbol database needs final database."""
    with pytest.raises(ValueError, match="pass `final_database`"):
        HarmonizeFeatures.default_final_database("danio_rerio")


def test_result_cache_header_holds_organism_database_and_releases(tmp_path):
    """The cache is only reused for the same organism, final database, releases and graph."""
    h = _harmonizer(tmp_path, {"d1": ["GENEA"]})
    header = json.loads(pathlib.Path(h.result_cache_path("d1")).read_text())["header"]
    assert header["organism"] == "homo_sapiens" and header["final_database"] == HGNC
    assert header["target_ensembl_release"] == 101 and header["graph_snapshot"] == 101
    assert header["source_release"] is None
    api = _api(tmp_path)
    _harmonizer(tmp_path, {"d1": ["GENEA"]}, api=api, target_ensembl_release=100)
    assert len(api.track.calls) > 0


def _write_rich_datasets(tmp_path):
    """Two datasets like the ones of the harmonizer fixture, with layers, obsm, obsp, uns, raw and var columns."""
    names = {
        "d1": ["GENE_OK_1", "GENE_OK_2", "FAIL_ALL", "FAIL_ONLY_D1", "TP53", "TP53_ALIAS"],
        "d2": ["GENE_OK_1", "GENE_OK_3", "FAIL_ALL", "FAIL_ONLY_D2", "TP53", "TP53_ALIAS"],
    }
    paths = {}
    for i, (handle, var_names) in enumerate(names.items()):
        x = csr_matrix(np.arange(18, dtype=np.float32).reshape(3, 6) + 100 * i)
        obs = pd.DataFrame(
            {
                "disease": ["normal"] * 3 if handle == "d1" else ["disease"] * 3,
                "study": [f"study_{handle}"] * 3,
                "n_counts": np.array([1.0, 2.0, np.nan]) + i,
                "only_in_d1": ["a", "b", "c"],
                "n_genes_d1": np.array([5, 6, 7], dtype=np.int64),
                "is_doublet_d1": [True, False, True],
                "batch_d1": pd.Categorical(["x", "y", "x"]),
            },
            index=[f"cell{handle.upper()}_{k}" for k in range(3)],
        )
        if handle == "d2":
            obs = obs.drop(columns=["only_in_d1", "n_genes_d1", "is_doublet_d1", "batch_d1"])
        var = pd.DataFrame(
            {
                "feature_type": ["Gene Expression"] * 6,
                "chrom": [f"c{k}{handle}" for k in range(6)],
                "length": np.arange(6, dtype=np.int64) * 10 + 100,
            },
            index=var_names,
        )
        adata = ad.AnnData(X=x, obs=obs, var=var)
        adata.layers["counts"] = x.copy() * 2
        adata.obsm["X_pca"] = np.ones((3, 2), dtype=np.float32) * (i + 1)
        adata.obsp["connectivities"] = identity(3, format="csr", dtype=np.float32)
        adata.uns["shared"] = "same"
        adata.uns["own"] = handle
        adata.raw = adata.copy()
        paths[handle] = tmp_path / f"{handle}_rich.h5ad"
        adata.write_h5ad(paths[handle])
    return paths


@pytest.fixture
def rich_harmonizer(tmp_path) -> HarmonizeFeatures:
    """The harmonizer fixture on datasets with every AnnData element."""
    paths = {k: str(v) for k, v in _write_rich_datasets(tmp_path).items()}
    answers = {**ANSWERS, "TP53": _q("ENSG00000141510", "TP53")}
    return HarmonizeFeatures(
        project_name="unit",
        data_h5ad_dict=paths,
        project_local_repository=str(tmp_path / "project_repo"),
        idtrack_local_repository=str(tmp_path / "idtrack_repo"),
        target_ensembl_release=101,
        final_database=HGNC,
        api=fake_api(tmp_path / "idtrack_repo", answers),
        verbose_level=0,
    )


def test_unify_keeps_every_obs_column_by_default(rich_harmonizer):
    """With the default `obs_join="outer"` no obs column is lost; absent ones are missing values of their kind."""
    adata = rich_harmonizer.unify_multiple_anndatas(mode="union")
    assert list(adata.obs.columns) == [
        "disease",
        "study",
        "n_counts",
        "only_in_d1",
        "n_genes_d1",
        "is_doublet_d1",
        "batch_d1",
        "handle_anndata",
    ]
    d2 = adata.obs["handle_anndata"] == "d2"
    for column in ("only_in_d1", "n_genes_d1", "is_doublet_d1", "batch_d1"):
        assert adata.obs.loc[d2, column].isna().all() and adata.obs.loc[~d2, column].notna().all()
    assert not adata.obs["only_in_d1"].astype(str).str.contains("PLACEHOLDER").any()
    assert str(adata.obs["n_genes_d1"].dtype) == "Int64" and adata.obs["n_genes_d1"].iloc[:3].tolist() == [5, 6, 7]
    assert str(adata.obs["is_doublet_d1"].dtype) == "boolean"
    assert adata.obs["is_doublet_d1"].iloc[:3].tolist() == [True, False, True]
    assert isinstance(adata.obs["batch_d1"].dtype, pd.CategoricalDtype)
    assert adata.obs["batch_d1"].iloc[:3].tolist() == ["x", "y", "x"]


def test_unify_obs_join_inner_and_selection(rich_harmonizer):
    """`obs_join="inner"` keeps the common columns; `obs_columns_to_keep` selects and names wrong names."""
    adata = rich_harmonizer.unify_multiple_anndatas(mode="intersect", obs_join="inner")
    assert list(adata.obs.columns) == ["disease", "study", "n_counts", "handle_anndata"]
    assert adata.obs["disease"].tolist() == ["normal"] * 3 + ["disease"] * 3
    assert adata.obs["n_counts"].dtype.kind == "f"
    assert np.isnan(adata.obs["n_counts"].iloc[2]) and adata.obs["n_counts"].iloc[3] == 2.0
    assert adata.obs["handle_anndata"].cat.categories.tolist() == ["d1", "d2"]
    assert adata.obs_names.tolist()[:2] == ["d1_cellD1_0", "d1_cellD1_1"]

    adata = rich_harmonizer.unify_multiple_anndatas(mode="union", obs_columns_to_keep=["only_in_d1", "disease"])
    assert list(adata.obs.columns) == ["only_in_d1", "disease", "handle_anndata"]
    assert adata.obs["only_in_d1"].isna().tolist() == [False] * 3 + [True] * 3
    with pytest.raises(ValueError, match=r"no dataset has: \['nowhere'\]"):
        rich_harmonizer.unify_multiple_anndatas(obs_columns_to_keep=["nowhere"])
    with pytest.raises(ValueError, match=r"not every dataset has: \['only_in_d1'\]"):
        rich_harmonizer.unify_multiple_anndatas(obs_join="inner", obs_columns_to_keep=["only_in_d1"])
    with pytest.raises(ValueError, match="already an obs column"):
        rich_harmonizer.unify_multiple_anndatas(handle_anndata_key="disease")


@pytest.mark.wrong_argument_type
def test_unify_names_an_unknown_join_or_mode(rich_harmonizer):
    """A join other than "outer" and "inner", or a mode other than "union" and "intersect", is refused by name."""
    with pytest.raises(ValueError, match="`obs_join` must be"):
        rich_harmonizer.unify_multiple_anndatas(obs_join="left")
    with pytest.raises(ValueError, match="`mode` must be"):
        rich_harmonizer.unify_multiple_anndatas(mode="inner")


def test_unify_keeps_layers_obsm_obsp_uns_and_var_columns(rich_harmonizer):
    """Everything aligned to cells is carried; var columns are merged; raw is dropped; X keeps its dtype."""
    adata = rich_harmonizer.unify_multiple_anndatas(mode="union")
    assert adata.X.dtype == np.float32
    assert [k for k in adata.layers if k is not None] == ["counts"]  # anndata 0.13 also lists X under None
    np.testing.assert_array_equal(adata.layers["counts"].toarray(), 2 * adata.X.toarray())
    assert adata.obsm["X_pca"].shape == (6, 2)
    assert adata.obsp["connectivities"].shape == (6, 6)
    assert adata.uns["shared"] == "same" and "own" not in adata.uns
    assert adata.raw is None
    assert adata.var["feature_type"].astype(str).unique().tolist() == ["Gene Expression"]
    assert {"chrom_d1", "chrom_d2"} <= set(adata.var.columns) and "chrom" not in adata.var.columns
    assert str(adata.var["length"].dtype) == "Int64"
    assert adata.var.loc["ENSG00000000001", "length"] == 100 and adata.var["length"].notna().all()
    assert pd.isna(adata.var.loc["ENSG00000000002", "chrom_d2"])
    assert adata.var.loc["ENSG00000000001", "converted_id"] == "G1"
    assert adata.var.loc["ENSG00000000002", "intersection"] == 0
    assert adata.var.loc["ENSG00000000001", "intersection"] == 1
    col = adata.var_names.get_loc("ENSG00000000003")
    assert adata.X[:3, col].toarray().sum() == 0 and adata.layers["counts"][:3, col].toarray().sum() == 0


def test_feature_harmonizer_keeps_the_dataset_elements(rich_harmonizer):
    """feature_harmonizer only re-indexes the features; the input var columns follow the conversion columns."""
    adata, t0, t1 = rich_harmonizer.feature_harmonizer("d1")
    assert (t0, t1) == (6, 6)
    assert list(adata.var.columns) == [
        HGNC,
        f"{HGNC}_alternatives",
        "Query ID",
        "matching_ensembl_gene",
        "idtrack_status",
        "idtrack_reason",
        "feature_type",
        "chrom",
        "length",
    ]
    assert adata.var.loc["ENSG00000000001", "chrom"] == "c0d1"
    assert (
        [k for k in adata.layers if k is not None] == ["counts"]
        and "X_pca" in adata.obsm
        and "connectivities" in adata.obsp
    )
    assert adata.uns["own"] == "d1"
    assert adata.raw is None
    summed, _, n = rich_harmonizer.feature_harmonizer("d1", merge="sum")
    assert n == 5 and summed.var.loc["ENSG00000141510", "Query ID"] == "TP53;TP53_ALIAS"
    assert summed.var.loc["ENSG00000141510", "feature_type"] == "Gene Expression"  # the merged inputs agree
    assert pd.isna(summed.var.loc["ENSG00000141510", "chrom"])  # they do not


def test_unify_rejects_unknown_numeric_columns(rich_harmonizer):
    """A numeric column that the merged object does not have is named in the error."""
    with pytest.raises(ValueError, match=r"numeric_obs_columns.*\['nowhere'\]"):
        rich_harmonizer.unify_multiple_anndatas(numeric_obs_columns={"nowhere"})
    adata = rich_harmonizer.unify_multiple_anndatas(numeric_obs_columns={"n_counts"})
    assert adata.obs["n_counts"].dtype == float and adata.obs["n_counts"].isna().sum() == 2


def test_unify_var_conflicts_are_kept_per_dataset_and_clashes_raise(rich_harmonizer):
    """Different values for one feature keep the column per dataset; a generated name already used raises."""
    path = rich_harmonizer.data_h5ad_dict["d2"]
    d2 = ad.read_h5ad(path)
    d2.var["length"] = d2.var["length"] + 1
    d2.var["intersection"] = "x"
    d2.write_h5ad(path)
    with pytest.raises(ValueError, match="'intersection' twice"):
        rich_harmonizer.unify_multiple_anndatas(mode="union")
    adata = rich_harmonizer.unify_multiple_anndatas(mode="intersect")
    assert "length" not in adata.var.columns
    assert adata.var.loc["ENSG00000000001", "length_d1"] == 100
    assert adata.var.loc["ENSG00000000001", "length_d2"] == 101


def test_unconverted_name_equal_to_a_converted_gene_never_shares_its_row(tmp_path):
    """An input kept unconverted in one dataset that is another dataset's converted gene id gets its own row."""
    answers = {**ANSWERS, "ENSG01.1": {"ENSG01.1": {HGNC: ["X"]}, "ENSG02.1": {HGNC: ["Y"]}}}
    api = fake_api(tmp_path / "idtrack_repo", answers)
    h = _harmonizer(tmp_path, {"a": (["OLDNAME"], [[1]]), "b": (["ENSG01.1"], [[2]])}, api=api)
    merged = h.unify_multiple_anndatas()
    assert merged.var_names.tolist() == ["ENSG01.1", "unconverted:ENSG01.1"]
    assert merged.X.toarray().tolist() == [[1, 0], [0, 2]]
