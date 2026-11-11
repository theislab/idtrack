#!/usr/bin/env python3
"""Unit tests for `idtrack._harmonize_features`.

These tests run the real harmonisation logic on tiny `.h5ad` files while patching
the expensive IDTrack mapping step to deterministic synthetic results.
"""

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from idtrack._db import DB
from idtrack._harmonize_features import HarmonizeFeatures


@dataclass(frozen=True)
class _SyntheticMatchings:
    dataset_to_matchings: dict[str, list[dict]]


def _make_matching(query_id: str, target_pairs: list[tuple[str, str]] | None, final_database: str | None) -> dict:
    target_pairs = [] if target_pairs is None else target_pairs
    return {
        "query_id": query_id,
        "target_id": sorted({j for _, j in target_pairs}),
        "last_node": target_pairs,
        "final_database": final_database,
        "graph_id": None,
        "no_corresponding": False,
        "no_conversion": final_database is None and len(target_pairs) == 0,
        "no_target": False,
    }


def _synthetic_matchings(final_db: str = "HGNC Symbol") -> _SyntheticMatchings:
    ensg1 = "ENSG00000000001"
    ensg2 = "ENSG00000000002"
    ensg3 = "ENSG00000000003"
    ensg_tp53 = "ENSG00000141510"

    d1 = [
        _make_matching("GENE_OK_1", [(ensg1, "G1")], final_db),
        _make_matching("GENE_OK_2", [(ensg2, "G2")], final_db),
        _make_matching("FAIL_ALL", None, None),
        _make_matching("FAIL_ONLY_D1", None, None),
        _make_matching("TP53", [(ensg_tp53, "TP53")], final_db),
        # n-to-1: maps to TP53 as well, should be retained unconverted by HarmonizeFeatures.
        _make_matching("TP53_ALIAS", [(ensg_tp53, "TP53")], final_db),
    ]
    d2 = [
        _make_matching("GENE_OK_1", [(ensg1, "G1")], final_db),
        _make_matching("GENE_OK_3", [(ensg3, "G3")], final_db),
        _make_matching("FAIL_ALL", None, None),
        _make_matching("FAIL_ONLY_D2", None, None),
        _make_matching("TP53", [(ensg_tp53, "TP53")], final_db),
        _make_matching("TP53_ALIAS", [(ensg_tp53, "TP53")], final_db),
    ]

    return _SyntheticMatchings(dataset_to_matchings={"d1": d1, "d2": d2})


def _make_harmonizer(
    tmp_path, monkeypatch, organism_name="homo_sapiens", final_database="HGNC Symbol"
) -> HarmonizeFeatures:
    """Create a HarmonizeFeatures instance backed by synthetic matchings."""
    expected_final = final_database or DB.symbol_databases[organism_name][0]
    synthetic = _synthetic_matchings(expected_final)

    d1_path = tmp_path / "d1.h5ad"
    d2_path = tmp_path / "d2.h5ad"

    d1_var_names = ["GENE_OK_1", "GENE_OK_2", "FAIL_ALL", "FAIL_ONLY_D1", "TP53", "TP53_ALIAS"]
    d2_var_names = ["GENE_OK_1", "GENE_OK_3", "FAIL_ALL", "FAIL_ONLY_D2", "TP53", "TP53_ALIAS"]

    def _adata(var_names: list[str], handle: str) -> ad.AnnData:
        x = csr_matrix(np.arange(12, dtype=np.float32).reshape(2, 6))
        cell_type: list[str | None] = ["T", "nan"] if handle == "d1" else ["B", None]
        obs = {
            "cell_type": cell_type,
            "age": ["10", "20"] if handle == "d1" else ["30", "40"],
        }
        return ad.AnnData(X=x, obs=pd.DataFrame(obs, index=["c0", "c1"]), var=pd.DataFrame(index=var_names))

    _adata(d1_var_names, "d1").write_h5ad(d1_path)
    _adata(d2_var_names, "d2").write_h5ad(d2_path)

    def _fake_get_idtrack_matchings_for_all_datasets(self: HarmonizeFeatures):
        result = {}
        for dataset_name in self.data_h5ad_dict:
            matching_list = synthetic.dataset_to_matchings[dataset_name]
            self.n_to_1_within_individual_dataset(dataset_name=dataset_name, dataset_matching_list=matching_list)
            result[dataset_name] = matching_list
        return result

    monkeypatch.setattr(
        HarmonizeFeatures, "get_idtrack_matchings_for_all_datasets", _fake_get_idtrack_matchings_for_all_datasets
    )

    project_repo = tmp_path / "project_repo"
    idtrack_repo = tmp_path / "idtrack_repo"
    project_repo.mkdir()
    idtrack_repo.mkdir()

    return HarmonizeFeatures(
        project_name="unit",
        data_h5ad_dict={"d1": str(d1_path), "d2": str(d2_path)},
        project_local_repository=str(project_repo),
        idtrack_local_repository=str(idtrack_repo),
        target_ensembl_release=101,
        final_database=final_database,
        organism_name=organism_name,
        graph_last_ensembl_release=101,
        verbose_level=0,
        debugging_variables=True,
        converted_id_column="converted_id",
    )


@pytest.fixture
def harmonizer(tmp_path, monkeypatch) -> HarmonizeFeatures:
    """Create a HarmonizeFeatures instance backed by synthetic matchings."""
    return _make_harmonizer(tmp_path, monkeypatch)


def test_import_harmonize_features():
    """Import smoke test for HarmonizeFeatures."""
    assert HarmonizeFeatures is not None


def test_initialization_builds_diagnostics(harmonizer):
    """Ensure HarmonizeFeatures computes diagnostic sets on init."""
    assert harmonizer.conversion_failed_identifiers == {"FAIL_ONLY_D1", "FAIL_ONLY_D2"}
    assert {"FAIL_ALL", "TP53_ALIAS"}.issubset(harmonizer.conversion_failed_but_consistent_identifiers)

    assert harmonizer.reporter_dict_creator_helper_reason_finder("FAIL_ONLY_D1") == "1-to-0"
    assert harmonizer.reporter_dict_creator_helper_reason_finder("TP53_ALIAS") == "n-to-1"

    assert "TP53" in harmonizer.dict_n_to_1_with_query
    assert ("TP53", "TP53_ALIAS") in harmonizer.dict_n_to_1_with_query["TP53"]
    assert set(harmonizer.dict_n_to_1["TP53_ALIAS"]) == {"d1", "d2"}


def test_feature_harmonizer_removes_inconsistent_failures(harmonizer):
    """Ensure inconsistent failures are removed from the feature set."""
    adata, t0, t1 = harmonizer.feature_harmonizer("d1")

    assert t0 == 6
    assert t1 == 5  # FAIL_ONLY_D1 removed
    assert "FAIL_ONLY_D1" not in adata.var["Query ID"].values
    assert "FAIL_ALL" in adata.var.index
    assert "TP53_ALIAS" in adata.var.index
    assert "TP53" in adata.var["Query ID"].values


def test_unify_multiple_anndatas_union_adds_intersection_flag(harmonizer):
    """Union mode should add an `intersection` column."""
    adata = harmonizer.unify_multiple_anndatas(
        mode="union",
        obs_columns_to_keep=["cell_type", "age"],
        numeric_obs_columns={"age"},
        handle_anndata_key="handle",
    )

    assert adata.n_obs == 4
    assert adata.obs["age"].dtype.kind == "f"
    assert adata.obs["handle"].dtype.name == "category"
    assert adata.obs_names.str.startswith(("d1_", "d2_")).all()

    assert set(adata.var.index) == {
        "ENSG00000000001",  # GENE_OK_1
        "ENSG00000000002",  # GENE_OK_2 (d1-only)
        "ENSG00000000003",  # GENE_OK_3 (d2-only)
        "ENSG00000141510",  # TP53
        "FAIL_ALL",  # kept consistent failure
        "TP53_ALIAS",  # kept to avoid n-to-1 collapse
    }

    assert "intersection" in adata.var.columns
    assert adata.var["intersection"].dtype.kind in {"i", "u"}
    intersection_ones = set(adata.var.index[adata.var["intersection"] == 1])
    assert {"ENSG00000000001", "ENSG00000141510", "FAIL_ALL", "TP53_ALIAS"}.issubset(intersection_ones)

    # Values are kept as they are: the text "nan" is a value, a missing value stays missing.
    assert adata.obs["cell_type"].astype(object).where(adata.obs["cell_type"].notna(), None).tolist() == [
        "T",
        "nan",
        "B",
        None,
    ]
    assert DB.placeholder_na not in set(adata.obs["cell_type"].cat.categories)


def test_unify_multiple_anndatas_intersect_keeps_shared_features(harmonizer):
    """Intersect mode should keep only shared features."""
    adata = harmonizer.unify_multiple_anndatas(
        mode="intersect",
        obs_columns_to_keep=["cell_type", "age"],
        numeric_obs_columns={"age"},
        handle_anndata_key="handle",
    )

    assert set(adata.var.index) == {
        "ENSG00000000001",
        "ENSG00000141510",
        "FAIL_ALL",
        "TP53_ALIAS",
    }
    assert "intersection" not in adata.var.columns


def test_get_column_as_series_handles_duplicates():
    """Ensure _get_column_as_series returns Series even with duplicate column names."""
    # Normal DataFrame with unique columns
    df_normal = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = HarmonizeFeatures._get_column_as_series(df_normal, "a")
    assert isinstance(result, pd.Series)
    assert list(result) == [1, 2, 3]

    # DataFrame with duplicate column names
    df_dup = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
    result_dup = HarmonizeFeatures._get_column_as_series(df_dup, "a")
    assert isinstance(result_dup, pd.Series)
    # Should return the first column named "a"
    assert list(result_dup) == [1, 4]

    # Test that KeyError is raised for non-existent column
    with pytest.raises(KeyError):
        HarmonizeFeatures._get_column_as_series(df_normal, "nonexistent")


@pytest.mark.parametrize(
    ("organism", "symbol_database"),
    [(organism, db) for organism, databases in sorted(DB.symbol_databases.items()) for db in databases],
)
def test_harmonizer_works_for_every_symbol_database(tmp_path, monkeypatch, organism, symbol_database):
    """The harmonisation runs the same way for each organism's symbol databases (formerly HGNC-only)."""
    h = _make_harmonizer(tmp_path, monkeypatch, organism_name=organism, final_database=symbol_database)
    assert h.final_database == symbol_database
    adata = h.unify_multiple_anndatas(
        mode="intersect",
        obs_columns_to_keep=["cell_type", "age"],
        numeric_obs_columns={"age"},
        handle_anndata_key="handle",
    )
    assert set(adata.var.index) == {"ENSG00000000001", "ENSG00000141510", "FAIL_ALL", "TP53_ALIAS"}


@pytest.mark.parametrize("organism", sorted(DB.symbol_databases))
def test_default_final_database_is_the_first_symbol_database(tmp_path, monkeypatch, organism):
    """Without `final_database`, the organism's first symbol database is used (no human default)."""
    h = _make_harmonizer(tmp_path, monkeypatch, organism_name=organism, final_database=None)
    assert h.final_database == DB.symbol_databases[organism][0]
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
        "graph_macaca_mulatta_min48_max114_narrow.pickle",
        "graph_macaca_mulatta_min48_max116_narrow.pickle",
        "graph_macaca_mulatta_min100_max116_narrow.pickle",  # not a graph build_graph would load
    ):
        (tmp_path / name).write_bytes(b"")
    h = _bare_harmonizer(tmp_path)
    h.organism_name = h._resolve_organism_name(None)
    assert h.organism_name == "macaca_mulatta"
    assert h._default_graph_last_release() == 116

    (tmp_path / "graph_homo_sapiens_min48_max115_narrow.pickle").write_bytes(b"")
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


def test_result_cache_key_includes_organism_database_and_releases(tmp_path):
    """Result cache key includes organism database and releases."""
    h = _bare_harmonizer(tmp_path)
    h.project_name, h.organism_name, h.final_database = "proj", "macaca_mulatta", "VGNC Symbol"
    h.target_ensembl_release, h.graph_last_ensembl_release = 110, 116
    path = h.result_cache_path("d1")
    assert path.endswith("idtrack_result_proj_d1_macaca_mulatta_VGNC-Symbol_rel110_graph116.pickle")
    h.final_database = "HGNC Symbol"
    assert h.result_cache_path("d1") != path


def _write_rich_datasets(tmp_path):
    """Two datasets like the ones of the harmonizer fixture, with layers, obsm, obsp, uns, raw and var columns."""
    from scipy.sparse import identity

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
def rich_harmonizer(tmp_path, monkeypatch) -> HarmonizeFeatures:
    """The harmonizer fixture on datasets with every AnnData element."""
    h = _make_harmonizer(tmp_path, monkeypatch)
    h.data_h5ad_dict = {k: str(v) for k, v in _write_rich_datasets(tmp_path).items()}
    return h


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
    assert DB.placeholder_na not in set(adata.obs["only_in_d1"].astype(str))
    # Integers and booleans become nullable (pd.NA), not floats or objects; categories stay categories.
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
    with pytest.raises(ValueError, match="`obs_join` must be"):
        rich_harmonizer.unify_multiple_anndatas(obs_join="left")
    with pytest.raises(ValueError, match="already an obs column"):
        rich_harmonizer.unify_multiple_anndatas(handle_anndata_key="disease")
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
    # feature_type agrees between the datasets: one column; chrom does not: one column per dataset.
    assert adata.var["feature_type"].astype(str).unique().tolist() == ["Gene Expression"]
    assert {"chrom_d1", "chrom_d2"} <= set(adata.var.columns) and "chrom" not in adata.var.columns
    # length agrees where both datasets have a feature: one column; missing (pd.NA, still integers) where none has it.
    assert str(adata.var["length"].dtype) == "Int64"
    assert adata.var.loc["ENSG00000000001", "length"] == 100 and adata.var["length"].notna().all()
    assert adata.var.loc["ENSG00000000002", "chrom_d2"] is np.nan or pd.isna(
        adata.var.loc["ENSG00000000002", "chrom_d2"]
    )
    assert adata.var.loc["ENSG00000000001", "converted_id"] == "G1"
    assert adata.var.loc["ENSG00000000002", "intersection"] == 0
    assert adata.var.loc["ENSG00000000001", "intersection"] == 1
    # A feature only in d2 has zeros for the d1 cells, in X and in the layer.
    col = adata.var_names.get_loc("ENSG00000000003")
    assert adata.X[:3, col].toarray().sum() == 0 and adata.layers["counts"][:3, col].toarray().sum() == 0


def test_feature_harmonizer_keeps_the_dataset_elements(rich_harmonizer):
    """feature_harmonizer only re-indexes the features; the input var columns follow the conversion columns."""
    adata, t0, t1 = rich_harmonizer.feature_harmonizer("d1")
    assert (t0, t1) == (6, 5)
    assert list(adata.var.columns) == ["HGNC Symbol", "Query ID", "feature_type", "chrom", "length"]
    assert adata.var.loc["ENSG00000000001", "chrom"] == "c0d1"
    assert (
        [k for k in adata.layers if k is not None] == ["counts"]
        and "X_pca" in adata.obsm
        and "connectivities" in adata.obsp
    )
    assert adata.uns["own"] == "d1"
    assert adata.raw is None


def test_unify_rejects_unknown_numeric_columns(rich_harmonizer):
    """A numeric column that the merged object does not have is named in the error."""
    with pytest.raises(ValueError, match=r"numeric_obs_columns.*\['nowhere'\]"):
        rich_harmonizer.unify_multiple_anndatas(numeric_obs_columns={"nowhere"})
    adata = rich_harmonizer.unify_multiple_anndatas(numeric_obs_columns={"n_counts"})
    assert adata.obs["n_counts"].dtype == float and adata.obs["n_counts"].isna().sum() == 2


def test_unify_var_conflicts_are_kept_per_dataset_and_clashes_raise(rich_harmonizer, tmp_path):
    """Different values for one feature keep the column per dataset; a generated name already used raises."""
    import anndata as ad

    path = rich_harmonizer.data_h5ad_dict["d2"]
    d2 = ad.read_h5ad(path)
    d2.var["length"] = d2.var["length"] + 1  # now d1 and d2 disagree on the shared features
    d2.var["intersection"] = "x"  # an input column with the name of a generated one
    d2.write_h5ad(path)
    with pytest.raises(ValueError, match="'intersection' twice"):
        rich_harmonizer.unify_multiple_anndatas(mode="union")
    adata = rich_harmonizer.unify_multiple_anndatas(mode="intersect")
    assert "length" not in adata.var.columns
    assert adata.var.loc["ENSG00000000001", "length_d1"] == 100
    assert adata.var.loc["ENSG00000000001", "length_d2"] == 101
