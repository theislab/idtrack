#!/usr/bin/env python3
"""
Unit tests for `idtrack._harmonize_features`.

These tests run the real harmonisation logic on tiny `.h5ad` files while patching
the expensive IDTrack mapping step to deterministic synthetic results.
"""

from __future__ import annotations

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


def _synthetic_matchings() -> _SyntheticMatchings:
    final_db = "HGNC Symbol"
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


@pytest.fixture
def harmonizer(tmp_path, monkeypatch) -> HarmonizeFeatures:
    synthetic = _synthetic_matchings()

    d1_path = tmp_path / "d1.h5ad"
    d2_path = tmp_path / "d2.h5ad"

    d1_var_names = ["GENE_OK_1", "GENE_OK_2", "FAIL_ALL", "FAIL_ONLY_D1", "TP53", "TP53_ALIAS"]
    d2_var_names = ["GENE_OK_1", "GENE_OK_3", "FAIL_ALL", "FAIL_ONLY_D2", "TP53", "TP53_ALIAS"]

    def _adata(var_names: list[str], handle: str) -> ad.AnnData:
        x = csr_matrix(np.arange(12, dtype=np.float32).reshape(2, 6))
        obs = {
            "cell_type": ["T", "nan"] if handle == "d1" else ["B", None],
            "age": ["10", "20"] if handle == "d1" else ["30", "40"],
        }
        return ad.AnnData(X=x, obs=pd.DataFrame(obs), var=pd.DataFrame(index=var_names))

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
        final_database="HGNC Symbol",
        organism_name="homo_sapiens",
        graph_last_ensembl_release=101,
        verbose_level=0,
        debugging_variables=True,
        converted_id_column="converted_id",
    )


def test_import_harmonize_features():
    assert HarmonizeFeatures is not None


def test_initialization_builds_diagnostics(harmonizer):
    assert harmonizer.conversion_failed_identifiers == {"FAIL_ONLY_D1", "FAIL_ONLY_D2"}
    assert {"FAIL_ALL", "TP53_ALIAS"}.issubset(harmonizer.conversion_failed_but_consistent_identifiers)

    assert harmonizer.reporter_dict_creator_helper_reason_finder("FAIL_ONLY_D1") == "1-to-0"
    assert harmonizer.reporter_dict_creator_helper_reason_finder("TP53_ALIAS") == "n-to-1"

    assert "TP53" in harmonizer.dict_n_to_1_with_query
    assert ("TP53", "TP53_ALIAS") in harmonizer.dict_n_to_1_with_query["TP53"]
    assert set(harmonizer.dict_n_to_1["TP53_ALIAS"]) == {"d1", "d2"}


def test_feature_harmonizer_removes_inconsistent_failures(harmonizer):
    adata, t0, t1 = harmonizer.feature_harmonizer("d1")

    assert t0 == 6
    assert t1 == 5  # FAIL_ONLY_D1 removed
    assert "FAIL_ONLY_D1" not in adata.var["Query ID"].values
    assert "FAIL_ALL" in adata.var.index
    assert "TP53_ALIAS" in adata.var.index
    assert "TP53" in adata.var["Query ID"].values


def test_unify_multiple_anndatas_union_adds_intersection_flag(harmonizer):
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

    assert DB.placeholder_na in set(adata.obs["cell_type"].cat.categories)


def test_unify_multiple_anndatas_intersect_keeps_shared_features(harmonizer):
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
