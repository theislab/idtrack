#!/usr/bin/env python3
"""High-value unit tests for `idtrack._track_tests.TrackTests`.

The goal is to exercise the developer-facing test harness on a tiny synthetic
graph (2 releases) without requiring large MySQL downloads.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._graph_maker import GraphMaker
from idtrack._track_tests import TrackTests


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
        }
    )
    gene_r101 = pd.DataFrame(
        {
            "gene_id": [1, 3],
            "stable_id": ["ENSG00000000001", "ENSG00000000003"],
            "version": [2, 1],
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

    tables_by_release = {
        100: {"gene": gene_r100, "stable_id_event": stable_id_event, "mapping_session": mapping_session},
        101: {"gene": gene_r101, "stable_id_event": stable_id_event, "mapping_session": mapping_session},
    }
    return _SyntheticGraphMySQL(tables_by_release=tables_by_release, available_releases=[100, 101])


@pytest.fixture
def synthetic_track_tests(tmp_path, monkeypatch) -> TrackTests:
    """Create a TrackTests instance backed by a tiny synthetic 2-release graph."""
    synthetic = _synthetic_graph_mysql()

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

    graph = GraphMaker(dm).get_graph(narrow=True, save_after_calculation=False, form_list=["gene"])
    graph.calculate_caches(for_test=True)

    tests = TrackTests.__new__(TrackTests)
    tests.log = logging.getLogger("track_tests")
    tests.db_manager = dm
    tests.graph = graph
    tests.version_info = graph.graph.get("version_info")
    tests._external_entrance_placeholder = {False: -1, True: 10001}
    tests._external_entrance_placeholders = sorted(tests._external_entrance_placeholder.values())
    return tests


def test_track_tests_basic_invariants(synthetic_track_tests):
    """Ensure basic TrackTests invariants hold for the synthetic graph."""
    assert synthetic_track_tests.is_node_consistency_robust(verbose=False)
    assert synthetic_track_tests.is_range_functions_robust(verbose=False)
    assert synthetic_track_tests.is_base_is_range_correct(verbose=False)
    assert synthetic_track_tests.is_combined_edges_dicts_overlapping_and_complete()
    assert synthetic_track_tests.is_edge_with_same_nts_only_at_backbone_nodes()


def test_track_tests_history_travel_testing_smoke(synthetic_track_tests):
    """Smoke-test history_travel_testing on the synthetic graph."""
    random.seed(0)
    graph = synthetic_track_tests.graph
    assembly = int(graph.graph["genome_assembly"])
    release = int(max(graph.graph["confident_for_release"]))

    database = sorted(graph.available_external_databases)[0]
    metrics = synthetic_track_tests.history_travel_testing(
        from_release=release,
        from_assembly=assembly,
        from_database=database,
        to_release=release,
        to_database=database,
        go_external=False,
        prioritize_to_one_filter=True,
        convert_using_release=True,
        from_fraction=1.0,
        verbose=False,
    )

    assert metrics["parameters"]["from_database"] == database
    assert metrics["ids"]["from"]  # at least one source ID
    assert "time" in metrics

    report = synthetic_track_tests._format_history_travel_testing_report(metrics, include_header=True)
    assert any("History-Travel-Testing" in line for line in report)


def test_track_tests_history_travel_testing_random_smoke(synthetic_track_tests):
    """Smoke-test history_travel_testing_random on the synthetic graph."""
    random.seed(0)
    res = synthetic_track_tests.history_travel_testing_random(
        from_fraction=1.0,
        include_ensembl_source=False,
        include_external_source=False,
        include_ensembl_destination=False,
        include_external_destination=True,
        verbose=False,
        strict_forward=True,
        convert_using_release=True,
        return_result=True,
    )
    assert isinstance(res, dict)
    assert "parameters" in res and "ids" in res
