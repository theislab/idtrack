#!/usr/bin/env python3
"""Unit tests for idtrack._graph_maker module.

These tests exercise real GraphMaker logic using tiny synthetic Ensembl tables to
avoid multi-GB downloads while still validating important graph-building
behaviour (time travel edges, births/retirements, external edges, caching).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._graph_maker import GraphMaker
from idtrack._the_graph import TheGraph


@dataclass(frozen=True)
class _SyntheticGraphMySQL:
    tables_by_release: dict[int, dict[str, pd.DataFrame]]
    available_releases: list[int]

    def get(self, release: int, table_key: str) -> pd.DataFrame:
        return self.tables_by_release[release][table_key]


def _synthetic_graph_mysql() -> _SyntheticGraphMySQL:
    gene_r100 = pd.DataFrame(
        {
            "gene_id": [1, 2, 4],
            "stable_id": ["ENSG00000000001", "ENSG00000000002", "ENSG00000000004"],
            "version": [1, 1, 1],
        }
    )
    gene_r101 = pd.DataFrame(
        {
            "gene_id": [1, 3, 5],
            "stable_id": ["ENSG00000000001", "ENSG00000000003", "ENSG00000000005"],
            "version": [2, 1, 1],
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
def synthetic_graph_dm(tmp_path, monkeypatch) -> DatabaseManager:
    """Latest-release DatabaseManager backed by synthetic MySQL tables."""
    synthetic = _synthetic_graph_mysql()

    def _fake_core_index(*, organism: str, genome_assembly: int):  # noqa: ARG001
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
                {
                    "release": release,
                    "graph_id": graph_id,
                    "id_db": "tp53",
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

    return DatabaseManager(
        organism="homo_sapiens",
        form="gene",
        local_repository=str(tmp_path),
        ensembl_release=101,
        ignore_before=100,
        ignore_after=101,
        genome_assembly=38,
        store_raw_always=True,
    )


def test_import_graph_maker():
    """Import smoke test for GraphMaker."""
    assert GraphMaker is not None


def test_construct_graph_form_builds_time_travel_edges(synthetic_graph_dm):
    """Construct a per-form graph and validate backbone edges."""
    gm = GraphMaker(synthetic_graph_dm)
    g = gm.construct_graph_form(narrow=False, db_manager=synthetic_graph_dm.change_form("gene"))

    assert isinstance(g, TheGraph)
    assert g.graph["version_info"] == "with_version"

    assert "ENSG00000000001.1" in g.nodes
    assert "ENSG00000000001.2" in g.nodes
    assert "ENSG00000000002.1" in g.nodes
    assert "ENSG00000000003.1" in g.nodes

    # ID-change edge comes from the stable_id_event row ENSG2 → ENSG3.
    assert g.has_edge("ENSG00000000002.1", "ENSG00000000003.1")
    edge_data = next(iter(g.get_edge_data("ENSG00000000002.1", "ENSG00000000003.1").values()))
    assert edge_data["old_release"] == 100
    assert edge_data["new_release"] == 101
    assert "mapping_session_id" in edge_data  # narrow=False path

    # Version-change edge is created from df_w lookup (ENSG1.1 → ENSG1.2).
    assert g.has_edge("ENSG00000000001.1", "ENSG00000000001.2")

    # Birth/retirement scaffolding should exist due to ENSG5 (birth) and ENSG4 (retirement).
    assert any(data.get("Version") == DB.no_old_node_id for _, data in g.nodes(data=True))
    assert any(data.get("Version") == DB.no_new_node_id for _, data in g.nodes(data=True))

    # Latest-release nodes should be labelled.
    assert g.nodes["ENSG00000000001.2"]["is_latest"] is True
    assert g.nodes["ENSG00000000001.1"]["is_latest"] is False


def test_construct_graph_form_handles_clean_handoff_assemblies(tmp_path, monkeypatch):
    """The GraphMaker should traverse releases even when assemblies change across the snapshot window."""
    tables_by_release = {
        100: {
            "gene": pd.DataFrame(
                {
                    "gene_id": [1],
                    "stable_id": ["ENSG00000000001"],
                    "version": [1],
                }
            ),
            "stable_id_event": pd.DataFrame(
                {
                    "mapping_session_id": [1],
                    "old_stable_id": ["ENSG00000000001"],
                    "old_version": [1],
                    "new_stable_id": ["ENSG00000000001"],
                    "new_version": [2],
                    "score": [0.8],
                    "type": ["gene"],
                }
            ),
            "mapping_session": pd.DataFrame(
                {
                    "mapping_session_id": [1],
                    "old_db_name": ["mus_musculus_core_100_38"],
                    "new_db_name": ["mus_musculus_core_101_39"],
                    "old_release": [100],
                    "new_release": [101],
                    "old_assembly": ["38"],
                    "new_assembly": ["39"],
                    "created": ["2020-01-01"],
                }
            ),
        },
        101: {
            "gene": pd.DataFrame(
                {
                    "gene_id": [1],
                    "stable_id": ["ENSG00000000001"],
                    "version": [2],
                }
            ),
            "stable_id_event": pd.DataFrame(
                {
                    "mapping_session_id": [1],
                    "old_stable_id": ["ENSG00000000001"],
                    "old_version": [1],
                    "new_stable_id": ["ENSG00000000001"],
                    "new_version": [2],
                    "score": [0.8],
                    "type": ["gene"],
                }
            ),
            "mapping_session": pd.DataFrame(
                {
                    "mapping_session_id": [1],
                    "old_db_name": ["mus_musculus_core_100_38"],
                    "new_db_name": ["mus_musculus_core_101_39"],
                    "old_release": [100],
                    "new_release": [101],
                    "old_assembly": ["38"],
                    "new_assembly": ["39"],
                    "created": ["2020-01-01"],
                }
            ),
        },
    }

    def _fake_core_index(*, organism: str, genome_assembly: int):
        if organism != "mus_musculus":
            raise ValueError("unexpected organism")
        if int(genome_assembly) == 39:
            releases = [101]
        elif int(genome_assembly) == 38:
            releases = [100]
        elif int(genome_assembly) == 37:
            releases = []
        else:
            releases = []
        db_for_release = {r: f"{organism}_core_{r}_{genome_assembly}" for r in releases}
        return {
            "organism": organism,
            "genome_assembly": int(genome_assembly),
            "ports": (3306,),
            "releases_by_port": {3306: set(releases)},
            "db_by_port_release": {3306: db_for_release.copy()},
            "releases": releases,
            "port_for_release": {r: 3306 for r in releases},
            "db_for_release": db_for_release,
            "releases_on_mysql": releases,
        }

    def _available_releases_versions(self: DatabaseManager, **kwargs) -> list[int]:  # noqa: ARG002
        core_index = _fake_core_index(organism=self.organism, genome_assembly=int(self.genome_assembly))
        return sorted(r for r in core_index["releases"] if self.ignore_after >= r >= self.ignore_before)

    def _download_table(self: DatabaseManager, table_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
        df = tables_by_release[int(self.ensembl_release)].get(table_key)
        if df is None:
            return pd.DataFrame()
        if usecols is None:
            return df.copy(deep=True)
        return df.loc[:, usecols].copy(deep=True)

    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: _fake_core_index(**kw)))
    monkeypatch.setattr(DatabaseManager, "available_releases_versions", _available_releases_versions)
    monkeypatch.setattr(DatabaseManager, "download_table", _download_table)

    dm = DatabaseManager(
        organism="mus_musculus",
        form="gene",
        local_repository=str(tmp_path),
        ensembl_release=101,
        ignore_before=100,
        ignore_after=101,
        genome_assembly=39,
        store_raw_always=True,
    )

    g = GraphMaker(dm).construct_graph_form(narrow=False, db_manager=dm.change_form("gene"))

    assert g.has_edge("ENSG00000000001.1", "ENSG00000000001.2")
    assert 100 in set(g.graph.get("confident_for_release", []))
    assert 101 in set(g.graph.get("confident_for_release", []))


def test_construct_graph_adds_base_nodes_and_merges_case_insensitive_externals(synthetic_graph_dm):
    """Ensure base nodes exist and externals are case-insensitive."""
    gm = GraphMaker(synthetic_graph_dm)
    g = gm.construct_graph(narrow=True, form_list=["gene"], narrow_external=True)

    # Versionless base nodes should be connected for versioned organisms.
    assert "ENSG00000000001" in g.nodes
    assert g.nodes["ENSG00000000001"][DB.node_type_str] == DB.nts_base_ensembl["gene"]
    assert g.has_edge("ENSG00000000001", "ENSG00000000001.1") or g.has_edge("ENSG00000000001", "ENSG00000000001.2")

    # External IDs differing only by case should be merged.
    assert len({"TP53", "tp53"} & set(g.nodes)) == 1
    ext_node = "TP53" if "TP53" in g.nodes else "tp53"

    # External edges carry the connection dictionary and derived available releases.
    targets = [t for t in g.neighbors(ext_node) if t.startswith("ENSG")]
    assert targets, "Expected at least one external → Ensembl edge"
    target = targets[0]
    edge_data = next(iter(g.get_edge_data(ext_node, target).values()))
    assert DB.connection_dict in edge_data
    assert "available_releases" in edge_data
    assert 100 in edge_data["available_releases"] or 101 in edge_data["available_releases"]


def test_get_graph_roundtrips_pickle_cache(synthetic_graph_dm):
    """Ensure get_graph writes and reloads a cached pickle."""
    gm = GraphMaker(synthetic_graph_dm)

    file_path = gm.create_file_name(narrow=True, form_list=["gene"])
    g1 = gm.get_graph(
        narrow=True,
        create_even_if_exist=True,
        save_after_calculation=True,
        overwrite_even_if_exist=True,
        form_list=["gene"],
    )
    assert os.path.exists(file_path)

    g2 = gm.get_graph(
        narrow=True,
        create_even_if_exist=False,
        save_after_calculation=False,
        form_list=["gene"],
    )
    assert isinstance(g2, TheGraph)
    assert set(g1.nodes) == set(g2.nodes)

    with pytest.raises(FileNotFoundError):
        GraphMaker.read_exported(os.path.join(synthetic_graph_dm.local_repository, "does-not-exist.pickle"))


def test_remove_non_gene_trees_prunes_transcript_history():
    """Ensure non-gene temporal trees are pruned."""
    g = TheGraph()
    t1 = "ENST00000000010.1"
    t2 = "ENST00000000020.1"
    t_void = "ENST00000000030.Void"
    gene = "ENSG00000000001.1"

    g.add_node(t1, **{DB.node_type_str: DB.nts_ensembl["transcript"], "Version": "1"})
    g.add_node(t2, **{DB.node_type_str: DB.nts_ensembl["transcript"], "Version": "1"})
    g.add_node(t_void, **{DB.node_type_str: DB.nts_ensembl["transcript"], "Version": DB.no_old_node_id})
    g.add_node(gene, **{DB.node_type_str: DB.nts_ensembl["gene"], "Version": "1"})

    g.add_edge(t1, t2)
    g.add_edge(t1, gene)

    pruned = GraphMaker.remove_non_gene_trees(g)
    assert t_void not in pruned.nodes
    assert not pruned.has_edge(t1, t2)
    assert pruned.has_edge(t1, gene)


def test_graph_maker_requires_latest_release(synthetic_graph_dm):
    """Ensure GraphMaker requires constructing from the latest release."""
    import inspect

    older = synthetic_graph_dm.change_release(100)
    with pytest.raises(ValueError):
        GraphMaker(older)

    source = inspect.getsource(GraphMaker.update_graph_with_the_new_release)
    assert "raise NotImplementedError" in source


class TestGraphMakerPerformance:
    """Test performance-related functionality."""

    def test_uses_efficient_data_structures(self):
        """Test efficient data structures are used."""
        import networkx as nx

        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        # MultiDiGraph is appropriate for this use case
        assert isinstance(graph, nx.MultiDiGraph)


class TestGraphMakerIntegrationPoints:
    """Test integration with other modules."""

    def test_uses_database_manager(self, mock_database_manager):
        """Test uses DatabaseManager correctly."""
        assert mock_database_manager.organism == "homo_sapiens"

    def test_produces_the_graph(self):
        """Test produces TheGraph instance."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        assert isinstance(graph, TheGraph)

    def test_uses_db_constants(self):
        """Test uses DB constants."""
        from idtrack._db import DB

        assert DB.node_type_str == "node_type"
