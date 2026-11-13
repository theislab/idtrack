#!/usr/bin/env python3
"""Unit tests for idtrack._graph_maker module.

These tests exercise real GraphMaker logic using tiny synthetic Ensembl tables to
avoid multi-GB downloads while still validating important graph-building
behaviour (time travel edges, births/retirements, external edges, caching).
"""

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
            "seq_region_id": [1, 1, 1],
        }
    )
    gene_r101 = pd.DataFrame(
        {
            "gene_id": [1, 3, 5],
            "stable_id": ["ENSG00000000001", "ENSG00000000003", "ENSG00000000005"],
            "version": [2, 1, 1],
            "seq_region_id": [1, 1, 1],
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
def synthetic_graph_dm(tmp_path, monkeypatch) -> DatabaseManager:
    """Latest-release DatabaseManager backed by synthetic MySQL tables."""
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


def test_construct_graph_form_builds_time_travel_edges(synthetic_graph_dm):
    """Construct a per-form graph and validate backbone edges."""
    gm = GraphMaker(synthetic_graph_dm)
    g = gm.construct_graph_form(narrow=False, db_manager=synthetic_graph_dm.change_form("gene"))

    assert isinstance(g, TheGraph)

    assert "ENSG00000000001.1" in g.nodes
    assert "ENSG00000000001.2" in g.nodes
    assert "ENSG00000000002.1" in g.nodes
    assert "ENSG00000000003.1" in g.nodes

    # ID-change edge comes from the stable_id_event row ENSG2 → ENSG3.
    assert g.has_edge("ENSG00000000002.1", "ENSG00000000003.1")
    edge_data = next(iter(g.get_edge_data("ENSG00000000002.1", "ENSG00000000003.1").values()))
    assert edge_data["old_release"] == 100
    assert edge_data["new_release"] == 101
    # narrow=False path: the session metadata is in the report of the history, not on every edge.
    (session,) = g.graph["history"]["sessions"]
    assert session["mapping_session_id"] == 1 and session["used"] == [100, 101]
    assert session["new_db_name"] == "homo_sapiens_core_101_38"

    # Version-change edge is created from df_w lookup (ENSG1.1 → ENSG1.2).
    assert g.has_edge("ENSG00000000001.1", "ENSG00000000001.2")

    # Birth/retirement scaffolding should exist due to ENSG5 (birth) and ENSG4 (retirement).
    assert any(data.get("Version") == DB.no_old_node_id for _, data in g.nodes(data=True))
    assert any(data.get("Version") == DB.no_new_node_id for _, data in g.nodes(data=True))

    # Ids of the latest release carry the self-loop to infinity; earlier versions do not.
    assert g.get_edge_data("ENSG00000000001.2", "ENSG00000000001.2")[0]["new_release"] == float("inf")
    assert not g.has_edge("ENSG00000000001.1", "ENSG00000000001.1")
    assert g.get_edge_data("ENSG00000000001.1", "ENSG00000000001.2")[0]["weight"] == 0.8


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

    def _fake_core_index(*, organism: str, genome_assembly: int, **_):
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

    # External IDs differing only by case are one node, named with the upper-case spelling here (both spellings
    # occur in the latest release with one link each).
    assert "TP53" in g.nodes and "tp53" not in g.nodes
    ext_node = "TP53"

    # External edges carry the connection dictionary and derived available releases.
    targets = [t for t in g.neighbors(ext_node) if t.startswith("ENSG")]
    assert targets, "Expected at least one external → Ensembl edge"
    target = targets[0]
    edge_data = next(iter(g.get_edge_data(ext_node, target).values()))
    assert DB.connection_dict in edge_data
    assert "available_releases" in edge_data
    assert 100 in edge_data["available_releases"] or 101 in edge_data["available_releases"]


def test_construct_graph_warns_once_about_releases_missing_from_externals_yaml(synthetic_graph_dm):
    """A release without an entry in the externals YAML is announced once and gets no external edges."""
    import warnings

    import yaml

    yaml_path = os.path.join(synthetic_graph_dm.local_repository, "homo_sapiens_externals_modified.yml")
    with open(yaml_path, "w") as handle:
        yaml.safe_dump(
            {
                "homo_sapiens": {
                    "gene": {
                        "HGNC": {
                            "Assembly": {"38": {"Ensembl release": "101", "Include": True}},
                            "Database Index": 0,
                            "Potential Synonymous": "",
                        }
                    }
                }
            },
            handle,
        )

    gm = GraphMaker(synthetic_graph_dm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = gm.construct_graph(narrow=True, form_list=["gene"], narrow_external=True)
    messages = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning) and "externals YAML" in str(w.message)
    ]
    assert len(messages) == 1
    assert "[100]" in messages[0] and "extend_modified_yaml" in messages[0]

    ext_node = "TP53" if "TP53" in g.nodes else "tp53"
    releases = set()
    for target in g.neighbors(ext_node):
        for data in g.get_edge_data(ext_node, target).values():
            releases |= set(data.get("available_releases", set()))
    assert releases == {101}


def test_construct_graph_without_externals_yaml_fails_before_building(synthetic_graph_dm, monkeypatch):
    """An organism without any externals YAML stops at once with a message naming the file to create."""
    from idtrack._external_databases import ExternalDatabases

    missing = os.path.join(synthetic_graph_dm.local_repository, "absent", "homo_sapiens_externals_modified.yml")
    monkeypatch.setattr(ExternalDatabases, "file_name_modified_yaml", lambda self, mode: missing)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("the history graph must not be built without an externals YAML")

    monkeypatch.setattr(GraphMaker, "history", _must_not_run)
    gm = GraphMaker(synthetic_graph_dm)
    with pytest.raises(FileNotFoundError, match="create_template_yaml"):
        gm.construct_graph(narrow=True, form_list=["gene"], narrow_external=True)


def test_get_graph_roundtrips_graph_file(synthetic_graph_dm):
    """Ensure get_graph writes the graph file and reads it back."""
    gm = GraphMaker(synthetic_graph_dm)

    file_path = gm.create_file_name(narrow=True, form_list=["gene"])
    g1 = gm.get_graph(
        narrow=True,
        create_even_if_exist=True,
        save_after_calculation=True,
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
    assert list(g1.nodes) == list(g2.nodes)
    assert all(repr(g1.get_edge_data(u, v)) == repr(g2.get_edge_data(u, v)) for u in g1.nodes for v in g1.neighbors(u))
    assert g2.manifest == g1.manifest

    with pytest.raises(FileNotFoundError):
        TheGraph.load(os.path.join(synthetic_graph_dm.local_repository, "does-not-exist.npz"))


def test_get_graph_without_saving_writes_nothing(synthetic_graph_dm):
    """save_after_calculation=False leaves the local repository without a graph file."""
    gm = GraphMaker(synthetic_graph_dm)
    gm.get_graph(narrow=True, save_after_calculation=False, form_list=["gene"])
    assert GraphMaker.list_graph_files(synthetic_graph_dm.local_repository) == []


def test_graph_maker_requires_latest_release(synthetic_graph_dm):
    """Ensure GraphMaker requires constructing from the latest release."""
    older = synthetic_graph_dm.change_release(100)
    with pytest.raises(ValueError):
        GraphMaker(older)


@pytest.mark.wrong_argument_type  # the manager is a stand-in that only carries the attributes the check reads
def test_graph_maker_explains_how_to_build_an_older_snapshot():
    """A manager anchored before the end of its window is refused with the call that builds that snapshot."""
    from types import SimpleNamespace

    from idtrack._graph_maker import GraphMaker

    dm = SimpleNamespace(ensembl_release=93, available_releases=[92, 93, 94])
    with pytest.raises(ValueError, match=r"newest release of the manager's window \(94\).*snapshot_release=93"):
        GraphMaker(dm)
    assert (
        GraphMaker(SimpleNamespace(ensembl_release=94, available_releases=[92, 93, 94])).db_manager.ensembl_release
        == 94
    )
