#!/usr/bin/env python3
"""
Unit tests for idtrack._database_manager module.

These tests avoid heavy MySQL downloads by patching `DatabaseManager.download_table`
to return tiny synthetic tables, while still exercising the real caching and
data-processing logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest


from idtrack._database_manager import DatabaseManager
from idtrack._db import DB


@dataclass(frozen=True)
class _SyntheticMySQL:
    tables_by_release: dict[int, dict[str, pd.DataFrame]]
    available_releases: list[int]


def _synthetic_mysql_tables() -> _SyntheticMySQL:
    def gene_table(with_versions: bool) -> pd.DataFrame:
        versions: list[Any] = [1, 1] if with_versions else [np.nan, np.nan]
        return pd.DataFrame(
            {
                "gene_id": [1, 2],
                "stable_id": ["ENSG00000000001", "ENSG00000000002"],
                "version": versions,
            }
        )

    transcript = pd.DataFrame(
        {
            "transcript_id": [10, 20],
            "gene_id": [1, 2],
            "stable_id": ["ENST00000000010", "ENST00000000020"],
            "version": [1, 1],
        }
    )
    translation = pd.DataFrame(
        {
            "translation_id": [100],
            "transcript_id": [10],
            "stable_id": ["ENSP00000000100"],
            "version": [1],
        }
    )

    stable_id_event = pd.DataFrame(
        {
            "mapping_session_id": [1, 1, 2],
            "old_stable_id": [np.nan, "ENSG00000000001", "ENSG00000000002"],
            "old_version": [np.nan, 1, 1],
            "new_stable_id": ["ENSG00000000001", "ENSG00000000001", "ENSG00000000002"],
            "new_version": [1, 2, 1],
            "score": [0.0, 0.5, 0.0],
            "type": ["gene", "gene", "gene"],
        }
    )
    mapping_session = pd.DataFrame(
        {
            "mapping_session_id": [1, 2],
            "old_db_name": ["homo_sapiens_core_100_38", "homo_sapiens_core_100_38"],
            "new_db_name": ["homo_sapiens_core_101_38", "homo_sapiens_core_101_38"],
            "old_release": [100, 100],
            "new_release": [101, 101],
            "old_assembly": ["38", "38"],
            "new_assembly": ["38", "38"],
            "created": ["2020-01-01", "2020-01-01"],
        }
    )

    object_xref = pd.DataFrame(
        {
            "ensembl_id": [1, 2],
            "ensembl_object_type": ["Gene", "Gene"],
            "xref_id": [1000, 1001],
            "object_xref_id": [5000, 5001],
        }
    )
    xref = pd.DataFrame(
        {
            "xref_id": [1000, 1001],
            "external_db_id": [1, 2],
            "dbprimary_acc": ["P12345", "7157"],
            "display_label": ["TP53", "BRCA1"],
        }
    )
    external_db = pd.DataFrame(
        {
            "external_db_id": [1, 2],
            "db_name": ["UniProtKB/Swiss-Prot", "EntrezGene"],
            "db_display_name": ["UniProt", "NCBI Gene"],
        }
    )
    identity_xref = pd.DataFrame(
        {
            "ensembl_identity": [100.0, 100.0],
            "xref_identity": [99.0, 98.0],
            "object_xref_id": [5000, 5001],
        }
    )
    external_synonym = pd.DataFrame(
        {
            "xref_id": [1000],
            "synonym": ["TRP53"],
        }
    )

    base_tables = {
        "transcript": transcript,
        "translation": translation,
        "stable_id_event": stable_id_event,
        "mapping_session": mapping_session,
        "object_xref": object_xref,
        "xref": xref,
        "external_db": external_db,
        "identity_xref": identity_xref,
        "external_synonym": external_synonym,
    }

    tables_by_release = {
        100: {"gene": gene_table(with_versions=True), **base_tables},
        101: {"gene": gene_table(with_versions=False), **base_tables},
    }
    return _SyntheticMySQL(tables_by_release=tables_by_release, available_releases=[100, 101])


@pytest.fixture
def synthetic_dm(tmp_path, monkeypatch) -> DatabaseManager:
    """DatabaseManager using tiny synthetic MySQL tables (no real MySQL access)."""
    synthetic = _synthetic_mysql_tables()

    def _available_releases_versions(self: DatabaseManager, **kwargs) -> list[int]:
        return list(synthetic.available_releases)

    def _download_table(self: DatabaseManager, table_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
        try:
            df = synthetic.tables_by_release[int(self.ensembl_release)][table_key]
        except KeyError:
            return pd.DataFrame()
        if usecols is None:
            return df.copy(deep=True)
        return df.loc[:, usecols].copy(deep=True)

    monkeypatch.setattr(DatabaseManager, "available_releases_versions", _available_releases_versions)
    monkeypatch.setattr(DatabaseManager, "download_table", _download_table)

    return DatabaseManager(
        organism="homo_sapiens",
        form="gene",
        local_repository=str(tmp_path),
        ensembl_release=100,
        ignore_before=100,
        ignore_after=101,
        genome_assembly=38,
        store_raw_always=True,
    )


def test_import_database_manager():
    assert DatabaseManager is not None


def test_init_rejects_unsupported_organism(tmp_path):
    with pytest.raises(NotImplementedError):
        DatabaseManager(
            organism="drosophila_melanogaster",
            form="gene",
            local_repository=str(tmp_path),
            ensembl_release=100,
            ignore_before=100,
            ignore_after=101,
            genome_assembly=38,
        )


def test_init_defaults_release_respects_ignore_after(tmp_path, monkeypatch):
    """When `ensembl_release=None`, the default release should not exceed `ignore_after`."""

    def _fake_core_index(*, organism: str, genome_assembly: int):  # noqa: ARG001
        releases = [100, 101, 102]
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
        }

    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: _fake_core_index(**kw)))

    dm = DatabaseManager(
        organism="homo_sapiens",
        form="gene",
        local_repository=str(tmp_path),
        ensembl_release=None,
        ignore_before=100,
        ignore_after=101,
        genome_assembly=38,
    )
    assert dm.ensembl_release == 101
    assert dm.mysql_settings["port"] == 3306
    assert dm.mysql_database == "homo_sapiens_core_101_38"


def test_init_infers_assembly_that_supports_requested_release(tmp_path, monkeypatch):
    """When `genome_assembly=None`, prefer the highest-priority assembly that contains the requested release."""

    def _fake_core_index(*, organism: str, genome_assembly: int):
        if organism != "mus_musculus":
            raise ValueError("unexpected organism")
        if genome_assembly == 39:
            releases = list(range(103, 116))
        elif genome_assembly == 38:
            releases = list(range(68, 103))
        else:
            releases = []
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
        }

    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: _fake_core_index(**kw)))

    dm = DatabaseManager(
        organism="mus_musculus",
        form="gene",
        local_repository=str(tmp_path),
        ensembl_release=100,
        ignore_before=68,
        ignore_after=100,
        genome_assembly=None,
    )
    assert dm.genome_assembly == 38
    assert dm.mysql_settings["port"] == 3306
    assert dm.mysql_database == "mus_musculus_core_100_38"


def test_get_table_downloads_and_caches(synthetic_dm, monkeypatch):
    calls: list[tuple[str, list[str] | None]] = []

    original_download = DatabaseManager.download_table

    def _wrapped_download(self: DatabaseManager, table_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
        calls.append((table_key, usecols))
        return original_download(self, table_key, usecols)

    monkeypatch.setattr(DatabaseManager, "download_table", _wrapped_download)

    df1 = synthetic_dm.get_table("gene", usecols=["gene_id", "stable_id", "version"])
    df2 = synthetic_dm.get_table("gene", usecols=["gene_id", "stable_id", "version"])

    assert df1.equals(df2)
    assert calls.count(("gene", ["gene_id", "stable_id", "version"])) == 1


def test_create_ids_and_release_id(synthetic_dm):
    idsraw = synthetic_dm.get_db("idsraw_gene")
    assert set(idsraw.columns) == {"gene_id", "gene_stable_id", "gene_version"}
    assert idsraw["gene_id"].dtype.kind in {"i", "u"}

    ids = synthetic_dm.get_db("ids")
    assert list(ids.columns) == ["gene_stable_id", "gene_version"]
    assert ids["gene_stable_id"].is_unique
    assert ids["gene_stable_id"].str.contains(DB.id_ver_delimiter, regex=False).sum() == 0


def test_create_relation_current_builds_three_node_columns(synthetic_dm):
    rel = synthetic_dm.get_db("relationcurrent")
    assert list(rel.columns) == ["gene", "transcript", "translation"]
    assert rel["gene"].str.startswith("ENSG").all()
    assert rel["transcript"].str.startswith("ENST").all()
    assert (rel["translation"].astype(str).str.startswith("ENSP") | (rel["translation"] == "")).all()


def test_create_id_history_narrow_and_full(synthetic_dm):
    full = synthetic_dm.get_db("idhistory")
    narrow = synthetic_dm.get_db("idhistory_narrow")

    assert {"old_stable_id", "new_stable_id", "old_release", "new_release"}.issubset(full.columns)
    assert {"mapping_session_id", "type"}.issubset(full.columns)
    assert {"mapping_session_id", "type"}.isdisjoint(narrow.columns)
    assert (narrow["old_release"] >= synthetic_dm.ignore_before).all()


def test_create_id_history_fixed_repairs_reappearing_versions(synthetic_dm, monkeypatch):
    df = pd.DataFrame(
        {
            "old_stable_id": ["ENSG00000000001", "ENSG00000000001", "ENSG00000000001"],
            "old_version": [1, 2, 1],
            "new_stable_id": ["ENSG00000000001", "ENSG00000000001", "ENSG00000000001"],
            "new_version": [2, 3, 2],
            "score": [np.nan, np.nan, np.nan],
            "old_release": [100, 101, 102],
            "new_release": [101, 102, 103],
        }
    )

    monkeypatch.setattr(synthetic_dm, "get_db", lambda *_args, **_kwargs: df.copy(deep=True))

    fixed = synthetic_dm.create_id_history_fixed(narrow=True, inspect=True)
    assert "unfixed_old_version" in fixed.columns
    assert fixed.loc[2, "old_version"] == 3


def test_external_db_all_and_filter_modes(synthetic_dm):
    synthetic_dm.external_inst = type(
        "_External",
        (),
        {"give_list_for_case": staticmethod(lambda give_type: ["UniProt", "EntrezGene", f"{DB.synonym_id_nodes_prefix}UniProt"])},
    )()

    all_rows = synthetic_dm.create_external_db(filter_mode="all")
    assert set(["release", "graph_id", "id_db", "name_db", "ensembl_identity", "xref_identity"]).issubset(all_rows.columns)
    assert (all_rows["release"] == 100).all()
    assert all_rows["graph_id"].str.startswith("ENSG").all()

    # Synonym rows should be prefixed both on the identifier and database name.
    synonym_rows = all_rows[all_rows["id_db"].str.startswith(DB.synonym_id_nodes_prefix)]
    assert len(synonym_rows) > 0
    assert synonym_rows["name_db"].str.startswith(DB.synonym_id_nodes_prefix).all()

    relevant = synthetic_dm.create_external_db(filter_mode="relevant")
    assert set(relevant["name_db"]).issubset({"UniProt", "EntrezGene", f"{DB.synonym_id_nodes_prefix}UniProt"})

    db_counts = synthetic_dm.create_external_db(filter_mode="database")
    assert set(db_counts.columns) == {"name_db", "count"}

    with pytest.raises(ValueError):
        synthetic_dm.create_external_db(filter_mode="not-a-mode")


def test_release_discovery_and_mysql_database_from_server_catalog(tmp_path, monkeypatch):
    import pymysql

    class _Cursor:
        def execute(self, _query: str) -> None:  # noqa: ARG002
            return None

        def fetchall(self):
            return [
                ("homo_sapiens_core_100_38",),
                ("homo_sapiens_core_101_38",),
                ("homo_sapiens_other_100_38",),
                ("unrelated_db",),
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    class _Conn:
        def cursor(self):
            return _Cursor()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    monkeypatch.setattr(pymysql, "connect", lambda **_kwargs: _Conn())

    dm = DatabaseManager(
        organism="homo_sapiens",
        form="gene",
        local_repository=str(tmp_path),
        ensembl_release=100,
        ignore_before=100,
        ignore_after=101,
        genome_assembly=38,
    )

    assert dm.available_releases == [100, 101]
    assert dm.mysql_database == "homo_sapiens_core_100_38"
