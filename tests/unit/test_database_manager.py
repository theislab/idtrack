#!/usr/bin/env python3
"""Unit tests for idtrack._database_manager module.

These tests avoid heavy MySQL downloads by patching `DatabaseManager.download_table`
to return tiny synthetic tables, while still exercising the real caching and
data-processing logic.
"""

from dataclasses import dataclass

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
    gene = pd.DataFrame(
        {
            "gene_id": [1, 2],
            "stable_id": ["ENSG00000000001", "ENSG00000000002"],
            "version": [1, 1],
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
        100: {"gene": gene, **base_tables},
        101: {"gene": gene, **base_tables},
    }
    return _SyntheticMySQL(tables_by_release=tables_by_release, available_releases=[100, 101])


@pytest.fixture
def synthetic_dm(tmp_path, monkeypatch) -> DatabaseManager:
    """Create a DatabaseManager backed by tiny synthetic MySQL tables."""
    synthetic = _synthetic_mysql_tables()

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
        try:
            df = synthetic.tables_by_release[int(self.ensembl_release)][table_key]
        except KeyError:
            return pd.DataFrame()
        if usecols is None:
            return df.copy(deep=True)
        return df.loc[:, usecols].copy(deep=True)

    monkeypatch.setattr(DatabaseManager, "available_releases_versions", _available_releases_versions)
    monkeypatch.setattr(DatabaseManager, "download_table", _download_table)
    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: _fake_core_index(**kw)))

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


def test_init_rejects_unsupported_organism(tmp_path):
    """Unsupported organisms should raise NotImplementedError."""
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

    def _fake_core_index(*, organism: str, genome_assembly: int, **_):  # noqa: ARG001
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

    def _fake_core_index(*, organism: str, genome_assembly: int, **_):
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
    """get_table should download once and cache subsequent calls."""
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
    """create_ids should produce ids and idsraw_gene tables."""
    idsraw = synthetic_dm.get_db("idsraw_gene")
    assert set(idsraw.columns) == {"gene_id", "gene_stable_id", "gene_version"}
    assert idsraw["gene_id"].dtype.kind in {"i", "u"}

    ids = synthetic_dm.get_db("ids")
    assert list(ids.columns) == ["gene_stable_id", "gene_version"]
    assert ids["gene_stable_id"].is_unique
    assert ids["gene_stable_id"].str.contains(DB.id_ver_delimiter, regex=False).sum() == 0


def test_create_relation_current_builds_three_node_columns(synthetic_dm):
    """create_relation_current should yield (gene, transcript, translation) columns."""
    rel = synthetic_dm.get_db("relationcurrent")
    assert list(rel.columns) == ["gene", "transcript", "translation"]
    assert rel["gene"].str.startswith("ENSG").all()
    assert rel["transcript"].str.startswith("ENST").all()
    assert (rel["translation"].astype(str).str.startswith("ENSP") | (rel["translation"] == "")).all()


def test_create_id_history_narrow_and_full(synthetic_dm):
    """create_id_history should support both narrow and full modes."""
    full = synthetic_dm.get_db("idhistory")
    narrow = synthetic_dm.get_db("idhistory_narrow")

    assert {"old_stable_id", "new_stable_id", "old_release", "new_release"}.issubset(full.columns)
    assert {"mapping_session_id", "type"}.issubset(full.columns)
    assert {"mapping_session_id", "type"}.isdisjoint(narrow.columns)
    assert (narrow["old_release"] >= synthetic_dm.ignore_before).all()


def test_external_db_all_and_filter_modes(synthetic_dm):
    """create_external_db should support all/relevant/database filter modes."""
    synthetic_dm.external_inst = type(
        "_External",
        (),
        {
            "give_list_for_case": staticmethod(
                lambda give_type: ["UniProt", "EntrezGene", f"{DB.synonym_id_nodes_prefix}UniProt"]
            )
        },
    )()

    all_rows = synthetic_dm.create_external_db(filter_mode="all")
    assert {"release", "graph_id", "id_db", "name_db", "ensembl_identity", "xref_identity"}.issubset(all_rows.columns)
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


def test_external_db_merge_coerces_string_external_db_id(synthetic_dm, monkeypatch):
    """create_external_db should tolerate object-typed external_db_id merge keys."""
    original_download_table = synthetic_dm.download_table

    def _download_table(table_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
        df = original_download_table(table_key, usecols=usecols)
        if table_key == "external_db":
            df["external_db_id"] = df["external_db_id"].astype(str)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "external_db_id": "\\",
                                "db_name": "BAD_ROW",
                                "db_display_name": "BAD_ROW",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        return df

    monkeypatch.setattr(synthetic_dm, "download_table", _download_table)

    all_rows = synthetic_dm.create_external_db(filter_mode="all")
    assert len(all_rows) > 0
    assert {"UniProt", "UniProtKB/Swiss-Prot"}.issubset(set(all_rows["name_db"]))


def test_release_discovery_and_mysql_database_from_server_catalog(tmp_path, monkeypatch):
    """available_releases/mysql_database should be discovered from server catalog."""
    # Offline: the MySQL port is reported usable and the (mocked) HTTPS scan answers that the dumps hold none of these
    # releases, so they come from the (mocked) MySQL catalogue.
    import pymysql

    from idtrack._ensembl_access import reachability

    monkeypatch.setattr(reachability, "mysql_port_usable", lambda port: True)
    monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})

    def _no_https(cls, *, organism, genome_assembly, ports):
        return {
            "organism": organism,
            "genome_assembly": genome_assembly,
            "ports": tuple(ports),
            "releases_by_port": {int(p): set() for p in ports},
            "db_by_port_release": {int(p): {} for p in ports},
            "releases": [],
            "port_for_release": {},
            "db_for_release": {},
            "db_dir_url_by_release": {},
            "scanned_up_to": 101,
        }

    monkeypatch.setattr(DatabaseManager, "_get_core_db_index_from_ftp", classmethod(_no_https))

    class _Cursor:
        def execute(self, _query: str, _params=None) -> None:  # noqa: ARG002
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


def test_create_ids_does_not_hide_a_failure_behind_the_stable_id_table_fallback(synthetic_dm, monkeypatch):
    """Only a schema answer leads to the `<form>_stable_id` fallback; any other error surfaces unchanged."""
    # Before, every `ValueError` (e.g. a cache write that failed under pandas 3) was taken to mean "old schema",
    # and the user saw `Table 'gene_stable_id' not found` instead of the real error.
    import idtrack._utils_hdf5 as hs

    original_write = hs.write_table

    def _failing_write(path, df, params):
        if "stable_id" in df.columns and "gene_stable_id" not in path:
            raise ValueError("Unsupported dtype for column 'stable_id'")
        return original_write(path, df, params)

    monkeypatch.setattr(hs, "write_table", _failing_write)
    with pytest.raises(ValueError, match="Unsupported dtype for column 'stable_id'"):
        synthetic_dm.create_ids("gene")


def test_create_ids_falls_back_to_the_stable_id_table_on_an_old_schema(synthetic_dm, monkeypatch):
    """A release whose `gene` table has no stable_id/version columns is read from `gene_stable_id`."""
    from idtrack._db import EnsemblSchemaError

    old_gene = pd.DataFrame({"gene_id": [1, 2]})
    old_stable = pd.DataFrame(
        {"gene_id": [1, 2], "stable_id": ["ENSG00000000001", "ENSG00000000002"], "version": [1, 1]}
    )

    def _download_table(self, table_key, usecols=None):
        if table_key == "gene":
            missing = [c for c in (usecols or []) if c not in old_gene.columns]
            if missing:
                raise EnsemblSchemaError("Missing columns", table="gene", missing_columns=tuple(missing))
            return old_gene.loc[:, usecols].copy()
        if table_key == "gene_stable_id":
            return old_stable.loc[:, usecols].copy() if usecols else old_stable.copy()
        raise EnsemblSchemaError(f"Table {table_key!r} not found", table=table_key)

    monkeypatch.setattr(DatabaseManager, "download_table", _download_table)
    df = synthetic_dm.create_ids("gene")
    assert sorted(df["gene_stable_id"]) == ["ENSG00000000001", "ENSG00000000002"]


def test_schema_error_is_not_a_value_error():
    """Broad `except ValueError` handlers must not catch the schema answer by accident (and vice versa)."""
    from idtrack._db import EnsemblSchemaError

    err = EnsemblSchemaError("m", table="t", missing_columns=("a",))
    assert not isinstance(err, ValueError)
    assert err.table == "t" and err.missing_columns == ("a",)


def test_schema_error_survives_pickle_and_copy():
    """The schema answer keeps its message and fields through pickling (e.g. from a worker process) and copying."""
    import copy
    import pickle  # noqa: S403

    from idtrack._db import EnsemblSchemaError

    err = EnsemblSchemaError("Table 'gene' lacks columns", table="gene", missing_columns=("version",))
    for clone in (pickle.loads(pickle.dumps(err)), copy.copy(err)):  # noqa: S301
        assert isinstance(clone, EnsemblSchemaError)
        assert str(clone) == "Table 'gene' lacks columns"
        assert clone.table == "gene" and clone.missing_columns == ("version",)


class TestSavedHttpsIndex:
    """The HTTPS-dump index is saved in the local repository and spares the network for windows it covers."""

    @staticmethod
    def _scan(calls):
        def _fake_ftp(cls, *, organism, genome_assembly, ports):
            calls.append((organism, genome_assembly))
            releases = [110, 111, 112]
            return {
                "organism": organism,
                "genome_assembly": int(genome_assembly),
                "ports": tuple(ports),
                "releases_by_port": {int(ports[0]): set(releases)},
                "db_by_port_release": {int(ports[0]): {r: f"{organism}_core_{r}_111" for r in releases}},
                "releases": releases,
                "port_for_release": {r: int(ports[0]) for r in releases},
                "db_for_release": {r: f"{organism}_core_{r}_111" for r in releases},
                "db_dir_url_by_release": {r: f"https://example/{r}/" for r in releases},
                "source": "ftp",
                "scanned_up_to": 112,
            }

        return classmethod(_fake_ftp)

    @pytest.fixture
    def scans(self, monkeypatch):
        """Record every faked HTTPS scan; the in-memory index starts empty and MySQL is unusable."""
        from idtrack._ensembl_access import reachability

        calls: list = []
        monkeypatch.setattr(DatabaseManager, "_get_core_db_index_from_ftp", self._scan(calls))
        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
        monkeypatch.setattr(reachability, "mysql_port_usable", lambda port: False)
        return calls

    def _index(self, repo, needed_up_to):
        return DatabaseManager._get_core_db_index(
            organism="sus_scrofa", genome_assembly=111, local_repository=str(repo), needed_up_to=needed_up_to
        )

    def test_saved_index_answers_a_covered_window_in_a_new_process(self, tmp_path, scans, monkeypatch):
        """A window ending at or before the saved scan is answered from disk, without scanning."""
        first = self._index(tmp_path, 112)
        assert len(scans) == 1 and (tmp_path / "ensembl_https_index_sus_scrofa_assembly-111.json").exists()
        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})  # as in a new process
        again = self._index(tmp_path, 111)
        assert len(scans) == 1
        for key in ("releases", "port_for_release", "db_for_release", "db_dir_url_by_release", "releases_by_port"):
            assert again[key] == first[key]

    def test_a_window_beyond_the_saved_scan_or_open_ended_scans_again(self, tmp_path, scans, monkeypatch):
        """The latest release is a network question; a saved scan never answers it."""
        self._index(tmp_path, 112)
        for need in (113, None):
            monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
            self._index(tmp_path, need)
        assert len(scans) == 3

    def test_saved_index_of_other_settings_is_ignored(self, tmp_path, scans, monkeypatch):
        """An index saved for another organism-table entry (here: another suffix) is not used."""
        import json

        self._index(tmp_path, 112)
        path = tmp_path / "ensembl_https_index_sus_scrofa_assembly-111.json"
        saved = json.loads(path.read_text())
        saved["signature"]["db_suffix"] = "999"
        path.write_text(json.dumps(saved))
        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
        self._index(tmp_path, 111)
        assert len(scans) == 2

    def test_manager_with_a_bounded_window_is_created_offline(self, tmp_path, scans, monkeypatch):
        """After one online run, a manager for the same window needs neither REST nor the dump scan."""
        import idtrack._ensembl_access as ea

        monkeypatch.setenv("IDTRACK_ENSEMBL_ACCESS", "https")
        DatabaseManager(
            organism="sus_scrofa",
            form="gene",
            local_repository=str(tmp_path),
            ignore_before=110,
            ignore_after=112,
            genome_assembly=111,
        )
        assert len(scans) == 1

        def _offline(*args, **kwargs):
            raise AssertionError("the network was used")

        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
        monkeypatch.setattr(DatabaseManager, "_get_core_db_index_from_ftp", classmethod(_offline))
        monkeypatch.setattr(ea, "rest_get_json", _offline)
        dm = DatabaseManager(
            organism="sus_scrofa",
            form="gene",
            local_repository=str(tmp_path),
            ignore_before=110,
            ignore_after=112,
            genome_assembly=111,
        )
        assert dm.ensembl_release == 112 and dm.available_releases == [110, 111, 112]
