#!/usr/bin/env python3
"""Unit tests for `idtrack._ensembl_access`: access mode and the per-process port reachability memory (no network)."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pandas as pd
import pymysql
import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._ensembl_access import PortReachability, ensembl_access_mode, reachability


@pytest.fixture
def fresh_reachability(monkeypatch):
    """Give the module-level memory an empty state and count probes."""
    probes: list[tuple[str, int]] = []
    answers: dict[tuple[str, int], bool] = {}

    def _probe(host, port, timeout_s):
        probes.append((host, int(port)))
        return answers.get((host, int(port)), False)

    reachability.reset()
    monkeypatch.setattr(PortReachability, "_probe", staticmethod(_probe))
    yield probes, answers
    reachability.reset()


class TestAccessMode:
    """Access Mode."""

    def test_default_is_auto(self, monkeypatch):
        """Default is auto."""
        monkeypatch.setattr(DB, "ensembl_access", None)
        monkeypatch.delenv(DB.ensembl_access_env_var, raising=False)
        assert ensembl_access_mode() == "auto"

    def test_environment_variable(self, monkeypatch):
        """Environment variable."""
        monkeypatch.setattr(DB, "ensembl_access", None)
        monkeypatch.setenv(DB.ensembl_access_env_var, " HTTPS ")
        assert ensembl_access_mode() == "https"

    def test_attribute_wins_over_environment(self, monkeypatch):
        """Attribute wins over environment."""
        monkeypatch.setenv(DB.ensembl_access_env_var, "https")
        monkeypatch.setattr(DB, "ensembl_access", "auto")
        assert ensembl_access_mode() == "auto"

    def test_invalid_value(self, monkeypatch):
        """Invalid value."""
        monkeypatch.setattr(DB, "ensembl_access", "mysql-only")
        with pytest.raises(ValueError, match="not a valid Ensembl access mode"):
            ensembl_access_mode()


class TestReachabilityMemory:
    """Reachability Memory."""

    def test_one_probe_per_port_per_process(self, fresh_reachability, monkeypatch):
        """One probe per port per process."""
        probes, answers = fresh_reachability
        monkeypatch.setattr(DB, "ensembl_access", "auto")
        answers[(DB.mysql_host, 3306)] = True
        for _ in range(5):
            assert reachability.mysql_port_usable(3306) is True
            assert reachability.mysql_port_usable(5306) is False
        assert probes == [(DB.mysql_host, 3306), (DB.mysql_host, 5306)]

    def test_https_mode_never_probes(self, fresh_reachability, monkeypatch):
        """Https mode never probes."""
        probes, answers = fresh_reachability
        answers[(DB.mysql_host, 3306)] = True
        monkeypatch.setattr(DB, "ensembl_access", "https")
        assert reachability.mysql_port_usable(3306) is False
        assert probes == []

    def test_failure_after_success_marks_unreachable(self, fresh_reachability, monkeypatch):
        """Failure after success marks unreachable."""
        probes, answers = fresh_reachability
        monkeypatch.setattr(DB, "ensembl_access", "auto")
        answers[(DB.mysql_host, 3306)] = True
        assert reachability.mysql_port_usable(3306)
        reachability.mark_unreachable(DB.mysql_host, 3306, "lost connection")
        assert not reachability.mysql_port_usable(3306)
        assert probes == [(DB.mysql_host, 3306)]

    def test_reset_probes_again(self, fresh_reachability, monkeypatch):
        """Reset probes again."""
        probes, _answers = fresh_reachability
        monkeypatch.setattr(DB, "ensembl_access", "auto")
        reachability.mysql_port_usable(3337)
        reachability.reset()
        reachability.mysql_port_usable(3337)
        assert probes == [(DB.mysql_host, 3337), (DB.mysql_host, 3337)]


class TestDatabaseManagerUsesTheMemory:
    """Repeated index look-ups and table downloads never re-probe a blocked port."""

    def test_index_built_once_and_cached_lookups_do_not_probe(self, fresh_reachability, monkeypatch):
        """Index built once and cached lookups do not probe."""
        probes, _answers = fresh_reachability
        monkeypatch.setattr(DB, "ensembl_access", "auto")
        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
        monkeypatch.setattr(
            DatabaseManager,
            "_get_core_db_index_from_ftp",
            classmethod(
                lambda cls, *, organism, genome_assembly, ports: {
                    "releases": [100],
                    "db_for_release": {100: "sus_scrofa_core_100_111"},
                    "port_for_release": {100: 3306},
                }
            ),
        )
        for _ in range(10):
            index = DatabaseManager._get_core_db_index(organism="sus_scrofa", genome_assembly=111)
        assert index["releases"] == [100]
        assert sorted(probes) == [(DB.mysql_host, 3306), (DB.mysql_host, 5306)]

    def test_https_mode_download_never_touches_mysql(self, fresh_reachability, monkeypatch):
        """Https mode download never touches mysql."""
        probes, _answers = fresh_reachability
        monkeypatch.setattr(DB, "ensembl_access", "https")
        dm = DatabaseManager.__new__(DatabaseManager)
        dm.organism, dm.genome_assembly, dm.ensembl_release = "sus_scrofa", 111, 100
        dm.mysql_settings = {"host": DB.mysql_host, "port": 3306}
        monkeypatch.setattr(
            DatabaseManager,
            "_get_core_db_index",
            classmethod(lambda cls, **kw: {"releases_on_mysql": [100]}),
        )
        connect = MagicMock(side_effect=AssertionError("MySQL must not be used in https mode"))
        monkeypatch.setattr(pymysql, "connect", connect)
        monkeypatch.setattr(
            DatabaseManager, "_download_table_from_ftp", lambda self, table_key, usecols=None: pd.DataFrame({"a": [1]})
        )
        df = dm.download_table("gene")
        assert list(df["a"]) == [1]
        assert probes == []
        connect.assert_not_called()

    def test_connection_failure_during_query_marks_port(self, fresh_reachability, monkeypatch):
        """Connection failure during query marks port."""
        probes, answers = fresh_reachability
        monkeypatch.setattr(DB, "ensembl_access", "auto")
        answers[(DB.mysql_host, 3306)] = True
        dm = DatabaseManager.__new__(DatabaseManager)
        dm.organism, dm.genome_assembly, dm.ensembl_release = "sus_scrofa", 111, 100
        dm.mysql_settings = {"host": DB.mysql_host, "port": 3306}
        dm.__dict__["mysql_database"] = "sus_scrofa_core_100_111"
        monkeypatch.setattr(
            DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: {"releases_on_mysql": [100]})
        )
        monkeypatch.setattr(
            pymysql, "connect", MagicMock(side_effect=pymysql.err.OperationalError(2013, "Lost connection"))
        )
        monkeypatch.setattr(
            DatabaseManager, "_download_table_from_ftp", lambda self, table_key, usecols=None: pd.DataFrame({"a": [1]})
        )
        dm.download_table("gene")
        assert reachability.known()[(DB.mysql_host, 3306)] is False
        dm.download_table("gene")
        assert probes == [(DB.mysql_host, 3306)]

    def test_query_error_does_not_mark_port(self, fresh_reachability, monkeypatch):
        """Query error does not mark port."""
        _probes, answers = fresh_reachability
        monkeypatch.setattr(DB, "ensembl_access", "auto")
        answers[(DB.mysql_host, 3306)] = True
        dm = DatabaseManager.__new__(DatabaseManager)
        dm.organism, dm.genome_assembly, dm.ensembl_release = "sus_scrofa", 111, 100
        dm.mysql_settings = {"host": DB.mysql_host, "port": 3306}
        dm.__dict__["mysql_database"] = "sus_scrofa_core_100_111"
        monkeypatch.setattr(
            DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: {"releases_on_mysql": [100]})
        )
        monkeypatch.setattr(
            pymysql, "connect", MagicMock(side_effect=pymysql.err.ProgrammingError(1146, "Table doesn't exist"))
        )
        monkeypatch.setattr(
            DatabaseManager, "_download_table_from_ftp", lambda self, table_key, usecols=None: pd.DataFrame({"a": [1]})
        )
        dm.download_table("gene")
        assert reachability.known()[(DB.mysql_host, 3306)] is True


def test_dump_reader_is_used_for_https_tables(monkeypatch):
    """`_download_table_from_ftp` decodes escapes through `read_mysql_dump` (an escaped newline stays one row)."""
    dm = DatabaseManager.__new__(DatabaseManager)
    dm.organism, dm.genome_assembly, dm.ensembl_release = "sus_scrofa", 111, 100
    dm.log = MagicMock()
    monkeypatch.setattr(DatabaseManager, "_ftp_db_dir_url", lambda self: "https://x/db/")
    monkeypatch.setattr(DatabaseManager, "_ftp_schema_url", lambda self: "https://x/db/db.sql.gz")
    monkeypatch.setattr(
        DatabaseManager, "_ftp_schema_for_sql_url", classmethod(lambda cls, url: {"external_db": ["id", "text"]})
    )

    class _Resp:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("idtrack._database_manager.requests.get", lambda *a, **k: _Resp())

    class _Ctx:
        def __enter__(self):
            return io.StringIO("1\tline one.\r\\\nline two\n2\t\\N\n")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(DatabaseManager, "_open_decompressed_http_text", staticmethod(lambda url: _Ctx()))
    df = dm._download_table_from_ftp("external_db")
    assert df["id"].tolist() == [1, 2]
    assert df["text"].iloc[0] == "line one.\r\nline two"
    assert pd.isna(df["text"].iloc[1])


class TestInvalidAccessMode:
    """An invalid access mode raises one clear ValueError at the first use and is never swallowed."""

    def test_database_manager_raises_the_mode_error(self, tmp_path, monkeypatch):
        """`DatabaseManager(...)` reports the bad value itself, not a failed assembly inference."""
        monkeypatch.setattr(DB, "ensembl_access", None)
        monkeypatch.setenv(DB.ensembl_access_env_var, "htps")
        monkeypatch.setattr(
            DatabaseManager,
            "_get_core_db_index",
            classmethod(lambda cls, **kw: (_ for _ in ()).throw(AssertionError("must not be reached"))),
        )
        with pytest.raises(ValueError, match="IDTRACK_ENSEMBL_ACCESS='htps' is not a valid Ensembl access mode"):
            DatabaseManager(organism="sus_scrofa", form="gene", local_repository=str(tmp_path))

    def test_create_available_databases_does_not_swallow_it(self, monkeypatch):
        """`create_available_databases` raises instead of returning an empty frame."""
        dm = DatabaseManager.__new__(DatabaseManager)
        dm.organism = "sus_scrofa"
        dm.log = MagicMock()
        monkeypatch.setattr(DB, "ensembl_access", "mysql-only")
        with pytest.raises(ValueError, match="not a valid Ensembl access mode"):
            dm.create_available_databases()
