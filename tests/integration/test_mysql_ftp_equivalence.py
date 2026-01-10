#!/usr/bin/env python3
"""Integration tests asserting FTP-dump and live-MySQL parity (when both are reachable).

These tests exist to guarantee that `DatabaseManager` returns the **same pandas
dataframes** (values + column order) whether data is fetched via:

1) direct MySQL queries to `ensembldb.ensembl.org`, or
2) Ensembl HTTPS/FTP MySQL dumps under `https://ftp.ensembl.org/pub/...`.

They are skipped automatically when either the FTP host or the MySQL service is
unreachable from the test environment.
"""

from __future__ import annotations

import tempfile

import pandas as pd
import pymysql
import pytest
import requests

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB


def _can_reach_ensembl_ftp() -> bool:
    try:
        r = requests.get(
            f"https://{DB.ensembl_ftp_base}/pub/",
            timeout=(DB.connection_timeout, DB.reading_timeout),
        )
        return bool(r.ok)
    except Exception:
        return False


def _can_connect_mysql_port(port: int) -> bool:
    try:
        with pymysql.connect(
            host=DB.mysql_host,
            user=DB.myqsl_user,
            password=DB.mysql_togo,
            port=int(port),
            connect_timeout=max(DB.connection_timeout, 10),
            read_timeout=max(DB.reading_timeout, 10),
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def human38_mysql_release() -> int:
    """Return a human GRCh38 Ensembl release reachable via both FTP and MySQL."""
    if not _can_reach_ensembl_ftp():
        pytest.skip(f"Unable to reach https://{DB.ensembl_ftp_base}/pub/ from the test environment.")

    ports = [int(p) for p in DB.assembly_mysqlport_priority["homo_sapiens"][38]["Ports"]]
    if not any(_can_connect_mysql_port(p) for p in ports):
        pytest.skip(f"Ensembl public MySQL not reachable on any of the human GRCh38 ports: {ports}.")

    core_index = DatabaseManager._get_core_db_index(organism="homo_sapiens", genome_assembly=38)
    mysql_releases = sorted(set(core_index.get("releases_on_mysql", []) or []))
    if not mysql_releases:
        pytest.skip("No MySQL-backed human GRCh38 releases were discovered in this environment.")

    return int(mysql_releases[-1])


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_download_table_mysql_matches_ftp_for_coord_system(human38_mysql_release: int, monkeypatch) -> None:
    """Assert that MySQL and FTP results match for `coord_system`."""
    usecols = ["coord_system_id", "name", "version"]

    with tempfile.TemporaryDirectory(prefix="idtrack_mysql_ftp_eq_") as tmpdir:
        dm = DatabaseManager(
            organism="homo_sapiens",
            form="gene",
            local_repository=tmpdir,
            ensembl_release=int(human38_mysql_release),
            genome_assembly=38,
            store_raw_always=False,
        )

        df_ftp = dm._download_table_from_ftp("coord_system", usecols=usecols)

        # Sanity-check that the target DB is queryable via MySQL before asserting parity.
        try:
            with pymysql.connect(
                host=dm.mysql_settings["host"],
                user=dm.mysql_settings["user"],
                password=dm.mysql_settings["password"],
                port=int(dm.mysql_settings["port"]),
                database=str(dm.mysql_database),
                connect_timeout=dm.mysql_settings["connect_timeout"],
                read_timeout=dm.mysql_settings["read_timeout"],
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT coord_system_id, name, version FROM coord_system LIMIT 1")
                    cur.fetchone()
        except Exception as exc:
            pytest.skip(
                f"MySQL sanity query failed for {dm.mysql_database!r} on port {dm.mysql_settings['port']}: {exc!r}"
            )

        def _forbid_ftp(*_args, **_kwargs):
            raise AssertionError("Unexpected FTP fallback while comparing MySQL vs FTP results.")

        monkeypatch.setattr(DatabaseManager, "_download_table_from_ftp", _forbid_ftp)
        df_mysql = dm.download_table("coord_system", usecols=usecols)

        df_mysql = df_mysql.sort_values(by="coord_system_id").reset_index(drop=True)
        df_ftp = df_ftp.sort_values(by="coord_system_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(df_mysql, df_ftp, check_dtype=True)


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_get_table_cache_overwrite_is_source_independent(human38_mysql_release: int, monkeypatch) -> None:
    """Ensure the on-disk cache key is the same for FTP and MySQL sources."""
    usecols = ["coord_system_id", "name", "version"]

    with tempfile.TemporaryDirectory(prefix="idtrack_mysql_ftp_cache_") as tmpdir:
        dm = DatabaseManager(
            organism="homo_sapiens",
            form="gene",
            local_repository=tmpdir,
            ensembl_release=int(human38_mysql_release),
            genome_assembly=38,
            store_raw_always=False,
        )

        # 1) Force FTP for the first materialisation.
        tcp_orig = DatabaseManager._tcp_can_connect

        def _tcp_mysql_block_only(host: str, port: int, timeout_s: float = 2.0) -> bool:  # noqa: ARG001
            if host == DB.mysql_host:
                return False
            return tcp_orig(host, port, timeout_s=timeout_s)

        monkeypatch.setattr(DatabaseManager, "_tcp_can_connect", staticmethod(_tcp_mysql_block_only))
        df_ftp_cached = dm.get_table(
            "coord_system",
            usecols=usecols,
            create_even_if_exist=True,
            save_after_calculation=True,
            overwrite_even_if_exist=True,
        )

        # 2) Force MySQL and overwrite the same cached key.
        monkeypatch.setattr(DatabaseManager, "_tcp_can_connect", staticmethod(tcp_orig))

        def _forbid_ftp(*_args, **_kwargs):
            raise AssertionError("Unexpected FTP fallback while forcing MySQL overwrite of the cached table.")

        monkeypatch.setattr(DatabaseManager, "_download_table_from_ftp", _forbid_ftp)
        df_mysql_cached = dm.get_table(
            "coord_system",
            usecols=usecols,
            create_even_if_exist=True,
            save_after_calculation=True,
            overwrite_even_if_exist=True,
        )

        df_mysql_cached = df_mysql_cached.sort_values(by="coord_system_id").reset_index(drop=True)
        df_ftp_cached = df_ftp_cached.sort_values(by="coord_system_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(df_mysql_cached, df_ftp_cached, check_dtype=True)
