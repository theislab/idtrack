#!/usr/bin/env python3
"""Live integration tests for the HTTPS/FTP MySQL-dump fallback.

These tests download small real tables from `https://ftp.ensembl.org/pub/...` to validate:

1) `DatabaseManager._get_core_db_index` can enumerate releases for each supported
   (organism, assembly) pair even when direct MySQL connectivity is unavailable.
2) `DatabaseManager.download_table` can fetch tables via the dump-based fallback for:
   - human GRCh38 (assembly 38)
   - human GRCh37 (assembly 37), including the GRCh37 archive path for release 78
   - human NCBI36 (assembly 36)
   - mouse GRCm37/38/39 (assemblies 37/38/39)
   - pig Sscrofa9.2/10.2/11.1 (assemblies 9/102/111)

The tests are marked as slow/network and will be skipped if the FTP host is unreachable.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable

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


@pytest.fixture(scope="session")
def ensembl_latest_release() -> int:
    if not _can_reach_ensembl_ftp():
        pytest.skip(f"Unable to reach https://{DB.ensembl_ftp_base}/pub/ from the test environment.")
    return int(DatabaseManager._ensembl_latest_release())


EXPECTED_RANGES: dict[tuple[str, int], dict[str, int | str]] = {
    # Human: GRCh37 and GRCh38 overlap for years (multi-assembly is essential for idtrack).
    ("homo_sapiens", 38): {"min": 76, "max": "latest"},
    ("homo_sapiens", 37): {"min": 55, "max": "latest"},
    # Human: NCBI36 is a historic assembly with a clean handoff to GRCh37 at release 55.
    ("homo_sapiens", 36): {"min": 48, "max": 54},
    # Mouse: clean handoffs (one assembly per release).
    ("mus_musculus", 39): {"min": 103, "max": "latest"},
    ("mus_musculus", 38): {"min": 68, "max": 102},
    ("mus_musculus", 37): {"min": 48, "max": 67},
    # Pig: clean handoff in FTP dumps (later archive releases can be MySQL-only).
    ("sus_scrofa", 111): {"min": 90, "max": "latest"},
    ("sus_scrofa", 102): {"min": 67, "max": 89},
    ("sus_scrofa", 9): {"min": 56, "max": 66},
}

REQUIRED_CORE_TABLES: set[str] = {
    # Core schema tables used throughout DatabaseManager / GraphMaker.
    "gene",
    "transcript",
    "translation",
    "coord_system",
    # History tables used to build idhistory graphs.
    "mapping_session",
    "stable_id_event",
    "gene_archive",
    # Cross-reference tables used for external mappings (even if externals are disabled for some species).
    "object_xref",
    "xref",
    "external_db",
    "identity_xref",
    "external_synonym",
}


def _iter_supported_pairs() -> Iterable[tuple[str, int]]:
    for organism in DB.supported_organisms:
        for assembly in sorted(DB.assembly_mysqlport_priority[organism]):
            yield organism, int(assembly)


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(("organism", "assembly"), sorted(EXPECTED_RANGES))
def test_ftp_core_index_release_ranges(ensembl_latest_release: int, organism: str, assembly: int) -> None:
    expected = EXPECTED_RANGES[(organism, int(assembly))]
    core_index = DatabaseManager._get_core_db_index(organism=organism, genome_assembly=int(assembly))

    # This integration module validates the HTTPS/FTP MySQL-dump availability.
    # `core_index["releases"]` can include MySQL-only archive releases that do not have published dumps,
    # so use `db_dir_url_by_release` (FTP-only) for range assertions here.
    releases = sorted(int(r) for r in (core_index.get("db_dir_url_by_release", {}) or {}).keys())
    assert releases, (organism, int(assembly), core_index.get("source"), "No FTP releases discovered.")

    assert int(releases[0]) == int(expected["min"])

    expected_max = expected["max"]
    if expected_max == "latest":
        assert int(releases[-1]) == int(ensembl_latest_release)
    else:
        assert int(releases[-1]) == int(expected_max)

    # Ensembl release coverage for these supported pairs is contiguous.
    assert releases == list(range(int(releases[0]), int(releases[-1]) + 1))


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
def test_grch37_ftp_archive_path_for_release_78(ensembl_latest_release: int) -> None:  # noqa: ARG001
    core_index = DatabaseManager._get_core_db_index(organism="homo_sapiens", genome_assembly=37)
    url = core_index.get("db_dir_url_by_release", {}).get(78)
    assert isinstance(url, str) and url
    assert "/pub/grch37/release-78/mysql/" in url
    assert url.endswith("/homo_sapiens_core_78_37/")


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(("organism", "assembly"), list(_iter_supported_pairs()))
def test_download_table_via_ftp_fallback_for_each_pair(
    ensembl_latest_release: int, organism: str, assembly: int
) -> None:
    expected = EXPECTED_RANGES.get((organism, int(assembly)))
    if expected is None:
        pytest.skip(f"Missing expected release range for {(organism, int(assembly))}.")

    # Use an older GRCh37 release to exercise the dedicated GRCh37 archive path (and non-MySQL releases).
    if organism == "homo_sapiens" and int(assembly) == 37:
        release = 78
    else:
        expected_max = expected["max"]
        release = int(ensembl_latest_release if expected_max == "latest" else expected_max)

    with tempfile.TemporaryDirectory(prefix="idtrack_ftp_test_") as tmpdir:
        dm = DatabaseManager(
            organism=organism,
            form="gene",
            local_repository=tmpdir,
            ensembl_release=int(release),
            genome_assembly=int(assembly),
            store_raw_always=False,
        )

        # Keep downloads tiny: `coord_system` is small and present in all core dumps.
        df = dm.download_table("coord_system", usecols=["coord_system_id", "name", "version"])
        assert list(df.columns) == ["coord_system_id", "name", "version"]
        assert len(df) > 0


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
def test_download_table_supports_nested_bz2_gzip_dumps() -> None:
    """Exercise the `.txt.gz.bz2` dump path used by some species/releases (notably pig archives)."""
    if not _can_reach_ensembl_ftp():
        pytest.skip(f"Unable to reach https://{DB.ensembl_ftp_base}/pub/ from the test environment.")

    with tempfile.TemporaryDirectory(prefix="idtrack_ftp_nested_dump_test_") as tmpdir:
        dm = DatabaseManager(
            organism="sus_scrofa",
            form="gene",
            local_repository=tmpdir,
            ensembl_release=89,
            genome_assembly=102,
            store_raw_always=False,
        )
        # `external_db` is shipped as `external_db.txt.gz.bz2` for this archive release.
        df = dm.download_table("external_db", usecols=["external_db_id", "db_name"])
        assert list(df.columns) == ["external_db_id", "db_name"]
        assert len(df) > 0


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(("organism", "assembly"), sorted(EXPECTED_RANGES))
def test_ftp_schema_contains_required_tables(ensembl_latest_release: int, organism: str, assembly: int) -> None:
    expected = EXPECTED_RANGES[(organism, int(assembly))]
    min_release = int(expected["min"])
    max_release = int(ensembl_latest_release if expected["max"] == "latest" else expected["max"])

    with tempfile.TemporaryDirectory(prefix="idtrack_ftp_schema_test_") as tmpdir:
        for rel in (min_release, max_release):
            dm = DatabaseManager(
                organism=organism,
                form="gene",
                local_repository=tmpdir,
                ensembl_release=int(rel),
                genome_assembly=int(assembly),
                store_raw_always=False,
            )

            sql_url = dm._ftp_schema_url()
            schema = DatabaseManager._ftp_schema_for_sql_url(sql_url)
            tables = set(schema)

            missing = sorted(REQUIRED_CORE_TABLES - tables)
            assert not missing, (organism, int(assembly), int(rel), missing)

            # `DatabaseManager.create_ids` expects either stable_id/version columns to exist on the main table,
            # or the historic `<form>_stable_id` split table to exist (older releases).
            for form in ("gene", "transcript", "translation"):
                assert form in schema
                cols = set(schema[form])
                if {"stable_id", "version"}.issubset(cols):
                    continue
                stable_table = f"{form}_stable_id"
                assert stable_table in schema, (organism, int(assembly), int(rel), form, stable_table)
                stable_cols = set(schema[stable_table])
                assert {"stable_id", "version"}.issubset(stable_cols), (
                    organism,
                    int(assembly),
                    int(rel),
                    stable_table,
                )
