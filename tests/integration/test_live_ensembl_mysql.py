#!/usr/bin/env python3
"""Live integration tests against the Ensembl public MySQL service.

These tests validate that (when the live service is reachable):

1) `DB.assembly_mysqlport_priority` matches the real organism/assembly coverage on
   `ensembldb.ensembl.org` across the configured ports **for the assemblies the live service exposes**.
2) `DatabaseManager._get_core_db_index` correctly discovers releases across multiple ports and
   returns the *actual* schema name for each release (including patch-letter suffixes) **for MySQL-backed releases**.
3) `DatabaseManager(...)` selects the correct port and schema for *every* discovered release for
   every configured organism/assembly pair **that is present on the live MySQL service**.

The tests are marked as integration/network/database and will be skipped if the public service is
unreachable from the test environment.
"""

from __future__ import annotations

import re
import tempfile
from collections import defaultdict
from typing import Any

import pymysql
import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _can_connect(port: int) -> bool:
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


def _fetch_core_schemas(*, organism: str, port: int) -> list[str]:
    like = f"{_escape_like(organism)}\\_core\\_%"
    last_exc: Exception | None = None

    for _attempt in range(2):
        try:
            with pymysql.connect(
                host=DB.mysql_host,
                user=DB.myqsl_user,
                password=DB.mysql_togo,
                port=int(port),
                connect_timeout=max(DB.connection_timeout, 30),
                read_timeout=max(DB.reading_timeout, 60),
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                        "WHERE SCHEMA_NAME LIKE %s ESCAPE '\\\\'",
                        (like,),
                    )
                    rows = cur.fetchall()
            names: list[str] = []
            for (name,) in rows:
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                names.append(str(name))
            return names
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(f"Unable to query information_schema on port {int(port)} for {organism!r}: {last_exc}")


@pytest.fixture(scope="session")
def live_mysql_core_catalog() -> dict[str, dict[int, dict[str, Any]]]:
    """Return live core schema metadata from Ensembl public MySQL.

    Structure:
        catalog[organism][port] = {
            "schemas": [schema_name, ...],
            "by_assembly": {assembly: {release: schema_name}},
            "releases_by_assembly": {assembly: {release, ...}},
        }

    Returns:
        Nested dictionary mapping organism -> port -> schema metadata.

    Raises:
        AssertionError: If core schemas do not match the expected naming pattern.
    """
    ports = sorted(
        {
            int(p)
            for org in DB.supported_organisms
            for cfg in DB.assembly_mysqlport_priority[org].values()
            for p in cfg["Ports"]
        }
    )

    unreachable = [p for p in ports if not _can_connect(p)]
    if unreachable:
        pytest.skip(f"Ensembl public MySQL not reachable on ports {unreachable}.")

    catalog: dict[str, dict[int, dict[str, Any]]] = {}
    for organism in DB.supported_organisms:
        catalog[organism] = {}
        for port in ports:
            schemas = _fetch_core_schemas(organism=organism, port=port)
            pattern = re.compile(
                rf"^{re.escape(organism)}_core_(?P<release>[0-9]+)_(?P<assembly>[0-9]+)(?P<patch>[a-z]*)$"
            )

            by_assembly: dict[int, dict[int, str]] = defaultdict(dict)
            releases_by_assembly: dict[int, set[int]] = defaultdict(set)
            mismatched: list[str] = []
            for schema in schemas:
                m = pattern.match(schema)
                if not m:
                    mismatched.append(schema)
                    continue
                rel = int(m.group("release"))
                asm = int(m.group("assembly"))
                releases_by_assembly[asm].add(rel)
                # Deterministic choice if Ensembl ever exposes multiple patch variants.
                existing = by_assembly[asm].get(rel)
                if existing is None or len(schema) < len(existing):
                    by_assembly[asm][rel] = schema

            if mismatched:
                raise AssertionError(
                    f"Found core schemas that do not match the expected naming pattern on port {int(port)} "
                    f"for {organism!r}: {mismatched[:5]} (n={len(mismatched)})"
                )

            catalog[organism][int(port)] = {
                "schemas": schemas,
                "by_assembly": dict(by_assembly),
                "releases_by_assembly": {asm: set(rels) for asm, rels in releases_by_assembly.items()},
            }
    return catalog


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_live_mysql_config_covers_all_discovered_assemblies(live_mysql_core_catalog):
    """Ensure our config includes every core assembly Ensembl exposes on the configured ports."""
    for organism in DB.supported_organisms:
        discovered: set[int] = set()
        for port_info in live_mysql_core_catalog[organism].values():
            discovered.update(port_info["releases_by_assembly"].keys())

        configured = set(DB.assembly_mysqlport_priority[organism].keys())
        # The live service often hosts only the current and previous release.
        # Older/discontinued assemblies can still be supported via the HTTPS/FTP MySQL dumps.
        assert discovered.issubset(configured), (organism, sorted(discovered), sorted(configured))


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_live_mysql_config_ports_match_discovered_ports(live_mysql_core_catalog):
    """Ensure each (organism, assembly) lists exactly the ports where Ensembl hosts it."""
    for organism in DB.supported_organisms:
        for assembly, cfg in DB.assembly_mysqlport_priority[organism].items():
            discovered_ports = {
                int(port)
                for port, port_info in live_mysql_core_catalog[organism].items()
                if int(assembly) in port_info["releases_by_assembly"]
                and port_info["releases_by_assembly"][int(assembly)]
            }
            if not discovered_ports:
                # This (organism, assembly) is not present on the live MySQL service (e.g. historic assemblies).
                # Access is expected to happen via the HTTPS/FTP dumps instead.
                continue
            assert discovered_ports == set(map(int, cfg["Ports"])), (
                organism,
                int(assembly),
                discovered_ports,
                cfg["Ports"],
            )


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_core_index_matches_live_catalog(live_mysql_core_catalog):
    """Cross-check DatabaseManager discovery against an independent live enumeration."""
    for organism in DB.supported_organisms:
        for assembly, cfg in DB.assembly_mysqlport_priority[organism].items():
            core_index = DatabaseManager._get_core_db_index(organism=organism, genome_assembly=int(assembly))

            assert tuple(core_index["ports"]) == tuple(int(p) for p in cfg["Ports"])

            # Focus on the releases that the live MySQL service actually provides.
            expected_mysql_union = sorted(
                {
                    int(rel)
                    for port in cfg["Ports"]
                    for rel in live_mysql_core_catalog[organism][int(port)]["releases_by_assembly"].get(
                        int(assembly), set()
                    )
                }
            )
            mysql_releases = sorted(set(core_index.get("releases_on_mysql", []) or []))
            assert mysql_releases == expected_mysql_union

            if not mysql_releases:
                # No MySQL-backed releases for this assembly on the live service; nothing further to compare.
                continue

            for port in cfg["Ports"]:
                port = int(port)
                live_db_by_release = live_mysql_core_catalog[organism][port]["by_assembly"].get(int(assembly), {})
                idx_db_by_release = core_index["db_by_port_release"].get(port, {})
                # Only compare the releases present on this port in the live catalog.
                for rel, live_db in live_db_by_release.items():
                    assert str(idx_db_by_release[int(rel)]) == str(live_db)

            # First-port-wins semantics for overlaps must match config order.
            for rel in mysql_releases:
                expected_port: int | None = None
                expected_db: str | None = None
                for port in cfg["Ports"]:
                    port = int(port)
                    live_db_by_release = live_mysql_core_catalog[organism][port]["by_assembly"].get(int(assembly), {})
                    if rel in live_db_by_release:
                        expected_port = port
                        expected_db = str(live_db_by_release[rel])
                        break

                assert expected_port is not None and expected_db is not None
                assert int(core_index["port_for_release"][rel]) == int(expected_port)
                assert str(core_index["db_for_release"][rel]) == str(expected_db)


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_database_manager_resolves_every_release_for_each_configured_assembly(live_mysql_core_catalog):
    """Ensure DatabaseManager picks the correct port and schema for every discovered release."""
    for organism in DB.supported_organisms:
        for assembly in DB.assembly_mysqlport_priority[organism]:
            core_index = DatabaseManager._get_core_db_index(organism=organism, genome_assembly=int(assembly))
            mysql_releases = sorted(set(core_index.get("releases_on_mysql", []) or []))
            if not mysql_releases:
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                for rel in mysql_releases:
                    dm = DatabaseManager(
                        organism=organism,
                        form="gene",
                        local_repository=tmpdir,
                        ensembl_release=int(rel),
                        genome_assembly=int(assembly),
                    )
                    assert int(dm.mysql_settings["port"]) == int(core_index["port_for_release"][int(rel)])
                    assert str(dm.mysql_database) == str(core_index["db_for_release"][int(rel)])
                    assert re.match(
                        rf"^{re.escape(organism)}_core_{int(rel)}_{int(assembly)}[a-z]*$",
                        str(dm.mysql_database),
                    ), (organism, int(assembly), int(rel), dm.mysql_database)


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_database_manager_infers_best_assembly_per_release_from_live_availability(live_mysql_core_catalog):
    """Ensure `genome_assembly=None` selects the highest-priority assembly that contains the requested release."""
    for organism in DB.supported_organisms:
        priorities = {int(asm): int(cfg["Priority"]) for asm, cfg in DB.assembly_mysqlport_priority[organism].items()}

        # Union of releases across all assemblies (from the live catalog).
        releases_all: set[int] = set()
        live_releases_by_assembly: dict[int, set[int]] = defaultdict(set)
        for port_info in live_mysql_core_catalog[organism].values():
            for asm, rels in port_info["releases_by_assembly"].items():
                live_releases_by_assembly[int(asm)].update(int(r) for r in rels)
                releases_all.update(int(r) for r in rels)

        expected_best: dict[int, int] = {}
        for rel in sorted(releases_all):
            candidates = [asm for asm, rels in live_releases_by_assembly.items() if rel in rels]
            assert candidates, (organism, rel)
            expected_best[int(rel)] = sorted(candidates, key=lambda a: priorities[int(a)])[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            for rel, expected_asm in expected_best.items():
                dm = DatabaseManager(
                    organism=organism,
                    form="gene",
                    local_repository=tmpdir,
                    ensembl_release=int(rel),
                    genome_assembly=None,
                )
                assert int(dm.genome_assembly) == int(expected_asm), (
                    organism,
                    int(rel),
                    expected_asm,
                    dm.genome_assembly,
                )
