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
from idtrack._ensembl_access import reachability


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


def _fetch_assembly_names(*, port: int, schemas: list[str]) -> dict[str, str]:
    """Return ``assembly.default`` from the ``meta`` table of each schema (``"?"`` where it cannot be read).

    Used only to explain a failure: the name tells which assembly a core database really holds, independently of
    the suffix-and-release rule of the organism table.

    Args:
        port: MySQL port of the Ensembl public service.
        schemas: Core database names found on that port (from ``information_schema``, so they are real schema names).

    Returns:
        Mapping of each schema to its declared assembly name.
    """
    names: dict[str, str] = {}
    if not schemas:
        return names
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
                for schema in schemas:
                    try:
                        # Identifiers cannot be bound as parameters; the names come from information_schema.
                        cur.execute(
                            f"SELECT meta_value FROM `{schema}`.meta "  # noqa: S608
                            "WHERE meta_key = 'assembly.default' LIMIT 1"
                        )
                        row = cur.fetchone()
                        value = row[0] if row else "?"
                        names[schema] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    except Exception:
                        names[schema] = "?"
    except Exception as exc:
        return {schema: f"? ({exc.__class__.__name__}: {exc})" for schema in schemas}
    return {schema: names.get(schema, "?") for schema in schemas}


def _describe(schemas_by_port: dict[int, list[str]]) -> dict[int, list[tuple[str, str]]]:
    """Pair each schema with the assembly name its ``meta`` table declares, per port."""
    return {
        port: sorted(_fetch_assembly_names(port=port, schemas=sorted(set(schemas))).items())
        for port, schemas in sorted(schemas_by_port.items())
        if schemas
    }


@pytest.fixture(scope="session")
def live_mysql_core_catalog() -> dict[str, dict[int, dict[str, Any]]]:
    """Return live core schema metadata from Ensembl public MySQL.

    Structure:
        catalog[organism][port] = {
            "schemas": [schema_name, ...],
            "by_assembly": {assembly key: {release: schema_name}},
            "releases_by_assembly": {assembly key: {release, ...}},
            "unconfigured": [schema_name, ...],
        }

    Schema names carry the core-database suffix, not the assembly key; a name is assigned to the assembly whose
    ``DbSuffix`` and ``ReleaseRange`` match (:py:meth:`idtrack.DB.assembly_for_core_db`), so rhesus
    ``macaca_mulatta_core_<r>_10`` is MMUL_1 (key 1) up to release 85 and Mmul_10 (key 10) from release 98.

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

    unreachable = [p for p in ports if not reachability.mysql_port_usable(p) or not _can_connect(p)]
    if unreachable:
        pytest.skip(f"Ensembl public MySQL not reachable on ports {unreachable}.")

    catalog: dict[str, dict[int, dict[str, Any]]] = {}
    for organism in DB.supported_organisms:
        catalog[organism] = {}
        for port in ports:
            schemas = _fetch_core_schemas(organism=organism, port=port)
            pattern = re.compile(
                rf"^{re.escape(organism)}_core_(?P<release>[0-9]+)_(?P<suffix>[0-9]+)(?P<patch>[a-z]*)$"
            )

            by_assembly: dict[int, dict[int, str]] = defaultdict(dict)
            releases_by_assembly: dict[int, set[int]] = defaultdict(set)
            mismatched: list[str] = []
            unconfigured: list[str] = []
            for schema in schemas:
                m = pattern.match(schema)
                if not m:
                    mismatched.append(schema)
                    continue
                rel = int(m.group("release"))
                asm = DB.assembly_for_core_db(organism, m.group("suffix"), rel)
                if asm is None:
                    unconfigured.append(schema)
                    continue
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
                "unconfigured": unconfigured,
            }
    return catalog


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_live_mysql_config_covers_all_discovered_assemblies(live_mysql_core_catalog):
    """Ensure every core database Ensembl exposes on the configured ports belongs to a configured assembly."""
    # All organisms are checked before the test fails; the failure names, per organism and port, each unassigned
    # core database with the assembly name its `meta` table declares.
    problems: dict[str, dict[int, list[tuple[str, str]]]] = {}
    for organism in DB.supported_organisms:
        unconfigured = {
            int(port): list(port_info["unconfigured"])
            for port, port_info in live_mysql_core_catalog[organism].items()
            if port_info["unconfigured"]
        }
        if unconfigured:
            problems[organism] = _describe(unconfigured)
    assert not problems, problems


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_live_mysql_config_ports_match_discovered_ports(live_mysql_core_catalog):
    """Ensure each (organism, assembly) lists exactly the ports where Ensembl hosts it."""
    # All assemblies are checked before the test fails. For each mismatch the failure gives the discovered and the
    # configured ports and, for every port that is discovered but not configured, the core databases found there
    # with the assembly name their `meta` table declares.
    problems: list[dict[str, Any]] = []
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
            configured_ports = set(map(int, cfg["Ports"]))
            if discovered_ports != configured_ports:
                extra = {
                    port: list(live_mysql_core_catalog[organism][port]["by_assembly"][int(assembly)].values())
                    for port in sorted(discovered_ports - configured_ports)
                }
                problems.append(
                    {
                        "organism": organism,
                        "assembly": int(assembly),
                        "discovered_ports": sorted(discovered_ports),
                        "configured_ports": list(cfg["Ports"]),
                        "on_unconfigured_ports": _describe(extra),
                    }
                )
    assert not problems, problems


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
                    match = DB.core_db_name_pattern(organism, int(assembly)).match(str(dm.mysql_database))
                    assert match is not None and int(match.group("release")) == int(rel), (
                        organism,
                        int(assembly),
                        int(rel),
                        dm.mysql_database,
                    )


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
