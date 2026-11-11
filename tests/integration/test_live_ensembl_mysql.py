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

import pprint
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

    The name tells which assembly a core database really holds, independently of the suffix-and-release rule of
    the organism table. The schemas are read in batches of ``UNION ALL`` queries over one connection.

    Args:
        port: MySQL port of the Ensembl public service.
        schemas: Core database names found on that port (from ``information_schema``, so they are real schema names).

    Returns:
        Mapping of each schema to its declared assembly name.
    """
    names: dict[str, str] = {}
    if not schemas:
        return names

    def _select(schema: str) -> str:
        # Identifiers cannot be bound as parameters; the names come from information_schema.
        return (
            f"(SELECT {schema!r}, meta_value FROM `{schema}`.meta "  # noqa: S608
            "WHERE meta_key = 'assembly.default' LIMIT 1)"
        )

    try:
        with pymysql.connect(
            host=DB.mysql_host,
            user=DB.myqsl_user,
            password=DB.mysql_togo,
            port=int(port),
            connect_timeout=max(DB.connection_timeout, 30),
            read_timeout=max(DB.reading_timeout, 120),
        ) as conn:
            with conn.cursor() as cur:
                for start in range(0, len(schemas), 50):
                    batch = schemas[start : start + 50]
                    rows: list[tuple[Any, ...]] = []
                    try:
                        cur.execute(" UNION ALL ".join(_select(schema) for schema in batch))
                        rows.extend(cur.fetchall())
                    except Exception:
                        rows = []
                        for schema in batch:  # one unreadable schema must not hide the others
                            try:
                                cur.execute(_select(schema))
                                rows.extend(cur.fetchall())
                            except Exception:
                                names[schema] = "?"
                    for schema, value in rows:
                        schema = schema.decode("utf-8") if isinstance(schema, bytes) else str(schema)
                        names[schema] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    except Exception as exc:
        return {schema: f"? ({exc.__class__.__name__}: {exc})" for schema in schemas}
    return {schema: names.get(schema, "?") for schema in schemas}


def _fail(title: str, problems: Any) -> None:
    """Fail with the whole of *problems* (pytest shortens an assertion's object, never a failure message)."""
    pytest.fail(f"{title}\n{pprint.pformat(problems, width=120)}", pytrace=False)


@pytest.fixture(scope="session")
def live_mysql_core_catalog() -> dict[str, dict[int, dict[str, Any]]]:
    """Return live core schema metadata from Ensembl public MySQL.

    Structure:
        catalog[organism][port] = {
            "schemas": [schema_name, ...],
            "by_assembly": {assembly key: {release: schema_name}},
            "releases_by_assembly": {assembly key: {release, ...}},
            "unconfigured": [schema_name, ...],
            "archive_copies": [schema_name, ...],
            "assembly_names": {schema_name: assembly.default of its meta table},
        }

    Schema names carry the core-database suffix, not the assembly key; a name is assigned to the assembly whose
    ``DbSuffix`` and ``ReleaseRange`` match (:py:meth:`idtrack.DB.assembly_for_core_db`), so rhesus
    ``macaca_mulatta_core_<r>_10`` is MMUL_1 (key 1) up to release 85 and Mmul_10 (key 10) from release 98.
    On a port that :py:attr:`idtrack.DB.mysql_port_assembly` reserves for one assembly (3337: human GRCh37), every
    other core database is an archive copy (an older assembly carried forward under a later release number): it
    is listed under ``archive_copies`` and attributed to no assembly, as idtrack's discovery does.

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
            owner = DB.mysql_port_assembly.get(port)
            pattern = re.compile(
                rf"^{re.escape(organism)}_core_(?P<release>[0-9]+)_(?P<suffix>[0-9]+)(?P<patch>[a-z]*)$"
            )

            by_assembly: dict[int, dict[int, str]] = defaultdict(dict)
            releases_by_assembly: dict[int, set[int]] = defaultdict(set)
            mismatched: list[str] = []
            unconfigured: list[str] = []
            archive_copies: list[str] = []
            for schema in schemas:
                m = pattern.match(schema)
                if not m:
                    mismatched.append(schema)
                    continue
                rel = int(m.group("release"))
                asm = DB.assembly_for_core_db(organism, m.group("suffix"), rel)
                if owner is not None and (asm is None or owner != (organism, asm)):
                    archive_copies.append(schema)
                    continue
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
                "archive_copies": archive_copies,
                "assembly_names": _fetch_assembly_names(port=port, schemas=sorted(schemas)),
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
            problems[organism] = {
                port: [(schema, live_mysql_core_catalog[organism][port]["assembly_names"][schema]) for schema in names]
                for port, names in unconfigured.items()
            }
    if problems:
        _fail("Core databases that no configured assembly claims (schema, meta assembly.default):", problems)


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
                    port: [
                        (schema, live_mysql_core_catalog[organism][port]["assembly_names"][schema])
                        for schema in sorted(
                            live_mysql_core_catalog[organism][port]["by_assembly"][int(assembly)].values()
                        )
                    ]
                    for port in sorted(discovered_ports - configured_ports)
                }
                problems.append(
                    {
                        "organism": organism,
                        "assembly": int(assembly),
                        "discovered_ports": sorted(discovered_ports),
                        "configured_ports": list(cfg["Ports"]),
                        "on_unconfigured_ports": extra,
                    }
                )
    if problems:
        _fail("Assemblies whose configured ports differ from the ports that hold them:", problems)


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_live_core_databases_hold_the_assembly_the_table_names(live_mysql_core_catalog):
    """Ensure every core database the table attributes to an assembly declares that assembly in its `meta` table."""
    # The suffix-and-release rule (`DB.assembly_for_core_db`) is checked against the truth, `assembly.default`, for
    # every database on every port idtrack may read for the organism.
    problems: list[tuple[str, int, int, str, str, str | None]] = []
    for organism in DB.supported_organisms:
        for port, port_info in sorted(live_mysql_core_catalog[organism].items()):
            for assembly, by_release in sorted(port_info["by_assembly"].items()):
                expected = DB.assembly_name(organism, int(assembly))
                for release, schema in sorted(by_release.items()):
                    declared = port_info["assembly_names"].get(schema, "?")
                    if declared != expected:
                        problems.append((organism, int(port), int(release), schema, declared, expected))
    if problems:
        _fail("(organism, port, release, schema, meta assembly.default, table Name) that disagree:", problems)


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_live_archive_copies_are_older_assemblies_carried_forward(live_mysql_core_catalog):
    """Ensure every archive copy on a reserved port is a configured assembly carried forward, never a new one."""
    # An archive copy declares, in its `meta` table, the name of one of the organism's configured assemblies, and its
    # release is not before that assembly's first release (copies are carried forward, never backward). Such copies
    # are expected (rhesus MMUL_1 as `_79_10` to `_99_10`, pig Sscrofa10.2 after 89, mouse GRCm38) and attributed
    # to no assembly; anything else on a reserved port fails, with the whole listing.
    listing: dict[str, dict[int, list[tuple[str, str]]]] = {}
    problems: list[tuple[str, int, str, str]] = []
    for organism in DB.supported_organisms:
        first_release_by_name = {
            DB.assembly_name(organism, key): DB.release_range(organism, key)[0]
            for key in DB.assembly_mysqlport_priority[organism]
        }
        for port, port_info in sorted(live_mysql_core_catalog[organism].items()):
            for schema in sorted(port_info["archive_copies"]):
                name = port_info["assembly_names"].get(schema, "?")
                listing.setdefault(organism, {}).setdefault(int(port), []).append((schema, name))
                release = int(schema.split("_core_", 1)[1].split("_", 1)[0])
                first = first_release_by_name.get(name)
                if first is None or release < first:
                    problems.append((organism, int(port), schema, name))
    if problems:
        _fail(
            "Databases on reserved ports that are no configured assembly carried forward; full listing follows:",
            {"problems": problems, "archive_copies": listing},
        )


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.database
def test_live_mysql_releases_are_releases_of_the_https_dumps(live_mysql_core_catalog):
    """Ensure each assembly's MySQL releases (archive copies excluded) are among the releases of its HTTPS dumps."""
    # In mode "auto" an assembly's releases are the union of both listings and in mode "https" those of the dumps,
    # so the two modes give the same releases exactly when nothing is on MySQL that the dumps lack. The dumps have no
    # archive copies, which is why those are attributed to no assembly.
    problems: list[tuple[str, int, list[int]]] = []
    for organism in DB.supported_organisms:
        for assembly, cfg in DB.assembly_mysqlport_priority[organism].items():
            on_mysql = {
                int(rel)
                for port_info in live_mysql_core_catalog[organism].values()
                for rel in port_info["releases_by_assembly"].get(int(assembly), set())
            }
            if not on_mysql:
                continue
            https = DatabaseManager._get_core_db_index_from_ftp(
                organism=organism, genome_assembly=int(assembly), ports=[int(p) for p in cfg["Ports"]]
            )
            missing = sorted(on_mysql - {int(r) for r in https["releases"]})
            if missing:
                problems.append((organism, int(assembly), missing))
    if problems:
        _fail("(organism, assembly, releases on MySQL that the HTTPS dumps lack):", problems)


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
