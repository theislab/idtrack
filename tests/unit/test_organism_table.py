#!/usr/bin/env python3
"""Unit tests for the organism table of `idtrack._db.DB` and its use by `DatabaseManager` (no network).

Covers the table validator, `DB.register_organism` (including the shared core-database suffix rule and an organism
added at run time), core database names with `DbSuffix` and patch letters, and the `ReleaseRange` filter on both
discovery paths (a mocked MySQL listing and a mocked HTTPS dump listing): rhesus key 10 (Mmul_10) must never see a
MMUL_1 release, and key 1 (MMUL_1) never a Mmul_10 release, although both are named `macaca_mulatta_core_<r>_10`.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._ensembl_access import reachability

# Release -> core database name of rhesus macaque, as in Ensembl's dump listings (patch letters up to 62).
RHESUS_CORE_DBS = {
    **{r: f"macaca_mulatta_core_{r}_10{letter}" for r, letter in zip(range(48, 63), "ghiijkklllmnopq")},
    **{r: f"macaca_mulatta_core_{r}_10" for r in range(63, 86)},
    **{r: f"macaca_mulatta_core_{r}_801" for r in range(86, 98)},
    **{r: f"macaca_mulatta_core_{r}_10" for r in range(98, 117)},
}


@contextmanager
def restored_organism_table():
    """Run a block that changes the organism table and put the original back afterwards."""
    table = copy.deepcopy(DB.assembly_mysqlport_priority)
    symbols = dict(DB.symbol_databases)
    try:
        yield
    finally:
        # Restore in place (as `register_organism` changes it) and move the version forward, never back, so that
        # index-cache entries made under the temporary table are not reused.
        DB.assembly_mysqlport_priority.clear()
        DB.assembly_mysqlport_priority.update(table)
        DB.symbol_databases.clear()
        DB.symbol_databases.update(symbols)
        DB.organism_table_version += 1
        DB._rebuild_organism_tables()


def _entry(**overrides):
    entry = {"Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [90, None]}
    entry.update(overrides)
    return entry


class TestShippedTable:
    """The shipped table passes its own validation and declares the five organisms."""

    def test_shipped_table_is_valid(self):
        """Shipped table is valid."""
        DB._validate_organism_table(DB.assembly_mysqlport_priority, DB.symbol_databases)

    def test_five_organisms(self):
        """Five organisms."""
        assert DB.supported_organisms == (
            "homo_sapiens",
            "macaca_fascicularis",
            "macaca_mulatta",
            "mus_musculus",
            "sus_scrofa",
        )

    def test_macaque_entries(self):
        """Macaque entries."""
        rhesus = DB.assembly_mysqlport_priority["macaca_mulatta"]
        assert {
            k: (v["Name"], DB.db_suffix("macaca_mulatta", k), DB.release_range("macaca_mulatta", k))
            for k, v in rhesus.items()
        } == {
            10: ("Mmul_10", "10", (98, None)),
            801: ("Mmul_8.0.1", "801", (86, 97)),
            1: ("MMUL_1", "10", (48, 85)),
        }
        cyno = DB.assembly_mysqlport_priority["macaca_fascicularis"]
        assert {k: (v["Name"], DB.release_range("macaca_fascicularis", k)) for k, v in cyno.items()} == {
            6: ("Macaca_fascicularis_6.0", (103, None)),
            5: ("Macaca_fascicularis_5.0", (91, 102)),
        }
        assert DB.assemblies_by_priority("macaca_mulatta") == [10, 801, 1]
        assert DB.assemblies_by_priority("macaca_fascicularis") == [6, 5]

    def test_every_assembly_has_a_name(self):
        """Every assembly has a name."""
        for organism, entries in DB.assembly_mysqlport_priority.items():
            for key in entries:
                assert DB.assembly_name(organism, key)

    def test_symbol_databases(self):
        """Symbol databases."""
        assert DB.symbol_databases["homo_sapiens"][0] == "HGNC Symbol"
        assert DB.symbol_databases["mus_musculus"][0] == "MGI Symbol"
        for organism in ("sus_scrofa", "macaca_mulatta", "macaca_fascicularis"):
            assert DB.symbol_databases[organism] == ("VGNC Symbol", "HGNC Symbol")

    def test_derived_attributes_cover_new_keys(self):
        """Derived attributes cover new keys."""
        for key in (1, 5, 6, 10, 801):
            assert key in DB.all_assemblies
            assert DB.nts_assembly[key]["gene"] == f"assembly_{key}_ensembl_gene"
            assert DB.nts_assembly_reverse[f"assembly_{key}_ensembl_transcript"] == DB.nts_ensembl["transcript"]
            assert f"assembly_{key}_ensembl_gene" in DB.nts_assembly_gene


class TestValidator:
    """Each rule of `_validate_organism_table` rejects what it should."""

    @staticmethod
    def _check(table, symbols=None):
        symbols = {org: () for org in table} if symbols is None else symbols
        DB._validate_organism_table(table, symbols)

    def test_accepts_minimal_table(self):
        """Accepts minimal table."""
        self._check({"danio_rerio": {11: _entry()}})

    @pytest.mark.parametrize(
        ("table", "message"),
        [
            ({"Danio rerio": {11: _entry()}}, "production name"),
            ({"danio_rerio": {}}, "at least one assembly"),
            ({"danio_rerio": {"11": _entry()}}, "positive integers"),
            ({"danio_rerio": {0: _entry()}}, "positive integers"),
            ({"danio_rerio": {11: _entry(Ports=[1234])}}, "unknown MySQL ports"),
            ({"danio_rerio": {11: _entry(Ports=[])}}, "non-empty list"),
            ({"danio_rerio": {11: _entry(Priority=2)}}, "start at 1"),
            ({"danio_rerio": {11: _entry(), 10: _entry()}}, "unique"),
            ({"danio_rerio": {11: _entry(ReleaseRange=[90])}}, "ReleaseRange"),
            ({"danio_rerio": {11: _entry(ReleaseRange=[90, 80])}}, "None or >= 90"),
            ({"danio_rerio": {11: _entry(DbSuffix="10a")}}, "string of digits"),
            ({"danio_rerio": {11: _entry(Name="")}}, "non-empty string"),
            ({"danio_rerio": {11: _entry(Typo=1)}}, "unknown fields"),
            ({"danio_rerio": {11: {"Ports": [3306], "Priority": 1}}}, "missing required fields"),
        ],
    )
    def test_rejects(self, table, message):
        """Rejects."""
        with pytest.raises(ValueError, match=message):
            self._check(table)

    def test_shared_suffix_requires_disjoint_ranges(self):
        """Shared suffix requires disjoint ranges."""
        ok = {
            "macaca_mulatta": {
                10: _entry(Priority=1, ReleaseRange=[98, None]),
                1: _entry(Priority=2, DbSuffix="10", ReleaseRange=[48, 85]),
            }
        }
        self._check(ok)
        overlapping = copy.deepcopy(ok)
        overlapping["macaca_mulatta"][1]["ReleaseRange"] = [48, 98]
        with pytest.raises(ValueError, match="share the core-database suffix '10'"):
            self._check(overlapping)
        open_ended = copy.deepcopy(ok)
        open_ended["macaca_mulatta"][1]["ReleaseRange"] = [99, None]
        with pytest.raises(ValueError, match="overlap"):
            self._check(open_ended)

    def test_symbol_table_must_match_organisms(self):
        """Symbol table must match organisms."""
        with pytest.raises(ValueError, match="exactly the organisms"):
            self._check({"danio_rerio": {11: _entry()}}, symbols={})
        with pytest.raises(ValueError, match="sequence of names"):
            self._check({"danio_rerio": {11: _entry()}}, symbols={"danio_rerio": "ZFIN"})


class TestRegisterOrganism:
    """`DB.register_organism` adds or replaces an organism and recomputes the derived attributes."""

    def test_adds_runtime_organism(self):
        """Adds runtime organism."""
        with restored_organism_table():
            version = DB.organism_table_version
            DB.register_organism(
                "danio_rerio",
                {11: {"Name": "GRCz11", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [92, None]}},
                symbol_databases=["ZFIN"],
            )
            assert "danio_rerio" in DB.supported_organisms
            assert 11 in DB.all_assemblies
            assert DB.nts_assembly[11]["gene"] == "assembly_11_ensembl_gene"
            assert "assembly_11_ensembl_gene" in DB.nts_assembly_gene
            assert DB.symbol_databases["danio_rerio"] == ("ZFIN",)
            assert DB.organism_table_version == version + 1
            # DatabaseManager now accepts the organism (it fails later, on the missing directory, not on the name).
            with pytest.raises(ValueError, match="existing directory"):
                DatabaseManager(organism="danio_rerio", form="gene", local_repository="/nonexistent/dir")
        assert "danio_rerio" not in DB.supported_organisms
        assert 11 not in DB.all_assemblies

    def test_rejected_entry_leaves_table_unchanged(self):
        """Rejected entry leaves table unchanged."""
        before = copy.deepcopy(DB.assembly_mysqlport_priority)
        with pytest.raises(ValueError, match="start at 1"):
            DB.register_organism("danio_rerio", {11: _entry(Priority=3)}, symbol_databases=["ZFIN"])
        assert DB.assembly_mysqlport_priority == before
        assert "danio_rerio" not in DB.supported_organisms

    def test_replaces_organism_and_drops_unused_keys(self):
        """Replaces organism and drops unused keys."""
        with restored_organism_table():
            DB.register_organism("macaca_fascicularis", {7: _entry(ReleaseRange=[117, None])}, ["VGNC Symbol"])
            assert list(DB.assembly_mysqlport_priority["macaca_fascicularis"]) == [7]
            assert 7 in DB.all_assemblies and 5 not in DB.all_assemblies and 6 not in DB.all_assemblies

    def test_replacing_without_symbol_databases_keeps_them(self):
        """Replacing an organism without `symbol_databases` keeps its symbol databases; references stay current."""
        table_ref = DB.assembly_mysqlport_priority
        symbols_ref = DB.symbol_databases
        with restored_organism_table():
            entries = copy.deepcopy(DB.assembly_mysqlport_priority["homo_sapiens"])
            entries[38]["Ports"] = [5306, 3306]
            DB.register_organism("homo_sapiens", entries)
            assert DB.symbol_databases["homo_sapiens"] == ("HGNC Symbol",)
            assert table_ref["homo_sapiens"][38]["Ports"] == [5306, 3306]
            assert symbols_ref is DB.symbol_databases

    def test_new_organism_needs_symbol_databases(self):
        """A new organism must name its symbol databases (an empty list is allowed)."""
        with pytest.raises(ValueError, match="pass `symbol_databases`"):
            DB.register_organism("danio_rerio", {11: _entry()})
        assert "danio_rerio" not in DB.supported_organisms
        with restored_organism_table():
            DB.register_organism("danio_rerio", {11: _entry()}, symbol_databases=[])
            assert DB.symbol_databases["danio_rerio"] == ()

    def test_unregistered_organism_error_points_to_register(self, tmp_path):
        """Unregistered organism error points to register."""
        with pytest.raises(NotImplementedError, match="register_organism"):
            DatabaseManager(organism="danio_rerio", form="gene", local_repository=str(tmp_path))


class TestCoreDatabaseNames:
    """Names are built and matched with `DbSuffix`; patch letters belong to the assembly."""

    def test_pattern_uses_suffix_and_patch_letters(self):
        """Pattern uses suffix and patch letters."""
        pattern = DB.core_db_name_pattern("macaca_mulatta", 1)
        match = pattern.match("macaca_mulatta_core_48_10g")
        assert match is not None and match.group("release") == "48" and match.group("patch") == "g"
        assert pattern.match("macaca_mulatta_core_70_10")
        assert not pattern.match("macaca_mulatta_core_90_801")
        assert not pattern.match("macaca_mulatta_core_70_101")
        assert DB.core_db_name_pattern("macaca_mulatta", 801).match("macaca_mulatta_core_90_801")

    def test_assembly_for_core_db(self):
        """Assembly for core db."""
        assert DB.assembly_for_core_db("macaca_mulatta", "10", 48) == 1
        assert DB.assembly_for_core_db("macaca_mulatta", "10", 85) == 1
        assert DB.assembly_for_core_db("macaca_mulatta", "801", 86) == 801
        assert DB.assembly_for_core_db("macaca_mulatta", "10", 98) == 10
        assert DB.assembly_for_core_db("macaca_mulatta", "10", 116) == 10
        assert DB.assembly_for_core_db("macaca_mulatta", "10", 90) is None
        assert DB.assembly_for_core_db("homo_sapiens", "37", 100) == 37
        assert DB.assembly_for_core_db("homo_sapiens", "99", 100) is None

    def test_suffix_defaults_to_key(self):
        """Suffix defaults to key."""
        assert DB.db_suffix("homo_sapiens", 38) == "38"
        assert DB.db_suffix("macaca_mulatta", 801) == "801"
        assert DB.db_suffix("macaca_mulatta", 1) == "10"


def _fake_mysql_connect(names: list[str]):
    """Return a `pymysql.connect` replacement whose cursor answers the information_schema query with *names*."""

    def _connect(**_kwargs):
        cursor = MagicMock()
        cursor.fetchall.return_value = [(n,) for n in names]
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cursor
        return conn

    return _connect


class TestReleaseRangeOnMySQLListing:
    """The MySQL probe keeps only releases inside the assembly's `ReleaseRange`."""

    @pytest.fixture(autouse=True)
    def _mysql_reachable(self, monkeypatch):
        monkeypatch.setattr(reachability, "mysql_port_usable", lambda port: True)

    @pytest.mark.parametrize(
        ("assembly", "expected"),
        [
            (10, list(range(98, 117))),
            (1, list(range(48, 86))),
            (801, list(range(86, 98))),
        ],
    )
    def test_rhesus_assemblies_get_only_their_releases(self, assembly, expected):
        """Rhesus assemblies get only their releases."""
        names = list(RHESUS_CORE_DBS.values())
        with patch("idtrack._database_manager.pymysql.connect", side_effect=_fake_mysql_connect(names)):
            releases_by_port, db_by_port_release, errors = DatabaseManager._probe_mysql_core_schemas_by_port(
                organism="macaca_mulatta", genome_assembly=assembly, ports=[3306]
            )
        assert not errors
        assert sorted(releases_by_port[3306]) == expected
        assert db_by_port_release[3306][expected[0]] == RHESUS_CORE_DBS[expected[0]]

    def test_like_pattern_uses_the_suffix(self):
        """Like pattern uses the suffix."""
        captured = {}

        def _connect(**_kwargs):
            cursor = MagicMock()

            def _execute(sql, params):
                captured["like"] = params[0]

            cursor.execute.side_effect = _execute
            cursor.fetchall.return_value = []
            conn = MagicMock()
            conn.__enter__.return_value = conn
            conn.cursor.return_value.__enter__.return_value = cursor
            return conn

        with patch("idtrack._database_manager.pymysql.connect", side_effect=_connect):
            DatabaseManager._probe_mysql_core_schemas_by_port(
                organism="macaca_mulatta", genome_assembly=1, ports=[3306]
            )
        assert captured["like"] == "macaca\\_mulatta\\_core\\_%\\_10%"


class TestReleaseRangeOnHTTPSListing:
    """The HTTPS scan asks only for releases inside `ReleaseRange` and names directories with `DbSuffix`."""

    @staticmethod
    def _fake_get(url, *args, **kwargs):
        response = MagicMock()
        response.__enter__.return_value = response
        name = url.rstrip("/").rsplit("/", 1)[-1]
        response.status_code = 200 if name in RHESUS_CORE_DBS.values() else 404
        if url.endswith("/mysql/"):
            release = int(url.split("/release-")[1].split("/")[0])
            db = RHESUS_CORE_DBS.get(release)
            response.text = f'<a href="{db}/">{db}/</a>' if db else ""
        return response

    @pytest.mark.parametrize(
        ("assembly", "expected"),
        [
            (10, list(range(98, 117))),
            (1, list(range(48, 86))),
            (801, list(range(86, 98))),
        ],
    )
    def test_rhesus_assemblies_get_only_their_releases(self, monkeypatch, assembly, expected):
        """Rhesus assemblies get only their releases."""
        monkeypatch.setattr(DatabaseManager, "_ensembl_latest_release_cache", 116)
        with patch("idtrack._database_manager.requests.get", side_effect=self._fake_get) as fake_get:
            index = DatabaseManager._get_core_db_index_from_ftp(
                organism="macaca_mulatta", genome_assembly=assembly, ports=[3306, 5306]
            )
        assert index["releases"] == expected
        assert index["db_for_release"][expected[0]] == RHESUS_CORE_DBS[expected[0]]
        first, last = DB.release_range("macaca_mulatta", assembly)
        asked = {int(c.args[0].split("/release-")[1].split("/")[0]) for c in fake_get.call_args_list}
        assert min(asked) >= max(first, 48) and max(asked) <= (last or 116)


class TestIndexCacheFollowsTable:
    """The core-database index cache is keyed by the table version, so `register_organism` invalidates it."""

    def test_cache_key_contains_table_version(self, monkeypatch):
        """Cache key contains table version."""
        built = []

        def _fake_ftp(cls, *, organism, genome_assembly, ports):
            built.append(DB.organism_table_version)
            return {"releases": [100], "db_for_release": {100: "x"}, "port_for_release": {100: 3306}}

        monkeypatch.setattr(reachability, "mysql_port_usable", lambda port: False)
        monkeypatch.setattr(DatabaseManager, "_get_core_db_index_from_ftp", classmethod(_fake_ftp))
        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
        with restored_organism_table():
            DatabaseManager._get_core_db_index(organism="macaca_fascicularis", genome_assembly=6)
            DatabaseManager._get_core_db_index(organism="macaca_fascicularis", genome_assembly=6)
            assert len(built) == 1
            DB.register_organism(
                "macaca_fascicularis",
                copy.deepcopy(DB.assembly_mysqlport_priority["macaca_fascicularis"]),
                DB.symbol_databases["macaca_fascicularis"],
            )
            DatabaseManager._get_core_db_index(organism="macaca_fascicularis", genome_assembly=6)
            assert len(built) == 2


class TestDefaultAssemblyFollowsWindow:
    """Without `genome_assembly`, the primary assembly is the highest-priority one with a release in the window."""

    @pytest.fixture(autouse=True)
    def _fake_indexes(self, monkeypatch):
        def _index(cls, *, organism, genome_assembly):
            first, last = DB.release_range(organism, genome_assembly)
            releases = list(range(max(first, 48), (last or 116) + 1))
            return {
                "ports": (3306,),
                "releases": releases,
                "port_for_release": {r: 3306 for r in releases},
                "db_for_release": {
                    r: f"{organism}_core_{r}_{DB.db_suffix(organism, genome_assembly)}" for r in releases
                },
                "releases_on_mysql": [],
            }

        monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(_index))

    @pytest.mark.parametrize(
        ("organism", "window", "assembly", "release"),
        [
            ("macaca_mulatta", (85, 86), 801, 86),
            ("macaca_mulatta", (97, 98), 10, 98),
            ("macaca_mulatta", (60, 70), 1, 70),
            ("macaca_fascicularis", (102, 103), 6, 103),
            ("macaca_fascicularis", (95, 100), 5, 100),
            ("mus_musculus", (48, 100), 38, 100),
            ("homo_sapiens", (48, 116), 38, 116),
        ],
    )
    def test_primary_assembly(self, tmp_path, organism, window, assembly, release):
        """Primary assembly."""
        dm = DatabaseManager(
            organism=organism,
            form="gene",
            local_repository=str(tmp_path),
            ignore_before=window[0],
            ignore_after=window[1],
        )
        assert (dm.genome_assembly, dm.ensembl_release) == (assembly, release)
        # Across a handoff, the release walk uses each release's own assembly.
        lo = dm.change_release_auto_assembly(window[0])
        assert lo.genome_assembly == DB.assembly_for_core_db(
            organism, DB.db_suffix(organism, lo.genome_assembly), window[0]
        )


class TestAvailableDatabasesInBothModes:
    """`create_available_databases` lists the same core databases over MySQL and over HTTPS (mocked listings)."""

    @pytest.mark.parametrize("mode", ["auto", "https"])
    def test_same_list(self, monkeypatch, mode):
        """Both modes give every rhesus core database of the configured ranges, with patch letters."""
        monkeypatch.setattr(DB, "ensembl_access", mode)
        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
        monkeypatch.setattr(DatabaseManager, "_ensembl_latest_release_cache", 116)
        reachability.reset()
        monkeypatch.setattr(type(reachability), "_probe", staticmethod(lambda host, port, timeout_s: True))
        names = list(RHESUS_CORE_DBS.values())
        connect = MagicMock(side_effect=_fake_mysql_connect(names))
        with patch("idtrack._database_manager.pymysql.connect", connect), patch(
            "idtrack._database_manager.requests.get", side_effect=TestReleaseRangeOnHTTPSListing._fake_get
        ):
            dm = DatabaseManager.__new__(DatabaseManager)
            dm.organism = "macaca_mulatta"
            dm.log = MagicMock()
            listed = dm.create_available_databases()["available_databases"].tolist()
        reachability.reset()
        assert listed == sorted(RHESUS_CORE_DBS.values())
        assert "macaca_mulatta_core_48_10g" in listed and "macaca_mulatta_core_90_801" in listed
        assert connect.called is (mode == "auto")
