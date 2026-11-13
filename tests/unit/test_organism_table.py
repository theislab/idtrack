#!/usr/bin/env python3
"""Unit tests for the organism table of `idtrack._db.DB` and its use by `DatabaseManager` (no network).

Covers the table validator, `DB.register_organism` (including the shared core-database suffix rule and an organism
added at run time), core database names with `DbSuffix` and patch letters, and the `ReleaseRange` filter on both
discovery paths (a mocked MySQL listing and a mocked HTTPS dump listing): rhesus key 10 (Mmul_10) must never see a
MMUL_1 release, and key 1 (MMUL_1) never a Mmul_10 release, although both are named `macaca_mulatta_core_<r>_10`.
"""

import copy
import time
from contextlib import contextmanager
from types import MappingProxyType
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
    aliases = dict(DB.organism_aliases)
    try:
        yield
    finally:
        DB.organism_aliases.clear()
        DB.organism_aliases.update(aliases)
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
            ({"sus_scrofa": {102: _entry(Ports=[3306, 5306, 3337])}}, r"serve only \[\('homo_sapiens', 37\)\]"),
            ({"homo_sapiens": {38: _entry(Ports=[3306, 3337])}}, r"serve only \[\('homo_sapiens', 37\)\]"),
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

    def test_grch37_archive_port_is_human_grch37_only(self):
        """Port 3337 (the GRCh37 archive) is accepted for human GRCh37 and listed by no other shipped assembly."""
        self._check({"homo_sapiens": {37: _entry(Ports=[3337, 3306, 5306])}})
        assert DB.mysql_port_assembly == {3337: ("homo_sapiens", 37)}
        for organism, entries in DB.assembly_mysqlport_priority.items():
            for key, entry in entries.items():
                assert (3337 in entry["Ports"]) == ((organism, key) == ("homo_sapiens", 37)), (organism, key)

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

    def test_read_only_mappings_are_stored_as_independent_dicts(self):
        """Entries given as read-only mappings are accepted and stored as plain dicts, detached from the input."""
        ports = [3306, 5306]
        entry = MappingProxyType({"Name": "GRCz11", "Ports": ports, "Priority": 1, "ReleaseRange": [92, None]})
        with restored_organism_table():
            DB.register_organism("danio_rerio", MappingProxyType({11: entry}), symbol_databases=["ZFIN"])
            stored = DB.assembly_mysqlport_priority["danio_rerio"]
            assert type(stored) is dict and type(stored[11]) is dict
            assert stored[11] == dict(entry)
            ports.append(3337)
            assert stored[11]["Ports"] == [3306, 5306]

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

    def test_aliases_are_declared_with_the_organism(self):
        """Decision 22: an alias belongs to one organism of the table and is registered with it."""
        assert DB.organism_aliases == {
            "homo_sapiens": ("human",),
            "mus_musculus": ("mouse",),
            "sus_scrofa": ("pig",),
            "macaca_mulatta": ("rhesus",),
            "macaca_fascicularis": ("cynomolgus",),
        }
        with restored_organism_table():
            DB.register_organism("danio_rerio", {11: _entry()}, symbol_databases=["ZFIN"], aliases=["zebrafish"])
            assert DB.organism_aliases["danio_rerio"] == ("zebrafish",)
            from idtrack._verify_organism import VerifyOrganism

            assert VerifyOrganism("Zebrafish").get_formal_name() == "danio_rerio"
            # replacing the organism without `aliases` keeps them
            DB.register_organism("danio_rerio", {11: _entry(ReleaseRange=[95, None])})
            assert DB.organism_aliases["danio_rerio"] == ("zebrafish",)
        assert "danio_rerio" not in DB.organism_aliases
        for bad, message in (
            (["pig"], "declared twice"),
            (["Zebrafish"], "lower-case"),
            (["homo_sapiens"], "not an organism name"),
            ([""], "non-empty"),
            ("zebrafish", "sequence of names"),
        ):
            before = (copy.deepcopy(DB.assembly_mysqlport_priority), dict(DB.organism_aliases))
            with pytest.raises(ValueError, match=message):
                DB.register_organism("danio_rerio", {11: _entry()}, symbol_databases=["ZFIN"], aliases=bad)
            assert (DB.assembly_mysqlport_priority, DB.organism_aliases) == before

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


class TestAssemblyKey:
    """One rule for every place that takes an assembly key from a caller."""

    @pytest.mark.parametrize("good", [38, "38", " 38 ", 38.0])
    def test_integral_values_are_keys(self, good):
        """Integers, integral floats and digit strings name the assembly."""
        assert DB.assembly_key(good) == 38

    def test_numpy_integers_and_integral_floats_are_keys(self):
        """Scalars of NumPy are accepted as their Python counterparts."""
        import numpy as np

        assert DB.assembly_key(np.int64(38)) == 38
        assert DB.assembly_key(np.float64(38.0)) == 38
        assert type(DB.assembly_key(np.int64(38))) is int

    def test_values_that_only_look_like_a_key_are_refused(self):
        """A boolean is not rhesus MMUL_1 (key 1) and `38.9` is not GRCh38."""
        import numpy as np

        for bad in (True, False, np.True_, 38.9, np.float64(38.9), "38a", "", None, float("nan"), [38]):
            with pytest.raises(ValueError, match="integer key"):
                DB.assembly_key(bad)

    @pytest.mark.parametrize("accessor", ["release_range", "db_suffix", "assembly_name", "core_db_name_pattern"])
    @pytest.mark.parametrize("bad", [True, 1.5])
    def test_accessors_refuse_booleans_and_fractions(self, accessor, bad):
        """An accessor asked for assembly `True` raised nothing before and answered for MMUL_1 (key 1)."""
        with pytest.raises(ValueError, match="integer key"):
            getattr(DB, accessor)("macaca_mulatta", bad)

    def test_accessors_accept_the_same_spellings_of_a_key(self):
        """A key given as text or as an integral float reads the same entry."""
        assert DB.release_range("macaca_mulatta", "801") == DB.release_range("macaca_mulatta", 801.0) == (86, 97)
        assert DB.db_suffix("macaca_mulatta", "1") == "10"
        assert DB.db_suffix("homo_sapiens", 38.0) == "38"

    def test_unknown_key_names_the_configured_assemblies(self):
        """A well-formed key that the organism does not have is reported with the keys it has."""
        with pytest.raises(ValueError, match=r"not configured for organism 'sus_scrofa'.*\[9, 102, 111\]"):
            DB.release_range("sus_scrofa", 38)


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
            response.status_code = 200
            release = int(url.split("/release-")[1].split("/")[0])
            db = RHESUS_CORE_DBS.get(release)
            response.text = f'<a href="{db}/">{db}/</a>' if db else ""
        response.ok = response.status_code == 200
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
        monkeypatch.setattr(DatabaseManager, "_ensembl_latest_release_cache", (116, time.monotonic()))
        fake_get = MagicMock(side_effect=self._fake_get)
        with patch("idtrack._ensembl_access._session", return_value=MagicMock(get=fake_get)):
            index = DatabaseManager._get_core_db_index_from_ftp(
                organism="macaca_mulatta", genome_assembly=assembly, ports=[3306, 5306]
            )
        assert index["releases"] == expected
        assert index["db_for_release"][expected[0]] == RHESUS_CORE_DBS[expected[0]]
        first, last = DB.release_range("macaca_mulatta", assembly)
        asked = {int(c.args[0].split("/release-")[1].split("/")[0]) for c in fake_get.call_args_list}
        assert min(asked) >= max(first, 48) and max(asked) <= (last or 116)


# What Ensembl's public MySQL showed in November 2026 (run 43109865432): the GRCh37 archive port 3337 holds rhesus
# MMUL_1 under the archive's release numbers 79 to 99 (`meta` assembly.default "MMUL_1" for all of them, including
# `_98_10` and `_99_10`, which the main ports name Mmul_10), and pig Sscrofa10.2 beyond its last main release 89.
ARCHIVE_PORT_CORE_DBS = {
    "macaca_mulatta": [f"macaca_mulatta_core_{r}_10" for r in range(79, 100)],
    "sus_scrofa": [f"sus_scrofa_core_{r}_102" for r in range(79, 100)],
}
PIG_MAIN_CORE_DBS = [f"sus_scrofa_core_{r}_102" for r in range(67, 90)] + [
    f"sus_scrofa_core_{r}_111" for r in range(90, 117)
]


def _fake_mysql_connect_by_port(names_by_port: dict[int, list[str]], connected: list[int]):
    """Return a `pymysql.connect` replacement answering the information_schema query per port."""

    def _connect(**kwargs):
        port = int(kwargs["port"])
        connected.append(port)
        return _fake_mysql_connect(names_by_port.get(port, []))(**kwargs)

    return _connect


class TestArchivePortCopies:
    """Port 3337 is read for human GRCh37 only, so its archive copies are never attributed to an assembly."""

    @pytest.fixture(autouse=True)
    def _mysql_reachable(self, monkeypatch):
        monkeypatch.setattr(reachability, "mysql_port_usable", lambda port: True)

    @pytest.mark.parametrize(
        ("organism", "assembly", "main", "expected"),
        [
            ("macaca_mulatta", 10, list(RHESUS_CORE_DBS.values()), list(range(98, 117))),
            ("macaca_mulatta", 801, list(RHESUS_CORE_DBS.values()), list(range(86, 98))),
            ("macaca_mulatta", 1, list(RHESUS_CORE_DBS.values()), list(range(48, 86))),
            ("sus_scrofa", 102, PIG_MAIN_CORE_DBS, list(range(67, 90))),
            ("sus_scrofa", 111, PIG_MAIN_CORE_DBS, list(range(90, 117))),
        ],
    )
    def test_probe_never_reads_the_archive_port(self, organism, assembly, main, expected):
        """Even with 3337 passed in, the probe reads only the main ports, so `_98_10` on 3337 never becomes Mmul_10."""
        listing = {3306: main, 5306: main, 3337: ARCHIVE_PORT_CORE_DBS[organism]}
        connected: list[int] = []
        with patch(
            "idtrack._database_manager.pymysql.connect",
            side_effect=_fake_mysql_connect_by_port(listing, connected),
        ):
            releases_by_port, db_by_port_release, errors = DatabaseManager._probe_mysql_core_schemas_by_port(
                organism=organism, genome_assembly=assembly, ports=[3306, 5306, 3337]
            )
        assert 3337 not in connected and not errors
        assert sorted(releases_by_port[3306]) == expected and not releases_by_port[3337]
        assert not db_by_port_release[3337]

    def test_archive_port_is_read_for_human_grch37_only(self):
        """GRCh37 reads its releases on 3337; GRCh38 never connects to 3337 even when it is passed in."""
        grch37 = [f"homo_sapiens_core_{r}_37" for r in range(79, 117)]
        listing = {3337: grch37, 3306: [f"homo_sapiens_core_{r}_38" for r in range(76, 117)]}
        connected: list[int] = []
        with patch(
            "idtrack._database_manager.pymysql.connect",
            side_effect=_fake_mysql_connect_by_port(listing, connected),
        ):
            releases_37, _, _ = DatabaseManager._probe_mysql_core_schemas_by_port(
                organism="homo_sapiens", genome_assembly=37, ports=[3337]
            )
            assert connected == [3337]
            releases_38, _, _ = DatabaseManager._probe_mysql_core_schemas_by_port(
                organism="homo_sapiens", genome_assembly=38, ports=[3306, 3337]
            )
        assert connected == [3337, 3306]
        assert sorted(releases_37[3337]) == list(range(79, 117))
        assert sorted(releases_38[3306]) == list(range(76, 117)) and not releases_38[3337]

    def test_both_access_modes_give_the_same_releases(self, monkeypatch):
        """The MySQL listing (main ports) and the HTTPS listing give each rhesus assembly the same releases."""
        monkeypatch.setattr(DatabaseManager, "_ensembl_latest_release_cache", (116, time.monotonic()))
        listing = {3306: list(RHESUS_CORE_DBS.values()), 5306: list(RHESUS_CORE_DBS.values())}
        listing[3337] = ARCHIVE_PORT_CORE_DBS["macaca_mulatta"]
        for assembly in (10, 801, 1):
            connected: list[int] = []
            with patch(
                "idtrack._database_manager.pymysql.connect",
                side_effect=_fake_mysql_connect_by_port(listing, connected),
            ):
                releases_by_port, _, _ = DatabaseManager._probe_mysql_core_schemas_by_port(
                    organism="macaca_mulatta",
                    genome_assembly=assembly,
                    ports=DB.assembly_mysqlport_priority["macaca_mulatta"][assembly]["Ports"],
                )
            session = MagicMock(get=MagicMock(side_effect=TestReleaseRangeOnHTTPSListing._fake_get))
            with patch("idtrack._ensembl_access._session", return_value=session):
                https_index = DatabaseManager._get_core_db_index_from_ftp(
                    organism="macaca_mulatta", genome_assembly=assembly, ports=[3306, 5306]
                )
            mysql_releases = sorted({r for rels in releases_by_port.values() for r in rels})
            assert mysql_releases == https_index["releases"], assembly
            assert 3337 not in connected


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


class TestFirstRelease:
    """Decision 22: one default start of the release window, the organism's first release in the table."""

    def test_first_release_is_over_all_assemblies_of_the_organism(self):
        """The first release of an organism is that of its earliest assembly."""
        assert DB.first_release("homo_sapiens") == 48
        assert DB.first_release("mus_musculus") == 48
        assert DB.first_release("sus_scrofa") == 56
        assert DB.first_release("macaca_mulatta") == 48
        assert DB.first_release("macaca_fascicularis") == 91
        with pytest.raises(ValueError, match="not configured"):
            DB.first_release("danio_rerio")

    @pytest.mark.parametrize(
        ("organism", "assembly", "first"),
        [("sus_scrofa", 111, 56), ("macaca_fascicularis", 6, 91), ("homo_sapiens", 38, 48), ("homo_sapiens", 37, 48)],
    )
    def test_manager_and_api_start_at_the_first_release_of_the_organism(
        self, tmp_path, monkeypatch, organism, assembly, first
    ):
        """`DatabaseManager` without `ignore_before`, and the manager the API makes, have the same window."""
        from idtrack import API

        def _index(cls, *, organism, genome_assembly, **_):
            low, high = DB.release_range(organism, genome_assembly)
            releases = list(range(low, (high or 116) + 1))
            names = {r: f"{organism}_core_{r}_{DB.db_suffix(organism, genome_assembly)}" for r in releases}
            return {
                "ports": (3306,),
                "releases": releases,
                "port_for_release": {r: 3306 for r in releases},
                "db_for_release": names,
                "releases_on_mysql": [],
            }

        monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(_index))
        direct = DatabaseManager(
            organism=organism, form="gene", local_repository=str(tmp_path), genome_assembly=assembly
        )
        through_api = API(str(tmp_path)).get_database_manager(organism, 115, genome_assembly=assembly)
        assert direct.ignore_before == through_api.ignore_before == first
        assert API(str(tmp_path)).get_database_manager(organism, 115, ignore_before=100).ignore_before == 100


class TestDefaultAssemblyFollowsWindow:
    """Without `genome_assembly`, the primary assembly is the highest-priority one with a release in the window."""

    @pytest.fixture(autouse=True)
    def _fake_indexes(self, monkeypatch):
        def _index(cls, *, organism, genome_assembly, **_):
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
        monkeypatch.setattr(DatabaseManager, "_ensembl_latest_release_cache", (116, time.monotonic()))
        reachability.reset()
        monkeypatch.setattr(type(reachability), "_probe", staticmethod(lambda host, port, timeout_s: True))
        names = list(RHESUS_CORE_DBS.values())
        connect = MagicMock(side_effect=_fake_mysql_connect(names))
        session = MagicMock(get=MagicMock(side_effect=TestReleaseRangeOnHTTPSListing._fake_get))
        with patch("idtrack._database_manager.pymysql.connect", connect), patch(
            "idtrack._ensembl_access._session", return_value=session
        ):
            dm = DatabaseManager.__new__(DatabaseManager)
            dm.organism = "macaca_mulatta"
            dm.log = MagicMock()
            listed = dm.create_available_databases()["available_databases"].tolist()
        reachability.reset()
        assert listed == sorted(RHESUS_CORE_DBS.values())
        assert "macaca_mulatta_core_48_10g" in listed and "macaca_mulatta_core_90_801" in listed
        assert connect.called is (mode == "auto")
