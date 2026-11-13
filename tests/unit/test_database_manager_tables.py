#!/usr/bin/env python3
"""The cached tables of DatabaseManager: complete keys, atomic storage, no stale table served, strict discovery.

A fake Ensembl (tables per release, faked downloads and core-database index) stands in for the network. Each test
names the review finding it guards.
"""

import logging
import os
import stat
import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._ensembl_access import EnsemblDownloadError

RELEASES = [110, 111, 112, 113, 114, 115]


def _gene(release: int, versions: bool = True) -> pd.DataFrame:
    """Genes 1..3 exist in every release; gene 1 changes version at every release (version = release - 109).

    Genes 1 and 2 lie on sequence region 7 (a chromosome), gene 3 on region 8 (a patch).
    """
    return pd.DataFrame(
        {
            "gene_id": [1, 2, 3],
            "stable_id": ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"],
            "version": [float(release - 109), 1.0, 1.0] if versions else [np.nan] * 3,
            "seq_region_id": [7, 7, 8],
        }
    )


def _history(release: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The history of release R: one mapping session r-1 -> r per earlier release, as Ensembl's tables hold it."""
    sessions = list(range(RELEASES[0] + 1, release + 1))
    ms = pd.DataFrame(
        {
            "mapping_session_id": list(range(1, len(sessions) + 1)),
            "old_db_name": [f"homo_sapiens_core_{r - 1}_38" for r in sessions],
            "new_db_name": [f"homo_sapiens_core_{r}_38" for r in sessions],
            "old_release": [str(r - 1) for r in sessions],
            "new_release": [str(r) for r in sessions],
            "old_assembly": ["GRCh38"] * len(sessions),
            "new_assembly": ["GRCh38"] * len(sessions),
            "created": ["2020-01-01 00:00:00"] * len(sessions),
        }
    )
    sie = pd.DataFrame(
        {
            "mapping_session_id": list(range(1, len(sessions) + 1)),
            "old_stable_id": ["ENSG00000000001"] * len(sessions),
            "old_version": [float(r - 1 - 109) for r in sessions],
            "new_stable_id": ["ENSG00000000001"] * len(sessions),
            "new_version": [float(r - 109) for r in sessions],
            "score": [0.9] * len(sessions),
            "type": ["gene"] * len(sessions),
        }
    )
    return sie, ms


def _tables(release: int, versions: bool = True) -> dict[str, pd.DataFrame]:
    sie, ms = _history(release)
    gene = _gene(release, versions)
    return {
        "gene": gene,
        "transcript": pd.DataFrame(
            {
                "transcript_id": [10, 20, 30],
                "gene_id": [1, 2, 3],
                "stable_id": ["ENST00000000010", "ENST00000000020", "ENST00000000030"],
                "version": [1.0, 1.0, 1.0] if versions else [np.nan] * 3,
            }
        ),
        "translation": pd.DataFrame(
            {
                "translation_id": [100],
                "transcript_id": [10],
                "stable_id": ["ENSP00000000100"],
                "version": [1.0] if versions else [np.nan],
            }
        ),
        "stable_id_event": sie,
        "mapping_session": ms,
        # Region 7 is a top-level chromosome; region 8 is top-level too and outside the reference sequence.
        "attrib_type": pd.DataFrame({"attrib_type_id": [6, 16, 312], "code": ["toplevel", "non_ref", "patch_fix"]}),
        "seq_region_attrib": pd.DataFrame({"seq_region_id": [7, 8, 8, 8], "attrib_type_id": [6, 6, 16, 312]}),
        "object_xref": pd.DataFrame(
            {
                "ensembl_id": [1, 2, 3, 1],
                "ensembl_object_type": ["Gene", "Gene", "Gene", "Gene"],
                "xref_id": [1000, 1001, 1002, 1003],
                "object_xref_id": [5000, 5001, 5002, 5003],
            }
        ),
        "xref": pd.DataFrame(
            {
                "xref_id": [1000, 1001, 1002, 1003],
                "external_db_id": [1, 2, 2, 3],
                "dbprimary_acc": ["P12345", "7157", "672", "HGNC:1"],
                "display_label": ["TP53 ", "BRCA1", "   ", "TP53"],
            }
        ),
        "external_db": pd.DataFrame(
            {
                "external_db_id": [1, 2, 3],
                "db_name": ["Uniprot/SWISSPROT", "EntrezGene", "HGNC"],
                "db_display_name": ["UniProtKB/Swiss-Prot", "NCBI gene (formerly Entrezgene)", "HGNC Symbol"],
            }
        ),
        "identity_xref": pd.DataFrame(
            {"ensembl_identity": [100.0, 99.0], "xref_identity": [98.0, 97.0], "object_xref_id": [5000, 5001]}
        ),
        "external_synonym": pd.DataFrame({"xref_id": [1000, 1003], "synonym": ["14-3-3 ", "p53"]}),
    }


def _write_yaml(repo: str, databases: list[str], releases: list[int] = RELEASES) -> None:
    """Write a local externals YAML enabling *databases* for gene on assembly 38 at *releases*."""
    from idtrack._external_databases import write_externals_yaml

    all_dbs = [
        "Uniprot/SWISSPROT",
        "UniProtKB/Swiss-Prot",
        "EntrezGene",
        "NCBI gene (formerly Entrezgene)",
        "HGNC",
        "HGNC Symbol",
        "synonym_id::UniProtKB/Swiss-Prot",
    ]
    all_dbs += [d for d in databases if d not in all_dbs]
    data = {
        "homo_sapiens": {
            "gene": {
                db: {
                    "Assembly": {"38": {"Ensembl release": ",".join(map(str, releases)), "Include": db in databases}},
                    "Database Index": i,
                    "Potential Synonymous": "",
                }
                for i, db in enumerate(all_dbs)
            }
        }
    }
    path = os.path.join(repo, "homo_sapiens_externals_modified.yml")
    old = os.stat(path).st_mtime_ns if os.path.exists(path) else None
    write_externals_yaml(data, path)
    if old is not None:  # make the change visible even within the file system's time resolution
        os.utime(path, ns=(old + 10**9, old + 10**9))


class FakeEnsembl:
    """Counts downloads and serves the synthetic tables; `versions[release] = False` drops a release's versions."""

    def __init__(self, releases=RELEASES):  # noqa: D107
        self.releases = list(releases)
        self.versions = {r: True for r in releases}
        self.downloads: list[tuple[int, str]] = []
        self.replaced: dict[str, pd.DataFrame] = {}  # tables served instead of the synthetic ones, by table name

    def core_index(self, *, organism, genome_assembly, **_):  # noqa: D102
        releases = self.releases if int(genome_assembly) == 38 else []
        db = {r: f"{organism}_core_{r}_{genome_assembly}" for r in releases}
        return {
            "organism": organism,
            "genome_assembly": int(genome_assembly),
            "ports": (3306,),
            "releases_by_port": {3306: set(releases)},
            "db_by_port_release": {3306: dict(db)},
            "releases": list(releases),
            "port_for_release": {r: 3306 for r in releases},
            "db_for_release": db,
            "releases_on_mysql": [],
        }

    def download(self, dm, table_key, usecols=None):  # noqa: D102
        self.downloads.append((int(dm.ensembl_release), table_key))
        if table_key in self.replaced:
            df = self.replaced[table_key]
        else:
            df = _tables(int(dm.ensembl_release), self.versions[int(dm.ensembl_release)])[table_key]
        return df.copy() if usecols is None else df.loc[:, list(usecols)].copy()


@pytest.fixture
def fake(monkeypatch):
    """Install a fake Ensembl."""
    ens = FakeEnsembl()
    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: ens.core_index(**kw)))
    monkeypatch.setattr(DatabaseManager, "download_table", lambda self, t, usecols=None: ens.download(self, t, usecols))
    return ens


def _dm(repo, release=115, before=110, after=115, form="gene") -> DatabaseManager:
    return DatabaseManager(
        organism="homo_sapiens",
        form=form,
        local_repository=str(repo),
        ensembl_release=release,
        ignore_before=before,
        ignore_after=after,
        genome_assembly=38,
    )


# ---------------------------------------------------------------- keys carry everything a table depends on


def test_id_history_of_a_wider_window_is_not_the_cached_narrower_one(tmp_path, fake):
    """access-2, accuracy-new-1, accuracy-11: a history cached for 113..115 is not served to 110..115."""
    narrow = _dm(tmp_path, before=113).get_db("idhistory_narrow")
    assert sorted(narrow["old_release"].unique()) == [113.0, 114.0]
    wide = _dm(tmp_path, before=110).get_db("idhistory_narrow")
    assert sorted(wide["old_release"].unique()) == [110.0, 111.0, 112.0, 113.0, 114.0]
    # The stored history is the release's whole history, so the second window needed no new download.
    assert fake.downloads.count((115, "stable_id_event")) == 1


def test_external_relevant_follows_the_externals_yaml(tmp_path, fake):
    """access-1, organisms_externals-5: a changed selection is applied, for a removed and for an added database."""
    _write_yaml(str(tmp_path), ["EntrezGene"])
    first = _dm(tmp_path).get_db("external_relevant")
    assert set(first["name_db"]) == {"EntrezGene"}
    _write_yaml(str(tmp_path), ["HGNC Symbol", "Uniprot/SWISSPROT"])
    second = _dm(tmp_path).get_db("external_relevant")
    assert set(second["name_db"]) == {"HGNC Symbol", "Uniprot/SWISSPROT"}
    _write_yaml(str(tmp_path), ["EntrezGene"])
    pd.testing.assert_frame_equal(_dm(tmp_path).get_db("external_relevant"), first)


def test_forced_rebuild_replaces_the_stored_table(tmp_path, fake, monkeypatch):
    """access-new-1: after `create_even_if_exist=True` the next ordinary call returns the rebuilt table."""
    dm = _dm(tmp_path)
    dm.get_db("idhistory")
    real = DatabaseManager.create_id_history

    def changed(self, narrow):
        out = real(self, narrow)
        out["score"] = 0.5
        return out

    monkeypatch.setattr(DatabaseManager, "create_id_history", changed)
    rebuilt = dm.get_db("idhistory", create_even_if_exist=True)
    monkeypatch.setattr(DatabaseManager, "create_id_history", real)
    pd.testing.assert_frame_equal(_dm(tmp_path).get_db("idhistory"), rebuilt)


# ---------------------------------------------------------------- ids without versions (decision 27)


def test_release_without_id_versions_is_refused_and_no_table_is_made_up_for_it(tmp_path, fake):
    """Decision 27: ids without versions are not supported; the manager serves no table of such a release."""
    fake.versions[112] = False
    refused = _dm(tmp_path, release=112)
    for table in ("idsraw_gene", "ids", "relationcurrent", "external", "generegion"):
        with pytest.raises(
            ValueError, match="Organisms whose Ensembl ids carry no versions are not supported"
        ) as error:
            refused.get_db(table)
        assert "the gene ids of homo_sapiens have no version at Ensembl release 112 (genome assembly 38)" in str(
            error.value
        )
    with pytest.raises(ValueError, match="the transcript ids of homo_sapiens have no version at Ensembl release 112"):
        _dm(tmp_path, release=112, form="transcript").get_db("ids")
    stored = os.listdir(refused.tables_directory)
    assert stored and all(name.startswith("ens112_mysql_") for name in stored)  # raw tables, nothing derived

    # a release with versions is served as ever, whatever another release of the window lacks
    served = _dm(tmp_path, release=115)
    assert served.get_db("ids").values.tolist() == [
        ["ENSG00000000001", 6],
        ["ENSG00000000002", 1],
        ["ENSG00000000003", 1],
    ]
    assert "ENSG00000000001.6" in set(served.get_db("relationcurrent")["gene"])
    served.get_db("idhistory")
    # no key carries a version policy: there is one state
    derived = sorted(n for n in os.listdir(served.tables_directory) if "relationcurrent" in n or "idhistory" in n)
    assert derived == ["ens115_idhistory_gene.h5", "ens115_relationcurrent.h5"]


def test_ids_of_which_only_some_carry_a_version_are_refused(tmp_path, fake):
    """A release in which some ids have a version and others have none is refused, with the count."""
    fake.replaced["gene"] = _gene(115).assign(version=[6.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="have no version for 1 of 3 ids at Ensembl release 115"):
        _dm(tmp_path).get_db("ids")


def test_id_history_refuses_an_event_of_an_id_without_version(tmp_path, fake):
    """An event whose id has no version is refused; an event side without an id (a birth) has none and is kept."""
    sie, ms = _history(115)
    born = sie.iloc[[0]].assign(old_stable_id=[np.nan], old_version=[np.nan], new_stable_id=["ENSG00000000003"])
    fake.replaced["stable_id_event"] = pd.concat([sie, born], ignore_index=True)
    history = _dm(tmp_path).get_db("idhistory")
    birth = history[history["new_stable_id"] == "ENSG00000000003"]
    assert len(birth) == 1 and birth["old_stable_id"].iloc[0] == "" and np.isnan(birth["old_version"].iloc[0])
    assert history["new_version"].notna().all()

    other = tmp_path / "other"
    other.mkdir()
    fake.replaced["stable_id_event"] = sie.assign(new_version=[np.nan] * len(sie))
    with pytest.raises(ValueError, match="carry no versions are not supported"):
        _dm(other).get_db("idhistory")


# ---------------------------------------------------------------- what is stored (access-7)


def test_ids_of_a_form_are_not_stored_per_asking_form(tmp_path, fake):
    """access-7: relationcurrent and the external tables of three forms store no per-form copies of the ids."""
    _write_yaml(str(tmp_path), ["EntrezGene"])
    for form in DB.forms_in_order:
        _dm(tmp_path, form=form).get_db("relationcurrent")
        _dm(tmp_path, form=form).get_db("external")
    names = os.listdir(_dm(tmp_path).tables_directory)
    assert not [n for n in names if "idsraw" in n]
    assert len([n for n in names if n.startswith("ens115_mysql_gene_")]) == 1
    assert len([n for n in names if n.startswith("ens115_relationcurrent")]) == 1


def test_a_cached_raw_table_comes_back_in_the_order_of_usecols(tmp_path, fake):
    """access-16: `get_table` returns the columns in the order asked for, whoever stored the table."""
    dm = _dm(tmp_path)
    dm.get_table("gene", usecols=["gene_id", "stable_id", "version"])
    again = dm.get_table("gene", usecols=["version", "gene_id", "stable_id"])
    assert list(again.columns) == ["version", "gene_id", "stable_id"]


# ---------------------------------------------------------------- genes beside the reference sequence (decision 18)


def test_gene_region_marks_the_genes_on_regions_outside_the_reference_sequence(tmp_path, fake):
    """Decision 18: a gene is marked when its sequence region carries Ensembl's `non_ref` attribute, and only then."""
    table = _dm(tmp_path).get_db("generegion")
    assert list(table.columns) == ["gene_stable_id", "gene_version", "non_reference"]
    assert table["non_reference"].dtype == np.dtype(bool)
    assert table.values.tolist() == [
        ["ENSG00000000001", 6, False],  # its region is top-level, like every region: that is no mark
        ["ENSG00000000002", 1, False],
        ["ENSG00000000003", 1, True],
    ]
    # The table is about genes: a manager of another form gives the same one.
    pd.testing.assert_frame_equal(_dm(tmp_path, form="translation").get_db("generegion"), table)
    # Each raw table is downloaded once and stored; the table itself is a function of them and is not stored.
    _dm(tmp_path).get_db("generegion")
    for raw in ("attrib_type", "seq_region_attrib"):
        assert fake.downloads.count((115, raw)) == 1
    assert fake.downloads.count((115, "gene")) == 2  # the ids (id, stable id, version) and the regions
    names = os.listdir(_dm(tmp_path).tables_directory)
    assert "ens115_mysql_gene__cols=gene_id+seq_region_id.h5" in names
    assert not [n for n in names if "generegion" in n]


def test_gene_region_of_a_release_that_marks_no_region(tmp_path, fake):
    """A core database without the attribute, or with no region that carries it, has genes of the reference only."""
    fake.replaced["attrib_type"] = pd.DataFrame({"attrib_type_id": [6], "code": ["toplevel"]})
    assert not _dm(tmp_path).get_db("generegion")["non_reference"].any()

    other = tmp_path / "other"
    other.mkdir()
    del fake.replaced["attrib_type"]
    fake.replaced["seq_region_attrib"] = pd.DataFrame({"seq_region_id": [7, 8], "attrib_type_id": [6, 6]})
    assert not _dm(other).get_db("generegion")["non_reference"].any()


def test_gene_region_refuses_a_gene_without_a_sequence_region(tmp_path, fake):
    """A gene of the id table that the `gene` table gives no region is an error, not a gene of the reference."""
    fake.replaced["gene"] = _gene(115).assign(seq_region_id=[7.0, np.nan, 8.0])
    with pytest.raises(
        ValueError, match="1 genes of homo_sapiens release 115 have no sequence region.*ENSG00000000002"
    ):
        _dm(tmp_path).get_db("generegion")


# ---------------------------------------------------------------- node names (access-12)


def test_id_ver_from_df_checks_the_columns_exactly(tmp_path, fake):
    """access-12: a frame with one wrong column is refused."""
    dm = _dm(tmp_path)
    with pytest.raises(ValueError, match="not as expected"):
        dm.id_ver_from_df(pd.DataFrame({"gene_stable_id": ["A"], "transcript_version": [1]}))


def test_id_ver_from_df_keeps_version_zero_and_equals_the_row_wise_names(tmp_path, fake):
    """access-12: version 0 names `<id>.0`, not the versionless base node; output equals the row-wise rule."""
    dm = _dm(tmp_path)
    df = pd.DataFrame({"gene_stable_id": ["A", "B", "C", "D"], "gene_version": [0, 3, np.nan, 12.0]})
    assert dm.id_ver_from_df(df) == ["A.0", "B.3", "C", "D.12"]
    rng = np.random.default_rng(1)
    big = pd.DataFrame(
        {"gene_stable_id": [f"G{i}" for i in range(500)], "gene_version": rng.integers(1, 30, 500).astype(float)}
    )
    big.loc[::7, "gene_version"] = np.nan
    row_wise = [DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(i, v)) for i, v in big.values]
    before = big.copy()
    assert dm.id_ver_from_df(big) == row_wise
    pd.testing.assert_frame_equal(big, before)  # the input frame is not changed
    with pytest.raises(ValueError, match="floating"):
        dm.id_ver_from_df(pd.DataFrame({"gene_stable_id": ["A"], "gene_version": [1.5]}))
    alt = pd.DataFrame({"gene_stable_id": ["A", "B"], "gene_version": pd.Series([DB.no_old_node_id, 2], dtype=object)})
    assert dm.id_ver_from_df(alt) == [f"A.{DB.no_old_node_id}", "B.2"]


# ---------------------------------------------------------------- external tables (access-8, access-18)


def test_external_ids_and_synonyms_are_stripped(tmp_path, fake):
    """access-18: padded labels and synonyms become the bare id once; a blank-only label is dropped."""
    ext = _dm(tmp_path).get_db("external")
    ids = set(ext["id_db"])
    assert "TP53" in ids and "TP53 " not in ids
    assert f"{DB.synonym_id_nodes_prefix}14-3-3" in ids
    assert not any(i != i.strip() or i.strip() in ("", DB.synonym_id_nodes_prefix.strip()) for i in ids)
    tp53 = ext[(ext["id_db"] == "TP53") & (ext["graph_id"] == "ENSG00000000001.6")]
    assert sorted(tp53["name_db"]) == ["HGNC Symbol", "UniProtKB/Swiss-Prot"]  # once per database


def test_relevant_rows_equal_the_filtered_full_join(tmp_path, fake):
    """access-8: restricting before the join gives the table the full join gives after filtering, row for row."""
    selection = ["EntrezGene", "HGNC Symbol", f"{DB.synonym_id_nodes_prefix}UniProtKB/Swiss-Prot"]
    _write_yaml(str(tmp_path), selection)
    dm = _dm(tmp_path)
    full = dm.create_external_db("all")
    relevant = dm.create_external_db("relevant")
    expected = full[full["name_db"].isin(selection)].reset_index(drop=True)
    pd.testing.assert_frame_equal(relevant, expected)
    counts = dm.get_db("external_relevant-database")
    assert counts["count"].sum() == len(relevant)


def test_create_external_db_checks_its_mode_first_and_names_missing_databases(tmp_path, fake):
    """access-8, access-16: a wrong mode raises before any download; a missing database is named."""
    dm = _dm(tmp_path)
    with pytest.raises(ValueError, match="'relevant'"):
        dm.create_external_db("everything")
    assert fake.downloads == []
    _write_yaml(str(tmp_path), ["EntrezGene", "RefSeq_peptide"])
    with pytest.raises(ValueError, match="RefSeq_peptide"):
        _dm(tmp_path).create_external_db("relevant")


# ---------------------------------------------------------------- repository access (access-4, storage)


def test_read_only_repository_serves_what_it_holds(tmp_path, fake):
    """access-4: a read-only repository is accepted; computing a new table names the reason."""
    _dm(tmp_path).get_db("idhistory")
    mode = os.stat(tmp_path).st_mode
    tables = _dm(tmp_path).tables_directory
    try:
        os.chmod(tables, stat.S_IRUSR | stat.S_IXUSR)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
        dm = _dm(tmp_path)
        assert len(dm.get_db("idhistory")) > 0
        with pytest.raises(PermissionError, match="not writable"):
            dm.get_db("external")
    finally:
        os.chmod(tmp_path, mode)
        os.chmod(tables, mode)


def test_tables_live_in_one_file_each(tmp_path, fake):
    """access-6, storage_net-1: every table is a file of its own (a crash costs at most the table being written)."""
    _dm(tmp_path).get_db("idhistory_narrow")
    names = sorted(os.listdir(_dm(tmp_path).tables_directory))
    assert "ens115_mysql_stable_id_event.h5" in names and "ens115_mysql_mapping_session.h5" in names
    assert all(n.endswith(".h5") for n in names)


# ---------------------------------------------------------------- discovery (access-3, access-15)


@pytest.fixture
def no_wait(monkeypatch):
    """No waits between retries, an empty index memory, a fresh latest release, the "https" mode."""
    monkeypatch.setattr(DB, "rest_retry_wait", 0.0)
    monkeypatch.setattr(DB, "rest_attempts", 3)
    monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
    monkeypatch.setattr(DatabaseManager, "_ensembl_latest_release_cache", (102, time.monotonic()))
    monkeypatch.setattr(DB, "ensembl_access", "https")


class _Transport:
    """A fake HTTPS transport over pig assembly 111 releases 100..102: `fail[url] = n` fails the first n requests."""

    def __init__(self, status_on_fail=None):
        self.fail: dict[str, int] = {}
        self.calls: list[str] = []
        self.status_on_fail = status_on_fail

    def get(self, url, timeout=None, stream=False, headers=None):
        import requests

        self.calls.append(url)
        if self.fail.get(url, 0) > 0:
            self.fail[url] -= 1
            if self.status_on_fail is None:
                raise requests.ConnectionError("connection reset")
            return self._resp(self.status_on_fail, "")
        name = url.rstrip("/").rsplit("/", 1)[-1]
        if url.endswith("/mysql/"):
            return self._resp(200, "")
        ok = name in {f"sus_scrofa_core_{r}_111" for r in (100, 101, 102)}
        return self._resp(200 if ok else 404, "")

    @staticmethod
    def _resp(status, text):
        r = MagicMock()
        r.status_code, r.ok, r.text, r.reason, r.headers = status, status == 200, text, "x", {}
        return r


def _scan(transport, monkeypatch):
    monkeypatch.setattr("idtrack._ensembl_access._session", lambda: transport)
    return DatabaseManager._get_core_db_index_from_ftp(organism="sus_scrofa", genome_assembly=111, ports=[3306])


def test_one_failed_probe_is_retried_not_read_as_absent(monkeypatch, no_wait):
    """access-3 (a): a release whose probe fails once is still in the index."""
    monkeypatch.setattr(DB, "assembly_mysqlport_priority", DB.assembly_mysqlport_priority)
    t = _Transport()
    url = "https://ftp.ensembl.org/pub/release-101/mysql/sus_scrofa_core_101_111/"
    t.fail[url] = 1
    monkeypatch.setattr(DatabaseManager, "_release_window", staticmethod(lambda o, a: (100, None)))
    assert _scan(t, monkeypatch)["releases"] == [100, 101, 102]


@pytest.mark.parametrize("status", [None, 503, 429])
def test_an_unanswered_release_makes_the_scan_raise(monkeypatch, no_wait, status):
    """access-3 (a): a release that stays unknown raises instead of vanishing from the index."""
    t = _Transport(status_on_fail=status)
    for url in (
        "https://ftp.ensembl.org/pub/release-101/mysql/sus_scrofa_core_101_111/",
        "https://ftp.ensembl.org/pub/release-101/mysql/",
    ):
        t.fail[url] = 99
    monkeypatch.setattr(DatabaseManager, "_release_window", staticmethod(lambda o, a: (100, None)))
    with pytest.raises(EnsemblDownloadError):
        _scan(t, monkeypatch)


def test_a_404_still_means_absent(monkeypatch, no_wait):
    """access-3: Ensembl's "not found" answer leaves a release out without an error."""
    monkeypatch.setattr(DatabaseManager, "_release_window", staticmethod(lambda o, a: (99, None)))
    assert _scan(_Transport(), monkeypatch)["releases"] == [100, 101, 102]


def test_a_failed_index_never_selects_another_primary_assembly(tmp_path, monkeypatch):
    """access-3 (c): if GRCh38's index cannot be built, the manager raises instead of choosing GRCh37."""

    def index(cls, *, organism, genome_assembly, **_):
        if int(genome_assembly) == 38:
            raise EnsemblDownloadError("Ensembl did not answer")
        return FakeEnsembl([115]).core_index(organism=organism, genome_assembly=38) | {"genome_assembly": 37}

    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(index))
    with pytest.raises(EnsemblDownloadError):
        DatabaseManager(organism="homo_sapiens", form="gene", local_repository=str(tmp_path), ensembl_release=115)


def test_all_assemblies_raise_when_one_index_fails(tmp_path, fake, monkeypatch):
    """access-3 (b): `available_releases_all_assemblies` does not leave a failed assembly out."""
    dm = _dm(tmp_path)
    real = DatabaseManager._get_core_db_index

    def index(cls, *, organism, genome_assembly, **kw):
        if int(genome_assembly) == 37:
            raise EnsemblDownloadError("Ensembl did not answer")
        return real(organism=organism, genome_assembly=genome_assembly, **kw)

    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(index))
    with pytest.raises(EnsemblDownloadError):
        _ = dm.available_releases_all_assemblies


def test_create_external_all_skips_only_assemblies_without_the_release(tmp_path, fake, monkeypatch):
    """access-3 (b), organisms_externals-2: an assembly without the release is passed over; a failure raises."""
    from idtrack._external_databases import ExternalDatabases

    _write_yaml(str(tmp_path), ["EntrezGene"])
    monkeypatch.setattr(
        ExternalDatabases,
        "give_list_for_case",
        lambda self, give_type: [37, 38] if give_type == "assembly" else ["EntrezGene"],
    )
    out = _dm(tmp_path).create_external_all(return_mode="all")
    assert set(out["assembly"]) == {38}
    real = DatabaseManager._get_core_db_index

    def index(cls, *, organism, genome_assembly, **kw):
        if int(genome_assembly) == 37:
            raise EnsemblDownloadError("Ensembl did not answer")
        return real(organism=organism, genome_assembly=genome_assembly, **kw)

    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(index))
    with pytest.raises(EnsemblDownloadError):
        _dm(tmp_path).create_external_all(return_mode="all")


def test_latest_release_memory_expires(monkeypatch):
    """access-15: a long-running process asks for the latest release again once the answer is old."""
    asked = []
    monkeypatch.setattr(
        "idtrack._database_manager.rest_get_json", lambda path, what: asked.append(path) or {"releases": [116]}
    )
    monkeypatch.setattr(DatabaseManager, "_ensembl_latest_release_cache", (115, time.monotonic()))
    assert DatabaseManager._ensembl_latest_release() == 115
    monkeypatch.setattr(
        DatabaseManager,
        "_ensembl_latest_release_cache",
        (115, time.monotonic() - 2 * float(DB.rest_species_cache_seconds)),
    )
    assert DatabaseManager._ensembl_latest_release() == 116
    assert asked == ["/info/data"]


def test_open_ended_index_is_scanned_again_when_old(monkeypatch):
    """access-15: an index for "the latest release" is not kept for ever."""
    fresh = {"scanned_up_to": 115, "scanned_live_at": time.monotonic()}
    old = {"scanned_up_to": 115, "scanned_live_at": time.monotonic() - 2 * float(DB.rest_species_cache_seconds)}
    assert DatabaseManager._index_covers(fresh, "sus_scrofa", 111, None)
    assert not DatabaseManager._index_covers(old, "sus_scrofa", 111, None)
    assert DatabaseManager._index_covers(old, "sus_scrofa", 111, 110)  # a closed window never changes


# ---------------------------------------------------------------- downloads (access-9, -10, -11, storage_net-10)


def test_schema_is_resolved_once_and_each_table_is_one_request(tmp_path, monkeypatch, no_wait):
    """access-10: n tables of one database cost n requests plus the schema (one probe, one download)."""
    import gzip as gz

    schema = (
        "CREATE TABLE `gene` (\n  `gene_id` int(10) unsigned NOT NULL,\n  `stable_id` varchar(128) DEFAULT NULL\n"
        ") ENGINE=MyISAM;\nCREATE TABLE `xref` (\n  `xref_id` int(10),\n  `dbprimary_acc` varchar(512)\n) ENGINE=MyISAM;"
    )
    calls = []

    class _T:
        def get(self, url, timeout=None, stream=False, headers=None):
            calls.append(url)
            r = MagicMock()
            r.status_code, r.ok, r.headers, r.reason = 200, True, {}, "OK"
            r.content = gz.compress(schema.encode())
            return r

    monkeypatch.setattr("idtrack._ensembl_access._session", lambda: _T())
    monkeypatch.setattr(DatabaseManager, "_ftp_schema_url_cache", {})
    monkeypatch.setattr(DatabaseManager, "_ftp_schema_cache", {})
    dm = DatabaseManager.__new__(DatabaseManager)
    dm.organism, dm.genome_assembly, dm.ensembl_release, dm.log = "sus_scrofa", 111, 115, logging.getLogger("t")
    dm.__dict__["mysql_database"] = "sus_scrofa_core_115_111"
    monkeypatch.setattr(DatabaseManager, "_ftp_db_dir_url", lambda self: "https://x/db/")

    class _Ctx:
        def __init__(self, text):
            self.text = text

        def __enter__(self):
            import io

            calls.append("table")
            return io.StringIO(self.text)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(DatabaseManager, "_open_decompressed_http_text", staticmethod(lambda url: _Ctx("1\t007\n")))
    a = dm._download_table_from_ftp("gene")
    b = dm._download_table_from_ftp("xref")
    assert calls.count("table") == 2
    assert len([c for c in calls if c != "table"]) == 2  # schema probe + schema download, once
    # storage_net-10: an all-digit varchar stays text, leading zeros included.
    assert a["stable_id"].tolist() == ["007"] and b["dbprimary_acc"].tolist() == ["007"]


def test_both_routes_give_the_same_types(tmp_path):
    """access-9, storage_net-10: rows from MySQL and the same rows from a dump give equal frames."""
    import datetime
    import io

    from idtrack._mysql_dump import read_mysql_dump

    typed = [
        ("mapping_session_id", "int(10) unsigned"),
        ("old_release", "varchar(5)"),
        ("score", "float"),
        ("created", "datetime"),
        ("acc", "varchar(40)"),
    ]
    rows = [
        (1, "106", 0.5, datetime.datetime(2020, 1, 2, 3, 4, 5), "007"),
        (2, b"107", None, datetime.datetime(2021, 1, 2, 3, 4, 5), None),
    ]
    mysql = DatabaseManager._typed_mysql_frame(rows, typed, "mapping_session")
    dump = "1\t106\t0.5\t2020-01-02 03:04:05\t007\n2\t107\t\\N\t2021-01-02 03:04:05\t\\N\n"
    text_cols = [c for c, t in typed if DatabaseManager._is_text_sql_type(t)]
    https = read_mysql_dump(io.StringIO(dump), [c for c, _ in typed], text_columns=text_cols)
    pd.testing.assert_frame_equal(mysql, https)
    assert https["old_release"].tolist() == ["106", "107"]


def test_a_mysql_connection_failure_falls_back_with_a_warning(monkeypatch, caplog):
    """access-9: only a connection failure moves to the dumps, and the warning names the cause."""
    import pymysql

    from idtrack._ensembl_access import reachability

    reachability.reset()
    monkeypatch.setattr(reachability, "mysql_port_usable", lambda port: True)
    monkeypatch.setattr(reachability, "mark_unreachable", lambda *a, **k: None)
    dm = DatabaseManager.__new__(DatabaseManager)
    dm.organism, dm.genome_assembly, dm.ensembl_release, dm.log = "sus_scrofa", 111, 100, logging.getLogger("dm")
    dm.mysql_settings = {"host": DB.mysql_host, "port": 3306}
    dm.__dict__["mysql_database"] = "sus_scrofa_core_100_111"
    monkeypatch.setattr(
        DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: {"releases_on_mysql": [100]})
    )
    monkeypatch.setattr(pymysql, "connect", MagicMock(side_effect=pymysql.err.OperationalError(2003, "refused")))
    monkeypatch.setattr(
        DatabaseManager, "_download_table_from_ftp", lambda self, t, usecols=None: pd.DataFrame({"a": [1]})
    )
    with caplog.at_level(logging.WARNING, logger="dm"):
        dm.download_table("gene")
    assert any("refused" in r.getMessage() and "HTTPS" in r.getMessage() for r in caplog.records)


def test_a_failed_gene_download_names_gene_not_an_old_schema(tmp_path, monkeypatch):
    """access-11: a time-out while reading `gene` is reported as such, never as a missing `gene_stable_id`."""
    ens = FakeEnsembl()
    monkeypatch.setattr(DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: ens.core_index(**kw)))

    def download(self, table_key, usecols=None):
        raise EnsemblDownloadError(f"Ensembl did not answer an Ensembl table dump of {table_key!r}: ReadTimeout")

    monkeypatch.setattr(DatabaseManager, "download_table", download)
    with pytest.raises(EnsemblDownloadError, match="'gene'.*ReadTimeout"):
        _dm(tmp_path).create_ids("gene")


# ---------------------------------------------------------------- arguments (organisms_externals-12, N4)


@pytest.mark.parametrize(
    "bad", [True, 38.9, "38a", pytest.param(None.__class__, marks=pytest.mark.wrong_argument_type)]
)
def test_assembly_keys_that_only_look_like_one_are_refused(tmp_path, fake, bad):
    """organisms_externals-N4: `genome_assembly=True` or `38.9` no longer names another assembly."""
    with pytest.raises(ValueError, match="integer key"):
        DatabaseManager(organism="homo_sapiens", form="gene", local_repository=str(tmp_path), genome_assembly=bad)


@pytest.mark.parametrize("good", [38, "38", 38.0, np.int64(38)])
def test_integral_assembly_keys_are_accepted(tmp_path, fake, good):
    """organisms_externals-N4: integers, integral floats and digit strings are assembly keys."""
    dm = DatabaseManager(
        organism="homo_sapiens", form="gene", local_repository=str(tmp_path), genome_assembly=good, ignore_before=110
    )
    assert dm.genome_assembly == 38


def test_class_docstring_names_existing_methods():
    """access-16: every method the class docstring points to exists."""
    import re

    names = re.findall(r":py:meth:`~?(?:DatabaseManager\.)?([A-Za-z_]+)`", DatabaseManager.__doc__)
    assert names and all(hasattr(DatabaseManager, n) for n in names), names
