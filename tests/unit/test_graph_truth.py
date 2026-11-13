#!/usr/bin/env python3
"""What a built graph must say, and how it is stored: tests of ``GraphMaker`` and ``TheGraph`` on small worlds.

A *world* is a synthetic organism: the gene tables of each assembly and release, Ensembl's id events with their
mapping sessions, and the external identifiers of each release. The ``DatabaseManager`` reads it through its usual
methods (no network); the graph is built by the real builder.
"""

from __future__ import annotations

import os
import pickle  # noqa: S403
import socket
import time

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._graph_maker import GraphMaker, HistoryValidationError
from idtrack._the_graph import Connection, GraphMismatchError, HistoryEdge, ReleaseSet, TheGraph
from idtrack._track import Track

INF = float("inf")


class World:
    """A synthetic organism served to ``DatabaseManager`` in place of Ensembl."""

    def __init__(
        self, organism, assemblies, events=(), sessions=(), externals=None, unversioned=False, non_reference=None
    ):
        """Describe the organism: gene tables per assembly and release, events, sessions, external ids, regions."""
        self.organism = organism
        self.assemblies = assemblies  # {assembly: {release: {gene id: version}}}
        self.events = list(events)  # (session id, old id, old version, new id, new version, score)
        self.sessions = list(sessions)  # (session id, old release, new release)
        self.externals = externals or {}  # {release: [(graph id, external id, database, assembly), ...]}
        self.unversioned = unversioned  # True, or the releases whose gene ids carry no version
        self.non_reference = non_reference or {}  # {assembly: {release: [gene id on a patch, ...]}}

    def core_index(self, *, organism, genome_assembly, **_):
        """Return the core-database index of an assembly, as ``DatabaseManager._get_core_db_index`` does.

        An assembly of the organism table that this world does not have gets an index without releases, as Ensembl's
        complete answer "no core database of this assembly" does.
        """
        if organism != self.organism:
            raise ValueError("unexpected organism")
        releases = sorted(self.assemblies.get(int(genome_assembly), {}))
        names = {r: f"{organism}_core_{r}_{genome_assembly}" for r in releases}
        return {
            "organism": organism,
            "genome_assembly": int(genome_assembly),
            "ports": (3306,),
            "releases_by_port": {3306: set(releases)},
            "db_by_port_release": {3306: names.copy()},
            "releases": releases,
            "port_for_release": {r: 3306 for r in releases},
            "db_for_release": names,
            "releases_on_mysql": releases,
        }

    def table(self, manager, table_key):
        """Return the raw Ensembl table that the manager asks for."""
        release, assembly = int(manager.ensembl_release), int(manager.genome_assembly)
        if table_key == "gene":
            genes = self.assemblies[assembly][release]
            beside = set(self.non_reference.get(assembly, {}).get(release, ()))
            return pd.DataFrame(
                {
                    "gene_id": list(range(1, len(genes) + 1)),
                    "stable_id": list(genes),
                    "version": [np.nan] * len(genes) if self.without_versions(release) else list(genes.values()),
                    # sequence region 1 is a chromosome, region 2 a patch beside the reference sequence
                    "seq_region_id": [2 if gene in beside else 1 for gene in genes],
                }
            )
        if table_key == "attrib_type":
            return pd.DataFrame({"attrib_type_id": [6, 16], "code": ["toplevel", "non_ref"]})
        if table_key == "seq_region_attrib":
            return pd.DataFrame({"seq_region_id": [1, 2, 2], "attrib_type_id": [6, 6, 16]})
        if table_key in ("transcript", "translation"):
            # one transcript and one translation per gene, named after it (ENSG1 -> ENST1 -> ENSP1), version 1
            genes = list(self.assemblies[assembly][release])
            letter, parent = ("T", "gene_id") if table_key == "transcript" else ("P", "transcript_id")
            return pd.DataFrame(
                {
                    f"{table_key}_id": list(range(1, len(genes) + 1)),
                    "stable_id": [g[::-1].replace("G", letter, 1)[::-1] for g in genes],
                    "version": [1] * len(genes),
                    parent: list(range(1, len(genes) + 1)),
                }
            )
        if table_key == "stable_id_event":
            return pd.DataFrame(
                {
                    "mapping_session_id": [e[0] for e in self.events],
                    "old_stable_id": [e[1] for e in self.events],
                    "old_version": [e[2] for e in self.events],
                    "new_stable_id": [e[3] for e in self.events],
                    "new_version": [e[4] for e in self.events],
                    "score": [e[5] for e in self.events],
                    "type": ["gene"] * len(self.events),
                }
            )
        if table_key == "mapping_session":
            return pd.DataFrame(
                {
                    "mapping_session_id": [s[0] for s in self.sessions],
                    "old_db_name": [f"{self.organism}_core_{s[1]}" for s in self.sessions],
                    "new_db_name": [f"{self.organism}_core_{s[2]}" for s in self.sessions],
                    "old_release": [s[1] for s in self.sessions],
                    "new_release": [s[2] for s in self.sessions],
                    "old_assembly": ["a"] * len(self.sessions),
                    "new_assembly": ["a"] * len(self.sessions),
                    "created": ["2020-01-01"] * len(self.sessions),
                }
            )
        return pd.DataFrame()

    def without_versions(self, release):
        """Tell whether the gene ids of a release carry no version."""
        return self.unversioned is True or (self.unversioned is not False and release in self.unversioned)

    def external_rows(self, manager):
        """Return the external-identifier table of the manager's release."""
        release = int(manager.ensembl_release)
        rows = [
            {
                "release": release,
                "graph_id": graph_id,
                "id_db": external_id,
                "name_db": database,
                "ensembl_identity": np.nan,
                "xref_identity": np.nan,
                "assembly": assembly,
            }
            for graph_id, external_id, database, assembly, *form in self.externals.get(release, [])
            if (form[0] if form else "gene") == manager.form
        ]
        columns = ["release", "graph_id", "id_db", "name_db", "ensembl_identity", "xref_identity", "assembly"]
        return pd.DataFrame(rows, columns=columns)

    def install(self, monkeypatch):
        """Make ``DatabaseManager`` read this world instead of Ensembl."""
        world = self

        def available_releases_versions(self, **kwargs):
            releases = sorted(world.assemblies.get(int(self.genome_assembly), {}))
            return [r for r in releases if self.ignore_after >= r >= self.ignore_before]

        def download_table(self, table_key, usecols=None):
            df = world.table(self, table_key)
            return df.copy(deep=True) if usecols is None or not len(df.columns) else df.loc[:, usecols].copy(deep=True)

        def create_external_all(self, return_mode, narrow_external=True, **kwargs):
            return world.external_rows(self)

        monkeypatch.setattr(
            DatabaseManager, "_get_core_db_index", classmethod(lambda cls, **kw: world.core_index(**kw))
        )
        monkeypatch.setattr(DatabaseManager, "available_releases_versions", available_releases_versions)
        monkeypatch.setattr(DatabaseManager, "download_table", download_table)
        monkeypatch.setattr(DatabaseManager, "create_external_all", create_external_all)

    def manager(self, repository, primary, first=None, last=None):
        """Return a manager at the newest release of the primary assembly."""
        releases = sorted(self.assemblies[primary])
        every = sorted({r for per in self.assemblies.values() for r in per})
        return DatabaseManager(
            organism=self.organism,
            form="gene",
            local_repository=str(repository),
            ensembl_release=releases[-1] if last is None else last,
            ignore_before=every[0] if first is None else first,
            ignore_after=every[-1] if last is None else last,
            genome_assembly=primary,
            store_raw_always=True,
        )


def build(world, monkeypatch, tmp_path, primary, **kwargs):
    """Build the gene graph of a world with the given primary assembly."""
    os.makedirs(tmp_path, exist_ok=True)
    world.install(monkeypatch)
    return GraphMaker(world.manager(tmp_path, primary)).construct_graph(form_list=["gene"], **kwargs)


def lifetimes(genes_by_release):
    """Return ``{release: {id: version}}`` from ``{id: [(first, last, version), ...]}`` over all releases named."""
    releases = sorted({r for spans in genes_by_release.values() for a, b, _ in spans for r in range(a, b + 1)})
    return {r: {g: v for g, spans in genes_by_release.items() for a, b, v in spans if a <= r <= b} for r in releases}


# --------------------------------------------------------------------------- the history tells the truth (graph-1)


def pig_like(recorded_old, recorded_new):
    """Gene A becomes gene B between releases 106 and 107; the session is recorded as given."""
    genes = lifetimes({"ENSSSCG1": [(104, 106, 1)], "ENSSSCG2": [(107, 108, 1)], "ENSSSCG9": [(104, 108, 3)]})
    return World(
        "sus_scrofa",
        {111: genes},
        events=[(7, "ENSSSCG1", 1, "ENSSSCG2", 1, 0.9)],
        sessions=[(7, recorded_old, recorded_new)],
    )


def test_session_recorded_with_wrong_releases_is_placed_where_the_ids_change(monkeypatch, tmp_path):
    """Ensembl's pig session 106 to 107 is recorded as 106 to 104: the new id must start at 107, not 104."""
    g = build(pig_like(106, 104), monkeypatch, tmp_path, 111)

    assert g.active_ranges("ENSSSCG2.1") == [[107, INF]]
    assert g.active_ranges("ENSSSCG1.1") == [[104, 106]]
    incoming = {e.node: (e.old_release, e.new_release) for e in g.history_edges("ENSSSCG2.1", reverse=True)}
    assert incoming == {"ENSSSCG1.1": (106, 107), "ENSSSCG2.Void": (106, 107), "ENSSSCG2.1": (107, INF)}
    assert g.history_edges("ENSSSCG1.1") == (
        HistoryEdge("ENSSSCG1.Retired", 106, 107, pytest.approx(float("nan"), nan_ok=True), 0),
        HistoryEdge("ENSSSCG2.1", 106, 107, 0.9, 0),
    )
    assert not g.is_active("ENSSSCG2.1", 104) and not g.is_active("ENSSSCG2.1", 106)
    assert g.manifest["history"]["corrections"] == [
        {"recorded": [106, 104], "used": [106, 107], "events": 1, "edges": 1}
    ]


def test_session_over_a_gap_is_placed_at_the_transition_of_the_ids(monkeypatch, tmp_path):
    """A session recorded over two releases (rhesus 50 to 52) belongs to the pair at which its ids change."""
    g = build(pig_like(105, 107), monkeypatch, tmp_path, 111)

    (edge,) = (e for e in g.history_edges("ENSSSCG1.1") if e.node == "ENSSSCG2.1")
    assert (edge.old_release, edge.new_release) == (106, 107)
    assert g.manifest["history"]["corrections"][0]["used"] == [106, 107]


def test_correctly_recorded_session_is_not_a_correction(monkeypatch, tmp_path):
    """A session whose recorded releases agree with the id tables is used as recorded."""
    g = build(pig_like(106, 107), monkeypatch, tmp_path, 111)

    assert g.manifest["history"]["corrections"] == []
    assert g.manifest["history"]["event_edges"] == 1


def test_every_history_edge_agrees_with_the_id_tables(monkeypatch, tmp_path):
    """Source alive at old_release, target alive at new_release, old before new: for every event edge."""
    genes = lifetimes(
        {
            "G1": [(100, 102, 1), (103, 105, 2)],
            "G2": [(100, 101, 1)],
            "G3": [(102, 105, 1)],
            "G4": [(100, 105, 1)],
        }
    )
    world = World(
        "homo_sapiens",
        {38: genes},
        events=[(1, "G2", 1, "G3", 1, 0.8), (1, "G2", 1, "G4", 1, 0.5), (2, "G1", 1, "G1", 2, 0.7)],
        sessions=[(1, 101, 101), (2, 102, 103)],  # the first one mislabelled like human 114 to 114
    )
    g = build(world, monkeypatch, tmp_path, 38)

    checked = 0
    for node in g.nodes:
        if g.is_backbone(node) and g.version(node) not in DB.alternative_versions:
            for edge in g.history_edges(node):
                if edge.new_release == INF or g.version(edge.node) in DB.alternative_versions:
                    continue
                assert edge.old_release < edge.new_release
                assert g.is_active(node, edge.old_release) and g.is_active(edge.node, edge.new_release)
                checked += 1
    assert checked == 3
    assert g.active_ranges("G3.1") == [[102, INF]]


def split_and_merge_world(g1_onto_itself):
    """Session 100 to 101 maps G1 to the new gene G2 and merges G3 into G4; G1, G4 and G9 live on unchanged.

    ``g1_onto_itself`` is the score with which the session also maps G1 onto itself (``None``: it does not).
    """
    genes = lifetimes(
        {
            "G1": [(100, 102, 1)],
            "G2": [(101, 102, 1)],
            "G3": [(100, 100, 1)],
            "G4": [(100, 102, 1)],
            "G9": [(100, 102, 1)],
        }
    )
    events = [
        (1, "G1", 1, "G2", 1, 0.4),
        (1, "G3", 1, "G4", 1, 0.6),
        (1, "G4", 1, "G4", 1, 1.0),
        (1, "G9", 1, "G9", 1, 1.0),
        (2, "G1", 1, "G1", 1, 1.0),
        (2, "G4", 1, "G4", 1, 1.0),
        (2, "G9", 1, "G9", 1, 1.0),
    ]
    if g1_onto_itself is not None:
        events.append((1, "G1", 1, "G1", 1, g1_onto_itself))
    return World("homo_sapiens", {38: genes}, events=events, sessions=[(1, 100, 101), (2, 101, 102)])


def self_events(g, node):
    """Return ``(old_release, new_release, weight)`` of the self events of a node (not the loop to infinity)."""
    loops = [e for e in g.history_edges(node) if e.node == node and e.new_release != INF]
    return [(e.old_release, e.new_release, e.weight) for e in loops]


def test_perfect_self_event_is_kept_beside_other_events_of_its_session(monkeypatch, tmp_path):
    """An id mapped onto itself with score 1 and to or from another id in one session: the self event is history."""
    g = build(split_and_merge_world(1.0), monkeypatch, tmp_path, 38)

    assert self_events(g, "G1.1") == [(100, 101, 1.0)]  # G1 lived on beside G2, which branched off
    assert self_events(g, "G4.1") == [(100, 101, 1.0)]  # G4 lived on and took G3 in
    assert self_events(g, "G9.1") == []  # nothing else happened to G9: its perfect self events say nothing
    assert g.active_ranges("G1.1") == g.active_ranges("G4.1") == [[100, INF]]
    assert {e.node for e in g.history_edges("G1.1")} == {"G1.1", "G2.1"}
    # the loop to infinity starts after the latest event of the id with itself
    assert [(e.old_release, e.new_release) for e in g.history_edges("G1.1") if e.new_release == INF] == [(101, INF)]


def test_id_mapped_only_to_another_id_has_no_self_event(monkeypatch, tmp_path):
    """Without Ensembl's self event the graph does not invent one: the id is mapped to the other id only."""
    g = build(split_and_merge_world(None), monkeypatch, tmp_path, 38)

    assert self_events(g, "G1.1") == []
    assert [(e.node, e.old_release, e.new_release, e.weight) for e in g.history_edges("G1.1") if e.node != "G1.1"] == [
        ("G2.1", 100, 101, 0.4)
    ]
    assert [(e.old_release, e.new_release) for e in g.history_edges("G1.1") if e.new_release == INF] == [(100, INF)]


def test_imperfect_self_event_is_kept_whatever_else_the_session_holds(monkeypatch, tmp_path):
    """A self event with a score below 1 is a re-mapping of the id and is kept, as before."""
    world = split_and_merge_world(0.8)
    world.events.append((2, "G9", 1, "G9", 1, 0.9))
    world.events.remove((2, "G9", 1, "G9", 1, 1.0))
    g = build(world, monkeypatch, tmp_path, 38)

    assert self_events(g, "G1.1") == [(100, 101, 0.8)]
    assert self_events(g, "G9.1") == [(101, 102, 0.9)]


def test_engine_walks_an_id_that_lived_on_beside_a_branch(monkeypatch, tmp_path):
    """The kept self event changes no answer: the id is its own counterpart, the branch the lesser successor."""
    track = Track.from_graph(build(split_and_merge_world(1.0), monkeypatch, tmp_path, 38))

    result = track.convert("G1.1", from_release=100, to_release=102)
    assert list(result) == ["G1.1", "G2.1"]
    assert (result["G1.1"]["score"], result["G2.1"]["score"]) == (1.0, 0.4)
    assert list(track.convert("G1.1", from_release=100, to_release=102, prioritize_to_one_filter=True)) == ["G1.1"]
    assert list(track.convert("G4.1", from_release=102, to_release=100)) == ["G4.1", "G3.1"]


def test_event_that_fits_no_release_pair_stops_the_build(monkeypatch, tmp_path):
    """A session whose ids never meet at consecutive releases cannot be reconciled with the id tables."""
    genes = lifetimes({"G1": [(100, 100, 1)], "G2": [(103, 104, 1)], "G9": [(100, 104, 1)]})
    world = World("homo_sapiens", {38: genes}, events=[(1, "G1", 1, "G2", 1, 0.9)], sessions=[(1, 100, 101)])
    world.install(monkeypatch)

    with pytest.raises(HistoryValidationError, match="fit no pair of consecutive releases"):
        GraphMaker(world.manager(tmp_path, 38)).construct_graph(form_list=["gene"])


def test_id_history_computed_for_another_window_stops_the_build(monkeypatch, tmp_path):
    """A stored id-history table that starts later than Ensembl's sessions lost history: never build from it."""
    genes = lifetimes({"G1": [(100, 100, 1)], "G2": [(101, 103, 1)], "G3": [(100, 102, 1)], "G4": [(103, 103, 1)]})
    world = World(
        "homo_sapiens",
        {38: genes},
        events=[(1, "G1", 1, "G2", 1, 0.9), (2, "G3", 1, "G4", 1, 0.9)],
        sessions=[(1, 100, 101), (2, 102, 103)],
    )
    world.install(monkeypatch)
    real_get_db = DatabaseManager.get_db

    def stale(self, df_indicator, *args, **kwargs):
        table = real_get_db(self, df_indicator, *args, **kwargs)
        return table[table["old_release"] >= 102] if df_indicator.startswith("idhistory") else table

    monkeypatch.setattr(DatabaseManager, "get_db", stale)
    with pytest.raises(HistoryValidationError, match="another release window"):
        GraphMaker(world.manager(tmp_path, 38)).construct_graph(form_list=["gene"])


def test_retired_reborn_and_retired_again_has_complete_ranges(monkeypatch, tmp_path):
    """The ``Retired`` node of an id with two stays ends its second range at the last release (graph-7)."""
    genes = lifetimes({"G1": [(77, 81, 1), (87, 91, 1)], "G9": [(76, 96, 1)]})
    g = build(World("homo_sapiens", {38: genes}), monkeypatch, tmp_path, 38)

    assert g.active_ranges("G1.Retired") == [[82, 86], [92, 96]]
    assert g.active_ranges("G1.1") == [[77, 81], [87, 91]]
    assert g.active_ranges("G1.Void") == [[76, 76]]
    g.calculate_caches(for_test=True)
    assert ("assembly_38_ensembl_gene", 38, 96) in g.trios("G1.Retired")
    assert all(len(pair) == 2 for node in g.nodes if g.is_backbone(node) for pair in g.active_ranges(node))


def test_open_range_of_a_version_is_an_error():
    """A version whose last range never ends is a malformed history, named in the error (graph-7)."""
    g = TheGraph(genome_assembly=38, confident_for_release=[60, 61, 62, 63, 64])
    for name in ("G.Void", "G.1", "G.Retired"):
        g.add_node(name, **{DB.node_type_str: DB.nts_ensembl["gene"]})
    g.add_edge("G.Void", "G.1", weight=float("nan"), old_release=60, new_release=61)
    g.add_edge("G.1", "G.Retired", weight=float("nan"), old_release=61, new_release=62)
    g.add_edge("G.Retired", "G.1", weight=float("nan"), old_release=62, new_release=63)  # and no end after 63

    with pytest.raises(ValueError, match="'G.1'.*never ends"):
        g.active_ranges("G.1")


# --------------------------------------------------------------------------- spelling (graph-2, graph-8, new-1)


def symbols_world():
    """Three pig genes with symbols in several spellings and databases."""
    genes = lifetimes({"ENSSSCG1": [(113, 115, 1)], "ENSSSCG2": [(113, 115, 1)], "ENSSSCG3": [(113, 115, 1)]})
    externals = {
        113: [("ENSSSCG1.1", "stat2", "UniProtKB Gene Name", 111), ("ENSSSCG2.1", "Wnt4", "Old Symbol", 111)],
        114: [("ENSSSCG1.1", "stat2", "UniProtKB Gene Name", 111), ("ENSSSCG1.1", "STAT2", "VGNC Symbol", 111)],
        115: [
            ("ENSSSCG1.1", "STAT2", "VGNC Symbol", 111),
            ("ENSSSCG1.1", "stat2", "UniProtKB Gene Name", 111),
            ("ENSSSCG2.1", "WNT4", "VGNC Symbol", 111),
            ("ENSSSCG3.1", "SLA", "VGNC Symbol", 111),
            ("ENSSSCG2.1", "SLA-1", "VGNC Symbol", 111),
            ("ENSSSCG3.1", "14-3-3 ", "Synonym", 111),
        ],
    }
    return World("sus_scrofa", {111: genes}, externals=externals)


def test_external_id_keeps_the_spelling_of_its_latest_release(monkeypatch, tmp_path):
    """One node per identifier whatever the case; its name is the spelling Ensembl gives last (graph-2)."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert "WNT4" in g and "Wnt4" not in g  # `Wnt4` was last seen at 113, `WNT4` at 115
    assert "STAT2" in g and "stat2" not in g  # both at 115: the spelling with more upper-case letters
    assert g.spelling("STAT2", "VGNC Symbol") == "STAT2"
    assert g.spelling("STAT2", "UniProtKB Gene Name") == "stat2"
    assert g.spelling("WNT4", "Old Symbol") == "Wnt4"
    # the merged node carries the links of every spelling
    assert g.connection("STAT2", "ENSSSCG1.1").as_dict == {
        "UniProtKB Gene Name": {111: {113, 114, 115}},
        "VGNC Symbol": {111: {114, 115}},
    }
    # both spellings are written at the last release, each by one database: each is the name exactly
    assert g.resolve_name("stat2") == ("STAT2", "exact")
    assert g.resolve_name("Stat2") == ("STAT2", "case")
    assert g.resolve_name("STAT2") == ("STAT2", "exact")


def test_spelling_of_a_release_is_how_the_database_wrote_the_id_at_that_release(monkeypatch, tmp_path):
    """A database that changed the letter case of a name is answered at each release as it wrote it there."""
    genes = lifetimes({"ENSSSCG1": [(113, 115, 1)], "ENSSSCG2": [(113, 115, 1)], "ENSSSCG3": [(113, 115, 1)]})
    name, symbol = "UniProtKB Gene Name", "VGNC Symbol"
    externals = {
        113: [("ENSSSCG1.1", "ZNF1", name, 111), ("ENSSSCG1.1", "ZNF1", symbol, 111)],
        114: [("ENSSSCG1.1", "Znf1", name, 111), ("ENSSSCG1.1", "ZNF1", symbol, 111)],
        115: [
            ("ENSSSCG1.1", "Znf1", name, 111),
            ("ENSSSCG1.1", "ZNF1", symbol, 111),
            # two spellings in one database at one release: the one with more links there
            ("ENSSSCG2.1", "Abc", name, 111),
            ("ENSSSCG3.1", "Abc", name, 111),
            ("ENSSSCG3.1", "ABC", name, 111),
        ],
    }
    world = World("sus_scrofa", {111: genes}, externals=externals)
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 111))
    built = maker.get_graph(form_list=["gene"])

    for g in (built, TheGraph.load(maker.create_file_name(form_list=["gene"]))):
        assert "ZNF1" in g and "Znf1" not in g  # both at 115, one link each: more upper-case letters
        assert g.spelling("ZNF1", symbol) == "ZNF1" and g.spelling("ZNF1", symbol, 113) == "ZNF1"
        assert g.spelling("ZNF1", name) == "Znf1"  # without a release: as the database writes it last
        assert [g.spelling("ZNF1", name, release) for release in (113, 114, 115)] == ["ZNF1", "Znf1", "Znf1"]
        assert g.spelling("ZNF1", name, 112) == "Znf1"  # a release at which the database does not have it
        assert "Abc" in g and "ABC" not in g and g.spelling("Abc", name, 115) == "Abc"
        assert g.spelling("ZNF1", "No Such Database", 115) == "ZNF1"
    track = Track.from_graph(built)
    assert track.gene_to_database("ENSSSCG1.1", name, 113) == (["ZNF1"], 0)
    assert track.gene_to_database("ENSSSCG1.1", name, 115) == (["Znf1"], 0)
    old = track.convert("ENSSSCG1.1", to_release=113, final_database=name)
    assert list(old["ENSSSCG1.1"]["final_conversion"]["final_elements"]) == ["ZNF1"]
    assert list(track.convert("znf1", final_database=name)["ENSSSCG1.1"]["final_conversion"]["final_elements"]) == [
        "Znf1"
    ]


def test_frozen_parallel_assembly_does_not_supply_the_spelling_of_a_release_it_does_not_serve(monkeypatch, tmp_path):
    """Decision 29: GRCh37 still lists ``FEAT`` at the releases of GRCh38, which writes ``feat`` now."""
    grch37 = lifetimes({"ENSG1": [(74, 77, 1)], "ENSG5": [(74, 77, 1)]})
    grch38 = lifetimes({"ENSG1": [(76, 77, 1)]})
    name = "UniProtKB Gene Name"
    externals = {
        74: [("ENSG1.1", "FEAT", name, 37)],
        75: [("ENSG1.1", "FEAT", name, 37)],
        76: [("ENSG1.1", "FEAT", name, 37), ("ENSG1.1", "FEAT", name, 38), ("ENSG5.1", "Only", name, 37)],
        # three links of GRCh37 against one of GRCh38
        77: [
            ("ENSG1.1", "FEAT", name, 37),
            ("ENSG5.1", "FEAT", name, 37),
            ("ENSG1.1", "FEAT", "Other Name", 37),
            ("ENSG1.1", "feat", name, 38),
            ("ENSG5.1", "ONLY", name, 37),
        ],
    }
    g = build(World("homo_sapiens", {37: grch37, 38: grch38}, externals=externals), monkeypatch, tmp_path, 38)

    assert g.backbone_assembly == {74: 37, 75: 37, 76: 38, 77: 38}
    assert "feat" in g and "FEAT" not in g  # the spelling of the assembly that serves the latest release
    assert [g.spelling("feat", name, release) for release in (74, 75, 76, 77)] == ["FEAT", "FEAT", "FEAT", "feat"]
    assert g.spelling("feat", name) == "feat"
    assert g.spelling("feat", "Other Name") == "FEAT"  # a database that only the frozen assembly has: as it writes it
    # an id that no serving assembly has at any release is spelled as its own assembly spells it last
    assert "ONLY" in g and "Only" not in g and g.spelling("ONLY", name, 76) == "Only"
    track = Track.from_graph(g)
    assert track.gene_to_database("ENSG1.1", name) == (["feat"], 0)
    assert track.gene_to_database("ENSG1.1", name, 76) == (["FEAT"], 0)
    assert track.gene_to_database("ENSG1.1", name, 75) == (["FEAT"], 0)


def respelled_names_world():
    """Pig genes whose databases changed the letter case of a name, and another gene with that string as a synonym.

    NCBI writes the name of ENSSSCG1 ``PRR2`` at 113 and 114; UniProt writes it ``prr2`` at 115, so the node is
    ``prr2``. ENSSSCG2 has the synonym ``PRR2`` at every release. ``STAT2`` (VGNC) and ``stat2`` (UniProt) are one
    name of ENSSSCG3 written in two ways at 115, and ``stat2`` is also a synonym of ENSSSCG2. ``Gone1`` names
    ENSSSCG3 at 113 only, where it was written ``GONE1`` by a second database.
    """
    genes = lifetimes({f"ENSSSCG{i}": [(113, 115, 1)] for i in (1, 2, 3)})
    synonym, ncbi, uniprot, vgnc = DB.synonym_id_nodes_prefix, "EntrezGene", "UniProtKB Gene Name", "VGNC Symbol"
    always = [
        ("ENSSSCG2.1", f"{synonym}PRR2", f"{synonym}{vgnc}", 111),
        ("ENSSSCG2.1", f"{synonym}stat2", f"{synonym}{vgnc}", 111),
    ]
    early = always + [("ENSSSCG1.1", "PRR2", ncbi, 111)]
    externals = {
        113: early + [("ENSSSCG3.1", "Gone1", uniprot, 111), ("ENSSSCG3.1", "GONE1", ncbi, 111)],
        114: early + [("ENSSSCG3.1", "STAT2", vgnc, 111)],
        115: always
        + [
            ("ENSSSCG1.1", "prr2", uniprot, 111),
            ("ENSSSCG3.1", "STAT2", vgnc, 111),
            ("ENSSSCG3.1", "stat2", uniprot, 111),
        ],
    }
    return World("sus_scrofa", {111: genes}, externals=externals)


def test_name_as_a_database_wrote_it_at_the_release_is_an_exact_match(monkeypatch, tmp_path):
    """Decision 38: the step "exact" is judged against the spelling of the release, before any synonym step."""
    from idtrack import API

    world = respelled_names_world()
    world.install(monkeypatch)
    g = GraphMaker(world.manager(tmp_path, 111, first=DB.first_release("sus_scrofa"))).get_graph()
    synonym, ncbi, uniprot, vgnc = DB.synonym_id_nodes_prefix, "EntrezGene", "UniProtKB Gene Name", "VGNC Symbol"
    assert "prr2" in g and "PRR2" not in g  # one node, named as the identifier is written at its last release
    assert [g.spelling("prr2", ncbi, release) for release in (113, 114)] == ["PRR2", "PRR2"]

    # typed as the database wrote the name at the release asked about: that name, not the other gene's synonym
    assert g.resolve_name("PRR2", 113) == ("prr2", "exact") and g.resolve_name("PRR2", 114) == ("prr2", "exact")
    # at 115 no database writes the name so, and the string is exactly the synonym
    assert g.resolve_name("PRR2", 115) == (f"{synonym}PRR2", "synonym:exact")
    assert g.resolve_name("PRR2") == (f"{synonym}PRR2", "synonym:exact")
    assert g.resolve_name("prr2", 115) == ("prr2", "exact") and g.resolve_name("prr2") == ("prr2", "exact")
    # today's spelling typed for a release that wrote it otherwise: the name by letter case (no synonym is `prr2`)
    assert g.resolve_name("prr2", 113) == ("prr2", "case") and g.resolve_name("Prr2", 113) == ("prr2", "case")
    # without a release the spelling is that of the release a conversion reads the name at (decision 1): the
    # target release where the name exists there
    assert g.resolve_name("PRR2", None, None, 113) == ("prr2", "exact")
    assert g.resolve_name("PRR2", None, None, 115) == (f"{synonym}PRR2", "synonym:exact")
    assert g.resolve_name("PRR2", 115, None, 113) == (f"{synonym}PRR2", "synonym:exact")  # a release was named

    # one name written in two ways by two databases at one release: each is the name exactly
    assert g.resolve_name("STAT2", 115) == ("STAT2", "exact") and g.resolve_name("stat2", 115) == ("STAT2", "exact")
    assert g.resolve_name("stat2", 114) == (f"{synonym}stat2", "synonym:exact")  # only VGNC's `STAT2` at 114
    assert g.resolve_name("Stat2", 115) == ("STAT2", "case")
    # with the database named, its own spelling of the release decides
    assert g.resolve_name("stat2", 115, uniprot) == ("STAT2", "exact")
    assert g.resolve_name("stat2", 115, vgnc) == ("STAT2", "case")
    assert g.resolve_name("STAT2", 115, uniprot) == ("STAT2", "case")
    assert g.resolve_name("PRR2", 113, ncbi) == ("prr2", "exact")
    assert g.resolve_name("PRR2", 115, ncbi) == ("prr2", "exact")  # no id of NCBI at 115: as NCBI wrote it last
    assert g.resolve_name("prr2", 115, ncbi) == ("prr2", "case")
    # an identifier that does not exist at the release asked about is judged as at its last release
    assert g.resolve_name("Gone1", 115) == ("GONE1", "exact") and g.resolve_name("GONE1", 115) == ("GONE1", "exact")
    assert g.resolve_name("gone1", 115) == ("GONE1", "case") and g.resolve_name("Gone1", 113, ncbi) == ("GONE1", "case")

    track = Track.from_graph(g)
    assert list(track.convert("PRR2", from_release=113)) == ["ENSSSCG1.1"]
    assert list(track.convert("PRR2", from_release=115)) == ["ENSSSCG2.1"]
    assert list(track.convert("PRR2")) == ["ENSSSCG2.1"]
    old = track.convert("PRR2", to_release=113)  # read at the target release, where it is the name
    assert list(old) == ["ENSSSCG1.1"] and old["ENSSSCG1.1"]["from_release"] == 113
    assert list(track.convert("stat2", from_release=115)) == ["ENSSSCG3.1"]
    assert list(track.convert("stat2", from_release=114)) == ["ENSSSCG2.1"]

    api = API(str(tmp_path))
    api.build_graph("sus_scrofa", 115)
    then = api.convert_identifier("PRR2", from_release=113, to_release=113)
    assert (then["graph_id"], then["match"], then["target_id"]) == ("prr2", "exact", ["ENSSSCG1.1"])
    now = api.convert_identifier("PRR2", from_release=115)
    assert (now["graph_id"], now["match"], now["target_id"]) == (f"{synonym}PRR2", "synonym:exact", ["ENSSSCG2.1"])
    back = api.convert_identifier("PRR2", to_release=113)
    assert (back["graph_id"], back["match"], back["target_id"], back["from_release"]) == (
        "prr2",
        "exact",
        ["ENSSSCG1.1"],
        113,
    )


def frozen_names_world():
    """GRCh37 serves the releases 74 and 75 and is still listed, frozen, at 76 and 77, which GRCh38 serves.

    ``OLDN`` names ENSG1 until 75; on GRCh38 ENSG1 is ``NEWN`` and ``OLDN`` is a synonym of ENSG2. ``GONE`` names
    ENSG3 until 75 and is nothing on GRCh38. The frozen assembly goes on listing both at 76 and 77. ``PARA`` is a
    name that only the frozen assembly has, and only at the releases that GRCh38 serves.
    """
    grch37 = lifetimes({"ENSG1": [(74, 77, 1)], "ENSG3": [(74, 77, 1)]})
    grch38 = lifetimes({"ENSG1": [(76, 77, 1)], "ENSG2": [(76, 77, 1)], "ENSG3": [(76, 77, 1)]})
    synonym, symbol = DB.synonym_id_nodes_prefix, "HGNC Symbol"
    frozen = [("ENSG1.1", "OLDN", symbol, 37), ("ENSG3.1", "GONE", symbol, 37)]
    today = frozen + [
        ("ENSG1.1", "PARA", symbol, 37),
        ("ENSG1.1", "NEWN", symbol, 38),
        ("ENSG2.1", f"{synonym}OLDN", f"{synonym}{symbol}", 38),
        ("ENSG3.1", "Third", symbol, 38),
    ]
    return World("homo_sapiens", {37: grch37, 38: grch38}, externals={74: frozen, 75: frozen, 76: today, 77: today})


def test_string_exists_at_a_release_only_on_the_assembly_that_serves_it(monkeypatch, tmp_path, caplog):
    """Decision 37: a name that only the frozen assembly still lists does not exist at a release of GRCh38."""
    import logging

    from idtrack import API

    world = frozen_names_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 38, first=DB.first_release("homo_sapiens")))
    g = maker.get_graph()
    # a graph read from its file answers without announcing a computation: what it needs came with the file
    loaded = TheGraph.load(maker.create_file_name())
    with caplog.at_level(logging.INFO, logger="idtrack"):
        caplog.clear()  # what the build logged before
        assert loaded.resolve_name("OLDN", 77)[1] == "synonym:exact" and loaded.serving_line == g.serving_line
        assert list(Track.from_graph(loaded).convert("OLDN")) == ["ENSG2.1"]
        assert "being calculated" not in caplog.text
    synonym, symbol = DB.synonym_id_nodes_prefix, "HGNC Symbol"
    assert g.backbone_assembly == {74: 37, 75: 37, 76: 38, 77: 38}
    assert g.serving_line == {37: {74, 75}, 38: {76, 77}}
    assert g.connection("OLDN", "ENSG1.1").as_dict == {symbol: {37: {74, 75, 76, 77}}}  # listed to the last release

    # the old name is today a synonym of another gene: read as the synonym where GRCh38 serves, as the name before
    for release in (None, 76, 77):
        assert g.resolve_name("OLDN", release) == (f"{synonym}OLDN", "synonym:exact")
        assert g.resolve_name("oldn", release) == (f"{synonym}OLDN", "synonym:case")
    assert g.resolve_name("OLDN", 75) == ("OLDN", "exact") and g.resolve_name("OLDN", 74) == ("OLDN", "exact")
    assert g.resolve_name("OLDN", None, symbol) == ("OLDN", "exact")  # named as a symbol: the symbol it was
    assert g.resolve_name("OLDN", 77, f"{synonym}{symbol}") == (f"{synonym}OLDN", "synonym:exact")
    # a name that GRCh38 gave up and nothing replaced: found where it exists, which is not at release 77
    assert g.resolve_name("GONE") == ("GONE", "exact") and g._exists_at("GONE", 77, None) == (False, True)
    assert g._exists_at("GONE", 75, None) == (True, True) and g._exists_at("GONE", 77, symbol) == (False, True)
    # an identifier that only the frozen assembly ever has exists at no release, and is still read
    assert g._exists_at("PARA", 77, None) == (False, True) and g._exists_at("PARA", 77, symbol) == (False, True)
    assert g.resolve_name("PARA") == ("PARA", "exact") and g.resolve_name("para", 77) == ("PARA", "case")
    assert g.resolve_name("PARA", 77, symbol) == ("PARA", "exact")

    track = Track.from_graph(g)
    assert list(track.convert("OLDN")) == ["ENSG2.1"]
    assert list(track.convert("OLDN", from_release=77)) == ["ENSG2.1"]
    assert list(track.convert("OLDN", from_release=75, to_release=75)) == ["ENSG1.1"]
    gone = track.convert("GONE")
    assert list(gone) == ["ENSG3.1"] and gone["ENSG3.1"]["from_release"] == 75  # read at its last release, followed
    para = track.convert("PARA")
    assert list(para) == ["ENSG1.1"] and para["ENSG1.1"]["from_release"] == 77

    api = API(str(tmp_path))
    api.build_graph("homo_sapiens", 77)
    for from_release in (None, 77):
        today = api.convert_identifier("OLDN", from_release=from_release)
        assert (today["graph_id"], today["match"], today["target_id"]) == (
            f"{synonym}OLDN",
            "synonym:exact",
            ["ENSG2.1"],
        )
        assert (today["from_release"], today["from_release_fallback"]) == (77, False)
    then = api.convert_identifier("OLDN", from_release=75, to_release=75)
    assert (then["graph_id"], then["match"], then["target_id"]) == ("OLDN", "exact", ["ENSG1.1"])
    only = api.convert_identifier("PARA", from_release=77)
    assert (only["graph_id"], only["match"], only["target_id"]) == ("PARA", "exact", ["ENSG1.1"])
    assert (only["from_release"], only["from_release_fallback"], only["no_corresponding"]) == (77, False, False)


def test_external_ids_are_stripped_of_surrounding_white_space(monkeypatch, tmp_path):
    """An identifier with a trailing space in Ensembl's table is the identifier without it (graph-16)."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert "14-3-3" in g and "14-3-3 " not in g
    assert not any(name != name.strip() for name in g.nodes)


def test_separator_variant_wins_over_a_cut_suffix(monkeypatch, tmp_path):
    """``SLA_1`` is ``SLA-1``, never ``SLA`` (graph-new-1); the match is reported as a separator variant."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert g.resolve_name("SLA_1") == ("SLA-1", "separator")
    assert g.resolve_name("sla_1") == ("SLA-1", "separator")
    assert g.resolve_name("SLA-1") == ("SLA-1", "exact")


def test_suffix_of_an_external_id_is_never_cut_off(monkeypatch, tmp_path):
    """An unknown ``X-12`` is not ``X`` and ``NM_713028147`` is not ``NM`` (graph-16)."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert g.resolve_name("SLA-12") == (None, None)
    assert g.resolve_name("SLA_12") == (None, None)
    assert g.resolve_name("WNT4_713028147") == (None, None)


def test_ensembl_versions_are_read_with_any_separator(monkeypatch, tmp_path):
    """``ENSG..._2`` is ``ENSG....2``; a symbol followed by ``.<digits>`` is not the symbol."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert g.resolve_name("ENSSSCG1.1") == ("ENSSSCG1.1", "exact")
    assert g.resolve_name("ENSSSCG1_1") == ("ENSSSCG1.1", "separator")
    assert g.resolve_name("ensssCG1-1") == ("ENSSSCG1.1", "separator")
    assert g.resolve_name("SLA.1") == ("SLA-1", "separator")  # as R writes `SLA-1`; not a version of `SLA`
    assert g.resolve_name("SLA.7") == (None, None)  # `SLA` is a VGNC symbol: nothing is cut off a symbol
    assert g.resolve_name("ENSSSCG1") == ("ENSSSCG1", "exact")


def versions_world():
    """Pig genes with several versions, a retired one, and ids of accession, symbol and gene-name databases.

    G1 had version 2 at 110 and 111 and has version 4 since 112 (versions 1 and 3 never existed); G2 was retired
    after 112; G3 is current. Transcripts and translations follow their genes with version 1.
    """
    genes = lifetimes(
        {"ENSSSCG1": [(110, 111, 2), (112, 115, 4)], "ENSSSCG2": [(110, 112, 6)], "ENSSSCG3": [(110, 115, 1)]}
    )
    per_release = [
        ("ENSSSCT3.1", "NM_0001", "RefSeq_mRNA", 111, "transcript"),
        ("ENSSSCP3.1", "P04637", "Uniprot/SWISSPROT", 111, "translation"),
        ("ENSSSCG3.1", "SLA", "VGNC Symbol", 111),
        ("ENSSSCG3.1", "ABC1", "UniProtKB Gene Name", 111),
        # one string that is a RefSeq accession and a gene symbol
        ("ENSSSCT3.1", "BOTH", "RefSeq_mRNA", 111, "transcript"),
        ("ENSSSCG3.1", "BOTH", "VGNC Symbol", 111),
    ]
    return World("sus_scrofa", {111: genes}, externals={115: per_release, 114: per_release})


def test_version_newer_than_the_graph_is_read_as_the_latest_version(monkeypatch, tmp_path):
    """Decision 14: a higher version of an id that is active at the last release may be newer than the graph."""
    world = versions_world()
    os.makedirs(tmp_path, exist_ok=True)
    world.install(monkeypatch)
    g = GraphMaker(world.manager(tmp_path, 111)).construct_graph()

    assert g.versions_of("ENSSSCG1") == ("ENSSSCG1.2", "ENSSSCG1.4")
    for written in ("ENSSSCG1.5", "ENSSSCG1.99", "ENSSSCG1_5", "ensssCG1-5"):
        assert g.resolve_name(written) == ("ENSSSCG1.4", "version_newer_than_graph"), written
        assert g.known_versions(written) == ("ENSSSCG1.2", "ENSSSCG1.4")
    # transcripts and translations have no node without version: the same rule through their versions
    assert g.resolve_name("ENSSSCT1.2") == ("ENSSSCT1.1", "version_newer_than_graph")
    assert g.resolve_name("ENSSSCP3.7") == ("ENSSSCP3.1", "version_newer_than_graph")
    # a version that the graph holds is itself, and has no "known versions" to report
    assert g.resolve_name("ENSSSCG1.2") == ("ENSSSCG1.2", "exact") and g.known_versions("ENSSSCG1.2") == ()

    track = Track.from_graph(g)
    assert list(track.convert("ENSSSCG1.99")) == ["ENSSSCG1.4"]
    assert track.convert("ENSSSCG1.99")["ENSSSCG1.4"]["from_id"] == "ENSSSCG1.4"


def test_other_unknown_versions_match_nothing_and_the_versions_that_exist_are_named(monkeypatch, tmp_path):
    """Decision 13 and 14: a lower version, one in a gap, or any version of a retired id is not another version."""
    from idtrack import UnknownVersionError

    world = versions_world()
    os.makedirs(tmp_path, exist_ok=True)
    world.install(monkeypatch)
    g = GraphMaker(world.manager(tmp_path, 111)).construct_graph()
    track = Track.from_graph(g)

    cases = {
        "ENSSSCG1.1": ("ENSSSCG1.2", "ENSSSCG1.4"),  # lower than every version
        "ENSSSCG1.3": ("ENSSSCG1.2", "ENSSSCG1.4"),  # in a gap
        "ENSSSCG1_3": ("ENSSSCG1.2", "ENSSSCG1.4"),
        "ENSSSCG2.7": ("ENSSSCG2.6",),  # higher, but the id was retired after 112: the graph knows all its versions
        "ENSSSCG2.5": ("ENSSSCG2.6",),
        "ENSSSCT2.9": ("ENSSSCT2.1",),  # a transcript of the retired gene
    }
    for written, held in cases.items():
        assert g.resolve_name(written) == (None, None), written
        assert g.known_versions(written) == held, written
        with pytest.raises(UnknownVersionError) as caught:
            track.convert(written)
        assert caught.value.versions == held and caught.value.identifier == written
        assert ", ".join(held) in str(caught.value)
    # not an Ensembl id of the graph at all: unknown, with nothing to name
    for written in ("ENSSSCG9.1", "NOPE.3", "SLA.7"):
        assert g.resolve_name(written) == (None, None) and g.known_versions(written) == ()
        assert track.convert(written) is None


def test_version_suffix_is_cut_only_off_ids_of_accession_databases(monkeypatch, tmp_path):
    """Decision 20: `NM_0001.9` is `NM_0001`; a symbol, a gene name, or a string that is also a symbol is not cut."""
    world = versions_world()
    os.makedirs(tmp_path, exist_ok=True)
    world.install(monkeypatch)
    g = GraphMaker(world.manager(tmp_path, 111)).construct_graph()

    assert g.resolve_name("NM_0001.9") == ("NM_0001", "version")
    assert g.resolve_name("nm_0001.12") == ("NM_0001", "version")
    assert g.resolve_name("P04637.4") == ("P04637", "version")
    assert g.resolve_name("NM_0001_9") == (None, None) and g.resolve_name("NM_0001-9") == (None, None)
    assert g.resolve_name("SLA.7") == (None, None)  # a symbol of the organism
    assert g.resolve_name("ABC1.2") == (None, None)  # a gene name of UniProt: not an accession
    assert g.resolve_name("BOTH.3") == (None, None)  # an accession that is also a symbol
    assert g.resolve_name("BOTH") == ("BOTH", "exact")
    track = Track.from_graph(g)
    assert list(track.convert("NM_0001.9")) == ["ENSSSCG3.1"] and track.convert("SLA.7") is None


def test_synonym_is_found_and_reported_as_such():
    """A name that is only a synonym resolves to the synonym node, and the way of matching says so."""
    g = TheGraph(genome_assembly=38, confident_for_release=[100])
    g.add_node(f"{DB.synonym_id_nodes_prefix}CBF-A", **{DB.node_type_str: DB.nts_external})
    g.add_node("NFYA", **{DB.node_type_str: DB.nts_external})

    assert g.resolve_name("CBF-A") == (f"{DB.synonym_id_nodes_prefix}CBF-A", "synonym:exact")
    assert g.resolve_name("cbf_a") == (f"{DB.synonym_id_nodes_prefix}CBF-A", "synonym:separator")
    assert g.resolve_name("nfya") == ("NFYA", "case")


def test_each_step_is_tried_on_names_and_then_on_synonyms_before_the_next_step(monkeypatch, tmp_path):
    """Decision 28: an exact synonym wins over a separator variant of another gene's name; a name wins in a step."""
    genes = lifetimes({"ENSMUSG1": [(103, 104, 1)], "ENSMUSG2": [(103, 104, 1)], "ENSMUSG3": [(103, 104, 1)]})
    synonym = f"{DB.synonym_id_nodes_prefix}"
    externals = {
        104: [
            ("ENSMUSG1.1", "H3-3b", "MGI Symbol", 39),  # a name of one gene
            ("ENSMUSG2.1", f"{synonym}H3.3B", f"{synonym}MGI Symbol", 39),  # a synonym of another
            ("ENSMUSG3.1", "Abc1", "MGI Symbol", 39),  # a string that is a name and a synonym
            ("ENSMUSG1.1", f"{synonym}Abc1", f"{synonym}MGI Symbol", 39),
            ("ENSMUSG3.1", f"{synonym}NM_77", f"{synonym}RefSeq_mRNA", 39),
        ]
    }
    g = build(World("mus_musculus", {39: genes}, externals=externals), monkeypatch, tmp_path, 39)

    assert g.resolve_name("H3.3B") == (f"{synonym}H3.3B", "synonym:exact")  # not the name `H3-3b`
    assert g.resolve_name("h3.3b") == (f"{synonym}H3.3B", "synonym:case")  # nor here: case comes before separators
    assert g.resolve_name("H3-3b") == ("H3-3b", "exact")
    assert g.resolve_name("h3-3B") == ("H3-3b", "case")
    assert g.resolve_name("H3_3b") == ("H3-3b", "separator")  # no synonym is written so: the name's variant
    assert g.resolve_name("Abc1") == ("Abc1", "exact") and g.resolve_name("ABC1") == ("Abc1", "case")
    assert g.resolve_name(f"{synonym}Abc1") == (f"{synonym}Abc1", "exact")
    track = Track.from_graph(g)
    assert list(track.convert("H3.3B")) == ["ENSMUSG2.1"] and list(track.convert("H3_3b")) == ["ENSMUSG1.1"]


def reused_strings_world():
    """Five mouse genes whose databases reuse strings over the releases 103 to 106.

    ``slit2`` is a synonym of ENSMUSG1 until 104, and ``Slit2`` is the name of ENSMUSG2 from 105; ``Apc10`` is the
    name of ENSMUSG3 until 104 and a synonym of ENSMUSG4 from 105; ``Ab-1`` names ENSMUSG1 until 104 and ``Ab_1``
    names ENSMUSG2 from 105; ENSMUSG5 is merged into ENSMUSG3 between 104 and 105, and the symbol database lists
    the retired id as a synonym of ENSMUSG4 afterwards.
    """
    genes = lifetimes({f"ENSMUSG{i}": [(103, 106, 1)] for i in (1, 2, 3, 4)} | {"ENSMUSG5": [(103, 104, 1)]})
    synonym, symbol = DB.synonym_id_nodes_prefix, "MGI Symbol"
    early = [
        ("ENSMUSG1.1", f"{synonym}slit2", f"{synonym}{symbol}", 39),
        ("ENSMUSG3.1", "Apc10", symbol, 39),
        ("ENSMUSG1.1", "Ab-1", symbol, 39),
    ]
    late = [
        ("ENSMUSG2.1", "Slit2", symbol, 39),
        ("ENSMUSG4.1", f"{synonym}Apc10", f"{synonym}{symbol}", 39),
        ("ENSMUSG2.1", "Ab_1", symbol, 39),
        ("ENSMUSG4.1", f"{synonym}ENSMUSG5", f"{synonym}{symbol}", 39),
    ]
    return World(
        "mus_musculus",
        {39: genes},
        events=[(1, "ENSMUSG5", 1, "ENSMUSG3", 1, 0.9)],
        sessions=[(1, 104, 105)],
        externals={103: early, 104: early, 105: late, 106: late},
    )


def test_string_means_the_node_that_exists_at_the_release_asked_about(monkeypatch, tmp_path):
    """Decision 30: existence at the release decides before the spelling does; then the order of the steps."""
    g = build(reused_strings_world(), monkeypatch, tmp_path, 39)
    synonym = DB.synonym_id_nodes_prefix

    # an old synonym of one gene and, in another letter case, the name of another gene today
    assert g.resolve_name("slit2") == ("Slit2", "case")  # no release named: the last release of the graph
    assert g.resolve_name("slit2", 106) == ("Slit2", "case") and g.resolve_name("slit2", 105) == ("Slit2", "case")
    assert g.resolve_name("slit2", 103) == (f"{synonym}slit2", "synonym:exact")
    # an old name of one gene and a synonym of another gene today
    assert g.resolve_name("Apc10") == (f"{synonym}Apc10", "synonym:exact")
    assert g.resolve_name("Apc10", 104) == ("Apc10", "exact")
    assert g.resolve_name("apc10") == (f"{synonym}Apc10", "synonym:case")
    assert g.resolve_name("apc10", 103) == ("Apc10", "case")
    # two names that differ in the separator only: the one of the release
    assert g.resolve_name("ab.1") == ("Ab_1", "separator") and g.resolve_name("ab.1", 103) == ("Ab-1", "separator")
    assert g.resolve_name("Ab-1") == ("Ab_1", "separator") and g.resolve_name("Ab-1", 103) == ("Ab-1", "exact")
    # where none of them exists (a release the graph does not have): the first in the order of the steps
    assert g.resolve_name("slit2", 90) == (f"{synonym}slit2", "synonym:exact")
    assert g.resolve_name("Apc10", 90) == ("Apc10", "exact") and g.resolve_name("ab.1", 90) == ("Ab-1", "separator")
    # a string that is spelt as a node says so when asked for it, and is still read by existence
    assert g.resolve_name(f"{synonym}slit2") == (f"{synonym}slit2", "exact")  # nothing else it can mean
    assert g.resolve_name("slit2", np.int64(103)) == (f"{synonym}slit2", "synonym:exact")

    # with the database named, a node exists where it is an id of that database
    symbol = "MGI Symbol"
    assert g.resolve_name("Apc10", None, symbol) == ("Apc10", "exact")  # today's synonym is no MGI symbol
    assert g.resolve_name("Apc10", None, f"{synonym}{symbol}") == (f"{synonym}Apc10", "synonym:exact")
    assert g.resolve_name("slit2", 103, symbol) == ("Slit2", "case")  # the old synonym is no MGI symbol either
    assert g.resolve_name("ab.1", None, symbol) == ("Ab_1", "separator")
    assert g.resolve_name("ab.1", 103, symbol) == ("Ab-1", "separator")
    assert g.resolve_name("Apc10", None, "RefSeq_mRNA") == ("Apc10", "exact")  # of no node: the order of the steps


def test_ensembl_id_is_never_passed_over_for_a_synonym_spelt_the_same(monkeypatch, tmp_path):
    """Decision 30: a retired Ensembl id is an id of Ensembl's history, whatever a database lists it as today."""
    g = build(reused_strings_world(), monkeypatch, tmp_path, 39)
    synonym = DB.synonym_id_nodes_prefix
    assert g.is_active(f"{synonym}ENSMUSG5", 106) and not g.is_active("ENSMUSG5.1", 106)

    for release in (None, 103, 106):
        assert g.resolve_name("ENSMUSG5", release) == ("ENSMUSG5", "exact")
        assert g.resolve_name("ensmusg5", release) == ("ENSMUSG5", "case")
        assert g.resolve_name("ENSMUSG5.1", release) == ("ENSMUSG5.1", "exact")
        assert g.resolve_name("ENSMUSG5_1", release) == ("ENSMUSG5.1", "separator")
    track = Track.from_graph(g)
    assert list(track.convert("ENSMUSG5")) == ["ENSMUSG3.1"]  # its successor, not the gene of today's synonym
    assert list(track.convert(f"{synonym}ENSMUSG5")) == ["ENSMUSG4.1"]


def test_engine_and_api_read_a_string_by_the_one_rule_of_the_graph(monkeypatch, tmp_path):
    """Decision 30: the engine does not take a string that happens to be a node as that node; the API agrees."""
    from idtrack import API

    world = reused_strings_world()
    world.install(monkeypatch)
    g = GraphMaker(world.manager(tmp_path, 39, first=DB.first_release("mus_musculus"))).get_graph()
    synonym = DB.synonym_id_nodes_prefix
    track = Track.from_graph(g)

    # `Apc10` is a node (an old name): without a release the string is today's synonym of another gene
    assert "Apc10" in g
    assert list(track.convert("Apc10")) == ["ENSMUSG4.1"]
    assert list(track.convert("Apc10", from_release=103)) == ["ENSMUSG3.1"]
    assert list(track.convert("slit2")) == ["ENSMUSG2.1"] and list(track.convert("slit2", from_release=104)) == [
        "ENSMUSG1.1"
    ]
    assert list(track.convert("ab.1")) == ["ENSMUSG2.1"] and list(track.convert("ab.1", from_release=103)) == [
        "ENSMUSG1.1"
    ]
    assert track.convert("Apc10", from_release=103)["ENSMUSG3.1"]["from_id"] == "Apc10"
    # named as a symbol of the database, the string is the symbol it was, read where it last was one
    named = track.convert("Apc10", from_database="MGI Symbol")
    assert list(named) == ["ENSMUSG3.1"] and named["ENSMUSG3.1"]["from_release"] == 104
    assert list(track.convert("slit2", from_database="MGI Symbol")) == ["ENSMUSG2.1"]
    as_synonym = track.convert("Slit2", from_database=f"{synonym}MGI Symbol")  # the name is no synonym: the old one
    assert list(as_synonym) == ["ENSMUSG1.1"] and as_synonym["ENSMUSG1.1"]["from_id"] == f"{synonym}slit2"
    assert track.convert("Ab_1", from_database=f"{synonym}MGI Symbol") is None  # an id of the other database only
    with pytest.raises(ValueError, match="not an external database of this graph"):
        track.convert("nothing at all", from_database="No Such Database")

    api = API(str(tmp_path))
    api.build_graph("mus_musculus", 106)
    today = api.convert_identifier("Apc10")
    assert (today["graph_id"], today["match"], today["target_id"]) == (
        f"{synonym}Apc10",
        "synonym:exact",
        ["ENSMUSG4.1"],
    )
    then = api.convert_identifier("Apc10", from_release=103)
    assert (then["graph_id"], then["match"], then["target_id"]) == ("Apc10", "exact", ["ENSMUSG3.1"])
    lower = api.convert_identifier("slit2")
    assert (lower["graph_id"], lower["match"], lower["target_id"]) == ("Slit2", "case", ["ENSMUSG2.1"])
    old = api.convert_identifier("slit2", from_release=104)
    assert (old["graph_id"], old["match"], old["target_id"]) == (f"{synonym}slit2", "synonym:exact", ["ENSMUSG1.1"])
    # spelt exactly as today's name, asked from a release that had only the synonym in another letter case
    spelt = api.convert_identifier("Slit2", from_release=103)
    assert (spelt["graph_id"], spelt["match"], spelt["target_id"]) == (
        f"{synonym}slit2",
        "synonym:case",
        ["ENSMUSG1.1"],
    )
    assert spelt["from_release_fallback"] is False
    # a release at which nothing the string can mean exists: read where it is held, by the rule without a release
    moved = api.convert_identifier(f"{synonym}ENSMUSG5", from_release=103)
    assert moved["from_release_fallback"] is True and moved["from_release"] == 106
    assert (moved["graph_id"], moved["match"]) == (f"{synonym}ENSMUSG5", "exact")
    assert moved["target_id"] == ["ENSMUSG4.1"]
    as_symbol = api.convert_identifier("Apc10", from_database="MGI Symbol")
    assert (as_symbol["graph_id"], as_symbol["match"], as_symbol["target_id"]) == ("Apc10", "exact", ["ENSMUSG3.1"])
    batch = api.convert_identifier_multiple(["Apc10", "ENSMUSG5"], verbose=False)
    assert [item["target_id"] for item in batch] == [["ENSMUSG4.1"], ["ENSMUSG3.1"]]

    # the source of a list is inferred from the nodes the strings mean at the last release of the graph
    counts = dict(map(tuple, api.infer_identifier_source(["Apc10"], mode="ensembl_release", report_only_winner=False)))
    assert counts == {105: 1, 106: 1}


def test_unknown_identifier_with_many_separators_is_answered_at_once(monkeypatch, tmp_path):
    """Resolution does not enumerate the 2^n dash/underscore variants (graph-8)."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)
    unknown = "-".join(["a_b"] * 20)  # 39 separators: 2^39 variants

    started = time.perf_counter()
    assert g.resolve_name(unknown) == (None, None)
    assert g.resolve_name(unknown.upper()) == (None, None)
    assert time.perf_counter() - started < 0.5


@pytest.mark.wrong_argument_type
def test_release_of_a_name_must_be_an_integer():
    """A release that is not an integer is refused, not read as "no release"."""
    g = TheGraph(genome_assembly=38, confident_for_release=[100])
    g.add_node("A", **{DB.node_type_str: DB.nts_external})

    for bad in (1.5, "100", True):
        with pytest.raises(TypeError, match="`release` must be an integer Ensembl release"):
            g.resolve_name("A", bad)
        with pytest.raises(TypeError, match="`target` must be an integer Ensembl release"):
            g.resolve_name("A", None, None, bad)
    assert g.resolve_name("A", np.int64(100), None, np.int64(100)) == ("A", "exact")


@pytest.mark.wrong_argument_type
def test_identifier_must_be_a_string():
    """Anything but a string is refused with a clear error."""
    g = TheGraph(genome_assembly=38, confident_for_release=[100])
    g.add_node("A", **{DB.node_type_str: DB.nts_external})

    for bad in (None, 0, 1.5, ["A"]):
        with pytest.raises(TypeError, match="must be a string"):
            g.resolve_name(bad)


def test_base_gene_id_listed_as_external_id_stays_a_base_id(monkeypatch, tmp_path):
    """A database that lists an Ensembl gene id as its identifier does not turn the base id into an external id."""
    genes = lifetimes({"ENSMUSG1": [(103, 104, 1)], "ENSMUSG2": [(103, 104, 2)]})
    externals = {104: [("ENSMUSG2.2", "ENSMUSG1", "Some Database", 39), ("ENSMUSG2.2", "Sym", "MGI Symbol", 39)]}
    g = build(World("mus_musculus", {39: genes}, externals=externals), monkeypatch, tmp_path, 39)

    assert g.node_type("ENSMUSG1") == DB.nts_base_ensembl["gene"]
    assert DB.nts_base_ensembl["gene"] not in g.available_external_databases
    assert g.available_external_databases == {"MGI Symbol"}
    assert g.graph["misplaced_external_entry"] == {"ENSMUSG1"}
    assert [n for n, _ in g.xref_edges("ENSMUSG1")] == ["ENSMUSG1.1"]


# --------------------------------------------------------------------------- assemblies (graph-6, 9, 10, 13)


def handoff_world(externals=None):
    """Sscrofa10.2 serves releases 88 and 89, Sscrofa11.1 releases 90 and 91."""
    old = lifetimes({"ENSSSCG1": [(88, 89, 1)], "ENSSSCG2": [(88, 89, 1)]})
    new = lifetimes({"ENSSSCG2": [(90, 91, 2)], "ENSSSCG3": [(90, 91, 1)]})
    return World("sus_scrofa", {102: old, 111: new}, externals=externals)


def test_assembly_without_external_ids_still_has_base_ids(monkeypatch, tmp_path):
    """Base ids exist for every assembly of the graph, not only for those with external edges (graph-13)."""
    world = handoff_world(externals={91: [("ENSSSCG3.1", "SYM", "VGNC Symbol", 111)]})
    g = build(world, monkeypatch, tmp_path, 111)

    assert g.node_type("ENSSSCG1") == DB.nts_base_ensembl["gene"]
    assert g.connection("ENSSSCG1", "ENSSSCG1.1").as_dict == {"base_ensembl_gene": {102: {88, 89}}}
    assert g.connection("ENSSSCG2", "ENSSSCG2.2").as_dict == {"base_ensembl_gene": {111: {90, 91}}}
    assert g.active_ranges("ENSSSCG2") == [[88, 91]]
    assert g.available_genome_assemblies == {102, 111}


def test_all_inclusive_ranges_of_a_gene_that_never_lived_on_the_primary_assembly(monkeypatch, tmp_path):
    """No ``KeyError`` for genes of releases that an earlier assembly serves (graph-9)."""
    g = build(handoff_world(), monkeypatch, tmp_path, 111)

    assert g.get_active_ranges_of_id_ensembl_all_inclusive("ENSSSCG1.1") == [[88, 89]]
    assert g.get_active_ranges_of_base_id_alternative("ENSSSCG1") == [[88, 89]]
    assert g.get_active_ranges_of_base_id_alternative("ENSSSCG2") == [[88, 91]]
    assert g.graph["backbone_assembly"] == g.backbone_assembly == {88: 102, 89: 102, 90: 111, 91: 111}
    assert g.assembly_releases == {102: (88, 89), 111: (90, 91)}
    assert g.manifest["releases_per_assembly"] == {"102": [88, 89], "111": [90, 91]}


def test_relation_of_a_transcript_is_named_after_the_assembly_of_the_edge():
    """A transcript of an earlier assembly is reported on that assembly, like its gene (graph-10)."""
    g = TheGraph(genome_assembly=111, confident_for_release=[73, 74, 90])
    g.add_node("ENSSSCT1.1", **{DB.node_type_str: DB.nts_ensembl["transcript"]})
    g.add_node("ENSSSCG1.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
    g.add_node("ENSSSCG1", **{DB.node_type_str: DB.nts_base_ensembl["gene"]})
    g.add_edge("ENSSSCT1.1", "ENSSSCG1.1", **{DB.connection_dict: {"ensembl_transcript": {102: {73, 74}, 111: {90}}}})
    g.add_edge("ENSSSCG1", "ENSSSCG1.1", **{DB.connection_dict: {"base_ensembl_gene": {102: {73, 74}, 111: {90}}}})

    assert g.trios("ENSSSCT1.1") == {
        ("assembly_102_ensembl_transcript", 102, 73),
        ("assembly_102_ensembl_transcript", 102, 74),
        ("assembly_111_ensembl_transcript", 111, 90),
    }
    assert g.get_id_list("ensembl_transcript", 102, 73) == ["ENSSSCT1.1"]
    assert g.get_id_list("ensembl_gene", 102, 73) == ["ENSSSCG1.1"]
    assert g.get_id_list("ensembl_transcript", 111, 73) == []
    assert g.assembly_releases == {102: (73, 74), 111: (90,)}
    assert g.backbone_assembly == {73: 102, 74: 102, 90: 111}  # derived from the edges of a graph made by hand


def test_backbone_follows_the_primary_assembly_where_it_has_the_release(monkeypatch, tmp_path):
    """With the older of two overlapping assemblies as primary, the backbone is that assembly's ids (graph-6)."""
    grch37 = lifetimes({"ENSG1": [(100, 101, 1)], "ENSG2": [(100, 101, 1)]})
    grch38 = lifetimes({"ENSG1": [(100, 100, 1), (101, 101, 2)], "ENSG2": [(100, 100, 1)]})
    world = World("homo_sapiens", {37: grch37, 38: grch38})

    g37 = build(world, monkeypatch, tmp_path / "a", 37)
    assert g37.graph["backbone_assembly"] == {100: 37, 101: 37}
    assert g37.active_ranges("ENSG1.1") == [[100, INF]]
    assert g37.active_ranges("ENSG2.1") == [[100, INF]]
    assert "ENSG1.2" not in g37

    g38 = build(world, monkeypatch, tmp_path / "b", 38)
    assert g38.graph["backbone_assembly"] == {100: 38, 101: 38}
    assert g38.active_ranges("ENSG1.1") == [[100, 100]]
    assert g38.active_ranges("ENSG1.2") == [[101, INF]]
    assert g38.active_ranges("ENSG2.1") == [[100, 100]]


def test_graph_of_an_earlier_primary_assembly_ends_at_its_last_release(monkeypatch, tmp_path):
    """The releases after the snapshot of the primary assembly are not part of the graph (graph-6)."""
    g = build(handoff_world(), monkeypatch, tmp_path, 102)

    assert g.releases == (88, 89)
    assert g.active_ranges("ENSSSCG1.1") == [[88, INF]]
    assert "ENSSSCG3.1" not in g


# --------------------------------------------------------------------------- what cannot be built (graph-14)


@pytest.mark.parametrize("unversioned", [True, {100}, {101}])
def test_organism_without_id_versions_is_refused_before_any_work(monkeypatch, tmp_path, unversioned):
    """Decision 27: ids without versions, in all releases or in one, are not supported; the error says so (graph-14)."""
    genes = lifetimes({"YAL001C": [(100, 101, 1)]})
    world = World("homo_sapiens", {38: genes}, unversioned=unversioned)
    world.install(monkeypatch)
    monkeypatch.setattr(GraphMaker, "history", lambda *a, **k: pytest.fail("the history must not be built"))
    maker = GraphMaker(world.manager(tmp_path, 38))

    with pytest.raises(ValueError, match="Organisms whose Ensembl ids carry no versions are not supported") as refusal:
        maker.get_graph(form_list=["gene"])
    release = 100 if unversioned is True else min(unversioned)
    assert f"the gene ids of homo_sapiens have no version at Ensembl release {release}" in str(refusal.value)
    assert not os.path.exists(maker.create_file_name(form_list=["gene"]))


def test_split_id_says_how_many_delimiters_it_found():
    """The error of a malformed node name states the number of delimiters found (graph-new-2)."""
    with pytest.raises(ValueError, match="'ENSG000001' has 0 delimiters"):
        GraphMaker.split_id("ENSG000001", "ID")
    with pytest.raises(ValueError, match="has 2 delimiters"):
        GraphMaker.split_id("ENSG000001.1.2", "ID")
    assert GraphMaker.split_id("ENSG000001.7", "ID") == "ENSG000001"
    assert GraphMaker.split_id("ENSG000001.7", "Version") == "7"


# --------------------------------------------------------------------------- compact and shared (graph-3, graph-4)


def test_release_sets_are_shared_immutable_and_carry_their_mask(monkeypatch, tmp_path):
    """Release sets are interned frozen sets with a bit mask; no edge owns a set of its own (graph-3)."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    stat2 = g.connection("STAT2", "ENSSSCG1.1")
    base = g.connection("ENSSSCG2", "ENSSSCG2.1")
    assert isinstance(stat2.releases, ReleaseSet) and isinstance(stat2.releases, frozenset)
    assert stat2.releases == {113, 114, 115} and stat2.releases.mask == (1 << 113) | (1 << 114) | (1 << 115)
    assert base.releases is stat2.releases  # one object per distinct set
    assert g.connection("ENSSSCG3", "ENSSSCG3.1") is base  # one object per distinct connection
    for node in g.nodes:
        for _, connection in g.xref_edges(node):
            union = set()
            for _database, _assembly, releases in connection.items:
                assert isinstance(releases, ReleaseSet)
                union |= releases
            assert connection.releases == union
    with pytest.raises(AttributeError):
        stat2.releases.add(116)


def test_release_set_and_connection_survive_pickling_as_the_same_objects():
    """Equal release sets and connections are one object, also after pickling."""
    releases = ReleaseSet.of([90, 91, 115])
    connection = Connection.of([("HGNC Symbol", 38, releases), ("EntrezGene", 37, [90])])

    assert ReleaseSet.of({115, 91, 90}) is releases and ReleaseSet.from_mask(releases.mask) is releases
    assert pickle.loads(pickle.dumps(releases)) is releases  # noqa: S301
    assert pickle.loads(pickle.dumps(connection)) is connection  # noqa: S301
    assert connection.as_dict == {"HGNC Symbol": {38: {90, 91, 115}}, "EntrezGene": {37: {90}}}
    assert connection.releases == {90, 91, 115} and connection.databases == {"HGNC Symbol", "EntrezGene"}
    assert sorted(releases) == [90, 91, 115] and 91 in releases and 92 not in releases
    with pytest.raises(ValueError):
        ReleaseSet.of([-1])


def test_no_table_over_all_nodes_is_kept_and_answers_are_the_same(monkeypatch, tmp_path):
    """After ``calculate_caches`` the per-node values are still computed on demand (graph-4)."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)
    before = {n: (g.trios(n), g.active_ranges(n)) for n in g.nodes}
    g.calculate_caches(for_test=True)

    for name, value in vars(g).items():
        if isinstance(value, dict) and name != "graph":
            assert len(value) < len(g), f"{name} holds an entry per node"
    assert g.trios("STAT2") == {
        ("UniProtKB Gene Name", 111, 113),
        ("UniProtKB Gene Name", 111, 114),
        ("UniProtKB Gene Name", 111, 115),
        ("VGNC Symbol", 111, 114),
        ("VGNC Symbol", 111, 115),
    }
    assert g.trios("ENSSSCG1.1") == {("assembly_111_ensembl_gene", 111, r) for r in (113, 114, 115)}
    assert before == {n: (g.trios(n), g.active_ranges(n)) for n in before}


def test_built_graph_reads_as_a_networkx_graph(monkeypatch, tmp_path):
    """A built (frozen) graph is read with networkx as it is with the accessors; both show the same shared objects."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert g.nodes["ENSSSCG1.1"] == {DB.node_type_str: "ensembl_gene", "ID": "ENSSSCG1", "Version": "1"}
    assert g.nodes["STAT2"] == {DB.node_type_str: "external"}
    assert list(g.neighbors("WNT4")) == ["ENSSSCG2.1"]
    data = g.get_edge_data("WNT4", "ENSSSCG2.1")
    connection = {"Old Symbol": {111: {113}}, "VGNC Symbol": {111: {115}}}
    assert data == {0: {"connection": connection, "available_releases": {113, 115}}}
    # the edge data are the objects of the accessors: immutable release sets, one object per distinct value
    assert data[0]["connection"] is g.connection("WNT4", "ENSSSCG2.1").as_dict
    assert data[0]["available_releases"] is g.connection("WNT4", "ENSSSCG2.1").releases
    assert isinstance(data[0]["available_releases"], ReleaseSet) and data[0]["available_releases"].mask == 5 << 113
    assert g.get_edge_data("ENSSSCG1.Void", "ENSSSCG1.1", 0)["new_release"] == 113
    assert g.has_edge("WNT4", "ENSSSCG2.1", 0) and not g.has_edge("ENSSSCG2.1", "WNT4")
    assert "ENSSSCG2.1" in g.rev
    assert set(g.rev.neighbors("ENSSSCG2.1")) == {"ENSSSCG2.Void", "ENSSSCG2.1", "ENSSSCG2", "WNT4", "SLA-1"}
    assert g.rev.get_edge_data("ENSSSCG2.1", "WNT4", 0)["available_releases"] == {113, 115}
    assert g.combined("WNT4") == {"Old Symbol": {111: {113}}, "VGNC Symbol": {111: {115}}}
    assert g.active_ranges("ENSSSCG1.1") == [[113, INF]]
    assert g.active_ranges("WNT4") == [[113, 113], [115, 115]]
    assert g.out_degree["ENSSSCG1"] == 1 and g.number_of_nodes() == len(g) == len(list(g.nodes))
    assert g.get_next_edge_releases("ENSSSCG1.1", reverse=False) == [INF]
    assert g.get_next_edge_releases("ENSSSCG1.1", reverse=True) == [113]
    for removed in ("combined_edges", "combined_edges_genes", "get_active_ranges_of_id", "node_trios",
                    "lower_chars_graph", "node_name_alternatives"):  # fmt: skip
        assert not hasattr(g, removed), removed


def test_compact_ranges_leaves_its_argument_alone():
    """``compact_ranges`` returns new ranges and does not change the list it is given (graph-16)."""
    ranges = [[1, 2], [3, 4], [8, 9]]

    assert TheGraph.compact_ranges(ranges) == [[1, 4], [8, 9]]
    assert ranges == [[1, 2], [3, 4], [8, 9]]


# --------------------------------------------------------------------------- the file (graph-5, graph-12, access-17)


def test_graph_file_is_self_describing_and_loads_without_network(monkeypatch, tmp_path):
    """The file carries its manifest and is read back identically with no manager and no connection."""
    world = symbols_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 111))
    built = maker.get_graph(form_list=["gene"])
    path = maker.create_file_name(form_list=["gene"])
    assert os.path.basename(path) == "graph_sus_scrofa_assembly111_min113_max115_narrow_formsgene.npz"

    def no_network(*args, **kwargs):
        raise AssertionError("loading a graph must not open a connection")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(DatabaseManager, "__init__", no_network)
    loaded = TheGraph.load(path)
    by_request = GraphMaker.load_graph(str(tmp_path), "sus_scrofa", 115, ignore_before=113, form_list=["gene"])
    assert by_request.manifest == loaded.manifest and by_request.graph["genome_assembly"] == 111
    os.link(path, GraphMaker.graph_file_path(str(tmp_path), "sus_scrofa", 102, 113, 115, form_list=["gene"]))
    with pytest.raises(GraphMismatchError, match="genome_assembly"):
        GraphMaker.load_graph(
            str(tmp_path), "sus_scrofa", 115, genome_assembly=102, ignore_before=113, form_list=["gene"]
        )
    with pytest.raises(FileNotFoundError):
        GraphMaker.load_graph(str(tmp_path), "sus_scrofa", 114, ignore_before=113, form_list=["gene"])

    manifest = loaded.manifest
    for field in ("format_version", "package_version", "organism", "genome_assembly", "ignore_before", "ignore_after",
                  "snapshot_release", "forms", "externals_fingerprint", "organism_table_fingerprint", "build_date",
                  "releases_per_assembly", "tables", "history", "content_digest"):  # fmt: skip
        assert field in manifest, field
    assert (manifest["organism"], manifest["genome_assembly"], manifest["snapshot_release"]) == ("sus_scrofa", 111, 115)
    assert manifest["releases_per_assembly"] == {"111": [113, 114, 115]}
    assert loaded.is_frozen and list(loaded.nodes) == list(built.nodes)
    for node in built.nodes:
        assert loaded.xref_edges(node) == built.xref_edges(node)
        assert repr(loaded.history_edges(node)) == repr(built.history_edges(node))
        assert loaded.nodes[node] == built.nodes[node]
    assert loaded.graph["confident_for_release"] == [113, 114, 115]
    assert loaded.hyperconnective_nodes == built.hyperconnective_nodes == {}
    assert loaded.shared_external_ids == built.shared_external_ids == {}
    assert loaded.available_external_databases == built.available_external_databases
    assert loaded.external_database_connection_form == dict.fromkeys(
        ["Old Symbol", "Synonym", "UniProtKB Gene Name", "VGNC Symbol"], "gene"
    )
    assert loaded.resolve_name("sla_1") == ("SLA-1", "separator")
    assert loaded.spelling("STAT2", "UniProtKB Gene Name") == "stat2"


def test_graphs_of_different_primary_assemblies_are_different_files(monkeypatch, tmp_path):
    """The file name carries the primary assembly; a file of another graph is never returned (graph-5)."""
    grch37 = lifetimes({"ENSG1": [(100, 101, 1)]})
    grch38 = lifetimes({"ENSG1": [(100, 100, 1), (101, 101, 2)]})
    world = World("homo_sapiens", {37: grch37, 38: grch38})
    world.install(monkeypatch)
    maker38, maker37 = GraphMaker(world.manager(tmp_path, 38)), GraphMaker(world.manager(tmp_path, 37))

    g38 = maker38.get_graph(form_list=["gene"])
    g37 = maker37.get_graph(form_list=["gene"])
    assert maker38.create_file_name(form_list=["gene"]) != maker37.create_file_name(form_list=["gene"])
    assert g38.graph["genome_assembly"] == 38 and "ENSG1.2" in g38
    assert g37.graph["genome_assembly"] == 37 and "ENSG1.2" not in g37
    assert maker37.get_graph(form_list=["gene"]).graph["genome_assembly"] == 37  # loaded from its own file
    found = GraphMaker.list_graph_files(str(tmp_path), "homo_sapiens")
    assert [(f["genome_assembly"], f["ignore_before"], f["ignore_after"], f["forms"]) for f in found] == [
        (37, 100, 101, ["gene"]),
        (38, 100, 101, ["gene"]),
    ]

    # the file of one graph under the name of the other is refused, with the difference named
    os.replace(maker38.create_file_name(form_list=["gene"]), maker37.create_file_name(form_list=["gene"]))
    with pytest.raises(GraphMismatchError, match="genome_assembly: file has 38, asked for 37"):
        maker37.get_graph(form_list=["gene"])


def test_graph_built_for_another_selection_of_external_databases_is_refused(monkeypatch, tmp_path):
    """A changed externals YAML makes the stored graph another graph (access-17, docs-1)."""
    import yaml

    world = symbols_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 111))
    maker.get_graph(form_list=["gene"])

    selection = {
        "sus_scrofa": {
            "gene": {
                "VGNC Symbol": {
                    "Assembly": {"111": {"Ensembl release": "113,114,115", "Include": True}},
                    "Database Index": 0,
                    "Potential Synonymous": "",
                }
            }
        }
    }
    with open(os.path.join(str(tmp_path), "sus_scrofa_externals_modified.yml"), "w") as handle:
        yaml.safe_dump(selection, handle)
    with pytest.raises(GraphMismatchError, match="externals_fingerprint"):
        GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"])
    rebuilt = GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"], create_even_if_exist=True)
    assert rebuilt.available_external_databases == {"VGNC Symbol"} or rebuilt.is_frozen
    assert GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"]).is_frozen  # the file was replaced


def pig_selection(repository, entries):
    """Write an externals YAML for pig genes: ``{database: {assembly: (releases, include)}}``."""
    import yaml

    selection = {
        "sus_scrofa": {
            "gene": {
                database: {
                    "Assembly": {
                        str(assembly): {"Ensembl release": ",".join(map(str, releases)), "Include": include}
                        for assembly, (releases, include) in per_assembly.items()
                    },
                    "Database Index": index,
                    "Potential Synonymous": "",
                }
                for index, (database, per_assembly) in enumerate(entries.items())
            }
        }
    }
    os.makedirs(repository, exist_ok=True)
    with open(os.path.join(str(repository), "sus_scrofa_externals_modified.yml"), "w") as handle:
        yaml.safe_dump(selection, handle)


def test_selection_entries_that_never_produce_an_edge_are_not_part_of_the_identity(monkeypatch, tmp_path):
    """Decision 24: two YAML files that differ only in entries without effect select the same graph."""
    tidy = {
        "VGNC Symbol": {111: ([113, 114, 115], True)},
        "UniProtKB Gene Name": {111: ([113, 114, 115], True)},
    }
    untidy = {
        # Sscrofa11.1 at a release it does not have (it begins at 90), and at one outside the window
        "VGNC Symbol": {111: ([80, 113, 114, 115, 116], True), 9: ([113, 114], True)},
        # Sscrofa10.2 at releases of Sscrofa11.1 (the 71 entries that the shipped pig YAML once had)
        "UniProtKB Gene Name": {111: ([113, 114, 115], True), 102: ([113, 114, 115], True)},
        "Old Symbol": {111: ([113, 114, 115], False)},  # not included
    }
    world = symbols_world()
    world.install(monkeypatch)
    first = DB.first_release("sus_scrofa")  # the window begins before the releases that Sscrofa11.1 has

    def graph(folder):
        return GraphMaker(world.manager(tmp_path / folder, 111, first=first)).get_graph(form_list=["gene"])

    pig_selection(tmp_path / "a", tidy)
    pig_selection(tmp_path / "b", untidy)
    one, two = graph("a"), graph("b")
    assert one.manifest["externals_fingerprint"] == two.manifest["externals_fingerprint"]
    assert one.manifest["content_digest"] == two.manifest["content_digest"]

    # each file is the graph of the other selection too
    pig_selection(tmp_path / "a", untidy)
    pig_selection(tmp_path / "b", tidy)
    assert list(graph("a").nodes) == list(one.nodes) and list(graph("b").nodes) == list(two.nodes)
    loaded = GraphMaker.load_graph(str(tmp_path / "a"), "sus_scrofa", 115, form_list=["gene"])
    assert loaded.manifest["build_date"] == one.manifest["build_date"]


@pytest.mark.parametrize(
    "changed",
    [
        {"VGNC Symbol": {111: ([114, 115], True)}},  # one release less, of the assembly and of the window
        {"VGNC Symbol": {111: ([113, 114, 115], True), 102: ([88, 89], True)}},  # an assembly at releases it has
        {"VGNC Symbol": {111: ([113, 114, 115], True)}, "Old Symbol": {111: ([113], True)}},  # one more database
    ],
)
def test_selection_difference_that_takes_effect_is_refused(monkeypatch, tmp_path, changed):
    """Decision 24: a release, an assembly or a database that the selection gains or loses is another graph."""
    world = symbols_world()
    world.install(monkeypatch)
    first = DB.first_release("sus_scrofa")
    pig_selection(tmp_path, {"VGNC Symbol": {111: ([113, 114, 115], True)}})
    GraphMaker(world.manager(tmp_path, 111, first=first)).get_graph(form_list=["gene"])

    pig_selection(tmp_path, changed)
    with pytest.raises(GraphMismatchError, match="externals_fingerprint"):
        GraphMaker(world.manager(tmp_path, 111, first=first)).get_graph(form_list=["gene"])
    with pytest.raises(GraphMismatchError, match="externals_fingerprint"):
        GraphMaker.load_graph(str(tmp_path), "sus_scrofa", 115, form_list=["gene"])


def test_graph_built_under_another_organism_table_is_refused(monkeypatch, tmp_path):
    """A graph built when an assembly had other releases is not loaded as the current one (accuracy-4)."""
    world = handoff_world()
    world.install(monkeypatch)
    GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"])

    table = {k: dict(v) for k, v in DB.assembly_mysqlport_priority["sus_scrofa"].items()}
    table[102]["ReleaseRange"] = [67, 99]
    monkeypatch.setitem(DB.assembly_mysqlport_priority, "sus_scrofa", table)
    with pytest.raises(GraphMismatchError, match="organism_table_fingerprint"):
        GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"])


def test_rebuilding_replaces_the_file(monkeypatch, tmp_path):
    """``create_even_if_exist`` leaves the new graph on disk, not the old one (graph-12)."""
    world = symbols_world()
    world.install(monkeypatch)
    first = GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"])
    assert "NEW" not in first

    world.externals[115].append(("ENSSSCG3.1", "NEW", "VGNC Symbol", 111))
    assert "NEW" not in GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"])  # the stored graph
    assert "NEW" in GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"], create_even_if_exist=True)
    assert "NEW" in GraphMaker(world.manager(tmp_path, 111)).get_graph(form_list=["gene"])


def test_interrupted_write_leaves_the_previous_file(monkeypatch, tmp_path):
    """A graph is written to a temporary file and renamed: no partial file under the final name (graph-12)."""
    world = symbols_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 111))
    graph = maker.get_graph(form_list=["gene"])
    path = maker.create_file_name(form_list=["gene"])
    with open(path, "rb") as handle:
        before = handle.read()

    def dies(handle, **arrays):
        handle.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez", dies)
    with pytest.raises(KeyboardInterrupt):
        graph.save(path)
    with open(path, "rb") as handle:
        assert handle.read() == before
    assert sorted(os.listdir(tmp_path)) == sorted(n for n in os.listdir(tmp_path) if not n.endswith(".tmp"))

    fresh = os.path.join(str(tmp_path), "fresh.npz")
    with pytest.raises(KeyboardInterrupt):
        graph.save(fresh)
    assert not os.path.exists(fresh)


def test_file_is_the_same_before_and_after_using_the_graph(monkeypatch, tmp_path):
    """Nothing computed on demand ends up in the file, and two writes of one graph are byte-identical."""
    world = symbols_world()
    world.install(monkeypatch)
    graph = GraphMaker(world.manager(tmp_path, 111)).construct_graph(form_list=["gene"])
    first, second = os.path.join(str(tmp_path), "a.npz"), os.path.join(str(tmp_path), "b.npz")

    graph.save(first)
    graph.calculate_caches(for_test=True)
    for node in graph.nodes:
        _ = (graph.xref_edges(node), graph.history_edges(node), graph.nodes[node], list(graph.neighbors(node)))
    graph.save(second)
    with open(first, "rb") as a, open(second, "rb") as b:
        assert a.read() == b.read()


def test_two_builds_of_the_same_tables_have_the_same_content(monkeypatch, tmp_path):
    """The build is deterministic: same tables, same content digest, nodes in sorted order."""
    world = symbols_world()
    one = build(world, monkeypatch, tmp_path / "a", 111)
    two = build(world, monkeypatch, tmp_path / "b", 111)

    assert one.manifest["content_digest"] == two.manifest["content_digest"]
    assert list(one.nodes) == sorted(one.nodes) == list(two.nodes)


def test_manifest_records_the_tables_the_graph_was_built_from(monkeypatch, tmp_path):
    """The manifest holds one fingerprint per assembly over the stored tables; other tables, another fingerprint."""
    import hashlib

    one = build(handoff_world(), monkeypatch, tmp_path / "a", 111)
    again = build(handoff_world(), monkeypatch, tmp_path / "b", 111)
    world = handoff_world()
    world.assemblies[102][88]["ENSSSCG9"] = 1  # one more gene on the earlier assembly
    other = build(world, monkeypatch, tmp_path / "c", 111)

    recorded = one.manifest["source_tables"]
    assert sorted(recorded) == ["102", "111"]
    for assembly in (102, 111):
        digests = (
            world.manager(tmp_path / "a", 111)
            .change_assembly(assembly, last_possible_ensembl_release=True)
            .table_digests()
        )
        assert digests, "the build stores its tables"
        folded = hashlib.sha256()
        for name in sorted(digests):
            folded.update(f"{name}\t{digests[name]}\n".encode())
        assert recorded[str(assembly)] == {"tables": len(digests), "fingerprint": folded.hexdigest()}
    assert again.manifest["source_tables"] == recorded
    assert other.manifest["source_tables"]["102"]["fingerprint"] != recorded["102"]["fingerprint"]
    assert other.manifest["source_tables"]["111"] == recorded["111"]


def test_external_table_naming_an_assembly_of_another_organism_stops_the_build(monkeypatch, tmp_path):
    """Assembly keys are checked against the organism of the graph, not against every organism's keys."""
    world = symbols_world()
    world.externals[115].append(("ENSSSCG3.1", "TP53", "HGNC Symbol", 38))  # 38 is a key of human and mouse only
    with pytest.raises(ValueError, match=r"Unknown assembly 38 in the external table of release 115.*\[9, 102, 111\]"):
        build(world, monkeypatch, tmp_path, 111)


def test_unreadable_graph_file_names_the_file_and_the_way_out(tmp_path):
    """A broken or missing file is reported with its path and how to rebuild (graph-12)."""
    path = tmp_path / "graph_broken.npz"
    path.write_bytes(b"not a graph")

    with pytest.raises(GraphMismatchError, match=r"graph_broken\.npz.*To build the graph again"):
        TheGraph.load(str(path))
    with pytest.raises(FileNotFoundError, match="absent.npz"):
        TheGraph.load(str(tmp_path / "absent.npz"))


def test_graph_file_with_a_missing_part_is_refused_with_the_way_out(monkeypatch, tmp_path):
    """A file that lacks a part of the format is an explicit error, not a ``KeyError`` from inside the loader."""
    world = symbols_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 111))
    maker.get_graph(form_list=["gene"])
    path = maker.create_file_name(form_list=["gene"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "in_order"}
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)

    with pytest.raises(GraphMismatchError, match="does not hold what a graph file.*To build the graph again"):
        TheGraph.load(path)


def test_graph_of_another_format_version_is_refused(monkeypatch, tmp_path):
    """A file of another graph format version is never loaded."""
    import idtrack._the_graph as module

    world = symbols_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 111))
    maker.get_graph(form_list=["gene"])

    monkeypatch.setattr(module, "GRAPH_FORMAT_VERSION", module.GRAPH_FORMAT_VERSION + 1)
    with pytest.raises(GraphMismatchError, match="format version"):
        TheGraph.load(maker.create_file_name(form_list=["gene"]))


def test_mismatch_message_names_a_way_out_that_exists(monkeypatch, tmp_path):
    """Every ``GraphMismatchError`` says what a user can do, in terms of calls and arguments that exist."""
    import inspect
    import re

    import idtrack
    import idtrack._the_graph as module
    from idtrack import API

    world = symbols_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 111, first=DB.first_release("sus_scrofa")))
    maker.get_graph()
    path = str(tmp_path / "graph_sus_scrofa_assembly111_min56_max115_narrow.npz")
    assert os.path.isfile(path)

    messages = []
    # the four ways a file can be refused: not a graph file, another graph than asked for, a part missing, another format
    broken = tmp_path / "graph_broken.npz"
    broken.write_bytes(b"not a graph")
    with pytest.raises(GraphMismatchError) as caught:
        TheGraph.load(str(broken))
    messages.append(str(caught.value))
    with pytest.raises(GraphMismatchError) as caught:
        TheGraph.load(path, expect={"organism": "homo_sapiens"})
    messages.append(str(caught.value))
    partial = str(tmp_path / "graph_partial.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "in_order"}
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(GraphMismatchError) as caught:
        TheGraph.load(partial)
    messages.append(str(caught.value))
    with monkeypatch.context() as patched:
        patched.setattr(module, "GRAPH_FORMAT_VERSION", module.GRAPH_FORMAT_VERSION + 1)
        with pytest.raises(GraphMismatchError) as caught:
            TheGraph.load(path)
        messages.append(str(caught.value))

    for message in messages:
        # every call the message names exists, with every argument it names
        named = re.findall(r"`(API|GraphMaker)\.(\w+)(?:\((\w+)=[^)]*\))?`", message)
        assert {(owner, method) for owner, method, _ in named} == {("API", "build_graph"), ("GraphMaker", "get_graph")}
        for owner, method, argument in named:
            call = getattr(getattr(idtrack, owner), method)
            if argument:
                assert argument in inspect.signature(call).parameters, (owner, method, argument)
        # and no argument is offered that the API's call does not have
        assert "create_even_if_exist" not in inspect.signature(API.build_graph).parameters
        assert not re.search(r"`API\.build_graph\(", message) and "delete or move the file" in message

    # the way out through the API works as the message says: with the file moved away the graph is built again
    os.replace(path, path + ".aside")
    os.link(path + ".aside", str(tmp_path / "graph_sus_scrofa_assembly111_min56_max114_narrow.npz"))
    api = API(str(tmp_path))
    with pytest.raises(GraphMismatchError, match="ignore_after.*delete or move the file"):
        api.build_graph("sus_scrofa", 114)  # the file of another graph under this graph's name
    os.remove(tmp_path / "graph_sus_scrofa_assembly111_min56_max114_narrow.npz")
    api.build_graph("sus_scrofa", 114)
    assert api.track.graph.graph["ensembl_release"] == 114 and api.graph_releases() == [113, 114]
    # and through the builder: `create_even_if_exist=True` replaces a file that is not the graph asked for
    os.link(path + ".aside", path)
    os.remove(tmp_path / "graph_sus_scrofa_assembly111_min56_max114_narrow.npz")
    os.link(path, str(tmp_path / "graph_sus_scrofa_assembly111_min56_max114_narrow.npz"))
    older = GraphMaker(world.manager(tmp_path, 111, first=DB.first_release("sus_scrofa"), last=114))
    with pytest.raises(GraphMismatchError):
        older.get_graph()
    assert older.get_graph(create_even_if_exist=True).graph["ensembl_release"] == 114


def test_narrow_false_records_the_sessions_and_builds_with_two_sessions_of_one_release(monkeypatch, tmp_path):
    """Two sessions that name the same new release (human 113 to 114 and 114 to "114") do not stop the build."""
    genes = lifetimes({"G1": [(100, 100, 1)], "G2": [(101, 102, 1)], "G3": [(100, 101, 1)], "G4": [(102, 102, 1)]})
    world = World(
        "homo_sapiens",
        {38: genes},
        events=[(424, "G1", 1, "G2", 1, 0.9), (425, "G3", 1, "G4", 1, 0.9)],
        sessions=[(424, 100, 101), (425, 101, 101)],
    )
    g = build(world, monkeypatch, tmp_path, 38, narrow=False)

    sessions = {s["mapping_session_id"]: s for s in g.manifest["history"]["sessions"]}
    assert sessions[424]["used"] == [100, 101] and sessions[425]["used"] == [101, 102]
    assert sessions[425]["recorded"] == [101, 101] and sessions[425]["old_db_name"] == "homo_sapiens_core_101"
    assert g.active_ranges("G4.1") == [[102, INF]]
    assert g.graph["name"] == "homo_sapiens_102_gene"


def test_frozen_graph_is_a_networkx_graph_that_cannot_be_changed(monkeypatch, tmp_path):
    """A built graph reads like a networkx graph, refuses every change, and survives pickling."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert isinstance(g, nx.MultiDiGraph) and g.is_frozen
    for change in (
        lambda: g.add_node("X"),
        lambda: g.add_edge("STAT2", "ENSSSCG2.1"),
        lambda: g.remove_node("STAT2"),
        lambda: g.remove_edge("STAT2", "ENSSSCG1.1"),
        lambda: g.clear(),
    ):
        with pytest.raises(nx.NetworkXError, match="frozen"):
            change()
    copy = pickle.loads(pickle.dumps(g))  # noqa: S301
    assert copy.is_frozen and copy.xref_edges("STAT2") == g.xref_edges("STAT2")


def test_versions_of_a_stable_id_are_found_without_a_base_node(monkeypatch, tmp_path):
    """Transcripts and translations have no base node; their versions are found by name."""
    g = build(symbols_world(), monkeypatch, tmp_path, 111)

    assert g.versions_of("ENSSSCG1") == ("ENSSSCG1.1",)
    assert g.versions_of("ENSSSCG") == ()
    assert g.versions_of("NOPE") == ()

    by_hand = TheGraph(genome_assembly=38, confident_for_release=[100])
    for name in ("ENST2.10", "ENST2.9", "ENST20.1", "ENST2.Void"):
        by_hand.add_node(name, **{DB.node_type_str: DB.nts_ensembl["transcript"]})
    assert by_hand.versions_of("ENST2") == ("ENST2.9", "ENST2.10")


def test_three_forms_and_an_id_that_exists_on_a_parallel_assembly_only(monkeypatch, tmp_path):
    """The whole build with transcripts and translations, and a GRCh37 gene that GRCh38 never had."""
    grch37 = lifetimes({"ENSG1": [(100, 101, 1)], "ENSG7": [(100, 101, 1)]})
    grch38 = lifetimes({"ENSG1": [(100, 100, 1), (101, 101, 2)]})
    externals = {
        100: [
            ("ENSG1.1", "SYM", "HGNC Symbol", 38),
            ("ENSG1.1", "SYM", "HGNC Symbol", 37),
            ("ENSG7.1", "OLD", "HGNC Symbol", 37),
            ("ENST1.1", "NM_1", "RefSeq_mRNA", 38, "transcript"),
            ("ENSP1.1", "P1", "Uniprot/SWISSPROT", 38, "translation"),
        ],
        101: [
            ("ENSG1.2", "SYM", "HGNC Symbol", 38),
            ("ENSG1.1", "SYM", "HGNC Symbol", 37),
            ("ENSG7.1", "OLD", "HGNC Symbol", 37),
            ("ENST1.1", "NM_1", "RefSeq_mRNA", 38, "transcript"),
        ],
    }
    world = World("homo_sapiens", {37: grch37, 38: grch38}, externals=externals)
    os.makedirs(tmp_path, exist_ok=True)
    world.install(monkeypatch)
    g = GraphMaker(world.manager(tmp_path, 38)).construct_graph()

    assert g.available_forms == ["gene", "transcript", "translation"]
    assert g.node_type("ENST1.1") == "ensembl_transcript" and g.node_type("ENSP1.1") == "ensembl_translation"
    assert g.history_edges("ENST1.1") == () and "ENST1.Void" not in g  # only genes carry a history
    assert g.connection("ENST1.1", "ENSG1.1").as_dict == {"ensembl_transcript": {38: {100}}}
    assert g.connection("ENST1.1", "ENSG1.2").as_dict == {"ensembl_transcript": {38: {101}}}
    assert g.connection("ENSP1.1", "ENST1.1").as_dict == {"ensembl_translation": {38: {100, 101}}}
    assert g.trios("ENST1.1") == {
        ("assembly_38_ensembl_transcript", 38, 100),
        ("assembly_38_ensembl_transcript", 38, 101),
    }
    assert g.versions_of("ENST1") == ("ENST1.1",)

    # the gene that only GRCh37 has: a node of that assembly, with its transcript, translation and base id
    assert g.node_type("ENSG7.1") == "assembly_37_ensembl_gene"
    assert g.node_type("ENST7.1") == "assembly_37_ensembl_transcript"
    assert g.node_type("ENSP7.1") == "assembly_37_ensembl_translation"
    assert g.connection("ENST7.1", "ENSG7.1").as_dict == {"assembly_37_ensembl_transcript": {37: {100, 101}}}
    assert g.connection("ENSG7", "ENSG7.1").as_dict == {"base_ensembl_gene": {37: {100, 101}}}
    assert g.combined("ENSG7.1") == {"assembly_37_ensembl_gene": {37: {100, 101}}}
    assert g.active_ranges("ENSG7.1") == [[100, 101]] and g.history_edges("ENSG7.1") == ()
    assert g.connection("OLD", "ENSG7.1").as_dict == {"HGNC Symbol": {37: {100, 101}}}

    # the gene of both assemblies: one backbone node, referred to on both
    assert g.connection("ENSG1", "ENSG1.1").as_dict == {"base_ensembl_gene": {37: {100, 101}, 38: {100}}}
    assert g.active_ranges("ENSG1.1") == [[100, 100]]
    assert g.get_active_ranges_of_id_ensembl_all_inclusive("ENSG1.1") == [[100, 101]]
    assert g.backbone_assembly == {100: 38, 101: 38} and g.assembly_releases == {37: (100, 101), 38: (100, 101)}
    assert g.available_external_databases == {"HGNC Symbol", "RefSeq_mRNA", "Uniprot/SWISSPROT"}
    assert g.external_database_connection_form == {
        "HGNC Symbol": "gene",
        "RefSeq_mRNA": "transcript",
        "Uniprot/SWISSPROT": "translation",
    }
    assert g.genes_linked_at_once("NM_1") == 1 and g.genes_linked_at_once("SYM") == 1
    g.calculate_caches(for_test=True)
    table = g.available_releases_given_database_assembly
    assert table[("HGNC Symbol", 37)] == {100, 101} and table[("assembly_37_ensembl_gene", 37)] == {100, 101}
    assert table[("ensembl_gene", 38)] == table[("assembly_38_ensembl_gene", 38)] == {100, 101}


# ---------------------------------------------------------------- genes outside the reference sequence (decision 18)


def patches_world(externals=None):
    """A human-like organism: GRCh37 serves releases 74 and 75 and stays beside GRCh38, which serves 76 and 77.

    ``ENSG1`` is a gene of the reference sequence of both assemblies. ``ENSG2`` is a gene on a patch of GRCh38 and
    ``LRG_1`` an LRG record there. ``ENSG3`` lay on a haplotype of GRCh37 and lies on a chromosome of GRCh38.
    ``ENSG5`` is a gene on a patch of GRCh37 that GRCh38 does not have.
    """
    grch37 = lifetimes({"ENSG1": [(74, 77, 1)], "ENSG3": [(74, 77, 1)], "ENSG5": [(74, 77, 1)]})
    grch38 = lifetimes({"ENSG1": [(76, 77, 1)], "ENSG2": [(76, 77, 1)], "ENSG3": [(76, 77, 1)], "LRG_1": [(76, 77, 1)]})
    beside = {
        37: {release: ["ENSG3", "ENSG5"] for release in (74, 75, 76, 77)},
        38: {release: ["ENSG2", "LRG_1"] for release in (76, 77)},
    }
    return World("homo_sapiens", {37: grch37, 38: grch38}, externals=externals, non_reference=beside)


def test_genes_outside_the_reference_sequence_are_marked_for_the_assembly_that_serves_the_release(
    monkeypatch, tmp_path
):
    """A gene on a patch, a haplotype or an LRG record carries the releases at which it lies there (decision 18)."""
    world = patches_world(externals={77: [("ENSG2.1", "SYM", "HGNC Symbol", 38)]})
    g = build(world, monkeypatch, tmp_path, 38)

    assert g.backbone_assembly == {74: 37, 75: 37, 76: 38, 77: 38}
    assert g.non_reference_releases("ENSG2.1") == {76, 77} and g.non_reference_releases("LRG_1.1") == {76, 77}
    assert isinstance(g.non_reference_releases("ENSG2.1"), ReleaseSet)
    assert g.non_reference_releases("ENSG3.1") == {74, 75}  # a haplotype of GRCh37, a chromosome of GRCh38
    # GRCh37 keeps ENSG5 on its patch at 76 and 77, but GRCh38 serves those releases
    assert g.active_ranges("ENSG5.1") == [[74, 75]] and g.non_reference_releases("ENSG5.1") == {74, 75}
    assert g.non_reference_releases("ENSG1.1") == set()
    assert g.is_non_reference("ENSG3.1", 75) and not g.is_non_reference("ENSG3.1", 76)
    assert not g.is_non_reference("ENSG1.1", 77) and not g.is_non_reference("ENSG2.1", 200)
    for other in ("ENSG2", "ENSG2.Void", "ENSG5.Retired", "SYM"):  # only genes with a version are marked
        assert g.non_reference_releases(other) == set(), other
    with pytest.raises(KeyError):
        g.non_reference_releases("ENSG9.1")
    assert g.manifest["counts"]["non_reference_genes"] == 4
    assert g.manifest["tables"]["generegion"] == {"tables": 4, "rows": 3 + 3 + 4 + 4}

    # the mark is part of what a graph holds: the same world without it is another graph
    plain = patches_world(externals=world.externals)
    plain.non_reference = {}
    unmarked = build(plain, monkeypatch, tmp_path / "plain", 38)
    assert unmarked.manifest["counts"]["non_reference_genes"] == 0
    assert unmarked.non_reference_releases("ENSG2.1") == set()
    assert unmarked.manifest["content_digest"] != g.manifest["content_digest"]
    assert list(unmarked.nodes) == list(g.nodes)


def test_marks_of_genes_outside_the_reference_sequence_are_stored_with_the_graph(monkeypatch, tmp_path):
    """The graph file holds the marks; a file without them is not a graph file of this format."""
    world = patches_world()
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 38))
    built = maker.get_graph(form_list=["gene"])
    path = maker.create_file_name(form_list=["gene"])

    loaded = TheGraph.load(path)
    marks = {node: loaded.non_reference_releases(node) for node in loaded.nodes if loaded.non_reference_releases(node)}
    assert marks == {"ENSG2.1": {76, 77}, "ENSG3.1": {74, 75}, "ENSG5.1": {74, 75}, "LRG_1.1": {76, 77}}
    assert all(loaded.non_reference_releases(node) is built.non_reference_releases(node) for node in built.nodes)
    copy = pickle.loads(pickle.dumps(loaded))  # noqa: S301
    assert copy.non_reference_releases("ENSG3.1") == {74, 75}

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "nonref_mask"}
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(GraphMismatchError, match="does not hold what a graph file.*To build the graph again"):
        TheGraph.load(path)


def test_graph_made_in_memory_has_no_gene_outside_the_reference_sequence():
    """The mark comes from Ensembl's tables through the builder; a graph made by hand has none."""
    g = TheGraph(genome_assembly=38, organism="homo_sapiens", confident_for_release=[100, 101])
    g.add_node("ENSG1.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
    g.add_node("ENSG1.Void", **{DB.node_type_str: DB.nts_ensembl["gene"]})
    g.add_edge("ENSG1.Void", "ENSG1.1", 0, weight=float("nan"), old_release=100, new_release=100)
    g.add_edge("ENSG1.1", "ENSG1.1", 0, weight=1.0, old_release=100, new_release=INF)

    assert g.non_reference_releases("ENSG1.1") == set() and not g.is_non_reference("ENSG1.1", 100)


# ---------------------------------------------------------------- the engine on a graph of the real builder


def test_final_conversion_returns_the_spelling_of_the_asked_database(monkeypatch, tmp_path):
    """One node per identifier; a conversion returns it as the database asked for writes it (graph-2)."""
    track = Track.from_graph(build(symbols_world(), monkeypatch, tmp_path, 111))

    def targets(result):
        return [t for gene in result for t in result[gene]["final_conversion"]["final_elements"]]

    assert targets(track.convert("ENSSSCG1.1", final_database="VGNC Symbol")) == ["STAT2"]
    assert targets(track.convert("ENSSSCG1.1", final_database="UniProtKB Gene Name")) == ["stat2"]
    assert track.gene_to_database("ENSSSCG1.1", "UniProtKB Gene Name") == (["stat2"], 0)
    assert track.gene_to_database("ENSSSCG1.1", "VGNC Symbol", 114) == (["STAT2"], 0)
    # either spelling is the same query, and it is preferred as the target whichever way it was written
    assert list(track.convert("stat2")) == list(track.convert("STAT2")) == ["ENSSSCG1.1"]
    assert track.convert("stat2")["ENSSSCG1.1"]["from_id"] == "STAT2"


def test_engine_reads_ids_without_version_of_every_form_from_a_built_graph(monkeypatch, tmp_path):
    """Gene ids without version have a base node, transcript and translation ids are found by their versions."""
    grch38 = lifetimes({"ENSG1": [(100, 100, 1), (101, 101, 2)], "ENSG2": [(100, 101, 1)]})
    world = World("homo_sapiens", {38: grch38}, externals={101: [("ENSG1.2", "SYM", "HGNC Symbol", 38)]})
    os.makedirs(tmp_path, exist_ok=True)
    world.install(monkeypatch)
    g = GraphMaker(world.manager(tmp_path, 38)).construct_graph()
    track = Track.from_graph(g)

    assert g.node_type("ENSG1") == "base_ensembl_gene" and "ENST1" not in g and "ENSP1" not in g
    for query in ("ENSG1", "ENST1", "ENSP1", "ENST1.1", "ENSP1.1"):
        assert list(track.convert(query)) == ["ENSG1.2"], query
        assert list(track.convert(query, to_release=100)) == ["ENSG1.1"], query
    assert track.convert("ENST1")["ENSG1.2"]["from_id"] == "ENST1.1"
    # what an Ensembl id is read through is its own relation to the gene; a gene id with version is the gene
    read_through = {query: track.convert(query)["ENSG1.2"]["from_databases"] for query in ("ENSG1", "ENST1", "ENSP1")}
    assert read_through == {
        "ENSG1": ["base_ensembl_gene"],
        "ENST1": ["ensembl_transcript"],
        "ENSP1": ["ensembl_translation"],
    }
    assert track.convert("ENSG1.2")["ENSG1.2"]["from_databases"] == []
    assert {track.convert(query)["ENSG1.2"]["entry_support"] for query in ("ENSG1", "ENST1", "ENSP1", "ENSG1.2")} == {1}
    assert track.convert("ENST2", final_database="HGNC Symbol")["ENSG2.1"]["final_conversion"]["no_target"] is True
    assert track.convert("ENST3") is None


def copies_world():
    """The organism of ``patches_world`` with names: what a gene of the reference sequence shares with its copies.

    ``SYM`` is the symbol of ENSG1, of its copy on a patch (ENSG2) and of its LRG record; ``ONLYPATCH`` is a symbol
    of the patch gene alone; the NCBI id ``TWO`` is on ENSG1, on ENSG3 and on the patch gene, in every release.
    """
    externals = {}
    for release in (74, 75):
        externals[release] = [("ENSG1.1", "TWO", "EntrezGene", 37), ("ENSG3.1", "TWO", "EntrezGene", 37)]
    for release in (76, 77):
        externals[release] = [
            ("ENSG1.1", "SYM", "HGNC Symbol", 38),
            ("ENSG2.1", "SYM", "HGNC Symbol", 38),
            ("LRG_1.1", "SYM", "HGNC Symbol", 38),
            ("ENSG2.1", "ONLYPATCH", "HGNC Symbol", 38),
            ("ENSG2.1", "COPIES", "HGNC Symbol", 38),
            ("LRG_1.1", "COPIES", "HGNC Symbol", 38),
            ("ENSG1.1", "TWO", "EntrezGene", 38),
            ("ENSG2.1", "TWO", "EntrezGene", 38),
            ("ENSG3.1", "TWO", "EntrezGene", 38),
        ]
    return patches_world(externals=externals)


def test_copies_on_patches_leave_an_answer_that_holds_a_gene_of_the_reference_sequence(monkeypatch, tmp_path):
    """Decision 18: a name of a gene and of its copy on a patch is the gene; the copy is reported apart."""
    track = Track.from_graph(build(copies_world(), monkeypatch, tmp_path, 38))

    for best in (False, True):
        result = track.convert("SYM", prioritize_to_one_filter=best)
        assert list(result) == ["ENSG1.1"]
        # decision 31: the LRG record is a copy like the gene on the patch, by the graph's mark
        assert result["ENSG1.1"]["alternative_locus_genes"] == ["ENSG2.1", "LRG_1.1"]
        assert result["ENSG1.1"]["tie"] is False
    with pytest.raises(TypeError, match="deprioritize_lrg_genes"):  # no rule of its own, so no switch for it
        track.convert("SYM", deprioritize_lrg_genes=False)

    # copies are the answer when nothing of the reference sequence is reached
    for query in ("ONLYPATCH", "ENSG2", "ENSG2.1"):
        result = track.convert(query, prioritize_to_one_filter=True)
        assert list(result) == ["ENSG2.1"] and result["ENSG2.1"]["alternative_locus_genes"] == [], query
    for query in ("LRG_1.1", "LRG_1", "lrg_1"):
        assert list(track.convert(query)) == ["LRG_1.1"], query
    both = track.convert("COPIES")  # a name of the gene on the patch and of the LRG record only
    assert list(both) == ["ENSG2.1", "LRG_1.1"] and both["LRG_1.1"]["alternative_locus_genes"] == []
    # the mark decides, not the id: an id that looks like an LRG record and is not marked would stay in the answer
    assert not track.graph.is_non_reference("ENSG1.1", 77) and track.graph.is_non_reference("LRG_1.1", 77)


def own_copy_world():
    """A gene that lay on a patch of GRCh37 and is a gene of the reference sequence of GRCh38 under the same id.

    Releases 74 and 75 are served by GRCh37, 76 and 77 by GRCh38. ``ENSG9`` lies on a patch at 74 and 75; so does
    ``ENSG6``. ``ENSG7`` is a gene of the reference sequence there. At release 76 Ensembl merges ``ENSG6`` and
    ``ENSG7`` into ``ENSG9``, which is a gene of the reference sequence from then on (version 2). ``SYM9`` is the
    symbol of ``ENSG9`` at release 77.
    """
    grch37 = lifetimes({"ENSG1": [(74, 77, 1)], "ENSG6": [(74, 75, 1)], "ENSG7": [(74, 75, 1)], "ENSG9": [(74, 75, 1)]})
    grch38 = lifetimes({"ENSG1": [(76, 77, 1)], "ENSG9": [(76, 77, 2)]})
    events = [
        (1, "ENSG9", 1, "ENSG9", 2, 0.8),
        (1, "ENSG7", 1, "ENSG9", 2, 0.9),
        (1, "ENSG6", 1, "ENSG9", 2, 0.7),
    ]
    return World(
        "homo_sapiens",
        {37: grch37, 38: grch38},
        events=events,
        sessions=[(1, 75, 76)],
        externals={77: [("ENSG9.2", "SYM9", "HGNC Symbol", 38)]},
        non_reference={37: {release: ["ENSG6", "ENSG9"] for release in (74, 75)}},
    )


def test_gene_id_that_exists_at_the_target_release_is_its_own_answer(monkeypatch, tmp_path):
    """Decision 35: the rule for copies never removes the queried id's own version at the target release."""
    g = build(own_copy_world(), monkeypatch, tmp_path, 38)
    track = Track.from_graph(g)
    assert g.is_non_reference("ENSG9.1", 75) and g.is_non_reference("ENSG6.1", 75)
    assert not g.is_non_reference("ENSG7.1", 75) and not g.is_non_reference("ENSG9.2", 77)

    for query in ("ENSG9", "ENSG9.2"):
        # every answer: the id's own version first, beside the gene of the reference sequence that its history
        # reaches; the other copy is reported apart
        every = track.convert(query, from_release=77, to_release=75)
        assert list(every) == ["ENSG9.1", "ENSG7.1"], query
        assert every["ENSG9.1"]["alternative_locus_genes"] == ["ENSG6.1"]
        # the best answer: the id itself, although another gene has the higher mapping score
        best = track.convert(query, from_release=77, to_release=75, prioritize_to_one_filter=True)
        assert list(best) == ["ENSG9.1"] and best["ENSG9.1"]["tie"] is False, query
        assert best["ENSG9.1"]["alternative_locus_genes"] == ["ENSG6.1"]
        assert every["ENSG7.1"]["score"] > every["ENSG9.1"]["score"]
    # forward the id is itself as well (two other genes merge into it)
    assert list(track.convert("ENSG9", from_release=75, to_release=77, prioritize_to_one_filter=True)) == ["ENSG9.2"]
    assert list(track.convert("ENSG9.1", to_release=77)) == ["ENSG9.2"]
    # the rule is about the queried gene id: another copy that leads here is still reported apart, and a name is
    # answered by the gene of the reference sequence, with the copies apart (decision 18)
    other = track.convert("ENSG6", from_release=75, to_release=75)
    assert list(other) == ["ENSG6.1"]
    named = track.convert("SYM9", to_release=75)
    assert list(named) == ["ENSG7.1"] and named["ENSG7.1"]["alternative_locus_genes"] == ["ENSG6.1", "ENSG9.1"]


def test_api_answers_an_id_that_exists_at_the_target_release_with_itself(monkeypatch, tmp_path):
    """Decision 35 through the API, for both strategies."""
    from idtrack import API

    world = own_copy_world()
    world.install(monkeypatch)
    GraphMaker(world.manager(tmp_path, 38, first=DB.first_release("homo_sapiens"))).get_graph()
    api = API(str(tmp_path))
    api.build_graph("homo_sapiens", 77)

    best = api.convert_identifier("ENSG9", from_release=77, to_release=75)
    assert best["target_id"] == ["ENSG9.1"] and best["tie"] is False and best["ambiguous"] is False
    assert best["alternative_locus_genes"] == ["ENSG6.1"]
    every = api.convert_identifier("ENSG9", from_release=77, to_release=75, strategy="all")
    assert every["target_id"] == ["ENSG9.1", "ENSG7.1"] and every["alternative_locus_genes"] == ["ENSG6.1"]


def test_genes_of_the_reference_sequence_that_carry_an_id_equally_are_a_tie(monkeypatch, tmp_path):
    """Decision 18: the tie is among the genes of the reference sequence, at the target release."""
    track = Track.from_graph(build(copies_world(), monkeypatch, tmp_path, 38))

    best = track.convert("TWO", prioritize_to_one_filter=True)
    assert list(best) == ["ENSG1.1", "ENSG3.1"] and best["ENSG1.1"]["tie"] is True
    assert best["ENSG3.1"]["alternative_locus_genes"] == ["ENSG2.1"]  # the LRG record does not carry `TWO`
    assert list(track.convert("TWO")) == ["ENSG1.1", "ENSG3.1"]  # every answer: the same genes, the copy apart

    # at release 75 GRCh37 serves, on which ENSG3 lies on a haplotype: there it is the copy
    earlier = track.convert("TWO", to_release=75, prioritize_to_one_filter=True)
    assert list(earlier) == ["ENSG1.1"] and earlier["ENSG1.1"]["tie"] is False
    assert earlier["ENSG1.1"]["alternative_locus_genes"] == ["ENSG3.1"]

    # the copy leaves before the final conversion: its own symbols are not an answer for the gene
    named = track.convert("TWO", final_database="HGNC Symbol")
    assert list(named) == ["ENSG1.1"] and list(named["ENSG1.1"]["final_conversion"]["final_elements"]) == ["SYM"]
    assert named["ENSG1.1"]["final_conversion"]["genes_without_target"] == ["ENSG3.1"]
    assert named["ENSG1.1"]["alternative_locus_genes"] == ["ENSG2.1"]


def test_api_reports_the_copies_on_alternative_loci_apart(monkeypatch, tmp_path):
    """The API hands the engine's list on: the copies are named, and they are not targets."""
    import json

    from idtrack import API

    world = copies_world()
    world.install(monkeypatch)
    GraphMaker(world.manager(tmp_path, 38, first=DB.first_release("homo_sapiens"))).get_graph()
    api = API(str(tmp_path))
    api.build_graph("homo_sapiens", 77)

    for strategy in ("best", "all"):
        result = api.convert_identifier("sym", strategy=strategy)
        assert result["target_id"] == ["ENSG1.1"] and result["alternative_locus_genes"] == ["ENSG2.1", "LRG_1.1"]
        assert result["ambiguous"] is False and result["tie"] is False
        assert json.loads(json.dumps(result, allow_nan=False)) == result
    tied = api.convert_identifier("TWO")
    assert tied["target_id"] == ["ENSG1.1", "ENSG3.1"] and tied["tie"] is True
    assert tied["alternative_locus_genes"] == ["ENSG2.1"]
    assert api.convert_identifier("ONLYPATCH")["target_id"] == ["ENSG2.1"]
    assert api.convert_identifier("ONLYPATCH")["alternative_locus_genes"] == []
    assert api.convert_identifier("ENST2")["target_id"] == ["ENSG2.1"]  # a transcript of the patch gene
    assert api.convert_identifier("nope")["alternative_locus_genes"] == []
    batch = api.convert_identifier_multiple(["SYM", "ENSG2"], verbose=False)
    assert [item["alternative_locus_genes"] for item in batch] == [["ENSG2.1", "LRG_1.1"], []]
    assert api.convert_identifier("COPIES", strategy="all")["target_id"] == ["ENSG2.1", "LRG_1.1"]
    assert api.convert_identifier("LRG_1")["target_id"] == ["LRG_1.1"]


def test_symbol_that_two_databases_give_to_two_genes_is_one_gene_when_the_database_is_named(monkeypatch, tmp_path):
    """Decision 33: the harmonizer flags the tie unless it is told which database a dataset's names are from."""
    import anndata as ad

    from idtrack import API, HarmonizeFeatures

    # `Rik1` is the MGI symbol of ENSMUSG1 and NCBI's name of ENSMUSG2, which MGI calls `Gm2`. `Arh1` is the MGI
    # symbol of ENSMUSG3 and ENSMUSG4 and NCBI's name of ENSMUSG3 only. `Both1` is the MGI symbol of ENSMUSG5 and
    # ENSMUSG6 and nothing else.
    genes = lifetimes({f"ENSMUSG{i}": [(103, 104, 1)] for i in range(1, 7)})
    rows = [
        ("ENSMUSG1.1", "Rik1", "MGI Symbol", 39),
        ("ENSMUSG2.1", "Rik1", "EntrezGene", 39),
        ("ENSMUSG2.1", "Gm2", "MGI Symbol", 39),
        ("ENSMUSG3.1", "Arh1", "MGI Symbol", 39),
        ("ENSMUSG3.1", "Arh1", "EntrezGene", 39),
        ("ENSMUSG4.1", "Arh1", "MGI Symbol", 39),
        ("ENSMUSG5.1", "Both1", "MGI Symbol", 39),
        ("ENSMUSG6.1", "Both1", "MGI Symbol", 39),
    ]
    world = World("mus_musculus", {39: genes}, externals={103: rows, 104: rows})
    world.install(monkeypatch)
    os.makedirs(tmp_path / "repo")
    GraphMaker(world.manager(tmp_path / "repo", 39, first=DB.first_release("mus_musculus"))).get_graph()
    api = API(str(tmp_path / "repo"))
    api.build_graph("mus_musculus", 104)

    unnamed = api.convert_identifier("Rik1", final_database="MGI Symbol")
    assert unnamed["target_id"] == ["Rik1", "Gm2"] and unnamed["tie"] is True
    named = api.convert_identifier("Rik1", final_database="MGI Symbol", from_database="MGI Symbol")
    assert named["target_id"] == ["Rik1"] and named["tie"] is False and named["from_databases"] == ["MGI Symbol"]

    # decision 36: the named database says which genes are entered, not which evidence counts. Both genes carry
    # `Arh1` in the named database; the one that a second database also calls so is the better supported, named
    # or not. Where nothing tells the genes apart the tie stays.
    for kwargs in ({}, {"from_database": "MGI Symbol"}):
        supported = api.track.convert("Arh1", prioritize_to_one_filter=True, **kwargs)
        assert list(supported) == ["ENSMUSG3.1"] and supported["ENSMUSG3.1"]["tie"] is False, kwargs
        assert supported["ENSMUSG3.1"]["entry_support"] == 2
        every = api.track.convert("Arh1", **kwargs)
        assert list(every) == ["ENSMUSG3.1", "ENSMUSG4.1"] and every["ENSMUSG4.1"]["entry_support"] == 1
        equal = api.track.convert("Both1", prioritize_to_one_filter=True, **kwargs)
        assert list(equal) == ["ENSMUSG5.1", "ENSMUSG6.1"] and equal["ENSMUSG5.1"]["tie"] is True, kwargs
    assert api.track.convert("Arh1", from_database="MGI Symbol")["ENSMUSG3.1"]["from_databases"] == ["MGI Symbol"]
    assert api.track.convert("Arh1")["ENSMUSG3.1"]["from_databases"] == ["EntrezGene", "MGI Symbol"]
    # as NCBI's name `Arh1` is the first gene alone: the second is not entered
    assert list(api.track.convert("Arh1", from_database="EntrezGene")) == ["ENSMUSG3.1"]

    path = tmp_path / "study.h5ad"
    names = ["Rik1", "Gm2", "Arh1", "Both1"]
    ad.AnnData(X=np.ones((2, 4), dtype=np.float32), var=pd.DataFrame(index=names)).write_h5ad(path)

    def statuses(folder, **kwargs):
        harmonizer = HarmonizeFeatures(
            "unit",
            {"study": str(path)},
            str(tmp_path / folder),
            str(tmp_path / "repo"),
            104,
            api=api,
            verbose_level=0,
            **kwargs,
        )
        return harmonizer.feature_table("study")["status"].to_dict()

    assert statuses("unnamed", final_database="MGI Symbol") == {
        "Rik1": "ambiguous",
        "Gm2": "converted",
        "Arh1": "converted",
        "Both1": "ambiguous",
    }
    # with the database named only the true tie is left
    assert statuses("named", final_database="MGI Symbol", source_databases={"study": "MGI Symbol"}) == {
        "Rik1": "converted",
        "Gm2": "converted",
        "Arh1": "converted",
        "Both1": "ambiguous",
    }
    # named as NCBI's names, `Rik1` is the other gene, and `Gm2` and `Both1` are no names of NCBI
    assert statuses("ncbi", final_database="MGI Symbol", source_databases={"study": "EntrezGene"}) == {
        "Rik1": "converted",
        "Gm2": "no_conversion",
        "Arh1": "converted",
        "Both1": "no_conversion",
    }


# ---------------------------------------------------------------- the API on a graph file of the real builder


def test_databases_that_attach_an_id_to_the_products_of_a_gene_count_for_the_gene(monkeypatch, tmp_path):
    """Decision 39: the databases an id is read through, and the support of its gene, count every attachment."""
    from idtrack import API

    # `P100` is UniProt's gene name of ENSSSCG1, the UniProt accession of its translation and a RefSeq id of its
    # transcript: three databases behind one gene. NCBI and UniProt's gene names give it to ENSSSCG2 as well:
    # two databases. `Q200` is attached to ENSSSCG3 on its translation only, and to ENSSSCG2 on the gene.
    genes = lifetimes({f"ENSSSCG{i}": [(113, 114, 1)] for i in (1, 2, 3)})
    name, accession, refseq, ncbi = "UniProtKB Gene Name", "Uniprot/SWISSPROT", "RefSeq_peptide", "EntrezGene"
    rows = [
        ("ENSSSCG1.1", "P100", name, 111),
        ("ENSSSCP1.1", "P100", accession, 111, "translation"),
        ("ENSSSCT1.1", "P100", refseq, 111, "transcript"),
        ("ENSSSCG2.1", "P100", name, 111),
        ("ENSSSCG2.1", "P100", ncbi, 111),
        ("ENSSSCP3.1", "Q200", accession, 111, "translation"),
        ("ENSSSCG2.1", "Q200", ncbi, 111),
    ]
    world = World("sus_scrofa", {111: genes}, externals={113: rows, 114: rows})
    world.install(monkeypatch)
    os.makedirs(tmp_path / "repo")
    GraphMaker(world.manager(tmp_path / "repo", 111, first=DB.first_release("sus_scrofa"))).get_graph()
    api = API(str(tmp_path / "repo"))
    api.build_graph("sus_scrofa", 114)
    track = api.track

    every = track.convert("P100")
    assert list(every) == ["ENSSSCG1.1", "ENSSSCG2.1"]  # the better supported gene first
    assert every["ENSSSCG1.1"]["entry_support"] == 3 and every["ENSSSCG2.1"]["entry_support"] == 2
    assert every["ENSSSCG1.1"]["from_databases"] == sorted([name, accession, refseq, ncbi])
    best = api.convert_identifier("P100")
    assert (best["target_id"], best["tie"]) == (["ENSSSCG1.1"], False)
    assert best["from_databases"] == sorted([name, accession, refseq, ncbi])
    # the chain reported for a gene is still the shortest one: the gene's own link
    path = track.convert("P100", return_path=True)["ENSSSCG1.1"]["the_path"]
    assert path[0][:2] == ("P100", "ENSSSCG1.1")

    # a named database says which genes are entered; every attachment still counts as support
    for database, entered in ((accession, ["ENSSSCG1.1"]), (refseq, ["ENSSSCG1.1"]), (ncbi, ["ENSSSCG2.1"])):
        named = track.convert("P100", from_database=database)
        assert list(named) == entered, database
        assert named[entered[0]]["from_databases"] == [database]
        assert named[entered[0]]["entry_support"] == (3 if entered == ["ENSSSCG1.1"] else 2)
    both = track.convert("P100", from_database=name)
    assert list(both) == ["ENSSSCG1.1", "ENSSSCG2.1"] and both["ENSSSCG1.1"]["entry_support"] == 3
    assert list(track.convert("P100", from_database=name, prioritize_to_one_filter=True)) == ["ENSSSCG1.1"]

    # an identifier attached to one gene through its translation only and to another on the gene: equal support
    tied = api.convert_identifier("Q200")
    assert (tied["target_id"], tied["tie"]) == (["ENSSSCG2.1", "ENSSSCG3.1"], True)
    assert tied["from_databases"] == sorted([accession, ncbi])
    assert track.convert("Q200", from_database=accession)["ENSSSCG3.1"]["entry_support"] == 1


def test_api_loads_a_graph_file_without_manager_or_network_and_converts(monkeypatch, tmp_path):
    """`API.build_graph` on an existing file needs nothing but the file; the answers are the engine's."""
    import json

    from idtrack import API
    from idtrack import GraphMismatchError as ExportedMismatch
    from idtrack import InactiveAtReleaseError

    world = symbols_world()
    world.install(monkeypatch)
    built = GraphMaker(world.manager(tmp_path, 111, first=DB.first_release("sus_scrofa"))).get_graph()
    assert os.path.exists(tmp_path / "graph_sus_scrofa_assembly111_min56_max115_narrow.npz")

    def refuse(*args, **kwargs):
        raise AssertionError("loading a stored graph must not open a connection or make a manager")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(DatabaseManager, "__init__", refuse)
    api = API(str(tmp_path))
    api.build_graph("sus_scrofa", 115)
    assert api.track.db_manager is None and list(api.track.graph.nodes) == list(built.nodes)
    assert api.graph_releases() == [113, 114, 115] and api.list_genome_assemblies() == [111]
    assert "VGNC Symbol" in api.valid_final_databases() and "ensembl_gene" in api.valid_final_databases(113)

    result = api.convert_identifier("stat2", final_database="UniProtKB Gene Name", explain=True)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert (result["graph_id"], result["match"]) == ("STAT2", "exact")  # as UniProt writes the name at 115
    assert api.convert_identifier("Stat2")["match"] == "case"
    assert result["target_id"] == ["stat2"] and result["last_node"] == [["ENSSSCG1.1", "stat2"]]
    assert result["from_release"] == 115 and result["from_release_inferred"] is True
    assert result["from_databases"] == ["UniProtKB Gene Name", "VGNC Symbol"]
    assert result["tie"] is False and result["no_target"] is False and result["no_target_genes"] == []
    assert api.convert_identifier("sla_1")["match"] == "separator"
    assert api.convert_identifier("ENSSSCT1")["target_id"] == ["ENSSSCG1.1"]  # a transcript id without version
    assert api.convert_identifier("ENSSSCT1")["graph_id"] == "ENSSSCT1.1"
    unnamed = api.convert_identifier("ENSSSCG3", final_database="UniProtKB Gene Name")
    assert unnamed["no_target"] is True and unnamed["final_database"] == "ensembl_gene"
    assert unnamed["target_id"] == ["ENSSSCG3.1"]
    # WNT4 exists at 113 and 115, STAT2 at 113 to 115: two releases hold both, and both are named
    assert api.infer_identifier_source(["WNT4", "STAT2"], mode="ensembl_release") == [113, 115]

    # decision 15: the engine is strict about a release at which the graph does not hold the id; the API reads the id
    # where the graph holds it and says so
    with pytest.raises(InactiveAtReleaseError, match="it is active at the releases: 115"):
        api.track.convert("SLA", from_release=113)
    lenient = api.convert_identifier("SLA", from_release=113)
    assert lenient["target_id"] == ["ENSSSCG3.1"] and lenient["error"] is None
    assert (lenient["from_release_given"], lenient["from_release"]) == (113, 115)
    assert lenient["from_release_fallback"] is True and lenient["from_release_inferred"] is True
    assert lenient["target_id"] == api.convert_identifier("SLA")["target_id"]
    held = api.convert_identifier("WNT4", from_release=113)
    assert (held["from_release_given"], held["from_release"], held["from_release_fallback"]) == (113, 113, False)
    batch = api.convert_identifier_multiple(["SLA", "WNT4", "nope"], from_release=113, verbose=False)
    assert [item["from_release_fallback"] for item in batch] == [True, False, False]
    assert [item["error"] for item in batch] == [None, None, None]
    assert batch[1]["target_id"] == ["ENSSSCG2.1"] and batch[2]["no_corresponding"] is True

    # decision 14: a version the graph cannot know is flagged; one that never existed is not another version's gene
    newer = api.convert_identifier("ENSSSCG1.9")
    assert (newer["graph_id"], newer["match"]) == ("ENSSSCG1.1", "version_newer_than_graph")
    assert newer["target_id"] == ["ENSSSCG1.1"] and newer["known_versions"] == []
    never = api.convert_identifier("ENSSSCG1.0")
    assert never["target_id"] == [] and never["no_corresponding"] is True and never["match"] is None
    assert never["known_versions"] == ["ENSSSCG1.1"]
    assert json.loads(json.dumps(never, allow_nan=False)) == never

    # a file that holds another graph than the one asked for is an error that names the difference
    os.link(
        tmp_path / "graph_sus_scrofa_assembly111_min56_max115_narrow.npz",
        tmp_path / "graph_sus_scrofa_assembly111_min56_max114_narrow.npz",
    )
    with pytest.raises(ExportedMismatch, match="ignore_after"):
        api.build_graph("sus_scrofa", 114)
    # decision 19: a graph is built on the organism's assembly order; the API takes no primary assembly
    with pytest.raises(TypeError, match="genome_assembly"):
        api.build_graph("sus_scrofa", 115, genome_assembly=102)
    # decision 25: nothing is computed in advance for a graph, so there is no parameter that asks for it
    with pytest.raises(TypeError, match="calculate_caches"):
        api.build_graph("sus_scrofa", 115, calculate_caches=True)
    # decision 32: nor a method that would compute it afterwards; the table that only the self-tests read is
    # computed when it is first read
    assert not hasattr(api, "calculate_graph_caches")
    assert "available_releases_given_database_assembly" not in api.track.graph._frozen().globals
    assert api.track.graph.available_releases_given_database_assembly[("VGNC Symbol", 111)] == {114, 115}
    assert ExportedMismatch is GraphMismatchError and api.track.graph.graph["ensembl_release"] == 115
