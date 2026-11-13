#!/usr/bin/env python3
"""The graph built from real Ensembl tables says what the tables say.

The graphs of pig and human are built by ``API.build_graph`` from the cut tables of ``tests/data/real``
(``tests/_real_tables.py``). Here the content of each graph is compared with the tables themselves, read by
``tests._real_tables.Ensembl``, which shares no code with the package: which gene ids exist at which release, which
external identifiers each database has at each release, which genes lie outside the reference sequence. The tests
of the answers are in ``test_real_truth.py`` and ``test_real_decisions.py``.
"""

from __future__ import annotations

import collections
import lzma

import pytest

from idtrack._db import DB, EnsemblSchemaError
from tests._real_tables import ORGANISMS, SNAPSHOT, RealTables, read_genes, read_truth, tables_path, truth_path

pytestmark = pytest.mark.parametrize("organism", [o for o in ORGANISMS if tables_path(o).is_file()])
SIZE_LIMIT = 500 * 1024  # bytes per table file, as tests/data/real/build_real_tables.py enforces
SYNONYM = DB.synonym_id_nodes_prefix


def test_cut_tables_hold_every_gene_of_the_list_and_stay_small(organism, real_world):
    """A gene of the list that Ensembl's tables lack, or a file over the limit, is an error of the test data."""
    world = real_world(organism)
    genes = read_genes(organism)
    seen = {
        gene
        for (assembly, release, name) in world.tables.tables
        if name == "xref"
        for gene in world.ensembl.genes(assembly, release)
    }

    # every gene of the list is in a gene table, or is an id that only Ensembl's history names (ids of patch
    # annotation, ``ASMPATCHG...``, that were renamed before a release carried them)
    events = world.ensembl.events(world.backbone[SNAPSHOT], SNAPSHOT)
    in_history = set(events["old_stable_id"].dropna()) | set(events["new_stable_id"].dropna())
    assert world.tables.genes == genes
    assert seen <= set(genes) and set(genes) - seen <= in_history
    assert len(seen) >= 0.9 * len(genes)
    assert tables_path(organism).stat().st_size <= SIZE_LIMIT
    with lzma.open(tables_path(organism), "rb") as handle:
        assert handle.read(1) == b"{"


def test_cut_tables_hold_the_events_the_queries_are_about(organism, real_world):
    """Version changes, splits, merges, retirements, an assembly change: each must be in the tables, not assumed."""
    world = real_world(organism)
    last = max(world.ensembl.releases(world.backbone[SNAPSHOT]))
    events = world.ensembl._frame(world.backbone[SNAPSHOT], last, "stable_id_event")
    events = events[events["type"] == "gene"]
    between = events[events["old_stable_id"].notna() & events["new_stable_id"].notna()]
    other = between[between["old_stable_id"] != between["new_stable_id"]]
    successors = other.groupby(["mapping_session_id", "old_stable_id"])["new_stable_id"].nunique()
    predecessors = other.groupby(["mapping_session_id", "new_stable_id"])["old_stable_id"].nunique()
    same = between[between["old_stable_id"] == between["new_stable_id"]]

    assert (successors > 1).sum() >= 5, "ids that were split"
    assert (predecessors > 1).sum() >= 5, "ids that were merged"
    assert (successors == 1).sum() >= 5, "ids that were replaced by one id"
    assert (same["old_version"] != same["new_version"]).sum() >= 20, "ids whose version changed"
    assert events["new_stable_id"].isna().sum() >= 5, "ids that were retired"
    assert events["old_stable_id"].isna().sum() >= 5, "ids that were born"
    assert len(set(world.backbone.values())) >= 3, "releases of three assemblies"
    kinds = (
        collections.Counter(query["kind"] for query in read_truth(organism)) if truth_path(organism).is_file() else {}
    )
    assert all(count >= 2 for count in kinds.values())


def test_replay_answers_as_ensembl_does_and_never_makes_a_table_up(organism):
    """A column that an old release lacks is a schema error; anything the file does not hold is a failure."""
    tables = RealTables.load(organism)
    first = min(release for (_assembly, release, name) in tables.tables if name == "gene")
    assembly = next(a for (a, release, name) in tables.tables if name == "gene" and release == first)

    with pytest.raises(EnsemblSchemaError) as error:
        tables.table(assembly, first, "gene", ["gene_id", "stable_id", "version"])  # in `gene_stable_id` then
    assert set(error.value.missing_columns) == {"stable_id", "version"}
    old = tables.table(assembly, first, "gene_stable_id", ["gene_id", "stable_id", "version"])
    assert list(old.columns) == ["gene_id", "stable_id", "version"] and len(old) > 0
    with pytest.raises(AssertionError, match="rebuild"):
        tables.table(assembly, first, "gene", ["gene_id", "biotype"])
    with pytest.raises(AssertionError, match="rebuild"):
        tables.table(assembly, first, "karyotype", None)
    with pytest.raises(AssertionError, match="rebuild"):
        tables.table(assembly, SNAPSHOT + 1, "gene", ["gene_id"])


def test_build_reads_only_the_cut_tables_and_the_file_gives_the_same_graph(organism, real_world):
    """The graph was built without the network, and loading its file gives the graph that was built."""
    world = real_world(organism)
    built, loaded = world.built.track.graph, world.api.track.graph

    assert world.tables_served > 500
    assert world.built.track.db_manager is not None and world.api.track.db_manager is None
    assert loaded.manifest == built.manifest
    assert loaded.manifest["organism"] == organism and loaded.manifest["releases"] == sorted(world.backbone)
    assert {int(r): a for r, a in loaded.manifest["backbone_assembly"].items()} == world.backbone
    assert list(loaded.nodes) == list(built.nodes)
    if truth_path(organism).is_file():
        for query in read_truth(organism):
            arguments = (
                query["id"],
                query["from_release"],
                query["to_release"],
                query["final_database"],
                query["strategy"],
            )
            assert world.built.convert_identifier(*arguments) == world.api.convert_identifier(*arguments)
    for node in list(built.nodes)[:: max(1, len(built.nodes) // 400)]:
        assert loaded.history_edges(node) == built.history_edges(node) or all(
            a[:3] == b[:3] for a, b in zip(loaded.history_edges(node), built.history_edges(node), strict=True)
        )
        assert loaded.xref_edges(node) == built.xref_edges(node)


def test_gene_ids_of_every_release_are_those_of_ensembls_tables(organism, real_world):
    """A gene id with its version is active in the graph exactly at the releases whose gene table holds it.

    This is the history told by the id tables: a mapping session recorded with wrong releases (pig 106 to 107 is
    recorded as 106 to 104, human 114 to 115 as 114 to 114) must not make an id active where it does not exist.
    """
    world = real_world(organism)
    graph = world.api.track.graph
    active: dict[int, set[str]] = {release: set() for release in world.backbone}
    for node in graph.nodes:
        if graph.node_type(node) == "ensembl_gene" and graph.version(node) not in DB.alternative_versions:
            for release in graph.active_releases(node):
                active[release].add(node)

    for release in sorted(world.backbone):
        assert active[release] == {f"{gene}.{version}" for gene, version in world.genes_at(release).items()}, release
    # The sessions that Ensembl records with releases that cannot be (the new release not after the old one) and
    # that change an id of the tables are corrected, each to the pair of releases at which its ids change.
    events = world.ensembl.events(world.backbone[SNAPSHOT], SNAPSHOT)
    changed = events[events["old_stable_id"].notna() & events["new_stable_id"].notna()]
    changed = changed[changed["old_stable_id"] != changed["new_stable_id"]]
    impossible = changed[changed["new_release"] <= changed["old_release"]]
    recorded = {(int(a), int(b)) for a, b in zip(impossible["old_release"], impossible["new_release"])}
    corrections = {tuple(c["recorded"]): tuple(c["used"]) for c in graph.manifest["history"]["corrections"]}
    assert set(corrections) == recorded
    assert corrections == {"sus_scrofa": {(106, 104): (106, 107)}, "homo_sapiens": {(114, 114): (114, 115)}}[organism]


def test_external_ids_of_every_release_are_those_of_ensembls_tables(organism, real_world):
    """Each database has, on each assembly at each release, the identifiers that Ensembl's tables give it there.

    Compared for every database, assembly and release of the graph: nothing is missing and nothing is added. The
    graph keeps one node for identifiers that differ only in letter case, so the comparison ignores case; an
    identifier that is itself an Ensembl id of the graph stays that Ensembl id and is no external identifier.
    """
    world = real_world(organism)
    graph = world.api.track.graph
    in_graph: dict[tuple[str, int, int], set[str]] = collections.defaultdict(set)
    for node in graph.nodes:
        if graph.node_type(node) == DB.nts_external:
            for database, assembly, release in graph.trios(node):
                in_graph[(database, assembly, release)].add(node.lower())
    misplaced = {name.lower() for name in graph.graph["misplaced_external_entry"]}

    in_tables: dict[tuple[str, int, int], set[str]] = collections.defaultdict(set)
    for assembly in sorted(set(world.backbone.values()) | {int(a) for a in world.tables.indices}):
        for release in world.ensembl.releases(assembly):
            for _gene, name, value, kind, form in world.ensembl.links(assembly, release):
                database, external_id = (name, value) if kind == "id" else (SYNONYM + name, SYNONYM + value)
                if world.ensembl.selected(form, database, assembly, release) and external_id.lower() not in misplaced:
                    in_tables[(database, assembly, release)].add(external_id.lower())

    assert len(in_graph) > 500
    assert set(in_graph) == set(in_tables)
    for key in sorted(in_graph):
        assert in_graph[key] == in_tables[key], key


def test_genes_outside_the_reference_sequence_are_those_ensembl_marks(organism, real_world):
    """The graph's mark of a gene on a patch, a haplotype or an LRG record is Ensembl's ``non_ref`` attribute."""
    world = real_world(organism)
    graph = world.api.track.graph
    marked_somewhere = 0
    for release in sorted(world.backbone):
        expected = world.ensembl.non_reference_genes(world.backbone[release], release)
        marked = {
            gene
            for gene, version in world.genes_at(release).items()
            if graph.is_non_reference(f"{gene}.{version}", release)
        }
        assert marked == expected, release
        marked_somewhere += len(marked)

    assert (marked_somewhere > 0) is (organism == "homo_sapiens")
    if organism == "homo_sapiens":
        at_snapshot = world.ensembl.non_reference_genes(world.backbone[SNAPSHOT], SNAPSHOT)
        assert any(gene.startswith("LRG_") for gene in at_snapshot) and any(
            gene.startswith("ENSG") for gene in at_snapshot
        )


def test_ids_of_a_database_and_the_releases_two_ids_share_are_those_of_the_tables(organism, real_world):
    """``get_external_database_nodes`` and ``get_two_nodes_coinciding_releases`` against the tables.

    The identifiers a database has at any release, and the releases at which two versions of genes exist together.
    """
    world = real_world(organism)
    graph = world.api.track.graph
    misplaced = {name.lower() for name in graph.graph["misplaced_external_entry"]}
    ids: dict[str, set[str]] = collections.defaultdict(set)
    for assembly in sorted({int(a) for a in world.tables.indices}):
        for release in world.ensembl.releases(assembly):
            for _gene, name, value, kind, form in world.ensembl.links(assembly, release):
                if kind == "id" and world.ensembl.selected(form, name, assembly, release):
                    ids[name].add(value.lower())
    assert len(ids) >= 10
    for database, expected in sorted(ids.items()):
        assert {node.lower() for node in graph.get_external_database_nodes(database)} == expected - misplaced, database

    releases_of: dict[str, list[int]] = collections.defaultdict(list)
    for release in sorted(world.backbone):
        for gene, version in world.genes_at(release).items():
            releases_of[f"{gene}.{version}"].append(release)
    nodes = sorted(releases_of)[:: max(1, len(releases_of) // 40)]
    shared = 0
    for first, second in zip(nodes, nodes[1:]):
        together = sorted(set(releases_of[first]) & set(releases_of[second]))
        ranges = graph.get_two_nodes_coinciding_releases(first, second)
        covered = [r for low, high in ranges for r in range(int(low), int(min(high, SNAPSHOT)) + 1)]
        assert covered == together, (first, second)
        shared += bool(together)
    assert shared >= 5
