#!/usr/bin/env python3
"""The graph built from real Ensembl tables says what the tables say.

The graphs of pig and human are built by ``API.build`` from the cut tables of ``tests/data/real``
(``tests/_real_tables.py``). Here the content of each graph is compared with the tables themselves, read by
``tests._real_tables.Ensembl``, which shares no code with the package: which gene ids exist at which release, which
external identifiers each database has at each release, which genes lie outside the reference sequence. The tests
of the answers are in ``test_real_truth.py`` and ``test_real_decisions.py``.
"""

from __future__ import annotations

import collections
import lzma

import pytest

from idtrack import ids, organisms
from idtrack.errors import EnsemblSchemaError
from tests._real_tables import (
    ORGANISMS,
    SNAPSHOT,
    RealTables,
    read_as_typed,
    read_genes,
    read_truth,
    tables_path,
    truth_path,
)

pytestmark = pytest.mark.parametrize("organism", [o for o in ORGANISMS if tables_path(o).is_file()])
SIZE_LIMIT = 600 * 1024  # bytes per table file: the repository's size check, as tools/cut_tables.py says why
SYNONYM = ids.SYNONYM_PREFIX


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
        tables.table(assembly, first, "gene", ["gene_id", "description"])
    with pytest.raises(AssertionError, match="rebuild"):
        tables.table(assembly, first, "karyotype", None)
    with pytest.raises(AssertionError, match="rebuild"):
        tables.table(assembly, SNAPSHOT + 1, "gene", ["gene_id"])


def test_build_reads_only_the_cut_tables_and_the_file_gives_the_same_graph(organism, real_world):
    """The graph was built without the network, and loading its file gives the graph that was built."""
    world = real_world(organism)
    built, loaded = world.built.graph, world.api.graph

    assert world.tables_served > 500
    assert built is not loaded
    assert loaded.record == built.record
    assert loaded.record.organism == organism and list(loaded.record.releases) == sorted(world.backbone)
    assert loaded.record.serving_assembly == world.backbone
    assert list(loaded.nodes) == list(built.nodes)
    if truth_path(organism).is_file():
        for query in read_truth(organism):
            arguments = (query["id"], query["from_release"], query["to_release"], query["final_database"])
            assert world.built.convert(*arguments, explain=True) == world.api.convert(*arguments, explain=True)
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
    graph = world.api.graph
    active: dict[int, set[str]] = {release: set() for release in world.backbone}
    for node in graph.nodes:
        if graph.node_type(node) == "ensembl_gene" and graph.version(node) not in ids.SENTINEL_VERSIONS:
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
    corrections = {tuple(c["recorded"]): tuple(c["used"]) for c in graph.record.history["corrections"]}
    assert set(corrections) == recorded
    assert corrections == {"sus_scrofa": {(106, 104): (106, 107)}, "homo_sapiens": {(114, 114): (114, 115)}}[organism]


def test_external_ids_of_every_release_are_those_of_ensembls_tables(organism, real_world):
    """Each database has, on each assembly at each release, the identifiers that Ensembl's tables give it there.

    Compared for every database, assembly and release of the graph: nothing is missing and nothing is added. The
    graph keeps one node for identifiers that differ only in letter case, so the comparison ignores case; an
    identifier that is itself an Ensembl id of the graph stays that Ensembl id and is no external identifier.
    """
    world = real_world(organism)
    graph = world.api.graph
    in_graph: dict[tuple[str, int, int], set[str]] = collections.defaultdict(set)
    for node in graph.nodes:
        if graph.node_type(node) == ids.EXTERNAL:
            for database, assembly, release in graph.trios(node):
                in_graph[(database, assembly, release)].add(node.lower())
    misplaced = {name.lower() for name in graph.misplaced_external_entries}

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


def test_mirbase_entries_left_out_are_those_of_genes_at_several_loci(organism, real_world):
    """Decision 73 by the tables: the entries the build leaves out are those that name genes at several loci.

    The oracle counts loci from the tables alone (:py:meth:`tests._real_tables.Ensembl.one_locus_count`: a copy on a
    ``non_ref`` region adds none, the ids of one locus over time are one, a pseudoautosomal region of Y is read on the
    region of X it repeats), apart from the package's rule. The cut tables hold entries of several stable ids that
    are one gene (pig: successive ids joined by Ensembl's history; human: a gene and its copies on patches, ids at
    overlapping places, and the X and Y copies of the pseudoautosomal ``MIR3690`` and ``MIR6089``) and, in pig,
    entries of genes at several loci, so that counting genes by stable id fails here.
    """
    world = real_world(organism)
    oracle = world.ensembl
    linked: dict[str, set[str]] = collections.defaultdict(set)
    for assembly, release, name in world.tables.tables:
        if name == "xref" and int(release) <= SNAPSHOT:
            for gene, database, value, kind, _form in oracle._links_of_release(assembly, release):
                if database in oracle._mirbase_names():
                    linked[(value if kind == "id" else SYNONYM + value).lower()].add(gene)
            for gene, database, value in oracle.display_names(assembly, release):  # I129: display names are held
                if database in oracle._mirbase_names():
                    linked[value.lower()].add(gene)
    several = {value for value, genes in linked.items() if len(genes) > 1}
    left_out = {value for value in several if oracle.one_locus_count(linked[value]) > 1}

    assert world.api.graph.entries_of_several_genes == left_out
    assert several - left_out and bool(left_out) is (organism == "sus_scrofa"), (sorted(several), sorted(left_out))


def test_ncbi_gene_symbols_are_an_authority_by_the_name_the_tables_give_them(organism, real_world):
    """Decision 94 reads NCBI's symbols as an authority under the name decision 53 gives them, not a typed name.

    The graph names the label rows of Ensembl's ``EntrezGene`` as the oracle reads the tables' ``external_db``
    (:py:meth:`tests._real_tables.Ensembl.label_names`), and that name is among the graph's authority databases. Fails
    if the authority keeps a typed name that the tables do not give.
    """
    world = real_world(organism)
    graph = world.api.graph
    name = world.ensembl.label_names()["EntrezGene"]
    assert graph.label_names["EntrezGene"] == name
    assert name in graph.available_external_databases
    assert graph.authority_databases == frozenset(organisms.SYMBOL_DATABASES[organism]) | {name}


def test_external_values_are_read_as_a_typed_identifier_is_read(organism, real_world):
    """Decision 83 on the side of the tables: the graph holds the name a database wrote with punctuation, as the name.

    The cut tables hold UniProt's gene names written with a semicolon (pig ``ACYP1;``, human ``ABCF1;``); the graph has
    no external id that ends in the punctuation of a sentence, and such values are in the graph as the plain names
    (which ids each database has at each release, read alike by the oracle, is
    :py:func:`test_external_ids_of_every_release_are_those_of_ensembls_tables`). Fails if the builder keeps the values
    as written (a typed ``A1BG;`` was read one way and the graph's value another).
    """
    world = real_world(organism)
    graph = world.api.graph
    punctuated: set[tuple[str, str]] = set()
    for assembly, release, name in world.tables.tables:
        if name != "xref" or int(release) > SNAPSHOT:
            continue
        xref = world.ensembl._frame(assembly, release, "xref")
        for value in set(xref["display_label"].dropna().astype(str)) | set(xref["dbprimary_acc"].dropna().astype(str)):
            text = value.strip()
            if text and text[-1] in "!,.:;?":
                punctuated.add((text, read_as_typed(text)))
    assert punctuated
    externals = [node for node in graph.nodes if graph.node_type(node) == ids.EXTERNAL]
    assert not [n for n in externals if n.removeprefix(SYNONYM)[-1] in "!,.:;?"]
    lower = {node.lower() for node in externals}
    assert {plain.lower() for _text, plain in punctuated} & lower


def test_genes_outside_the_reference_sequence_are_those_ensembl_marks(organism, real_world):
    """The graph's mark of a gene on a patch, a haplotype or an LRG record is Ensembl's ``non_ref`` attribute."""
    world = real_world(organism)
    graph = world.api.graph
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
    """The identifiers a database has at any release, and the releases at which two gene versions exist together.

    Both read from the graph's public accessors (cross-reference edges, active ranges) and compared with the tables.
    """
    world = real_world(organism)
    graph = world.api.graph
    misplaced = {name.lower() for name in graph.misplaced_external_entries}
    ids: dict[str, set[str]] = collections.defaultdict(set)
    for assembly in sorted({int(a) for a in world.tables.indices}):
        for release in world.ensembl.releases(assembly):
            for _gene, name, value, kind, form in world.ensembl.links(assembly, release):
                if kind == "id" and world.ensembl.selected(form, name, assembly, release):
                    ids[name].add(value.lower())
    assert len(ids) >= 10
    externals = [node for node in graph.nodes if graph.node_type(node) == "external"]

    def nodes_of(database: str) -> set[str]:
        return {
            node
            for node in externals
            if any(database in connection.databases for _, connection in graph.xref_edges(node))
        }

    for database, expected in sorted(ids.items()):
        assert {node.lower() for node in nodes_of(database)} == expected - misplaced, database

    releases_of: dict[str, list[int]] = collections.defaultdict(list)
    for release in sorted(world.backbone):
        for gene, version in world.genes_at(release).items():
            releases_of[f"{gene}.{version}"].append(release)
    nodes = sorted(releases_of)[:: max(1, len(releases_of) // 40)]
    shared = 0
    for first, second in zip(nodes, nodes[1:]):
        together = sorted(set(releases_of[first]) & set(releases_of[second]))
        ranges = graph.get_intersecting_ranges(graph.active_ranges(first), graph.active_ranges(second))
        covered = [r for low, high in ranges for r in range(int(low), int(min(high, SNAPSHOT)) + 1)]
        assert covered == together, (first, second)
        shared += bool(together)
    assert shared >= 5
