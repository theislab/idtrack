#!/usr/bin/env python3
"""The package's own self-tests (``TrackTests``) on the graphs of real organisms.

``API.build_graph(..., return_test=True)`` attaches a ``TrackTests``: checks that compare the graph with Ensembl's
tables, for developers and for the end of a graph build. Here they run on the graphs built from the cut tables of
``tests/data/real`` (three genome assemblies per organism), with the tables replayed to the manager. What each
check must find is known: the graph was compared with the tables by independent code in ``test_real_graph.py``.
Every check holds on both graphs.
"""

from __future__ import annotations

import collections

import pytest

import idtrack
from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._track_tests import CheckResult, TrackTests
from tests._real_tables import ORGANISMS, SNAPSHOT, tables_path

AVAILABLE = [organism for organism in ORGANISMS if tables_path(organism).is_file()]
pytestmark = pytest.mark.parametrize("organism", AVAILABLE)


@pytest.fixture(scope="module")
def self_tests(real_world):
    """Return ``get(organism)``: the self-tests of an organism's graph, their manager reading the cut tables."""
    patches = pytest.MonkeyPatch()
    made: dict[str, TrackTests] = {}

    def get(organism: str) -> TrackTests:
        if organism not in made:
            world = real_world(organism)
            world.tables.install(patches, world.repository)
            patches.setattr(DatabaseManager, "_core_db_index_cache", {})
            api = idtrack.API(local_repository=world.repository)
            api.build_graph(organism, SNAPSHOT, return_test=True)
            assert isinstance(api.track, TrackTests)
            made[organism] = api.track
        return made[organism]

    yield get
    patches.undo()


def test_invariants_of_a_built_graph_hold(self_tests, organism):
    """Node types, edge summaries, the ranges of versions and of ids without version are consistent."""
    tests = self_tests(organism)
    assert tests.db_manager.organism == organism and int(tests.db_manager.ensembl_release) == SNAPSHOT
    for result in (
        tests.is_node_consistency_robust(verbose=False),
        tests.is_combined_summary_complete(),
        tests.is_edge_with_same_nts_only_at_backbone_nodes(),
        tests.is_range_functions_robust(verbose=False),
        tests.is_base_is_range_correct(verbose=False),
    ):
        assert isinstance(result, CheckResult) and result.holds is True, (result.check, result.details)
        assert result.details["compared"] >= 400


def test_final_conversions_of_the_last_release_equal_the_tables(self_tests, real_world, organism):
    """The check of gene-to-database conversions agrees with the tables, read here by independent code.

    Every gene of the release is converted to the symbol database: the conversions are the symbols the tables give
    the genes, and the only genes the check lists as lost are those without a symbol.
    """
    tests, world = self_tests(organism), real_world(organism)
    assembly = world.backbone[SNAPSHOT]
    database = next(d for d in DB.symbol_databases[organism] if d in world.api.valid_final_databases(SNAPSHOT))
    symbols: dict[str, set[str]] = collections.defaultdict(set)
    for gene, name, value, kind, form in world.ensembl.links(assembly, SNAPSHOT):
        if kind == "id" and name == database and world.ensembl.selected(form, name, assembly, SNAPSHOT):
            symbols[f"{gene}.{world.genes_at(SNAPSHOT)[gene]}"].add(value)

    result = tests.is_final_external_conversion_robust(
        database=database, ens_rel=SNAPSHOT, verbose=False, convert_using_release=True
    )
    details = result.details
    assert result.holds is True, {key: value for key, value in details.items() if key != "run"}
    assert (details["database"], details["assembly"], details["release"]) == (database, assembly, SNAPSHOT)
    assert details["different"] == [] and details["unexpected"] == [] and details["lost_with_id"] == []
    assert {gene: set(found) for gene, found in details["run"]["conversion"].items()} == dict(symbols)
    assert set(details["lost"]) == {f"{g}.{v}" for g, v in world.genes_at(SNAPSHOT).items()} - set(symbols)
    assert len(symbols) >= 100


def test_final_conversions_are_checked_at_a_release_of_an_earlier_assembly(self_tests, real_world, organism):
    """The same check at the last release that an earlier assembly served: its genes, its tables, its databases."""
    tests, world = self_tests(organism), real_world(organism)
    release = max(r for r, assembly in world.backbone.items() if assembly != world.backbone[SNAPSHOT])
    assembly = world.backbone[release]
    view = collections.defaultdict(set)  # the databases that attach ids to genes at the release, from the tables
    for _gene, name, _value, kind, form in world.ensembl.links(assembly, release):
        if kind == "id" and form == "gene" and world.ensembl.selected(form, name, assembly, release):
            view[name].add(_gene)
    assert len(view) >= 3
    checked = 0
    for database in sorted(world.api.valid_final_databases(release)):
        if tests.graph.external_database_connection_form.get(database) != "gene" or database.startswith("synonym"):
            continue
        result = tests.is_final_external_conversion_robust(
            database=database, ens_rel=release, verbose=False, convert_using_release=True
        )
        if database in view:
            assert result.holds is True, {key: value for key, value in result.details.items() if key != "run"}
            assert (result.details["assembly"], result.details["release"]) == (assembly, release)
            assert len(result.details["run"]["conversion"]) == len(view[database])
            checked += 1
        else:
            # a database of the assembly that has no id at this release: nothing to check, and the result says so
            assert result.holds is False and result.details["error"].startswith("ValueError: No external database")
    assert checked == len(view)


def test_database_that_the_graph_does_not_hold_is_refused_with_what_is_valid(self_tests, real_world, organism):
    """A conversion run or a check asked for a database that cannot work raises, and lists what the API offers."""
    tests, world = self_tests(organism), real_world(organism)
    release = max(r for r, assembly in world.backbone.items() if assembly != world.backbone[SNAPSHOT])
    assembly = world.backbone[release]  # an earlier assembly, at the last release it served

    def run(from_database, to_database):
        return tests.history_travel_testing(
            from_release=release, from_assembly=assembly, from_database=from_database, to_release=SNAPSHOT,
            to_database=to_database, go_external=False, prioritize_to_one_filter=True, convert_using_release=True,
            from_fraction=0.01, verbose=False,
        )  # fmt: skip

    own_genes = DB.nts_assembly[assembly][DB.backbone_form]
    targets = world.api.valid_final_databases(SNAPSHOT)
    for target in ("No Such Database", "gene", own_genes, DB.nts_ensembl["transcript"], targets[0].swapcase()):
        with pytest.raises(ValueError) as refused:
            run(own_genes, target)
        assert str(refused.value) == (
            f"`to_database`={target!r} is not a database of the loaded graph at Ensembl release {SNAPSHOT}; "
            f"valid: {targets}."
        )
    sources = set(world.api.list_external_databases_by_assembly()[assembly])
    sources |= {DB.nts_ensembl[form] for form in DB.forms_in_order}
    sources |= {DB.nts_assembly[assembly][form] for form in DB.forms_in_order} | {DB.nts_base_ensembl["gene"]}
    other_genes = DB.nts_assembly[world.backbone[SNAPSHOT]][DB.backbone_form]  # the gene ids of another assembly
    for source in ("No Such Database", "gene", other_genes):
        with pytest.raises(ValueError) as refused:
            run(source, DB.nts_ensembl["gene"])
        assert str(refused.value).startswith(
            f"`from_database`={source!r} is not a database of the loaded graph on genome assembly {assembly} "
        )
        assert str(refused.value).endswith(f"valid: {sorted(sources)}.")
    # what is valid still runs: every id drawn gets one outcome (which ids convert depends on the draw: not asserted)
    report = run(own_genes, DB.nts_ensembl["gene"])
    stopped = [the_id for the_id, _ in report["history_voyage_failed_gracefully"]]
    outcomes = [*report["conversion"], *report["lost_item"], *stopped]
    assert report["ids"]["from"] and sorted(outcomes) == report["ids"]["from"]
    assert not report["query_not_in_the_graph"] and not report["history_voyage_failed_unknown"]

    forms = tests.graph.external_database_connection_form
    checkable = sorted(d for d in world.api.list_external_databases() if forms[d] == "gene" and "::" not in d)
    not_of_genes = next(d for d in world.api.list_external_databases() if forms[d] != "gene")
    for database in ("No Such Database", DB.nts_ensembl["gene"], not_of_genes):
        with pytest.raises(ValueError) as refused:
            tests.is_final_external_conversion_robust(database=database, ens_rel=SNAPSHOT, verbose=False)
        assert str(refused.value) == (
            f"`database`={database!r} is not an external database of the loaded graph that attaches its ids to "
            f"genes; valid: {checkable}."
        )


def test_gene_id_lists_are_checked_at_every_release(self_tests, real_world, organism):
    """The gene ids of the graph equal the id tables on every assembly at every release it has.

    The same is asserted independently in ``test_real_graph``; here the package's own check must see it, for the
    releases of the earlier assemblies too.
    """
    tests, world = self_tests(organism), real_world(organism)
    result = tests.is_id_functions_consistent_ensembl(verbose=False)
    assert result.holds is True, result.details
    pairs = {(a, r) for a in (int(key) for key in world.tables.indices) for r in world.ensembl.releases(a)}
    assert result.details["compared"] == len(pairs) >= len(world.backbone)
    if organism == "homo_sapiens":
        assert len(pairs) > len(world.backbone)  # also the releases at which GRCh37 is listed beside GRCh38


def test_ranges_of_versions_agree_with_their_edges(self_tests, organism):
    """The releases of a gene version's edges are the releases of its history, on whatever assembly it lived."""
    result = self_tests(organism).is_id_functions_consistent_ensembl_2(verbose=False)
    assert result.holds is True, result.details
    assert result.details["compared"] >= 500


def test_external_id_lists_are_checked_at_every_release(self_tests, organism):
    """The external ids of the graph equal the tables (asserted independently in test_real_graph)."""
    result = self_tests(organism).is_id_functions_consistent_external(verbose=False)
    assert result.holds is True, result.details
    assert result.details["compared"] > 500
