#!/usr/bin/env python3
"""
High-value end-to-end integration tests for idtrack.

These tests intentionally avoid genome-wide graph builds (which can require tens of GB of RAM).
Instead they use a tiny but real gene-only snapshot spanning a bounded Ensembl-release window,
generated from targeted MySQL queries and cached on disk.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from idtrack._db import DB
from idtrack._the_graph import TheGraph


def _flatten_final_targets(conversion: dict) -> set[str]:
    """Return the union of all external IDs in `final_conversion.final_elements`."""
    targets: set[str] = set()
    for ensembl_gene in conversion:
        targets.update(conversion[ensembl_gene]["final_conversion"]["final_elements"].keys())
    return targets


def _active_gene_nodes(graph: TheGraph, *, release: int) -> list[str]:
    """Return Ensembl gene nodes active at `release`."""
    nodes: list[str] = []
    for n in graph.nodes:
        if graph.nodes[n].get(DB.node_type_str) != DB.nts_ensembl["gene"]:
            continue
        if graph.nodes[n].get("Version") in DB.alternative_versions:
            continue
        if TheGraph.is_point_in_range(graph.get_active_ranges_of_id[n], release):
            nodes.append(n)
    return nodes


def _stable_ids_active_at_release(graph: TheGraph, *, release: int) -> set[str]:
    return {graph.nodes[n]["ID"] for n in _active_gene_nodes(graph, release=release)}


def _pick_stable_id_active_in_both(graph: TheGraph, *, release_a: int, release_b: int) -> str:
    common = sorted(_stable_ids_active_at_release(graph, release=release_a) & _stable_ids_active_at_release(graph, release=release_b))
    if not common:
        pytest.skip(f"No stable IDs active in both releases {release_a} and {release_b}.")
    return common[0]


def _external_ids_by_database_for_gene(
    graph: TheGraph, *, gene_node: str, assembly: int, release: int
) -> dict[str, set[str]]:
    """Collect external identifiers grouped by database for `gene_node` at a given (assembly, release)."""
    by_db: dict[str, set[str]] = {}
    for ext in graph.predecessors(gene_node):
        if graph.nodes[ext].get(DB.node_type_str) != DB.nts_external:
            continue
        edge_data = graph.get_edge_data(ext, gene_node)
        if not edge_data or 0 not in edge_data:
            continue
        conn = edge_data[0].get(DB.connection_dict, {})
        for database, asm_map in conn.items():
            rels = asm_map.get(assembly, set())
            if release in rels:
                by_db.setdefault(database, set()).add(ext)
    return by_db


def _pick_gene_with_multiple_external_dbs(
    graph: TheGraph, *, release: int, assembly: int, min_dbs: int = 2
) -> tuple[str, dict[str, set[str]]]:
    for gene in _active_gene_nodes(graph, release=release):
        by_db = {
            db: ids
            for db, ids in _external_ids_by_database_for_gene(graph, gene_node=gene, assembly=assembly, release=release).items()
            if ids
        }
        if len(by_db) >= min_dbs:
            return gene, by_db
    pytest.skip(f"No gene node with >= {min_dbs} external databases found in the snapshot at release {release}.")


def _pick_any_external_node(graph: TheGraph, *, release: int) -> str:
    assembly = graph.graph["genome_assembly"]
    for ext in graph.nodes:
        if graph.nodes[ext].get(DB.node_type_str) != DB.nts_external:
            continue
        for gene in graph.neighbors(ext):
            edge = graph.get_edge_data(ext, gene)
            if not edge or 0 not in edge:
                continue
            conn = edge[0].get(DB.connection_dict, {})
            if any(release in conn[db].get(assembly, set()) for db in conn):
                return ext
    pytest.skip(f"No external node with an active mapping at release {release}.")


@pytest.mark.integration
class TestRealGraphSnapshot:
    def test_graph_shape_and_metadata(self, real_gene_graph, organism_under_test):
        graph, fixtures = real_gene_graph

        assert isinstance(graph, TheGraph)
        assert graph.graph["organism"] == organism_under_test
        assert graph.graph["ensembl_release"] == max(graph.graph["confident_for_release"])
        assert graph.graph["confident_for_release"] == sorted(graph.graph["confident_for_release"])
        assert graph.graph["genome_assembly"] in DB.all_assemblies

        assert fixtures.get("seed_gene_ids") is not None
        assert any(graph.nodes[n].get(DB.node_type_str) == DB.nts_ensembl["gene"] for n in graph.nodes)
        assert any(graph.nodes[n].get(DB.node_type_str) == DB.nts_base_ensembl["gene"] for n in graph.nodes)
        assert graph.available_external_databases

    def test_snapshot_is_cached_on_disk(self, real_gene_graph, real_graph_cache_root: Path, organism_under_test):
        graph, fixtures = real_gene_graph
        min_rel = min(graph.graph["confident_for_release"])
        max_rel = max(graph.graph["confident_for_release"])
        assembly = int(graph.graph["genome_assembly"])

        path = (
            real_graph_cache_root
            / f"graph_{organism_under_test}_asm{assembly}_min{min_rel}_max{max_rel}_gene_small.pickle"
        )
        assert path.exists()

        payload = pickle.loads(path.read_bytes())
        assert isinstance(payload, dict)
        assert "graph" in payload and "fixtures" in payload
        cached = payload["graph"]
        assert isinstance(cached, TheGraph)
        assert cached.number_of_nodes() == graph.number_of_nodes()
        assert payload["fixtures"].get("seed_gene_ids") == fixtures.get("seed_gene_ids")

    def test_graph_caches_compute(self, real_gene_graph):
        graph, _ = real_gene_graph
        graph.calculate_caches(for_test=True)

        assert isinstance(graph.combined_edges, dict)
        assert isinstance(graph.combined_edges_genes, dict)
        assert isinstance(graph.get_active_ranges_of_id, dict)
        assert isinstance(graph.node_trios, dict)

    def test_connection_edges_have_release_sets(self, real_gene_graph):
        graph, _ = real_gene_graph

        for _u, _v, _k, data in graph.edges(keys=True, data=True):
            if DB.connection_dict not in data:
                continue
            assert "available_releases" in data
            assert isinstance(data["available_releases"], set)
            assert all(isinstance(r, int) for r in data["available_releases"])


@pytest.mark.integration
class TestTimeTravelAndExternalDatabases:
    def test_time_travel_on_base_id(self, real_track, real_gene_graph):
        graph, _ = real_gene_graph
        releases = graph.graph["confident_for_release"]
        min_rel, max_rel = min(releases), max(releases)

        stable_id = _pick_stable_id_active_in_both(graph, release_a=min_rel, release_b=max_rel)

        for from_release, to_release in [(min_rel, max_rel), (max_rel, min_rel)]:
            converted = real_track.convert(
                from_id=stable_id,
                from_release=from_release,
                to_release=to_release,
                final_database=None,
                prioritize_to_one_filter=False,
            )
            assert converted is not None
            assert any(
                real_track.graph.nodes[target]["ID"] == stable_id
                and TheGraph.is_point_in_range(real_track.graph.get_active_ranges_of_id[target], to_release)
                for target in converted
            )

    def test_versioned_gene_node_time_travel(self, real_track, real_gene_graph):
        graph, _ = real_gene_graph
        releases = graph.graph["confident_for_release"]
        min_rel, max_rel = min(releases), max(releases)

        gene_node = sorted(_active_gene_nodes(graph, release=min_rel))[0]
        stable_id = graph.nodes[gene_node]["ID"]

        converted = real_track.convert(
            from_id=gene_node,
            from_release=min_rel,
            to_release=max_rel,
            final_database=None,
            prioritize_to_one_filter=False,
        )
        assert converted is not None
        assert any(
            real_track.graph.nodes[target]["ID"] == stable_id
            and TheGraph.is_point_in_range(real_track.graph.get_active_ranges_of_id[target], max_rel)
            for target in converted
        )

    def test_external_database_roundtrip(self, real_track, real_gene_graph):
        graph, _ = real_gene_graph
        release = int(graph.graph["ensembl_release"])
        assembly = int(graph.graph["genome_assembly"])

        _gene_node, by_db = _pick_gene_with_multiple_external_dbs(graph, release=release, assembly=assembly, min_dbs=2)
        db1, db2 = sorted(by_db)[:2]
        ext1 = sorted(by_db[db1])[0]
        ext2 = sorted(by_db[db2])[0]

        converted = real_track.convert(
            from_id=ext1,
            from_release=release,
            to_release=release,
            final_database=db2,
            prioritize_to_one_filter=False,
        )
        assert converted is not None
        assert ext2 in _flatten_final_targets(converted)

        converted_back = real_track.convert(
            from_id=ext2,
            from_release=release,
            to_release=release,
            final_database=db1,
            prioritize_to_one_filter=False,
        )
        assert converted_back is not None
        assert ext1 in _flatten_final_targets(converted_back)

    def test_stable_id_change_is_traversable(self, real_track, real_gene_graph):
        graph, fixtures = real_gene_graph

        old_node = fixtures.get("changed_old_node")
        new_node = fixtures.get("changed_new_node")
        old_rel = fixtures.get("changed_old_release")
        new_rel = fixtures.get("changed_new_release")

        if not (old_node and new_node and old_rel and new_rel):
            pytest.skip("No stable-id change event recorded in the snapshot.")

        converted = real_track.convert(
            from_id=old_node,
            from_release=int(old_rel),
            to_release=int(new_rel),
            final_database=None,
            prioritize_to_one_filter=False,
        )
        assert converted is not None
        assert new_node in converted

    def test_synonymous_nodes_finds_gene_from_external(self, real_track, real_gene_graph):
        graph, _ = real_gene_graph
        release = int(graph.graph["ensembl_release"])

        external_node = _pick_any_external_node(graph, release=release)
        paths = real_track.synonymous_nodes(
            the_id=external_node,
            depth_max=2,
            filter_node_type={DB.nts_ensembl["gene"]},
            from_release=release,
        )
        assert paths
        assert any(
            p[0][-1] in graph.nodes and graph.nodes[p[0][-1]][DB.node_type_str] == DB.nts_ensembl["gene"]
            for p in paths
        )

    def test_unknown_identifier_returns_none(self, real_track, real_gene_graph):
        graph, _ = real_gene_graph
        release = int(graph.graph["ensembl_release"])

        unknown = "THIS_IS_NOT_A_REAL_IDENTIFIER"
        resolved, converted = graph.node_name_alternatives(unknown)
        assert resolved is None
        assert converted is False

        assert (
            real_track.convert(
                from_id=unknown,
                from_release=release,
                to_release=release,
                final_database=None,
                prioritize_to_one_filter=False,
            )
            is None
        )


@pytest.mark.integration
class TestTrackTestsHarness:
    def test_invariants_hold_on_real_snapshot(self, real_track_tests, real_gene_graph):
        graph, _fixtures = real_gene_graph
        graph.calculate_caches(for_test=True)

        assert real_track_tests.is_id_functions_consistent_ensembl_2(verbose=False)
        assert real_track_tests.is_base_is_range_correct(verbose=False)
        assert real_track_tests.is_combined_edges_dicts_overlapping_and_complete()
        assert real_track_tests.is_edge_with_same_nts_only_at_backbone_nodes()


@pytest.mark.integration
class TestAPIIntegration:
    def test_api_convert_identifier_works_end_to_end(self, tmp_path, real_track, real_gene_graph):
        from idtrack._api import API

        graph, _ = real_gene_graph
        release = int(graph.graph["ensembl_release"])

        stable_id = _pick_stable_id_active_in_both(graph, release_a=release, release_b=release)

        api = API(str(tmp_path))
        api.track = real_track

        result = api.convert_identifier(
            stable_id,
            from_release=release,
            to_release=release,
            final_database=None,
            strategy="best",
            explain=True,
        )
        assert result["query_id"] == stable_id
        assert result["no_corresponding"] is False
        assert result["no_conversion"] is False
        assert result["graph_id"] == stable_id
        assert result["final_database"] == DB.nts_ensembl[DB.backbone_form]
        assert result["target_id"]
        assert result["the_path"]
