#!/usr/bin/env python3
"""Unit tests for idtrack._the_graph module.

Tests TheGraph class which extends NetworkX MultiDiGraph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


class TestTheGraphInitialization:
    """Test TheGraph initialization."""

    def test_can_create_empty_graph(self):
        """Test creating an empty TheGraph."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        assert isinstance(graph, nx.MultiDiGraph)
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0

    def test_available_forms_initially_none(self):
        """Test available_forms is None initially."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        assert graph.available_forms is None

    def test_can_initialize_with_data(self):
        """Test initializing with node/edge data."""
        from idtrack._the_graph import TheGraph

        # Create with incoming graph
        source = nx.MultiDiGraph()
        source.add_node("A", type="test")
        source.add_node("B", type="test")
        source.add_edge("A", "B", weight=1)

        graph = TheGraph(source)
        assert "A" in graph.nodes()
        assert "B" in graph.nodes()
        assert graph.has_edge("A", "B")


class TestAttachIncludedForms:
    """Test _attach_included_forms method."""

    def test_attach_forms(self):
        """Test attaching available forms."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        forms = ["gene", "transcript", "translation"]
        graph._attach_included_forms(forms)
        assert graph.available_forms == forms

    def test_attach_partial_forms(self):
        """Test attaching partial form list."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        forms = ["gene"]
        graph._attach_included_forms(forms)
        assert graph.available_forms == ["gene"]


class TestReverseGraph:
    """Test reverse graph property."""

    def test_rev_property_returns_reversed_graph(self):
        """Test rev property returns reversed graph."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("A", **{DB.node_type_str: "test"})
        graph.add_node("B", **{DB.node_type_str: "test"})
        graph.add_edge("A", "B")

        rev = graph.rev
        assert rev.has_edge("B", "A")
        assert not rev.has_edge("A", "B")

    def test_rev_is_cached(self):
        """Test rev property is cached."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_edge("A", "B")

        rev1 = graph.rev
        rev2 = graph.rev
        # Same object due to caching
        assert rev1 is rev2


class TestListToRanges:
    """Test list_to_ranges static method."""

    def test_empty_list(self):
        """Test empty list returns empty ranges."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.list_to_ranges([])
        assert result == []

    def test_single_element(self):
        """Test single element list."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.list_to_ranges([5])
        assert result == [[5, 5]]

    def test_consecutive_elements(self):
        """Test consecutive elements form one range."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.list_to_ranges([1, 2, 3, 4, 5])
        assert result == [[1, 5]]

    def test_non_consecutive_elements(self):
        """Test non-consecutive elements form multiple ranges."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.list_to_ranges([1, 2, 3, 5, 6, 10])
        assert result == [[1, 3], [5, 6], [10, 10]]

    def test_gaps_in_sequence(self):
        """Test handling gaps in sequence."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.list_to_ranges([100, 101, 105, 106, 107])
        assert result == [[100, 101], [105, 107]]


class TestCompactRanges:
    """Test compact_ranges static method."""

    def test_empty_ranges(self):
        """Test empty range list."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.compact_ranges([])
        assert result == []

    def test_single_range(self):
        """Test single range unchanged."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.compact_ranges([[1, 5]])
        assert result == [[1, 5]]

    def test_adjacent_ranges_merge(self):
        """Test adjacent ranges are merged."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.compact_ranges([[1, 3], [4, 6]])
        assert result == [[1, 6]]

    def test_overlapping_ranges_merge(self):
        """Test overlapping ranges are merged."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.compact_ranges([[1, 5], [3, 8]])
        assert result == [[1, 8]]

    def test_non_adjacent_ranges_preserved(self):
        """Test non-adjacent ranges are preserved."""
        from idtrack._the_graph import TheGraph

        result = TheGraph.compact_ranges([[1, 3], [5, 7]])
        assert result == [[1, 3], [5, 7]]


class TestGetIntersectingRanges:
    """Test get_intersecting_ranges static method."""

    def test_no_overlap(self):
        """Test non-overlapping ranges return empty."""
        from idtrack._the_graph import TheGraph

        lor1 = [[1, 5]]
        lor2 = [[10, 15]]
        result = TheGraph.get_intersecting_ranges(lor1, lor2)
        assert result == []

    def test_full_overlap(self):
        """Test fully overlapping ranges."""
        from idtrack._the_graph import TheGraph

        lor1 = [[1, 10]]
        lor2 = [[3, 7]]
        result = TheGraph.get_intersecting_ranges(lor1, lor2)
        assert result == [[3, 7]]

    def test_partial_overlap(self):
        """Test partially overlapping ranges."""
        from idtrack._the_graph import TheGraph

        lor1 = [[1, 5]]
        lor2 = [[3, 8]]
        result = TheGraph.get_intersecting_ranges(lor1, lor2)
        assert result == [[3, 5]]

    def test_multiple_ranges(self):
        """Test multiple ranges with intersections."""
        from idtrack._the_graph import TheGraph

        lor1 = [[1, 5], [10, 15]]
        lor2 = [[3, 12]]
        result = TheGraph.get_intersecting_ranges(lor1, lor2)
        assert result == [[3, 5], [10, 12]]

    def test_no_compact(self):
        """Test without compacting."""
        from idtrack._the_graph import TheGraph

        lor1 = [[1, 5], [6, 10]]
        lor2 = [[1, 10]]
        result = TheGraph.get_intersecting_ranges(lor1, lor2, compact=False)
        assert len(result) == 2


class TestIsPointInRange:
    """Test is_point_in_range static method."""

    def test_point_in_single_range(self):
        """Test point within single range."""
        from idtrack._the_graph import TheGraph

        lor = [[1, 10]]
        assert TheGraph.is_point_in_range(lor, 5) is True

    def test_point_at_range_start(self):
        """Test point at range start."""
        from idtrack._the_graph import TheGraph

        lor = [[5, 10]]
        assert TheGraph.is_point_in_range(lor, 5) is True

    def test_point_at_range_end(self):
        """Test point at range end."""
        from idtrack._the_graph import TheGraph

        lor = [[5, 10]]
        assert TheGraph.is_point_in_range(lor, 10) is True

    def test_point_outside_range(self):
        """Test point outside range."""
        from idtrack._the_graph import TheGraph

        lor = [[5, 10]]
        assert TheGraph.is_point_in_range(lor, 3) is False
        assert TheGraph.is_point_in_range(lor, 15) is False

    def test_point_in_multiple_ranges(self):
        """Test point in one of multiple ranges."""
        from idtrack._the_graph import TheGraph

        lor = [[1, 5], [10, 15], [20, 25]]
        assert TheGraph.is_point_in_range(lor, 12) is True
        assert TheGraph.is_point_in_range(lor, 7) is False


class TestResolveName:
    """Test how a user's string is matched to a node of a graph made by hand."""

    def test_exact_match(self):
        """An exact node name is itself."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{DB.node_type_str: "ensembl_gene"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("ENSG00000141510") == ("ENSG00000141510", "exact")

    def test_case_insensitive_match(self):
        """A name in another letter case is matched and reported as such."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{DB.node_type_str: "ensembl_gene"})
        graph.add_node("TP53", **{DB.node_type_str: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("tp53") == ("TP53", "case")
        assert graph.resolve_name("ensg00000141510") == ("ENSG00000141510", "case")

    def test_not_found(self):
        """A string that is no node in any spelling matches nothing."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("EXISTING", **{DB.node_type_str: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("NONEXISTENT") == (None, None)

    def test_names_equal_in_lower_case_are_reported_by_the_store(self):
        """Two nodes that differ only in letter case are what a build refuses; the store names them."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("TEST", **{DB.node_type_str: "external"})
        graph.add_node("Test", **{DB.node_type_str: "external"})

        assert graph.freeze()._frozen().lower_collisions() == [("TEST", "Test")]


class TestCombined:
    """Test the per-node summary of cross-reference edges."""

    def test_combined_with_mock_graph(self, mock_the_graph):
        """An external id and a base gene id are summed over the edges that leave them, a gene over those that enter."""
        assert mock_the_graph.combined("TP53") == {"HGNC Symbol": {38: {110}}}
        assert mock_the_graph.combined("ENSG00000141510") == {"base_ensembl_gene": {38: {70, 80, 110}}}
        assert set(mock_the_graph.combined("ENSG00000141510.15")) == {"assembly_38_ensembl_gene"}


class TestHyperconnectiveNodes:
    """Test hyperconnective_nodes cached property."""

    def test_empty_for_small_graph(self):
        """Test empty hyperconnective nodes for small graph."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        # Add a few nodes with low degree
        graph.add_node("A", **{DB.node_type_str: DB.nts_external})
        graph.add_node("B", **{DB.node_type_str: "ensembl_gene"})
        graph.add_edge("A", "B")

        hcn = graph.hyperconnective_nodes
        assert len(hcn) == 0

    def test_id_shared_by_many_genes_at_once_is_hyperconnective(self):
        """An external id linked to more genes than the threshold in one release is hyper-connective."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph(genome_assembly=38, confident_for_release=[100, 101])
        threshold = TheGraph.genes_at_once_threshold
        for name, genes in (("FAMILY", threshold + 1), ("PAIR", threshold)):
            graph.add_node(name, **{DB.node_type_str: DB.nts_external})
            for i in range(genes):
                target = f"{name}GENE{i}.1"
                graph.add_node(target, **{DB.node_type_str: "ensembl_gene"})
                graph.add_edge(name, target, **{DB.connection_dict: {"DB1": {38: {100}}}})

        assert graph.hyperconnective_nodes == {"FAMILY": threshold + 1}
        assert graph.genes_linked_at_once("PAIR") == threshold
        assert graph.shared_external_ids == {"FAMILY": threshold + 1, "PAIR": threshold}

    def test_threshold_is_read_when_the_nodes_are_asked_for(self, monkeypatch):
        """Another threshold takes effect on a graph that exists already (the counts are kept, not the verdict)."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph(genome_assembly=38, confident_for_release=[100])
        graph.add_node("TWO", **{DB.node_type_str: DB.nts_external})
        for i in range(2):
            graph.add_node(f"GENE{i}.1", **{DB.node_type_str: "ensembl_gene"})
            graph.add_edge("TWO", f"GENE{i}.1", **{DB.connection_dict: {"DB1": {38: {100}}}})

        assert graph.hyperconnective_nodes == {}
        monkeypatch.setattr(TheGraph, "genes_at_once_threshold", 1)
        assert graph.hyperconnective_nodes == {"TWO": 2}

    def test_versions_and_years_do_not_make_an_id_hyperconnective(self):
        """Many versions of one gene are one gene; genes of different releases are one at a time."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph(genome_assembly=38, confident_for_release=list(range(60, 100)))
        graph.add_node("SYMBOL", **{DB.node_type_str: DB.nts_external})
        graph.add_node("FAMILY", **{DB.node_type_str: DB.nts_external})
        graph.add_node("MOVING", **{DB.node_type_str: DB.nts_external})
        for i in range(30):
            for target in (f"GENE.{i + 1}", f"MEMBER{i}.1", f"OTHER{i}.1"):
                graph.add_node(target, **{DB.node_type_str: "ensembl_gene"})
            graph.add_edge("SYMBOL", f"GENE.{i + 1}", **{DB.connection_dict: {"DB1": {38: {60 + i}}}})
            graph.add_edge("FAMILY", f"MEMBER{i}.1", **{DB.connection_dict: {"DB1": {38: {70, 71}}}})
            graph.add_edge("MOVING", f"OTHER{i}.1", **{DB.connection_dict: {"DB1": {38: {60 + i}}}})

        assert graph.out_degree["SYMBOL"] == graph.out_degree["MOVING"] == 30
        assert graph.genes_linked_at_once("SYMBOL") == 1
        assert graph.genes_linked_at_once("FAMILY") == 30
        assert graph.genes_linked_at_once("MOVING") == 1
        assert graph.hyperconnective_nodes == {"FAMILY": 30}

    def test_genes_linked_at_once_counts_a_transcript_as_its_gene(self):
        """An id linked to two transcripts of one gene links one gene."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph(genome_assembly=38, confident_for_release=[100])
        graph.add_node("NM_1", **{DB.node_type_str: DB.nts_external})
        graph.add_node("GENE.1", **{DB.node_type_str: "ensembl_gene"})
        for transcript in ("T1.1", "T2.1"):
            graph.add_node(transcript, **{DB.node_type_str: "ensembl_transcript"})
            graph.add_edge("NM_1", transcript, **{DB.connection_dict: {"RefSeq": {38: {100}}}})
            graph.add_edge(transcript, "GENE.1", **{DB.connection_dict: {"ensembl_transcript": {38: {100}}}})

        assert graph.genes_linked_at_once("NM_1") == 1


class TestCalculateCaches:
    """Test calculate_caches method."""

    def test_empty_graph_caches(self):
        """Test calculate_caches on empty graph."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 105, 110]

        # An empty graph has nothing to compute and nothing to find
        graph.calculate_caches(for_test=False)
        assert len(graph) == 0 and graph.is_frozen
        assert graph.resolve_name("anything") == (None, None)
        assert graph.available_external_databases == set()


class TestRangesToList:
    """Test ranges_to_list method."""

    def test_single_range(self):
        """Test single range expansion."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 110]

        result = graph.ranges_to_list([[1, 5]])
        assert result == [1, 2, 3, 4, 5]

    def test_multiple_ranges(self):
        """Test multiple range expansion."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 110]

        result = graph.ranges_to_list([[1, 3], [5, 7]])
        assert result == [1, 2, 3, 5, 6, 7]

    def test_infinite_upper_bound(self):
        """Test infinite upper bound uses max confident release."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 105, 110]

        result = graph.ranges_to_list([[108, np.inf]])
        assert 108 in result
        assert 109 in result
        assert 110 in result
