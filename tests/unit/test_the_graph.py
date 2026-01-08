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

    def test_inherits_from_multidigraph(self):
        """Test TheGraph inherits from MultiDiGraph."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        assert isinstance(graph, nx.MultiDiGraph)

    def test_has_logger(self):
        """Test TheGraph has a logger."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        assert hasattr(graph, "log")
        assert graph.log is not None

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


class TestNodeNameAlternatives:
    """Test node_name_alternatives method."""

    def test_exact_match(self):
        """Test exact match returns node."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{DB.node_type_str: "ensembl_gene"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        # Need to build lower_chars_graph cache
        _ = graph.lower_chars_graph

        result, was_converted = graph.node_name_alternatives("ENSG00000141510")
        assert result == "ENSG00000141510"
        assert was_converted is False

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("TP53", **{DB.node_type_str: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        _ = graph.lower_chars_graph

        result, was_converted = graph.node_name_alternatives("tp53")
        assert result == "TP53"
        assert was_converted is True

    def test_not_found(self):
        """Test node not found returns None."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("EXISTING", **{DB.node_type_str: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        _ = graph.lower_chars_graph

        result, was_converted = graph.node_name_alternatives("NONEXISTENT")
        assert result is None
        assert was_converted is False


class TestLowerCharsGraph:
    """Test lower_chars_graph cached property."""

    def test_builds_lowercase_mapping(self):
        """Test lowercase mapping is built correctly."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{DB.node_type_str: "ensembl_gene"})
        graph.add_node("TP53", **{DB.node_type_str: "external"})

        lower_map = graph.lower_chars_graph

        assert "ensg00000141510" in lower_map
        assert "tp53" in lower_map
        assert lower_map["ensg00000141510"] == "ENSG00000141510"
        assert lower_map["tp53"] == "TP53"

    def test_caching(self):
        """Test lower_chars_graph is cached."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.add_node("TEST", **{DB.node_type_str: "external"})

        map1 = graph.lower_chars_graph
        map2 = graph.lower_chars_graph

        assert map1 is map2


class TestCombinedEdges:
    """Test combined_edges cached property."""

    def test_combined_edges_with_mock_graph(self, mock_the_graph):
        """Test combined_edges property with mock graph."""
        # The mock_the_graph fixture creates a graph with edges
        # Test that combined_edges runs without error
        combined = mock_the_graph.combined_edges
        assert isinstance(combined, dict)


class TestGraphMetadata:
    """Test graph metadata handling."""

    def test_graph_attributes(self, mock_the_graph):
        """Test graph-level attributes."""
        assert "genome_assembly" in mock_the_graph.graph
        assert "confident_for_release" in mock_the_graph.graph
        assert mock_the_graph.graph["genome_assembly"] == 38

    def test_confident_for_release_is_list(self, mock_the_graph):
        """Test confident_for_release is a list."""
        releases = mock_the_graph.graph["confident_for_release"]
        assert isinstance(releases, list)
        assert all(isinstance(r, int) for r in releases)


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

    def test_identifies_high_degree_nodes(self):
        """Test high-degree nodes are identified."""
        from idtrack._db import DB
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        # Add a hyperconnective external node
        hub = "HUB_NODE"
        graph.add_node(hub, **{DB.node_type_str: DB.nts_external})

        # Add many outgoing edges (more than threshold)
        for i in range(DB.hyperconnecting_threshold + 10):
            target = f"TARGET_{i}"
            graph.add_node(target, **{DB.node_type_str: "ensembl_gene"})
            graph.add_edge(hub, target)

        hcn = graph.hyperconnective_nodes
        assert hub in hcn
        assert hcn[hub] > DB.hyperconnecting_threshold


class TestGetExternalDatabaseNodes:
    """Test get_external_database_nodes method."""

    def test_method_exists(self):
        """Test get_external_database_nodes method exists."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        assert hasattr(graph, "get_external_database_nodes")
        assert callable(graph.get_external_database_nodes)

    def test_returns_set(self):
        """Test method returns a set for valid graph."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        # Empty graph should return empty set
        result = graph.get_external_database_nodes("HGNC Symbol")
        assert isinstance(result, set)


class TestCalculateCaches:
    """Test calculate_caches method."""

    def test_method_exists(self):
        """Test calculate_caches method exists."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        assert hasattr(graph, "calculate_caches")
        assert callable(graph.calculate_caches)

    def test_empty_graph_caches(self):
        """Test calculate_caches on empty graph."""
        from idtrack._the_graph import TheGraph

        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 105, 110]

        # Empty graph should have empty caches
        assert graph.combined_edges == {}
        assert len(graph.lower_chars_graph) == 0


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


class TestGetTwoNodesCoincidingReleases:
    """Test get_two_nodes_coinciding_releases method."""

    def test_overlapping_releases(self, mock_the_graph):
        """Test finding overlapping releases between two nodes."""
        # Add test nodes with known ranges
        # This requires proper setup of get_active_ranges_of_id

        # For now, just verify the method exists and is callable
        assert hasattr(mock_the_graph, "get_two_nodes_coinciding_releases")
        assert callable(mock_the_graph.get_two_nodes_coinciding_releases)
