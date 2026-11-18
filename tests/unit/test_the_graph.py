#!/usr/bin/env python3
"""Unit tests of :py:class:`idtrack.graph.TheGraph` on graphs made by hand.

The construction of a graph, its reversed view, the range helpers, name lookup by letter case, the rules of
hyper-connectivity and the caches of an empty graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from idtrack import ids
from idtrack.graph import TheGraph
from idtrack.settings import GraphSettings


class TestTheGraphInitialization:
    """Test TheGraph initialization."""

    def test_can_create_empty_graph(self):
        """Test creating an empty TheGraph."""
        graph = TheGraph()
        assert isinstance(graph, nx.MultiDiGraph)
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0

    def test_available_forms_initially_none(self):
        """Test available_forms is None initially."""
        graph = TheGraph()
        assert graph.available_forms is None

    def test_can_initialize_with_data(self):
        """Test initializing with node/edge data."""
        source = nx.MultiDiGraph()
        source.add_node("A", type="test")
        source.add_node("B", type="test")
        source.add_edge("A", "B", weight=1)

        graph = TheGraph(source)
        assert "A" in graph.nodes()
        assert "B" in graph.nodes()
        assert graph.has_edge("A", "B")


class TestReverseGraph:
    """Test reverse graph property."""

    def test_rev_property_returns_reversed_graph(self):
        """Test rev property returns reversed graph."""
        graph = TheGraph()
        graph.add_node("A", **{ids.NODE_TYPE_KEY: "test"})
        graph.add_node("B", **{ids.NODE_TYPE_KEY: "test"})
        graph.add_edge("A", "B")

        rev = graph.rev
        assert rev.has_edge("B", "A")
        assert not rev.has_edge("A", "B")

    def test_rev_is_cached(self):
        """Test rev property is cached."""
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
        assert TheGraph.list_to_ranges([]) == []

    def test_single_element(self):
        """Test single element list."""
        assert TheGraph.list_to_ranges([5]) == [[5, 5]]

    def test_consecutive_elements(self):
        """Test consecutive elements form one range."""
        assert TheGraph.list_to_ranges([1, 2, 3, 4, 5]) == [[1, 5]]

    def test_non_consecutive_elements(self):
        """Test non-consecutive elements form multiple ranges."""
        assert TheGraph.list_to_ranges([1, 2, 3, 5, 6, 10]) == [[1, 3], [5, 6], [10, 10]]

    def test_gaps_in_sequence(self):
        """Test handling gaps in sequence."""
        assert TheGraph.list_to_ranges([100, 101, 105, 106, 107]) == [[100, 101], [105, 107]]


class TestCompactRanges:
    """Test compact_ranges static method."""

    def test_empty_ranges(self):
        """Test empty range list."""
        assert TheGraph.compact_ranges([]) == []

    def test_single_range(self):
        """Test single range unchanged."""
        assert TheGraph.compact_ranges([[1, 5]]) == [[1, 5]]

    def test_adjacent_ranges_merge(self):
        """Test adjacent ranges are merged."""
        assert TheGraph.compact_ranges([[1, 3], [4, 6]]) == [[1, 6]]

    def test_overlapping_ranges_merge(self):
        """Test overlapping ranges are merged."""
        assert TheGraph.compact_ranges([[1, 5], [3, 8]]) == [[1, 8]]

    def test_non_adjacent_ranges_preserved(self):
        """Test non-adjacent ranges are preserved."""
        assert TheGraph.compact_ranges([[1, 3], [5, 7]]) == [[1, 3], [5, 7]]


class TestGetIntersectingRanges:
    """Test get_intersecting_ranges static method."""

    def test_no_overlap(self):
        """Test non-overlapping ranges return empty."""
        assert TheGraph.get_intersecting_ranges([[1, 5]], [[10, 15]]) == []

    def test_full_overlap(self):
        """Test fully overlapping ranges."""
        assert TheGraph.get_intersecting_ranges([[1, 10]], [[3, 7]]) == [[3, 7]]

    def test_partial_overlap(self):
        """Test partially overlapping ranges."""
        assert TheGraph.get_intersecting_ranges([[1, 5]], [[3, 8]]) == [[3, 5]]

    def test_multiple_ranges(self):
        """Test multiple ranges with intersections."""
        assert TheGraph.get_intersecting_ranges([[1, 5], [10, 15]], [[3, 12]]) == [[3, 5], [10, 12]]

    def test_no_compact(self):
        """Test without compacting."""
        result = TheGraph.get_intersecting_ranges([[1, 5], [6, 10]], [[1, 10]], compact=False)
        assert len(result) == 2


class TestIsPointInRange:
    """Test is_point_in_range static method."""

    def test_point_in_single_range(self):
        """Test point within single range."""
        assert TheGraph.is_point_in_range([[1, 10]], 5) is True

    def test_point_at_range_start(self):
        """Test point at range start."""
        assert TheGraph.is_point_in_range([[5, 10]], 5) is True

    def test_point_at_range_end(self):
        """Test point at range end."""
        assert TheGraph.is_point_in_range([[5, 10]], 10) is True

    def test_point_outside_range(self):
        """Test point outside range."""
        assert TheGraph.is_point_in_range([[5, 10]], 3) is False
        assert TheGraph.is_point_in_range([[5, 10]], 15) is False

    def test_point_in_multiple_ranges(self):
        """Test point in one of multiple ranges."""
        lor = [[1, 5], [10, 15], [20, 25]]
        assert TheGraph.is_point_in_range(lor, 12) is True
        assert TheGraph.is_point_in_range(lor, 7) is False


class TestResolveName:
    """Test how a user's string is matched to a node of a graph made by hand."""

    def test_exact_match(self):
        """An exact node name is itself."""
        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("ENSG00000141510") == ("ENSG00000141510", "exact")

    def test_case_insensitive_match(self):
        """A name in another letter case is matched and reported as such."""
        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        graph.add_node("TP53", **{ids.NODE_TYPE_KEY: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("tp53") == ("TP53", "case")
        assert graph.resolve_name("ensg00000141510") == ("ENSG00000141510", "case")

    def test_not_found(self):
        """A string that is no node in any spelling matches nothing."""
        graph = TheGraph()
        graph.add_node("EXISTING", **{ids.NODE_TYPE_KEY: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("NONEXISTENT") == (None, None)

    def test_names_equal_in_lower_case_are_reported_by_the_store(self):
        """Two nodes that differ only in letter case are what a build refuses; the store names them."""
        graph = TheGraph()
        graph.add_node("TEST", **{ids.NODE_TYPE_KEY: "external"})
        graph.add_node("Test", **{ids.NODE_TYPE_KEY: "external"})

        assert graph.freeze()._frozen().lower_collisions() == [("TEST", "Test")]


class TestCombined:
    """Test the per-node summary of cross-reference edges."""

    def test_combined_with_mock_graph(self, mock_the_graph):
        """An external id and a base gene id are summed over the edges that leave them, a gene over those that enter."""
        assert mock_the_graph.combined("TP53") == {"HGNC Symbol": {38: {110}}}
        assert mock_the_graph.combined("ENSG00000141510") == {"base_ensembl_gene": {38: {70, 80, 110}}}
        assert set(mock_the_graph.combined("ENSG00000141510.15")) == {"assembly_38_ensembl_gene"}


class TestHyperconnectiveNodes:
    """Test the hyper-connective nodes and the count of genes an id is linked to at one time."""

    def test_empty_for_small_graph(self):
        """Test empty hyperconnective nodes for small graph."""
        graph = TheGraph()
        # Add a few nodes with low degree
        graph.add_node("A", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("B", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        graph.add_edge("A", "B")

        assert len(graph.hyperconnective_nodes) == 0

    def test_id_shared_by_many_genes_at_once_is_hyperconnective(self):
        """An external id linked to more genes than the threshold in one release is hyper-connective."""
        graph = TheGraph(assembly=38, confident_for_release=[100, 101])
        threshold = GraphSettings().hyperconnective_threshold
        for name, genes in (("FAMILY", threshold + 1), ("PAIR", threshold)):
            graph.add_node(name, **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
            for i in range(genes):
                target = f"{name}GENE{i}.1"
                graph.add_node(target, **{ids.NODE_TYPE_KEY: "ensembl_gene"})
                graph.add_edge(name, target, **{ids.CONNECTION_KEY: {"DB1": {38: {100}}}})

        assert graph.hyperconnective_nodes == {"FAMILY": threshold + 1}
        assert graph.genes_linked_at_once("PAIR") == threshold
        assert graph.shared_external_ids == {"FAMILY": threshold + 1, "PAIR": threshold}

    def test_threshold_is_read_when_the_nodes_are_asked_for(self):
        """Another threshold takes effect on a graph that exists already (the counts are kept, not the verdict)."""
        graph = TheGraph(assembly=38, confident_for_release=[100])
        graph.add_node("TWO", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        for i in range(2):
            graph.add_node(f"GENE{i}.1", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
            graph.add_edge("TWO", f"GENE{i}.1", **{ids.CONNECTION_KEY: {"DB1": {38: {100}}}})

        assert graph.hyperconnective_nodes == {}
        graph.graph_settings = GraphSettings(hyperconnective_threshold=1)
        assert graph.hyperconnective_nodes == {"TWO": 2}

    def test_versions_and_years_do_not_make_an_id_hyperconnective(self):
        """Many versions of one gene are one gene; genes of different releases are one at a time."""
        graph = TheGraph(assembly=38, confident_for_release=list(range(60, 100)))
        graph.add_node("SYMBOL", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("FAMILY", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("MOVING", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        for i in range(30):
            for target in (f"GENE.{i + 1}", f"MEMBER{i}.1", f"OTHER{i}.1"):
                graph.add_node(target, **{ids.NODE_TYPE_KEY: "ensembl_gene"})
            graph.add_edge("SYMBOL", f"GENE.{i + 1}", **{ids.CONNECTION_KEY: {"DB1": {38: {60 + i}}}})
            graph.add_edge("FAMILY", f"MEMBER{i}.1", **{ids.CONNECTION_KEY: {"DB1": {38: {70, 71}}}})
            graph.add_edge("MOVING", f"OTHER{i}.1", **{ids.CONNECTION_KEY: {"DB1": {38: {60 + i}}}})

        assert graph.out_degree["SYMBOL"] == graph.out_degree["MOVING"] == 30
        assert graph.genes_linked_at_once("SYMBOL") == 1
        assert graph.genes_linked_at_once("FAMILY") == 30
        assert graph.genes_linked_at_once("MOVING") == 1
        assert graph.hyperconnective_nodes == {"FAMILY": 30}

    def test_genes_linked_at_once_counts_a_transcript_as_its_gene(self):
        """An id linked to two transcripts of one gene links one gene."""
        graph = TheGraph(assembly=38, confident_for_release=[100])
        graph.add_node("NM_1", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("GENE.1", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        for transcript in ("T1.1", "T2.1"):
            graph.add_node(transcript, **{ids.NODE_TYPE_KEY: "ensembl_transcript"})
            graph.add_edge("NM_1", transcript, **{ids.CONNECTION_KEY: {"RefSeq": {38: {100}}}})
            graph.add_edge(transcript, "GENE.1", **{ids.CONNECTION_KEY: {"ensembl_transcript": {38: {100}}}})

        assert graph.genes_linked_at_once("NM_1") == 1


class TestCalculateCaches:
    """Test calculate_caches method."""

    def test_empty_graph_caches(self):
        """Test calculate_caches on empty graph."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 105, 110]

        # An empty graph has nothing to compute and nothing to find; afterwards it is frozen (cannot be changed).
        graph.calculate_caches()
        assert len(graph) == 0
        try:
            graph.add_node("X")
        except nx.NetworkXError:
            frozen = True
        else:
            frozen = False
        assert frozen
        assert graph.resolve_name("anything") == (None, None)
        assert graph.available_external_databases == set()


class TestRangesToList:
    """Test ranges_to_list method."""

    def test_single_range(self):
        """Test single range expansion."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 110]
        assert graph.ranges_to_list([[1, 5]]) == [1, 2, 3, 4, 5]

    def test_multiple_ranges(self):
        """Test multiple range expansion."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 110]
        assert graph.ranges_to_list([[1, 3], [5, 7]]) == [1, 2, 3, 5, 6, 7]

    def test_infinite_upper_bound(self):
        """Test infinite upper bound uses max confident release."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 105, 110]

        result = graph.ranges_to_list([[108, np.inf]])
        assert 108 in result
        assert 109 in result
        assert 110 in result
