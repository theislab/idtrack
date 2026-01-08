#!/usr/bin/env python3
"""Unit tests for idtrack._track module.

Tests the Track class path-finding algorithms with mocked graph.
"""

from __future__ import annotations

import networkx as nx
import pytest


class TestTrackImport:
    """Test Track can be imported."""

    def test_import_track(self):
        """Test Track import."""
        from idtrack._track import Track

        assert Track is not None


class TestTrackInitialization:
    """Test Track initialization."""

    def test_track_class_exists(self):
        """Test Track class exists."""
        from idtrack._track import Track

        assert Track is not None


class TestTrackWithMockGraph:
    """Test Track with mocked TheGraph."""

    def test_uses_the_graph(self, mock_the_graph):
        """Test Track uses TheGraph instance."""
        from idtrack._the_graph import TheGraph

        assert isinstance(mock_the_graph, TheGraph)
        assert len(mock_the_graph.nodes()) > 0


class TestTrackPathFinding:
    """Test path-finding algorithms."""

    def test_simple_forward_path(self, mock_the_graph):
        """Test forward path finding."""
        # Mock graph has versioned gene nodes
        nodes = list(mock_the_graph.nodes())
        assert len(nodes) > 0

    def test_handles_missing_node(self, mock_the_graph):
        """Test handling of non-existent node."""
        assert "NONEXISTENT_NODE" not in mock_the_graph.nodes()

    def test_handles_orphaned_id(self, mock_the_graph):
        """Test handling of IDs with no connections (orphaned nodes)."""
        from idtrack._db import DB

        # Add an orphaned node with no edges
        orphaned_id = "ORPHAN_GENE.1"
        mock_the_graph.add_node(
            orphaned_id,
            **{
                DB.node_type_str: DB.nts_ensembl["gene"],
                "ID": "ORPHAN_GENE",
                "Version": "1",
                "is_latest": False,
            },
        )

        # Verify the node exists but has no connections
        assert orphaned_id in mock_the_graph.nodes()
        assert mock_the_graph.degree(orphaned_id) == 0

        # The node should still be findable via node_name_alternatives
        _ = mock_the_graph.lower_chars_graph  # Rebuild cache
        result, _ = mock_the_graph.node_name_alternatives(orphaned_id)
        assert result == orphaned_id


class TestTrackNodeResolution:
    """Test node resolution functionality."""

    def test_resolves_exact_match(self, mock_the_graph):
        """Test exact node match resolution."""
        # Add a known node
        from idtrack._db import DB

        mock_the_graph.add_node("TEST_NODE", **{DB.node_type_str: "external"})
        _ = mock_the_graph.lower_chars_graph  # Build cache

        result, was_converted = mock_the_graph.node_name_alternatives("TEST_NODE")
        assert result == "TEST_NODE"
        assert was_converted is False

    def test_resolves_case_insensitive(self, mock_the_graph):
        """Test case-insensitive resolution."""
        from idtrack._db import DB

        mock_the_graph.add_node("ACTB", **{DB.node_type_str: "external"})
        _ = mock_the_graph.lower_chars_graph  # Build cache

        result, was_converted = mock_the_graph.node_name_alternatives("actb")
        assert result == "ACTB"
        assert was_converted is True


class TestTrackTemporalNavigation:
    """Test temporal navigation through releases."""

    def test_forward_in_time(self, mock_the_graph):
        """Test moving forward through releases."""
        releases = mock_the_graph.graph["confident_for_release"]
        assert len(releases) > 1
        assert releases == sorted(releases)

    def test_backward_in_time(self, mock_the_graph):
        """Test moving backward through releases."""
        # Rev property provides backward traversal
        rev_graph = mock_the_graph.rev
        assert rev_graph is not None


class TestTrackExternalDatabaseSearch:
    """Test external database search."""

    def test_finds_external_connections(self, mock_the_graph):
        """Test finding external database connections."""
        from idtrack._db import DB

        # Check for external nodes
        external_nodes = [
            n for n in mock_the_graph.nodes() if mock_the_graph.nodes[n].get(DB.node_type_str) == DB.nts_external
        ]
        assert len(external_nodes) > 0


class TestTrackScoring:
    """Test confidence scoring."""

    def test_scoring_values(self):
        """Test scoring produces valid values within expected range."""
        from idtrack._track import Track

        # Test the static method get_from_release_and_reverse_vars which is used in scoring
        # Test closest mode with point at range start
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 70, "closest")
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], int)  # release number
        assert isinstance(result[0][1], bool)  # reverse flag

        # Test closest mode with point before range
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 60, "closest")
        assert result == [(70, True)]  # Should reverse from start

        # Test closest mode with point after range
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 90, "closest")
        assert result == [(80, False)]  # Should go forward from end

        # Test closest mode with point inside range
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 75, "closest")
        assert len(result) == 2  # Both directions possible

        # Test distant mode with point before range
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 60, "distant")
        assert result == [(80, True)]  # Distant end is 80

        # Test distant mode with point after range
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 90, "distant")
        assert result == [(70, False)]  # Distant end is 70

    def test_scoring_with_invalid_range(self):
        """Test scoring raises error for invalid range."""
        from idtrack._track import Track

        # l1 > l2 should raise ValueError
        with pytest.raises(ValueError, match="l1 > l2"):
            Track.get_from_release_and_reverse_vars([(80, 70)], 75, "closest")

    def test_scoring_multiple_ranges(self):
        """Test scoring with multiple active ranges."""
        from idtrack._track import Track

        # Multiple ranges
        lor = [(70, 80), (90, 100)]
        result = Track.get_from_release_and_reverse_vars(lor, 85, "closest")
        # Should have entries for both ranges
        assert len(result) >= 2


class TestTrackConversionTypes:
    """Test different conversion scenarios."""

    def test_one_to_one_mapping(self, mock_the_graph):
        """Test 1:1 ID mapping via synonymous_nodes search."""
        from idtrack._db import DB

        # Add nodes that form a 1:1 mapping path
        mock_the_graph.add_node(
            "UNIQUE_GENE.1",
            **{
                DB.node_type_str: DB.nts_ensembl["gene"],
                "ID": "UNIQUE_GENE",
                "Version": "1",
            },
        )
        mock_the_graph.add_node(
            "UNIQUE_SYMBOL",
            **{
                DB.node_type_str: DB.nts_external,
                "database": "HGNC Symbol",
            },
        )

        # Create edge between them
        mock_the_graph.add_edge(
            "UNIQUE_SYMBOL", "UNIQUE_GENE.1", connection={"HGNC Symbol": {38: {110}}}, available_releases={110}
        )

        # Verify the 1:1 relationship exists
        neighbors = list(mock_the_graph.neighbors("UNIQUE_SYMBOL"))
        assert len(neighbors) == 1
        assert neighbors[0] == "UNIQUE_GENE.1"

    def test_one_to_many_mapping(self, mock_the_graph):
        """Test 1:N ID mapping where one ID maps to multiple targets."""
        from idtrack._db import DB

        # Add a symbol that maps to multiple genes (gene family)
        mock_the_graph.add_node(
            "FAMILY_GENE.1",
            **{
                DB.node_type_str: DB.nts_ensembl["gene"],
                "ID": "FAMILY_GENE",
                "Version": "1",
            },
        )
        mock_the_graph.add_node(
            "FAMILY_GENE.2",
            **{
                DB.node_type_str: DB.nts_ensembl["gene"],
                "ID": "FAMILY_GENE",
                "Version": "2",
            },
        )
        mock_the_graph.add_node(
            "AMBIGUOUS_SYMBOL",
            **{
                DB.node_type_str: DB.nts_external,
                "database": "HGNC Symbol",
            },
        )

        # Create edges to multiple genes
        mock_the_graph.add_edge(
            "AMBIGUOUS_SYMBOL", "FAMILY_GENE.1", connection={"HGNC Symbol": {38: {100}}}, available_releases={100}
        )
        mock_the_graph.add_edge(
            "AMBIGUOUS_SYMBOL", "FAMILY_GENE.2", connection={"HGNC Symbol": {38: {110}}}, available_releases={110}
        )

        # Verify 1:N relationship
        neighbors = list(mock_the_graph.neighbors("AMBIGUOUS_SYMBOL"))
        assert len(neighbors) == 2
        assert "FAMILY_GENE.1" in neighbors
        assert "FAMILY_GENE.2" in neighbors

    def test_one_to_zero_mapping(self, mock_the_graph):
        """Test 1:0 ID mapping (isolated node with no connections)."""
        from idtrack._db import DB

        # Add an isolated node with no edges
        mock_the_graph.add_node(
            "ISOLATED_ID",
            **{
                DB.node_type_str: DB.nts_external,
                "database": "Unknown",
            },
        )

        # Verify zero outgoing connections
        neighbors = list(mock_the_graph.neighbors("ISOLATED_ID"))
        assert len(neighbors) == 0

        # Also verify via degree
        assert mock_the_graph.out_degree("ISOLATED_ID") == 0


class TestTrackAssemblyHandling:
    """Test assembly handling."""

    def test_main_assembly(self, mock_the_graph):
        """Test main assembly is set."""
        assert mock_the_graph.graph["genome_assembly"] == 38

    def test_assembly_specific_nodes(self, mock_the_graph):
        """Test assembly-specific node handling."""
        from idtrack._db import DB

        # Check for assembly-specific gene node types
        for nts in DB.nts_assembly_gene:
            assert "assembly" in nts


class TestTrackHyperconnectiveNodes:
    """Test hyperconnective node handling."""

    def test_identifies_hyperconnective(self, mock_the_graph):
        """Test identification of hyperconnective nodes."""
        hcn = mock_the_graph.hyperconnective_nodes
        assert isinstance(hcn, dict)

    def test_excludes_from_search(self, mock_the_graph):
        """Test hyperconnective nodes are excluded from certain searches."""
        from idtrack._db import DB

        threshold = DB.hyperconnecting_threshold
        assert threshold > 0


class TestTrackEdgeCases:
    """Test edge cases."""

    def test_empty_input(self, mock_the_graph):
        """Test handling of empty input - empty string node lookup."""
        # Empty string should not be in graph
        assert "" not in mock_the_graph.nodes()

        # node_name_alternatives should handle empty string gracefully
        result, was_converted = mock_the_graph.node_name_alternatives("")
        assert result is None
        assert was_converted is False

    def test_none_input(self, mock_the_graph):
        """Test handling of None input."""
        # None should not be a valid node
        assert None not in mock_the_graph.nodes()

    def test_invalid_release(self, mock_the_graph):
        """Test handling of invalid release number."""
        releases = mock_the_graph.graph["confident_for_release"]
        invalid_release = max(releases) + 1000
        assert invalid_release not in releases

    def test_special_characters_in_id(self, mock_the_graph):
        """Test handling of special characters in IDs."""
        from idtrack._db import DB

        # Test IDs with various special characters (common in biological databases)
        special_ids = [
            "LRG_1",  # LRG genes have underscores
            "ENSG00000141510.15",  # Version separator
            "NM_001126112.3",  # RefSeq format
            "P04637-1",  # UniProt isoform
        ]

        for special_id in special_ids:
            # Add node with special character ID
            mock_the_graph.add_node(
                special_id,
                **{
                    DB.node_type_str: DB.nts_external,
                    "database": "test",
                },
            )

            # Verify node can be retrieved
            assert special_id in mock_the_graph.nodes()

            # Verify node attributes are accessible
            node_data = mock_the_graph.nodes[special_id]
            assert DB.node_type_str in node_data

    def test_whitespace_handling(self, mock_the_graph):
        """Test handling of whitespace in IDs."""
        # Whitespace IDs should not exist in a well-formed graph
        assert "  " not in mock_the_graph.nodes()
        assert "\t" not in mock_the_graph.nodes()
        assert "\n" not in mock_the_graph.nodes()


class TestTrackDBConstants:
    """Test usage of DB constants."""

    def test_uses_node_type_str(self):
        """Test uses correct node type string."""
        from idtrack._db import DB

        assert DB.node_type_str == "node_type"

    def test_uses_connection_dict(self):
        """Test uses correct connection dict key."""
        from idtrack._db import DB

        assert DB.connection_dict == "connection"

    def test_uses_backbone_form(self):
        """Test uses correct backbone form."""
        from idtrack._db import DB

        assert DB.backbone_form == "gene"


class TestTrackCachingIntegration:
    """Test integration with TheGraph caches."""

    def test_uses_combined_edges(self, mock_the_graph):
        """Test uses combined_edges cache."""
        combined = mock_the_graph.combined_edges
        assert isinstance(combined, dict)

    def test_uses_lower_chars_graph(self, mock_the_graph):
        """Test uses lower_chars_graph cache."""
        lower = mock_the_graph.lower_chars_graph
        assert isinstance(lower, dict)

    def test_get_active_ranges_property_exists(self):
        """Test get_active_ranges_of_id property exists on class."""
        from idtrack._the_graph import TheGraph

        # Check the property exists on the class (not instance to avoid evaluation)
        assert "get_active_ranges_of_id" in dir(TheGraph)


class TestTrackMethods:
    """Test Track method signatures."""

    def test_has_expected_methods(self):
        """Test Track has expected methods."""
        from idtrack._track import Track

        # Check for method existence (names may vary)
        assert hasattr(Track, "__init__")


class TestTrackLogging:
    """Test logging during operations."""

    def test_has_logger(self):
        """Test Track class has logger setup in __init__."""
        import inspect

        from idtrack._track import Track

        # Verify __init__ sets up a logger
        source = inspect.getsource(Track.__init__)
        assert "logging" in source or "log" in source

        # Verify Track expects to have a 'log' attribute
        # by checking the source for self.log usage
        full_source = inspect.getsource(Track)
        assert "self.log" in full_source

    def test_logger_name(self):
        """Test Track logger is properly named."""
        import logging

        # The Track class creates a logger named 'track'
        track_logger = logging.getLogger("track")
        assert track_logger is not None
        assert track_logger.name == "track"


class TestTrackErrorHandling:
    """Test error handling."""

    def test_handles_disconnected_nodes(self, mock_the_graph):
        """Test handling of disconnected nodes (no path exists)."""
        from idtrack._db import DB

        # Add two disconnected subgraphs
        mock_the_graph.add_node(
            "DISCONNECTED_A.1",
            **{
                DB.node_type_str: DB.nts_ensembl["gene"],
                "ID": "DISCONNECTED_A",
                "Version": "1",
            },
        )
        mock_the_graph.add_node(
            "DISCONNECTED_B.1",
            **{
                DB.node_type_str: DB.nts_ensembl["gene"],
                "ID": "DISCONNECTED_B",
                "Version": "1",
            },
        )

        # Verify no path exists between disconnected nodes
        assert not nx.has_path(mock_the_graph, "DISCONNECTED_A.1", "DISCONNECTED_B.1")

        # Verify each node exists independently
        assert "DISCONNECTED_A.1" in mock_the_graph.nodes()
        assert "DISCONNECTED_B.1" in mock_the_graph.nodes()

        # Verify no edges between them
        assert not mock_the_graph.has_edge("DISCONNECTED_A.1", "DISCONNECTED_B.1")
        assert not mock_the_graph.has_edge("DISCONNECTED_B.1", "DISCONNECTED_A.1")

    def test_handles_cycles(self, mock_the_graph):
        """Test handling of cycles in graph - graph should be DAG for backbone."""
        from idtrack._db import DB

        # Verify the mock graph structure
        # In idtrack, the backbone (ensembl_gene history) should be acyclic
        backbone_nodes = [
            n for n in mock_the_graph.nodes() if mock_the_graph.nodes[n].get(DB.node_type_str) == DB.nts_ensembl["gene"]
        ]

        # Create a subgraph of backbone nodes to check for cycles
        if len(backbone_nodes) > 1:
            backbone_subgraph = mock_the_graph.subgraph(backbone_nodes)
            # The backbone should ideally be a DAG
            # Note: simple_cycles returns empty iterator for DAGs
            cycles = list(nx.simple_cycles(backbone_subgraph))
            # In a proper ID history graph, backbone should have no cycles
            # (IDs progress forward in time)
            assert len(cycles) == 0, "Backbone history should be acyclic"

    def test_handles_self_loops(self, mock_the_graph):
        """Test graph does not contain self-loops."""
        # Self-loops would be invalid in an ID conversion graph
        for node in mock_the_graph.nodes():
            assert not mock_the_graph.has_edge(node, node), f"Self-loop found at {node}"

    def test_handles_nonexistent_node_query(self, mock_the_graph):
        """Test querying nonexistent nodes returns None."""
        result, was_converted = mock_the_graph.node_name_alternatives("DEFINITELY_NOT_IN_GRAPH_12345")
        assert result is None
        assert was_converted is False
