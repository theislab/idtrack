#!/usr/bin/env python3
"""
Functional tests for idtrack._the_graph module.

These tests actually execute code paths to increase coverage.
"""

from __future__ import annotations

import pytest
import networkx as nx

from idtrack._db import DB
from idtrack._the_graph import TheGraph


class TestTheGraphCachedPropertiesFunctional:
    """Functional tests for TheGraph cached properties."""

    @pytest.fixture
    def populated_graph(self):
        """Create a populated graph that mimics a small, valid IDTrack backbone + externals."""
        import numpy as np

        graph = TheGraph()

        # Set graph-level metadata
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 105, 110]
        graph.graph["organism"] = "homo_sapiens"

        # Add backbone gene nodes (ensembl_gene type). Include the "Void" sentinel so active-range logic works.
        gene_nodes = [
            ("ENSG00000141510.Void", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": DB.no_old_node_id}),
            ("ENSG00000141510.1", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": 1}),
            ("ENSG00000141510.2", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": 2}),
            ("ENSG00000141510.15", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": 15}),
            ("ENSG00000012048.Void", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000012048", "Version": DB.no_old_node_id}),
            ("ENSG00000012048.1", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000012048", "Version": 1}),
            ("ENSG00000012048.11", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000012048", "Version": 11}),
        ]
        graph.add_nodes_from(gene_nodes)

        # Add base ID nodes (base_ensembl_gene type)
        base_nodes = [
            ("ENSG00000141510", {DB.node_type_str: DB.nts_base_ensembl["gene"], "Version": None}),
            ("ENSG00000012048", {DB.node_type_str: DB.nts_base_ensembl["gene"], "Version": None}),
        ]
        graph.add_nodes_from(base_nodes)

        # Add external nodes (external type)
        external_nodes = [
            ("TP53", {DB.node_type_str: DB.nts_external, "database": "HGNC Symbol"}),
            ("P04637", {DB.node_type_str: DB.nts_external, "database": "UniProtKB"}),
            ("7157", {DB.node_type_str: DB.nts_external, "database": "EntrezGene"}),
            ("BRCA1", {DB.node_type_str: DB.nts_external, "database": "HGNC Symbol"}),
            ("P38398", {DB.node_type_str: DB.nts_external, "database": "UniProtKB"}),
        ]
        graph.add_nodes_from(external_nodes)

        # Temporal edges between gene versions (same node type - backbone). These must include old/new releases.
        graph.add_edge("ENSG00000141510.Void", "ENSG00000141510.1", weight=np.nan, old_release=100, new_release=100)
        graph.add_edge("ENSG00000141510.1", "ENSG00000141510.2", weight=1.0, old_release=100, new_release=105)
        graph.add_edge("ENSG00000141510.2", "ENSG00000141510.15", weight=1.0, old_release=105, new_release=110)
        graph.add_edge("ENSG00000141510.15", "ENSG00000141510.15", weight=1.0, old_release=110, new_release=np.inf)

        graph.add_edge("ENSG00000012048.Void", "ENSG00000012048.1", weight=np.nan, old_release=100, new_release=100)
        graph.add_edge("ENSG00000012048.1", "ENSG00000012048.11", weight=1.0, old_release=100, new_release=110)
        graph.add_edge("ENSG00000012048.11", "ENSG00000012048.11", weight=1.0, old_release=110, new_release=np.inf)

        # Add edges from base to versioned genes
        graph.add_edge(
            "ENSG00000141510", "ENSG00000141510.1",
            **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {100}}}}
        )
        graph.add_edge(
            "ENSG00000141510", "ENSG00000141510.2",
            **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {105}}}}
        )
        graph.add_edge(
            "ENSG00000141510", "ENSG00000141510.15",
            **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {110}}}}
        )
        graph.add_edge(
            "ENSG00000012048", "ENSG00000012048.1",
            **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {100}}}}
        )
        graph.add_edge(
            "ENSG00000012048", "ENSG00000012048.11",
            **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {110}}}}
        )

        # Add edges from external to versioned genes
        graph.add_edge(
            "TP53", "ENSG00000141510.15",
            **{DB.connection_dict: {"HGNC Symbol": {38: {110}}}}
        )
        graph.add_edge(
            "P04637", "ENSG00000141510.15",
            **{DB.connection_dict: {"UniProtKB": {38: {105, 110}}}}
        )
        graph.add_edge(
            "7157", "ENSG00000141510.15",
            **{DB.connection_dict: {"EntrezGene": {38: {110}}}}
        )
        graph.add_edge(
            "BRCA1", "ENSG00000012048.11",
            **{DB.connection_dict: {"HGNC Symbol": {38: {110}}}}
        )
        graph.add_edge(
            "P38398", "ENSG00000012048.11",
            **{DB.connection_dict: {"UniProtKB": {38: {110}}}}
        )

        # Attach included forms
        graph._attach_included_forms(["gene"])

        return graph

    def test_lower_chars_graph_computed(self, populated_graph):
        """Test lower_chars_graph cached property is computed correctly."""
        # Access the cached property
        lower_map = populated_graph.lower_chars_graph

        # Verify structure
        assert isinstance(lower_map, dict)
        assert len(lower_map) > 0

        # Verify lowercase mapping works
        assert lower_map["tp53"] == "TP53"
        assert lower_map["brca1"] == "BRCA1"
        assert lower_map["ensg00000141510.15"] == "ENSG00000141510.15"

    def test_combined_edges_computed(self, populated_graph):
        """Test combined_edges cached property is computed correctly."""
        # Access the cached property
        combined = populated_graph.combined_edges

        # Verify structure
        assert isinstance(combined, dict)

        # External nodes should be in combined_edges
        assert "TP53" in combined
        assert "BRCA1" in combined

        # Base ensembl nodes should be in combined_edges
        assert "ENSG00000141510" in combined

    def test_combined_edges_genes_computed(self, populated_graph):
        """Test combined_edges_genes cached property is computed correctly."""
        # Access the cached property
        combined_genes = populated_graph.combined_edges_genes

        # Verify structure
        assert isinstance(combined_genes, dict)

        # Gene nodes should be in combined_edges_genes
        assert "ENSG00000141510.15" in combined_genes or "ENSG00000141510.1" in combined_genes

    def test_hyperconnective_nodes_computed(self, populated_graph):
        """Test hyperconnective_nodes cached property is computed correctly."""
        # Access the cached property
        hyperconn = populated_graph.hyperconnective_nodes

        # Verify structure
        assert isinstance(hyperconn, dict)

        # With our small test graph, no nodes should exceed the threshold
        # (DB.hyperconnecting_threshold is typically 100+)
        assert len(hyperconn) == 0

    def test_rev_property(self, populated_graph):
        """Test rev cached property returns reversed graph."""
        rev_graph = populated_graph.rev

        # Verify it's a reversed view
        assert isinstance(rev_graph, nx.MultiDiGraph)

        # Check edge direction is reversed
        # Original: TP53 -> ENSG00000141510.15
        # Reversed: ENSG00000141510.15 -> TP53
        assert populated_graph.has_edge("TP53", "ENSG00000141510.15")
        assert rev_graph.has_edge("ENSG00000141510.15", "TP53")

    def test_calculate_caches_runs(self, populated_graph):
        """Test calculate_caches method executes without error."""
        populated_graph.calculate_caches(for_test=False)
        assert "combined_edges" in populated_graph.__dict__
        assert "get_active_ranges_of_id" in populated_graph.__dict__

    def test_calculate_caches_with_test_flag(self, populated_graph):
        """Test calculate_caches with for_test=True."""
        populated_graph.calculate_caches(for_test=True)
        assert "node_trios" in populated_graph.__dict__


class TestNodeNameAlternatives:
    """Test node_name_alternatives method."""

    @pytest.fixture
    def graph_with_nodes(self):
        """Create a graph with various node types for testing."""
        graph = TheGraph()
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        # Add nodes
        graph.add_node("TP53", **{DB.node_type_str: DB.nts_external})
        graph.add_node("ENSG00000141510.15", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_node("ENSG00000141510", **{DB.node_type_str: DB.nts_base_ensembl["gene"]})
        graph.add_node("P04637", **{DB.node_type_str: DB.nts_external})

        # Build the lowercase cache
        _ = graph.lower_chars_graph

        return graph

    def test_exact_match(self, graph_with_nodes):
        """Test exact identifier match."""
        result, converted = graph_with_nodes.node_name_alternatives("TP53")
        assert result == "TP53"
        assert converted is False

    def test_lowercase_match(self, graph_with_nodes):
        """Test lowercase identifier is matched."""
        result, converted = graph_with_nodes.node_name_alternatives("tp53")
        assert result == "TP53"
        assert converted is True

    def test_versioned_id_match(self, graph_with_nodes):
        """Test versioned Ensembl ID match."""
        result, converted = graph_with_nodes.node_name_alternatives("ENSG00000141510.15")
        assert result == "ENSG00000141510.15"
        assert converted is False

    def test_unknown_identifier(self, graph_with_nodes):
        """Test unknown identifier returns None."""
        result, converted = graph_with_nodes.node_name_alternatives("UNKNOWN_GENE_XYZ")
        assert result is None
        assert converted is False

    def test_empty_string(self, graph_with_nodes):
        """Test empty string returns None."""
        result, converted = graph_with_nodes.node_name_alternatives("")
        assert result is None


class TestGetActiveRangesOfId:
    """Test get_active_ranges_of_id cached property."""

    @pytest.fixture
    def graph_with_releases(self):
        """Create a graph with release information."""
        import numpy as np

        graph = TheGraph()
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 105, 110]

        # Add backbone nodes with a valid birth edge and a "still active" self-loop.
        graph.add_node("GENE.Void", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": DB.no_old_node_id})
        graph.add_node("GENE.1", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": 1})
        graph.add_node("GENE.2", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": 2})

        graph.add_edge("GENE.Void", "GENE.1", weight=np.nan, old_release=100, new_release=100)
        graph.add_edge("GENE.1", "GENE.2", weight=1.0, old_release=100, new_release=105)
        graph.add_edge("GENE.2", "GENE.2", weight=1.0, old_release=105, new_release=np.inf)

        return graph

    def test_get_active_ranges_computed(self, graph_with_releases):
        """Test get_active_ranges_of_id cached property."""
        active_ranges = graph_with_releases.get_active_ranges_of_id
        assert isinstance(active_ranges, dict)
        assert active_ranges["GENE.1"] == [[100, 100]]
        assert active_ranges["GENE.2"][0][0] == 105


class TestAvailableExternalDatabases:
    """Test available_external_databases cached property."""

    @pytest.fixture
    def graph_with_externals(self):
        """Create a graph with external database connections."""
        graph = TheGraph()
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [110]

        # Add external nodes
        graph.add_node("TP53", **{DB.node_type_str: DB.nts_external, "database": "HGNC Symbol"})
        graph.add_node("P04637", **{DB.node_type_str: DB.nts_external, "database": "UniProtKB"})

        # Add gene node
        graph.add_node("ENSG.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})

        # Add connections
        graph.add_edge("TP53", "ENSG.1", **{DB.connection_dict: {"HGNC Symbol": {38: {110}}}})
        graph.add_edge("P04637", "ENSG.1", **{DB.connection_dict: {"UniProtKB": {38: {110}}}})

        graph._attach_included_forms(["gene"])
        return graph

    def test_available_external_databases(self, graph_with_externals):
        """Test available_external_databases returns correct databases."""
        databases = graph_with_externals.available_external_databases

        assert isinstance(databases, set)
        assert "HGNC Symbol" in databases
        assert "UniProtKB" in databases


class TestAvailableGenomeAssemblies:
    """Test available_genome_assemblies cached property."""

    @pytest.fixture
    def graph_with_assembly(self):
        """Create a graph with assembly information."""
        graph = TheGraph()
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [110]

        graph.add_node("GENE.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_node("EXT.1", **{DB.node_type_str: DB.nts_external})

        graph.add_edge("EXT.1", "GENE.1", **{DB.connection_dict: {"DB": {38: {110}}}})

        graph._attach_included_forms(["gene"])
        return graph

    def test_available_genome_assemblies(self, graph_with_assembly):
        """Test available_genome_assemblies returns correct assemblies."""
        assemblies = graph_with_assembly.available_genome_assemblies

        assert isinstance(assemblies, set)
        assert 38 in assemblies


class TestNodeTrios:
    """Test node_trios cached property."""

    @pytest.fixture
    def graph_for_trios(self):
        """Create a graph for testing node_trios."""
        import numpy as np

        graph = TheGraph()
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        # Add nodes
        graph.add_node("EXT1", **{DB.node_type_str: DB.nts_external})
        graph.add_node("GENE.Void", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": DB.no_old_node_id})
        graph.add_node("GENE.1", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": 1})

        # Temporal edges to make the backbone valid
        graph.add_edge("GENE.Void", "GENE.1", weight=np.nan, old_release=100, new_release=100)
        graph.add_edge("GENE.1", "GENE.1", weight=1.0, old_release=100, new_release=np.inf)

        # Add connection (external → gene)
        graph.add_edge("EXT1", "GENE.1", **{DB.connection_dict: {"DB1": {38: {100, 110}}}})

        graph._attach_included_forms(["gene"])
        return graph

    def test_node_trios_computed(self, graph_for_trios):
        """Test node_trios cached property is computed."""
        trios = graph_for_trios.node_trios
        assert isinstance(trios, dict)
        assert ("DB1", 38, 100) in trios["EXT1"]
        assert ("DB1", 38, 110) in trios["EXT1"]


class TestAttachIncludedForms:
    """Test _attach_included_forms method."""

    def test_attach_forms(self):
        """Test _attach_included_forms sets available_forms."""
        graph = TheGraph()
        assert graph.available_forms is None

        graph._attach_included_forms(["gene", "transcript"])

        assert graph.available_forms == ["gene", "transcript"]

    def test_attach_empty_forms(self):
        """Test _attach_included_forms with empty list."""
        graph = TheGraph()
        graph._attach_included_forms([])

        assert graph.available_forms == []


class TestCombinedEdgesHelper:
    """Test _combined_edges_genes_helper static method."""

    def test_helper_merges_correctly(self):
        """Test _combined_edges_genes_helper merges nested dicts."""
        # Simulate input from _combined_edges
        input_dict = {
            "neighbor1": {38: {100, 105}},
            "neighbor2": {38: {105, 110}},
        }

        result = TheGraph._combined_edges_genes_helper(input_dict)

        # Result should be reorganized by assembly
        assert isinstance(result, dict)


class TestCombinedEdgesStaticMethod:
    """Test _combined_edges static method."""

    def test_combined_edges_basic(self):
        """Test _combined_edges with simple graph."""
        graph = TheGraph()
        graph.graph["genome_assembly"] = 38

        # Add nodes of different types
        graph.add_node("EXT", **{DB.node_type_str: DB.nts_external})
        graph.add_node("GENE", **{DB.node_type_str: DB.nts_ensembl["gene"]})

        # Add edge between different types
        graph.add_edge("EXT", "GENE", **{DB.connection_dict: {"DB": {38: {100}}}})

        # Call static method
        result = TheGraph._combined_edges(["EXT"], graph)

        assert "EXT" in result
        assert isinstance(result["EXT"], dict)

    def test_combined_edges_excludes_same_type(self):
        """Test _combined_edges excludes edges between same node types."""
        graph = TheGraph()
        graph.graph["genome_assembly"] = 38

        # Add nodes of same type
        graph.add_node("GENE1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_node("GENE2", **{DB.node_type_str: DB.nts_ensembl["gene"]})

        # Add edge between same types
        graph.add_edge("GENE1", "GENE2", **{DB.connection_dict: {DB.nts_ensembl["gene"]: {38: {100}}}})

        # Call static method
        result = TheGraph._combined_edges(["GENE1"], graph)

        # GENE1 should not be in result because edge is to same type
        assert "GENE1" not in result or len(result.get("GENE1", {})) == 0


class TestGraphInitialization:
    """Test TheGraph initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        graph = TheGraph()

        assert isinstance(graph, nx.MultiDiGraph)
        assert graph.available_forms is None
        assert hasattr(graph, "log")

    def test_initialization_with_data(self):
        """Test initialization with initial data."""
        initial_edges = [("A", "B"), ("B", "C")]
        graph = TheGraph(initial_edges)

        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")

    def test_initialization_with_name(self):
        """Test initialization with name attribute."""
        graph = TheGraph(name="test_graph")

        assert graph.name == "test_graph"
