#!/usr/bin/env python3
"""Functional tests for idtrack._the_graph module.

These tests actually execute code paths to increase coverage.
"""

from __future__ import annotations

import networkx as nx
import pytest

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
            (
                "ENSG00000141510.Void",
                {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": DB.no_old_node_id},
            ),
            ("ENSG00000141510.1", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": 1}),
            ("ENSG00000141510.2", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": 2}),
            ("ENSG00000141510.15", {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000141510", "Version": 15}),
            (
                "ENSG00000012048.Void",
                {DB.node_type_str: DB.nts_ensembl["gene"], "ID": "ENSG00000012048", "Version": DB.no_old_node_id},
            ),
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
            "ENSG00000141510", "ENSG00000141510.1", **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {100}}}}
        )
        graph.add_edge(
            "ENSG00000141510", "ENSG00000141510.2", **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {105}}}}
        )
        graph.add_edge(
            "ENSG00000141510", "ENSG00000141510.15", **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {110}}}}
        )
        graph.add_edge(
            "ENSG00000012048", "ENSG00000012048.1", **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {100}}}}
        )
        graph.add_edge(
            "ENSG00000012048", "ENSG00000012048.11", **{DB.connection_dict: {DB.nts_base_ensembl["gene"]: {38: {110}}}}
        )

        # Add edges from external to versioned genes
        graph.add_edge("TP53", "ENSG00000141510.15", **{DB.connection_dict: {"HGNC Symbol": {38: {110}}}})
        graph.add_edge("P04637", "ENSG00000141510.15", **{DB.connection_dict: {"UniProtKB": {38: {105, 110}}}})
        graph.add_edge("7157", "ENSG00000141510.15", **{DB.connection_dict: {"EntrezGene": {38: {110}}}})
        graph.add_edge("BRCA1", "ENSG00000012048.11", **{DB.connection_dict: {"HGNC Symbol": {38: {110}}}})
        graph.add_edge("P38398", "ENSG00000012048.11", **{DB.connection_dict: {"UniProtKB": {38: {110}}}})

        # Attach included forms
        graph._attach_included_forms(["gene"])

        return graph

    def test_names_are_found_in_any_letter_case(self, populated_graph):
        """A node is found by its name in another letter case, and the match says so."""
        assert populated_graph.resolve_name("tp53") == ("TP53", "case")
        assert populated_graph.resolve_name("brca1") == ("BRCA1", "case")
        assert populated_graph.resolve_name("ensg00000141510.15") == ("ENSG00000141510.15", "case")
        assert populated_graph.resolve_name("TP53") == ("TP53", "exact")

    def test_combined_of_external_and_base_nodes(self, populated_graph):
        """External ids and base gene ids are summed over the edges that leave them."""
        assert set(populated_graph.combined("TP53")) == {"HGNC Symbol"}
        assert set(populated_graph.combined("BRCA1")) == {"HGNC Symbol"}
        assert set(populated_graph.combined("ENSG00000141510")) == {"base_ensembl_gene"}

    def test_combined_of_a_gene(self, populated_graph):
        """A gene is summed over the edges that enter it, under the gene type of each assembly."""
        assert set(populated_graph.combined("ENSG00000141510.15")) == {"assembly_38_ensembl_gene"}

    def test_hyperconnective_nodes_computed(self, populated_graph):
        """Test hyperconnective_nodes cached property is computed correctly."""
        # Access the cached property
        hyperconn = populated_graph.hyperconnective_nodes

        # Verify structure
        assert isinstance(hyperconn, dict)

        # No external id of the small test graph is linked to more genes at once than
        # `TheGraph.genes_at_once_threshold`.
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
        """calculate_caches freezes the graph; no table over all nodes is kept."""
        populated_graph.calculate_caches(for_test=False)
        assert populated_graph.is_frozen
        tables = [
            k for k, v in populated_graph.__dict__.items() if isinstance(v, dict) and len(v) >= len(populated_graph)
        ]
        assert [k for k in tables if k not in ("graph",)] == []

    def test_calculate_caches_with_test_flag(self, populated_graph):
        """calculate_caches with for_test=True also computes the table of the integrity tests."""
        populated_graph.calculate_caches(for_test=True)
        table = populated_graph.available_releases_given_database_assembly
        assert all(isinstance(key, tuple) and len(key) == 2 for key in table)

    def test_frozen_graph_refuses_changes(self, populated_graph):
        """After the first derived value nothing can be changed, so nothing derived can be stale."""
        _ = populated_graph.combined("TP53")
        with pytest.raises(nx.NetworkXError, match="frozen"):
            populated_graph.add_node("LATE", **{DB.node_type_str: DB.nts_external})
        with pytest.raises(nx.NetworkXError, match="frozen"):
            populated_graph.add_edge("TP53", "ENSG00000141510.1")


class TestResolveNameFunctional:
    """Test `resolve_name` on a graph with several node types."""

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

        return graph

    def test_exact_match(self, graph_with_nodes):
        """Test exact identifier match."""
        assert graph_with_nodes.resolve_name("TP53") == ("TP53", "exact")

    def test_lowercase_match(self, graph_with_nodes):
        """Test lowercase identifier is matched."""
        assert graph_with_nodes.resolve_name("tp53") == ("TP53", "case")

    def test_versioned_id_match(self, graph_with_nodes):
        """Test versioned Ensembl ID match."""
        assert graph_with_nodes.resolve_name("ENSG00000141510.15") == ("ENSG00000141510.15", "exact")

    def test_unknown_identifier(self, graph_with_nodes):
        """Test unknown identifier returns None."""
        assert graph_with_nodes.resolve_name("UNKNOWN_GENE_XYZ") == (None, None)

    def test_empty_string(self, graph_with_nodes):
        """Test empty string returns None."""
        assert graph_with_nodes.resolve_name("") == (None, None)


class TestActiveRanges:
    """Test the release ranges of a gene, from its history edges."""

    @pytest.fixture
    def graph_with_releases(self):
        """Create a graph with release information."""
        import numpy as np

        graph = TheGraph()
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 105, 110]

        # Add backbone nodes with a valid birth edge and a "still active" self-loop.
        graph.add_node(
            "GENE.Void", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": DB.no_old_node_id}
        )
        graph.add_node("GENE.1", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": 1})
        graph.add_node("GENE.2", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": 2})

        graph.add_edge("GENE.Void", "GENE.1", weight=np.nan, old_release=100, new_release=100)
        graph.add_edge("GENE.1", "GENE.2", weight=1.0, old_release=100, new_release=105)
        graph.add_edge("GENE.2", "GENE.2", weight=1.0, old_release=105, new_release=np.inf)

        return graph

    def test_active_ranges_computed(self, graph_with_releases):
        """A version is active from the release it is entered at to the release it is left at."""
        assert graph_with_releases.active_ranges("GENE.1") == [[100, 100]]
        assert graph_with_releases.active_ranges("GENE.2") == [[105, float("inf")]]
        assert graph_with_releases.is_active("GENE.2", 110) and not graph_with_releases.is_active("GENE.1", 105)


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
        graph.add_node(
            "GENE.Void", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": DB.no_old_node_id}
        )
        graph.add_node("GENE.1", **{DB.node_type_str: DB.nts_ensembl["gene"], "ID": "GENE", "Version": 1})

        # Temporal edges to make the backbone valid
        graph.add_edge("GENE.Void", "GENE.1", weight=np.nan, old_release=100, new_release=100)
        graph.add_edge("GENE.1", "GENE.1", weight=1.0, old_release=100, new_release=np.inf)

        # Add connection (external → gene)
        graph.add_edge("EXT1", "GENE.1", **{DB.connection_dict: {"DB1": {38: {100, 110}}}})

        graph._attach_included_forms(["gene"])
        return graph

    def test_trios_computed(self, graph_for_trios):
        """The (database, assembly, release) triples of a node come from its own edges."""
        trios = graph_for_trios.trios("EXT1")
        assert ("DB1", 38, 100) in trios
        assert ("DB1", 38, 110) in trios


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


class TestCombined:
    """Test the per-node summary of cross-reference edges."""

    def test_external_node_is_summed_per_database(self):
        """The edges that leave an external node are summed per database and assembly."""
        graph = TheGraph(genome_assembly=38, confident_for_release=[100, 105, 110])
        graph.add_node("EXT", **{DB.node_type_str: DB.nts_external})
        graph.add_node("GENE.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_node("GENE.2", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_edge("EXT", "GENE.1", **{DB.connection_dict: {"DB": {38: {100}}}})
        graph.add_edge("EXT", "GENE.2", **{DB.connection_dict: {"DB": {38: {105, 110}}, "DB2": {37: {105}}}})

        assert graph.combined("EXT") == {"DB": {38: {100, 105, 110}}, "DB2": {37: {105}}}

    def test_gene_is_summed_per_assembly(self):
        """The edges that enter a gene are reported under the gene type of each assembly."""
        graph = TheGraph(genome_assembly=38, confident_for_release=[100, 105, 110])
        graph.add_node("EXT", **{DB.node_type_str: DB.nts_external})
        graph.add_node("GENE", **{DB.node_type_str: DB.nts_base_ensembl["gene"]})
        graph.add_node("GENE.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_edge("EXT", "GENE.1", **{DB.connection_dict: {"DB": {38: {100}, 37: {100, 105}}}})
        graph.add_edge("GENE", "GENE.1", **{DB.connection_dict: {"base_ensembl_gene": {38: {100, 105}}}})

        expected = {"assembly_38_ensembl_gene": {38: {100, 105}}, "assembly_37_ensembl_gene": {37: {100, 105}}}
        assert graph.combined("GENE.1") == expected
        assert graph.combined("GENE") == {"base_ensembl_gene": {38: {100, 105}}}

    def test_edges_between_nodes_of_one_type_are_not_summed(self):
        """History edges are not cross-references."""
        graph = TheGraph(genome_assembly=38, confident_for_release=[100, 101])
        graph.add_node("GENE1.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_node("GENE2.1", **{DB.node_type_str: DB.nts_ensembl["gene"]})
        graph.add_edge("GENE1.1", "GENE2.1", weight=1.0, old_release=100, new_release=101)

        with pytest.raises(KeyError):
            graph.combined("GENE1.1")


class TestGraphInitialization:
    """Test TheGraph initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        graph = TheGraph()

        assert isinstance(graph, nx.MultiDiGraph)
        assert graph.available_forms is None

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
