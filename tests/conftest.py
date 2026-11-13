#!/usr/bin/env python3
"""Shared pytest fixtures for idtrack test suite.

This module provides:
- Session-scoped fixtures for expensive operations (graph construction, mock data)
- Function-scoped fixtures for isolated test data
- Mock fixtures to avoid network calls in unit tests
- Utility fixtures for common test patterns
"""

import importlib.util
import os
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from idtrack._db import DB

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow.",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped unless --runslow)")
    config.addinivalue_line("markers", "network: marks tests requiring network access")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default (opt-in via `--runslow`), and wrong-argument-type tests under typeguard."""
    if config.getoption("typeguard_packages", default=None):
        # With `--typeguard-packages` every annotated argument is checked before the function runs, so a test that
        # passes a value of another type on purpose never reaches the package's own check and error message.
        skip_typed = pytest.mark.skip(reason="typeguard rejects the argument before the package's own check runs")
        for item in items:
            if "wrong_argument_type" in item.keywords:
                item.add_marker(skip_typed)

    if config.getoption("runslow"):
        return

    env_override = os.getenv("IDTRACK_SKIP_SLOW")
    if env_override is not None and env_override.strip().lower() in {"0", "false", "no"}:
        return

    skip_slow = pytest.mark.skip(reason="pass --runslow (or set IDTRACK_SKIP_SLOW=0) to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# =============================================================================
# BASIC FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[str]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# GRAPH FIXTURES
# =============================================================================


@pytest.fixture
def minimal_multigraph() -> nx.MultiDiGraph:
    """Create a minimal NetworkX MultiDiGraph for testing TheGraph operations."""
    graph = nx.MultiDiGraph()

    # Add gene nodes with proper attributes
    gene_nodes = [
        ("ENSG00000141510.1", {"node_type": "ensembl_gene", "Version": 1}),
        ("ENSG00000141510.2", {"node_type": "ensembl_gene", "Version": 2}),
        ("ENSG00000141510.15", {"node_type": "ensembl_gene", "Version": 15}),
        ("ENSG00000012048.1", {"node_type": "ensembl_gene", "Version": 1}),
    ]
    graph.add_nodes_from(gene_nodes)

    # Add base ID nodes
    base_nodes = [
        ("ENSG00000141510", {"node_type": "base_ensembl_gene", "Version": None}),
        ("ENSG00000012048", {"node_type": "base_ensembl_gene", "Version": None}),
    ]
    graph.add_nodes_from(base_nodes)

    # Add external nodes
    external_nodes = [
        ("TP53", {"node_type": "external", "database": "HGNC Symbol"}),
        ("P04637", {"node_type": "external", "database": "UniProtKB"}),
        ("BRCA1", {"node_type": "external", "database": "HGNC Symbol"}),
    ]
    graph.add_nodes_from(external_nodes)

    # Add edges between versions (temporal edges)
    graph.add_edge(
        "ENSG00000141510.1",
        "ENSG00000141510.2",
        old_release=70,
        new_release=80,
        connection={"ensembl_gene": {38: {70, 80}}},
    )
    graph.add_edge(
        "ENSG00000141510.2",
        "ENSG00000141510.15",
        old_release=80,
        new_release=110,
        connection={"ensembl_gene": {38: {80, 110}}},
    )

    # Add edges from base to versioned
    graph.add_edge("ENSG00000141510", "ENSG00000141510.1", connection={"base_ensembl_gene": {38: {70}}})
    graph.add_edge("ENSG00000141510", "ENSG00000141510.2", connection={"base_ensembl_gene": {38: {80}}})
    graph.add_edge("ENSG00000141510", "ENSG00000141510.15", connection={"base_ensembl_gene": {38: {110}}})

    # Add external edges
    graph.add_edge("TP53", "ENSG00000141510.15", connection={"HGNC Symbol": {38: {110}}})
    graph.add_edge("P04637", "ENSG00000141510.15", connection={"UniProtKB": {38: {110}}})

    # Add graph-level metadata
    graph.graph["genome_assembly"] = 38
    graph.graph["confident_for_release"] = [70, 80, 90, 100, 110]
    graph.graph["organism"] = "homo_sapiens"

    return graph


@pytest.fixture
def mock_the_graph(minimal_multigraph):
    """Create a mock TheGraph instance for testing."""
    from idtrack._the_graph import TheGraph

    # Create TheGraph and copy data from minimal graph
    graph = TheGraph()
    graph.add_nodes_from(minimal_multigraph.nodes(data=True))
    graph.add_edges_from(minimal_multigraph.edges(data=True, keys=True))
    graph.graph.update(minimal_multigraph.graph)

    return graph


# =============================================================================
# MOCK NETWORK FIXTURES
# =============================================================================


@pytest.fixture
def mock_requests_session():
    """Mock requests.Session for testing network calls."""
    with patch("requests.Session") as mock_session:
        session_instance = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        yield session_instance


@pytest.fixture
def mock_ensembl_rest_response() -> dict[str, Any]:
    """Mock response from Ensembl REST API /info/species endpoint."""
    return {
        "species": [
            {
                "name": "homo_sapiens",
                "common_name": "human",
                "display_name": "Human",
                "taxon_id": "9606",
                "assembly": "GRCh38.p14",
                "accession": "GCA_000001405.29",
                "release": "110",
                "groups": ["core", "funcgen", "variation"],
                "aliases": ["hsapiens", "Homo sapiens"],
            },
            {
                "name": "mus_musculus",
                "common_name": "mouse",
                "display_name": "Mouse",
                "taxon_id": "10090",
                "assembly": "GRCm39",
                "accession": "GCA_000001635.9",
                "release": "110",
                "groups": ["core", "funcgen", "variation"],
                "aliases": ["mmusculus", "Mus musculus"],
            },
        ]
    }


# =============================================================================
# YAML CONFIG FIXTURES
# =============================================================================


@pytest.fixture
def sample_external_db_yaml() -> dict:
    """Sample external database YAML configuration."""
    return {
        "homo_sapiens": {
            "gene": {
                "HGNC Symbol": {
                    "Database Index": 0,
                    "Potential Synonymous": "",
                    "Assembly": {
                        "38": {
                            "Ensembl release": "100,101,102,103,104,105,106,107,108,109,110",
                            "Include": True,
                        },
                        "37": {
                            "Ensembl release": "79,80,81,82,83,84,85,86,87",
                            "Include": True,
                        },
                    },
                },
                "UniProtKB": {
                    "Database Index": 1,
                    "Potential Synonymous": "",
                    "Assembly": {
                        "38": {
                            "Ensembl release": "100,101,102,103,104,105,106,107,108,109,110",
                            "Include": True,
                        },
                    },
                },
            },
            "transcript": {
                "RefSeq mRNA": {
                    "Database Index": 0,
                    "Potential Synonymous": "",
                    "Assembly": {
                        "38": {
                            "Ensembl release": "100,101,102,103,104,105,106,107,108,109,110",
                            "Include": True,
                        },
                    },
                },
            },
        }
    }


# =============================================================================
# SKIP FIXTURES FOR OPTIONAL DEPENDENCIES
# =============================================================================


@pytest.fixture
def skip_if_no_mygene():
    """Skip test if mygene is not installed."""
    if importlib.util.find_spec("mygene") is None:
        pytest.skip("mygene not installed")


@pytest.fixture
def skip_if_no_gprofiler():
    """Skip test if gprofiler-official is not installed."""
    if importlib.util.find_spec("gprofiler") is None:
        pytest.skip("gprofiler-official not installed")


# =============================================================================
# GRAPHS OF REAL ORGANISMS (BUILT FROM THE CUT ENSEMBL TABLES OF tests/data/real, NO NETWORK)
# =============================================================================


@pytest.fixture(scope="session")
def real_world(tmp_path_factory):
    """Return ``get(organism)``: the graph of a real organism, built once per session from its cut tables.

    The graph is built by ``API.build_graph`` from the tables of ``tests/data/real`` with the network refused
    (:py:class:`tests._real_tables.RealWorld`); organisms: ``tests._real_tables.ORGANISMS``.
    """
    from tests._real_tables import RealWorld

    worlds: dict[str, Any] = {}

    def get(organism: str):
        if organism not in worlds:
            worlds[organism] = RealWorld(organism, tmp_path_factory.mktemp(f"real_{organism}"))
        return worlds[organism]

    return get


@pytest.fixture(autouse=True)
def _fresh_species_cache(monkeypatch):
    """Give every test its own (empty) per-process cache of the Ensembl species list, and no retry waits."""
    from idtrack._verify_organism import VerifyOrganism

    monkeypatch.setattr(VerifyOrganism, "_species_cache", {})
    monkeypatch.setattr(DB, "rest_retry_wait", 0.0)
