#!/usr/bin/env python3
"""
Shared pytest fixtures for idtrack test suite.

This module provides:
- Session-scoped fixtures for expensive operations (graph construction, mock data)
- Function-scoped fixtures for isolated test data
- Mock fixtures to avoid network calls in unit tests
- Utility fixtures for common test patterns
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# Add idtrack to path for imports
_IDTRACK_PKG_DIR = Path(__file__).parent.parent / "idtrack"
if str(_IDTRACK_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_IDTRACK_PKG_DIR))

from idtrack._db import DB


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (network/integration)")
    config.addinivalue_line("markers", "network: marks tests requiring network access")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Optionally skip slow tests via `IDTRACK_SKIP_SLOW=1`."""
    if config.getoption("-m"):
        return

    if os.getenv("IDTRACK_SKIP_SLOW", "").strip().lower() not in {"1", "true", "yes"}:
        return

    skip_slow = pytest.mark.skip(reason="unset IDTRACK_SKIP_SLOW (or set it to 0) to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# =============================================================================
# BASIC FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_ensembl_gene_ids() -> list[str]:
    """Sample Ensembl gene IDs for testing."""
    return [
        "ENSG00000141510",  # TP53
        "ENSG00000012048",  # BRCA1
        "ENSG00000139618",  # BRCA2
        "ENSG00000157764",  # BRAF
        "ENSG00000133703",  # KRAS
    ]


@pytest.fixture
def sample_ensembl_transcript_ids() -> list[str]:
    """Sample Ensembl transcript IDs for testing."""
    return [
        "ENST00000269305",  # TP53
        "ENST00000357654",  # BRCA1
        "ENST00000380152",  # BRCA2
    ]


@pytest.fixture
def sample_external_ids() -> dict[str, list[str]]:
    """Sample external database IDs for testing."""
    return {
        "uniprot": ["P04637", "P38398", "P51587"],
        "hgnc_symbol": ["TP53", "BRCA1", "BRCA2"],
        "entrez_gene": ["7157", "672", "675"],
        "refseq_mrna": ["NM_000546", "NM_007294", "NM_000059"],
    }


@pytest.fixture
def versioned_ensembl_ids() -> list[str]:
    """Ensembl IDs with version suffixes."""
    return [
        "ENSG00000141510.15",
        "ENST00000269305.8",
        "ENSP00000269305.4",
    ]


# =============================================================================
# MOCK DATA FIXTURES
# =============================================================================


@pytest.fixture
def mock_ensembl_gene_table() -> pd.DataFrame:
    """Mock Ensembl gene table for testing without database access."""
    return pd.DataFrame({
        "stable_id": ["ENSG00000141510", "ENSG00000012048", "ENSG00000139618"],
        "version": [15, 11, 12],
        "biotype": ["protein_coding", "protein_coding", "protein_coding"],
        "analysis_id": [1, 1, 1],
        "seq_region_id": [1, 17, 13],
        "seq_region_start": [7661779, 43044295, 32315474],
        "seq_region_end": [7687538, 43170245, 32400266],
        "seq_region_strand": [-1, -1, 1],
    })


@pytest.fixture
def mock_ensembl_transcript_table() -> pd.DataFrame:
    """Mock Ensembl transcript table for testing."""
    return pd.DataFrame({
        "stable_id": ["ENST00000269305", "ENST00000357654", "ENST00000380152"],
        "version": [8, 7, 7],
        "gene_id": ["ENSG00000141510", "ENSG00000012048", "ENSG00000139618"],
        "biotype": ["protein_coding", "protein_coding", "protein_coding"],
    })


@pytest.fixture
def mock_xref_table() -> pd.DataFrame:
    """Mock external reference table for testing."""
    return pd.DataFrame({
        "ensembl_id": [
            "ENSG00000141510", "ENSG00000141510", "ENSG00000141510",
            "ENSG00000012048", "ENSG00000012048",
            "ENSG00000139618", "ENSG00000139618",
        ],
        "external_id": [
            "TP53", "P04637", "7157",
            "BRCA1", "P38398",
            "BRCA2", "P51587",
        ],
        "external_db": [
            "HGNC Symbol", "UniProtKB", "EntrezGene",
            "HGNC Symbol", "UniProtKB",
            "HGNC Symbol", "UniProtKB",
        ],
    })


@pytest.fixture
def mock_stable_id_event_table() -> pd.DataFrame:
    """Mock stable_id_event table for ID history tracking."""
    return pd.DataFrame({
        "old_stable_id": [
            "ENSG00000141510.1", "ENSG00000141510.2", "ENSG00000141510.3",
            "ENSG00000012048.1", "ENSG00000012048.2",
        ],
        "new_stable_id": [
            "ENSG00000141510.2", "ENSG00000141510.3", "ENSG00000141510.15",
            "ENSG00000012048.2", "ENSG00000012048.11",
        ],
        "old_release": [70, 80, 90, 70, 85],
        "new_release": [80, 90, 110, 85, 110],
        "type": ["gene", "gene", "gene", "gene", "gene"],
    })


# =============================================================================
# GRAPH FIXTURES
# =============================================================================


@pytest.fixture
def minimal_multigraph() -> nx.MultiDiGraph:
    """Create a minimal NetworkX MultiDiGraph for testing TheGraph operations."""
    G = nx.MultiDiGraph()

    # Add gene nodes with proper attributes
    gene_nodes = [
        ("ENSG00000141510.1", {"node_type": "ensembl_gene", "Version": 1}),
        ("ENSG00000141510.2", {"node_type": "ensembl_gene", "Version": 2}),
        ("ENSG00000141510.15", {"node_type": "ensembl_gene", "Version": 15}),
        ("ENSG00000012048.1", {"node_type": "ensembl_gene", "Version": 1}),
    ]
    G.add_nodes_from(gene_nodes)

    # Add base ID nodes
    base_nodes = [
        ("ENSG00000141510", {"node_type": "base_ensembl_gene", "Version": None}),
        ("ENSG00000012048", {"node_type": "base_ensembl_gene", "Version": None}),
    ]
    G.add_nodes_from(base_nodes)

    # Add external nodes
    external_nodes = [
        ("TP53", {"node_type": "external", "database": "HGNC Symbol"}),
        ("P04637", {"node_type": "external", "database": "UniProtKB"}),
        ("BRCA1", {"node_type": "external", "database": "HGNC Symbol"}),
    ]
    G.add_nodes_from(external_nodes)

    # Add edges between versions (temporal edges)
    G.add_edge("ENSG00000141510.1", "ENSG00000141510.2",
               old_release=70, new_release=80, connection={"ensembl_gene": {38: {70, 80}}})
    G.add_edge("ENSG00000141510.2", "ENSG00000141510.15",
               old_release=80, new_release=110, connection={"ensembl_gene": {38: {80, 110}}})

    # Add edges from base to versioned
    G.add_edge("ENSG00000141510", "ENSG00000141510.1", connection={"base_ensembl_gene": {38: {70}}})
    G.add_edge("ENSG00000141510", "ENSG00000141510.2", connection={"base_ensembl_gene": {38: {80}}})
    G.add_edge("ENSG00000141510", "ENSG00000141510.15", connection={"base_ensembl_gene": {38: {110}}})

    # Add external edges
    G.add_edge("TP53", "ENSG00000141510.15", connection={"HGNC Symbol": {38: {110}}})
    G.add_edge("P04637", "ENSG00000141510.15", connection={"UniProtKB": {38: {110}}})

    # Add graph-level metadata
    G.graph["genome_assembly"] = 38
    G.graph["confident_for_release"] = [70, 80, 90, 100, 110]
    G.graph["organism"] = "homo_sapiens"

    return G


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


@pytest.fixture
def mock_mysql_connection():
    """Mock PyMySQL connection for testing database access."""
    with patch("pymysql.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn
        yield mock_cursor


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


@pytest.fixture
def temp_yaml_config(temp_dir, sample_external_db_yaml) -> str:
    """Create a temporary YAML config file for testing."""
    import yaml

    yaml_path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
    with open(yaml_path, "w") as f:
        yaml.dump(sample_external_db_yaml, f)
    return yaml_path


# =============================================================================
# DATABASE MANAGER FIXTURES
# =============================================================================


@pytest.fixture
def mock_database_manager(
    mock_ensembl_gene_table,
    mock_ensembl_transcript_table,
    mock_xref_table,
    mock_stable_id_event_table,
):
    """Create a mock DatabaseManager that returns pre-defined tables."""
    mock_dm = MagicMock()

    # Configure table returns
    def get_table(table_name):
        tables = {
            "gene": mock_ensembl_gene_table,
            "transcript": mock_ensembl_transcript_table,
            "xref": mock_xref_table,
            "stable_id_event": mock_stable_id_event_table,
        }
        return tables.get(table_name, pd.DataFrame())

    mock_dm.get_table = MagicMock(side_effect=get_table)
    mock_dm.organism = "homo_sapiens"
    mock_dm.form = "gene"
    mock_dm.ensembl_release = 110
    mock_dm.genome_assembly = 38
    mock_dm.local_repository = "/tmp/mock_repo"
    mock_dm.available_releases = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

    return mock_dm


# =============================================================================
# ANNDATA FIXTURES (for harmonize_features tests)
# =============================================================================


@pytest.fixture
def mock_anndata():
    """Create mock AnnData objects for testing harmonization."""
    try:
        import anndata as ad
        import scipy.sparse as sp
    except ImportError:
        pytest.skip("anndata or scipy not installed")

    # Create sample AnnData with gene expression data
    n_obs, n_vars = 100, 50
    X = sp.random(n_obs, n_vars, density=0.3, format="csr")

    # Gene IDs as var names
    var_names = [f"ENSG{str(i).zfill(11)}" for i in range(n_vars)]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=var_names),
    )

    return adata


@pytest.fixture
def mock_anndata_dict(mock_anndata):
    """Create dictionary of mock AnnData objects for multi-dataset testing."""
    return {
        "dataset1": mock_anndata.copy(),
        "dataset2": mock_anndata.copy(),
    }


# =============================================================================
# HELPER FIXTURES
# =============================================================================


@pytest.fixture
def assert_dataframe_structure():
    """Fixture providing a function to assert DataFrame structure."""

    def _assert_structure(df: pd.DataFrame, expected_columns: list[str], min_rows: int = 0):
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

    return _assert_structure


@pytest.fixture
def compare_graphs():
    """Fixture providing a function to compare two graphs."""

    def _compare(g1: nx.Graph, g2: nx.Graph, check_data: bool = True) -> bool:
        if set(g1.nodes()) != set(g2.nodes()):
            return False
        if set(g1.edges()) != set(g2.edges()):
            return False
        if check_data:
            for node in g1.nodes():
                if g1.nodes[node] != g2.nodes[node]:
                    return False
        return True

    return _compare


# =============================================================================
# SKIP FIXTURES FOR OPTIONAL DEPENDENCIES
# =============================================================================


@pytest.fixture
def skip_if_no_mygene():
    """Skip test if mygene is not installed."""
    try:
        import mygene
    except ImportError:
        pytest.skip("mygene not installed")


@pytest.fixture
def skip_if_no_gprofiler():
    """Skip test if gprofiler-official is not installed."""
    try:
        import gprofiler
    except ImportError:
        pytest.skip("gprofiler-official not installed")


@pytest.fixture
def skip_if_no_pybiomart():
    """Skip test if pybiomart is not installed."""
    try:
        import pybiomart
    except ImportError:
        pytest.skip("pybiomart not installed")


@pytest.fixture
def skip_if_no_gget():
    """Skip test if gget is not installed."""
    try:
        import gget
    except ImportError:
        pytest.skip("gget not installed")


@pytest.fixture
def skip_if_no_anndata():
    """Skip test if anndata is not installed."""
    try:
        import anndata
    except ImportError:
        pytest.skip("anndata not installed")


# =============================================================================
# REAL GRAPH FIXTURES (NETWORK-BACKED, RELEASE-BOUNDED)
# =============================================================================


def _discover_default_config_organisms() -> list[str]:
    """Discover organisms shipped with default external-db configs."""
    default_config_dir = Path(__file__).parent.parent / "idtrack" / "default_config"
    configs = sorted(default_config_dir.glob("*_externals_modified.yml"))
    organisms = [cfg.name.removesuffix("_externals_modified.yml") for cfg in configs]
    return sorted(set(organisms))


def _selected_real_graph_organisms() -> list[str]:
    """Return organisms to exercise in network-backed tests.

    Defaults to organisms that ship a `*_externals_modified.yml` under `idtrack/default_config`.
    Override via `IDTRACK_TEST_ORGANISM=...` or `IDTRACK_TEST_ORGANISMS=a,b,c`.
    """
    env = os.environ.get("IDTRACK_TEST_ORGANISMS") or os.environ.get("IDTRACK_TEST_ORGANISM")
    if env:
        parts = [p.strip() for p in re.split(r"[,\\s]+", env) if p.strip()]
        return parts

    discovered = _discover_default_config_organisms()
    return discovered if discovered else ["homo_sapiens"]


def _default_real_graph_release_window() -> tuple[int, int]:
    """Default Ensembl release window used by integration tests."""
    min_rel = int(os.environ.get("IDTRACK_TEST_IGNORE_BEFORE", "100"))
    max_rel = int(os.environ.get("IDTRACK_TEST_IGNORE_AFTER", "103"))
    if min_rel > max_rel:
        raise ValueError("IDTRACK_TEST_IGNORE_BEFORE must be <= IDTRACK_TEST_IGNORE_AFTER")
    return min_rel, max_rel


@pytest.fixture(scope="session")
def real_graph_cache_root() -> Path:
    """Persistent cache dir for network-backed tests (defaults to `.pytest_cache/idtrack-real`)."""
    env = os.environ.get("IDTRACK_TEST_CACHE_DIR")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path(__file__).parent.parent / ".pytest_cache" / "idtrack-real"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def supported_default_config_organisms() -> list[str]:
    """Organisms that currently ship a default externals config."""
    return _discover_default_config_organisms()


@pytest.fixture(scope="session", params=_selected_real_graph_organisms())
def organism_under_test(request) -> str:
    """Organism name for network-backed tests (parametrized)."""
    return str(request.param)


@pytest.fixture(scope="session")
def real_gene_graph(real_graph_cache_root: Path, organism_under_test: str) -> tuple["Any", dict[str, str]]:
    """Small, real gene graph for Ensembl releases 100–103 (cached on disk).

    Builds the graph via targeted MySQL queries for a curated set of genes and external IDs, so it stays tiny and
    avoids multi-GB genome-wide builds.
    """
    from tests._real_graph_builder import RealGraphSpec, load_or_build_real_gene_graph

    organism = organism_under_test
    genome_assembly = int(os.environ.get("IDTRACK_TEST_ASSEMBLY", str(DB.main_assembly)))
    ignore_before, ignore_after = _default_real_graph_release_window()

    # Prefer a small, portable subset of externals from the organism's default config.
    external_dbs: tuple[str, ...]
    try:
        from idtrack._external_databases import ExternalDatabases

        ed = ExternalDatabases(
            organism=organism,
            ensembl_release=ignore_after,
            form="gene",
            local_repository=str(real_graph_cache_root),
            genome_assembly=genome_assembly,
        )
        candidates = sorted(set(ed.give_list_for_case("db")))
        preferred = [
            "HGNC Symbol",
            "MGI Symbol",
            "EntrezGene",
            "UniProtKB",
        ]
        selected: list[str] = [db for db in preferred if db in candidates]
        selected.extend([db for db in candidates if db not in selected])
        external_dbs = tuple(selected[:6]) if selected else ("EntrezGene",)
    except Exception:
        external_dbs = ("EntrezGene",)

    releases = tuple(range(ignore_before, ignore_after + 1))
    spec = RealGraphSpec(
        organism=organism,
        genome_assembly=genome_assembly,
        releases=releases,
        seed_gene_ids=(
            (
                "ENSG00000141510",  # TP53
                "ENSG00000012048",  # BRCA1
                "ENSG00000139618",  # BRCA2
                "ENSG00000157764",  # BRAF
                "ENSG00000133703",  # KRAS
            )
            if organism == "homo_sapiens"
            else ()
        ),
        external_databases=external_dbs,
    )

    graph_path = (
        real_graph_cache_root
        / f"graph_{organism}_asm{genome_assembly}_min{ignore_before}_max{ignore_after}_gene_small.pickle"
    )
    try:
        return load_or_build_real_gene_graph(graph_path, spec)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Unable to build/load real gene graph: {exc}")


@pytest.fixture(scope="session")
def real_track(real_gene_graph) -> "Any":
    """`Track` instance wired to the session-scoped real graph (no DatabaseManager required)."""
    import logging

    from idtrack._track import Track

    graph, _fixtures = real_gene_graph
    graph.calculate_caches(for_test=True)
    track = Track.__new__(Track)
    track.log = logging.getLogger("track")
    track.db_manager = None
    track.graph = graph
    track.version_info = graph.graph.get("version_info")
    track._external_entrance_placeholder = {False: -1, True: 10001}
    track._external_entrance_placeholders = sorted(track._external_entrance_placeholder.values())
    track._ensure_assembly_priority_cache()
    return track


@pytest.fixture(scope="session")
def real_track_tests(real_gene_graph) -> "Any":
    """`TrackTests` instance wired to the session-scoped real graph (no DatabaseManager required)."""
    import logging

    from idtrack._track_tests import TrackTests

    graph, _fixtures = real_gene_graph
    graph.calculate_caches(for_test=True)
    tests = TrackTests.__new__(TrackTests)
    tests.log = logging.getLogger("track_tests")
    tests.db_manager = None
    tests.graph = graph
    tests.version_info = graph.graph.get("version_info")
    tests._external_entrance_placeholder = {False: -1, True: 10001}
    tests._external_entrance_placeholders = sorted(tests._external_entrance_placeholder.values())
    tests._ensure_assembly_priority_cache()
    return tests
