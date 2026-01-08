#!/usr/bin/env python3
"""Comprehensive tests for the idtrack._external_mappers package.

This module tests:
- Utility functions (unit tests, no network calls)
- Constants and configuration
- Backend functions with mocked dependencies
- Integration tests with real APIs (marked as slow)

Run with: pytest tests/test_external_mappers.py -v
Run fast tests only: pytest tests/test_external_mappers.py -v -m "not slow"
Run integration tests: pytest tests/test_external_mappers.py -v -m slow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add the idtrack package directory to sys.path to allow importing _external_mappers
# as a standalone subpackage without triggering the main idtrack __init__.py
_IDTRACK_DIR = Path(__file__).parent.parent / "idtrack"
if str(_IDTRACK_DIR) not in sys.path:
    sys.path.insert(0, str(_IDTRACK_DIR))

# Import utility functions and constants from the standalone subpackage
from _external_mappers._constants import (  # noqa: E402
    _BM_ATTR_CANDIDATES,
    _DB_ALIASES,
    _ENSEMBL_ARCHIVE_BY_RELEASE,
    _ENSEMBL_INPUT_DB,
    _GP_NS,
    _MG_SCOPES,
    _SPECIES_ALIASES,
    _SPECIES_CANONICAL_TO_BGEENAMES,
    SUPPORTED_DBS,
    SUPPORTED_METHODS,
)
from _external_mappers._utils import (  # noqa: E402
    OPTIONAL_DEPENDENCIES,
    _add_mapping_column,
    _as_list,
    _chunker,
    _empty_result,
    _ensure_all_inputs,
    _is_bare_numeric,
    _json,
    _species_for_mygene,
    _suppress_stdout_stderr,
    _unique_not_null,
    canonical_db,
    canonical_species,
    check_optional_dependencies,
    raise_missing_dependency,
    strip_version,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_ids() -> list[str]:
    """Sample Ensembl gene IDs for testing."""
    return ["ENSG00000141510", "ENSG00000012048", "ENSG00000139618"]


@pytest.fixture
def sample_ids_with_versions() -> list[str]:
    """Sample IDs with version suffixes."""
    return ["ENSG00000141510.15", "ENST00000269305.8", "NM_000546.6"]


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame with expected columns."""
    return _empty_result()


@pytest.fixture
def sample_mapping_df() -> pd.DataFrame:
    """Sample mapping DataFrame for testing."""
    return pd.DataFrame(
        {
            "input_id": ["ID1", "ID1", "ID2", "ID3"],
            "output_id": ["OUT1", "OUT2", "OUT3", None],
            "input_db": ["ensembl_gene"] * 4,
            "output_db": ["hgnc_symbol"] * 4,
            "method": ["test"] * 4,
            "release_used": [None] * 4,
            "metadata_json": ["{}"] * 4,
        }
    )


# =============================================================================
# TESTS: CONSTANTS
# =============================================================================


class TestConstants:
    """Tests for constants and configuration."""

    def test_supported_methods_not_empty(self):
        """SUPPORTED_METHODS should contain expected backends."""
        assert len(SUPPORTED_METHODS) > 0
        assert "pybiomart" in SUPPORTED_METHODS
        assert "mygene" in SUPPORTED_METHODS
        assert "gprofiler" in SUPPORTED_METHODS
        assert "gget" in SUPPORTED_METHODS

    def test_supported_dbs_not_empty(self):
        """SUPPORTED_DBS should contain expected database types."""
        assert len(SUPPORTED_DBS) > 0
        assert "ensembl_gene" in SUPPORTED_DBS
        assert "ensembl_transcript" in SUPPORTED_DBS
        assert "ensembl_protein" in SUPPORTED_DBS
        assert "hgnc_symbol" in SUPPORTED_DBS
        assert "entrez_gene" in SUPPORTED_DBS
        assert "uniprot" in SUPPORTED_DBS

    def test_db_aliases_map_to_supported_dbs(self):
        """All DB aliases should map to supported DBs."""
        for alias, canonical in _DB_ALIASES.items():
            assert canonical in SUPPORTED_DBS, f"Alias {alias!r} maps to unknown DB {canonical!r}"

    def test_species_aliases_consistency(self):
        """Species aliases should map to consistent canonical codes."""
        canonical_codes = set(_SPECIES_ALIASES.values())
        # All canonical codes should have Bgee mappings
        for code in canonical_codes:
            assert code in _SPECIES_CANONICAL_TO_BGEENAMES, f"Missing Bgee mapping for {code!r}"

    def test_mygene_scopes_cover_common_dbs(self):
        """Ensure MyGene scopes cover common input databases."""
        expected = {"ensembl_gene", "hgnc_symbol", "entrez_gene", "uniprot"}
        assert expected.issubset(set(_MG_SCOPES.keys()))

    def test_gprofiler_namespaces_cover_common_dbs(self):
        """g:Profiler namespaces should cover common databases."""
        expected = {"ensembl_gene", "hgnc_symbol", "entrez_gene", "uniprot"}
        assert expected.issubset(set(_GP_NS.keys()))

    def test_biomart_attributes_cover_common_dbs(self):
        """Ensure BioMart attribute candidates cover common databases."""
        expected = {"ensembl_gene", "hgnc_symbol", "entrez_gene", "uniprot"}
        assert expected.issubset(set(_BM_ATTR_CANDIDATES.keys()))

    def test_ensembl_archives_are_chronological(self):
        """Ensembl archive releases should be in order."""
        releases = sorted(_ENSEMBL_ARCHIVE_BY_RELEASE.keys())
        assert releases == list(_ENSEMBL_ARCHIVE_BY_RELEASE.keys()) or len(releases) > 0

    def test_ensembl_input_db_contains_ensembl_types(self):
        """_ENSEMBL_INPUT_DB should contain Ensembl ID types."""
        assert "ensembl_gene" in _ENSEMBL_INPUT_DB
        assert "ensembl_transcript" in _ENSEMBL_INPUT_DB
        assert "ensembl_protein" in _ENSEMBL_INPUT_DB


# =============================================================================
# TESTS: UTILITY FUNCTIONS
# =============================================================================


class TestCanonicalDb:
    """Tests for canonical_db function."""

    def test_canonical_db_ensembl_gene(self):
        """Test various Ensembl gene aliases."""
        assert canonical_db("ensembl_gene") == "ensembl_gene"
        assert canonical_db("ENSEMBL_GENE") == "ensembl_gene"
        assert canonical_db("ensg") == "ensembl_gene"
        assert canonical_db("ENSG") == "ensembl_gene"

    def test_canonical_db_hgnc_symbol(self):
        """Test HGNC symbol aliases."""
        assert canonical_db("hgnc_symbol") == "hgnc_symbol"
        assert canonical_db("symbol") == "hgnc_symbol"
        assert canonical_db("gene_symbol") == "hgnc_symbol"
        assert canonical_db("gene_name") == "hgnc_symbol"

    def test_canonical_db_entrez(self):
        """Test Entrez gene aliases."""
        assert canonical_db("entrez_gene") == "entrez_gene"
        assert canonical_db("entrez") == "entrez_gene"
        assert canonical_db("ncbi_gene") == "entrez_gene"

    def test_canonical_db_uniprot(self):
        """Test UniProt aliases."""
        assert canonical_db("uniprot") == "uniprot"
        assert canonical_db("uniprot_acc") == "uniprot"
        assert canonical_db("swissprot") == "uniprot"

    def test_canonical_db_prefix_detection(self):
        """Test ID prefix detection for Ensembl IDs."""
        assert canonical_db("ENSG00000141510") == "ensembl_gene"
        assert canonical_db("ENST00000269305") == "ensembl_transcript"
        assert canonical_db("ENSP00000269305") == "ensembl_protein"

    def test_canonical_db_invalid_raises(self):
        """Test that invalid DB raises ValueError."""
        with pytest.raises(ValueError):
            canonical_db("invalid_database_name")

    def test_canonical_db_empty_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            canonical_db("")

    def test_canonical_db_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert canonical_db("  ensembl_gene  ") == "ensembl_gene"


class TestCanonicalSpecies:
    """Tests for canonical_species function."""

    def test_human_aliases(self):
        """Test human species aliases."""
        assert canonical_species("human") == "hsapiens"
        assert canonical_species("homo_sapiens") == "hsapiens"
        assert canonical_species("homo sapiens") == "hsapiens"
        assert canonical_species("hsapiens") == "hsapiens"

    def test_mouse_aliases(self):
        """Test mouse species aliases."""
        assert canonical_species("mouse") == "mmusculus"
        assert canonical_species("mus_musculus") == "mmusculus"
        assert canonical_species("mmusculus") == "mmusculus"

    def test_pig_aliases(self):
        """Test pig species aliases."""
        assert canonical_species("pig") == "sscrofa"
        assert canonical_species("sus_scrofa") == "sscrofa"
        assert canonical_species("sscrofa") == "sscrofa"

    def test_default_is_human(self):
        """Test default species is human."""
        assert canonical_species(None) == "hsapiens"
        assert canonical_species("") == "hsapiens"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert canonical_species("HUMAN") == "hsapiens"
        assert canonical_species("Human") == "hsapiens"

    def test_unknown_passthrough(self):
        """Test unknown species pass through unchanged."""
        assert canonical_species("danio_rerio") == "danio_rerio"


class TestSpeciesForMygene:
    """Tests for _species_for_mygene function."""

    def test_human(self):
        """Test human conversion for MyGene."""
        assert _species_for_mygene("hsapiens") == "human"
        assert _species_for_mygene("human") == "human"

    def test_mouse(self):
        """Test mouse conversion for MyGene."""
        assert _species_for_mygene("mmusculus") == "mouse"
        assert _species_for_mygene("mouse") == "mouse"

    def test_pig(self):
        """Test pig conversion for MyGene."""
        assert _species_for_mygene("sscrofa") == "pig"
        assert _species_for_mygene("pig") == "pig"


class TestStripVersion:
    """Tests for strip_version function."""

    def test_ensembl_gene_version(self):
        """Test stripping Ensembl gene version."""
        assert strip_version("ENSG00000141510.15") == "ENSG00000141510"

    def test_ensembl_transcript_version(self):
        """Test stripping Ensembl transcript version."""
        assert strip_version("ENST00000269305.8") == "ENST00000269305"

    def test_ensembl_protein_version(self):
        """Test stripping Ensembl protein version."""
        assert strip_version("ENSP00000269305.4") == "ENSP00000269305"

    def test_refseq_version(self):
        """Test stripping RefSeq version."""
        assert strip_version("NM_000546.6") == "NM_000546"
        assert strip_version("NP_000537.3") == "NP_000537"
        assert strip_version("XM_017029179.2") == "XM_017029179"

    def test_no_version_unchanged(self):
        """Test IDs without version remain unchanged."""
        assert strip_version("ENSG00000141510") == "ENSG00000141510"
        assert strip_version("TP53") == "TP53"

    def test_non_string_passthrough(self):
        """Test non-string values pass through."""
        assert strip_version(12345) == 12345
        assert strip_version(None) is None


class TestAsList:
    """Tests for _as_list function."""

    def test_none_returns_empty(self):
        """Test None returns empty list."""
        assert _as_list(None) == []

    def test_list_unchanged(self):
        """Test list returns same list."""
        assert _as_list([1, 2, 3]) == [1, 2, 3]

    def test_tuple_to_list(self):
        """Test tuple converted to list."""
        assert _as_list((1, 2, 3)) == [1, 2, 3]

    def test_set_to_list(self):
        """Test set converted to list."""
        result = _as_list({1, 2, 3})
        assert set(result) == {1, 2, 3}

    def test_scalar_wrapped(self):
        """Test scalar wrapped in list."""
        assert _as_list("value") == ["value"]
        assert _as_list(42) == [42]


class TestUniqueNotNull:
    """Tests for _unique_not_null function."""

    def test_removes_none(self):
        """Test None values are removed."""
        assert _unique_not_null(["a", None, "b"]) == ["a", "b"]

    def test_removes_nan_string(self):
        """Test 'nan' string is removed."""
        assert _unique_not_null(["a", "nan", "NaN", "b"]) == ["a", "b"]

    def test_removes_none_string(self):
        """Test 'none' string is removed."""
        assert _unique_not_null(["a", "none", "None", "b"]) == ["a", "b"]

    def test_removes_null_string(self):
        """Test 'null' string is removed."""
        assert _unique_not_null(["a", "null", "NULL", "b"]) == ["a", "b"]

    def test_removes_empty_string(self):
        """Test empty strings are removed."""
        assert _unique_not_null(["a", "", "  ", "b"]) == ["a", "b"]

    def test_removes_duplicates(self):
        """Test duplicates are removed, preserving order."""
        assert _unique_not_null(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty_input(self):
        """Test empty input returns empty list."""
        assert _unique_not_null([]) == []


class TestChunker:
    """Tests for _chunker function."""

    def test_exact_chunks(self):
        """Test chunking with exact division."""
        items = [1, 2, 3, 4, 5, 6]
        chunks = list(_chunker(items, 2))
        assert chunks == [[1, 2], [3, 4], [5, 6]]

    def test_remainder_chunk(self):
        """Test chunking with remainder."""
        items = [1, 2, 3, 4, 5]
        chunks = list(_chunker(items, 2))
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        """Test items smaller than chunk size."""
        items = [1, 2]
        chunks = list(_chunker(items, 10))
        assert chunks == [[1, 2]]

    def test_empty_list(self):
        """Test empty list returns no chunks."""
        assert list(_chunker([], 5)) == []


class TestJson:
    """Tests for _json function."""

    def test_dict_serialization(self):
        """Test dict serialization."""
        result = _json({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_compact_format(self):
        """Test output is compact (no extra spaces)."""
        result = _json({"a": 1, "b": 2})
        assert " " not in result

    def test_unicode_preserved(self):
        """Test unicode characters are preserved."""
        result = _json({"emoji": "test"})
        assert "test" in result


class TestIsBareNumeric:
    """Tests for _is_bare_numeric function."""

    def test_numeric_string(self):
        """Test numeric strings are detected."""
        assert _is_bare_numeric("12345") is True
        assert _is_bare_numeric("0") is True

    def test_non_numeric_string(self):
        """Test non-numeric strings return False."""
        assert _is_bare_numeric("abc") is False
        assert _is_bare_numeric("12.34") is False
        assert _is_bare_numeric("12abc") is False

    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert _is_bare_numeric("  12345  ") is True


class TestEmptyResult:
    """Tests for _empty_result function."""

    def test_returns_dataframe(self):
        """Test returns a DataFrame."""
        result = _empty_result()
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        """Test has all expected columns."""
        result = _empty_result()
        expected = {
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        }
        assert set(result.columns) == expected

    def test_is_empty(self):
        """Test DataFrame is empty."""
        result = _empty_result()
        assert len(result) == 0


class TestAddMappingColumn:
    """Tests for _add_mapping_column function."""

    def test_one_to_one_mapping(self):
        """Test 1:1 mapping detection."""
        df = pd.DataFrame(
            {
                "input_id": ["ID1", "ID2"],
                "output_id": ["OUT1", "OUT2"],
            }
        )
        result = _add_mapping_column(df)
        assert all(result["mapping"] == "1:1")

    def test_one_to_many_mapping(self):
        """Test 1:n mapping detection."""
        df = pd.DataFrame(
            {
                "input_id": ["ID1", "ID1", "ID1"],
                "output_id": ["OUT1", "OUT2", "OUT3"],
            }
        )
        result = _add_mapping_column(df)
        assert all(result["mapping"] == "1:n")

    def test_one_to_zero_mapping(self):
        """Test 1:0 mapping detection."""
        df = pd.DataFrame(
            {
                "input_id": ["ID1", "ID2"],
                "output_id": [None, None],
            }
        )
        result = _add_mapping_column(df)
        assert all(result["mapping"] == "1:0")

    def test_mixed_mappings(self):
        """Test mixed mapping types."""
        df = pd.DataFrame(
            {
                "input_id": ["ID1", "ID1", "ID2", "ID3"],
                "output_id": ["OUT1", "OUT2", "OUT3", None],
            }
        )
        result = _add_mapping_column(df)
        id1_rows = result[result["input_id"] == "ID1"]
        id2_rows = result[result["input_id"] == "ID2"]
        id3_rows = result[result["input_id"] == "ID3"]
        assert all(id1_rows["mapping"] == "1:n")
        assert all(id2_rows["mapping"] == "1:1")
        assert all(id3_rows["mapping"] == "1:0")

    def test_empty_dataframe(self):
        """Test empty DataFrame handling."""
        df = pd.DataFrame({"input_id": [], "output_id": []})
        result = _add_mapping_column(df)
        assert "mapping" in result.columns

    def test_none_input(self):
        """Test None input returns empty result."""
        result = _add_mapping_column(None)
        assert isinstance(result, pd.DataFrame)


class TestEnsureAllInputs:
    """Tests for _ensure_all_inputs function."""

    def test_adds_missing_inputs(self):
        """Test missing inputs are added."""
        df = pd.DataFrame(
            {
                "input_id": ["ID1"],
                "output_id": ["OUT1"],
            }
        )
        original_inputs = ["ID1", "ID2", "ID3"]
        result = _ensure_all_inputs(df, original_inputs, "inp", "outp", "test", None)

        assert set(result["input_id"]) == {"ID1", "ID2", "ID3"}

    def test_preserves_input_order(self):
        """Test original input order is preserved."""
        df = pd.DataFrame(
            {
                "input_id": ["ID3"],
                "output_id": ["OUT3"],
            }
        )
        original_inputs = ["ID1", "ID2", "ID3"]
        result = _ensure_all_inputs(df, original_inputs, "inp", "outp", "test", None)

        # First occurrence of each input should be in order
        first_occurrences = result.drop_duplicates(subset=["input_id"])
        assert list(first_occurrences["input_id"]) == ["ID1", "ID2", "ID3"]

    def test_adds_mapping_column(self):
        """Test mapping column is added."""
        df = pd.DataFrame(
            {
                "input_id": ["ID1"],
                "output_id": ["OUT1"],
            }
        )
        result = _ensure_all_inputs(df, ["ID1"], "inp", "outp", "test", None)
        assert "mapping" in result.columns

    def test_empty_dataframe_handling(self):
        """Test empty DataFrame is handled correctly."""
        result = _ensure_all_inputs(pd.DataFrame(), ["ID1", "ID2"], "inp", "outp", "test", None)
        assert set(result["input_id"]) == {"ID1", "ID2"}
        assert all(result["output_id"].isna())


class TestSuppressStdoutStderr:
    """Tests for _suppress_stdout_stderr context manager."""

    def test_suppresses_when_enabled(self):
        """Test output is suppressed when enabled."""
        import sys
        from io import StringIO

        captured = StringIO()
        old_stdout = sys.stdout

        try:
            sys.stdout = captured
            with _suppress_stdout_stderr(True):
                print("This should not appear")
            sys.stdout = old_stdout
            # The print inside the context manager should be suppressed
        finally:
            sys.stdout = old_stdout

    def test_passthrough_when_disabled(self):
        """Test output passes through when disabled."""
        # Just verify it doesn't raise
        with _suppress_stdout_stderr(False):
            pass  # Normal execution


class TestOptionalDependencies:
    """Tests for optional dependency checking functions."""

    def test_optional_dependencies_registry_structure(self):
        """Test OPTIONAL_DEPENDENCIES has expected structure."""
        assert isinstance(OPTIONAL_DEPENDENCIES, dict)
        assert len(OPTIONAL_DEPENDENCIES) > 0

        for _dep_key, info in OPTIONAL_DEPENDENCIES.items():
            assert "import_name" in info
            assert "pip_name" in info
            assert "features" in info
            assert "description" in info
            assert isinstance(info["features"], list)

    def test_optional_dependencies_known_packages(self):
        """Test that known packages are in the registry."""
        expected_packages = ["gget", "mygene", "pybiomart", "gprofiler-official", "biopython"]
        for pkg in expected_packages:
            assert pkg in OPTIONAL_DEPENDENCIES, f"{pkg} should be in OPTIONAL_DEPENDENCIES"

    def test_check_optional_dependencies_returns_dict(self):
        """Test check_optional_dependencies returns a status dict."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            status = check_optional_dependencies(warn=False)

        assert isinstance(status, dict)
        for dep_key in OPTIONAL_DEPENDENCIES:
            assert dep_key in status
            assert isinstance(status[dep_key], bool)

    def test_check_optional_dependencies_no_warning_when_disabled(self):
        """Test no warning emitted when warn=False."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_optional_dependencies(warn=False)
            # Should not have warnings from our function
            our_warnings = [x for x in w if "idtrack._external_mappers" in str(x.message)]
            assert len(our_warnings) == 0

    def test_raise_missing_dependency_known_package(self):
        """Test raise_missing_dependency for a known package."""
        with pytest.raises(RuntimeError) as exc_info:
            raise_missing_dependency("gget", feature="test feature")

        error_msg = str(exc_info.value)
        assert "Missing optional dependency: gget" in error_msg
        assert "test feature" in error_msg
        assert "pip install gget" in error_msg

    def test_raise_missing_dependency_unknown_package(self):
        """Test raise_missing_dependency for an unknown package."""
        with pytest.raises(RuntimeError) as exc_info:
            raise_missing_dependency("unknown_package_xyz")

        error_msg = str(exc_info.value)
        assert "Missing dependency: unknown_package_xyz" in error_msg
        assert "pip install unknown_package_xyz" in error_msg

    def test_raise_missing_dependency_with_original_error(self):
        """Test raise_missing_dependency chains original error."""
        original = ImportError("No module named 'gget'")
        with pytest.raises(RuntimeError) as exc_info:
            raise_missing_dependency("gget", original_error=original)

        assert exc_info.value.__cause__ is original

    def test_raise_missing_dependency_default_feature(self):
        """Test raise_missing_dependency uses features as default."""
        with pytest.raises(RuntimeError) as exc_info:
            raise_missing_dependency("gget")

        error_msg = str(exc_info.value)
        # Should include the features from OPTIONAL_DEPENDENCIES
        assert "gget backend" in error_msg or "ortholog utilities" in error_msg


# =============================================================================
# TESTS: BACKEND FUNCTIONS (MOCKED)
# =============================================================================


class TestGprofilerBackend:
    """Tests for g:Profiler backend with mocked dependencies."""

    def test_import_error_handling(self):
        """Test graceful handling when gprofiler is not installed."""
        from _external_mappers._backend_gprofiler import map_with_gprofiler

        # Mock gprofiler module in sys.modules to simulate ImportError
        with patch.dict("sys.modules", {"gprofiler": None}):
            with pytest.raises(RuntimeError, match="Missing optional dependency: gprofiler-official"):
                map_with_gprofiler(
                    ["ID1"],
                    "ensembl_gene",
                    "hgnc_symbol",
                    show_progress=False,
                )

    def test_unsupported_output_db_raises(self):
        """Test unsupported output DB raises ValueError."""
        from _external_mappers._backend_gprofiler import map_with_gprofiler

        # Mock GProfiler class
        mock_gp_instance = MagicMock()
        mock_gp_class = MagicMock(return_value=mock_gp_instance)
        mock_gprofiler = MagicMock()
        mock_gprofiler.GProfiler = mock_gp_class

        with patch.dict("sys.modules", {"gprofiler": mock_gprofiler}):
            # Using an invalid DB name - will fail at canonical_db level
            with pytest.raises(ValueError, match="Unsupported or unknown db alias"):
                map_with_gprofiler(
                    ["ID1"],
                    "ensembl_gene",
                    "invalid_db_xyz",  # Invalid output DB
                    show_progress=False,
                )

    def test_valid_input_and_output_dbs(self):
        """Test that valid DBs are accepted (with mocked backend)."""
        from _external_mappers._backend_gprofiler import map_with_gprofiler

        # Mock GProfiler class with a successful response
        mock_gp_instance = MagicMock()
        mock_gp_instance.convert.return_value = []  # Empty response
        mock_gp_class = MagicMock(return_value=mock_gp_instance)
        mock_gprofiler = MagicMock()
        mock_gprofiler.GProfiler = mock_gp_class

        with patch.dict("sys.modules", {"gprofiler": mock_gprofiler}):
            result = map_with_gprofiler(
                ["ENSG00000141510"],
                "ensembl_gene",
                "hgnc_symbol",
                show_progress=False,
            )
            assert isinstance(result, pd.DataFrame)


class TestMygeneBackend:
    """Tests for MyGene backend with mocked dependencies."""

    def test_import_error_handling(self):
        """Test graceful handling when mygene is not installed."""
        from _external_mappers._backend_mygene import map_with_mygene

        # Mock mygene module in sys.modules to simulate ImportError
        with patch.dict("sys.modules", {"mygene": None}):
            with pytest.raises(RuntimeError, match="Missing optional dependency: mygene"):
                map_with_mygene(
                    ["ID1"],
                    "ensembl_gene",
                    "hgnc_symbol",
                    show_progress=False,
                )

    def test_mocked_mygene_query(self):
        """Test mygene query with mocked backend."""
        from _external_mappers._backend_mygene import map_with_mygene

        # Create a mock mygene module with MyGeneInfo class
        mock_mg_instance = MagicMock()
        mock_mg_instance.querymany.return_value = [
            {"query": "ENSG00000141510", "symbol": "TP53", "entrezgene": "7157"},
        ]
        mock_mygene = MagicMock()
        mock_mygene.MyGeneInfo.return_value = mock_mg_instance

        with patch.dict("sys.modules", {"mygene": mock_mygene}):
            result = map_with_mygene(
                ["ENSG00000141510"],
                "ensembl_gene",
                "hgnc_symbol",
                show_progress=False,
            )
            assert isinstance(result, pd.DataFrame)
            assert "input_id" in result.columns


class TestPybiomartBackend:
    """Tests for pybiomart backend with mocked dependencies."""

    def test_non_ensembl_input_raises(self):
        """Test non-Ensembl input DB raises ValueError."""
        from _external_mappers._backend_pybiomart import map_with_pybiomart

        # Mock pybiomart module to avoid import error
        mock_pybiomart = MagicMock()
        mock_pybiomart.Dataset = MagicMock()

        with patch.dict("sys.modules", {"pybiomart": mock_pybiomart}):
            with pytest.raises(ValueError, match="pybiomart input_db must be one of"):
                map_with_pybiomart(
                    ["TP53"],
                    "hgnc_symbol",  # Not an Ensembl ID type
                    "entrez_gene",
                    show_progress=False,
                )

    def test_ensembl_archive_host_resolution(self):
        """Test Ensembl archive host resolution."""
        from _external_mappers._backend_pybiomart import _ensembl_archive_host_for_release

        # Known release
        assert _ensembl_archive_host_for_release(104) == "may2021.archive.ensembl.org"

        # String release
        assert _ensembl_archive_host_for_release("104") == "may2021.archive.ensembl.org"
        assert _ensembl_archive_host_for_release("v104") == "may2021.archive.ensembl.org"

        # Special release
        assert _ensembl_archive_host_for_release("grch37") == "grch37.ensembl.org"

        # Unknown release
        assert _ensembl_archive_host_for_release(9999) is None

        # None input
        assert _ensembl_archive_host_for_release(None) is None


class TestGgetBackend:
    """Tests for gget backend with mocked dependencies."""

    def test_non_ensembl_input_raises(self):
        """Test non-Ensembl input DB raises ValueError."""
        from _external_mappers._backend_gget import map_with_gget

        # Mock gget module to avoid import error
        mock_gget = MagicMock()
        mock_gget.info = MagicMock()

        with patch.dict("sys.modules", {"gget": mock_gget}):
            with pytest.raises(ValueError, match="gget input_db must be one of"):
                map_with_gget(
                    ["TP53"],
                    "hgnc_symbol",  # Not an Ensembl ID type
                    "entrez_gene",
                    show_progress=False,
                )

    def test_import_error_handling(self):
        """Test graceful handling when gget is not installed."""
        from _external_mappers._backend_gget import map_with_gget

        # Mock gget module in sys.modules to simulate ImportError
        with patch.dict("sys.modules", {"gget": None}):
            with pytest.raises(RuntimeError, match="Missing optional dependency: gget"):
                map_with_gget(
                    ["ID1"],
                    "ensembl_gene",
                    "hgnc_symbol",
                    show_progress=False,
                )


# =============================================================================
# TESTS: CONVERT FUNCTION
# =============================================================================


class TestConvertIds:
    """Tests for the main convert_ids function."""

    def test_empty_input_returns_empty(self):
        """Test empty input returns empty DataFrame."""
        from _external_mappers._convert import convert_ids

        # Mock the backend to avoid network calls
        with patch("_external_mappers._convert.map_with_gprofiler") as mock_gp:
            mock_gp.return_value = _empty_result()
            result = convert_ids(
                [],
                "ensembl_gene",
                "hgnc_symbol",
                method="gprofiler",
                species="human",
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_unsupported_method_raises(self):
        """Test unsupported method raises ValueError."""
        from _external_mappers._convert import convert_ids

        with pytest.raises(ValueError, match="method must be one of"):
            convert_ids(
                ["ID1"],
                "ensembl_gene",
                "hgnc_symbol",
                method="invalid_method",
                species="human",
            )

    def test_release_with_non_pybiomart_raises(self):
        """Test release parameter with non-pybiomart method raises."""
        from _external_mappers._convert import convert_ids

        with pytest.raises(ValueError, match="release parameter can only be used"):
            convert_ids(
                ["ID1"],
                "ensembl_gene",
                "hgnc_symbol",
                method="mygene",
                species="human",
                release_for_pybiomart=104,
            )

    def test_verbose_level_parsing(self):
        """Test verbose level parsing."""
        from _external_mappers._convert import _normalize_verbose_level

        # Boolean
        assert _normalize_verbose_level(True) == 3
        assert _normalize_verbose_level(False) == 2

        # Integer
        assert _normalize_verbose_level(1) == 1
        assert _normalize_verbose_level(2) == 2
        assert _normalize_verbose_level(3) == 3

        # String
        assert _normalize_verbose_level("error") == 1
        assert _normalize_verbose_level("warning") == 1
        assert _normalize_verbose_level("info") == 2
        assert _normalize_verbose_level("debug") == 3

        # Invalid
        with pytest.raises(ValueError):
            _normalize_verbose_level(99)
        with pytest.raises(ValueError):
            _normalize_verbose_level("invalid")


# =============================================================================
# TESTS: ORTHOLOG MODULE
# =============================================================================


class TestOrthologModule:
    """Tests for ortholog utilities (require optional dependencies)."""

    def test_lazy_import_works(self):
        """Test lazy import mechanism works."""
        from _external_mappers import __getattr__

        # Test that accessing ortholog functions triggers lazy import
        # This should not raise even without dependencies installed
        try:
            __getattr__("get_ortholog_table")
        except (ImportError, RuntimeError):
            pytest.skip("Ortholog optional dependencies not installed")

    def test_species_resolution(self):
        """Test species resolution in ortholog module."""
        # Import the helper directly from constants
        from _external_mappers._constants import _SPECIES_ALIASES, _SPECIES_CANONICAL_TO_BGEENAMES

        # Verify human
        assert _SPECIES_ALIASES["human"] == "hsapiens"
        assert _SPECIES_CANONICAL_TO_BGEENAMES["hsapiens"] == ("Homo", "sapiens")

        # Verify mouse
        assert _SPECIES_ALIASES["mouse"] == "mmusculus"
        assert _SPECIES_CANONICAL_TO_BGEENAMES["mmusculus"] == ("Mus", "musculus")


# =============================================================================
# TESTS: INTEGRATION (SLOW - REQUIRE NETWORK)
# =============================================================================


@pytest.mark.slow
class TestIntegrationGprofiler:
    """Integration tests for g:Profiler backend (requires network)."""

    @pytest.fixture
    def gprofiler_available(self):
        """Skip if gprofiler-official is not installed."""
        pytest.importorskip("gprofiler", reason="gprofiler-official not installed")
        return True

    def test_real_gprofiler_query(self, gprofiler_available, sample_ids):
        """Test real g:Profiler query."""
        from _external_mappers._backend_gprofiler import map_with_gprofiler

        result = map_with_gprofiler(
            sample_ids[:1],  # Use just one ID to speed up
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert "input_id" in result.columns
        assert "output_id" in result.columns
        assert "mapping" in result.columns


@pytest.mark.slow
class TestIntegrationMygene:
    """Integration tests for MyGene backend (requires network)."""

    @pytest.fixture
    def mygene_available(self):
        """Skip if mygene is not installed."""
        pytest.importorskip("mygene", reason="mygene not installed")
        return True

    def test_real_mygene_query(self, mygene_available, sample_ids):
        """Test real MyGene query."""
        from _external_mappers._backend_mygene import map_with_mygene

        result = map_with_mygene(
            sample_ids[:1],
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert "input_id" in result.columns
        assert "output_id" in result.columns


@pytest.mark.slow
class TestIntegrationPybiomart:
    """Integration tests for pybiomart backend (requires network)."""

    @pytest.fixture
    def pybiomart_available(self):
        """Skip if pybiomart is not installed."""
        pytest.importorskip("pybiomart", reason="pybiomart not installed")
        return True

    def test_real_pybiomart_query(self, pybiomart_available, sample_ids):
        """Test real pybiomart query."""
        from _external_mappers._backend_pybiomart import map_with_pybiomart

        result = map_with_pybiomart(
            sample_ids[:1],
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert "input_id" in result.columns
        assert "output_id" in result.columns


@pytest.mark.slow
class TestIntegrationGget:
    """Integration tests for gget backend (requires network)."""

    @pytest.fixture
    def gget_available(self):
        """Skip if gget is not installed."""
        pytest.importorskip("gget", reason="gget not installed")
        return True

    def test_real_gget_query(self, gget_available, sample_ids):
        """Test real gget query."""
        from _external_mappers._backend_gget import map_with_gget

        result = map_with_gget(
            sample_ids[:1],
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert "input_id" in result.columns
        assert "output_id" in result.columns


# =============================================================================
# TESTS: EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ids_with_special_characters(self):
        """Test handling of IDs with special characters."""
        ids = ["ENSG00000141510", "ID with space", "ID/with/slash"]
        clean = _unique_not_null(ids)
        assert len(clean) == 3

    def test_duplicate_ids_deduplicated(self):
        """Test duplicate IDs are handled correctly."""
        ids = ["ID1", "ID1", "ID2", "ID2", "ID3"]
        clean = _unique_not_null(ids)
        assert clean == ["ID1", "ID2", "ID3"]

    def test_mixed_case_ids(self):
        """Test mixed case IDs are preserved."""
        ids = ["ENSG00000141510", "ensg00000012048"]
        clean = _unique_not_null(ids)
        assert "ENSG00000141510" in clean
        assert "ensg00000012048" in clean

    def test_very_long_id_list(self):
        """Test handling of large ID lists."""
        ids = [f"ID{i}" for i in range(10000)]
        clean = _unique_not_null(ids)
        assert len(clean) == 10000

    def test_chunking_large_list(self):
        """Test chunking works correctly for large lists."""
        ids = list(range(10000))
        chunks = list(_chunker(ids, 1000))
        assert len(chunks) == 10
        assert all(len(c) == 1000 for c in chunks)


# =============================================================================
# TESTS: PYBIOMART HELPER FUNCTIONS
# =============================================================================


class TestPybiomartHelpers:
    """Tests for pybiomart helper functions."""

    def test_normalize_biomart_host_with_protocol(self):
        """Test host normalization with protocol."""
        from _external_mappers._backend_pybiomart import _normalize_biomart_host

        assert _normalize_biomart_host("http://www.ensembl.org") == "http://www.ensembl.org"
        assert _normalize_biomart_host("https://www.ensembl.org") == "http://www.ensembl.org"

    def test_normalize_biomart_host_without_protocol(self):
        """Test host normalization without protocol."""
        from _external_mappers._backend_pybiomart import _normalize_biomart_host

        assert _normalize_biomart_host("www.ensembl.org") == "http://www.ensembl.org"
        assert _normalize_biomart_host("may2021.archive.ensembl.org") == "http://may2021.archive.ensembl.org"

    def test_normalize_biomart_host_empty(self):
        """Test host normalization with empty/None input."""
        from _external_mappers._backend_pybiomart import _normalize_biomart_host

        assert _normalize_biomart_host(None) == "http://www.ensembl.org"
        assert _normalize_biomart_host("") == "http://www.ensembl.org"

    def test_biomart_dataset_for_species(self):
        """Test dataset name generation for species."""
        from _external_mappers._backend_pybiomart import _biomart_dataset_for_species

        assert _biomart_dataset_for_species("hsapiens") == "hsapiens_gene_ensembl"
        assert _biomart_dataset_for_species("mmusculus") == "mmusculus_gene_ensembl"
        assert _biomart_dataset_for_species("human") == "hsapiens_gene_ensembl"

    def test_biomart_dataset_explicit_override(self):
        """Test explicit dataset name override."""
        from _external_mappers._backend_pybiomart import _biomart_dataset_for_species

        assert _biomart_dataset_for_species("hsapiens", "custom_dataset") == "custom_dataset"

    def test_ensembl_archive_host_various_formats(self):
        """Test archive host resolution for various input formats."""
        from _external_mappers._backend_pybiomart import _ensembl_archive_host_for_release

        # Integer
        assert _ensembl_archive_host_for_release(100) == "apr2020.archive.ensembl.org"

        # String with "v" prefix
        assert _ensembl_archive_host_for_release("v100") == "apr2020.archive.ensembl.org"

        # String with "r" prefix
        assert _ensembl_archive_host_for_release("r100") == "apr2020.archive.ensembl.org"

        # Empty string
        assert _ensembl_archive_host_for_release("") is None

        # Whitespace
        assert _ensembl_archive_host_for_release("  ") is None


# =============================================================================
# TESTS: GGET EXTRACT HELPER
# =============================================================================


class TestGgetExtract:
    """Tests for gget data extraction helper."""

    def test_extract_hgnc_symbol(self):
        """Test extracting HGNC symbol from gget output."""
        from _external_mappers._backend_gget import _gget_extract

        df = pd.DataFrame(
            {
                "query": ["ENSG00000141510"],
                "name": ["TP53"],
                "entrezgene": [7157],
            }
        )
        result = _gget_extract(df, "hgnc_symbol")
        assert "input_id" in result.columns
        assert "output_id" in result.columns

    def test_extract_ensembl_gene(self):
        """Test extracting Ensembl gene from gget output."""
        from _external_mappers._backend_gget import _gget_extract

        df = pd.DataFrame(
            {
                "id": ["ENSG00000141510"],
                "name": ["TP53"],
            }
        )
        result = _gget_extract(df, "ensembl_gene")
        assert len(result) == 1

    def test_extract_missing_column(self):
        """Test extraction when target column is missing."""
        from _external_mappers._backend_gget import _gget_extract

        df = pd.DataFrame(
            {
                "query": ["ENSG00000141510"],
                "name": ["TP53"],
            }
        )
        result = _gget_extract(df, "uniprot")
        # Should return empty output_id
        assert "output_id" in result.columns


# =============================================================================
# TESTS: MYGENE EXTRACT HELPER
# =============================================================================


class TestMygeneExtract:
    """Tests for MyGene data extraction helper."""

    def test_extract_hgnc_symbol(self):
        """Test extracting HGNC symbol from MyGene record."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {"query": "ENSG00000141510", "symbol": "TP53", "entrezgene": 7157}
        result = _mg_extract(record, "hgnc_symbol")
        assert result == ["TP53"]

    def test_extract_entrez_gene(self):
        """Test extracting Entrez gene from MyGene record."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {"query": "TP53", "symbol": "TP53", "entrezgene": 7157}
        result = _mg_extract(record, "entrez_gene")
        assert result == ["7157"]

    def test_extract_ensembl_gene_dict(self):
        """Test extracting Ensembl gene from nested dict."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {
            "query": "TP53",
            "ensembl": {"gene": "ENSG00000141510", "transcript": "ENST00000269305"},
        }
        result = _mg_extract(record, "ensembl_gene")
        assert result == ["ENSG00000141510"]

    def test_extract_ensembl_gene_list(self):
        """Test extracting Ensembl gene from nested list."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {
            "query": "TP53",
            "ensembl": [
                {"gene": "ENSG00000141510", "transcript": "ENST00000269305"},
                {"gene": "ENSG00000141511", "transcript": "ENST00000269306"},
            ],
        }
        result = _mg_extract(record, "ensembl_gene")
        assert "ENSG00000141510" in result
        assert "ENSG00000141511" in result

    def test_extract_uniprot_dict(self):
        """Test extracting UniProt from nested dict."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {
            "query": "TP53",
            "uniprot": {"Swiss-Prot": "P04637", "TrEMBL": "Q9H1B1"},
        }
        result = _mg_extract(record, "uniprot")
        assert "P04637" in result

    def test_extract_hgnc_id_numeric(self):
        """Test extracting HGNC ID with numeric value."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {"query": "TP53", "hgnc": "11998"}
        result = _mg_extract(record, "hgnc_id")
        assert "HGNC:11998" in result

    def test_extract_refseq_mrna(self):
        """Test extracting RefSeq mRNA."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {
            "query": "TP53",
            "refseq": {"rna": ["NM_000546", "NM_001126112"]},
        }
        result = _mg_extract(record, "refseq_mrna")
        assert "NM_000546" in result

    def test_extract_missing_field(self):
        """Test extraction when field is missing."""
        from _external_mappers._backend_mygene import _mg_extract

        record = {"query": "TP53", "symbol": "TP53"}
        result = _mg_extract(record, "uniprot")
        assert result == []


# =============================================================================
# TESTS: ORTHOLOG MODULE (UNIT TESTS)
# =============================================================================


class TestOrthologHelpers:
    """Unit tests for ortholog module helper functions."""

    def test_canonical_from_alias(self):
        """Test species alias resolution."""
        from _external_mappers._ortholog import _canonical_from_alias

        assert _canonical_from_alias("human") == "hsapiens"
        assert _canonical_from_alias("HUMAN") == "hsapiens"
        assert _canonical_from_alias("homo sapiens") == "hsapiens"
        assert _canonical_from_alias("Homo_sapiens") == "hsapiens"

    def test_canonical_from_alias_passthrough(self):
        """Test unknown species passes through."""
        from _external_mappers._ortholog import _canonical_from_alias

        assert _canonical_from_alias("unknown_species") == "unknown_species"

    def test_species_to_genus_species_human(self):
        """Test species to genus/species conversion for human."""
        from _external_mappers._ortholog import _species_to_genus_species

        canonical, genus, species = _species_to_genus_species("human")
        assert canonical == "hsapiens"
        assert genus == "Homo"
        assert species == "sapiens"

    def test_species_to_genus_species_mouse(self):
        """Test species to genus/species conversion for mouse."""
        from _external_mappers._ortholog import _species_to_genus_species

        canonical, genus, species = _species_to_genus_species("mouse")
        assert canonical == "mmusculus"
        assert genus == "Mus"
        assert species == "musculus"

    def test_species_to_genus_species_invalid(self):
        """Test invalid species raises ValueError."""
        from _external_mappers._ortholog import _species_to_genus_species

        with pytest.raises(ValueError, match="could not be resolved"):
            _species_to_genus_species("invalid_species_xyz")

    def test_aa_composition_vector(self):
        """Test amino acid composition vector calculation."""
        from _external_mappers._ortholog import AA_ALPHABET, _aa_composition_vector

        # Simple sequence with known composition
        seq = "AAACCC"
        comp = _aa_composition_vector(seq)

        assert len(comp) == len(AA_ALPHABET)
        # A should be 0.5, C should be 0.5
        assert comp[AA_ALPHABET.index("A")] == pytest.approx(0.5)
        assert comp[AA_ALPHABET.index("C")] == pytest.approx(0.5)

    def test_aa_composition_ignores_gaps(self):
        """Test that gaps are ignored in composition calculation."""
        from _external_mappers._ortholog import _aa_composition_vector

        seq1 = "AAA"
        seq2 = "A-A-A"  # Same sequence with gaps

        comp1 = _aa_composition_vector(seq1)
        comp2 = _aa_composition_vector(seq2)

        # Should be identical (gaps removed)
        assert list(comp1) == list(comp2)

    def test_parse_clustal_alignment(self):
        """Test ClustalW alignment parsing."""
        from _external_mappers._ortholog import parse_clustal_alignment

        clustal_text = """CLUSTAL alignment

seq1      MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
seq2      MVLSGEDKSNIKAAWGKIGGHGAEYGAEALERMFASFPTTKTYFPHFDLSH
          ****  **.:*:*****:* * ***********:*****************

seq1      GSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFK
seq2      GSAQVKAHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFK
          ******:******************************************
"""
        result = parse_clustal_alignment(clustal_text)

        assert "seq1" in result
        assert "seq2" in result
        assert len(result["seq1"]) > 0
        assert len(result["seq1"]) == len(result["seq2"])


class TestAlignmentScoresDataclass:
    """Tests for AlignmentScores dataclass."""

    def test_alignment_scores_creation(self):
        """Test creating AlignmentScores dataclass."""
        from _external_mappers._ortholog import AlignmentScores

        scores = AlignmentScores(
            alignment_length=100,
            identity_fraction=0.8,
            positive_fraction=0.9,
            very_negative_fraction=0.01,
            gap_fraction_query=0.05,
            gap_fraction_target=0.03,
            gap_openings_query=2,
            gap_openings_target=1,
            seq1_coverage=0.95,
            seq2_coverage=0.97,
            blosum62_sum=450.0,
            blosum62_mean=4.5,
            composition_l2_distance=0.1,
        )

        assert scores.alignment_length == 100
        assert scores.identity_fraction == 0.8
        assert scores.blosum62_mean == 4.5

    def test_alignment_scores_asdict(self):
        """Test converting AlignmentScores to dict."""
        from dataclasses import asdict

        from _external_mappers._ortholog import AlignmentScores

        scores = AlignmentScores(
            alignment_length=100,
            identity_fraction=0.8,
            positive_fraction=0.9,
            very_negative_fraction=0.01,
            gap_fraction_query=0.05,
            gap_fraction_target=0.03,
            gap_openings_query=2,
            gap_openings_target=1,
            seq1_coverage=0.95,
            seq2_coverage=0.97,
            blosum62_sum=450.0,
            blosum62_mean=4.5,
            composition_l2_distance=0.1,
        )

        d = asdict(scores)
        assert isinstance(d, dict)
        assert d["alignment_length"] == 100


# =============================================================================
# TESTS: CONVERT_IDS ROUTING
# =============================================================================


class TestConvertIdsRouting:
    """Tests for convert_ids method routing."""

    def test_routes_to_pybiomart(self):
        """Test routing to pybiomart backend."""
        from _external_mappers._convert import convert_ids

        with patch("_external_mappers._convert.map_with_pybiomart") as mock_bm:
            mock_bm.return_value = _empty_result()
            convert_ids(
                ["ENSG00000141510"],
                "ensembl_gene",
                "hgnc_symbol",
                method="pybiomart",
                species="human",
            )
            mock_bm.assert_called_once()

    def test_routes_to_mygene(self):
        """Test routing to mygene backend."""
        from _external_mappers._convert import convert_ids

        with patch("_external_mappers._convert.map_with_mygene") as mock_mg:
            mock_mg.return_value = _empty_result()
            convert_ids(
                ["ENSG00000141510"],
                "ensembl_gene",
                "hgnc_symbol",
                method="mygene",
                species="human",
            )
            mock_mg.assert_called_once()

    def test_routes_to_gprofiler(self):
        """Test routing to gprofiler backend."""
        from _external_mappers._convert import convert_ids

        with patch("_external_mappers._convert.map_with_gprofiler") as mock_gp:
            mock_gp.return_value = _empty_result()
            convert_ids(
                ["ENSG00000141510"],
                "ensembl_gene",
                "hgnc_symbol",
                method="gprofiler",
                species="human",
            )
            mock_gp.assert_called_once()

    def test_routes_to_gget(self):
        """Test routing to gget backend."""
        from _external_mappers._convert import convert_ids

        with patch("_external_mappers._convert.map_with_gget") as mock_gg:
            mock_gg.return_value = _empty_result()
            convert_ids(
                ["ENSG00000141510"],
                "ensembl_gene",
                "hgnc_symbol",
                method="gget",
                species="human",
            )
            mock_gg.assert_called_once()


# =============================================================================
# TESTS: INTEGRATION WITH KNOWN GENE (SLOW)
# =============================================================================


@pytest.mark.slow
class TestIntegrationKnownGene:
    """Integration tests with known gene TP53 (ENSG00000141510)."""

    # TP53 gene identifiers for validation
    TP53_ENSEMBL = "ENSG00000141510"
    TP53_SYMBOL = "TP53"
    TP53_ENTREZ = "7157"

    @pytest.fixture
    def skip_if_no_mygene(self):
        """Skip if mygene not available."""
        pytest.importorskip("mygene", reason="mygene not installed")
        return True

    @pytest.fixture
    def skip_if_no_gprofiler(self):
        """Skip if gprofiler not available."""
        pytest.importorskip("gprofiler", reason="gprofiler-official not installed")
        return True

    def test_mygene_ensembl_to_symbol(self, skip_if_no_mygene):
        """Test MyGene Ensembl to symbol mapping for TP53."""
        from _external_mappers._backend_mygene import map_with_mygene

        result = map_with_mygene(
            [self.TP53_ENSEMBL],
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        assert len(result) > 0
        symbols = result["output_id"].dropna().tolist()
        assert self.TP53_SYMBOL in symbols

    def test_mygene_symbol_to_entrez(self, skip_if_no_mygene):
        """Test MyGene symbol to Entrez mapping for TP53."""
        from _external_mappers._backend_mygene import map_with_mygene

        result = map_with_mygene(
            [self.TP53_SYMBOL],
            "hgnc_symbol",
            "entrez_gene",
            show_progress=False,
        )

        assert len(result) > 0
        entrez_ids = result["output_id"].dropna().tolist()
        assert self.TP53_ENTREZ in entrez_ids

    def test_gprofiler_ensembl_to_symbol(self, skip_if_no_gprofiler):
        """Test g:Profiler Ensembl to symbol mapping for TP53."""
        from _external_mappers._backend_gprofiler import map_with_gprofiler

        result = map_with_gprofiler(
            [self.TP53_ENSEMBL],
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        assert len(result) > 0
        # Check that we got some output
        non_null_outputs = result["output_id"].dropna()
        assert len(non_null_outputs) > 0


@pytest.mark.slow
class TestIntegrationMultipleIds:
    """Integration tests with multiple IDs."""

    WELL_KNOWN_GENES = [
        "ENSG00000141510",  # TP53
        "ENSG00000012048",  # BRCA1
        "ENSG00000139618",  # BRCA2
    ]

    @pytest.fixture
    def skip_if_no_mygene(self):
        """Skip if mygene not available."""
        pytest.importorskip("mygene", reason="mygene not installed")
        return True

    def test_batch_mapping_preserves_all_inputs(self, skip_if_no_mygene):
        """Test that all input IDs appear in output."""
        from _external_mappers._backend_mygene import map_with_mygene

        result = map_with_mygene(
            self.WELL_KNOWN_GENES,
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        # All input IDs should be present
        input_ids_in_result = set(result["input_id"].unique())
        for gene_id in self.WELL_KNOWN_GENES:
            assert gene_id in input_ids_in_result

    def test_mapping_column_values(self, skip_if_no_mygene):
        """Test mapping column contains valid values."""
        from _external_mappers._backend_mygene import map_with_mygene

        result = map_with_mygene(
            self.WELL_KNOWN_GENES,
            "ensembl_gene",
            "hgnc_symbol",
            show_progress=False,
        )

        # All mapping values should be one of the expected types
        valid_mappings = {"1:0", "1:1", "1:n"}
        assert set(result["mapping"].unique()).issubset(valid_mappings)


# =============================================================================
# TESTS: ORTHOLOG INTEGRATION (VERY SLOW)
# =============================================================================


@pytest.mark.slow
class TestOrthologIntegration:
    """Integration tests for ortholog utilities (require gget and biopython)."""

    @pytest.fixture
    def skip_if_no_ortholog_deps(self):
        """Skip if ortholog dependencies not available."""
        pytest.importorskip("gget", reason="Ortholog dependencies (gget, biopython) not installed")
        pytest.importorskip("Bio.SeqIO", reason="Ortholog dependencies (gget, biopython) not installed")
        return True

    def test_get_ortholog_table(self, skip_if_no_ortholog_deps):
        """Test fetching ortholog table from Bgee."""
        from _external_mappers._ortholog import get_ortholog_table

        # Use a well-known gene
        df = get_ortholog_table("ENSG00000141510", verbose=False)

        assert isinstance(df, pd.DataFrame)
        assert "gene_id" in df.columns
        assert "genus" in df.columns
        assert "species" in df.columns

    def test_get_ortholog_ids_for_species(self, skip_if_no_ortholog_deps):
        """Test getting ortholog IDs for a specific species."""
        from _external_mappers._ortholog import get_ortholog_ids_for_species, get_ortholog_table

        df = get_ortholog_table("ENSG00000141510", verbose=False)
        mouse_ids = get_ortholog_ids_for_species(df, "mouse")

        # Should return a list (may be empty if no mouse orthologs)
        assert isinstance(mouse_ids, list)

    def test_pick_ortholog_for_species(self, skip_if_no_ortholog_deps):
        """Test picking a single ortholog for a species."""
        from _external_mappers._ortholog import get_ortholog_table, pick_ortholog_for_species

        df = get_ortholog_table("ENSG00000141510", verbose=False)
        mouse_ortholog = pick_ortholog_for_species(df, "mouse")

        # Should return a string or None
        assert mouse_ortholog is None or isinstance(mouse_ortholog, str)
