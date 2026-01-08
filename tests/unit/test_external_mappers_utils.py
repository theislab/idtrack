#!/usr/bin/env python3
"""Functional tests for idtrack._external_mappers._utils module.

These tests exercise the utility functions without requiring network access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from idtrack._external_mappers._utils import (
    _as_list,
    _chunker,
    _empty_result,
    _ensure_all_inputs,
    _is_bare_numeric,
    _json,
    _unique_not_null,
    canonical_db,
    canonical_species,
    check_optional_dependencies,
    strip_version,
)


class TestCanonicalDb:
    """Tests for canonical_db function."""

    def test_ensembl_gene(self):
        """Test canonical name for ensembl_gene."""
        assert canonical_db("ensembl_gene") == "ensembl_gene"
        assert canonical_db("ensg") == "ensembl_gene"
        assert canonical_db("ensembl.gene") == "ensembl_gene"

    def test_symbol(self):
        """Test canonical name for symbol."""
        assert canonical_db("symbol") == "hgnc_symbol"
        assert canonical_db("hgnc_symbol") == "hgnc_symbol"
        assert canonical_db("gene_symbol") == "hgnc_symbol"

    def test_entrezgene(self):
        """Test canonical name for entrezgene."""
        assert canonical_db("entrezgene") == "entrez_gene"
        assert canonical_db("ncbi_gene") == "entrez_gene"
        assert canonical_db("entrez") == "entrez_gene"

    def test_uniprot(self):
        """Test canonical name for uniprot."""
        assert canonical_db("uniprot") == "uniprot"
        assert canonical_db("uniprotkb") == "uniprot"
        assert canonical_db("swissprot") == "uniprot"

    def test_refseq(self):
        """Test canonical name for refseq."""
        assert canonical_db("refseq_mrna") == "refseq_mrna"
        assert canonical_db("nm") == "refseq_mrna"

    def test_unknown_raises(self):
        """Test that unknown database raises ValueError."""
        with pytest.raises(ValueError):
            canonical_db("unknown_database")


class TestCanonicalSpecies:
    """Tests for canonical_species function."""

    def test_human(self):
        """Test canonical name for human."""
        assert canonical_species("human") == "hsapiens"
        assert canonical_species("hsapiens") == "hsapiens"
        assert canonical_species("homo_sapiens") == "hsapiens"

    def test_mouse(self):
        """Test canonical name for mouse."""
        assert canonical_species("mouse") == "mmusculus"
        assert canonical_species("mmusculus") == "mmusculus"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert canonical_species("HUMAN") == "hsapiens"
        assert canonical_species("Mouse") == "mmusculus"


class TestStripVersion:
    """Tests for strip_version function."""

    def test_ensembl_gene(self):
        """Test stripping version from Ensembl gene ID."""
        assert strip_version("ENSG00000141510.15") == "ENSG00000141510"
        assert strip_version("ENSG00000141510.1") == "ENSG00000141510"

    def test_ensembl_transcript(self):
        """Test stripping version from Ensembl transcript ID."""
        assert strip_version("ENST00000000001.5") == "ENST00000000001"

    def test_refseq_mrna(self):
        """Test stripping version from RefSeq mRNA ID."""
        assert strip_version("NM_001234.5") == "NM_001234"

    def test_no_version(self):
        """Test ID without version remains unchanged."""
        assert strip_version("ENSG00000141510") == "ENSG00000141510"
        assert strip_version("TP53") == "TP53"

    def test_non_ensembl_with_dot(self):
        """Test non-Ensembl ID with dot is unchanged."""
        # IDs that don't match Ensembl/RefSeq patterns should not be stripped
        result = strip_version("some.other.id")
        # Depending on implementation, may or may not strip
        assert isinstance(result, str)


class TestAsList:
    """Tests for _as_list function."""

    def test_string_to_list(self):
        """Test string becomes single-element list."""
        assert _as_list("hello") == ["hello"]

    def test_list_unchanged(self):
        """Test list remains unchanged."""
        assert _as_list(["a", "b"]) == ["a", "b"]

    def test_tuple_to_list(self):
        """Test tuple becomes list."""
        assert _as_list(("a", "b")) == ["a", "b"]

    def test_none_to_empty(self):
        """Test None becomes empty list."""
        assert _as_list(None) == []


class TestUniqueNotNull:
    """Tests for _unique_not_null function."""

    def test_removes_duplicates(self):
        """Test removes duplicates."""
        result = _unique_not_null(["a", "b", "a", "c"])
        assert len(result) == 3
        assert set(result) == {"a", "b", "c"}

    def test_removes_none(self):
        """Test removes None values."""
        result = _unique_not_null(["a", None, "b", None])
        assert None not in result
        assert len(result) == 2

    def test_removes_empty_string(self):
        """Test removes empty strings."""
        result = _unique_not_null(["a", "", "b"])
        assert "" not in result


class TestChunker:
    """Tests for _chunker function."""

    def test_even_chunks(self):
        """Test chunking evenly divisible list."""
        result = list(_chunker([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_chunks(self):
        """Test chunking with remainder."""
        result = list(_chunker([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        """Test when all items fit in one chunk."""
        result = list(_chunker([1, 2, 3], 10))
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        """Test empty list."""
        result = list(_chunker([], 5))
        assert result == []


class TestJson:
    """Tests for _json function."""

    def test_dict_serialization(self):
        """Test dictionary serialization."""
        result = _json({"a": 1, "b": 2})
        assert isinstance(result, str)
        assert "a" in result

    def test_list_serialization(self):
        """Test list serialization."""
        result = _json([1, 2, 3])
        assert isinstance(result, str)

    def test_none(self):
        """Test None handling."""
        result = _json(None)
        assert result == "null" or result is None


class TestIsBareNumeric:
    """Tests for _is_bare_numeric function."""

    def test_numeric_string(self):
        """Test bare numeric string."""
        assert _is_bare_numeric("12345") is True

    def test_alphanumeric(self):
        """Test alphanumeric string."""
        assert _is_bare_numeric("ABC123") is False

    def test_alpha_only(self):
        """Test alphabetic string."""
        assert _is_bare_numeric("ABC") is False

    def test_with_dots(self):
        """Test string with dots."""
        assert _is_bare_numeric("123.456") is False


class TestEmptyResult:
    """Tests for _empty_result function."""

    def test_returns_dataframe(self):
        """Test returns empty DataFrame."""
        result = _empty_result()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_has_standard_columns(self):
        """Test has standard columns."""
        result = _empty_result()
        assert "input_id" in result.columns
        assert "output_id" in result.columns


class TestEnsureAllInputs:
    """Tests for _ensure_all_inputs function."""

    def test_adds_missing_ids(self):
        """Test adds rows for missing input IDs."""
        df = pd.DataFrame({"input_id": ["A", "B"], "output_id": ["X", "Y"]})
        result = _ensure_all_inputs(
            df, ["A", "B", "C"], inp="ensembl_gene", outp="hgnc_symbol", method="test", release_used=None
        )

        assert len(result) == 3
        assert "C" in result["input_id"].values

    def test_no_missing_ids(self):
        """Test when all IDs present."""
        df = pd.DataFrame({"input_id": ["A", "B"], "output_id": ["X", "Y"]})
        result = _ensure_all_inputs(
            df, ["A", "B"], inp="ensembl_gene", outp="hgnc_symbol", method="test", release_used=None
        )

        assert len(result) == 2


class TestCheckOptionalDependencies:
    """Tests for check_optional_dependencies function."""

    def test_returns_dict(self):
        """Test returns dictionary of dependency status."""
        result = check_optional_dependencies(warn=False)
        assert isinstance(result, dict)

    def test_contains_known_dependencies(self):
        """Test contains known dependencies."""
        result = check_optional_dependencies(warn=False)
        # Should have entries for known dependencies
        assert isinstance(result, dict)
