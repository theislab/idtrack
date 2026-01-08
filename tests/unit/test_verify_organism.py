#!/usr/bin/env python3
"""
Unit tests for idtrack._verify_organism module.

Tests the VerifyOrganism class for organism name resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestVerifyOrganismInitialization:
    """Test VerifyOrganism initialization."""

    def test_organism_query_lowercased(self, mock_requests_session, mock_ensembl_rest_response):
        """Test organism query is converted to lowercase."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("HOMO_SAPIENS")
            assert resolver.organism_query == "homo_sapiens"

    def test_creates_dataframes(self, mock_requests_session, mock_ensembl_rest_response):
        """Test initialization creates required dataframes."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("homo_sapiens")

            assert hasattr(resolver, "name_synonyms_dataframe")
            assert hasattr(resolver, "ensembl_release_dataframe")
            assert isinstance(resolver.name_synonyms_dataframe, pd.DataFrame)
            assert isinstance(resolver.ensembl_release_dataframe, pd.DataFrame)


class TestGetFormalName:
    """Test get_formal_name method."""

    def test_returns_formal_name(self, mock_requests_session, mock_ensembl_rest_response):
        """Test get_formal_name returns canonical name."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("homo_sapiens")

            formal_name = resolver.get_formal_name()
            assert formal_name == "homo_sapiens"

    def test_resolves_common_name(self, mock_requests_session, mock_ensembl_rest_response):
        """Test resolving from common name."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("human")

            formal_name = resolver.get_formal_name()
            assert formal_name == "homo_sapiens"

    def test_raises_key_error_for_unknown(self, mock_requests_session, mock_ensembl_rest_response):
        """Test KeyError raised for unknown organism."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("unknown_organism")

            with pytest.raises(KeyError):
                resolver.get_formal_name()


class TestGetLatestRelease:
    """Test get_latest_release method."""

    def test_returns_integer_release(self, mock_requests_session, mock_ensembl_rest_response):
        """Test get_latest_release returns an integer."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("homo_sapiens")

            release = resolver.get_latest_release()
            assert isinstance(release, int)
            assert release == 110


class TestFetchOrganismAndLatestRelease:
    """Test fetch_organism_and_latest_release method."""

    def test_handles_timeout(self, mock_requests_session):
        """Test TimeoutError on connection timeout."""
        import requests

        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            session_instance.get.return_value.__enter__.side_effect = requests.exceptions.ConnectTimeout()

            from idtrack._verify_organism import VerifyOrganism

            with pytest.raises(TimeoutError):
                VerifyOrganism("homo_sapiens")

    def test_validates_core_group(self, mock_requests_session):
        """Test ValueError when organism lacks 'core' group."""
        invalid_response = {
            "species": [
                {
                    "name": "test_organism",
                    "common_name": "test",
                    "display_name": "Test",
                    "taxon_id": "12345",
                    "assembly": "Test1",
                    "accession": "GCA_000000000.0",
                    "release": "110",
                    "groups": ["funcgen"],  # Missing 'core'
                    "aliases": [],
                }
            ]
        }

        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = invalid_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism

            with pytest.raises(ValueError, match="core"):
                VerifyOrganism("test_organism")

    def test_validates_duplicate_organism(self, mock_requests_session):
        """Test ValueError for duplicate organism entries."""
        duplicate_response = {
            "species": [
                {
                    "name": "homo_sapiens",
                    "common_name": "human",
                    "display_name": "Human",
                    "taxon_id": "9606",
                    "assembly": "GRCh38",
                    "accession": "GCA_000001405.29",
                    "release": "110",
                    "groups": ["core"],
                    "aliases": [],
                },
                {
                    "name": "homo_sapiens",  # Duplicate
                    "common_name": "human",
                    "display_name": "Human 2",
                    "taxon_id": "9606",
                    "assembly": "GRCh38",
                    "accession": "GCA_000001405.29",
                    "release": "111",
                    "groups": ["core"],
                    "aliases": [],
                },
            ]
        }

        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = duplicate_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism

            with pytest.raises(ValueError, match="more than one"):
                VerifyOrganism("homo_sapiens")


class TestSynonymDataframe:
    """Test synonym dataframe structure."""

    def test_dataframe_columns(self, mock_requests_session, mock_ensembl_rest_response):
        """Test name_synonyms_dataframe has correct columns."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("homo_sapiens")

            df = resolver.name_synonyms_dataframe
            assert "synonym" in df.columns
            assert "formal_name" in df.columns
            assert "ambiguous" in df.columns

    def test_contains_multiple_synonyms(self, mock_requests_session, mock_ensembl_rest_response):
        """Test dataframe contains multiple synonyms per organism."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("homo_sapiens")

            df = resolver.name_synonyms_dataframe
            human_rows = df[df["formal_name"] == "homo_sapiens"]

            # Should have multiple synonyms (common_name, name, aliases, etc.)
            assert len(human_rows) >= 2


class TestReleaseDataframe:
    """Test release dataframe structure."""

    def test_dataframe_indexed_by_formal_name(self, mock_requests_session, mock_ensembl_rest_response):
        """Test ensembl_release_dataframe is indexed by formal_name."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("homo_sapiens")

            df = resolver.ensembl_release_dataframe
            assert "homo_sapiens" in df.index
            assert "ensembl_release" in df.columns

    def test_release_is_integer(self, mock_requests_session, mock_ensembl_rest_response):
        """Test release values are integers."""
        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = mock_ensembl_rest_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("homo_sapiens")

            df = resolver.ensembl_release_dataframe
            assert df.loc["homo_sapiens", "ensembl_release"] == 110


class TestAmbiguousSynonyms:
    """Test handling of ambiguous synonyms."""

    def test_marks_ambiguous_synonyms(self, mock_requests_session):
        """Test ambiguous synonyms are properly marked."""
        # Create response with ambiguous common name
        ambiguous_response = {
            "species": [
                {
                    "name": "species_one",
                    "common_name": "shared_name",  # Shared
                    "display_name": "Species One",
                    "taxon_id": "1",
                    "assembly": "S1",
                    "accession": "GCA_1",
                    "release": "110",
                    "groups": ["core"],
                    "aliases": [],
                },
                {
                    "name": "species_two",
                    "common_name": "shared_name",  # Shared - ambiguous
                    "display_name": "Species Two",
                    "taxon_id": "2",
                    "assembly": "S2",
                    "accession": "GCA_2",
                    "release": "110",
                    "groups": ["core"],
                    "aliases": [],
                },
            ]
        }

        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = ambiguous_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("species_one")

            df = resolver.name_synonyms_dataframe
            shared_rows = df[df["synonym"] == "shared_name"]

            # All rows with shared synonym should be marked ambiguous
            assert all(shared_rows["ambiguous"])

    def test_raises_value_error_for_ambiguous_query(self, mock_requests_session):
        """Test ValueError raised when query matches ambiguous synonym."""
        ambiguous_response = {
            "species": [
                {
                    "name": "species_one",
                    "common_name": "shared_name",
                    "display_name": "Species One",
                    "taxon_id": "1",
                    "assembly": "S1",
                    "accession": "GCA_1",
                    "release": "110",
                    "groups": ["core"],
                    "aliases": [],
                },
                {
                    "name": "species_two",
                    "common_name": "shared_name",
                    "display_name": "Species Two",
                    "taxon_id": "2",
                    "assembly": "S2",
                    "accession": "GCA_2",
                    "release": "110",
                    "groups": ["core"],
                    "aliases": [],
                },
            ]
        }

        with patch("requests.Session") as mock_session:
            session_instance = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            response = MagicMock()
            response.ok = True
            response.json.return_value = ambiguous_response
            session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
            session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

            from idtrack._verify_organism import VerifyOrganism
            resolver = VerifyOrganism("shared_name")

            with pytest.raises(ValueError, match="ambiguous"):
                resolver.get_formal_name()
