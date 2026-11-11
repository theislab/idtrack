#!/usr/bin/env python3
"""Unit tests for idtrack._verify_organism module.

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


def _resolver_for(species: list[dict], query):
    """Build a VerifyOrganism for *query* against a mocked `/info/species` payload."""
    with patch("requests.Session") as mock_session:
        session_instance = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=session_instance)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        response = MagicMock()
        response.ok = True
        response.json.return_value = {"species": species}
        session_instance.get.return_value.__enter__ = MagicMock(return_value=response)
        session_instance.get.return_value.__exit__ = MagicMock(return_value=False)

        from idtrack._verify_organism import VerifyOrganism

        return VerifyOrganism(query)


def _species(name, common, taxon, assembly, aliases=(), display=None):
    return {
        "name": name,
        "common_name": common,
        "display_name": display or common.capitalize(),
        "taxon_id": taxon,
        "assembly": assembly,
        "accession": f"GCA_{assembly}",
        "release": "116",
        "groups": ["core"],
        "aliases": list(aliases),
    }


# Shaped like Ensembl's payload: aliases only on the reference entries, breeds share the common name and taxon ID.
SPECIES = [
    _species("homo_sapiens", "human", "9606", "GRCh38", aliases=["hsapiens", "homo sapiens", "9606", "human"]),
    _species("sus_scrofa", "pig", "9823", "Sscrofa11.1", aliases=["sscrofa", "pig", "9823", "sus scrofa"]),
    _species("sus_scrofa_bamei", "pig", "9823", "Bamei_pig_v1", display="Pig - Bamei"),
    _species("sus_scrofa_meishan", "pig", "9823", "Meishan_pig_v1", display="Pig - Meishan"),
    _species("macaca_mulatta", "Macaque", "9544", "Mmul_10"),
    _species("macaca_fascicularis", "Crab-eating macaque", "9541", "Macaca_fascicularis_6.0"),
    _species("mus_musculus_aj", "house mouse", "10090", "A_J_v3", display="Mouse A/J"),
    _species("mus_musculus_balbcj", "house mouse", "10090", "BALB_cJ_v3", display="Mouse BALB/cJ"),
]


class TestResolution:
    """Case-insensitive matching, taxon IDs as int or str, and ambiguity rules."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("macaca_mulatta", "macaca_mulatta"),
            ("Macaca mulatta", "macaca_mulatta"),
            ("MACACA_MULATTA", "macaca_mulatta"),
            ("Mmul_10", "macaca_mulatta"),
            ("macaque", "macaca_mulatta"),
            (9544, "macaca_mulatta"),
            ("9541", "macaca_fascicularis"),
            ("Crab-eating macaque", "macaca_fascicularis"),
            ("Macaca_fascicularis_6.0", "macaca_fascicularis"),
            ("Homo Sapiens", "homo_sapiens"),
            ("HSAPIENS", "homo_sapiens"),
            (9606, "homo_sapiens"),
            # A taxon ID shared with breeds resolves to the species-level entry.
            (9823, "sus_scrofa"),
            ("9823", "sus_scrofa"),
            # A formal name is never ambiguous.
            ("sus_scrofa_bamei", "sus_scrofa_bamei"),
            ("Sus scrofa", "sus_scrofa"),
        ],
    )
    def test_resolves(self, query, expected):
        """Resolves."""
        assert _resolver_for(SPECIES, query).get_formal_name() == expected

    def test_ambiguous_common_name_lists_candidates(self):
        """Ambiguous common name lists candidates."""
        resolver = _resolver_for(SPECIES, "house mouse")
        with pytest.raises(ValueError, match="ambiguous") as info:
            resolver.get_formal_name()
        assert "mus_musculus_aj" in str(info.value) and "mus_musculus_balbcj" in str(info.value)

    def test_taxon_without_species_level_entry_is_ambiguous(self):
        """Taxon without species level entry is ambiguous."""
        with pytest.raises(ValueError, match="mus_musculus_aj, mus_musculus_balbcj"):
            _resolver_for(SPECIES, 10090).get_formal_name()

    def test_unknown_query(self):
        """Unknown query."""
        with pytest.raises(KeyError):
            _resolver_for(SPECIES, "zebrafish").get_formal_name()

    def test_integer_query_is_normalised(self):
        """Integer query is normalised."""
        resolver = _resolver_for(SPECIES, 9544)
        assert resolver.organism_query == "9544"
        assert resolver.get_latest_release() == 116
