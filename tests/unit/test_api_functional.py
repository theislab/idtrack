#!/usr/bin/env python3
"""
Functional tests for idtrack._api module.

These tests actually execute code paths to increase coverage.
"""

from __future__ import annotations

import pytest

from idtrack._api import API


class TestClassifyMultipleConversion:
    """Functional tests for classify_multiple_conversion method."""

    @pytest.fixture
    def api(self, tmp_path):
        """Create an API instance without building graph."""
        return API(str(tmp_path))

    def test_no_corresponding_case(self, api):
        """Test handling of no_corresponding matches."""
        matchings = [
            {
                "query_id": "INVALID",
                "target_id": [],
                "no_corresponding": True,
                "no_conversion": False,
                "no_target": False,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["matching_1_to_0"]) == 1
        assert len(result["input_identifiers"]) == 1
        assert len(result["matching_1_to_1"]) == 0

    def test_no_conversion_case(self, api):
        """Test handling of no_conversion matches."""
        matchings = [
            {
                "query_id": "OLD_ID",
                "target_id": [],
                "no_corresponding": False,
                "no_conversion": True,
                "no_target": False,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["matching_1_to_0"]) == 1

    def test_one_to_one_same_id(self, api):
        """Test 1:1 mapping where target equals query."""
        matchings = [
            {
                "query_id": "GENE1",
                "target_id": ["GENE1"],
                "no_corresponding": False,
                "no_conversion": False,
                "no_target": False,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["matching_1_to_1"]) == 1
        assert len(result["changed_only_1_to_1"]) == 0  # Same ID, not changed

    def test_one_to_one_changed(self, api):
        """Test 1:1 mapping where target differs from query."""
        matchings = [
            {
                "query_id": "GENE1",
                "target_id": ["GENE2"],
                "no_corresponding": False,
                "no_conversion": False,
                "no_target": False,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["matching_1_to_1"]) == 1
        assert len(result["changed_only_1_to_1"]) == 1

    def test_one_to_n_with_original(self, api):
        """Test 1:n mapping where original ID is in targets."""
        matchings = [
            {
                "query_id": "GENE1",
                "target_id": ["GENE1", "GENE2", "GENE3"],
                "no_corresponding": False,
                "no_conversion": False,
                "no_target": False,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["matching_1_to_n"]) == 1
        assert len(result["changed_only_1_to_n"]) == 0  # Original ID present

    def test_one_to_n_all_changed(self, api):
        """Test 1:n mapping where all targets differ."""
        matchings = [
            {
                "query_id": "GENE1",
                "target_id": ["GENE2", "GENE3"],
                "no_corresponding": False,
                "no_conversion": False,
                "no_target": False,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["matching_1_to_n"]) == 1
        assert len(result["changed_only_1_to_n"]) == 1

    def test_alternative_target_one_to_one(self, api):
        """Test alternative target (no_target=True) with single result."""
        matchings = [
            {
                "query_id": "GENE1",
                "target_id": ["ENSG123"],  # Ensembl fallback
                "no_corresponding": False,
                "no_conversion": False,
                "no_target": True,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["alternative_target_1_to_1"]) == 1
        assert len(result["matching_1_to_1"]) == 0  # Goes to alternative_target

    def test_alternative_target_one_to_n(self, api):
        """Test alternative target with multiple results."""
        matchings = [
            {
                "query_id": "GENE1",
                "target_id": ["ENSG123", "ENSG456"],
                "no_corresponding": False,
                "no_conversion": False,
                "no_target": True,
            }
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["alternative_target_1_to_n"]) == 1

    def test_empty_target_raises(self, api):
        """Test that empty target_id with valid flags raises error."""
        matchings = [
            {
                "query_id": "GENE1",
                "target_id": [],
                "no_corresponding": False,
                "no_conversion": False,
                "no_target": False,
            }
        ]
        with pytest.raises(ValueError, match="Unexpected"):
            api.classify_multiple_conversion(matchings)

    def test_multiple_mixed_results(self, api):
        """Test classification with multiple varied results."""
        matchings = [
            # No corresponding
            {"query_id": "A", "target_id": [], "no_corresponding": True, "no_conversion": False, "no_target": False},
            # 1:1 same
            {"query_id": "B", "target_id": ["B"], "no_corresponding": False, "no_conversion": False, "no_target": False},
            # 1:1 changed
            {"query_id": "C", "target_id": ["D"], "no_corresponding": False, "no_conversion": False, "no_target": False},
            # 1:n all changed
            {"query_id": "E", "target_id": ["F", "G"], "no_corresponding": False, "no_conversion": False, "no_target": False},
            # Alternative target
            {"query_id": "H", "target_id": ["I"], "no_corresponding": False, "no_conversion": False, "no_target": True},
        ]
        result = api.classify_multiple_conversion(matchings)

        assert len(result["input_identifiers"]) == 5
        assert len(result["matching_1_to_0"]) == 1
        assert len(result["matching_1_to_1"]) == 2
        assert len(result["changed_only_1_to_1"]) == 1
        assert len(result["matching_1_to_n"]) == 1
        assert len(result["changed_only_1_to_n"]) == 1
        assert len(result["alternative_target_1_to_1"]) == 1


class TestAPIInitialization:
    """Functional tests for API initialization."""

    def test_init_creates_instance(self, tmp_path):
        """Test API can be initialized."""
        api = API(str(tmp_path))
        assert api.local_repository == str(tmp_path)
        assert api.logger_configured is False
        assert api.track is None

    def test_configure_logger_first_time(self, tmp_path):
        """Test logger configuration on first call."""
        api = API(str(tmp_path))
        api.configure_logger()
        assert api.logger_configured is True

    def test_configure_logger_second_time(self, tmp_path):
        """Test logger configuration on subsequent call."""
        api = API(str(tmp_path))
        api.configure_logger()
        # Second call should not reconfigure
        api.configure_logger()
        assert api.logger_configured is True

    def test_configure_logger_with_level(self, tmp_path):
        """Test logger configuration with custom level."""
        import logging

        api = API(str(tmp_path))
        api.configure_logger(level=logging.DEBUG)
        assert api.logger_configured is True


class TestAPIRequireTrack:
    """Functional tests for _require_track method."""

    def test_require_track_without_graph(self, tmp_path):
        """Test _require_track raises when no graph is built."""
        api = API(str(tmp_path))
        with pytest.raises(RuntimeError, match="No graph is attached"):
            api._require_track()


class TestPrintBinnedConversion:
    """Functional tests for print_binned_conversion method."""

    def test_print_binned_empty(self, tmp_path):
        """Test print_binned_conversion with empty results."""
        api = API(str(tmp_path))
        classified = {
            "input_identifiers": [],
            "matching_1_to_0": [],
            "matching_1_to_1": [],
            "matching_1_to_n": [],
            "changed_only_1_to_1": [],
            "changed_only_1_to_n": [],
            "alternative_target_1_to_1": [],
            "alternative_target_1_to_n": [],
        }
        # Should not raise
        api.print_binned_conversion(classified)

    def test_print_binned_with_data(self, tmp_path):
        """Test print_binned_conversion with sample data."""
        api = API(str(tmp_path))
        api.configure_logger()

        classified = {
            "input_identifiers": [
                {"query_id": "A", "target_id": ["B"], "no_corresponding": False, "no_conversion": False, "no_target": False},
                {"query_id": "C", "target_id": [], "no_corresponding": True, "no_conversion": False, "no_target": False},
            ],
            "matching_1_to_0": [
                {"query_id": "C", "target_id": [], "no_corresponding": True, "no_conversion": False, "no_target": False},
            ],
            "matching_1_to_1": [
                {"query_id": "A", "target_id": ["B"], "no_corresponding": False, "no_conversion": False, "no_target": False},
            ],
            "matching_1_to_n": [],
            "changed_only_1_to_1": [
                {"query_id": "A", "target_id": ["B"], "no_corresponding": False, "no_conversion": False, "no_target": False},
            ],
            "changed_only_1_to_n": [],
            "alternative_target_1_to_1": [],
            "alternative_target_1_to_n": [],
        }
        # Should not raise
        api.print_binned_conversion(classified)
