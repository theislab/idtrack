#!/usr/bin/env python3
"""
Functional tests for idtrack._track module static methods.

These tests actually execute code paths to increase coverage.
"""

from __future__ import annotations

import pytest
import numpy as np

from idtrack._track import Track


class TestGetFromReleaseAndReverseVars:
    """Functional tests for get_from_release_and_reverse_vars static method."""

    def test_closest_at_start_of_range(self):
        """Test closest mode when pivot equals range start."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 70, "closest")
        assert len(result) == 1
        assert result[0] == (70, False)  # At start, forward direction

    def test_closest_before_range(self):
        """Test closest mode when pivot is before range."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 60, "closest")
        assert len(result) == 1
        assert result[0] == (70, True)  # Start of range, reverse direction

    def test_closest_after_range(self):
        """Test closest mode when pivot is at or after range end."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 90, "closest")
        assert len(result) == 1
        assert result[0] == (80, False)  # End of range, forward direction

    def test_closest_at_end_of_range(self):
        """Test closest mode when pivot equals range end."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 80, "closest")
        assert len(result) == 1
        assert result[0] == (80, False)  # End of range, forward direction

    def test_closest_inside_range(self):
        """Test closest mode when pivot is strictly inside range."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 75, "closest")
        assert len(result) == 2
        assert (70, True) in result
        assert (80, False) in result

    def test_distant_before_or_at_start(self):
        """Test distant mode when pivot <= range start."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 70, "distant")
        assert len(result) == 1
        assert result[0] == (80, True)  # End of range, reverse

    def test_distant_after_or_at_end(self):
        """Test distant mode when pivot >= range end."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 80, "distant")
        assert len(result) == 1
        assert result[0] == (70, False)  # Start of range, forward

    def test_distant_inside_range(self):
        """Test distant mode when pivot is strictly inside range."""
        result = Track.get_from_release_and_reverse_vars([(70, 80)], 75, "distant")
        assert len(result) == 2
        assert (80, True) in result
        assert (70, False) in result

    def test_multiple_ranges(self):
        """Test with multiple ranges."""
        ranges = [(60, 70), (80, 90), (100, 110)]
        result = Track.get_from_release_and_reverse_vars(ranges, 85, "closest")
        # Should process all ranges
        assert len(result) >= 3

    def test_invalid_range_raises(self):
        """Test that invalid range (l1 > l2) raises ValueError."""
        with pytest.raises(ValueError, match="l1 > l2"):
            Track.get_from_release_and_reverse_vars([(80, 70)], 75, "closest")


class TestPathScoreSorterSingleTarget:
    """Functional tests for _path_score_sorter_single_target static method."""

    def test_selects_best_score(self):
        """Test that method selects the dictionary with lowest scores."""
        scores = [
            {
                "assembly_jump": 1,
                "external_jump": 2,
                "external_step": 3,
                "edge_scores_reduced": 1.0,
                "ensembl_step": 5,
            },
            {
                "assembly_jump": 0,
                "external_jump": 1,
                "external_step": 2,
                "edge_scores_reduced": 0.5,
                "ensembl_step": 3,
            },
        ]
        result = Track._path_score_sorter_single_target(scores)
        assert result["assembly_jump"] == 0
        assert result["external_jump"] == 1

    def test_ties_broken_by_next_field(self):
        """Test that ties are broken by subsequent fields."""
        scores = [
            {
                "assembly_jump": 0,
                "external_jump": 2,
                "external_step": 1,
                "edge_scores_reduced": 0.5,
                "ensembl_step": 3,
            },
            {
                "assembly_jump": 0,
                "external_jump": 1,
                "external_step": 2,
                "edge_scores_reduced": 0.5,
                "ensembl_step": 3,
            },
        ]
        result = Track._path_score_sorter_single_target(scores)
        assert result["external_jump"] == 1

    def test_single_score(self):
        """Test with single score dict."""
        scores = [
            {
                "assembly_jump": 0,
                "external_jump": 0,
                "external_step": 0,
                "edge_scores_reduced": 0.0,
                "ensembl_step": 1,
            }
        ]
        result = Track._path_score_sorter_single_target(scores)
        assert result["ensembl_step"] == 1

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="not len"):
            Track._path_score_sorter_single_target([])


class TestFinalConversionDictPrepare:
    """Functional tests for _final_conversion_dict_prepare static method."""

    def test_with_paths(self):
        """Test preparation with paths included."""
        result = Track._final_conversion_dict_prepare(
            confidence=0,
            sysns=["SYN1", "SYN2"],
            paths=[[["A", "B"], ["B", "C"]], [["X", "Y"]]],
            min_priority_list=[1, 2],
            len_priority_list=[2, 1],
            add_ass_jump_list=[0, 1],
            final_database="HGNC Symbol",
        )

        assert result["final_conversion_confidence"] == 0
        assert result["final_database"] == "HGNC Symbol"
        assert "SYN1" in result["final_elements"]
        assert "SYN2" in result["final_elements"]
        assert "the_path" in result["final_elements"]["SYN1"]

    def test_without_paths(self):
        """Test preparation without paths (paths=None)."""
        result = Track._final_conversion_dict_prepare(
            confidence=1,
            sysns=["SYN1"],
            paths=None,
            min_priority_list=[1],
            len_priority_list=[1],
            add_ass_jump_list=[0],
            final_database="UniProtKB",
        )

        assert result["final_conversion_confidence"] == 1
        assert result["final_database"] == "UniProtKB"
        assert "SYN1" in result["final_elements"]
        assert "the_path" not in result["final_elements"]["SYN1"]

    def test_empty_synonyms(self):
        """Test with empty synonyms list."""
        result = Track._final_conversion_dict_prepare(
            confidence=np.inf,
            sysns=[],
            paths=None,
            min_priority_list=[],
            len_priority_list=[],
            add_ass_jump_list=[],
            final_database="RefSeq",
        )

        assert result["final_conversion_confidence"] == np.inf
        assert result["final_elements"] == {}


class TestMinimumAssemblyJumpsHelper:
    """Functional tests for _minimum_assembly_jumps_helper static method."""

    def test_simple_case(self):
        """Test simple priority traversal."""
        # DB.assembly_priority = [1, 2, 3, ...] typically
        # This method expects sorted priority lists
        step_pri = [38]
        current_priority = 38
        priorities = [[38], [38]]
        assembly_priority = [38]

        result = Track._minimum_assembly_jumps_helper(step_pri, current_priority, priorities, assembly_priority)
        # Returns (penalty, final_step_pri, final_current_priority)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_empty_priorities(self):
        """Test with no remaining priorities."""
        step_pri = [38]
        current_priority = 38
        priorities = []
        assembly_priority = [38]

        penalty, final_step, final_priority = Track._minimum_assembly_jumps_helper(
            step_pri, current_priority, priorities, assembly_priority
        )
        assert penalty == 0
        assert final_step == [38]
        assert final_priority == 38


class TestTrackHelperMethods:
    """Additional functional tests for Track helper methods."""

    def test_get_from_release_and_reverse_vars_edge_cases(self):
        """Test edge cases for get_from_release_and_reverse_vars."""
        # Single point range
        result = Track.get_from_release_and_reverse_vars([(100, 100)], 100, "closest")
        assert len(result) == 1
        assert result[0] == (100, False)

        # Distant mode at exact start
        result = Track.get_from_release_and_reverse_vars([(50, 100)], 50, "distant")
        assert result[0] == (100, True)

    def test_path_score_sorter_preserves_all_keys(self):
        """Test that path score sorter preserves all dictionary keys."""
        scores = [
            {
                "assembly_jump": 0,
                "external_jump": 0,
                "external_step": 0,
                "edge_scores_reduced": 0.0,
                "ensembl_step": 1,
                "extra_key": "preserved",
            }
        ]
        result = Track._path_score_sorter_single_target(scores)
        assert result.get("extra_key") == "preserved"

    def test_final_conversion_dict_multiple_synonyms(self):
        """Test final conversion dict with multiple synonyms."""
        result = Track._final_conversion_dict_prepare(
            confidence=0,
            sysns=["A", "B", "C"],
            paths=[[[1, 2]], [[3, 4]], [[5, 6]]],
            min_priority_list=[38, 37, 38],
            len_priority_list=[1, 2, 1],
            add_ass_jump_list=[0, 1, 0],
            final_database="HGNC",
        )

        assert len(result["final_elements"]) == 3
        assert result["final_elements"]["B"]["additional_assembly_jump"] == 1
        assert result["final_elements"]["B"]["final_assembly_priority_count"] == 2
