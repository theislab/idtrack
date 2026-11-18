"""Functional tests of :py:meth:`idtrack.API.classify` and :py:meth:`idtrack.API.classification_summary`.

The results are made by hand.
"""

from __future__ import annotations

import pytest

from idtrack import errors
from idtrack.api import API
from idtrack.results import Candidate, ConversionResult


def result(
    query_id: str, targets: list[str], status: str = "converted", best: list[bool] | None = None
) -> ConversionResult:
    """A conversion result of *query_id* with *targets* (each through a gene of its own name), made by hand."""
    found = status in ("converted", "tie", "no_target")
    flags = best if best is not None else [True] * len(targets)
    candidates = tuple(
        Candidate(
            gene=t,
            target=t,
            best=flag,
            entry_gene=t,
            entry_support=1,
            own_version=False,
            final_confidence=0,
            bridges=0,
            bridge_gap=0,
            score=1.0,
            same_as_query=t == query_id,
            external_steps=0,
            history_steps=0,
            annotation=None,
        )
        for t, flag in zip(targets, flags)
    )
    return ConversionResult(
        query_id=query_id,
        graph_id=None if status == "not_found" else query_id,
        match=None if status == "not_found" else "exact",
        known_versions=(),
        from_release=None,
        from_release_inferred=True,
        from_release_given=None,
        from_release_fallback=False,
        from_databases=(),
        to_release=101,
        final_database="ensembl_gene" if found else None,
        candidates=candidates if found else (),
        tie=found and len(targets) > 1,
        decided_by=None,
        alternative_locus_genes=(),
        no_target=status == "no_target",
        no_target_genes=(),
        status=status,
        reason=None if found else ("not_in_graph" if status == "not_found" else "no_path_to_target"),
    )


class TestClassify:
    """API.classify bins results by the targets of a view."""

    def test_not_found_case(self):
        """An id not found goes to the 1-to-0 bin and is counted as an input."""
        out = API.classify([result("INVALID", [], "not_found")])
        assert len(out["matching_1_to_0"]) == 1
        assert len(out["input_identifiers"]) == 1
        assert len(out["matching_1_to_1"]) == 0

    def test_no_conversion_case(self):
        """A result without a conversion goes to the 1-to-0 bin."""
        out = API.classify([result("OLD_ID", [], "no_conversion")])
        assert len(out["matching_1_to_0"]) == 1

    def test_one_to_one_same_id(self):
        """A single target equal to the query is 1-to-1 and not changed."""
        out = API.classify([result("GENE1", ["GENE1"])])
        assert len(out["matching_1_to_1"]) == 1
        assert len(out["changed_only_1_to_1"]) == 0  # Same ID, not changed

    def test_one_to_one_changed(self):
        """A single target other than the query is 1-to-1 and changed."""
        out = API.classify([result("GENE1", ["GENE2"])])
        assert len(out["matching_1_to_1"]) == 1
        assert len(out["changed_only_1_to_1"]) == 1

    def test_one_to_n_with_original(self):
        """Several targets that include the query are 1-to-n and not changed."""
        out = API.classify([result("GENE1", ["GENE1", "GENE2", "GENE3"], "tie")])
        assert len(out["matching_1_to_n"]) == 1
        assert len(out["changed_only_1_to_n"]) == 0  # Original ID present

    def test_one_to_n_all_changed(self):
        """Several targets that exclude the query are 1-to-n and changed."""
        out = API.classify([result("GENE1", ["GENE2", "GENE3"], "tie")])
        assert len(out["matching_1_to_n"]) == 1
        assert len(out["changed_only_1_to_n"]) == 1

    def test_the_view_decides(self):
        """With strategy "best" only the best candidates count; with "all" every candidate."""
        mixed = result("GENE1", ["GENE2", "GENE3"], best=[True, False])
        assert len(API.classify([mixed], "best")["matching_1_to_1"]) == 1
        assert len(API.classify([mixed], "all")["matching_1_to_n"]) == 1

    def test_alternative_target_one_to_one(self):
        """A single candidate outside the target database goes to the alternative-target 1-to-1 bin."""
        out = API.classify([result("GENE1", ["ENSG123"], "no_target")])
        assert len(out["alternative_target_1_to_1"]) == 1
        assert len(out["matching_1_to_1"]) == 0  # Goes to alternative_target

    def test_alternative_target_one_to_n(self):
        """Several candidates outside the target database go to the alternative-target 1-to-n bin."""
        out = API.classify([result("GENE1", ["ENSG123", "ENSG456"], "no_target")])
        assert len(out["alternative_target_1_to_n"]) == 1

    def test_result_without_a_best_target_is_an_integrity_error(self):
        """A result with candidates but none in the view is an internal error, never a silent 1-to-0."""
        broken = result("GENE1", ["GENE2"], best=[False])
        with pytest.raises(errors.IntegrityError, match="Unexpected"):
            API.classify([broken])

    def test_multiple_mixed_results(self):
        """A mix of results fills each bin with its own results."""
        out = API.classify(
            [
                result("A", [], "not_found"),
                result("B", ["B"]),
                result("C", ["D"]),
                result("E", ["F", "G"], "tie"),
                result("H", ["I"], "no_target"),
            ]
        )
        assert len(out["input_identifiers"]) == 5
        assert len(out["matching_1_to_0"]) == 1
        assert len(out["matching_1_to_1"]) == 2
        assert len(out["changed_only_1_to_1"]) == 1
        assert len(out["matching_1_to_n"]) == 1
        assert len(out["changed_only_1_to_n"]) == 1
        assert len(out["alternative_target_1_to_1"]) == 1


class TestClassificationSummary:
    """API.classification_summary counts bins and statuses."""

    def test_summary_of_nothing(self):
        """The summary of no results reports zero processed."""
        summary = API.classification_summary(API.classify([]))
        assert "Total processed: 0" in summary

    def test_summary_with_data(self):
        """The summary reports the totals, the share of each bin and the count of each status."""
        classified = API.classify([result("A", ["B"]), result("C", [], "not_found")])
        summary = API.classification_summary(classified)
        assert "Total processed: 2" in summary
        assert "1→0: 1 (50.0%)" in summary and "1→1: 1 (50.0%)" in summary
        assert "converted: 1" in summary and "not_found: 1" in summary
