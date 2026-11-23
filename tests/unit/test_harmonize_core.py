"""The pure core of harmonization (:py:mod:`idtrack.harmonize.core`): needs pandas only, not the ``harmonize`` extra."""

from __future__ import annotations

from idtrack.harmonize import feature_table, output_plan
from tests.unit._fake_engine import fake_api

HGNC = "HGNC Symbol"


def test_an_unconverted_feature_named_like_a_converted_gene_is_renamed(tmp_path):
    """A kept unconverted feature whose name is that of a converted output gets the prefix ``unconverted:``.

    Two outputs of one dataset may never share a name: the gene ``ENSG_NEW.1`` (converted from ``OLD``) and the input
    feature ``ENSG_NEW.1`` that converts to nothing are both kept, the latter renamed.
    """
    answers = {"OLD": {"ENSG_NEW.1": {HGNC: ["NEW"]}}, "ENSG_NEW.1": {}}
    api = fake_api(tmp_path, answers)
    names = ["OLD", "ENSG_NEW.1"]
    table = feature_table(names, api.convert_many(names, final_database=HGNC), {"ENSG_NEW.1": ["NEW"]})
    assert table["status"].tolist()[0] == "converted" and table["status"].tolist()[1] != "converted"
    plan = output_plan(table, "flag", keep_unconverted=True)
    assert sorted(plan.index) == ["ENSG_NEW.1", "unconverted:ENSG_NEW.1"]
    assert plan.loc["ENSG_NEW.1", "sources"] == [0] and plan.loc["unconverted:ENSG_NEW.1", "sources"] == [1]
    assert not plan.index.duplicated().any()


def test_a_missing_value_is_left_unconverted_with_its_reason(tmp_path):
    """Decision 81: ``nan`` beside a gene is never summed into a gene; it stays ``missing_value`` with the reason.

    Fails if the harmonizer reads a missing value as a name (``nan`` is a synonym of SCN11A in human), or if the
    status says only ``error``.
    """
    answers = {"TP53": {"ENSG1.1": {HGNC: ["TP53"]}}, "nan": {"ENSG2.1": {HGNC: ["SCN11A"]}}}
    api = fake_api(tmp_path, answers)
    names = ["TP53", "nan"]
    table = feature_table(names, api.convert_many(names, final_database=HGNC), {"ENSG1.1": ["TP53"]})
    assert table["status"].tolist() == ["converted", "missing_value"]
    assert "missing value" in table.loc["nan", "reason"]
    plan = output_plan(table, "sum", keep_unconverted=True)
    assert plan.loc["nan", "status"] == "unconverted_missing_value" and plan.loc["ENSG1.1", "sources"] == [0]


def test_a_feature_made_unique_is_a_copy_of_its_feature(tmp_path):
    """Decision 82: ``TP53.1`` beside ``TP53`` is TP53's copy; with ``merge="sum"`` the two are summed into TP53.

    Fails if the copy is read on its own (here: it would be another gene, ``TP53-1``'s), or if no reason says so.
    """
    answers = {"TP53": {"ENSG1.1": {HGNC: ["TP53"]}}, "TP53-1": {"ENSG9.1": {HGNC: ["OTHER"]}}}
    api = fake_api(tmp_path, answers)
    names = ["TP53", "TP53.1"]
    results = api.convert_many(names, final_database=HGNC)
    assert results[1].match == "copy:exact" and results[1].targets() == ("TP53",)
    table = feature_table(names, results, {"ENSG1.1": ["TP53"]})
    assert table["ensembl_gene"].tolist() == ["ENSG1.1", "ENSG1.1"]
    assert "copy of the feature 'TP53'" in table.loc["TP53.1", "reason"]
    plan = output_plan(table, "sum", keep_unconverted=True)
    assert plan.loc["ENSG1.1", "sources"] == [0, 1] and plan.loc["ENSG1.1", "status"] == "merged"
