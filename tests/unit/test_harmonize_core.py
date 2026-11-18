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
