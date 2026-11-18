"""The explained path as data, on two real conversions of the complete graphs (human and pig, snapshot 115).

The two examples cross every kind of step a path has: an entry through a gene id without version, Ensembl's history
across assembly handoffs (NCBI36 to GRCh37 to GRCh38 in human; Sscrofa10.2 to Sscrofa11.1 in pig), splits and merges,
a scored self-loop, and a bridge through a synonym both genes carry. The complete graphs are not part of the
repository (85 to 345 MB): the tests run when ``IDTRACK_TEST_GRAPHS`` names a folder holding
``<organism>/graph_<organism>_assembly<key>_min<first>_max115_narrow.npz`` and are skipped otherwise.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from idtrack import API, ConversionResult

FOLDER = os.environ.get("IDTRACK_TEST_GRAPHS", "")


def _api(organism: str) -> API:
    found = sorted(glob.glob(os.path.join(FOLDER, organism, f"graph_{organism}_assembly*_max115_narrow.npz")))
    if not FOLDER or not found:
        pytest.skip(f"no complete graph of {organism} (set IDTRACK_TEST_GRAPHS)")
    return API.load(found[0])


def _steps(result: ConversionResult) -> list[tuple]:
    path = result.path
    return [
        (
            s.kind,
            s.source,
            s.target,
            s.release,
            s.old_release,
            s.new_release,
            s.weight,
            s.splits,
            s.merges,
            s.databases,
            s.assembly,
            s.old_assembly,
            s.reason,
        )
        for s in (path.steps[i] for i in result.best[0].path)
    ]


def test_human_gene_id_without_version_across_two_assembly_handoffs_and_a_bridge():
    """``ENSG00000196540``: read at 54 (NCBI36), split into GRCh37, retired, bridged through ``C9ORF36A`` to GRCh38."""
    r = _api("homo_sapiens").convert("ENSG00000196540", to_release=115, explain=True)
    assert (r.status, r.graph_id, r.from_release, r.from_release_inferred) == ("converted", "ENSG00000196540", 54, True)
    assert [(c.gene, c.best, c.bridges, c.bridge_gap) for c in r.candidates] == [
        ("ENSG00000204849.10", True, 1, 0),
        ("ENSG00000276040.5", False, 1, 1),
        ("ENSG00000276581.3", False, 1, 2),
    ]
    assert r.decided_by == "bridge_gap" and r.tie is False
    assert round(r.best[0].score, 6) == 0.923357
    assert _steps(r) == [
        (
            "entry",
            "ENSG00000196540",
            "ENSG00000196540.4",
            54,
            None,
            None,
            None,
            0,
            0,
            ("base_ensembl_gene",),
            36,
            None,
            "entry",
        ),
        (
            "history",
            "ENSG00000196540.4",
            "ENSG00000234734.1",
            None,
            54,
            55,
            0.99239,
            2,
            2,
            (),
            37,
            36,
            "split_and_merge",
        ),
        ("history", "ENSG00000234734.1", "ENSG00000234734.2", None, 55, 56, 0.978144, 0, 0, (), 37, 37, "new_version"),
        ("history", "ENSG00000234734.2", "ENSG00000234734.3", None, 57, 58, 0.985837, 0, 0, (), 37, 37, "new_version"),
        (
            "bridge_near",
            "ENSG00000234734.3",
            "synonym_id::C9ORF36A",
            70,
            None,
            None,
            None,
            0,
            0,
            ("synonym_id::EntrezGene",),
            37,
            None,
            "bridge_near",
        ),
        (
            "bridge_far",
            "synonym_id::C9ORF36A",
            "ENSG00000204849.5",
            70,
            None,
            None,
            None,
            0,
            0,
            ("synonym_id::EntrezGene", "synonym_id::HGNC Symbol"),
            37,
            None,
            "bridge_far",
        ),
        ("history", "ENSG00000204849.5", "ENSG00000204849.6", None, 78, 79, 0.996658, 0, 0, (), 38, 38, "new_version"),
        ("history", "ENSG00000204849.6", "ENSG00000204849.7", None, 80, 81, 0.999895, 0, 0, (), 38, 38, "new_version"),
        ("history", "ENSG00000204849.7", "ENSG00000204849.8", None, 95, 96, 0.992651, 0, 0, (), 38, 38, "new_version"),
        ("history", "ENSG00000204849.8", "ENSG00000204849.9", None, 97, 98, 0.991236, 0, 0, (), 38, 38, "new_version"),
        ("self_loop", "ENSG00000204849.9", "ENSG00000204849.9", None, 98, 99, 0.991402, 0, 0, (), 38, 38, "self_loop"),
        (
            "history",
            "ENSG00000204849.9",
            "ENSG00000204849.10",
            None,
            99,
            100,
            0.992557,
            0,
            0,
            (),
            38,
            38,
            "new_version",
        ),
    ]
    nodes = {n.id: n for n in r.path.nodes}
    # the graph's names, as d4c2dda reported them (the clean labels of decision 56 are stage S2's)
    assert (nodes["synonym_id::C9ORF36A"].kind, nodes["synonym_id::C9ORF36A"].label) == (
        "synonym",
        "synonym_id::C9ORF36A",
    )
    assert nodes["ENSG00000204849.5"].active == ((62, 78),) and nodes["ENSG00000204849.5"].assemblies == (37, 38)
    assert nodes["ENSG00000204849.10"].active == ((100, None),)
    assert ConversionResult.from_dict(json.loads(json.dumps(r.to_dict(), allow_nan=False))) == r


def test_pig_gene_retired_at_the_assembly_change_bridged_through_a_synonym():
    """``ENSSSCG00000024116.1``: read at 89, bridged through ``H2B.1`` at 86, then a split and merge at 89 to 90."""
    r = _api("sus_scrofa").convert("ENSSSCG00000024116.1", to_release=115, explain=True)
    assert (r.status, r.from_release, len(r.candidates), r.tie) == ("converted", 89, 8, False)
    best, runner_up = r.candidates[0], r.candidates[1]
    assert (best.gene, best.best, best.bridges, best.bridge_gap, round(best.score, 6)) == (
        "ENSSSCG00000063423.1", True, 1, 0, 0.962092,
    )  # fmt: skip
    # Equal in confidence, bridges, gap, support and score: the richer annotation decides (7 against 6 external ids).
    assert (runner_up.gene, runner_up.best, round(runner_up.score, 6)) == ("ENSSSCG00000057021.1", False, 0.962092)
    assert (best.annotation, runner_up.annotation, r.decided_by) == ((-7, -1, -1), (-6, -1, -1), "annotation")
    assert _steps(r) == [
        (
            "bridge_near",
            "ENSSSCG00000024116.1",
            "synonym_id::H2B.1",
            86,
            None,
            None,
            None,
            0,
            0,
            ("synonym_id::HGNC Symbol",),
            102,
            None,
            "bridge_near",
        ),
        (
            "bridge_far",
            "synonym_id::H2B.1",
            "ENSSSCG00000026668.1",
            86,
            None,
            None,
            None,
            0,
            0,
            ("synonym_id::HGNC Symbol",),
            102,
            None,
            "bridge_far",
        ),
        (
            "history",
            "ENSSSCG00000026668.1",
            "ENSSSCG00000036045.1",
            None,
            89,
            90,
            0.962092,
            3,
            3,
            (),
            111,
            102,
            "split_and_merge",
        ),
        (
            "history",
            "ENSSSCG00000036045.1",
            "ENSSSCG00000063423.1",
            None,
            106,
            107,
            1.0,
            4,
            1,
            (),
            111,
            111,
            "split_and_merge",
        ),
    ]
    nodes = {n.id: n for n in r.path.nodes}
    assert nodes["ENSSSCG00000024116.1"].active == ((67, 89),) and nodes["ENSSSCG00000024116.1"].assemblies == (102,)
    assert nodes["synonym_id::H2B.1"].databases == ("synonym_id::HGNC Symbol",)
    assert ConversionResult.from_dict(json.loads(json.dumps(r.to_dict(), allow_nan=False))) == r
