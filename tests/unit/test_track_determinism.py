#!/usr/bin/env python3
"""The answers of ``Track.convert`` do not depend on the hash seed of the process (tracking-4, docs-13)."""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404
import sys
from pathlib import Path

import pytest

SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from test_track import GraphBuilder

builder = GraphBuilder(range(1, 10))
builder.gene("A", [(1, 1, 4)])
names = ["zeta", "Alpha", "mid", "B2", "b10", "Gamma", "delta", "E"]
for name in names:
    builder.gene(name, [(1, 5, 9)]).event("A.1", f"{name}.1", 4, 0.5)
    for k in range(3):
        builder.xref(f"sym_{name}_{k}", f"{name}.1", "HGNC Symbol", range(5, 10))
builder.gene("R", [(1, 1, 4)])
for name in names:
    builder.xref(f"bridge_{name}", "R.1", "EntrezGene", range(1, 5))
    builder.xref(f"bridge_{name}", f"{name}.1", "EntrezGene", range(5, 10))
track = builder.track()
answers = []
for query, arguments in (
    ("A.1", {}),
    ("A.1", {"prioritize_to_one_filter": True}),
    ("A.1", {"final_database": "HGNC Symbol", "return_path": True}),
    ("A.1", {"final_database": "HGNC Symbol", "prioritize_to_one_filter": True, "return_path": True}),
    ("A", {"to_release": 9}),
    ("R.1", {"return_path": True}),
    ("R.1", {"prioritize_to_one_filter": True, "final_database": "EntrezGene"}),
):
    answers.append(track.convert(query, **arguments))
answers.append(track.identify_source([f"{name}.1" for name in names] + ["A.1"], "ensembl_release"))
answers.append(track.identify_source([f"sym_{name}_0" for name in names], "complete"))
print(json.dumps(answers))
"""


@pytest.mark.parametrize("seeds", [("0", "1", "2", "12345")])
def test_answers_are_identical_across_hash_seeds(seeds, tmp_path):
    """The complete output, order included, is byte-identical in processes with different hash seeds."""
    script = tmp_path / "convert_many.py"
    script.write_text(SCRIPT)
    here = Path(__file__).resolve().parent
    outputs = []
    for seed in seeds:
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(here.parent.parent)] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        completed = subprocess.run(  # noqa: S603
            [sys.executable, str(script), str(here)], capture_output=True, text=True, env=environment, check=False
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1
    answers = json.loads(outputs[0])
    # The split into eight equal genes is truly several: all are returned, in one fixed order.
    assert list(answers[1]) == sorted(answers[1])
    assert len(answers[1]) == 8
