#!/usr/bin/env python3
"""Write the consumer-contract file of the tests, ``tests/data/real/consumer_contract.json``, from its two sources.

Usage (from the root of the repository, with the package importable)::

    python tools/consumer_contract.py [--answers <answers.json>] [--out <file>] [--check]

The file holds, per organism of the cut test tables, two kinds of expectations (``tests/unit/test_consumer_contract.py``
reads them):

- the **answers** of the consumer's calls (every key of an organism but ``describe``, and the top-level keys): made
  by ``tools/contract_answers.py`` with the API they were written for, which runs only in a checkout of commit
  ``d4c2dda`` (its usage says how). They change only when the cut tables change. ``--answers`` names the file it
  wrote; without it, the answers of the present file are kept.
- the **description** of the cut graph the answers are compared on (``describe``: content digest, releases, the
  assembly serving each release, the releases and external databases of each assembly): made here, from the graphs
  this checkout builds from the cut tables (``tests/_real_tables.py``, no network), so that the test knows it runs on
  the graph the answers were made for. They change with any change of what a graph build makes.

Reads: the present file, ``--answers`` if given, and the cut tables ``tests/data/real/<organism>.tables.json.xz``.
Writes: the file (``--out``, default the present file's place). With ``--check``, nothing is written: the file it
would write is compared with the present one (sha256), and the exit status is 1 when they differ. A build of the two
cut graphs takes a few minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONTRACT = ROOT / "tests" / "data" / "real" / "consumer_contract.json"


def describe_of(api: Any) -> dict[str, Any]:
    """Return the ``describe`` expectations of a loaded graph (:py:class:`idtrack.API`)."""
    described = api.describe()
    return {
        "content_digest": described.content_digest,
        "releases": [int(r) for r in described.releases],
        "serving_assembly": {str(r): int(a) for r, a in described.serving_assembly.items()},
        "assemblies": {str(a.assembly): [int(r) for r in a.releases] for a in described.assemblies},
        "external_databases": {str(a.assembly): list(a.external_databases) for a in described.assemblies},
    }


def merged(present: dict[str, Any], answers: dict[str, Any] | None, describe: dict[str, dict]) -> dict[str, Any]:
    """Return the contract: the answers (of *answers*, else of *present*) with the given descriptions."""
    source = answers if answers is not None else present
    contract = {key: value for key, value in source.items() if key != "organisms"}
    contract["organisms"] = {}
    for organism, expected in source["organisms"].items():
        entry = {key: value for key, value in expected.items() if key != "describe"}
        entry["describe"] = describe[organism]
        contract["organisms"][organism] = entry
    return contract


def render(contract: dict[str, Any]) -> str:
    """Return the text of the contract file (compact, keys sorted, one value per line)."""
    return json.dumps(contract, indent=0, sort_keys=True, separators=(",", ":")) + "\n"


def describe_cut_graphs(organisms: list[str]) -> dict[str, dict]:
    """Build the cut graph of each organism (no network) and return its ``describe`` expectations."""
    from tests._real_tables import RealWorld

    described = {}
    for organism in organisms:
        with tempfile.TemporaryDirectory(prefix=f"contract_{organism}_") as repository:
            described[organism] = describe_of(RealWorld(organism, repository).api)
    return described


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--answers", help="the answers written by tools/contract_answers.py (default: keep the present)")
    parser.add_argument("--out", default=str(CONTRACT), help="the file to write (default: the present file)")
    parser.add_argument("--check", action="store_true", help="write nothing; compare with the present file")
    arguments = parser.parse_args(argv)
    present = json.loads(CONTRACT.read_text(encoding="utf-8"))
    answers = None
    if arguments.answers:
        answers = json.loads(Path(arguments.answers).read_text(encoding="utf-8"))
    organisms = list((answers or present)["organisms"])
    text = render(merged(present, answers, describe_cut_graphs(organisms)))
    if arguments.check:
        made, shipped = (hashlib.sha256(t.encode("utf-8")).hexdigest() for t in (text, CONTRACT.read_text("utf-8")))
        print(f"made {made}, present {shipped}: {'equal' if made == shipped else 'DIFFERENT'}")
        return 0 if made == shipped else 1
    Path(arguments.out).write_text(text, encoding="utf-8")
    print(arguments.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
