#!/usr/bin/env python3
"""The maintainer's tools of ``tools/`` on the cut tables of the tests, against the files of the repository.

Each tool runs on the smallest real input the repository holds (the cut Ensembl tables of ``tests/data/real``, served
without network by :py:class:`tests._real_tables.RealTables`) and its result is compared with the file it writes:
the shipped externals YAML files, the cut tables themselves, the consumer-contract file. ``tools/README.md`` says how
each is run on Ensembl's complete tables. ``tools/contract_answers.py`` runs only in a checkout of ``d4c2dda`` (the
API its answers were written for) and has no test here; its output passes through ``tools/consumer_contract.py``,
whose merge is tested below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from idtrack.build.database_manager import DatabaseManager
from tests._real_tables import SNAPSHOT, RealTables, read_genes, tables_path
from tests._tools import TOOLS, load_tool

ORGANISM = "sus_scrofa"  # the smaller of the two organisms of the cut tables
# The test of an installed wheel copies the tests without the rest of the repository (.github/scripts/test_wheel.sh).
pytestmark = pytest.mark.skipif(not TOOLS.is_dir(), reason="tools/ is not beside the tests")


@pytest.fixture
def cut_repository(monkeypatch, tmp_path):
    """Return a local repository whose managers read the cut tables of pig, and no other source."""
    monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
    repository = tmp_path / "repository"
    repository.mkdir()
    RealTables.load(ORGANISM).install(monkeypatch, repository)
    return str(repository)


def _entries(path: str | Path, organism: str, enabled: bool) -> dict[tuple[str, str, str], list[int]]:
    """Return ``{(form, database, assembly): releases up to the snapshot}`` of an externals file (enabled only)."""
    with open(path, encoding="utf-8") as handle:
        forms = yaml.safe_load(handle)[organism]
    entries = {}
    for form, databases in (forms or {}).items():
        for database, entry in (databases or {}).items():
            for assembly, item in entry["Assembly"].items():
                if item["Include"] or not enabled:
                    releases = [int(r) for r in str(item["Ensembl release"]).split(",") if int(r) <= SNAPSHOT]
                    if releases:
                        entries[(form, database, str(assembly))] = releases
    return entries


def test_make_externals_enables_on_the_cut_tables_what_the_shipped_file_enables(cut_repository, tmp_path):
    """From the cut tables, the modified file enables the shipped entries at the same releases (up to the snapshot).

    The cut tables keep the cross references of every database the shipped selection enables, at every assembly and
    release (the test build stops otherwise), and few others: so the enabled entries are equal, and the template is a
    part of the shipped template.
    """
    tool = load_tool("make_externals")
    paths = tool.make(ORGANISM, cut_repository, str(tmp_path / "out"), SNAPSHOT, SNAPSHOT)
    shipped = {kind: tool.SHIPPED / tool.file_name(ORGANISM, kind) for kind in tool.KINDS}
    made = _entries(paths["modified"], ORGANISM, enabled=True)
    assert made and made == _entries(shipped["modified"], ORGANISM, enabled=True)
    template, shipped_template = (
        _entries(p, ORGANISM, enabled=False) for p in (paths["template"], shipped["template"])
    )
    assert set(template) <= set(shipped_template)
    assert all(set(releases) <= set(shipped_template[key]) for key, releases in template.items())
    assert tool.check(ORGANISM, shipped)  # the shipped files equal themselves (the comparison's own check)


def test_cut_coverage_finds_no_gap_in_the_gene_list(cut_repository):
    """Every assembly and release at which the selection enables RFAM or miRBase on genes has a gene of the list."""
    tool = load_tool("cut_coverage")
    genes = tool.read_gene_list(ORGANISM)
    assert genes == set(read_genes(ORGANISM))
    assert tool.wanted(ORGANISM, SNAPSHOT)  # pig's selection has such databases: the check is not empty
    gaps, chosen, left = tool.coverage(ORGANISM, cut_repository, SNAPSHOT, genes)
    assert (gaps, chosen, left) == ([], [], set())


def test_cut_coverage_names_a_gene_that_closes_a_gap(cut_repository):
    """Without the genes that carry miRBase, the gaps are found and closed by genes of the tables."""
    tool = load_tool("cut_coverage")
    genes = tool.read_gene_list(ORGANISM)
    carriers = set()
    for assembly, release in tool.wanted(ORGANISM, SNAPSHOT):
        carriers |= tool.carriers_at(ORGANISM, cut_repository, assembly, release)["miRBase"]
    gaps, chosen, left = tool.coverage(ORGANISM, cut_repository, SNAPSHOT, genes - carriers)
    assert {name for _, _, name, _ in gaps} >= {"miRBase", "miRBase label"}
    assert chosen and {gene for gene, _ in chosen} <= carriers and left == set()


def test_cut_tables_replays_the_shipped_file_into_the_graph_of_the_contract(capsys):
    """``cut_tables.py --check-only`` builds the graph of the shipped file, the one the contract describes."""
    tool = load_tool("cut_tables")
    assert tool.main(["--organism", ORGANISM, "--check-only"]) == 0
    printed = capsys.readouterr().out
    contract = json.loads(load_tool("consumer_contract").CONTRACT.read_text(encoding="utf-8"))
    digest = contract["organisms"][ORGANISM]["describe"]["content_digest"]
    assert f"replayed from {tables_path(ORGANISM)}: " in printed and f"graph content {digest}" in printed


def test_consumer_contract_describes_the_cut_graphs_as_the_present_file(real_world):
    """The file written from the present answers and the descriptions of the cut graphs is the present file."""
    tool = load_tool("consumer_contract")
    present = json.loads(tool.CONTRACT.read_text(encoding="utf-8"))
    described = {organism: tool.describe_of(real_world(organism).api) for organism in present["organisms"]}
    assert tool.render(tool.merged(present, None, described)) == tool.CONTRACT.read_text(encoding="utf-8")


def test_consumer_contract_takes_the_answers_of_an_answers_file_and_describes_with_this_version():
    """An answers file gives every expectation but the descriptions, which are this version's."""
    tool = load_tool("consumer_contract")
    present = json.loads(tool.CONTRACT.read_text(encoding="utf-8"))
    answers = json.loads(json.dumps(present))
    for organism, expected in answers["organisms"].items():
        expected["describe"] = {"content_digest": "of d4c2dda's graph"}
        expected["answer"] = [{"changed": organism}]
    described = {organism: present["organisms"][organism]["describe"] for organism in present["organisms"]}
    contract = tool.merged(present, answers, described)
    for organism, expected in contract["organisms"].items():
        assert expected["answer"] == [{"changed": organism}]
        assert expected["describe"] == present["organisms"][organism]["describe"]
    assert {key: value for key, value in contract.items() if key != "organisms"} == {
        key: value for key, value in present.items() if key != "organisms"
    }
