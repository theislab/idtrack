#!/usr/bin/env python3
"""Conversions on graphs of real organisms, each compared with an answer taken from Ensembl's own files.

The graphs of pig and human are built by the production code from real Ensembl tables cut down to a few hundred
genes (``tests/_real_tables.py``, ``tests/data/real/``). The queries and their expected answers are in
``tests/data/real/<organism>.truth.jsonl``: one line per query with its kind (what is asked: a current id, an id
that was split, merged or retired, an id of an earlier assembly, a name, a synonym, an id of an external database,
a spelling variant, an id that never existed ...), the arguments of the call and the expected targets. The expected
targets were derived from Ensembl's files and never from idtrack: the gene ids of a release from its GTF file, the
fate of a retired id from the events of the ``stable_id_event`` table of release 115 read by code of its own (a
session that maps an id onto itself continues that id alone; otherwise all its successors), names and external ids
from the BioMart archive of the release. ``source`` identifies a line in the larger sets of such queries it was drawn
from (they are kept outside the repository; every line here is complete in itself).

Every query is asked through ``API.convert_identifier``; the ids returned and the flags of the result are asserted.
"""

from __future__ import annotations

import json

import pytest

from tests._real_tables import ORGANISMS, SNAPSHOT, answer_of, read_truth, truth_path, verdict, without_version

TRUTH = [
    (organism, query) for organism in ORGANISMS if truth_path(organism).is_file() for query in read_truth(organism)
]


def queries(*kinds):
    """Return the parameters of the queries of the given kinds (of every kind when none is named)."""
    return [
        pytest.param(organism, query, id=f"{organism}:{query['source']}")
        for organism, query in TRUTH
        if not kinds or query["kind"] in kinds
    ]


NOT_AN_ID = ("nonexistent", "version_unknown")
GENE_ID_KINDS = (
    "gene_versioned_to_self", "gene_unversioned_to_self", "gene_versioned_to_self_altloci",
    "gene_unversioned_to_self_altloci", "retired_split", "retired_merged", "retired_renamed", "retired_none",
    "retired_none_branch", "unchanged_old_to_current", "reverse_present", "reverse_absent", "cross_assembly",
    "gene_to_db", "old_gene_to_db",
)  # fmt: skip


def ask(world, query, **more):
    """Ask a query of the truth file through the public API."""
    return world.api.convert_identifier(
        query["id"],
        from_release=query["from_release"],
        to_release=query["to_release"],
        final_database=query["final_database"],
        strategy=query["strategy"],
        from_database=query.get("from_database"),
        **more,
    )


@pytest.mark.parametrize("organism, query", queries())
def test_conversion_returns_the_ids_that_ensembl_gives(real_world, organism, query):
    """The targets are the expected ones, and the flags of the result say what happened."""
    world = real_world(organism)
    result = ask(world, query)

    assert verdict(query, result) is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["error"] is None
    assert (result["query_id"], result["to_release"], result["strategy"]) == (
        query["id"],
        query["to_release"],
        query["strategy"],
    )
    assert result["from_release_given"] == query["from_release"]
    targets = result["target_id"]
    assert result["ambiguous"] is (len(targets) > 1)
    assert [pair[1] for pair in result["last_node"]] == targets or len(result["last_node"]) > len(targets)
    # every gene of the answer is a gene of the target release in Ensembl's table, with that version
    there = world.genes_at(query["to_release"])
    assert all(there.get(without_version(gene)) == int(gene.rpartition(".")[2]) for gene, _ in result["last_node"])

    if query["expect"]:
        # an answer in the database asked for: none of the three "nothing" flags
        assert not (result["no_corresponding"] or result["no_conversion"] or result["no_target"])
        assert result["final_database"] == (query["final_database"] or "ensembl_gene")
        assert result["graph_id"] is not None and result["match"] is not None
        assert result["from_release_inferred"] is (query["from_release"] is None or result["from_release_fallback"])
        if not result["from_release_fallback"] and query["from_release"] is not None:
            assert result["from_release"] == query["from_release"]
        if query["strategy"] == "best" and len(result["last_node"]) > 1:
            # several targets as the best answer are a tie; read and answered at one release, nothing but the ids
            # tells them apart, and they come in ascending order of gene and target
            assert result["tie"] is True
            if result["from_release"] == query["to_release"]:
                assert result["last_node"] == sorted(result["last_node"])
    elif query["kind"] in NOT_AN_ID:
        assert result["no_corresponding"] is True and result["graph_id"] is None and result["match"] is None
        assert (result["no_conversion"], result["no_target"], targets) == (False, False, [])
        assert result["final_database"] is None
    else:
        # the identifier is known, and Ensembl gives it no counterpart (in the database asked for)
        assert result["no_corresponding"] is False and result["graph_id"] is not None
        assert result["no_conversion"] is not result["no_target"]
        if result["no_conversion"]:
            assert (targets, result["last_node"], result["final_database"]) == ([], [], None)
        else:
            # genes were reached, none with an id in the database: the genes are named instead, flagged
            assert query["final_database"] and result["final_database"] == "ensembl_gene" and targets


@pytest.mark.parametrize("organism, query", queries(*GENE_ID_KINDS))
def test_gene_id_is_read_at_the_release_the_rule_says(real_world, organism, query):
    """Without ``from_release`` a gene id is read at the target release if it exists there, else at its last one.

    Where the id exists is read from Ensembl's gene tables; an id written with a version means that version.
    """
    world = real_world(organism)
    result = ask(world, query)
    releases = world.releases_of_gene(query["id"])
    assert releases, "the cut tables must hold the queried gene"

    if not answer_of(result, None) and result["no_conversion"]:
        assert result["from_release"] is None or result["from_release"] in releases
        return
    if query["from_release"] is None:
        expected = query["to_release"] if query["to_release"] in releases else releases[-1]
        assert (result["from_release"], result["from_release_inferred"]) == (expected, True)
    else:
        assert query["from_release"] in releases
        assert (result["from_release"], result["from_release_inferred"], result["from_release_fallback"]) == (
            query["from_release"],
            False,
            False,
        )
    # a gene id is a node of the graph as it is written, with its version or without
    assert (result["graph_id"], result["match"]) == (query["id"], "exact")


EXTERNAL_ID_KINDS = (
    "name_unique_to_gene", "name_ambiguous_to_set", "name_old_to_current", "name_changed_gene", "synonym_to_gene",
    "ncbi_to_gene", "refseq_to_gene", "uniprot_to_gene", "separator_variant",
)  # fmt: skip


@pytest.mark.parametrize("organism, query", queries(*EXTERNAL_ID_KINDS))
def test_external_id_is_matched_and_read_as_ensembls_tables_say(real_world, organism, query):
    """How a name was matched, in which databases it was read, and the fall-back of the release, are reported.

    What the string means at the release asked about is worked out from Ensembl's tables: an identifier that
    exists at the release (``from_release``, without one the last release) is taken before one that does not,
    exact before letter case before separators, names before synonyms. When no meaning of the string exists at
    a ``from_release`` that was given, the identifier is read where the graph holds it and the result says so.
    """
    world = real_world(organism)
    result = ask(world, query)
    asked = SNAPSHOT if query["from_release"] is None else query["from_release"]
    match, here, databases = world.ensembl.meaning(query["id"], asked)

    assert result["from_release_fallback"] is (query["from_release"] is not None and not here)
    if result["from_release_fallback"]:
        assert result["from_release"] != query["from_release"] and result["from_release_inferred"] is True
        match, here, _ = world.ensembl.meaning(query["id"], SNAPSHOT)
        databases = world.ensembl.meaning(query["id"], result["from_release"])[2]
    elif query["from_release"] is None and not here:
        # not an identifier of the last release: read at the latest release that has it
        databases = world.ensembl.meaning(query["id"], result["from_release"])[2]
    if match is not None:
        assert result["match"] == match
    if result["no_conversion"]:
        assert result["from_databases"] == []  # read, but it leads to no gene at the target: nothing to report
    elif databases is not None:
        assert sorted(result["from_databases"]) == sorted(databases)

    # The node read and the release it is read at, for a string matched as it is written or up to letter case.
    kind = "synonym" if str(result["match"]).startswith("synonym:") else "id"
    if match is None or result["match"].rpartition(":")[2] not in ("exact", "case"):
        return
    spellings = world.ensembl.spellings_on(world.backbone, query["id"], kind)
    if not spellings:
        return  # an identifier that only an assembly beside the serving one has
    latest = spellings[max(spellings)]
    if len(latest) == 1:
        # one node per identifier, named as Ensembl writes it at the last release that has it
        assert result["graph_id"] == ("synonym_id::" if kind == "synonym" else "") + next(iter(latest))
    if query["from_release"] is None:
        # read at the target release if it exists there, else at the last release that has it
        expected = query["to_release"] if query["to_release"] in spellings else max(spellings)
        assert (result["from_release"], result["from_release_inferred"]) == (expected, True)
    elif not result["from_release_fallback"]:
        assert query["from_release"] in spellings and result["from_release"] == query["from_release"]


@pytest.mark.parametrize("organism", [o for o in ORGANISMS if o == "homo_sapiens" and truth_path(o).is_file()])
def test_string_that_only_the_frozen_assembly_lists_as_a_name_is_read_on_the_serving_one(real_world, organism):
    """What a string means at a release is what the assembly that serves the release says.

    Human GRCh37 lists its names of release 75 at every release up to the last. The lines of the truth file whose
    string is, at the release asked about, a name on that frozen assembly only and a synonym on the serving one:
    read as today's synonym, at the release asked about, without fall-back.
    """
    world = real_world(organism)
    wrong = []
    for _organism, query in TRUTH:
        if _organism != organism or query["kind"] not in EXTERNAL_ID_KINDS:
            continue
        asked = SNAPSHOT if query["from_release"] is None else query["from_release"]
        if not world.ensembl.only_beside(query["id"], asked):
            continue
        match, here, _databases = world.ensembl.meaning(query["id"], asked)
        result = ask(world, query)
        if here and (result["match"], result["from_release"], result["from_release_fallback"]) != (match, asked, False):
            wrong.append((query["source"], result["match"], result["from_release"], result["from_release_fallback"]))
    assert not wrong, wrong


@pytest.mark.parametrize("organism", [o for o in ORGANISMS if o == "homo_sapiens" and truth_path(o).is_file()])
def test_truth_holds_strings_that_only_the_frozen_assembly_lists_as_names(real_world, organism):
    """The case of the test before exists in the truth file (it would otherwise pass for want of a case)."""
    world = real_world(organism)
    cases = [
        query["source"]
        for _organism, query in TRUTH
        if _organism == organism
        and query["kind"] in EXTERNAL_ID_KINDS
        and world.ensembl.only_beside(query["id"], SNAPSHOT if query["from_release"] is None else query["from_release"])
    ]
    assert len(cases) >= 2, cases


@pytest.mark.parametrize("organism, query", queries("version_unknown"))
def test_unknown_version_names_the_versions_that_ensembl_had(real_world, organism, query):
    """A version that no release had is answered with nothing, and the versions that exist are named."""
    world = real_world(organism)
    result = ask(world, query)
    base = without_version(query["id"])
    held = sorted({world.genes_at(r)[base] for r in world.backbone if base in world.genes_at(r)})

    assert result["known_versions"] == [f"{base}.{version}" for version in held]
    assert int(query["id"].rpartition(".")[2]) not in held
    assert (result["target_id"], result["no_corresponding"]) == ([], True)


@pytest.mark.parametrize("organism, query", queries("version_newer_than_graph"))
def test_version_newer_than_the_graph_is_flagged(real_world, organism, query):
    """A version above every version Ensembl had may belong to a later release: the gene, with the flag."""
    world = real_world(organism)
    result = ask(world, query)
    base = without_version(query["id"])
    current = world.genes_at(SNAPSHOT)[base]

    assert int(query["id"].rpartition(".")[2]) > max(world.genes_at(r).get(base, 0) for r in world.backbone)
    assert (result["match"], result["graph_id"], result["target_id"]) == (
        "version_newer_than_graph",
        f"{base}.{current}",
        [f"{base}.{current}"],
    )
    assert result["known_versions"] == []


@pytest.mark.parametrize("organism", [o for o in ORGANISMS if truth_path(o).is_file()])
def test_truth_file_covers_every_kind_of_query(organism):
    """The truth file must keep the kinds of queries it was made for (a regenerated file that lost one fails here)."""
    kinds = {query["kind"] for query in read_truth(organism)}
    needed = {
        "gene_versioned_to_self", "gene_unversioned_to_self", "transcript_to_gene", "translation_to_gene",
        "gene_to_db", "old_gene_to_db", "name_unique_to_gene", "name_ambiguous_to_set", "name_old_to_current",
        "name_changed_gene", "synonym_to_gene", "ncbi_to_gene", "refseq_to_gene", "uniprot_to_gene", "case_variant",
        "separator_variant", "nonexistent", "version_newer_than_graph", "version_unknown", "retired_split",
        "retired_merged", "retired_renamed", "retired_none", "unchanged_old_to_current", "reverse_present",
        "reverse_absent", "cross_assembly",
    }  # fmt: skip
    if organism == "homo_sapiens":
        needed |= {"gene_versioned_to_self_altloci", "gene_unversioned_to_self_altloci"}
    assert needed <= kinds, sorted(needed - kinds)
