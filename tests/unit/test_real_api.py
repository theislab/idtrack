#!/usr/bin/env python3
"""The public API on the graphs of real organisms: what it lists, how it reports, what it refuses.

The graphs are built from the cut Ensembl tables of ``tests/data/real`` (``tests/_real_tables.py``); every
expectation is read from those tables or from the truth file, not from idtrack. The conversions themselves are
tested in ``test_real_truth.py`` and ``test_real_decisions.py``; here are the listings of the graph, batches, the
explanation of an answer, the inference of where a list of identifiers comes from, the refusal of wrong arguments
and the harmonization of a dataset.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest

import idtrack
from idtrack import errors, ids, organisms
from idtrack.results import ConversionResult
from tests._real_tables import (
    ORGANISMS,
    SNAPSHOT,
    read_truth,
    tables_path,
    truth_path,
    verdict,
    with_y_copies_apart,
    without_version,
)
from tests.unit._paths import raw_steps

AVAILABLE = [organism for organism in ORGANISMS if tables_path(organism).is_file() and truth_path(organism).is_file()]
pytestmark = pytest.mark.parametrize("organism", AVAILABLE)
SYNONYM = ids.SYNONYM_PREFIX


def selected_links(world):
    """Yield ``(assembly, release, gene, database, id, form)`` of every selected cross reference of the tables."""
    for assembly in sorted({int(a) for a in world.tables.indices}):
        for release in world.ensembl.releases(assembly):
            for gene, name, value, kind, form in world.ensembl.links(assembly, release):
                database = name if kind == "id" else SYNONYM + name
                if world.ensembl.selected(form, database, assembly, release):
                    yield assembly, release, gene, database, value, form


# --------------------------------------------------------------------------- what the graph holds


def test_listings_of_the_graph_are_those_of_ensembls_tables(real_world, organism):
    """Releases, assemblies, databases per assembly and the form each database is attached to."""
    world = real_world(organism)
    api = world.api
    databases: dict[int, set[str]] = collections.defaultdict(set)
    forms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    synonyms: dict[int, set[str]] = collections.defaultdict(set)
    for assembly, _release, _gene, database, _value, form in selected_links(world):
        # decision 56: a database's synonyms are listed under its own name, the database marked as recording them
        databases[assembly].add(ids.public_name(database))
        if database.startswith(SYNONYM):
            synonyms[assembly].add(ids.public_name(database))
        else:
            forms[database][form] += 1

    assert api.releases() == sorted(world.backbone)
    assert sorted(a.assembly for a in api.describe().assemblies) == sorted(databases)
    assert api.databases_by_assembly() == {str(a): sorted(names) for a, names in sorted(databases.items())}
    assert api.external_databases() == sorted(set().union(*databases.values()))
    reported = api.database_forms()
    for database, counts in forms.items():
        assert reported[database] in counts, database
        if len(counts) == 1:
            assert reported[database] == next(iter(counts)), database
    # the lists are copies: changing one changes nothing in the graph
    api.external_databases().clear()
    assert api.external_databases()

    # the description of the graph says the same, without the network
    described = api.describe()
    assert described.organism == organism and described.snapshot_release == SNAPSHOT
    assert list(described.releases) == sorted(world.backbone)
    assert {str(r): a for r, a in world.backbone.items()} == described.serving_assembly
    assert [a.assembly for a in described.assemblies] == [
        a for a in organisms.assemblies_by_priority(organism) if a in databases
    ]
    for assembly in described.assemblies:
        held = databases[assembly.assembly]
        assert list(assembly.external_databases) == sorted(held)
        assert list(assembly.synonym_databases) == sorted(synonyms[assembly.assembly])
        assert assembly.name == organisms.assembly_name(organism, assembly.assembly)
        assert list(assembly.serves) == sorted(r for r, a in world.backbone.items() if a == assembly.assembly)
    every = set().union(*databases.values())
    assert list(described.external_databases) == sorted(every) == api.external_databases()
    assert list(described.synonym_databases) == sorted(set().union(*synonyms.values()))
    assert not any(d.startswith(SYNONYM) for d in every | set(api.final_databases()))
    assert list(described.final_databases) == api.final_databases(SNAPSHOT)
    assert described.content_digest and described.format_version == idtrack.graph.GRAPH_FORMAT_VERSION
    assert type(described).from_dict(json.loads(json.dumps(described.to_dict(), allow_nan=False))) == described


def test_source_of_a_list_of_identifiers_is_inferred_from_where_they_exist(real_world, organism):
    """The gene ids of one release, with their versions, exist together at the releases Ensembl had them all.

    Decision 79: in the mode ``release`` a release counts for an id where the assembly that serves it has the id
    there, each id once; the answer is the band of releases that a one-sided sign test on the ids found at one
    release and not at the other cannot tell from the best (:py:data:`idtrack.api.SOURCE_SIGNIFICANCE`), best first.
    """
    from idtrack.api import SOURCE_SIGNIFICANCE

    world = real_world(organism)
    api = world.api
    releases = sorted(world.backbone)
    for release in (releases[0], releases[len(releases) // 2], SNAPSHOT):
        genes = world.genes_at(release)
        ids = [f"{gene}.{version}" for gene, version in sorted(genes.items())]
        found = {
            other: {
                gene
                for gene, version in genes.items()
                if world.ensembl.genes(world.backbone[other], other).get(gene) == version
            }
            for other in releases
        }
        counts = {other: len(held) for other, held in found.items()}
        ranking = sorted(([r, c] for r, c in counts.items() if c), key=lambda pair: (-pair[1], pair[0]))
        best = ranking[0][0]

        def worse(other: int) -> bool:
            fewer, more = len(found[best] - found[other]), len(found[other] - found[best])
            n = fewer + more
            if fewer <= more:
                return False
            if n <= 1000:
                p = sum(math.comb(n, k) for k in range(fewer, n + 1)) / 2**n
            else:
                p = 0.5 * math.erfc((fewer - 0.5 - n / 2) / math.sqrt(n / 4) / math.sqrt(2))
            return p < SOURCE_SIGNIFICANCE

        band = [r for r, _c in ranking if r == best or not worse(r)]
        assert release in band
        assert api.infer_source(ids, mode="release") == band
        assert api.infer_source(ids, mode="release", report_only_winner=False) == ranking
    with pytest.raises(errors.UnknownIdentifierError, match="None of the 2 identifiers"):
        api.infer_source(["no such id", "nor this"])
    with pytest.raises(errors.InvalidArgumentError, match="is not a mode"):
        api.infer_source(["x"], mode="somewhere")


# --------------------------------------------------------------------------- batches


def test_batch_gives_the_answers_of_single_conversions_in_the_order_asked(real_world, organism):
    """A batch is the single conversions of its identifiers, repeated identifiers included, and sorts into bins."""
    world = real_world(organism)
    api = world.api
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for query in read_truth(organism):
        key = (query["from_release"], query["to_release"], query["final_database"], query["strategy"])
        groups[key].append(query)
    asked = 0
    for (from_release, to_release, final_database, strategy), queries in sorted(groups.items(), key=str):
        if len(queries) < 3:
            continue
        ids = [query["id"] for query in queries]
        ids = ids + ids[:2] + ["  " + ids[0] + " "]
        arguments = dict(from_release=from_release, to_release=to_release, final_database=final_database)
        batch = api.convert_many(ids, **arguments)
        single = [api.convert(identifier, **arguments) for identifier in ids]
        assert batch == single
        assert [result.query_id for result in batch] == ids
        for query, result in zip(queries, batch):
            assert verdict(with_y_copies_apart(world, query), result) is None, query["source"]
        with pytest.raises(AttributeError):
            batch[-3].query_id = "changed by the caller"  # results are frozen: a repeated identifier is safe

        bins = api.classify(batch, strategy)
        assert bins["input_identifiers"] == batch
        nothing = [r for r in batch if r.status in ("not_found", "unknown_version", "no_conversion")]
        elsewhere = [r for r in batch if r not in nothing and r.from_release_fallback]
        stand_in = [r for r in batch if r.no_target and r not in elsewhere]
        answered = [r for r in batch if r not in nothing and r not in elsewhere and not r.no_target]
        assert bins["matching_1_to_0"] == nothing
        assert bins["read_at_another_release"] == elsewhere
        assert all(r.reason == "read_at_another_release" for r in elsewhere if not r.no_target)
        assert bins["alternative_target_1_to_1"] + bins["alternative_target_1_to_n"] == [
            r for n in (1, 2) for r in stand_in if (len(r.targets(strategy)) == 1) is (n == 1)
        ]
        assert bins["matching_1_to_1"] == [r for r in answered if len(r.targets(strategy)) == 1]
        assert bins["matching_1_to_n"] == [r for r in answered if len(r.targets(strategy)) > 1]
        for result in bins["matching_1_to_1"]:
            same = without_version(result.targets(strategy)[0]).lower() == without_version(result.graph_id).lower()
            assert (result in bins["changed_only_1_to_1"]) is not same
        summary = api.classification_summary(bins)
        assert f"Total processed: {len(batch)}" in summary and f"1→0: {len(nothing)} " in summary
        asked += len(batch)
    assert asked >= 100


# --------------------------------------------------------------------------- the explanation of an answer


def test_explanation_leads_from_the_gene_read_to_the_target_through_real_edges(real_world, organism):
    """``explain=True``: for each target the gene the identifier was read as, the bridges used and the edges walked.

    Asked for the retired ids of the truth file that Ensembl's history gives a successor: the walk is the history
    alone (no bridge through an external id), it begins at the gene read and ends at the target, and every edge of
    it is an edge of the graph.
    """
    world = real_world(organism)
    api, graph = world.api, world.api.graph
    asked = 0
    for query in read_truth(organism):
        if query["kind"] not in ("retired_renamed", "retired_merged", "retired_split") or query["final_database"]:
            continue
        plain = api.convert(query["id"], query["from_release"], query["to_release"])
        result = api.convert(query["id"], query["from_release"], query["to_release"], explain=True)
        stripped = dataclasses.replace(
            result,
            path=None,
            candidates=tuple(dataclasses.replace(c, path=(), final_path=()) for c in result.candidates),
        )
        assert stripped == plain and plain.path is None
        assert ConversionResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result
        for candidate in result.view(query["strategy"]):
            edges = raw_steps(result.path, candidate.path + candidate.final_path)
            assert (candidate.bridges, candidate.bridge_gap) == (0, 0), query["source"]
            assert 0.0 < candidate.score <= 1.0
            assert without_version(candidate.entry_gene) == without_version(query["id"])
            assert graph.is_active(candidate.entry_gene, result.from_release)
            assert edges and edges[-1][1] == candidate.gene
            assert edges[0][0] in (candidate.entry_gene, result.graph_id)
            steps = [result.path.steps[i] for i in candidate.path]
            assert {s.kind for s in steps} <= {"entry", "history", "self_loop", "identity"}, query["source"]
            for step in steps:
                if step.kind == "history":
                    assert step.old_release < step.new_release and step.reason
            for source, target, *_rest in edges:
                assert target in graph[source], (source, target)
            for (_a, joint, *_x), (following, *_y) in zip(edges, edges[1:]):
                assert joint == following
        asked += 1
    assert asked >= 6


# --------------------------------------------------------------------------- wrong arguments


def test_wrong_arguments_are_refused_before_any_search_and_say_what_is_valid(real_world, organism):
    """Releases outside the graph, unknown databases, identifiers that are no strings: errors that name the limits."""
    world = real_world(organism)
    api = world.api
    gene = sorted(world.genes_at(SNAPSHOT))[0]
    first = min(world.backbone)

    with pytest.raises(
        errors.UnknownReleaseError,
        match=rf"`to_release`={SNAPSHOT + 1} .* covers the Ensembl releases {first} to {SNAPSHOT}",
    ):
        api.convert(gene, to_release=SNAPSHOT + 1)
    with pytest.raises(errors.UnknownReleaseError, match=rf"`from_release`={first - 1} "):
        api.convert(gene, from_release=first - 1)
    with pytest.raises(errors.UnknownDatabaseError, match="`final_database`='No Such Database' is not a database"):
        api.convert(gene, final_database="No Such Database")
    with pytest.raises(errors.UnknownDatabaseError, match="`from_database`='No Such Database' is not an external"):
        api.convert(gene, from_database="No Such Database")
    with pytest.raises(errors.MissingValueError, match="`identifier` is empty"):
        api.convert("   ")
    # decisions 81 and 46 (g): in a batch, an empty string is that item's error
    refused = api.convert_many([gene, " "])[1]
    assert (refused.status, refused.error.type) == ("error", "MissingValueError")
    # the same refusals for an identifier that is not in the graph: nothing is searched first
    with pytest.raises(errors.UnknownReleaseError, match="`to_release`"):
        api.convert("not an identifier of anything", to_release=SNAPSHOT + 1)

    # a database of the earlier assembly is no target at a release of the last one, and the other way round
    by_assembly = api.databases_by_assembly()
    last, earliest = str(world.backbone[SNAPSHOT]), str(world.backbone[first])
    only_earlier = sorted(set(by_assembly[earliest]) - set(by_assembly[last]))
    only_last = sorted(set(by_assembly[last]) - set(by_assembly[earliest]))
    for database, release in [(d, SNAPSHOT) for d in only_earlier[:1]] + [(d, first) for d in only_last[:1]]:
        with pytest.raises(errors.UnknownDatabaseError, match="`final_database`"):
            api.convert(gene, to_release=release, final_database=database)
    assert only_last


@pytest.mark.wrong_argument_type
def test_arguments_of_the_wrong_type_are_type_errors(real_world, organism):
    """An API of no graph, a release that is no integer, an identifier that is no string, a strategy that is none."""
    world = real_world(organism)
    api = world.api
    gene = sorted(world.genes_at(SNAPSHOT))[0]

    with pytest.raises(errors.ArgumentTypeError, match="API.load"):
        idtrack.API(world.repository)  # type: ignore[arg-type]

    for release in (True, 100.0, "100", None.__class__):
        with pytest.raises(errors.ArgumentTypeError, match="`to_release` must be an integer"):
            api.convert(gene, to_release=release)
    with pytest.raises(errors.ArgumentTypeError, match="`identifier` must be a string"):
        api.convert(12345)
    with pytest.raises(errors.InvalidArgumentError, match="`strategy`='first' is not a strategy"):
        api.convert(gene).view("first")
    with pytest.raises(errors.ArgumentTypeError, match=r"non-string items, at positions \[1\]"):
        api.convert_many([gene, 12345])
    assert api.convert_many([gene, None])[1].error.type == "MissingValueError"  # an empty cell: that item's
    assert api.convert(gene, to_release=np.int64(SNAPSHOT)).to_release == SNAPSHOT


# --------------------------------------------------------------------------- a dataset


def test_dataset_of_an_old_release_keeps_every_count_and_sums_what_ensembl_merged(real_world, organism, tmp_path):
    """``HarmonizeFeatures`` on the gene ids of an old release, counts summed where several ids became one gene.

    The dataset holds every gene id that the tables have at the release from which the truth file has the most
    merged ids. Read from the tables and the truth file: an id that Ensembl still has at the target release is its
    own feature there (with the symbol the symbol database gives it), an id of the truth file that was merged into
    another gene is part of that gene's feature, and no count is lost.
    """
    ad = pytest.importorskip("anndata")
    pytest.importorskip("scipy")

    from idtrack import HarmonizeFeatures

    world = real_world(organism)
    merged = [
        q for q in read_truth(organism) if q["kind"] == "retired_merged" and q["from_release"] and len(q["expect"]) == 1
    ]
    release = collections.Counter(q["from_release"] for q in merged).most_common(1)[0][0]
    merged = {q["id"]: q["expect"][0] for q in merged if q["from_release"] == release and q["to_release"] == SNAPSHOT}
    assert merged
    then, now = world.genes_at(release), world.genes_at(SNAPSHOT)
    names = sorted(then)
    counts = np.arange(1, 2 * len(names) + 1, dtype=np.float32).reshape(2, len(names))
    path = tmp_path / "old.h5ad"
    ad.AnnData(X=counts, var=pd.DataFrame(index=names)).write_h5ad(path)
    symbol_database = next(d for d in organisms.SYMBOL_DATABASES[organism] if d in world.api.final_databases(SNAPSHOT))

    harmonizer = HarmonizeFeatures(
        "real", {"old": str(path)}, str(tmp_path / "project"), world.repository, SNAPSHOT,
        from_release_by_dataset={"old": release}, api=world.api, verbosity=0,
    )  # fmt: skip
    data, n_in, n_out = harmonizer.feature_harmonizer("old", merge="sum")
    column = {name: i for i, name in enumerate(names)}
    values = np.asarray(data.X)

    assert (n_in, n_out) == (len(names), data.n_vars) and n_out <= n_in
    assert float(values.sum()) == float(counts.sum())  # nothing dropped, nothing counted twice
    assert data.var_names.is_unique
    symbols: dict[str, set[str]] = collections.defaultdict(set)
    for gene, name, value, kind, form in world.ensembl.links(world.backbone[SNAPSHOT], SNAPSHOT):
        if (
            kind == "id"
            and name == symbol_database
            and world.ensembl.selected(form, name, world.backbone[SNAPSHOT], SNAPSHOT)
        ):
            symbols[gene].add(value)
    kept = 0
    for gene in names:
        if gene not in now:
            continue
        if gene.startswith("LRG_"):
            # decision 96: an LRG record's counts go to the gene the record describes, never to the record
            holders = [i for i, query in enumerate(data.var["Query ID"]) if gene in str(query).split(";")]
            assert len(holders) == 1 and not data.var_names[holders[0]].startswith("LRG_"), gene
            continue
        feature = f"{gene}.{now[gene]}"
        assert feature in data.var_names, gene
        row = data.var.loc[feature]
        sources = row["Query ID"].split(";")
        assert gene in sources and row["matching_ensembl_gene"] == feature
        assert row["idtrack_status"] in ("converted", "converted_no_final_id", "merged", "merged_no_final_id")
        np.testing.assert_array_equal(
            values[:, data.var_names.get_loc(feature)], counts[:, [column[s] for s in sources]].sum(axis=1)
        )
        if symbols[gene]:
            assert row[symbol_database] in symbols[gene] and not row["idtrack_status"].endswith("no_final_id")
        else:
            assert pd.isna(row[symbol_database]) and row["idtrack_status"].endswith("no_final_id")
        kept += 1
    assert kept >= 50
    for old, successor in merged.items():
        row = data.var.loc[successor]
        assert old in row["Query ID"].split(";") and row["idtrack_status"].startswith("merged"), old


# --------------------------------------------------------------------------- defaults, strategies, determinism


def test_default_target_is_the_snapshot_release_and_best_is_part_of_all(real_world, organism):
    """Without ``to_release`` the target is the last release of the graph; the best answer is among all answers."""
    world = real_world(organism)
    api = world.api
    gene, version = sorted(world.genes_at(SNAPSHOT).items())[0]
    assert api.convert(gene).to_release == SNAPSHOT
    assert api.convert(gene).targets() == (f"{gene}.{version}",)
    assert [c.gene for c in api.engine.convert(gene).candidates] == [f"{gene}.{version}"]

    asked = 0
    for query in read_truth(organism):
        arguments = dict(
            from_release=query["from_release"], to_release=query["to_release"], final_database=query["final_database"]
        )
        result = api.convert(query["id"], **arguments)
        best = {(c.gene, c.target) for c in result.view("best")}
        everyone = {(c.gene, c.target) for c in result.view("all")}
        assert best <= everyone, query["source"]
        assert bool(best) == bool(everyone), query["source"]
        asked += 1
    assert asked >= 100


def test_every_answer_comes_best_first(real_world, organism):
    """With ``strategy="all"`` the answers come best first: the best answer leads, in its own order.

    Asked for every query of the truth file with both strategies: the first pair of all answers is the first pair
    of the best answer, and the genes of the best answer are the first genes of all answers, in the same order.
    The pig id ``ENSSSCG00000042716`` (retired after release 106) was split into two genes with the same mapping
    score, of which the richer annotated one is the best answer although the other comes first by id.
    """
    world = real_world(organism)
    api = world.api
    wrong = []
    with_more = 0
    for query in read_truth(organism):
        arguments = dict(
            from_release=query["from_release"], to_release=query["to_release"], final_database=query["final_database"]
        )
        result = api.convert(query["id"], **arguments)
        best = [(c.gene, c.target) for c in result.view("best")]
        everyone = [(c.gene, c.target) for c in result.view("all")]
        best_genes = list(dict.fromkeys(gene for gene, _target in best))
        all_genes = list(dict.fromkeys(gene for gene, _target in everyone))
        with_more += len(all_genes) > len(best_genes) > 0
        if best and not (
            everyone[0] == best[0]
            and all_genes[: len(best_genes)] == best_genes
            and result.targets("all")[0] == result.targets("best")[0]
        ):
            wrong.append((query["source"], best, everyone))
    assert not wrong, wrong[:5]
    assert with_more >= 5  # queries with more genes among all answers than in the best one

    if organism == "sus_scrofa":
        split = dict(from_release=106, to_release=115)
        result = api.convert("ENSSSCG00000042716", explain=True, **split)
        assert (result.targets("best"), result.tie) == (("ENSSSCG00000062505.1",), False)
        assert result.targets("all") == ("ENSSSCG00000062505.1", "ENSSSCG00000058225.1")
        scores = {c.gene: c.score for c in result.candidates}
        assert scores["ENSSSCG00000062505.1"] == scores["ENSSSCG00000058225.1"]
        assert result.decided_by == "annotation"  # the annotation decided


SEEDED = r"""
import json, sys
import idtrack
api = idtrack.API.build(sys.argv[2], int(sys.argv[3]), sys.argv[1])
answers = []
for line in open(sys.argv[4], encoding="utf-8"):
    q = json.loads(line)
    arguments = (q["id"], q["from_release"], q["to_release"], q["final_database"])
    answers.append(api.convert(*arguments, explain=True).to_dict())
print(json.dumps(answers))
"""


def test_answers_do_not_depend_on_the_hash_seed_of_the_process(real_world, organism, tmp_path):
    """Every query of the truth file, with its explanation: byte-identical output under three hash seeds."""
    import os
    import subprocess  # noqa: S404
    import sys
    from pathlib import Path

    world = real_world(organism)
    script = tmp_path / "answers.py"
    script.write_text(SEEDED)
    root = str(Path(idtrack.__file__).resolve().parents[1])
    outputs = []
    for seed in ("0", "1", "4242"):
        environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        environment["PYTHONPATH"] = os.pathsep.join(
            [root] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        completed = subprocess.run(  # noqa: S603
            [sys.executable, str(script), world.repository, organism, str(SNAPSHOT), str(truth_path(organism))],
            capture_output=True, text=True, env=environment, check=False,
        )  # fmt: skip
        assert completed.returncode == 0, completed.stderr[-2000:]
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1
    here = [
        world.api.convert(q["id"], q["from_release"], q["to_release"], q["final_database"], explain=True).to_dict()
        for q in read_truth(organism)
    ]
    assert json.loads(outputs[0]) == here  # and the process that built the graph answers as the ones that loaded it
