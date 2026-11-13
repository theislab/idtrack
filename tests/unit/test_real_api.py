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
import json

import numpy as np
import pandas as pd
import pytest

import idtrack
from idtrack._db import DB
from tests._real_tables import ORGANISMS, SNAPSHOT, read_truth, tables_path, truth_path, verdict, without_version

AVAILABLE = [organism for organism in ORGANISMS if tables_path(organism).is_file() and truth_path(organism).is_file()]
pytestmark = pytest.mark.parametrize("organism", AVAILABLE)
SYNONYM = DB.synonym_id_nodes_prefix


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
    for assembly, _release, _gene, database, _value, form in selected_links(world):
        databases[assembly].add(database)
        forms[database][form] += 1

    assert api.list_ensembl_releases() == api.graph_releases() == sorted(world.backbone)
    assert api.list_genome_assemblies() == sorted(databases)
    assert api.list_external_databases_by_assembly() == {a: sorted(names) for a, names in sorted(databases.items())}
    assert api.list_external_databases() == sorted(set().union(*databases.values()))
    reported = api.external_database_forms()
    for database, counts in forms.items():
        assert reported[database] in counts, database
        if len(counts) == 1:
            assert reported[database] == next(iter(counts)), database
    # the lists are copies: changing one changes nothing in the graph
    api.list_external_databases().clear()
    assert api.list_external_databases()


def test_source_of_a_list_of_identifiers_is_inferred_from_where_they_exist(real_world, organism):
    """The gene ids of one release, with their versions, exist together at the releases Ensembl had them all."""
    world = real_world(organism)
    api = world.api
    releases = sorted(world.backbone)
    for release in (releases[0], releases[len(releases) // 2], SNAPSHOT):
        genes = world.genes_at(release)
        ids = [f"{gene}.{version}" for gene, version in sorted(genes.items())]
        # a release counts for an id when some assembly has the id there (human GRCh37 beside GRCh38)
        counts = collections.Counter()
        for other in releases:
            tables = [
                world.ensembl.genes(assembly, other)
                for assembly in sorted({int(a) for a in world.tables.indices})
                if other in world.ensembl.releases(assembly)
            ]
            counts[other] = sum(
                1 for gene, version in genes.items() if any(there.get(gene) == version for there in tables)
            )
        best = [r for r in releases if counts[r] == max(counts.values())]
        assert release in best
        assert api.infer_identifier_source(ids, mode="ensembl_release") == best
        ranking = api.infer_identifier_source(ids, mode="ensembl_release", report_only_winner=False)
        assert [pair for pair in ranking if pair[1]] == sorted(
            ([r, c] for r, c in counts.items() if c), key=lambda pair: (-pair[1], pair[0])
        )
    with pytest.raises(ValueError, match="None of the 2 identifiers"):
        api.infer_identifier_source(["no such id", "nor this"])
    with pytest.raises(ValueError, match="Unknown mode"):
        api.infer_identifier_source(["x"], mode="somewhere")


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
        arguments = dict(
            from_release=from_release, to_release=to_release, final_database=final_database, strategy=strategy
        )
        batch = api.convert_identifier_multiple(ids, verbose=False, **arguments)
        single = [api.convert_identifier(identifier, **arguments) for identifier in ids]
        assert batch == single
        assert [result["query_id"] for result in batch] == ids
        for query, result in zip(queries, batch):
            assert verdict(query, result) is None, query["source"]
        batch[-3]["target_id"].append("changed by the caller")
        assert batch[0]["target_id"] == single[0]["target_id"]  # a repeated identifier has a result of its own

        bins = api.classify_multiple_conversion(batch)
        assert bins["input_identifiers"] == batch
        nothing = [r for r in batch if r["no_corresponding"] or r["no_conversion"]]
        stand_in = [r for r in batch if r["no_target"]]
        answered = [r for r in batch if r not in nothing and not r["no_target"]]
        assert bins["matching_1_to_0"] == nothing
        assert bins["alternative_target_1_to_1"] + bins["alternative_target_1_to_n"] == [
            r for n in (1, 2) for r in stand_in if (len(r["target_id"]) == 1) is (n == 1)
        ]
        assert bins["matching_1_to_1"] == [r for r in answered if len(r["target_id"]) == 1]
        assert bins["matching_1_to_n"] == [r for r in answered if len(r["target_id"]) > 1]
        for result in bins["matching_1_to_1"]:
            same = without_version(result["target_id"][0]).lower() == without_version(result["graph_id"]).lower()
            assert (result in bins["changed_only_1_to_1"]) is not same
        summary = api.binned_conversion_summary(bins)
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
    api, graph = world.api, world.api.track.graph
    asked = 0
    for query in read_truth(organism):
        if query["kind"] not in ("retired_renamed", "retired_merged", "retired_split") or query["final_database"]:
            continue
        plain = api.convert_identifier(
            query["id"], query["from_release"], query["to_release"], strategy=query["strategy"]
        )
        result = api.convert_identifier(
            query["id"], query["from_release"], query["to_release"], strategy=query["strategy"], explain=True
        )
        assert {k: v for k, v in result.items() if k != "the_path"} == plain and "the_path" not in plain
        assert json.loads(json.dumps(result, allow_nan=False)) == result
        assert [[p["ensembl_gene"], p["target_id"]] for p in result["the_path"]] == result["last_node"]
        for path in result["the_path"]:
            edges = path["edges"]
            assert (path["external_jump"], path["external_gap"]) == (0, 0), query["source"]
            assert 0.0 < path["score"] <= 1.0
            assert without_version(path["entry_gene"]) == without_version(query["id"])
            assert graph.is_active(path["entry_gene"], result["from_release"])
            assert edges and edges[-1][1] == path["ensembl_gene"]
            assert edges[0][0] in (path["entry_gene"], result["graph_id"])
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

    with pytest.raises(ValueError, match=rf"`to_release`={SNAPSHOT + 1} .* valid releases are {first} to {SNAPSHOT}"):
        api.convert_identifier(gene, to_release=SNAPSHOT + 1)
    with pytest.raises(ValueError, match=rf"`from_release`={first - 1} "):
        api.convert_identifier(gene, from_release=first - 1)
    with pytest.raises(ValueError, match="`final_database`='No Such Database' is not a database"):
        api.convert_identifier(gene, final_database="No Such Database")
    with pytest.raises(ValueError, match="`from_database`='No Such Database' is not an external database"):
        api.convert_identifier(gene, from_database="No Such Database")
    with pytest.raises(ValueError, match="`identifier` is empty"):
        api.convert_identifier("   ")
    with pytest.raises(ValueError, match=r"1 empty strings, at positions \[1\]"):
        api.convert_identifier_multiple([gene, " "], verbose=False)
    # the same refusals for an identifier that is not in the graph: nothing is searched first
    with pytest.raises(ValueError, match="`to_release`"):
        api.convert_identifier("not an identifier of anything", to_release=SNAPSHOT + 1)

    with pytest.raises(RuntimeError, match="Call `API.build_graph"):
        idtrack.API(local_repository=world.repository).convert_identifier(gene)

    # a database of the earlier assembly is no target at a release of the last one, and the other way round
    by_assembly = api.list_external_databases_by_assembly()
    last, earliest = world.backbone[SNAPSHOT], world.backbone[first]
    only_earlier = sorted(set(by_assembly[earliest]) - set(by_assembly[last]))
    only_last = sorted(set(by_assembly[last]) - set(by_assembly[earliest]))
    for database, release in [(d, SNAPSHOT) for d in only_earlier[:1]] + [(d, first) for d in only_last[:1]]:
        with pytest.raises(ValueError, match="`final_database`"):
            api.convert_identifier(gene, to_release=release, final_database=database)
    assert only_last


@pytest.mark.wrong_argument_type
def test_arguments_of_the_wrong_type_are_type_errors(real_world, organism):
    """A release that is no integer, an identifier that is no string, a strategy that is none of the two."""
    world = real_world(organism)
    api = world.api
    gene = sorted(world.genes_at(SNAPSHOT))[0]

    for release in (True, 100.0, "100", None.__class__):
        with pytest.raises(TypeError, match="`to_release` must be an integer"):
            api.convert_identifier(gene, to_release=release)
    with pytest.raises(TypeError, match="`identifier` must be a string"):
        api.convert_identifier(12345)
    with pytest.raises(ValueError, match="Invalid strategy='first'"):
        api.convert_identifier(gene, strategy="first")
    with pytest.raises(TypeError, match=r"non-string items, at positions \[1\]"):
        api.convert_identifier_multiple([gene, None], verbose=False)
    assert api.convert_identifier(gene, to_release=np.int64(SNAPSHOT))["to_release"] == SNAPSHOT


# --------------------------------------------------------------------------- a dataset


def test_dataset_of_an_old_release_keeps_every_count_and_sums_what_ensembl_merged(real_world, organism, tmp_path):
    """``HarmonizeFeatures`` on the gene ids of an old release, counts summed where several ids became one gene.

    The dataset holds every gene id that the tables have at the release from which the truth file has the most
    merged ids. Read from the tables and the truth file: an id that Ensembl still has at the target release is its
    own feature there (with the symbol the symbol database gives it), an id of the truth file that was merged into
    another gene is part of that gene's feature, and no count is lost.
    """
    import anndata as ad

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
    symbol_database = next(d for d in DB.symbol_databases[organism] if d in world.api.valid_final_databases(SNAPSHOT))

    harmonizer = HarmonizeFeatures(
        "real", {"old": str(path)}, str(tmp_path / "project"), world.repository, SNAPSHOT,
        source_releases={"old": release}, api=world.api, verbose_level=0,
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
    assert api.convert_identifier(gene)["to_release"] == SNAPSHOT
    assert api.convert_identifier(gene)["target_id"] == [f"{gene}.{version}"]
    assert list(api.track.convert(gene)) == [f"{gene}.{version}"]

    asked = 0
    for query in read_truth(organism):
        arguments = dict(
            from_release=query["from_release"], to_release=query["to_release"], final_database=query["final_database"]
        )
        best = api.convert_identifier(query["id"], strategy="best", **arguments)
        everyone = api.convert_identifier(query["id"], strategy="all", **arguments)
        assert set(map(tuple, best["last_node"])) <= set(map(tuple, everyone["last_node"])), query["source"]
        for key in (
            "graph_id",
            "match",
            "from_release",
            "alternative_locus_genes",
            "no_corresponding",
            "no_conversion",
            "tie",
        ):
            assert best[key] == everyone[key], (query["source"], key)
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
        best = api.convert_identifier(query["id"], strategy="best", **arguments)
        everyone = api.convert_identifier(query["id"], strategy="all", **arguments)
        best_genes = list(dict.fromkeys(gene for gene, _target in best["last_node"]))
        all_genes = list(dict.fromkeys(gene for gene, _target in everyone["last_node"]))
        with_more += len(all_genes) > len(best_genes) > 0
        if best["last_node"] and not (
            everyone["last_node"][0] == best["last_node"][0]
            and all_genes[: len(best_genes)] == best_genes
            and everyone["target_id"][0] == best["target_id"][0]
        ):
            wrong.append((query["source"], best["last_node"], everyone["last_node"]))
    assert not wrong, wrong[:5]
    assert with_more >= 5  # queries with more genes among all answers than in the best one

    if organism == "sus_scrofa":
        split = dict(from_release=106, to_release=115)
        best = api.convert_identifier("ENSSSCG00000042716", strategy="best", **split)
        everyone = api.convert_identifier("ENSSSCG00000042716", strategy="all", explain=True, **split)
        assert (best["target_id"], best["tie"]) == (["ENSSSCG00000062505.1"], False)
        assert everyone["target_id"] == ["ENSSSCG00000062505.1", "ENSSSCG00000058225.1"]
        scores = {path["ensembl_gene"]: path["score"] for path in everyone["the_path"]}
        assert scores["ENSSSCG00000062505.1"] == scores["ENSSSCG00000058225.1"]  # the annotation decided


SEEDED = r"""
import json, sys
import idtrack
api = idtrack.API(local_repository=sys.argv[1])
api.build_graph(sys.argv[2], int(sys.argv[3]))
answers = []
for line in open(sys.argv[4], encoding="utf-8"):
    q = json.loads(line)
    for strategy in ("best", "all"):
        arguments = (q["id"], q["from_release"], q["to_release"], q["final_database"], strategy)
        answers.append(api.convert_identifier(*arguments, explain=True))
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
        world.api.convert_identifier(
            q["id"], q["from_release"], q["to_release"], q["final_database"], strategy, explain=True
        )
        for q in read_truth(organism)
        for strategy in ("best", "all")
    ]
    assert json.loads(outputs[0]) == here  # and the process that built the graph answers as the ones that loaded it
