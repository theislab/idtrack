"""The consumer contract: every call that rosetta (idtrack's one dependent) makes, through the public API only.

rosetta reads idtrack in two modules: its graph units (build, record, self-tests, a check against another build,
and a reader that converts gene ids) and its gene mapping (names per database and release, a gene's symbols at a
release, every path from genes of one release to another). This module makes each of those calls on the graphs
built from the cut tables of ``tests/data/real`` and compares the answers with
``tests/data/real/consumer_contract.json``: the answers of the same calls made with the API they replace (the
release before this layout), so that a change of what rosetta gets is a failing test. The functions below are
written as rosetta writes them, against the public names; nothing here reads a private member.

This is a **change detector, not a test of correctness**: its expected answers are an earlier run of idtrack itself,
not Ensembl's records. A difference passes only where it is explained (:py:func:`changed_only_by_refused_bridges`):
every gene lost is the target of a bridge whose refusal Ensembl's tables confirm through an oracle that shares no code
with the engine (:py:func:`refusal_confirmed`), and every gene gained is reached through a bridge (the gain is the
engine's word, not confirmed from the tables). Whether an answer is right is tested against Ensembl's records
elsewhere (``test_real_truth.py``, ``test_real_decisions.py``, ``test_history_travel.py``).
"""

from __future__ import annotations

import importlib.resources
import inspect
import json
import os
import random
import re
from pathlib import Path

import pytest

import idtrack
from idtrack import API, GraphChecks, ids, organisms, settings
from idtrack.build.builder import GraphMaker
from idtrack.build.database_manager import DatabaseManager
from idtrack.build.externals import load_externals_yaml
from idtrack.graph import files
from tests._real_tables import SNAPSHOT, tables_path

EXPECTED_PATH = Path(__file__).resolve().parents[1] / "data" / "real" / "consumer_contract.json"
EXPECTED = json.loads(EXPECTED_PATH.read_text()) if EXPECTED_PATH.is_file() else {"organisms": {}}
ORGANISMS = [organism for organism in sorted(EXPECTED["organisms"]) if tables_path(organism).is_file()]
pytestmark = pytest.mark.parametrize("organism", ORGANISMS)

TIERS = ("symbol", "name", "synonym")
VERSIONED = re.compile(r"^(ENS[A-Z]*G\d{11})\.\d+$")
CLASSES = (
    "matching_1_to_0",
    "matching_1_to_1",
    "matching_1_to_n",
    "alternative_target_1_to_1",
    "alternative_target_1_to_n",
)
CHANGED = {"matching_1_to_1": "changed_only_1_to_1", "matching_1_to_n": "changed_only_1_to_n"}
SELF_TESTS = (
    "is_id_functions_consistent_ensembl",
    "is_id_functions_consistent_ensembl_2",
    "is_id_functions_consistent_external",
    "is_range_functions_robust",
    "is_base_is_range_correct",
    "is_combined_summary_complete",
    "is_edge_with_same_nts_only_at_backbone_nodes",
    "is_node_consistency_robust",
)


def unversioned(target: str) -> str:
    """Return the target without its version."""
    return VERSIONED.sub(r"\1", str(target))


def mask_of(releases) -> int:
    """Return the bit mask of the releases."""
    return sum(1 << r for r in releases)


# ------------------------------------------------------------------------------------- rosetta's calls, on the new API


def no_corresponding(result) -> bool:
    """The identifier is not in the graph (status ``not_found`` or ``unknown_version``)."""
    return result.status in ("not_found", "unknown_version")


def draw_queries(api: API, organism: str, snapshot: int, assembly: int, n: int, seed: int, old_releases: int):
    """Rosetta's `idtrack_graph.draw_queries`: symbols of the snapshot and gene ids of earlier releases."""
    graph = api.graph
    earlier = [r for r in api.releases() if r < snapshot]
    picked = sorted({earlier[round(i * (len(earlier) - 1) / max(old_releases - 1, 1))] for i in range(old_releases)})
    serving = graph.serving_assembly
    wanted = {(organisms.ASSEMBLY_LABELS[serving[r]][ids.BACKBONE_FORM], serving[r], r): ("gene", r) for r in picked}
    symbol_dbs = [d for d in organisms.SYMBOL_DATABASES[organism] if d in api.external_databases()]
    wanted |= {(d, assembly, snapshot): ("symbol", snapshot) for d in symbol_dbs}
    found: dict[tuple, set] = {v: set() for v in wanted.values()}
    for name in graph.nodes:
        if graph.is_backbone(name) and graph.version(name) in ids.SENTINEL_VERSIONS:
            continue
        try:
            trios = graph.trios(name)
        except KeyError:
            continue
        for trio in trios:
            if trio in wanted:
                kind, release = wanted[trio]
                found[(kind, release)].add(graph.base_id(name) if kind == "gene" else name)
    rng = random.Random(seed)  # noqa: S311 - a seeded draw of test queries, not a secret
    pool = sorted(found[("symbol", snapshot)])
    symbols = [{"id": i, "kind": "symbol", "from_release": None} for i in rng.sample(pool, min(n // 2, len(pool)))]
    genes = n - len(symbols)
    per_release = [genes // len(picked) + (1 if i < genes % len(picked) else 0) for i in range(len(picked))]
    queries = []
    for release, k in zip(picked, per_release, strict=True):
        pool = sorted(found[("gene", release)])
        queries += [{"id": i, "kind": "gene", "from_release": release} for i in rng.sample(pool, min(k, len(pool)))]
    return queries + symbols


def answer(api: API, queries: list[dict], to_release: int) -> list[dict]:
    """Rosetta's `idtrack_graph.answer`: one conversion per query, its best answer and how it was read."""
    out = []
    for q in queries:
        try:
            r = api.convert(q["id"], from_release=q["from_release"], to_release=to_release)
            out.append(
                {
                    "graph_id": r.graph_id,
                    "match": r.match,
                    "from_release": r.from_release,
                    "target_id": list(r.targets("best")),
                    "tie": r.tie,
                    "no_corresponding": no_corresponding(r),
                    "no_conversion": r.status == "no_conversion",
                    "no_target": r.no_target,
                }
            )
        except idtrack.IdtrackError as error:
            out.append({"error": f"{type(error).__name__}: {error}"})
    return out


def bins(results: list, strategy: str) -> tuple[list[str], list[bool]]:
    """Rosetta's `idtrack_graph.bins`: the class of each result and whether it is in a `changed_only` bin."""
    classified = API.classify(results, strategy)
    of = {}
    for name in CLASSES:
        for item in classified[name]:
            of[id(item)] = name
    changed = {id(item) for name in CHANGED.values() for item in classified[name]}
    return [of[id(r)] for r in results], [id(r) in changed for r in results]


def convert(api: API, ids_: list[str], to_release: int, from_release=None, from_database=None) -> list[dict]:
    """Rosetta's `idtrack_graph.convert`: one search per id, its best answer and every target it reaches."""
    distinct = list(dict.fromkeys(ids_))
    results = api.convert_many(
        distinct, from_release=from_release, to_release=to_release, final_database=ids.GENE, from_database=from_database
    )
    best_class, best_changed = bins(results, "best")
    all_class, _ = bins(results, "all")
    rows = []
    for r, cls, changed, cls_all in zip(results, best_class, best_changed, all_class, strict=True):
        rows.append(
            {
                "query_id": r.query_id,
                "from_release": r.from_release,
                "from_release_inferred": r.from_release_inferred,
                "from_release_fallback": r.from_release_fallback,
                "graph_id": r.graph_id,
                "match": r.match,
                "from_databases": list(r.from_databases),
                "targets": list(r.targets("best")),
                "target_genes": [c.gene for c in r.view("best")],
                "tie": r.tie,
                "class": cls,
                "changed": changed,
                "targets_all": list(r.targets("all")),
                "class_all": cls_all,
                "no_corresponding": no_corresponding(r),
                "no_conversion": r.status == "no_conversion",
                "no_target": r.no_target,
                "no_target_genes": list(r.no_target_genes),
                "alternative_locus_genes": list(r.alternative_locus_genes),
                "known_versions": list(r.known_versions),
                "error": "" if r.error is None else f"{r.error.type}: {r.error.message}",
            }
        )
    return rows


class GraphView:
    """rosetta's `genes.GraphView`, on the public API: versions, names per tier and release, symbols, paths."""

    def __init__(self, api: API, symbol_databases: list[str], name_databases: list[str]):
        """Keep the API, its graph and the database tiers of names."""
        self.api = api
        self.G = api.graph
        self.releases = [int(r) for r in self.G.releases]
        self.symbol_databases = set(symbol_databases)
        self.tier = {db: "symbol" for db in symbol_databases} | {db: "name" for db in name_databases}
        self._versions: dict = {}

    def versions(self, base: str) -> list[tuple[str, int]]:
        """Return the gene versions of a base id with the mask of their reference releases."""
        if base not in self._versions:
            out = []
            if base in self.G.nodes:
                for v in self.G.versions_of(base):
                    if self.G.node_type(v) == ids.GENE:
                        out.append((v, self.G.active_releases(v).mask & ~self.G.non_reference_releases(v).mask))
            self._versions[base] = out
        return self._versions[base]

    def name_tiers(self, text: str, fold: bool) -> dict[str, dict[str, int]]:
        """Return the genes that a text names, per tier, with the mask of the releases of each."""
        out: dict[str, dict[str, int]] = {t: {} for t in TIERS}
        for match in self.api.genes_named(text, fold_case=fold):
            # decision 56: a synonym is reported under its database's own name, marked `synonym`
            tier = self.tier.get(match.database)
            if tier is not None and match.synonym:
                tier = "synonym"
            if tier is not None:
                base = self.G.base_id(match.gene)
                out[tier][base] = out[tier].get(base, 0) | mask_of(match.releases)
        return out

    def symbols(self, base: str, release: int, synonyms: bool) -> set[str]:
        """Return the lower-case symbols of a base id at a release."""
        out = set()
        for v, mask in self.versions(base):
            if not mask >> release & 1:
                continue
            for match in self.api.names_of(v, release):
                if match.database in self.symbol_databases and (synonyms or not match.synonym):
                    out.add(match.name.lower())
        return out

    def convert(self, genes: list[str], from_release, to_release: int) -> dict[str, dict]:
        """Convert genes to a release and return, per gene, every path with its bridges, the error and the release."""
        if not genes:
            return {}
        results = self.api.convert_many(
            genes, from_release=from_release, to_release=to_release, final_database=ids.GENE
        )
        out = {}
        for g, r in zip(genes, results, strict=True):
            out[g] = {
                "paths": [[unversioned(c.target), int(c.bridges)] for c in r.view("all")],
                "error": "" if r.error is None else r.error.type,
                "from_release": r.from_release,
            }
        return out


# --------------------------------------------------------------------------------------------------------------- tests


def decided(world, rows: list[dict]) -> list[dict]:
    """Return d4c2dda's answers as this version's decisions change them: decision 95.

    A name that the X and Y copies of a pseudoautosomal gene share answers the X gene, the Y copy listed apart (from
    release 110 Ensembl gives the Y copies ids of their own); a Y copy is at the place of an X gene of the answer, read
    from the tables' places (:py:meth:`tests._real_tables.Ensembl.y_twins`).
    """
    assembly = world.backbone[SNAPSHOT]
    twins = world.ensembl.y_twins(assembly, SNAPSHOT)
    out = []
    for row in rows:
        targets = row.get("targets", row.get("target_id"))
        if not targets:
            out.append(row)
            continue
        genes = {t.rpartition(".")[0] for t in targets}
        y = [t for t in targets if twins.get(t.rpartition(".")[0], set()) & genes]
        if not y:
            out.append(row)
            continue
        row = dict(row)
        kept = [t for t in targets if t not in y]
        if "target_id" in row:
            row.update(target_id=kept, tie=len(kept) > 1)
        else:
            row.update(
                targets=kept,
                target_genes=[g for g in row["target_genes"] if g not in y],
                targets_all=[t for t in row["targets_all"] if t not in y],
                tie=len(kept) > 1,
                alternative_locus_genes=sorted(set(row["alternative_locus_genes"]) | set(y)),
            )
            for key in ("class", "class_all"):
                one = len(kept if key == "class" else row["targets_all"]) == 1
                row[key] = row[key].replace("1_to_n", "1_to_1") if one else row[key]
        out.append(row)
    return out


def tiers_decided(world, view: GraphView, text: str, fold: bool, got: dict, expected: dict) -> None:
    """Check rosetta's names of a text per tier against d4c2dda's, as decisions 83 and 104 on the side of the tables change them.

    The graph holds a value a database wrote with the punctuation of a sentence as the plain name (UniProt's ``CLCN5;``
    is ``CLCN5``), and a row of a database that continues a selected one under the selected database (decision 104:
    pig HGNC projections filed as ``HGNC_curated_gene`` up to release 60 are ``HGNC Symbol``'s), so a gene can be
    named by the text at more releases than before. A release a gene gained in a tier is accepted only where the tables
    give the gene, at that release and in a database of that tier, a value that reads as the text and is written
    otherwise (:py:meth:`tests._real_tables.Ensembl.read_changed`), or that Ensembl wrote under another database that
    continues it (:py:meth:`tests._real_tables.Ensembl.refiled`); nothing may be lost.
    """
    assert set(got) == set(expected), (text, got, expected)
    for tier in expected:
        for gene in set(got[tier]) | set(expected[tier]):
            before, now = int(expected[tier].get(gene, 0)), int(got[tier].get(gene, 0))
            assert before & ~now == 0, (text, tier, gene, "lost releases")
            gained = now & ~before
            for release in (r for r in range(gained.bit_length()) if gained >> r & 1):
                assembly = world.backbone.get(release)
                confirmed = assembly is not None and any(
                    row[0] == gene
                    and (row[2].lower() == text.lower() if fold else row[2] == text)
                    and view.tier.get(row[1]) is not None
                    and ("synonym" if row[3] == "synonym" else view.tier[row[1]]) == tier
                    for row in world.ensembl.read_changed(assembly, release) | world.ensembl.refiled(assembly, release)
                )
                assert confirmed, (text, tier, gene, release)


def symbols_decided(world, view: GraphView, key: str, got: list[str], expected: list[str]) -> None:
    """Check rosetta's symbols of a gene at a release against d4c2dda's, as decision 104 on the side of the tables changes them.

    A symbol gained is accepted only where the tables give a version of the gene, at that release, the symbol under a
    database that Ensembl wrote otherwise and that continues a symbol database (:py:meth:`tests._real_tables.Ensembl.
    refiled`: pig HGNC projections of releases 56 to 60, filed as ``HGNC_curated_gene``); nothing may be lost.
    """
    base, release, synonyms = key.split(":")
    assert set(expected) <= set(got), (key, "lost", sorted(set(expected) - set(got)))
    assembly = world.backbone.get(int(release))
    genes = {unversioned(v) for v, mask in view.versions(base) if mask >> int(release) & 1}  # the tables' gene ids
    for symbol in sorted(set(got) - set(expected)):
        confirmed = assembly is not None and any(
            row[0] in genes
            and row[2].lower() == symbol
            and row[1].removeprefix(ids.SYNONYM_PREFIX) in view.symbol_databases
            and (synonyms == "True" or row[3] != "synonym")
            for row in world.ensembl.refiled(assembly, int(release))
        )
        assert confirmed, (key, symbol)


def _plain(value):
    return json.loads(json.dumps(value))


def _genes(record: dict) -> set[str]:
    """Every gene (without version) a rosetta record names as an answer: its targets, best and all, and paths."""
    found: set[str] = set()
    for key in ("target_id", "targets", "targets_all", "target_genes"):
        found.update(unversioned(str(t)) for t in record.get(key) or ())
    found.update(unversioned(str(path[0])) for path in record.get("paths") or ())
    return found


def release_of_no_answer_confirmed(world, got: dict, expected: dict, conversion: tuple) -> bool:
    """Tell whether a record differs only by the release a conversion without an answer reports, as the tables say.

    The API before this version reported no release (``None``) for a gene id that leads to no gene; this version
    reports the release the id was read at. Accepted only where nothing else of the record changed and the release is
    the one the rule gives by Ensembl's id tables (the oracle): the release given where the tables have the id there,
    else the target release where they have it there, else the latest release that has it.

    Args:
        world: The test world (graph and tables).
        got: The record now.
        expected: The expected record.
        conversion: ``(identifier, release given)`` of the record's conversion.

    Returns:
        bool: ``True`` when that is the only difference and the tables confirm the release.
    """
    if expected.get("from_release") is not None or not isinstance(got.get("from_release"), int):
        return False
    if {k: v for k, v in got.items() if k != "from_release"} != {
        k: v for k, v in expected.items() if k != "from_release"
    }:
        return False
    if _genes(got) or _genes(expected):
        return False
    identifier, given = conversion
    base = unversioned(identifier)
    held = [release for release in sorted(world.backbone) if base in world.genes_at(release)]
    if not held:
        return False
    if given is not None and given in held:
        rule = given
    else:
        rule = SNAPSHOT if SNAPSHOT in held else held[-1]
    return bool(got["from_release"] == rule)


def refusal_confirmed(world, check) -> bool:
    """Tell whether Ensembl's tables, read by the oracle, confirm every rule that refused a bridge.

    The oracle (:py:class:`tests._real_tables.Ensembl`) shares no code with the engine: ``synonym``, the tables list
    the id as a synonym of one of the two genes; ``places_apart``, the two genes lie on the reference sequence of a
    common assembly and no places of them overlap there; ``copy_id_elsewhere``, one of them lies on a ``non_ref`` region
    at some release and no release gives the id to both; ``copy_id_of_several_genes``, one of them lies on a
    ``non_ref`` region at some release and at a release that gives the id to both, the genes it is given to are more
    than one locus (:py:meth:`tests._real_tables.Ensembl.one_locus_count`). Any other rule (a rule that takes a bridge,
    or ``shared_at_one_release``, which a graph with places never gives) is not confirmed.

    Args:
        world: The test world (graph and tables).
        check: A :py:class:`idtrack.results.BridgeCheck` the engine refused.

    Returns:
        bool: ``True`` when the tables confirm the rule of every id between the two genes.
    """
    oracle = world.ensembl
    source, target = unversioned(check.source), unversioned(check.target)
    for middle, rule in check.through:
        text = re.sub(r"^(synonym_id::|synonym:)", "", middle)
        if rule == "synonym":
            confirmed = oracle.is_synonym(text, {source, target})
        elif rule == "places_apart":
            confirmed = oracle.apart(source, target)
        elif rule == "copy_id_elsewhere":
            held = oracle.holders(unversioned(text))
            confirmed = (oracle.outside_somewhere(source) or oracle.outside_somewhere(target)) and not (
                held.get(source, set()) & held.get(target, set())
            )
        elif rule == "copy_id_of_several_genes":
            held = oracle.holders(unversioned(text))
            confirmed = (oracle.outside_somewhere(source) or oracle.outside_somewhere(target)) and any(
                oracle.one_locus_count({gene for gene, releases in held.items() if release in releases}) > 1
                for release in held.get(source, set()) & held.get(target, set())
            )
        else:
            confirmed = False
        if not confirmed:
            return False
    return True


def changed_only_by_refused_bridges(world, got, expected, conversion_of) -> None:
    """Check that rosetta's answers are the expected ones, or differ only where the rule of bridges says why.

    The expected answers are those of the API before this version. A record that differs only by the release a
    conversion without an answer now reports is accepted where the tables confirm that release
    (:py:func:`release_of_no_answer_confirmed`). Any other record that differs is accepted only when the
    same conversion, explained, shows the cause and Ensembl's tables confirm it: every gene the record lost is the
    target of a bridge the rules refused, for reasons the tables confirm (:py:func:`refusal_confirmed`, not the
    engine's word for it), every gene it gained is reached through a bridge, and it lost or gained at least one gene
    (a changed order or class alone is no explanation).

    Args:
        world: The test world (graph and tables).
        got: The answers now (a list or a dict of records).
        expected: The expected answers, of the same shape.
        conversion_of: ``(index or key, expected record) -> (identifier, from_release)`` of the record's conversion.
    """
    if got == expected:
        return
    assert type(got) is type(expected) and len(got) == len(expected), "a record was added or removed"
    keys = range(len(got)) if isinstance(got, list) else sorted(expected)
    for key in keys:
        if got[key] == expected[key] or release_of_no_answer_confirmed(
            world, got[key], expected[key], conversion_of(key, expected[key])
        ):
            continue
        lost, gained = _genes(expected[key]) - _genes(got[key]), _genes(got[key]) - _genes(expected[key])
        assert lost or gained, (key, expected[key], got[key])
        identifier, from_release = conversion_of(key, expected[key])
        result = world.api.convert(identifier, from_release=from_release, to_release=SNAPSHOT, explain=True)
        taken = {unversioned(c.target) for c in result.path.bridges if c.taken}
        refused = {
            unversioned(c.target) for c in result.path.bridges if not c.taken and refusal_confirmed(world, c)
        } - taken
        bridged = {unversioned(c.gene) for c in result.candidates if c.bridges}
        assert lost <= refused and gained <= bridged, (key, lost, refused, gained, bridged)


def test_graph_is_the_one_the_expectations_were_made_on(real_world, organism):
    """The cut graph of the expectations: same content digest, releases, serving assemblies and databases."""
    expected = EXPECTED["organisms"][organism]["describe"]
    api = real_world(organism).api
    described = api.describe()
    assert described.content_digest == expected["content_digest"]
    assert list(described.releases) == expected["releases"]
    assert described.serving_assembly == expected["serving_assembly"]
    assert {str(a.assembly): list(a.releases) for a in described.assemblies} == expected["assemblies"]
    assert {str(a.assembly): list(a.external_databases) for a in described.assemblies} == {
        a: sorted({ids.public_name(d) for d in names}) for a, names in expected["external_databases"].items()
    }
    for a in described.assemblies:  # decision 56: the databases that record synonyms, by their own names
        assert set(a.synonym_databases) <= set(a.external_databases)
        assert not any(d.startswith(ids.SYNONYM_PREFIX) for d in a.external_databases)
    assert api.databases_by_assembly() == {str(a.assembly): list(a.external_databases) for a in described.assemblies}
    assert described.organism == organism and described.snapshot_release == SNAPSHOT
    record = api.graph.record  # rosetta's record of a graph unit copies these from the graph file's record
    for key in ("counts", "content_digest", "history", "source_tables", "tables", "build_date"):
        assert key in record.to_json()
    assert described.format_version == idtrack.graph.GRAPH_FORMAT_VERSION


def test_draw_queries_and_answer(real_world, organism):
    """`idtrack_graph.draw_queries` and `answer` give the same queries and answers."""
    expected = EXPECTED["organisms"][organism]
    api = real_world(organism).api
    assembly = int(api.describe().primary_assembly)
    queries = draw_queries(api, organism, SNAPSHOT, assembly, 60, 0, 5)
    assert _plain(queries) == expected["queries"]
    changed_only_by_refused_bridges(
        real_world(organism),
        _plain(answer(api, queries, SNAPSHOT)),
        decided(real_world(organism), expected["answer"]),
        lambda i, record: (queries[i]["id"], queries[i]["from_release"]),
    )


def test_convert(real_world, organism):
    """`idtrack_graph.convert`: gene ids read at their release and symbols, best and every target, classes."""
    expected = EXPECTED["organisms"][organism]
    api = real_world(organism).api
    genes: dict[int, list[str]] = {}
    for q in expected["queries"]:
        if q["kind"] == "gene":
            genes.setdefault(q["from_release"], []).append(q["id"])
    world = real_world(organism)
    for release, ids_ in sorted(genes.items()):
        changed_only_by_refused_bridges(
            world,
            _plain(convert(api, ids_, SNAPSHOT, from_release=release)),
            decided(world, expected["convert_genes"][str(release)]),
            lambda i, record, release=release: (record["query_id"], release),
        )
    symbols = [q["id"] for q in expected["queries"] if q["kind"] == "symbol"]
    changed_only_by_refused_bridges(
        world,
        _plain(convert(api, symbols, SNAPSHOT)),
        decided(world, expected["convert_symbols"]),
        lambda i, record: (record["query_id"], None),
    )


def test_graph_view(real_world, organism):
    """`genes.GraphView`: versions, names per tier and release (both letter-case rules), symbols, every path."""
    expected = EXPECTED["organisms"][organism]
    api = real_world(organism).api
    view = GraphView(api, list(organisms.SYMBOL_DATABASES.get(organism, ())), expected["name_databases"])
    assert view.releases == expected["describe"]["releases"]  # rosetta's map_features reads them
    for key, tiers in expected["name_tiers"].items():
        fold, text = key.split(":", 1)
        got = view.name_tiers(text, fold == "True")
        tiers_decided(real_world(organism), view, text, fold == "True", got, tiers)
    for base, versions in expected["versions"].items():
        assert _plain(view.versions(base)) == versions, base
    for key, symbols in expected["symbols"].items():
        base, release, synonyms = key.split(":")
        got = sorted(view.symbols(base, int(release), synonyms == "True"))
        symbols_decided(real_world(organism), view, key, got, symbols)
    bases = sorted(expected["versions"])
    older = min(int(k) for k in expected["convert_genes"])
    changed_only_by_refused_bridges(
        real_world(organism),
        _plain(view.convert(bases, older, SNAPSHOT)),
        expected["graph_view_convert"],
        lambda gene, record: (gene, older),
    )


def test_graph_file_name_manager_and_self_tests(real_world, organism, monkeypatch):
    """The graph unit's build and record: the file name, the externals file, the manager and the eight self-tests."""
    world = real_world(organism)
    first = organisms.first_release(organism)
    assembly = organisms.primary_assembly(organism, SNAPSHOT)
    name = files.graph_file_path(str(world.repository), organism, assembly, first, SNAPSHOT)
    assert os.path.isfile(name) and name == API.graph_path(organism, SNAPSHOT, str(world.repository))

    shipped = importlib.resources.files("idtrack") / "default_config" / f"{organism}_externals_modified.yml"
    content = load_externals_yaml(str(shipped), shipped=True)[organism]
    included = {
        form: sorted(db for db, entry in dbs.items() if any(a["Include"] for a in entry["Assembly"].values()))
        for form, dbs in content.items()
    }
    assert set(organisms.SYMBOL_DATABASES[organism]) & set(included["gene"])

    table = organisms.ORGANISM_TABLE[organism]
    lowest, highest = table[assembly]["ReleaseRange"]
    assert lowest <= SNAPSHOT and (highest is None or SNAPSHOT <= highest)

    monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
    world.tables.install(monkeypatch, world.repository)
    manager = API.database_manager(organism, SNAPSHOT, str(world.repository), assembly=assembly, first_release=first)
    assert GraphMaker(manager).create_file_name() == name
    checks = GraphChecks(world.api.graph, manager)
    for check in SELF_TESTS:
        method = getattr(checks, check)
        # as rosetta calls a check: with `progress=False` where the check has a progress bar
        result = method(**({"progress": False} if "progress" in inspect.signature(method).parameters else {}))
        assert result.holds is True, (check, {k: v for k, v in result.details.items() if k != "traceback"})
        json.dumps(result.details, allow_nan=False, default=str)


def test_access_settings(organism, monkeypatch):
    """`idtrack_graph.configure_access`: access mode, timeouts and retries, checked and set for the process."""
    for key in ("access", "connection_timeout", "reading_timeout", "rest_attempts", "rest_retry_wait"):
        monkeypatch.setattr(settings.ENSEMBL, key, getattr(settings.ENSEMBL, key))  # restored after the test
    settings.configure_ensembl(
        access="https", connection_timeout=20, reading_timeout=120, rest_attempts=5, rest_retry_wait=3.0
    )
    assert (settings.ENSEMBL.access, settings.ENSEMBL.reading_timeout, settings.ENSEMBL.rest_attempts) == (
        "https",
        120,
        5,
    )
    with pytest.raises(idtrack.InvalidArgumentError):
        settings.configure_ensembl(access="mysql-only")
