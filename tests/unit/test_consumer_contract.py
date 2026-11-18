"""The consumer contract: every call that rosetta (idtrack's one dependent) makes, through the public API only.

rosetta reads idtrack in two modules: its graph units (build, record, self-tests, a check against another build,
and a reader that converts gene ids) and its gene mapping (names per database and release, a gene's symbols at a
release, every path from genes of one release to another). This module makes each of those calls on the graphs
built from the cut tables of ``tests/data/real`` and compares the answers with
``tests/data/real/consumer_contract.json``: the answers of the same calls made with the API they replace (the
release before this layout), so that a change of what rosetta gets is a failing test. The functions below are
written as rosetta writes them, against the public names; nothing here reads a private member.
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
    wanted = {
        (organisms.ASSEMBLY_NODE_TYPES[serving[r]][ids.BACKBONE_FORM], serving[r], r): ("gene", r) for r in picked
    }
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
        # The graph names a database of synonyms with its prefix (as NameMatch reports it).
        self.tier |= {ids.SYNONYM_PREFIX + db: "synonym" for db in list(symbol_databases) + list(name_databases)}
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
            tier = self.tier.get(match.database)
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
                plain = match.database.removeprefix(ids.SYNONYM_PREFIX)
                if plain in self.symbol_databases and (synonyms or not match.synonym):
                    out.add(match.name.removeprefix(ids.SYNONYM_PREFIX).lower())
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


def _plain(value):
    return json.loads(json.dumps(value))


def test_graph_is_the_one_the_expectations_were_made_on(real_world, organism):
    """The cut graph of the expectations: same content digest, releases, serving assemblies and databases."""
    expected = EXPECTED["organisms"][organism]["describe"]
    api = real_world(organism).api
    described = api.describe()
    assert described.content_digest == expected["content_digest"]
    assert list(described.releases) == expected["releases"]
    assert described.serving_assembly == expected["serving_assembly"]
    assert {str(a.assembly): list(a.releases) for a in described.assemblies} == expected["assemblies"]
    assert {str(a.assembly): list(a.external_databases) for a in described.assemblies} == expected["external_databases"]
    for a in described.assemblies:  # the databases of synonyms are those with the graph's prefix
        assert list(a.synonym_databases) == [d for d in a.external_databases if d.startswith(ids.SYNONYM_PREFIX)]
    assert api.databases_by_assembly() == expected["external_databases"]
    assert described.organism == organism and described.snapshot_release == SNAPSHOT
    for key in ("counts", "content_digest", "format_version", "history", "source_tables", "tables"):
        assert key in api.graph.manifest  # rosetta's record copies these from the graph file's manifest


def test_draw_queries_and_answer(real_world, organism):
    """`idtrack_graph.draw_queries` and `answer` give the same queries and answers."""
    expected = EXPECTED["organisms"][organism]
    api = real_world(organism).api
    assembly = int(api.describe().primary_assembly)
    queries = draw_queries(api, organism, SNAPSHOT, assembly, 60, 0, 5)
    assert _plain(queries) == expected["queries"]
    assert _plain(answer(api, queries, SNAPSHOT)) == expected["answer"]


def test_convert(real_world, organism):
    """`idtrack_graph.convert`: gene ids read at their release and symbols, best and every target, classes."""
    expected = EXPECTED["organisms"][organism]
    api = real_world(organism).api
    genes: dict[int, list[str]] = {}
    for q in expected["queries"]:
        if q["kind"] == "gene":
            genes.setdefault(q["from_release"], []).append(q["id"])
    for release, ids_ in sorted(genes.items()):
        assert _plain(convert(api, ids_, SNAPSHOT, from_release=release)) == expected["convert_genes"][str(release)]
    symbols = [q["id"] for q in expected["queries"] if q["kind"] == "symbol"]
    assert _plain(convert(api, symbols, SNAPSHOT)) == expected["convert_symbols"]


def test_graph_view(real_world, organism):
    """`genes.GraphView`: versions, names per tier and release (both letter-case rules), symbols, every path."""
    expected = EXPECTED["organisms"][organism]
    api = real_world(organism).api
    view = GraphView(api, list(organisms.SYMBOL_DATABASES.get(organism, ())), expected["name_databases"])
    assert view.releases == expected["describe"]["releases"]  # rosetta's map_features reads them
    for key, tiers in expected["name_tiers"].items():
        fold, text = key.split(":", 1)
        got = view.name_tiers(text, fold == "True")
        assert {t: {g: m for g, m in sorted(v.items())} for t, v in got.items()} == tiers, key
    for base, versions in expected["versions"].items():
        assert _plain(view.versions(base)) == versions, base
    for key, symbols in expected["symbols"].items():
        base, release, synonyms = key.split(":")
        assert sorted(view.symbols(base, int(release), synonyms == "True")) == symbols, key
    bases = sorted(expected["versions"])
    older = min(int(k) for k in expected["convert_genes"])
    assert _plain(view.convert(bases, older, SNAPSHOT)) == expected["graph_view_convert"]


def test_graph_file_name_manager_and_self_tests(real_world, organism, monkeypatch):
    """The graph unit's build and record: the file name, the externals file, the manager and the eight self-tests."""
    world = real_world(organism)
    first = organisms.first_release(organism)
    assembly = organisms.primary_assembly(organism, SNAPSHOT)
    name = files.graph_file_path(
        str(world.repository), organism, assembly, first, SNAPSHOT, narrow=True, narrow_external=True
    )
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
    assert GraphMaker(manager).create_file_name(narrow=True, narrow_external=True) == name
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
