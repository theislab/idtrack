#!/usr/bin/env python3
"""Write the expected answers of the consumer-contract test: ``tests/data/real/consumer_contract.json``.

The test (``tests/unit/test_consumer_contract.py``) makes the calls of idtrack's main consumer, the rosetta project
(``src/rosetta/reference/idtrack_graph.py`` and ``src/rosetta/preprocessing/genes.py``), with the API of this version
on the cut test graphs, and compares the answers with those that the calls rosetta made before the API changed gave.
This script made those answers: it holds rosetta's functions as they were (``draw_queries``, ``answer``, ``convert``,
``bins``, ``GraphView``, read on 2026-11-17, with only rosetta's imports and its ``Graph`` record removed), written
for the API of idtrack before version 0.1.0, and runs them on the cut graphs of ``tests/_real_tables.py``.

Usage, from a checkout of commit ``d4c2dda`` (the last commit with that API)::

    git worktree add ../idtrack_d4c2dda d4c2dda
    cd ../idtrack_d4c2dda && PYTHONPATH=. python <this repository>/tests/data/real/build_consumer_contract.py out.json

The answers do not change unless the cut tables change; the file is then written again in the same way.
"""

import json
import random
import re
import sys
import tempfile

from tests._real_tables import SNAPSHOT, RealWorld

SYNONYM = "synonym_id::"
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
# rosetta's configs/preparation.toml, [gene_mapping] name_databases (2026-11-17)
ROSETTA_NAME_DATABASES = [
    "Clone_based_ensembl_gene",
    "Clone_based_vega_gene",
    "EntrezGene",
    "NCBI gene",
    "NCBI gene (formerly Entrezgene)",
    "UniProtKB Gene Name",
    "Vega gene",
    "Vega_gene",
]
NAME_DATABASES = {"sus_scrofa": ROSETTA_NAME_DATABASES, "homo_sapiens": ROSETTA_NAME_DATABASES}


def unversioned(target):
    """Return a gene id without its version (rosetta's helper)."""
    return VERSIONED.sub(r"\1", str(target))


def releases_of(mask):
    """Return the releases of a bit mask, ascending."""
    return [r for r in range(mask.bit_length()) if mask >> r & 1]


class GraphView:
    """rosetta's ``genes.GraphView`` as it was: versions, names per tier and release, symbols, paths."""

    def __init__(self, api, symbol_databases, name_databases):
        """Keep the API, its graph and the database tiers of names."""
        self.api = api
        self.G = api.track.graph
        self.store = self.G._frozen()
        self.releases = [int(r) for r in self.G.releases]
        self.respelled = set(self.store.respelled()[0])
        self.line = {int(a): rs.mask for a, rs in self.G.serving_line.items()}
        self.symbol_databases = set(symbol_databases)
        self.tier = {db: "symbol" for db in symbol_databases} | {db: "name" for db in name_databases}
        self.tier |= {SYNONYM + db: "synonym" for db in list(symbol_databases) + list(name_databases)}
        self._versions = {}

    def versions(self, base):
        """Return the gene versions of a base id with the mask of their reference releases."""
        if base not in self._versions:
            out = []
            if base in self.G.nodes:
                for v in self.G.versions_of(base):
                    if self.G.node_type(v) == "ensembl_gene":
                        out.append((v, self.G.active_releases(v).mask & ~self.G.non_reference_releases(v).mask))
            self._versions[base] = out
        return self._versions[base]

    def name_tiers(self, text, fold):
        """Return the genes that a text names, per tier, with the mask of the releases of each."""
        out = {t: {} for t in TIERS}
        st, graph = self.store, self.G
        for query in (text, SYNONYM + text):
            numbers = set(st.lower_candidates(query.lower()))
            exact = st.index.get(query)
            if exact is not None:
                numbers.add(exact)
            for number in numbers:
                name = st.names[number]
                if graph.node_type(name) != "external":
                    continue
                if not fold and name != query and number not in self.respelled:
                    continue
                for other, connection in graph.xref_edges(name):
                    if graph.node_type(other) != "ensembl_gene":
                        continue
                    base = graph.base_id(other)
                    nonref = graph.non_reference_releases(other).mask
                    for db, assembly, rs in connection.items:
                        tier = self.tier.get(db)
                        if tier is None:
                            continue
                        mask = rs.mask & self.line.get(int(assembly), 0) & ~nonref
                        if not fold and name != query:
                            mask &= sum(1 << r for r in releases_of(mask) if graph.spelling(name, db, r) == query)
                        if mask:
                            out[tier][base] = out[tier].get(base, 0) | mask
        return out

    def symbols(self, base, release, synonyms):
        """Return the symbols of a base id at a release (with synonyms or not), lower case."""
        out = set()
        for v, mask in self.versions(base):
            if not mask >> release & 1:
                continue
            for other, connection in self.G.xref_edges(v, reverse=True):
                for db, assembly, rs in connection.items:
                    plain = db.removeprefix(SYNONYM)
                    if (
                        plain in self.symbol_databases
                        and (synonyms or db == plain)
                        and rs.mask >> release & 1
                        and self.line.get(int(assembly), 0) >> release & 1
                    ):
                        out.add(other.removeprefix(SYNONYM).lower())
        return out

    def convert(self, genes, from_release, to_release):
        """Return the paths (target, bridges), the error and the starting release of each gene's conversion."""
        if not genes:
            return {}
        results = self.api.convert_identifier_multiple(
            genes,
            from_release=from_release,
            to_release=to_release,
            final_database="ensembl_gene",
            strategy="all",
            explain=True,
            verbose=False,
        )
        out = {}
        for g, r in zip(genes, results):
            out[g] = {
                "paths": [(unversioned(p["target_id"]), int(p["external_jump"])) for p in (r.get("the_path") or [])],
                "error": (r["error"] or {}).get("type", "") if r.get("error") else "",
                "from_release": r.get("from_release"),
            }
        return out


def draw_queries(graph, organism, snapshot, genome_assembly, n, seed, old_releases):
    """Draw gene ids of earlier releases and symbols of the snapshot (rosetta's ``draw_queries``)."""
    from idtrack import DB

    earlier = [r for r in graph.releases if r < snapshot]
    picked = sorted({earlier[round(i * (len(earlier) - 1) / max(old_releases - 1, 1))] for i in range(old_releases)})
    serving = graph.backbone_assembly
    wanted = {(f"assembly_{serving[r]}_ensembl_gene", serving[r], r): ("gene", r) for r in picked}
    symbol_dbs = [d for d in DB.symbol_databases[organism] if d in graph.available_external_databases]
    wanted |= {(d, genome_assembly, snapshot): ("symbol", snapshot) for d in symbol_dbs}
    found = {v: set() for v in wanted.values()}
    for name in graph.nodes:
        if graph.is_backbone(name) and graph.version(name) in DB.alternative_versions:
            continue
        try:
            trios = graph.trios(name)
        except KeyError:
            continue
        for trio in trios:
            if trio in wanted:
                kind, release = wanted[trio]
                found[(kind, release)].add(graph.base_id(name) if kind == "gene" else name)
    rng = random.Random(seed)  # noqa: S311 - a reproducible draw, not a secret
    pool = sorted(found[("symbol", snapshot)])
    symbols = [{"id": i, "kind": "symbol", "from_release": None} for i in rng.sample(pool, min(n // 2, len(pool)))]
    genes = n - len(symbols)
    per_release = [genes // len(picked) + (1 if i < genes % len(picked) else 0) for i in range(len(picked))]
    queries = []
    for release, k in zip(picked, per_release):
        pool = sorted(found[("gene", release)])
        queries += [{"id": i, "kind": "gene", "from_release": release} for i in rng.sample(pool, min(k, len(pool)))]
    queries += symbols
    return queries


def answer(api, queries, to_release):
    """Convert each query and keep the fields rosetta reads (rosetta's ``answer``)."""
    out = []
    for q in queries:
        try:
            r = api.convert_identifier(q["id"], from_release=q["from_release"], to_release=to_release)
            out.append(
                {
                    k: r[k]
                    for k in (
                        "graph_id",
                        "match",
                        "from_release",
                        "target_id",
                        "tie",
                        "no_corresponding",
                        "no_conversion",
                        "no_target",
                    )
                }
            )
        except Exception as error:  # noqa: BLE001
            out.append({"error": f"{type(error).__name__}: {error}"})
    return out


def bins(api, results):
    """Return the class of each result and whether it changed (rosetta's ``bins``)."""
    classified = api.classify_multiple_conversion(results)
    of = {}
    for name in CLASSES:
        for item in classified[name]:
            of[id(item)] = name
    changed = {id(item) for name in CHANGED.values() for item in classified[name]}
    return [of[id(r)] for r in results], [id(r) in changed for r in results]


def convert(api, ids, to_release, from_release=None, from_database=None):
    """Convert a batch twice, best and all, into rosetta's rows (rosetta's ``convert``)."""
    distinct = list(dict.fromkeys(ids))
    common = {
        "from_release": from_release,
        "to_release": to_release,
        "final_database": "ensembl_gene",
        "from_database": from_database,
        "verbose": False,
    }
    best = api.convert_identifier_multiple(distinct, strategy="best", **common)
    every = api.convert_identifier_multiple(distinct, strategy="all", **common)
    best_class, best_changed = bins(api, best)
    all_class, _ = bins(api, every)
    rows = []
    for b, a, cls, changed, cls_all in zip(best, every, best_class, best_changed, all_class):
        rows.append(
            {
                "query_id": b["query_id"],
                "from_release": b["from_release"],
                "from_release_inferred": b["from_release_inferred"],
                "from_release_fallback": b["from_release_fallback"],
                "graph_id": b["graph_id"],
                "match": b["match"],
                "from_databases": b["from_databases"],
                "targets": b["target_id"],
                "target_genes": [g for g, _ in b["last_node"]],
                "tie": b["tie"],
                "class": cls,
                "changed": changed,
                "targets_all": a["target_id"],
                "class_all": cls_all,
                "no_corresponding": b["no_corresponding"],
                "no_conversion": b["no_conversion"],
                "no_target": b["no_target"],
                "no_target_genes": b["no_target_genes"],
                "alternative_locus_genes": b["alternative_locus_genes"],
                "known_versions": b["known_versions"],
                "error": (b["error"] or {}).get("type", "") + (": " + b["error"]["message"] if b["error"] else ""),
            }
        )
    return rows


def main(out_path):
    """Write the expected answers of both organisms to *out_path*."""
    from idtrack import DB

    result = {"snapshot": SNAPSHOT, "organisms": {}}
    for organism in ("sus_scrofa", "homo_sapiens"):
        world = RealWorld(organism, tempfile.mkdtemp(prefix=f"contract_{organism}_"))
        api = world.api
        graph = api.track.graph
        genome_assembly = int(graph.graph["genome_assembly"])
        queries = draw_queries(graph, organism, SNAPSHOT, genome_assembly, 60, 0, 5)
        genes = [q for q in queries if q["kind"] == "gene"]
        symbols = [q["id"] for q in queries if q["kind"] == "symbol"]
        by_release = {}
        for q in genes:
            by_release.setdefault(q["from_release"], []).append(q["id"])
        view = GraphView(api, list(DB.symbol_databases.get(organism, ())), NAME_DATABASES[organism])
        names = sorted(set(symbols) | {s.lower() for s in symbols[:10]} | {s.upper() for s in symbols[:10]})
        bases = sorted({q["id"] for q in genes})
        older = sorted(by_release)[0]
        result["organisms"][organism] = {
            "queries": queries,
            "answer": answer(api, queries, SNAPSHOT),
            "convert_genes": {
                str(r): convert(api, ids, SNAPSHOT, from_release=r) for r, ids in sorted(by_release.items())
            },
            "convert_symbols": convert(api, symbols, SNAPSHOT),
            "name_databases": NAME_DATABASES[organism],
            "name_tiers": {
                f"{fold}:{text}": {
                    t: {g: m for g, m in sorted(v.items())} for t, v in view.name_tiers(text, fold).items()
                }
                for text in names
                for fold in (False, True)
            },
            "versions": {base: view.versions(base) for base in bases},
            "symbols": {
                f"{base}:{release}:{syn}": sorted(view.symbols(base, release, syn))
                for base in bases
                for release in (older, SNAPSHOT)
                for syn in (False, True)
            },
            "graph_view_convert": view.convert(bases, older, SNAPSHOT),
            "describe": {
                "releases": [int(r) for r in graph.releases],
                "serving_assembly": {str(r): int(a) for r, a in sorted(graph.backbone_assembly.items())},
                "assemblies": {str(a): [int(r) for r in rel] for a, rel in sorted(graph.assembly_releases.items())},
                "external_databases": {
                    str(a): sorted(d) for a, d in sorted(graph.available_external_databases_assembly.items())
                },
                "content_digest": graph.manifest.get("content_digest"),
            },
        }
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
