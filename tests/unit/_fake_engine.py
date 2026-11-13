"""A small stand-in for :py:class:`idtrack.Track` and its graph, for tests of the API and the harmonizer.

The fake answers :py:meth:`idtrack.Track.convert` from a table, in the engine's result shape and with the engine's
errors, so that the real :py:class:`idtrack.API` and :py:class:`idtrack.HarmonizeFeatures` code runs on it without a
graph or a network.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from idtrack import API, Track

GENE_DATABASES = ("ensembl_gene", "base_ensembl_gene")


class FakeGraph:
    """The parts of :py:class:`idtrack.TheGraph` the API reads."""

    def __init__(self, nodes: list[str], releases: list[int], databases: set[str], organism: str = "homo_sapiens"):
        """Hold *nodes* over *releases*, with *databases* on the primary assembly 38."""
        self.nodes = set(nodes)
        self.releases = tuple(releases)
        self.graph = {
            "organism": organism,
            "ensembl_release": max(releases),
            "confident_for_release": list(releases),
            "genome_assembly": 38,
            "misplaced_external_entry": set(),
        }
        self.available_external_databases_assembly = {38: set(databases), 37: {"Old DB"}}
        self.available_external_databases = set(databases) | {"Old DB"}
        self.available_genome_assemblies = {37, 38}
        self.external_database_connection_form = {d: "gene" for d in sorted(databases)}
        self.lookups: Counter = Counter()

    @property
    def last_release(self) -> int:
        """The last release of the fake."""
        return max(self.releases)

    def resolve_name(
        self, identifier: str, release: int | None = None, database: str | None = None, target: int | None = None
    ) -> tuple[str | None, str | None]:
        """Resolve exactly, then case-insensitively (the first match in sorted order), whatever the release."""
        self.lookups[identifier] += 1
        if identifier in self.nodes:
            return identifier, "exact"
        folded = [n for n in sorted(self.nodes) if n.casefold() == identifier.casefold()]
        return (folded[0], "case") if folded else (None, None)

    def known_versions(self, identifier: str) -> tuple[str, ...]:
        """Return the versions of an id written with a version that the fake does not hold."""
        stable_id, _, version = identifier.rpartition(".")
        held = self.versions_of(stable_id) if version.isdigit() and identifier not in self.nodes else ()
        return held

    def versions_of(self, stable_id: str) -> tuple[str, ...]:
        """Return the versioned nodes of an id without version (the fake has no such ids besides its nodes)."""
        versions = [n for n in self.nodes if n.rpartition(".")[0] == stable_id and n.rpartition(".")[2].isdigit()]
        return tuple(sorted(versions, key=lambda n: int(n.rpartition(".")[2])))

    def trios(self, node: str) -> frozenset:
        """Return the ``(database, assembly, release)`` triples of a node: database ``db`` at every release."""
        return frozenset(("db", 38, r) for r in self.releases)

    def calculate_caches(self, for_test: bool = False) -> None:
        """Do nothing (the fake has no graph-wide tables to compute)."""
        pass


class FakeTrack(Track):
    """Answers ``convert`` from ``answers``: ``{node: {gene: {database: [targets]}}}``.

    It is a :py:class:`idtrack.Track` by type only (so that it can stand where the API expects one); every method the
    API calls is replaced, and the constructor of ``Track`` is not run.

    As the engine does: the genes are returned in the given order (the engine's ranking, best first); with a final
    database only the genes that have an id in it are returned, and if none has one all genes are returned as
    themselves with ``no_target``; with ``prioritize_to_one_filter`` every pair is kept and the answer is a tie
    when there are several (the fake has no evidence to choose by). ``errors`` maps a node (or
    ``(node, from_release)``) to an exception raised by ``convert``.
    """

    def __init__(
        self,
        answers: dict[str, dict[str, dict[str, list[str]]]],
        releases: tuple[int, ...] | list[int] = (100, 101),
        databases: set[str] | frozenset[str] = frozenset({"HGNC Symbol"}),
        errors: dict[Any, Exception] | None = None,
        organism: str = "homo_sapiens",
    ):
        """Keep the table and the errors; record every call in ``calls``."""
        self.answers = answers
        self.errors = errors or {}
        self.calls: list[dict[str, Any]] = []
        self.name_lookups: list[tuple[str, str, int | None]] = []
        nodes = set(answers)
        self.db_manager = None
        self.graph = FakeGraph(sorted(nodes), list(releases), set(databases), organism=organism)  # type: ignore[assignment]

    def final_databases(self, to_release: int | None = None) -> list[str]:
        """Return the accepted final databases: those of assembly 38 and the two gene namespaces."""
        return sorted(self.graph.available_external_databases_assembly[38] | set(GENE_DATABASES))

    def gene_to_database(self, gene: str, final_database: str, release: int | None = None) -> tuple[list[str], int]:
        """Return the ids of a gene in a database: what the table says under the gene itself."""
        self.name_lookups.append((gene, final_database, release))
        return sorted(self.answers.get(gene, {}).get(gene, {}).get(final_database, [])), 0

    def convert(
        self,
        from_id: str,
        from_release: int | None = None,
        to_release: int | None = None,
        final_database: str | None = None,
        prioritize_to_one_filter: bool = False,
        return_path: bool = False,
        from_database: str | None = None,
    ) -> dict[str, Any] | None:
        """Answer like :py:meth:`idtrack.Track.convert` from the table (``None`` when there is no gene).

        As the engine does, the string is read as the node the graph says it means at ``from_release``; ``calls``
        records the string as given.
        """
        given = from_id
        resolved, _ = self.graph.resolve_name(from_id, from_release, from_database)
        from_id = given if resolved is None else resolved
        self.calls.append(
            {
                "from_id": given,
                "from_release": from_release,
                "to_release": to_release,
                "final_database": final_database,
                "from_database": from_database,
            }
        )
        for key in ((from_id, from_release), from_id):
            if key in self.errors:
                raise self.errors[key]
        genes = self.answers.get(from_id, {})
        if not genes:
            return None
        as_gene = final_database is None or final_database == "ensembl_gene"
        with_target: dict[str, list[str]] = (
            {}
            if final_database is None or as_gene
            else {g: by_db.get(final_database, []) for g, by_db in genes.items()}
        )
        no_target = not as_gene and not any(with_target.values())
        out: dict[str, dict[str, Any]] = {}
        for gene in genes:
            targets = [gene] if as_gene or no_target else with_target[gene]
            if not targets:
                continue
            out[gene] = {
                "from_id": from_id,
                "from_release": from_release if from_release is not None else to_release,
                "from_release_inferred": from_release is None,
                "from_databases": [] if from_database is None else [from_database],
                "entry_gene": gene,
                "entry_support": 1,
                "to_release": to_release,
                "external_jump": 0,
                "external_gap": 0,
                "external_step": 0,
                "ensembl_step": 0,
                "score": 1.0,
                "tie": False,
                "alternative_locus_genes": [],
                **({"the_path": ((from_id, gene, 0, to_release),)} if return_path else {}),
                "final_conversion": {
                    "final_conversion_confidence": None if no_target else 0,
                    "final_database": "ensembl_gene" if as_gene or no_target else final_database,
                    "no_target": no_target,
                    "genes_without_target": (
                        [] if as_gene or no_target else sorted(g for g, t in with_target.items() if not t)
                    ),
                    "final_elements": {
                        t: ({"the_path": ((gene, t, 0, to_release),)} if return_path and t != gene else {})
                        for t in targets
                    },
                },
            }
        pairs = sum(len(value["final_conversion"]["final_elements"]) for value in out.values())
        tie = prioritize_to_one_filter and pairs > 1
        for value in out.values():
            value["tie"] = tie
        return out

    def identify_source(self, dataset_ids: list[str], mode: str) -> list[tuple[Any, int]]:
        """Count the origins of the ids like :py:meth:`idtrack.Track.identify_source`."""
        counts: Counter = Counter()
        for i in dataset_ids:
            for db, asm, rel in self.graph.trios(i):
                counts[{"complete": (db, asm, rel), "ensembl_release": rel, "assembly": asm}.get(mode, (asm, rel))] += 1
        return list(counts.most_common())


def fake_api(tmp_path, answers, **kwargs) -> API:
    """Return an :py:class:`idtrack.API` whose tracker is a :py:class:`FakeTrack` on *answers*."""
    api = API(str(tmp_path))
    api.track = FakeTrack(answers, **kwargs)  # type: ignore[assignment]
    return api
