"""A small stand-in for the engine (:py:class:`idtrack.engine.Track`) and its graph.

It serves the tests of the API and the harmonizer.

The fake answers :py:meth:`idtrack.engine.Track.convert` from a table, as the engine's
:py:class:`idtrack.engine.Conversion` and with the engine's errors, so that the real :py:class:`idtrack.API` and
:py:class:`idtrack.harmonize.HarmonizeFeatures` code runs on it without a graph or a network.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from idtrack import API
from idtrack.engine import Conversion, Track
from idtrack.results import Candidate, ExplainedPath, PathStep
from idtrack.settings import SearchSettings


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
        self.available_assemblies = {37, 38}
        self.external_database_connection_form = {d: "gene" for d in sorted(databases)}
        self.lookups: Counter = Counter()

    @property
    def snapshot_release(self) -> int:
        """The last release of the fake."""
        return max(self.releases)

    @property
    def serving_assembly(self) -> dict[int, int]:
        """Every release of the fake is served by assembly 38."""
        return {r: 38 for r in self.releases}

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


class FakeTrack(Track):
    """Answers ``convert`` from ``answers``: ``{node: {gene: {database: [targets]}}}``.

    It is an :py:class:`idtrack.engine.Track` (so that it can stand where the API expects one) whose search is
    replaced: the engine's argument checks run as they are (on the fake graph), and the conversion of checked
    arguments (``_convert``) and the counting of origins (``_identify_source``) answer from the table. The
    constructor of ``Track`` is not run.

    As the engine does: the genes are returned in the given order (the engine's ranking, best first); with a final
    database only the genes that have an id in it are returned, and if none has one all genes are returned as
    themselves with ``no_target``; every pair is best and the answer is a tie when there are several (the fake has
    no evidence to choose by). ``errors`` maps a node (or ``(node, from_release)``) to an exception raised by
    ``convert``.
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
        self.search = SearchSettings()
        self.graph = FakeGraph(sorted(nodes), list(releases), set(databases), organism=organism)  # type: ignore[assignment]

    def gene_to_database(self, gene: str, final_database: str, release: int | None = None) -> tuple[list[str], int]:
        """Return the ids of a gene in a database: what the table says under the gene itself."""
        self.name_lookups.append((gene, final_database, release))
        return sorted(self.answers.get(gene, {}).get(gene, {}).get(final_database, [])), 0

    def _convert(
        self,
        identifier: str,
        *,
        from_release: int | None,
        to_release: int,
        final_database: str,
        from_database: str | None,
        allow_bridges: bool,
        return_ensembl_alternative: bool,
        explain: bool,
        search: SearchSettings,
    ) -> Conversion:
        """Answer like :py:meth:`idtrack.engine.Track.convert` from the table (arguments checked by the engine).

        As the engine does, the string is read as the node the graph says it means at ``from_release``; ``calls``
        records the string as given.
        """
        given = identifier
        resolved, _ = self.graph.resolve_name(identifier, from_release, from_database)
        node = given if resolved is None else resolved
        self.calls.append(
            {
                "identifier": given,
                "from_release": from_release,
                "to_release": to_release,
                "final_database": final_database,
                "from_database": from_database,
                "search": search,
            }
        )
        for key in ((node, from_release), node):
            if key in self.errors:
                raise self.errors[key]
        genes = self.answers.get(node, {})
        nothing = Conversion(
            graph_id=node,
            from_release=None,
            from_release_inferred=from_release is None,
            from_databases=(),
            to_release=to_release,
            final_database=None,
            candidates=(),
            tie=False,
            decided_by=None,
            alternative_locus_genes=(),
            no_target=False,
            no_target_genes=(),
            reason="no_path_to_target",
            path=ExplainedPath(nodes=(), steps=()) if explain else None,
        )
        if not genes:
            return nothing
        as_gene = final_database is None or final_database == "ensembl_gene"
        with_target: dict[str, list[str]] = (
            {}
            if final_database is None or as_gene
            else {g: by_db.get(final_database, []) for g, by_db in genes.items()}
        )
        no_target = not as_gene and not any(with_target.values())
        steps: list[PathStep] = []

        def step(kind: str, source: str, target: str) -> int:
            steps.append(PathStep(kind=kind, source=source, target=target, release=to_release))
            return len(steps) - 1

        candidates: list[Candidate] = []
        for gene in genes:
            targets = [gene] if as_gene or no_target else with_target[gene]
            for t in targets:
                path = (step("entry", node, gene),) if explain else ()
                final_path = (step("final", gene, t),) if explain and t != gene else ()
                candidates.append(
                    Candidate(
                        gene=gene,
                        target=t,
                        best=True,
                        entry_gene=gene,
                        entry_support=1,
                        own_version=False,
                        final_confidence=None if no_target else 0,
                        bridges=0,
                        bridge_gap=0,
                        score=1.0,
                        same_as_query=t == node,
                        external_steps=0,
                        history_steps=0,
                        annotation=None,
                        path=path,
                        final_path=final_path,
                    )
                )
        return Conversion(
            graph_id=node,
            from_release=from_release if from_release is not None else to_release,
            from_release_inferred=from_release is None,
            from_databases=() if from_database is None else (from_database,),
            to_release=to_release,
            final_database="ensembl_gene" if as_gene or no_target else final_database,
            candidates=tuple(candidates),
            tie=len(candidates) > 1,
            decided_by=None,
            alternative_locus_genes=(),
            no_target=no_target,
            no_target_genes=() if as_gene or no_target else tuple(sorted(g for g, t in with_target.items() if not t)),
            reason=None,
            path=ExplainedPath(nodes=(), steps=tuple(steps)) if explain else None,
        )

    def _identify_source(self, nodes: list[str], mode: str) -> list[tuple[Any, int]]:
        """Count the origins of the nodes like :py:meth:`idtrack.engine.Track.identify_source`."""
        counts: Counter = Counter()
        for i in nodes:
            for db, asm, rel in self.graph.trios(i):
                counts[{"complete": (db, asm, rel), "release": rel, "assembly": asm}.get(mode, (asm, rel))] += 1
        return list(counts.most_common())


def fake_api(tmp_path, answers, **kwargs) -> API:
    """Return an :py:class:`idtrack.API` whose graph and engine are a :py:class:`FakeGraph` and a :py:class:`FakeTrack`.

    The API's constructor accepts graphs only (:py:class:`idtrack.graph.TheGraph`), so the fake is put in place of
    what the constructor sets; every method of the API then runs as it does on a real graph.
    """
    engine = FakeTrack(answers, **kwargs)
    api = API.__new__(API)
    api.log = logging.getLogger("idtrack.api")
    api.graph = engine.graph  # type: ignore[assignment]
    api.engine = engine
    return api
