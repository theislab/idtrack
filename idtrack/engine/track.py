"""The conversion engine: the search over a graph and the ranking of what it finds, as functions of the graph.

:py:class:`Track` answers one question: given an identifier (an Ensembl gene id with or without version, a
transcript or translation id, or an id of an external database such as a gene symbol), optionally the release it
comes from, a target release and optionally a target database, which genes correspond to it at the target release,
and which ids of the target database those genes carry. It reads the graph only, needs no network and no table, and
keeps no state between calls.

The arguments of a conversion are checked here, once, by one rule each (:py:meth:`Track.convert`): a release must
be a release of the graph, a database one of the graph at the target release (matched without letter case and
surrounding white space). :py:class:`idtrack.API` uses the same checks; it also reads a user's identifier (surrounding
white space removed, an empty string refused), where the engine reads the string as it is given.
"""

from __future__ import annotations

import bisect
import heapq
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, SupportsIndex

from idtrack import _arguments, ids, organisms
from idtrack.engine import ranking
from idtrack.engine.explain import PathBuilder
from idtrack.errors import (
    InactiveAtReleaseError,
    InvalidArgumentError,
    SearchBudgetExceededError,
    UnknownDatabaseError,
    UnknownIdentifierError,
    UnknownOrganismError,
    UnknownReleaseError,
    UnknownVersionError,
)
from idtrack.graph import genes
from idtrack.graph.types import ReleaseSet, served_mask
from idtrack.results import BRIDGE_RULES_TAKEN, BridgeCheck, Candidate, ExplainedPath
from idtrack.settings import SearchSettings, check_search

if TYPE_CHECKING:
    from idtrack.graph.graph import TheGraph


class Written(NamedTuple):
    """How an external identifier was written, where that restricts the links that carry it.

    Where the query's spelling, or a version written after an accession, limits which links of its node carry it, a
    conversion follows only those links from the node it read the query as (:py:meth:`idtrack.TheGraph.resolve_name`);
    this records what limits them. It lives within one conversion: no result holds it.

    Attributes:
        text: The spelling as typed, where the string matched exactly a node that some link spells otherwise (only the
            links spelt so carry it; for a synonym, with the synonym prefix); ``None`` otherwise.
        version: The version of an accession read with one (only the links that attach that version carry it, at the
            releases at which they do: :py:meth:`idtrack.graph.TheGraph.reading_version`); ``None`` otherwise.
    """

    text: str | None = None
    version: int | None = None


#: The modes of :py:meth:`Track.identify_source`: what an origin of an identifier is.
SOURCE_MODES: tuple[str, ...] = ("complete", "release", "assembly", "assembly_release")

# A ``(database, assembly, release)`` triple of a node as the origin of each mode.
_ORIGIN_OF = {
    "complete": lambda trio: trio,
    "release": lambda trio: trio[2],
    "assembly": lambda trio: trio[1],
    "assembly_release": lambda trio: trio[1:],
}


class _GraphReader:
    """The engine's view of the graph: its accessors in the engine's order, and the primary line of releases.

    The engine never touches the attribute layout of nodes and edges. It reads the graph through the accessor
    methods of :py:class:`idtrack.TheGraph` (``node_type``, ``base_id``, ``version``, ``history_edges``,
    ``xref_edges``, ``active_ranges``, ``trios``, ``resolve_name``, ``versions_of``, ``spelling``,
    ``hyperconnective_nodes``, ``assembly_releases``, ``serving_assembly``); this class adds the order in which the
    search visits edges and the question whether an edge holds at a release on the primary line. Nothing is computed
    for the whole graph.
    """

    def __init__(self, graph: TheGraph):
        """Bind the reader to a graph and read the facts that hold for the whole graph.

        Args:
            graph (TheGraph): The graph that is queried.

        Raises:
            UnknownOrganismError: If the graph's organism is not registered in the organism table, so that its
                assemblies cannot be ordered.
        """
        self.graph = graph
        self.log = logging.getLogger(__name__)
        self.backbone_type: str = ids.GENE
        self.base_type: str = ids.BASE_ENSEMBL_TYPE[ids.BACKBONE_FORM]

        self.releases: tuple[int, ...] = graph.releases
        self.release_set = frozenset(self.releases)
        self.first_release = self.releases[0]
        self.snapshot_release = self.releases[-1]

        organism = graph.organism
        # Assembly keys carry no order of their own (rhesus Mmul_8.0.1 is 801 but older than Mmul_10, 10): the order
        # comes from the organism table only.
        if not isinstance(organism, str) or organism not in organisms.ORGANISM_TABLE:
            raise UnknownOrganismError(
                f"The graph's organism {organism!r} is not registered in the organism table "
                f"(supported: {organisms.supported_organisms()}), so its assemblies cannot be ordered. Register it "
                "with `idtrack.organisms.register_organism` before loading the graph.",
                argument="organism",
                value=organism,
            )
        self._organism_table = organisms.ORGANISM_TABLE[organism]
        self._assembly_releases: dict[int, frozenset] | None = None
        self._serving_line: dict[int, ReleaseSet] | None = None
        self._serving_assembly: dict[int, int] | None = None
        self._priority: dict[int, int] | None = None
        self._final_databases: dict[int, tuple[str, ...]] = {}
        self._external_databases: tuple[str, ...] | None = None

    # Nodes

    def has_node(self, n: Any) -> bool:
        """Tell whether ``n`` is a node name of the graph.

        Args:
            n (Any): Candidate node name.

        Returns:
            bool: ``True`` for an exact node name.
        """
        try:
            return n in self.graph.nodes
        except TypeError:
            return False

    def node_type(self, n: str) -> str:
        """Return the node type of ``n``.

        Args:
            n (str): Node name.

        Returns:
            str: For example ``ensembl_gene``, ``base_ensembl_gene``, ``ensembl_transcript`` or ``external``.
        """
        return self.graph.node_type(n)

    def base_id(self, n: str) -> str:
        """Return the stable id of an Ensembl node without its version.

        Args:
            n (str): Name of an Ensembl node.

        Returns:
            str: The stable id.
        """
        return self.graph.base_id(n)

    def version(self, n: str) -> str:
        """Return the version of an Ensembl node.

        Args:
            n (str): Name of an Ensembl node.

        Returns:
            str: The version, or ``Void`` / ``Retired`` for the two artificial nodes of a gene id.
        """
        return str(self.graph.version(n))

    def is_backbone(self, n: str) -> bool:
        """Tell whether ``n`` is a gene node of the history backbone.

        Args:
            n (str): Node name.

        Returns:
            bool: ``True`` for ``id.version`` gene nodes and for their ``Void`` and ``Retired`` nodes.
        """
        return self.graph.node_type(n) == self.backbone_type

    def is_real_gene(self, n: str) -> bool:
        """Tell whether ``n`` is a gene id with a version (not ``Void`` or ``Retired``).

        Args:
            n (str): Name of a backbone node.

        Returns:
            bool: ``True`` for a gene that exists in some release.
        """
        return self.version(n) not in ids.SENTINEL_VERSIONS

    def is_hyperconnective(self, n: str) -> bool:
        """Tell whether the external id ``n`` is shared by so many genes at once that it identifies none of them.

        Args:
            n (str): Name of an external node.

        Returns:
            bool: ``True`` when the graph counts the node as hyper-connective
            (:py:attr:`idtrack.TheGraph.hyperconnective_nodes`).
        """
        return n in self.graph.hyperconnective_nodes

    def resolve_name(
        self,
        identifier: str,
        from_release: int | None = None,
        from_database: str | None = None,
        to_release: int | None = None,
    ) -> tuple[str | None, str | None]:
        """Find the node an identifier written by a user stands for at a release.

        Args:
            identifier (str): The identifier as given.
            from_release (int | None): The release the identifier is read at; ``None``: the last release of the
                graph.
            from_database (str | None): The external database the identifier is meant as; ``None``: any.
            to_release (int | None): The release the conversion goes to (it matters without ``from_release`` only).

        Returns:
            tuple[str | None, str | None]: The node name and how the graph matched it (``exact``, ``case``,
            ``separator``, ``version``, each also as ``synonym:<how>``), or ``(None, None)`` when nothing matches.
        """
        return self.graph.resolve_name(identifier, from_release, from_database, to_release)

    # Edges

    def history_edges(self, n: str, reverse: bool) -> list[tuple]:
        """Return the history edges of a backbone node in the order in which the search visits them.

        Args:
            n (str): Name of a backbone node.
            reverse (bool): ``False`` for the edges that leave ``n``, ``True`` for those that enter it.

        Returns:
            list[tuple]: ``(other node, old_release, new_release, weight, key)`` per edge, sorted by the release of
            the event (``old_release`` for leaving, ``new_release`` for entering edges), the other node and the key.
        """
        position = 2 if reverse else 1
        return sorted(
            (tuple(edge) for edge in self.graph.history_edges(n, reverse=reverse)),
            key=lambda e: (e[position], e[0], e[4]),
        )

    def xref_edges(self, n: str, reverse: bool) -> list[tuple]:
        """Return the cross-reference edges of a node, ordered by the node at the other end.

        These are all edges that are not history: external id to Ensembl id, base gene id to gene, transcript to
        gene, translation to transcript. They point towards the backbone.

        Args:
            n (str): Node name.
            reverse (bool): ``False`` for the edges that leave ``n`` (towards the backbone), ``True`` for those that
                enter it.

        Returns:
            list[tuple]: ``(other node, items)`` sorted by the other node; ``items`` is the tuple of
            ``(database, assembly, releases)`` of the edge's :py:class:`idtrack.graph.Connection`.
        """
        return sorted(
            ((other, connection.items) for other, connection in self.graph.xref_edges(n, reverse=reverse)),
            key=lambda e: e[0],
        )

    def active_ranges(self, n: str) -> list[list]:
        """Return the release ranges in which a backbone node exists.

        Args:
            n (str): Name of a backbone node.

        Returns:
            list[list]: Inclusive ``[first, last]`` ranges in ascending order; ``last`` is ``inf`` for a gene that is
            alive at the last release of the graph.
        """
        return self.graph.active_ranges(n)

    def trios(self, n: str) -> frozenset:
        """Return every ``(database, assembly, release)`` in which a node occurs.

        Args:
            n (str): Node name.

        Returns:
            frozenset: The triples of the node, computed by the graph from the node's own edges.
        """
        return self.graph.trios(n)

    # Assemblies and the primary line of releases

    @property
    def priority(self) -> dict[int, int]:
        """Priority rank per assembly of the graph (1 is the primary assembly of the organism).

        Returns:
            dict[int, int]: Assembly key to rank; assemblies missing from the organism table rank last.
        """
        if self._priority is None:
            assemblies = sorted(self.assembly_releases)
            priority = {a: int(self._organism_table[a]["Priority"]) for a in assemblies if a in self._organism_table}
            missing = [a for a in assemblies if a not in self._organism_table]
            if missing:
                self.log.warning(
                    "Graph contains genome assemblies not configured for organism %r: %s. "
                    "Treating them as lowest priority; add them with `idtrack.organisms.register_organism` to set an "
                    "explicit order.",
                    self.graph.organism,
                    missing,
                )
                start_rank = (max(priority.values()) if priority else 0) + 1
                for offset, assembly in enumerate(missing):
                    priority[assembly] = start_rank + offset
            self._priority = priority
        return self._priority

    @property
    def assembly_releases(self) -> dict[int, frozenset]:
        """Releases that each assembly has in the graph.

        Returns:
            dict[int, frozenset]: Assembly key to the releases at which the graph holds data of that assembly.
        """
        if self._assembly_releases is None:
            self._assembly_releases = {
                int(assembly): frozenset(releases) & self.release_set
                for assembly, releases in self.graph.assembly_releases.items()
            }
        return self._assembly_releases

    @property
    def serving_assembly(self) -> dict[int, int]:
        """The assembly that Ensembl served as primary at each release of the graph.

        Returns:
            dict[int, int]: Release to the assembly that serves it in the backbone of the graph
            (:py:attr:`idtrack.TheGraph.serving_assembly`): the highest-priority assembly of the organism
            that has the release.
        """
        if self._serving_assembly is None:
            backbone = self.graph.serving_assembly
            self._serving_assembly = {r: int(backbone[r]) for r in self.releases if r in backbone}
        return self._serving_assembly

    @property
    def serving_line(self) -> dict[int, ReleaseSet]:
        """The releases at which each assembly is the primary one.

        Returns:
            dict[int, ReleaseSet]: Assembly key to its releases on the primary line: the graph's own statement of
            which assembly serves which release (:py:attr:`idtrack.graph.TheGraph.serving_line`), within the
            releases of the graph.
        """
        if self._serving_line is None:
            within = ReleaseSet.of(self.releases).mask
            self._serving_line = {
                int(assembly): ReleaseSet.from_mask(releases.mask & within)
                for assembly, releases in self.graph.serving_line.items()
            }
        return self._serving_line

    # Databases

    @property
    def external_databases(self) -> tuple[str, ...]:
        """The external databases of the graph, sorted.

        Returns:
            tuple[str, ...]: The databases.
        """
        if self._external_databases is None:
            # A database's synonyms are read under the database's own name: their bookkeeping name
            # `synonym_id::<database>` is no database of its own.
            self._external_databases = tuple(
                sorted({ids.public_name(d) for d in self.graph.available_external_databases})
            )
        return self._external_databases

    def final_databases(self, release: int) -> tuple[str, ...]:
        """Return the databases a conversion to a release can end in, sorted.

        Args:
            release (int): A release of the graph.

        Returns:
            tuple[str, ...]: The external databases of the assembly that serves the release (not the synonyms a
            database records: a synonym is no target database), and ``ensembl_gene`` and ``base_ensembl_gene``.
        """
        found = self._final_databases.get(release)
        if found is None:
            allowed = {
                database
                for database in self.graph.available_external_databases_assembly.get(
                    self.serving_assembly[release], set()
                )
                if not ids.is_synonym_database(database)  # synonyms are no target
            }
            allowed.update({ids.ENSEMBL_TYPE[ids.BACKBONE_FORM], ids.BASE_ENSEMBL_TYPE[ids.BACKBONE_FORM]})
            found = self._final_databases[release] = tuple(sorted(allowed))
        return found

    def primary_releases(self, items: tuple, exclude_databases: frozenset = frozenset()) -> frozenset:
        """Return the releases at which an edge holds on the primary line.

        Args:
            items (tuple): ``(database, assembly, releases)`` of one cross-reference edge.
            exclude_databases (frozenset): Databases whose release sets do not count.

        Returns:
            frozenset: Releases ``r`` listed by the edge for the assembly that is primary at ``r``
            (:py:func:`idtrack.graph.served_mask`, the graph's one rule for where a link counts).
        """
        return ReleaseSet.from_mask(served_mask(items, self.serving_line, exclude=exclude_databases))

    def primary_mask(self, items: tuple) -> int:
        """Return the releases at which an edge holds on the primary line, as a bit mask (every database counts).

        Args:
            items (tuple): ``(database, assembly, releases)`` of one cross-reference edge.

        Returns:
            int: Bit ``r`` set for every release ``r`` of :py:meth:`primary_releases`.
        """
        return served_mask(items, self.serving_line)

    def holds(self, items: tuple, release: int, database: str | None = None, any_assembly: bool = False) -> bool:
        """Tell whether an edge holds at a release on the primary line.

        Args:
            items (tuple): ``(database, assembly, releases)`` of one cross-reference edge.
            release (int): The release asked about.
            database (str | None): If given, only this database's release sets count.
            any_assembly (bool): If ``True``, the release may be listed for any assembly (for an identifier that
                no assembly has at a release it serves).

        Returns:
            bool: ``True`` when the edge lists the release for the assembly that is primary at that release.
        """
        if any_assembly:
            return any(
                (database is None or item_database == database) and release in releases
                for item_database, _, releases in items
            )
        return bool(served_mask(items, self.serving_line, database) >> release & 1)


class _Search:
    """Scratch memory of one conversion: facts of the nodes it touches, its settings and budget, and its counters."""

    def __init__(self, reader: _GraphReader, identifier: str, settings: SearchSettings):
        """Start the clock of a conversion.

        Args:
            reader (_GraphReader): The reader of the graph.
            identifier (str): The query node (for error messages).
            settings (SearchSettings): The budget and the rules of bridges of this conversion.
        """
        self.reader = reader
        self.identifier = identifier
        self.settings = settings
        self.budget_states = settings.budget_states
        self.budget_seconds = settings.budget_seconds
        self.started = time.monotonic()
        self.states = 0
        self._history: dict[tuple, list] = {}
        self._xrefs: dict[tuple, list] = {}
        self._ranges: dict[str, list] = {}
        self._chains: dict[str, list] = {}
        self._places: dict[Any, genes.GeneFacts] = {}
        self._relations: dict[tuple, str] = {}
        self._sides: dict[str, dict[str, frozenset[str]]] = {}
        self._holders: dict[str, dict[str, int]] = {}
        self._held_by_version: dict[str, dict[str, int]] = {}
        #: Every bridge considered, per (gene left, gene entered): ``{middle node: rule}``.
        self.bridge_checks: dict[tuple[str, str], dict[str, str]] = {}
        #: Every history step the walk did not follow, as the walk would take it ``(gene, other, key)``, with its rule
        #: (a key of :py:data:`idtrack.results.STEP_REASONS`).
        self.refused_steps: dict[tuple[str, str, int], str] = {}

    def spend(self, amount: int = 1) -> None:
        """Count work against the budget of the conversion.

        Args:
            amount (int): Number of units of work (expanded states or bridge candidates).

        Raises:
            SearchBudgetExceededError: If the conversion is over its budget of states or seconds.
        """
        self.states += amount
        if self.states > self.budget_states or (
            self.states % 256 < amount and time.monotonic() - self.started > self.budget_seconds
        ):
            raise SearchBudgetExceededError(
                self.identifier,
                self.states,
                time.monotonic() - self.started,
                self.budget_states,
                self.budget_seconds,
            )

    def history(self, n: str, reverse: bool) -> list:
        """History edges of a node, read once per conversion.

        Args:
            n (str): Name of a backbone node.
            reverse (bool): Direction, as in :py:meth:`_GraphReader.history_edges`.

        Returns:
            list: The edges in the reader's order.
        """
        key = (n, reverse)
        if key not in self._history:
            self._history[key] = self.reader.history_edges(n, reverse)
        return self._history[key]

    def xrefs(self, n: str, reverse: bool) -> list:
        """Cross-reference edges of a node, read once per conversion.

        Args:
            n (str): Node name.
            reverse (bool): Direction, as in :py:meth:`_GraphReader.xref_edges`.

        Returns:
            list: The edges in the reader's order.
        """
        key = (n, reverse)
        if key not in self._xrefs:
            self._xrefs[key] = self.reader.xref_edges(n, reverse)
        return self._xrefs[key]

    def ranges(self, n: str) -> list:
        """Active ranges of a backbone node, read once per conversion.

        Args:
            n (str): Name of a backbone node.

        Returns:
            list: Inclusive ``[first, last]`` ranges.
        """
        if n not in self._ranges:
            self._ranges[n] = self.reader.active_ranges(n)
        return self._ranges[n]

    def _versions(self, stable_id: str) -> list[str]:
        """Return the gene versions of a stable id (backbone nodes of real genes)."""
        reader = self.reader
        return [v for v, _ in self.xrefs(stable_id, False) if reader.is_backbone(v) and reader.is_real_gene(v)]

    def sides(self, stable_id: str) -> dict[str, frozenset[str]]:
        """Return the sides of a stable id that Ensembl reused across the mitochondrial genome and a nuclear sequence.

        A history step between two versions of the id that joins a gene of the mitochondrial genome and a gene of a
        nuclear sequence (:py:func:`idtrack.graph.genes.joins_mitochondrial_and_nuclear`, read from the kinds of the
        places of each version) separates two genes that share the stable id. The versions
        joined by the other history steps of the id are one side.

        Args:
            stable_id (str): A stable gene id (the node of the gene id without version).

        Returns:
            dict[str, frozenset[str]]: ``{version: the versions of its side}``; empty for an id without such a step
            (one gene: every version).
        """
        found = self._sides.get(stable_id)
        if found is None:
            graph = self.reader.graph
            versions = self._versions(stable_id)
            parent = {v: v for v in versions}

            def root(v: str) -> str:
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v

            reused = False
            for version in versions:
                for edge in graph.history_edges(version):
                    if edge.node not in parent or not math.isfinite(edge.new_release):
                        continue
                    old, new = int(edge.old_release), int(edge.new_release)
                    if genes.joins_mitochondrial_and_nuclear(graph, version, old, edge.node, new):
                        reused = True
                    else:
                        parent[root(edge.node)] = root(version)
            found = {}
            if reused:
                groups: dict[str, set[str]] = {}
                for version in versions:
                    groups.setdefault(root(version), set()).add(version)
                found = {version: frozenset(groups[root(version)]) for version in versions}
            self._sides[stable_id] = found
        return found

    def _side_of(self, n: str) -> tuple[str, frozenset[str] | None]:
        """Return the gene a backbone node stands for: its stable id, and its side where the id was reused."""
        stable_id = self.reader.base_id(n)
        return stable_id, self.sides(stable_id).get(n)

    def places(self, stable_id: str, side: frozenset[str] | None = None) -> genes.GeneFacts:
        """Where a gene lies, over every version of its stable id (:py:func:`idtrack.graph.genes.facts_of_nodes`), once.

        Args:
            stable_id (str): A stable gene id (the node of the gene id without version).
            side (frozenset[str] | None): For an id Ensembl reused across the mitochondrial genome and a nuclear
                sequence, the versions of one side (:py:meth:`sides`): the places of those versions only.

        Returns:
            GeneFacts: Its places on the reference sequence by assembly, whether it lies outside the reference
            sequence, whether the graph holds a place of it.
        """
        key: Any = stable_id if side is None else (stable_id, side)
        found = self._places.get(key)
        if found is None:
            versions = self._versions(stable_id) if side is None else sorted(side)
            found = self._places[key] = genes.facts_of_nodes(self.reader.graph, versions)
        return found

    def relation(self, n: str, other: str) -> str:
        """Where two genes lie relative to each other, as stable ids (:py:func:`idtrack.graph.genes.relation`).

        A stable id that Ensembl reused across the mitochondrial genome and a nuclear sequence is two genes: a node
        stands for the versions of its side (:py:meth:`sides`); every other stable id is one gene over its versions.

        Args:
            n (str): Name of a backbone node.
            other (str): Name of another backbone node.

        Returns:
            str: A key of :py:data:`idtrack.results.PLACE_RELATIONS`.
        """
        mine, theirs = self._side_of(n), self._side_of(other)
        key = (mine, theirs) if repr(mine) <= repr(theirs) else (theirs, mine)
        found = self._relations.get(key)
        if found is None:
            found = self._relations[key] = genes.relation(self.places(*mine), self.places(*theirs))
        return found

    def range_at(self, n: str, release: int) -> tuple | None:
        """Return the active range of a backbone node that contains a release.

        Args:
            n (str): Name of a backbone node.
            release (int): The release asked about.

        Returns:
            tuple | None: ``(first, last)`` or ``None`` when the node does not exist at that release.
        """
        for a, b in self.ranges(n):
            if a <= release <= b:
                return a, b
        return None


@dataclass(frozen=True)
class Conversion:
    """What the engine found for one identifier (the API adds how the string was read and wraps it in a result).

    Attributes:
        graph_id (str): The node the identifier was read as (for a transcript or translation id without version: its
            version of the starting release).
        from_release (int | None): The starting release (``None`` when nothing was read).
        from_release_inferred (bool): The starting release was chosen by the engine.
        from_databases (tuple[str, ...]): For an external id, the databases through which it was read.
        to_release (int): The target release.
        final_database (str | None): The database of the targets (``ensembl_gene`` with ``no_target``); ``None``
            without candidates.
        candidates (tuple[Candidate, ...]): Every candidate, ranked, the best marked.
        tie (bool): The best answer holds several pairs.
        decided_by (str | None): The criterion between the best answer and the runner-up.
        alternative_locus_genes (tuple[str, ...]): Copies outside the reference sequence left out.
        no_target (bool): No gene of the best answer (ranked as Ensembl genes) has an id in the final database; the
            genes are the targets.
        no_target_genes (tuple[str, ...]): Genes without an id in the final database while others have one.
        reason (str | None): Why there is no candidate (a key of :py:data:`idtrack.results.REASONS`).
        path (ExplainedPath | None): The explanation, when asked for.
    """

    graph_id: str
    from_release: int | None
    from_release_inferred: bool
    from_databases: tuple[str, ...]
    to_release: int
    final_database: str | None
    candidates: tuple[Candidate, ...]
    tie: bool
    decided_by: str | None
    alternative_locus_genes: tuple[str, ...]
    no_target: bool
    no_target_genes: tuple[str, ...]
    reason: str | None
    path: ExplainedPath | None


class Track:
    """Convert a gene identifier of one Ensembl release into its counterpart at another release.

    ``Track(graph)`` holds a graph (:py:class:`idtrack.graph.TheGraph`) and answers from it. How a conversion works
    (:py:meth:`convert`):

    1. **Starting release.** With ``from_release`` the identifier is read at that release and must be active there.
       Without it the identifier is read at the target release if it is active there, and at the latest release at
       which it is active otherwise. The rule is the same for every kind of identifier.
    2. **Entry.** An identifier that is not a versioned gene id is replaced by the genes it stands for at the
       starting release: a gene id without version by its version of that release, any other id by the genes that
       Ensembl's cross-references of that release attach it to.
    3. **History.** From those genes the search follows Ensembl's stable-id history (version changes, merges,
       splits, retirements) forward or backward in time until the target release. It is a best-first search over
       states ``(gene, release)``: every state is expanded once, so the work is bounded by the size of the part of
       the history that the query can reach, and every call has a budget. A step between a gene of the mitochondrial
       genome and a gene of a nuclear sequence is no lineage and is not followed (the explained path
       lists it among the refused steps).
    4. **Bridges.** Only when the history leads to no gene at the target release, the search may pass from a gene to
       another gene that carried the same external id (or the same transcript) at the same time or later (the gap in
       releases is reported), at most :py:attr:`idtrack.settings.SearchSettings.bridge_limit` times, and it uses as few
       bridges as possible. A bridge is taken only where it is evidence that one gene became the other: never
       through a synonym; where the two genes lie on the reference sequence of one assembly, only when their places
       overlap there; from or to a copy outside the reference sequence, only through an id both carried at one
       release that names one gene at each such release; elsewhere (their places cannot be compared), only when no
       release gave the id to both
       (:py:data:`idtrack.results.BRIDGE_RULES`). The explained path lists every bridge considered with the rule that
       took or refused it (:py:class:`idtrack.results.BridgeCheck`).
    5. **Copies beside the reference sequence.** Genes on alternative loci and assembly patches, and LRG records,
       leave the answer when a gene of the reference sequence was reached too; they are reported apart.
    6. **Final conversion.** Each gene found is reported as itself or converted to the ids of ``final_database``
       that are attached to it at the target release.
    7. **Ranking.** Every (gene, target) pair is ranked, best first, and the best answer is marked; it is several
       pairs when the identifier stands for several genes equally (a tie). :py:mod:`idtrack.engine.ranking` holds the
       rules.

    Throughout, an edge of the graph counts at a release only on the assembly that serves that release (the
    *primary line* of releases): an earlier assembly that Ensembl keeps serving beside a newer one (GRCh37 beside
    GRCh38) is not part of a conversion.

    Results never depend on the order of sets or on the hash seed of the process.

    Attributes:
        graph (TheGraph): The graph that is queried.
        search (SearchSettings): The default budget and bridge rules of a conversion (a call may give its own).
    """

    def __init__(self, graph: TheGraph, search: SearchSettings | None = None):
        """Create the engine of a graph.

        Args:
            graph: A graph built by :py:class:`idtrack.build.GraphMaker` or loaded from its file.
            search: The default search settings; ``None`` for :py:class:`idtrack.settings.SearchSettings`'s defaults.

        Raises:
            ArgumentTypeError: If *search* is not a :py:class:`idtrack.settings.SearchSettings`.
        """
        self.log = logging.getLogger(__name__)
        self.graph = graph
        self.search = check_search(search) or SearchSettings()

    @property
    def _reader(self) -> _GraphReader:
        """The reader of the graph, created at first use.

        Returns:
            _GraphReader: The reader bound to :py:attr:`graph`.
        """
        reader = self.__dict__.get("_reader_cache")
        if reader is None or reader.graph is not self.graph:
            reader = _GraphReader(self.graph)
            self.__dict__["_reader_cache"] = reader
        return reader

    # Arguments: the one place where the arguments of a conversion are checked (the API calls these as well)

    def _check_release(self, value: Any, name: str) -> int:
        """Return a release argument as ``int``, making sure the graph has that release.

        Args:
            value (Any): The value given (``int`` or another integer type such as ``numpy.int64``).
            name (str): The parameter's name, for the error message.

        Returns:
            int: The release.

        Raises:
            ArgumentTypeError: If the value is not an integer (``bool``, ``float`` and ``str`` are refused).
            UnknownReleaseError: If the graph does not contain that release.
        """
        release = _arguments.release(value, name)
        reader = self._reader
        if release not in reader.release_set:
            raise UnknownReleaseError(
                f"`{name}`={release} is not a release of this graph, which covers the Ensembl releases "
                f"{reader.first_release} to {reader.snapshot_release}.",
                argument=name,
                value=release,
                allowed=reader.releases,
            )
        return release

    def _check_target_release(self, value: Any) -> int:
        """Return the target release of a call: the graph's last release for ``None``, else a checked release."""
        return self._reader.snapshot_release if value is None else self._check_release(value, "to_release")

    def _check_final_database(self, value: Any, to_release: int) -> str:
        """Return the database a conversion to a release ends in, as the graph spells it.

        Args:
            value (Any): The database given; ``None`` means ``ensembl_gene`` (the genes themselves). A name that is
                not one of :py:meth:`final_databases` exactly is matched without letter case and surrounding white
                space, when that leaves one database.
            to_release (int): The target release (checked).

        Returns:
            str: The database, as :py:meth:`final_databases` lists it.

        Raises:
            ArgumentTypeError: If the value is neither a string nor ``None``.
            UnknownDatabaseError: If it is not one of :py:meth:`final_databases` of the release.
        """
        if value is None:
            return ids.GENE
        allowed = self._reader.final_databases(to_release)
        if value in allowed:
            return value
        _arguments.string(value, "final_database")
        folded = [database for database in allowed if database.casefold() == value.strip().casefold()]
        if len(folded) == 1:
            return folded[0]
        synonyms = " The synonyms a database records are no target database." if ids.is_synonym_database(value) else ""
        raise UnknownDatabaseError(
            f"`final_database`={value!r} is not a database of this graph at Ensembl release {to_release} (genome "
            f"assembly {self._reader.serving_assembly[to_release]});{_renamed_note(self.graph, value)}{synonyms} "
            f"valid: {list(allowed)}.",
            argument="final_database",
            value=value,
            allowed=allowed,
        )

    def _check_from_database(self, value: Any) -> str | None:
        """Return the external database an identifier is read as, as the graph spells it (``None``: any).

        Args:
            value (Any): The database given; matched as in :py:meth:`_check_final_database`.

        Returns:
            str | None: The database, or ``None``.

        Raises:
            ArgumentTypeError: If the value is neither a string nor ``None``.
            UnknownDatabaseError: If it is not an external database of the graph.
        """
        if value is None:
            return None
        allowed = self._reader.external_databases
        if value in allowed:
            return value
        _arguments.string(value, "from_database")
        folded = [database for database in allowed if database.casefold() == value.strip().casefold()]
        if len(folded) == 1:
            return folded[0]
        synonyms = (
            f" A database's synonyms are read under its own name ({ids.public_name(value)!r})."
            if ids.is_synonym_database(value)
            else ""
        )
        raise UnknownDatabaseError(
            f"`from_database`={value!r} is not an external database of this graph;{_renamed_note(self.graph, value)}"
            f"{synonyms} valid: {list(allowed)}.",
            argument="from_database",
            value=value,
            allowed=allowed,
        )

    def _check_search(self, search: Any) -> SearchSettings:
        """Return the search settings of a call: :py:attr:`search` for ``None``, else the given ones (checked)."""
        checked = check_search(search)
        return self.search if checked is None else checked

    def final_databases(self, to_release: int | None = None) -> list[str]:
        """Return the databases a conversion to a release can end in.

        These are the external databases of the genome assembly that serves the release, and the two forms of gene
        ids (``ensembl_gene`` with version, ``base_ensembl_gene`` without).

        Args:
            to_release (int | None): The target release. ``None``: the last release of the graph.

        Returns:
            list[str]: The accepted values of ``final_database`` of :py:meth:`convert`, sorted.

        Raises:
            ArgumentTypeError: If the release is not an integer.
            UnknownReleaseError: If the graph does not contain that release.
        """
        return list(self._reader.final_databases(self._check_target_release(to_release)))

    # Starting release and entry onto the backbone

    def _active_releases(
        self, search: _Search, node: str, database: str | None = None, spelt: Written | None = None
    ) -> tuple[frozenset, bool]:
        """Return the releases at which a node is active.

        A versioned gene is active in its history ranges, a gene id without version where one of its own versions is,
        any other node where one of its edges towards the backbone holds on the primary line. A node that never is
        on the primary line (an id of an archive assembly only) is active wherever any of its edges holds.

        Args:
            search (_Search): Scratch memory of the conversion.
            node (str): The node.
            database (str | None): For an external id, the database it is meant as: only edges of that database
                count.
            spelt (Written | None): For an external id matched exactly as typed, or an accession read with a version:
                only the releases at which an edge spells it so, or attaches that version, count
                (:py:meth:`_spelt_edges`).

        Returns:
            tuple[frozenset, bool]: The releases, and ``True`` when they had to be taken from all assemblies.
        """
        reader = self._reader
        node_type = reader.node_type(node)

        def of_gene(gene: str) -> set:
            return {
                r for a, b in search.ranges(gene) for r in range(int(a), int(min(b, reader.snapshot_release)) + 1)
            } & reader.release_set

        if node_type == reader.backbone_type:
            return frozenset(of_gene(node)), False

        edges = search.xrefs(node, False) if spelt is None else self._spelt_edges(search, node, spelt)
        if database is not None:
            edges = [
                (other, tuple(item for item in items if item[0] == database))
                for other, items in edges
                if any(item[0] == database for item in items)
            ]
        if node_type == reader.base_type:
            releases: set = set()
            for version, _ in edges:
                if reader.is_backbone(version) and reader.is_real_gene(version):
                    releases |= of_gene(version)
            if releases:
                return frozenset(releases), False

        releases = set()
        for _, items in edges:
            releases |= reader.primary_releases(items)
        if releases:
            return frozenset(releases), False
        for _, items in edges:
            for _, _, edge_releases in items:
                releases.update(edge_releases)
        return frozenset(releases) & reader.release_set, True

    def _entries(
        self,
        search: _Search,
        node: str,
        release: int,
        any_assembly: bool,
        database: str | None = None,
        spelt: Written | None = None,
    ) -> tuple[list[tuple], dict[str, set]]:
        """Return the genes a node stands for at a release, each with the chain of edges that leads to it.

        An identifier stands for a gene when a database attaches it, at the release, to the gene itself or to a
        transcript or translation of the gene. Every such attachment is followed, and the databases of all of them
        are reported per gene: one database may attach the identifier to the gene and another to its translation.

        Args:
            search (_Search): Scratch memory of the conversion.
            node (str): The query node.
            release (int): The starting release.
            any_assembly (bool): ``True`` for a node that exists in an archive assembly only.
            database (str | None): For an external id, the database it is meant as: only the genes that this
                database attaches the identifier to are entered.
            spelt (Written | None): For an external id matched exactly as typed, or an accession read with a version:
                only the genes that an edge spelling it so, or attaching that version, at the release reaches are
                entered (:py:meth:`_spelt_edges`); every database that attaches the identifier to such a gene counts,
                whatever its spelling or version.

        Returns:
            tuple[list[tuple], dict[str, set]]: ``(gene, steps)`` sorted by gene, where ``steps`` is the shortest
            chain of ``(source, target, 0, release)`` to the gene (empty when the node is itself a gene); and, per
            gene entered, every database that attaches the node to it at the release, whether the gene was entered
            through that database or not (empty for a gene id with version; for another Ensembl id the Ensembl
            relation that attaches it to the gene: ``base_ensembl_gene``, ``ensembl_transcript``, ...).
        """
        reader = self._reader
        if reader.is_backbone(node):
            return [(node, ())], {}

        found: dict[str, tuple] = {}
        attached: dict[str, set] = {}

        def keep(gene: str, steps: tuple) -> None:
            if gene not in found or (len(steps), steps) < (len(found[gene]), found[gene]):
                found[gene] = steps

        if reader.node_type(node) == reader.base_type:
            # a gene id without version stands for its version of the release; it is read through its own relation
            for version, items in search.xrefs(node, False):
                if reader.is_backbone(version) and reader.is_real_gene(version):
                    if search.range_at(version, release) is not None:
                        keep(version, ((node, version, 0, release),))
                        attached[version] = {
                            item_database
                            for item_database, _, _ in items
                            if reader.holds(items, release, database=item_database, any_assembly=any_assembly)
                        }
            if found:
                return sorted(found.items()), attached

        def walk(current: str, steps: tuple, depth: int, databases: frozenset, entered: bool, first: bool) -> None:
            edges = self._spelt_edges(search, current, spelt) if depth == 1 and first and spelt is not None else None
            for other, items in search.xrefs(current, False) if edges is None else edges:
                if not reader.holds(items, release, any_assembly=any_assembly):
                    continue
                if depth == 1:
                    # the databases of the first edge that hold at the release: they attach the identifier
                    databases = frozenset(
                        item_database
                        for item_database, _, _ in items
                        if reader.holds(items, release, database=item_database, any_assembly=any_assembly)
                    )
                    entered = database is None or database in databases
                step = steps + ((current, other, 0, release),)
                if reader.is_backbone(other):
                    if reader.is_real_gene(other) and search.range_at(other, release) is not None:
                        if first:
                            attached.setdefault(other, set()).update(databases)
                            if entered:
                                keep(other, step)
                        elif other in attached:  # a gene entered through a link spelt as typed
                            attached[other].update(databases)
                elif depth < 3 and reader.node_type(other) != ids.EXTERNAL:
                    walk(other, step, depth + 1, databases, entered, first)

        walk(node, (), 1, frozenset(), False, True)
        if spelt is not None:
            # The genes are those of the links spelt as typed; every database that attaches the identifier to such a
            # gene supports it, whatever its spelling (two spellings of one value for one gene are one value).
            walk(node, (), 1, frozenset(), False, False)
        return sorted(found.items()), {gene: attached[gene] for gene in found}

    def _start(
        self,
        search: _Search,
        node: str,
        from_release: int | None,
        to_release: int,
        database: str | None = None,
        spelt: Written | None = None,
    ) -> tuple[int | None, list, dict[str, set], bool]:
        """Decide the release a query is read at and the genes it stands for there.

        Args:
            search (_Search): Scratch memory of the conversion.
            node (str): The query node.
            from_release (int | None): The release given by the caller, or ``None``.
            to_release (int): The target release.
            database (str | None): For an external id, the database it is meant as.
            spelt (Written | None): How an external id was written, where that restricts its links.

        Returns:
            tuple[int | None, list, dict[str, set], bool]: The starting release (``None`` for a node that is never
            active), the entries of :py:meth:`_entries` and the databases that attach the node to each of them, and
            ``True`` when the node exists in an archive assembly only.

        Raises:
            InactiveAtReleaseError: If ``from_release`` is given and the node is not active at it.
        """
        active, any_assembly = self._active_releases(search, node, database, spelt)
        start = self._starting_release(node, active, from_release, to_release)
        if start is None:
            return None, [], {}, any_assembly
        entries, attached = self._entries(search, node, start, any_assembly, database, spelt)
        return start, entries, attached, any_assembly

    def _spelt_edges(self, search: _Search, node: str, spelt: Written) -> list[tuple[str, tuple]]:
        """Return the edges of an external node that carry it as written, each with the releases at which they do.

        A string matched exactly as typed (``Tc``, a synonym of one mouse gene) means the identifier as the links
        that write it so carry it, not as the links that write it otherwise (``tc``, a synonym of Noto): the graph
        keeps one node for both spellings and the spelling of every link (:py:meth:`idtrack.TheGraph.spelling`). An
        accession read with a version (``XM_003129161.3``) means the accession as the links that attached that version
        carry it, at the releases at which they did (:py:meth:`idtrack.TheGraph.version_releases`).

        Args:
            search (_Search): Scratch memory of the conversion.
            node (str): The external node.
            spelt (Written): The spelling as typed (for a synonym, with the synonym prefix) and the version, either
                ``None`` where it does not restrict.

        Returns:
            list[tuple[str, tuple]]: ``(other node, items)`` as :py:meth:`_Search.xrefs` gives them, with the releases
            of each ``(database, assembly, releases)`` cut to those that carry it as written; edges with no such
            release left out.
        """
        graph = self.graph
        kept = []
        for other, items in search.xrefs(node, False):
            cut = []
            for database, assembly, releases in items:
                mask = releases.mask
                if spelt.text is not None:
                    mask = graph.spelt_releases(node, other, database, mask, spelt.text)
                if mask and spelt.version is not None:
                    mask = graph.version_releases(node, other, database, mask, spelt.version)
                if mask:
                    cut.append((database, assembly, ReleaseSet.from_mask(mask)))
            if cut:
                kept.append((other, tuple(cut)))
        return kept

    @staticmethod
    def _starting_release(identifier: str, active: frozenset, from_release: int | None, to_release: int) -> int | None:
        """Apply the one rule for the release an identifier is read at.

        Args:
            identifier (str): The query, for the error message.
            active (frozenset): The releases at which the identifier is active.
            from_release (int | None): The release given by the caller, or ``None``.
            to_release (int): The target release.

        Returns:
            int | None: ``from_release`` when given; else the target release if the identifier is active there, else
            its latest active release; ``None`` for an identifier that is never active.

        Raises:
            InactiveAtReleaseError: If ``from_release`` is given and the identifier is not active at it.
        """
        if from_release is not None:
            if from_release not in active:
                raise InactiveAtReleaseError(identifier, from_release, _to_ranges(active))
            return from_release
        if not active:
            return None
        return to_release if to_release in active else int(max(active))

    def _version_at_start(
        self, search: _Search, stable_id: str, versions: tuple, from_release: int | None, to_release: int
    ) -> tuple[str, int] | None:
        """Choose the version that an Ensembl id without version stands for, and the release it is read at.

        Transcript and translation ids have no node without version (gene ids have the base gene id). Such an id is
        active where one of its versions is, its starting release follows the rule of every identifier, and it
        stands for its version of that release (the highest one, should a parallel assembly hold another version at
        the same release).

        Args:
            search (_Search): Scratch memory of the conversion.
            stable_id (str): The id as given, without version.
            versions (tuple): Its ``<id>.<version>`` nodes by ascending version.
            from_release (int | None): The release given by the caller, or ``None``.
            to_release (int): The target release.

        Returns:
            tuple[str, int] | None: The version node and the starting release; ``None`` when no version is ever
            active.
        """
        active = {version: self._active_releases(search, version)[0] for version in versions}
        start = self._starting_release(stable_id, frozenset().union(*active.values()), from_release, to_release)
        if start is None:
            return None
        return [version for version in versions if start in active[version]][-1], start

    # The search

    def _bridges(self, search: _Search, gene: str) -> list[tuple]:
        """Return the chains of cross-reference edges that link a gene to other genes.

        A chain has one external id or one shared transcript in the middle. For each chain the releases at which
        its two sides hold on the primary line are returned, so that the caller can place the bridge in time; the
        caller then takes a chain it can place only where the rules take it (:py:meth:`_taken`).

        Args:
            search (_Search): Scratch memory of the conversion; the result is kept there per gene.
            gene (str): The gene the walk stands on.

        Returns:
            list[tuple]: ``(other gene, releases of the near side, releases of the far side at which the other gene
            is active, near-side edges, far-side edges)`` in a fixed order. Edges are ``(source, target)`` pairs in
            the direction of the walk.
        """
        if gene in search._chains:
            return search._chains[gene]
        reader = self._reader
        skip = search.settings.non_bridging_databases
        chains: list[tuple] = []

        def far_releases(other: str, releases: frozenset) -> tuple:
            return tuple(r for r in sorted(releases) if search.range_at(other, r) is not None)

        def add(other: str, near: frozenset, far: frozenset, near_edges: tuple, far_edges: tuple) -> None:
            if other == gene or not near or not far or not reader.is_real_gene(other):
                return
            far_active = far_releases(other, far)
            if far_active:
                chains.append((other, tuple(sorted(near)), far_active, near_edges, far_edges))

        def through_external(external: str, near: frozenset, near_edges: tuple, direct_only: bool) -> None:
            if not near or reader.is_hyperconnective(external):
                return
            search.spend()
            for other, items in search.xrefs(external, False):
                far = reader.primary_releases(items, skip)
                if not far:
                    continue
                if reader.is_backbone(other):
                    add(other, near, far, near_edges, ((external, other),))
                elif not direct_only:
                    for upper, upper_items in search.xrefs(other, False):
                        far_2 = far & reader.primary_releases(upper_items)
                        if not far_2:
                            continue
                        if reader.is_backbone(upper):
                            add(upper, near, far_2, near_edges, ((external, other), (other, upper)))
                        else:
                            for top, top_items in search.xrefs(upper, False):
                                far_3 = far_2 & reader.primary_releases(top_items)
                                if far_3 and reader.is_backbone(top):
                                    add(top, near, far_3, near_edges, ((external, other), (other, upper), (upper, top)))

        for first, items in search.xrefs(gene, True):
            first_type = reader.node_type(first)
            if first_type == ids.EXTERNAL:
                through_external(first, reader.primary_releases(items, skip), ((gene, first),), False)
            elif first_type != reader.base_type:  # a transcript of the gene
                near_1 = reader.primary_releases(items)
                if not near_1:
                    continue
                for other, other_items in search.xrefs(first, False):  # a transcript shared with another gene
                    if reader.is_backbone(other) and reader.base_id(other) != reader.base_id(gene):
                        add(other, near_1, reader.primary_releases(other_items), ((gene, first),), ((first, other),))
                for second, second_items in search.xrefs(first, True):
                    second_type = reader.node_type(second)
                    if second_type == ids.EXTERNAL:
                        near_2 = near_1 & reader.primary_releases(second_items, skip)
                        through_external(second, near_2, ((gene, first), (first, second)), True)
                    else:  # a translation of the transcript
                        near_2 = near_1 & reader.primary_releases(second_items)
                        if not near_2:
                            continue
                        for third, third_items in search.xrefs(second, True):
                            if reader.node_type(third) == ids.EXTERNAL:
                                near_3 = near_2 & reader.primary_releases(third_items, skip)
                                through_external(third, near_3, ((gene, first), (first, second), (second, third)), True)
        chains.sort()
        search._chains[gene] = chains
        return chains

    def _taken(self, search: _Search, gene: str, other: str, middle: str) -> bool:
        """Tell whether the rules take a bridge the walk can place in time, and record it with its rule.

        Args:
            search (_Search): Scratch memory of the conversion; every bridge decided is kept in
                ``search.bridge_checks``.
            gene (str): The gene the walk leaves.
            other (str): The gene the walk would enter.
            middle (str): The external id, or the transcript, that the two genes carry.

        Returns:
            bool: ``True`` when the rule of the bridge is one that takes it (:py:meth:`_bridge_rule`).
        """
        checks = search.bridge_checks.setdefault((gene, other), {})
        rule = checks.get(middle)
        if rule is None:
            rule = checks[middle] = self._bridge_rule(search, gene, other, middle)
        return rule in BRIDGE_RULES_TAKEN

    def _bridge_rule(self, search: _Search, gene: str, other: str, middle: str) -> str:
        """Decide whether a chain from one gene through an id to another gene is a bridge.

        A bridge is evidence that one gene became the other only when the id leaves one gene for the other at the same
        place, so the rules are, in order:

        1. A synonym is no bridge (``synonym``): a synonym names other genes far more often than it stands for one.
           Nor is any id between the two sides of a stable id that Ensembl reused across the mitochondrial genome and
           a nuclear sequence (``mitochondrial_genome``: two genes, as the history step between them is
           refused); the places of such an id are compared per side (:py:meth:`_Search.sides`).
        2. Where the two genes lie on the reference sequence of one assembly, or Ensembl's mapping of their two
           assemblies carries the earlier place onto the later one, the bridge is taken when the places overlap
           (``same_place``) and refused when they do not (``places_apart``), whatever else the id names; so is a
           copy whose assembly exception's reference region shares no base with the other gene.
        3. Where a gene lies outside the reference sequence (a patch, a haplotype, an alternative locus, an LRG
           record) and the reference region its assembly exception replaces holds the other gene (or it has none: an
           LRG record), places are not compared base by base (a copy outside that region lies apart: rule 2). Such a
           copy stands for the gene whose names it carries while it exists: the
           bridge is taken when the two stable ids carry the id at one release and, at every release at which both
           carry it, the genes that carry it are one gene (:py:meth:`idtrack.TheGraph.count_genes` at that release: a
           copy adds no gene) (``outside_reference``); it is refused when the id names several genes at such a release
           (``copy_id_of_several_genes``: a family label or a shared alias does not say which gene the copy stands
           for), and when the id came to the other gene only at another time (``copy_id_elsewhere``: a name Ensembl
           moved to another gene after the copy's end, or before its start).
        4. Elsewhere the places cannot be compared (the genes never lie on one assembly and no published mapping
           carries one onto the other's assembly; or the graph holds no place of one of them). The bridge is refused
           when the two stable ids
           carry the id at one release (``shared_at_one_release``: a shared alias, a duplicated symbol or a family
           label names two living genes at once, and the end of one says nothing about the other), and taken when no
           release gives it to both (``never_shared``: the id left one gene for the other). On a built graph, which
           holds a place of every gene version, only ``never_shared`` comes here: two stable ids that carry the id at
           one release on the primary line both lie on that release's assembly, so rule 2 or 3 decides first;
           ``shared_at_one_release`` decides for a graph without places (one made in memory).

        Args:
            search (_Search): Scratch memory of the conversion.
            gene (str): The gene the walk leaves.
            other (str): The gene the walk would enter.
            middle (str): The external id, or the transcript, that the two genes carry.

        Returns:
            str: A key of :py:data:`idtrack.results.BRIDGE_RULES`.
        """
        if middle.startswith(ids.SYNONYM_PREFIX):
            return "synonym"
        reader = self._reader
        sides = search.sides(reader.base_id(gene)) if reader.base_id(gene) == reader.base_id(other) else {}
        if sides and sides.get(gene) != sides.get(other):
            # the two sides of an id Ensembl reused across the mitochondrial genome and a nuclear sequence:
            # two genes, joined by no id, as the step between them joins nothing
            return "mitochondrial_genome"
        relation = search.relation(gene, other)
        if relation in ("same_place", "places_apart"):
            return relation
        mine, theirs = reader.base_id(gene), reader.base_id(other)
        holders = self._holders(search, middle) if mine != theirs else {}
        shared = self._held(search, middle, gene, holders) & self._held(search, middle, other, holders)
        if relation == "outside_reference":
            if mine == theirs:
                return "outside_reference"
            if not shared:
                return "copy_id_elsewhere"
            return "outside_reference" if self._one_gene_holds(holders, shared) else "copy_id_of_several_genes"
        return "shared_at_one_release" if shared else "never_shared"

    def _one_gene_holds(self, holders: dict[str, int], releases: int) -> bool:
        """Tell whether, at every release of a mask, the genes that carry an id there are one gene.

        At a release, the holders are counted by the one rule of a gene at that release
        (:py:meth:`idtrack.TheGraph.count_genes` with ``release``): ids that live then are one gene only where their
        places overlap then, and a holder that lies outside the reference sequence at that release is a copy and adds
        no gene (copies alone are one gene).

        Args:
            holders (dict[str, int]): Per stable gene id, the releases at which it carries the id
                (:py:meth:`_holders`).
            releases (int): A bit mask of releases.

        Returns:
            bool: ``True`` when the holders of each release of the mask are at most one gene.
        """
        for release in ReleaseSet.from_mask(releases):
            group = [stable_id for stable_id, mask in holders.items() if mask >> release & 1]
            if self.graph.count_genes(group, release=release) > 1:
                return False
        return True

    def _holders(self, search: _Search, middle: str) -> dict[str, int]:
        """Return the releases at which each stable id carries an id, on the primary line.

        Args:
            search (_Search): Scratch memory of the conversion; the result is kept there per id.
            middle (str): An external id or a transcript.

        Returns:
            dict[str, int]: Per stable gene id, the bit mask of the releases at which the id is attached (by any
            database) to one of its versions: to the gene itself, or to its transcript or translation.
        """
        found = search._holders.get(middle)
        if found is not None:
            return found
        reader = self._reader
        found = {}
        by_version: dict[str, int] = {}

        def walk(node: str, mask: int, depth: int) -> None:
            for upper, items in search.xrefs(node, False):
                held = mask & reader.primary_mask(items)
                if not held:
                    continue
                if reader.is_backbone(upper):
                    if reader.is_real_gene(upper):
                        stable_id = reader.base_id(upper)
                        found[stable_id] = found.get(stable_id, 0) | held
                        by_version[upper] = by_version.get(upper, 0) | held
                elif depth < 3 and reader.node_type(upper) != ids.EXTERNAL:
                    walk(upper, held, depth + 1)

        walk(middle, -1, 1)
        search._holders[middle] = found
        search._held_by_version[middle] = by_version
        return found

    def _held(self, search: _Search, middle: str, gene: str, holders: dict[str, int]) -> int:
        """Return the releases at which the gene a node stands for carries an id, on the primary line.

        The gene is its stable id (:py:meth:`_holders`), or, for an id Ensembl reused across the mitochondrial genome
        and a nuclear sequence, the versions of the node's side (:py:meth:`_Search.sides`).

        Args:
            search (_Search): Scratch memory of the conversion (after :py:meth:`_holders` of the id).
            middle (str): The external id or transcript.
            gene (str): A gene version.
            holders (dict[str, int]): :py:meth:`_holders` of the id (empty where no release needs comparing).

        Returns:
            int: A bit mask of releases.
        """
        if not holders:
            return 0
        stable_id, side = search._side_of(gene)
        if side is None:
            return holders.get(stable_id, 0)
        by_version = search._held_by_version.get(middle, {})
        mask = 0
        for version in side:
            mask |= by_version.get(version, 0)
        return mask

    @staticmethod
    def _place_bridge(near: tuple, far: tuple, low: int, high: int, target: int, reverse: bool) -> tuple | None:
        """Place a bridge in time.

        Forward, the walk leaves the gene at a release ``x`` of the near side within ``[low, high]`` and enters the
        other gene at a release ``y`` of the far side with ``x <= y <= target``. The pair with the smallest gap is
        chosen, and among those the latest ``x``. Reverse is the mirror image.

        Args:
            near (tuple): Sorted releases at which the near side of the chain holds.
            far (tuple): Sorted releases at which the far side holds and the other gene is active.
            low (int): First release of the window in which the walk stands on the gene.
            high (int): Last release of that window.
            target (int): The target release.
            reverse (bool): Direction of the walk.

        Returns:
            tuple | None: ``(x, y)`` or ``None`` when the bridge cannot be placed.
        """
        best: tuple | None = None
        if not reverse:
            for i in range(bisect.bisect_right(near, high) - 1, -1, -1):
                x = near[i]
                if x < low:
                    break
                j = bisect.bisect_left(far, x)
                if j < len(far) and far[j] <= target:
                    gap = far[j] - x
                    if best is None or gap < best[0]:
                        best = (gap, x, far[j])
                        if gap == 0:
                            break
        else:
            for i in range(bisect.bisect_left(near, low), len(near)):
                x = near[i]
                if x > high:
                    break
                j = bisect.bisect_right(far, x) - 1
                if j >= 0 and far[j] >= target:
                    gap = x - far[j]
                    if best is None or gap < best[0]:
                        best = (gap, x, far[j])
                        if gap == 0:
                            break
        return None if best is None else (int(best[1]), int(best[2]))

    def _search_paths(
        self, search: _Search, entries: list, start: int, to_release: int, allow_bridges: bool
    ) -> dict[str, dict]:
        """Find the genes at the target release that the entry genes lead to, each with its best path.

        Args:
            search (_Search): Scratch memory of the conversion.
            entries (list): ``(gene, steps)`` pairs of :py:meth:`_entries`.
            start (int): The starting release.
            to_release (int): The target release.
            allow_bridges (bool): Whether bridges through external ids may be used when the history leads nowhere.

        Returns:
            dict[str, dict]: Per destination gene: ``entry_gene`` (the entry gene its path starts from),
            ``entry_genes`` (every entry gene from which a path of the search reaches it, the path's own among them:
            several where the history merges genes), ``external_jump``, ``external_gap``, ``external_step``,
            ``unscored_steps``, ``score``, ``ensembl_step`` and ``the_path``.
        """
        reader = self._reader
        reverse = start > to_release
        bridge_limit = int(search.settings.bridge_limit) if allow_bridges else 0

        # cost = (bridges, bridge gap, external steps, steps without a score, -score, history steps);
        # state = (gene, release, bridges, beamed)
        heap: list = []
        settled: dict[tuple, tuple] = {}
        destinations: dict[str, tuple] = {}
        # Every step the search considered, also into a state already reached (one path is kept per state), and every
        # state that arrived at a gene: the entry genes whose paths reach a gene are read from them.
        children: dict[tuple, set] = {}
        arrivals: dict[str, set] = {}
        counter = 0

        def push(cost: tuple, state: tuple, parent: tuple | None, steps: tuple) -> None:
            nonlocal counter
            if parent is not None:
                children.setdefault(parent, set()).add(state)
            if state not in settled:
                counter += 1
                heapq.heappush(heap, (cost, state, counter, parent, steps))

        def loops_passed(gene: str, low: int, high: int) -> tuple[float, tuple]:
            """Score and steps of the scored self loops of a gene whose event lies in [low, high)."""
            factor: float = 1.0
            steps: tuple = ()
            for other, old, new, weight, key in search.history(gene, False):
                if other == gene and not math.isinf(new):
                    event = new if reverse else old
                    inside = (low < event <= high) if reverse else (low <= event < high)
                    if inside:
                        factor *= _weight(weight)
                        steps += ((gene, gene, key),)
            return factor, steps

        def arrive(gene: str, cost: tuple, state: tuple, steps: tuple) -> None:
            if not reader.is_real_gene(gene):
                return
            candidate = (cost, state, steps)
            arrivals.setdefault(gene, set()).add(state)
            if gene not in destinations or candidate[0] < destinations[gene][0]:
                destinations[gene] = candidate

        def expand(cost: tuple, state: tuple, level_states: list) -> None:
            gene, release, bridges, beamed = state
            found = search.range_at(gene, release)
            if found is None:
                return
            first, last = found
            n_bridges, gap, external_steps, unscored, negative_score, history_steps = cost

            if (not reverse and to_release <= last) or (reverse and to_release >= first):
                if not reverse:
                    factor, loop_steps = loops_passed(gene, release, to_release)
                else:
                    factor, loop_steps = loops_passed(gene, to_release, release)
                arrive(
                    gene,
                    (
                        n_bridges,
                        gap,
                        external_steps,
                        unscored,
                        negative_score * factor,
                        history_steps + len(loop_steps),
                    ),
                    state,
                    loop_steps,
                )

            for other, old, new, weight, key in search.history(gene, reverse):
                if other == gene:
                    continue
                if not reverse:
                    if not (release <= old <= last and old < to_release):
                        continue
                    factor, loop_steps = loops_passed(gene, release, old)
                    step, entered = (gene, other, key), new
                else:
                    if not (first <= new <= release and new > to_release):
                        continue
                    factor, loop_steps = loops_passed(gene, new, release)
                    step, entered = (gene, other, key), old
                if math.isinf(entered):
                    continue
                if self._joins_mitochondrial_and_nuclear(search, gene, other, old, new, key, reverse):
                    continue  # no lineage between the mitochondrial genome and a nuclear sequence
                # Ensembl's score 0 between two different ids (stored as no score) is no measured score;
                # such a step ranks a path below every path whose scores are measured
                no_score = weight != weight and reader.base_id(other) != reader.base_id(gene)
                push(
                    (
                        n_bridges,
                        gap,
                        external_steps,
                        unscored + no_score,
                        negative_score * factor * _weight(weight),
                        history_steps + len(loop_steps) + 1,
                    ),
                    (other, int(entered), bridges, False),
                    state,
                    loop_steps + (step,),
                )
            if not beamed and reader.is_real_gene(gene):
                level_states.append((cost, state))

        def run(level_states: list) -> None:
            while heap:
                cost, state, _, parent, steps = heapq.heappop(heap)
                if state in settled:
                    continue
                search.spend()
                settled[state] = (parent, steps)
                expand(cost, state, level_states)

        def bridge_out(cost: tuple, state: tuple) -> None:
            gene, release, bridges, _ = state
            first, last = search.range_at(gene, release)  # type: ignore[misc]
            n_bridges, gap, external_steps, unscored, negative_score, history_steps = cost
            # The gene is one entity throughout its active range: evidence from any release of that range counts.
            if not reverse:
                low, high = int(first), int(min(last, to_release))
            else:
                low, high = int(max(first, to_release)), int(min(last, reader.snapshot_release))
            for other, near, far, near_edges, far_edges in self._bridges(search, gene):
                placed = self._place_bridge(near, far, low, high, to_release, reverse)
                if placed is None or not self._taken(search, gene, other, far_edges[0][0]):
                    continue
                left, entered = placed
                steps = tuple((u, v, 0, left) for u, v in near_edges) + tuple((u, v, 0, entered) for u, v in far_edges)
                push(
                    (
                        n_bridges + 1,
                        gap + abs(entered - left),
                        external_steps + len(near_edges) + len(far_edges),
                        unscored,
                        negative_score,
                        history_steps,
                    ),
                    (other, entered, bridges + 1, True),
                    state,
                    steps,
                )

        for gene, steps in entries:
            push((0, 0, len(steps), 0, -1.0, 0), (gene, start, 0, False), None, steps)

        level_states: list = []
        run(level_states)
        level = 0
        while not destinations and level < bridge_limit:
            sources, level_states = level_states, []
            for cost, state in sources:
                bridge_out(cost, state)
            run(level_states)
            level += 1

        # The entry genes that reach each state, carried along every step considered until nothing grows.
        reaching: dict[tuple, set] = {}
        pending = [((gene, start, 0, False), gene) for gene, _ in entries]
        while pending:
            state, entry = pending.pop()
            if entry in reaching.setdefault(state, set()):
                continue
            reaching[state].add(entry)
            pending.extend((child, entry) for child in children.get(state, ()))

        result: dict[str, dict] = {}
        for gene in sorted(destinations):
            cost, state, last_steps = destinations[gene]
            path: list = list(last_steps)
            cursor: tuple | None = state
            entry_gene = gene
            while cursor is not None:
                entry_gene = cursor[0]
                parent, steps = settled[cursor]
                path[0:0] = steps
                cursor = parent
            result[gene] = {
                "entry_gene": entry_gene,
                "entry_genes": tuple(sorted(set().union(*(reaching.get(each, set()) for each in arrivals[gene])))),
                "external_jump": cost[0],
                "external_gap": cost[1],
                "external_step": cost[2],
                "unscored_steps": cost[3],
                "score": -cost[4],
                "ensembl_step": cost[5],
                "the_path": tuple(path) if path else ((None, gene, None),),
            }
        return result

    def _joins_mitochondrial_and_nuclear(
        self, search: _Search, gene: str, other: str, old: float, new: float, key: int, reverse: bool
    ) -> bool:
        """Tell whether a history step joins a gene of the mitochondrial genome and a gene of a nuclear sequence.

        At assembly changes Ensembl gave the ids of mitochondrial genes to nuclear genes (rhesus MT-ND1 to a gene of
        chromosome 8 of Mmul_10); such a step is no lineage: the walk does not follow it, the id has no
        successor across it (the other rules answer, such as a bridge), and the step is recorded with its rule for the
        explained path. The rule reads the kinds of the places of the two versions
        (:py:func:`idtrack.graph.genes.joins_mitochondrial_and_nuclear`, from Ensembl's sequence regions). A version
        without a place there is no evidence either way, and the step is followed. Every other move of place is
        followed. The versions on the two sides of such a step are two genes that share a stable id by
        Ensembl's reuse (:py:meth:`_Search.sides`): bridges compare the places of each side, and a bridge between the
        two sides is refused like the step.

        Args:
            search (_Search): Scratch memory of the conversion; a refused step is recorded there.
            gene (str): The version the walk stands on.
            other (str): The version the step leads to.
            old (float): The old side of the event (the last release of its earlier version).
            new (float): The new side of the event (the first release of its later version).
            key (int): The key of the history edge.
            reverse (bool): Whether the walk goes back in time.

        Returns:
            bool: ``True`` when the step is refused.
        """
        older, newer = (other, gene) if reverse else (gene, other)
        reader = self._reader
        if not (reader.is_real_gene(older) and reader.is_real_gene(newer)):
            return False
        if not genes.joins_mitochondrial_and_nuclear(self.graph, older, int(old), newer, int(new)):
            return False
        search.refused_steps[(gene, other, int(key))] = "mitochondrial_genome"
        return True

    # Final conversion

    def _final_targets(self, search: _Search, gene: str, database: str, release: int) -> tuple[dict, int]:
        """Return the ids of a database that a gene carries at a release.

        Args:
            search (_Search): Scratch memory of the conversion.
            gene (str): A gene that exists at the release.
            database (str): An external database, or the node type of gene ids without version.
            release (int): The target release.

        Returns:
            tuple[dict, int]: ``{target: path}`` and the confidence: 0 when the ids are attached to the gene (or to
            its transcripts and translations) at the release; 1 when they are attached at the release to a transcript
            or translation that belonged to the gene at another release. The dict is empty when there is none.
        """
        reader = self._reader
        if database == reader.base_type:
            for other, _ in search.xrefs(gene, True):
                if reader.node_type(other) == reader.base_type:
                    return {other: ((gene, other, 0, release),)}, 0
            return {}, 0

        def collect(strict: bool) -> dict:
            found: dict[str, tuple] = {}

            def keep(target: str, steps: tuple) -> None:
                if target not in found or (len(steps), steps) < (len(found[target]), found[target]):
                    found[target] = steps

            def inner(items: tuple) -> bool:
                return reader.holds(items, release) if strict else True

            def mark(a: str, b: str) -> tuple:
                return (a, b, 0, release) if strict else (a, b, 0, None)

            for first, items in search.xrefs(gene, True):
                first_type = reader.node_type(first)
                if first_type == ids.EXTERNAL:
                    if strict and reader.holds(items, release, database=database):
                        keep(first, ((gene, first, 0, release),))
                elif first_type != reader.base_type and inner(items):
                    step_1 = (mark(gene, first),)
                    for second, second_items in search.xrefs(first, True):
                        if reader.node_type(second) == ids.EXTERNAL:
                            if reader.holds(second_items, release, database=database):
                                keep(second, step_1 + ((first, second, 0, release),))
                        elif inner(second_items):
                            step_2 = step_1 + (mark(first, second),)
                            for third, third_items in search.xrefs(second, True):
                                if reader.node_type(third) == ids.EXTERNAL and reader.holds(
                                    third_items, release, database=database
                                ):
                                    keep(third, step_2 + ((second, third, 0, release),))
            return found

        targets = collect(strict=True)
        if targets:
            return targets, 0
        return collect(strict=False), 1

    def gene_to_database(self, gene: str, final_database: str, release: int | None = None) -> tuple[list[str], int]:
        """Return the ids of a database that a gene carries at a release, without any search.

        This is the final conversion of :py:meth:`convert` for one gene. The answer depends on the gene, the
        database and the release only, never on the identifier a conversion started from, and it costs a few
        look-ups: a caller that needs the name of a gene it already has (the one fixed name per gene of a data set)
        asks here instead of converting again.

        Args:
            gene (str): A gene id with version that exists at the release.
            final_database (str): A database of :py:meth:`final_databases` of the release (``ensembl_gene``: the
                gene itself).
            release (int | None): The Ensembl release. ``None``: the last release of the graph.

        Returns:
            tuple[list[str], int]: The ids as the database wrote them at the release, in ascending order, and the
            confidence as in :py:meth:`convert` (0: attached at the release; 1: attached at the release to a
            transcript or translation the gene had at another release). An empty list when the gene has no id in
            the database.

        Raises:
            ArgumentTypeError: If ``gene`` or the database is not a string, or the release not an integer.
            UnknownReleaseError: If the release is not one of the graph.
            UnknownDatabaseError: If the database is not one of the graph at the release.
            UnknownIdentifierError: If ``gene`` is not a gene id with version that is active at the release.
        """
        _arguments.string(gene, "gene")
        reader = self._reader
        target = reader.snapshot_release if release is None else self._check_release(release, "release")
        final = self._check_final_database(final_database, target)
        database = None if final == ids.GENE else final
        search = _Search(reader, gene, self.search)
        if (
            not reader.has_node(gene)
            or not reader.is_backbone(gene)
            or not reader.is_real_gene(gene)
            or search.range_at(gene, target) is None
        ):
            raise UnknownIdentifierError(
                f"{gene!r} is not a gene id with version that is active at Ensembl release {target}.",
                argument="gene",
                value=gene,
            )
        if database is None:
            return [gene], 0
        targets, confidence = self._final_targets(search, gene, database, target)
        if database == reader.base_type:
            return sorted(targets), confidence
        return (
            sorted(
                self.graph.spelling(target_id, database, target, via=_linked_from(steps))
                for target_id, steps in targets.items()
            ),
            confidence,
        )

    def _node_scores(self, search: _Search, gene: str, release: int) -> list[int]:
        """Count how richly a gene is annotated at a release.

        The counts are the last criterion of the best answer of :py:meth:`convert`: between the successors of one
        gene that are equal in every other respect, the one with more annotation is preferred (never between the
        genes that an identifier stands for: those are a tie). Only the gene's own transcripts and translations of that
        release and the external ids attached to them or to the gene are counted; hyper-connective external ids are
        left out.

        Args:
            search (_Search): Scratch memory of the conversion.
            gene (str): A gene node.
            release (int): The release at which the annotation is counted.

        Returns:
            list[int]: ``[-external ids, -transcripts, -translations]``.
        """
        reader = self._reader
        externals: set = set()
        transcripts: set = set()
        translations: set = set()
        # An external id counts through external databases only, not through an Ensembl relation that it also has.
        ensembl_relations = frozenset(ids.ENSEMBL_TYPE.values()) | frozenset(ids.BASE_ENSEMBL_TYPE.values())

        def count(node: str, depth: int) -> None:
            for other, items in search.xrefs(node, True):
                if not reader.holds(items, release):
                    continue
                other_type = reader.node_type(other)
                if other_type == ids.EXTERNAL:
                    if release in reader.primary_releases(items, ensembl_relations) and not reader.is_hyperconnective(
                        other
                    ):
                        externals.add(other)
                elif other_type != reader.base_type and depth < 2:
                    (transcripts if depth == 0 else translations).add(other)
                    count(other, depth + 1)

        count(gene, 0)
        return [-len(externals), -len(transcripts), -len(translations)]

    # The conversion

    def _bridge_report(self, search: _Search) -> tuple[BridgeCheck, ...]:
        """Return every bridge the search considered, per pair of genes, with the rule of each id between them.

        Args:
            search (_Search): Scratch memory of the conversion.

        Returns:
            tuple[BridgeCheck, ...]: Ordered by source and target; the ids of each in order.
        """
        return tuple(
            BridgeCheck(
                source=source,
                target=target,
                places=search.relation(source, target),
                taken=any(rule in BRIDGE_RULES_TAKEN for rule in through.values()),
                through=tuple(sorted(through.items())),
            )
            for (source, target), through in sorted(search.bridge_checks.items())
        )

    def _explained_nothing(self, search: _Search, start: int, target: int, explain: bool) -> ExplainedPath | None:
        """Return the explanation of a search that found no candidate: the bridges it considered (with ``explain``)."""
        if not explain:
            return None
        builder = PathBuilder(self.graph, start > target, target)
        builder.bridges(self._bridge_report(search))
        builder.refused(search.refused_steps)
        return builder.result()

    def _nothing(
        self,
        node: str,
        start: int | None,
        inferred: bool,
        to_release: int,
        reason: str,
        explain: bool,
        path: ExplainedPath | None = None,
    ) -> Conversion:
        """Return the conversion of an identifier that leads to no candidate."""
        if explain and path is None:
            path = ExplainedPath(nodes=(), steps=())
        return Conversion(
            graph_id=node,
            from_release=start,
            from_release_inferred=inferred,
            from_databases=(),
            to_release=to_release,
            final_database=None,
            candidates=(),
            tie=False,
            decided_by=None,
            alternative_locus_genes=(),
            no_target=False,
            no_target_genes=(),
            reason=reason,
            path=path if explain else None,
        )

    def convert(
        self,
        identifier: str,
        from_release: SupportsIndex | None = None,
        to_release: SupportsIndex | None = None,
        final_database: str | None = None,
        *,
        from_database: str | None = None,
        allow_bridges: bool = True,
        return_ensembl_alternative: bool = True,
        explain: bool = False,
        search: SearchSettings | None = None,
    ) -> Conversion:
        """Convert an identifier to the target release and, if asked, to a target database.

        Starting release: with ``from_release`` the identifier is read at that release and must be active there.
        Without it, the identifier is read at ``to_release`` if it is active there (the answer is then what the
        identifier means at the target release), and at the latest release at which it is active otherwise. A gene
        id without version stands for its version of the starting release. The rule is the same for Ensembl ids and
        for ids of external databases.

        Search: from the genes the identifier stands for, Ensembl's stable-id history is followed to the target
        release. Only when that leads to no gene, and ``allow_bridges`` allows it, the search passes through an
        external id or transcript that two genes carried at the same time or one after the other, with as few such
        bridges as possible (:py:attr:`idtrack.results.Candidate.bridges`: 0 is history only). A bridge is taken only
        where it is evidence that one gene became the other (never through a synonym; on one assembly only between
        genes whose places overlap; from or to a copy outside the reference sequence only through an id both carried
        at one release that names one gene at each such release; elsewhere only through an id that no release gave to
        both:
        :py:data:`idtrack.results.BRIDGE_RULES`).

        Final conversion: with ``final_database`` the answer is first a gene: the genes found are ranked as Ensembl
        genes (below), and a target in the final database is an id of that database attached at the target release to
        a gene of that best answer. One result has one final database: the genes of the best answer that have such an
        id are returned with it (where only some of a tie have one, those); if none of them has one, the genes
        themselves are returned (``no_target``), or nothing with ``return_ensembl_alternative=False``. Another gene
        the search reached is never answered by its id (it stays a candidate of the view ``all`` where it has one).

        Copies beside the reference sequence: Ensembl also annotates genes on alternative loci (haplotypes) and
        assembly patches, and LRG records (``LRG_<number>``, fixed reference sequences of single genes), each with
        a stable id of its own and with the names and accessions of the gene it is a copy of. Which genes these
        are is Ensembl's statement, kept in the graph (:py:meth:`idtrack.graph.TheGraph.is_non_reference` at the
        target release); nothing is read off an id. When the search reaches genes of the reference sequence and such
        copies, the copies are not candidates; they are listed in ``alternative_locus_genes``. When it reaches copies
        only (the identifier is such a gene or an LRG record, or a name that only they carry), they are the answer.

        Identity: a gene id that exists at the target release is its own answer. The queried gene id's own version
        at the target release is never taken out as a copy (a gene that lay on a patch of the assembly of the
        target release is still that gene), it comes first, and it is the best answer whatever else its history
        reaches.

        Ranking (:py:mod:`idtrack.engine.ranking`): every (gene, target) pair is ranked by the confidence of the
        final conversion, the number of bridges, the release gap of the bridges, the query being the entry gene's
        display name at the starting release (Ensembl's own name of the gene there:
        :py:meth:`idtrack.graph.TheGraph.display_releases`), the support of the query on the
        entry gene (the number of databases that attach the identifier to that gene at the starting release, to the
        gene itself or to one of its transcripts or translations), both the best over every entry gene from which the
        search reaches the gene (several where the history merges genes), the path score, being the query itself, the
        number of cross-reference and history steps, and the ids. The best answer: the queried gene id's own version
        at the target release when the search reached it; otherwise the entry genes with the best confidence,
        bridges, gap, display name and support each contribute their best successor (by score, then being the query itself, then,
        between successors of the one gene that are still equal, the richest annotation). If more than one gene or
        target remains, the answer is a tie. The candidates come best first, the pairs of one gene together.

        Args:
            identifier (str): The identifier. The graph says which node the string means
                (:py:meth:`idtrack.graph.TheGraph.resolve_name` for ``from_release``, without one for the last
                release of the graph, and for ``from_database``). A transcript or translation id without version
                stands for its version of the starting release.
            from_release (SupportsIndex | None): The Ensembl release the identifier comes from, an ``int`` or another
                integer type (``numpy.int64``). ``None``: see above.
            to_release (SupportsIndex | None): The target Ensembl release. ``None``: the last release of the graph.
            final_database (str | None): The database to convert the genes into. ``None`` or ``ensembl_gene``:
                the genes themselves; ``base_ensembl_gene``: gene ids without version; else an external database of
                the genome assembly of the target release.
            from_database (str | None): The external database the identifier is meant as (its ids, and where none of
                them matches, the synonyms it records): only the genes that this database attaches it to are entered,
                and an identifier that is not an id of that database (an Ensembl id, an id of other databases only)
                has no conversion. Between the genes entered, the one that more databases attach the identifier to is
                still the better supported. ``None``: all databases.
            allow_bridges (bool): Allow bridges through external ids when the history leads to no gene.
            return_ensembl_alternative (bool): When no gene found has an id in ``final_database``, return the genes
                themselves instead of nothing.
            explain (bool): Record the paths (:py:attr:`Conversion.path`, and the step indices of each candidate).
            search (SearchSettings | None): The budget and bridge rules of this call; ``None``: :py:attr:`search`.

        Returns:
            Conversion: The candidates, ranked, the best marked; without candidates, the reason.

        Raises:
            ArgumentTypeError: If ``identifier`` or a database is not a string, a release not an integer, a flag not
                a boolean, or ``search`` not a SearchSettings.
            UnknownReleaseError: If a release is not one of the graph.
            UnknownDatabaseError: If ``final_database`` is not available at the target release or ``from_database``
                is not an external database of the graph.
            InactiveAtReleaseError: If ``from_release`` is given and the identifier is not active at that release.
            UnknownVersionError: If the identifier is an Ensembl id with a version that the graph does not hold and
                that cannot be newer than the graph, or an accession with a version that its links never attached and
                that cannot be newer than the graph (below the latest, or of an accession not attached at the last
                release).
            SearchBudgetExceededError: If the search needs more states or time than the budget of the call.
        """
        _arguments.string(identifier, "identifier")
        target = self._check_target_release(to_release)
        return self._convert(
            identifier,
            from_release=None if from_release is None else self._check_release(from_release, "from_release"),
            to_release=target,
            final_database=self._check_final_database(final_database, target),
            from_database=self._check_from_database(from_database),
            allow_bridges=_arguments.boolean(allow_bridges, "allow_bridges"),
            return_ensembl_alternative=_arguments.boolean(return_ensembl_alternative, "return_ensembl_alternative"),
            explain=_arguments.boolean(explain, "explain"),
            search=self._check_search(search),
        )

    def _read(
        self, identifier: str, given: int | None, from_database: str | None, target: int
    ) -> tuple[str | None, Written | None, tuple, tuple]:
        """Return which node a string means for a conversion (:py:meth:`idtrack.TheGraph.resolve_name`).

        Args:
            identifier (str): The string as read.
            given (int | None): The release given, or ``None``.
            from_database (str | None): The database named as its source.
            target (int): The target release.

        Returns:
            tuple[str | None, Written | None, tuple, tuple]: The node (``None`` when no node is named so); how it was
            written where that restricts its links (:py:class:`Written`: the typed spelling where the string matched
            exactly a node spelt otherwise too, only the links spelt so being the identifier, the case step taking
            all; the version of an accession read with one); and for a string that names no node the versions it
            stands for (a transcript or translation id without version) and the versions held of an id or accession
            written with a version the graph does not have.
        """
        resolved, how = self._reader.resolve_name(identifier, given, from_database, target)
        if resolved is not None:
            text = None
            if how in ("exact", "synonym:exact") and self.graph.respelled(resolved):
                text = identifier if how == "exact" else f"{ids.SYNONYM_PREFIX}{identifier}"
            version = self.graph.reading_version(identifier, resolved, how) if how is not None else None
            return resolved, (None if text is None and version is None else Written(text, version)), (), ()
        held = tuple(self.graph.known_versions(identifier))
        return None, None, (() if held else tuple(self.graph.versions_of(identifier))), held

    def reads_at(self, identifier: str, release: int, to_release: int, from_database: str | None = None) -> bool:
        """Tell whether a conversion reads a string at a release given (:py:meth:`convert` raises otherwise).

        The string is read at the release where the node it means there is active then
        (:py:meth:`_active_releases`: an id that only an archive assembly has is active wherever its links hold); a
        transcript or translation id without version, where one of its versions is. A string that names nothing
        raises no such error: ``True``.

        Args:
            identifier (str): The string as read.
            release (int): The release given.
            to_release (int): The target release.
            from_database (str | None): The database named as its source.

        Returns:
            bool: ``False`` when the graph does not hold the string at the release (it is read by the rule without a
            release).
        """
        node, spelt, versions, _ = self._read(identifier, release, from_database, to_release)
        if node is None:
            if not versions:
                return True
            memory = _Search(self._reader, identifier, self.search)
            return any(release in self._active_releases(memory, version)[0] for version in versions)
        memory = _Search(self._reader, node, self.search)
        if from_database is not None and self._reader.node_type(node) != ids.EXTERNAL:
            return True  # not an id of a database: the conversion says so, wherever it is read
        if from_database is not None and node.startswith(ids.SYNONYM_PREFIX):
            from_database = ids.SYNONYM_PREFIX + from_database
        return release in self._active_releases(memory, node, from_database, spelt)[0]

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
        """Convert with arguments checked by :py:meth:`convert`'s checks (the API checks them with the same methods).

        Returns:
            Conversion: See :py:meth:`convert`.
        """
        reader = self._reader
        target, given = to_release, from_release
        database = None if final_database == ids.GENE else final_database
        resolved, spelt, versions, held = self._read(identifier, given, from_database, target)
        node: str = identifier if resolved is None else resolved
        if resolved is None:
            if held:
                raise UnknownVersionError(identifier, held)
            if not versions:
                return self._nothing(identifier, None, given is None, target, "not_in_graph", explain)

        if from_database is not None and (versions or reader.node_type(node) != ids.EXTERNAL):
            return self._nothing(node, None, given is None, target, "not_an_id_of_database", explain)
        if from_database is not None and node.startswith(ids.SYNONYM_PREFIX):
            from_database = ids.SYNONYM_PREFIX + from_database  # the synonyms of the database named

        memory = _Search(reader, node, search)
        read_at = given
        if versions:
            chosen = self._version_at_start(memory, identifier, versions, given, target)
            if chosen is None:
                return self._nothing(identifier, None, given is None, target, "never_active", explain)
            node, read_at = chosen
            memory.identifier = node
        start, entries, attached, _ = self._start(memory, node, read_at, target, from_database, spelt)
        if start is None or not entries:
            # read at its starting release (where it has one), it stands for no gene there
            return self._nothing(node, start, given is None, target, "no_gene_at_start", explain)
        # Through which databases the query belongs to each entry gene at the starting release: every database that
        # attaches it to the gene itself or to a transcript or translation of the gene. A named database says which
        # genes are entered (`_start`); the support of an entry gene is every database that attaches the identifier
        # to it, named or not: what the other databases say about a gene is evidence all the same.
        read_through = set().union(*attached.values()) if attached else set()
        from_databases = sorted(
            {ids.public_name(d) for d in (read_through if from_database is None else read_through & {from_database})}
        )
        support = {gene: max(1, len(attached.get(gene, ()))) for gene, _ in entries}
        if identifier.isdigit() and len(support) > 1:
            # a number is an id of every database whose ids are numbers; the databases of one gene are no
            # evidence against a gene that other databases (other numbers spelt alike) attach it to
            strongest = max(sorted(support), key=lambda gene: support[gene])
            for gene in support:
                if not attached.get(gene, set()) & attached.get(strongest, set()):
                    support[gene] = support[strongest]
        authorities = self.graph.authority_databases
        authority = {gene: bool(attached.get(gene, set()) & authorities) for gene, _ in entries}
        # Ensembl's display name of a gene at the starting release: its own statement of the gene's name there, which
        # ranks a gene first among the genes that carry the name (decided before support: :py:mod:`ranking`), for the
        # string as typed (its letter case is evidence where it has both cases)
        external = reader.node_type(node) == ids.EXTERNAL
        typed = identifier
        if spelt is not None and spelt.version is not None:
            suffix = ids.NUMBERED_SUFFIX.match(identifier)
            typed = suffix.group(1) if suffix is not None else identifier
        display = {gene: self.graph.display_rank(node, gene, start, typed) if external else 2 for gene, _ in entries}
        found = self._search_paths(memory, entries, start, target, allow_bridges)
        if not found:
            explained = self._explained_nothing(memory, start, target, explain)
            return self._nothing(node, start, given is None, target, "no_path_to_target", explain, explained)

        found, own, alternative_locus_genes = self._without_copies(node, versions, found, target)
        entry_of = {gene: record["entry_gene"] for gene, record in found.items()}
        # Where the history merges several entry genes into one gene, the query is that gene's display
        # name and supported on it as on the best of them, not only as on the entry gene of the one path kept.
        display_of = {gene: min(display[entry] for entry in record["entry_genes"]) for gene, record in found.items()}
        support_of = {gene: max(support[entry] for entry in record["entry_genes"]) for gene, record in found.items()}
        query_base = node if reader.node_type(node) == reader.base_type else None

        def is_query(gene: str, target_id: str) -> bool:
            return target_id == node or (query_base is not None and reader.base_id(gene) == query_base)

        def pair_of(gene: str, target_id: str, confidence: int | None) -> ranking.Pair:
            record = found[gene]
            return ranking.make_pair(
                own=gene in own,
                confidence=confidence,
                bridges=record["external_jump"],
                gap=record["external_gap"],
                support=support_of[gene],
                unscored=record["unscored_steps"],
                score=record["score"],
                same_as_query=is_query(gene, target_id),
                external_steps=record["external_step"],
                history_steps=record["ensembl_step"],
                gene=gene,
                target=target_id,
                display_rank=display_of[gene],
            )

        def annotation(gene: str) -> list[int]:
            return self._node_scores(memory, gene, target)

        # The answer is a gene before it is an id of a database: the genes of the best answer are ranked as Ensembl
        # genes, and a target in a final database is an id attached to one of them; where none of them has one, the
        # genes are the answer (no_target), and another gene the search reached is never answered by its id.
        answered: set[str] | None = None
        if database is not None:
            gene_pairs = sorted(pair_of(gene, gene, 0) for gene in found)
            answered = {pair.gene for pair in ranking.select_best(gene_pairs, own, entry_of, annotation)[0]}
        finals = self._finals(memory, found, database, target, return_ensembl_alternative, answered)
        if finals is None:
            explained = self._explained_nothing(memory, start, target, explain)
            return self._nothing(node, start, given is None, target, "no_id_in_final_database", explain, explained)
        per_gene, final_name, no_target, without_target = finals

        # An external id is reported as its database wrote it at the target release: the node name is the graph's one
        # spelling of the id, and a database may write it with other letter case, also from release to release.
        spell_as = None if no_target or database is None or database == reader.base_type else database
        pairs = sorted(
            pair_of(gene, target_id, confidence)
            for gene, (targets, confidence) in per_gene.items()
            for target_id in targets
        )
        # the best answer is among the pairs of the genes answered (all pairs where the targets are the genes)
        leading_pairs = pairs if answered is None or no_target else [pair for pair in pairs if pair.gene in answered]
        best, counts = ranking.select_best(leading_pairs, own, entry_of, annotation)
        ordered = ranking.ranked(pairs, best)
        decided = ranking.decided_by(ordered, best, own, entry_of, counts)
        leading = set(best)
        builder = PathBuilder(self.graph, start > target, target) if explain else None

        candidates: list[Candidate] = []
        position: dict[tuple[str, str], int] = {}
        for pair in ordered:
            gene, target_id = pair.gene, pair.target
            record = found[gene]
            reported = (
                target_id
                if spell_as is None
                else self.graph.spelling(target_id, spell_as, target, via=_linked_from(per_gene[gene][0][target_id]))
            )
            path: tuple[int, ...] = ()
            final_path: tuple[int, ...] = ()
            if builder is not None:
                path = builder.search_path(record["the_path"])
                final_path = builder.final_path(per_gene[gene][0][target_id])
            candidate = Candidate(
                gene=gene,
                target=reported,
                best=pair in leading,
                entry_gene=record["entry_gene"],
                entry_databases=tuple(sorted({ids.public_name(d) for d in attached.get(record["entry_gene"], ())})),
                authority=authority[record["entry_gene"]],
                display_name=display_of[gene] < 2,
                display_rank=display_of[gene],
                entry_support=support_of[gene],
                own_version=gene in own,
                final_confidence=per_gene[gene][1],
                bridges=int(record["external_jump"]),
                bridge_gap=int(record["external_gap"]),
                score=float(record["score"]),
                unscored_steps=int(record["unscored_steps"]),
                same_as_query=pair.within[2] == 0,
                external_steps=int(record["external_step"]),
                history_steps=int(record["ensembl_step"]),
                annotation=None if gene not in counts else tuple(counts[gene]),
                path=path,
                final_path=final_path,
            )
            key = (gene, reported)
            if key in position:  # two nodes of one gene spelt alike: one target, as a result lists it once
                candidates[position[key]] = candidate
            else:
                position[key] = len(candidates)
                candidates.append(candidate)

        return Conversion(
            graph_id=node,
            from_release=start,
            from_release_inferred=given is None,
            from_databases=tuple(from_databases),
            to_release=target,
            final_database=final_name,
            candidates=tuple(candidates),
            tie=len(best) > 1,
            decided_by=decided,
            alternative_locus_genes=tuple(alternative_locus_genes),
            no_target=no_target,
            no_target_genes=tuple(without_target),
            reason=None,
            path=None if builder is None else self._explained(builder, memory),
        )

    def _explained(self, builder: PathBuilder, search: _Search) -> ExplainedPath:
        """Return the explanation of a conversion with candidates: their paths, the bridges and the steps refused."""
        builder.bridges(self._bridge_report(search))
        builder.refused(search.refused_steps)
        return builder.result()

    def _without_copies(
        self, node: str, versions: tuple, found: dict[str, dict], target: int
    ) -> tuple[dict[str, dict], set[str], list[str]]:
        """Take the copies outside the reference sequence out of an answer that holds a gene of the reference sequence.

        The graph's mark says which genes these are. The Y copy of a pseudoautosomal gene (a gene in a pseudoautosomal
        region of Y at the place of its gene on X, :py:meth:`idtrack.graph.TheGraph.y_copies`) is taken out likewise
        beside that gene. The queried gene id's own version at the target
        release is never taken out: an id that exists at the target release is its own answer.

        Args:
            node: The query node.
            versions: The versions an id without version stands for (empty for any other query).
            found: The genes the search reached, with their records.
            target: The target release.

        Returns:
            tuple: The genes kept, the genes that are the query's own version, and the copies taken out.
        """
        reader = self._reader
        own_id = None
        if not versions and reader.node_type(node) == reader.base_type:
            own_id = node
        elif not versions and reader.is_backbone(node):
            own_id = reader.base_id(node)
        own = {gene for gene in found if own_id is not None and reader.base_id(gene) == own_id}
        copies = sorted(gene for gene in found if gene not in own and self.graph.is_non_reference(gene, target))
        alternative_locus_genes: list[str] = []
        if copies and any(not self.graph.is_non_reference(gene, target) for gene in found):
            alternative_locus_genes = copies
            found = {gene: record for gene, record in found.items() if gene not in set(copies)}
        # The Y copy of a pseudoautosomal gene, beside its gene on X, is listed apart as a copy is.
        y_copies = self.graph.y_copies(found, target) - own
        if y_copies:
            alternative_locus_genes = sorted(set(alternative_locus_genes) | y_copies)
            found = {gene: record for gene, record in found.items() if gene not in y_copies}
        return found, own, alternative_locus_genes

    def _finals(
        self,
        search: _Search,
        found: dict[str, dict],
        database: str | None,
        target: int,
        ensembl_alternative: bool,
        answered: set[str] | None = None,
    ) -> tuple[dict[str, tuple], str, bool, list[str]] | None:
        """Convert the genes found to the final database: one database and one confidence for the whole result.

        A target in the final database is an id attached to a gene of the answer (``answered``: the genes of the best
        answer ranked as Ensembl genes). Where none of them has one, the genes are the targets (``no_target``): a gene
        of a lower rank is never answered by its id. Where some have one, the genes of the answer with an id at the
        best confidence are the answer, and every gene with an id at that confidence is a candidate.

        Args:
            search: Scratch memory of the conversion.
            found: The genes and their records.
            database: The final database (``None``: the genes themselves).
            target: The target release.
            ensembl_alternative: Return the genes when none of the answer has an id in the database.
            answered: The genes of the best answer as Ensembl genes; ``None``: every gene found.

        Returns:
            tuple | None: ``({gene: ({target: steps}, confidence)}, final database, no_target, genes without
            target)``; ``None`` when no gene of the answer has an id and ``ensembl_alternative`` is false.
        """
        gene_form = ids.GENE
        if database is None:
            return {gene: ({gene: ()}, 0) for gene in found}, gene_form, False, []
        converted = {gene: self._final_targets(search, gene, database, target) for gene in found}
        with_target = {gene: value for gene, value in converted.items() if value[0]}
        leading = {gene: value for gene, value in with_target.items() if answered is None or gene in answered}
        if leading:
            best_confidence = min(value[1] for value in leading.values())
            finals = {gene: value for gene, value in with_target.items() if value[1] == best_confidence}
            without = sorted(gene for gene, value in converted.items() if not value[0])
            return finals, database, False, without
        if ensembl_alternative:
            return {gene: ({gene: ()}, None) for gene in found}, gene_form, True, []
        return None

    # Where a list of identifiers comes from

    @staticmethod
    def _check_mode(mode: Any) -> str:
        """Return the mode of :py:meth:`identify_source`, checked.

        Raises:
            InvalidArgumentError: If it is not one of :py:data:`SOURCE_MODES`.
        """
        if mode not in SOURCE_MODES:
            raise InvalidArgumentError(
                f"`mode`={mode!r} is not a mode; valid: {list(SOURCE_MODES)}.",
                argument="mode",
                value=mode,
                allowed=SOURCE_MODES,
            )
        return mode

    def identify_source(self, identifiers: list[str], mode: str) -> list[tuple]:
        """Count in which assemblies, databases and releases the identifiers of a data set occur.

        Only the edges of the given identifiers are read; nothing is computed for the rest of the graph.

        Args:
            identifiers (list[str]): Nodes of the graph (:py:meth:`idtrack.graph.TheGraph.resolve_name` turns a
                string into one).
            mode (str): What to count (:py:data:`SOURCE_MODES`): ``complete`` for ``(database, assembly, release)``,
                ``release``, ``assembly`` or ``assembly_release``.

        Returns:
            list[tuple]: ``(origin, count)`` pairs, the most frequent first. Origins with the same count are tied;
            among them the later release, then the higher-priority assembly, then the database name comes first, so
            the order does not depend on the input order or on the process.

        Raises:
            InvalidArgumentError: If ``mode`` is not one of the four values.
            ArgumentTypeError: If ``identifiers`` is not a sequence of strings.
            UnknownIdentifierError: If an identifier is not a node of the graph.
        """
        mode = self._check_mode(mode)
        identifiers = _arguments.strings(identifiers, "identifiers")
        reader = self._reader
        unknown = [identifier for identifier in identifiers if not reader.has_node(identifier)]
        if unknown:
            raise UnknownIdentifierError(
                f"{len(unknown)} identifiers are not nodes of the graph, e.g. {unknown[:5]}; resolve strings first "
                "(`TheGraph.resolve_name`).",
                argument="identifiers",
                value=unknown[:20],
            )
        return self._identify_source(identifiers, mode)

    def _identify_source(self, nodes: list[str], mode: str) -> list[tuple]:
        """Count the origins of nodes of the graph (arguments checked as in :py:meth:`identify_source`)."""
        reader = self._reader
        counts: Counter = Counter()
        for node in nodes:
            counts.update({_ORIGIN_OF[mode](trio) for trio in reader.trios(node)})

        priority = reader.priority
        worst = max(priority.values(), default=0) + 1

        def tie_order(origin: Any) -> tuple:
            if mode == "complete":
                return (-origin[2], priority.get(origin[1], worst), origin[1], origin[0])
            if mode == "release":
                return (-origin,)
            if mode == "assembly":
                return (priority.get(origin, worst), origin)
            return (-origin[1], priority.get(origin[0], worst), origin[0])

        return sorted(counts.items(), key=lambda item: (-item[1], tie_order(item[0])))


def _weight(weight: float) -> float:
    """Return the mapping score of a history edge as a factor of a path's score, 1.0 where Ensembl gives none.

    A step between two different ids without a score (Ensembl writes 0 where it measured nothing) is counted apart,
    before the product of the scores (:py:attr:`idtrack.results.Candidate.unscored_steps`).

    Args:
        weight (float): The ``weight`` of a history edge (NaN when the event has no score).

    Returns:
        float: The score as a factor of a path's score.
    """
    return 1.0 if weight != weight else float(weight)


def _to_ranges(releases: frozenset) -> list[list[int]]:
    """Write a set of releases as inclusive ranges.

    Args:
        releases (frozenset): Release numbers.

    Returns:
        list[list[int]]: ``[first, last]`` runs of consecutive releases in ascending order.
    """
    ranges: list[list[int]] = []
    for r in sorted(int(release) for release in releases):
        if ranges and r == ranges[-1][1] + 1:
            ranges[-1][1] = r
        else:
            ranges.append([r, r])
    return ranges


def _renamed_note(graph: Any, database: Any) -> str:
    """Return the sentence that names the graph's name of a database name Ensembl used and the graph does not."""
    renamed = getattr(graph, "renamed_databases", None) or {}
    new = renamed.get(database) if isinstance(database, str) else None
    return f" It is named {new!r} in this graph." if new else ""


def _linked_from(steps: tuple) -> str | None:
    """Return the Ensembl id from which the last step of a final conversion reaches the external id (``None``: none)."""
    return steps[-1][0] if steps else None


__all__ = ["SOURCE_MODES", "Conversion", "Track"]
