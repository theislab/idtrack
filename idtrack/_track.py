#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import bisect
import heapq
import logging
import math
import operator
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, SupportsIndex

from idtrack._db import DB
from idtrack._the_graph import ReleaseSet, served_mask

if TYPE_CHECKING:
    from idtrack._database_manager import DatabaseManager
    from idtrack._the_graph import TheGraph


class InactiveAtReleaseError(ValueError):
    """The identifier is not active at the release the caller gave as ``from_release``.

    Attributes:
        identifier (str): The graph node of the query.
        release (int): The release that was given.
        active_ranges (list[list[int]]): Inclusive ``[first, last]`` release ranges in which the identifier is active.
    """

    def __init__(self, identifier: str, release: int, active_ranges: list[list[int]]):
        """Keep the query, the release and the releases at which the identifier is active.

        Args:
            identifier (str): The graph node of the query.
            release (int): The release that was given.
            active_ranges (list[list[int]]): Inclusive release ranges in which the identifier is active.
        """
        super().__init__(identifier, release, active_ranges)
        self.identifier = identifier
        self.release = release
        self.active_ranges = active_ranges

    def __str__(self) -> str:
        """Name the releases at which the identifier is active.

        Returns:
            str: The message of the error.
        """
        ranges = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in self.active_ranges) or "none"
        return (
            f"{self.identifier!r} is not active at Ensembl release {self.release}; it is active at the releases: "
            f"{ranges}. Give one of these as `from_release`, or leave it out to start from the target release when "
            "the identifier is active there and from its latest active release otherwise."
        )


class UnknownVersionError(ValueError):
    """The identifier is an Ensembl id with a version that the graph does not hold.

    The version never existed, or the graph cannot know it (a version lower than the latest, or of an id that is no
    longer active). It is not answered with another version's gene.

    Attributes:
        identifier (str): The identifier as given.
        versions (tuple[str, ...]): The ``<id>.<version>`` ids that the graph holds, by ascending version.
    """

    def __init__(self, identifier: str, versions: tuple[str, ...]):
        """Keep the query and the versions that exist.

        Args:
            identifier (str): The identifier as given.
            versions (tuple[str, ...]): The versions of the id that the graph holds.
        """
        super().__init__(identifier, versions)
        self.identifier = identifier
        self.versions = versions

    def __str__(self) -> str:
        """Name the versions that exist.

        Returns:
            str: The message of the error.
        """
        return (
            f"{self.identifier!r} names a version that this graph does not hold; the versions that exist are: "
            f"{', '.join(self.versions)}. Give one of them, or the id without version."
        )


class SearchBudgetExceededError(RuntimeError):
    """A conversion needed more search states or more time than its budget allows.

    Attributes:
        identifier (str): The graph node of the query.
        states (int): Number of search states expanded when the search stopped.
        seconds (float): Seconds spent when the search stopped.
        budget_states (int): The budget of states of the call.
        budget_seconds (float): The budget of seconds of the call.
    """

    def __init__(self, identifier: str, states: int, seconds: float, budget_states: int, budget_seconds: float):
        """Keep the work done and the budget.

        Args:
            identifier (str): The graph node of the query.
            states (int): Number of search states expanded when the search stopped.
            seconds (float): Seconds spent when the search stopped.
            budget_states (int): The budget of states of the call.
            budget_seconds (float): The budget of seconds of the call.
        """
        super().__init__(identifier, states, seconds, budget_states, budget_seconds)
        self.identifier = identifier
        self.states = states
        self.seconds = seconds
        self.budget_states = budget_states
        self.budget_seconds = budget_seconds

    def __str__(self) -> str:
        """Say how far the search got and what its budget was.

        Returns:
            str: The message of the error.
        """
        return (
            f"The conversion of {self.identifier!r} was stopped after {self.states} search states and "
            f"{self.seconds:.1f} s (budget: {self.budget_states} states, {self.budget_seconds:.1f} s). Raise "
            "`Track.search_budget_states` or `Track.search_budget_seconds` to let it run longer."
        )


class _GraphReader:
    """The engine's view of the graph: its accessors in the engine's order, and the primary line of releases.

    The engine never touches the attribute layout of nodes and edges. It reads the graph through the accessor
    methods of :py:class:`idtrack.TheGraph` (``node_type``, ``base_id``, ``version``, ``history_edges``,
    ``xref_edges``, ``active_ranges``, ``trios``, ``resolve_name``, ``versions_of``, ``spelling``,
    ``hyperconnective_nodes``, ``assembly_releases``, ``backbone_assembly``); this class adds the order in which the
    search visits edges and the question whether an edge holds at a release on the primary line. Nothing is computed
    for the whole graph.
    """

    def __init__(self, graph: "TheGraph"):
        """Bind the reader to a graph and read the facts that hold for the whole graph.

        Args:
            graph (TheGraph): The graph that is queried.

        Raises:
            ValueError: If the graph's organism is not registered in the organism table, so that its assemblies
                cannot be ordered.
        """
        self.graph = graph
        self.log = logging.getLogger(__name__)
        self.backbone_type: str = DB.external_search_settings["nts_backbone"]
        self.base_type: str = DB.nts_base_ensembl[DB.backbone_form]

        self.releases: tuple[int, ...] = tuple(sorted(int(r) for r in graph.graph["confident_for_release"]))
        self.release_set = frozenset(self.releases)
        self.first_release = self.releases[0]
        self.last_release = self.releases[-1]

        organism = graph.graph.get("organism")
        # Assembly keys carry no order of their own (rhesus Mmul_8.0.1 is 801 but older than Mmul_10, 10): the order
        # comes from the organism table only.
        if not isinstance(organism, str) or organism not in DB.assembly_mysqlport_priority:
            raise ValueError(
                f"The graph's organism {organism!r} is not registered in the organism table "
                f"(supported: {DB.supported_organisms}), so its assemblies cannot be ordered. Register it with "
                "`idtrack.DB.register_organism` before loading the graph."
            )
        self._organism_table = DB.assembly_mysqlport_priority[organism]
        self._assembly_releases: dict[int, frozenset] | None = None
        self._primary_line: dict[int, ReleaseSet] | None = None
        self._primary_assembly: dict[int, int] | None = None
        self._priority: dict[int, int] | None = None

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
        return self.version(n) not in DB.alternative_versions

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
        self, identifier: str, release: int | None = None, database: str | None = None, target: int | None = None
    ) -> tuple[str | None, str | None]:
        """Find the node an identifier written by a user stands for at a release.

        Args:
            identifier (str): The identifier as given.
            release (int | None): The release the identifier is read at; ``None``: the last release of the graph.
            database (str | None): The external database the identifier is meant as; ``None``: any.
            target (int | None): The release the conversion goes to (it matters without ``release`` only).

        Returns:
            tuple[str | None, str | None]: The node name and how the graph matched it (``exact``, ``case``,
            ``separator``, ``version``, each also as ``synonym:<how>``), or ``(None, None)`` when nothing matches.
        """
        return self.graph.resolve_name(identifier, release, database, target)

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
            ``(database, assembly, releases)`` of the edge's :py:class:`idtrack._the_graph.Connection`.
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
                    "Treating them as lowest priority; add them with `idtrack.DB.register_organism` to set an "
                    "explicit order.",
                    self.graph.graph.get("organism"),
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
    def primary_assembly(self) -> dict[int, int]:
        """The assembly that Ensembl served as primary at each release of the graph.

        Returns:
            dict[int, int]: Release to the assembly that serves it in the backbone of the graph
            (:py:attr:`idtrack.TheGraph.backbone_assembly`): the highest-priority assembly of the organism
            that has the release.
        """
        if self._primary_assembly is None:
            backbone = self.graph.backbone_assembly
            self._primary_assembly = {r: int(backbone[r]) for r in self.releases if r in backbone}
        return self._primary_assembly

    @property
    def primary_line(self) -> dict[int, ReleaseSet]:
        """The releases at which each assembly is the primary one.

        Returns:
            dict[int, ReleaseSet]: Assembly key to its releases on the primary line: the graph's own statement of
            which assembly serves which release (:py:attr:`idtrack._the_graph.TheGraph.serving_line`), within the
            releases of the graph.
        """
        if self._primary_line is None:
            within = ReleaseSet.of(self.releases).mask
            self._primary_line = {
                int(assembly): ReleaseSet.from_mask(releases.mask & within)
                for assembly, releases in self.graph.serving_line.items()
            }
        return self._primary_line

    def primary_releases(self, items: tuple, exclude_databases: frozenset = frozenset()) -> frozenset:
        """Return the releases at which an edge holds on the primary line.

        Args:
            items (tuple): ``(database, assembly, releases)`` of one cross-reference edge.
            exclude_databases (frozenset): Databases whose release sets do not count.

        Returns:
            frozenset: Releases ``r`` listed by the edge for the assembly that is primary at ``r``
            (:py:func:`idtrack._the_graph.served_mask`, the graph's one rule for where a link counts).
        """
        return ReleaseSet.from_mask(served_mask(items, self.primary_line, exclude=exclude_databases))

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
        return bool(served_mask(items, self.primary_line, database) >> release & 1)


class _Search:
    """Scratch memory of one conversion: facts of the nodes it touches, its budget, and its counters."""

    def __init__(self, reader: _GraphReader, identifier: str, budget_states: int, budget_seconds: float):
        """Start the clock of a conversion.

        Args:
            reader (_GraphReader): The reader of the graph.
            identifier (str): The query node (for error messages).
            budget_states (int): Maximum number of search states the conversion may expand.
            budget_seconds (float): Maximum number of seconds the conversion may take.
        """
        self.reader = reader
        self.identifier = identifier
        self.budget_states = budget_states
        self.budget_seconds = budget_seconds
        self.started = time.monotonic()
        self.states = 0
        self._history: dict[tuple, list] = {}
        self._xrefs: dict[tuple, list] = {}
        self._ranges: dict[str, list] = {}
        self._chains: dict[str, list] = {}

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


class Track:
    """Convert a gene identifier of one Ensembl release into its counterpart at another release.

    ``Track`` holds the graph (:py:class:`idtrack.TheGraph`) and answers one question: given an identifier
    (an Ensembl gene id with or without version, a transcript or translation id, or an id of an external database
    such as a gene symbol), optionally the release it comes from, a target release and optionally a target database,
    which genes correspond to it at the target release, and which ids of the target database those genes carry.

    How a conversion works (:py:meth:`convert`):

    1. **Starting release.** With ``from_release`` the identifier is read at that release and must be active there.
       Without it the identifier is read at the target release if it is active there, and at the latest release at
       which it is active otherwise. The rule is the same for every kind of identifier.
    2. **Entry.** An identifier that is not a versioned gene id is replaced by the genes it stands for at the
       starting release: a gene id without version by its version of that release, any other id by the genes that
       Ensembl's cross-references of that release attach it to.
    3. **History.** From those genes the search follows Ensembl's stable-id history (version changes, merges,
       splits, retirements) forward or backward in time until the target release. It is a best-first search over
       states ``(gene, release)``: every state is expanded once, so the work is bounded by the size of the part of
       the history that the query can reach, and every call has a budget.
    4. **Bridges.** Only when the history leads to no gene at the target release, the search may pass from a gene to
       another gene that carried the same external id (or the same transcript) at the same time or later (the gap in
       releases is reported), at most ``DB.external_search_settings["jump_limit"]`` times, and it uses as few
       bridges as possible.
    5. **Copies beside the reference sequence.** Genes on alternative loci and assembly patches, and LRG records,
       leave the answer when a gene of the reference sequence was reached too; they are reported apart.
    6. **Final conversion.** Each gene found is reported as itself or converted to the ids of ``final_database``
       that are attached to it at the target release.
    7. **Order.** The result is ordered, best first; ``prioritize_to_one_filter`` keeps only the best, which is
       several genes when the identifier stands for several genes equally (a tie).

    Throughout, an edge of the graph counts at a release only on the assembly that was Ensembl's primary assembly
    at that release (the *primary line* of releases): an earlier assembly that Ensembl keeps serving beside a newer
    one (GRCh37 beside GRCh38) is not part of a conversion.

    Results never depend on the order of sets or on the hash seed of the process.

    Attributes:
        graph (TheGraph): The graph that is queried.
        search_budget_states (int): Largest number of search states one conversion may expand before
            :py:class:`SearchBudgetExceededError` is raised.
        search_budget_seconds (float): Longest time one conversion may take before the same error is raised.
        non_bridging_databases (frozenset[str]): External databases whose ids are names of a clone or region rather
            than of a gene; they are never used as a bridge between two genes.
    """

    search_budget_states: int = 200_000
    search_budget_seconds: float = 30.0
    non_bridging_databases: frozenset = frozenset(
        {
            "Clone_based_ensembl_gene",
            "Clone_based_vega_gene",
            "Clone_based_ensembl_transcript",
            "Clone_based_vega_transcript",
        }
    )

    def __init__(self, db_manager: "DatabaseManager", **kwargs):
        """Create a ``Track`` and load (or build) its graph.

        Args:
            db_manager (DatabaseManager): The manager of the organism's tables; it is handed to
                :py:class:`idtrack.GraphMaker`.
            kwargs: Keyword arguments for :py:meth:`idtrack.GraphMaker.get_graph`: ``narrow``,
                ``create_even_if_exist``, ``save_after_calculation``, ``form_list`` and ``narrow_external``.
        """
        from idtrack._graph_maker import GraphMaker  # needed only to load or build; conversions do not use it

        self.log = logging.getLogger(__name__)
        # ``None`` for a ``Track`` made by :py:meth:`from_graph`.
        self.db_manager: DatabaseManager | None = db_manager
        self.graph = GraphMaker(db_manager).get_graph(**kwargs)

    @classmethod
    def from_graph(cls, graph: "TheGraph") -> "Track":
        """Create a ``Track`` on a graph that is already in memory, without a database manager.

        Args:
            graph (TheGraph): A graph built by :py:class:`idtrack.GraphMaker` (or loaded from a file).

        Returns:
            Track: A resolver that answers from that graph; no network and no tables are needed.
        """
        track = cls.__new__(cls)
        track.log = logging.getLogger(__name__)
        track.db_manager = None
        track.graph = graph
        return track

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

    # Arguments

    def _check_release(self, release: Any, name: str) -> int:
        """Turn a release argument into a plain ``int`` and make sure the graph has that release.

        Args:
            release (Any): The value given by the caller (``int`` or an integer type such as ``numpy.int64``).
            name (str): The parameter's name, for the error message.

        Returns:
            int: The release as ``int``.

        Raises:
            TypeError: If the value is not an integer (``bool``, ``float`` and ``str`` are rejected).
            ValueError: If the graph does not contain that release.
        """
        if isinstance(release, bool):
            raise TypeError(f"`{name}` must be an integer Ensembl release, not {release!r}.")
        try:
            value = operator.index(release)
        except TypeError:
            raise TypeError(
                f"`{name}` must be an integer Ensembl release, not {release!r} of type {type(release).__name__}."
            ) from None
        reader = self._reader
        if value not in reader.release_set:
            raise ValueError(
                f"`{name}`={value} is not a release of this graph, which covers the Ensembl releases "
                f"{reader.first_release} to {reader.last_release}."
            )
        return value

    def final_databases(self, to_release: int | None = None) -> list[str]:
        """Return the databases a conversion to a release can end in.

        These are the external databases of the genome assembly that is primary at the release, and the two forms
        of gene ids (``ensembl_gene`` with version, ``base_ensembl_gene`` without).

        Args:
            to_release (int | None): The target release. ``None``: the last release of the graph.

        Returns:
            list[str]: The accepted values of ``final_database`` of :py:meth:`convert`, sorted.

        Raises:
            TypeError: If the release is not an integer.
            ValueError: If the graph does not contain that release.
        """
        reader = self._reader
        target = reader.last_release if to_release is None else self._check_release(to_release, "to_release")
        allowed = set(self.graph.available_external_databases_assembly.get(reader.primary_assembly[target], set()))
        allowed.update({DB.nts_ensembl[DB.backbone_form], DB.nts_base_ensembl[DB.backbone_form]})
        return sorted(allowed)

    def _check_final_database(self, final_database: str | None, to_release: int) -> str | None:
        """Make sure the target database exists on the assembly of the target release.

        Args:
            final_database (str | None): The database asked for, or ``None``.
            to_release (int): The target release.

        Returns:
            str | None: ``None`` when the gene itself is the target, else the database name.

        Raises:
            ValueError: If the database is not one of :py:meth:`final_databases` of the target release.
        """
        if final_database is None or final_database == DB.nts_ensembl[DB.backbone_form]:
            return None
        allowed = self.final_databases(to_release)
        if final_database not in allowed:
            raise ValueError(
                f"`final_database`={final_database!r} is not a database of this graph at Ensembl release "
                f"{to_release} (genome assembly {self._reader.primary_assembly[to_release]}). Available: {allowed}."
            )
        return final_database

    # Starting release and entry onto the backbone

    def _active_releases(self, search: _Search, node: str, database: str | None = None) -> tuple[frozenset, bool]:
        """Return the releases at which a node is active.

        A versioned gene is active in its history ranges, a gene id without version where one of its own versions is,
        any other node where one of its edges towards the backbone holds on the primary line. A node that never is
        on the primary line (an id of an archive assembly only) is active wherever any of its edges holds.

        Args:
            search (_Search): Scratch memory of the conversion.
            node (str): The node.
            database (str | None): For an external id, the database it is meant as: only edges of that database
                count.

        Returns:
            tuple[frozenset, bool]: The releases, and ``True`` when they had to be taken from all assemblies.
        """
        reader = self._reader
        node_type = reader.node_type(node)

        def of_gene(gene: str) -> set:
            return {
                r for a, b in search.ranges(gene) for r in range(int(a), int(min(b, reader.last_release)) + 1)
            } & reader.release_set

        if node_type == reader.backbone_type:
            return frozenset(of_gene(node)), False

        edges = search.xrefs(node, False)
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
        self, search: _Search, node: str, release: int, any_assembly: bool, database: str | None = None
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

        def walk(current: str, steps: tuple, depth: int, databases: frozenset, entered: bool) -> None:
            for other, items in search.xrefs(current, False):
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
                        attached.setdefault(other, set()).update(databases)
                        if entered:
                            keep(other, step)
                elif depth < 3 and reader.node_type(other) != DB.nts_external:
                    walk(other, step, depth + 1, databases, entered)

        walk(node, (), 1, frozenset(), False)
        return sorted(found.items()), {gene: attached[gene] for gene in found}

    def _start(
        self, search: _Search, node: str, from_release: int | None, to_release: int, database: str | None = None
    ) -> tuple[int, list, dict[str, set], bool]:
        """Decide the release a query is read at and the genes it stands for there.

        Args:
            search (_Search): Scratch memory of the conversion.
            node (str): The query node.
            from_release (int | None): The release given by the caller, or ``None``.
            to_release (int): The target release.
            database (str | None): For an external id, the database it is meant as.

        Returns:
            tuple[int, list, dict[str, set], bool]: The starting release, the entries of :py:meth:`_entries` and
            the databases that attach the node to each of them, and ``True`` when the node exists in an archive
            assembly only.

        Raises:
            InactiveAtReleaseError: If ``from_release`` is given and the node is not active at it.
        """
        active, any_assembly = self._active_releases(search, node, database)
        start = self._starting_release(node, active, from_release, to_release)
        if start is None:
            return to_release, [], {}, any_assembly
        entries, attached = self._entries(search, node, start, any_assembly, database)
        return start, entries, attached, any_assembly

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
        its two sides hold on the primary line are returned, so that the caller can place the bridge in time.

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
        skip = self.non_bridging_databases
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
            if first_type == DB.nts_external:
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
                    if second_type == DB.nts_external:
                        near_2 = near_1 & reader.primary_releases(second_items, skip)
                        through_external(second, near_2, ((gene, first), (first, second)), True)
                    else:  # a translation of the transcript
                        near_2 = near_1 & reader.primary_releases(second_items)
                        if not near_2:
                            continue
                        for third, third_items in search.xrefs(second, True):
                            if reader.node_type(third) == DB.nts_external:
                                near_3 = near_2 & reader.primary_releases(third_items, skip)
                                through_external(third, near_3, ((gene, first), (first, second), (second, third)), True)
        chains.sort()
        search._chains[gene] = chains
        return chains

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
        self, search: _Search, entries: list, start: int, to_release: int, go_external: bool
    ) -> dict[str, dict]:
        """Find the genes at the target release that the entry genes lead to, each with its best path.

        Args:
            search (_Search): Scratch memory of the conversion.
            entries (list): ``(gene, steps)`` pairs of :py:meth:`_entries`.
            start (int): The starting release.
            to_release (int): The target release.
            go_external (bool): Whether bridges through external ids may be used when the history leads nowhere.

        Returns:
            dict[str, dict]: Per destination gene: ``entry_gene`` (the entry gene its path starts from),
            ``external_jump``, ``external_gap``, ``external_step``, ``score``, ``ensembl_step`` and ``the_path``.
        """
        reader = self._reader
        reverse = start > to_release
        jump_limit = int(DB.external_search_settings["jump_limit"]) if go_external else 0

        # cost = (bridges, bridge gap, external steps, -score, history steps); state = (gene, release, bridges, beamed)
        heap: list = []
        settled: dict[tuple, tuple] = {}
        destinations: dict[str, tuple] = {}
        counter = 0

        def push(cost: tuple, state: tuple, parent: tuple | None, steps: tuple) -> None:
            nonlocal counter
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
            if gene not in destinations or candidate[0] < destinations[gene][0]:
                destinations[gene] = candidate

        def expand(cost: tuple, state: tuple, level_states: list) -> None:
            gene, release, bridges, beamed = state
            found = search.range_at(gene, release)
            if found is None:
                return
            first, last = found
            n_bridges, gap, external_steps, negative_score, history_steps = cost

            if (not reverse and to_release <= last) or (reverse and to_release >= first):
                if not reverse:
                    factor, loop_steps = loops_passed(gene, release, to_release)
                else:
                    factor, loop_steps = loops_passed(gene, to_release, release)
                arrive(
                    gene,
                    (n_bridges, gap, external_steps, negative_score * factor, history_steps + len(loop_steps)),
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
                push(
                    (
                        n_bridges,
                        gap,
                        external_steps,
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
            n_bridges, gap, external_steps, negative_score, history_steps = cost
            # The gene is one entity throughout its active range: evidence from any release of that range counts.
            if not reverse:
                low, high = int(first), int(min(last, to_release))
            else:
                low, high = int(max(first, to_release)), int(min(last, reader.last_release))
            for other, near, far, near_edges, far_edges in self._bridges(search, gene):
                placed = self._place_bridge(near, far, low, high, to_release, reverse)
                if placed is None:
                    continue
                left, entered = placed
                steps = tuple((u, v, 0, left) for u, v in near_edges) + tuple((u, v, 0, entered) for u, v in far_edges)
                push(
                    (
                        n_bridges + 1,
                        gap + abs(entered - left),
                        external_steps + len(near_edges) + len(far_edges),
                        negative_score,
                        history_steps,
                    ),
                    (other, entered, bridges + 1, True),
                    state,
                    steps,
                )

        for gene, steps in entries:
            push((0, 0, len(steps), -1.0, 0), (gene, start, 0, False), None, steps)

        level_states: list = []
        run(level_states)
        level = 0
        while not destinations and level < jump_limit:
            sources, level_states = level_states, []
            for cost, state in sources:
                bridge_out(cost, state)
            run(level_states)
            level += 1

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
                "external_jump": cost[0],
                "external_gap": cost[1],
                "external_step": cost[2],
                "score": -cost[3],
                "ensembl_step": cost[4],
                "the_path": tuple(path) if path else ((None, gene, None),),
            }
        return result

    def get_possible_paths(
        self, from_id: str, from_release: int, to_release: int, reverse: bool, go_external: bool = True
    ) -> tuple:
        """Return the path to every gene that an identifier of one release leads to at another release.

        This is the search of :py:meth:`convert` without the final conversion and without ranking: one path per
        destination gene, the best one.

        Args:
            from_id (str): A node of the graph.
            from_release (int): The release the identifier is read at; it must be active there.
            to_release (int): The target release.
            reverse (bool): The direction of the walk, ``True`` when ``from_release`` is later than ``to_release``.
            go_external (bool): Whether bridges through external ids may be used when the history leads nowhere.

        Returns:
            tuple: One path per destination gene, ordered by gene. A path is a tuple of steps: ``(source, target,
            key)`` for a history edge, ``(source, target, 0, release)`` for a cross-reference edge at a release, and
            the single step ``(None, gene, None)`` for a gene that is its own counterpart.

        Raises:
            ValueError: If ``reverse`` contradicts the two releases.
        """
        start_release = self._check_release(from_release, "from_release")
        target = self._check_release(to_release, "to_release")
        if start_release != target and bool(reverse) != (start_release > target):
            raise ValueError(
                f"`reverse`={reverse!r} contradicts the releases: from {start_release} to {target} is "
                f"{'backward' if start_release > target else 'forward'} in time."
            )
        search = _Search(self._reader, from_id, self.search_budget_states, self.search_budget_seconds)
        start, entries, _, _ = self._start(search, from_id, start_release, target)
        found = self._search_paths(search, entries, start, target, go_external)
        return tuple(found[gene]["the_path"] for gene in sorted(found))

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
                if first_type == DB.nts_external:
                    if strict and reader.holds(items, release, database=database):
                        keep(first, ((gene, first, 0, release),))
                elif first_type != reader.base_type and inner(items):
                    step_1 = (mark(gene, first),)
                    for second, second_items in search.xrefs(first, True):
                        if reader.node_type(second) == DB.nts_external:
                            if reader.holds(second_items, release, database=database):
                                keep(second, step_1 + ((first, second, 0, release),))
                        elif inner(second_items):
                            step_2 = step_1 + (mark(first, second),)
                            for third, third_items in search.xrefs(second, True):
                                if reader.node_type(third) == DB.nts_external and reader.holds(
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
            final_database (str): An external database of the genome assembly of the release, or
                ``base_ensembl_gene``.
            release (int | None): The Ensembl release. ``None``: the last release of the graph.

        Returns:
            tuple[list[str], int]: The ids as the database wrote them at the release, in ascending order, and the
            confidence as in :py:meth:`convert` (0: attached at the release; 1: attached at the release to a
            transcript or translation the gene had at another release). An empty list when the gene has no id in
            the database.

        Raises:
            ValueError: If the release or the database is not one of the graph, or ``gene`` is not a gene id with
                version that is active at the release.
        """
        reader = self._reader
        target = reader.last_release if release is None else self._check_release(release, "release")
        database = self._check_final_database(final_database, target)
        search = _Search(reader, gene, self.search_budget_states, self.search_budget_seconds)
        if (
            not reader.has_node(gene)
            or not reader.is_backbone(gene)
            or not reader.is_real_gene(gene)
            or search.range_at(gene, target) is None
        ):
            raise ValueError(f"{gene!r} is not a gene id with version that is active at Ensembl release {target}.")
        if database is None:
            return [gene], 0
        targets, confidence = self._final_targets(search, gene, database, target)
        if database == reader.base_type:
            return sorted(targets), confidence
        return sorted(self.graph.spelling(target_id, database, target) for target_id in targets), confidence

    def calculate_node_scores(self, the_id: str, ens_release: int) -> list[int]:
        """Count how richly a gene is annotated at a release.

        The counts are the last criterion of :py:meth:`convert` with ``prioritize_to_one_filter``: between the
        successors of one gene that are equal in every other respect, the one with more annotation is preferred
        (never between the genes that an identifier stands for: those are a tie). Only the gene's own
        transcripts and translations of that release and the external ids attached to them or to the gene are
        counted; hyper-connective external ids are left out.

        Args:
            the_id (str): A gene node.
            ens_release (int): The release at which the annotation is counted.

        Returns:
            list[int]: ``[-external ids, -transcripts, -translations]`` (negative, so that smaller is better).
        """
        search = _Search(self._reader, the_id, self.search_budget_states, self.search_budget_seconds)
        return self._node_scores(search, the_id, ens_release)

    def _node_scores(self, search: _Search, gene: str, release: int) -> list[int]:
        """Worker of :py:meth:`calculate_node_scores` that shares the scratch memory of a conversion.

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
        ensembl_relations = frozenset(DB.nts_ensembl.values()) | frozenset(DB.nts_base_ensembl.values())

        def count(node: str, depth: int) -> None:
            for other, items in search.xrefs(node, True):
                if not reader.holds(items, release):
                    continue
                other_type = reader.node_type(other)
                if other_type == DB.nts_external:
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

    def convert(
        self,
        from_id: str,
        from_release: SupportsIndex | None = None,
        to_release: SupportsIndex | None = None,
        final_database: str | None = None,
        go_external: bool = True,
        prioritize_to_one_filter: bool = False,
        return_path: bool = False,
        return_ensembl_alternative: bool = True,
        from_database: str | None = None,
        budget_states: int | None = None,
        budget_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        """Convert an identifier to the target release and, if asked, to a target database.

        Starting release: with ``from_release`` the identifier is read at that release and must be active there.
        Without it, the identifier is read at ``to_release`` if it is active there (the answer is then what the
        identifier means at the target release), and at the latest release at which it is active otherwise. A gene
        id without version stands for its version of the starting release. The rule is the same for Ensembl ids and
        for ids of external databases.

        Search: from the genes the identifier stands for, Ensembl's stable-id history is followed to the target
        release. Only when that leads to no gene, and ``go_external`` allows it, the search passes through an
        external id or transcript that two genes carried at the same time or one after the other, with as few such
        bridges as possible. ``external_jump`` in the result tells how many were used (0: history only).

        Final conversion: with ``final_database`` each gene found is converted to the ids of that database attached
        to it at the target release. One result has one final database: genes that have such an id are returned
        with it; only if none has one, the genes themselves are returned (``no_target`` is then ``True``), or
        nothing with ``return_ensembl_alternative=False``.

        Order: the returned dict is ordered, best first. The genes of the best answer come first, in the order in
        which ``prioritize_to_one_filter`` returns them alone (see "The best answer" below), so the first gene of
        every answer is the first gene of the best answer. The other genes follow by their evidence: the
        confidence of the final conversion, the number of bridges, the release gap of the bridges, the support of
        the query on the entry gene (the number of databases that attach the identifier to that gene at the
        starting release, to the gene itself or to one of its transcripts or translations), the path score, being
        the query itself, the number of cross-reference and history steps, and finally the ids themselves.

        Copies beside the reference sequence: Ensembl also annotates genes on alternative loci (haplotypes) and
        assembly patches, and LRG records (``LRG_<number>``, fixed reference sequences of single genes), each with
        a stable id of its own and with the names and accessions of the gene it is a copy of. Which genes these
        are is Ensembl's statement, kept in the graph
        (:py:meth:`idtrack.TheGraph.is_non_reference` at the target release); nothing is read off an
        id. When the search reaches genes of the reference sequence and such copies, the copies are not part of
        the answer, in any strategy; they are listed in ``alternative_locus_genes``. When it reaches copies only
        (the identifier is such a gene or an LRG record, or a name that only they carry), they are the answer.

        Identity: a gene id that exists at the target release is its own answer. The queried gene id's own version
        at the target release is never taken out as a copy (a gene that lay on a patch of the assembly of the
        target release is still that gene), it comes first in the result, and it is the best answer whatever else
        its history reaches.

        The best answer (``prioritize_to_one_filter``): the queried gene id's own version at the target release when
        the search reached it; otherwise the entry genes with the best confidence, bridges, gap and
        support each contribute their best successor (by score, then being the query itself, then, between
        successors of the one gene that are still equal, the richest annotation); the mapping score and the
        annotation are compared among the successors of one entry gene only, because neither says which of two
        genes an identifier stands for. If more than one gene or target remains (an identifier that several genes
        carry with equal support, a gene that Ensembl split into equal parts, a gene with several ids in the target
        database), the answer is a tie: all of them are kept and ``tie`` is ``True``. The pairs of a tie are equal
        in the evidence between entry genes; they are ordered by the mapping score of each to its own entry gene
        (the higher first), then being the query itself, then the number of steps, then gene id and target id.
        The assembly cannot differ between candidates, because every candidate lies on the primary assembly of
        the target release.

        Args:
            from_id (str): The identifier. The graph says which node the string means
                (:py:meth:`idtrack.TheGraph.resolve_name` for ``from_release``, without one for the last
                release of the graph, and for ``from_database``): a node that exists at that release (as an id of
                that database, on the assembly that serves the release) before one that does not, whatever the
                spelling, so a string that is itself the name of a node is not always read as that node. Among
                those the string is matched exactly before any other way, and "exactly" means as a database wrote
                the identifier at the release it is read at (the starting release of the rule above), so a name
                typed as it was written then is that name, not another gene's synonym spelt the same. A
                transcript or translation id without version stands for its version of the starting release.
            from_release (SupportsIndex | None): The Ensembl release the identifier comes from, an ``int`` or another
                integer type (``numpy.int64``). ``None``: see above.
            to_release (SupportsIndex | None): The target Ensembl release (as ``from_release``). ``None``: the last
                release of the graph.
            final_database (str | None): The database to convert the genes into. ``None`` or ``ensembl_gene``:
                the genes themselves; ``base_ensembl_gene``: gene ids without version; else an external database of
                the genome assembly of the target release.
            go_external (bool): Allow bridges through external ids when the history leads to no gene.
            prioritize_to_one_filter (bool): Keep only the best gene and target.
            return_path (bool): Include the path to every gene and target in the result.
            return_ensembl_alternative (bool): When no gene found has an id in ``final_database``, return the genes
                themselves instead of nothing.
            from_database (str | None): The external database the identifier is meant as. The same string can be
                an id of several databases (a number that is an NCBI gene id and an id of another source); with
                this argument the identifier is read as an id of that database: only the genes that this database
                attaches it to are entered, and an identifier that is not an id of that database (an Ensembl id, an
                id of other databases only) has no conversion. The database says which genes are meant, not which
                evidence counts: between the genes entered, the one that more databases attach the identifier to
                is still the better supported (``entry_support``). ``None``: all databases.
            budget_states (int | None): Largest number of search states this call may expand. ``None``:
                :py:attr:`search_budget_states`.
            budget_seconds (float | None): Longest time this call may take. ``None``:
                :py:attr:`search_budget_seconds`.

        Returns:
            dict | None: ``None`` when the identifier is unknown or leads to nothing. Otherwise, per gene at the
            target release (keys in the order described above):

            * ``from_id`` (the node the identifier was read as), ``from_release`` (the starting release),
              ``from_release_inferred``, ``to_release``,
              ``from_databases`` (for an external id: the databases through which it was read at the starting
              release, sorted: every database that attaches it there to one of the genes entered, to the gene
              itself or to a transcript or translation of the gene; with ``from_database`` that database; more
              than one means the string is an id of several databases there; for an Ensembl id that is not a
              gene id with version, the Ensembl relation it is read through, e.g. ``base_ensembl_gene``);
            * ``entry_gene`` (the gene the identifier was read as at the starting release and from which this gene
              was reached), ``entry_support`` (the number of databases that attach the identifier to it there, to
              the gene itself or to one of its transcripts or translations; 1 for Ensembl identifiers);
            * ``external_jump`` (bridges used), ``external_gap`` (releases between leaving a gene and entering the
              next, summed over the bridges), ``external_step`` (cross-reference edges on the path),
              ``ensembl_step`` (history edges on the path);
            * ``score``: the product of Ensembl's mapping scores along the history edges of the path (1.0 when the
              path has no scored event);
            * ``tie``: ``True`` when the best answer of this conversion is shared by several genes or targets (the
              same value in every entry of a result);
            * ``alternative_locus_genes``: the genes on alternative loci and patches and the LRG records that the
              search reached beside genes of the reference sequence, in ascending order; they are not part of the
              result (the same list in every entry, empty when there is none);
            * ``the_path`` (with ``return_path``): a tuple of steps, ``(source, target, key)`` for a history edge,
              ``(source, target, 0, release)`` for a cross-reference edge used at a release, and
              ``((None, gene, None),)`` for a gene that is its own counterpart;
            * ``final_conversion``: ``final_database``, ``final_conversion_confidence`` (0: attached at the target
              release; 1: attached at the target release to a transcript or translation the gene had at another
              release; ``None``: no gene has an id in the database and the genes themselves are returned),
              ``no_target`` (``True`` in that last case), ``genes_without_target`` (the genes the search reached
              that have no id in the database while other genes have one; they are not part of the result; the same
              list in every entry), and
              ``final_elements``: per target id (an id of an external database as that database wrote it at the
              target release, :py:meth:`idtrack.TheGraph.spelling`) a dict with ``the_path`` (with ``return_path``) and,
              with ``prioritize_to_one_filter``, ``filter_scores``.

        Raises:
            TypeError: If ``from_id`` is not a string or a release is not an integer.
            ValueError: If a release is outside the graph, ``final_database`` is not available at the target
                release, ``from_database`` is not an external database of the graph, or a budget is not positive.
            InactiveAtReleaseError: If ``from_release`` is given and the identifier is not active at that release.
            UnknownVersionError: If the identifier is an Ensembl id with a version that the graph does not hold and
                that cannot be newer than the graph (an id with a version higher than all the graph holds, active
                at the graph's last release, is read as its latest version instead).
            SearchBudgetExceededError: If the search needs more states or time than the budget of a call.
        """
        if not isinstance(from_id, str):
            raise TypeError(f"`from_id` must be a string, not {from_id!r} of type {type(from_id).__name__}.")
        reader = self._reader
        target = reader.last_release if to_release is None else self._check_release(to_release, "to_release")
        given = None if from_release is None else self._check_release(from_release, "from_release")
        database = self._check_final_database(final_database, target)

        if from_database is not None and from_database not in self.graph.available_external_databases:
            raise ValueError(
                f"`from_database`={from_database!r} is not an external database of this graph. Available: "
                f"{sorted(self.graph.available_external_databases)}."
            )

        node: str = from_id
        versions: tuple = ()
        resolved, _ = reader.resolve_name(from_id, given, from_database, target)
        if resolved is not None:
            node = resolved
        else:
            held = self.graph.known_versions(from_id)
            if held:
                raise UnknownVersionError(from_id, held)
            # A transcript or translation id without version has no node of its own: it stands for a version.
            versions = self.graph.versions_of(from_id)
            if not versions:
                return None

        if from_database is not None and (versions or reader.node_type(node) != DB.nts_external):
            return None  # an Ensembl identifier is not an id of an external database

        states = self.search_budget_states if budget_states is None else budget_states
        seconds = self.search_budget_seconds if budget_seconds is None else budget_seconds
        if not states > 0 or not seconds > 0:
            raise ValueError(f"The budget of a conversion must be positive (states: {states!r}, seconds: {seconds!r}).")

        search = _Search(reader, node, states, seconds)
        read_at = given
        if versions:
            chosen = self._version_at_start(search, from_id, versions, given, target)
            if chosen is None:
                return None
            node, read_at = chosen
            search.identifier = node
        start, entries, attached, _ = self._start(search, node, read_at, target, from_database)
        if not entries:
            return None
        # Through which databases the query belongs to each entry gene at the starting release: every database that
        # attaches it to the gene itself or to a transcript or translation of the gene. A named database says which
        # genes are entered (`_start`); the support of an entry gene is every database that attaches the identifier
        # to it, named or not: what the other databases say about a gene is evidence all the same.
        read_through = set().union(*attached.values()) if attached else set()
        from_databases = sorted(read_through if from_database is None else read_through & {from_database})
        support = {gene: max(1, len(attached.get(gene, ()))) for gene, _ in entries}
        found = self._search_paths(search, entries, start, target, go_external)

        if not found:
            return None

        # Copies on alternative loci and patches, and LRG records, are not part of an answer that holds a gene of
        # the reference sequence. The graph's mark says which genes these are. The queried gene id's own version at
        # the target release is never taken out: an id that exists at the target release is its own answer.
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

        # Final conversion: one database and one confidence for the whole result.
        gene_form = DB.nts_ensembl[DB.backbone_form]
        finals: dict[str, tuple] = {}
        without_target: list[str] = []
        if database is None:
            finals = {gene: ({gene: ()}, 0) for gene in found}
            final_name, no_target = gene_form, False
        else:
            converted = {gene: self._final_targets(search, gene, database, target) for gene in found}
            with_target = {gene: value for gene, value in converted.items() if value[0]}
            if with_target:
                best_confidence = min(value[1] for value in with_target.values())
                finals = {gene: value for gene, value in with_target.items() if value[1] == best_confidence}
                final_name, no_target = database, False
                without_target = sorted(gene for gene, value in converted.items() if not value[0])
            elif return_ensembl_alternative:
                finals = {gene: ({gene: ()}, None) for gene in found}
                final_name, no_target = gene_form, True
            else:
                return None

        # An external id is reported as its database wrote it at the target release: the node name is the graph's one
        # spelling of the id, and a database may write it with other letter case, also from release to release.
        spell_as = None if no_target or database is None or database == reader.base_type else database

        query_base = node if reader.node_type(node) == reader.base_type else None

        def is_query(gene: str, target_id: str) -> bool:
            return target_id == node or (query_base is not None and reader.base_id(gene) == query_base)

        # A pair is ranked by: being the queried gene id's own version at the target release; then how the entry gene
        # compares with the other entry genes (final confidence, bridges, gap of the bridges, support of the query on
        # the entry gene), then how the gene compares with the other successors of the same entry gene (mapping
        # score, being the query itself), then steps and ids.
        pairs = []
        for gene, (targets, confidence) in finals.items():
            record = found[gene]
            lineage = (confidence, record["external_jump"], record["external_gap"], -support[record["entry_gene"]])
            for target_id in targets:
                within = (-round(record["score"], 12), 0 if is_query(gene, target_id) else 1)
                rest = (record["external_step"], record["ensembl_step"], gene, target_id)
                order = (0 if gene in own else 1,) + lineage + within + rest
                pairs.append((order, lineage, within, gene, target_id))
        pairs.sort()

        # The best answer. Between entry genes only the lineage evidence counts: the entry genes that are equal in it
        # all belong to the answer. The mapping score compares the successors of one gene (it says how much of that
        # gene a successor keeps, not which of two genes a name stands for), then being the query itself, and between
        # successors that are still equal the richness of annotation decides, as the last resort. What remains is the
        # answer; more than one pair is a tie.
        best_pairs: list[tuple] = []
        by_entry: dict[str, list] = {}
        candidates = [pair for pair in pairs if pair[3] in own] or pairs  # the id's own version is the best answer
        for pair in candidates:
            if pair[1] == candidates[0][1]:
                by_entry.setdefault(found[pair[3]]["entry_gene"], []).append(pair)
        node_scores: dict[str, list] = {}
        for entry_gene in sorted(by_entry):
            successors = [pair for pair in by_entry[entry_gene] if pair[2] == by_entry[entry_gene][0][2]]
            equal_genes = sorted({pair[3] for pair in successors})
            if len(equal_genes) > 1:
                for gene in equal_genes:
                    node_scores[gene] = self._node_scores(search, gene, target)
                richest = min(node_scores[gene] for gene in equal_genes)
                successors = [pair for pair in successors if node_scores[pair[3]] == richest]
            best_pairs.extend(successors)
        best_pairs.sort()
        tie = len(best_pairs) > 1

        filter_scores: dict[tuple, dict] = {}
        if prioritize_to_one_filter:
            pairs = best_pairs
            for _, lineage, within, gene, target_id in pairs:
                filter_scores[(gene, target_id)] = {
                    "initial_filter": [lineage[0], lineage[1], lineage[2], -lineage[3], -within[0]],
                    "same_as_input_filter": within[1] == 0,
                    "node_importance_filter": node_scores.get(gene),
                }
        else:
            # Every answer, best first by the order that chose the best answer: its pairs come first, as they are
            # returned alone, and the others follow by their evidence. The evidence alone does not put the best
            # answer first: the mapping score and the annotation are compared among the successors of one entry
            # gene only.
            leading = set(best_pairs)
            pairs = best_pairs + [pair for pair in pairs if pair not in leading]

        result: dict[str, dict] = {}
        for _, _, _, gene, target_id in pairs:
            if gene not in result:
                record = found[gene]
                entry: dict[str, Any] = {
                    "from_id": node,
                    "from_release": start,
                    "from_release_inferred": given is None,
                    "from_databases": from_databases,
                    "entry_gene": record["entry_gene"],
                    "entry_support": support[record["entry_gene"]],
                    "to_release": target,
                    "external_jump": record["external_jump"],
                    "external_gap": record["external_gap"],
                    "external_step": record["external_step"],
                    "ensembl_step": record["ensembl_step"],
                    "score": record["score"],
                    "tie": tie,
                    "alternative_locus_genes": list(alternative_locus_genes),
                    "final_conversion": {
                        "final_database": final_name,
                        "final_conversion_confidence": finals[gene][1],
                        "no_target": no_target,
                        "genes_without_target": list(without_target),
                        "final_elements": {},
                    },
                }
                if return_path:
                    entry["the_path"] = record["the_path"]
                result[gene] = entry
            element: dict[str, Any] = {}
            if return_path:
                element["the_path"] = finals[gene][0][target_id]
            if prioritize_to_one_filter:
                element["filter_scores"] = filter_scores[(gene, target_id)]
            reported = target_id if spell_as is None else self.graph.spelling(target_id, spell_as, target)
            result[gene]["final_conversion"]["final_elements"][reported] = element
        return result

    # Where a list of identifiers comes from

    def identify_source(self, dataset_ids: list[str], mode: str) -> list[tuple]:
        """Count in which assemblies, databases and releases the identifiers of a data set occur.

        Only the edges of the given identifiers are read; nothing is computed for the rest of the graph.

        Args:
            dataset_ids (list[str]): Nodes of the graph.
            mode (str): What to count: ``complete`` for ``(database, assembly, release)``, ``ensembl_release``,
                ``assembly`` or ``assembly_ensembl_release``.

        Returns:
            list[tuple]: ``(origin, count)`` pairs, the most frequent first. Origins with the same count are tied;
            among them the later release, then the higher-priority assembly, then the database name comes first, so
            the order does not depend on the input order or on the process.

        Raises:
            ValueError: If ``mode`` is not one of the four values.
        """
        pick = {
            "complete": lambda trio: trio,
            "ensembl_release": lambda trio: trio[2],
            "assembly": lambda trio: trio[1],
            "assembly_ensembl_release": lambda trio: trio[1:],
        }
        if mode not in pick:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected one of: 'complete', "
                "'ensembl_release', 'assembly', 'assembly_ensembl_release'."
            )
        reader = self._reader
        counts: Counter = Counter()
        for identifier in dataset_ids:
            counts.update({pick[mode](trio) for trio in reader.trios(identifier)})

        priority = reader.priority
        worst = max(priority.values(), default=0) + 1

        def tie_order(origin: Any) -> tuple:
            if mode == "complete":
                return (-origin[2], priority.get(origin[1], worst), origin[1], origin[0])
            if mode == "ensembl_release":
                return (-origin,)
            if mode == "assembly":
                return (priority.get(origin, worst), origin)
            return (-origin[1], priority.get(origin[0], worst), origin[0])

        return sorted(counts.items(), key=lambda item: (-item[1], tie_order(item[0])))


def _weight(weight: float) -> float:
    """Return the mapping score of a history edge, 1.0 where Ensembl gives none.

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
