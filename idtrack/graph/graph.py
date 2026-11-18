"""The identifier graph of one organism (:py:class:`TheGraph`): typed accessors over its store, and its file."""

import functools
import itertools
import json
import logging
import math
import os
import zipfile
from collections.abc import Iterator, Mapping
from typing import Any

import networkx as nx
import numpy as np

from idtrack import ids, organisms
from idtrack._atomic import replacing
from idtrack.errors import (
    GraphFileError,
    GraphMismatchError,
    InputError,
    IntegrityError,
    InvalidArgumentError,
    UnknownIdentifierError,
)
from idtrack.graph.names import NameResolution
from idtrack.graph.store import (
    _NAME_SEPARATOR,
    _STORED_GLOBALS,
    _connection_or_none,
    _from_json_globals,
    _from_json_graph,
    _masks_to_words,
    _Store,
    _to_json,
    _words_to_masks,
)
from idtrack.graph.types import (
    _ASSEMBLY_TYPE,
    _INF,
    _KIND_OTHER,
    _KIND_XREF,
    _NO_RELEASES,
    Connection,
    HistoryEdge,
    ReleaseSet,
    _assembly_label,
    _is_assembly_gene,
    _is_release,
    serving_line,
)
from idtrack.settings import GraphSettings, check_graph_settings

#: Version of the graph file layout and of the build rules behind it. It is raised whenever a graph built by an
#: earlier version of the code must not be loaded any more (a change of the file layout, of what an edge means, of
#: how names are chosen or of how the history is validated).
GRAPH_FORMAT_VERSION = 2

#: The steps of name resolution that are tried on names and then on synonyms, in this order.
_NETWORKX_VIEWS = (
    "nodes",
    "adj",
    "succ",
    "pred",
    "edges",
    "out_edges",
    "in_edges",
    "degree",
    "out_degree",
    "in_degree",
)


# What a user can do about a graph file that cannot be used, named by what the public calls offer.
_REBUILD_HINT = (
    "To build the graph again: call `API.build(..., rebuild=True)` or "
    "`GraphMaker.get_graph(create_even_if_exist=True)`, which replace the file."
)


class _NodeMap(Mapping):
    """The node dictionary of a frozen graph: ``{name: attributes}``, computed per node."""

    def __init__(self, store: _Store, attributes):
        self._store = store
        self._attributes = attributes

    def __getitem__(self, name):
        return self._attributes(name)

    def __contains__(self, name) -> bool:
        try:
            return name in self._store.index
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.names)

    def __len__(self) -> int:
        return self._store.n


class _AdjacencyMap(Mapping):
    """The adjacency dictionary of a frozen graph: ``{node: {neighbour: {key: attributes}}}``, computed per node."""

    def __init__(self, store: _Store, adjacency):
        self._store = store
        self._adjacency = adjacency

    def __getitem__(self, name):
        return self._adjacency(name)

    def __contains__(self, name) -> bool:
        try:
            return name in self._store.index
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.names)

    def __len__(self) -> int:
        return self._store.n


class TheGraph(NameResolution, nx.MultiDiGraph):
    """The identifier graph of one organism: Ensembl gene history plus cross-references, with typed accessors.

    Nodes are identifiers: versioned Ensembl genes (the *backbone*, with ``<id>.Void`` and ``<id>.Retired`` marking
    births and retirements), versionless base gene ids, transcripts, translations and identifiers of external
    databases. Two families of edges exist. *History edges* connect backbone genes and carry the release pair of the
    event and Ensembl's mapping score. *Cross-reference edges* (external id to Ensembl id, base id to versioned gene,
    transcript to gene, translation to transcript) carry a :py:class:`Connection`: the releases in which each
    database links the two nodes on each assembly.

    The class is built by :py:class:`idtrack.GraphMaker` and read by :py:class:`idtrack.engine.Track`. A graph
    has two states. While it is being made it is an ordinary :py:class:`networkx.MultiDiGraph`. :py:meth:`freeze`
    (called by the first use of any accessor) turns the content into compact arrays: node names and types, edge
    arrays, and a small table of distinct connections whose release sets are bit masks. A frozen graph cannot be
    changed, is what :py:meth:`save` writes and :py:meth:`load` reads back in seconds without network access, and
    answers every question per node on demand behind bounded caches, so that nothing derived from the graph can be
    stale and no table over all nodes is kept in memory.

    The networkx reading interface (``g.nodes[n]``, ``g.neighbors(n)``, ``g.get_edge_data(u, v)``, ``g[u][v][k]``)
    keeps working on a frozen graph; the accessors of this class (:py:meth:`history_edges`, :py:meth:`xref_edges`,
    :py:meth:`active_ranges`, :py:meth:`combined`, :py:meth:`resolve_name` ...) are the faster way to read it.
    """

    #: How the graph caches what it computes per node, and its hyper-connectivity threshold (per graph: a graph
    #: loaded with other settings gets its own, :py:meth:`load`).
    graph_settings: GraphSettings = GraphSettings()

    def __init__(self, *args, **kwargs):
        """Create an empty, mutable graph; arguments are those of :py:class:`networkx.MultiDiGraph`.

        Args:
            args (Any): Positional arguments accepted by :py:meth:`networkx.MultiDiGraph.__init__`.
            kwargs (Any): Keyword arguments accepted by :py:meth:`networkx.MultiDiGraph.__init__`; they become graph
                attributes (``organism``, ``assembly``, ``ensembl_release``, ``confident_for_release`` ...).
        """
        self._store: _Store | None = None
        self._reverse_view = False
        self.log = logging.getLogger(__name__)
        self.available_forms = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ state

    def _refuse_if_frozen(self) -> None:
        if self.__dict__.get("_store") is not None:
            raise nx.NetworkXError(
                "This graph is frozen and cannot be changed: everything derived from it would be stale. Change a "
                "graph before its first use, or build a new one."
            )

    def add_node(self, node_for_adding, **attr):
        """Add a node (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().add_node(node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        """Add nodes (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().add_nodes_from(nodes_for_adding, **attr)

    def remove_node(self, n):
        """Remove a node (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().remove_node(n)

    def remove_nodes_from(self, nodes):
        """Remove nodes (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().remove_nodes_from(nodes)

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        """Add an edge (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().add_edge(u_for_edge, v_for_edge, key, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        """Add edges (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().add_edges_from(ebunch_to_add, **attr)

    def remove_edge(self, u, v, key=None):
        """Remove an edge (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().remove_edge(u, v, key)

    def remove_edges_from(self, ebunch):
        """Remove edges (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().remove_edges_from(ebunch)

    def clear(self):
        """Remove everything (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().clear()

    def clear_edges(self):
        """Remove all edges (only while the graph is not frozen)."""
        self._refuse_if_frozen()
        return super().clear_edges()

    def freeze(self) -> "TheGraph":
        """Turn the graph into its compact, read-only form (idempotent).

        Node names, node types and edges are moved into arrays, release sets become interned
        :py:class:`ReleaseSet` objects inside interned :py:class:`Connection` objects, and the networkx dictionaries
        are replaced by read-only views over the arrays. From then on the graph cannot be changed and all accessors
        and derived values are available.

        Returns:
            TheGraph: The graph itself.

        Raises:
            ValueError: If node names are not strings or edge keys are not integers.
        """
        if self.__dict__.get("_store") is None:
            if self.__dict__.get("_graph") is not None and self.__dict__.get("_graph") is not self:
                raise nx.NetworkXError("A view of another graph cannot be frozen; freeze the graph itself.")
            self._install(_Store.from_networkx(self))
        return self

    def _install(self, store: _Store) -> None:
        """Make ``store`` the content of this graph and set up the per-node caches."""
        self._store = store
        index = store.index
        size = self.graph_settings.cache_sizes

        @functools.lru_cache(maxsize=size["node"])
        def node_attributes(name: str) -> dict:
            return store.node_attributes(index[name])

        @functools.lru_cache(maxsize=size["adjacency"])
        def successors(name: str) -> dict:
            return store.adjacency(index[name], False)

        @functools.lru_cache(maxsize=size["adjacency"])
        def predecessors(name: str) -> dict:
            return store.adjacency(index[name], True)

        @functools.lru_cache(maxsize=size["edges"])
        def history(name: str, reverse: bool) -> tuple:
            return store.history_edges(index[name], reverse)

        @functools.lru_cache(maxsize=size["edges"])
        def xref(name: str, reverse: bool) -> tuple:
            return store.xref_edges(index[name], reverse)

        @functools.lru_cache(maxsize=size["summary"])
        def combined(name: str, family: str) -> dict:
            return self._combined_uncached(name, family)

        @functools.lru_cache(maxsize=size["summary"])
        def ranges(name: str) -> list:
            return self._active_ranges_uncached(name)

        @functools.lru_cache(maxsize=size["summary"])
        def active(name: str) -> ReleaseSet:
            return self._active_releases_uncached(name)

        @functools.lru_cache(maxsize=size["trios"])
        def trios(name: str) -> frozenset:
            return self._trios_uncached(name)

        self._c_node, self._c_succ, self._c_pred = node_attributes, successors, predecessors
        self._c_history, self._c_xref = history, xref
        self._c_combined = combined
        self._c_ranges, self._c_active, self._c_trios = ranges, active, trios
        # What a string means at a release: asked twice per conversion (for the report and by the engine).
        self._c_resolved = functools.lru_cache(maxsize=size["names"])(self._resolve_uncached)
        # What every resolution asks for: the identifiers that are written in more than one way, the last release.
        self._c_respelled, self._c_respelled_lower = store.respelled()
        self._c_last: int | None = None
        self._c_line: dict[int, ReleaseSet] | None = None

        self._node = _NodeMap(store, node_attributes)
        self._adj = _AdjacencyMap(store, successors)  # sets `_succ` as well
        self._pred = _AdjacencyMap(store, predecessors)
        for cached_view in (*_NETWORKX_VIEWS, "rev"):
            self.__dict__.pop(cached_view, None)

    def _frozen(self) -> _Store:
        """Return the store, freezing the graph first if needed."""
        store = self.__dict__.get("_store")
        if store is None:
            self.freeze()
            store = self._store
        if store is None:
            raise nx.NetworkXError("The graph could not be frozen.")
        return store

    def __getstate__(self):
        """Leave the per-node caches out of pickles and copies (they are rebuilt on demand)."""
        state = self.__dict__.copy()
        if state.get("_store") is not None:
            for name in list(state):
                if name.startswith("_c_") or name in ("_node", "_adj", "_succ", "_pred", "rev"):
                    del state[name]
            for cached_view in _NETWORKX_VIEWS:
                state.pop(cached_view, None)
        return state

    def __setstate__(self, state):
        """Restore a pickled graph; a frozen one gets fresh caches."""
        store = state.get("_store")
        self.__dict__.update(state)
        self.__dict__.setdefault("_store", None)
        self.__dict__.setdefault("_reverse_view", False)
        if store is not None:
            self._install(store)

    def calculate_caches(self) -> None:
        """Freeze the graph and compute the few graph-wide tables that are not stored with it.

        Nothing per node is precomputed: the per-node values (:py:meth:`combined`, :py:meth:`active_ranges`,
        :py:meth:`trios`) are computed when asked for and kept in bounded caches. The graph-wide tables
        (:py:attr:`hyperconnective_nodes`, the lists of databases and assemblies, the name indices) come with a
        loaded graph; for a graph made in memory they are computed here.
        """
        store = self._frozen()
        if store.lower_hash is None:
            store.build_name_indices()
        _ = self.hyperconnective_nodes
        _ = self.available_external_databases
        _ = self.available_assemblies
        _ = self.available_external_databases_assembly

    # ------------------------------------------------------------------ graph-level values

    @property
    def releases(self) -> tuple[int, ...]:
        """The Ensembl releases the graph covers, ascending."""
        return tuple(sorted(int(r) for r in self.graph["confident_for_release"]))

    @property
    def first_release(self) -> int:
        """The first release the graph covers."""
        return min(int(r) for r in self.graph["confident_for_release"])

    @property
    def snapshot_release(self) -> int:
        """The last release the graph covers."""
        return int(max(self.graph["confident_for_release"]))

    @property
    def manifest(self) -> dict:
        """What the graph was built from (empty for a graph made in memory); see :py:meth:`save`."""
        return self.graph.get("manifest", {})

    @property
    def assembly_releases(self) -> dict[int, tuple[int, ...]]:
        """The releases in which each genome assembly occurs on the cross-reference edges of the graph.

        Returns:
            dict[int, tuple[int, ...]]: ``{assembly: releases}``, ascending. For human this includes GRCh37 at the
            releases after GRCh38 took over (Ensembl maintains it in parallel); see :py:attr:`serving_assembly` for
            the assembly that serves each release in the backbone.
        """
        return self._global("assembly_releases", self._compute_assembly_releases)

    def _compute_assembly_releases(self) -> dict[int, tuple[int, ...]]:
        _src, _dst, ref, connections = self._xref_arrays()
        masks: dict[int, int] = {}
        for code in np.unique(ref).tolist():
            for _database, assembly, releases in connections[code].items:
                masks[assembly] = masks.get(assembly, 0) | releases.mask
        return {assembly: tuple(sorted(ReleaseSet.from_mask(mask))) for assembly, mask in sorted(masks.items())}

    @property
    def serving_assembly(self) -> dict[int, int]:
        """The genome assembly that serves each release in the backbone of the graph.

        A built graph records it: the primary assembly where it has the release, otherwise the organism's preferred
        assembly among those that have it. For a graph made in memory it is derived from the edges in the same way,
        with the order of the organism table (:py:attr:`idtrack.organisms.ORGANISM_TABLE`).

        Returns:
            dict[int, int]: ``{release: assembly}`` for the releases that some assembly has.
        """
        recorded = self.graph.get("backbone_assembly")
        if recorded:
            return recorded
        return self._global("backbone_assembly", self._compute_serving_assembly)

    def _compute_serving_assembly(self) -> dict[int, int]:
        primary = self.graph.get("genome_assembly")
        table = organisms.ORGANISM_TABLE.get(self.graph.get("organism"), {})
        result: dict[int, int] = {}
        for assembly, releases in self.assembly_releases.items():
            rank = (assembly != primary, table.get(assembly, {}).get("Priority", math.inf), assembly)
            for release in releases:
                have = result.get(release)
                if have is None or rank < (have != primary, table.get(have, {}).get("Priority", math.inf), have):
                    result[release] = assembly
        return dict(sorted(result.items()))

    @property
    def serving_line(self) -> dict[int, ReleaseSet]:
        """The releases that each genome assembly serves in the backbone (:py:func:`serving_line`).

        Returns:
            dict[int, ReleaseSet]: ``{assembly: releases}``: :py:attr:`serving_assembly` read the other way round.
            A link of the graph counts at a release only on the assembly that has the release here
            (:py:func:`idtrack.graph.served_mask`).
        """
        store = self._frozen()
        line = store.globals.get("serving_line")
        if line is None:  # the recorded assignment read the other way round: kept, not announced as a computation
            line = store.globals["serving_line"] = serving_line(self.serving_assembly)
        return line

    @property
    def rev(self) -> "TheGraph":
        """A read-only view of the graph with every edge reversed (no copy).

        The view shares the arrays and the per-node caches of the graph; ``rev.neighbors(n)`` are the predecessors
        of ``n``. The typed accessors of the view (:py:meth:`history_edges`, :py:meth:`xref_edges`) are reversed as
        well; everything that describes a node (:py:meth:`active_ranges`, :py:meth:`combined` ...) is that of the
        graph itself.

        Returns:
            TheGraph: The reversed view (itself frozen).
        """
        found = self.__dict__.get("rev")
        if found is None:
            self._frozen()
            found = self.reverse(copy=False)
            for name, value in self.__dict__.items():
                if name.startswith("_c_") or name in ("_store", "available_forms"):
                    found.__dict__[name] = value
            found.__dict__["_reverse_view"] = not self._reverse_view
            found.__dict__["rev"] = self
            self.__dict__["rev"] = found
        return found

    def _global(self, name: str, compute):
        store = self._frozen()
        if name not in store.globals:
            self.log.info(f"Graph-wide table being calculated: {name}")
            store.globals[name] = compute()
        return store.globals[name]

    def _xref_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Connection]]:
        """Return source, target and connection number of every key-0 edge that carries a connection."""
        store = self._frozen()
        chosen = np.flatnonzero((store.e_kind == _KIND_XREF) & (store.e_key == 0))
        src, dst, ref = store.e_src[chosen], store.e_dst[chosen], store.e_ref[chosen].astype(np.int64)
        connections = list(store.connections)
        extra = np.flatnonzero((store.e_kind == _KIND_OTHER) & (store.e_key == 0))
        if len(extra):
            more_src, more_dst, more_ref = [], [], []
            for edge in extra.tolist():
                connection = _connection_or_none(store.other_attributes[int(store.e_ref[edge])].get(ids.CONNECTION_KEY))
                if connection is not None:
                    more_src.append(int(store.e_src[edge]))
                    more_dst.append(int(store.e_dst[edge]))
                    more_ref.append(len(connections))
                    connections.append(connection)
            src = np.concatenate([src, np.asarray(more_src, dtype=src.dtype)])
            dst = np.concatenate([dst, np.asarray(more_dst, dtype=dst.dtype)])
            ref = np.concatenate([ref, np.asarray(more_ref, dtype=np.int64)])
        cross = store.ntype[src] != store.ntype[dst]
        return src[cross], dst[cross], ref[cross], connections

    @property
    def shared_external_ids(self) -> dict[str, int]:
        """External identifiers that several genes carry at the same time, with the largest number of such genes.

        For every external id that is linked to two or more different genes in some release:
        :py:meth:`genes_linked_at_once`. The table is computed when a graph is built and stored with it;
        :py:attr:`hyperconnective_nodes` is its part above :py:attr:`idtrack.settings.GraphSettings.hyperconnective_threshold`.

        Returns:
            dict[str, int]: ``{external node: largest number of genes linked in one release}``, numbers from 2.
        """
        return self._global("shared_external_ids", self._compute_shared_external_ids)

    def _compute_shared_external_ids(self) -> dict[str, int]:
        store = self._frozen()
        if store.external_code < 0:
            return {}
        degree = np.diff(store.out_ptr)  # an id cannot link more genes at once than it has edges
        candidates = np.flatnonzero((store.ntype == store.external_code) & (degree > 1))
        result: dict[str, int] = {}
        for i in candidates.tolist():
            name = store.names[i]
            count = self.genes_linked_at_once(name)
            if count > 1:
                result[name] = count
        return result

    @property
    def hyperconnective_nodes(self) -> dict[str, int]:
        """External identifiers that more genes share at one time than an identity can, with that number of genes.

        An external id is hyper-connective when, in a single Ensembl release, it is linked to more than
        :py:attr:`idtrack.settings.GraphSettings.hyperconnective_threshold` different genes
        (:py:meth:`genes_linked_at_once`: genes by stable id, whatever their versions; a transcript or a translation
        counts as its gene). Such ids are names of gene families and accessions shared by paralogues (RFAM families,
        ``U6``, ``5S_rRNA``, snoRNA names): they say what kind of gene something is, not which gene. Following them in
        a synonym search multiplies the candidates and bridges unrelated genes, so the search does not follow them.

        The count is per release, so the set does not grow with the number of releases in the graph: an id linked
        to many successive versions of one gene, or to a few genes that change over the years, is not
        hyper-connective, however many edges it has. The threshold was chosen by measurement against Ensembl's own
        data (pig: with 5, no kind of query is answered worse and 15 kinds better than with the earlier rule, more
        than 20 edges over all releases; with 10 or 20 the family ids shared by 6 to 20 genes become bridges and
        conversions of retired ids get worse). It is read when this property is read, so it can be changed on the
        class without building a graph again.

        Returns:
            dict[str, int]: ``{external node: largest number of genes linked in one release}`` for the
            hyper-connective nodes.
        """
        store = self._frozen()
        threshold = self.graph_settings.hyperconnective_threshold
        cached = store.globals.get("_hyperconnective_nodes")
        if cached is None or cached[0] != threshold:
            shared = self.shared_external_ids
            cached = (threshold, {name: count for name, count in shared.items() if count > threshold})
            store.globals["_hyperconnective_nodes"] = cached
        return cached[1]

    def genes_linked_at_once(self, node: str) -> int:
        """Return the largest number of different genes that a node is linked to in a single release.

        Genes are counted by stable id, whatever their versions; a transcript or a translation counts as its gene.
        For an external identifier this says how specific it is: 1 for an id of one gene, however many versions of
        that gene it was linked to over the years, and hundreds for the name of a gene family.

        Args:
            node (str): A node with outgoing cross-reference edges (an external identifier).

        Returns:
            int: The largest number of gene stable ids linked in one release; 0 if the node links no gene.
        """
        gene_mask: dict[str, int] = {}

        def follow(current: str, mask: int, depth: int) -> None:
            node_type = self.node_type(current)
            found = _ASSEMBLY_TYPE.match(node_type)
            form = ids.FORM_OF_ENSEMBL_TYPE.get(node_type) if found is None else found.group(2)
            if form == ids.BACKBONE_FORM:
                base = self.base_id(current)
                gene_mask[base] = gene_mask.get(base, 0) | mask
            elif form is not None and depth < 2:
                for parent, connection in self.xref_edges(current):
                    shared = mask & connection.releases.mask
                    if shared:
                        follow(parent, shared, depth + 1)

        for neighbour, connection in self.xref_edges(node):
            follow(neighbour, connection.releases.mask, 0)
        union = 0
        for mask in gene_mask.values():
            union |= mask
        best = 0
        for release in ReleaseSet.from_mask(union):
            best = max(best, sum(1 for mask in gene_mask.values() if mask >> release & 1))
        return best

    @property
    def available_external_databases(self) -> set[str]:
        """The names of the external databases whose identifiers are in the graph.

        Returns:
            set[str]: Database names found on the edges that leave external nodes.
        """
        return self._global("available_external_databases", self._compute_external_databases)

    def _compute_external_databases(self) -> set[str]:
        store = self._frozen()
        src, _dst, ref, connections = self._xref_arrays()
        result: set[str] = set()
        if store.external_code >= 0:
            for code in np.unique(ref[store.ntype[src] == store.external_code]).tolist():
                for database, assembly, _releases in connections[code].items:
                    form = ids.FORM_OF_ENSEMBL_TYPE.get(database)
                    result.add(database if form is None else _assembly_label(form, assembly))
        return result

    @property
    def available_assemblies(self) -> set[int]:
        """The genome assemblies (keys of the organism table) that occur on the edges of the graph.

        Returns:
            set[int]: Assembly keys, e.g. ``{36, 37, 38}`` for human.
        """
        return self._global("available_genome_assemblies", self._compute_assemblies)

    def _compute_assemblies(self) -> set[int]:
        _src, _dst, ref, connections = self._xref_arrays()
        return {asm for code in np.unique(ref).tolist() for _db, asm, _rs in connections[code].items}

    @property
    def available_external_databases_assembly(self) -> dict[int, set[str]]:
        """The external databases present on each genome assembly.

        Returns:
            dict[int, set[str]]: ``{assembly: database names}`` for every assembly of the graph.
        """
        return self._global("available_external_databases_assembly", self._compute_databases_per_assembly)

    def _compute_databases_per_assembly(self) -> dict[int, set[str]]:
        external = self.available_external_databases
        _src, _dst, ref, connections = self._xref_arrays()
        pairs = set()
        for code in np.unique(ref).tolist():
            for database, assembly, _releases in connections[code].items:
                form = ids.FORM_OF_ENSEMBL_TYPE.get(database)
                label = database if form is None else _assembly_label(form, assembly)
                if label in external:
                    pairs.add((assembly, label))
        return {asm: {label for a, label in pairs if a == asm} for asm in self.available_assemblies}

    @property
    def external_database_connection_form(self) -> dict[str, str]:
        """The Ensembl form (gene, transcript or translation) that each external database is linked to.

        For every database the Ensembl neighbours of its identifiers are counted by form and the commonest form
        wins (a few identifiers are annotated on another form than the rest of their database); a tie goes to the
        form that comes first in gene, transcript, translation.

        Returns:
            dict[str, str]: ``{database: form}``.

        Raises:
            ValueError: If an external identifier is linked to a node that is not an Ensembl identifier.
        """
        return self._global("external_database_connection_form", self._compute_connection_form)

    def _compute_connection_form(self) -> dict[str, str]:
        store = self._frozen()
        forms = list(ids.FORMS)
        form_of_type = []
        for node_type in store.node_types:
            found = _ASSEMBLY_TYPE.match(node_type) if isinstance(node_type, str) else None
            form = ids.FORM_OF_ENSEMBL_TYPE.get(node_type) if found is None else found.group(2)
            form_of_type.append(forms.index(form) if form in forms else -1)
        form_of_type_array = np.asarray(form_of_type, dtype=np.int64)
        src, dst, ref, connections = self._xref_arrays()
        keep = store.ntype[src] == store.external_code
        src, dst, ref = src[keep], dst[keep], ref[keep]
        result: dict[str, str] = {}
        if not len(src):
            return result
        dst_form = form_of_type_array[store.ntype[dst]]
        if np.any(dst_form < 0):
            bad = int(np.flatnonzero(dst_form < 0)[0])
            raise IntegrityError(
                f"Not connected only to Ensembl ID. From:{store.names[int(src[bad])]}, To:{store.names[int(dst[bad])]}, "
                f"{store.node_type(int(dst[bad]))}"
            )
        nodes, compact = np.unique(src, return_inverse=True)
        counts = np.zeros((len(nodes), len(forms)), dtype=np.int64)
        np.add.at(counts, (compact, dst_form), 1)
        has_database: dict[str, np.ndarray] = {}
        order = np.argsort(ref, kind="stable")
        boundaries = np.flatnonzero(np.diff(ref[order])) + 1
        for group in np.split(order, boundaries):
            for database in connections[int(ref[group[0]])].databases:
                flags = has_database.get(database)
                if flags is None:
                    flags = has_database[database] = np.zeros(len(nodes), dtype=bool)
                flags[compact[group]] = True
        for database in sorted(has_database):
            totals = counts[has_database[database]].sum(axis=0)
            result[database] = forms[int(np.argmax(totals))]
        return result

    @property
    def available_releases_given_database_assembly(self) -> dict[tuple[str, int], set]:
        """The releases in which each (database, assembly) pair occurs on some node; for the integrity tests.

        Keys are the external databases of each assembly, the Ensembl node types of each assembly of the graph
        (``assembly_<key>_ensembl_<form>``), the base gene type on the primary assembly, and the generic Ensembl
        types (``ensembl_gene`` ...) on the primary assembly, which stand for the releases of the corresponding
        assembly-specific type there. The table scans every node.

        Returns:
            dict[tuple[str, int], set[int]]: ``{(database, assembly): releases}``.
        """
        return self._global("available_releases_given_database_assembly", self._compute_releases_per_pair)

    def _compute_releases_per_pair(self) -> dict[tuple[str, int], set]:
        store = self._frozen()
        masks: dict[tuple[str, int], int] = {}
        for name in store.names:
            try:
                trios = self.trios(name)
            except KeyError:  # a node without cross-reference edges
                continue
            for database, assembly, release in trios:
                key = (database, assembly)
                masks[key] = masks.get(key, 0) | (1 << release)
        main_assembly = int(self.graph["genome_assembly"])
        result: dict[tuple[str, int], set] = {}
        for assembly, database_names in self.available_external_databases_assembly.items():
            for database in database_names:
                result[(database, assembly)] = set(ReleaseSet.from_mask(masks.get((database, assembly), 0)))
        for assembly in sorted(self.available_assemblies):
            for form in ids.FORMS:
                key = (_assembly_label(form, assembly), assembly)
                result[key] = set(ReleaseSet.from_mask(masks.get(key, 0)))
        for database in ids.BASE_ENSEMBL_TYPE.values():
            result[(database, main_assembly)] = set(ReleaseSet.from_mask(masks.get((database, main_assembly), 0)))
        for form, database in ids.ENSEMBL_TYPE.items():
            key = (_assembly_label(form, main_assembly), main_assembly)
            result[(database, main_assembly)] = set(ReleaseSet.from_mask(masks.get(key, 0)))
        return result

    # ------------------------------------------------------------------ nodes

    @staticmethod
    def _unknown(node: Any) -> UnknownIdentifierError:
        """Return the error for a node that is not in the graph."""
        return UnknownIdentifierError(f"{node!r} is not a node of the graph.", argument="node", value=node)

    def node_type(self, node: str) -> str:
        """Return the type of a node (``ensembl_gene``, ``base_ensembl_gene``, ``external`` ...).

        Raises:
            UnknownIdentifierError: If the node is not in the graph.
            KeyError: If (in a graph made by hand) the node has no type.
        """
        store = self._frozen()
        try:
            number = store.index[node]
        except KeyError:
            raise self._unknown(node) from None
        node_type = store.node_types[store.ntype_list[number]]
        if node_type is None:
            raise KeyError(ids.NODE_TYPE_KEY)
        return node_type

    def is_backbone(self, node: str) -> bool:
        """Tell whether the node is a gene of the history backbone (``<id>.Void`` and ``<id>.Retired`` included).

        Raises:
            UnknownIdentifierError: If the node is not in the graph.
        """
        store = self._frozen()
        try:
            return store.ntype_list[store.index[node]] == store.backbone_code
        except KeyError:
            raise self._unknown(node) from None

    def non_reference_releases(self, node: str) -> ReleaseSet:
        """Return the releases at which a gene lies outside the reference sequence of its genome assembly.

        Ensembl annotates genes on the chromosomes and scaffolds of an assembly (the reference sequence) and on
        regions beside it: alternative loci (haplotypes), assembly patches and LRG records. A gene on such a region
        is a copy of, or an alternative to, a gene of the reference sequence, with a stable id of its own. The
        builder takes the fact from Ensembl's core database of every release, for the assembly that serves the
        release in the backbone (:py:attr:`serving_assembly`): the sequence region of the gene carries the
        attribute :py:data:`idtrack.build.database_manager.NON_REFERENCE_ATTRIBUTE`.

        Args:
            node (str): A node of the graph.

        Returns:
            ReleaseSet: The releases; empty for a gene of the reference sequence, for a node that is not a gene of
            the backbone, and for every node of a graph that was made in memory rather than built.

        Raises:
            UnknownIdentifierError: If the node is not in the graph.
        """
        store = self._frozen()
        try:
            return store.non_reference.get(store.index[node], _NO_RELEASES)
        except KeyError:
            raise self._unknown(node) from None

    def is_non_reference(self, node: str, release: int) -> bool:
        """Tell whether a gene lies outside the reference sequence at a release (:py:meth:`non_reference_releases`)."""
        return bool(self.non_reference_releases(node).mask >> release & 1) if release >= 0 else False

    def base_id(self, node: str) -> str:
        """Return the stable id of an Ensembl node, without its version.

        Raises:
            UnknownIdentifierError: If the node is not in the graph.
            InvalidArgumentError: If the node is not an Ensembl ``<id>.<version>`` node.
        """
        self._frozen()
        try:
            return self._c_node(node)["ID"]
        except KeyError:
            raise self._not_versioned(node) from None

    def version(self, node: str) -> str:
        """Return the version of an Ensembl node as a string (``"7"``, ``"Void"``, ``"Retired"``).

        Raises:
            UnknownIdentifierError: If the node is not in the graph.
            InvalidArgumentError: If the node is not an Ensembl ``<id>.<version>`` node.
        """
        self._frozen()
        try:
            return self._c_node(node)["Version"]
        except KeyError:
            raise self._not_versioned(node) from None

    def _not_versioned(self, node: str) -> InputError:
        """Return the error for a node without version: unknown, or not an Ensembl ``<id>.<version>`` node."""
        if node not in self._frozen().index:
            return self._unknown(node)
        return InvalidArgumentError(f"{node!r} is not an Ensembl `<id>.<version>` node.", argument="node", value=node)

    def history_edges(self, node: str, reverse: bool = False) -> tuple[HistoryEdge, ...]:
        """Return the gene-history edges that leave a node (``reverse=True``: that enter it).

        Args:
            node (str): A node of the graph.
            reverse (bool): ``False`` for the edges towards later releases, ``True`` for those from earlier ones.

        Returns:
            tuple[HistoryEdge, ...]: One entry per edge, in the stored (deterministic) order; empty for nodes that
            are not backbone genes.

        Raises:
            UnknownIdentifierError: If the node is not in the graph.
        """
        store = self._frozen()
        if node not in store.index:
            raise self._unknown(node)
        return self._c_history(node, reverse != self._reverse_view)

    def xref_edges(self, node: str, reverse: bool = False) -> tuple[tuple[str, Connection], ...]:
        """Return the cross-reference edges that leave a node (``reverse=True``: that enter it).

        Args:
            node (str): A node of the graph.
            reverse (bool): ``False`` for edges from this node (external id to Ensembl id, base id to gene,
                transcript to gene, translation to transcript), ``True`` for edges to it.

        Returns:
            tuple[tuple[str, Connection], ...]: ``(other node, connection)`` per edge, in the stored order.

        Raises:
            UnknownIdentifierError: If the node is not in the graph.
        """
        store = self._frozen()
        if node not in store.index:
            raise self._unknown(node)
        return self._c_xref(node, reverse != self._reverse_view)

    def connection(self, source: str, target: str) -> Connection | None:
        """Return the connection of the cross-reference edge from ``source`` to ``target`` (``None`` if none)."""
        for other, connection in self.xref_edges(source):
            if other == target:
                return connection
        return None

    def get_next_edge_releases(self, node: str, reverse: bool) -> list:
        """List the releases at which the history of a backbone gene continues (or began).

        With ``reverse=False`` these are the releases at which the node is left: the ``old_release`` of its outgoing
        history edges, and infinity when it carries the self-loop of an id that is alive at the last release. With
        ``reverse=True`` these are the releases at which the node is entered: the ``new_release`` of its incoming
        history edges. Self-loops that record a re-mapping of the same id and version are not events.

        Args:
            node (str): A backbone gene node.
            reverse (bool): Direction, see above.

        Returns:
            list: The distinct releases, ascending (infinity last).

        Raises:
            ValueError: If ``node`` is not a backbone node.
        """
        if not self.is_backbone(node):
            raise UnknownIdentifierError(f"The method should be called only for backbone nodes: `{node}`.")
        releases = set()
        for edge in self.history_edges(node, reverse=reverse):
            if self.node_type(edge.node) != self.node_type(node):
                continue
            is_infinite = edge.new_release == _INF
            if edge.node == node and not (is_infinite and not reverse):
                continue
            releases.add(edge.old_release if (not is_infinite and not reverse) else edge.new_release)
        return sorted(releases)

    # ------------------------------------------------------------------ per-node summaries

    def _combined_uncached(self, name: str, family: str) -> dict:
        store = self._frozen()
        i = store.index[name]
        is_gene = store.ntype_list[i] == store.backbone_code
        if family == "out":
            found = store.combine(i, False)
            if found is None:
                raise KeyError(name)
            return {db: {asm: ReleaseSet.from_mask(mask) for asm, mask in per.items()} for db, per in found.items()}
        if family == "genes":
            if not is_gene:
                raise KeyError(name)
        elif is_gene or store.combine(i, False) is not None:
            raise KeyError(name)
        found = store.combine(i, True)
        if found is None:
            raise KeyError(name)
        per_assembly: dict[int, int] = {}
        for per in found.values():
            for assembly, mask in per.items():
                per_assembly[assembly] = per_assembly.get(assembly, 0) | mask
        return {
            _assembly_label(ids.BACKBONE_FORM, assembly): {assembly: ReleaseSet.from_mask(mask)}
            for assembly, mask in per_assembly.items()
        }

    def _family(self, name: str) -> str:
        store = self._frozen()
        code = store.ntype_list[store.index[name]]
        if code == store.backbone_code:
            return "genes"
        return "assembly" if store.type_is_assembly_gene[code] else "out"

    def combined(self, node: str) -> dict[str, dict[int, ReleaseSet]]:
        """Sum what all cross-reference edges of a node say: ``{database: {assembly: releases}}``.

        For a gene (of the backbone or of one assembly) the edges that enter it are summed and reported under the
        gene type of each assembly (``assembly_<key>_ensembl_gene``): the releases in which the gene is referred to
        on that assembly. For every other node the edges that leave it are summed per database; the relation of a
        transcript to its gene, or of a translation to its transcript, is named after the assembly of the edge
        (``assembly_<key>_ensembl_transcript``).

        Args:
            node (str): A node of the graph.

        Returns:
            dict[str, dict[int, ReleaseSet]]: The summary. Do not change it (it is cached).

        Raises:
            KeyError: If the node has no cross-reference edge (``<id>.Void`` and ``<id>.Retired`` nodes).
        """
        self._frozen()
        return self._c_combined(node, self._family(node))

    def active_ranges(self, node: str) -> list[list]:
        """Return the release ranges in which a node exists: ``[[first, last], ...]``, inclusive and ascending.

        For a backbone gene the ranges come from its history edges: a range starts at the ``new_release`` of an
        edge that enters the node and ends at the ``old_release`` of the last edge that leaves it; the last range of
        an id that is alive at the last release of the graph ends at infinity. For every other node the ranges are
        the releases of its cross-reference edges, whatever the database and assembly.

        Args:
            node (str): A node of the graph.

        Returns:
            list[list]: The ranges. Do not change the list (it is cached).

        Raises:
            ValueError: If the history edges of a backbone node do not form ranges (a node without edges, a history
                that starts with an end, a version whose last range never ends).
            UnknownIdentifierError: If the node is not in the graph.
            KeyError: If a node that is not a backbone gene has no cross-reference edge.
        """
        if node not in self._frozen().index:
            raise self._unknown(node)
        return self._c_ranges(node)

    def _active_ranges_uncached(self, name: str) -> list[list]:
        if not self.is_backbone(name):
            summary = self._c_combined(name, "assembly" if self._family(name) == "assembly" else "out")
            mask = 0
            for per in summary.values():
                for releases in per.values():
                    mask |= releases.mask
            return TheGraph.list_to_ranges(sorted(ReleaseSet.from_mask(mask)))

        starts = self.get_next_edge_releases(node=name, reverse=True)
        ends = self.get_next_edge_releases(node=name, reverse=False)
        version = self._c_node(name).get("Version")
        if not starts and not ends:
            raise IntegrityError(
                f"Node {name!r} has no incoming or outgoing edges. "
                f"This may indicate an orphaned node or data integrity issue in the graph."
            )
        if not starts:
            if version != ids.VOID:
                raise IntegrityError(
                    f"The history of {name!r} has no edge that enters it: only `<id>.{ids.VOID}` nodes "
                    "may be without one."
                )
            starts = [self.first_release]
        elif not ends:
            if version != ids.RETIRED:
                raise IntegrityError(
                    f"The history of {name!r} has no edge that leaves it: only `<id>.{ids.RETIRED}` nodes "
                    "may be without one (a version ends with an outgoing edge or the self-loop of the last release)."
                )
            ends = [self.snapshot_release]

        events = sorted(
            itertools.chain(zip(starts, itertools.repeat(True)), zip(ends, itertools.repeat(False))),
            key=lambda event: (event[0], not event[1]),
        )
        flat: list = []
        is_active = False
        for position, (release, is_start) in enumerate(events):
            if position == 0 and not is_start:
                raise IntegrityError(f"The range building should start with an in-edge: {name!r}.")
            if is_start:
                if not is_active:
                    flat.append(release)
                    is_active = True
            elif is_active:
                flat.append(release)
                is_active = False
            else:
                # The id was left before (a split or merge that it survived): its range goes on to this release.
                flat[-1] = release
        if is_active:
            if version != ids.RETIRED:
                raise IntegrityError(
                    f"The history of {name!r} has a range that starts at release {flat[-1]} and never ends: a "
                    "version must end with an outgoing edge or with the self-loop of the last release."
                )
            # A retired id stays retired until the last release of the graph.
            flat.append(max(self.snapshot_release, flat[-1]))
        return [flat[i : i + 2] for i in range(0, len(flat), 2)]

    def active_releases(self, node: str) -> ReleaseSet:
        """Return the releases of the graph in which a node exists, as a :py:class:`ReleaseSet`."""
        self._frozen()
        return self._c_active(node)

    def _active_releases_uncached(self, name: str) -> ReleaseSet:
        last = self.snapshot_release
        mask = 0
        for first, end in self.active_ranges(name):
            for release in range(int(first), (last if end == _INF else int(end)) + 1):
                mask |= 1 << release
        return ReleaseSet.from_mask(mask)

    def is_active(self, node: str, release: int) -> bool:
        """Tell whether a node exists at a release."""
        return bool(self.active_releases(node).mask >> release & 1) if release >= 0 else False

    def trios(self, node: str) -> frozenset:
        """Return every ``(database, assembly, release)`` in which a node occurs.

        ``database`` is the database name for external identifiers and the node type on the assembly for Ensembl
        nodes (``assembly_<key>_ensembl_gene`` ...); ``<id>.Void`` and ``<id>.Retired`` are reported on the primary
        assembly for the releases of their ranges.

        Args:
            node (str): A node of the graph.

        Returns:
            frozenset[tuple[str, int, int]]: The trios.

        Raises:
            ValueError: If a range of a ``Void`` or ``Retired`` node is not a valid release range.
        """
        self._frozen()
        return self._c_trios(node)

    def _trios_uncached(self, name: str) -> frozenset:
        if self.is_backbone(name) and self._c_node(name).get("Version") in ids.SENTINEL_VERSIONS:
            assembly = self.graph["genome_assembly"]
            label = _assembly_label(ids.BACKBONE_FORM, assembly)
            found = set()
            for first, end in self.active_ranges(name):
                if not _is_release(first) or not 0 < first <= end:
                    raise IntegrityError(f"Invalid release range: l1={first}, l2={end}. Expected 0 < l1 <= l2.")
                if end != _INF and not _is_release(end):
                    raise IntegrityError(f"Unexpected error: {end!r} should be either np.inf or integer.")
                for release in range(int(first), (self.snapshot_release if end == _INF else int(end)) + 1):
                    found.add((label, assembly, release))
            return frozenset(found)
        return frozenset(
            (database, assembly, release)
            for database, per in self.combined(name).items()
            for assembly, releases in per.items()
            for release in releases
        )

    def get_active_ranges_of_id_ensembl_all_inclusive(self, node: str) -> list[list]:
        """Return the release ranges in which a gene exists on any assembly.

        :py:meth:`active_ranges` of a backbone gene describes the backbone: each release as the organism's preferred
        assembly of that release has it. A gene can also exist at a release on another assembly (human GRCh37 keeps
        its genes in every release after GRCh38 took over). This method adds those releases. It also checks the two
        sources against each other: the releases in which the gene is referred to on the assembly that serves the
        backbone at that release must lie inside the backbone ranges.

        Args:
            node (str): A backbone gene or a gene of one assembly.

        Returns:
            list[list[int]]: Inclusive, ascending ranges of all releases in which the gene exists on any assembly.

        Raises:
            ValueError: If the cross-references contradict the backbone ranges, or if the node is not a gene.
        """
        node_type = self.node_type(node)
        if node_type == ids.ENSEMBL_TYPE["gene"]:
            ranges = self.active_ranges(node)
            try:
                summary = self.combined(node)
            except KeyError:  # a gene that no cross-reference edge enters
                summary = {}
            per_assembly: dict[int, set] = {}
            for per in summary.values():
                for assembly, release_set in per.items():
                    per_assembly.setdefault(assembly, set()).update(release_set)
            line = self.serving_line
            others: set = set()
            for assembly, releases in per_assembly.items():
                served = line.get(assembly, _NO_RELEASES)
                for release in releases:
                    if release in served:
                        if not TheGraph.is_point_in_range(ranges, release):
                            raise IntegrityError(
                                f"Inconsistency in ID range vs combined edges: {node}, {ranges}, {per_assembly}."
                            )
                    else:
                        others.add(release)
            return TheGraph.list_to_ranges(sorted(set(self.ranges_to_list(ranges)) | others))
        if _is_assembly_gene(node_type):
            return self.active_ranges(node)
        raise UnknownIdentifierError(
            f"Cannot get active ranges for {node!r}: node type {node_type!r} is not a gene type. "
            f"Expected 'ensembl_gene' or an assembly-specific gene type."
        )

    def get_active_ranges_of_base_id_alternative(self, base_id: str) -> list[list]:
        """Return the release ranges in which any version of a base gene id exists, on any assembly.

        Args:
            base_id (str): A base gene id (node type ``base_ensembl_gene``).

        Returns:
            list[list[int]]: Inclusive, ascending ranges; the union of
            :py:meth:`get_active_ranges_of_id_ensembl_all_inclusive` over the versions of the id.
        """
        all_ranges = [
            r for version, _ in self.xref_edges(base_id) for r in self.get_active_ranges_of_id_ensembl_all_inclusive(version)
        ]  # fmt: skip
        return self.list_to_ranges(self.ranges_to_list(all_ranges))

    @staticmethod
    def list_to_ranges(releases: list[int]) -> list[list]:
        """Turn an ascending list of distinct releases into inclusive ranges (``[1, 2, 3, 5]`` to ``[[1, 3], [5, 5]]``).

        Args:
            releases (list[int]): Releases, strictly increasing.

        Returns:
            list[list[int]]: The ranges that cover exactly these releases.
        """
        res = list()
        for _, a in itertools.groupby(enumerate(releases), lambda pair: pair[1] - pair[0]):
            b = list(a)
            res.append([b[0][1], b[-1][1]])
        return res

    def ranges_to_list(self, ranges: list[list]) -> list[int]:
        """Turn inclusive ranges into the ascending list of their releases; infinity stands for the last release.

        Args:
            ranges (list[list[int | float]]): Inclusive ``[first, last]`` pairs; ``last`` may be infinity.

        Returns:
            list[int]: The distinct releases of all ranges, ascending.
        """
        last = self.snapshot_release
        return sorted({k for i, j in ranges for k in range(i, (j if j != _INF else last) + 1)})

    @staticmethod
    def compact_ranges(ranges: list[list[int]]) -> list[list[int]]:
        """Merge ranges that touch or overlap; the argument is left unchanged.

        Args:
            ranges (list[list[int]]): Inclusive ranges ordered by their first release.

        Returns:
            list[list[int]]: New ranges that cover the same releases with as few ranges as possible.
        """
        result: list[list[int]] = []
        for first, last in ranges:
            if result and result[-1][1] + 1 >= first:
                result[-1][1] = max(result[-1][1], last)
            else:
                result.append([first, last])
        return result

    @staticmethod
    def get_intersecting_ranges(
        ranges: list[list[int]], other_ranges: list[list[int]], compact: bool = True
    ) -> list[list[int]]:
        """Return the ranges that two lists of ranges have in common.

        Args:
            ranges (list[list[int]]): Inclusive, ascending, non-overlapping ranges.
            other_ranges (list[list[int]]): Ranges of the same kind.
            compact (bool): Merge touching ranges of the result (default).

        Returns:
            list[list[int]]: The common ranges; empty when there are none.
        """
        result = [
            [max(first[0], second[0]), min(first[1], second[1])]
            for first in ranges
            for second in other_ranges
            if max(first[0], second[0]) <= min(first[1], second[1])
        ]
        return TheGraph.compact_ranges(result) if compact else result

    @staticmethod
    def is_point_in_range(ranges: list[list], release: int) -> bool:
        """Tell whether a release lies in one of the ranges.

        Args:
            ranges (list[list[int]]): Inclusive ranges.
            release (int): The release.

        Returns:
            bool: ``True`` if some range contains ``release``.
        """
        for l1, l2 in ranges:
            if l1 <= release <= l2:
                return True
        return False

    # ------------------------------------------------------------------ inspection helpers

    def get_id_list(self, database: str, assembly: int, release: int) -> list[str]:
        """Return the identifiers of a database on an assembly at a release (scans every node; for tests).

        Args:
            database (str): An external database name, or an Ensembl node type (``ensembl_gene``,
                ``ensembl_transcript``, ``ensembl_translation``).
            assembly (int): Genome assembly key.
            release (int): Ensembl release.

        Returns:
            list[str]: Node names in the order of the graph; ``<id>.Void`` and ``<id>.Retired`` are left out.
        """
        store = self._frozen()
        form = ids.FORM_OF_ENSEMBL_TYPE.get(database)
        key = (database if form is None else _assembly_label(form, assembly), assembly, release)
        found = []
        for name in store.names:
            if self.is_backbone(name) and self._c_node(name).get("Version") in ids.SENTINEL_VERSIONS:
                continue
            try:
                trios = self.trios(name)
            except KeyError:
                continue
            if key in trios:
                found.append(name)
        return found

    # ------------------------------------------------------------------ files

    def save(self, path: str) -> None:
        """Write the graph to a file, atomically.

        The file is a NumPy ``.npz`` archive (no pickled objects): the node names, node types, edge arrays,
        the table of distinct connections with their release bit masks, the genes outside the reference sequence
        with their releases, the name indices, and one JSON document with the graph attributes, the manifest and the
        graph-wide tables. It is written to a temporary file in the same
        folder and renamed, so an interrupted run leaves either the previous file or none, never a part of one.

        Args:
            path (str): File to write.

        Raises:
            ValueError: If the graph holds edges or node attributes that the file cannot express (graphs made by
                hand with other attributes than a connection or a release pair).
        """
        store = self._frozen()
        if store.other_attributes or store.node_overrides:
            raise IntegrityError(
                "Only graphs made of history edges and cross-reference edges, with nodes that carry nothing but "
                "their type, can be saved."
            )
        self.calculate_caches()
        _ = self.external_database_connection_form
        for name in store.names:
            if _NAME_SEPARATOR in name:
                raise IntegrityError(f"A node name contains a NUL character: {name!r}.")

        databases: list[str] = []
        database_code: dict[str, int] = {}
        c_ptr, c_db, c_asm, c_mask = [0], [], [], []
        for connection in store.connections:
            for database, assembly, releases in connection.items:
                code = database_code.get(database)
                if code is None:
                    code = database_code[database] = len(databases)
                    databases.append(database)
                c_db.append(code)
                c_asm.append(assembly)
                c_mask.append(releases.mask)
            c_ptr.append(len(c_db))
        masks = _masks_to_words(c_mask)
        marked = sorted(store.non_reference)

        attributes = dict(self.graph)
        document = {
            "format_version": GRAPH_FORMAT_VERSION,
            "graph": _to_json(attributes),
            "available_forms": self.available_forms,
            "node_types": store.node_types,
            "databases": databases,
            "globals": _to_json({k: store.globals[k] for k in _STORED_GLOBALS}),
            "spellings": [
                [i, db, latest, [[format(mask, "x"), text] for mask, text in otherwise]]
                for (i, db), (latest, otherwise) in sorted(store.spellings.items())
            ],
        }
        arrays = {
            "document": np.frombuffer(json.dumps(document, sort_keys=True).encode("utf-8"), dtype=np.uint8),
            "names": np.frombuffer(_NAME_SEPARATOR.join(store.names).encode("utf-8"), dtype=np.uint8),
            "ntype": store.ntype,
            "e_src": store.e_src,
            "e_dst": store.e_dst,
            "e_key": store.e_key,
            "e_kind": store.e_kind,
            "e_ref": store.e_ref,
            "in_order": store.in_order,
            "h_old": store.h_old,
            "h_new": store.h_new,
            "h_weight": store.h_weight,
            "c_ptr": np.asarray(c_ptr, dtype=np.int64),
            "c_db": np.asarray(c_db, dtype=np.int32),
            "c_asm": np.asarray(c_asm, dtype=np.int64),
            "c_mask": masks,
            "nonref_node": np.asarray(marked, dtype=np.int32),
            "nonref_mask": _masks_to_words([store.non_reference[i].mask for i in marked]),
            "lower_hash": store.lower_hash,
            "lower_idx": store.lower_idx,
            "separator_hash": store.separator_hash,
            "separator_idx": store.separator_idx,
        }
        with replacing(path) as tmp, open(tmp, "wb") as handle:
            np.savez(handle, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(
        cls, path: str, expect: Mapping[str, Any] | None = None, graph_settings: GraphSettings | None = None
    ) -> "TheGraph":
        """Read a graph file written by :py:meth:`save`. Needs no network, no table and no build module.

        Args:
            path (str): The graph file.
            expect (Mapping[str, Any] | None): Fields of the manifest that the graph must have, e.g.
                ``{"organism": "homo_sapiens", "genome_assembly": 38}``. See
                :py:meth:`idtrack.build.GraphMaker.graph_identity`.
            graph_settings (GraphSettings | None): The caches and the hyper-connectivity threshold of this graph;
                ``None`` for the defaults.

        Returns:
            TheGraph: The frozen graph.

        Raises:
            ArgumentTypeError: If *graph_settings* is not a :py:class:`idtrack.settings.GraphSettings`.
            GraphFileError: If there is no such file or it cannot be read as a graph file.
            GraphMismatchError: If the file was written by another version of the graph format, or differs from
                ``expect`` (the message names the fields).
        """
        check_graph_settings(graph_settings)
        if not os.path.isfile(path):
            raise GraphFileError(f"No graph file at `{path}`.", path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            document = json.loads(arrays["document"].tobytes().decode("utf-8"))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as error:
            raise GraphFileError(
                f"`{path}` cannot be read as an idtrack graph file ({type(error).__name__}: {error}). {_REBUILD_HINT}",
                path,
            ) from error
        if document.get("format_version") != GRAPH_FORMAT_VERSION:
            raise GraphMismatchError(
                f"`{path}` has graph format version {document.get('format_version')!r}; this version of idtrack "
                f"reads version {GRAPH_FORMAT_VERSION}. {_REBUILD_HINT}",
                path,
            )
        attributes = _from_json_graph(document["graph"])
        manifest = attributes.get("manifest", {})
        if expect:
            differing = {k: (manifest.get(k), v) for k, v in expect.items() if _to_json(manifest.get(k)) != _to_json(v)}
            if differing:
                details = "; ".join(
                    f"{k}: file has {have!r}, asked for {want!r}" for k, (have, want) in differing.items()
                )
                raise GraphMismatchError(
                    f"The graph file `{path}` is not the graph that was asked for ({details}). Ask for the graph "
                    f"that the file holds, or build this one. {_REBUILD_HINT}",
                    path,
                )

        try:
            names = arrays["names"].tobytes().decode("utf-8").split(_NAME_SEPARATOR) if len(arrays["ntype"]) else []
            databases = document["databases"]
            c_ptr, c_db, c_asm = arrays["c_ptr"].tolist(), arrays["c_db"].tolist(), arrays["c_asm"].tolist()
            mask_values = _words_to_masks(arrays["c_mask"])
            connections = [
                Connection.of((databases[c_db[j]], c_asm[j], ReleaseSet.from_mask(mask_values[j])) for j in range(a, b))
                for a, b in zip(c_ptr, c_ptr[1:])
            ]
            store = _Store(
                names, document["node_types"], arrays["ntype"], arrays["e_src"], arrays["e_dst"], arrays["e_key"],
                arrays["e_kind"], arrays["e_ref"], arrays["in_order"], arrays["h_old"], arrays["h_new"],
                arrays["h_weight"], connections,
            )  # fmt: skip
            store.lower_hash, store.lower_idx = arrays["lower_hash"], arrays["lower_idx"]
            store.separator_hash, store.separator_idx = arrays["separator_hash"], arrays["separator_idx"]
            store.globals = _from_json_globals(document["globals"])
            store.spellings = {
                (int(i), db): (latest, tuple((int(mask, 16), text) for mask, text in otherwise))
                for i, db, latest, otherwise in document["spellings"]
            }
            marked = arrays["nonref_node"].tolist()
            if any(not 0 <= i < store.n for i in marked):
                raise ValueError("a gene outside the reference sequence is not a node of the file")
            store.non_reference = dict(
                zip(marked, (ReleaseSet.from_mask(mask) for mask in _words_to_masks(arrays["nonref_mask"])))
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise GraphFileError(
                f"`{path}` does not hold what a graph file of format version {GRAPH_FORMAT_VERSION} holds "
                f"({type(error).__name__}: {error}). {_REBUILD_HINT}",
                path,
            ) from error

        graph = cls()
        if graph_settings is not None:
            graph.graph_settings = graph_settings
        graph.graph.update(attributes)
        graph.available_forms = document.get("available_forms")
        graph._install(store)
        return graph

    @classmethod
    def _from_store(cls, store: _Store, attributes: dict, available_forms: list[str] | None) -> "TheGraph":
        """Make a frozen graph from a store (used by the builder)."""
        graph = cls()
        graph.graph.update(attributes)
        graph.available_forms = available_forms
        graph._install(store)
        return graph


__all__ = ["GRAPH_FORMAT_VERSION", "TheGraph"]
