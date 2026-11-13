#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import bisect
import functools
import hashlib
import itertools
import json
import logging
import math
import operator
import os
import re
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, NamedTuple, SupportsIndex

import networkx as nx
import numpy as np

from idtrack._db import DB

#: Version of the graph file layout and of the build rules behind it. It is raised whenever a graph built by an
#: earlier version of the code must not be loaded any more (a change of the file layout, of what an edge means, of
#: how names are chosen or of how the history is validated).
GRAPH_FORMAT_VERSION = 2

_INF = float("inf")
_NAN = float("nan")
_NAME_SEPARATOR = "\x00"
_ASSEMBLY_TYPE = re.compile(r"^assembly_(-?[0-9]+)_ensembl_([a-z]+)$")
_VERSION_SUFFIX = re.compile(r"^(.+)([._-])([0-9]+)$")
_SEPARATORS = "-_."
_FOLD_SEPARATORS = str.maketrans("-.", "__")
_KIND_HISTORY, _KIND_XREF, _KIND_OTHER = 0, 1, 2
#: The steps of name resolution that are tried on names and then on synonyms, in this order.
_RESOLVE_STEPS = ("exact", "case", "separator", "version")
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


class GraphMismatchError(ValueError):
    """Raised when a graph file is not the graph that was asked for (or cannot be read as a graph file).

    The message names the file, what differs, and the way out (:py:data:`_REBUILD_HINT`): a graph file is never
    replaced silently.
    """


# What a user can do about a graph file that cannot be used, named by what the public calls offer:
# `API.build_graph` has no switch to rebuild (it loads the file when there is one and builds only when there is none);
# `GraphMaker.get_graph` has `create_even_if_exist`.
_REBUILD_HINT = (
    "To build the graph again: delete or move the file and call `API.build_graph` again (it builds the graph when "
    "the local repository holds no file for it), or call `GraphMaker.get_graph(create_even_if_exist=True)`, which "
    "replaces the file."
)


class ReleaseSet(frozenset):
    """An immutable set of Ensembl releases with its bit mask, one object per distinct value.

    It is a :py:class:`frozenset` (membership, iteration, ``|``, ``&``, ``min``, ``max`` and comparison with any set
    work as usual) that also carries :py:attr:`mask`, an ``int`` whose bit ``r`` is set when release ``r`` is in the
    set, so that ``mask >> r & 1`` tests membership and ``|``/``&`` on masks combine sets without building them. Equal
    sets are the same object: a graph with millions of edges holds a few thousand of them.
    """

    __slots__ = ("mask",)
    mask: int
    _interned: dict[int, "ReleaseSet"] = {}

    @classmethod
    def from_mask(cls, mask: int) -> "ReleaseSet":
        """Return the set of the releases whose bits are set in ``mask``."""
        found = cls._interned.get(mask)
        if found is None:
            if mask < 0:
                raise ValueError("A release mask cannot be negative.")
            releases, rest, release = [], mask, 0
            while rest:
                if rest & 1:
                    releases.append(release)
                rest >>= 1
                release += 1
            found = frozenset.__new__(cls, releases)
            found.mask = mask
            cls._interned[mask] = found
        return found

    @classmethod
    def of(cls, releases: Iterable[int]) -> "ReleaseSet":
        """Return the set of the given releases (non-negative integers)."""
        if isinstance(releases, ReleaseSet):
            return releases
        mask = 0
        for release in releases:
            if isinstance(release, bool) or not isinstance(release, (int, np.integer)) or release < 0:
                raise ValueError(f"A release must be a non-negative integer, got {release!r}.")
            mask |= 1 << int(release)
        return cls.from_mask(mask)

    def __reduce__(self):
        """Pickle and copy through the mask, so that a restored set is the interned object."""
        return ReleaseSet.from_mask, (self.mask,)


_NO_RELEASES = ReleaseSet.from_mask(0)


class Connection:
    """What one cross-reference edge says: in which releases which database links the two nodes on which assembly.

    Instances are immutable and shared: all edges that carry the same information hold the same object.

    Attributes:
        items (tuple[tuple[str, int, ReleaseSet], ...]): ``(database, assembly, releases)`` triples.
        releases (ReleaseSet): All releases of the edge, whatever the database and assembly.
        as_dict (dict[str, dict[int, ReleaseSet]]): The same as ``{database: {assembly: releases}}``. Do not change.
        databases (frozenset[str]): The database names of the edge.
    """

    __slots__ = ("items", "releases", "as_dict", "databases", "_attr")
    items: tuple[tuple[str, int, ReleaseSet], ...]
    releases: ReleaseSet
    as_dict: dict[str, dict[int, ReleaseSet]]
    databases: frozenset[str]
    _attr: dict | None
    _interned: dict[tuple, "Connection"] = {}

    @classmethod
    def of(cls, items: Iterable[tuple[str, int, Any]]) -> "Connection":
        """Return the connection of the ``(database, assembly, releases)`` triples, in the order given.

        ``releases`` is a :py:class:`ReleaseSet`, any iterable of releases, or a bit mask (an ``int``).
        """
        normal = tuple(
            (str(db), int(asm), ReleaseSet.from_mask(rs) if isinstance(rs, int) else ReleaseSet.of(rs))
            for db, asm, rs in items
        )
        key = tuple((db, asm, rs.mask) for db, asm, rs in normal)
        found = cls._interned.get(key)
        if found is None:
            found = object.__new__(cls)
            as_dict: dict[str, dict[int, ReleaseSet]] = {}
            mask = 0
            for db, asm, rs in normal:
                per_db = as_dict.setdefault(db, {})
                if asm in per_db:
                    raise ValueError(f"Database {db!r} and assembly {asm!r} occur twice in one connection.")
                per_db[asm] = rs
                mask |= rs.mask
            found.items = normal
            found.releases = ReleaseSet.from_mask(mask)
            found.as_dict = as_dict
            found.databases = frozenset(as_dict)
            found._attr = None
            cls._interned[key] = found
        return found

    @classmethod
    def of_dict(cls, connection: Mapping) -> "Connection":
        """Return the connection of a ``{database: {assembly: releases}}`` mapping, in its own order."""
        return cls.of((db, asm, rs) for db, per_db in connection.items() for asm, rs in per_db.items())

    @property
    def attr(self) -> dict:
        """The attributes of the edge as networkx shows them.

        ``connection`` is :py:attr:`as_dict` and ``available_releases`` is :py:attr:`releases`: the same shared,
        immutable objects that the accessors return.
        """
        if self._attr is None:
            self._attr = {DB.connection_dict: self.as_dict, "available_releases": self.releases}
        return self._attr

    def __reduce__(self):
        """Pickle and copy through the triples, so that a restored connection is the interned object."""
        return Connection.of, (tuple((db, asm, tuple(sorted(rs))) for db, asm, rs in self.items),)

    def __repr__(self) -> str:
        """Show the triples."""
        return f"Connection({[(db, asm, sorted(rs)) for db, asm, rs in self.items]})"


def serving_line(backbone_assembly: Mapping[int, int]) -> dict[int, ReleaseSet]:
    """Return, per genome assembly, the releases that the assembly serves in the backbone.

    A release belongs to one assembly: the one that serves it in the backbone of the graph
    (:py:attr:`TheGraph.backbone_assembly`). An earlier assembly that Ensembl keeps listing beside the primary one
    (human GRCh37, frozen with its annotation of release 75 and listed at every later release) serves none of the
    releases that the primary assembly has.

    Args:
        backbone_assembly (Mapping[int, int]): ``{release: assembly}``.

    Returns:
        dict[int, ReleaseSet]: ``{assembly: releases}`` for the assemblies that serve a release.
    """
    masks: dict[int, int] = {}
    for release, assembly in backbone_assembly.items():
        masks[int(assembly)] = masks.get(int(assembly), 0) | (1 << int(release))
    return {assembly: ReleaseSet.from_mask(mask) for assembly, mask in masks.items()}


def served_mask(
    items: Iterable[tuple[str, int, ReleaseSet]],
    line: Mapping[int, ReleaseSet],
    database: str | None = None,
    exclude: frozenset = frozenset(),
) -> int:
    """Return the releases at which a cross-reference link holds, each counted on the assembly that serves it.

    This is the one rule for where a link counts. The builder chooses the spelling of a release by it, the engine
    follows a link by it, and the name resolution says by it whether an identifier exists at a release: a link
    that an assembly lists at a release it does not serve (:py:func:`serving_line`) is not a link of that release.

    Args:
        items (Iterable[tuple[str, int, ReleaseSet]]): ``(database, assembly, releases)`` of a link
            (:py:attr:`Connection.items`).
        line (Mapping[int, ReleaseSet]): The releases each assembly serves (:py:func:`serving_line`).
        database (str | None): Count this database only. ``None``: every database.
        exclude (frozenset): Databases that do not count.

    Returns:
        int: The bit mask of the releases (bit ``r`` set: the link holds at release ``r``).
    """
    mask = 0
    for item_database, assembly, releases in items:
        if (database is None or item_database == database) and item_database not in exclude:
            served = line.get(assembly)
            if served is not None:
                mask |= releases.mask & served.mask
    return mask


class HistoryEdge(NamedTuple):
    """One edge of the gene history as seen from a node: the other end and the release pair.

    The fields: ``node``, the node at the other end of the edge; ``old_release``, the last release of the source side
    of the event; ``new_release``, the first release of the target side (``float("inf")`` on the self-loop of an id
    that is alive at the last release of the graph); ``weight``, Ensembl's mapping score of the event (NaN where there
    is none); ``key``, the key of the edge among the parallel edges between the two nodes.
    """

    node: str
    old_release: Any
    new_release: Any
    weight: float
    key: int


def _stable_hash(text: str) -> int:
    """Return a 64-bit hash of ``text`` that is the same in every process (unlike ``hash``)."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def _has_id_and_version(node_type: Any) -> bool:
    """Tell whether nodes of this type are Ensembl ``<id>.<version>`` nodes."""
    return isinstance(node_type, str) and (node_type in DB.nts_ensembl_reverse or bool(_ASSEMBLY_TYPE.match(node_type)))


def _is_assembly_gene(node_type: Any) -> bool:
    """Tell whether the type is the gene type of one assembly (``assembly_<key>_ensembl_gene``)."""
    if not isinstance(node_type, str):
        return False
    found = _ASSEMBLY_TYPE.match(node_type)
    return found is not None and found.group(2) == DB.backbone_form


def _assembly_label(form: str, assembly: int) -> str:
    """Return the database name of an Ensembl relation on one assembly (``assembly_<key>_ensembl_<form>``)."""
    return f"assembly_{assembly}_ensembl_{form}"


def _release_number(value: Any, name: str) -> int:
    """Read an argument that names an Ensembl release (any integer type; never a ``bool``, ``float`` or ``str``)."""
    try:
        if not isinstance(value, bool):
            return operator.index(value)
    except TypeError:
        pass
    raise TypeError(f"`{name}` must be an integer Ensembl release or None, not {value!r}.")


def _is_release(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


class _Store:
    """The frozen content of a graph: node names and types, and all edges as arrays.

    Nodes are numbered ``0 .. n-1``. Every edge has a source, a target, a key (parallel edges), a kind and a
    reference into the payload of its kind: the release pair and weight of a history edge, the
    :py:class:`Connection` of a cross-reference edge, or an attribute dictionary for anything else (graphs made by
    hand in tests). Edges are stored in the order of their source node; ``in_order`` lists them in the order of
    their target node.
    """

    def __init__(
        self,
        names: list[str],
        node_types: list[Any],
        ntype: np.ndarray,
        e_src: np.ndarray,
        e_dst: np.ndarray,
        e_key: np.ndarray,
        e_kind: np.ndarray,
        e_ref: np.ndarray,
        in_order: np.ndarray,
        h_old: np.ndarray,
        h_new: np.ndarray,
        h_weight: np.ndarray,
        connections: list[Connection],
        other_attributes: list[dict] | None = None,
        node_overrides: dict[int, dict] | None = None,
    ):
        self.names = names
        self.n = len(names)
        self.index: dict[str, int] = dict(zip(names, range(self.n)))
        if len(self.index) != self.n:
            raise ValueError("Node names of a graph must be unique.")
        self.node_types = node_types
        self.ntype = np.ascontiguousarray(ntype, dtype=np.int16)
        self.ntype_list: list[int] = self.ntype.tolist()
        self.e_src = np.ascontiguousarray(e_src, dtype=np.int32)
        self.e_dst = np.ascontiguousarray(e_dst, dtype=np.int32)
        self.e_key = np.ascontiguousarray(e_key, dtype=np.int32)
        self.e_kind = np.ascontiguousarray(e_kind, dtype=np.int8)
        self.e_ref = np.ascontiguousarray(e_ref, dtype=np.int32)
        self.in_order = np.ascontiguousarray(in_order, dtype=np.int32)
        self.h_old = np.ascontiguousarray(h_old, dtype=np.int32)
        self.h_new = np.ascontiguousarray(h_new, dtype=np.int32)  # -1 stands for infinity
        self.h_weight = np.ascontiguousarray(h_weight, dtype=np.float64)
        self.connections = connections
        self.other_attributes = other_attributes or []
        self.node_overrides = node_overrides or {}
        self.m = len(self.e_src)

        if self.m and (np.any(np.diff(self.e_src) < 0) or np.any(np.diff(self.e_dst[self.in_order]) < 0)):
            raise ValueError("Edges must be ordered by their source node, and `in_order` by their target node.")
        self.out_ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.e_src, minlength=self.n), out=self.out_ptr[1:])
        self.in_ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.e_dst, minlength=self.n), out=self.in_ptr[1:])

        self.backbone_code = node_types.index(_backbone_type()) if _backbone_type() in node_types else -1
        self.external_code = node_types.index(DB.nts_external) if DB.nts_external in node_types else -1
        self.type_has_version = [_has_id_and_version(t) for t in node_types]
        self.type_is_assembly_gene = [_is_assembly_gene(t) for t in node_types]

        self.lower_hash: np.ndarray | None = None
        self.lower_idx: np.ndarray | None = None
        self.separator_hash: np.ndarray | None = None
        self.separator_idx: np.ndarray | None = None
        self.globals: dict[str, Any] = {}
        self.spellings: dict[tuple[int, str], tuple[str, tuple[tuple[int, str], ...]]] = {}
        self.non_reference: dict[int, ReleaseSet] = {}
        self.names_sorted: bool | None = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_parts(
        cls,
        names: list[str],
        node_types: list[str],
        ntype: np.ndarray,
        history: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        xref: tuple[np.ndarray, np.ndarray, np.ndarray],
        connections: list[Connection],
    ) -> "_Store":
        """Make a store from the two edge families of a built graph, in a canonical edge order.

        Args:
            names: Node names; the position is the node number.
            node_types: The distinct node types.
            ntype: For every node the position of its type in ``node_types``.
            history: ``(source, target, key, old_release, new_release, weight)`` of the history edges;
                ``new_release`` is ``-1`` for infinity.
            xref: ``(source, target, connection)`` of the cross-reference edges, ``connection`` being a position in
                ``connections``.
            connections: The distinct connections.

        Returns:
            _Store: Edges ordered by source, kind, target and key; incoming edges by target, kind, source and key.
        """
        h_src, h_dst, h_key, h_old, h_new, h_weight = history
        x_src, x_dst, x_conn = xref
        n_h, n_x = len(h_src), len(x_src)
        e_src = np.concatenate([np.asarray(h_src, dtype=np.int64), np.asarray(x_src, dtype=np.int64)])
        e_dst = np.concatenate([np.asarray(h_dst, dtype=np.int64), np.asarray(x_dst, dtype=np.int64)])
        e_key = np.concatenate([np.asarray(h_key, dtype=np.int64), np.zeros(n_x, dtype=np.int64)])
        e_kind = np.concatenate([np.full(n_h, _KIND_HISTORY, dtype=np.int8), np.full(n_x, _KIND_XREF, dtype=np.int8)])
        e_ref = np.concatenate([np.arange(n_h, dtype=np.int64), np.asarray(x_conn, dtype=np.int64)])
        order = np.lexsort((e_key, e_dst, e_kind, e_src))
        e_src, e_dst, e_key, e_kind, e_ref = e_src[order], e_dst[order], e_key[order], e_kind[order], e_ref[order]
        in_order = np.lexsort((e_key, e_src, e_kind, e_dst))
        return cls(
            names, list(node_types), ntype, e_src, e_dst, e_key, e_kind, e_ref, in_order, h_old, h_new, h_weight,
            connections,
        )  # fmt: skip

    @classmethod
    def from_networkx(cls, graph: nx.MultiDiGraph) -> "_Store":
        """Make a store from the dictionaries of a networkx graph, keeping its node and adjacency orders."""
        node = graph._node
        names = list(node)
        index = {name: i for i, name in enumerate(names)}
        node_types: list[Any] = []
        type_code: dict[Any, int] = {}
        ntype = np.empty(len(names), dtype=np.int16)
        overrides: dict[int, dict] = {}
        for i, (name, attributes) in enumerate(node.items()):
            if not isinstance(name, str):
                raise ValueError(f"Node names of a graph must be strings, got {name!r}.")
            node_type = attributes.get(DB.node_type_str)
            code = type_code.get(node_type)
            if code is None:
                code = type_code[node_type] = len(node_types)
                node_types.append(node_type)
            ntype[i] = code
            derived = _derived_node_attributes(name, node_type)
            if any(key not in derived or derived[key] != value for key, value in attributes.items()):
                # A node made by hand with attributes of its own: they win over what the name says.
                overrides[i] = {**derived, **attributes}

        e_src: list[int] = []
        e_dst: list[int] = []
        e_key: list[int] = []
        e_kind: list[int] = []
        e_ref: list[int] = []
        h_old: list[int] = []
        h_new: list[int] = []
        h_weight: list[float] = []
        connections: list[Connection] = []
        connection_code: dict[int, int] = {}
        others: list[dict] = []
        edge_number: dict[tuple[int, int, int], int] = {}
        history_keys = {"weight", "old_release", "new_release"}
        for source, neighbours in graph._succ.items():
            si = index[source]
            for target, keyed in neighbours.items():
                ti = index[target]
                for key, attributes in keyed.items():
                    if not _is_release(key):
                        raise ValueError(f"Edge keys must be integers, got {key!r} between {source!r} and {target!r}.")
                    edge_number[(si, ti, int(key))] = len(e_src)
                    e_src.append(si)
                    e_dst.append(ti)
                    e_key.append(int(key))
                    connection = None
                    if (
                        DB.connection_dict in attributes
                        and key == 0
                        and all(k in (DB.connection_dict, "available_releases") for k in attributes)
                    ):
                        connection = _connection_or_none(attributes[DB.connection_dict])
                        if connection is not None and "available_releases" in attributes:
                            if attributes["available_releases"] != connection.releases:
                                connection = None
                    if connection is not None:
                        code = connection_code.get(id(connection))
                        if code is None:
                            code = connection_code[id(connection)] = len(connections)
                            connections.append(connection)
                        e_kind.append(_KIND_XREF)
                        e_ref.append(code)
                    elif (
                        attributes.keys() == history_keys
                        and _is_release(attributes["old_release"])
                        and attributes["old_release"] >= 0
                        and (
                            (_is_release(attributes["new_release"]) and attributes["new_release"] >= 0)
                            or attributes["new_release"] == _INF
                        )
                        and isinstance(attributes["weight"], (int, float, np.floating, np.integer))
                    ):
                        e_kind.append(_KIND_HISTORY)
                        e_ref.append(len(h_old))
                        h_old.append(int(attributes["old_release"]))
                        h_new.append(-1 if attributes["new_release"] == _INF else int(attributes["new_release"]))
                        h_weight.append(float(attributes["weight"]))
                    else:
                        e_kind.append(_KIND_OTHER)
                        e_ref.append(len(others))
                        others.append(attributes)
        in_order: list[int] = []
        for target, neighbours in graph._pred.items():
            ti = index[target]
            for source, keyed in neighbours.items():
                si = index[source]
                for key in keyed:
                    in_order.append(edge_number[(si, ti, int(key))])
        return cls(
            names, node_types, ntype,
            np.asarray(e_src, dtype=np.int32), np.asarray(e_dst, dtype=np.int32), np.asarray(e_key, dtype=np.int32),
            np.asarray(e_kind, dtype=np.int8), np.asarray(e_ref, dtype=np.int32), np.asarray(in_order, dtype=np.int32),
            np.asarray(h_old, dtype=np.int32), np.asarray(h_new, dtype=np.int32),
            np.asarray(h_weight, dtype=np.float64), connections, others, overrides,
        )  # fmt: skip

    # ------------------------------------------------------------------ nodes

    def node_type(self, i: int) -> Any:
        return self.node_types[self.ntype_list[i]]

    def node_attributes(self, i: int) -> dict:
        found = self.node_overrides.get(i)
        return found if found is not None else _derived_node_attributes(self.names[i], self.node_type(i))

    # ------------------------------------------------------------------ edges

    def _edges(self, i: int, reverse: bool) -> tuple[list[int], list[int], list[int], list[int]]:
        """Return (other end, key, kind, reference) of the edges leaving (``reverse``: entering) node ``i``."""
        if not reverse:
            a, b = int(self.out_ptr[i]), int(self.out_ptr[i + 1])
            if a == b:
                return [], [], [], []
            return (
                self.e_dst[a:b].tolist(),
                self.e_key[a:b].tolist(),
                self.e_kind[a:b].tolist(),
                self.e_ref[a:b].tolist(),
            )
        a, b = int(self.in_ptr[i]), int(self.in_ptr[i + 1])
        if a == b:
            return [], [], [], []
        edges = self.in_order[a:b]
        return (
            self.e_src[edges].tolist(),
            self.e_key[edges].tolist(),
            self.e_kind[edges].tolist(),
            self.e_ref[edges].tolist(),
        )

    def _edge_attributes(self, kind: int, ref: int) -> dict:
        if kind == _KIND_XREF:
            return self.connections[ref].attr
        if kind == _KIND_HISTORY:
            new_release = int(self.h_new[ref])
            return {
                "weight": float(self.h_weight[ref]),
                "old_release": int(self.h_old[ref]),
                "new_release": new_release if new_release >= 0 else _INF,
            }
        return self.other_attributes[ref]

    def adjacency(self, i: int, reverse: bool) -> dict[str, dict[int, dict]]:
        """Return ``{neighbour: {key: attributes}}`` of node ``i``, as networkx keeps it."""
        result: dict[str, dict[int, dict]] = {}
        names = self.names
        for other, key, kind, ref in zip(*self._edges(i, reverse)):
            result.setdefault(names[other], {})[key] = self._edge_attributes(kind, ref)
        return result

    def history_edges(self, i: int, reverse: bool) -> tuple[HistoryEdge, ...]:
        result = []
        names = self.names
        for other, key, kind, ref in zip(*self._edges(i, reverse)):
            if kind == _KIND_HISTORY:
                new_release = int(self.h_new[ref])
                result.append(
                    HistoryEdge(
                        names[other],
                        int(self.h_old[ref]),
                        new_release if new_release >= 0 else _INF,
                        float(self.h_weight[ref]),
                        key,
                    )
                )
            elif kind == _KIND_OTHER:
                attributes = self.other_attributes[ref]
                if "old_release" in attributes and "new_release" in attributes:
                    result.append(
                        HistoryEdge(
                            names[other],
                            attributes["old_release"],
                            attributes["new_release"],
                            attributes.get("weight", _NAN),
                            key,
                        )
                    )
        return tuple(result)

    def _connection_of(self, kind: int, ref: int) -> Connection | None:
        if kind == _KIND_XREF:
            return self.connections[ref]
        if kind == _KIND_OTHER:
            return _connection_or_none(self.other_attributes[ref].get(DB.connection_dict))
        return None

    def xref_edges(self, i: int, reverse: bool) -> tuple[tuple[str, Connection], ...]:
        result = []
        names = self.names
        for other, _key, kind, ref in zip(*self._edges(i, reverse)):
            connection = self._connection_of(kind, ref)
            if connection is not None:
                result.append((names[other], connection))
        return tuple(result)

    # ------------------------------------------------------------------ per-node summaries

    def combine(self, i: int, reverse: bool) -> dict[str, dict[int, int]] | None:
        """Sum the connections of the edges between node ``i`` and nodes of another type.

        Returns:
            ``{database: {assembly: release mask}}``, or ``None`` when the node has no such edge. The generic names
            of Ensembl relations (``ensembl_transcript`` ...) are replaced by the name of the relation on the
            assembly of the edge (``assembly_<key>_ensembl_transcript``).
        """
        own = self.ntype_list[i]
        ntype = self.ntype_list
        result: dict[str, dict[int, int]] | None = None
        for other, key, kind, ref in zip(*self._edges(i, reverse)):
            if ntype[other] == own or key != 0:
                continue
            connection = self._connection_of(kind, ref)
            if connection is None:
                raise KeyError(DB.connection_dict)
            if result is None:
                result = {}
            for database, assembly, releases in connection.items:
                form = DB.nts_ensembl_reverse.get(database)
                label = database if form is None else _assembly_label(form, assembly)
                per = result.setdefault(label, {})
                per[assembly] = per.get(assembly, 0) | releases.mask
        return result

    # ------------------------------------------------------------------ name indices

    def build_name_indices(self) -> None:
        """Build the two look-up tables of name resolution: by lower-case name, and with ``-``, ``_``, ``.`` folded."""
        lower_hash = np.empty(self.n, dtype=np.uint64)
        separator_hash: list[int] = []
        separator_idx: list[int] = []
        for i, name in enumerate(self.names):
            lower = name.lower()
            lower_hash[i] = _stable_hash(lower)
            if any(separator in lower for separator in _SEPARATORS):
                separator_hash.append(_stable_hash(lower.translate(_FOLD_SEPARATORS)))
                separator_idx.append(i)
        order = np.argsort(lower_hash, kind="stable")
        self.lower_hash = lower_hash[order]
        self.lower_idx = order.astype(np.int32)
        sep_hash = np.asarray(separator_hash, dtype=np.uint64)
        sep_idx = np.asarray(separator_idx, dtype=np.int32)
        order = np.argsort(sep_hash, kind="stable")
        self.separator_hash = sep_hash[order]
        self.separator_idx = sep_idx[order]

    def _candidates(self, hashes: np.ndarray, indices: np.ndarray, key: str, separator: bool) -> list[int]:
        value = np.uint64(_stable_hash(key))
        a = int(np.searchsorted(hashes, value, side="left"))
        found = []
        for b in range(a, len(hashes)):  # the names with this hash lie side by side: nearly always none or one
            if hashes[b] != value:
                break
            i = int(indices[b])
            lower = self.names[i].lower()
            if (lower.translate(_FOLD_SEPARATORS) if separator else lower) == key:
                found.append(i)
        return sorted(found, key=lambda j: self.names[j]) if len(found) > 1 else found

    def respelled(self) -> tuple[frozenset[int], frozenset[str]]:
        """Return the nodes that some database writes otherwise than they are named, and their names in lower case.

        These are the nodes of :py:attr:`spellings`; for every other node the name is its one spelling.
        """
        numbers = frozenset(number for number, _database in self.spellings)
        return numbers, frozenset(self.names[number].lower() for number in numbers)

    def spelling(self, number: int, database: str, release: int | None) -> str:
        """Return how a database writes a node at a release (``None``, or a release without it: at its latest)."""
        found = self.spellings.get((number, database))
        if found is None:
            return self.names[number]
        latest, otherwise = found
        if release is not None and release >= 0:
            for mask, text in otherwise:
                if mask >> release & 1:
                    return text
        return latest

    def lower_candidates(self, lower: str) -> list[int]:
        if self.lower_hash is None:
            self.build_name_indices()
        return self._candidates(self.lower_hash, self.lower_idx, lower, separator=False)  # type: ignore[arg-type]

    def separator_candidates(self, folded: str) -> list[int]:
        if self.separator_hash is None:
            self.build_name_indices()
        return self._candidates(self.separator_hash, self.separator_idx, folded, separator=True)  # type: ignore[arg-type]

    def lower_collisions(self) -> list[tuple[str, str]]:
        """Return pairs of different node names that are equal in lower case (none in a built graph)."""
        if self.lower_hash is None or self.lower_idx is None:
            self.build_name_indices()
        hashes, indices = self.lower_hash, self.lower_idx
        if hashes is None or indices is None:
            return []
        same = np.flatnonzero(hashes[1:] == hashes[:-1])
        found = []
        for position in same.tolist():
            first, second = self.names[int(indices[position])], self.names[int(indices[position + 1])]
            if first.lower() == second.lower():
                found.append((first, second))
        return found


def _backbone_type() -> str:
    return DB.external_search_settings["nts_backbone"]


def _derived_node_attributes(name: str, node_type: Any) -> dict:
    """Return the attributes that follow from a node's name and type."""
    if _has_id_and_version(node_type) and name.count(DB.id_ver_delimiter) == 1:
        base, version = name.split(DB.id_ver_delimiter)
        return {DB.node_type_str: node_type, "ID": base, "Version": version}
    return {DB.node_type_str: node_type}


def _connection_or_none(value: Any) -> Connection | None:
    """Return the connection of a ``{database: {assembly: releases}}`` value, or ``None`` if it is not one."""
    if not isinstance(value, Mapping):
        return None
    try:
        for per_database in value.values():
            if not isinstance(per_database, Mapping):
                return None
        return Connection.of_dict(value)
    except (TypeError, ValueError):
        return None


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


class TheGraph(nx.MultiDiGraph):
    """The identifier graph of one organism: Ensembl gene history plus cross-references, with typed accessors.

    Nodes are identifiers: versioned Ensembl genes (the *backbone*, with ``<id>.Void`` and ``<id>.Retired`` marking
    births and retirements), versionless base gene ids, transcripts, translations and identifiers of external
    databases. Two families of edges exist. *History edges* connect backbone genes and carry the release pair of the
    event and Ensembl's mapping score. *Cross-reference edges* (external id to Ensembl id, base id to versioned gene,
    transcript to gene, translation to transcript) carry a :py:class:`Connection`: the releases in which each
    database links the two nodes on each assembly.

    The class is built by :py:class:`idtrack.GraphMaker` and read by :py:class:`idtrack.Track`. A graph
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

    #: An external identifier linked to more genes than this in one release is hyper-connective (a family name, not
    #: an identity); see :py:attr:`hyperconnective_nodes`.
    genes_at_once_threshold = 5

    #: Upper bounds of the per-node caches of a frozen graph (entries).
    cache_sizes = {
        "adjacency": 200_000,
        "node": 200_000,
        "edges": 200_000,
        "summary": 200_000,
        "trios": 20_000,
        "names": 20_000,
    }

    def __init__(self, *args, **kwargs):
        """Create an empty, mutable graph; arguments are those of :py:class:`networkx.MultiDiGraph`.

        Args:
            args (Any): Positional arguments accepted by :py:meth:`networkx.MultiDiGraph.__init__`.
            kwargs (Any): Keyword arguments accepted by :py:meth:`networkx.MultiDiGraph.__init__`; they become graph
                attributes (``organism``, ``genome_assembly``, ``ensembl_release``, ``confident_for_release`` ...).
        """
        self._store: _Store | None = None
        self._reverse_view = False
        self.log = logging.getLogger(__name__)
        self.available_forms = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ state

    @property
    def is_frozen(self) -> bool:
        """Whether the graph has been frozen (see :py:meth:`freeze`)."""
        return self.__dict__.get("_store") is not None

    def _attach_included_forms(self, available_forms: list[str]) -> None:
        """Record which Ensembl forms (``gene``, ``transcript``, ``translation``) the graph contains.

        Args:
            available_forms (list[str]): The included forms, in the order they were merged.
        """
        self.available_forms = available_forms

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
        size = self.cache_sizes

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

    def calculate_caches(self, for_test: bool = False) -> None:
        """Freeze the graph and compute the few graph-wide tables that are not stored with it.

        Nothing per node is precomputed: the per-node values (:py:meth:`combined`, :py:meth:`active_ranges`,
        :py:meth:`trios`) are computed when asked for and kept in bounded caches. The graph-wide tables
        (:py:attr:`hyperconnective_nodes`, the lists of databases and assemblies, the name indices) come with a
        loaded graph; for a graph made in memory they are computed here.

        Args:
            for_test (bool): If ``True``, also compute :py:attr:`available_releases_given_database_assembly`, which
                scans every node and is needed by the integrity tests only. Defaults to ``False``.
        """
        store = self._frozen()
        if store.lower_hash is None:
            store.build_name_indices()
        _ = self.hyperconnective_nodes
        _ = self.available_external_databases
        _ = self.available_genome_assemblies
        _ = self.available_external_databases_assembly
        if for_test:
            _ = self.external_database_connection_form
            _ = self.available_releases_given_database_assembly

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
    def last_release(self) -> int:
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
            releases after GRCh38 took over (Ensembl maintains it in parallel); see :py:attr:`backbone_assembly` for
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
    def backbone_assembly(self) -> dict[int, int]:
        """The genome assembly that serves each release in the backbone of the graph.

        A built graph records it: the primary assembly where it has the release, otherwise the organism's preferred
        assembly among those that have it. For a graph made in memory it is derived from the edges in the same way,
        with the order of the organism table (:py:attr:`idtrack.DB.assembly_mysqlport_priority`).

        Returns:
            dict[int, int]: ``{release: assembly}`` for the releases that some assembly has.
        """
        recorded = self.graph.get("backbone_assembly")
        if recorded:
            return recorded
        return self._global("backbone_assembly", self._compute_backbone_assembly)

    def _compute_backbone_assembly(self) -> dict[int, int]:
        primary = self.graph.get("genome_assembly")
        table = DB.assembly_mysqlport_priority.get(self.graph.get("organism"), {})
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
            dict[int, ReleaseSet]: ``{assembly: releases}``: :py:attr:`backbone_assembly` read the other way round.
            A link of the graph counts at a release only on the assembly that has the release here
            (:py:func:`served_mask`).
        """
        store = self._frozen()
        line = store.globals.get("serving_line")
        if line is None:  # the recorded assignment read the other way round: kept, not announced as a computation
            line = store.globals["serving_line"] = serving_line(self.backbone_assembly)
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
                connection = _connection_or_none(store.other_attributes[int(store.e_ref[edge])].get(DB.connection_dict))
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
        :py:attr:`hyperconnective_nodes` is its part above :py:attr:`genes_at_once_threshold`.

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
        :py:attr:`genes_at_once_threshold` different genes (:py:meth:`genes_linked_at_once`: genes by stable id,
        whatever their versions; a transcript or a translation counts as its gene). Such ids are names of gene
        families and accessions shared by paralogues (RFAM families, ``U6``, ``5S_rRNA``, snoRNA names): they say
        what kind of gene something is, not which gene. Following them in a synonym search multiplies the candidates
        and bridges unrelated genes, so the search does not follow them.

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
        threshold = self.genes_at_once_threshold
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
            form = DB.nts_ensembl_reverse.get(node_type) if found is None else found.group(2)
            if form == DB.backbone_form:
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
                    form = DB.nts_ensembl_reverse.get(database)
                    result.add(database if form is None else _assembly_label(form, assembly))
        return result

    @property
    def available_genome_assemblies(self) -> set[int]:
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
                form = DB.nts_ensembl_reverse.get(database)
                label = database if form is None else _assembly_label(form, assembly)
                if label in external:
                    pairs.add((assembly, label))
        return {asm: {label for a, label in pairs if a == asm} for asm in self.available_genome_assemblies}

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
        forms = list(DB.forms_in_order)
        form_of_type = []
        for node_type in store.node_types:
            found = _ASSEMBLY_TYPE.match(node_type) if isinstance(node_type, str) else None
            form = DB.nts_ensembl_reverse.get(node_type) if found is None else found.group(2)
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
            raise ValueError(
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
        for assembly in sorted(self.available_genome_assemblies):
            for form in DB.forms_in_order:
                key = (_assembly_label(form, assembly), assembly)
                result[key] = set(ReleaseSet.from_mask(masks.get(key, 0)))
        for database in DB.nts_base_ensembl.values():
            result[(database, main_assembly)] = set(ReleaseSet.from_mask(masks.get((database, main_assembly), 0)))
        for form, database in DB.nts_ensembl.items():
            key = (_assembly_label(form, main_assembly), main_assembly)
            result[(database, main_assembly)] = set(ReleaseSet.from_mask(masks.get(key, 0)))
        return result

    # ------------------------------------------------------------------ nodes

    def node_type(self, node: str) -> str:
        """Return the type of a node (``ensembl_gene``, ``base_ensembl_gene``, ``external`` ...).

        Raises:
            KeyError: If the node is not in the graph, or (in graphs made by hand) has no type.
        """
        store = self._frozen()
        node_type = store.node_types[store.ntype_list[store.index[node]]]
        if node_type is None:
            raise KeyError(DB.node_type_str)
        return node_type

    def is_backbone(self, node: str) -> bool:
        """Tell whether the node is a gene of the history backbone (``<id>.Void`` and ``<id>.Retired`` included)."""
        store = self._frozen()
        return store.ntype_list[store.index[node]] == store.backbone_code

    def non_reference_releases(self, node: str) -> ReleaseSet:
        """Return the releases at which a gene lies outside the reference sequence of its genome assembly.

        Ensembl annotates genes on the chromosomes and scaffolds of an assembly (the reference sequence) and on
        regions beside it: alternative loci (haplotypes), assembly patches and LRG records. A gene on such a region
        is a copy of, or an alternative to, a gene of the reference sequence, with a stable id of its own. The
        builder takes the fact from Ensembl's core database of every release, for the assembly that serves the
        release in the backbone (:py:attr:`backbone_assembly`): the sequence region of the gene carries the
        attribute :py:attr:`idtrack.DB.non_reference_attribute`.

        Args:
            node (str): A node of the graph.

        Returns:
            ReleaseSet: The releases; empty for a gene of the reference sequence, for a node that is not a gene of
            the backbone, and for every node of a graph that was made in memory rather than built.

        Raises:
            KeyError: If the node is not in the graph.
        """
        store = self._frozen()
        return store.non_reference.get(store.index[node], _NO_RELEASES)

    def is_non_reference(self, node: str, release: int) -> bool:
        """Tell whether a gene lies outside the reference sequence at a release (:py:meth:`non_reference_releases`)."""
        return bool(self.non_reference_releases(node).mask >> release & 1) if release >= 0 else False

    def base_id(self, node: str) -> str:
        """Return the stable id of an Ensembl node, without its version.

        Raises:
            KeyError: If the node is not an Ensembl ``<id>.<version>`` node.
        """
        self._frozen()
        return self._c_node(node)["ID"]

    def version(self, node: str) -> str:
        """Return the version of an Ensembl node as a string (``"7"``, ``"Void"``, ``"Retired"``).

        Raises:
            KeyError: If the node is not an Ensembl ``<id>.<version>`` node.
        """
        self._frozen()
        return self._c_node(node)["Version"]

    def versions_of(self, stable_id: str) -> tuple[str, ...]:
        """Return the ``<id>.<version>`` nodes of a stable id, by ascending version.

        Genes have a base node that links their versions; transcripts and translations do not. This finds the
        versioned nodes of any Ensembl stable id by name (``<id>.Void`` and ``<id>.Retired`` are left out).

        Args:
            stable_id (str): A stable id without version, e.g. ``ENST00000489289``.

        Returns:
            tuple[str, ...]: The node names; empty if the graph has no version of the id.
        """
        store = self._frozen()
        prefix = f"{stable_id}{DB.id_ver_delimiter}"
        if store.names_sorted is None:
            names = store.names
            store.names_sorted = all(names[i] <= names[i + 1] for i in range(len(names) - 1))
        if store.names_sorted:
            first = bisect.bisect_left(store.names, prefix)
            candidates = []
            for name in itertools.islice(store.names, first, None):
                if not name.startswith(prefix):
                    break
                candidates.append(name)
        else:
            candidates = [name for name in store.names if name.startswith(prefix)]
        found = [
            name
            for name in candidates
            if name[len(prefix) :].isdigit() and store.type_has_version[store.ntype_list[store.index[name]]]
        ]
        return tuple(sorted(found, key=lambda name: int(name[len(prefix) :])))

    def spelling(self, node: str, database: str, release: int | None = None) -> str:
        """Return the spelling that ``database`` uses for an external identifier, at a release.

        The name of an external node is its canonical spelling: the one Ensembl gives at the latest release in which
        the identifier exists. Where databases, or the releases of one database, write an identifier with different
        letter case, the graph keeps one node and remembers how each database wrote it at each release.

        A release counts as the assembly that serves it in the backbone has it (:py:attr:`backbone_assembly`): an
        earlier assembly that Ensembl keeps beside the primary one (human GRCh37, frozen with the spellings of its
        time) does not supply the spelling of a release that the primary assembly serves.

        Args:
            node (str): An external node.
            database (str): The database whose spelling is wanted.
            release (int | None): The release at which the database is read. ``None``: the latest release at which
                the database has the identifier.

        Returns:
            str: The spelling of that database at that release; at a release at which the database does not have
            the identifier, its spelling at its latest release; the node name where neither differs from it.

        Raises:
            KeyError: If the node is not in the graph.
        """
        store = self._frozen()
        return store.spelling(store.index[node], database, release)

    def resolve_name(
        self,
        identifier: str,
        release: SupportsIndex | None = None,
        database: str | None = None,
        target: SupportsIndex | None = None,
    ) -> tuple[str | None, str | None]:
        """Find the node a user's string means at a release, and say how it was matched.

        This is the one place that says what a typed string means; the engine, the API, the inference of the source
        of a list and the self-tests all ask it, each naming the release it means.

        Two things decide, in this order:

        *Existence at the release the question is about.* Between the nodes the string can mean, one that exists at
        ``release`` (without one: at the last release of the graph) is taken before one that does not. Databases
        reuse strings: ``slit2`` was a synonym of one gene years ago and is, in another letter case, the name of
        another gene today; ``APC10`` was the name of one gene and is a synonym of another today. Which of the two
        is meant depends on the release asked about, not on how the string happens to be spelt. An Ensembl stable
        id (of a gene, with or without version, a transcript or a translation) is never passed over for an external
        identifier or synonym spelt the same, at whatever release it exists: it is an id of Ensembl's own history,
        which the graph follows through the releases. An external identifier exists at a release where a database
        has it on the genome assembly that serves the release (:py:attr:`serving_line`), as the engine counts a
        link: what an earlier assembly that Ensembl keeps listing beside the primary one says at such a release
        (human GRCh37, with its names of release 75 at every later release) does not make an identifier exist
        there, so a name that GRCh38 gave up and that is a synonym of a gene today is that synonym today. An
        identifier that only such an assembly ever has exists at no release and is still found (below). When the
        caller says which external database the string is
        an id of (``database``), "exists" means "is an id of that database at the release", and among the nodes
        that are not, one that is an id of that database at some release comes before one that never is.

        *The steps*, among the nodes that are equal in existence. They are tried in this order, each only if the
        one before found nothing. Each of the first four is tried on the names of the graph and then on its
        synonyms (the synonym node of the string, ``synonym_id::<string>``; the match is then reported as
        ``"synonym:<step>"``) before the next step begins, so that a string that is exactly a synonym of one gene is
        that synonym, not a spelling variant of another gene's name (``H3.3B``, a synonym, is not the name
        ``H3-3b``):

        1. ``"exact"``: the string is an identifier exactly as it is written. For an Ensembl id that is the name
           of its node. For an external identifier it is how a database (the one named, when ``database`` is given)
           writes it at the release the identifier is read at: ``release`` when the identifier exists there;
           without ``release`` the release a conversion reads it at, which is ``target`` when the identifier
           exists there and its latest release otherwise (the same when ``release`` does not hold it). Databases
           change the letter case of a name over the releases, and two databases may write one name in two ways
           at one release; the graph keeps one node per identifier, named as at its last release, and how each
           database wrote it at each release (:py:meth:`spelling`). So ``PRR2``, typed for a release at which a
           database wrote the name so, is that name exactly, although the node is called ``prr2`` after today's
           spelling, and it is not taken for another gene's synonym ``PRR2``; typed ``prr2`` for that release it
           is the same name by letter case.
        2. ``"case"``: it is a node name when letter case is ignored.
        3. ``"separator"``: it is a node name when ``-``, ``_`` and ``.`` are taken as the same character (and case
           is ignored): ``HLA_A`` and ``HLA.A`` for ``HLA-A``, ``ENSG00000123456_2`` for ``ENSG00000123456.2``.
        4. ``"version"``: the string ends in ``.<digits>`` and the identifier without that suffix is an id of a
           database of sequence accessions (:py:attr:`idtrack.DB.accession_database_prefixes`: RefSeq, UniProt,
           CCDS and the like) and of no symbol database of the organism
           (:py:attr:`idtrack.DB.symbol_databases`): ``NM_000546.9`` for ``NM_000546``. Nothing else is ever
           cut off the name of an external identifier: ``SLA.7`` and ``SLA_11`` are not ``SLA``, ``NM_713028147``
           is not ``NM``.
        5. ``"version_newer_than_graph"``: the string is an Ensembl id with a version higher than every version the
           graph holds of it (written with ``.``, ``_`` or ``-``), and the id is active at the last release of the
           graph: the version may belong to a release newer than the graph, so the string stands for the latest
           version the graph holds, ``ENSG00000123456.99`` for ``ENSG00000123456.7``. Any other version that the
           graph does not hold (a lower one, one in a gap, any version of a retired id) matches nothing;
           :py:meth:`known_versions` then names the versions that exist.

        When no node the string can mean exists at the release, the first one in the order of the steps is returned
        (with ``database``: the first that is an id of that database at all, else the first), and the caller learns
        from the node where it does exist. The cost is a few dictionary look-ups, whatever the string.

        Args:
            identifier (str): The string to resolve.
            release (SupportsIndex | None): The Ensembl release the question is about (for a conversion: the release
                the identifier is read at, ``from_release``). ``None`` means the last release of the graph. A
                release the graph does not have is one at which no external identifier exists.
            database (str | None): The external database the string is meant as an id of (for a conversion:
                ``from_database``). ``None``: any database.
            target (SupportsIndex | None): Only without ``release``: the release a conversion goes to
                (``to_release``), at which an identifier is read when it exists there. It says at which release
                the spelling of an identifier is judged in the step ``"exact"``; which node exists is asked at the
                last release of the graph all the same. ``None``: the last release of the graph.

        Returns:
            tuple[str | None, str | None]: The node name and how it was matched, or ``(None, None)``.

        Raises:
            TypeError: If ``identifier`` is not a string, or ``release`` or ``target`` is not an integer.
        """
        if not isinstance(identifier, str):
            raise TypeError(f"An identifier must be a string, got {type(identifier).__name__}: {identifier!r}.")
        at = release if release is None or type(release) is int else _release_number(release, "release")
        goal = target if target is None or type(target) is int else _release_number(target, "target")
        self._frozen()
        return self._c_resolved(identifier, at, database, goal if at is None else None)

    def _resolve_uncached(
        self, identifier: str, at: int | None, database: str | None, goal: int | None = None
    ) -> tuple[str | None, str | None]:
        """Do the work of :py:meth:`resolve_name` for checked arguments.

        Args:
            identifier (str): The string.
            at (int | None): The release asked about; ``None``: the last release of the graph.
            database (str | None): The database the string is meant as an id of.
            goal (int | None): Without ``at``, the release a conversion goes to; ``None``: the last release.

        Returns:
            tuple[str | None, str | None]: As :py:meth:`resolve_name`.
        """
        store = self._frozen()
        external, kinds, names = store.external_code, store.ntype_list, store.names
        spellings = ((identifier, ""), (f"{DB.synonym_id_nodes_prefix}{identifier}", "synonym:"))
        given = at
        candidates: Iterable[int]
        # the first match that does not exist at the release: of the database named at another release, and any
        elsewhere: list[tuple[str, str] | None] = [None, None]
        for step in _RESOLVE_STEPS:
            for query, prefix in spellings:
                if step == "exact":
                    # A node is named as its identifier is written at its last release. An Ensembl id is its name.
                    # Nearly every external identifier is written in one way by every database at every release
                    # and is then exactly its name too; the others are known, and for them the spelling of the
                    # release the identifier is read at decides (`_written_so`).
                    exact = store.index.get(query)
                    if exact is not None and kinds[exact] != external:
                        return names[exact], f"{prefix}{step}"
                    if exact is not None and exact not in self._c_respelled:
                        candidates = (exact,)
                    elif query.lower() in self._c_respelled_lower:
                        if at is None:
                            at = self._last_release_kept()
                        # the release a conversion reads the identifier at, when it exists there
                        wanted = at if given is not None or goal is None else goal
                        candidates = self._written_so(store, query, exact, wanted, database)
                    else:
                        continue
                else:
                    candidates = self._matches(store, query, step)
                for found in candidates:
                    name = names[found]
                    if kinds[found] != external:
                        return name, f"{prefix}{step}"
                    if at is None:  # no release named, and only now needed: the last release of the graph
                        at = self._c_last
                        if at is None:
                            at = self._last_release_kept()
                    here, known = self._exists_at(name, at, database)
                    if here:
                        return name, f"{prefix}{step}"
                    rank = 0 if known else 1
                    if elsewhere[rank] is None:
                        elsewhere[rank] = (name, f"{prefix}{step}")
        for match in elsewhere:
            if match is not None:
                return match
        unknown = self._unknown_version(identifier)
        if unknown is not None:
            asked, held = unknown
            latest = held[-1]
            if asked > int(self.version(latest)) and self.is_active(latest, self.last_release):
                return latest, "version_newer_than_graph"
        return None, None

    def _unknown_version(self, identifier: str) -> tuple[int, tuple[str, ...]] | None:
        """Read a string as an Ensembl id with a version that the graph does not hold.

        Args:
            identifier (str): The string, which is not a node.

        Returns:
            tuple[int, tuple[str, ...]] | None: The version asked for and the ``<id>.<version>`` nodes the graph
            holds of the id, ascending; ``None`` when the string is not an Ensembl id of the graph followed by
            ``.``, ``_`` or ``-`` and digits.
        """
        suffix = _VERSION_SUFFIX.match(identifier)
        if suffix is None:
            return None
        core, _separator, digits = suffix.groups()
        store = self._frozen()
        hit = store.index.get(core)
        if hit is None:
            candidates = store.lower_candidates(core.lower())
            hit = candidates[0] if candidates else None
        # A gene id without version is a node (in any letter case); other Ensembl ids are found by their versions.
        stable_id = store.names[hit] if hit is not None and store.node_type(hit) != DB.nts_external else core
        held = self.versions_of(stable_id)
        return (int(digits), held) if held else None

    def known_versions(self, identifier: str) -> tuple[str, ...]:
        """Return the versions the graph holds of an Ensembl id that was written with a version it does not hold.

        An Ensembl id with a version that never existed (or that the graph cannot know) is not answered with
        another version's gene (:py:meth:`resolve_name` matches nothing, unless the version may be newer than the
        graph); this method says which versions exist, for the error message.

        Args:
            identifier (str): The string as given, e.g. ``ENSG00000123456.99``.

        Returns:
            tuple[str, ...]: The ``<id>.<version>`` nodes of the id by ascending version; empty when the string is a
            node, or is not an Ensembl id of the graph followed by a version.

        Raises:
            TypeError: If ``identifier`` is not a string.
        """
        if not isinstance(identifier, str):
            raise TypeError(f"An identifier must be a string, got {type(identifier).__name__}: {identifier!r}.")
        if identifier in self._frozen().index:
            return ()
        unknown = self._unknown_version(identifier)
        return unknown[1] if unknown is not None else ()

    def _is_accession(self, store: _Store, number: int) -> bool:
        """Tell whether an external node is an id of an accession database and of no symbol database."""
        databases = {
            database for _other, connection in store.xref_edges(number, False) for database in connection.databases
        }
        symbols = set(DB.symbol_databases.get(self.graph.get("organism"), ()))
        return (
            bool(databases)
            and not databases & symbols
            and any(database.startswith(DB.accession_database_prefixes) for database in databases)
        )

    def _exists_at(self, name: str, release: int, database: str | None) -> tuple[bool, bool]:
        """Tell whether an external identifier exists at a release, and whether it is an id of a database at all.

        An identifier exists at a release where one of its links holds on the assembly that serves the release
        (:py:func:`served_mask`), as the engine reads a link and as the spelling of a release is chosen. What an
        assembly lists at a release it does not serve does not make an identifier exist there: human GRCh37 lists
        its names of release 75 at every later release, and a name that GRCh38 gave up long ago does not exist
        today. An identifier that only such an assembly ever has exists at no release in this sense; it is still
        found, as an identifier that exists elsewhere (:py:meth:`resolve_name`).

        Args:
            name (str): An external node.
            release (int): The release.
            database (str | None): The database the identifier must be an id of; ``None``: any.

        Returns:
            tuple[bool, bool]: Whether the node exists at the release (as an id of ``database`` when one is named),
            and whether it is an id of ``database`` at any release (``True`` without a database).
        """
        # The links are read as the engine reads them next (:py:meth:`xref_edges`, kept per node), so nothing is
        # computed twice.
        edges = self._c_xref(name, self._reverse_view)
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        if database is None:
            if release < 0:
                return False, True
            for _, connection in edges:
                if connection.releases.mask >> release & 1 and served_mask(connection.items, line) >> release & 1:
                    return True, True
            return False, True
        here = known = False
        for _, connection in edges:
            if database in connection.databases:
                known = True
                if release >= 0 and served_mask(connection.items, line, database) >> release & 1:
                    here = True
                    break
        return here, known

    def _last_release_kept(self) -> int:
        """Return the last release of a frozen graph, kept after the first resolution asked for it."""
        last = self._c_last
        if last is None:
            last = self._c_last = self.last_release
        return last

    def _written_so(self, store: _Store, query: str, exact: int | None, wanted: int, database: str | None) -> list[int]:
        """Return the external identifiers that a database writes exactly as ``query`` at the release they are read at.

        This is the step ``"exact"`` of :py:meth:`resolve_name` for the identifiers that some database writes
        otherwise than their node is named. The release is the one a conversion reads the identifier at: ``wanted``
        (the release asked about; without one the release the conversion goes to) when the identifier exists there,
        as an id of ``database`` when one is named; otherwise its latest release. A link counts at a release on
        the assembly that serves it (:py:func:`served_mask`); an identifier that no assembly has at a release it
        serves is judged where it is listed.

        Args:
            store (_Store): The frozen store of the graph.
            query (str): The string as typed (for a synonym: the name of its synonym node).
            exact (int | None): The node that is named ``query``, if there is one.
            wanted (int): The release at which the identifier is read if it exists there.
            database (str | None): The database the string is meant as an id of; ``None``: any.

        Returns:
            list[int]: The nodes (at most one in a built graph: identifiers that differ in letter case only are
            one node), each written as ``query`` at its release by a database that has it there.
        """
        respelled = self._c_respelled
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        written = []
        for number in (exact,) if exact is not None else store.lower_candidates(query.lower()):
            if number not in respelled:
                continue  # written in one way, and that is not `query` (the caller looked the name up)
            name = store.names[number]
            edges = self._c_xref(name, self._reverse_view)
            # The databases that have the identifier at the wanted release (few links of a node hold at one release).
            release, databases = wanted, set()
            if wanted >= 0:
                for _, connection in edges:
                    if connection.releases.mask >> wanted & 1:
                        for item_database in connection.databases if database is None else (database,):
                            if served_mask(connection.items, line, item_database) >> wanted & 1:
                                databases.add(item_database)
            if not databases:  # not at that release: at its latest one
                masks: dict[str, int] = {}
                for on_line in (True, False):
                    for _, connection in edges:
                        for item in connection.items:
                            if database is None or item[0] == database:
                                mask = served_mask((item,), line) if on_line else item[2].mask
                                if mask:
                                    masks[item[0]] = masks.get(item[0], 0) | mask
                    if masks:
                        break
                if not masks:  # not an id of the database named: nothing but its name says how it is written
                    if query == name:
                        written.append(number)
                    continue
                release = max(mask.bit_length() for mask in masks.values()) - 1
                databases = {item_database for item_database, mask in masks.items() if mask >> release & 1}
            if any(store.spelling(number, item_database, release) == query for item_database in databases):
                written.append(number)
        return written

    def _matches(self, store: _Store, query: str, step: str) -> list[int]:
        """Try one of the steps after ``"exact"`` on one string; return the numbers of the nodes it can mean.

        Args:
            store (_Store): The frozen store of the graph.
            query (str): The string, or its synonym node name.
            step (str): One of ``"case"``, ``"separator"``, ``"version"``.

        Returns:
            list[int]: The nodes in a fixed order (by name); empty when the step finds nothing.
        """
        lower = query.lower()
        if step == "case":
            return store.lower_candidates(lower)
        if step == "separator":
            if any(separator in query for separator in _SEPARATORS):
                return store.separator_candidates(lower.translate(_FOLD_SEPARATORS))
            return []
        suffix = _VERSION_SUFFIX.match(query)  # the step "version"
        if suffix and suffix.group(2) == DB.id_ver_delimiter:
            core = suffix.group(1)
            found = store.index.get(core)
            if found is None:
                candidates = store.lower_candidates(core.lower())
                found = candidates[0] if candidates else None
            if found is not None and store.node_type(found) == DB.nts_external and self._is_accession(store, found):
                return [found]
        return []

    # ------------------------------------------------------------------ edges

    def history_edges(self, node: str, reverse: bool = False) -> tuple[HistoryEdge, ...]:
        """Return the gene-history edges that leave a node (``reverse=True``: that enter it).

        Args:
            node (str): A node of the graph.
            reverse (bool): ``False`` for the edges towards later releases, ``True`` for those from earlier ones.

        Returns:
            tuple[HistoryEdge, ...]: One entry per edge, in the stored (deterministic) order; empty for nodes that
            are not backbone genes.
        """
        store = self._frozen()
        if node not in store.index:
            raise KeyError(node)
        return self._c_history(node, reverse != self._reverse_view)

    def xref_edges(self, node: str, reverse: bool = False) -> tuple[tuple[str, Connection], ...]:
        """Return the cross-reference edges that leave a node (``reverse=True``: that enter it).

        Args:
            node (str): A node of the graph.
            reverse (bool): ``False`` for edges from this node (external id to Ensembl id, base id to gene,
                transcript to gene, translation to transcript), ``True`` for edges to it.

        Returns:
            tuple[tuple[str, Connection], ...]: ``(other node, connection)`` per edge, in the stored order.
        """
        store = self._frozen()
        if node not in store.index:
            raise KeyError(node)
        return self._c_xref(node, reverse != self._reverse_view)

    def connection(self, source: str, target: str) -> Connection | None:
        """Return the connection of the cross-reference edge from ``source`` to ``target`` (``None`` if none)."""
        for other, connection in self.xref_edges(source):
            if other == target:
                return connection
        return None

    def get_next_edge_releases(self, from_id: str, reverse: bool) -> list:
        """List the releases at which the history of a backbone gene continues (or began).

        With ``reverse=False`` these are the releases at which the node is left: the ``old_release`` of its outgoing
        history edges, and infinity when it carries the self-loop of an id that is alive at the last release. With
        ``reverse=True`` these are the releases at which the node is entered: the ``new_release`` of its incoming
        history edges. Self-loops that record a re-mapping of the same id and version are not events.

        Args:
            from_id (str): A backbone gene node.
            reverse (bool): Direction, see above.

        Returns:
            list: The distinct releases, ascending (infinity last).

        Raises:
            ValueError: If ``from_id`` is not a backbone node.
        """
        if not self.is_backbone(from_id):
            raise ValueError(f"The method should be called only for backbone nodes: `{from_id}`.")
        releases = set()
        for edge in self.history_edges(from_id, reverse=reverse):
            if self.node_type(edge.node) != self.node_type(from_id):
                continue
            is_infinite = edge.new_release == _INF
            if edge.node == from_id and not (is_infinite and not reverse):
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
            _assembly_label(DB.backbone_form, assembly): {assembly: ReleaseSet.from_mask(mask)}
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
            KeyError: If a node that is not a backbone gene has no cross-reference edge.
        """
        self._frozen()
        return self._c_ranges(node)

    def _active_ranges_uncached(self, name: str) -> list[list]:
        if not self.is_backbone(name):
            summary = self._c_combined(name, "assembly" if self._family(name) == "assembly" else "out")
            mask = 0
            for per in summary.values():
                for releases in per.values():
                    mask |= releases.mask
            return TheGraph.list_to_ranges(sorted(ReleaseSet.from_mask(mask)))

        starts = self.get_next_edge_releases(from_id=name, reverse=True)
        ends = self.get_next_edge_releases(from_id=name, reverse=False)
        version = self._c_node(name).get("Version")
        if not starts and not ends:
            raise ValueError(
                f"Node {name!r} has no incoming or outgoing edges. "
                f"This may indicate an orphaned node or data integrity issue in the graph."
            )
        if not starts:
            if version != DB.no_old_node_id:
                raise ValueError(
                    f"The history of {name!r} has no edge that enters it: only `<id>.{DB.no_old_node_id}` nodes "
                    "may be without one."
                )
            starts = [self.first_release]
        elif not ends:
            if version != DB.no_new_node_id:
                raise ValueError(
                    f"The history of {name!r} has no edge that leaves it: only `<id>.{DB.no_new_node_id}` nodes "
                    "may be without one (a version ends with an outgoing edge or the self-loop of the last release)."
                )
            ends = [self.last_release]

        events = sorted(
            itertools.chain(zip(starts, itertools.repeat(True)), zip(ends, itertools.repeat(False))),
            key=lambda event: (event[0], not event[1]),
        )
        flat: list = []
        is_active = False
        for position, (release, is_start) in enumerate(events):
            if position == 0 and not is_start:
                raise ValueError(f"The range building should start with an in-edge: {name!r}.")
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
            if version != DB.no_new_node_id:
                raise ValueError(
                    f"The history of {name!r} has a range that starts at release {flat[-1]} and never ends: a "
                    "version must end with an outgoing edge or with the self-loop of the last release."
                )
            # A retired id stays retired until the last release of the graph.
            flat.append(max(self.last_release, flat[-1]))
        return [flat[i : i + 2] for i in range(0, len(flat), 2)]

    def active_releases(self, node: str) -> ReleaseSet:
        """Return the releases of the graph in which a node exists, as a :py:class:`ReleaseSet`."""
        self._frozen()
        return self._c_active(node)

    def _active_releases_uncached(self, name: str) -> ReleaseSet:
        last = self.last_release
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
        if self.is_backbone(name) and self._c_node(name).get("Version") in DB.alternative_versions:
            assembly = self.graph["genome_assembly"]
            label = _assembly_label(DB.backbone_form, assembly)
            found = set()
            for first, end in self.active_ranges(name):
                if not _is_release(first) or not 0 < first <= end:
                    raise ValueError(f"Invalid release range: l1={first}, l2={end}. Expected 0 < l1 <= l2.")
                if end != _INF and not _is_release(end):
                    raise ValueError(f"Unexpected error: {end!r} should be either np.inf or integer.")
                for release in range(int(first), (self.last_release if end == _INF else int(end)) + 1):
                    found.add((label, assembly, release))
            return frozenset(found)
        return frozenset(
            (database, assembly, release)
            for database, per in self.combined(name).items()
            for assembly, releases in per.items()
            for release in releases
        )

    def get_active_ranges_of_id_ensembl_all_inclusive(self, the_id: str) -> list[list]:
        """Return the release ranges in which a gene exists on any assembly.

        :py:meth:`active_ranges` of a backbone gene describes the backbone: each release as the organism's preferred
        assembly of that release has it. A gene can also exist at a release on another assembly (human GRCh37 keeps
        its genes in every release after GRCh38 took over). This method adds those releases. It also checks the two
        sources against each other: the releases in which the gene is referred to on the assembly that serves the
        backbone at that release must lie inside the backbone ranges.

        Args:
            the_id (str): A backbone gene or a gene of one assembly.

        Returns:
            list[list[int]]: Inclusive, ascending ranges of all releases in which the gene exists on any assembly.

        Raises:
            ValueError: If the cross-references contradict the backbone ranges, or if the node is not a gene.
        """
        node_type = self.node_type(the_id)
        if node_type == DB.nts_ensembl["gene"]:
            ranges = self.active_ranges(the_id)
            try:
                summary = self.combined(the_id)
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
                            raise ValueError(
                                f"Inconsistency in ID range vs combined edges: {the_id}, {ranges}, {per_assembly}."
                            )
                    else:
                        others.add(release)
            return TheGraph.list_to_ranges(sorted(set(self.ranges_to_list(ranges)) | others))
        if _is_assembly_gene(node_type):
            return self.active_ranges(the_id)
        raise ValueError(
            f"Cannot get active ranges for {the_id!r}: node type {node_type!r} is not a gene type. "
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
    def list_to_ranges(lst: list[int]) -> list[list]:
        """Turn an ascending list of distinct releases into inclusive ranges (``[1, 2, 3, 5]`` to ``[[1, 3], [5, 5]]``).

        Args:
            lst (list[int]): Releases, strictly increasing.

        Returns:
            list[list[int]]: The ranges that cover exactly these releases.
        """
        res = list()
        for _, a in itertools.groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
            b = list(a)
            res.append([b[0][1], b[-1][1]])
        return res

    def ranges_to_list(self, lor: list[list]) -> list[int]:
        """Turn inclusive ranges into the ascending list of their releases; infinity stands for the last release.

        Args:
            lor (list[list[int | float]]): Inclusive ``[first, last]`` pairs; ``last`` may be infinity.

        Returns:
            list[int]: The distinct releases of all ranges, ascending.
        """
        last = self.last_release
        return sorted({k for i, j in lor for k in range(i, (j if j != _INF else last) + 1)})

    @staticmethod
    def compact_ranges(list_of_ranges: list[list[int]]) -> list[list[int]]:
        """Merge ranges that touch or overlap; the argument is left unchanged.

        Args:
            list_of_ranges (list[list[int]]): Inclusive ranges ordered by their first release.

        Returns:
            list[list[int]]: New ranges that cover the same releases with as few ranges as possible.
        """
        result: list[list[int]] = []
        for first, last in list_of_ranges:
            if result and result[-1][1] + 1 >= first:
                result[-1][1] = max(result[-1][1], last)
            else:
                result.append([first, last])
        return result

    @staticmethod
    def get_intersecting_ranges(lor1: list[list[int]], lor2: list[list[int]], compact: bool = True) -> list[list[int]]:
        """Return the ranges that two lists of ranges have in common.

        Args:
            lor1 (list[list[int]]): Inclusive, ascending, non-overlapping ranges.
            lor2 (list[list[int]]): Ranges of the same kind.
            compact (bool): Merge touching ranges of the result (default).

        Returns:
            list[list[int]]: The common ranges; empty when there are none.
        """
        result = [
            [max(first[0], second[0]), min(first[1], second[1])]
            for first in lor1
            for second in lor2
            if max(first[0], second[0]) <= min(first[1], second[1])
        ]
        return TheGraph.compact_ranges(result) if compact else result

    @staticmethod
    def is_point_in_range(lor: list[list], p: int) -> bool:
        """Tell whether a release lies in one of the ranges.

        Args:
            lor (list[list[int]]): Inclusive ranges.
            p (int): The release.

        Returns:
            bool: ``True`` if some range contains ``p``.
        """
        for l1, l2 in lor:
            if l1 <= p <= l2:
                return True
        return False

    def get_two_nodes_coinciding_releases(self, id1: str, id2: str, compact: bool = True) -> list[list]:
        """Return the release ranges in which two nodes exist at the same time.

        Args:
            id1 (str): A node.
            id2 (str): Another node.
            compact (bool): Merge touching ranges of the result (default).

        Returns:
            list[list[int]]: The common ranges; empty when the nodes never coexist.
        """
        return TheGraph.get_intersecting_ranges(self.active_ranges(id1), self.active_ranges(id2), compact=compact)

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
        form = DB.nts_ensembl_reverse.get(database)
        key = (database if form is None else _assembly_label(form, assembly), assembly, release)
        found = []
        for name in store.names:
            if self.is_backbone(name) and self._c_node(name).get("Version") in DB.alternative_versions:
                continue
            try:
                trios = self.trios(name)
            except KeyError:
                continue
            if key in trios:
                found.append(name)
        return found

    def get_external_database_nodes(self, database_name: str) -> set[str]:
        """Return the external identifiers that a database has in any release and on any assembly.

        Args:
            database_name (str): One of :py:attr:`available_external_databases`.

        Returns:
            set[str]: Node names.
        """
        store = self._frozen()
        src, _dst, ref, connections = self._xref_arrays()
        codes = [code for code in np.unique(ref).tolist() if database_name in connections[code].databases]
        chosen = src[np.isin(ref, codes) & (store.ntype[src] == store.external_code)]
        return {store.names[i] for i in np.unique(chosen).tolist()}

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
            raise ValueError(
                "Only graphs made of history edges and cross-reference edges, with nodes that carry nothing but "
                "their type, can be saved."
            )
        self.calculate_caches(for_test=False)
        _ = self.external_database_connection_form
        for name in store.names:
            if _NAME_SEPARATOR in name:
                raise ValueError(f"A node name contains a NUL character: {name!r}.")

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
        folder = os.path.dirname(os.path.abspath(path))
        handle = tempfile.NamedTemporaryFile(
            dir=folder, prefix=os.path.basename(path) + ".", suffix=".tmp", delete=False
        )
        try:
            with handle:
                np.savez(handle, **arrays)  # type: ignore[arg-type]
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, path)
        except BaseException:
            if os.path.exists(handle.name):
                os.remove(handle.name)
            raise

    @classmethod
    def load(cls, path: str, expect: Mapping[str, Any] | None = None) -> "TheGraph":
        """Read a graph file written by :py:meth:`save`. Needs no network and no :py:class:`DatabaseManager`.

        Args:
            path (str): The graph file.
            expect (Mapping[str, Any] | None): Fields of the manifest that the graph must have, e.g.
                ``{"organism": "homo_sapiens", "genome_assembly": 38}``. See
                :py:meth:`idtrack.GraphMaker.graph_identity`.

        Returns:
            TheGraph: The frozen graph.

        Raises:
            FileNotFoundError: If there is no such file.
            GraphMismatchError: If the file cannot be read as a graph file, was written by another version of the
                graph format, or differs from ``expect`` (the message names the fields).
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No graph file at `{path}`.")
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            document = json.loads(arrays["document"].tobytes().decode("utf-8"))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as error:
            raise GraphMismatchError(
                f"`{path}` cannot be read as an idtrack graph file ({type(error).__name__}: {error}). {_REBUILD_HINT}"
            ) from error
        if document.get("format_version") != GRAPH_FORMAT_VERSION:
            raise GraphMismatchError(
                f"`{path}` has graph format version {document.get('format_version')!r}; this version of idtrack "
                f"reads version {GRAPH_FORMAT_VERSION}. {_REBUILD_HINT}"
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
                    f"that the file holds, or build this one. {_REBUILD_HINT}"
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
            raise GraphMismatchError(
                f"`{path}` does not hold what a graph file of format version {GRAPH_FORMAT_VERSION} holds "
                f"({type(error).__name__}: {error}). {_REBUILD_HINT}"
            ) from error

        graph = cls()
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


_STORED_GLOBALS = (
    "shared_external_ids",
    "available_external_databases",
    "available_genome_assemblies",
    "available_external_databases_assembly",
    "external_database_connection_form",
)


def _masks_to_words(masks: list[int]) -> np.ndarray:
    """Return release bit masks as rows of 64-bit words (lowest word first), as a graph file stores them."""
    words = max(1, max((mask.bit_length() for mask in masks), default=1) + 63 >> 6)
    rows = np.zeros((len(masks), words), dtype=np.uint64)
    for row, mask in enumerate(masks):
        for word in range(words):
            rows[row, word] = (mask >> (64 * word)) & 0xFFFFFFFFFFFFFFFF
    return rows


def _words_to_masks(rows: np.ndarray) -> list[int]:
    """Return the release bit masks stored as rows of 64-bit words."""
    return [sum(int(word) << (64 * position) for position, word in enumerate(row)) for row in rows.tolist()]


def _to_json(value: Any) -> Any:
    """Turn graph attributes into JSON values (sets become sorted lists, integer keys become strings)."""
    if isinstance(value, Mapping):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_to_json(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return value


def _from_json_graph(attributes: dict) -> dict:
    """Restore the types of the graph attributes read from a file."""
    result = dict(attributes)
    if "misplaced_external_entry" in result:
        result["misplaced_external_entry"] = set(result["misplaced_external_entry"])
    if "backbone_assembly" in result:
        result["backbone_assembly"] = {int(k): int(v) for k, v in result["backbone_assembly"].items()}
    return result


def _from_json_globals(stored: dict) -> dict:
    """Restore the types of the graph-wide tables read from a file."""
    return {
        "shared_external_ids": {str(k): int(v) for k, v in stored["shared_external_ids"].items()},
        "available_external_databases": set(stored["available_external_databases"]),
        "available_genome_assemblies": {int(a) for a in stored["available_genome_assemblies"]},
        "available_external_databases_assembly": {
            int(a): set(names) for a, names in stored["available_external_databases_assembly"].items()
        },
        "external_database_connection_form": dict(stored["external_database_connection_form"]),
    }


__all__ = ["TheGraph", "ReleaseSet", "Connection", "HistoryEdge", "GraphMismatchError", "GRAPH_FORMAT_VERSION"]
