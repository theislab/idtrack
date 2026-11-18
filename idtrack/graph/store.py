"""The array store of a graph and the pieces of its file format.

A frozen graph keeps its content in :py:class:`_Store`: node names and types, edge arrays, a table of distinct
connections, the genes outside the reference sequence, the name indices and the per-release spellings of external ids.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

from idtrack import ids
from idtrack.errors import IntegrityError
from idtrack.graph.types import (
    _INF,
    _KIND_HISTORY,
    _KIND_OTHER,
    _KIND_XREF,
    _NAN,
    Connection,
    HistoryEdge,
    ReleaseSet,
    _assembly_label,
    _has_id_and_version,
    _is_assembly_gene,
    _is_release,
    _stable_hash,
)

_NAME_SEPARATOR = "\x00"
_FOLD_SEPARATORS = str.maketrans("-.", "__")
_SEPARATORS = "-_."


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
            raise IntegrityError("Node names of a graph must be unique.")
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
            raise IntegrityError("Edges must be ordered by their source node, and `in_order` by their target node.")
        self.out_ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.e_src, minlength=self.n), out=self.out_ptr[1:])
        self.in_ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.e_dst, minlength=self.n), out=self.in_ptr[1:])

        self.backbone_code = node_types.index(_backbone_type()) if _backbone_type() in node_types else -1
        self.external_code = node_types.index(ids.EXTERNAL) if ids.EXTERNAL in node_types else -1
        self.type_has_version = [_has_id_and_version(t) for t in node_types]
        self.type_is_assembly_gene = [_is_assembly_gene(t) for t in node_types]

        self.lower_hash: np.ndarray | None = None
        self.lower_idx: np.ndarray | None = None
        self.separator_hash: np.ndarray | None = None
        self.separator_idx: np.ndarray | None = None
        self.globals: dict[str, Any] = {}
        self.spellings: dict[tuple[int, str], tuple[str, tuple[tuple[int, str], ...]]] = {}
        self.non_reference: dict[int, ReleaseSet] = {}
        # The node names in sorted order, made at the first lookup by prefix (:py:meth:`names_with_prefix`).
        self.sorted_names: list[str] | None = None

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
    ) -> _Store:
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
    def from_networkx(cls, graph: nx.MultiDiGraph) -> _Store:
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
                raise IntegrityError(f"Node names of a graph must be strings, got {name!r}.")
            node_type = attributes.get(ids.NODE_TYPE_KEY)
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
                        raise IntegrityError(
                            f"Edge keys must be integers, got {key!r} between {source!r} and {target!r}."
                        )
                    edge_number[(si, ti, int(key))] = len(e_src)
                    e_src.append(si)
                    e_dst.append(ti)
                    e_key.append(int(key))
                    connection = None
                    if (
                        ids.CONNECTION_KEY in attributes
                        and key == 0
                        and all(k in (ids.CONNECTION_KEY, "available_releases") for k in attributes)
                    ):
                        connection = _connection_or_none(attributes[ids.CONNECTION_KEY])
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
            return _connection_or_none(self.other_attributes[ref].get(ids.CONNECTION_KEY))
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
                raise KeyError(ids.CONNECTION_KEY)
            if result is None:
                result = {}
            for database, assembly, releases in connection.items:
                form = ids.FORM_OF_ENSEMBL_TYPE.get(database)
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

    def names_with_prefix(self, prefix: str) -> list[str]:
        """Return the node names that begin with *prefix*, in sorted order (a binary search in the sorted names)."""
        if self.sorted_names is None:
            self.sorted_names = sorted(self.names)
        names = self.sorted_names
        found = []
        for position in range(bisect.bisect_left(names, prefix), len(names)):  # from the first match on, no scan
            name = names[position]
            if not name.startswith(prefix):
                break
            found.append(name)
        return found

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
    return ids.GENE


def _derived_node_attributes(name: str, node_type: Any) -> dict:
    """Return the attributes that follow from a node's name and type."""
    if _has_id_and_version(node_type) and ids.is_versioned_name(name):
        base, version = ids.split(name)
        return {ids.NODE_TYPE_KEY: node_type, "ID": base, "Version": version}
    return {ids.NODE_TYPE_KEY: node_type}


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
