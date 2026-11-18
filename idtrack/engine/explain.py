"""The explained path of a conversion: the steps the search took, typed, and the nodes they pass, as data.

The search records each path as a tuple of raw steps: ``(source, target, key)`` for a history edge,
``(source, target, 0, release)`` for a cross-reference edge used at a release (``None`` for the final hop of
confidence 1), and ``(None, gene, None)`` for a gene that is its own answer. :py:class:`PathBuilder` turns them into
:py:class:`idtrack.results.PathStep` and :py:class:`idtrack.results.PathNode` records with what the graph says about
each (releases, scores, splits and merges, databases and assemblies), every step and node once per conversion.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from idtrack import ids
from idtrack.errors import IntegrityError
from idtrack.results import ExplainedPath, PathNode, PathStep

if TYPE_CHECKING:
    from idtrack.graph.graph import TheGraph


def _finite(value: float) -> int | None:
    return None if value is None or (isinstance(value, float) and not math.isfinite(value)) else int(value)


class PathBuilder:
    """Collect the steps and nodes of every candidate of one conversion.

    Args:
        graph: The graph.
        reverse: Whether the walk goes back in time (a history step ``(u, v, key)`` then follows the edge
            ``v -> u``).
        to_release: The target release.
    """

    def __init__(self, graph: TheGraph, reverse: bool, to_release: int):
        """Start an empty explanation."""
        self.graph = graph
        self.reverse = reverse
        self.to_release = to_release
        self._steps: list[PathStep] = []
        self._step_index: dict[tuple, int] = {}
        self._nodes: dict[str, PathNode] = {}
        self._serving = graph.serving_assembly
        # The releases each assembly serves, as bit masks (``serving_assembly`` read the other way round).
        self._serving_masks = tuple((int(assembly), releases.mask) for assembly, releases in graph.serving_line.items())
        self._last = graph.snapshot_release
        self._real: dict[str, bool] = {}

    # ------------------------------------------------------------------ nodes

    def _node(self, name: str) -> None:
        if name in self._nodes:
            return
        graph = self.graph
        node_type = graph.node_type(name)
        kind = str(node_type)
        databases: tuple[str, ...] = ()
        active: tuple = ()
        assemblies: tuple[int, ...] = ()
        non_reference: tuple = ()
        if node_type == ids.EXTERNAL:
            if name.startswith(ids.SYNONYM_PREFIX):
                kind = "synonym"
            found: set[str] = set()
            for _, connection in graph.xref_edges(name):
                found.update(database for database, _, _ in connection.items)
            databases = tuple(sorted(found))
        elif node_type == ids.GENE and self._is_real(name):
            ranges = graph.active_ranges(name)
            active = tuple((int(a), _finite(b)) for a, b in ranges)
            alive = 0  # the releases of the ranges up to the last release of the graph, as a bit mask
            for a, b in ranges:
                low, high = int(a), int(min(b, self._last))
                if high >= low:
                    alive |= (1 << (high + 1)) - (1 << low)
            assemblies = tuple(sorted(assembly for assembly, served in self._serving_masks if served & alive))
            outside = graph.non_reference_releases(name)
            if outside.mask:
                non_reference = tuple((int(a), int(b)) for a, b in graph.list_to_ranges(sorted(outside)))
        self._nodes[name] = PathNode(
            id=name,
            kind=kind,
            label=name,
            databases=databases,
            active=active,
            assemblies=assemblies,
            non_reference=non_reference,
        )

    def _is_real(self, name: str) -> bool:
        """Whether a backbone node is a gene id with a version (not ``<id>.Void`` or ``<id>.Retired``)."""
        real = self._real.get(name)
        if real is None:
            real = self._real[name] = self.graph.version(name) not in ids.SENTINEL_VERSIONS
        return real

    # ------------------------------------------------------------------ steps

    def _history(self, kind: str, source: str, target: str, key: int) -> PathStep:
        graph = self.graph
        # The walk's step (u, v) follows the edge u -> v forward in time and v -> u backward.
        tail, head = (target, source) if self.reverse else (source, target)
        edge = next((e for e in graph.history_edges(tail) if e[0] == head and e[4] == key), None)
        if edge is None:  # pragma: no cover - the search only takes edges of the graph
            raise IntegrityError(f"No history edge {tail} -> {head} (key {key}) in the graph.")
        _, old, new, weight, _ = edge
        splits = merges = 0
        if kind == "history":
            splits = sum(
                1
                for e in graph.history_edges(tail)
                if e[1] == old and e[0] != head and e[0] != tail and math.isfinite(e[2]) and self._is_real(e[0])
            )
            merges = sum(
                1
                for e in graph.history_edges(head, reverse=True)
                if e[2] == new and e[0] != head and e[0] != tail and self._is_real(e[0])
            )
        weight_value = None if weight != weight else float(weight)
        new_release = _finite(new)
        if kind == "self_loop":
            reason = "self_loop"
        elif splits and merges:
            reason = "split_and_merge"
        elif splits:
            reason = "split"
        elif merges:
            reason = "merge"
        elif ids.split(tail)[0] == ids.split(head)[0]:
            reason = "new_version"
        else:
            reason = "replaced"
        old_release = _finite(old)
        return PathStep(
            kind=kind,
            source=source,
            target=target,
            key=int(key),
            old_release=old_release,
            new_release=new_release,
            weight=weight_value,
            splits=splits,
            merges=merges,
            assembly=None if new_release is None else self._assembly(new_release),
            old_assembly=None if old_release is None else self._assembly(old_release),
            reason=reason,
        )

    def _assembly(self, release: int) -> int | None:
        assembly = self._serving.get(release)
        return None if assembly is None else int(assembly)

    def _xref(self, kind: str, source: str, target: str, release: int | None) -> PathStep:
        graph = self.graph
        connection = graph.connection(source, target) or graph.connection(target, source)
        databases: tuple[str, ...] = ()
        assembly = None
        if release is not None:
            assembly = self._assembly(release)
            if connection is not None:
                databases = tuple(
                    sorted(
                        {
                            database
                            for database, edge_assembly, releases in connection.items
                            if release in releases and (assembly is None or int(edge_assembly) == assembly)
                        }
                    )
                )
        elif connection is not None:
            databases = tuple(sorted({d for d, _, _ in connection.items}))
        reason = "final_other_release" if kind == "final" and release is None else kind
        return PathStep(
            kind=kind,
            source=source,
            target=target,
            key=None,
            release=None if release is None else int(release),
            databases=databases,
            assembly=assembly,
            reason=reason,
        )

    def _add(self, raw: tuple, kind: str) -> int:
        token = (kind, *raw)
        index = self._step_index.get(token)
        if index is not None:
            return index
        if kind == "identity":
            step = PathStep(kind="identity", source=None, target=raw[1], reason="identity")
        elif kind in ("history", "self_loop"):
            step = self._history(kind, raw[0], raw[1], raw[2])
        else:
            step = self._xref(kind, raw[0], raw[1], raw[3])
        for name in (step.source, step.target):
            if name is not None:
                self._node(name)
        index = len(self._steps)
        self._steps.append(step)
        self._step_index[token] = index
        return index

    def _inward(self, source: str, target: str) -> bool:
        """Whether a cross-reference step walks towards the gene (along the edge ``source -> target``)."""
        return any(other == target for other, _ in self.graph.xref_edges(source))

    def search_path(self, raw_path: tuple[Any, ...]) -> tuple[int, ...]:
        """Return the indices of the steps of a path from the identifier to a gene.

        Args:
            raw_path: The engine's steps.

        Returns:
            tuple[int, ...]: Indices into :py:attr:`steps`.
        """
        indices = []
        # A path from a gene (a gene id with version as the query) has no entry steps: a cross-reference step that
        # leaves it is a bridge. Any other query enters through cross-reference steps up to its first gene.
        first = raw_path[0] if raw_path else None
        seen_gene = first is not None and first[0] is not None and self.graph.is_backbone(first[0])
        for raw in raw_path:
            if raw[0] is None:
                indices.append(self._add(raw, "identity"))
                seen_gene = True
            elif len(raw) == 3:
                indices.append(self._add(raw, "self_loop" if raw[0] == raw[1] else "history"))
                seen_gene = True
            elif not seen_gene:
                indices.append(self._add(raw, "entry"))
                if self.graph.is_backbone(raw[1]):
                    seen_gene = True
            else:
                indices.append(self._add(raw, "bridge_far" if self._inward(raw[0], raw[1]) else "bridge_near"))
        return tuple(indices)

    def final_path(self, raw_path: tuple[Any, ...]) -> tuple[int, ...]:
        """Return the indices of the steps of a final hop.

        Args:
            raw_path: The engine's steps from the gene to the target.

        Returns:
            tuple[int, ...]: Indices into :py:attr:`steps`.
        """
        return tuple(self._add(raw, "final") for raw in raw_path)

    def result(self) -> ExplainedPath:
        """Return the explanation collected.

        Returns:
            ExplainedPath: Nodes and steps.
        """
        return ExplainedPath(nodes=tuple(self._nodes.values()), steps=tuple(self._steps))
