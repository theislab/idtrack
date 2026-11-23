"""Value types of the graph: release sets, cross-reference records, history edges, and the serving line.

:py:class:`ReleaseSet` (a set of Ensembl releases kept as a bit mask), :py:class:`Connection` (which databases link two
nodes on which assemblies at which releases), :py:class:`HistoryEdge` (one event of Ensembl's id history), and the one
rule for where a cross-reference counts (:py:func:`served_mask`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np

from idtrack import ids
from idtrack.errors import IntegrityError

_INF = float("inf")
_NAN = float("nan")
# The kinds of edges of a store: an event of the id history, a cross-reference, anything else (graphs made by hand).
_KIND_HISTORY, _KIND_XREF, _KIND_OTHER = 0, 1, 2


class ReleaseSet(frozenset):
    """An immutable set of Ensembl releases with its bit mask, one object per distinct value.

    It is a :py:class:`frozenset` (membership, iteration, ``|``, ``&``, ``min``, ``max`` and comparison with any set
    work as usual) that also carries :py:attr:`mask`, an ``int`` whose bit ``r`` is set when release ``r`` is in the
    set, so that ``mask >> r & 1`` tests membership and ``|``/``&`` on masks combine sets without building them. Equal
    sets are the same object: a graph with millions of edges holds a few thousand of them.
    """

    __slots__ = ("mask",)
    mask: int
    _interned: dict[int, ReleaseSet] = {}

    @classmethod
    def from_mask(cls, mask: int) -> ReleaseSet:
        """Return the set of the releases whose bits are set in ``mask``."""
        found = cls._interned.get(mask)
        if found is None:
            if mask < 0:
                raise IntegrityError("A release mask cannot be negative.")
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

    def ranges(self) -> tuple[tuple[int, int], ...]:
        """Return the inclusive ranges of consecutive releases of the set, ascending (``{1, 2, 3, 5}``: ``(1, 3), (5, 5)``).

        Returns:
            tuple[tuple[int, int], ...]: ``(first, last)`` of every run of consecutive releases.
        """
        ranges = []
        mask = self.mask
        while mask:
            low = (mask & -mask).bit_length() - 1
            length = ((mask >> low) & ~((mask >> low) + 1)).bit_length()  # the run of set bits that starts at ``low``
            ranges.append((low, low + length - 1))
            mask &= ~(((1 << length) - 1) << low)
        return tuple(ranges)

    @classmethod
    def of(cls, releases: Iterable[int]) -> ReleaseSet:
        """Return the set of the given releases (non-negative integers)."""
        if isinstance(releases, ReleaseSet):
            return releases
        mask = 0
        for release in releases:
            if isinstance(release, bool) or not isinstance(release, (int, np.integer)) or release < 0:
                raise IntegrityError(f"A release must be a non-negative integer, got {release!r}.")
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
    _interned: dict[tuple, Connection] = {}

    @classmethod
    def of(cls, items: Iterable[tuple[str, int, Any]]) -> Connection:
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
                    raise IntegrityError(f"Database {db!r} and assembly {asm!r} occur twice in one connection.")
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
    def of_dict(cls, connection: Mapping) -> Connection:
        """Return the connection of a ``{database: {assembly: releases}}`` mapping, in its own order."""
        return cls.of((db, asm, rs) for db, per_db in connection.items() for asm, rs in per_db.items())

    @property
    def attr(self) -> dict:
        """The attributes of the edge as networkx shows them.

        ``connection`` is :py:attr:`as_dict` and ``available_releases`` is :py:attr:`releases`: the same shared,
        immutable objects that the accessors return.
        """
        if self._attr is None:
            self._attr = {ids.CONNECTION_KEY: self.as_dict, "available_releases": self.releases}
        return self._attr

    def __reduce__(self):
        """Pickle and copy through the triples, so that a restored connection is the interned object."""
        return Connection.of, (tuple((db, asm, tuple(sorted(rs))) for db, asm, rs in self.items),)

    def __repr__(self) -> str:
        """Show the triples."""
        return f"Connection({[(db, asm, sorted(rs)) for db, asm, rs in self.items]})"


def serving_line(serving_assembly: Mapping[int, int]) -> dict[int, ReleaseSet]:
    """Return, per genome assembly, the releases that the assembly serves in the backbone.

    A release belongs to one assembly: the one that serves it in the backbone of the graph
    (:py:attr:`TheGraph.serving_assembly`). An earlier assembly that Ensembl keeps listing beside the primary one
    (human GRCh37, frozen with its annotation of release 75 and listed at every later release) serves none of the
    releases that the primary assembly has.

    Args:
        serving_assembly (Mapping[int, int]): ``{release: assembly}``.

    Returns:
        dict[int, ReleaseSet]: ``{assembly: releases}`` for the assemblies that serve a release.
    """
    masks: dict[int, int] = {}
    for release, assembly in serving_assembly.items():
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


class Place(NamedTuple):
    """Where a gene lies: on an assembly, a sequence region, from start to end (1-based, inclusive), on a strand.

    The fields: ``assembly``, the key of the assembly (:py:data:`idtrack.organisms.ORGANISM_TABLE`); ``region``, the
    name of the sequence region (a chromosome, a scaffold, a patch); ``start`` and ``end``, the first and last base of
    the gene as Ensembl counts them; ``strand``, ``1`` or ``-1``; ``releases``, the releases at which the gene lies
    there on the assembly that serves the release; ``kind``, what Ensembl's tables say the sequence region is, one of
    :py:data:`idtrack.graph.genes.PLACE_KINDS` (a chromosome, the mitochondrial genome, or a scaffold or another
    sequence below a chromosome; ``"scaffold"`` where the graph does not say). Whether two places overlap, and so
    whether two genes lie at one place, is the rule of :py:func:`idtrack.graph.genes.overlap`; the kind lets two places
    on different assemblies be compared (:py:func:`idtrack.graph.genes.relation`).
    """

    assembly: int
    region: str
    start: int
    end: int
    strand: int
    releases: ReleaseSet
    kind: str = "scaffold"


class AssemblyException(NamedTuple):
    """Which region of the reference sequence a patch, a haplotype or a pseudoautosomal region stands for.

    One row of Ensembl's ``assembly_exception`` table (:py:meth:`idtrack.DatabaseManager.create_assembly_exceptions`)
    at the releases that list it. The fields: ``assembly``, the key of the assembly; ``region``, ``start`` and
    ``end``, the bases of the sequence region outside the reference sequence (or of the pseudoautosomal region of Y);
    ``type``, Ensembl's type of the exception (``PATCH_FIX``, ``PATCH_NOVEL``, ``HAP``, ``PAR``); ``ref_region``,
    ``ref_start`` and ``ref_end``, the region of the reference sequence it stands for; ``ref_kind``, what that region
    is (:py:data:`idtrack.graph.genes.PLACE_KINDS`); ``releases``, the releases that list it, on the assembly that
    serves them; ``carried``, the reference region carried onto later assemblies by Ensembl's mapping of the
    assemblies, ``((assembly, region, start, end), ...)`` (decision 103).
    """

    assembly: int
    region: str
    start: int
    end: int
    type: str
    ref_region: str
    ref_start: int
    ref_end: int
    ref_kind: str
    releases: ReleaseSet
    carried: tuple[tuple[int, str, int, int], ...] = ()


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
    return isinstance(node_type, str) and node_type in ids.FORM_OF_ENSEMBL_TYPE


def _assembly_label(form: str, assembly: int) -> str:
    """Return the label of the Ensembl ids of a form on one assembly (``assembly_<key>_ensembl_<form>``)."""
    return f"assembly_{assembly}_ensembl_{form}"


def _is_release(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
