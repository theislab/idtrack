"""Name resolution of a graph: which node a string means at a release, and which genes a name names.

The methods of :py:class:`NameResolution` are part of :py:class:`idtrack.graph.TheGraph`; they read its store only.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, NamedTuple, SupportsIndex

from idtrack import _arguments, ids, organisms, reading, settings
from idtrack.errors import InvalidArgumentError, UnknownIdentifierError
from idtrack.graph.store import _FOLD_SEPARATORS, _SEPARATORS, _Store
from idtrack.graph.types import Connection, Place, ReleaseSet, served_mask
from idtrack.results import NameMatch

_VERSION_SUFFIX = ids.NUMBERED_SUFFIX  # a version or a make-unique suffix after a separator
_RESOLVE_STEPS = ("exact", "case", "separator", "version")


class Readings(NamedTuple):
    """What a string means in a graph at a release (:py:meth:`NameResolution.readings`).

    Attributes:
        taken: ``(node, how)`` of each reading taken: the first is :py:meth:`NameResolution.resolve_name`'s answer;
            several when the step that matched found several identifiers that exist at the release (a tie of
            readings, decision 87); empty when the string means nothing.
        set_aside: ``(node, how, reason)`` of the readings that a rule put after the one taken (``reason``: a key of
            :py:data:`idtrack.results.READING_REASONS`).
        here: The readings taken exist at the release asked about (``False``: the string means an identifier that
            exists only at other releases, the first in the order of the steps).
        other_releases: ``(node, how, releases)`` of the readings that the steps reached before the one taken and
            that do not exist at the release read but at other releases (the releases, ascending): what the string
            meant at another time (a name of one gene then, a synonym of another gene at the release read).
    """

    taken: tuple[tuple[str, str], ...]
    set_aside: tuple[tuple[str, str, str], ...] = ()
    here: bool = True
    other_releases: tuple[tuple[str, str, tuple[int, ...]], ...] = ()


def _settled(
    taken: tuple[tuple[str, str], ...],
    aside: list[tuple[str, str, str]],
    here: bool = True,
    other: tuple[tuple[str, str, tuple[int, ...]], ...] = (),
) -> Readings:
    """Return the readings taken and those set aside, each node set aside once and none that was also taken.

    A reading of the node taken as an id of another database (``"unversioned_database"``: the string before a version
    read as an id of a database that writes no version) is another reading of the same node, kept beside it.

    Args:
        taken: ``(node, how)`` of each reading taken.
        aside: ``(node, how, reason)`` of each reading a rule set aside, in the order of the steps (the case and the
            separator steps can reach one node twice, and the step order can take a node set aside before, read
            exactly at another release).
        here: As :py:attr:`Readings.here`.
        other: As :py:attr:`Readings.other_releases` (a node taken or set aside is left out of it).

    Returns:
        Readings: The readings.
    """
    held = {node for node, _ in taken}
    kept: dict[tuple[str, bool], tuple[str, str, str]] = {}
    for node, how, reason in aside:
        other_database = reason == "unversioned_database"
        if (other_database or node not in held) and (node, other_database) not in kept:
            kept[(node, other_database)] = (node, how, reason)
    seen = held | {node for node, _ in kept}
    elsewhere: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    for node, how, releases in other:
        if node not in seen and node not in elsewhere:
            elsewhere[node] = (node, how, releases)
    return Readings(taken, tuple(kept.values()), here, tuple(elsewhere.values()))


def _in_one_case(text: str) -> bool:
    """Tell whether a string has letters with case and all of them in one case (``AR``, ``nos2``; not ``Ar``)."""
    return text.upper() != text.lower() and text in (text.upper(), text.lower())


def _version_written_so(identifier: str, node: str) -> bool:
    """Tell whether a string that the step ``"separator"`` reads as an Ensembl id writes its version as Ensembl does.

    Ensembl writes a version after ``.`` (``ENSG00000141510.5``); ``-<digits>`` and ``_<digits>`` after an Ensembl id
    are as often the suffixes that tools append to make repeated names unique (decision 82) as versions written
    otherwise, so they are read as no version: such a form stays not found, never another version's answer.

    Args:
        identifier (str): The string as typed.
        node (str): The Ensembl node (not an external identifier) the step reached.

    Returns:
        bool: ``False`` when the node has a version and the string writes another separator in front of it.
    """
    if not ids.is_versioned_name(node) or len(identifier) != len(node):
        return True
    return identifier[node.index(ids.ID_VERSION_DELIMITER)] == ids.ID_VERSION_DELIMITER


def _version_change_shares(versions: Any) -> dict[str, tuple[int, int]]:
    """Count, per database, the pairs of consecutive releases of one link and those in which its version differs.

    Args:
        versions: The version table of a store (:py:class:`idtrack.graph.store._LinkMarks`), rows sorted by link.

    Returns:
        dict[str, tuple[int, int]]: ``{database: (pairs, pairs in which the version differs)}``.
    """
    import numpy as np

    if not len(versions):
        return {}
    masks = versions.masks

    def shifted(words: Any) -> Any:  # bit r of the result is bit r + 1 of the mask
        out = words >> np.uint64(1)
        out[:, :-1] |= words[:, 1:] << np.uint64(63)
        return out

    keys = np.stack([versions.source, versions.target, versions.database], axis=1)
    starts = np.flatnonzero(np.concatenate([[True], np.any(keys[1:] != keys[:-1], axis=1)]))
    held = np.bitwise_or.reduceat(masks, starts, axis=0)  # the releases of each link
    changes = np.bitwise_or.reduceat(masks ^ shifted(masks), starts, axis=0)  # a version at r and not r + 1, or so
    both = held & shifted(held)  # the link holds at r and at r + 1
    pairs = np.bitwise_count(both).sum(axis=1).astype(np.int64)
    differ = np.bitwise_count(both & changes).sum(axis=1).astype(np.int64)
    codes = versions.database[starts]
    found = {}
    for code in np.unique(codes).tolist():
        chosen = codes == code
        found[versions.databases[code]] = (int(pairs[chosen].sum()), int(differ[chosen].sum()))
    return found


class NameResolution:
    """What a string means in a graph (mixed into :py:class:`idtrack.graph.TheGraph`)."""

    if TYPE_CHECKING:
        # What the methods below use of :py:class:`idtrack.graph.TheGraph` (and of networkx), with its signatures.
        graph: Any  # the graph attributes of networkx (a dict)
        _reverse_view: bool
        _c_readings: Callable[[str, int | None, str | None, int | None, bool, bool], Readings]
        _c_respelled: frozenset[int]
        _c_respelled_lower: frozenset[str]
        _c_xref: Callable[[str, bool], tuple[tuple[str, Connection], ...]]
        _c_last: int | None
        _c_line: dict[int, ReleaseSet] | None

        @property
        def nodes(self) -> Any:
            """The nodes of the graph (networkx)."""

        @property
        def snapshot_release(self) -> int:
            """See :py:attr:`idtrack.graph.TheGraph.snapshot_release`."""

        @property
        def organism(self) -> str | None:
            """See :py:attr:`idtrack.graph.TheGraph.organism`."""

        @property
        def serving_line(self) -> dict[int, ReleaseSet]:
            """See :py:attr:`idtrack.graph.TheGraph.serving_line`."""

        @property
        def authority_databases(self) -> frozenset[str]:
            """See :py:attr:`idtrack.graph.TheGraph.authority_databases`."""

        @property
        def repeated_regions(self) -> dict[int, tuple[tuple[str, int, int, str, int], ...]]:
            """See :py:attr:`idtrack.graph.TheGraph.repeated_regions`."""

        def _frozen(self) -> _Store:
            """See :py:meth:`idtrack.graph.TheGraph._frozen`."""

        def node_type(self, node: str) -> str:
            """See :py:meth:`idtrack.graph.TheGraph.node_type`."""

        def version(self, node: str) -> str:
            """See :py:meth:`idtrack.graph.TheGraph.version`."""

        def is_active(self, node: str, release: int) -> bool:
            """See :py:meth:`idtrack.graph.TheGraph.is_active`."""

        def non_reference_releases(self, node: str) -> ReleaseSet:
            """See :py:meth:`idtrack.graph.TheGraph.non_reference_releases`."""

        def xref_edges(self, node: str, reverse: bool = False) -> tuple[tuple[str, Connection], ...]:
            """See :py:meth:`idtrack.graph.TheGraph.xref_edges`."""

        def places(self, node: str) -> tuple[Place, ...]:
            """See :py:meth:`idtrack.graph.TheGraph.places`."""

    def versions_of(self, stable_id: str) -> tuple[str, ...]:
        """Return the ``<id>.<version>`` nodes of a stable id, by ascending version.

        Genes have a base node that links their versions; transcripts and translations do not. This finds the
        versioned nodes of any Ensembl stable id by name (``<id>.Void`` and ``<id>.Retired`` are left out). Ensembl
        writes its stable ids in capitals only, so an id in another letter case (``enst00000489289``) finds the
        versions of the id in capitals when it has none as written.

        Args:
            stable_id (str): A stable id without version, e.g. ``ENST00000489289``.

        Returns:
            tuple[str, ...]: The node names; empty if the graph has no version of the id.
        """
        store = self._frozen()
        for written in dict.fromkeys((stable_id, stable_id.upper())):
            prefix = ids.versions_prefix(written)
            found = [
                name
                for name in store.names_with_prefix(prefix)
                if name[len(prefix) :].isdigit() and store.type_has_version[store.ntype_list[store.index[name]]]
            ]
            if found:
                return tuple(sorted(found, key=lambda name: int(name[len(prefix) :])))
        return ()

    def spelling(self, node: str, database: str, release: int | None = None, via: str | None = None) -> str:
        """Return how ``database`` spells an external identifier at a release, on one link or over all.

        The name of an external node is its canonical spelling: the one Ensembl gives at the latest release in which
        the identifier exists. Where databases, the releases of one database, or the genes a database attaches it
        to write an identifier with different letter case, the graph keeps one node and remembers how each link (the
        identifier, an Ensembl id, a database) wrote it at each release: ``Tc`` (a synonym of one mouse gene) and
        ``tc`` (a synonym of Noto) are one node whose links say which gene carries which spelling.

        A release counts as the assembly that serves it in the backbone has it (:py:attr:`serving_assembly`): an
        earlier assembly that Ensembl keeps beside the primary one (human GRCh37, frozen with the spellings of its
        time) does not supply the spelling of a release that the primary assembly serves.

        Args:
            node (str): An external node.
            database (str): The database whose spelling is wanted.
            release (int | None): The release at which the database is read. ``None``: the latest release at which
                the database has the identifier (on the link, with ``via``).
            via (str | None): The Ensembl id at the other end of the link (a gene, transcript or translation). ``None``:
                the spelling that most links of the database give at the release, then the one with more upper-case
                letters, then the first in sort order.

        Returns:
            str: The spelling at that release; at a release at which the link (or the database) does not have the
            identifier, its spelling at its latest release; the node name where nothing differs from it.

        Raises:
            UnknownIdentifierError: If the node, or ``via``, is not in the graph.
        """
        store = self._frozen()
        try:
            number = store.index[node]
            target = None if via is None else store.index[via]
        except KeyError:
            missing = node if node not in store.index else via
            raise UnknownIdentifierError(
                f"{missing!r} is not a node of the graph.", argument="node", value=missing
            ) from None
        if number not in self._c_respelled:
            return node
        if target is not None:
            return store.link_spelling(number, target, database, release)
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        masks: dict[str, int] = {}
        for on_line in (True, False):
            for other, connection in self._c_xref(node, self._reverse_view):
                if database in connection.databases:
                    mask = (
                        served_mask(connection.items, line, database)
                        if on_line
                        else functools.reduce(operator.or_, (r.mask for r in connection.as_dict[database].values()), 0)
                    )
                    if mask:
                        masks[other] = mask
            if masks:
                break
        if not masks:
            return node
        holding = [
            other for other, mask in masks.items() if release is not None and release >= 0 and mask >> release & 1
        ]
        if not holding:  # at the latest release of the database's links
            release = max(mask.bit_length() for mask in masks.values()) - 1
            holding = [other for other, mask in masks.items() if mask >> release & 1]
        counts: dict[str, int] = {}
        for other in holding:
            text = store.link_spelling(number, store.index[other], database, release)
            counts[text] = counts.get(text, 0) + 1
        return max(sorted(counts), key=lambda text: (counts[text], sum(c.isupper() for c in text)))

    def respelled(self, node: str) -> bool:
        """Tell whether some link writes an external node otherwise than the node is named (:py:meth:`spelling`)."""
        store = self._frozen()
        number = store.index.get(node)
        return number is not None and number in self._c_respelled

    def spelt_releases(self, node: str, via: str, database: str, releases: int, text: str) -> int:
        """Return the releases of a mask at which the link of an external node to ``via`` in a database spells ``text``.

        Args:
            node (str): An external node.
            via (str): The Ensembl id at the other end of the link.
            database (str): The database of the link.
            releases (int): A release bit mask (of the link).
            text (str): A spelling.

        Returns:
            int: The bit mask of the releases of ``releases`` at which the link writes the identifier as ``text``.
        """
        store = self._frozen()
        return store.spelt_mask(store.index[node], store.index[via], database, releases, text)

    def accession_versions(self, node: str, database: str | None = None) -> dict[int, int]:
        """Return the versions that the links of an accession attach, each with the releases at which one does.

        Ensembl's cross references give the version of the sequence they attach (``xref.version``: ``NM_000546.6``),
        and the graph keeps it per link and release, as the assembly that serves a release has it (a link that no
        serving assembly has, where it is listed). A database that gives no version (a symbol, an NCBI gene id) has
        none here.

        Args:
            node (str): An external node.
            database (str | None): Only the links of this database; ``None``: of every database.

        Returns:
            dict[int, int]: ``{version: release bit mask}``; empty for an identifier without versions.
        """
        store = self._frozen()
        number = store.index.get(node)
        found: dict[int, int] = {}
        if number is None:
            return found
        for _target, link_database, version, mask in store.versions.of_node(number):
            if database is None or link_database == database:
                found[version] = found.get(version, 0) | mask
        return found

    def edit_counter_databases(self) -> frozenset[str]:
        """Return the databases whose stored number of an accession is an edit counter, not a version of the sequence.

        Decision 101: a version dates an accession only where it is a property of the sequence record that holds over
        time. Measured from the graph's own versions (:py:meth:`version_change_shares`): a database in which the
        version differs in more than :py:data:`idtrack.settings.VERSION_EDIT_COUNTER_SHARE` of the pairs of
        consecutive releases of one link counts the record's edits (UniProt's entry version: ``Q9TSV4.120``); a version
        written after such an accession is not judged (:py:meth:`readings`). Computed once per graph, at first use.

        Returns:
            frozenset[str]: The databases.
        """
        found = self.__dict__.get("_c_edit_counters")
        if found is None:
            bound = settings.VERSION_EDIT_COUNTER_SHARE
            found = self.__dict__["_c_edit_counters"] = frozenset(
                database
                for database, (pairs, differ) in self.version_change_shares().items()
                if pairs and differ / pairs > bound
            )
        return found

    def version_change_shares(self) -> dict[str, tuple[int, int]]:
        """Return, per database, the pairs of consecutive releases of one link and those in which the version differs.

        A link is an accession, one version of an Ensembl id and a database (:py:meth:`accession_versions`); a pair
        is two consecutive releases at both of which the link attaches a version, and it differs where the versions
        attached at the two releases are not the same.

        Returns:
            dict[str, tuple[int, int]]: ``{database: (pairs, pairs in which the version differs)}``.
        """
        found = self.__dict__.get("_c_version_shares")
        if found is None:
            found = self.__dict__["_c_version_shares"] = _version_change_shares(self._frozen().versions)
        return found

    def _judged_versions(self, node: str) -> dict[int, int]:
        """Return the versions of an accession attached by links of databases whose versions date it (decision 101)."""
        counters = self.edit_counter_databases()
        store = self._frozen()
        number = store.index.get(node)
        found: dict[int, int] = {}
        if number is None:
            return found
        for _target, database, version, mask in store.versions.of_node(number):
            if database not in counters:
                found[version] = found.get(version, 0) | mask
        return found

    def version_releases(self, node: str, via: str, database: str, releases: int, version: int) -> int:
        """Return the releases of a mask at which the link of an accession to ``via`` in a database attaches a version.

        Args:
            node (str): An external node.
            via (str): The Ensembl id at the other end of the link.
            database (str): The database of the link.
            releases (int): A release bit mask (of the link).
            version (int): The version.

        Returns:
            int: The bit mask of the releases of ``releases`` at which the link attaches that version.
        """
        store = self._frozen()
        target = store.index[via]
        mask = 0
        for other, link_database, value, held in store.versions.of_node(store.index[node]):
            if other == target and link_database == database and value == version:
                mask |= held
        return releases & mask

    def display_releases(self, node: str, gene: str) -> int:
        """Return the releases at which an external identifier is a gene's display name (``gene.display_xref_id``).

        Ensembl gives every gene at every release one display name, whatever database it came from (an authority's
        symbol, a projection of another organism's, NCBI's, UniProt's): Ensembl's own statement of the gene's name
        there. It ranks a gene first among the genes that carry a name at one release (:py:mod:`idtrack.engine.ranking`).

        Args:
            node (str): An external node.
            gene (str): A gene id with version.

        Returns:
            int: The release bit mask, on the assembly that serves each release; 0 where it never is.
        """
        store = self._frozen()
        number, target = store.index.get(node), store.index.get(gene)
        if number is None or target is None:
            return 0
        return functools.reduce(
            operator.or_, (mask for other, _db, _value, mask in store.displays.of_node(number) if other == target), 0
        )

    def display_links(self, node: str, gene: str) -> dict[str, int]:
        """Return the databases whose link of an external identifier to a gene is the gene's display name, with releases.

        Ensembl's display name of a gene is one cross reference (``gene.display_xref_id``), of one database, spelt as
        that database writes it (:py:meth:`spelling` of the link). A graph that does not keep the database of its
        display names gives them under ``""`` (the spelling is then not known).

        Args:
            node (str): An external node.
            gene (str): A gene id with version.

        Returns:
            dict[str, int]: ``{database: release bit mask}``; empty where the identifier never is the gene's display name.
        """
        store = self._frozen()
        number, target = store.index.get(node), store.index.get(gene)
        found: dict[str, int] = {}
        if number is None or target is None:
            return found
        for other, database, _value, mask in store.displays.of_node(number):
            if other == target:
                found[database] = found.get(database, 0) | mask
        return found

    def display_rank(self, node: str, gene: str, release: int, typed: str) -> int:
        """Rank how a typed string is a gene's display name at a release (decision 102; :py:mod:`idtrack.engine.ranking`).

        The display name holds for the spelling typed. Letter case is evidence only where the string has letters of
        both cases: then the gene's display name is the string only where its database spelt it so at the release
        (mouse ``Foxl1`` at 70 is the display name of the gene MGI names so, not of the gene UniProt displays as
        ``FOXL1``); separators are not compared (the step ``"separator"`` reads them as one). A string in one letter
        case carries no evidence of case (decisions 55 and 80): it is read without case, and in an organism whose
        symbols are written in mixed case (:py:meth:`symbol_case`) the display name of a name authority
        (:py:attr:`authority_databases`) ranks before another source's (mouse ``GAPDH`` is the gene MGI displays as
        ``Gapdh`` before the gene UniProt displays as ``GAPDH``).

        Args:
            node (str): The external node the string was read as.
            gene (str): A gene id with version.
            release (int): The release the string is read at.
            typed (str): The string as typed (without a version written after an accession).

        Returns:
            int: 0 the display name; 1 another source's display name, for a string in one letter case where symbols
            are written in mixed case; 2 not the gene's display name there.
        """
        if release < 0:
            return 2
        links = {database for database, mask in self.display_links(node, gene).items() if mask >> release & 1}
        if not links:
            return 2
        if "" in links:
            return 0  # the graph does not say which database displays it: no spelling to compare
        if _in_one_case(typed):
            if self.symbol_case() != "mixed":
                return 0
            return 0 if links & self.authority_databases else 1
        store = self._frozen()
        number, target = store.index[node], store.index[gene]
        written = typed.translate(_FOLD_SEPARATORS)
        spelt = any(
            store.link_spelling(number, target, database, release).translate(_FOLD_SEPARATORS) == written
            for database in links
        )
        return 0 if spelt else 2

    def reading_version(self, identifier: str, node: str, how: str) -> int | None:
        """Return the version of an accession that a reading of a string stands for (:py:meth:`readings`).

        Args:
            identifier (str): The string as read (``NM_000546.6``).
            node (str): The node it was read as.
            how (str): How it matched.

        Returns:
            int | None: For a versioned accession (the step ``"version"``) the version written; for one written with a
            version above every version its links attach (``"version_newer_than_graph"``) the latest of those;
            ``None`` for every other reading (``"version_not_judged"``: the accession without the version).
        """
        if how not in ("version", "version_newer_than_graph") or self.node_type(node) != ids.EXTERNAL:
            return None
        held = self._judged_versions(node)
        if how == "version_newer_than_graph":
            return max(held) if held else None
        suffix = _VERSION_SUFFIX.match(identifier)
        return int(suffix.group(3)) if suffix is not None and int(suffix.group(3)) in held else None

    def resolve_name(
        self,
        identifier: str,
        from_release: SupportsIndex | None = None,
        from_database: str | None = None,
        to_release: SupportsIndex | None = None,
    ) -> tuple[str | None, str | None]:
        """Find the node a user's string means at a release, and say how it was matched.

        This is the one place that says what a typed string means; the engine, the API, the inference of the source
        of a list and the self-tests all ask it, each naming the release it means.

        Two things decide, in this order:

        *Existence at the release the question is about.* Between the nodes the string can mean, one that exists at
        ``from_release`` (without one: at ``to_release``, the release a conversion reads it at when it exists there;
        without either, the last release of the graph) is taken before one that does not. One release decides both
        which node the string means and where a conversion reads it. Databases
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
        an id of (``from_database``), "exists" means "is an id of that database at the release", and among the nodes
        that are not, one that is an id of that database at some release comes before one that never is. The synonyms
        a database records are read under its name too, after all its names (every step on the names first): a
        string that the database gives no gene as a name at any release is read as one of its synonyms.

        *The steps*, among the nodes that are equal in existence. They are tried in this order, each only if the
        one before found nothing. Each of the first four is tried on the names of the graph and then on its
        synonyms (the synonym node of the string, ``synonym_id::<string>``; the match is then reported as
        ``"synonym:<step>"``) before the next step begins, so that a string that is exactly a synonym of one gene is
        that synonym, not a spelling variant of another gene's name (``H3.3B``, a synonym, is not the name
        ``H3-3b``):

        1. ``"exact"``: the string is an identifier exactly as it is written. For an Ensembl id that is the name
           of its node. For an external identifier it is how a link of it (to an Ensembl id, in a database: the one
           named, when ``from_database`` is given) writes it at the release the string is read at (above; for an
           identifier that does not exist there, at its latest release). Databases change the letter case of a name
           over the releases, two databases may write
           one name in two ways at one release, and one database may give two genes one name in two cases (``Tc``
           and ``tc``); the graph keeps one node per identifier, named as at its last release, and how each link
           wrote it at each release (:py:meth:`spelling`). So ``PRR2``, typed for a release at which a database
           wrote the name so, is that name exactly, although the node is called ``prr2`` after today's spelling,
           and it is not taken for another gene's synonym ``PRR2``; typed ``prr2`` for that release it is the same
           name by letter case. A conversion that matched a string exactly follows only the links spelt as typed
           (:py:meth:`idtrack.engine.Track.convert`).
        2. ``"case"``: it is a node name when letter case is ignored.
        3. ``"separator"``: it is a node name when ``-``, ``_`` and ``.`` are taken as the same character (and case
           is ignored): ``HLA_A`` and ``HLA.A`` for ``HLA-A``. An Ensembl id's version is read only after ``.``, as
           Ensembl writes it: ``ENSG00000123456_2`` and ``ENSG00000123456-2`` are not ``ENSG00000123456.2`` (single-cell
           tools append such suffixes to make repeated names unique), and stay not found.
        4. ``"version"``: the string ends in ``.<digits>`` and the identifier without that suffix is a versioned
           accession (a link of it attaches a version, ``xref.version`` of Ensembl's cross references: RefSeq,
           UniProt, CCDS, the EMBL protein ids; :py:meth:`accession_versions`) and no id of a symbol database of the
           organism (:py:attr:`idtrack.organisms.SYMBOL_DATABASES`), and some link attached that version:
           ``NM_000546.5`` for ``NM_000546`` as the links that attached version 5 carry it, at the releases at which
           they did (decision 14's rule for Ensembl ids, applied to accessions). Where the accession's numbers come only
           from databases whose number is an edit counter of their records (:py:meth:`edit_counter_databases`:
           UniProt's entry version), the version is not judged: the accession is read without it, as
           ``"version_not_judged"`` (decision 101). Nothing else is ever cut off the name
           of an external identifier: ``SLA.7`` and ``SLA_11`` are not ``SLA``, ``NM_713028147`` is not ``NM``.
        5. ``"version_newer_than_graph"``: the string is an Ensembl id with a version higher than every version the
           graph holds of it (written with ``.``), and the id is active at the last release of the
           graph: the version may belong to a release newer than the graph, so the string stands for the latest
           version the graph holds, ``ENSG00000123456.99`` for ``ENSG00000123456.7``; or a versioned accession written
           with a version higher than every version its links attached (the step ``"version"``), where a link of it
           holds at the last release of the graph (it can then be newer than the graph), read as the latest of them,
           as the links that attached it carry it. Any other version that the graph does not hold (a lower one, one in
           a gap, any version of a retired id; of an accession, one its links never attached, and one above all of an
           accession no link attaches at the last release) matches nothing; :py:meth:`known_versions` then names the
           versions that exist. Where the identifier before the version is, at the release read, also an id of a
           database that gives it no version (an NCBI gene id spelt like a numeric Havana id), and no link has it with
           that version there, the string has two readings: the version reading (taken where it exists, by the rule
           below) and that database's id, reported (:py:meth:`readings`, ``"unversioned_database"``).

        When no node the string can mean exists at the release, the string is read at the latest release at which one
        of them exists (with ``from_database``: as an id of that database), by the steps and rules of that release, so
        that what it means and where it is read are again one release; the caller learns from the node where it does
        exist (:py:attr:`Readings.here` is ``False``). A node that exists at no release on the assemblies that serve
        the graph's releases (or, with ``from_database``, is no id of that database at all) is taken, first in the
        order of the steps, only when no other node exists anywhere. The cost is a few dictionary look-ups, whatever
        the string.

        Args:
            identifier (str): The string to resolve.
            from_release (SupportsIndex | None): The Ensembl release the question is about (for a conversion: the
                release the identifier is read at). ``None`` means the last release of the graph. A release the graph
                does not have is one at which no external identifier exists.
            from_database (str | None): The external database the string is meant as an id of. ``None``: any
                database.
            to_release (SupportsIndex | None): Only without ``from_release``: the release a conversion goes to, at
                which the string is read (which node exists, and how it is spelt in the step ``"exact"``) when one of
                its readings exists there. ``None``: the last release of the graph.

        Returns:
            tuple[str | None, str | None]: The node name and how it was matched, or ``(None, None)``.

        Raises:
            ArgumentTypeError: If ``identifier`` is not a string, or ``from_release`` or ``to_release`` is not an
                integer.
        """
        _arguments.string(identifier, "identifier")
        at = (
            from_release
            if from_release is None or type(from_release) is int
            else _arguments.release(from_release, "from_release")
        )
        goal = (
            to_release
            if to_release is None or type(to_release) is int
            else _arguments.release(to_release, "to_release")
        )
        self._frozen()
        taken = self._c_readings(identifier, at, from_database, goal if at is None else None, True, True).taken
        return taken[0] if taken else (None, None)

    def readings(
        self,
        identifier: str,
        from_release: SupportsIndex | None = None,
        from_database: str | None = None,
        to_release: SupportsIndex | None = None,
    ) -> Readings:
        """Return every reading of a string that :py:meth:`resolve_name` took, and those its rules set aside.

        :py:meth:`resolve_name` returns the first reading taken. Two rules take a reading before the order of the
        steps would, and the reading the steps would have taken is then *set aside*, so that a caller can report it:

        - **Symbols without letter case first** (decisions 55 and 80). In an organism whose symbol databases write
          symbols in mixed case (:py:meth:`symbol_case`, read from the graph: MGI writes ``Trp53``), a string in one
          letter case (``AR``, ``nos2``) that is not exactly a name of the release is read first as a symbol of the
          organism's symbol databases without letter case; where that
          names exactly one gene at the release, it is the reading taken (match ``"case"``), and the reading of the
          steps (an exact synonym of another gene, ``AR`` of three) is set aside. A named ``from_database`` that is
          not a symbol database of the organism turns the rule off.
        - **No separator variant joins a make-unique suffix into another gene's synonym** (decision 82). A string
          that ends in ``-<digits>``, ``.<digits>`` or ``_<digits>`` (``SRF.1``: single-cell tools append such
          suffixes to repeated names) is not read by the steps ``"separator"`` and ``"case"`` as a synonym whose genes
          at the release are none of the genes to which a name authority gives the string before the suffix, written
          exactly so, as its symbol there (``SRF-1``, a synonym of SPATA6, while ``SRF`` is SRF's HGNC symbol); that
          reading is set aside. A name reached so is read (``MIR138_2``
          for the gene ``MIR138-2``, ``H1_0`` for ``H1-0``: numbered names are written with these separators too).

        A version is read where some database has it. Where the identifier before ``.<digits>`` is, at the release
        read, also an id of a database that gives it no version (:py:meth:`accession_versions`: an NCBI gene id
        ``2633`` beside a numeric Havana translation id ``2633`` with version 1 at releases 50 and 51) and no link
        attaches it with that version at that release, the version reading is taken by the rules below and the
        identifier as that other database's id is set aside beside it (``"unversioned_database"``): never one reading
        silently.

        A step that finds several identifiers that exist at the release takes them all (decision 87: a separator
        reading that reaches several names is a tie, ``AY172335_1`` for ``AY172335-1`` and ``AY172335.1`` where both
        are names at the release read). Readings are joined only where they exist at one release: when none exists at
        the release asked about, the string is read at the latest release at which one does (:py:meth:`resolve_name`),
        and the readings of that release are taken (``AY172335-1`` is a name at releases 54 to 56 only and
        ``AY172335.1`` at 57 to 60 only: without a release, ``AY172335_1`` is read at 60, as ``AY172335.1``).

        A reading the steps reached before the one taken that exists only at other releases is reported with those
        releases (:py:attr:`Readings.other_releases`): a name of one gene then, read as a synonym of another gene at
        the release asked about, says what the string meant at another time (decision 99: the release asked about
        decides; ask at the other release for the other meaning).

        Args:
            identifier (str): The string.
            from_release (SupportsIndex | None): As :py:meth:`resolve_name`.
            from_database (str | None): As :py:meth:`resolve_name`.
            to_release (SupportsIndex | None): As :py:meth:`resolve_name`.

        Returns:
            Readings: The readings taken (empty when the string means nothing) and those set aside.

        Raises:
            ArgumentTypeError: As :py:meth:`resolve_name`.
            InvalidArgumentError: If the string is an Ensembl id with a version longer than any Ensembl version.
        """
        _arguments.string(identifier, "identifier")
        at = None if from_release is None else _arguments.release(from_release, "from_release")
        goal = None if to_release is None else _arguments.release(to_release, "to_release")
        self._frozen()
        return self._c_readings(identifier, at, from_database, goal if at is None else None, True, True)

    def _readings_uncached(
        self,
        identifier: str,
        at: int | None,
        database: str | None,
        goal: int | None = None,
        rules: bool = True,
        settle: bool = True,
    ) -> Readings:
        """Do the work of :py:meth:`readings` and :py:meth:`resolve_name` for checked arguments.

        Args:
            identifier (str): The string.
            at (int | None): The release asked about; ``None``: ``goal``.
            database (str | None): The database the string is meant as an id of.
            goal (int | None): Without ``at``, the release a conversion goes to; ``None``: the last release.
            rules (bool): Apply the rules of decisions 55, 80 and 82 (:py:meth:`readings`); ``False``: the order of
                the steps alone.
            settle (bool): When no reading exists at the release, read the string at the latest release at which one
                does (:py:meth:`readings`); ``False``: return the first reading in the order of the steps (the
                reading at that latest release asks for it, so that it is asked once).

        Returns:
            Readings: As :py:meth:`readings`.
        """
        store = self._frozen()
        external, kinds, names = store.external_code, store.ntype_list, store.names
        spellings = ((identifier, ""), (f"{ids.SYNONYM_PREFIX}{identifier}", "synonym:"))
        if at is None:
            # One release decides both which node the string means and where it is read: without a release asked
            # about, the release a conversion goes to (decision 94 for calls that name only the target release).
            at = goal if goal is not None else self._last_release_kept()
        candidates: Iterable[int]
        # every match that does not exist at the release, in the order of the steps, with the database it is read in:
        # [0] ids of the database named (any match without one), [1] matches that are no id of the database named
        elsewhere: list[list[tuple[str, str, str | None, int | None]]] = [[], []]
        aside: list[tuple[str, str, str]] = []
        # Each step on names, then on synonyms; with a database named, every step on its names before any on the
        # synonyms it records (decision 56: a database's synonyms are read under its name, after all its names)
        order = (
            [(step, spelling) for step in _RESOLVE_STEPS for spelling in spellings]
            if database is None
            else [(step, spelling) for spelling in spellings for step in _RESOLVE_STEPS]
        )
        for step, (query, prefix) in order:
            if database is not None and prefix and step == _RESOLVE_STEPS[0] and elsewhere[0]:
                break  # a name of the database named, at another release, before any of its synonyms
            # a named database's synonyms are ids of `synonym_id::<database>` in the graph
            db = database if database is None or not prefix else f"{ids.SYNONYM_PREFIX}{database}"
            if rules and step == "exact" and prefix:
                # decisions 55 and 80: before the exact synonyms, a string against the organism's convention of
                # letter case is read as one of its symbols without letter case where that names one gene
                symbol = self._symbol_without_case(store, identifier, at, database)
                if symbol is not None:
                    plain = self._c_readings(identifier, at, database, None, False, True).taken
                    aside.extend((node, how, "symbol_without_case") for node, how in plain[:1] if node != symbol)
                    return _settled(((symbol, "case"),), aside, other=self._other_releases(elsewhere[0]))
            if step == "exact":
                # A node is named as its identifier is written at its last release. An Ensembl id is its name.
                # Nearly every external identifier is written in one way by every database at every release
                # and is then exactly its name too; the others are known, and for them the spelling of the
                # release the identifier is read at decides (`_written_so`).
                exact = store.index.get(query)
                if exact is not None and kinds[exact] != external:
                    return _settled(((names[exact], f"{prefix}{step}"),), aside)
                if exact is not None and exact not in self._c_respelled:
                    candidates = (exact,)
                elif query.lower() in self._c_respelled_lower:
                    candidates = self._written_so(store, query, exact, at, db)
                else:
                    continue
            else:
                candidates = self._matches(store, query, step)
            tie: list[tuple[str, str]] = []
            for found in candidates:
                name = names[found]
                if kinds[found] != external:
                    if step == "separator" and not _version_written_so(identifier, name):
                        continue  # an Ensembl version is written after `.`, never `-` or `_`
                    return _settled(((name, f"{prefix}{step}"),), aside)
                if (
                    rules
                    and prefix
                    and step in ("case", "separator")
                    and self._joins_suffix(identifier, name, at, database)
                ):
                    aside.append((name, f"{prefix}{step}", "make_unique_suffix"))
                    continue
                how, version = f"{prefix}{step}", None
                if step == "version":
                    # decision 14's rule for accessions: the version written, where the accession's links attached it;
                    # one above every version they attached, the latest (flagged), where the accession is attached at
                    # the graph's last release (the version may be newer than the graph); any other version, no
                    # reading. Decision 101: where the accession's numbers are edit counters only, the version is not
                    # judged.
                    written = int(_VERSION_SUFFIX.match(query).group(3))  # type: ignore[union-attr]
                    attached = self._judged_versions(name)
                    if attached and not self._exists_at(name, at, db, written)[0] and self._unversioned_at(name, at):
                        # the string before the version is, at the release read, an id of a database that writes no
                        # version (an NCBI gene id beside a numeric Havana id): the version reading is taken only
                        # where its database has the form with that version at the release; elsewhere the other
                        # reading is reported beside whatever is taken (never one reading silently)
                        aside.append((name, f"{prefix}exact", "unversioned_database"))
                    if not attached:
                        how = f"{prefix}version_not_judged"
                    elif written in attached:
                        version = written
                    elif written > max(attached) and self._attached_at_last(name):
                        how, version = f"{prefix}version_newer_than_graph", max(attached)
                    else:
                        continue  # a version the accession never had, or above all of one retired (:py:meth:`known_versions`)
                here, known = self._exists_at(name, at, db, version)
                if here:
                    if step != "separator":
                        return _settled(((name, how),), aside, other=self._other_releases(elsewhere[0]))
                    tie.append((name, how))  # decision 87: every name the step reaches
                    continue
                elsewhere[0 if known else 1].append((name, how, db, version))
            if tie:
                return _settled(tuple(tie), aside, other=self._other_releases(elsewhere[0]))
        if settle and elsewhere[0]:
            # No reading exists at the release: the string is read at the latest release at which one of its readings
            # exists, by the steps and rules of that release, so that the readings taken exist at one release
            # (decision 87's tie joins only such readings) and the node and the release it is read at are one choice.
            newest = max((self._latest_release(name, db, v) for name, _how, db, v in elsewhere[0]), default=-1)
            if newest >= 0:
                there = self._c_readings(identifier, newest, database, None, rules, False)
                if there.taken and there.here:
                    # the other database's reading at the release asked about stays reported
                    unversioned = [entry for entry in aside if entry[2] == "unversioned_database"]
                    joined = _settled(there.taken, [*there.set_aside, *unversioned])
                    return Readings(there.taken, joined.set_aside, False, there.other_releases)
        for matches in elsewhere:
            if matches:  # readings that exist at no release of the graph's line (or not at one): the first alone
                return _settled(((matches[0][0], matches[0][1]),), aside, here=False)
        unknown = self._unknown_version(identifier)
        if unknown is not None:
            asked, held = unknown
            latest = held[-1]
            if asked > int(self.version(latest)) and self.is_active(latest, self.snapshot_release):
                return _settled(((latest, "version_newer_than_graph"),), aside)
        return _settled((), aside)

    def _symbol_without_case(self, store: _Store, identifier: str, release: int, database: str | None) -> str | None:
        """Read a string against the organism's convention of letter case as one of its symbols (decisions 55, 80).

        Args:
            store (_Store): The frozen store of the graph.
            identifier (str): The string.
            release (int): The release it is read at.
            database (str | None): The database named as its source; the rule holds only without one or for a
                symbol database of the organism.

        Returns:
            str | None: The external node whose name is the string without letter case and that the organism's
            symbol databases attach to exactly one gene at the release; ``None`` when the rule does not apply.
        """
        organism = self.organism or ""
        if not _in_one_case(identifier) or self.symbol_case() != "mixed":
            return None
        symbols = organisms.SYMBOL_DATABASES.get(organism, ())
        if database is not None and database not in symbols:
            return None
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        for number in store.lower_candidates(identifier.lower()):
            if store.ntype_list[number] != store.external_code:
                continue
            name = store.names[number]
            genes = {
                gene
                for gene, connection in self.xref_edges(name)
                if self.node_type(gene) == ids.GENE
                and any(
                    symbol in connection.databases and served_mask(connection.items, line, symbol) >> release & 1
                    for symbol in symbols
                )
            }
            if len(genes) == 1:
                return name
        return None

    def symbol_case(self) -> str | None:
        """Return how the organism's symbol databases write their symbols, read from the graph (decisions 55 and 80).

        ``"mixed"`` when more than half of the names that the organism's symbol databases
        (:py:data:`idtrack.organisms.SYMBOL_DATABASES`) attach to genes, at any release, are written with a small
        letter (MGI writes ``Trp53``: 98 % of mouse symbols), else ``"capitals"`` (HGNC and VGNC write ``TP53``: 3 % of
        pig symbols have a small letter, as ``C1orf112`` has). In an organism that writes its symbols in mixed case, a
        string in one letter case is read first as a symbol without letter case (:py:meth:`readings`).

        Returns:
            str | None: The convention; ``None`` when the graph has no symbol of a symbol database.
        """
        known = self.__dict__.get("_c_symbol_case", False)
        if known is False:
            known = self.__dict__["_c_symbol_case"] = self._symbol_case_uncached()
        return known

    def _symbol_case_uncached(self) -> str | None:
        """Do the work of :py:meth:`symbol_case`, once per graph."""
        import numpy as np

        from idtrack.graph.store import _connection_or_none
        from idtrack.graph.types import _KIND_OTHER, _KIND_XREF

        store = self._frozen()
        symbols = set(organisms.SYMBOL_DATABASES.get(self.organism or "", ()))
        if not symbols:
            return None
        ends: set[int] = set()
        refs = [i for i, connection in enumerate(store.connections) if not symbols.isdisjoint(connection.databases)]
        chosen = (store.e_kind == _KIND_XREF) & np.isin(store.e_ref, np.asarray(refs, dtype=np.int64))
        ends.update(store.e_src[chosen].tolist())
        ends.update(store.e_dst[chosen].tolist())
        for edge in np.flatnonzero(store.e_kind == _KIND_OTHER).tolist():  # graphs made by hand
            connection = _connection_or_none(store.other_attributes[int(store.e_ref[edge])].get(ids.CONNECTION_KEY))
            if connection is not None and not symbols.isdisjoint(connection.databases):
                ends.update((int(store.e_src[edge]), int(store.e_dst[edge])))
        written = [store.names[i] for i in ends if store.ntype_list[i] == store.external_code]
        if not written:
            return None
        small = sum(1 for name in written if name != name.upper())
        return "mixed" if 2 * small > len(written) else "capitals"

    def _joins_suffix(self, identifier: str, name: str, release: int, database: str | None) -> bool:
        """Tell whether a separator reading joins a make-unique suffix into another gene's name (decision 82).

        Args:
            identifier (str): The string as typed (``SRF.1``).
            name (str): The external node the step ``"separator"`` found for it (``SRF-1``, or its synonym node).
            release (int): The release the string is read at.
            database (str | None): The database named as its source.

        Returns:
            bool: ``True`` when the string ends in ``[-._]<digits>``, the string before the suffix is exactly a symbol
            that a name authority (:py:attr:`idtrack.TheGraph.authority_databases`) gives to genes at the release,
            and the node's genes at the release are none of them.
        """
        suffix = _VERSION_SUFFIX.match(identifier)
        if suffix is None:
            return False
        # the string before the suffix is a symbol of a name authority written exactly so (a feature name of a
        # matrix), not a string that only another reading finds (`Gal` for `GAL`)
        base = self._c_readings(suffix.group(1), release, database, None, True, True).taken
        if not base or base[0][1] != "exact":
            return False
        authorities = self.authority_databases
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        base_genes = {
            gene
            for gene, connection in self.xref_edges(base[0][0])
            if self.node_type(gene) == ids.GENE
            and any(served_mask(connection.items, line, d) >> release & 1 for d in connection.databases & authorities)
        }
        return bool(base_genes) and not base_genes & self._genes_at(name, release)

    def _genes_at(self, node: str, release: int) -> set[str]:
        """Return the genes a node stands for at a release: itself, or the genes its links reach there.

        Args:
            node (str): A node.
            release (int): The release.

        Returns:
            set[str]: Gene ids with version (directly, or through a transcript or translation) at the release, on
            the assembly that serves it.
        """
        kind = self.node_type(node)
        if kind == ids.GENE:
            return {node} if self.is_active(node, release) else set()
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        genes: set[str] = set()
        frontier, depth = [node], 0
        while frontier and depth < 3:
            following = []
            for current in frontier:
                for other, connection in self.xref_edges(current):
                    if not served_mask(connection.items, line) >> release & 1:
                        continue
                    other_kind = self.node_type(other)
                    if other_kind == ids.GENE:
                        if self.is_active(other, release):
                            genes.add(other)
                    elif other_kind != ids.EXTERNAL:
                        following.append(other)
            frontier, depth = following, depth + 1
        return genes

    def _unknown_version(self, identifier: str) -> tuple[int, tuple[str, ...]] | None:
        """Read a string as an Ensembl id with a version that the graph does not hold.

        Args:
            identifier (str): The string, which is not a node.

        Returns:
            tuple[int, tuple[str, ...]] | None: The version asked for and the ``<id>.<version>`` nodes the graph
            holds of the id, ascending; ``None`` when the string is not an Ensembl id of the graph followed by
            ``.``, ``_`` or ``-`` and digits.

        Raises:
            InvalidArgumentError: If the version has more digits than any Ensembl version
                (:py:data:`idtrack.reading.MAX_VERSION_DIGITS`).
        """
        suffix = _VERSION_SUFFIX.match(identifier)
        if suffix is None:
            return None
        core, separator, digits = suffix.groups()
        if separator != ids.ID_VERSION_DELIMITER:
            return None  # an Ensembl version is written after `.` (:py:func:`_version_written_so`)
        store = self._frozen()
        hit = store.index.get(core)
        if hit is None:
            candidates = store.lower_candidates(core.lower())
            hit = candidates[0] if candidates else None
        # A gene id without version is a node (in any letter case); other Ensembl ids are found by their versions.
        stable_id = store.names[hit] if hit is not None and store.node_type(hit) != ids.EXTERNAL else core
        held = self.versions_of(stable_id)
        if not held:
            return None
        reading.version_digits_refused(identifier, digits)  # read as a number only within the bound of a version
        return int(digits), held

    def known_versions(self, identifier: str) -> tuple[str, ...]:
        """Return the versions the graph holds of an Ensembl id or an accession written with a version it does not hold.

        An Ensembl id with a version that never existed (or that the graph cannot know) is not answered with
        another version's gene (:py:meth:`resolve_name` matches nothing, unless the version may be newer than the
        graph), nor is an accession written with a version that its links never attached, unless it is above all of
        them and the accession is attached at the graph's last release; this method says which versions exist, for
        the error message.

        Args:
            identifier (str): The string as given, e.g. ``ENSG00000123456.99`` or ``NM_000546.2``.

        Returns:
            tuple[str, ...]: The ``<id>.<version>`` nodes of the id, or the ``<accession>.<version>`` that the links of
            the accession attached, by ascending version; empty when the string is a node, or is not an Ensembl id or
            a versioned accession of the graph followed by a version it never had.

        Raises:
            ArgumentTypeError: If ``identifier`` is not a string.
        """
        _arguments.string(identifier, "identifier")
        store = self._frozen()
        if identifier in store.index:
            return ()
        unknown = self._unknown_version(identifier)
        if unknown is not None:
            return unknown[1]
        suffix = _VERSION_SUFFIX.match(identifier)
        if suffix is None or suffix.group(2) != ids.ID_VERSION_DELIMITER:
            return ()
        for number in self._matches(store, identifier, "version"):
            name = store.names[number]
            held = self._judged_versions(name)
            written = int(suffix.group(3))
            if held and written not in held and (written < max(held) or not self._attached_at_last(name)):
                return tuple(f"{name}{ids.ID_VERSION_DELIMITER}{version}" for version in sorted(held))
        return ()

    def region_at(self, gene: str, release: int) -> str | None:
        """Return the sequence region (chromosome, scaffold, patch) a gene lies on at a release (its place).

        Args:
            gene (str): A gene id with version.
            release (int): The release.

        Returns:
            str | None: The region; ``None`` when the graph holds no place of the gene at the release.
        """
        for place in self.places(gene):
            if place.releases.mask >> release & 1:
                return place.region
        return None

    def in_pseudoautosomal_y(self, gene: str, release: int) -> bool:
        """Tell whether a gene lies in a pseudoautosomal region of chromosome Y at a release (decisions 93, 95).

        Such a region repeats a region of X (:py:attr:`idtrack.TheGraph.repeated_regions`); a gene in it is the Y copy
        of the gene at its place on X, where there is one (:py:meth:`y_copies`).

        Args:
            gene (str): A gene id with version.
            release (int): The release.

        Returns:
            bool: Whether its place at the release lies inside such a region of Y of its assembly.
        """
        masks = self.__dict__.get("_c_par")
        if masks is None:
            masks = self.__dict__["_c_par"] = self._pseudoautosomal_masks()
        number = self._frozen().index.get(gene)
        return number is not None and bool(masks.get(number, 0) >> release & 1)

    def y_copies(self, genes: Iterable[str], release: int) -> set[str]:
        """Return the genes of a set that are the Y copies of other genes of the set at a release (decisions 93, 95).

        A Y copy lies in a pseudoautosomal region of Y at the release (:py:meth:`in_pseudoautosomal_y`) at the place of
        another gene of the set: the one rule of a gene reads its place on the region of X that the region of Y repeats
        (:py:func:`idtrack.graph.genes.same_place`, :py:attr:`idtrack.TheGraph.repeated_regions`). A gene of that region
        of Y beside a gene of X elsewhere is no copy of it.

        Args:
            genes (Iterable[str]): Gene ids with version.
            release (int): The release.

        Returns:
            set[str]: The Y copies among them.
        """
        from idtrack.graph import genes as genes_rule

        found = list(dict.fromkeys(genes))
        on_y = {gene for gene in found if self.in_pseudoautosomal_y(gene, release)}
        if not on_y:
            return set()
        facts = {gene: genes_rule.facts_of_nodes(self, [gene]) for gene in found}  # type: ignore[arg-type]
        return {
            gene
            for gene in on_y
            if any(other not in on_y and genes_rule.same_place(facts[gene], facts[other]) for other in found)
        }

    def _pseudoautosomal_masks(self) -> dict[int, int]:
        """Return ``{node: release mask}`` of the genes in the pseudoautosomal regions of Y, read once."""
        places = self._frozen().places
        masks: dict[int, int] = {}
        if not len(places):
            return masks
        import numpy as np

        from idtrack.graph.store import _words_to_masks

        for assembly, regions in self.repeated_regions.items():
            for region, first, last, _repeated, _there in regions:
                if region not in places.regions:
                    continue
                inside = (
                    (places.assembly == assembly)
                    & (places.region == places.regions.index(region))
                    & (places.start >= first)
                    & (places.end <= last)
                )
                rows = np.flatnonzero(inside)
                for row, mask in zip(rows.tolist(), _words_to_masks(places.masks[rows])):
                    node = int(places.node[row])
                    masks[node] = masks.get(node, 0) | mask
        return masks

    def _name_match(self, name: str, gene: str, database: str, mask: int) -> NameMatch:
        """Return the public record of a name attached to a gene at the releases of a mask.

        A synonym and its database are named without the graph's prefix and the record is marked as a synonym
        (decision 56).
        """
        return NameMatch(
            name=ids.public_name(name),
            gene=gene,
            database=ids.public_name(database),
            synonym=ids.is_synonym_database(database),
            releases=tuple(sorted(ReleaseSet.from_mask(mask))),
        )

    def genes_named(self, name: str, fold_case: bool = False) -> list[NameMatch]:
        """Return every gene that a name of an external database names, with the database and the releases.

        This is name resolution per database and release, as data: the caller decides which databases count (a
        symbol database, a database of names, synonyms) and at which release. A name counts at a release only on the
        assembly that serves that release, and only for a gene of the reference sequence there; a name a database
        recorded as a synonym of the gene is marked :py:attr:`idtrack.results.NameMatch.synonym`.

        Args:
            name (str): The name, as written (surrounding white space is part of it).
            fold_case (bool): Match without letter case; otherwise the name must be written as the database wrote it
                at the release (a database may write a name in another case at another release).

        Returns:
            list[NameMatch]: One record per (name, gene, database) with the releases at which it holds, sorted by
            database, gene and name; empty when the name names no gene.

        Raises:
            ArgumentTypeError: If *name* is not a string.
            InvalidArgumentError: If *name* begins with the graph's internal prefix of synonyms (decision 56).
        """
        _arguments.string(name, "name")
        if name.casefold().startswith(ids.SYNONYM_PREFIX.casefold()):
            raise InvalidArgumentError(
                f"`name`={name!r} begins with the graph's internal prefix {ids.SYNONYM_PREFIX!r}; give the name alone "
                "(synonyms are found as names, marked `synonym`).",
                argument="name",
                value=name,
            )
        store = self._frozen()
        line = {int(assembly): releases.mask for assembly, releases in self.serving_line.items()}
        found: dict[tuple[str, str, str], int] = {}
        for query in (name, ids.SYNONYM_PREFIX + name):
            numbers = set(store.lower_candidates(query.lower()))
            exact = store.index.get(query)
            if exact is not None:
                numbers.add(exact)
            for number in sorted(numbers):
                node = store.names[number]
                if self.node_type(node) != ids.EXTERNAL:
                    continue
                if not fold_case and node != query and number not in self._c_respelled:
                    continue
                for gene, connection in self.xref_edges(node):
                    if self.node_type(gene) != ids.GENE:
                        continue
                    reference = ~self.non_reference_releases(gene).mask
                    for database, assembly, releases in connection.items:
                        mask = releases.mask & line.get(int(assembly), 0) & reference
                        if not fold_case and number in self._c_respelled:  # the links spelt as the name is written
                            mask = store.spelt_mask(number, store.index[gene], database, mask, query)
                        for text, spelt in self._by_spelling(store, number, gene, database, mask).items():
                            key = (text, gene, database)
                            found[key] = found.get(key, 0) | spelt
        matches = [self._name_match(text, gene, database, mask) for (text, gene, database), mask in found.items()]
        return sorted(matches, key=lambda m: (m.database, m.synonym, m.gene, m.name))

    def names_of(self, gene: str, release: int | None = None) -> list[NameMatch]:
        """Return the names of external databases attached to a gene itself, with the releases at which they are.

        A name counts at a release only on the assembly that serves that release, and only while the gene lies on
        the reference sequence there. Names attached to the gene's transcripts or translations are not included.

        Args:
            gene (str): An Ensembl gene id with version.
            release (int | None): Only names that hold at this release; ``None`` for every release.

        Returns:
            list[NameMatch]: One record per (name, database), sorted by database and name.

        Raises:
            UnknownIdentifierError: If *gene* is not a gene id with version of the graph.
            ArgumentTypeError: If *release* is not an integer.
        """
        if not (isinstance(gene, str) and gene in self.nodes and self.node_type(gene) == ids.GENE):
            raise UnknownIdentifierError(
                f"{gene!r} is not an Ensembl gene id with version of this graph.", argument="gene", value=gene
            )
        wanted = _arguments.optional_release(release, "release")
        line = {int(assembly): releases.mask for assembly, releases in self.serving_line.items()}
        reference = ~self.non_reference_releases(gene).mask
        store = self._frozen()
        found: dict[tuple[str, str], int] = {}
        for name, connection in self.xref_edges(gene, reverse=True):
            if self.node_type(name) != ids.EXTERNAL:
                continue
            for database, assembly, releases in connection.items:
                mask = releases.mask & line.get(int(assembly), 0) & reference
                for text, spelt in self._by_spelling(store, store.index[name], gene, database, mask).items():
                    if wanted is None or spelt >> wanted & 1:
                        found[(text, database)] = found.get((text, database), 0) | spelt
        matches = [self._name_match(text, gene, database, mask) for (text, database), mask in found.items()]
        return sorted(matches, key=lambda m: (m.database, m.synonym, m.name))

    def _by_spelling(self, store: _Store, number: int, gene: str, database: str, mask: int) -> dict[str, int]:
        """Split the releases of a link of an external node to a gene by how the link spells the node there."""
        if not mask:
            return {}
        if number not in self._c_respelled:
            return {store.names[number]: mask}
        target = store.index[gene]
        split: dict[str, int] = {}
        for release in ReleaseSet.from_mask(mask):
            text = store.link_spelling(number, target, database, release)
            split[text] = split.get(text, 0) | (1 << release)
        return split

    def _is_accession(self, store: _Store, number: int) -> bool:
        """Tell whether an external node is a versioned accession and no id of a symbol database.

        A versioned accession is one to which a link attaches a version (``xref.version``, :py:meth:`accession_versions`):
        which databases give versions is Ensembl's statement, read from its cross references.
        """
        if not len(store.versions.of_node(number)):
            return False
        symbols = set(organisms.SYMBOL_DATABASES.get(self.organism or "", ()))
        return not any(
            database in symbols
            for _other, connection in store.xref_edges(number, False)
            for database in connection.databases
        )

    def _attached_at_last(self, name: str) -> bool:
        """Tell whether a link of a database whose versions date it attaches an accession at the graph's last release.

        Decision 14's condition, for accessions as for Ensembl ids: a version above every version the graph holds can
        be newer than the graph only for an accession that Ensembl still attaches at the last release (on the assembly
        that serves it); one that it no longer attaches cannot have a version newer than Ensembl's last import of it.
        """
        last = self._last_release_kept()
        return any(mask >> last & 1 for mask in self._judged_masks(name).values())

    def _judged_masks(self, name: str) -> dict[int, int]:
        """Return ``{version: releases}`` of an accession on the assembly that serves each release (decision 101's)."""
        store = self._frozen()
        number = store.index.get(name)
        if number is None:
            return {}
        counters = self.edit_counter_databases()
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        links = dict(self._c_xref(name, self._reverse_view))
        found: dict[int, int] = {}
        for target, database, version, held in store.versions.of_node(number):
            connection = links.get(store.names[target])
            if database in counters or connection is None:
                continue
            found[version] = found.get(version, 0) | (held & served_mask(connection.items, line, database))
        return found

    def _unversioned_at(self, name: str, release: int) -> bool:
        """Tell whether an external node is, at a release, an id of a database that gives it no version.

        A number can be the id of two numbering schemes (decision 86): a numeric Havana translation id of the NCBI36
        releases, to which Ensembl's cross references give a version, and NCBI's gene id, which has none. Which
        databases give an identifier versions is read from the graph (:py:meth:`accession_versions`): a database none of
        whose links of the node attaches a version writes it without one.

        Args:
            name (str): An external node.
            release (int): The release.

        Returns:
            bool: Whether a link of such a database holds at the release, on the assembly that serves it.
        """
        if release < 0:
            return False
        store = self._frozen()
        number = store.index.get(name)
        if number is None:
            return False
        versioned = {database for _target, database, _version, _mask in store.versions.of_node(number)}
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        for _other, connection in self._c_xref(name, self._reverse_view):
            for database in connection.databases - versioned:
                if served_mask(connection.items, line, database) >> release & 1:
                    return True
        return False

    def _exists_at(
        self, name: str, release: int, database: str | None, version: int | None = None
    ) -> tuple[bool, bool]:
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
            version (int | None): For an accession read with a version, only the links that attach that version count.

        Returns:
            tuple[bool, bool]: Whether the node exists at the release (as an id of ``database`` when one is named),
            and whether it is an id of ``database`` at any release (``True`` without a database).
        """
        if version is not None:
            mask = self._existence_mask(name, database, version)
            if database is None:
                return release >= 0 and bool(mask >> release & 1), True
            known = any(db == database for _t, db, v, _m in self._frozen().versions.of_node(self._frozen().index[name]))
            return release >= 0 and bool(mask >> release & 1), known
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

    def _other_releases(
        self, matches: list[tuple[str, str, str | None, int | None]]
    ) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
        """Return the readings, not existing at the release read, that exist at other releases, with those releases.

        Args:
            matches: ``(node, how, database, version)`` of the readings the steps reached that do not exist at the
                release read.

        Returns:
            tuple: ``(node, how, releases)`` of those that exist at some release (:py:meth:`_existence_mask`).
        """
        found = []
        for name, how, database, version in matches:
            mask = self._existence_mask(name, database, version)
            if mask:
                found.append((name, how, tuple(sorted(ReleaseSet.from_mask(mask)))))
        return tuple(found)

    def _latest_release(self, name: str, database: str | None, version: int | None = None) -> int:
        """Return the latest release at which an external identifier exists, as :py:meth:`_exists_at` counts it.

        Args:
            name (str): An external node.
            database (str | None): The database the identifier must be an id of; ``None``: any.
            version (int | None): For an accession read with a version, only the links that attach that version count.

        Returns:
            int: The release; ``-1`` when it exists at no release on the assembly that serves the release.
        """
        return self._existence_mask(name, database, version).bit_length() - 1

    def _existence_mask(self, name: str, database: str | None, version: int | None = None) -> int:
        """Return the releases at which an external identifier exists, as :py:meth:`_exists_at` counts it (a mask).

        Args:
            name (str): An external node.
            database (str | None): The database the identifier must be an id of; ``None``: any.
            version (int | None): For an accession read with a version, only the links that attach that version count
                (at the releases at which they attach it).

        Returns:
            int: The bit mask of the releases.
        """
        line = self._c_line
        if line is None:
            line = self._c_line = self.serving_line
        mask = 0
        if version is not None:
            store = self._frozen()
            links = dict(self._c_xref(name, self._reverse_view))
            for target, link_database, value, held in store.versions.of_node(store.index[name]):
                connection = links.get(store.names[target])
                if value == version and connection is not None and (database is None or link_database == database):
                    mask |= held & served_mask(connection.items, line, link_database)
            return mask
        for _, connection in self._c_xref(name, self._reverse_view):
            if database is None:
                mask |= connection.releases.mask & served_mask(connection.items, line)
            elif database in connection.databases:
                mask |= served_mask(connection.items, line, database)
        return mask

    def _last_release_kept(self) -> int:
        """Return the last release of a frozen graph, kept after the first resolution asked for it."""
        last = self._c_last
        if last is None:
            last = self._c_last = self.snapshot_release
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
            # The links that hold at the wanted release (few links of a node hold at one release).
            release, links = wanted, set()
            if wanted >= 0:
                for other, connection in edges:
                    if connection.releases.mask >> wanted & 1:
                        for item_database in connection.databases if database is None else (database,):
                            if served_mask(connection.items, line, item_database) >> wanted & 1:
                                links.add((other, item_database))
            if not links:  # not at that release: at its latest one
                masks: dict[tuple[str, str], int] = {}
                for on_line in (True, False):
                    for other, connection in edges:
                        for item in connection.items:
                            if database is None or item[0] == database:
                                mask = served_mask((item,), line) if on_line else item[2].mask
                                if mask:
                                    masks[(other, item[0])] = masks.get((other, item[0]), 0) | mask
                    if masks:
                        break
                if not masks:  # not an id of the database named: nothing but its name says how it is written
                    if query == name:
                        written.append(number)
                    continue
                release = max(mask.bit_length() for mask in masks.values()) - 1
                links = {link for link, mask in masks.items() if mask >> release & 1}
            index = store.index
            if any(store.link_spelling(number, index[other], db, release) == query for other, db in links):
                written.append(number)
        return written

    def name_candidates(self, identifier: str) -> tuple[str, ...]:
        """Return every external node that name resolution can read a string as, at some release.

        The nodes of the steps of :py:meth:`readings` (exact, without letter case, with separators read as one, without
        a version the graph does not hold), names and synonyms alike; which of them a release reads is the business of
        :py:meth:`readings` at that release.

        Args:
            identifier (str): The string.

        Returns:
            tuple[str, ...]: The nodes, in the order of the steps (each once).
        """
        store = self._frozen()
        found: list[int] = []
        for query in (identifier, f"{ids.SYNONYM_PREFIX}{identifier}"):
            exact = store.index.get(query)
            if exact is not None:
                found.append(exact)
            for step in _RESOLVE_STEPS[1:]:
                found.extend(self._matches(store, query, step))
        external = store.external_code
        return tuple(dict.fromkeys(store.names[i] for i in found if store.ntype_list[i] == external))

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
        if suffix and suffix.group(2) == ids.ID_VERSION_DELIMITER:
            core = suffix.group(1)
            found = store.index.get(core)
            if found is None:
                candidates = store.lower_candidates(core.lower())
                found = candidates[0] if candidates else None
            if found is not None and store.node_type(found) == ids.EXTERNAL and self._is_accession(store, found):
                return [found]
        return []

    # ------------------------------------------------------------------ edges
