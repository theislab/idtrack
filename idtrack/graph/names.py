"""Name resolution of a graph: which node a string means at a release, and which genes a name names.

The methods of :py:class:`NameResolution` are part of :py:class:`idtrack.graph.TheGraph`; they read its store only.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, SupportsIndex

from idtrack import _arguments, ids, organisms
from idtrack.errors import UnknownIdentifierError
from idtrack.graph.store import _FOLD_SEPARATORS, _SEPARATORS, _Store
from idtrack.graph.types import Connection, ReleaseSet, served_mask
from idtrack.results import NameMatch

_VERSION_SUFFIX = re.compile(r"^(.+)([._-])([0-9]+)$")
_RESOLVE_STEPS = ("exact", "case", "separator", "version")


class NameResolution:
    """What a string means in a graph (mixed into :py:class:`idtrack.graph.TheGraph`)."""

    if TYPE_CHECKING:
        # What the methods below use of :py:class:`idtrack.graph.TheGraph` (and of networkx), with its signatures.
        graph: Any  # the graph attributes of networkx (a dict)
        _reverse_view: bool
        _c_resolved: Callable[[str, int | None, str | None, int | None], tuple[str | None, str | None]]
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
        def serving_line(self) -> dict[int, ReleaseSet]:
            """See :py:attr:`idtrack.graph.TheGraph.serving_line`."""

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
        prefix = ids.versions_prefix(stable_id)
        found = [
            name
            for name in store.names_with_prefix(prefix)
            if name[len(prefix) :].isdigit() and store.type_has_version[store.ntype_list[store.index[name]]]
        ]
        return tuple(sorted(found, key=lambda name: int(name[len(prefix) :])))

    def spelling(self, node: str, database: str, release: int | None = None) -> str:
        """Return the spelling that ``database`` uses for an external identifier, at a release.

        The name of an external node is its canonical spelling: the one Ensembl gives at the latest release in which
        the identifier exists. Where databases, or the releases of one database, write an identifier with different
        letter case, the graph keeps one node and remembers how each database wrote it at each release.

        A release counts as the assembly that serves it in the backbone has it (:py:attr:`serving_assembly`): an
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
            UnknownIdentifierError: If the node is not in the graph.
        """
        store = self._frozen()
        try:
            number = store.index[node]
        except KeyError:
            raise UnknownIdentifierError(f"{node!r} is not a node of the graph.", argument="node", value=node) from None
        return store.spelling(number, database, release)

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
        ``from_release`` (without one: at the last release of the graph) is taken before one that does not. Databases
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
        that are not, one that is an id of that database at some release comes before one that never is.

        *The steps*, among the nodes that are equal in existence. They are tried in this order, each only if the
        one before found nothing. Each of the first four is tried on the names of the graph and then on its
        synonyms (the synonym node of the string, ``synonym_id::<string>``; the match is then reported as
        ``"synonym:<step>"``) before the next step begins, so that a string that is exactly a synonym of one gene is
        that synonym, not a spelling variant of another gene's name (``H3.3B``, a synonym, is not the name
        ``H3-3b``):

        1. ``"exact"``: the string is an identifier exactly as it is written. For an Ensembl id that is the name
           of its node. For an external identifier it is how a database (the one named, when ``from_database`` is given)
           writes it at the release the identifier is read at: ``from_release`` when the identifier exists there;
           without ``from_release`` the release a conversion reads it at, which is ``to_release`` when the identifier
           exists there and its latest release otherwise (the same when ``from_release`` does not hold it). Databases
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
           database of sequence accessions (:py:attr:`idtrack.ids.ACCESSION_DATABASE_PREFIXES`: RefSeq, UniProt,
           CCDS and the like) and of no symbol database of the organism
           (:py:attr:`idtrack.organisms.SYMBOL_DATABASES`): ``NM_000546.9`` for ``NM_000546``. Nothing else is ever
           cut off the name of an external identifier: ``SLA.7`` and ``SLA_11`` are not ``SLA``, ``NM_713028147``
           is not ``NM``.
        5. ``"version_newer_than_graph"``: the string is an Ensembl id with a version higher than every version the
           graph holds of it (written with ``.``, ``_`` or ``-``), and the id is active at the last release of the
           graph: the version may belong to a release newer than the graph, so the string stands for the latest
           version the graph holds, ``ENSG00000123456.99`` for ``ENSG00000123456.7``. Any other version that the
           graph does not hold (a lower one, one in a gap, any version of a retired id) matches nothing;
           :py:meth:`known_versions` then names the versions that exist.

        When no node the string can mean exists at the release, the first one in the order of the steps is returned
        (with ``from_database``: the first that is an id of that database at all, else the first), and the caller learns
        from the node where it does exist. The cost is a few dictionary look-ups, whatever the string.

        Args:
            identifier (str): The string to resolve.
            from_release (SupportsIndex | None): The Ensembl release the question is about (for a conversion: the
                release the identifier is read at). ``None`` means the last release of the graph. A release the graph
                does not have is one at which no external identifier exists.
            from_database (str | None): The external database the string is meant as an id of. ``None``: any
                database.
            to_release (SupportsIndex | None): Only without ``from_release``: the release a conversion goes to, at
                which an identifier is read when it exists there. It says at which release
                the spelling of an identifier is judged in the step ``"exact"``; which node exists is asked at the
                last release of the graph all the same. ``None``: the last release of the graph.

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
        return self._c_resolved(identifier, at, from_database, goal if at is None else None)

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
        spellings = ((identifier, ""), (f"{ids.SYNONYM_PREFIX}{identifier}", "synonym:"))
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
            if asked > int(self.version(latest)) and self.is_active(latest, self.snapshot_release):
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
        stable_id = store.names[hit] if hit is not None and store.node_type(hit) != ids.EXTERNAL else core
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
            ArgumentTypeError: If ``identifier`` is not a string.
        """
        _arguments.string(identifier, "identifier")
        if identifier in self._frozen().index:
            return ()
        unknown = self._unknown_version(identifier)
        return unknown[1] if unknown is not None else ()

    def _name_match(self, name: str, gene: str, database: str, mask: int) -> NameMatch:
        """Return the public record of a name attached to a gene at the releases of a mask."""
        return NameMatch(
            name=name,
            gene=gene,
            database=database,
            synonym=database.startswith(ids.SYNONYM_PREFIX),
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
        """
        _arguments.string(name, "name")
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
                        if not fold_case and node != query:  # written otherwise by some database at some release
                            mask &= sum(
                                1 << r for r in ReleaseSet.from_mask(mask) if self.spelling(node, database, r) == query
                            )
                        if mask:
                            key = (node, gene, database)
                            found[key] = found.get(key, 0) | mask
        matches = [self._name_match(node, gene, database, mask) for (node, gene, database), mask in found.items()]
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
        found: dict[tuple[str, str], int] = {}
        for name, connection in self.xref_edges(gene, reverse=True):
            if self.node_type(name) != ids.EXTERNAL:
                continue
            for database, assembly, releases in connection.items:
                mask = releases.mask & line.get(int(assembly), 0) & reference
                if mask and (wanted is None or mask >> wanted & 1):
                    found[(name, database)] = found.get((name, database), 0) | mask
        matches = [self._name_match(name, gene, database, mask) for (name, database), mask in found.items()]
        return sorted(matches, key=lambda m: (m.database, m.synonym, m.name))

    def _is_accession(self, store: _Store, number: int) -> bool:
        """Tell whether an external node is an id of an accession database and of no symbol database."""
        databases = {
            database for _other, connection in store.xref_edges(number, False) for database in connection.databases
        }
        symbols = set(organisms.SYMBOL_DATABASES.get(self.graph.get("organism"), ()))
        return (
            bool(databases)
            and not databases & symbols
            and any(database.startswith(ids.ACCESSION_DATABASE_PREFIXES) for database in databases)
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
