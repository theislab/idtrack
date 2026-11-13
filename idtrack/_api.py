#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import copy
import logging
import math
import operator
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from tqdm import tqdm

from idtrack._db import DB
from idtrack._track import InactiveAtReleaseError, SearchBudgetExceededError, Track
from idtrack._verify_organism import VerifyOrganism

if TYPE_CHECKING:
    from idtrack._database_manager import DatabaseManager

_PACKAGE_LOGGER = "idtrack"
_HANDLER_MARK = "_idtrack_console_handler"

# The error of `Track.convert` that concerns one query only and that the API does not answer itself: a search that
# exceeded its budget. It is recorded per item in batches; anything else propagates. (An identifier that is not
# active at the given `from_release` is converted from the release at which the graph holds it, and flagged.)
_PER_QUERY_ERRORS = (SearchBudgetExceededError,)

_INFER_MODES = ("complete", "ensembl_release", "assembly", "assembly_ensembl_release")


def _plain(obj: Any) -> Any:
    """Return *obj* as plain, JSON-serialisable Python data.

    Tuples become lists, sets and frozensets become sorted lists, NumPy scalars become Python scalars, mappings get
    string keys (tuples are joined by ``"|"``), and non-finite floats become the strings ``"inf"``, ``"-inf"`` and
    ``"nan"``.

    Args:
        obj: Any nested structure of containers and scalars.

    Returns:
        Any: The same content made of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and ``None`` only.
    """
    if isinstance(obj, dict):
        return {
            ("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _plain(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (set, frozenset)):
        return sorted((_plain(v) for v in obj), key=lambda v: (type(v).__name__, str(v)))
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _stable_id(identifier: str) -> str:
    """Return *identifier* without the ``synonym_id::`` prefix and, for Ensembl ids, without the version."""
    identifier = identifier.removeprefix(DB.synonym_id_nodes_prefix)
    if identifier.upper().startswith(("ENS", "LRG_")):
        head, sep, tail = identifier.rpartition(".")
        if sep and tail.isdigit():
            return head
    return identifier


class API:
    """Provide a high-level façade for building graphs and converting biological identifiers with IDTrack.

    One :py:class:`API` holds at most one loaded graph (one organism, one snapshot release). Once
    :py:meth:`build_graph` has returned, the object holds no other state that changes between calls: every
    conversion is a function of its arguments and the graph, results are plain data (``dict``, ``list``, ``str``,
    ``int``, ``bool``, ``None``; ``json.dumps`` works on every result), and the values returned by the ``list_*``
    methods are copies. Arguments are validated before any search starts, and an invalid argument raises an error
    that names it and the accepted values.

    Conversions only read the graph, so several threads may call :py:meth:`convert_identifier` at once with the same
    answers as one; the search holds the interpreter lock, so threads do not run conversions in parallel.
    :py:meth:`build_graph` prepares the new graph completely before it replaces the old one in a single assignment.
    """

    def __init__(self, local_repository: str) -> None:
        """Bind the interface to a local repository used for data downloads and on-disk caches.

        Constructing an :py:class:`API` is inexpensive: the graph is loaded by :py:meth:`build_graph`.

        Args:
            local_repository (str): Path to a writable directory where the package stores downloaded resources and
                graphs. The caller is responsible for ensuring the path exists and is accessible.

        Attributes:
            log (logging.Logger): Logger ``"idtrack._api"`` (a child of the package logger ``"idtrack"``).
            local_repository (str): The given repository path.
            track (idtrack.Track | None): The active tracker; set by :py:meth:`build_graph`.
        """
        self.log = logging.getLogger(__name__)
        self.local_repository = local_repository
        self.track: Track | None = None

    def _require_track(self) -> Track:
        """Return the active tracker or raise a clear error if the graph is not built yet."""
        track = self.track
        if track is None:
            raise RuntimeError("No graph is attached to this API instance. Call `API.build_graph(...)` first.")
        return track

    @staticmethod
    def configure_logger(level: int | str | None = None) -> None:
        """Show the messages of idtrack on the console, at *level* and above.

        Attaches one console handler (time, level, logger name, message) to the package logger ``"idtrack"`` and
        sets that logger's level. Only idtrack's loggers are affected: the root logger and the loggers of other
        libraries keep their handlers and levels. Calling it again changes the level and never adds a second
        handler. Applications that configure logging themselves can instead set
        ``logging.getLogger("idtrack").setLevel(...)`` and leave this method alone.

        Args:
            level (int | str | None): Logging level (e.g. ``logging.INFO``, ``"DEBUG"``). ``None`` means
                :py:data:`logging.INFO`.
        """
        logger = logging.getLogger(_PACKAGE_LOGGER)
        logger.setLevel(logging.INFO if level is None else level)
        if not any(getattr(h, _HANDLER_MARK, False) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            setattr(handler, _HANDLER_MARK, True)
            logger.addHandler(handler)

    def resolve_organism(self, tentative_organism_name: str | int) -> tuple[str, int]:
        """Normalize a tentative organism name and fetch the latest supported Ensembl release.

        Resolves a user-provided synonym (common name, shorthand, taxon ID) to the Ensembl species name (e.g.
        ``"homo_sapiens"``) and the newest Ensembl release of that species, through
        :py:class:`idtrack.VerifyOrganism` (Ensembl REST; the species tables are kept for the process for a day).

        Args:
            tentative_organism_name (str | int): Organism descriptor in any supported synonym form (e.g. ``"human"``,
                ``"hsapiens"``, ``9606``, ``"Macaca mulatta"``, ``"Mmul_10"`` or ``"homo_sapiens"``). Matching is
                case-insensitive. Ensembl species names and taxon IDs always resolve; a common name that several
                Ensembl species share (``"pig"`` is shared by the pig breeds) raises a :py:class:`ValueError` that
                lists the candidates.

        Returns:
            tuple[str, int]: ``(formal_name, latest_release)``.
        """
        vdf = VerifyOrganism(tentative_organism_name)
        return vdf.get_formal_name(), int(vdf.get_latest_release())

    def get_database_manager(
        self,
        organism_name: str,
        snapshot_release: int,
        genome_assembly: int | None = None,
        ignore_before: int | None = None,
    ) -> "DatabaseManager":
        """Create a database manager configured for an organism and a release-bounded snapshot.

        Args:
            organism_name (str): Canonical Ensembl species name (e.g. ``"homo_sapiens"``).
            snapshot_release (int): Most recent Ensembl release to include; later releases are ignored for
                reproducibility.
            genome_assembly (int | None): Assembly key of the organism table
                (:py:attr:`idtrack.DB.assembly_mysqlport_priority`; usually the code in Ensembl core schema names
                ``<organism>_core_<release>_<code>``): the assembly whose tables the manager reads (e.g. ``38`` =
                human GRCh38, ``37`` = human GRCh37, ``39`` = mouse GRCm39, ``111`` = pig Sscrofa11.1, ``10`` =
                rhesus Mmul_10, ``6`` = cynomolgus Macaca_fascicularis_6.0). If ``None`` (default), the
                highest-priority assembly of the organism that has *snapshot_release*, as :py:meth:`build_graph`
                uses; a graph always holds the organism's other assemblies of its window too
                (:py:meth:`list_genome_assemblies`).
            ignore_before (int | None): Earliest Ensembl release to include in the snapshot window. ``None``
                (default) is the first release of the organism in the organism table
                (:py:meth:`idtrack.DB.first_release`), the default of :py:class:`idtrack.DatabaseManager`.

        Returns:
            idtrack.DatabaseManager: A manager ready for graph building and conversions.
        """
        from idtrack._database_manager import DatabaseManager  # needed only to build or to test a graph

        return DatabaseManager(
            organism=organism_name,
            ensembl_release=None,
            ignore_before=ignore_before,
            ignore_after=snapshot_release,
            form=DB.backbone_form,
            local_repository=self.local_repository,
            genome_assembly=genome_assembly,
        )

    def build_graph(
        self,
        organism_name: str,
        snapshot_release: int,
        return_test: bool = False,
    ) -> None:
        """Load (or build) the graph of an organism and snapshot release and attach it to this object.

        If the local repository holds the graph file, it is loaded from the file alone: no
        :py:class:`idtrack.DatabaseManager`, no table and no network connection are needed
        (:py:meth:`idtrack.GraphMaker.load_graph`). A file that holds another graph than the one asked for (another
        selection of external databases, another organism table, another format) is an error that names the
        difference and the way out (:py:class:`idtrack.GraphMismatchError`), never a silent rebuild. Only if there is
        no file, the graph is built from Ensembl's tables through a :py:class:`idtrack.DatabaseManager` that ignores
        releases newer than *snapshot_release*, and saved. The new tracker replaces the previous one only when it
        is ready.

        Args:
            organism_name (str): Canonical Ensembl species name, typically the output of :py:meth:`resolve_organism`.
            snapshot_release (int): Ensembl release anchoring this build. Data from later releases are ignored. The
                graph is built on the organism's assemblies in their order of the organism table
                (:py:attr:`idtrack.DB.assembly_mysqlport_priority`): its primary assembly is the preferred one
                that has *snapshot_release*, and the ids of the other assemblies (earlier ones, and human GRCh37
                beside GRCh38) are in the graph as ids to convert from.
            return_test (bool): Attach a :py:class:`idtrack.TrackTests` (the graph's self-tests, which read Ensembl's
                tables and therefore always need a database manager) instead of a :py:class:`idtrack.Track`.

        Nothing is computed in advance for conversions: a graph, built or read from its file, carries its graph-wide
        tables, and what a conversion reads is computed for the nodes of the query.

        See Also:
            :py:meth:`get_database_manager`, :py:class:`idtrack.Track`.
        """
        track: Track
        if return_test:
            from idtrack._track_tests import TrackTests

            track = TrackTests(
                self.get_database_manager(organism_name=organism_name, snapshot_release=snapshot_release)
            )
        else:
            from idtrack._graph_maker import GraphMaker

            try:
                graph = GraphMaker.load_graph(self.local_repository, organism_name, snapshot_release)
            except FileNotFoundError:
                track = Track(self.get_database_manager(organism_name=organism_name, snapshot_release=snapshot_release))
            else:
                track = Track.from_graph(graph)
        self.track = track

    # ------------------------------------------------------------------------------------------------------------
    # Validation at the boundary

    def graph_releases(self) -> list[int]:
        """Return the Ensembl releases the loaded graph covers, ascending (valid ``from_release``/``to_release``)."""
        return sorted(int(r) for r in self._require_track().graph.graph["confident_for_release"])

    def _validated_release(self, value: Any, name: str, releases: list[int]) -> int:
        """Return *value* as an ``int`` release of the graph, or raise naming the argument and the valid range."""
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"`{name}` must be an integer Ensembl release, got {value!r}.")
        try:
            release = operator.index(value)
        except TypeError:
            raise TypeError(f"`{name}` must be an integer Ensembl release, got {value!r}.") from None
        if release not in releases:
            raise ValueError(
                f"`{name}`={release} is not a release of the loaded graph; valid releases are "
                f"{releases[0]} to {releases[-1]}."
            )
        return int(release)

    def valid_final_databases(self, to_release: int | None = None) -> list[str]:
        """Return the values accepted as ``final_database`` for a target release, sorted.

        These are the external databases of the genome assembly that is primary at *to_release* (a release of an
        earlier assembly has that assembly's databases) and the two Ensembl gene namespaces (versioned
        ``ensembl_gene`` and ``base_ensembl_gene``); see :py:meth:`idtrack.Track.final_databases`.

        Args:
            to_release (int | None): The target release; ``None`` means the graph's snapshot release.
        """
        return self._require_track().final_databases(to_release)

    def _validated_database(self, final_database: str | None, to_release: int) -> str:
        """Return the canonical spelling of *final_database* (``None`` means ``ensembl_gene``), or raise."""
        if final_database is None:
            return DB.nts_ensembl[DB.backbone_form]
        if not isinstance(final_database, str):
            raise TypeError(f"`final_database` must be a string or None, got {final_database!r}.")
        allowed = self.valid_final_databases(to_release)
        if final_database in allowed:
            return final_database
        folded = [d for d in allowed if d.casefold() == final_database.strip().casefold()]
        if len(folded) == 1:
            return folded[0]
        raise ValueError(
            f"`final_database`={final_database!r} is not a database of the loaded graph at Ensembl release "
            f"{to_release}; valid: {allowed}."
        )

    def _validated_source_database(self, from_database: str | None) -> str | None:
        """Return the canonical spelling of *from_database* (an external database of the graph), or raise."""
        if from_database is None:
            return None
        if not isinstance(from_database, str):
            raise TypeError(f"`from_database` must be a string or None, got {from_database!r}.")
        allowed = self.list_external_databases()
        folded = [d for d in allowed if d.casefold() == from_database.strip().casefold()]
        if len(folded) == 1:
            return folded[0]
        raise ValueError(
            f"`from_database`={from_database!r} is not an external database of the graph; valid: {allowed}."
        )

    @staticmethod
    def _validated_identifier(identifier: Any) -> str:
        """Return *identifier* without surrounding white space, or raise for a non-string or an empty string."""
        if not isinstance(identifier, str):
            raise TypeError(f"`identifier` must be a string, got {type(identifier).__name__} {identifier!r}.")
        stripped = identifier.strip()
        if not stripped:
            raise ValueError(f"`identifier` is empty: {identifier!r}.")
        return stripped

    def _validated_arguments(
        self, from_release: Any, to_release: Any, final_database: str | None, strategy: str
    ) -> tuple[int | None, int, str, bool]:
        """Validate the conversion arguments shared by single and batch conversions.

        Returns:
            tuple: ``(from_release, to_release, final_database, prioritize_to_one_filter)``.
        """
        if strategy not in ("best", "all"):
            raise ValueError(f"Invalid strategy={strategy!r}. Valid options are 'best' or 'all'.")
        track = self._require_track()
        releases = self.graph_releases()
        to_rel = (
            int(track.graph.graph["ensembl_release"])
            if to_release is None
            else self._validated_release(to_release, "to_release", releases)
        )
        from_rel = None if from_release is None else self._validated_release(from_release, "from_release", releases)
        return from_rel, to_rel, self._validated_database(final_database, to_rel), strategy == "best"

    # ------------------------------------------------------------------------------------------------------------
    # Conversion

    def convert_identifier(
        self,
        identifier: str,
        from_release: int | None = None,
        to_release: int | None = None,
        final_database: str | None = None,
        strategy: Literal["all", "best"] = "best",
        explain: bool = False,
        from_database: str | None = None,
    ) -> dict[str, Any]:
        """Convert one identifier to a target Ensembl release and database.

        The identifier is resolved to a node of the graph (:py:meth:`idtrack.TheGraph.resolve_name`): between the
        nodes the string can mean, one that exists at *from_release* (without one: at the last release of the graph)
        is taken before one that does not (with *from_database*: as an id of that database), and an Ensembl id is
        never passed over. An identifier exists at a release on the genome assembly that serves the release (a name
        that only human GRCh37, listed beside GRCh38, still carries does not exist today). Among those it is matched
        exactly (as a database wrote the identifier at the release it is read at), then ignoring letter case, then
        taking ``-``, ``_`` and ``.`` as one character, then without the
        version of a sequence accession, each step on names and then on synonyms before the next begins. How it
        matched is reported. :py:meth:`idtrack.Track.convert`, which reads the identifier by the same rule, finds
        its Ensembl gene(s) at *to_release* and their ids in *final_database*. Every argument is validated before
        the search starts.

        Args:
            identifier (str): Ensembl id (with or without version), symbol or other external id. Surrounding white
                space is ignored.
            from_release (int | None): Ensembl release at which *identifier* is read. If ``None``, it is read at
                *to_release* when it is active there, otherwise at the latest release in which it is active. If a
                release is given at which the graph does not hold the identifier (a real id of that release that
                the graph's selection of external databases lacks there, or a wrong release), the identifier is
                read by the same rule as without *from_release*, and the result says so:
                ``"from_release_fallback"`` is true, ``"from_release"`` is the release used and
                ``"from_release_given"`` the one given. (:py:meth:`idtrack.Track.convert` itself is strict and
                raises :py:class:`idtrack.InactiveAtReleaseError`.)
            to_release (int | None): Target Ensembl release; ``None`` means the graph's snapshot release. Must be a
                release of the graph (:py:meth:`graph_releases`).
            final_database (str | None): Database of the targets (:py:meth:`valid_final_databases` of *to_release*,
                matched case-insensitively); ``None`` means ``"ensembl_gene"``.
            strategy (Literal["all", "best"]): ``"best"`` (default) keeps the best answer: one target, or the
                targets that are equal in all evidence (``"tie"`` is then true); ``"all"`` keeps every target
                reached.
            explain (bool): Add ``"the_path"``, the edges by which each target was reached.
            from_database (str | None): The external database *identifier* is meant as (an id that names
                different things in two databases is then read in this one only); one of
                :py:meth:`list_external_databases`. ``None`` (default) reads it in any database.

        Returns:
            dict[str, Any]: Plain data (``json.loads(json.dumps(r)) == r``):

                - ``"query_id"`` (str): *identifier* as given.
                - ``"graph_id"`` (str | None): The graph node the identifier was read as (``None``: not found). For
                  an Ensembl transcript or translation id without version this is its version of the starting
                  release.
                - ``"match"`` (str | None): How the identifier matched ``"graph_id"``: ``"exact"``, ``"case"``,
                  ``"separator"``, ``"version"`` (each also after ``"synonym:"`` for a synonym), or
                  ``"version_newer_than_graph"`` (an Ensembl id with a version higher than all the graph holds,
                  active at the graph's last release: read as its latest version; see
                  :py:meth:`idtrack.TheGraph.resolve_name`); ``None`` when it is not in the graph.
                - ``"known_versions"`` (list[str]): Empty, except for an Ensembl id written with a version that the
                  graph does not hold and that cannot be newer than the graph: such an id is not answered with
                  another version's gene (``"no_corresponding"`` is true), and this lists the versions that exist.
                - ``"from_release"`` (int | None): The release at which the identifier was read: the argument when
                  given and usable, otherwise the release the engine chose (``None`` when nothing was read).
                - ``"from_release_inferred"`` (bool): The release was chosen by the engine, not given.
                - ``"from_release_given"`` (int | None), ``"from_release_fallback"`` (bool): The argument, and
                  whether it was set aside because the graph does not hold the identifier at that release.
                - ``"from_databases"`` (list[str]): For an external id, the databases through which it was read at
                  that release: every database that attaches it there to one of its genes, to the gene itself or to
                  one of its transcripts or translations (with *from_database*: that database).
                - ``"to_release"`` (int), ``"strategy"`` (str): The arguments as used (``to_release`` resolved).
                - ``"final_database"`` (str | None): The database of the values in ``"target_id"``: the requested
                  one when a target in it was found, ``"ensembl_gene"`` when the genes reached have no id in it
                  (``no_target``), ``None`` when nothing was found.
                - ``"target_id"`` (list[str]): The targets, in the order of ``"last_node"``, each once. One
                  element for an unambiguous answer.
                - ``"last_node"`` (list[list[str]]): ``[ensembl_gene, target]`` for each target; the gene is the
                  Ensembl gene at *to_release* through which the target was reached. The order is the engine's,
                  best first: with ``"all"`` the genes of the best answer lead, in the order in which ``"best"``
                  returns them, and the others follow by their evidence. The pairs of a tie are ordered by the
                  mapping score of each to the gene the identifier was read as, then by gene id and target id.
                - ``"ambiguous"`` (bool): More than one target (a 1-to-n answer).
                - ``"tie"`` (bool): The best answer of this conversion is shared by several genes or targets that
                  are equal in all evidence (with ``"best"`` these are the pairs returned). A tie is judged among
                  the genes that have an id in the requested database: genes without one are not in the result
                  (``"no_target_genes"``), so a symbol that two databases give to two genes is a tie when converted
                  to Ensembl gene ids and one answer when converted to a database in which only one of the two
                  genes has an id.
                - ``"alternative_locus_genes"`` (list[str]): Genes on alternative loci (haplotypes) and assembly
                  patches, and LRG records, that the identifier also leads to, beside genes of the reference
                  sequence: copies with stable ids of their own, which are not in ``"target_id"``. Empty when there
                  is none, and when the copies are themselves the answer (nothing of the reference sequence was
                  reached).
                - ``"no_corresponding"`` (bool): The identifier is not in the graph.
                - ``"no_conversion"`` (bool): It is in the graph, but no path to *to_release* exists (1-to-0).
                - ``"no_target"`` (bool): Genes were reached, but none has an id in the requested database; then
                  ``"target_id"`` holds the Ensembl gene ids and ``"final_database"`` is ``"ensembl_gene"``.
                - ``"no_target_genes"`` (list[str]): Genes that were reached but have no id in the requested
                  database while others have (they are not in ``"target_id"``); otherwise empty.
                - ``"error"`` (None | dict): Always ``None`` here (errors raise); in
                  :py:meth:`convert_identifier_multiple` a per-query failure as ``{"type", "message"}``.
                - ``"the_path"`` (list[dict]): Only with *explain*: for each pair of ``"last_node"`` (same order),
                  ``{"ensembl_gene", "target_id", "entry_gene", "external_jump", "external_gap", "score",
                  "edges"}``: the gene the identifier was read as and from which this gene was reached, the number
                  of bridges through external ids (0: Ensembl's history only), the releases between leaving a gene
                  and entering the next over those bridges, the product of Ensembl's mapping scores, and the edges
                  of the history walk and then of the final hop, as lists.

        Raises:
            TypeError: If *identifier* is not a string or a release is not an integer.
            ValueError: If *identifier* is empty, a release is not one of the graph, *final_database* is not a
                database of the graph at *to_release* or *strategy* is not ``"best"``/``"all"``.
            idtrack.SearchBudgetExceededError: If the search needs more than its budget
                (:py:attr:`idtrack.Track.search_budget_states`, :py:attr:`idtrack.Track.search_budget_seconds`).
            RuntimeError: If no graph is loaded.
        """
        query = self._validated_identifier(identifier)
        from_rel, to_rel, database, to_one = self._validated_arguments(
            from_release, to_release, final_database, strategy
        )
        source = self._validated_source_database(from_database)
        return self._convert(identifier, query, from_rel, to_rel, database, to_one, explain, source)

    def _convert(
        self,
        identifier: str,
        query: str,
        from_release: int | None,
        to_release: int,
        final_database: str,
        prioritize_to_one: bool,
        explain: bool,
        from_database: str | None = None,
    ) -> dict[str, Any]:
        """Convert a validated query (see :py:meth:`convert_identifier`)."""
        track = self._require_track()
        graph = track.graph

        def read(release: int | None) -> tuple[str | None, str | None, list[str]]:
            """Say which node the query means at a release, as the engine will read it (for the report)."""
            node, how = graph.resolve_name(query, release, from_database, to_release)
            versions: list[str] = []
            if node is None:
                versions = list(graph.known_versions(query))
                if not versions and graph.versions_of(query):
                    # A transcript or translation id without version: the engine reads it as its version of the
                    # release.
                    node, how = query, "exact"
            return node, how, versions

        graph_id, match, known_versions = read(from_release)
        raw = None
        fallback = False
        if graph_id is not None:
            # The engine is given the string and reads it by the same rule for the same release.
            arguments: dict[str, Any] = {
                "from_id": query,
                "to_release": to_release,
                "final_database": final_database,
                "prioritize_to_one_filter": prioritize_to_one,
                "return_path": explain,
                "from_database": from_database,
            }
            try:
                raw = track.convert(from_release=from_release, **arguments)
            except InactiveAtReleaseError:
                # The graph does not hold the identifier at the release given: read it where the graph holds it.
                raw = track.convert(from_release=None, **arguments)
                fallback = True
                graph_id, match, known_versions = read(None)
        pairs = self._candidates(raw, explain) if raw else []
        first = next(iter(raw.values())) if raw else None
        target_ids = list(dict.fromkeys(pair["target"] for pair in pairs))
        result: dict[str, Any] = {
            "query_id": identifier,
            "graph_id": first["from_id"] if first else graph_id,
            "match": match,
            "known_versions": known_versions,
            "from_release": first["from_release"] if first else (None if fallback else from_release),
            "from_release_inferred": first["from_release_inferred"] if first else (fallback or from_release is None),
            "from_release_given": from_release,
            "from_release_fallback": fallback,
            "from_databases": list(first["from_databases"]) if first else [],
            "to_release": to_release,
            "strategy": "best" if prioritize_to_one else "all",
            "final_database": first["final_conversion"]["final_database"] if first else None,
            "target_id": target_ids,
            "last_node": [[pair["gene"], pair["target"]] for pair in pairs],
            "ambiguous": len(target_ids) > 1,
            "tie": bool(first["tie"]) if first else False,
            "alternative_locus_genes": list(first["alternative_locus_genes"]) if first else [],
            "no_corresponding": graph_id is None,
            "no_conversion": graph_id is not None and not pairs,
            "no_target": bool(first["final_conversion"]["no_target"]) if first else False,
            "no_target_genes": list(first["final_conversion"]["genes_without_target"]) if first else [],
            "error": None,
        }
        if explain:
            result["the_path"] = [
                {
                    "ensembl_gene": pair["gene"],
                    "target_id": pair["target"],
                    "entry_gene": pair["entry_gene"],
                    "external_jump": pair["external_jump"],
                    "external_gap": pair["external_gap"],
                    "score": pair["score"],
                    "edges": _plain(pair["path"]),
                }
                for pair in pairs
            ]
        return result

    @staticmethod
    def _candidates(raw: dict, explain: bool) -> list[dict[str, Any]]:
        """Flatten the answer of :py:meth:`idtrack.Track.convert` into one record per (gene, target) pair.

        This is the only place that reads the engine's nested answer. The order of the engine (best first) is kept.

        Args:
            raw: The engine's result: per gene its evidence, ``the_path`` and ``final_conversion`` with
                ``final_elements`` per target.
            explain: Whether paths were requested.

        Returns:
            list[dict]: ``{"gene", "target", "entry_gene", "external_jump", "external_gap", "score", "path"}``.
        """
        records = []
        for gene, value in raw.items():
            for target, element in value["final_conversion"]["final_elements"].items():
                path = None
                if explain:
                    path = list(value.get("the_path") or ()) + list(element.get("the_path") or ())
                records.append(
                    {
                        "gene": str(gene),
                        "target": str(target),
                        "entry_gene": value["entry_gene"],
                        "external_jump": int(value["external_jump"]),
                        "external_gap": int(value["external_gap"]),
                        "score": float(value["score"]),
                        "path": path,
                    }
                )
        return records

    def convert_identifier_multiple(
        self,
        identifier_list: list[str],
        from_release: int | None = None,
        to_release: int | None = None,
        final_database: str | None = None,
        strategy: Literal["all", "best"] = "best",
        explain: bool = False,
        from_database: str | None = None,
        verbose: bool = True,
        pbar_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Convert a batch of identifiers with the same arguments.

        The arguments are validated once, before any conversion: a non-string or empty identifier raises with the
        positions of the offending items, as do invalid releases or databases. A failure that concerns one query
        only (a search that exceeds its budget) does not stop the batch: that item gets
        ``"error": {"type": ..., "message": ...}``, empty targets and all three ``no_*`` flags false. Each distinct
        identifier is converted once; repeated identifiers get equal, independent copies of the result.

        Args:
            identifier_list (list[str]): Identifiers to convert.
            from_release (int | None): See :py:meth:`convert_identifier`.
            to_release (int | None): See :py:meth:`convert_identifier`.
            final_database (str | None): See :py:meth:`convert_identifier`.
            strategy (Literal["all", "best"]): See :py:meth:`convert_identifier`.
            explain (bool): See :py:meth:`convert_identifier`.
            from_database (str | None): See :py:meth:`convert_identifier`.
            verbose (bool): Show a :py:mod:`tqdm` progress bar. Defaults to ``True``.
            pbar_prefix (str): Label of the progress bar.

        Returns:
            list[dict[str, Any]]: One result per input identifier, in input order, each of the shape returned by
            :py:meth:`convert_identifier`.

        Raises:
            TypeError: If *identifier_list* is not a list or tuple, or holds non-strings (positions named).
            ValueError: If it holds empty strings (positions named) or an argument is invalid.
        """
        if not isinstance(identifier_list, (list, tuple)):
            raise TypeError(f"`identifier_list` must be a list of strings, got {type(identifier_list).__name__}.")
        not_strings = [i for i, x in enumerate(identifier_list) if not isinstance(x, str)]
        if not_strings:
            raise TypeError(
                f"`identifier_list` holds {len(not_strings)} non-string items, at positions {not_strings[:20]} "
                f"(e.g. {identifier_list[not_strings[0]]!r})."
            )
        empty = [i for i, x in enumerate(identifier_list) if not x.strip()]
        if empty:
            raise ValueError(f"`identifier_list` holds {len(empty)} empty strings, at positions {empty[:20]}.")
        from_rel, to_rel, database, to_one = self._validated_arguments(
            from_release, to_release, final_database, strategy
        )
        source = self._validated_source_database(from_database)

        by_identifier: dict[str, dict[str, Any]] = {}
        distinct = list(dict.fromkeys(identifier_list))
        with tqdm(distinct, mininterval=0.25, disable=not verbose, desc=pbar_prefix, ncols=100) as loop_obj:
            for identifier in loop_obj:
                try:
                    by_identifier[identifier] = self._convert(
                        identifier, identifier.strip(), from_rel, to_rel, database, to_one, explain, source
                    )
                except _PER_QUERY_ERRORS as error:
                    by_identifier[identifier] = self._error_result(identifier, from_rel, to_rel, to_one, error)
        seen: set[str] = set()
        results = []
        for identifier in identifier_list:
            result = by_identifier[identifier]
            results.append(copy.deepcopy(result) if identifier in seen else result)
            seen.add(identifier)
        return results

    @staticmethod
    def _error_result(
        identifier: str, from_release: int | None, to_release: int, to_one: bool, error: BaseException
    ) -> dict[str, Any]:
        """Return the batch result of a query whose conversion raised a per-query error."""
        return {
            "query_id": identifier,
            "graph_id": getattr(error, "identifier", None),
            "match": None,
            "known_versions": [],
            "from_release": from_release,
            "from_release_inferred": from_release is None,
            "from_release_given": from_release,
            "from_release_fallback": False,
            "from_databases": [],
            "to_release": to_release,
            "strategy": "best" if to_one else "all",
            "final_database": None,
            "target_id": [],
            "last_node": [],
            "ambiguous": False,
            "tie": False,
            "alternative_locus_genes": [],
            "no_corresponding": False,
            "no_conversion": False,
            "no_target": False,
            "no_target_genes": [],
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    def classify_multiple_conversion(self, matchings: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """Group batch-conversion results into bins for reporting.

        Every result is in ``"input_identifiers"`` and in exactly one of the bins ``"matching_1_to_0"`` (not found,
        no conversion, or an error), ``"matching_1_to_1"``, ``"matching_1_to_n"``, ``"alternative_target_1_to_1"``
        and ``"alternative_target_1_to_n"`` (the last two: ``no_target``, the targets are Ensembl genes standing in
        for a missing id of the requested database). A result of ``"matching_1_to_1"`` is also in
        ``"changed_only_1_to_1"`` when its target is not the identifier it resolved to, and one of
        ``"matching_1_to_n"`` is also in ``"changed_only_1_to_n"`` when none of its targets is. "The identifier it
        resolved to" is ``graph_id``, compared without the ``synonym_id::`` prefix and, for Ensembl ids, without the
        version, so that a query without version or in another case that maps to itself is not "changed".

        Args:
            matchings (list[dict[str, Any]]): Results of :py:meth:`convert_identifier_multiple` or
                :py:meth:`convert_identifier`.

        Returns:
            dict[str, list[dict[str, Any]]]: The bins named above.

        Raises:
            ValueError: If a result has no target although none of ``no_corresponding``, ``no_conversion`` or
                ``error`` is set (an internal error).
        """
        result: dict[str, list[dict]] = {
            "changed_only_1_to_n": [],
            "changed_only_1_to_1": [],
            "alternative_target_1_to_1": [],
            "alternative_target_1_to_n": [],
            "matching_1_to_0": [],
            "matching_1_to_1": [],
            "matching_1_to_n": [],
            "input_identifiers": [],
        }

        for i in matchings:
            result["input_identifiers"].append(i)
            if i["no_corresponding"] or i["no_conversion"] or i.get("error"):
                result["matching_1_to_0"].append(i)
                continue
            if len(i["target_id"]) == 0:
                raise ValueError(
                    f"Unexpected conversion result: query_id={i.get('query_id')!r} returned an empty target_id list. "
                    "This indicates an internal error in the conversion logic. Please report this issue."
                )
            one = len(i["target_id"]) == 1
            if i["no_target"]:
                result["alternative_target_1_to_1" if one else "alternative_target_1_to_n"].append(i)
                continue
            resolved = _stable_id(i.get("graph_id") or i["query_id"])
            unchanged = {_stable_id(t) for t in i["target_id"]}
            if one:
                result["matching_1_to_1"].append(i)
                if resolved not in unchanged:
                    result["changed_only_1_to_1"].append(i)
            else:
                result["matching_1_to_n"].append(i)
                if resolved not in unchanged:
                    result["changed_only_1_to_n"].append(i)
        return result

    def print_binned_conversion(self, classified: dict[str, list[dict]]) -> None:
        """Log a multi-line summary of :py:meth:`classify_multiple_conversion` at INFO level.

        Args:
            classified (dict[str, list[dict]]): Output of :py:meth:`classify_multiple_conversion`.
        """
        self.log.info(self.binned_conversion_summary(classified))

    @staticmethod
    def binned_conversion_summary(classified: dict[str, list[dict]]) -> str:
        """Return the summary that :py:meth:`print_binned_conversion` logs.

        Args:
            classified (dict[str, list[dict]]): Output of :py:meth:`classify_multiple_conversion`.

        Returns:
            str: Counts and percentages per bin, and the counts of the ``no_*`` flags and of errors.
        """
        items = classified.get("input_identifiers", [])
        total = len(items)
        one_to_zero = len(classified.get("matching_1_to_0", []))
        one_to_one_target = len(classified.get("matching_1_to_1", []))
        one_to_n_target = len(classified.get("matching_1_to_n", []))
        changed_1_to_1 = len(classified.get("changed_only_1_to_1", []))
        changed_1_to_n = len(classified.get("changed_only_1_to_n", []))
        alt_1_to_1 = len(classified.get("alternative_target_1_to_1", []))
        alt_1_to_n = len(classified.get("alternative_target_1_to_n", []))
        one_to_one_total = one_to_one_target + alt_1_to_1
        one_to_n_total = one_to_n_target + alt_1_to_n
        rest_1_to_1 = one_to_one_target - changed_1_to_1

        def pct(part, whole):
            return (part / whole * 100) if whole else 0

        return (
            f"\nIDTrack conversion summary:\n"
            f"  Total processed: {total}\n"
            f"  1→0: {one_to_zero} ({pct(one_to_zero, total):.1f}%)\n"
            f"  1→1: {one_to_one_total} ({pct(one_to_one_total, total):.1f}%)\n"
            f"    Changed only: {changed_1_to_1} ({pct(changed_1_to_1, one_to_one_total):.1f}%)\n"
            f"    Alternative targets: {alt_1_to_1} ({pct(alt_1_to_1, one_to_one_total):.1f}%)\n"
            f"    Rest: {rest_1_to_1} ({pct(rest_1_to_1, one_to_one_total):.1f}%)\n"
            f"  1→n: {one_to_n_total} ({pct(one_to_n_total, total):.1f}%)\n"
            f"    Changed only: {changed_1_to_n} ({pct(changed_1_to_n, one_to_n_total):.1f}%)\n"
            f"    Alternative targets: {alt_1_to_n} ({pct(alt_1_to_n, one_to_n_total):.1f}%)\n"
            f"  Diagnostics:\n"
            f"    no_corresponding: {sum(bool(x['no_corresponding']) for x in items)}\n"
            f"    no_conversion:   {sum(bool(x['no_conversion']) for x in items)}\n"
            f"    no_target:       {sum(bool(x['no_target']) for x in items)}\n"
            f"    errors:          {sum(bool(x.get('error')) for x in items)}"
        )

    def infer_identifier_source(
        self, id_list: list[str], mode: str = "assembly_ensembl_release", report_only_winner: bool = True
    ) -> list:
        """Infer the origins (database, assembly, release) that best explain a list of identifiers.

        Each identifier found in the graph counts once for every origin in which it exists
        (:py:meth:`idtrack.TheGraph.trios`, read for the given identifiers only). Identifiers not in the graph are
        counted and logged, not used. A string is read as the node it means at the last release of the graph
        (:py:meth:`idtrack.TheGraph.resolve_name`): the release of the list is what is asked for, so no other
        release can be named; a string that meant one identifier once and means another today counts for today's.

        Args:
            id_list (list[str]): Identifiers (strings).
            mode (str): Granularity of an origin: ``"complete"`` → ``[database, assembly, release]``,
                ``"ensembl_release"`` → ``release``, ``"assembly"`` → ``assembly``, ``"assembly_ensembl_release"``
                (default) → ``[assembly, release]``.
            report_only_winner (bool): ``True`` (default) returns every origin with the highest count (all origins
                tied for first place, sorted: e.g. all releases of the span in which the identifiers exist);
                ``False`` returns every origin with its count, by descending count, then origin.

        Returns:
            list: With *report_only_winner*, the sorted list of best origins; otherwise ``[origin, count]`` pairs.

        Raises:
            ValueError: If *mode* is unknown, or none of the identifiers is in the graph.
            TypeError: If an identifier is not a string.
        """
        if mode not in _INFER_MODES:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of: {', '.join(_INFER_MODES)}.")
        not_strings = [i for i, x in enumerate(id_list) if not isinstance(x, str)]
        if not_strings:
            raise TypeError(f"`id_list` holds non-strings at positions {not_strings[:20]}.")
        track = self._require_track()
        found_id_list = []
        unfound = 0
        for identifier in id_list:
            stripped = identifier.strip()
            found_id, _ = track.graph.resolve_name(stripped, track.graph.last_release) if stripped else (None, None)
            if found_id is None:
                unfound += 1
            else:
                found_id_list.append(found_id)
        if unfound:
            self.log.warning(f"{unfound} of {len(id_list)} identifiers are not in the graph and are not used.")
        if not found_id_list:
            raise ValueError(
                f"None of the {len(id_list)} identifiers is in the graph, so their source cannot be inferred."
            )
        counts = Counter({_plain_origin(o): c for o, c in track.identify_source(found_id_list, mode=mode)})
        ranking = sorted(counts.items(), key=lambda oc: (-oc[1], _origin_key(oc[0])))
        if not report_only_winner:
            return [[_plain(origin), count] for origin, count in ranking]
        best = ranking[0][1]
        return [_plain(origin) for origin, count in ranking if count == best]

    def list_external_databases(self) -> list[str]:
        """Return the external (non-Ensembl) databases of the graph, sorted (a new list)."""
        return sorted(self._require_track().graph.available_external_databases)

    def list_genome_assemblies(self) -> list[int]:
        """Return the genome assemblies of the graph (e.g. ``38`` for GRCh38), sorted (a new list)."""
        return sorted(int(a) for a in self._require_track().graph.available_genome_assemblies)

    def list_external_databases_by_assembly(self) -> dict[int, list[str]]:
        """Return ``{assembly: sorted external databases}`` of the graph (new objects)."""
        by_assembly = self._require_track().graph.available_external_databases_assembly
        return {int(a): sorted(by_assembly[a]) for a in sorted(by_assembly)}

    def external_database_forms(self) -> dict[str, str]:
        """Return ``{external database: Ensembl form it connects through}`` (``"gene"``, ``"transcript"``, ...).

        The table (:py:attr:`idtrack.TheGraph.external_database_connection_form`) comes with the graph file. The
        returned dict is a new object.
        """
        forms = self._require_track().graph.external_database_connection_form
        return {str(k): str(forms[k]) for k in sorted(forms)}

    def list_ensembl_releases(self) -> list[int]:
        """Return the Ensembl releases the loaded graph covers, ascending; the same as :py:meth:`graph_releases`.

        These are all releases of the graph's window, of every assembly it holds (for pig 56 to 115, three
        assemblies), not only those of the primary assembly; each is a valid ``from_release`` and ``to_release``.
        No network is used.
        """
        return self.graph_releases()


def _plain_origin(origin: Any) -> Any:
    """Return an origin of :py:meth:`idtrack.Track.identify_source` as a hashable of plain scalars."""
    if isinstance(origin, tuple):
        return tuple(_plain(v) for v in origin)
    return _plain(origin)


def _origin_key(origin: Any) -> tuple:
    """Total order on origins (scalars or tuples of mixed scalars)."""
    values = origin if isinstance(origin, tuple) else (origin,)
    return tuple((type(v).__name__, v) for v in values)
