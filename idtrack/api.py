"""The public API of idtrack: load (or build) a graph, convert identifiers, describe the graph, look up names.

An :py:class:`API` holds one loaded graph (one organism, one snapshot release) and nothing else that changes between
calls: every conversion is a function of its arguments and the graph, every result is a frozen dataclass of
:py:mod:`idtrack.results` with a strict-JSON ``to_dict()``, and every argument is checked once, here, before any
search starts. A wrong argument raises an :py:class:`idtrack.errors.InputError` that names it and the accepted
values; nothing else is the caller's fault.

Loading a graph file (:py:meth:`API.load`) reads that one file and nothing else: no network, no table, and none of
the build stack (pandas, h5py, PyMySQL, PyYAML) is imported. Building (:py:meth:`API.build`) is its own call.

Conversions only read the graph, so several threads may convert at once with the same answers as one; the search
holds the interpreter lock, so a server scales with processes, not threads.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from idtrack import _arguments, ids, organisms
from idtrack.engine.track import Conversion, Track
from idtrack.errors import (
    ArgumentTypeError,
    GraphMismatchError,
    InactiveAtReleaseError,
    InputError,
    InvalidArgumentError,
    MissingValueError,
    UnknownIdentifierError,
)
from idtrack.graph import files
from idtrack.graph.graph import TheGraph
from idtrack.results import (
    REASONS,
    AssemblyDescription,
    ConversionResult,
    ErrorInfo,
    ExplainedPath,
    GraphDescription,
    NameMatch,
    check_strategy,
)
from idtrack.settings import GraphSettings, SearchSettings, check_graph_settings, check_search

if TYPE_CHECKING:
    from idtrack.build.database_manager import DatabaseManager

_PACKAGE_LOGGER = "idtrack"
_HANDLER_MARK = "_idtrack_console_handler"

# The explained path of a result without a path (asked for with `explain`): no node, no step.
_NO_PATH = ExplainedPath(nodes=(), steps=())


#: The bins of :py:meth:`API.classify`.
BINS = (
    "changed_only_1_to_n",
    "changed_only_1_to_1",
    "alternative_target_1_to_1",
    "alternative_target_1_to_n",
    "matching_1_to_0",
    "matching_1_to_1",
    "matching_1_to_n",
    "input_identifiers",
)


def _plain(obj: Any) -> Any:
    """Return an origin of :py:meth:`API.infer_source` as plain JSON data (tuples to lists, NumPy to Python)."""
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj


def _plain_origin(origin: Any) -> Any:
    """Return an origin as a hashable of plain scalars."""
    if isinstance(origin, tuple):
        return tuple(_plain(v) for v in origin)
    return _plain(origin)


def _origin_key(origin: Any) -> tuple:
    """Total order on origins (scalars or tuples of mixed scalars)."""
    values = origin if isinstance(origin, tuple) else (origin,)
    return tuple((type(v).__name__, v) for v in values)


class API:
    """The public interface of idtrack over one loaded graph.

    Make one with :py:meth:`load` (a graph file), :py:meth:`build` (from Ensembl, or the file of the same graph in a
    local repository) or ``API(graph)`` (a graph in memory).

    Attributes:
        graph (TheGraph): The graph.
        engine (Track): The conversion engine over the graph.
    """

    def __init__(self, graph: TheGraph, *, search: SearchSettings | None = None) -> None:
        """Hold a graph.

        Args:
            graph: The graph (:py:meth:`idtrack.graph.TheGraph.load`, or built by :py:class:`idtrack.build.GraphMaker`).
            search: The default search settings; ``None`` for the defaults of :py:class:`idtrack.settings.SearchSettings`.

        Raises:
            ArgumentTypeError: If *graph* is not a :py:class:`idtrack.graph.TheGraph`, or *search* not a
                :py:class:`idtrack.settings.SearchSettings`.
        """
        if not isinstance(graph, TheGraph):
            raise ArgumentTypeError(
                f"`graph` must be an idtrack graph (TheGraph), got {type(graph).__name__}; load a file with "
                "`API.load(path)`.",
                argument="graph",
            )
        self.log = logging.getLogger(__name__)
        self.graph = graph
        self.engine = Track(graph, search)

    @property
    def search(self) -> SearchSettings:
        """The default budget and bridge rules of a conversion (the engine's; a call can give its own budget).

        Returns:
            SearchSettings: The settings.
        """
        return self.engine.search

    # ------------------------------------------------------------------------------------------------------------
    # Loading and building

    @classmethod
    def load(
        cls, path: str, *, search: SearchSettings | None = None, graph_settings: GraphSettings | None = None
    ) -> API:
        """Load the graph in a file. Never builds, never uses the network, never imports the build stack.

        The file's own record says which graph it holds (organism, window, selection of external databases); it is
        checked against the format this version of idtrack reads and against the organism table of this process
        (an organism registered otherwise than when the graph was built would order its assemblies otherwise).

        Args:
            path: The graph file (``graph_<organism>_assembly<key>_min<first>_max<last>_narrow.npz``).
            search: The default search settings of the API.
            graph_settings: The caches and hyper-connectivity threshold of the graph.

        Returns:
            API: An API over the graph.

        Raises:
            ArgumentTypeError: If *path* is not a string or path-like.
            idtrack.errors.GraphFileError: If there is no such file or it cannot be read as a graph file.
            idtrack.errors.GraphMismatchError: If the file is of another format, or its organism is not registered as
                it was when the graph was built.
            idtrack.errors.UnknownOrganismError: If the graph's organism is not in the organism table.
        """
        path = _arguments.path(path, "path")
        check_graph_settings(graph_settings)
        check_search(search)
        graph = TheGraph.load(path, graph_settings=graph_settings)
        manifest = graph.manifest
        organism = organisms.check_organism(graph.graph.get("organism"))
        recorded = manifest.get("organism_table_fingerprint")
        if recorded is not None and recorded != files.table_fingerprint(organism):
            raise GraphMismatchError(
                f"The graph file `{path}` was built with another organism table for {organism} (its assemblies, "
                "their releases or their order differ from `idtrack.organisms.ORGANISM_TABLE` of this process); "
                "build the graph again with this table.",
                path,
            )
        return cls(graph, search=search)

    @staticmethod
    def graph_path(
        organism: str,
        snapshot_release: int,
        local_repository: str,
        *,
        first_release: int | None = None,
    ) -> str:
        """Return the file a graph of :py:meth:`build` is written to (and read from).

        Args:
            organism: The organism (:py:func:`idtrack.organisms.supported_organisms`).
            snapshot_release: The last release of the graph.
            local_repository: The folder of the graph files and the downloaded tables.
            first_release: The first release of the window; ``None``: the organism's first release.

        Returns:
            str: The path.

        Raises:
            idtrack.errors.InputError: For an unknown organism, a release that is not an integer, or a window that ends
                before it begins.
        """
        organism = organisms.check_organism(organism)
        first, snapshot = API._window(organism, first_release, snapshot_release)
        local_repository = _arguments.path(local_repository, "local_repository")
        # The primary assembly of a graph is the organism's preferred assembly that has the snapshot.
        assembly = organisms.primary_assembly(organism, snapshot)
        return files.graph_file_path(local_repository, organism, assembly, first, snapshot)

    @staticmethod
    def _window(organism: str, first_release: Any, snapshot_release: Any) -> tuple[int, int]:
        """Return ``(first release, snapshot release)`` of a graph's window, checked.

        The first release defaults to the organism's first release.
        """
        snapshot = _arguments.release(snapshot_release, "snapshot_release")
        if first_release is None:
            first = organisms.first_release(organism)
        else:
            first = _arguments.release(first_release, "first_release")
        if first > snapshot:
            raise InvalidArgumentError(
                f"`first_release`={first} is after `snapshot_release`={snapshot}.",
                argument="first_release",
                value=first,
            )
        return first, snapshot

    @staticmethod
    def database_manager(
        organism: str,
        snapshot_release: int,
        local_repository: str,
        *,
        assembly: int | None = None,
        first_release: int | None = None,
    ) -> DatabaseManager:
        """Return the manager of an organism's Ensembl tables (needed to build a graph and to run its self-tests).

        Args:
            organism: The organism.
            snapshot_release: The last release to include; later releases are ignored.
            local_repository: The folder of the downloaded tables (it must exist).
            assembly: The assembly whose tables the manager reads; ``None``: the preferred assembly that has
                *snapshot_release*.
            first_release: The first release of the window; ``None``: the organism's first release.

        Returns:
            idtrack.build.DatabaseManager: The manager.
        """
        from idtrack.build.database_manager import DatabaseManager

        organism = organisms.check_organism(organism)
        # Checked here; the default of the first release stays the manager's (one default for both).
        API._window(organism, first_release, snapshot_release)
        return DatabaseManager(
            organism=organism,
            release=None,
            first_release=first_release,
            snapshot_release=snapshot_release,
            form=ids.BACKBONE_FORM,
            local_repository=local_repository,
            assembly=assembly,
        )

    @classmethod
    def build(
        cls,
        organism: str,
        snapshot_release: int,
        local_repository: str,
        *,
        first_release: int | None = None,
        rebuild: bool = False,
        search: SearchSettings | None = None,
        graph_settings: GraphSettings | None = None,
    ) -> API:
        """Build the graph of an organism and snapshot release from Ensembl's tables, and load it.

        The graph holds every release from *first_release* to *snapshot_release*, on the organism's assemblies in
        their order of the organism table; its primary assembly is the preferred one that has *snapshot_release*.
        Tables already in *local_repository* are not downloaded again. If the repository holds the file of this graph
        (:py:meth:`graph_path`), and it is the graph asked for (the same selection of external databases and organism
        table), it is loaded instead of built, unless *rebuild*; a file of another graph under the same name is an
        error (:py:class:`idtrack.errors.GraphMismatchError`), never silently replaced.

        Args:
            organism: The organism.
            snapshot_release: The last release of the graph; later releases are ignored.
            local_repository: An existing folder for the tables and the graph file.
            first_release: The first release of the window; ``None``: the organism's first release.
            rebuild: Build even if the file exists, and replace it.
            search: The default search settings of the API.
            graph_settings: The caches and hyper-connectivity threshold of the graph.

        Returns:
            API: An API over the graph.

        Raises:
            idtrack.errors.InputError: For an unknown organism, release or assembly, or a folder that does not exist.
            idtrack.errors.GraphMismatchError: If the file holds another graph than the one asked for.
            idtrack.errors.BuildError: If Ensembl cannot be reached or its tables cannot be built into a graph.
        """
        import os

        from idtrack.build.builder import GraphMaker

        organism = organisms.check_organism(organism)
        rebuild = _arguments.boolean(rebuild, "rebuild")
        check_search(search)
        check_graph_settings(graph_settings)
        local_repository = _arguments.path(local_repository, "local_repository")
        if not os.path.isdir(local_repository):
            raise InvalidArgumentError(
                f"`local_repository` must be an existing folder, got {local_repository!r}.",
                argument="local_repository",
                value=local_repository,
            )
        path = cls.graph_path(organism, snapshot_release, local_repository, first_release=first_release)
        if os.path.isfile(path) and not rebuild:
            # The file of this graph: loaded and checked against what is asked for, with no manager and no network.
            graph = GraphMaker.load_graph(
                local_repository,
                organism,
                int(snapshot_release),
                first_release=first_release,
                graph_settings=graph_settings,
            )
        else:
            manager = cls.database_manager(
                organism,
                snapshot_release,
                local_repository,
                first_release=first_release,
            )
            maker = GraphMaker(manager)
            graph = maker.get_graph(create_even_if_exist=True)
            if graph_settings is not None:
                graph = TheGraph.load(maker.create_file_name(), graph_settings=graph_settings)
        return cls(graph, search=search)

    # ------------------------------------------------------------------------------------------------------------
    # Process-wide helpers

    @staticmethod
    def configure_logger(level: int | str | None = None) -> None:
        """Show the messages of idtrack on the console, at *level* and above.

        Attaches one console handler (time, level, logger name, message) to the package logger ``"idtrack"`` and
        sets that logger's level. Only idtrack's loggers are affected. Calling it again changes the level and never
        adds a second handler. Applications that configure logging themselves can instead set
        ``logging.getLogger("idtrack").setLevel(...)`` and leave this method alone.

        Args:
            level: Logging level (e.g. ``logging.INFO``, ``"DEBUG"``). ``None`` means :py:data:`logging.INFO`.
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

    @staticmethod
    def ensembl_organism(name: str | int) -> tuple[str, int]:
        """Resolve an organism's name and fetch Ensembl's latest release of it (Ensembl REST: needs the network).

        A name of idtrack's own table (``homo_sapiens``, ``human``) resolves without the network
        (:py:func:`idtrack.organisms.resolve_organism`); for the latest release, and for other names (common names,
        taxon ids, assembly names), Ensembl's species list is read (kept for the process for a day).

        Args:
            name: Organism descriptor in any supported form (e.g. ``"human"``, ``"hsapiens"``, ``9606``,
                ``"Macaca mulatta"``, ``"Mmul_10"`` or ``"homo_sapiens"``). Matching is case-insensitive. A common name
                that several Ensembl species share raises an error that lists the candidates.

        Returns:
            tuple[str, int]: ``(formal_name, latest_release)``.

        Raises:
            idtrack.errors.ArgumentTypeError: If *name* is neither a string nor an integer.
            idtrack.errors.MissingValueError: If *name* is an empty string.
            idtrack.errors.UnknownOrganismError: If no species of Ensembl has that name.
            idtrack.errors.InvalidArgumentError: If several species have it.
            idtrack.errors.EnsemblRestError: If Ensembl's species list cannot be read.
        """
        from idtrack.build.species import VerifyOrganism

        found = VerifyOrganism(name)
        return found.get_formal_name(), int(found.get_latest_release())

    # ------------------------------------------------------------------------------------------------------------
    # What the graph holds

    def releases(self) -> list[int]:
        """Return the Ensembl releases the graph covers, ascending (the valid ``from_release`` and ``to_release``).

        Returns:
            list[int]: The releases.
        """
        return [int(r) for r in self.graph.releases]

    def external_databases(self) -> list[str]:
        """Return the external databases of the graph, sorted (the valid ``from_database``).

        Returns:
            list[str]: The databases.
        """
        return sorted(self.graph.available_external_databases)

    def final_databases(self, to_release: int | None = None) -> list[str]:
        """Return the values accepted as ``final_database`` for a target release, sorted.

        These are the external databases of the genome assembly that serves *to_release* (a release of an earlier
        assembly has that assembly's databases) and the two Ensembl gene namespaces (``ensembl_gene`` with version
        and ``base_ensembl_gene`` without).

        Args:
            to_release: The target release; ``None`` means the graph's snapshot release.

        Returns:
            list[str]: The databases.
        """
        return self.engine.final_databases(to_release)

    def databases_by_assembly(self) -> dict[str, list[str]]:
        """Return ``{assembly (as a string, a JSON key): sorted external databases}`` of the graph.

        Returns:
            dict[str, list[str]]: The databases of each assembly.
        """
        by_assembly = self.graph.available_external_databases_assembly
        return {str(int(a)): sorted(by_assembly[a]) for a in sorted(by_assembly)}

    def database_forms(self) -> dict[str, str]:
        """Return ``{external database: Ensembl form it connects through}`` (``"gene"``, ``"transcript"``, ...).

        Returns:
            dict[str, str]: The form of each database.
        """
        forms = self.graph.external_database_connection_form
        return {str(k): str(forms[k]) for k in sorted(forms)}

    def describe(self) -> GraphDescription:
        """Describe the loaded graph: organism, releases, assemblies, databases and what built it. No network.

        Returns:
            GraphDescription: The description.
        """
        graph = self.graph
        manifest = graph.manifest
        organism = str(graph.graph["organism"])
        serving = {int(r): int(a) for r, a in sorted(graph.serving_assembly.items())}
        serves: dict[int, list[int]] = {}
        for release, assembly in serving.items():
            serves.setdefault(assembly, []).append(release)
        by_assembly = graph.available_external_databases_assembly
        table = organisms.ORGANISM_TABLE[organism]
        order = {a: i for i, a in enumerate(organisms.assemblies_by_priority(organism))}

        def every(databases) -> tuple[str, ...]:
            return tuple(sorted(databases))

        def synonyms(databases) -> tuple[str, ...]:
            return tuple(sorted(d for d in databases if d.startswith(ids.SYNONYM_PREFIX)))

        assemblies = tuple(
            AssemblyDescription(
                assembly=int(a),
                name=organisms.assembly_name(organism, a) if int(a) in table else None,
                releases=tuple(int(r) for r in releases),
                serves=tuple(serves.get(int(a), ())),
                external_databases=every(by_assembly.get(a, ())),
                synonym_databases=synonyms(by_assembly.get(a, ())),
            )
            for a, releases in sorted(graph.assembly_releases.items(), key=lambda kv: order.get(int(kv[0]), len(order)))
        )
        counts = {str(k): int(v) for k, v in (manifest.get("counts") or {}).items()}
        return GraphDescription(
            organism=organism,
            primary_assembly=int(graph.graph["genome_assembly"]),
            snapshot_release=int(graph.graph["ensembl_release"]),
            releases=tuple(self.releases()),
            assemblies=assemblies,
            serving_assembly={str(r): a for r, a in serving.items()},
            external_databases=every(self.external_databases()),
            synonym_databases=synonyms(self.external_databases()),
            final_databases=tuple(self.final_databases()),
            forms=tuple(graph.available_forms or manifest.get("forms") or ()),
            format_version=manifest.get("format_version"),
            package_version=manifest.get("package_version"),
            package_commit=manifest.get("package_commit"),
            content_digest=manifest.get("content_digest"),
            build_date=manifest.get("build_date"),
            counts=counts,
        )

    # ------------------------------------------------------------------------------------------------------------
    # Names

    def genes_named(self, name: str, fold_case: bool = False) -> list[NameMatch]:
        """Return every gene a name of an external database names, per database and release.

        See :py:meth:`idtrack.graph.TheGraph.genes_named`.

        Args:
            name: The name as written.
            fold_case: Match without letter case.

        Returns:
            list[NameMatch]: The genes, databases and releases.
        """
        return self.graph.genes_named(name, fold_case=_arguments.boolean(fold_case, "fold_case"))

    def names_of(self, gene: str, release: int | None = None) -> list[NameMatch]:
        """Return the names external databases attach to a gene, per database and release.

        See :py:meth:`idtrack.graph.TheGraph.names_of`.

        Args:
            gene: An Ensembl gene id with version.
            release: Only names that hold at this release; ``None`` for all.

        Returns:
            list[NameMatch]: The names.
        """
        gene = _arguments.string(gene, "gene")
        return self.graph.names_of(gene, None if release is None else self.engine._check_release(release, "release"))

    def gene_to_database(self, gene: str, final_database: str, release: int | None = None) -> tuple[list[str], int]:
        """Return the ids of a database that a gene carries at a release (the final hop of a conversion alone).

        Args:
            gene: An Ensembl gene id with version that exists at the release.
            final_database: A database of :py:meth:`final_databases` of the release.
            release: The release; ``None``: the graph's snapshot release.

        Returns:
            tuple[list[str], int]: The ids, sorted, and the confidence (0: attached at the release; 1: attached to a
            transcript or translation the gene had at another release).
        """
        return self.engine.gene_to_database(gene, final_database, release)

    # ------------------------------------------------------------------------------------------------------------
    # Arguments: the engine's checks (one rule and one message each), and the reading of a user's identifier

    @staticmethod
    def _identifier(identifier: Any) -> str:
        """Return *identifier* without surrounding white space, or raise for a non-string or an empty string."""
        stripped = _arguments.string(identifier, "identifier").strip()
        if not stripped:
            raise MissingValueError(f"`identifier` is empty: {identifier!r}.", argument="identifier", value=identifier)
        return stripped

    def _search(self, budget_states: int | None, budget_seconds: float | None) -> SearchSettings:
        """Return the search settings of one call: :py:attr:`search` with the call's budget (checked by the settings)."""
        if budget_states is None and budget_seconds is None:
            return self.search
        changes: dict[str, Any] = {}
        if budget_states is not None:
            changes["budget_states"] = budget_states
        if budget_seconds is not None:
            changes["budget_seconds"] = budget_seconds
        return self.search.replace(**changes)

    def _conversion_arguments(
        self, from_release: Any, to_release: Any, final_database: Any, from_database: Any
    ) -> tuple[int | None, int, str, str | None]:
        """Check the arguments shared by single and batch conversions with the engine's checks.

        Returns:
            tuple: ``(from_release, to_release, final_database, from_database)``.
        """
        engine = self.engine
        to_rel = engine._check_target_release(to_release)
        from_rel = None if from_release is None else engine._check_release(from_release, "from_release")
        return (
            from_rel,
            to_rel,
            engine._check_final_database(final_database, to_rel),
            engine._check_from_database(from_database),
        )

    # ------------------------------------------------------------------------------------------------------------
    # Conversion

    def convert(
        self,
        identifier: str,
        from_release: int | None = None,
        to_release: int | None = None,
        final_database: str | None = None,
        *,
        from_database: str | None = None,
        explain: bool = False,
        allow_bridges: bool = True,
        budget_states: int | None = None,
        budget_seconds: float | None = None,
    ) -> ConversionResult:
        """Convert one identifier to a target release and database: every candidate, ranked, the best marked.

        The identifier is resolved to a node of the graph (:py:meth:`idtrack.graph.TheGraph.resolve_name`): between
        the nodes the string can mean, one that exists at *from_release* (without one: at the last release of the
        graph) is taken before one that does not (with *from_database*: as an id of that database), and an Ensembl id
        is never passed over. Among those it is matched exactly (as a database wrote the identifier at the release
        it is read at), then ignoring letter case, then taking ``-``, ``_`` and ``.`` as one character, then without
        the version of a sequence accession, each step on names and then on synonyms before the next begins. How it
        matched is reported (:py:attr:`idtrack.results.ConversionResult.match`). The engine
        (:py:meth:`idtrack.engine.Track.convert`) then finds its Ensembl gene(s) at *to_release* and their ids in
        *final_database*, ranks them and marks the best answer: ``result.view("best")`` and ``result.view("all")`` are
        the two views of the one search.

        Args:
            identifier: Ensembl id (with or without version), symbol or other external id. Surrounding white space is
                ignored.
            from_release: Ensembl release at which *identifier* is read. If ``None``, it is read at *to_release* when
                it is active there, otherwise at the latest release in which it is active. If a release is given at
                which the graph does not hold the identifier, the identifier is read by the same rule as without
                *from_release*, and the result says so (``from_release_fallback``).
            to_release: Target Ensembl release; ``None`` means the graph's snapshot release.
            final_database: Database of the targets (:py:meth:`final_databases` of *to_release*, matched case-insensitively);
                ``None`` means ``"ensembl_gene"``.
            from_database: The external database *identifier* is meant as (one of :py:meth:`external_databases`);
                ``None`` reads it in any database.
            explain: Add :py:attr:`idtrack.results.ConversionResult.path`, the paths of the candidates as data (empty
                when there is no candidate).
            allow_bridges: Allow bridges through ids two genes share when the history leads to no gene.
            budget_states: The search budget of this call in states (default: :py:attr:`search`).
            budget_seconds: The search budget of this call in seconds (default: :py:attr:`search`).

        Returns:
            ConversionResult: The result.

        Raises:
            idtrack.errors.InputError: For an argument that is not valid (the subclass names which).
            idtrack.errors.SearchBudgetExceededError: If the search needs more than its budget.
        """
        query = self._identifier(identifier)
        from_rel, to_rel, database, source = self._conversion_arguments(
            from_release, to_release, final_database, from_database
        )
        search = self._search(budget_states, budget_seconds)
        return self._convert(
            identifier, query, from_rel, to_rel, database, source, _arguments.boolean(explain, "explain"),
            _arguments.boolean(allow_bridges, "allow_bridges"), search,
        )  # fmt: skip

    def _convert(
        self,
        identifier: str,
        query: str,
        from_release: int | None,
        to_release: int,
        final_database: str,
        from_database: str | None,
        explain: bool,
        allow_bridges: bool,
        search: SearchSettings,
    ) -> ConversionResult:
        """Convert a checked query (see :py:meth:`convert`)."""
        graph = self.graph

        def read(release: int | None) -> tuple[str | None, str | None, tuple[str, ...]]:
            """Say which node the query means at a release, as the engine will read it (for the report)."""
            node, how = graph.resolve_name(query, release, from_database, to_release)
            versions: tuple[str, ...] = ()
            if node is None:
                versions = tuple(graph.known_versions(query))
                if not versions and graph.versions_of(query):
                    # A transcript or translation id without version: the engine reads it as its version of the
                    # release.
                    node, how = query, "exact"
            return node, how, versions

        graph_id, match, known_versions = read(from_release)
        conversion: Conversion | None = None
        fallback = False
        if graph_id is not None:
            # The engine is given the string (its arguments checked by the engine's checks) and reads it by the same
            # rule for the same release.
            arguments: dict[str, Any] = {
                "to_release": to_release,
                "final_database": final_database,
                "from_database": from_database,
                "allow_bridges": allow_bridges,
                "return_ensembl_alternative": True,
                "explain": explain,
                "search": search,
            }
            try:
                conversion = self.engine._convert(query, from_release=from_release, **arguments)
            except InactiveAtReleaseError:
                # The graph does not hold the identifier at the release given: read it where the graph holds it.
                conversion = self.engine._convert(query, from_release=None, **arguments)
                fallback = True
                graph_id, match, known_versions = read(None)
        # The conversion when it found candidates, else None.
        hit = conversion if conversion is not None and conversion.candidates else None
        reason: str | None
        if graph_id is None:
            status, reason = ("unknown_version", "unknown_version") if known_versions else ("not_found", "not_in_graph")
        elif hit is None:
            status, reason = "no_conversion", (conversion.reason if conversion is not None else "no_path_to_target")
        elif hit.no_target:
            # The candidates are genes standing in for the missing id of the requested database.
            status, reason = "no_target", "no_id_in_final_database"
        elif hit.tie:
            status, reason = "tie", None
        else:
            status, reason = "converted", None
        return ConversionResult(
            query_id=identifier,
            graph_id=hit.graph_id if hit is not None else graph_id,
            match=match,
            known_versions=known_versions,
            from_release=hit.from_release if hit is not None else (None if fallback else from_release),
            from_release_inferred=hit.from_release_inferred if hit is not None else (fallback or from_release is None),
            from_release_given=from_release,
            from_release_fallback=fallback,
            from_databases=hit.from_databases if hit is not None else (),
            to_release=to_release,
            final_database=hit.final_database if hit is not None else None,
            candidates=hit.candidates if hit is not None else (),
            tie=hit.tie if hit is not None else False,
            decided_by=hit.decided_by if hit is not None else None,
            alternative_locus_genes=hit.alternative_locus_genes if hit is not None else (),
            no_target=hit.no_target if hit is not None else False,
            no_target_genes=hit.no_target_genes if hit is not None else (),
            status=status,
            reason=reason,
            error=None,
            path=(conversion.path if conversion is not None else _NO_PATH) if explain else None,
        )

    def convert_many(
        self,
        identifiers: Iterable[str],
        from_release: int | None = None,
        to_release: int | None = None,
        final_database: str | None = None,
        *,
        from_database: str | None = None,
        explain: bool = False,
        allow_bridges: bool = True,
        budget_states: int | None = None,
        budget_seconds: float | None = None,
        progress: bool = False,
    ) -> list[ConversionResult]:
        """Convert a batch of identifiers with the same arguments; one identifier's error never stops the batch.

        The arguments are checked once, before any conversion: a batch that is not a sequence of strings, or holds
        an empty string, raises with the positions of the offending items, as do invalid releases or databases.
        Any error of one identifier (an exceeded budget, an internal error) is that item's
        :py:attr:`idtrack.results.ConversionResult.error`, with status ``error``, and the batch goes on. Each
        distinct identifier is converted once; repeated identifiers get the same (immutable) result.

        Args:
            identifiers: The identifiers (a list, tuple, NumPy array, pandas Index or Series of strings; not a str).
            from_release: See :py:meth:`convert`.
            to_release: See :py:meth:`convert`.
            final_database: See :py:meth:`convert`.
            from_database: See :py:meth:`convert`.
            explain: See :py:meth:`convert`.
            allow_bridges: See :py:meth:`convert`.
            budget_states: See :py:meth:`convert` (per identifier).
            budget_seconds: See :py:meth:`convert` (per identifier).
            progress: Show a progress bar on the console.

        Returns:
            list[ConversionResult]: One result per input identifier, in input order.

        Raises:
            idtrack.errors.InputError: For a batch or an argument that is not valid.
        """
        items = _arguments.strings(identifiers, "identifiers")
        empty = [i for i, x in enumerate(items) if not x.strip()]
        if empty:
            raise MissingValueError(
                f"`identifiers` holds {len(empty)} empty strings, at positions {empty[:20]}.",
                argument="identifiers",
                value=empty[:20],
            )
        from_rel, to_rel, database, source = self._conversion_arguments(
            from_release, to_release, final_database, from_database
        )
        search = self._search(budget_states, budget_seconds)
        explain, allow_bridges = _arguments.boolean(explain, "explain"), _arguments.boolean(
            allow_bridges, "allow_bridges"
        )
        distinct = list(dict.fromkeys(items))
        loop: Iterable[str] = distinct
        if _arguments.boolean(progress, "progress"):
            from tqdm import tqdm

            loop = tqdm(distinct, mininterval=0.25, ncols=100)
        by_identifier: dict[str, ConversionResult] = {}
        for identifier in loop:
            try:
                by_identifier[identifier] = self._convert(
                    identifier, identifier.strip(), from_rel, to_rel, database, source, explain, allow_bridges, search
                )
            except Exception as error:  # noqa: BLE001 - one identifier's error is that item's answer
                by_identifier[identifier] = self._error_result(identifier, from_rel, to_rel, error, explain)
        return [by_identifier[identifier] for identifier in items]

    @staticmethod
    def _error_result(
        identifier: str, from_release: int | None, to_release: int, error: BaseException, explain: bool
    ) -> ConversionResult:
        """Return the batch result of an identifier whose conversion raised (with *explain*, an empty path)."""
        return ConversionResult(
            query_id=identifier,
            graph_id=getattr(error, "identifier", None),
            match=None,
            known_versions=(),
            from_release=from_release,
            from_release_inferred=from_release is None,
            from_release_given=from_release,
            from_release_fallback=False,
            from_databases=(),
            to_release=to_release,
            final_database=None,
            candidates=(),
            tie=False,
            decided_by=None,
            alternative_locus_genes=(),
            no_target=False,
            no_target_genes=(),
            status="error",
            reason="error",
            error=ErrorInfo(type=type(error).__name__, message=str(error), input_error=isinstance(error, InputError)),
            path=_NO_PATH if explain else None,
        )

    @staticmethod
    def reason_text(result: ConversionResult) -> str | None:
        """Return the reason of a result without a target in words (:py:data:`idtrack.results.REASONS`).

        Args:
            result: A result.

        Returns:
            str | None: The text; ``None`` for a result with a target in the requested database.
        """
        return None if result.reason is None else REASONS[result.reason]

    @staticmethod
    def classify(results: list[ConversionResult], strategy: str = "best") -> dict[str, list[ConversionResult]]:
        """Group conversion results into bins for reporting, by the targets of one view (``best`` or ``all``).

        Every result is in ``"input_identifiers"`` and in exactly one of the bins ``"matching_1_to_0"`` (not found,
        no conversion, or an error), ``"matching_1_to_1"``, ``"matching_1_to_n"``, ``"alternative_target_1_to_1"``
        and ``"alternative_target_1_to_n"`` (the last two: ``no_target``, the targets are Ensembl genes standing in
        for a missing id of the requested database). A result of ``"matching_1_to_1"`` is also in
        ``"changed_only_1_to_1"`` when its target is not the identifier it resolved to, and one of
        ``"matching_1_to_n"`` is also in ``"changed_only_1_to_n"`` when none of its targets is. "The identifier it
        resolved to" is ``graph_id``, compared without the graph's synonym prefix and, for Ensembl ids, without the
        version (:py:func:`idtrack.ids.stable_id_of`), so that a query without version or in another case that maps
        to itself is not "changed".

        Args:
            results: Results of :py:meth:`convert_many` or :py:meth:`convert`.
            strategy: ``"best"`` or ``"all"``: which view's targets count.

        Returns:
            dict[str, list[ConversionResult]]: The bins (:py:data:`BINS`).

        Raises:
            idtrack.errors.InvalidArgumentError: For another strategy.
            idtrack.errors.ArgumentTypeError: If *results* is not a list of results.
            idtrack.errors.IntegrityError: If a result has no target although it is neither not found, nor without
                conversion, nor an error.
        """
        from idtrack.errors import IntegrityError

        strategy = check_strategy(strategy)
        if isinstance(results, (str, bytes, ConversionResult)) or not isinstance(results, Iterable):
            raise ArgumentTypeError(
                f"`results` must be a list of ConversionResult, got {type(results).__name__}.",
                argument="results",
                value=results,
            )
        results = list(results)
        wrong = [i for i, r in enumerate(results) if not isinstance(r, ConversionResult)]
        if wrong:
            raise ArgumentTypeError(
                f"`results` must hold ConversionResult objects only; positions {wrong[:20]} hold other values.",
                argument="results",
                value=wrong[:20],
            )
        bins: dict[str, list[ConversionResult]] = {name: [] for name in BINS}
        for result in results:
            bins["input_identifiers"].append(result)
            if result.graph_id is None or not result.candidates or result.error is not None:
                bins["matching_1_to_0"].append(result)
                continue
            targets = result.targets(strategy)
            if not targets:
                raise IntegrityError(
                    f"Unexpected conversion result: query_id={result.query_id!r} returned no target. "
                    "This indicates an internal error in the conversion logic. Please report this issue."
                )
            one = len(targets) == 1
            if result.no_target:
                bins["alternative_target_1_to_1" if one else "alternative_target_1_to_n"].append(result)
                continue
            resolved = ids.stable_id_of(result.graph_id or result.query_id)
            unchanged = {ids.stable_id_of(t) for t in targets}
            if one:
                bins["matching_1_to_1"].append(result)
                if resolved not in unchanged:
                    bins["changed_only_1_to_1"].append(result)
            else:
                bins["matching_1_to_n"].append(result)
                if resolved not in unchanged:
                    bins["changed_only_1_to_n"].append(result)
        return bins

    @staticmethod
    def classification_summary(classified: dict[str, list[ConversionResult]]) -> str:
        """Return counts and percentages per bin of :py:meth:`classify`, and the counts of each status.

        Args:
            classified: Output of :py:meth:`classify`.

        Returns:
            str: The summary.
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

        def pct(part: int, whole: int) -> float:
            return (part / whole * 100) if whole else 0

        statuses = Counter(item.status for item in items)
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
            f"  Statuses:\n" + "".join(f"    {status}: {statuses[status]}\n" for status in sorted(statuses))
        ).rstrip("\n")

    # ------------------------------------------------------------------------------------------------------------
    # Where a list comes from

    def infer_source(
        self, identifiers: Iterable[str], mode: str = "assembly_release", report_only_winner: bool = True
    ) -> list:
        """Infer the origins (database, assembly, release) that best explain a list of identifiers.

        Each identifier found in the graph counts once for every origin in which it exists
        (:py:meth:`idtrack.graph.TheGraph.trios`, read for the given identifiers only). Identifiers not in the graph
        are counted and logged, not used. A string is read as the node it means at the last release of the graph
        (:py:meth:`idtrack.graph.TheGraph.resolve_name`).

        Args:
            identifiers: Identifiers (a sequence of strings, not a str).
            mode: Granularity of an origin (:py:data:`idtrack.engine.SOURCE_MODES`): ``"complete"`` →
                ``[database, assembly, release]``, ``"release"`` → ``release``, ``"assembly"`` → ``assembly``,
                ``"assembly_release"`` (default) →
                ``[assembly, release]``.
            report_only_winner: ``True`` (default) returns every origin with the highest count, sorted; ``False``
                returns every origin with its count, by descending count, then origin.

        Returns:
            list: With *report_only_winner*, the sorted list of best origins; otherwise ``[origin, count]`` pairs.

        Raises:
            idtrack.errors.InvalidArgumentError: If *mode* is unknown.
            idtrack.errors.ArgumentTypeError: If *identifiers* is not a sequence of strings.
            idtrack.errors.UnknownIdentifierError: If none of the identifiers is in the graph.
        """
        mode = self.engine._check_mode(mode)
        items = _arguments.strings(identifiers, "identifiers")
        report_only_winner = _arguments.boolean(report_only_winner, "report_only_winner")
        found_id_list = []
        unfound = 0
        for identifier in items:
            stripped = identifier.strip()
            found_id, _ = self.graph.resolve_name(stripped, self.graph.snapshot_release) if stripped else (None, None)
            if found_id is None:
                unfound += 1
            else:
                found_id_list.append(found_id)
        if unfound:
            self.log.warning(f"{unfound} of {len(items)} identifiers are not in the graph and are not used.")
        if not found_id_list:
            raise UnknownIdentifierError(
                f"None of the {len(items)} identifiers is in the graph, so their source cannot be inferred.",
                argument="identifiers",
            )
        counts = Counter({_plain_origin(o): c for o, c in self.engine._identify_source(found_id_list, mode)})
        ranking = sorted(counts.items(), key=lambda oc: (-oc[1], _origin_key(oc[0])))
        if not report_only_winner:
            return [[_plain(origin), count] for origin, count in ranking]
        best = ranking[0][1]
        return [_plain(origin) for origin, count in ranking if count == best]


__all__ = ["API", "BINS"]
