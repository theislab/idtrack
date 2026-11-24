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

import dataclasses
import logging
import math
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from idtrack import _arguments, ids, organisms
from idtrack.engine import ranking
from idtrack.engine.track import _ORIGIN_OF, Conversion, Track
from idtrack.errors import (
    ArgumentTypeError,
    GraphMismatchError,
    InactiveAtReleaseError,
    InputError,
    InvalidArgumentError,
    MissingValueError,
    UnknownIdentifierError,
)
from idtrack.graph import GRAPH_FORMAT_VERSION, files
from idtrack.graph.graph import TheGraph
from idtrack.graph.names import Readings
from idtrack.graph.types import served_mask
from idtrack.reading import Reading, read_identifier
from idtrack.results import (
    REASONS,
    AssemblyDescription,
    BridgeCheck,
    Candidate,
    ConversionResult,
    ErrorInfo,
    ExplainedPath,
    GraphDescription,
    NameMatch,
    OtherReading,
    SourceInference,
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
    "read_at_another_release",
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


def _origin_key(origin: Any) -> tuple:
    """Total order on origins (scalars or tuples of mixed scalars)."""
    values = origin if isinstance(origin, tuple) else (origin,)
    return tuple((type(v).__name__, v) for v in values)


def _public(node: str | None) -> str | None:
    """Return a node as results name it (without the graph's synonym prefix, no part of any identifier)."""
    return None if node is None else ids.public_name(node)


def _best_genes(conversion: Conversion | None) -> tuple[str, ...]:
    """Return the genes of a conversion's best answer, ascending (empty without one)."""
    if conversion is None:
        return ()
    return tuple(sorted({candidate.gene for candidate in conversion.candidates if candidate.best}))


def _lineage(candidate: Candidate) -> tuple:
    """Return the key by which one conversion compares the lineages of its entry genes, for a candidate.

    Being the queried id's own version, then the lineage of :py:func:`idtrack.engine.ranking.make_pair` (final
    confidence, bridges, their gap, the identifier being the entry gene's display name, support of the identifier on
    the entry gene); a smaller key is better.
    """
    pair = ranking.make_pair(
        own=candidate.own_version,
        confidence=candidate.final_confidence if candidate.final_confidence is not None else 0,
        bridges=candidate.bridges,
        gap=candidate.bridge_gap,
        support=candidate.entry_support,
        unscored=candidate.unscored_steps,
        score=candidate.score,
        same_as_query=candidate.same_as_query,
        external_steps=candidate.external_steps,
        history_steps=candidate.history_steps,
        gene=candidate.gene,
        target=candidate.target,
        display_rank=candidate.display_rank,
    )
    return (pair.order[0], *pair.lineage)


def _tie_of_readings(conversions: list[Conversion]) -> Conversion:
    """Join the conversions of several readings of one identifier into one answer.

    The readings exist at one release (:py:meth:`idtrack.graph.TheGraph.readings`). Their candidates are ranked as the
    candidates of one conversion rank the genes an identifier enters (:py:mod:`idtrack.engine.ranking`): the best
    answer of every reading whose best is equal in the lineage evidence to the best of all (:py:func:`_lineage`: a
    reading whose best needs a bridge does not tie with one that needs none) belongs to the answer; a tie when that
    holds more than one pair. The first conversion's own fields stay; the candidates follow, each pair once (best where
    one reading has it best), the best first (in ascending order of gene id and target id, as one conversion gives a
    tie) and the others by their lineage, with their explained paths joined into one (nodes, steps, and the bridges
    each search considered, each pair of genes once). Only readings whose answer is of the same kind as the first's
    (targets in the final database, or genes standing in for them) join it.

    Args:
        conversions: The conversions with candidates, the reading taken first.

    Returns:
        Conversion: The joined conversion.
    """
    first = conversions[0]
    joined = [c for c in conversions if c.no_target == first.no_target and c.final_database == first.final_database]
    leading = {id(c): _lineage(next(x for x in c.candidates if x.best)) for c in joined}
    winning = min(leading.values())
    found: dict[tuple[str, str], tuple[Candidate, Conversion]] = {}
    nodes: list = []
    steps: list = []
    step_index: dict = {}
    node_ids: set[str] = set()
    bridges: dict[tuple[str, str], BridgeCheck] = {}
    for conversion in joined:
        remap: list[int] = []
        if conversion.path is not None:
            for node in conversion.path.nodes:
                if node.id not in node_ids:
                    node_ids.add(node.id)
                    nodes.append(node)
            for check in conversion.path.bridges:
                have = bridges.get((check.source, check.target))
                if have is not None and have != check:
                    check = dataclasses.replace(
                        have,
                        taken=have.taken or check.taken,
                        through=tuple(sorted(set(have.through) | set(check.through))),
                    )
                bridges[(check.source, check.target)] = check
            for step in conversion.path.steps:
                if step not in step_index:
                    step_index[step] = len(steps)
                    steps.append(step)
                remap.append(step_index[step])
        for candidate in conversion.candidates:
            key = (candidate.gene, candidate.target)
            best = candidate.best and leading[id(conversion)] == winning
            if key in found and (found[key][0].best or not best):
                continue  # each pair once: where a reading has it best, as that reading has it
            found[key] = (
                (
                    dataclasses.replace(
                        candidate,
                        best=best,
                        path=tuple(remap[i] for i in candidate.path),
                        final_path=tuple(remap[i] for i in candidate.final_path),
                    )
                    if conversion.path is not None
                    else dataclasses.replace(candidate, best=best)
                ),
                conversion,
            )
    order = list(found.values())
    ordered = sorted(
        (pair for pair in order if pair[0].best), key=lambda pair: ranking.best_order(pair[0].gene, pair[0].target)
    ) + sorted((pair for pair in order if not pair[0].best), key=lambda pair: _lineage(pair[0]))
    candidates = tuple(candidate for candidate, _ in ordered)
    tie = len({(c.gene, c.target) for c in candidates if c.best}) > 1
    decided_by = None
    runner_up = next((pair for pair in ordered if not pair[0].best), None)
    if not tie and runner_up is not None:
        names = ("own_version", *ranking.LINEAGE_CRITERIA)
        differs = [name for name, a, b in zip(names, _lineage(candidates[0]), _lineage(runner_up[0])) if a != b]
        # equal in lineage: the runner-up lost within its own reading, by that reading's criterion
        decided_by = differs[0] if differs else runner_up[1].decided_by
    return dataclasses.replace(
        first,
        from_databases=tuple(sorted(set().union(*(c.from_databases for c in joined)))),
        candidates=candidates,
        tie=tie,
        decided_by=decided_by,
        alternative_locus_genes=tuple(sorted(set().union(*(c.alternative_locus_genes for c in joined)))),
        no_target_genes=tuple(sorted(set().union(*(c.no_target_genes for c in joined)))),
        path=(
            None
            if first.path is None
            else ExplainedPath(
                nodes=tuple(nodes), steps=tuple(steps), bridges=tuple(bridges[pair] for pair in sorted(bridges))
            )
        ),
    )


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
            path: The graph file (``graph_<organism>_assembly<key>_min<first>_max<last>.npz``).
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
        record = graph.record
        organism = organisms.check_organism(graph.organism)
        if record is not None and record.organism_table_fingerprint != files.table_fingerprint(organism):
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
        return list(self.engine._reader.external_databases)

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
        return {str(int(a)): sorted({ids.public_name(d) for d in by_assembly[a]}) for a in sorted(by_assembly)}

    def database_forms(self) -> dict[str, str]:
        """Return ``{external database: Ensembl form it connects through}`` (``"gene"``, ``"transcript"``, ...).

        Returns:
            dict[str, str]: The form of each database.
        """
        forms = self.graph.external_database_connection_form
        found: dict[str, str] = {}
        for database in sorted(forms, key=lambda d: (ids.is_synonym_database(str(d)), str(d))):
            found.setdefault(ids.public_name(str(database)), str(forms[database]))  # a database's own form first
        return dict(sorted(found.items()))

    def describe(self) -> GraphDescription:
        """Describe the loaded graph: organism, releases, assemblies, databases and what built it. No network.

        Returns:
            GraphDescription: The description.
        """
        graph = self.graph
        record = graph.record
        organism = str(graph.organism)
        serving = {int(r): int(a) for r, a in sorted(graph.serving_assembly.items())}
        serves: dict[int, list[int]] = {}
        for release, assembly in serving.items():
            serves.setdefault(assembly, []).append(release)
        by_assembly = graph.available_external_databases_assembly
        table = organisms.ORGANISM_TABLE[organism]
        order = {a: i for i, a in enumerate(organisms.assemblies_by_priority(organism))}

        def every(databases) -> tuple[str, ...]:
            return tuple(sorted({ids.public_name(d) for d in databases}))

        def synonyms(databases) -> tuple[str, ...]:
            # the databases that record synonyms, by their own names (decision 56)
            return tuple(sorted({ids.public_name(d) for d in databases if ids.is_synonym_database(d)}))

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
        counts = {str(k): int(v) for k, v in (record.counts if record is not None else {}).items()}
        return GraphDescription(
            organism=organism,
            primary_assembly=graph._primary(),
            snapshot_release=int(graph.snapshot_release),
            releases=tuple(self.releases()),
            assemblies=assemblies,
            serving_assembly={str(r): a for r, a in serving.items()},
            external_databases=every(graph.available_external_databases),
            synonym_databases=synonyms(graph.available_external_databases),
            final_databases=tuple(self.final_databases()),
            forms=tuple(graph.available_forms or ()),
            format_version=None if record is None else GRAPH_FORMAT_VERSION,
            package_version=None if record is None else record.package_version,
            package_commit=None if record is None else record.package_commit,
            content_digest=None if record is None else record.content_digest,
            build_date=None if record is None else record.build_date,
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
    def _identifier(identifier: Any, from_database: str | None) -> Reading:
        """Read a user's identifier as it is looked up (:py:func:`idtrack.reading.read_identifier`), or refuse it."""
        return read_identifier(identifier, "identifier", from_database)

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
        the nodes the string can mean, one that exists at *from_release* (without one: at *to_release*, or, where none
        exists there, at the latest release at which one does; one release decides both the node and where it is read)
        is taken before one that does not (with *from_database*: as an id of that database), and an Ensembl id
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
                *from_release*, and the result says so (``from_release_fallback``, and ``reasons`` holds
                ``"read_at_another_release"``, the ``reason`` of a result with targets: an answer to what the identifier
                meant where it was read).
            to_release: Target Ensembl release; ``None`` means the graph's snapshot release.
            final_database: Database of the targets (:py:meth:`final_databases` of *to_release*, matched case-insensitively);
                ``None`` means ``"ensembl_gene"``.
            from_database: The external database *identifier* is meant as (one of :py:meth:`external_databases`);
                ``None`` reads it in any database.
            explain: Add :py:attr:`idtrack.results.ConversionResult.path`, the paths of the candidates as data (no
                step when there is no candidate), with every bridge the search considered and the rule that took or
                refused it.
            allow_bridges: Allow bridges through ids two genes share when the history leads to no gene (only where
                the bridge is evidence that one gene became the other: :py:meth:`idtrack.engine.track.Track.convert`).
            budget_states: The search budget of this call in states (default: :py:attr:`search`).
            budget_seconds: The search budget of this call in seconds (default: :py:attr:`search`).

        Returns:
            ConversionResult: The result.

        Raises:
            idtrack.errors.InputError: For an argument that is not valid (the subclass names which).
            idtrack.errors.SearchBudgetExceededError: If the search needs more than its budget.
        """
        from_rel, to_rel, database, source = self._conversion_arguments(
            from_release, to_release, final_database, from_database
        )
        query = self._identifier(identifier, source)
        search = self._search(budget_states, budget_seconds)
        return self._convert(
            identifier, query, from_rel, to_rel, database, source, _arguments.boolean(explain, "explain"),
            _arguments.boolean(allow_bridges, "allow_bridges"), search,
        )  # fmt: skip

    def _convert(
        self,
        identifier: str,
        reading: Reading,
        from_release: int | None,
        to_release: int,
        final_database: str,
        from_database: str | None,
        explain: bool,
        allow_bridges: bool,
        search: SearchSettings,
    ) -> ConversionResult:
        """Convert a checked query (see :py:meth:`convert`), read as :py:func:`idtrack.reading.read_identifier` reads it."""
        graph = self.graph
        query = reading.text
        arguments: dict[str, Any] = {
            "to_release": to_release,
            "final_database": final_database,
            "from_database": from_database,
            "allow_bridges": allow_bridges,
            "return_ensembl_alternative": True,
            "explain": explain,
            "search": search,
        }

        found, fallback = self._read_where(query, from_release, from_database, to_release)
        read_at = None if fallback else from_release
        graph_id, match = found.taken[0] if found.taken else (None, None)
        known_versions: tuple[str, ...] = ()
        if graph_id is None:
            known_versions = tuple(graph.known_versions(query))
            held = () if known_versions else graph.versions_of(query)
            if held:
                # A transcript or translation id without version: the engine reads it as its version of the
                # release (an id typed in another letter case than Ensembl's capitals matches by case).
                graph_id, match = query, "exact" if held[0].startswith(ids.versions_prefix(query)) else "case"
        conversion: Conversion | None = None
        if graph_id is not None:
            # The engine is given the string (its arguments checked by the engine's checks) and reads it by the same
            # rule at the release decided above.
            conversion = self.engine._convert(query, from_release=read_at, **arguments)
        # The other readings: those of the same step (a tie, decision 87) and those a rule set aside (55, 80, 82).
        others: list[OtherReading] = []
        partners: list[Conversion] = []
        for node, how in found.taken[1:]:
            partner = self._reading(node, read_at, arguments)
            if partner is not None and partner.candidates:
                partners.append(partner)
            others.append(OtherReading(ids.public_name(node), reading.prefix + how, "same_step", _best_genes(partner)))
        for node, how, why in found.set_aside:
            other = self._reading(node, read_at, arguments)
            others.append(OtherReading(ids.public_name(node), reading.prefix + how, why, _best_genes(other)))
        for node, how, releases in found.other_releases:
            # what the string meant at other releases only: read where it had that meaning (decision 99)
            other = self._reading(node, releases[-1], arguments)
            others.append(
                OtherReading(
                    ids.public_name(node), reading.prefix + how, "other_release", _best_genes(other), tuple(releases)
                )
            )
        if partners:
            conversion = _tie_of_readings([c for c in (conversion, *partners) if c is not None and c.candidates])
        lrg = False
        record = self._lrg_record(conversion)
        if record is not None and conversion is not None:
            # Decision 96: every id of an LRG record (the record's, its transcripts' and translations') answers the gene
            # the record describes, the record itself listed apart.
            described = self._lrg_conversion(conversion, arguments, record)
            if described is not None:
                conversion, lrg = described, True
        if "par_y" in reading.kinds and conversion is not None and conversion.candidates:
            # Decision 93: GENCODE's Y copy of a pseudoautosomal gene answers the gene; its Y copy is listed apart.
            twins = self._par_y_twins(conversion, to_release)
            if twins:
                conversion = dataclasses.replace(
                    conversion, alternative_locus_genes=tuple(sorted(set(conversion.alternative_locus_genes) | twins))
                )
        # The conversion when it found candidates, else None.
        hit = conversion if conversion is not None and conversion.candidates else None
        # Every reason that applies, the one of the status first: why there is no target, then that the identifier was
        # read at another release than the one given, then that a version written was not judged (decision 101).
        reasons: list[str] = []
        if graph_id is None:
            status = "unknown_version" if known_versions else "not_found"
            reasons.append("unknown_version" if known_versions else "not_in_graph")
        elif hit is None:
            status = "no_conversion"
            reasons.append(conversion.reason if conversion is not None and conversion.reason else "no_path_to_target")
        elif hit.no_target:
            # The candidates are genes standing in for the missing id of the requested database.
            status = "no_target"
            reasons.append("no_id_in_final_database")
        else:
            status = "tie" if hit.tie else "converted"
        if graph_id is not None and fallback:
            reasons.append("read_at_another_release")
        if graph_id is not None and match is not None and match.endswith("version_not_judged"):
            reasons.append("version_not_judged")  # said in the result, never silently
        return ConversionResult(
            query_id=identifier,
            graph_id=_public(hit.graph_id if hit is not None else graph_id),
            match=None if match is None else reading.prefix + ("lrg:" if lrg else "") + match,
            known_versions=known_versions,
            # the release the identifier was read at, also where it leads to no gene (the engine's reading)
            from_release=(conversion.from_release if conversion is not None else (None if fallback else from_release)),
            from_release_inferred=(
                conversion.from_release_inferred if conversion is not None else (fallback or from_release is None)
            ),
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
            reason=reasons[0] if reasons else None,
            reasons=tuple(reasons),
            other_readings=tuple(others),
            error=None,
            path=(conversion.path if conversion is not None else _NO_PATH) if explain else None,
        )

    def _lrg_record(self, conversion: Conversion | None) -> str | None:
        """Return the LRG record an Ensembl id of a conversion is an id of, at the release it was read at.

        An LRG record (a fixed reference sequence of one gene) is annotated by Ensembl as a gene of its own on a sequence
        region of its own, both named by the record's id (``LRG_321``), with its transcripts and translations; in the
        GRCh37 releases the gene had an id of its own (``LRG_7_g1`` on the region ``LRG_7``). An Ensembl id (of the
        record, its gene, a transcript or a translation) whose one entry gene is such a record at the starting release
        (named by a record's id, or lying on a region so named) is an id of the record: whatever its form, it answers as
        the record does.

        Args:
            conversion: The engine's conversion of the identifier.

        Returns:
            str | None: The record's gene (with version) at the starting release; ``None`` for any other conversion.
        """
        graph = self.graph
        if conversion is None or not conversion.candidates or conversion.from_release is None:
            return None
        if graph.node_type(conversion.graph_id) == ids.EXTERNAL:
            return None  # a name or an accession is no id of a record
        entries = {candidate.entry_gene for candidate in conversion.candidates}
        if len(entries) != 1:
            return None
        (entry,) = entries
        start = conversion.from_release
        on_record = bool(ids.LRG_ID.match(entry)) or any(
            ids.LRG_ID.match(place.region) and place.releases.mask >> start & 1 for place in graph.places(entry)
        )
        return entry if on_record else None

    def _lrg_conversion(self, conversion: Conversion, arguments: dict[str, Any], record: str) -> Conversion | None:
        """Convert the gene an LRG record describes in place of the record.

        The record's gene at the release it is read at is the gene of the reference sequence to which a symbol database
        of the organism gives the record's symbol there (Ensembl gives an LRG record the HGNC symbol of its gene).

        Args:
            conversion: The conversion of an id of the LRG record (the record's own gene its answer).
            arguments: The engine's arguments of the conversion.
            record: The record's gene at the starting release (:py:meth:`_lrg_record`).

        Returns:
            Conversion | None: The conversion of the described gene from the same release, with the LRG id as its
            ``graph_id`` and the record's gene at the target release listed apart; ``None`` when no such gene exists.
        """
        graph = self.graph
        start = conversion.from_release
        if start is None:
            return None
        symbols = set(organisms.SYMBOL_DATABASES.get(graph.organism or "", ()))
        line = graph.serving_line

        def symbol_at(connection: Any) -> bool:
            return any(served_mask(connection.items, line, symbol) >> start & 1 for symbol in symbols)

        genes: set[str] = set()
        # the record's own links (an LRG record lies outside the reference sequence, so `names_of` leaves it out)
        for name, connection in graph.xref_edges(record, reverse=True):
            if graph.node_type(name) != ids.EXTERNAL or ids.is_synonym_database(name) or not symbol_at(connection):
                continue
            for gene, link in graph.xref_edges(name):
                if gene != record and graph.node_type(gene) == ids.GENE and symbol_at(link):
                    if graph.is_active(gene, start) and not graph.is_non_reference(gene, start):
                        genes.add(gene)
        # decision 95: the Y copy of a pseudoautosomal gene is not the gene the record describes, beside its X gene
        y_copies = graph.y_copies(genes, start)
        genes -= y_copies
        found = [
            c for c in (self._reading(g, start, arguments) for g in sorted(genes)) if c is not None and c.candidates
        ]
        if not found:
            return None
        described = _tie_of_readings(found) if len(found) > 1 else found[0]
        own = {candidate.gene for candidate in conversion.candidates}
        own |= {gene for gene in y_copies - genes if graph.is_active(gene, arguments["to_release"])}
        return dataclasses.replace(
            described,
            graph_id=conversion.graph_id,
            from_release_inferred=conversion.from_release_inferred,  # the LRG id's own starting release
            alternative_locus_genes=tuple(sorted(set(described.alternative_locus_genes) | own)),
        )

    def _par_y_twins(self, conversion: Conversion, to_release: int) -> set[str]:
        """Return the Y copies of the genes of a conversion's best answer at the target release.

        A Y copy is a gene in a pseudoautosomal region of Y (:py:meth:`idtrack.graph.TheGraph.in_pseudoautosomal_y`)
        to which a symbol database gives a symbol of the gene on X at that release.

        Args:
            conversion: The conversion of the id read from ``<id>_PAR_Y``.
            to_release: The target release.

        Returns:
            set[str]: The Y copies (empty before Ensembl gave them ids of their own, release 110).
        """
        graph = self.graph
        symbols = set(organisms.SYMBOL_DATABASES.get(graph.organism or "", ()))
        twins: set[str] = set()
        best = {candidate.gene for candidate in conversion.candidates if candidate.best}
        for gene in best:
            for name in graph.names_of(gene, to_release):
                if name.synonym or name.database not in symbols:
                    continue
                for match in graph.genes_named(name.name):
                    if to_release in match.releases and graph.in_pseudoautosomal_y(match.gene, to_release):
                        twins.add(match.gene)
        return twins - best

    def _reading(self, node: str, from_release: int | None, arguments: dict[str, Any]) -> Conversion | None:
        """Convert one other reading of an identifier (a node of the graph) with the arguments of the conversion.

        Args:
            node: The node.
            from_release: The release the identifier was read at (``None``: as the engine chooses).
            arguments: The engine's arguments of the conversion.

        Returns:
            Conversion | None: The engine's conversion; ``None`` when the node never is active.
        """
        try:
            return self.engine._convert(node, from_release=from_release, **arguments)
        except InactiveAtReleaseError:
            return self.engine._convert(node, from_release=None, **arguments)

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

        The arguments are checked once, before any conversion: a batch that is not a sequence of strings raises with
        the positions of the items that are neither strings nor missing values, as do invalid releases or databases.
        Each item is read as :py:meth:`convert` reads its identifier; an item that is refused there (an empty string
        or a spelling of a missing value: :py:class:`idtrack.errors.MissingValueError`; a string longer than any
        identifier or one of the graph's internal names: :py:class:`idtrack.errors.InvalidArgumentError`), an item
        that is a missing value as Python, NumPy or pandas hold it (``None``, NaN, ``pd.NA``, ``NaT``: what a column of
        a table holds in an empty cell; a :py:class:`idtrack.errors.MissingValueError`, its ``query_id`` the value
        written as text) and any other error of one identifier (an exceeded budget, an internal error) is that item's
        :py:attr:`idtrack.results.ConversionResult.error`, with status ``error``, and the batch goes on. Each
        distinct identifier is converted once; repeated identifiers get the same (immutable) result. An item is
        answered as :py:meth:`convert` answers it alone, but for the copies of other items made unique by a single-cell
        tool (below).

        Args:
            identifiers: The identifiers (a list, tuple, NumPy array, pandas Index or Series of strings, which may hold
                missing values; not a str).
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
        items = _arguments.strings(identifiers, "identifiers", missing=True)
        from_rel, to_rel, database, source = self._conversion_arguments(
            from_release, to_release, final_database, from_database
        )
        search = self._search(budget_states, budget_seconds)
        explain, allow_bridges = _arguments.boolean(explain, "explain"), _arguments.boolean(
            allow_bridges, "allow_bridges"
        )
        distinct = list(dict.fromkeys(items))
        # Decision 82: an item that is another item with a make-unique suffix (and no exact name of its own) is that
        # item's copy; it is answered after the item it copies.
        copies = self._copies(distinct, from_rel, source, to_rel)
        distinct = [identifier for identifier in distinct if identifier not in copies]
        loop: Iterable[str] = distinct
        if _arguments.boolean(progress, "progress"):
            from tqdm import tqdm

            loop = tqdm(distinct, mininterval=0.25, ncols=100)
        by_identifier: dict[str, ConversionResult] = {}
        for identifier in loop:
            if not isinstance(identifier, str):  # a missing value as a table holds it (decision 81)
                error = MissingValueError(
                    f"`identifiers` holds a missing value ({identifier!r}), not an identifier.",
                    argument="identifiers",
                    value=str(identifier),
                )
                by_identifier[identifier] = self._error_result(str(identifier), from_rel, to_rel, error, explain)
                continue
            try:
                by_identifier[identifier] = self._convert(
                    identifier,
                    self._identifier(identifier, source),
                    from_rel,
                    to_rel,
                    database,
                    source,
                    explain,
                    allow_bridges,
                    search,
                )
            except Exception as error:  # noqa: BLE001 - one identifier's error is that item's answer
                by_identifier[identifier] = self._error_result(identifier, from_rel, to_rel, error, explain)
        for identifier in sorted(copies, key=len):  # a copy of a copy follows the copy it copies
            copied = by_identifier[copies[identifier]]
            match = None if copied.match is None else f"copy:{copied.match}"
            by_identifier[identifier] = dataclasses.replace(copied, query_id=identifier, match=match)
        return [by_identifier[identifier] for identifier in items]

    def _read_where(
        self, text: str, from_release: int | None, from_database: str | None, to_release: int
    ) -> tuple[Readings, bool]:
        """Decide where a string is read, for a conversion and for the batch's copies alike: one rule, one place.

        At *from_release* where the graph holds the string there (:py:meth:`idtrack.engine.Track.reads_at`, the
        engine's own test); otherwise (a release given at which the graph does not hold the string, or no release) by
        the rule without *from_release*: the target release if one of its readings exists there, else the latest
        release at which one does (:py:meth:`idtrack.graph.TheGraph.readings`).

        Args:
            text: The string as read (:py:func:`idtrack.reading.read_identifier`).
            from_release: The release given, or ``None``.
            from_database: The database named as its source.
            to_release: The target release.

        Returns:
            tuple[Readings, bool]: The readings at the release decided, and ``True`` when the release given was set
            aside (``from_release_fallback``).
        """
        if from_release is not None and not self.engine.reads_at(text, from_release, to_release, from_database):
            return self.graph.readings(text, None, from_database, to_release), True
        return self.graph.readings(text, from_release, from_database, to_release), False

    def _copies(
        self, identifiers: list[str], from_release: int | None, from_database: str | None, to_release: int
    ) -> dict[str, str]:
        """Find the items of a batch that are copies of other items made unique by a single-cell tool.

        An item is a copy when it is another item followed by ``-<digits>``, ``.<digits>`` or ``_<digits>`` (scanpy's,
        R's and other tools' suffixes for repeated names), is not itself an exact name at the release its own
        conversion reads it at (``IGHV3-30-2``, ``MIR486-2`` are names of their own), and the item it extends is no
        Ensembl or LRG id (``ENSG00000141510.5`` beside ``ENSG00000141510`` is a version) and no versioned accession
        that the item writes with a version (``NM_000546.5`` beside ``NM_000546`` is a version of it, read as one,
        not judged, or a version the accession never had,
        :py:meth:`idtrack.graph.TheGraph.known_versions`). The release is the one
        :py:meth:`convert` reads the item at: *from_release* where the graph holds the item there, else (as without
        *from_release*) the target release if one of its readings exists there, else the latest release at which one
        does, decided by the one rule of :py:meth:`_read_where`. So every item that is not a copy is answered in a batch
        as :py:meth:`convert` answers it alone.

        Args:
            identifiers: The distinct items, as given.
            from_release: The release the items are read at (``None``: as the conversion chooses).
            from_database: The database named as their source.
            to_release: The target release.

        Returns:
            dict[str, str]: ``{copy: the item it copies}``.
        """
        given = set(identifiers)
        copies: dict[str, str] = {}
        for identifier in identifiers:
            if not isinstance(identifier, str):
                continue  # a missing value
            suffix = ids.NUMBERED_SUFFIX.match(identifier.strip())
            if suffix is None or suffix.group(1) not in given or ids.STABLE_ID.match(suffix.group(1).strip()):
                continue  # an Ensembl or LRG id with a version is that id's version, never a copy
            try:
                text = read_identifier(identifier, "identifier", from_database).text
                found, _ = self._read_where(text, from_release, from_database, to_release)  # where convert reads it
            except InputError:  # an item refused by the reading is not a copy; its conversion says why
                continue
            # an exact name, or an Ensembl id of the graph that names no node (a transcript or translation id without
            # version, as `_convert` reads it): an id of its own, never a copy
            exact = (bool(found.taken) and found.taken[0][1] == "exact") or (
                not found.taken and bool(self.graph.versions_of(text))
            )
            version = bool(found.taken) and (
                self.graph.reading_version(text, *found.taken[0]) is not None
                or found.taken[0][1] == "version_not_judged"
            )
            if not (exact or version or (not found.taken and self.graph.known_versions(text))):
                copies[identifier] = suffix.group(1)
        return copies

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
            reasons=("error",),
            error=ErrorInfo(type=type(error).__name__, message=str(error), input_error=isinstance(error, InputError)),
            path=_NO_PATH if explain else None,
        )

    @staticmethod
    def reason_text(result: ConversionResult) -> str | None:
        """Return the reason of a result in words (:py:data:`idtrack.results.REASONS`).

        Args:
            result: A result.

        Returns:
            str | None: The text; ``None`` for a result with a target in the requested database, read at the release
            asked about.
        """
        return None if result.reason is None else REASONS[result.reason]

    @staticmethod
    def classify(results: list[ConversionResult], strategy: str = "best") -> dict[str, list[ConversionResult]]:
        """Group conversion results into bins for reporting, by the targets of one view (``best`` or ``all``).

        Every result is in ``"input_identifiers"`` and in exactly one of the bins ``"matching_1_to_0"`` (not found,
        no conversion, or an error), ``"read_at_another_release"`` (a result with candidates that the graph could not
        read at the release given, read where it holds the identifier: an answer to another question,
        :py:attr:`idtrack.results.ConversionResult.from_release_fallback`), ``"matching_1_to_1"``, ``"matching_1_to_n"``,
        ``"alternative_target_1_to_1"`` and ``"alternative_target_1_to_n"`` (the last two: ``no_target``, the targets
        are Ensembl genes standing in for a missing id of the requested database). A result of ``"matching_1_to_1"`` is
        also in ``"changed_only_1_to_1"`` when its target is not the identifier it resolved to, and one of
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
            if result.from_release_fallback:
                bins["read_at_another_release"].append(result)
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
        elsewhere = len(classified.get("read_at_another_release", []))
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
            f"  Read at another release: {elsewhere} ({pct(elsewhere, total):.1f}%)\n"
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

    def source_inference(self, identifiers: Iterable[str], mode: str = "assembly_release") -> SourceInference:
        """Infer where a list of identifiers comes from: the origins it cannot tell apart, and by how much.

        Each distinct identifier counts once for every origin (database, assembly, release) at which it
        is an identifier, read as it is at that release: a string counts at a release where name
        resolution at that release (:py:meth:`idtrack.graph.TheGraph.readings`) reads it as a name of that release,
        so that a name renamed later counts as the name it was where it was one, and not at the releases where it is
        only a synonym (a synonym says the string was once a name or an alias, not that a table of that release wrote
        it). An Ensembl id counts at every origin of its node
        (:py:meth:`idtrack.graph.TheGraph.trios`). Strings refused as identifiers (missing values, the graph's
        internal names), missing values as a table holds them (``None``, NaN, ``pd.NA``, ``NaT``) and strings not in
        the graph are counted in :py:attr:`SourceInference.not_found`.

        The best origin is the one where most identifiers are found. Another origin belongs to the *band* of origins
        the list cannot tell apart from the best unless the identifiers found at one of the two and not at the other
        make it significantly worse: a one-sided sign test on those identifiers at the level
        :py:data:`SOURCE_SIGNIFICANCE` (a list of the names of release R is found whole at R, and its neighbours lose
        only the few names renamed between them). The margin is the best count minus the count of the best origin
        outside the band.

        Args:
            identifiers: Identifiers (a sequence of strings, which may hold missing values; not a str).
            mode: Granularity of an origin (:py:data:`idtrack.engine.SOURCE_MODES`): ``"complete"`` →
                ``[database, assembly, release]``, ``"release"`` → ``release``, ``"assembly"`` → ``assembly``,
                ``"assembly_release"`` (default) → ``[assembly, release]``.

        Returns:
            SourceInference: The band, the margin and the count of every origin.

        Raises:
            idtrack.errors.InvalidArgumentError: If *mode* is unknown.
            idtrack.errors.ArgumentTypeError: If *identifiers* is not a sequence of strings.
            idtrack.errors.UnknownIdentifierError: If none of the identifiers is in the graph.
        """
        mode = self.engine._check_mode(mode)
        items = _arguments.strings(identifiers, "identifiers", missing=True)
        project = _ORIGIN_OF[mode]
        origins_of: list[frozenset] = []
        seen: set[str] = set()
        not_found = 0
        for item in dict.fromkeys(items):
            try:
                text = read_identifier(item).text
            except InputError:
                not_found += 1
                continue
            if text in seen:  # two spellings of one identifier (white space, quotes) are one identifier
                continue
            seen.add(text)
            origins = self._origins(text, project, served_only=mode == "release")
            if origins:
                origins_of.append(origins)
            else:
                not_found += 1
        if not_found:
            self.log.warning(f"{not_found} of {len(items)} identifiers are not in the graph and are not used.")
        if not origins_of:
            raise UnknownIdentifierError(
                f"None of the {len(items)} identifiers is in the graph, so their source cannot be inferred.",
                argument="identifiers",
            )
        counts: Counter = Counter()
        for origins in origins_of:
            counts.update(origins)
        ranking = sorted(counts.items(), key=lambda oc: (-oc[1], _origin_key(oc[0])))
        best, top = ranking[0]
        shared: Counter = Counter()  # identifiers found both at the best origin and at each other
        for origins in origins_of:
            if best in origins:
                shared.update(origins)
        band: list = [best]
        outside: list = []
        for origin, count in ranking[1:]:
            fewer = _significantly_fewer(top - shared[origin], count - shared[origin])
            (outside if fewer else band).append(origin)
        return SourceInference(
            mode=mode,
            band=tuple(_plain(origin) for origin in band),
            margin=None if not outside else int(top - counts[outside[0]]),
            counts=tuple((_plain(origin), int(count)) for origin, count in ranking),
            identifiers=len(origins_of),
            not_found=not_found,
        )

    def _origins(self, text: str, project: Any, served_only: bool = False) -> frozenset:
        """Return the origins of one identifier, read at each release as name resolution reads it there.

        An Ensembl id counts at every origin of its node. A string counts at a release, on the assembly that serves
        it, where :py:meth:`idtrack.graph.TheGraph.readings` at that release reads it as a name of that release (its
        exact name, a symbol read without letter case, a reading by case or separators); where it reads it as a
        synonym, or as an identifier of another release, the string does not count there: a synonym records that the
        string was once a name or an alias, not that a table of that release wrote it. The releases are read once
        per run of releases at which the same nodes the string can mean exist (:py:meth:`name_candidates`).

        Args:
            text: The identifier, as read (:py:func:`idtrack.reading.read_identifier`).
            project: The origin of a ``(database, assembly, release)`` trio in the mode asked for.
            served_only: Count a release only on the assembly that serves it (the mode ``release``: a release is its
                own annotation); otherwise the string counts too where it is exactly a name of an assembly Ensembl
                keeps beside the one that serves the release (human GRCh37 with its names of release 75).

        Returns:
            frozenset: The origins; empty when the identifier is not in the graph.
        """
        graph = self.graph
        serving = graph.serving_assembly
        nodes = [node for node, _ in graph.readings(text).taken] or list(graph.versions_of(text))
        if not nodes:  # the string means nothing at any release
            return frozenset()
        if any(graph.node_type(node) != ids.EXTERNAL for node in nodes):
            return frozenset(
                project(trio)
                for node in nodes
                for trio in graph.trios(node)
                if not served_only or serving.get(int(trio[2])) == int(trio[1])
            )
        candidates = graph.name_candidates(text)
        if not candidates:
            return frozenset()
        trios = {node: graph.trios(node) for node in candidates}
        served = {node: {int(r) for _d, a, r in trios[node] if serving.get(int(r)) == int(a)} for node in candidates}
        runs: dict[frozenset, list[int]] = {}  # releases by the nodes that exist there
        for release in sorted(set().union(*served.values())):
            runs.setdefault(frozenset(node for node in candidates if release in served[node]), []).append(release)
        origins: set = set()
        for releases in runs.values():
            found = graph.readings(text, releases[0])
            if not found.here or not found.taken:
                continue
            names = [node for node, how in found.taken if not how.startswith("synonym:") and node in trios]
            if len(names) < len(found.taken):
                continue  # read as a synonym there
            wanted = set(releases)
            origins.update(
                project(trio)
                for node in names
                for trio in trios[node]
                if int(trio[2]) in wanted and serving.get(int(trio[2])) == int(trio[1])
            )
        if not served_only and text in trios:
            origins.update(project(trio) for trio in trios[text] if serving.get(int(trio[2])) != int(trio[1]))
        return frozenset(origins)

    def infer_source(
        self, identifiers: Iterable[str], mode: str = "assembly_release", report_only_winner: bool = True
    ) -> list:
        """Infer the origins (database, assembly, release) of a list of identifiers (:py:meth:`source_inference`).

        Args:
            identifiers: Identifiers (a sequence of strings, not a str).
            mode: Granularity of an origin (:py:data:`idtrack.engine.SOURCE_MODES`): ``"complete"`` →
                ``[database, assembly, release]``, ``"release"`` → ``release``, ``"assembly"`` → ``assembly``,
                ``"assembly_release"`` (default) → ``[assembly, release]``.
            report_only_winner: ``True`` (default) returns the band: the origins the list cannot tell apart from the
                best one, best first (:py:attr:`SourceInference.band`); ``False`` returns every origin with the number
                of distinct identifiers found there, by descending count, then origin (the margin is the band's
                first count minus the first count outside it).

        Returns:
            list: The band, or ``[origin, count]`` pairs.

        Raises:
            idtrack.errors.InvalidArgumentError: If *mode* is unknown.
            idtrack.errors.ArgumentTypeError: If *identifiers* is not a sequence of strings.
            idtrack.errors.UnknownIdentifierError: If none of the identifiers is in the graph.
        """
        report_only_winner = _arguments.boolean(report_only_winner, "report_only_winner")
        inference = self.source_inference(identifiers, mode)
        if not report_only_winner:
            return [[origin, count] for origin, count in inference.counts]
        return list(inference.band)


#: The level of the sign test that tells the origins of a list apart (:py:meth:`API.source_inference`).
SOURCE_SIGNIFICANCE = 0.05


def _significantly_fewer(fewer: int, more: int) -> bool:
    """Tell whether an origin is significantly worse than the best for a list (a one-sided sign test).

    Args:
        fewer: The identifiers found at the best origin and not at the other.
        more: The identifiers found at the other origin and not at the best.

    Returns:
        bool: Whether ``fewer`` exceeds ``more`` more than chance would at :py:data:`SOURCE_SIGNIFICANCE`.
    """
    n = fewer + more
    if fewer <= more:
        return False
    if n <= 1000:
        p = sum(math.comb(n, k) for k in range(fewer, n + 1)) / 2**n
    else:
        z = (fewer - 0.5 - n / 2) / math.sqrt(n / 4)
        p = 0.5 * math.erfc(z / math.sqrt(2))
    return p < SOURCE_SIGNIFICANCE


__all__ = ["API", "BINS", "SOURCE_SIGNIFICANCE"]
