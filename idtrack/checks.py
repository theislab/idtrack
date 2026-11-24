"""The self-tests of a graph: checks that compare a built graph with Ensembl's tables and with itself."""

import itertools
import logging
import os
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from idtrack import _arguments, ids, organisms
from idtrack.engine.track import Track, _renamed_note
from idtrack.errors import (
    ArgumentTypeError,
    InactiveAtReleaseError,
    IntegrityError,
    InvalidArgumentError,
    SearchBudgetExceededError,
    UnknownAssemblyError,
    UnknownDatabaseError,
)
from idtrack.graph import ReleaseSet, TheGraph, served_mask
from idtrack.results import check_strategy
from idtrack.settings import SearchSettings

if TYPE_CHECKING:
    from idtrack.build.database_manager import DatabaseManager


@dataclass(frozen=True)
class CheckResult:
    """What one self-test of :py:class:`GraphChecks` found.

    Every check (the methods whose names begin with ``is_``) returns one. Its truth value is :py:attr:`holds`, so
    ``if tests.is_node_consistency_robust():`` and ``assert tests.is_node_consistency_robust()`` mean what they say.

    Attributes:
        check (str): The name of the check (the method).
        holds (bool): ``True`` when the invariant holds for the graph. ``False`` when a violation was found, and
            when the check could not be completed (then :py:attr:`details` has ``"error"``).
        details (dict[str, Any]): What was compared and what was found: the counts of the things compared, and each
            kind of violation with its number and its first examples (at most
            :py:attr:`idtrack.checks.GraphChecks.details_limit` of them). For a check that could not be completed: ``"error"`` (the
            type and message of the exception) and ``"traceback"``.
    """

    check: str
    holds: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return whether the invariant holds."""
        return self.holds


class GraphChecks:
    """The self-tests of a graph: checks that compare a built graph with Ensembl's tables and with itself.

    ``GraphChecks(graph, database_manager)`` holds a graph, the manager of the tables it was built from, and an engine
    (:py:class:`idtrack.engine.Track`) over the graph. It has two groups of methods, for developers and for the end
    of a graph build. Nothing here changes the graph.

    *Checks* (``is_...``). Each states one invariant of a built graph and returns a :py:class:`CheckResult`:
    whether the invariant holds, and what was compared and found. A check does not raise for what it finds: a
    violation is a result that does not hold, and so is anything that keeps the check from being completed (a
    table that cannot be read, a release without identifiers), with the error in the result's details. So the
    checks can be run one after the other, and a result is never mistaken for success: its truth value is the
    verdict.

    *Conversion runs* (``history_...``). They convert the identifiers of a database at a release and count what
    happened to them; they return the counts and raise an :py:class:`idtrack.errors.InputError` for arguments that
    cannot be used, before anything is searched (a database that the graph does not hold, a release or assembly it
    does not have), naming the argument and what is valid.

    Ensembl keeps one genome assembly per release as the primary one, and an earlier assembly can stay listed
    beside it (human GRCh37). A release therefore exists per assembly: the checks ask the manager for an assembly
    first and for the release second, and compare assembly by assembly.

    Typical use::

        checks = GraphChecks(api.graph, API.database_manager("homo_sapiens", 115, "./idtrack_cache"))
        result = checks.is_id_functions_consistent_ensembl()
        if not result:
            print(result.details)

    Attributes:
        graph (TheGraph): The graph.
        database_manager (DatabaseManager): The manager of the tables the graph was built from.
        track (Track): The engine over the graph, used by the conversion runs.
        details_limit (int): How many examples of each kind of violation a result keeps (the number found is
            always reported).
        random_draw_attempts (int): How often the random arguments of a conversion run are drawn before
            :py:meth:`history_travel_testing_random_arguments_generator` gives up.
    """

    random_draw_attempts: int = 100

    def __init__(
        self,
        graph: TheGraph,
        database_manager: "DatabaseManager",
        *,
        details_limit: int = 50,
        search: SearchSettings | None = None,
    ):
        """Create the self-tests of a graph.

        Args:
            graph: The graph.
            database_manager: The manager of the organism's tables (:py:meth:`idtrack.API.database_manager`).
            details_limit: How many examples of each kind of violation a result keeps.
            search: The search settings of the conversion runs.

        Raises:
            ArgumentTypeError: If *graph* is not a graph or *details_limit* not a positive integer.
        """
        if not isinstance(graph, TheGraph):
            raise ArgumentTypeError(f"`graph` must be an idtrack graph, got {type(graph).__name__}.", argument="graph")
        if isinstance(details_limit, bool) or not isinstance(details_limit, int) or details_limit < 1:
            raise ArgumentTypeError(f"`details_limit` must be a positive integer, got {details_limit!r}.")
        self.log = logging.getLogger(__name__)
        self.graph = graph
        self.database_manager = database_manager
        self.track = Track(graph, search)
        self.details_limit = details_limit

    # ------------------------------------------------------------------ what every check shares

    def _run(self, name: str, check: Callable[[], tuple[bool, dict[str, Any]]]) -> CheckResult:
        """Run the work of one check and turn what it finds, or what stops it, into a result.

        Args:
            name (str): The name of the check.
            check (Callable[[], tuple[bool, dict[str, Any]]]): The work: it returns whether the invariant holds
                and the details.

        Returns:
            CheckResult: The verdict. An exception of the work is a result that does not hold, with ``"error"``
            and ``"traceback"`` in its details; it is logged, as a violation is.
        """
        try:
            holds, details = check()
        except Exception as error:
            details = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            self.log.warning(f"Self-test `{name}` could not be completed: {details['error']}")
            return CheckResult(name, False, details)
        if not holds:
            found = {key: value for key, value in details.items() if key.startswith("n_") and value}
            self.log.warning(f"Self-test `{name}`: the invariant does not hold: {found}.")
        return CheckResult(name, bool(holds), details)

    def _examples(self, found: list) -> list:
        """Return the first examples of a kind of violation (:py:attr:`details_limit` of them)."""
        return found[: self.details_limit]

    def _manager(self, release: int, assembly: int) -> "DatabaseManager":
        """Return the manager of a genome assembly at a release.

        The manager is moved to the assembly first, at a release that the assembly has, and to the release second:
        a release exists per assembly, and asking the manager of one assembly for a release that only another one
        has is an error.

        Args:
            release (int): The Ensembl release.
            assembly (int): The genome assembly.

        Returns:
            DatabaseManager: A manager of that assembly at that release.
        """
        return self.database_manager.change_assembly(int(assembly), last_possible_release=True).change_release(
            int(release)
        )

    def _assembly_releases(self) -> list[tuple[int, int]]:
        """Return every ``(assembly, release)`` of the graph: each release that an assembly has, by assembly."""
        releases = set(self.graph.releases)
        return [
            (int(assembly), int(release))
            for assembly, of_assembly in sorted(self.graph.assembly_releases.items())
            for release in of_assembly
            if int(release) in releases
        ]

    def _serving_assembly(self, release: Any, name: str) -> int:
        """Return the assembly that serves a release argument in the backbone of the graph (checked by the engine).

        Raises:
            ArgumentTypeError: If the release is not an integer.
            UnknownReleaseError: If the release is not one of the graph.
        """
        return int(self.graph.serving_assembly[self.track._check_release(release, name)])

    def _source_databases(self, assembly: int) -> list[str]:
        """Return the names under which the graph holds identifiers on a genome assembly, sorted.

        The external databases of the assembly, and the Ensembl ids of every form of the graph by their node type
        (``ensembl_<form>``) and by their label on the assembly (``assembly_<key>_ensembl_<form>``), and the gene ids
        without version (``base_ensembl_gene``). Empty for an assembly that the graph does not have.
        """
        graph = self.graph
        if assembly not in graph.available_assemblies:
            return []
        held = set(graph.available_external_databases_assembly.get(assembly, ()))
        for form in graph.available_forms:
            held.update((ids.ENSEMBL_TYPE[form], organisms.ASSEMBLY_LABELS[assembly][form]))
        held.add(ids.BASE_ENSEMBL_TYPE[ids.BACKBONE_FORM])
        return sorted(held)

    def _gene_id_databases(self) -> list[str]:
        """Return the external databases of the graph that attach their ids to genes (synonyms excepted), sorted."""
        forms = self.graph.external_database_connection_form
        return sorted(
            name
            for name in self.graph.available_external_databases
            if forms.get(name) == ids.BACKBONE_FORM and not name.startswith(ids.SYNONYM_PREFIX)
        )

    def _refuse_unknown_database(self, argument: str, value: Any, allowed: list[str], what: str) -> None:
        """Refuse a database argument that is not one of the databases that can be used.

        Args:
            argument (str): The name of the argument.
            value (Any): What was given.
            allowed (list[str]): The databases that can be used.
            what (str): What they are, for the message.

        Raises:
            ArgumentTypeError: If the value is not a string.
            UnknownDatabaseError: If it is not one of *allowed*; the message names the argument and lists what is valid.
        """
        _arguments.string(value, argument)
        if value not in allowed:
            raise UnknownDatabaseError(
                f"`{argument}`={value!r} is not {what};{_renamed_note(self.graph, value)} valid: {allowed}.",
                argument=argument,
                value=value,
                allowed=allowed,
            )

    def _is_real_gene(self, node: str) -> bool:
        """Tell whether a node is a gene id with version of the backbone (not ``<id>.Void`` or ``<id>.Retired``)."""
        return self.graph.is_backbone(node) and self.graph.version(node) not in ids.SENTINEL_VERSIONS

    # ------------------------------------------------------------------ the graph against Ensembl's tables

    def is_id_functions_consistent_ensembl(self, progress: bool = True) -> CheckResult:
        """Check that the gene ids of the graph are those of Ensembl's id tables, on every assembly at every release.

        For each genome assembly and each release that the assembly has in the graph, the gene ids with their
        versions are read from the id table of that assembly and release (through the manager) and from the graph
        (the genes that the graph has on that assembly at that release,
        :py:meth:`idtrack.TheGraph.trios`). The two sets must be equal: no gene of the table is missing in the
        graph and the graph has no gene that the table lacks.

        Args:
            progress (bool): Show a progress bar.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of assembly-release pairs), ``n_differing`` and
            ``differing`` (per differing pair its ``assembly`` and ``release``, the numbers ``n_missing`` and
            ``n_extra``, and the first ids ``missing`` in the graph and ``extra`` in it).
        """
        return self._run("is_id_functions_consistent_ensembl", lambda: self._gene_ids_against_tables(progress))

    def _gene_ids_against_tables(self, progress: bool) -> tuple[bool, dict[str, Any]]:
        labels = {
            organisms.ASSEMBLY_LABELS[assembly][ids.BACKBONE_FORM]: assembly
            for assembly in self.graph.available_assemblies
        }
        in_graph: dict[tuple[int, int], set[str]] = {}
        for node in self.graph.nodes:
            node_type = self.graph.node_type(node)
            if node_type in labels or self._is_real_gene(node):
                for label, assembly, release in self.graph.trios(node):
                    if label in labels:
                        in_graph.setdefault((assembly, release), set()).add(node)

        pairs = self._assembly_releases()
        differing = []
        with tqdm(pairs, mininterval=0.25, disable=not progress, ncols=100) as loop_obj:
            for assembly, release in loop_obj:
                loop_obj.set_postfix_str(f"Item:{assembly},{release}", refresh=False)
                manager = self._manager(release, assembly).change_form(ids.BACKBONE_FORM)
                in_tables = set(manager.id_ver_from_df(manager.get_db("ids", save_after_calculation=False)))
                missing = sorted(in_tables - in_graph.get((assembly, release), set()))
                extra = sorted(in_graph.get((assembly, release), set()) - in_tables)
                if missing or extra:
                    differing.append(
                        {
                            "assembly": assembly,
                            "release": release,
                            "n_missing": len(missing),
                            "missing": self._examples(missing),
                            "n_extra": len(extra),
                            "extra": self._examples(extra),
                        }
                    )
        details = {"compared": len(pairs), "n_differing": len(differing), "differing": self._examples(differing)}
        return not differing, details

    def is_id_functions_consistent_ensembl_2(self, progress: bool = True) -> CheckResult:
        """Check that every gene version is referred to exactly at the releases of its history.

        For every gene id with version of the backbone, two things the graph says independently must agree:

        1. *The releases of its edges*: the releases at which the gene is referred to (by its id without version,
           its transcripts, external identifiers) on the assembly that serves the release
           (:py:meth:`idtrack.TheGraph.combined`, :py:attr:`idtrack.TheGraph.serving_line`). A gene that lived
           on an earlier assembly is referred to there at the releases that assembly served.
        2. *The releases of its history*: where Ensembl's id history, validated against the id tables, has the
           version (:py:meth:`idtrack.TheGraph.active_releases`).

        Args:
            progress (bool): Show a progress bar.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of gene versions), ``n_differing`` and ``differing``
            (per gene its ``node`` and the release ranges ``edges`` and ``history``).
        """
        return self._run("is_id_functions_consistent_ensembl_2", lambda: self._edges_against_history(progress))

    def _edges_against_history(self, progress: bool) -> tuple[bool, dict[str, Any]]:
        genes = [node for node in self.graph.nodes if self._is_real_gene(node)]
        line = self.graph.serving_line
        differing = []
        with tqdm(genes, mininterval=0.25, disable=not progress, ncols=100) as loop_obj:
            for gene in loop_obj:
                loop_obj.set_postfix_str(f"Item:{gene}", refresh=False)
                try:
                    summary = self.graph.combined(gene)
                except KeyError:  # a gene that no cross-reference edge enters
                    summary = {}
                edges = 0
                for label, per_assembly in summary.items():
                    for assembly, releases in per_assembly.items():
                        edges |= served_mask(((label, assembly, releases),), line)
                history = self.graph.active_releases(gene).mask
                if edges != history:
                    differing.append(
                        {
                            "node": gene,
                            "edges": self.graph.list_to_ranges(sorted(ReleaseSet.from_mask(edges))),
                            "history": self.graph.list_to_ranges(sorted(ReleaseSet.from_mask(history))),
                        }
                    )
        details = {"compared": len(genes), "n_differing": len(differing), "differing": self._examples(differing)}
        return not differing, details

    def is_id_functions_consistent_external(self, progress: bool = True) -> CheckResult:
        """Check that the external identifiers of the graph are those of Ensembl's tables.

        For each genome assembly and each release that the assembly has in the graph, and for every external
        database, the identifiers of the manager's table of external identifiers (of every form of the graph) are
        compared with the identifiers the graph has for that database, assembly and release
        (:py:meth:`idtrack.TheGraph.trios`). Both differences count: an identifier of the tables that the graph
        lacks, and an identifier of the graph that the tables do not have (the kind of error that produces wrong
        mappings). The graph keeps one node for identifiers that differ in letter case only, so the comparison
        ignores letter case; an identifier of the tables that is itself an Ensembl id of the graph is no external
        identifier there (:py:attr:`idtrack.TheGraph.misplaced_external_entries`) and is left out, and so is an entry of
        :py:data:`idtrack.build.database_manager.ONE_GENE_ONLY` that the build left out because the tables give it to
        several genes over the releases of the graph (:py:attr:`idtrack.TheGraph.entries_of_several_genes`).

        Args:
            progress (bool): Show a progress bar per assembly.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of database-assembly-release triples), ``n_differing``
            and ``differing`` (per differing triple its ``database``, ``assembly`` and ``release``, the numbers
            ``n_missing`` and ``n_extra``, and the first identifiers ``missing`` in the graph and ``extra`` in it,
            in lower case).
        """
        return self._run("is_id_functions_consistent_external", lambda: self._external_ids_against_tables(progress))

    def _external_ids_against_tables(self, progress: bool) -> tuple[bool, dict[str, Any]]:
        table_name = "external" if self.graph.all_external else "external_relevant"
        misplaced = {name.lower() for name in self.graph.misplaced_external_entries}
        several = self.graph.entries_of_several_genes
        in_graph: dict[tuple[int, int], dict[str, set[str]]] = {}
        for node in self.graph.nodes:
            if self.graph.node_type(node) == ids.EXTERNAL:
                for database, assembly, release in self.graph.trios(node):
                    in_graph.setdefault((assembly, release), {}).setdefault(database, set()).add(node.lower())

        pairs = self._assembly_releases()
        compared = 0
        differing = []
        with tqdm(pairs, mininterval=0.25, disable=not progress, ncols=100) as loop_obj:
            for assembly, release in loop_obj:
                loop_obj.set_postfix_str(f"Item:{assembly},{release}", refresh=False)
                manager = self._manager(release, assembly)
                judged = manager.one_gene_only_names() if several else set()
                in_tables: dict[str, set[str]] = {}
                for form in self.graph.available_forms:
                    table = manager.change_form(form).get_db(table_name)
                    for database, identifier in zip(table["name_db"].tolist(), table["id_db"].tolist()):
                        lower = str(identifier).lower()
                        if lower in misplaced or (
                            lower in several and str(database).removeprefix(ids.SYNONYM_PREFIX) in judged
                        ):
                            continue
                        in_tables.setdefault(str(database), set()).add(lower)
                of_graph = in_graph.get((assembly, release), {})
                for database in sorted(set(in_tables) | set(of_graph)):
                    compared += 1
                    missing = sorted(in_tables.get(database, set()) - of_graph.get(database, set()))
                    extra = sorted(of_graph.get(database, set()) - in_tables.get(database, set()))
                    if missing or extra:
                        differing.append(
                            {
                                "database": database,
                                "assembly": assembly,
                                "release": release,
                                "n_missing": len(missing),
                                "missing": self._examples(missing),
                                "n_extra": len(extra),
                                "extra": self._examples(extra),
                            }
                        )
        details = {"compared": compared, "n_differing": len(differing), "differing": self._examples(differing)}
        return not differing, details

    # ------------------------------------------------------------------ the graph against itself

    def is_range_functions_robust(self, progress: bool = True) -> CheckResult:
        """Check that no two versions of one gene id are active at the same release.

        The versions of a gene id follow one another in Ensembl's history: at every release of the backbone an id
        has one version at most. For every gene id without version the release ranges of its versions on the
        backbone (:py:meth:`idtrack.TheGraph.active_ranges`) are compared pair by pair; any release that two of
        them share is a violation.

        Args:
            progress (bool): Show a progress bar.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of gene ids), ``n_overlapping`` and ``overlapping``
            (per pair the ``id``, the two ``versions`` and the ``releases`` they share, as ranges).
        """
        return self._run("is_range_functions_robust", lambda: self._overlapping_versions(progress))

    def _base_ids(self) -> list[str]:
        base_type = ids.BASE_ENSEMBL_TYPE[ids.BACKBONE_FORM]
        return [node for node in self.graph.nodes if self.graph.node_type(node) == base_type]

    def _overlapping_versions(self, progress: bool) -> tuple[bool, dict[str, Any]]:
        base_ids = self._base_ids()
        overlapping = []
        with tqdm(base_ids, mininterval=0.25, disable=not progress, ncols=100) as loop_obj:
            for base_id in loop_obj:
                loop_obj.set_postfix_str(f"Item:{base_id}", refresh=False)
                versions = [version for version, _ in self.graph.xref_edges(base_id) if self._is_real_gene(version)]
                ranges = {version: self.graph.active_ranges(version) for version in versions}
                for first, second in itertools.combinations(versions, 2):
                    shared = self.graph.get_intersecting_ranges(ranges[first], ranges[second], compact=False)
                    if shared:
                        # Open ranges end at the last release of the graph (strict JSON holds no infinity).
                        last = self.graph.snapshot_release
                        shared = [[a, last if np.isinf(b) else int(b)] for a, b in shared]
                        overlapping.append({"id": base_id, "versions": [first, second], "releases": shared})
        details = {
            "compared": len(base_ids),
            "n_overlapping": len(overlapping),
            "overlapping": self._examples(overlapping),
        }
        return not overlapping, details

    def is_base_is_range_correct(self, progress: bool = True) -> CheckResult:
        """Check that a gene id without version exists exactly where one of its versions does.

        For every gene id without version two ways of saying where it exists must agree:

        1. the releases of the edges from the id to its versions, on any assembly
           (:py:meth:`idtrack.TheGraph.combined`);
        2. the union of the releases of its versions on any assembly
           (:py:meth:`idtrack.TheGraph.get_active_ranges_of_base_id_alternative`, which also refuses a version
           whose edges contradict its history; that is reported as a difference too).

        Args:
            progress (bool): Show a progress bar.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of gene ids), ``n_differing`` and ``differing`` (per
            gene id its ``id``, the ranges of its ``edges`` and of its ``versions``, or the ``contradiction`` of a
            version).
        """
        return self._run("is_base_is_range_correct", lambda: self._base_ranges(progress))

    def _base_ranges(self, progress: bool) -> tuple[bool, dict[str, Any]]:
        base_ids = self._base_ids()
        last = self.graph.snapshot_release
        differing: list[dict[str, Any]] = []
        with tqdm(base_ids, mininterval=0.25, disable=not progress, ncols=100) as loop_obj:
            for base_id in loop_obj:
                loop_obj.set_postfix_str(f"Item:{base_id}", refresh=False)
                summary = self.graph.combined(base_id)
                edges = self.graph.list_to_ranges(
                    sorted({release for per in summary.values() for releases in per.values() for release in releases})
                )
                try:
                    of_versions = self.graph.get_active_ranges_of_base_id_alternative(base_id)
                except IntegrityError as error:
                    differing.append({"id": base_id, "contradiction": str(error)})
                    continue
                versions = [[first, end if not np.isinf(end) else last] for first, end in of_versions]
                if edges != versions:
                    differing.append({"id": base_id, "edges": edges, "versions": versions})
        details = {"compared": len(base_ids), "n_differing": len(differing), "differing": self._examples(differing)}
        return not differing, details

    def is_combined_summary_complete(self) -> CheckResult:
        """Check that every node has a summary of its cross-reference edges, except the two placeholders of an id.

        :py:meth:`idtrack.TheGraph.combined` sums what the cross-reference edges of a node say, by the kind of the
        node (the edges that enter a gene, the edges that leave any other node). Every node of a built graph has
        such edges, with two exceptions by construction: ``<id>.Void`` and ``<id>.Retired``, the nodes that stand
        for "before the id existed" and "after it was retired", which only the history touches.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of nodes), ``n_without_summary`` and
            ``without_summary`` (nodes that no cross-reference edge touches, or whose summary is empty),
            ``n_placeholders_with_summary`` and ``placeholders_with_summary``.
        """
        return self._run("is_combined_summary_complete", self._summaries)

    def _summaries(self) -> tuple[bool, dict[str, Any]]:
        compared = 0
        without_summary: list[str] = []
        placeholders_with_summary: list[str] = []
        for node in self.graph.nodes:
            compared += 1
            placeholder = self.graph.is_backbone(node) and self.graph.version(node) in ids.SENTINEL_VERSIONS
            try:
                summary = self.graph.combined(node)
            except KeyError:  # a node that no cross-reference edge touches
                if not placeholder:
                    without_summary.append(node)
                continue
            if placeholder:
                placeholders_with_summary.append(node)
            elif not summary:
                without_summary.append(node)
        details = {
            "compared": compared,
            "n_without_summary": len(without_summary),
            "without_summary": self._examples(without_summary),
            "n_placeholders_with_summary": len(placeholders_with_summary),
            "placeholders_with_summary": self._examples(placeholders_with_summary),
        }
        return not without_summary and not placeholders_with_summary, details

    def is_edge_with_same_nts_only_at_backbone_nodes(self) -> CheckResult:
        """Check that only genes of the backbone are linked to nodes of their own type.

        The graph has layers: the history links gene versions of the backbone with each other; every other edge
        links nodes of two different types (an external identifier and a gene, a transcript and its gene, a gene id
        without version and its versions). An edge between two nodes of one type that is not the backbone's is a
        violation.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of nodes), ``n_same_type`` and ``same_type`` (pairs
            of linked nodes of one type that is not the backbone's, with the type).
        """
        return self._run("is_edge_with_same_nts_only_at_backbone_nodes", lambda: self._neighbours(False, False))

    def is_node_consistency_robust(self, progress: bool = True) -> CheckResult:
        """Check the two rules of the graph's edges: no edge within a layer but the backbone's, one edge between layers.

        Beside the rule of :py:meth:`is_edge_with_same_nts_only_at_backbone_nodes`, two nodes of different types
        are linked by exactly one edge: what the databases say about the pair at all releases is the content of
        that one edge (:py:class:`idtrack.graph.Connection`). Several edges are allowed only between gene
        versions of the backbone (one per event of the history).

        Args:
            progress (bool): Show a progress bar.

        Returns:
            CheckResult: ``details``: ``compared`` (the number of nodes), ``n_same_type`` and ``same_type`` as in
            :py:meth:`is_edge_with_same_nts_only_at_backbone_nodes`, ``n_several_edges`` and ``several_edges``
            (pairs of nodes of different types with more than one edge, with their types and the number of edges).
        """
        return self._run("is_node_consistency_robust", lambda: self._neighbours(True, progress))

    def _neighbours(self, count_edges: bool, progress: bool) -> tuple[bool, dict[str, Any]]:
        backbone_type = ids.GENE
        compared = 0
        same_type = []
        several_edges = []
        with tqdm(self.graph.nodes, mininterval=0.25, disable=not progress, ncols=100) as loop_obj:
            for node in loop_obj:
                compared += 1
                node_type = self.graph.node_type(node)
                for other in self.graph.neighbors(node):
                    other_type = self.graph.node_type(other)
                    if node_type == other_type:
                        if node_type != backbone_type:
                            same_type.append([node, other, node_type])
                    elif count_edges and len(self.graph[node][other]) != 1:
                        several_edges.append([node, node_type, other, other_type, len(self.graph[node][other])])
        details: dict[str, Any] = {
            "compared": compared,
            "n_same_type": len(same_type),
            "same_type": self._examples(same_type),
        }
        if count_edges:
            details["n_several_edges"] = len(several_edges)
            details["several_edges"] = self._examples(several_edges)
        return not same_type and not several_edges, details

    def history_travel_testing(
        self,
        from_release: int,
        from_assembly: int,
        from_database: str,
        to_release: int,
        final_database: str,
        allow_bridges: bool,
        strategy: str,
        convert_using_release: bool,
        from_fraction: float = 1.0,
        progress: bool = True,
        progress_details: bool = False,
        return_ensembl_alternative: bool = False,
    ) -> dict[str, Any]:
        """Run an end-to-end Ensembl-history conversion and collect granular QA metrics.

        The routine samples identifiers from *from_database*/*from_release*
        (optionally down-sampling via *from_fraction*) and converts each one to
        *final_database*/*to_release* using :py:meth:`idtrack.engine.Track.convert`.  It
        is intentionally **non-fatal**: every failure mode is caught, logged
        and tallied so that large regression suites can run unattended.  All
        results are returned in a single nested *metrics* dictionary whose
        structure mirrors the printable report that :py:meth:`history_travel_testing_random` logs.

        The statistics fall into **four conceptual groups**; each counter not only records an absolute event
        count but also serves as a red-flag indicator for specific classes of mapping pathology.  Use the
        guidelines below to interpret the numbers and decide whether a run is *healthy*, *questionable*, or
        *action-required*.

        * **Failure / anomaly counters**

          * ``history_voyage_failed_gracefully`` - the converter declined with one of its two named errors:
            :py:class:`idtrack.InactiveAtReleaseError` (the id is not active in the graph at the release
            it was taken from) or :py:class:`idtrack.SearchBudgetExceededError`.
          * ``history_voyage_failed_unknown`` - any *other* unexpected exception.
          * ``lost_item`` - traversal finished but produced **no** final IDs.
          * ``lost_item_but_the_same_id_exists`` - special case of *lost_item*
            when the target and source DB are both Ensembl-gene and the target
            ID still exists in the graph.
          * ``found_ids_not_accurate`` - at least one returned target ID is not
            part of the authoritative *ids_to* reference set.

        * **Mapping quality**

          * ``one_to_one_ids`` - queries that resolved to *exactly* one target ID.
          * ``one_to_multiple_ids`` - queries with > 1 admissible targets.
          * ``one_to_multiple_final_conversion`` - subset of the above where
            *exactly one* traversal path was found (heuristics eliminated alternatives).

        * **Collision analysis**

          The list ``clashing_id_type == [clash_one_one, clash_multi_multi,
          clash_multi_one]`` classifies *target IDs* that were reached by
          more than one query:

          * ``clash_one_one`` - every colliding query was 1→1.
          * ``clash_multi_multi`` - every colliding query was 1→many.
          * ``clash_multi_one`` - mixture of 1→1 and 1→many queries (most
            alarming category).

        * **Timings & book-keeping**

          * ``time`` - wall-clock runtime in seconds.
          * ``conversion`` - per-query mapping result for *all* successful traversals.
          * ``converted_item_dict`` / ``converted_item_dict_reversed`` - raw
            per-ID caches used to derive the higher-level counters above.
          * ``parameters`` - echo of the function arguments.
          * ``ids`` - the sampled *from* and reference *to* ID sets.

        Args:
            from_release (int): Ensembl release number of the *source* IDs.
            from_assembly (int): Genome assembly code of the source IDs.
            from_database (str): Node-type / database of the source IDs.
            to_release (int): Ensembl release number of the *target* IDs.
            final_database (str): Node-type / database to convert **into**.
            allow_bridges (bool): Permit temporary detours through external IDs when native Ensembl history edges break.
            strategy (str): Count the best answer of each conversion (``best``) or every candidate (``all``)
                (:py:data:`idtrack.results.STRATEGIES`).
            convert_using_release (bool): Pass *from_release* straight into
                :py:meth:`idtrack.engine.Track.convert` instead of letting it infer the starting point.
            from_fraction (float): Fraction (0 < x ≤ 1) of the
                *ids_from* population to sample; speeds up smoke tests.
            progress (bool): Show tqdm progress bar (coarse).
            progress_details (bool): Embed live metric counters in the tqdm postfix.
            return_ensembl_alternative (bool): Forwarded to :py:meth:`idtrack.engine.Track.convert`.

        Raises:
            UnknownDatabaseError: If a database is not one that the graph holds; nothing is converted then, and the message
                lists what is valid. A source is an external database of *from_assembly* or the Ensembl ids of a
                form of the graph, named by their node type (``ensembl_gene``, ``base_ensembl_gene``) or by their
                label on the assembly (``assembly_<key>_ensembl_<form>``; the name of a form, ``gene``, is not a
                database). A target is a database that :py:meth:`idtrack.engine.Track.convert` takes as
                ``final_database`` at *to_release* (:py:meth:`idtrack.engine.Track.final_databases`). Also if
                *to_release* is not a release of the graph, or *from_fraction* is outside the interval (0, 1].
            ArgumentTypeError: If a database is not a string.
            InvalidArgumentError: If *strategy* is not ``best`` or ``all``.

        Returns:
            dict: Nested *metrics* dictionary with the layout described above.
                :py:meth:`history_travel_testing_random` logs it as a human-readable summary.

        Notes:
            * All counters are **absolute counts** - divide by
              ``len(metrics['ids']['from'])`` to obtain rates.
            * The collision analysis is inspired by the *clash statistics*
              logic implemented at the end of the function and helps spot
              discrepant “unique” IDs that suddenly become ambiguous. Keeping
              all three clash counters at zero is the gold standard for a
              healthy build.
        """
        strategy = check_strategy(strategy)
        # A database that the graph does not hold is refused here: converted, every id of such a run would be
        # reported as lost or failed.
        to_assembly = self._serving_assembly(to_release, "to_release")
        self._refuse_unknown_database(
            "from_database",
            from_database,
            self._source_databases(from_assembly),
            f"a database of the loaded graph on genome assembly {from_assembly!r} (the graph has the assemblies "
            f"{sorted(self.graph.available_assemblies)})",
        )
        # A target is checked as every conversion checks it (the engine's one rule for `final_database`).
        final_database = self.track._check_final_database(final_database, to_release)

        ids_from = sorted(self.graph.get_id_list(from_database, from_assembly, from_release))
        # The ids a conversion may end in: those of the target database on the assembly that serves the target
        # release (an earlier assembly where the release is one of its own).
        ids_to = set(self.graph.get_id_list(final_database, to_assembly, to_release))
        gene_type = ids.ENSEMBL_TYPE[ids.BACKBONE_FORM]
        ids_to_s = {self.graph.base_id(i) for i in ids_to} if final_database == gene_type else set()

        # Input sampling
        if from_fraction == 1.0:
            pass
        elif 0.0 < from_fraction < 1.0:
            from_faction_count = ceil(len(ids_from) * from_fraction)
            ids_from = sorted(random.sample(ids_from, from_faction_count))  # noqa: S311
        else:
            raise InvalidArgumentError(
                f"from_fraction must be in range (0.0, 1.0], got {from_fraction!r}.", argument="from_fraction"
            )

        # Metric scaffold
        parameters: dict[str, bool | str | int | float] = {
            "from_release": from_release,
            "from_assembly": from_assembly,
            "from_database": from_database,
            "to_release": to_release,
            "final_database": final_database,
            "allow_bridges": allow_bridges,
            "from_fraction": from_fraction,
            "strategy": strategy,
        }

        metrics: dict[str, Any] = {
            "parameters": parameters,
            "ids": {"from": ids_from, "to": ids_to},
            "lost_item": [],
            "one_to_one_ids": {},
            "history_voyage_failed_gracefully": [],
            "history_voyage_failed_unknown": [],
            "lost_item_but_the_same_id_exists": [],
            "found_ids_not_accurate": {},
            "conversion": {},
            "one_to_multiple_ids": {},
            "one_to_multiple_final_conversion": [],
            "converted_item_dict": {},
            "converted_item_dict_reversed": {},
        }

        t1 = time.time()

        with tqdm(ids_from, mininterval=0.25, disable=not progress, desc="Mapping", ncols=100) as loop_obj:
            for the_id in loop_obj:
                if not progress_details:
                    loop_obj.set_postfix_str(f"Item:{the_id}", refresh=False)
                else:
                    suffix = (
                        f"Item:{the_id}, "
                        f"{len(metrics['one_to_one_ids'])},"
                        f"{len(metrics['one_to_multiple_ids'])},"
                        f"{len(metrics['lost_item'])},"
                        f"{len(metrics['found_ids_not_accurate'])},"
                        f"{len(metrics['history_voyage_failed_unknown'])}"
                    )
                    loop_obj.set_postfix_str(suffix, refresh=False)

                # Conversion attempt
                try:
                    conversion = self.track.convert(
                        identifier=the_id,
                        from_release=from_release if convert_using_release else None,
                        to_release=to_release,
                        final_database=final_database,
                        allow_bridges=allow_bridges,
                        return_ensembl_alternative=return_ensembl_alternative,
                    )
                    view = [c for c in conversion.candidates if c.best or strategy == "all"]
                    converted_item = [[c.gene, c.target] for c in view] or None
                    metrics["converted_item_dict"][the_id] = converted_item

                except (InactiveAtReleaseError, SearchBudgetExceededError):
                    full_tb = traceback.format_exc()
                    metrics["history_voyage_failed_gracefully"].append((the_id, full_tb))
                    continue
                except Exception:
                    full_tb = traceback.format_exc()
                    metrics["history_voyage_failed_unknown"].append((the_id, full_tb))
                    continue

                # Metrics aggregation
                if converted_item is None:
                    if final_database == from_database == gene_type and self.graph.base_id(the_id) in ids_to_s:
                        metrics["lost_item_but_the_same_id_exists"].append(the_id)
                    metrics["lost_item"].append(the_id)
                else:
                    covr = {target for _, target in converted_item}
                    metrics["conversion"][the_id] = list(covr)
                    if len(covr) == 1:
                        metrics["one_to_one_ids"][the_id] = list(covr)
                    elif len(covr) > 1:
                        metrics["one_to_multiple_ids"][the_id] = list(covr)
                        if len({gene for gene, _ in converted_item}) == 1:
                            metrics["one_to_multiple_final_conversion"].append(the_id)
                    else:
                        raise IntegrityError(f"Unexpected empty conversion result for ID {the_id!r}.")

                    # How much of the converted IDs show the same ID
                    for c in covr:
                        if c in metrics["converted_item_dict_reversed"]:
                            metrics["converted_item_dict_reversed"][c].append(the_id)
                        else:
                            metrics["converted_item_dict_reversed"][c] = [the_id]

                    for c in covr:
                        if c not in ids_to:
                            if the_id not in metrics["found_ids_not_accurate"]:
                                metrics["found_ids_not_accurate"][the_id] = list()
                            metrics["found_ids_not_accurate"][the_id].append(c)

        # ── Clash statistics:
        # Each counter tracks how many *target* IDs were reached by more than
        # one *source* ID, broken down by the **type of mapping** that led to
        # the collision:
        clash_multi_multi: int = 0  # all colliding sources were 1→many mappings
        clash_multi_one: int = 0  # mix of 1→many and 1→1 sources
        clash_one_one: int = 0  # all colliding sources were clean 1→1

        # `converted_item_dict_reversed` maps every target ID to the list of
        # source IDs that converted to it (built earlier inside the big mapping
        # loop).  We inspect only those targets hit by ≥ 2 sources - a “clash”.
        for target_id, cidr_val in metrics["converted_item_dict_reversed"].items():

            # Skip non-clashing targets (exactly one source mapped here).
            if len(cidr_val) > 1:
                # Does *any* of the sources belong to the 1→many bucket?
                s1 = any([cv in metrics["one_to_multiple_ids"] for cv in cidr_val])
                # Does *any* of the sources belong to the 1→1 bucket?
                s2 = any([cv in metrics["one_to_one_ids"] for cv in cidr_val])
                # Categorise clash by the mixture of mapping types.
                if s1 and s2:  # at least one 1→many **and** one 1→1
                    clash_multi_one += 1
                elif s1:  # only 1→many sources present
                    clash_multi_multi += 1
                elif s2:  # only 1→1 sources present
                    clash_one_one += 1
                else:
                    raise IntegrityError(f"Unexpected state: no sources found for clashing ID {target_id!r}.")
        metrics["clashing_id_type"] = [clash_one_one, clash_multi_multi, clash_multi_one]

        t2 = time.time()
        metrics["time"] = t2 - t1
        return metrics

    def history_travel_testing_random_arguments_generator(
        self, strict_forward: bool, include_exclude_list: list[bool]
    ) -> dict[str, Any]:
        """Draw the arguments of a conversion run: a source and a target that exist in the graph.

        A source is a database (or a kind of Ensembl id) of a genome assembly at a release at which the graph has
        it; a target likewise, on the graph's primary assembly. With *strict_forward* the target release is not
        earlier than the source release. A draw that has no target (the source lies after every release of every
        target) is drawn again, :py:attr:`random_draw_attempts` times at most.

        Args:
            strict_forward (bool): The target release is at or after the source release.
            include_exclude_list (list[bool]): Four switches: Ensembl ids as source, external databases as source,
                Ensembl ids as target, external databases as target. With both switches of a side off, that side
                is the gene ids of the backbone (``ensembl_gene``).

        Returns:
            dict[str, Any]: ``from_assembly``, ``from_release``, ``to_release``, ``from_database`` and
            ``final_database``, to be passed to :py:meth:`history_travel_testing`.

        Raises:
            InvalidArgumentError: If no source and target could be drawn in :py:attr:`random_draw_attempts` attempts (the
                graph has no such pair, or nearly none).
        """
        include_ensembl_1, include_external_1, include_ensembl_2, include_external_2 = include_exclude_list
        only_backbone_tests_1 = not any([include_ensembl_1, include_external_1])
        only_backbone_tests_2 = not any([include_ensembl_2, include_external_2])

        main_assembly = self.graph._primary()
        assemblies = sorted(self.graph.available_assemblies) if not only_backbone_tests_1 else [main_assembly]
        # Final conversions are always evaluated on the graph's primary assembly.
        to_assembly = main_assembly
        for _ in range(self.random_draw_attempts):
            from_assembly = random.choice(assemblies)  # noqa: S311
            source = self.random_dataset_source_generator(
                from_assembly,
                include_ensembl=include_ensembl_1,
                include_external=include_external_1,
                only_backbone_tests=only_backbone_tests_1,
                for_final_database=False,
            )
            if source is None:
                continue
            target = self.random_dataset_source_generator(
                to_assembly,
                include_ensembl=include_ensembl_2,
                include_external=include_external_2,
                for_final_database=True,
                only_backbone_tests=only_backbone_tests_2,
                release_lower_limit=source[2] if strict_forward else None,
            )
            if target is None:
                continue
            return {
                "from_assembly": from_assembly,
                "from_release": source[2],
                "to_release": target[2],
                "from_database": source[0],
                "final_database": target[0],
            }
        raise InvalidArgumentError(
            f"No source and target of a conversion run could be drawn in {self.random_draw_attempts} attempts "
            f"(strict_forward={strict_forward!r}, switches {include_exclude_list!r})."
        )

    def history_travel_testing_random(
        self,
        from_fraction: float,
        include_ensembl_source=True,
        include_external_source=True,
        include_ensembl_destination=True,
        include_external_destination=True,
        progress: bool = True,
        progress_details: bool = False,
        strict_forward: bool = False,
        convert_using_release: bool = False,
        strategy: str = "best",
        allow_bridges: bool = True,
    ) -> dict[str, Any]:
        """Convenience wrapper around :py:meth:`history_travel_testing`.

        The routine generates a *random* but internally consistent test case
        via :py:meth:`history_travel_testing_random_arguments_generator`, logs
        the chosen parameters (unless *progress* is *False*) and delegates the
        heavy lifting to :py:meth:`history_travel_testing`.

        Args:
            from_fraction: Fraction of IDs to sample from the source set.
            strict_forward: Forwarded to the argument generator.
            convert_using_release: Forwarded to
                :py:meth:`history_travel_testing`.
            strategy: Forwarded to :py:meth:`history_travel_testing`.
            progress (bool): Show coarse progress information.
            progress_details (bool): Include extended per-ID counters in the progress bar.
            allow_bridges (bool): Allow bridges through external ids (see :py:meth:`idtrack.engine.Track.convert`).
            include_ensembl_source (bool): Include Ensembl databases as valid sources.
            include_external_source (bool): Include external databases as valid sources.
            include_ensembl_destination (bool): Include Ensembl databases as valid destinations.
            include_external_destination (bool): Include external databases as valid destinations.

        Returns:
            dict: The *metrics* dictionary returned by :py:meth:`history_travel_testing` (with *progress*, the report
            is also logged).

        Raises:
            InvalidArgumentError: If no arguments could be drawn
                (:py:meth:`history_travel_testing_random_arguments_generator`).
        """
        include_exclude_list = [
            include_ensembl_source,
            include_external_source,
            include_ensembl_destination,
            include_external_destination,
        ]

        parameters = self.history_travel_testing_random_arguments_generator(
            strict_forward=strict_forward, include_exclude_list=include_exclude_list
        )

        if progress:
            printable1 = os.linesep + os.linesep.join(self._format_history_travel_testing_report_header(parameters))
            self.log.info(printable1)

        res = self.history_travel_testing(
            **parameters,
            allow_bridges=allow_bridges,
            strategy=strategy,
            convert_using_release=convert_using_release,
            from_fraction=from_fraction,
            progress=progress,
            progress_details=progress_details,
            return_ensembl_alternative=False,
        )

        if progress:
            printable2 = os.linesep + os.linesep.join(
                self._format_history_travel_testing_report(res, include_header=False)
            )
            self.log.info(printable2)

        return res

    def is_final_external_conversion_robust(
        self,
        convert_using_release: bool = False,
        final_database: str | None = None,
        to_release: int | None = None,
        progress: bool = True,
        from_fraction: float = 1.0,
        strategy: str = "all",
    ) -> CheckResult:
        """Check that every gene of a release converts to the ids that an external database gives it there.

        The genes of a release are converted, at that release, into an external database that attaches its ids to
        genes, and each conversion is compared with the manager's table of external identifiers of the assembly
        that serves the release. The invariant holds when every gene that the table gives an id converts to
        exactly those ids (letter case ignored), and no gene converts to an id that the table does not give it. A
        gene without an id in the database converts to nothing; that is the only loss that is not a violation. The
        entries the build left out because Ensembl gives them to several genes
        (:py:attr:`idtrack.TheGraph.entries_of_several_genes`) are no
        ids of the table here.

        Args:
            convert_using_release (bool): Give the release to the conversion as ``from_release`` (otherwise the
                engine chooses where the gene id is read).
            final_database (str | None): The external database. ``None``: drawn at random among the external
                databases that attach ids to genes (at *to_release*, when that is given).
            to_release (int | None): The Ensembl release. ``None``: drawn at random among the releases at which the
                database has ids (of any such database, when *final_database* is ``None`` too).
            progress (bool): Show a progress bar and log what is tested.
            from_fraction (float): The fraction of the genes that is converted (0 < x <= 1).
            strategy (str): ``best``: compare the engine's best answer only; ``all``: every candidate.

        Returns:
            CheckResult: ``details``: ``database``, ``assembly``, ``release``, ``compared`` (the number of genes
            converted), and four kinds of violation, each with its number (``n_...``) and first examples:
            ``different`` (converted to other ids than the table's: ``id``, ``converted``, ``expected``),
            ``unexpected`` (converted although the table gives the gene no id), ``lost_with_id`` (not converted
            although the table gives the gene an id) and ``failed`` (the conversion raised). Also ``lost`` (every
            gene that was not converted, with or without an id in the table) and ``run`` (the counts of
            :py:meth:`history_travel_testing`). A check that could not be made (no database and release to draw,
            a release at which the database has no ids, a release the graph does not have) does not hold and
            says why in ``error``.

        Raises:
            UnknownDatabaseError: If *final_database* is not an external database of the graph that attaches its ids to
                genes; nothing is converted then, and the message lists what is valid.
            ArgumentTypeError: If *final_database* is neither a string nor ``None``.
            InvalidArgumentError: If *strategy* is not ``best`` or ``all``.
        """
        strategy = check_strategy(strategy)
        if final_database is not None:
            self._refuse_unknown_database(
                "final_database",
                final_database,
                self._gene_id_databases(),
                "an external database of the loaded graph that attaches its ids to genes",
            )
        return self._run(
            "is_final_external_conversion_robust",
            lambda: self._final_conversions(
                convert_using_release, final_database, to_release, progress, from_fraction, strategy
            ),
        )

    def _final_conversions(
        self,
        convert_using_release: bool,
        final_database: str | None,
        to_release: int | None,
        progress: bool,
        from_fraction: float,
        strategy: str,
    ) -> tuple[bool, dict[str, Any]]:
        serving = self.graph.serving_assembly
        forms = self.graph.external_database_connection_form
        candidates = sorted(
            (name, assembly, held)
            for (name, assembly), releases in self.graph.available_releases_given_database_assembly.items()
            for held in releases
            if forms.get(name) == ids.BACKBONE_FORM
            and not name.startswith(ids.SYNONYM_PREFIX)
            and serving.get(held) == assembly
            and (final_database is None or name == final_database)
            and (to_release is None or held == to_release)
        )
        if not candidates:
            raise UnknownDatabaseError(
                f"No external database that attaches ids to genes was found for final_database={final_database!r} "
                f"and to_release={to_release!r}: the check needs such a database at a release at which it has ids.",
                argument="final_database",
                value=final_database,
            )
        database, assembly, release = random.choice(candidates)  # noqa: S311
        if progress:
            self.log.info(f"Assembly: {assembly}, Database: {database}, Release: {release}")

        manager = self._manager(release, assembly).change_form(ids.BACKBONE_FORM)
        table = manager.get_db("external" if self.graph.all_external else "external_relevant")
        table = table[table["name_db"] == database]
        # an entry the build left out by decision 73's rule is no id the gene can convert to
        several = self.graph.entries_of_several_genes
        left_out = several if several and database in manager.one_gene_only_names() else frozenset()
        expected: dict[str, set[str]] = {}
        for gene, identifier in zip(table["graph_id"].tolist(), table["id_db"].tolist()):
            if str(identifier).lower() not in left_out:
                expected.setdefault(gene, set()).add(identifier)

        run = self.history_travel_testing(
            from_fraction=from_fraction,
            from_release=release,
            from_assembly=assembly,
            from_database=organisms.ASSEMBLY_LABELS[assembly][ids.BACKBONE_FORM],
            to_release=release,
            final_database=database,
            allow_bridges=True,
            strategy=strategy,
            convert_using_release=convert_using_release,
            progress=progress,
            progress_details=False,
        )

        raised = {identifier for kind in ("history_voyage_failed_gracefully", "history_voyage_failed_unknown")
                  for identifier, _ in run[kind]}  # fmt: skip
        different, unexpected, lost, lost_with_id, failed = [], [], [], [], []
        for gene in run["ids"]["from"]:
            if gene in raised:
                failed.append(gene)
            elif gene not in run["conversion"]:
                lost.append(gene)
                if gene in expected:
                    lost_with_id.append(gene)
            elif gene not in expected:
                unexpected.append({"id": gene, "converted": sorted(run["conversion"][gene])})
            elif {i.lower() for i in run["conversion"][gene]} != {i.lower() for i in expected[gene]}:
                different.append(
                    {"id": gene, "converted": sorted(run["conversion"][gene]), "expected": sorted(expected[gene])}
                )
        details = {
            "database": database,
            "assembly": assembly,
            "release": release,
            "compared": len(run["ids"]["from"]),
            "n_different": len(different),
            "different": self._examples(different),
            "n_unexpected": len(unexpected),
            "unexpected": self._examples(unexpected),
            "n_lost_with_id": len(lost_with_id),
            "lost_with_id": self._examples(lost_with_id),
            "n_failed": len(failed),
            "failed": self._examples(failed),
            "lost": lost,
            "run": run,
        }
        return not (different or unexpected or lost_with_id or failed), details

    def random_dataset_source_generator(
        self,
        assembly: int,
        include_external: bool,
        include_ensembl: bool,
        for_final_database: bool,
        only_backbone_tests: bool,
        release_lower_limit: int | None = None,
        form: str | None = None,
    ) -> tuple[str, int, int] | None:
        """Draw a database of a genome assembly and a release at which the graph has it.

        The databases to draw from are chosen by the switches; of them only those that the graph has on the
        assembly at some release (at or after *release_lower_limit*, when that is given) can be drawn, and the
        release is one of theirs.

        Args:
            assembly (int): The genome assembly.
            include_external (bool): External databases of the assembly can be drawn (synonyms excepted).
            include_ensembl (bool): Ensembl ids can be drawn: the gene ids of the backbone with and without version
                (``ensembl_gene``, ``base_ensembl_gene``) and, unless *for_final_database*, the ids of the assembly
                (``assembly_<key>_ensembl_<form>``).
            for_final_database (bool): The draw is a target: the ids of one assembly are no target of a conversion.
            only_backbone_tests (bool): Draw ``ensembl_gene`` whatever the other switches say.
            release_lower_limit (int | None): The earliest release that can be drawn.
            form (str | None): Only databases that attach their ids to this form (``gene``, ``transcript``,
                ``translation``), and the Ensembl ids of this form.

        Returns:
            tuple[str, int, int] | None: ``(database, assembly, release)``; ``None`` when none of the databases to
            draw from has a release that can be drawn.

        Raises:
            InvalidArgumentError: If the switches leave no database to draw from (neither external databases nor Ensembl ids,
                and not the backbone alone).
        """
        if assembly not in self.graph.available_assemblies:
            raise UnknownAssemblyError(
                f"Genome assembly {assembly!r} is not one of the graph: {sorted(self.graph.available_assemblies)}.",
                argument="assembly",
                value=assembly,
                allowed=sorted(self.graph.available_assemblies),
            )
        if form is not None and form not in ids.FORMS:
            raise InvalidArgumentError(
                f"`form`={form!r} is not an Ensembl form; valid: {list(ids.FORMS)}.", argument="form", allowed=ids.FORMS
            )
        if only_backbone_tests:
            sources = [ids.ENSEMBL_TYPE[ids.BACKBONE_FORM]]
        elif not include_external and not include_ensembl:
            raise InvalidArgumentError("The switches leave no database to draw from.")
        else:
            sources = []
            if include_external:
                sources.extend(
                    name
                    for name in self.graph.available_external_databases_assembly.get(assembly, ())
                    if not name.startswith(ids.SYNONYM_PREFIX)
                    and (form is None or self.graph.external_database_connection_form[name] == form)
                )
            if include_ensembl and not for_final_database:
                of_assembly = organisms.ASSEMBLY_LABELS[assembly]
                sources.extend(of_assembly.values() if form is None else [of_assembly[form]])
            if include_ensembl and (form == ids.BACKBONE_FORM or form is None):
                sources.extend([ids.ENSEMBL_TYPE[ids.BACKBONE_FORM], ids.BASE_ENSEMBL_TYPE[ids.BACKBONE_FORM]])

        table = self.graph.available_releases_given_database_assembly
        releases_of: dict[str, list[int]] = {}
        for name in sources:
            releases = sorted(
                release
                for release in table.get((name, assembly), ())
                if release_lower_limit is None or release >= release_lower_limit
            )
            if releases:
                releases_of[name] = releases
        if not releases_of:
            return None
        selected_database = random.choice(sorted(releases_of))  # noqa: S311
        selected_release = random.choice(releases_of[selected_database])  # noqa: S311
        return selected_database, assembly, selected_release

    def _format_history_travel_testing_report_header(self, p: dict[str, Any]) -> list[str]:
        """Format the header section for a history travel testing report.

        Args:
            p (dict[str, Any]): Parameters dictionary containing ``from_database``,
                ``from_assembly``, ``from_release``, ``final_database``, and ``to_release``.

        Returns:
            list[str]: Lines of formatted text for the report header.
        """
        header = [
            "╔═ History-Travel-Testing Report ═╗",
            f"Source  : {p.get('from_database')} "
            f"(Assembly {p.get('from_assembly')}, Release {p.get('from_release')})",
            f"Target  : {p.get('final_database')} " f"(Release {p.get('to_release')})",
        ]

        return header

    def _format_history_travel_testing_report(
        self, res: dict[str, Any], include_header=False, line_separation_at_end=True
    ) -> list[str]:
        """Format a complete history travel testing report with metrics.

        Args:
            res (dict[str, Any]): Results dictionary from history_travel_testing
                containing conversion metrics and parameters.
            include_header (bool): If ``True``, include the header section with
                source/target information. Defaults to ``False``.
            line_separation_at_end (bool): If ``True``, append a blank line
                separator at the end of the report. Defaults to ``True``.

        Returns:
            list[str]: Lines of formatted text for the complete report.
        """

        def cnt(key: str) -> int:
            """Return length of list/dict at res[key] or zero if absent."""
            return len(res.get(key, []))

        def block(title: str, rows: list[tuple]) -> list[str]:
            pad = max(len(k) for k, _ in rows)
            return [f"{title}:"] + [f"  - {k.ljust(pad)} : {v:,}" for k, v in rows]

        if include_header:
            p = res.get("parameters", {})
            header = self._format_history_travel_testing_report_header(p)
            header_extension = [
                f"External: {p.get('allow_bridges')}   " f"Strategy: {p.get('strategy')}",
                f"Sample  : {p.get('from_fraction'):g} of source IDs",
            ]
            header.extend(header_extension)
        else:
            header = []

        failure_rows = [
            ("Voyage failed (graceful)      ", cnt("history_voyage_failed_gracefully")),
            ("Voyage failed (unknown)       ", cnt("history_voyage_failed_unknown")),
            ("Lost item                     ", cnt("lost_item")),
            ("Lost item, but ID exists      ", cnt("lost_item_but_the_same_id_exists")),
            ("Found IDs not accurate        ", cnt("found_ids_not_accurate")),
        ]

        clashes = res.get("clashing_id_type", [0, 0, 0])
        mapping_rows = [
            ("One→one IDs                   ", cnt("one_to_one_ids")),
            ("One→many IDs                  ", cnt("one_to_multiple_ids")),
            ("One→many (single conv.)       ", cnt("one_to_multiple_final_conversion")),
            ("Successfully converted IDs    ", cnt("conversion")),
            ("Clash one→one                 ", clashes[0]),
            ("Clash many→many               ", clashes[1]),
            ("Clash mixed                   ", clashes[2]),
        ]

        report_lines: list[str] = (
            header
            + block("Failure / Anomaly Counts", failure_rows)
            + block("Mapping Statistics", mapping_rows)
            + [f"Total runtime: {res.get('time', 0):.2f} s"]
        )

        if line_separation_at_end:
            report_lines.append("")

        return report_lines
