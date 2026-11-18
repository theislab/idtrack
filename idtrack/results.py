"""The typed results of idtrack's public calls: frozen dataclasses with a lossless, strict-JSON ``to_dict()``.

Every result is plain data: strings, integers, floats (never NaN or infinity: a missing number is ``None``),
booleans, ``None``, tuples and nested results. ``to_dict()`` turns a result into ``dict``/``list``/scalars that
``json.dumps(..., allow_nan=False)`` writes, and ``from_dict()`` reads it back into an equal result.

- :py:class:`ConversionResult`: one identifier converted (:py:meth:`idtrack.API.convert`). Its
  :py:attr:`~ConversionResult.candidates` are every target the search reached, ranked, the best marked
  (:py:attr:`Candidate.best`); ``best`` and ``all`` are two views of the one search
  (:py:meth:`ConversionResult.view`). :py:attr:`~ConversionResult.decided_by` names the criterion that put the best
  answer before the runner-up, and :py:attr:`~ConversionResult.status` with :py:attr:`~ConversionResult.reason`
  says why a result has no target.
- :py:class:`ExplainedPath`: the path of every candidate as data, one graph per conversion (nodes and steps).
- :py:class:`GraphDescription`: what a loaded graph holds, without the network.
- :py:class:`NameMatch`: a name of an external database attached to a gene at releases.

This module imports only :py:mod:`idtrack.errors`.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from dataclasses import dataclass, field
from typing import Any

from idtrack.errors import IntegrityError, InvalidArgumentError

#: The outcome of a conversion.
STATUSES: tuple[str, ...] = ("converted", "tie", "no_target", "not_found", "unknown_version", "no_conversion", "error")

#: Why a conversion has no target (:py:attr:`ConversionResult.reason`).
REASONS: dict[str, str] = {
    "not_in_graph": "The identifier is not an identifier of this graph at any release.",
    "unknown_version": "The identifier is an Ensembl id with a version this graph does not hold.",
    "not_an_id_of_database": "The identifier is an Ensembl id, not an id of the database named as its source.",
    "never_active": "The identifier is never active on the assemblies that serve the releases of this graph.",
    "no_gene_at_start": "At the release it is read at, the identifier is attached to no gene.",
    "no_path_to_target": "Ensembl's history (and the bridges allowed) leads to no gene at the target release.",
    "no_id_in_final_database": "The genes reached have no id in the requested database.",
    "error": "The conversion of this identifier raised an error (see `error`).",
}

#: Why a step of an explained path was taken (:py:attr:`PathStep.reason`).
STEP_REASONS: dict[str, str] = {
    "identity": "The gene exists at the target release.",
    "entry": "The identifier is attached to this gene at the release it is read at.",
    "new_version": "Ensembl's id history: a new version of the id.",
    "replaced": "Ensembl's id history: the id was replaced by another.",
    "split": "Ensembl's id history: a split.",
    "merge": "Ensembl's id history: a merge.",
    "split_and_merge": "Ensembl's id history: a split and a merge.",
    "self_loop": "Ensembl re-mapped the gene to itself with a score.",
    "bridge_near": (
        "The history reaches no gene at the target release: the walk leaves the gene through an id it carries."
    ),
    "bridge_far": "The walk enters another gene that carries the same id.",
    "final": "The gene carries the id in the final database at the target release.",
    "final_other_release": (
        "The id is attached at the target release to a transcript or translation the gene had at another release."
    ),
}

#: The criteria that order candidates, in the order in which they decide (:py:attr:`ConversionResult.decided_by`).
CRITERIA: tuple[str, ...] = (
    "own_version",
    "final_confidence",
    "bridges",
    "bridge_gap",
    "entry_support",
    "score",
    "same_as_query",
    "annotation",
)


#: The two views of the candidates of a conversion (:py:meth:`ConversionResult.view`): the best answer, and every
#: candidate ranked.
STRATEGIES: tuple[str, ...] = ("best", "all")


def check_strategy(strategy: Any) -> str:
    """Return a ``strategy`` argument, checked: one of :py:data:`STRATEGIES`.

    Raises:
        idtrack.errors.InvalidArgumentError: For anything else.
    """
    if strategy not in STRATEGIES:
        raise InvalidArgumentError(
            f"`strategy`={strategy!r} is not a strategy; valid: {list(STRATEGIES)}.",
            argument="strategy",
            value=strategy,
            allowed=STRATEGIES,
        )
    return strategy


def _plain(value: Any) -> Any:
    """Turn a field value into JSON data (tuples to lists, results to dicts); refuse what JSON cannot hold exactly."""
    if isinstance(value, _Result):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise IntegrityError(f"A result holds a mapping with keys that are not strings: {sorted(map(repr, value))}")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise IntegrityError(f"A result holds a number that is not finite: {value!r}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise IntegrityError(f"A result holds a value of type {type(value).__name__}: {value!r}")


def _read(annotation: Any, value: Any, scope: dict) -> Any:
    """Read a JSON value back into the type of a field (the inverse of :py:func:`_plain`)."""
    if isinstance(annotation, str):
        annotation = eval(annotation, scope)  # noqa: S307 - the annotations of this module's own dataclasses
    origin = typing.get_origin(annotation)
    if value is None:
        return None
    if origin in (typing.Union, types.UnionType):
        options = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _read(options[0], value, scope) if len(options) == 1 else value
    if origin is tuple:
        args = typing.get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_read(args[0], v, scope) for v in value)
        return tuple(_read(a, v, scope) for a, v in zip(args, value))
    if origin is dict:
        _, value_type = typing.get_args(annotation)
        return {k: _read(value_type, v, scope) for k, v in value.items()}
    if isinstance(annotation, type) and issubclass(annotation, _Result):
        return annotation.from_dict(value)
    return value


class _Result:
    """Common behaviour of the result dataclasses: strict JSON out and back."""

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain JSON data (``json.dumps(..., allow_nan=False)`` writes it).

        Returns:
            dict[str, Any]: One key per field, in the order of the fields.
        """
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Read a result back from :py:meth:`to_dict`'s data.

        Args:
            data: The data.

        Returns:
            The result, equal to the one that wrote the data.
        """
        hints = typing.get_type_hints(cls)
        scope = globals()
        # Every subclass is a dataclass, which the type of this mixin cannot say (as in `to_dict`).
        fields = dataclasses.fields(cls)  # type: ignore[arg-type]
        return cls(**{f.name: _read(hints[f.name], data[f.name], scope) for f in fields})


@dataclass(frozen=True)
class ErrorInfo(_Result):
    """An error that stopped the conversion of one identifier in a batch.

    Attributes:
        type (str): The class of the error.
        message (str): Its message.
        input_error (bool): ``True`` when the error is the caller's (:py:class:`idtrack.errors.InputError`), ``False``
            for any other (an exceeded budget, an internal error).
    """

    type: str
    message: str
    input_error: bool


@dataclass(frozen=True)
class Candidate(_Result):
    """One target of a conversion: a gene at the target release and its id in the requested database.

    The fields after :py:attr:`best` are the ranking record of the candidate: what the engine compared.

    Attributes:
        gene (str): The Ensembl gene (``<id>.<version>``) at the target release through which the target was reached.
        target (str): The id in the final database, as that database wrote it at the target release (the gene itself
            for ``ensembl_gene``).
        best (bool): Part of the best answer (several candidates are best in a tie).
        entry_gene (str): The gene the identifier was read as at the starting release, from which this gene was
            reached.
        entry_support (int): The number of databases that attach the identifier to the entry gene at the starting
            release (1 for an Ensembl id).
        own_version (bool): The gene is the queried gene id's own version at the target release (an id that exists
            at the target release is its own answer).
        final_confidence (int | None): 0: the target is attached to the gene at the target release; 1: attached to a
            transcript or translation that belonged to the gene at another release; ``None``: no gene has an id in
            the requested database and the gene itself is the target.
        bridges (int): The bridges through external ids (or shared transcripts) on the path; 0 is Ensembl's history
            alone.
        bridge_gap (int): The releases between leaving a gene and entering the next, summed over the bridges.
        score (float): The product of Ensembl's mapping scores along the history of the path (1.0 without a scored
            event).
        same_as_query (bool): The target is the identifier itself (or the gene id it was read as).
        external_steps (int): Cross-reference edges on the path.
        history_steps (int): History edges on the path.
        annotation (tuple[int, ...] | None): ``(-external ids, -transcripts, -translations)`` of the gene at the
            target release, where the engine compared it (between successors of one entry gene that are equal in
            everything else); ``None`` where it did not.
        path (tuple[int, ...]): With ``explain``: the indices of :py:attr:`ExplainedPath.steps` from the identifier
            to :py:attr:`gene`, in order; empty otherwise.
        final_path (tuple[int, ...]): With ``explain``: the indices of the steps from :py:attr:`gene` to
            :py:attr:`target` (empty when the gene is the target).
    """

    gene: str
    target: str
    best: bool
    entry_gene: str
    entry_support: int
    own_version: bool
    final_confidence: int | None
    bridges: int
    bridge_gap: int
    score: float
    same_as_query: bool
    external_steps: int
    history_steps: int
    annotation: tuple[int, ...] | None
    path: tuple[int, ...] = ()
    final_path: tuple[int, ...] = ()


@dataclass(frozen=True)
class PathNode(_Result):
    """A node of an explained path.

    Attributes:
        id (str): The node as the graph names it.
        kind (str): ``ensembl_gene``, ``base_ensembl_gene``, ``ensembl_transcript``, ``ensembl_translation``,
            ``external`` or ``synonym`` (an external id recorded as a synonym), or the graph's type of a node of an
            assembly-only form.
        label (str): The name to show for the node. This version shows the graph's name, the same as :py:attr:`id`
            (a synonym with the graph's prefix ``synonym_id::``).
        databases (tuple[str, ...]): For an external id, the databases that attach it to an Ensembl id at some
            release, as the graph names them; empty otherwise.
        active (tuple[tuple[int, int | None], ...]): For an Ensembl gene with version, the inclusive release ranges in
            which it exists (``None``: still active at the last release of the graph); empty otherwise.
        assemblies (tuple[int, ...]): For an Ensembl gene with version, the assemblies that serve its active releases.
        non_reference (tuple[tuple[int, int], ...]): For an Ensembl gene, the release ranges in which it lies outside
            the reference sequence (an alternative locus, a patch, an LRG record).
    """

    id: str
    kind: str
    label: str
    databases: tuple[str, ...] = ()
    active: tuple[tuple[int, int | None], ...] = ()
    assemblies: tuple[int, ...] = ()
    non_reference: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class PathStep(_Result):
    """A step of an explained path.

    Attributes:
        kind (str): ``identity`` (the gene is its own answer), ``entry`` (from the identifier to the gene it stands
            for), ``history`` (an event of Ensembl's id history), ``self_loop`` (a scored re-mapping of a gene to
            itself), ``bridge_near`` and ``bridge_far`` (the two sides of a bridge through an id two genes carry),
            ``final`` (from the gene to its id in the final database).
        source (str | None): The node the step leaves (``None`` for ``identity``).
        target (str): The node the step enters.
        key (int | None): The key of a history edge (Ensembl events between the same two ids); ``None`` otherwise.
        release (int | None): The release at which a cross-reference step holds (``None`` for a final step of
            confidence 1, which holds at another release).
        old_release (int | None): For a history step: the old side of Ensembl's event, the last release of the
            event's earlier id. Ensembl's events run forward in time, the walk may not: walking forward the earlier
            id is :py:attr:`source`, walking backward (to an earlier target release) it is :py:attr:`target`.
        new_release (int | None): For a history step: the new side of the event, the first release of the event's
            later id (:py:attr:`target` walking forward, :py:attr:`source` walking backward).
        weight (float | None): For a history step: Ensembl's mapping score (``None``: none given).
        splits (int): For a history step: the other ids that the event's earlier id passed to at
            :py:attr:`old_release` (a split), whatever the direction of the walk.
        merges (int): For a history step: the other ids that passed to the event's later id at
            :py:attr:`new_release` (a merge), whatever the direction of the walk.
        databases (tuple[str, ...]): For a cross-reference step: the databases that link the two nodes at the release
            on the assembly that serves it, as the graph names them (a synonym's database with the prefix
            ``synonym_id::``).
        assembly (int | None): The assembly that serves :py:attr:`release` (or, for a history step,
            :py:attr:`new_release`, the event's new side).
        old_assembly (int | None): For a history step: the assembly that serves :py:attr:`old_release`, the event's
            old side (it differs from :py:attr:`assembly` where the event crosses from one assembly to the next);
            ``None`` otherwise.
        reason (str): Why the step was taken: a key of :py:data:`STEP_REASONS`.
    """

    kind: str
    source: str | None
    target: str
    key: int | None = None
    release: int | None = None
    old_release: int | None = None
    new_release: int | None = None
    weight: float | None = None
    splits: int = 0
    merges: int = 0
    databases: tuple[str, ...] = ()
    assembly: int | None = None
    old_assembly: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class ExplainedPath(_Result):
    """The paths of a conversion as one graph: every node and step once; candidates refer to steps by index.

    Attributes:
        nodes (tuple[PathNode, ...]): The nodes of every step, ordered by first use.
        steps (tuple[PathStep, ...]): The steps of every candidate's path, each once, ordered by first use.
    """

    nodes: tuple[PathNode, ...]
    steps: tuple[PathStep, ...]


@dataclass(frozen=True)
class ConversionResult(_Result):
    """One identifier converted to a target release and database (:py:meth:`idtrack.API.convert`).

    Attributes:
        query_id (str): The identifier as given.
        graph_id (str | None): The graph node the identifier was read as (``None``: not found). For an Ensembl
            transcript or translation id without version this is its version of the starting release. A synonym is
            named as the graph names it, with the prefix ``synonym_id::``.
        match (str | None): How the identifier matched ``graph_id``: ``exact``, ``case``, ``separator``, ``version``
            (each also after ``synonym:`` for a synonym), or ``version_newer_than_graph``; ``None`` when it is not in
            the graph.
        known_versions (tuple[str, ...]): For an Ensembl id written with a version the graph does not hold: the
            versions that exist (the identifier is not answered with another version's gene).
        from_release (int | None): The release at which the identifier was read.
        from_release_inferred (bool): The release was chosen by the engine, not given.
        from_release_given (int | None): The release given.
        from_release_fallback (bool): The release given was set aside because the graph does not hold the identifier
            there; it was read where the graph holds it.
        from_databases (tuple[str, ...]): For an external id, the databases through which it was read at that release
            (a synonym's database with the graph's prefix ``synonym_id::``).
        to_release (int): The target release.
        final_database (str | None): The database of the targets: the requested one, ``ensembl_gene`` when the genes
            reached have no id in it (``no_target``), ``None`` when nothing was found.
        candidates (tuple[Candidate, ...]): Every target reached, ranked, the best first and marked.
        tie (bool): The best answer is shared by several genes or targets equal in all evidence.
        decided_by (str | None): The criterion (:py:data:`CRITERIA`) that put the best answer before the runner-up;
            ``None`` without a runner-up.
        alternative_locus_genes (tuple[str, ...]): Copies on alternative loci and patches, and LRG records, that the
            identifier also leads to beside genes of the reference sequence; not candidates.
        no_target (bool): Genes were reached, but none has an id in the requested database; the candidates are the
            genes.
        no_target_genes (tuple[str, ...]): Genes reached without an id in the requested database while others have
            one; not candidates.
        status (str): One of :py:data:`STATUSES`.
        reason (str | None): For a result without a target in the requested database (no candidates, or
            ``no_target``: genes stand in for the missing id), a key of :py:data:`REASONS`; ``None`` otherwise.
        error (ErrorInfo | None): In a batch, the error that stopped this identifier's conversion.
        path (ExplainedPath | None): With ``explain``: the paths of the candidates (no node and no step when there
            is no candidate); ``None`` without ``explain``.
    """

    query_id: str
    graph_id: str | None
    match: str | None
    known_versions: tuple[str, ...]
    from_release: int | None
    from_release_inferred: bool
    from_release_given: int | None
    from_release_fallback: bool
    from_databases: tuple[str, ...]
    to_release: int
    final_database: str | None
    candidates: tuple[Candidate, ...]
    tie: bool
    decided_by: str | None
    alternative_locus_genes: tuple[str, ...]
    no_target: bool
    no_target_genes: tuple[str, ...]
    status: str
    reason: str | None
    error: ErrorInfo | None = None
    path: ExplainedPath | None = None

    def view(self, strategy: str = "best") -> tuple[Candidate, ...]:
        """Return the candidates of a strategy: ``best`` (the best answer) or ``all`` (every candidate, ranked).

        Args:
            strategy: ``"best"`` or ``"all"``.

        Returns:
            tuple[Candidate, ...]: The candidates, in rank order.

        Raises:
            idtrack.errors.InvalidArgumentError: For another strategy.
        """
        if check_strategy(strategy) == "best":
            return tuple(c for c in self.candidates if c.best)
        return self.candidates

    def targets(self, strategy: str = "best") -> tuple[str, ...]:
        """Return the target ids of a strategy, each once, in rank order.

        Args:
            strategy: ``"best"`` or ``"all"``.

        Returns:
            tuple[str, ...]: The targets.
        """
        return tuple(dict.fromkeys(c.target for c in self.view(strategy)))

    @property
    def best(self) -> tuple[Candidate, ...]:
        """The candidates of the best answer (several in a tie)."""
        return self.view("best")


@dataclass(frozen=True)
class NameMatch(_Result):
    """A name of an external database attached to an Ensembl gene, and the releases at which it is.

    Attributes:
        name (str): The name as the graph holds it (a synonym with the graph's prefix ``synonym_id::``).
        gene (str): The Ensembl gene with version.
        database (str): The database as the graph names it (a synonym's database with the prefix ``synonym_id::``).
        synonym (bool): The name is one of the database's synonyms of the gene, not its name.
        releases (tuple[int, ...]): The releases at which the database attaches the name to the gene on the assembly
            that serves each, and the gene lies on the reference sequence, ascending.
    """

    name: str
    gene: str
    database: str
    synonym: bool
    releases: tuple[int, ...]


@dataclass(frozen=True)
class AssemblyDescription(_Result):
    """One genome assembly of a loaded graph.

    Attributes:
        assembly (int): The assembly key of idtrack's organism table.
        name (str | None): Ensembl's name of the assembly.
        releases (tuple[int, ...]): The releases at which the graph holds data of it.
        serves (tuple[int, ...]): The releases it serves (is the assembly whose annotation counts).
        external_databases (tuple[str, ...]): The external databases of the graph on it, as the graph names them
            (the databases of synonyms with the prefix ``synonym_id::``).
        synonym_databases (tuple[str, ...]): Those of :py:attr:`external_databases` that hold synonyms.
    """

    assembly: int
    name: str | None
    releases: tuple[int, ...]
    serves: tuple[int, ...]
    external_databases: tuple[str, ...]
    synonym_databases: tuple[str, ...]


@dataclass(frozen=True)
class GraphDescription(_Result):
    """What a loaded graph holds (:py:meth:`idtrack.API.describe`), read from the graph without the network.

    Attributes:
        organism (str): The organism.
        primary_assembly (int): The assembly of the graph's snapshot release.
        snapshot_release (int): The last release of the graph.
        releases (tuple[int, ...]): Every release the graph covers.
        assemblies (tuple[AssemblyDescription, ...]): The assemblies, by the organism table's priority.
        serving_assembly (dict[str, int]): Release (as a string, a JSON key) to the assembly that serves it.
        external_databases (tuple[str, ...]): The external databases of the graph (any assembly), as
            :py:meth:`idtrack.API.external_databases` names them.
        synonym_databases (tuple[str, ...]): Those of :py:attr:`external_databases` that hold synonyms.
        final_databases (tuple[str, ...]): The databases a conversion to the snapshot release can end in.
        forms (tuple[str, ...]): The Ensembl forms of the graph (``gene``, ``transcript``, ``translation``).
        format_version (int | None): The graph file's format version (``None`` for a graph made in memory).
        package_version (str | None): The idtrack version that built the graph.
        package_commit (str | None): The git commit of idtrack that built the graph, where it was known.
        content_digest (str | None): The digest of the graph's content.
        build_date (str | None): When the graph was built.
        counts (dict[str, int]): Counts of the build (nodes, edges, ...), as the graph file records them.
    """

    organism: str
    primary_assembly: int
    snapshot_release: int
    releases: tuple[int, ...]
    assemblies: tuple[AssemblyDescription, ...]
    serving_assembly: dict[str, int]
    external_databases: tuple[str, ...]
    synonym_databases: tuple[str, ...]
    final_databases: tuple[str, ...]
    forms: tuple[str, ...]
    format_version: int | None
    package_version: str | None
    package_commit: str | None
    content_digest: str | None
    build_date: str | None
    counts: dict[str, int] = field(default_factory=dict)


__all__ = [
    "CRITERIA",
    "REASONS",
    "STATUSES",
    "STEP_REASONS",
    "STRATEGIES",
    "AssemblyDescription",
    "Candidate",
    "ConversionResult",
    "ErrorInfo",
    "ExplainedPath",
    "GraphDescription",
    "NameMatch",
    "PathNode",
    "PathStep",
    "check_strategy",
]
