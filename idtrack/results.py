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
- :py:class:`ExplainedPath`: the path of every candidate as data, one graph per conversion (nodes and steps),
  with every bridge the search considered and the rule that took or refused it (:py:class:`BridgeCheck`).
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

#: Why a conversion has no target in the requested database, or answers another question than the one asked
#: (:py:attr:`ConversionResult.reason`).
REASONS: dict[str, str] = {
    "not_in_graph": "The identifier is not an identifier of this graph at any release.",
    "unknown_version": (
        "The identifier is an Ensembl id, or a sequence accession, with a version this graph does not hold."
    ),
    "version_not_judged": (
        "The version written after the accession was not judged: Ensembl's records hold only an edit counter of the "
        "records of its database, not a version of the sequence; the accession was read without it."
    ),
    "not_an_id_of_database": "The identifier is an Ensembl id, not an id of the database named as its source.",
    "never_active": "The identifier is never active on the assemblies that serve the releases of this graph.",
    "no_gene_at_start": "At the release it is read at, the identifier is attached to no gene.",
    "no_path_to_target": "Ensembl's history (and the bridges allowed) leads to no gene at the target release.",
    "no_id_in_final_database": "The genes reached have no id in the requested database.",
    "error": "The conversion of this identifier raised an error (see `error`).",
    "read_at_another_release": (
        "The graph does not hold the identifier at the release given (`from_release_given`); it was read where the "
        "graph holds it (`from_release`), so the targets are what the identifier meant there, which can be another "
        "gene than the one it meant at the release given."
    ),
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
    "mitochondrial_genome": (
        "Refused: Ensembl's history gives the id of a gene of the mitochondrial genome to a gene of a nuclear sequence, "
        "or the reverse; that is no lineage, and the walk does not follow it."
    ),
}

#: Where two genes lie relative to each other (:py:attr:`BridgeCheck.places` and :py:attr:`PathStep.places`), from
#: the places the graph holds (:py:meth:`idtrack.TheGraph.places`). Two places are compared base by base on one
#: assembly and on its reference sequence (they overlap when they share a base on one sequence region and one strand),
#: and across an assembly change where Ensembl's mapping of the two assemblies carries the earlier place onto the later
#: assembly; a copy on a patch or haplotype is compared by the reference region its assembly exception replaces; across
#: assemblies otherwise, two genes that lie on chromosomes or the mitochondrial genome only, on sequences of different
#: names, lie apart (:py:func:`idtrack.graph.genes.relation`).
PLACE_RELATIONS: dict[str, str] = {
    "same_place": (
        "On one assembly the two genes overlap (one sequence region, one strand), or a place carried onto the other's "
        "assembly by Ensembl's mapping of the two assemblies overlaps it."
    ),
    "places_apart": (
        "On one assembly the two genes do not overlap: another sequence region, another strand, or apart; or a place "
        "carried onto the other's assembly by Ensembl's mapping of the two assemblies shares no base with it; or a "
        "copy's reference region (Ensembl's assembly exception) shares no base with the other gene; or, on two "
        "assemblies that no published mapping joins, they lie on chromosomes of different names (the mitochondrial "
        "genome against a chromosome among them), which no assembly change turns one into the other."
    ),
    "other_assemblies": (
        "The two genes never lie on one assembly at the same time, and nothing tells them apart (a sequence of one "
        "name, or a scaffold): their places cannot be compared."
    ),
    "outside_reference": (
        "A gene lies outside the reference sequence (a patch, a haplotype, an alternative locus, an LRG record), and "
        "nothing places it apart from the other gene (it lies inside its assembly exception's reference region, or it "
        "is an LRG record, which has none): its place is not compared base by base with the reference sequence."
    ),
    "no_place": "The graph holds no place for one of the two genes.",
}

#: Why a bridge between two genes was taken or refused (:py:attr:`BridgeCheck.through`). A bridge is evidence that one
#: gene became the other only when the id leaves one gene for the other at the same place.
BRIDGE_RULES: dict[str, str] = {
    "same_place": (
        "Taken: the two genes overlap on one assembly, or Ensembl's mapping of their two assemblies carries one onto "
        "the other."
    ),
    "never_shared": (
        "Taken: the places of the two genes cannot be compared (two assemblies, a sequence of one name or a scaffold), "
        "and no release gives the id to both: the id left one gene for the other."
    ),
    "outside_reference": (
        "Taken: a gene lies outside the reference sequence (a copy on a patch, a haplotype, an alternative locus, an "
        "LRG record), where places are not compared, both genes carried the id at one release, and at every such "
        "release the genes that carried it are one gene: the copy and the gene it stands for."
    ),
    "synonym": "Refused: a synonym is no bridge (it names other genes too often to stand for one).",
    "places_apart": (
        "Refused: on one assembly the two genes do not overlap, or Ensembl's mapping of their two assemblies carries "
        "one elsewhere, or a copy's reference region does not hold the other gene, or on two assemblies that no "
        "published mapping joins they lie on chromosomes of different names."
    ),
    "shared_at_one_release": (
        "Refused: the graph holds no place of a gene (a graph made in memory) and both genes carry the id at one "
        "release, so it names two living genes, not one gene that became the other."
    ),
    "copy_id_of_several_genes": (
        "Refused: a gene lies outside the reference sequence, where places are not compared, and at a release at which "
        "both genes carried the id it named several genes (a family label, a shared alias): it does not say which "
        "gene the copy stands for."
    ),
    "mitochondrial_genome": (
        "Refused: the two genes share a stable id that Ensembl gave first to a gene of the mitochondrial genome and then "
        "to a gene of a nuclear sequence (or the reverse); they are two genes, and no id makes one the other (the "
        "history step between them is refused alike)."
    ),
    "copy_id_elsewhere": (
        "Refused: a gene lies outside the reference sequence, and no release gave the id to both genes: a copy stands "
        "for the gene that carries its names while it exists, not for a gene the id reached at another time."
    ),
}

#: The rules of :py:data:`BRIDGE_RULES` under which a bridge is taken.
BRIDGE_RULES_TAKEN: frozenset[str] = frozenset({"same_place", "never_shared", "outside_reference"})

#: The criteria that order candidates, in the order in which they decide (:py:attr:`ConversionResult.decided_by`).
CRITERIA: tuple[str, ...] = (
    "own_version",
    "final_confidence",
    "bridges",
    "bridge_gap",
    "display_name",
    "entry_support",
    "score",
    "same_as_query",
    "annotation",
)


#: How an identifier matched the graph node it was read as (:py:attr:`ConversionResult.match`): the steps of name
#: resolution (:py:meth:`idtrack.graph.TheGraph.resolve_name`). A match is one step, after the prefixes of
#: :py:data:`MATCH_PREFIXES` that apply, each followed by ``:`` (``"unicode:punctuation:synonym:exact"``).
MATCH_STEPS: dict[str, str] = {
    "exact": "The identifier as written (for an external identifier: as a database wrote it at the release read).",
    "case": "The identifier when letter case is ignored.",
    "separator": "The identifier when '-', '_' and '.' are taken as one character (and letter case is ignored).",
    "version": (
        "A sequence accession with the version written after it, read where Ensembl's cross references attached that "
        "version."
    ),
    "version_newer_than_graph": (
        "An Ensembl id or a sequence accession with a version newer than every version the graph holds, read as its "
        "latest version (an id active, an accession attached, at the graph's last release)."
    ),
    "version_not_judged": (
        "A sequence accession with a version that is not judged: Ensembl's records hold only an edit counter of the "
        "records of its database (UniProt's entry version), so the accession is read without it."
    ),
}

#: What else a match says, as prefixes of :py:attr:`ConversionResult.match` in this order.
MATCH_PREFIXES: dict[str, str] = {
    "copy": (
        "In a batch: a name made unique by a single-cell tool (another item of the batch with -1, .1 or _1 appended, "
        "itself no exact name at the release it is read at alone); the result is that item's (decision 82)."
    ),
    "unicode": (
        "Unicode was normalised before matching: invisible and control characters (zero-width characters, byte-order "
        "marks, bidirectional marks, soft hyphens) removed, compatibility forms (NFKC: full-width letters and digits) "
        "and Unicode dashes read as their plain characters (decision 84)."
    ),
    "punctuation": (
        "Stray punctuation was removed before matching: quotes around the identifier, a trailing semicolon, comma or "
        "full stop (decision 83)."
    ),
    "par_y": (
        "GENCODE's form of the Y copy of a pseudoautosomal gene (<id>_PAR_Y) was read as the gene's id; the Y copy is "
        "reported apart (decision 93)."
    ),
    "lrg": "An LRG record's id, answered with the gene the record describes (decision 96).",
    "synonym": "A synonym that a database records for the gene, not the name the database gives it.",
}

#: Why a reading of an identifier is reported beside the one taken (:py:attr:`OtherReading.reason`).
READING_REASONS: dict[str, str] = {
    "symbol_without_case": (
        "In an organism whose symbols follow a letter-case convention, a string written otherwise that is not exactly "
        "a name is read first as a symbol of the organism without letter case, where that names exactly one gene "
        "(decisions 55 and 80); this reading came after it."
    ),
    "make_unique_suffix": (
        "The string ends in a suffix that single-cell tools append to make names unique (-1, .1, _1); reading it, "
        "with its separators taken as one or without letter case, as a synonym of another gene than the string before "
        "the suffix names is not taken (decision 82)."
    ),
    "other_release": (
        "The string means this identifier at other releases only (the releases listed), not at the release it was read "
        "at: what it meant at another time (a name of one gene then, a synonym of another gene now). Give "
        "from_release to read it there."
    ),
    "unversioned_database": (
        "The string is an identifier followed by a version, and the identifier is, at the release read, also an id of a "
        "database that writes its ids without a version (an NCBI gene id spelt like a numeric Havana id of the early "
        "releases): where no database has the identifier with that version at that release, the string has two "
        "readings, and this is the other one (the identifier as that database's id, without the suffix)."
    ),
    "same_step": (
        "The step that matched reached this identifier as well, at the same release (decision 87); its candidates are "
        "part of the answer, ranked with those of the reading taken by the same evidence of lineage."
    ),
}

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
        entry_databases (tuple[str, ...]): The databases that attach the identifier to the entry gene at the starting
            release (for an Ensembl id, the Ensembl relation that reads it: ``base_ensembl_gene``, ...); a synonym's
            database by its own name.
        authority (bool): A name authority gives the identifier to the entry gene as its symbol at the starting
            release: HGNC, MGI or VGNC (the symbol databases of the organism) or NCBI's official symbol
            (``NCBI gene (formerly Entrezgene)``), not as a synonym (decision 94: such a symbol is read before any
            synonym of another gene, :py:meth:`idtrack.graph.TheGraph.resolve_name`).
        display_name (bool): The identifier is the display name of an entry gene at the starting release
            (``gene.display_xref_id``: Ensembl's own name of the gene there, whatever database it came from). Among
            the genes that carry a name at one release such a gene ranks first, before :py:attr:`entry_support`
            (:py:mod:`idtrack.engine.ranking`; ``decided_by`` is ``"display_name"`` where it decided). It holds for the
            string as typed (decision 102): where the string has letters of both cases, only a display name that its
            database spelt so; a string in one letter case carries no evidence of case and is read without it. Where
            the history merges several entry genes into :py:attr:`gene`, this and :py:attr:`display_rank` and
            :py:attr:`entry_support` are the best over all of them, whichever one :py:attr:`entry_gene` names (the
            path shown starts from one; decision 106).
        display_rank (int): How the identifier is the display name of an entry gene, as the ranking compares it: 0 the
            display name (for a string in one letter case in an organism whose symbols are written in mixed case, the
            display name of a name authority, read without letter case); 1 for such a string, the display name of
            another source; 2 not its display name (:py:attr:`display_name` is ``False`` only then).
        entry_support (int): The number of databases that attach the identifier to an entry gene at the starting
            release (1 for an Ensembl id). For an identifier of digits only, databases of other numbers (an NCBI gene
            id and a Havana number spelt alike) are not counted against each other: their genes tie (decision 86).
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
        unscored_steps (int): The history steps of the path between two different ids that Ensembl gave no score
            (it writes 0 where it measured nothing); a path with such a step ranks below every path whose scores are
            measured, before :py:attr:`score` compares them (decision 57).
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
    entry_databases: tuple[str, ...]
    authority: bool
    display_name: bool
    display_rank: int
    entry_support: int
    own_version: bool
    final_confidence: int | None
    bridges: int
    bridge_gap: int
    score: float
    unscored_steps: int
    same_as_query: bool
    external_steps: int
    history_steps: int
    annotation: tuple[int, ...] | None
    path: tuple[int, ...] = ()
    final_path: tuple[int, ...] = ()


@dataclass(frozen=True)
class GenePlace(_Result):
    """Where a gene lies at some releases: an assembly, a sequence region, its first and last base, a strand.

    Attributes:
        assembly (int): The assembly key of idtrack's organism table.
        region (str): The sequence region as Ensembl names it (a chromosome, a scaffold, a patch).
        start (int): The first base of the gene (1-based).
        end (int): The last base of the gene (inclusive).
        strand (int): ``1`` or ``-1``.
        releases (tuple[tuple[int, int], ...]): The inclusive release ranges at which the gene lies there.
        reference (bool): The region is part of the reference sequence (not a patch, a haplotype, an alternative locus
            or an LRG record).
    """

    assembly: int
    region: str
    start: int
    end: int
    strand: int
    releases: tuple[tuple[int, int], ...]
    reference: bool


@dataclass(frozen=True)
class BridgeCheck(_Result):
    """A bridge the search considered from one gene to another, and the rule that took or refused it.

    The search considers bridges only where Ensembl's history leads to no gene at the target release, from every gene
    the history reached, and only those it can place in time (the far side holds at a release between the near side's
    and the target release); every such pair of genes is listed once.

    Attributes:
        source (str): The gene the walk would leave.
        target (str): The gene the walk would enter.
        places (str): Where the two genes lie relative to each other: a key of :py:data:`PLACE_RELATIONS` (the nodes
            of the path hold both places).
        taken (bool): The rules allow at least one of the ids in :py:attr:`through` as a bridge (whether a path then
            uses it depends on the releases and on the other candidates).
        through (tuple[tuple[str, str], ...]): Every id the two genes carried in common, as ``(node, rule)`` with a
            key of :py:data:`BRIDGE_RULES`; the node is an external id or a transcript.
    """

    source: str
    target: str
    places: str
    taken: bool
    through: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PathNode(_Result):
    """A node of an explained path.

    Attributes:
        id (str): The node's id within the explanation: the identifier (a synonym written as it is, without the
            graph's bookkeeping; ``synonym:<text>`` only should the same text name another node of the explanation).
        kind (str): ``ensembl_gene``, ``base_ensembl_gene``, ``ensembl_transcript``, ``ensembl_translation``,
            ``external`` or ``synonym`` (an external id recorded as a synonym).
        label (str): The name to show for the node: the identifier.
        databases (tuple[str, ...]): For an external id, the databases that attach it to an Ensembl id at some
            release (for a synonym, the databases that record it); empty otherwise.
        active (tuple[tuple[int, int | None], ...]): For an Ensembl gene with version, the inclusive release ranges in
            which it exists (``None``: still active at the last release of the graph); empty otherwise.
        assemblies (tuple[int, ...]): For an Ensembl gene with version, the assemblies that serve its active releases.
        non_reference (tuple[tuple[int, int], ...]): For an Ensembl gene, the release ranges in which it lies outside
            the reference sequence (an alternative locus, a patch, an LRG record).
        places (tuple[GenePlace, ...]): For an Ensembl gene with version, where it lies at its releases (on the
            assembly that serves each release); empty otherwise.
    """

    id: str
    kind: str
    label: str
    databases: tuple[str, ...] = ()
    active: tuple[tuple[int, int | None], ...] = ()
    assemblies: tuple[int, ...] = ()
    non_reference: tuple[tuple[int, int], ...] = ()
    places: tuple[GenePlace, ...] = ()


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
            on the assembly that serves it (for a synonym: the databases that record it).
        assembly (int | None): The assembly that serves :py:attr:`release` (or, for a history step,
            :py:attr:`new_release`, the event's new side).
        old_assembly (int | None): For a history step: the assembly that serves :py:attr:`old_release`, the event's
            old side (it differs from :py:attr:`assembly` where the event crosses from one assembly to the next);
            ``None`` otherwise.
        reason (str): Why the step was taken (or, for a step of :py:attr:`ExplainedPath.refused`, why it was not): a
            key of :py:data:`STEP_REASONS`.
        places (str | None): For a history step between two genes: where the event's later gene lies relative to the
            earlier one (a key of :py:data:`PLACE_RELATIONS`; ``places_apart``: Ensembl's history moves the gene to
            another place, which the walk follows); ``None`` for other steps.
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
    places: str | None = None


@dataclass(frozen=True)
class ExplainedPath(_Result):
    """The paths of a conversion as one graph: every node and step once; candidates refer to steps by index.

    Attributes:
        nodes (tuple[PathNode, ...]): The nodes of every step and of every bridge considered, ordered by first use.
        steps (tuple[PathStep, ...]): The steps of every candidate's path, each once, ordered by first use.
        bridges (tuple[BridgeCheck, ...]): Every bridge the search considered, taken or refused, with its rule (empty
            when Ensembl's history alone reached the target release); ordered by source and target.
        refused (tuple[PathStep, ...]): The history steps of Ensembl the search did not follow, in the direction of the
            walk, each with the rule that refused it as its :py:attr:`PathStep.reason` (a step between the
            mitochondrial genome and a nuclear sequence); ordered by source, target and key.
    """

    nodes: tuple[PathNode, ...]
    steps: tuple[PathStep, ...]
    bridges: tuple[BridgeCheck, ...] = ()
    refused: tuple[PathStep, ...] = ()


@dataclass(frozen=True)
class OtherReading(_Result):
    """Another graph node an identifier was read as, beside :py:attr:`ConversionResult.graph_id`.

    Attributes:
        graph_id (str): The node (as :py:attr:`ConversionResult.graph_id` names a node).
        match (str): How the identifier matched it (:py:data:`MATCH_STEPS`, :py:data:`MATCH_PREFIXES`).
        reason (str): Why it is reported: a key of :py:data:`READING_REASONS`.
        genes (tuple[str, ...]): The genes of the best answer that a conversion of this reading gives (the same
            target release and arguments), ascending; empty when it gives none.
        releases (tuple[int, ...]): For a reading of reason ``other_release``, the releases at which the string has
            this meaning, ascending; empty otherwise.
    """

    graph_id: str
    match: str
    reason: str
    genes: tuple[str, ...] = ()
    releases: tuple[int, ...] = ()


@dataclass(frozen=True)
class ConversionResult(_Result):
    """One identifier converted to a target release and database (:py:meth:`idtrack.API.convert`).

    Attributes:
        query_id (str): The identifier as given.
        graph_id (str | None): The identifier of the graph the identifier was read as (``None``: not found). For an
            Ensembl transcript or translation id without version this is its version of the starting release. A
            synonym is written as it is (:py:attr:`match` says it is a synonym).
        match (str | None): How the identifier matched ``graph_id``: a step of :py:data:`MATCH_STEPS` after the
            prefixes of :py:data:`MATCH_PREFIXES` that apply (``"synonym:exact"``, ``"unicode:case"``); ``None`` when it
            is not in the graph.
        known_versions (tuple[str, ...]): For an Ensembl id, or a sequence accession, written with a version the
            graph does not hold: the versions that exist (the identifier is not answered with another version's gene).
        from_release (int | None): The release at which the identifier was read.
        from_release_inferred (bool): The release was chosen by the engine, not given.
        from_release_given (int | None): The release given.
        from_release_fallback (bool): The release given was set aside because the graph does not hold the identifier
            there; it was read where the graph holds it.
        from_databases (tuple[str, ...]): For an external id, the databases through which it was read at that release
            (for a synonym: the databases that record it).
        to_release (int): The target release.
        final_database (str | None): The database of the targets: the requested one, ``ensembl_gene`` when the genes
            reached have no id in it (``no_target``), ``None`` when nothing was found.
        candidates (tuple[Candidate, ...]): Every target reached, ranked, the best first and marked.
        tie (bool): The best answer is shared by several genes or targets equal in all evidence.
        decided_by (str | None): The criterion (:py:data:`CRITERIA`) that put the best answer before the runner-up;
            ``None`` without a runner-up.
        alternative_locus_genes (tuple[str, ...]): Copies on alternative loci and patches, LRG records, and the Y
            copies of pseudoautosomal genes (beside their gene on X), that the identifier also leads to beside genes
            of the reference sequence; not candidates.
        no_target (bool): Genes were reached, but none has an id in the requested database; the candidates are the
            genes.
        no_target_genes (tuple[str, ...]): Genes reached without an id in the requested database while others have
            one; not candidates.
        status (str): One of :py:data:`STATUSES`.
        reason (str | None): The first of :py:attr:`reasons`; ``None`` when none applies.
        reasons (tuple[str, ...]): Every reason that applies, keys of :py:data:`REASONS`, in this order: why the result
            has no target in the requested database (no candidates, or ``no_target``: genes stand in for the missing
            id); ``"read_at_another_release"`` where the identifier was read at another release than the one given
            (:py:attr:`from_release_fallback`); ``"version_not_judged"`` where a version written after an accession
            was not judged (decision 101). Empty for a result with targets read as asked.
        other_readings (tuple[OtherReading, ...]): The other nodes the identifier was read as: those a rule of name
            resolution put after :py:attr:`graph_id`, with what they would answer, those read by the same step,
            whose candidates are part of a tie, and those the string has at other releases only, with those releases
            (:py:meth:`idtrack.graph.TheGraph.readings`).
        error (ErrorInfo | None): In a batch, the error that stopped this identifier's conversion.
        path (ExplainedPath | None): With ``explain``: the paths of the candidates (no step when there is no
            candidate) and the bridges the search considered; ``None`` without ``explain``.
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
    reasons: tuple[str, ...] = ()
    other_readings: tuple[OtherReading, ...] = ()
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
        name (str): The name as the link of the gene writes it at these releases.
        gene (str): The Ensembl gene with version.
        database (str): The database (for a synonym: the database that records it).
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
class SourceInference(_Result):
    """Where a list of identifiers comes from (:py:meth:`idtrack.API.source_inference`).

    An origin is ``[database, assembly, release]``, ``release``, ``assembly`` or ``[assembly, release]`` by the mode.

    Attributes:
        mode (str): The granularity of an origin.
        band (tuple[Any, ...]): The origins the list cannot tell apart from the best one, best first.
        margin (int | None): The best origin's count minus the count of the best origin outside the band; ``None``
            when every origin is in the band.
        counts (tuple[tuple[Any, int], ...]): Every origin with the number of distinct identifiers found there, by
            descending count, then origin.
        identifiers (int): The distinct identifiers found in the graph.
        not_found (int): The distinct strings not in the graph, or refused as identifiers (missing values).
    """

    mode: str
    band: tuple[Any, ...]
    margin: int | None
    counts: tuple[tuple[Any, int], ...]
    identifiers: int
    not_found: int


@dataclass(frozen=True)
class AssemblyDescription(_Result):
    """One genome assembly of a loaded graph.

    Attributes:
        assembly (int): The assembly key of idtrack's organism table.
        name (str | None): Ensembl's name of the assembly.
        releases (tuple[int, ...]): The releases at which the graph holds data of it.
        serves (tuple[int, ...]): The releases it serves (is the assembly whose annotation counts).
        external_databases (tuple[str, ...]): The external databases of the graph on it.
        synonym_databases (tuple[str, ...]): Those of :py:attr:`external_databases` that record synonyms (read under
            the database's name: ``from_database`` covers them).
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
        synonym_databases (tuple[str, ...]): Those of :py:attr:`external_databases` that record synonyms.
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
    "BRIDGE_RULES",
    "BRIDGE_RULES_TAKEN",
    "CRITERIA",
    "MATCH_PREFIXES",
    "MATCH_STEPS",
    "PLACE_RELATIONS",
    "READING_REASONS",
    "REASONS",
    "STATUSES",
    "STEP_REASONS",
    "STRATEGIES",
    "AssemblyDescription",
    "BridgeCheck",
    "Candidate",
    "ConversionResult",
    "ErrorInfo",
    "ExplainedPath",
    "GenePlace",
    "GraphDescription",
    "NameMatch",
    "OtherReading",
    "PathNode",
    "PathStep",
    "SourceInference",
    "check_strategy",
]
