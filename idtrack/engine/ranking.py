"""The ranking of the candidates of a conversion.

The order of every (gene, target) pair, the best answer, and the criterion that decided between the best answer and
the runner-up.

Pure functions of what the search found; nothing here reads the graph except through the annotation counts that the
caller passes as a function.

A pair is ordered by its key (:py:class:`Pair`): being the queried gene id's own version at the target release;
then how its entry gene compares with the other entry genes (the *lineage*: final confidence, bridges, gap of the
bridges, the query being the entry gene's display name at the starting release, support of the query on the entry
gene; where the history merges several entry genes into the gene, the best display rank and support of them all),
then how its gene compares with the other successors of the same entry gene (*within*: steps without a score,
mapping score, being the query itself), then steps and ids.

Of the evidence of which gene a name stands for at a release, Ensembl's display name of a gene (``gene.display_xref_id``)
is Ensembl's own statement of the gene's name there; support counts the databases that attach the name to a gene, of
which a projection or a symbol that NCBI attached to another gene's translation is one like any other. So among the
genes that carry a name at one release the gene whose display name it is ranks first, and support decides between genes
that are equal in it (two genes that both display the name, or neither). The display name holds for the string as it
is typed: letter case is evidence where the string has mixed case (``Foxl1`` is not the display name
``FOXL1`` of another gene); a string in one letter case carries none, and in an organism whose symbols are written in
mixed case the display name of a name authority, read without letter case, ranks before another source's (mouse
``GAPDH`` is the gene that MGI displays as ``Gapdh`` before the gene that UniProt displays as ``GAPDH``). A name
authority's symbol against another
gene's synonym is decided before both, when the string is read (every name of a release before any
synonym, :py:meth:`idtrack.graph.TheGraph.resolve_name`).

The best answer: the pairs of the queried gene id's own version when the search reached it, else all pairs; of
those, the entry genes that are equal in the lineage evidence all belong to the answer. The mapping score compares
the successors of one entry gene (it says how much of that gene a successor keeps, not which of two genes a name
stands for), then being the query itself, and between successors that are still equal the richness of annotation
decides, as the last resort. What remains is the answer; more than one pair is a tie, given in ascending order of
gene id and target id.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

#: The criteria of the lineage, in the order of :py:attr:`Pair.lineage` (named as
#: :py:data:`idtrack.results.CRITERIA` names them).
LINEAGE_CRITERIA: tuple[str, ...] = ("final_confidence", "bridges", "bridge_gap", "display_name", "entry_support")


class Pair(NamedTuple):
    """One (gene, target) pair with the parts of its ranking key.

    The tuple order is the ranking order: ``order`` first (which holds the other parts and is unique per pair), so
    two pairs compare as their keys do.
    """

    #: The full key: ``(own, *lineage, *within, external steps, history steps, gene, target)``.
    order: tuple
    #: ``(final confidence, bridges, bridge gap, display rank of the query on the entry gene, -entry support)``: the
    #: display rank is 0 where the query is the entry gene's display name as the string is read, 1 where it is another
    #: source's display name read without letter case, 2 where it is not its display name
    #: (:py:attr:`idtrack.results.Candidate.display_rank`).
    lineage: tuple
    #: ``(history steps between two ids without a score, -score rounded to 12 places, 0 if the target is the query
    #: itself else 1)``: a step that Ensembl scored 0 (no score) ranks a path below every measured score.
    within: tuple
    #: The gene at the target release.
    gene: str
    #: The target id (the graph's node name; spelt as the database wrote it only in results).
    target: str


def make_pair(
    own: bool,
    confidence: int | None,
    bridges: int,
    gap: int,
    support: int,
    unscored: int,
    score: float,
    same_as_query: bool,
    external_steps: int,
    history_steps: int,
    gene: str,
    target: str,
    display_rank: int = 2,
) -> Pair:
    """Return the ranking pair of a candidate.

    Args:
        own: The gene is the queried gene id's own version at the target release.
        confidence: The final confidence of the gene (0 or 1; ``None`` is never compared, all genes of a no-target
            result share it).
        bridges: The bridges of the gene's path.
        gap: The bridge gap of the gene's path.
        support: The support of the query on the gene's entry genes (the best of them).
        unscored: The history steps of the path between two different ids without a score (Ensembl's 0).
        score: The path score (the product of the measured scores).
        same_as_query: The target is the query itself.
        external_steps: Cross-reference steps of the path.
        history_steps: History steps of the path.
        gene: The gene.
        target: The target.
        display_rank: How the query is the display name of the gene's entry genes at the starting release (the best)
            (:py:attr:`idtrack.results.Candidate.display_rank`: 0 first, 2 not at all).

    Returns:
        Pair: The pair.
    """
    lineage = (confidence, bridges, gap, display_rank, -support)
    within = (unscored, -round(score, 12), 0 if same_as_query else 1)
    rest = (external_steps, history_steps, gene, target)
    return Pair((0 if own else 1,) + lineage + within + rest, lineage, within, gene, target)


def select_best(
    pairs: list[Pair], own: set[str], entry_of: dict[str, str], annotation: Callable[[str], list[int]]
) -> tuple[list[Pair], dict[str, list[int]]]:
    """Return the best pairs and the annotation counts that were compared.

    Args:
        pairs: Every pair, sorted.
        own: The genes that are the queried gene id's own version at the target release.
        entry_of: Gene to its entry gene.
        annotation: Gene to its annotation counts (``[-external ids, -transcripts, -translations]``).

    Returns:
        tuple[list[Pair], dict[str, list[int]]]: The best pairs, sorted, and the counts per gene compared.
    """
    best: list[Pair] = []
    by_entry: dict[str, list[Pair]] = {}
    candidates = [pair for pair in pairs if pair.gene in own] or pairs  # the id's own version is the best answer
    for pair in candidates:
        if pair.lineage == candidates[0].lineage:
            by_entry.setdefault(entry_of[pair.gene], []).append(pair)
    counts: dict[str, list[int]] = {}
    for entry_gene in sorted(by_entry):
        successors = [pair for pair in by_entry[entry_gene] if pair.within == by_entry[entry_gene][0].within]
        equal_genes = sorted({pair.gene for pair in successors})
        if len(equal_genes) > 1:
            for gene in equal_genes:
                counts[gene] = annotation(gene)
            richest = min(counts[gene] for gene in equal_genes)
            successors = [pair for pair in successors if counts[pair.gene] == richest]
        best.extend(successors)
    best.sort()
    return best, counts


def best_order(gene: str, target: str) -> tuple[str, str]:
    """Return the key that orders the pairs of the best answer: gene id, then target id, ascending.

    Args:
        gene: The gene at the target release.
        target: The target id.

    Returns:
        tuple[str, str]: The key.
    """
    return (gene, target)


def ranked(pairs: list[Pair], best: list[Pair]) -> list[Pair]:
    """Return every pair in the order of results: the best pairs first, the others by their key, grouped by gene.

    The best pairs are equal in the evidence that decides; what still tells them apart (steps, the mapping score of
    successors of different entry genes) did not decide, so a tie is given in ascending order of gene id and target id
    (:py:func:`best_order`), a stated order that does not depend on how each was reached. The pairs of one gene stay
    together, in the order in which the gene first appears (a result lists each gene's targets together).

    Args:
        pairs: Every pair, sorted.
        best: The best pairs, sorted.

    Returns:
        list[Pair]: The ranked pairs.
    """
    leading = set(best)
    order = sorted(best, key=lambda pair: best_order(pair.gene, pair.target)) + [
        pair for pair in pairs if pair not in leading
    ]
    by_gene: dict[str, list[Pair]] = {}
    for pair in order:
        by_gene.setdefault(pair.gene, []).append(pair)
    return [pair for gene_pairs in by_gene.values() for pair in gene_pairs]


def decided_by(
    ordered: list[Pair], best: list[Pair], own: set[str], entry_of: dict[str, str], counts: dict[str, list[int]]
) -> str | None:
    """Name the criterion that put the best answer before the runner-up (the first pair that is not best).

    Args:
        ordered: Every pair in the order of results (:py:func:`ranked`).
        best: The best pairs.
        own: The genes that are the queried gene id's own version.
        entry_of: Gene to its entry gene.
        counts: The annotation counts compared (:py:func:`select_best`).

    Returns:
        str | None: ``own_version``, ``final_confidence``, ``bridges``, ``bridge_gap``, ``display_name``,
        ``entry_support``, ``score``, ``same_as_query`` or ``annotation``; ``None`` when every pair is best.
    """
    leading = set(best)
    runner_up = next((pair for pair in ordered if pair not in leading), None)
    if runner_up is None:
        return None
    first = best[0]
    if own and (first.gene in own) != (runner_up.gene in own):
        return "own_version"
    for name, a, b in zip(LINEAGE_CRITERIA, first.lineage, runner_up.lineage):
        if a != b:
            return name
    # Same lineage: the runner-up is a successor of its entry gene that lost against that entry gene's best.
    entry = entry_of[runner_up.gene]
    rival = next((pair for pair in best if entry_of[pair.gene] == entry), first)
    for name, a, b in zip(("score", "score", "same_as_query"), rival.within, runner_up.within):
        if a != b:
            return name
    if counts.get(runner_up.gene) is not None:
        return "annotation"
    return None
