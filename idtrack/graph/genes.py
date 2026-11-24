"""What one gene is: the one rule of the package for telling whether gene ids name the same gene.

Ensembl gives a gene a new stable id more often than the gene changes: a model is replaced by one at the same
place, two models are merged, a gene is annotated again on an alternative locus, an assembly patch or an LRG record
beside the reference sequence. Several stable ids can therefore stand for one gene, and a rule that counts genes by
stable id counts these as several. The rules that ask how many genes an identifier names over the releases
(an entry of :py:data:`idtrack.build.database_manager.ONE_GENE_ONLY` links genes only where it names one gene; a
label of more genes than an identity can is a family name, :py:attr:`idtrack.TheGraph.label_families`), the rule of
bridges (an id is evidence of a lineage only where it leaves one gene for another at the same place) and the Y copies
of pseudoautosomal genes (:py:meth:`idtrack.TheGraph.y_copies`) read it here, from what the graph holds of Ensembl's
own tables (the count of genes at one release, :py:meth:`idtrack.TheGraph.genes_linked_at_once`, counts stable ids;
it says why):

- **places** (:py:meth:`idtrack.TheGraph.places`, from Ensembl's ``gene`` and ``seq_region`` tables): two genes are
  at one place when, on one assembly, a place of one shares a base with a place of the other on one sequence region
  and one strand (:py:func:`overlap`). Places are compared base by base on the reference sequence; a patch, a
  haplotype, an alternative locus or an LRG record lies on a sequence of its own, whose coordinates are not those of
  the reference sequence;
- **places carried across an assembly change** (:py:meth:`idtrack.TheGraph.carried_places`): where
  Ensembl publishes a mapping of an earlier assembly onto a later one (its chain files,
  :py:mod:`idtrack.build.assembly_chain`), a place on the earlier assembly is carried onto the later one by its aligned
  bases at build, and compared there by overlap as places on one assembly are; where Ensembl publishes none (pig
  Sscrofa10.2 to Sscrofa11.1, the macaques), places of two assemblies are compared by the names of their sequences
  only (below);
- **the region a patch or a haplotype stands for** (:py:meth:`idtrack.TheGraph.assembly_exceptions`, Ensembl's
  ``assembly_exception`` table of each release): a gene on a patch or a haplotype is a copy of, or an alternative to,
  a gene of that region of the reference sequence, so it and a gene whose places share no base with the region (on
  either strand) lie apart; inside the region the bases are not compared (the table maps the region, not each base);
- **what a sequence is** (:py:attr:`idtrack.graph.types.Place.kind`, from Ensembl's ``coord_system`` and
  ``seq_region_attrib`` tables): a chromosome (:py:data:`CHROMOSOME`), the mitochondrial genome
  (:py:data:`MITOCHONDRIAL`) or a sequence below them (:py:data:`SCAFFOLD`: a scaffold, a supercontig, a contig). The
  assemblies of one organism name its chromosomes alike from one to the next (by its karyotype), and an organism has
  one mitochondrial genome; so across assemblies, where no base can be compared, two genes whose places
  all lie on such whole sequences, and on no sequence of the same name (the mitochondrial genome compared by its
  kind), lie apart. A scaffold's name says nothing of where its bases lie in another assembly;
- **outside the reference sequence** (:py:meth:`idtrack.TheGraph.non_reference_releases`, Ensembl's ``non_ref``
  attribute of a sequence region): a gene that never lies on the reference sequence is a copy of, or an alternative
  to, a gene of the reference sequence, never a gene of its own;
- **Ensembl's history** (the stable-id events between two stable ids): an id that Ensembl's history carried to
  another id that follows it (no release at which both live) is that gene under another id. Two ids that live at one
  release are two genes unless their places overlap at that release, whatever history they share: a split of one id
  into two ids that then live side by side is Ensembl saying "two genes";
- **a region that repeats another**: a pseudoautosomal region of Y is the same sequence as the region of X it repeats,
  so a place in it is read on that region of X (:py:func:`idtrack.organisms.pseudoautosomal_repeats`, GRC's regions,
  equal to Ensembl's ``PAR`` assembly exceptions): the Y copy of a pseudoautosomal gene, which Ensembl gives an id of
  its own from release 110, lies at its gene's place.

The rule of bridges compares two genes by their places only (:py:func:`relation`, the keys of
:py:data:`idtrack.results.PLACE_RELATIONS`: the walk follows Ensembl's history itself), over every version of each
stable id; the explained path compares the two versions of a history step the same way
(:py:attr:`idtrack.results.PathStep.places`). The count of genes (:py:func:`count_genes`) counts what the graph shows
apart on one assembly and by the names of whole sequences, and no more: ids that nothing tells apart (holders on two
assemblies without a history link and without a comparable place) are not counted as two genes; it does not read the
places carried across an assembly change or the regions of the copies, which only :py:func:`relation` compares.

Copies are not grouped with their own gene of the reference sequence: Ensembl's ``alt_allele`` table says which gene
of the reference sequence a copy is an allele of, but the graph does not read it (Ensembl's core databases fill it for
human, for mouse at some releases and not at 115, and not for pig or the macaques at 115); a copy therefore adds no
gene to any count.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from idtrack import ids

if TYPE_CHECKING:
    from idtrack.graph.graph import TheGraph
    from idtrack.graph.types import Place

#: A place on the sequence of one assembly: ``(region, start, end, strand)``.
Segment = tuple[str, int, int, int]
#: A region of the sequence of one assembly, on both strands: ``(region, start, end)``.
Region = tuple[str, int, int]
#: Ensembl's assembly exceptions of a graph, per assembly and sequence region outside the reference sequence:
#: ``{(assembly, region): ((start, end, release mask, reference region, its start, its end, its kind, carried), ...)}``,
#: ``carried`` the reference region carried onto later assemblies, ``((assembly, region, start, end), ...)``.
Exceptions = Mapping[
    tuple[int, str], tuple[tuple[int, int, int, str, int, int, str, tuple[tuple[int, str, int, int], ...]], ...]
]
#: Places carried onto later assemblies: ``(assembly, region, start, end, strand)`` (:py:attr:`GeneFacts.carried`).
Carried = Iterable[tuple[int, str, int, int, int]]
#: Per assembly, the regions that repeat another: ``(region, first base, last base, region repeated, its first base)``.
Repeats = Mapping[int, tuple[tuple[str, int, int, str, int], ...]]

_EMPTY: Mapping = MappingProxyType({})

#: The kind of a sequence that is a chromosome of the organism: a region of Ensembl's coordinate system ``chromosome``,
#: or a region Ensembl ranks in the organism's karyotype (attribute ``karyotype_rank``: the chromosomes of an assembly
#: that Ensembl loads with one level of sequences, ``primary_assembly``).
CHROMOSOME = "chromosome"
#: The kind of the mitochondrial genome: a region that Ensembl translates with an alternate genetic code (attribute
#: ``codon_table``; in a vertebrate core database the mitochondrial genome alone).
MITOCHONDRIAL = "mitochondrial"
#: The kind of every other sequence: a scaffold, a supercontig, a contig (a part of a chromosome, or of none known).
SCAFFOLD = "scaffold"
#: The kinds of :py:attr:`idtrack.graph.types.Place.kind`.
PLACE_KINDS: tuple[str, ...] = (CHROMOSOME, MITOCHONDRIAL, SCAFFOLD)
#: The kinds of sequences that are whole, and named alike, in every assembly of an organism.
WHOLE_SEQUENCES: frozenset[str] = frozenset({CHROMOSOME, MITOCHONDRIAL})


def overlap(one: Segment, other: Segment) -> bool:
    """Tell whether two places on one assembly share a base on one sequence region and one strand.

    Args:
        one: ``(region, start, end, strand)``, start and end 1-based and inclusive, as Ensembl counts.
        other: The same of the other place.

    Returns:
        bool: ``True`` when the regions and strands are the same and the two ranges have a base in common.
    """
    return one[0] == other[0] and one[3] == other[3] and one[1] <= other[2] and other[1] <= one[2]


class GeneFacts(NamedTuple):
    """Where a gene lies and when it lives, as far as :py:mod:`idtrack.graph.genes` compares genes.

    The fields: ``reference``, its places on the reference sequence by assembly (``{assembly: (segment, ...)}``, every
    version together); ``outside``, whether it lies outside the reference sequence at some release; ``placed``,
    whether the graph holds any place of it (a graph made in memory holds none); ``spans``, every place on the
    reference sequence with the releases at which it is there (``(assembly, segment, release mask)``); ``alive``, the
    releases at which it lives (a bit mask; 0 when the graph does not say); ``outside_at``, the releases at which it
    lies outside the reference sequence; ``sequences``, the names of the sequences of its places on the reference
    sequence when every one of them is a whole sequence (:py:data:`WHOLE_SEQUENCES`; the mitochondrial genome named by
    its kind), else ``None``; ``carried``, its places on the reference sequence carried onto later assemblies by
    Ensembl's mapping of the assemblies (``{assembly: (segment, ...)}``); ``regions``, the regions of the reference
    sequence that its places outside it stand for (Ensembl's assembly exceptions), on their own assembly and carried
    onto later ones (``{assembly: ((region, first base, last base), ...)}``, on both strands); ``region_sequences``,
    the names of the sequences of those regions on their own assembly when every one is a whole sequence, else
    ``None``.
    """

    reference: dict[int, tuple[Segment, ...]]
    outside: bool
    placed: bool
    spans: tuple[tuple[int, Segment, int], ...] = ()
    alive: int = 0
    outside_at: int = 0
    sequences: frozenset[str] | None = None
    carried: Mapping[int, tuple[Segment, ...]] = _EMPTY
    regions: Mapping[int, tuple[Region, ...]] = _EMPTY
    region_sequences: frozenset[str] | None = None

    @property
    def copy(self) -> bool:
        """Whether the gene lies only outside the reference sequence (a copy, never a gene of its own)."""
        return self.outside and not self.reference


#: The facts of a gene of which the graph holds nothing (a graph made in memory).
NO_FACTS = GeneFacts({}, False, False)


def _sequence_name(region: str, kind: str) -> str:
    """Return the name a whole sequence is compared by across assemblies: its name, the mitochondrial genome its kind."""
    return MITOCHONDRIAL if kind == MITOCHONDRIAL else region


def facts_of(
    places: Iterable[tuple],
    outside: int,
    repeats: Repeats | None = None,
    alive: int | None = None,
    carried: Carried = (),
    exceptions: Exceptions | None = None,
) -> GeneFacts:
    """Return the facts of one gene version from its places.

    Args:
        places: ``(assembly, region, start, end, strand, release mask)`` of every place of the version, optionally
            followed by the kind of the region (:py:data:`PLACE_KINDS`; :py:data:`SCAFFOLD` when not given).
        outside: The bit mask of the releases at which the version lies outside the reference sequence.
        repeats: The regions that repeat another, per assembly (:py:data:`Repeats`); a place that lies in one is read
            on the region it repeats.
        alive: The bit mask of the releases at which the version lives; by default the releases of its places.
        carried: Its places on the reference sequence carried onto later assemblies by Ensembl's mapping
            (:py:data:`Carried`, :py:meth:`idtrack.TheGraph.carried_places`).
        exceptions: Ensembl's assembly exceptions (:py:data:`Exceptions`, :py:meth:`idtrack.TheGraph.assembly_exceptions`):
            a place outside the reference sequence on a region they list, at a release they list it, stands for
            the reference region they give.

    Returns:
        GeneFacts: Its places at releases at which it lies on the reference sequence, by assembly.
    """
    reference: dict[int, set[Segment]] = {}
    spans: set[tuple[int, Segment, int]] = set()
    names: set[str] = set()
    regions: dict[int, set[Region]] = {}
    region_names: set[str] = set()
    whole_regions = True
    whole = True
    placed = False
    lives = 0
    for place in places:
        assembly, region, start, end, strand, releases = place[:6]
        kind = place[6] if len(place) > 6 else SCAFFOLD
        placed = True
        lives |= int(releases)
        if int(releases) & int(outside):
            for first, last, mask, ref, ref_start, ref_end, ref_kind, onto in (exceptions or {}).get(
                (int(assembly), str(region)), ()
            ):
                if mask & int(releases) and first <= int(end) and int(start) <= last:
                    regions.setdefault(int(assembly), set()).add((ref, ref_start, ref_end))
                    for later, onto_region, onto_start, onto_end in onto:
                        regions.setdefault(int(later), set()).add((onto_region, onto_start, onto_end))
                    whole_regions = whole_regions and ref_kind in WHOLE_SEQUENCES
                    region_names.add(_sequence_name(ref, ref_kind))
            continue
        segment = (str(region), int(start), int(end), int(strand))
        for repeating, first, last, repeated, there in (repeats or {}).get(int(assembly), ()):
            if region == repeating and first <= start and end <= last:
                segment = (repeated, int(start) - first + there, int(end) - first + there, int(strand))
        reference.setdefault(int(assembly), set()).add(segment)
        spans.add((int(assembly), segment, int(releases)))
        whole = whole and kind in WHOLE_SEQUENCES
        names.add(_sequence_name(segment[0], kind))
    moved: dict[int, set[Segment]] = {}
    for assembly, region, start, end, strand in carried:
        moved.setdefault(int(assembly), set()).add((str(region), int(start), int(end), int(strand)))
    return GeneFacts(
        {a: tuple(sorted(s)) for a, s in reference.items()},
        bool(outside),
        placed,
        tuple(sorted(spans)),
        lives if alive is None else int(alive),
        int(outside),
        frozenset(names) if names and whole else None,
        {a: tuple(sorted(s)) for a, s in moved.items()},
        {a: tuple(sorted(s)) for a, s in regions.items()},
        frozenset(region_names) if region_names and whole_regions else None,
    )


def facts_of_places(
    places: Iterable[Place],
    outside: int,
    repeats: Repeats | None = None,
    alive: int | None = None,
    carried: Iterable[Place] = (),
    exceptions: Exceptions | None = None,
) -> GeneFacts:
    """Return the facts of one gene version from its places as the graph gives them (:py:meth:`idtrack.TheGraph.places`).

    Args:
        places: The places of the version.
        outside: The bit mask of the releases at which the version lies outside the reference sequence.
        repeats: The regions that repeat another, per assembly (:py:data:`Repeats`).
        alive: The bit mask of the releases at which the version lives; by default the releases of its places.
        carried: Its places carried onto later assemblies (:py:meth:`idtrack.TheGraph.carried_places`).
        exceptions: Ensembl's assembly exceptions of the graph (:py:attr:`idtrack.TheGraph.exception_regions`).

    Returns:
        GeneFacts: Its facts (:py:func:`facts_of`).
    """
    return facts_of(
        ((p.assembly, p.region, p.start, p.end, p.strand, p.releases.mask, p.kind) for p in places),
        outside,
        repeats,
        alive,
        ((p.assembly, p.region, p.start, p.end, p.strand) for p in carried),
        exceptions,
    )


def merged(facts: Iterable[GeneFacts]) -> GeneFacts:
    """Return the facts of a stable id from those of its versions.

    Args:
        facts: The facts of every version.

    Returns:
        GeneFacts: Every version's places on the reference sequence together; outside or placed when one version is;
        the releases at which a version lives or lies outside, together; the whole sequences when every version's are.
    """
    reference: dict[int, set[Segment]] = {}
    carried: dict[int, set[Segment]] = {}
    regions: dict[int, set[Region]] = {}
    spans: set[tuple[int, Segment, int]] = set()
    names: set[str] = set()
    region_names: set[str] = set()
    whole = whole_regions = True
    outside = placed = False
    alive = outside_at = 0
    for one in facts:
        outside = outside or one.outside
        placed = placed or one.placed
        alive |= one.alive
        outside_at |= one.outside_at
        spans.update(one.spans)
        for assembly, segments in one.reference.items():
            reference.setdefault(assembly, set()).update(segments)
        for assembly, segments in one.carried.items():
            carried.setdefault(assembly, set()).update(segments)
        for assembly, found in one.regions.items():
            regions.setdefault(assembly, set()).update(found)
        if one.reference:
            if one.sequences is None:
                whole = False
            else:
                names.update(one.sequences)
        if one.regions:
            if one.region_sequences is None:
                whole_regions = False
            else:
                region_names.update(one.region_sequences)
    return GeneFacts(
        {a: tuple(sorted(s)) for a, s in reference.items()},
        outside,
        placed,
        tuple(sorted(spans)),
        alive,
        outside_at,
        frozenset(names) if names and whole else None,
        {a: tuple(sorted(s)) for a, s in carried.items()},
        {a: tuple(sorted(s)) for a, s in regions.items()},
        frozenset(region_names) if region_names and whole_regions else None,
    )


def facts_by_stable_id(
    places: Mapping[str, Iterable[tuple]],
    outside: Mapping[str, int],
    history: Iterable[tuple[str, str]],
    repeats: Repeats | None = None,
    alive_of: Callable[[str], int] | None = None,
    unplaced: Iterable[str] = (),
    carried: Mapping[str, Carried] | None = None,
    exceptions: Exceptions | None = None,
) -> tuple[dict[str, GeneFacts], dict[str, frozenset[str]]]:
    """Return the facts of every gene by stable id and the stable ids Ensembl's history joins (:py:func:`count_genes`).

    The one place where the facts of the one rule are assembled: the builder calls it with what it collected
    (:py:func:`idtrack.build.builder._gene_counter`), a graph with what it holds (:py:meth:`idtrack.TheGraph.count_genes`).

    Args:
        places: ``{gene version: (assembly, region, start, end, strand, release mask, kind), ...}``; a version lives at
            the releases of its places.
        outside: ``{gene version: release mask}`` of the versions outside the reference sequence.
        history: ``(gene version, gene version)`` of every step of Ensembl's history between two versions.
        repeats: The regions that repeat another, per assembly (:py:data:`Repeats`).
        alive_of: The releases at which a version that holds no place lives (a mask); ``None``: at none.
        unplaced: The gene versions that hold no place (a graph made in memory; a built graph places every version).
        carried: ``{gene version: carried places}`` (:py:data:`Carried`).
        exceptions: Ensembl's assembly exceptions (:py:data:`Exceptions`).

    Returns:
        tuple: ``{stable id: facts}`` (every version of the id together, :py:func:`merged`) and ``{stable id: the
        stable ids Ensembl's history joins to it}``.
    """
    versions: dict[str, list[GeneFacts]] = {}
    for version in set(places) | set(outside) | set(unplaced):
        held = places.get(version, ())
        alive = None if version in places or alive_of is None else alive_of(version)
        versions.setdefault(_stable_id_of(version), []).append(
            facts_of(held, outside.get(version, 0), repeats, alive, (carried or {}).get(version, ()), exceptions)
        )
    facts = {stable_id: merged(found) for stable_id, found in versions.items()}
    joined: dict[str, set[str]] = {}
    for one, other in history:
        a, b = _stable_id_of(one), _stable_id_of(other)
        if a != b:
            joined.setdefault(a, set()).add(b)
            joined.setdefault(b, set()).add(a)
    return facts, {stable_id: frozenset(found) for stable_id, found in joined.items()}


def _stable_id_of(version: str) -> str:
    """Return the stable id of a gene version ``<id>.<version>`` (a name without a version is its own)."""
    return ids.split(version)[0] if ids.is_versioned_name(version) else version


def facts_of_nodes(graph: TheGraph, nodes: Iterable[str]) -> GeneFacts:
    """Return the facts of a gene from the graph: the places and marks of the given versions together.

    Args:
        graph: The graph.
        nodes: Gene versions (backbone nodes): one version, or every version of a stable id.

    Returns:
        GeneFacts: Their places on the reference sequence by assembly (:py:meth:`idtrack.TheGraph.places`), whether one
        lies outside the reference sequence at some release (:py:meth:`idtrack.TheGraph.non_reference_releases`),
        whether the graph holds any place of them, their places carried onto later assemblies
        (:py:meth:`idtrack.TheGraph.carried_places`) and the reference regions their copies stand for
        (:py:attr:`idtrack.TheGraph.exception_regions`).
    """
    exceptions = graph.exception_regions
    return merged(
        facts_of_places(
            graph.places(node),
            graph.non_reference_releases(node).mask,
            graph.repeated_regions,
            carried=graph.carried_places(node),
            exceptions=exceptions,
        )
        for node in nodes
    )


def place_kinds(graph: TheGraph, node: str, release: int) -> set[str]:
    """Return the kinds of the places of a gene version at a release.

    Args:
        graph: The graph.
        node: A gene version (a backbone node).
        release: The release.

    Returns:
        set[str]: The kinds (:py:data:`PLACE_KINDS`) of the places that hold the release; empty for a version without a
        place there and for a graph that holds no places.
    """
    return {place.kind for place in graph.places(node) if release in place.releases}


def joins_mitochondrial_and_nuclear(
    graph: TheGraph, older: str, old_release: int, newer: str, new_release: int
) -> bool:
    """Tell whether a history step joins a gene of the mitochondrial genome and a gene of a nuclear sequence.

    One version lies on the mitochondrial genome at its side of the event, the other has a place at its side and none
    on the mitochondrial genome (a chromosome or a scaffold: the kind says it is nuclear, whatever the relation of the
    two places across an assembly change). A version without a place there is no evidence either way.

    Args:
        graph: The graph.
        older: The version on the old side of the event.
        old_release: The last release of the old side.
        newer: The version on the new side.
        new_release: The first release of the new side.

    Returns:
        bool: ``True`` when the step joins the two genomes (Ensembl reused a mitochondrial gene's id for a nuclear gene,
        or the reverse).
    """
    kinds_older = place_kinds(graph, older, old_release)
    kinds_newer = place_kinds(graph, newer, new_release)
    on_older, on_newer = MITOCHONDRIAL in kinds_older, MITOCHONDRIAL in kinds_newer
    return on_older != on_newer and bool(kinds_newer if on_older else kinds_older)


def lies_on_mitochondrial_genome(graph: TheGraph, node: str, release: int) -> bool:
    """Tell whether a gene version lies on the organism's mitochondrial genome at a release.

    Args:
        graph: The graph.
        node: A gene version (a backbone node).
        release: The release.

    Returns:
        bool: ``True`` when a place of the version at that release lies on a region of kind :py:data:`MITOCHONDRIAL`;
        ``False`` for a graph that holds no places.
    """
    return MITOCHONDRIAL in place_kinds(graph, node, release)


def same_place(one: GeneFacts, other: GeneFacts) -> bool:
    """Tell whether two genes lie at one place: a place of each on one assembly's reference sequence overlaps.

    Args:
        one: The facts of one gene.
        other: The facts of the other.

    Returns:
        bool: ``True`` when, on an assembly on whose reference sequence both lie, two of their places overlap
        (:py:func:`overlap`).
    """
    for assembly in one.reference.keys() & other.reference.keys():
        for segment in one.reference[assembly]:
            for segment_2 in other.reference[assembly]:
                if overlap(segment, segment_2):
                    return True
    return False


def on_other_sequences(one: GeneFacts, other: GeneFacts) -> bool:
    """Tell whether two genes lie on whole sequences of different names (the comparison across assemblies).

    Args:
        one: The facts of one gene.
        other: The facts of the other.

    Returns:
        bool: ``True`` when every place of each on the reference sequence lies on a chromosome or the mitochondrial
        genome (:py:attr:`GeneFacts.sequences`) and no sequence of one is a sequence of the other.
    """
    return one.sequences is not None and other.sequences is not None and not one.sequences & other.sequences


def _carried_relation(one: GeneFacts, other: GeneFacts) -> str | None:
    """Compare a gene's places carried onto another assembly with the other gene's places there.

    Returns:
        str | None: ``same_place`` when a carried place of one overlaps a place of the other on its assembly,
        ``places_apart`` when places were carried onto an assembly of the other and none overlaps, ``None`` when no
        place of either was carried onto an assembly on whose reference sequence the other lies.
    """
    compared = False
    for mine, theirs in ((one, other), (other, one)):
        for assembly in mine.carried.keys() & theirs.reference.keys():
            compared = True
            if any(overlap(a, b) for a in mine.carried[assembly] for b in theirs.reference[assembly]):
                return "same_place"
    return "places_apart" if compared else None


def _stands(facts: GeneFacts) -> dict[int, set[Region]]:
    """Return what a gene's places stand for on each assembly, on both strands: its places and its copies' regions."""
    found: dict[int, set[Region]] = {}
    for table in (facts.reference, facts.carried):
        for assembly, segments in table.items():
            found.setdefault(assembly, set()).update((s[0], s[1], s[2]) for s in segments)
    for assembly, regions in facts.regions.items():
        found.setdefault(assembly, set()).update(regions)
    return found


def _regions_relation(one: GeneFacts, other: GeneFacts) -> str | None:
    """Compare two genes through the reference regions that their copies stand for (Ensembl's assembly exceptions).

    Returns:
        str | None: ``places_apart`` when, on an assembly where both stand (a copy by its region, a gene of the
        reference sequence by its places, on either strand), nothing of one shares a base with the other, or when one
        copy's regions lie on whole sequences of other names than every place of the other (across assemblies);
        ``None`` when they share a base (the bases inside a region are not compared) or nothing compares them.
    """
    if not (one.regions or other.regions):
        return None
    mine, theirs = _stands(one), _stands(other)
    common = mine.keys() & theirs.keys()
    if common:
        for assembly in common:
            if any(a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2] for a in mine[assembly] for b in theirs[assembly]):
                return None
        return "places_apart"
    names = [f.region_sequences if f.regions else f.sequences for f in (one, other)]
    if names[0] is not None and names[1] is not None and not names[0] & names[1]:
        return "places_apart"
    return None


def relation(one: GeneFacts, other: GeneFacts) -> str:
    """Return where two genes lie relative to each other (a key of :py:data:`idtrack.results.PLACE_RELATIONS`).

    Args:
        one: The facts of one gene.
        other: The facts of the other.

    Returns:
        str: In this order: ``same_place`` when, on an assembly on whose reference sequence both lie, a place of one
        overlaps a place of the other (:py:func:`same_place`); ``places_apart`` when they lie on the reference sequence
        of a common assembly and no places overlap there; else, where a place of one was carried onto an assembly on
        whose reference sequence the other lies (Ensembl's mapping of the assemblies), ``same_place``
        when a carried place overlaps a place of the other there and ``places_apart`` when none does; else, where
        either lies outside the reference sequence, ``places_apart`` when the reference region a copy stands for
        (Ensembl's assembly exception) shares no base with what the other stands for, or lies on whole sequences of
        other names, and ``outside_reference`` otherwise; else ``no_place`` when the graph holds no place of either;
        else ``places_apart`` when they lie on whole sequences of different names (:py:func:`on_other_sequences`: two
        chromosomes, or a chromosome and the mitochondrial genome, which no assembly change turns one into the other);
        else ``other_assemblies``.
    """
    if one.reference.keys() & other.reference.keys():
        return "same_place" if same_place(one, other) else "places_apart"
    carried = _carried_relation(one, other)
    if carried is not None:
        return carried
    if one.outside or other.outside:
        return _regions_relation(one, other) or "outside_reference"
    if not one.placed or not other.placed:
        return "no_place"
    if on_other_sequences(one, other):
        return "places_apart"
    return "other_assemblies"


class _Union:
    """Components of the members of one frame (union-find)."""

    def __init__(self, members: Iterable[str]):
        self.parent = {member: member for member in members}

    def root(self, member: str) -> str:
        parent = self.parent
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def join(self, one: str, other: str) -> None:
        a, b = self.root(one), self.root(other)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def join_overlaps(self, segments: Iterable[tuple[tuple, int, int, str]]) -> None:
        """Join the members whose places overlap: ``(run key, start, end, member)``, one run per region and strand.

        The components of the overlap graph of each run, by one sweep over the places sorted by start (a place that
        starts before the reach of the run overlaps the place that reaches furthest in it).
        """
        run: tuple | None = None
        reach, furthest = 0, ""
        for key, start, end, member in sorted(segments):
            if run == key and start <= reach:
                self.join(member, furthest)
                if end > reach:
                    reach, furthest = end, member
            else:
                run, reach, furthest = key, end, member

    def count(self) -> int:
        return len({self.root(member) for member in self.parent})


def _at_release(members: list[str], facts: Mapping[str, GeneFacts], release: int) -> dict[str, str]:
    """Return the loci of the ids that live at a release, as ``{id: locus}``: overlapping places there are one locus.

    An id that lies outside the reference sequence at the release is not among them; an id that lives at the release
    without a place there is a locus of its own.
    """
    bit = 1 << release
    present = [member for member in members if facts[member].alive & bit and not facts[member].outside_at & bit]
    union = _Union(present)
    union.join_overlaps(
        ((assembly, segment[0], segment[3]), segment[1], segment[2], member)
        for member in present
        for assembly, segment, mask in facts[member].spans
        if mask & bit
    )
    return {member: union.root(member) for member in present}


def _successive(one: GeneFacts, other: GeneFacts) -> bool:
    """Tell whether two ids follow one another: no release at which both live (or the graph does not say when)."""
    return not one.alive & other.alive


def _lineages(members: list[str], facts: Mapping[str, GeneFacts], joined: Mapping[str, Collection[str]]) -> _Union:
    """Return the lineages of the ids: ids that Ensembl's history carried one to the other where they follow one another."""
    union = _Union(members)
    for member in members:
        for other in joined.get(member, ()):
            if other in union.parent and _successive(facts[member], facts[other]):
                union.join(member, other)
    return union


def _on_assembly(members: list[str], facts: Mapping[str, GeneFacts], lineage: _Union, assembly: int) -> dict[str, str]:
    """Return the loci of the ids placed on one assembly's reference sequence, over its releases, as ``{id: locus}``.

    Ids whose places on the assembly overlap are one locus, and so are ids of one lineage (:py:func:`_lineages`).
    """
    present = [member for member in members if assembly in facts[member].reference]
    union = _Union(present)
    first_of: dict[str, str] = {}
    for member in present:
        root = lineage.root(member)
        if root in first_of:
            union.join(member, first_of[root])
        else:
            first_of[root] = member
    union.join_overlaps(
        ((segment[0], segment[3]), segment[1], segment[2], member)
        for member in present
        for segment in facts[member].reference[assembly]
    )
    return {member: union.root(member) for member in present}


def _by_sequence(members: list[str], facts: Mapping[str, GeneFacts], lineage: _Union) -> dict[str, str]:
    """Return the groups, across assemblies, of the ids that lie on whole sequences only, as ``{id: group}``.

    Ids on a sequence of one name (a chromosome, the mitochondrial genome) are one group, and so are ids of one
    lineage (:py:func:`_lineages`); an id with a place on a scaffold is not among them.
    """
    present = [member for member in members if facts[member].sequences is not None]
    union = _Union(present)
    first_on: dict[str, str] = {}
    for member in present:
        for key in (
            f"lineage:{lineage.root(member)}",
            *(f"sequence:{n}" for n in sorted(facts[member].sequences or ())),
        ):
            if key in first_on:
                union.join(member, first_on[key])
            else:
                first_on[key] = member
    return {member: union.root(member) for member in present}


def count_genes(
    stable_ids: Collection[str],
    facts: Mapping[str, GeneFacts],
    joined: Mapping[str, Collection[str]],
    release: int | None = None,
) -> int:
    """Return how many genes some stable ids name: the size of a set of them that the graph tells pairwise apart.

    A copy (:py:attr:`GeneFacts.copy`: it lies only outside the reference sequence) adds no gene: it stands for a gene
    of the reference sequence; copies alone stand for one gene. The other ids are compared in each frame in which the
    graph can tell loci apart:

    - **one release**: the ids that live at it and lie on the reference sequence there; ids whose places at that release
      overlap (one region, one strand; a chain of overlaps among the ids counted) are one locus, every other id is a
      locus of its own (two ids that live at once are two genes unless their places overlap, whatever history they
      share: a split of one id into two that then live side by side is two genes);
    - **one assembly**: the ids placed on its reference sequence, over its releases; ids whose places overlap are one
      locus, and so are ids of one lineage (Ensembl's history carried one to the other where they follow one another,
      no release at which both live); a later id at another place without a history link is another locus;
    - **the names of whole sequences across assemblies**: the ids all of whose places lie on chromosomes or the
      mitochondrial genome; ids on a sequence of one name, or of one lineage, are one group (a gene of chromosome 15 in
      one assembly and a gene of chromosome 6 in the next are two).

    Two ids are told apart when a frame that holds both puts them at two loci. The count is the size of a set of ids
    that are pairwise told apart, built from the frame that shows the most loci (one id of each of its loci, the first
    by id) and extended, in the order of the ids, by every other id told apart from each id of the set so far, the
    other ids of that frame among them (a gene of chromosome 1 of the next assembly beside five genes that live apart
    on other chromosomes at once is a sixth gene; two genes at one place at one release that lie apart at the next are
    two). The set is built in this order, not searched for the largest, so the count is a lower bound of the most ids
    told pairwise apart. Ids that nothing tells apart are one gene:
    holders on two assemblies without a history link and without a comparable place, or ids of which the graph holds
    neither a place nor the releases at which they live, are not evidence of two loci.

    Args:
        stable_ids: The stable gene ids (without version).
        facts: ``{stable id: facts}``; an id missing here has :py:data:`NO_FACTS`.
        joined: ``{stable id: stable ids}`` that Ensembl's history carried it to or from; an id missing here has none.
        release: Count at this release only (the frame of one release; an id that does not live at it adds none).

    Returns:
        int: The number of genes; 0 for no id, at least 1 for some.
    """
    given = set(stable_ids)
    if not given:
        return 0
    known = {member: facts.get(member, NO_FACTS) for member in given}
    members = sorted(member for member in given if not known[member].copy)
    if len(members) < 2:
        return 1
    if release is not None:
        return max(1, len(set(_at_release(members, known, int(release)).values())))
    lineage = _lineages(members, known, joined)
    frames: list[dict[str, str]] = []
    seen: set[frozenset] = set()

    def add(frame: dict[str, str]) -> None:
        signature = frozenset(frame.items())
        if len(frame) > 1 and signature not in seen:
            seen.add(signature)
            frames.append(frame)

    lives = 0
    assemblies: set[int] = set()
    for member in members:
        lives |= known[member].alive
        assemblies.update(known[member].reference)
    release_now = 0
    while lives:
        if lives & 1:
            add(_at_release(members, known, release_now))
        lives >>= 1
        release_now += 1
    for assembly in sorted(assemblies):
        add(_on_assembly(members, known, lineage, assembly))
    add(_by_sequence(members, known, lineage))
    if not frames:
        return 1
    best = max(frames, key=lambda frame: len(set(frame.values())))
    # one id of each locus of that frame (the first by id): ids of two loci of one frame are told apart
    chosen = list({best[member]: member for member in sorted(best, reverse=True)}.values())
    if len(chosen) == len(members):
        return len(chosen)
    where: dict[str, dict[int, str]] = {member: {} for member in members}
    for index, frame in enumerate(frames):
        for member, locus in frame.items():
            where[member][index] = locus

    def apart(one: str, other: str) -> bool:
        mine, theirs = where[one], where[other]
        return any(index in theirs and theirs[index] != locus for index, locus in mine.items())

    # every id not chosen, those of the frame of most loci among them (an id that shares a locus with the chosen id of
    # that frame may be told apart from it by another frame)
    for member in members:
        if member not in chosen and all(apart(member, other) for other in chosen):
            chosen.append(member)
    return len(chosen)
