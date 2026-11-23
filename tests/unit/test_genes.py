#!/usr/bin/env python3
"""What one gene is (:py:mod:`idtrack.graph.genes`): the one rule behind decisions 72, 73 and 74.

Each test names the plausible wrong rule it tells apart from the one built.
"""

from __future__ import annotations

from idtrack.graph import genes
from idtrack.graph.genes import GeneFacts, count_genes, facts_of, overlap, relation, same_place

ALL = (1 << 120) - 1  # every release


def at(*segments: tuple[int, str, int, int, int], outside: bool = False) -> GeneFacts:
    """The facts of a gene lying at the given places ``(assembly, region, start, end, strand)``."""
    reference: dict[int, list] = {}
    for assembly, region, start, end, strand in segments:
        reference.setdefault(assembly, []).append((region, start, end, strand))
    return GeneFacts({a: tuple(sorted(s)) for a, s in reference.items()}, outside, True)


COPY = GeneFacts({}, True, True)  # a gene that lies only on a patch


class TestOverlap:
    """Two places overlap when they share a base on one region and one strand."""

    def test_a_shared_base_overlaps_and_a_neighbour_does_not(self):
        """Ensembl counts 1-based and inclusive: ending where the other starts is a shared base, one before is not.

        Fails if the ends are read exclusive (the first pair would not overlap) or as touching (the second would).
        """
        assert overlap(("1", 100, 200, 1), ("1", 200, 300, 1))
        assert not overlap(("1", 100, 199, 1), ("1", 200, 300, 1))

    def test_the_strand_and_the_region_must_be_the_same(self):
        """Fails if the strand or the region is ignored (an antisense neighbour, a gene of another chromosome)."""
        assert not overlap(("1", 100, 200, 1), ("1", 150, 300, -1))
        assert not overlap(("1", 100, 200, 1), ("2", 150, 300, 1))


class TestSamePlace:
    """Two genes lie at one place when places of both on one assembly's reference sequence overlap."""

    def test_places_on_two_assemblies_are_not_compared(self):
        """Fails if places are compared across assemblies (the same numbers on GRCh37 and GRCh38 are other bases)."""
        assert not same_place(at((37, "1", 100, 200, 1)), at((38, "1", 100, 200, 1)))
        assert same_place(at((37, "1", 100, 200, 1), (38, "1", 900, 950, 1)), at((38, "1", 940, 990, 1)))

    def test_places_at_releases_outside_the_reference_sequence_are_not_compared(self):
        """A version on a patch at 80 and on chromosome 1 at 81 compares by its place on chromosome 1 only.

        Fails if a place of a release at which the gene lies outside the reference sequence counts (the patch's own
        coordinates are not those of the chromosome).
        """
        version = facts_of([(38, "CHR_PATCH", 100, 200, 1, 1 << 80), (38, "1", 5000, 5100, 1, 1 << 81)], 1 << 80)
        assert version.reference == {38: (("1", 5000, 5100, 1),)} and version.outside and not version.copy
        assert not same_place(version, at((38, "CHR_PATCH", 150, 160, 1)))
        assert same_place(version, at((38, "1", 5050, 5060, 1)))


class TestRelation:
    """Where two genes lie relative to each other, the relation the rule of bridges reads (decision 72)."""

    def test_every_relation(self):
        """Each key in the order of the rule: one assembly's reference decides first, then copies, then no place.

        Fails if a copy's place is compared on the reference sequence (``outside_reference`` would be
        ``places_apart``), if a gene without places is taken as a copy, or if two assemblies are compared.
        """
        a = at((38, "1", 100, 200, 1))
        assert relation(a, at((38, "1", 150, 300, 1))) == "same_place"
        assert relation(a, at((38, "1", 900, 950, 1))) == "places_apart"
        assert relation(a, COPY) == "outside_reference"
        assert relation(a, genes.NO_FACTS) == "no_place"
        assert relation(a, at((37, "1", 100, 200, 1))) == "other_assemblies"
        assert relation(GeneFacts({38: (("1", 100, 200, 1),)}, True, True), a) == "same_place"


class TestCountGenes:
    """How many genes some stable ids name."""

    def test_a_copy_outside_the_reference_sequence_adds_no_gene(self):
        """A gene and its copy on a patch are one gene; copies alone stand for one.

        Fails if genes are counted by stable id (2: the review's 247 human miRBase entries of decision 73 and 636
        labels of decision 74).
        """
        facts = {"REF": at((38, "1", 100, 200, 1)), "COPY": COPY, "COPY2": COPY}
        assert count_genes(["REF", "COPY"], facts, {}) == 1
        assert count_genes(["REF", "COPY", "COPY2"], facts, {}) == 1
        assert count_genes(["COPY", "COPY2"], facts, {}) == 1

    def test_a_copy_beside_two_genes_at_two_loci_leaves_two(self):
        """Fails if a copy merges the genes it stands beside (one gene instead of two)."""
        facts = {"A": at((38, "1", 100, 200, 1)), "B": at((38, "1", 900, 950, 1)), "COPY": COPY}
        assert count_genes(["A", "B", "COPY"], facts, {}) == 2

    def test_the_ids_of_one_locus_over_time_are_one_gene(self):
        """An id Ensembl's history carried to another id, or an id whose place the next one overlaps, is one gene.

        ``OLD`` and ``NEW`` lie apart (Ensembl's history moved the gene); ``FIRST`` and ``SECOND`` overlap without an
        event. Fails if either join is left out (the review's 500 human miRBase entries of successive ids).
        """
        facts = {
            "OLD": at((38, "1", 100, 200, 1)),
            "NEW": at((38, "1", 5000, 5100, 1)),
            "FIRST": at((38, "2", 100, 200, 1)),
            "SECOND": at((38, "2", 150, 260, 1)),
        }
        joined = {"OLD": {"NEW"}, "NEW": {"OLD"}}
        assert count_genes(["OLD", "NEW"], facts, joined) == 1
        assert count_genes(["OLD", "NEW"], facts, {}) == 2
        assert count_genes(["FIRST", "SECOND"], facts, {}) == 1

    def test_genes_at_two_loci_and_genes_on_two_strands_are_two(self):
        """Fails if the rule takes genes of one region as one gene (the second pair) or ignores the strand (the third)."""
        facts = {
            "A": at((38, "1", 100, 200, 1)),
            "B": at((38, "1", 900, 950, 1)),
            "C": at((38, "1", 150, 260, -1)),
        }
        assert count_genes(["A", "B"], facts, {}) == 2
        assert count_genes(["A", "C"], facts, {}) == 2

    def test_only_the_ids_counted_join(self):
        """``LONG`` overlaps ``A`` and ``B``, which do not overlap each other (a host gene and two nested genes).

        ``A`` and ``B`` are two genes; with ``LONG`` among the ids counted, a chain joins all three. Fails if every
        gene of the graph joins (``A`` and ``B`` would be one gene through ``LONG``) or if only direct pairs join.
        """
        facts = {
            "LONG": at((38, "1", 100, 10_000, 1)),
            "A": at((38, "1", 200, 300, 1)),
            "B": at((38, "1", 5000, 5100, 1)),
        }
        assert count_genes(["A", "B"], facts, {}) == 2
        assert count_genes(["A", "B", "LONG"], facts, {}) == 1

    def test_a_chain_of_overlaps_joins_in_any_order(self):
        """``A`` reaches over ``B``, ``D`` and ``C``, which do not overlap each other: one gene, in any order of the ids.

        Fails if the sweep compares a place only with the one before it in start order rather than with the furthest
        reach of the run (``D`` starts after ``B`` ends, but before ``A`` ends).
        """
        facts = {
            "A": at((38, "1", 100, 1000, 1)),
            "B": at((38, "1", 200, 300, 1)),
            "C": at((38, "1", 900, 1200, 1)),
            "D": at((38, "1", 400, 500, 1)),
        }
        for order in (["A", "B", "C", "D"], ["D", "C", "B", "A"], ["C", "A", "D", "B"]):
            assert count_genes(order, facts, {}) == 1
        assert count_genes(["B", "D"], facts, {}) == 2

    def test_ids_without_places_are_two_genes_only_where_they_live_at_once(self):
        """A graph made in memory holds no places: ids that live at one release are two genes, others are no evidence.

        Fails if an id without facts is counted as a gene of its own whatever its releases (the rule before I131:
        ``X`` and ``Y`` that never live at once, without a history link, would be two), or if two ids that live at
        one release without a place are taken as one.
        """
        at_once = {
            "X": GeneFacts({}, False, False, alive=1 << 60 | 1 << 61),
            "Y": GeneFacts({}, False, False, alive=1 << 61),
        }
        one_after = {"X": GeneFacts({}, False, False, alive=1 << 60), "Y": GeneFacts({}, False, False, alive=1 << 61)}
        assert count_genes(["X", "Y"], at_once, {}) == 2
        assert count_genes(["X", "Y"], at_once, {"X": {"Y"}, "Y": {"X"}}) == 2
        assert count_genes(["X", "Y"], one_after, {}) == 1
        assert count_genes(["X", "Y"], {}, {}) == 1
        assert count_genes(["X"], {}, {}) == 1 and count_genes([], {}, {}) == 0

    def test_a_gene_on_the_reference_sequence_at_some_releases_is_no_copy(self):
        """A gene that lay on a haplotype of GRCh37 and lies on a chromosome of GRCh38 counts by its GRCh38 place.

        Fails if a gene outside the reference sequence at any release is taken as a copy.
        """
        moved = genes.merged(
            [facts_of([(37, "CHR_HAP", 100, 200, 1, 1 << 75)], 1 << 75), facts_of([(38, "1", 100, 200, 1, 1 << 76)], 0)]
        )
        assert not moved.copy and moved.outside
        facts = {"MOVED": moved, "OTHER": at((38, "1", 900, 950, 1))}
        assert count_genes(["MOVED", "OTHER"], facts, {}) == 2


class TestRepeatedRegions:
    """A pseudoautosomal region of Y is read on the region of X it repeats (decisions 93 and 95 on the one rule)."""

    REPEATS = {38: (("Y", 10_001, 2_781_479, "X", 10_001), ("Y", 56_887_903, 57_217_415, "X", 155_701_383))}

    def test_a_y_copy_lies_at_its_x_genes_place(self):
        """The Y copy of a gene of PAR2 (another offset than PAR1) is one gene with it; a gene elsewhere in PAR1 is not.

        Fails if the region of Y is compared as it is (the copy would be a second gene), if the offset of PAR2 is lost,
        or if a place outside a region is moved.
        """
        x_gene = facts_of([(38, "X", 155_800_000, 155_810_000, 1, ALL)], 0, self.REPEATS)
        y_copy = facts_of([(38, "Y", 56_986_520, 56_996_520, 1, ALL)], 0, self.REPEATS)
        elsewhere = facts_of([(38, "Y", 2_000_000, 2_100_000, 1, ALL)], 0, self.REPEATS)
        outside_par = facts_of([(38, "Y", 20_000_000, 20_100_000, 1, ALL)], 0, self.REPEATS)
        assert y_copy.reference == {38: (("X", 155_800_000, 155_810_000, 1),)}
        assert relation(x_gene, y_copy) == "same_place"
        facts = {"X1": x_gene, "Y1": y_copy, "Y2": elsewhere}
        assert count_genes(["X1", "Y1"], facts, {}) == 1 and count_genes(["X1", "Y2"], facts, {}) == 2
        assert outside_par.reference == {38: (("Y", 20_000_000, 20_100_000, 1),)}


def lives(assembly: int, region: str, start: int, end: int, strand: int, releases: range, kind: str = genes.CHROMOSOME):
    """The facts of a gene version that lies at one place at some releases (``releases``: a range of releases)."""
    mask = sum(1 << r for r in releases)
    return facts_of([(assembly, region, start, end, strand, mask, kind)], 0)


class TestOneGeneOverTime:
    """I131: history joins only ids that follow one another; ids that live at once are one gene only at one place."""

    def test_a_split_into_two_ids_that_live_side_by_side_is_two_genes(self):
        """Ensembl's history splits ``OLD`` into ``PLUS`` and ``MINUS``, which live together on the same bases, two strands.

        The mouse ``mmu-mir-30f`` case (F3): the two current ids are two genes, with or without their ancestor among the
        ids counted. Fails if history joins through a shared ancestor over all releases (the rule before: 1 gene).
        """
        facts = {
            "OLD": lives(38, "4", 100, 190, 1, range(68, 103)),
            "PLUS": lives(39, "4", 100, 190, 1, range(103, 116)),
            "MINUS": lives(39, "4", 100, 190, -1, range(103, 116)),
        }
        joined = {"OLD": {"PLUS", "MINUS"}, "PLUS": {"OLD"}, "MINUS": {"OLD"}}
        assert count_genes(["OLD", "PLUS", "MINUS"], facts, joined) == 2
        assert count_genes(["PLUS", "MINUS"], facts, joined) == 2
        assert count_genes(["OLD", "PLUS"], facts, joined) == 1

    def test_a_split_on_one_assembly_into_ids_apart_is_two_genes(self):
        """The same split without an assembly change: the history joins ``OLD`` to both, the two children lie apart.

        Fails if the frame of one assembly alone decides (history and overlaps chain the three into one gene).
        """
        facts = {
            "OLD": lives(39, "7", 1000, 2000, 1, range(103, 105)),
            "LEFT": lives(39, "7", 1000, 1400, 1, range(105, 116)),
            "RIGHT": lives(39, "7", 5000, 5400, 1, range(105, 116)),
        }
        joined = {"OLD": {"LEFT", "RIGHT"}, "LEFT": {"OLD"}, "RIGHT": {"OLD"}}
        assert count_genes(["OLD", "LEFT", "RIGHT"], facts, joined) == 2
        assert count_genes(["OLD", "LEFT"], facts, joined) == 1

    def test_two_ids_merged_into_one_were_two_genes_while_they_lived_at_once(self):
        """``B`` and ``C`` live together apart, then Ensembl merges both into ``A`` at ``B``'s place.

        Fails if the merge makes ``B`` and ``C`` one gene (history through ``A``), or if ``A`` and ``B`` (one after the
        other, joined by history) are two.
        """
        facts = {
            "B": lives(39, "1", 100, 200, 1, range(103, 106)),
            "C": lives(39, "1", 900, 950, 1, range(103, 106)),
            "A": lives(39, "1", 100, 950, 1, range(106, 116)),
        }
        joined = {"A": {"B", "C"}, "B": {"A"}, "C": {"A"}}
        assert count_genes(["B", "C"], facts, joined) == 2
        assert count_genes(["A", "B", "C"], facts, joined) == 2
        assert count_genes(["A", "B"], facts, joined) == 1

    def test_holders_on_two_assemblies_that_nothing_compares_are_one_gene(self):
        """The rhesus ``mml-mir-1232`` case (``MI0006322``, F4): one gene id on MMUL_1, another on Mmul_10, no history.

        The two lie on chromosomes of one name, or one of them on a scaffold: no frame tells them apart. Fails if
        holders without a comparable place are taken as two loci (the rule before: 2).
        """
        facts = {
            "OLD": lives(1, "X", 145_365_733, 145_365_830, 1, range(63, 86)),
            "NEW": lives(10, "X", 150_000_000, 150_000_090, -1, range(98, 116)),
            "ON_SCAFFOLD": lives(10, "QNVO02000259.1", 100, 190, 1, range(98, 116), genes.SCAFFOLD),
        }
        assert count_genes(["OLD", "NEW"], facts, {}) == 1
        assert count_genes(["OLD", "ON_SCAFFOLD"], facts, {}) == 1

    def test_holders_on_chromosomes_of_different_names_are_two_genes(self):
        """I133 in the count: a gene of chromosome 15 in one assembly and a gene of chromosome 6 in the next are two.

        Fails if chromosome names are not compared across assemblies (1), or if a scaffold's name is (the third pair).
        """
        facts = {
            "ON_15": lives(9, "15", 100, 900, 1, range(56, 67)),
            "ON_6": lives(111, "6", 100, 900, 1, range(90, 116)),
            "ON_MT": lives(9, "MT", 100, 900, 1, range(56, 67), genes.MITOCHONDRIAL),
            "ON_SCAFFOLD": lives(111, "AEMK02000452.1", 100, 900, 1, range(90, 116), genes.SCAFFOLD),
        }
        assert count_genes(["ON_15", "ON_6"], facts, {}) == 2
        assert count_genes(["ON_MT", "ON_6"], facts, {}) == 2
        assert count_genes(["ON_15", "ON_SCAFFOLD"], facts, {}) == 1
        assert count_genes(["ON_15", "ON_6"], facts, {"ON_15": {"ON_6"}, "ON_6": {"ON_15"}}) == 1

    def test_evidence_of_two_frames_adds_up(self):
        """Cynomolgus ``SNORD81``: five genes apart at once, and a sixth on another chromosome than each of them.

        The five live apart on one assembly; the sixth lies on chromosome 1 of the next assembly. No single frame shows
        six (the release shows five, the chromosome names five: two of the five share chromosome 3), but every pair of
        the six is told apart by some frame: six genes. Fails if the count is the
        most loci of one frame (5), or if an id that a frame does not compare with a locus is added (the gene of
        chromosome 3 of the next assembly, which no frame tells from the genes of chromosome 3).
        """
        five = {
            f"G{c}": lives(5, c, 100 * i, 100 * i + 50, 1, range(91, 103)) for i, c in enumerate("3 6 X 11".split())
        }
        five["G3B"] = lives(5, "3", 900_000, 900_050, 1, range(91, 103))
        facts = {
            **five,
            "G1": lives(6, "1", 100, 150, -1, range(103, 116)),
            "N3": lives(6, "3", 5, 55, 1, range(103, 116)),
        }
        assert count_genes(list(five), facts, {}) == 5
        assert count_genes([*five, "G1"], facts, {}) == 6
        assert count_genes([*five, "N3"], facts, {}) == 5
        assert count_genes([*five, "G1", "N3"], facts, {}) == 6

    def test_the_count_is_of_ids_told_pairwise_apart(self):
        """A and B lie at one place and live together at 10; C is told apart from A only, D from B only.

        The frame of release 10 shows the most loci ({A, B}, G, H, I); C lives with A and G at 11, D with B and G at
        12, C and D lie apart on the assembly that H and I move to. No frame holds C and B or D and A, so C and D
        cannot both join A or B: of the six ids no set of six is told apart pairwise, and {A, G, H, I, C} is. Fails if
        an id joins the set where it is told apart from some id of a locus and not from the id the set holds (6).
        """

        def at(*places):
            return facts_of([(*place[:5], sum(1 << r for r in place[5]), genes.SCAFFOLD) for place in places], 0)

        facts = {
            "A": at((5, "s1", 100, 200, 1, (10, 11))),
            "B": at((5, "s1", 100, 200, 1, (10, 12))),
            "G": at((5, "s9", 100, 200, 1, (10, 11, 12, 13))),
            "C": at((6, "c", 100, 200, 1, (11, 13))),
            "D": at((6, "d", 100, 200, 1, (12, 13))),
            "H": at((5, "h", 100, 200, 1, (10,)), (6, "h6", 100, 200, 1, (14,))),
            "I": at((5, "i", 100, 200, 1, (10,)), (6, "i6", 100, 200, 1, (14,))),
        }
        assert count_genes(list(facts), facts, {}) == 5
        assert count_genes(["A", "B", "G", "H", "I"], facts, {}) == 4

    def test_an_id_that_shares_a_locus_of_the_frame_of_most_loci_is_still_compared(self):
        """A and B overlap at release 10 and lie apart at 11 (B's model moved on one assembly); C lives at 10 apart from both.

        Every pair is told apart by some frame (A and B at 11, C from each at 10): three genes. The frame of release 10
        shows the most loci (two: {A, B} and C), and B, which shares A's locus there, must still be compared with the
        set. Fails if the ids of that frame that are not chosen are never compared (2).
        """
        facts = {
            "A": facts_of([(1, "1", 100, 200, 1, (1 << 10) | (1 << 11), genes.CHROMOSOME)], 0),
            "B": facts_of(
                [(1, "1", 150, 250, 1, 1 << 10, genes.CHROMOSOME), (1, "1", 900, 950, 1, 1 << 11, genes.CHROMOSOME)], 0
            ),
            "C": facts_of([(1, "1", 5000, 5100, 1, 1 << 10, genes.CHROMOSOME)], 0),
        }
        assert count_genes(["A", "B", "C"], facts, {}) == 3
        assert count_genes(["A", "B", "C"], facts, {}, release=10) == 2
        assert count_genes(["A", "B"], facts, {}) == 2

    def test_one_after_the_other_apart_on_one_assembly_is_two_genes(self):
        """The pig ``ssc-mir-16`` swap: an entry on one gene, then on another gene apart on one assembly.

        Whether or not the two genes live at once, places apart on one assembly are two loci. Fails if only genes that
        hold the entry at once are compared.
        """
        facts = {
            "FIRST": lives(111, "13", 1000, 1100, 1, range(90, 100)),
            "THEN": lives(111, "13", 90_000, 90_100, 1, range(100, 116)),
        }
        assert count_genes(["FIRST", "THEN"], facts, {}) == 2

    def test_the_count_at_one_release(self):
        """``release=`` counts the loci of the ids that live at the release only (the frame of one release).

        Fails if ids that do not live at the release are counted, or if a copy there adds a gene.
        """
        facts = {
            "A": lives(38, "1", 100, 200, 1, range(80, 90)),
            "B": lives(38, "1", 900, 950, 1, range(85, 116)),
            "C": lives(38, "1", 150, 250, 1, range(85, 116)),
            "COPY": COPY,
        }
        assert count_genes(["A", "B"], facts, {}, release=80) == 1
        assert count_genes(["A", "B"], facts, {}, release=86) == 2
        assert count_genes(["A", "C", "COPY"], facts, {}, release=86) == 1
        assert count_genes(["COPY"], facts, {}, release=86) == 1


class TestRelationAcrossAssemblies:
    """I133 and I134: across assemblies, genes on whole sequences of different names lie apart."""

    def test_chromosomes_of_different_names_are_places_apart(self):
        """The pig ``RPL28`` bridge of F16 (Sscrofa9 chromosome 15 to Sscrofa11.1 chromosome 6) and rhesus ``MT-ND1``.

        Fails if two assemblies are never compared (``other_assemblies``, the rule before), if a scaffold's name is
        compared, or if the mitochondrial genome is compared by its name rather than by its kind.
        """
        on_15 = lives(9, "15", 100, 900, 1, range(56, 67))
        on_6 = lives(111, "6", 100, 900, 1, range(90, 116))
        also_15 = lives(111, "15", 5000, 9000, 1, range(90, 116))
        scaffold = lives(111, "AEMK02000452.1", 100, 900, 1, range(90, 116), genes.SCAFFOLD)
        mt = lives(801, "MT", 3259, 4214, 1, range(86, 98), genes.MITOCHONDRIAL)
        mt_named_otherwise = lives(10, "chrM", 3259, 4214, 1, range(98, 116), genes.MITOCHONDRIAL)
        on_8 = lives(10, "8", 88_190_642, 88_220_984, -1, range(98, 116))
        assert relation(on_15, on_6) == "places_apart"
        assert relation(on_15, also_15) == "other_assemblies"
        assert relation(on_15, scaffold) == "other_assemblies"
        assert relation(mt, on_8) == "places_apart"
        assert relation(mt, mt_named_otherwise) == "other_assemblies"

    def test_a_gene_on_two_chromosomes_over_its_assemblies_shares_either(self):
        """A gene placed on chromosome 15 in one assembly and 6 in the next is not apart from a gene of chromosome 6.

        Fails if one pair of assemblies decides alone (any pair of different names would make them apart).
        """
        moved = genes.merged([lives(9, "15", 100, 900, 1, range(56, 67)), lives(102, "6", 100, 900, 1, range(67, 90))])
        assert relation(moved, lives(111, "6", 100, 900, 1, range(90, 116))) == "other_assemblies"
        assert relation(moved, lives(111, "2", 100, 900, 1, range(90, 116))) == "places_apart"

    def test_kinds_come_from_places_and_default_to_a_scaffold(self):
        """A place without a kind is a scaffold (a graph made in memory): never compared across assemblies.

        Fails if a place without a kind is read as a chromosome.
        """
        no_kind = facts_of([(9, "15", 100, 900, 1, 1 << 60)], 0)
        assert no_kind.sequences is None
        assert relation(no_kind, lives(111, "6", 100, 900, 1, range(90, 116))) == "other_assemblies"
