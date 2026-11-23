"""History travel by its rules, on small graphs written by hand and on the graphs of real organisms.

The rules of :py:class:`idtrack.engine.Track` that move through time, each tested where it decides: the placing of a
bridge in time (forward and backward, at the target release), the version an id without version stands for, the order
of the lineage evidence, the edges of the window (self-loops and history events at the starting and the target
release), the number of bridges a path may take, Ensembl's score 0 against a version change without score, travel
backward across an assembly change, and the copies' rule of bridges. Then three guards that hold by Ensembl's own
records whatever the graph: a gene version converted at a release where it exists is itself; a stable id that exists
at two releases converts from one to the other as its own version there; and composed, there and back, it returns.
Each test names the plausible wrong version it fails under.
"""

from __future__ import annotations

import random

import pytest

from idtrack import ids
from idtrack.engine import ranking
from idtrack.engine.track import Track
from idtrack.settings import SearchSettings
from tests._real_tables import ORGANISMS, SNAPSHOT, tables_path
from tests.unit.test_track import NAN, GraphBuilder, cand, checks, genes, nothing, path

ALL = range(1, 10)


def bridge_releases(result) -> list[tuple[str, int]]:
    """The kind and release of every bridge step of the best candidate's path."""
    return [(step[0], step[3]) for step in path(result, genes(result, best=True)[0]) if len(step) == 4]


# ----------------------------------------------------------------------------------------- placing a bridge


class TestPlacingABridge:
    """``Track._place_bridge``: the pair of releases of a bridge (leave one gene at x, enter the other at y)."""

    def test_forward_among_equal_gaps_the_latest_leaving_release_is_taken(self):
        """A (1-6) carries ``X`` at 2 and 6, B (3-9) at 3 and 7, both at one place: gaps 2->3 and 6->7 are equal.

        The walk leaves A at 6 and enters B at 7 (the latest x among the smallest gaps). Fails if the first pair of
        equal gap wins instead (2->3).
        """
        builder = GraphBuilder(ALL)
        builder.gene("A", [(1, 1, 6)]).gene("B", [(1, 3, 9)])
        builder.place("A.1", "1", 100, 200, 1, range(1, 7)).place("B.1", "1", 120, 220, 1, range(3, 10))
        builder.xref("X", "A.1", "EntrezGene", [2, 6]).xref("X", "B.1", "EntrezGene", [3, 7])
        result = builder.track().convert("A.1", from_release=1, to_release=9, explain=True)
        assert genes(result) == ["B.1"] and cand(result, "B.1").bridge_gap == 1
        assert [release for _, release in bridge_releases(result)] == [6, 7]
        assert Track._place_bridge((2, 6), (3, 7), 1, 6, 9, False) == (6, 7)

    def test_backward_a_bridge_may_enter_exactly_at_the_target_release(self):
        """Walking back from 9 to 3: A (5-9, born at 5) carries ``X`` at 5 to 9, B (1-3, retired at 3) at 1 to 3.

        B lives at the target release 3 and carried ``X`` there: the bridge enters B at 3. Fails if the far side must
        lie strictly after the target (nothing would be found).
        """
        builder = GraphBuilder(ALL)
        builder.gene("A", [(1, 5, 9)]).gene("B", [(1, 1, 3)])
        builder.place("A.1", "1", 100, 200, 1, range(5, 10)).place("B.1", "1", 120, 220, 1, range(1, 4))
        builder.xref("X", "A.1", "EntrezGene", range(5, 10)).xref("X", "B.1", "EntrezGene", range(1, 4))
        result = builder.track().convert("A.1", from_release=9, to_release=3, explain=True)
        assert genes(result) == ["B.1"] and cand(result, "B.1").bridges == 1
        assert [release for _, release in bridge_releases(result)] == [5, 3]
        assert Track._place_bridge((5, 6, 7, 8, 9), (1, 2, 3), 5, 9, 3, True) == (5, 3)


# ------------------------------------------------------------------------------------------ the starting version


def test_an_id_without_version_stands_for_its_highest_version_at_the_starting_release():
    """Assembly 37 lists ``T.1`` and assembly 38 ``T.2`` of one transcript at 5 to 9: ``T`` reads as ``T.2``.

    Fails if the lowest version of the starting release is taken.
    """
    builder = GraphBuilder(ALL)
    builder.gene("G", [(1, 1, 9)])
    builder._ensembl_node("T.1", "ensembl_transcript")
    builder._xref_edge("T.1", "G.1", "ensembl_transcript", list(range(5, 10)), 37)
    builder.transcript("T.2", "G.1", range(5, 10))
    result = builder.track().convert("T", to_release=9)
    assert result.graph_id == "T.2" and result.from_release == 9 and genes(result) == ["G.1"]


# ------------------------------------------------------------------------------------------ the lineage order


def test_the_lineage_compares_the_final_confidence_before_the_bridges():
    """The lineage compares the final confidence first, then the bridges, and names the criterion that differs.

    Of two pairs, one of confidence 1 without a bridge and one of confidence 0 with a bridge, the second ranks first,
    decided by ``final_confidence``; of two equal but for the bridges, ``bridges`` decides.

    Fails if the bridges are compared before the confidence (or the criterion named is not the one that differs).
    """

    def pair(confidence: int, bridges: int, gene: str):
        return ranking.make_pair(False, confidence, bridges, 0, 1, 0, 1.0, False, 1, 0, gene, gene)

    late, bridged = pair(1, 0, "A"), pair(0, 1, "B")
    pairs = sorted([late, bridged])
    assert pairs == [bridged, late]
    best, counts = ranking.select_best(pairs, set(), {"A": "A", "B": "B"}, lambda gene: [0, 0, 0])
    assert best == [bridged]
    assert ranking.decided_by(ranking.ranked(pairs, best), best, set(), {"A": "A", "B": "B"}, counts) == (
        "final_confidence"
    )
    one, two = pair(0, 1, "A"), pair(0, 2, "B")
    best, counts = ranking.select_best([one, two], set(), {"A": "A", "B": "B"}, lambda gene: [0, 0, 0])
    assert ranking.decided_by([one, two], best, set(), {"A": "A", "B": "B"}, counts) == "bridges"


# -------------------------------------------------------------------------------------- the edges of the window


class TestWindowEdges:
    """A scored self-loop of A (re-mapped between 4 and 5, score 0.5) is crossed only by a walk that passes 4 -> 5."""

    @pytest.fixture
    def looped(self) -> Track:
        """A gene of 1 to 9 with a scored self-loop at 4/5."""
        builder = GraphBuilder(ALL)
        builder.gene("A", [(1, 1, 9)]).event("A.1", "A.1", 4, 0.5)
        return builder.track()

    @pytest.mark.parametrize(
        ("start", "target", "score"),
        [(1, 4, 1.0), (1, 5, 0.5), (4, 9, 0.5), (5, 9, 1.0), (9, 5, 1.0), (9, 4, 0.5), (5, 1, 0.5), (4, 1, 1.0)],
    )
    def test_a_self_loop_counts_where_the_walk_crosses_it(self, looped, start, target, score):
        """Forward from 1 to 4 and backward from 9 to 5 the loop is not crossed; from or to its other side it is.

        Fails if the window's edge is counted on the wrong side in either direction (the loop at the target release
        or at the starting release counted, or left out).
        """
        result = looped.convert("A.1", from_release=start, to_release=target)
        assert genes(result) == ["A.1"] and cand(result, "A.1").score == score

    def test_a_history_event_at_the_target_release_is_not_followed(self):
        """A (1-4) was replaced by B (5-9) at 4/5; B's symbol read at 7 and carried back to 5 is B alone.

        Fails if the walk backward follows the event whose new side is the target release (A would join the
        candidates, though it does not exist at 5), or forward the event whose old side is the target.
        """
        builder = GraphBuilder(ALL)
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).event("A.1", "B.1", 4, 0.9)
        builder.xref("SB", "B.1", "HGNC Symbol", range(5, 10)).xref("SA", "A.1", "HGNC Symbol", range(1, 5))
        track = builder.track()
        assert genes(track.convert("SB", from_release=7, to_release=5)) == ["B.1"]
        assert genes(track.convert("SA", from_release=2, to_release=4)) == ["A.1"]
        assert genes(track.convert("SA", from_release=2, to_release=5)) == ["B.1"]


# -------------------------------------------------------------------------------------------- the bridge limit


class TestBridgeLimit:
    """A (1-3) -> X -> B.1 (4-5) -> B.2 (6) -> Y -> C (7-9), at one place: C is two bridges away from A.

    (A walk leaves a gene by a bridge only from a gene it reached through Ensembl's history, so the second bridge
    leaves B.2, B's next version.)
    """

    @pytest.fixture
    def chain(self) -> GraphBuilder:
        """The chain of the class."""
        builder = GraphBuilder(ALL)
        builder.gene("A", [(1, 1, 3)]).gene("B", [(1, 4, 5), (2, 6, 6)]).gene("C", [(1, 7, 9)])
        for gene, releases in (("A.1", range(1, 4)), ("B.1", [4, 5]), ("B.2", [6]), ("C.1", range(7, 10))):
            builder.place(gene, "1", 100, 200, 1, releases)
        builder.xref("X", "A.1", "EntrezGene", range(1, 4)).xref("X", "B.1", "EntrezGene", [4, 5])
        builder.xref("Y", "B.2", "UniProtKB Gene Name", [6]).xref("Y", "C.1", "UniProtKB Gene Name", [7, 8, 9])
        return builder

    @pytest.mark.parametrize(("limit", "found"), [(0, []), (1, []), (2, ["C.1"]), (3, ["C.1"])])
    def test_a_path_takes_at_most_the_bridges_allowed(self, chain, limit, found):
        """Two bridges reach C; one bridge reaches only B, which does not live at 9. Fails off by one either way."""
        result = chain.track(SearchSettings(bridge_limit=limit)).convert("A.1", from_release=1, to_release=9)
        assert genes(result) == found
        if found:
            assert cand(result, "C.1").bridges == 2
        else:
            assert nothing(result)

    def test_without_bridges_allowed_there_are_none(self, chain):
        """``allow_bridges=False`` is a limit of 0, whatever the settings."""
        assert nothing(chain.track().convert("A.1", from_release=1, to_release=9, allow_bridges=False))


# ----------------------------------------------------------------------------------------- decision 57's scope


def test_a_version_change_without_a_score_is_no_unscored_step():
    """A version change without a score ranks as measured; a step between two ids without a score does not.

    A.1 -> A.2 at 4/5 has no score (Ensembl gives none to a version change it did not measure): it ranks as
    measured; A.1 -> B.1 between two ids without a score is one unscored step.

    Fails if decision 57 is applied to a change of version of one id.
    """
    builder = GraphBuilder(ALL)
    builder.gene("A", [(1, 1, 4), (2, 5, 9)]).gene("B", [(1, 5, 9)]).event("A.1", "B.1", 4, NAN)
    result = builder.track().convert("A.1", from_release=1, to_release=9)
    assert cand(result, "A.2").unscored_steps == 0 and cand(result, "B.1").unscored_steps == 1
    assert genes(result, best=True) == ["A.2"]


# ------------------------------------------------------------------------------- backward across an assembly change


class TestBackwardAcrossAnAssemblyChange:
    """Assembly 37 serves 1 to 4 and 38 serves 5 to 9; Ensembl's history crosses the change at 4/5."""

    @pytest.fixture
    def handoff(self) -> Track:
        """A new version, a split and a merge at the assembly change."""
        builder = GraphBuilder(ALL, assembly_of={r: (37 if r <= 4 else 38) for r in ALL})
        builder.gene("A", [(1, 1, 4), (2, 5, 9)])  # a new version at the change
        builder.gene("S", [(1, 1, 4)]).gene("T", [(1, 5, 9)]).gene("U", [(1, 5, 9)])  # a split at the change
        builder.event("S.1", "T.1", 4, 0.6).event("S.1", "U.1", 4, 0.4)
        builder.gene("M", [(1, 1, 4)]).gene("N", [(1, 1, 4)]).gene("O", [(1, 5, 9)])  # a merge at the change
        builder.event("M.1", "O.1", 4, 0.9).event("N.1", "O.1", 4, 0.9)
        for gene, region, releases in (
            ("A.1", "1", range(1, 5)),
            ("A.2", "1", range(5, 10)),
            ("S.1", "2", range(1, 5)),
            ("T.1", "2", range(5, 10)),
            ("U.1", "2", range(5, 10)),
            ("M.1", "3", range(1, 5)),
            ("N.1", "3", range(1, 5)),
            ("O.1", "3", range(5, 10)),
        ):
            builder.place(gene, region, 100, 200, 1, releases)
        return builder.track()

    @pytest.mark.parametrize("target", [1, 4])
    def test_backward_the_history_is_followed_to_the_older_assembly(self, handoff, target):
        """A.2 back to A.1, T.1 and U.1 back to S.1, O.1 back to M.1 and N.1 (a tie), to 4 (the edge) and to 1.

        Fails if a walk backward does not cross the assembly change, or stops one release short of it.
        """
        assert genes(handoff.convert("A.2", from_release=9, to_release=target)) == ["A.1"]
        assert genes(handoff.convert("T.1", from_release=9, to_release=target)) == ["S.1"]
        assert genes(handoff.convert("U.1", from_release=5, to_release=target)) == ["S.1"]
        merged = handoff.convert("O.1", from_release=9, to_release=target)
        assert sorted(genes(merged, best=True)) == ["M.1", "N.1"] and merged.tie

    def test_forward_and_back_return(self, handoff):
        """A.1 at 2 to 9 is A.2, and A.2 at 9 back to 2 is A.1 (composition of an unchanged id)."""
        there = handoff.convert("A.1", from_release=2, to_release=9)
        assert genes(there, best=True) == ["A.2"]
        back = handoff.convert(genes(there, best=True)[0], from_release=9, to_release=2)
        assert genes(back, best=True) == ["A.1"]


# ------------------------------------------------------------------------------------- the copies' rule (interim)


class TestCopiesBridgeOnlyThroughAnIdOfOneGene:
    """A copy on a patch (1-4) carries ids at the releases a reference gene does; its history ends at 4."""

    @staticmethod
    def graph(second_place: tuple[str, int, int]) -> GraphBuilder:
        """P on a patch carries ``L`` with R (chromosome 1) and a second reference gene Q at ``second_place``."""
        builder = GraphBuilder(ALL)
        builder.gene("P", [(1, 1, 4)]).gene("R", [(1, 1, 9)]).gene("Q", [(1, 1, 9)])
        builder.place("P.1", "CHR_PATCH_1", 100, 200, 1, range(1, 5)).outside("P.1", range(1, 5))
        builder.place("R.1", "1", 100, 200, 1, ALL).place("Q.1", second_place[0], *second_place[1:], 1, ALL)
        for gene in ("P.1", "R.1", "Q.1"):
            builder.xref("L", gene, "UniProtKB Gene Name", range(1, 5) if gene == "P.1" else ALL)
        return builder

    def test_an_id_two_reference_genes_carry_is_no_bridge_from_a_copy(self):
        """R (chromosome 1) and Q (chromosome 2) both carry ``L`` while the copy does: two genes, no bridge.

        Fails if a copy bridges through any id it carried at a release with another gene (the rule before: R and Q
        both answer, one of them a gene elsewhere).
        """
        result = self.graph(("2", 100, 200)).track().convert("P.1", to_release=9, explain=True)
        assert nothing(result)
        assert checks(result) == {
            ("P.1", "Q.1"): ("outside_reference", False, (("L", "copy_id_of_several_genes"),)),
            ("P.1", "R.1"): ("outside_reference", False, (("L", "copy_id_of_several_genes"),)),
        }

    def test_a_holder_outside_the_reference_at_that_release_adds_no_gene(self):
        """A gene that was a copy at the releases both carried the id adds no gene there.

        R (chromosome 1 of assembly 37, 1-4) ends; P lay on a patch at 1 to 4, where both carried ``L``, and on
        chromosome 3 of assembly 38 from 5 (human LYPD4 moved so between GRCh37 and GRCh38).

        At the releases both carried ``L``, P was a copy: ``L`` named one gene there, and R bridges to P. Fails if the
        holders are counted over their whole lives (two genes: their places on the reference are on two assemblies).
        """
        builder = GraphBuilder(ALL, assembly_of={r: (37 if r <= 4 else 38) for r in ALL})
        builder.gene("R", [(1, 1, 4)]).gene("P", [(1, 1, 9)])
        builder.place("R.1", "1", 100, 200, 1, range(1, 5)).place("P.1", "CHR_PATCH_1", 100, 200, 1, range(1, 5))
        builder.outside("P.1", range(1, 5)).place("P.1", "3", 500, 600, 1, range(5, 10))
        builder.xref("L", "R.1", "UniProtKB Gene Name", range(1, 5)).xref("L", "P.1", "UniProtKB Gene Name", ALL)
        result = builder.track().convert("R.1", from_release=2, to_release=9, explain=True)
        assert genes(result) == ["P.1"]
        assert checks(result)[("R.1", "P.1")] == ("outside_reference", True, (("L", "outside_reference"),))

    def test_an_id_of_one_gene_under_two_ids_still_bridges(self):
        """Two ids of one gene that carry the id are one gene: the copy bridges to both.

        Q overlaps R on chromosome 1 (one strand): the two ids are one gene by the one rule of a gene, so ``L`` is
        an id of one gene and the copy bridges to both ids.

        Fails if the holders are counted by stable id instead of by gene.
        """
        result = self.graph(("1", 150, 250)).track().convert("P.1", to_release=9, explain=True)
        assert genes(result) == ["Q.1", "R.1"] or genes(result) == ["R.1", "Q.1"]
        assert {rule for (_, _), (_, taken, through) in checks(result).items() for _, rule in through} == {
            "outside_reference"
        }


# ---------------------------------------------------------------------------------------------- the guards


@pytest.fixture(scope="module")
def world() -> tuple[Track, dict[str, list[int]]]:
    """A graph with versions, an assembly change, a split and a retirement, and the releases of each gene version."""
    builder = GraphBuilder(ALL, assembly_of={r: (37 if r <= 4 else 38) for r in ALL})
    builder.gene("A", [(1, 1, 3), (2, 4, 6), (3, 7, 9)]).gene("B", [(1, 1, 5)]).gene("C", [(1, 6, 9)])
    builder.gene("D", [(1, 2, 9)]).event("B.1", "C.1", 5, 0.7).event("B.1", "D.1", 5, 0.3)
    lives = {"A.1": [1, 2, 3], "A.2": [4, 5, 6], "A.3": [7, 8, 9], "B.1": [1, 2, 3, 4, 5]}
    lives |= {"C.1": [6, 7, 8, 9], "D.1": list(range(2, 10))}
    return builder.track(), lives


class TestGuardsOnAWrittenGraph:
    """Identity, the return of unchanged ids, and composition, on the graph of :py:func:`world`.

    Every gene version at every release where it exists, every pair of releases.
    """

    def test_a_gene_version_at_a_release_where_it_exists_is_itself(self, world):
        """Fails if anything outranks a gene's own version at its own release."""
        track, lives = world
        for version, releases in lives.items():
            for release in releases:
                result = track.convert(version, from_release=release, to_release=release)
                assert genes(result, best=True) == [version], (version, release)

    def test_a_stable_id_that_lives_at_two_releases_is_its_own_version_there_and_returns(self, world):
        """For every stable id and every pair of releases at which it lives: its version there, and back again.

        Fails if a split, a merge or a bridge takes an id that exists at the target release elsewhere.
        """
        track, lives = world
        by_id: dict[str, dict[int, str]] = {}
        for version, releases in lives.items():
            for release in releases:
                by_id.setdefault(version.split(".")[0], {})[release] = version
        for stable_id, at in by_id.items():
            for one in at:
                for other in at:
                    there = track.convert(at[one], from_release=one, to_release=other)
                    assert genes(there, best=True) == [at[other]], (stable_id, one, other)
                    back = track.convert(at[other], from_release=other, to_release=one)
                    assert genes(back, best=True) == [at[one]], (stable_id, other, one)
                    assert genes(track.convert(stable_id, from_release=one, to_release=other), best=True) == [at[other]]


REAL = [organism for organism in ORGANISMS if tables_path(organism).is_file()]


@pytest.mark.parametrize("organism", REAL)
def test_guards_on_the_graph_of_a_real_organism(real_world, organism):
    """The guards on the graphs built from the cut tables of real organisms, judged by Ensembl's gene tables.

    For a seeded sample of genes of the tables: a gene version at a release where the tables have it is itself, and a
    stable id that the tables have at two releases (half of the pairs across an assembly change where there is one)
    converts from one to the other as the version the tables give there, and back. Fails as the tests above, on real
    histories (versions, splits, merges, re-uses, assembly changes). LRG records are left out: an LRG id answers the
    gene its record describes, by design.
    """
    world = real_world(organism)
    api, oracle = world.api, world.ensembl
    serving = api.graph.serving_assembly
    present: dict[str, dict[int, str]] = {}
    for release in api.releases():
        for gene, version in oracle.genes(int(serving[release]), release).items():
            if not ids.LRG_ID.match(gene):  # an LRG record answers the gene it describes (decision 96)
                present.setdefault(gene, {})[release] = f"{gene}.{version}"
    rng = random.Random(115)  # noqa: S311 - a seeded sample of test cases, not a secret
    sample = rng.sample(sorted(present), min(60, len(present)))
    for gene in sample:
        at = present[gene]
        releases = sorted(at)
        release = rng.choice(releases)
        result = api.convert(at[release], from_release=release, to_release=release)
        assert result.targets() == (at[release],), (gene, release)
        one, other = rng.choice(releases), rng.choice(releases)
        across = [r for r in releases if serving[r] != serving[one]]
        if across and rng.random() < 0.5:
            other = rng.choice(across)
        there = api.convert(at[one], from_release=one, to_release=other)
        assert there.targets() == (at[other],), (gene, one, other)
        back = api.convert(at[other], from_release=other, to_release=one)
        assert back.targets() == (at[one],), (gene, other, one)
        assert api.convert(gene, from_release=one, to_release=other).targets() == (at[other],), (gene, one, other)
    assert SNAPSHOT in api.releases()
