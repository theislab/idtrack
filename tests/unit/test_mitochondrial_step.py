"""Decisions 91 and 100 on graphs written by hand: the mitochondrial genome against a nuclear sequence.

A history step between a gene of the mitochondrial genome and a gene of a nuclear sequence, and the two genes that
share a stable id across such a step.

Assembly 37 serves releases 1 to 4 and 38 serves 5 to 9; Ensembl's history crosses the change at 4/5. Each test names
the plausible wrong version it fails under.
"""

from __future__ import annotations

from idtrack.engine.track import _Search
from idtrack.graph.genes import CHROMOSOME, MITOCHONDRIAL, SCAFFOLD
from idtrack.graph.store import _Places
from idtrack.results import BRIDGE_RULES, STEP_REASONS
from tests.unit.test_track import GraphBuilder, genes, nothing

ALL = range(1, 10)
EARLY, LATE = range(1, 5), range(5, 10)


def mask(releases) -> int:
    """The bit mask of releases."""
    return sum(1 << r for r in releases)


def handoff():
    """Three genes across the assembly change, their places of a kind.

    M: mitochondrial at 37, a chromosome-8 gene at 38 (rhesus MT-ND1); N: mitochondrial on both; O: chromosome 2, then
    chromosome 5 (another move of place, which the walk follows: decision 89); S: mitochondrial at 37, a gene of a
    nuclear scaffold at 38 (cynomolgus ND4 from 102); U: mitochondrial at 37, no place at 38.
    """
    builder = GraphBuilder(ALL, assembly_of={r: (37 if r <= 4 else 38) for r in ALL})
    builder.gene("M", [(1, 1, 4), (2, 5, 9)]).gene("N", [(1, 1, 4), (2, 5, 9)]).gene("O", [(1, 1, 4), (2, 5, 9)])
    builder.gene("S", [(1, 1, 4), (2, 5, 9)]).gene("U", [(1, 1, 4), (2, 5, 9)])
    track = builder.track()
    store = builder.graph._frozen()
    index = store.index
    store.places = _Places.of(
        {
            (index["M.1"], 37, "MT", 3259, 4214, 1, MITOCHONDRIAL): mask(EARLY),
            (index["M.2"], 38, "8", 88190642, 88220984, -1, CHROMOSOME): mask(LATE),
            (index["N.1"], 37, "MT", 5000, 6000, 1, MITOCHONDRIAL): mask(EARLY),
            (index["N.2"], 38, "MT", 5000, 6000, 1, MITOCHONDRIAL): mask(LATE),
            (index["O.1"], 37, "2", 100, 200, 1, CHROMOSOME): mask(EARLY),
            (index["O.2"], 38, "5", 100, 200, 1, CHROMOSOME): mask(LATE),
            (index["S.1"], 37, "MT", 10760, 12137, 1, MITOCHONDRIAL): mask(EARLY),
            (index["S.2"], 38, "JAANEP010000702.1", 1000, 2377, 1, SCAFFOLD): mask(LATE),
            (index["U.1"], 37, "MT", 7000, 8000, 1, MITOCHONDRIAL): mask(EARLY),
        }
    )
    store.non_reference = {}
    store.globals.clear()
    return track


def test_a_step_from_the_mitochondrial_genome_to_a_nuclear_gene_is_not_followed():
    """M.1 (mitochondrial) -> M.2 (chromosome 8): no successor across the step, and the explained path shows it refused.

    Fails if the step is followed as an ordinary conversion (M.2 answered, ``converted``).
    """
    track = handoff()
    result = track.convert("M.1", from_release=2, to_release=9, explain=True)
    assert nothing(result)
    refused = result.path.refused
    assert [(s.source, s.target, s.reason, s.places) for s in refused] == [
        ("M.1", "M.2", "mitochondrial_genome", "places_apart")
    ]
    assert "mitochondrial genome" in STEP_REASONS["mitochondrial_genome"]
    back = track.convert("M.2", from_release=9, to_release=2, explain=True)
    assert nothing(back) and [(s.source, s.target) for s in back.path.refused] == [("M.2", "M.1")]


def test_steps_within_the_mitochondrial_genome_and_between_chromosomes_are_followed():
    """N stays on the mitochondrial genome; O moves from chromosome 2 to 5 (decision 89 follows it and marks it).

    Fails if the rule refuses every step across an assembly change or every move of place.
    """
    track = handoff()
    assert genes(track.convert("N.1", from_release=2, to_release=9)) == ["N.2"]
    moved = track.convert("O.1", from_release=2, to_release=9, explain=True)
    assert genes(moved) == ["O.2"] and moved.path.refused == ()


def test_a_step_to_a_gene_of_a_nuclear_scaffold_is_not_followed_across_an_assembly_change():
    """S.1 (mitochondrial at 37) -> S.2 (a nuclear scaffold at 38): refused, though the two places cannot be compared.

    Across an assembly change a scaffold has no name that the next assembly keeps, so the relation of the two places is
    ``other_assemblies``; the kind of the place says the gene is nuclear all the same. Fails if the rule asks the
    places to lie apart besides (the step followed, S.2 answered).
    """
    track = handoff()
    result = track.convert("S.1", from_release=2, to_release=9, explain=True)
    assert nothing(result)
    assert [(s.source, s.target, s.reason, s.places) for s in result.path.refused] == [
        ("S.1", "S.2", "mitochondrial_genome", "other_assemblies")
    ]


def test_a_step_to_a_version_without_a_place_is_followed():
    """U.1 (mitochondrial) -> U.2 (no place at 38): nothing says U.2 is nuclear, so the step is followed.

    Fails if a missing place is read as a nuclear one.
    """
    assert genes(handoff().convert("U.1", from_release=2, to_release=9)) == ["U.2"]


# ------------------------------------------------------------------- an id reused across the two genomes (decision 100)


def reused():
    """Ids that Ensembl gave first to a mitochondrial gene of 37 and then to a nuclear gene of 38, and their kin.

    R (rhesus MT-ND1): R.1 mitochondrial at 1 to 4, R.2 on chromosome 8 at 5 to 9; T, the mitochondrial gene of 38
    (from 6), carries R.1's name ``ND1``. C (cynomolgus ND4): C.1 mitochondrial, C.2 on a nuclear scaffold, both named
    ``ND4``, as is M, the mitochondrial gene of 38. P and Q: a gene of chromosome 3 on both assemblies and another gene
    of chromosome 3 of 38, apart from P there, with P.1's name from 6 (no reuse: the stable id is one gene).
    """
    builder = GraphBuilder(ALL, assembly_of={r: (37 if r <= 4 else 38) for r in ALL})
    builder.gene("R", [(1, 1, 4), (2, 5, 9)]).gene("T", [(1, 6, 9)])
    builder.gene("C", [(1, 1, 4), (2, 5, 9)]).gene("M", [(1, 5, 9)])
    builder.gene("P", [(1, 1, 4), (2, 5, 9)]).gene("Q", [(1, 6, 9)])
    builder.xref("ND1", "R.1", "HGNC Symbol", EARLY).xref("ND1", "T.1", "HGNC Symbol", range(6, 10))
    builder.xref("ND4", "C.1", "UniProtKB Gene Name", EARLY).xref("ND4", "C.2", "UniProtKB Gene Name", LATE)
    builder.xref("ND4", "M.1", "UniProtKB Gene Name", LATE)
    builder.xref("PN", "P.1", "HGNC Symbol", EARLY).xref("PN", "Q.1", "HGNC Symbol", range(6, 10))
    track = builder.track()
    store = builder.graph._frozen()
    index = store.index
    store.places = _Places.of(
        {
            (index["R.1"], 37, "MT", 3259, 4214, 1, MITOCHONDRIAL): mask(EARLY),
            (index["R.2"], 38, "8", 88190642, 88220984, -1, CHROMOSOME): mask(LATE),
            (index["T.1"], 38, "MT", 2700, 3655, 1, MITOCHONDRIAL): mask(range(6, 10)),
            (index["C.1"], 37, "MT", 10760, 12137, 1, MITOCHONDRIAL): mask(EARLY),
            (index["C.2"], 38, "JAANEP010000702.1", 1000, 2377, 1, SCAFFOLD): mask(LATE),
            (index["M.1"], 38, "MT", 10700, 12077, 1, MITOCHONDRIAL): mask(LATE),
            (index["P.1"], 37, "3", 1000, 2000, 1, CHROMOSOME): mask(EARLY),
            (index["P.2"], 38, "3", 1000, 2000, 1, CHROMOSOME): mask(LATE),
            (index["Q.1"], 38, "3", 90000, 91000, 1, CHROMOSOME): mask(range(6, 10)),
        }
    )
    store.non_reference = {}
    store.globals.clear()
    return track


def bridge_rules(result) -> dict:
    """``{(source, target): {id: rule}}`` of the bridges an explained conversion considered."""
    return {(b.source, b.target): dict(b.through) for b in result.path.bridges}


def test_the_mitochondrial_gene_of_the_new_assembly_is_reached_by_its_name():
    """The rhesus shape: R.1's step to R.2 (chromosome 8) is refused, and its name leads to T, the new MT gene.

    The places of R are those of R.1's side (the mitochondrial genome of 37), not R.2's chromosome 8 of 38, so the
    bridge to T is not ``places_apart``. Fails if bridges compare the places of the whole stable id (no answer), and
    forward as backward.
    """
    track = reused()
    forward = track.convert("R.1", from_release=2, to_release=9, explain=True)
    assert genes(forward) == ["T.1"] and bridge_rules(forward)[("R.1", "T.1")] == {"ND1": "never_shared"}
    assert [(s.source, s.target, s.reason) for s in forward.path.refused] == [("R.1", "R.2", "mitochondrial_genome")]
    back = track.convert("T.1", from_release=9, to_release=2, explain=True)
    assert genes(back) == ["R.1"] and bridge_rules(back)[("T.1", "R.1")] == {"ND1": "never_shared"}


def test_a_bridge_between_the_two_genes_of_a_reused_id_is_refused():
    """The cynomolgus shape: C.1 (MT) and C.2 (a nuclear scaffold) share the name ND4; M is the MT gene of 38.

    The bridge from C.1 to its own nuclear version through ND4 is refused like the step (``mitochondrial_genome``),
    the bridge to M is taken: C.1 answers M.1, and back from C.2 nothing leads to C.1. Fails if the shared stable id
    makes the two sides one gene (C.2 answered through the name).
    """
    track = reused()
    forward = track.convert("C.1", from_release=2, to_release=9, explain=True)
    assert genes(forward) == ["M.1"]
    rules = bridge_rules(forward)
    assert rules[("C.1", "C.2")] == {"ND4": "mitochondrial_genome"} and rules[("C.1", "M.1")] == {"ND4": "never_shared"}
    assert "mitochondrial genome" in BRIDGE_RULES["mitochondrial_genome"]
    back = track.convert("C.2", from_release=9, to_release=2, explain=True)
    assert "C.1" not in genes(back) and bridge_rules(back)[("C.2", "C.1")] == {"ND4": "mitochondrial_genome"}


def test_an_id_without_such_a_step_is_one_gene_over_all_its_versions():
    """P is one gene on chromosome 3 of both assemblies; Q lies apart from P.2 on chromosome 3 of 38.

    The stable id has no step between the genomes, so its places are those of every version, as before: the bridge
    from P.1 to Q through PN is ``places_apart``. Fails if the places of one version are compared for every id (the
    bridge taken across the assembly change).
    """
    track = reused()
    result = track.convert("P.1", from_release=2, to_release=9, explain=True)
    assert genes(result) == ["P.2"] and bridge_rules(result).get(("P.1", "Q.1"), {}) in ({}, {"PN": "places_apart"})
    search = _Search(track._reader, "P.1", track.search)
    assert search.sides("P") == {} and search.relation("P.1", "Q.1") == "places_apart"
    assert search.sides("R") == {"R.1": frozenset({"R.1"}), "R.2": frozenset({"R.2"})}
