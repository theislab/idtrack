"""What the graph says of places and families, as conversions read it (I131, I133, I134), on graphs written by hand.

Releases 1 to 9; assembly 37 serves 1 to 4 and 38 serves 5 to 9 (the writer of :py:mod:`tests.unit.test_explained_path`).
Each test names the plausible wrong version it fails under.
"""

from __future__ import annotations

from idtrack import API
from idtrack.graph import genes
from idtrack.graph.store import _Places
from tests.unit.test_explained_path import _Writer

EARLY, LATE = range(1, 5), range(5, 10)


def mask(releases) -> int:
    """The bit mask of releases."""
    return sum(1 << r for r in releases)


def placed(writer: _Writer, places: dict[str, tuple[int, str, int, int, str, range]]) -> API:
    """Freeze the graph with the places ``{gene version: (assembly, region, start, end, kind, releases)}``."""
    graph = writer.graph
    store = graph._frozen()
    store.places = _Places.of(
        {
            (store.index[node], assembly, region, start, end, 1, kind): mask(releases)
            for node, (assembly, region, start, end, kind, releases) in places.items()
        }
    )
    return API(graph)


def retired_and_named_again(database: str, name: str, early_region: str, late_region: str) -> API:
    """``OLD`` lives on assembly 37 and retires; ``NEW`` lives on assembly 38; ``name`` is OLD's, then NEW's."""
    w = _Writer()
    w.gene("OLD", [(1, 1, 4)]).gene("NEW", [(1, 5, 9)])
    w.xref(name, "OLD.1", database, EARLY).xref(name, "NEW.1", database, LATE)
    return placed(
        w,
        {
            "OLD.1": (37, early_region, 1000, 2000, genes.CHROMOSOME, EARLY),
            "NEW.1": (38, late_region, 5000, 6000, genes.CHROMOSOME, LATE),
        },
    )


def genes_of(api: API, query: str) -> list[str]:
    """The genes of the best answer of ``query`` from release 4 to 9."""
    return sorted({c.gene for c in api.convert(query, from_release=4, to_release=9).best})


def test_a_bridge_across_an_assembly_change_to_another_chromosome_is_refused():
    """I133: the pig ``RPL28`` bridge of F16 (a retired gene of chromosome 15, a gene of chromosome 6 in the next assembly).

    Fails if places on two assemblies are never compared (the rule before: the bridge was taken, ``never_shared``).
    The same name on chromosomes of one name is taken as before (the control).
    """
    apart = retired_and_named_again("HGNC Symbol", "RPL28", "15", "6")
    assert genes_of(apart, "OLD.1") == []
    result = apart.convert("OLD.1", from_release=4, to_release=9, explain=True)
    assert [b.places for b in result.path.bridges] == ["places_apart"]
    same = retired_and_named_again("HGNC Symbol", "RPL28", "15", "15")
    assert genes_of(same, "OLD.1") == ["NEW.1"]


def test_an_rfam_accession_is_no_bridge():
    """I133: an Rfam accession names a family of RNA genes; a retired gene does not travel through it.

    Fails if a family accession bridges like a gene's id (pig ENSSSCG00000020242.1 to a gene of another chromosome
    through ``RF00049``). The label of the same family, the display name of each gene, bridges as before.
    """
    accession = retired_and_named_again("RFAM", "RF00049", "14", "14")
    assert genes_of(accession, "OLD.1") == []
    label = retired_and_named_again("RFAM label", "SNORD36", "14", "14")
    assert genes_of(label, "OLD.1") == ["NEW.1"]


def test_a_move_from_the_mitochondrial_genome_to_a_chromosome_is_apart():
    """I134: rhesus ``MT-ND1``: version 1 on the mitochondrial genome of one assembly, version 2 on chromosome 8 of the next.

    The relation of the two versions is ``places_apart``, the region's kind read from the graph, not its name (here
    ``chrM``). Fails if the mitochondrial genome is compared by its name or not at all (``other_assemblies``).
    """
    w = _Writer()
    w.gene("MTND1", [(1, 1, 4), (2, 5, 9)])
    api = placed(
        w,
        {
            "MTND1.1": (37, "chrM", 3259, 4214, genes.MITOCHONDRIAL, EARLY),
            "MTND1.2": (38, "8", 88_190_642, 88_220_984, genes.CHROMOSOME, LATE),
        },
    )
    graph = api.graph
    one, other = (genes.facts_of_nodes(graph, [node]) for node in ("MTND1.1", "MTND1.2"))
    assert genes.relation(one, other) == "places_apart"
    assert genes.lies_on_mitochondrial_genome(graph, "MTND1.1", 3)
    assert not genes.lies_on_mitochondrial_genome(graph, "MTND1.2", 6)
    assert not genes.lies_on_mitochondrial_genome(graph, "MTND1.1", 6)  # it lies there at its own releases only
    assert graph.places("MTND1.1")[0].kind == genes.MITOCHONDRIAL


def test_the_graph_counts_two_genes_after_a_split_whatever_their_shared_ancestor():
    """I131 on a graph: ``A`` splits into ``B`` and ``C``, which live side by side apart: two genes, at 7 and overall.

    Fails if history joins through the shared ancestor (one gene), or if ``count_genes(release=)`` counts ids that do
    not live at the release.
    """
    w = _Writer()
    w.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
    w.event("A.1", "B.1", 4, 0.9).event("A.1", "C.1", 4, 0.8)
    api = placed(
        w,
        {
            "A.1": (37, "7", 1000, 9000, genes.CHROMOSOME, EARLY),
            "B.1": (38, "7", 1000, 2000, genes.CHROMOSOME, LATE),
            "C.1": (38, "7", 8000, 9000, genes.CHROMOSOME, LATE),
        },
    )
    graph = api.graph
    assert graph.count_genes(["A", "B", "C"]) == 2
    assert graph.count_genes(["A", "B"]) == 1
    assert graph.count_genes(["B", "C"], release=7) == 2
    assert graph.count_genes(["A", "B", "C"], release=3) == 1
