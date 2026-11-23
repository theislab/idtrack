"""Decision 103 and I148: places carried across an assembly change by Ensembl's mapping, copies at their region.

Hand-made chain files and facts: :py:mod:`idtrack.build.assembly_chain`, :py:func:`idtrack.graph.genes.relation`,
the builder's carrying (:py:func:`idtrack.build.builder._carried_places`) and the manager's table of assembly
exceptions (:py:meth:`idtrack.DatabaseManager.create_assembly_exceptions`).
"""

from __future__ import annotations

import pandas as pd

from idtrack.build import builder
from idtrack.build.assembly_chain import AssemblyChain, parse_chain
from idtrack.graph import genes
from idtrack.graph.genes import facts_of, relation

#: Assembly 1 maps onto assembly 2: chromosome 1's bases 0-1000 lie at 5000-6000 of chromosome 1 (forward), its
#: bases 2000-3000 at chromosome 7's 100-1100 (forward coordinates) on its reverse strand, a sequence of 10,000 bases.
CHAIN = """chain 1000 1 100000 + 0 1000 1 100000 + 5000 6000 1
1000

chain 900 1 100000 + 2000 3000 7 10000 - 8900 9900 2
1000
"""


def _chain() -> AssemblyChain:
    return AssemblyChain(parse_chain(CHAIN.splitlines()))


def at(assembly: int, region: str, start: int, end: int, strand: int = 1, releases=(10,), outside: int = 0, **extra):
    """The facts of a version that lies at one place at some releases."""
    mask = sum(1 << r for r in releases)
    return facts_of([(assembly, region, start, end, strand, mask, genes.CHROMOSOME)], outside, **extra)


class TestChain:
    """Reading and carrying by a chain file (:py:class:`idtrack.build.assembly_chain.AssemblyChain`)."""

    def test_carry_forward_and_reverse(self):
        """A place is carried by its aligned bases; on a block of the reverse strand its strand turns.

        Fails if the reverse strand's coordinates are not counted from the end of the sequence, or the strand kept.
        """
        chain = _chain()
        assert chain.carry("1", 101, 200, 1) == [("1", 5101, 5200, 1)]
        assert chain.carry("chr1", 101, 200, -1) == [("1", 5101, 5200, -1)]
        # bases 2101-2200 of 1: 100 bases into the block, which lies at 8900-9900 counted from the end of 7 (10,000
        # bases): 9000-9100 from the end, 900-1000 from the start (forward strand, 1-based 901-1000)
        assert chain.carry("1", 2101, 2200, 1) == [("7", 901, 1000, -1)]
        assert chain.carry("1", 1201, 1300, 1) == []  # no aligned base
        assert chain.carry("2", 1, 10, 1) == []

    def test_a_place_over_two_blocks_spans_their_bases(self):
        """A place whose bases lie in two blocks on one region spans from the first to the last carried base."""
        text = "chain 1 1 100 + 0 100 1 100 + 0 120 1\n40 10 30\n50\n"
        chain = AssemblyChain(parse_chain(text.splitlines()))
        assert chain.carry("1", 31, 60, 1) == [("1", 31, 80, 1)]


class TestRelationThroughTheMapping:
    """``genes.relation`` compares a carried place with the places of the later assembly by overlap."""

    def test_a_bridge_the_mapping_puts_elsewhere_is_apart(self):
        """L-bridges: one chromosome name, but the old gene's place carried lies 500 kb from the gene entered.

        Fails if two genes on chromosomes of one name across an assembly change are compared by the name only
        (``other_assemblies``, the bridge taken).
        """
        old = at(1, "X", 30_479_508, 30_480_285, carried=[(2, "X", 32_159_591, 32_160_368, 1)])
        new = at(2, "X", 31_618_216, 31_618_920, -1, releases=(20,))
        assert relation(old, new) == "places_apart"
        assert relation(new, old) == "places_apart"

    def test_a_bridge_the_mapping_puts_on_the_gene_is_one_place(self):
        """The carried place overlaps the later gene on one strand: ``same_place``. Fails if carried places are unread."""
        old = at(1, "1", 100, 200, carried=[(2, "1", 5100, 5200, 1)])
        new = at(2, "1", 5150, 5400, releases=(20,))
        assert relation(old, new) == "same_place"

    def test_a_place_the_mapping_does_not_carry_keeps_the_names_rule(self):
        """Where Ensembl publishes no mapping (or the place is not aligned), the names rule stays.

        Two chromosomes of one name stay comparable only by name (``other_assemblies``) and two of different names are
        apart. Fails if a missing mapping refuses.
        """
        old = at(1, "1", 100, 200)
        assert relation(old, at(2, "1", 900_000, 900_100, releases=(20,))) == "other_assemblies"
        assert relation(old, at(2, "6", 100, 200, releases=(20,))) == "places_apart"


class TestCopiesAtTheirRegion:
    """I148: a copy on a patch or a haplotype stands for its assembly exception's region of the reference."""

    EXCEPTIONS = {
        (1, "CHR_PATCH"): ((49_000_000, 49_600_000, 1 << 10, "Y", 49_216_909, 49_556_418, genes.CHROMOSOME, ()),),
        (1, "HSCHR17_1"): (
            (
                1,
                1_000_000,
                1 << 10,
                "17",
                43_000_000,
                44_000_000,
                genes.CHROMOSOME,
                ((2, "17", 45_000_000, 46_000_000),),
            ),
        ),
    }

    def copy(self, region: str, start: int, end: int):
        """The facts of a copy at a place of a region outside the reference sequence, at release 10."""
        return at(1, region, start, end, outside=1 << 10, exceptions=self.EXCEPTIONS)

    def test_a_copy_and_a_gene_outside_its_region_are_apart(self):
        """H5: the copy on CHR_MG4213_PATCH stands for Y:49.2-49.6 Mb; a Y gene 29 Mb away is apart.

        Fails if a copy is compared with nothing (``outside_reference``: the interim rule takes the bridge).
        """
        copy = self.copy("CHR_PATCH", 49_392_584, 49_394_900)
        assert copy.regions == {1: (("Y", 49_216_909, 49_556_418),)}
        assert relation(copy, at(1, "Y", 20_000_000, 20_001_000)) == "places_apart"
        assert relation(at(1, "Y", 20_000_000, 20_001_000), copy) == "places_apart"

    def test_inside_its_region_the_interim_rule_stays(self):
        """A gene inside the region (either strand) is not told apart: ``outside_reference`` (round 1's interim rule)."""
        copy = self.copy("CHR_PATCH", 49_392_584, 49_394_900)
        assert relation(copy, at(1, "Y", 49_300_000, 49_301_000, -1)) == "outside_reference"

    def test_across_an_assembly_change_through_the_carried_region_or_the_names(self):
        """The human GRCh37 haplotype copy of 17q21.31 and a GRCh38 gene of chromosome 11 are apart.

        Through the region carried onto the later assembly where the mapping carries it (a gene there outside it is
        apart, one inside not), else by the names of whole sequences. Fails if a copy is compared on its own assembly
        only.
        """
        copy = self.copy("HSCHR17_1", 500, 900)
        assert copy.regions[2] == (("17", 45_000_000, 46_000_000),)
        assert relation(copy, at(2, "11", 100, 200, releases=(20,))) == "places_apart"
        assert relation(copy, at(2, "17", 45_500_000, 45_500_100, releases=(20,))) == "outside_reference"
        uncarried = at(1, "CHR_PATCH", 49_392_584, 49_394_900, outside=1 << 10, exceptions=self.EXCEPTIONS)
        assert relation(uncarried, at(2, "11", 100, 200, releases=(20,))) == "places_apart"
        assert relation(uncarried, at(2, "Y", 100, 200, releases=(20,))) == "outside_reference"

    def test_a_copy_without_an_exception_keeps_the_interim_rule(self):
        """An LRG record (no assembly exception) is compared with nothing: ``outside_reference``."""
        lrg = at(1, "LRG_1", 1, 100, outside=1 << 10, exceptions=self.EXCEPTIONS)
        assert not lrg.regions
        assert relation(lrg, at(1, "Y", 20_000_000, 20_001_000)) == "outside_reference"


class TestCarryingAtBuild:
    """The builder carries every place of the reference sequence onto every later assembly it can."""

    def test_places_are_carried_onto_every_later_assembly_through_published_steps(self):
        """Mouse: a place of NCBIM37 is carried onto GRCm39 through GRCm38 (no file maps NCBIM37 onto GRCm39).

        NCBIM37 onto GRCm38 and GRCm38 onto GRCm39 are published. A place outside the reference sequence is not
        carried.

        Fails if only pairs with a file of their own are carried, or copies' places are carried as genes' places.
        """
        step = "chain 1 1 100000 + 0 5000 1 100000 + 1000 6000 1\n5000\n"
        chains = {
            (37, 38): AssemblyChain(parse_chain(step.splitlines())),
            (38, 39): AssemblyChain(parse_chain(step.splitlines())),
        }
        serving = {48: 37, 68: 38, 103: 39}
        places = {
            (0, 37, "1", 101, 200, 1, genes.CHROMOSOME): 1 << 48,
            (1, 37, "PATCH", 101, 200, 1, genes.SCAFFOLD): 1 << 48,
        }
        carried = builder._carried_places(places, {1: 1 << 48}, chains, serving)
        assert carried == {
            (0, 38, "1", 1101, 1200, 1, genes.SCAFFOLD): 1 << 48,
            (0, 39, "1", 2101, 2200, 1, genes.SCAFFOLD): 1 << 48,
        }


def test_the_assembly_exceptions_of_a_release(tmp_path, monkeypatch):
    """``create_assembly_exceptions`` names both regions and the reference region's kind (I148)."""
    from idtrack.build.database_manager import DatabaseManager

    tables = {
        "assembly_exception": pd.DataFrame(
            {
                "seq_region_id": [2, 3],
                "seq_region_start": [1, 10_001],
                "seq_region_end": [500, 2_781_479],
                "exc_type": ["PATCH_FIX", "PAR"],
                "exc_seq_region_id": [1, 4],
                "exc_seq_region_start": [1000, 10_001],
                "exc_seq_region_end": [1600, 2_781_479],
            }
        ),
        "seq_region": pd.DataFrame(
            {"seq_region_id": [1, 2, 3, 4], "name": ["7", "CHR_HSCHR7_1_CTG1", "Y", "X"], "coord_system_id": [1] * 4}
        ),
        "coord_system": pd.DataFrame({"coord_system_id": [1], "name": ["chromosome"]}),
        "attrib_type": pd.DataFrame({"attrib_type_id": [16], "code": ["non_ref"]}),
        "seq_region_attrib": pd.DataFrame({"seq_region_id": [2], "attrib_type_id": [16]}),
    }

    def get_table(self, table_key, usecols=None, **kwargs):
        frame = tables[table_key]
        return frame if usecols is None else frame[list(usecols)]

    monkeypatch.setattr(DatabaseManager, "get_table", get_table)
    manager = DatabaseManager.__new__(DatabaseManager)
    manager.organism, manager.release, manager.store_raw_always = "homo_sapiens", 110, False
    found = manager.create_assembly_exceptions()
    assert found.to_dict("records") == [
        {
            "seq_region": "CHR_HSCHR7_1_CTG1",
            "start": 1,
            "end": 500,
            "type": "PATCH_FIX",
            "ref_region": "7",
            "ref_start": 1000,
            "ref_end": 1600,
            "ref_kind": genes.CHROMOSOME,
        },
        {
            "seq_region": "Y",
            "start": 10_001,
            "end": 2_781_479,
            "type": "PAR",
            "ref_region": "X",
            "ref_start": 10_001,
            "ref_end": 2_781_479,
            "ref_kind": genes.CHROMOSOME,
        },
    ]


def test_a_built_graph_keeps_carried_places_exceptions_and_display_databases(monkeypatch, tmp_path):
    """The builder carries GRCh37 places onto GRCh38 by the mapping, reads the exceptions; the file keeps both.

    ENSG1 lives on GRCh37 (74, 75) at chromosome 1, 1000-1500; the hand-made mapping puts 1's bases 0-10,000 at
    100,000-110,000 of GRCh38's 1. ENSG5 lies on the patch CHR_PATCH_1 at 77, whose exception stands for 1:5,000-6,000.
    Fails if the graph does not carry places, does not keep the exceptions, or loses either (or the record of the
    mapping, or the database of a display mark) in its file.
    """
    import os

    from idtrack.build.builder import GraphMaker
    from idtrack.build.database_manager import DatabaseManager
    from idtrack.graph import TheGraph
    from tests.unit.test_graph_truth import World, lifetimes

    class Mapped(World):
        def table(self, manager, table_key):
            if table_key == "assembly_exception":
                rows = [(2, 1, 9_000, "PATCH_FIX", 1, 5_000, 6_000)] if int(manager.assembly) == 38 else []
                return pd.DataFrame(rows, columns=list(ASSEMBLY_EXCEPTION_COLUMNS))
            return super().table(manager, table_key)

    from idtrack.build.database_manager import ASSEMBLY_EXCEPTION_COLUMNS

    world = Mapped(
        "homo_sapiens",
        {
            37: lifetimes({"ENSG1": [(74, 75, 1)], "ENSG9": [(74, 77, 1)]}),
            38: lifetimes({"ENSG2": [(76, 77, 1)], "ENSG5": [(77, 77, 1)], "ENSG9": [(76, 77, 1)]}),
        },
        non_reference={38: {77: ["ENSG5"]}},
        places={"ENSG1": (1000, 1500), "ENSG2": (101_200, 101_300), "ENSG5": (100, 200), "ENSG9": (50_000, 50_100)},
        externals={77: [("ENSG2.1", "NAME2", "HGNC Symbol", 38, {"display": True})]},
    )
    blocks = pd.DataFrame(
        [("1", 0, 10_000, "1", 100_000, 1)], columns=["region", "start", "end", "onto", "onto_start", "strand"]
    )
    info = {"file": "GRCh37_to_GRCh38.chain.gz", "from": 37, "to": 38, "release": 77, "sha256": "0" * 64}
    monkeypatch.setattr(
        DatabaseManager, "assembly_chain", lambda self, a, b: (blocks, info) if (a, b) == (37, 38) else None
    )
    world.install(monkeypatch)
    maker = GraphMaker(world.manager(tmp_path, 38))
    graph = maker.construct_graph(forms=["gene"])
    carried = graph.carried_places("ENSG1.1")
    assert [(p.assembly, p.region, p.start, p.end, p.strand) for p in carried] == [(38, "1", 101_000, 101_500, 1)]
    assert [
        (e.region, e.ref_region, e.ref_start, e.ref_end, sorted(e.releases)) for e in graph.assembly_exceptions
    ] == [("CHR_PATCH_1", "1", 5_000, 6_000, [76, 77])]
    assert graph.record.assembly_mappings == {
        "GRCh37_to_GRCh38.chain.gz": {k: info[k] for k in ("from", "to", "release", "sha256")}
    }
    facts = genes.facts_of_nodes(graph, ["ENSG1.1"]), genes.facts_of_nodes(graph, ["ENSG2.1"])
    assert relation(*facts) == "same_place"
    copy = genes.facts_of_nodes(graph, ["ENSG5.1"])
    assert copy.regions == {38: (("1", 5_000, 6_000),)}
    assert relation(copy, genes.facts_of_nodes(graph, ["ENSG9.1"])) == "places_apart"
    path = os.path.join(str(tmp_path), "mapped.npz")
    graph.save(path)
    loaded = TheGraph.load(path)
    assert loaded.carried_places("ENSG1.1") == carried
    assert loaded.assembly_exceptions == graph.assembly_exceptions
    assert loaded.record == graph.record
    store = loaded._frozen()
    node = store.index["NAME2"]
    assert [(store.names[t], db) for t, db, _v, _m in store.displays.of_node(node)] == [("ENSG2.1", "HGNC Symbol")]
