#!/usr/bin/env python3
"""Unit tests of :py:class:`idtrack.graph.TheGraph` on graphs made by hand.

The construction of a graph, its reversed view, the range helpers, name lookup by letter case, the rules of
hyper-connectivity and the caches of an empty graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from idtrack import ids
from idtrack.graph import TheGraph
from idtrack.settings import GraphSettings


class TestTheGraphInitialization:
    """Test TheGraph initialization."""

    def test_can_create_empty_graph(self):
        """Test creating an empty TheGraph."""
        graph = TheGraph()
        assert isinstance(graph, nx.MultiDiGraph)
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0

    def test_available_forms_initially_none(self):
        """Test available_forms is None initially."""
        graph = TheGraph()
        assert graph.available_forms is None

    def test_can_initialize_with_data(self):
        """Test initializing with node/edge data."""
        source = nx.MultiDiGraph()
        source.add_node("A", type="test")
        source.add_node("B", type="test")
        source.add_edge("A", "B", weight=1)

        graph = TheGraph(source)
        assert "A" in graph.nodes()
        assert "B" in graph.nodes()
        assert graph.has_edge("A", "B")


class TestReverseGraph:
    """Test reverse graph property."""

    def test_rev_property_returns_reversed_graph(self):
        """Test rev property returns reversed graph."""
        graph = TheGraph()
        graph.add_node("A", **{ids.NODE_TYPE_KEY: "test"})
        graph.add_node("B", **{ids.NODE_TYPE_KEY: "test"})
        graph.add_edge("A", "B")

        rev = graph.rev
        assert rev.has_edge("B", "A")
        assert not rev.has_edge("A", "B")

    def test_rev_is_cached(self):
        """Test rev property is cached."""
        graph = TheGraph()
        graph.add_edge("A", "B")

        rev1 = graph.rev
        rev2 = graph.rev
        # Same object due to caching
        assert rev1 is rev2


class TestListToRanges:
    """Test list_to_ranges static method."""

    def test_empty_list(self):
        """Test empty list returns empty ranges."""
        assert TheGraph.list_to_ranges([]) == []

    def test_single_element(self):
        """Test single element list."""
        assert TheGraph.list_to_ranges([5]) == [[5, 5]]

    def test_consecutive_elements(self):
        """Test consecutive elements form one range."""
        assert TheGraph.list_to_ranges([1, 2, 3, 4, 5]) == [[1, 5]]

    def test_non_consecutive_elements(self):
        """Test non-consecutive elements form multiple ranges."""
        assert TheGraph.list_to_ranges([1, 2, 3, 5, 6, 10]) == [[1, 3], [5, 6], [10, 10]]

    def test_gaps_in_sequence(self):
        """Test handling gaps in sequence."""
        assert TheGraph.list_to_ranges([100, 101, 105, 106, 107]) == [[100, 101], [105, 107]]


class TestCompactRanges:
    """Test compact_ranges static method."""

    def test_empty_ranges(self):
        """Test empty range list."""
        assert TheGraph.compact_ranges([]) == []

    def test_single_range(self):
        """Test single range unchanged."""
        assert TheGraph.compact_ranges([[1, 5]]) == [[1, 5]]

    def test_adjacent_ranges_merge(self):
        """Test adjacent ranges are merged."""
        assert TheGraph.compact_ranges([[1, 3], [4, 6]]) == [[1, 6]]

    def test_overlapping_ranges_merge(self):
        """Test overlapping ranges are merged."""
        assert TheGraph.compact_ranges([[1, 5], [3, 8]]) == [[1, 8]]

    def test_non_adjacent_ranges_preserved(self):
        """Test non-adjacent ranges are preserved."""
        assert TheGraph.compact_ranges([[1, 3], [5, 7]]) == [[1, 3], [5, 7]]


class TestGetIntersectingRanges:
    """Test get_intersecting_ranges static method."""

    def test_no_overlap(self):
        """Test non-overlapping ranges return empty."""
        assert TheGraph.get_intersecting_ranges([[1, 5]], [[10, 15]]) == []

    def test_full_overlap(self):
        """Test fully overlapping ranges."""
        assert TheGraph.get_intersecting_ranges([[1, 10]], [[3, 7]]) == [[3, 7]]

    def test_partial_overlap(self):
        """Test partially overlapping ranges."""
        assert TheGraph.get_intersecting_ranges([[1, 5]], [[3, 8]]) == [[3, 5]]

    def test_multiple_ranges(self):
        """Test multiple ranges with intersections."""
        assert TheGraph.get_intersecting_ranges([[1, 5], [10, 15]], [[3, 12]]) == [[3, 5], [10, 12]]

    def test_no_compact(self):
        """Test without compacting."""
        result = TheGraph.get_intersecting_ranges([[1, 5], [6, 10]], [[1, 10]], compact=False)
        assert len(result) == 2


class TestIsPointInRange:
    """Test is_point_in_range static method."""

    def test_point_in_single_range(self):
        """Test point within single range."""
        assert TheGraph.is_point_in_range([[1, 10]], 5) is True

    def test_point_at_range_start(self):
        """Test point at range start."""
        assert TheGraph.is_point_in_range([[5, 10]], 5) is True

    def test_point_at_range_end(self):
        """Test point at range end."""
        assert TheGraph.is_point_in_range([[5, 10]], 10) is True

    def test_point_outside_range(self):
        """Test point outside range."""
        assert TheGraph.is_point_in_range([[5, 10]], 3) is False
        assert TheGraph.is_point_in_range([[5, 10]], 15) is False

    def test_point_in_multiple_ranges(self):
        """Test point in one of multiple ranges."""
        lor = [[1, 5], [10, 15], [20, 25]]
        assert TheGraph.is_point_in_range(lor, 12) is True
        assert TheGraph.is_point_in_range(lor, 7) is False


class TestResolveName:
    """Test how a user's string is matched to a node of a graph made by hand."""

    def test_exact_match(self):
        """An exact node name is itself."""
        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("ENSG00000141510") == ("ENSG00000141510", "exact")

    def test_case_insensitive_match(self):
        """A name in another letter case is matched and reported as such."""
        graph = TheGraph()
        graph.add_node("ENSG00000141510", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        graph.add_node("TP53", **{ids.NODE_TYPE_KEY: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("tp53") == ("TP53", "case")
        assert graph.resolve_name("ensg00000141510") == ("ENSG00000141510", "case")

    def test_not_found(self):
        """A string that is no node in any spelling matches nothing."""
        graph = TheGraph()
        graph.add_node("EXISTING", **{ids.NODE_TYPE_KEY: "external"})
        graph.graph["genome_assembly"] = 38
        graph.graph["confident_for_release"] = [100, 110]

        assert graph.resolve_name("NONEXISTENT") == (None, None)

    def test_names_equal_in_lower_case_are_reported_by_the_store(self):
        """Two nodes that differ only in letter case are what a build refuses; the store names them."""
        graph = TheGraph()
        graph.add_node("TEST", **{ids.NODE_TYPE_KEY: "external"})
        graph.add_node("Test", **{ids.NODE_TYPE_KEY: "external"})

        assert graph.freeze()._frozen().lower_collisions() == [("TEST", "Test")]


class TestCombined:
    """Test the per-node summary of cross-reference edges."""

    def test_combined_with_mock_graph(self, mock_the_graph):
        """An external id and a base gene id are summed over the edges that leave them, a gene over those that enter."""
        assert mock_the_graph.combined("TP53") == {"HGNC Symbol": {38: {110}}}
        assert mock_the_graph.combined("ENSG00000141510") == {"base_ensembl_gene": {38: {70, 80, 110}}}
        assert set(mock_the_graph.combined("ENSG00000141510.15")) == {"assembly_38_ensembl_gene"}


class TestHyperconnectiveNodes:
    """Test the hyper-connective nodes and the count of genes an id is linked to at one time."""

    def test_empty_for_small_graph(self):
        """Test empty hyperconnective nodes for small graph."""
        graph = TheGraph()
        # Add a few nodes with low degree
        graph.add_node("A", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("B", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        graph.add_edge("A", "B")

        assert len(graph.hyperconnective_nodes) == 0

    def test_id_shared_by_many_genes_at_once_is_hyperconnective(self):
        """An external id linked to more genes than the threshold in one release is hyper-connective."""
        graph = TheGraph(assembly=38, confident_for_release=[100, 101])
        threshold = GraphSettings().hyperconnective_threshold
        for name, genes in (("FAMILY", threshold + 1), ("PAIR", threshold)):
            graph.add_node(name, **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
            for i in range(genes):
                target = f"{name}GENE{i}.1"
                graph.add_node(target, **{ids.NODE_TYPE_KEY: "ensembl_gene"})
                graph.add_edge(name, target, **{ids.CONNECTION_KEY: {"DB1": {38: {100}}}})

        assert graph.hyperconnective_nodes == {"FAMILY": threshold + 1}
        assert graph.genes_linked_at_once("PAIR") == threshold
        assert graph.shared_external_ids == {"FAMILY": threshold + 1, "PAIR": threshold}

    def test_threshold_is_read_when_the_nodes_are_asked_for(self):
        """Another threshold takes effect on a graph that exists already (the counts are kept, not the verdict)."""
        graph = TheGraph(assembly=38, confident_for_release=[100])
        graph.add_node("TWO", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        for i in range(2):
            graph.add_node(f"GENE{i}.1", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
            graph.add_edge("TWO", f"GENE{i}.1", **{ids.CONNECTION_KEY: {"DB1": {38: {100}}}})

        assert graph.hyperconnective_nodes == {}
        graph.graph_settings = GraphSettings(hyperconnective_threshold=1)
        assert graph.hyperconnective_nodes == {"TWO": 2}

    def test_versions_and_years_do_not_make_an_id_hyperconnective(self):
        """Many versions of one gene are one gene; genes of different releases are one at a time."""
        graph = TheGraph(assembly=38, confident_for_release=list(range(60, 100)))
        graph.add_node("SYMBOL", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("FAMILY", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("MOVING", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        for i in range(30):
            for target in (f"GENE.{i + 1}", f"MEMBER{i}.1", f"OTHER{i}.1"):
                graph.add_node(target, **{ids.NODE_TYPE_KEY: "ensembl_gene"})
            graph.add_edge("SYMBOL", f"GENE.{i + 1}", **{ids.CONNECTION_KEY: {"DB1": {38: {60 + i}}}})
            graph.add_edge("FAMILY", f"MEMBER{i}.1", **{ids.CONNECTION_KEY: {"DB1": {38: {70, 71}}}})
            graph.add_edge("MOVING", f"OTHER{i}.1", **{ids.CONNECTION_KEY: {"DB1": {38: {60 + i}}}})

        assert graph.out_degree["SYMBOL"] == graph.out_degree["MOVING"] == 30
        assert graph.genes_linked_at_once("SYMBOL") == 1
        assert graph.genes_linked_at_once("FAMILY") == 30
        assert graph.genes_linked_at_once("MOVING") == 1
        assert graph.hyperconnective_nodes == {"FAMILY": 30}

    def test_a_label_that_named_many_genes_over_its_releases_is_a_family_name(self):
        """Decision 74: a label is a family name when it named more genes than the threshold over its releases.

        ``SCARNA``, a label, names one or two genes at a time, six genes over the releases (mouse ``SCARNA15``:
        9 genes, at most 5 at once, carried a retired id to copies at other loci); the same pattern in a database of
        accessions, and a label of as many genes as the threshold, stay identities: the comparison is the one of the
        count at once (more genes than the threshold).
        """
        threshold = GraphSettings().hyperconnective_threshold
        graph = TheGraph(assembly=38, confident_for_release=list(range(60, 80)), label_databases=["RFAM label"])
        places: dict[str, tuple[str, int, int]] = {}
        releases: dict[str, set[int]] = {}
        for name, database, genes in (
            ("SCARNA", "RFAM label", threshold + 1),
            ("RF00001", "RFAM", threshold + 1),
            ("FEW", "RFAM label", threshold),
        ):
            graph.add_node(name, **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
            for i in range(genes):
                target = f"{name}GENE{i}.1"
                graph.add_node(target, **{ids.NODE_TYPE_KEY: "ensembl_gene"})
                graph.add_edge(name, target, **{ids.CONNECTION_KEY: {database: {38: {60 + 2 * i, 61 + 2 * i}}}})
                # each gene at a locus of its own, living at the releases at which the label names it
                places[target] = ("1", 10_000 * (i + 1), 10_000 * (i + 1) + 500)
                releases[target] = {60 + 2 * i, 61 + 2 * i}
        self.placed(graph, places, set(), releases)

        assert graph.genes_linked_at_once("SCARNA") == 1
        assert graph.label_families == {"SCARNA": threshold + 1, "FEW": threshold}
        assert graph.hyperconnective_nodes == {"SCARNA": threshold + 1}

    @staticmethod
    def placed(
        graph: TheGraph,
        places: dict[str, tuple[str, int, int]],
        outside: set[str],
        releases: dict[str, set[int]] | None = None,
    ) -> TheGraph:
        """Give a graph made by hand the places of its genes (assembly 38, forward strand) and marks.

        ``places``: ``{gene version: (region, start, end)}``; ``outside``: the versions on a patch at every release
        they live; ``releases``: the releases at which each version lies there (a version lives at the releases of
        its places), every release where not given.
        """
        from idtrack.graph.store import _Places
        from idtrack.graph.types import ReleaseSet

        store = graph._frozen()
        every = (1 << 120) - 1

        def mask(node: str) -> int:
            return every if releases is None or node not in releases else sum(1 << r for r in releases[node])

        store.places = _Places.of(
            {
                (store.index[node], 38, region, start, end, 1): mask(node)
                for node, (region, start, end) in places.items()
            }
        )
        store.non_reference = {store.index[node]: ReleaseSet.from_mask(mask(node)) for node in outside}
        return graph

    def test_a_label_of_one_gene_its_copies_and_its_ids_over_time_is_no_family_name(self):
        """Decisions 74 and 73's notion of one gene: copies outside the reference sequence and successive ids are one.

        ``ACTB``-like: one gene of chromosome 1, its five copies on patches (one per release). ``MOVED``: six stable
        ids at one locus, each the next one's predecessor in Ensembl's history (one per release). ``FAMILY``: six
        genes at six loci. Only ``FAMILY`` is a family name. Fails if genes are counted by stable id (the review's 636
        human labels flagged through patch and haplotype copies; ``MOVED`` would be one too).
        """
        threshold = GraphSettings().hyperconnective_threshold
        graph = TheGraph(assembly=38, confident_for_release=list(range(60, 80)), label_databases=["HGNC Symbol"])
        places: dict[str, tuple[str, int, int]] = {}
        releases: dict[str, set[int]] = {}
        outside: set[str] = set()
        for name in ("ACTB", "MOVED", "FAMILY"):
            graph.add_node(name, **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        for i in range(threshold + 1):
            for name in ("ACTB", "MOVED", "FAMILY"):
                gene = f"{name}GENE{i}.1"
                graph.add_node(gene, **{ids.NODE_TYPE_KEY: "ensembl_gene"})
                graph.add_edge(name, gene, **{ids.CONNECTION_KEY: {"HGNC Symbol": {38: {60 + i}}}})
            places[f"ACTBGENE{i}.1"] = ("1", 100, 900) if i == 0 else (f"CHR_PATCH_{i}", 100, 900)
            if i:
                outside.add(f"ACTBGENE{i}.1")
            places[f"MOVEDGENE{i}.1"] = ("2", 10_000 * (i + 1), 10_000 * (i + 1) + 500)
            places[f"FAMILYGENE{i}.1"] = ("3", 10_000 * (i + 1), 10_000 * (i + 1) + 500)
            for name in ("ACTB", "MOVED", "FAMILY"):  # each gene lives at the release at which the label names it
                releases[f"{name}GENE{i}.1"] = {60 + i}
            if i:
                graph.add_edge(f"MOVEDGENE{i - 1}.1", f"MOVEDGENE{i}.1", key=60 + i, weight=0.9, old_release=59 + i,
                               new_release=60 + i)  # fmt: skip
        self.placed(graph, places, outside, releases)

        assert graph.label_families == {"FAMILY": threshold + 1}
        assert graph.count_genes(["ACTBGENE0", "ACTBGENE1"]) == 1
        assert graph.count_genes(["MOVEDGENE0", "MOVEDGENE5"]) == 2  # only the ids counted join: no event between
        assert graph.count_genes(["MOVEDGENE0", "MOVEDGENE1"]) == 1
        assert graph.hyperconnective_nodes == {"FAMILY": threshold + 1}

    def test_the_count_at_once_counts_a_genes_copies(self):
        """The count at once counts stable ids: a gene and its seven haplotype copies carrying one id at once are 8.

        Such an id (the MHC genes' ids on the human haplotypes) stays no bridge, as Ensembl's history gives for an id
        retired without a successor; the count over the releases (decision 74) reads the one rule of a gene. Fails if
        the count at once reads :py:meth:`idtrack.TheGraph.count_genes` (the copies would bridge retired copies to
        other copies) or counts genes at several loci as one.
        """
        graph = TheGraph(assembly=38, confident_for_release=[100], label_databases=["HGNC Symbol"])
        places: dict[str, tuple[str, int, int]] = {}
        outside: set[str] = set()
        for name in ("HLA", "SPREAD"):
            graph.add_node(name, **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        for i in range(8):
            for name in ("HLA", "SPREAD"):
                gene = f"{name}GENE{i}.1"
                graph.add_node(gene, **{ids.NODE_TYPE_KEY: "ensembl_gene"})
                graph.add_edge(name, gene, **{ids.CONNECTION_KEY: {"HGNC Symbol": {38: {100}}}})
            places[f"HLAGENE{i}.1"] = ("6", 100, 900) if i == 0 else (f"CHR_HSCHR6_MHC_{i}", 100, 900)
            if i:
                outside.add(f"HLAGENE{i}.1")
            places[f"SPREADGENE{i}.1"] = ("6", 10_000 * (i + 1), 10_000 * (i + 1) + 500)
        self.placed(graph, places, outside)

        assert graph.genes_linked_at_once("HLA") == 8 and graph.count_genes(graph._gene_masks("HLA")) == 1
        assert graph.genes_linked_at_once("SPREAD") == 8
        assert graph.label_families == {"SPREAD": 8}
        assert graph.hyperconnective_nodes == {"HLA": 8, "SPREAD": 8}

    def test_genes_linked_at_once_counts_a_transcript_as_its_gene(self):
        """An id linked to two transcripts of one gene links one gene."""
        graph = TheGraph(assembly=38, confident_for_release=[100])
        graph.add_node("NM_1", **{ids.NODE_TYPE_KEY: ids.EXTERNAL})
        graph.add_node("GENE.1", **{ids.NODE_TYPE_KEY: "ensembl_gene"})
        for transcript in ("T1.1", "T2.1"):
            graph.add_node(transcript, **{ids.NODE_TYPE_KEY: "ensembl_transcript"})
            graph.add_edge("NM_1", transcript, **{ids.CONNECTION_KEY: {"RefSeq": {38: {100}}}})
            graph.add_edge(transcript, "GENE.1", **{ids.CONNECTION_KEY: {"ensembl_transcript": {38: {100}}}})

        assert graph.genes_linked_at_once("NM_1") == 1


class TestCalculateCaches:
    """Test calculate_caches method."""

    def test_empty_graph_caches(self):
        """Test calculate_caches on empty graph."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 105, 110]

        # An empty graph has nothing to compute and nothing to find; afterwards it is frozen (cannot be changed).
        graph.calculate_caches()
        assert len(graph) == 0
        try:
            graph.add_node("X")
        except nx.NetworkXError:
            frozen = True
        else:
            frozen = False
        assert frozen
        assert graph.resolve_name("anything") == (None, None)
        assert graph.available_external_databases == set()


class TestRangesToList:
    """Test ranges_to_list method."""

    def test_single_range(self):
        """Test single range expansion."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 110]
        assert graph.ranges_to_list([[1, 5]]) == [1, 2, 3, 4, 5]

    def test_multiple_ranges(self):
        """Test multiple range expansion."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 110]
        assert graph.ranges_to_list([[1, 3], [5, 7]]) == [1, 2, 3, 5, 6, 7]

    def test_infinite_upper_bound(self):
        """Test infinite upper bound uses max confident release."""
        graph = TheGraph()
        graph.graph["confident_for_release"] = [100, 105, 110]

        result = graph.ranges_to_list([[108, np.inf]])
        assert 108 in result
        assert 109 in result
        assert 110 in result
