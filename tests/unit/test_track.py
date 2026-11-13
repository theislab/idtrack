#!/usr/bin/env python3
"""Behaviour tests of the conversion engine (``idtrack._track.Track``) on small hand-written graphs.

Every test builds a graph in the layout of a real one (history edges with ``old_release``, ``new_release`` and
``weight``; cross-reference edges with ``connection`` and ``available_releases``) and asserts the ids that a
conversion returns. No network, no files.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from idtrack._the_graph import TheGraph
from idtrack._track import InactiveAtReleaseError, SearchBudgetExceededError, Track

NAN = float("nan")
INF = float("inf")


class GraphBuilder:
    """Write a small graph by hand: genes with their lives, history events and cross-references.

    The organism is human, whose table knows the assemblies 37 (older, priority 2) and 38 (primary, priority 1).
    ``assembly_of`` maps a release to the assembly that is primary at it; by default every release is on 38.
    """

    def __init__(self, releases: range | list[int], assembly_of: dict[int, int] | None = None):
        """Start an empty graph over the given releases."""
        self.releases = list(releases)
        self.first, self.last = self.releases[0], self.releases[-1]
        self.assembly_of = assembly_of or {r: 38 for r in self.releases}
        self.graph = TheGraph()
        self.graph.graph.update(
            {
                "organism": "homo_sapiens",
                "genome_assembly": 38,
                "ensembl_release": self.last,
                "confident_for_release": list(self.releases),
                "narrow": True,
                "narrow_external": True,
                "misplaced_external_entry": set(),
            }
        )
        self.graph.available_forms = ["gene", "transcript", "translation"]

    def _connection(self, database: str, releases: list[int], assembly: int | None) -> dict:
        per_assembly: dict[int, set] = {}
        for r in releases:
            per_assembly.setdefault(assembly if assembly is not None else self.assembly_of[r], set()).add(r)
        return {database: per_assembly}

    def _xref_edge(self, source: str, target: str, database: str, releases: list[int], assembly: int | None):
        connection = self._connection(database, releases, assembly)
        if self.graph.has_edge(source, target, 0):
            data = self.graph[source][target][0]
            for assembly_key, release_set in connection[database].items():
                data["connection"].setdefault(database, {}).setdefault(assembly_key, set()).update(release_set)
            data["available_releases"] = set(data["available_releases"]) | set(releases)
        else:
            self.graph.add_edge(source, target, 0, connection=connection, available_releases=set(releases))

    def _ensembl_node(self, name: str, node_type: str) -> None:
        stable_id, version = name.rsplit(".", 1)
        self.graph.add_node(name, node_type=node_type, ID=stable_id, Version=version, is_latest=False)

    def gene(self, stable_id: str, lives: list[tuple], scores: dict | None = None) -> GraphBuilder:
        """Add a gene id with its versions.

        Args:
            stable_id: The id without version.
            lives: ``(version, first_release, last_release)`` in the order of time. Consecutive lives are joined
                by a version change, a gap between two lives by a retirement and a re-birth.
            scores: Mapping score of the version change that ends the life of a version, by version.

        Returns:
            The builder.
        """
        scores = scores or {}
        void, retired = f"{stable_id}.Void", f"{stable_id}.Retired"
        self._ensembl_node(void, "ensembl_gene")
        self.graph.add_node(stable_id, node_type="base_ensembl_gene")
        previous: tuple | None = None
        for version, first, last in lives:
            node = f"{stable_id}.{version}"
            if node not in self.graph:
                self._ensembl_node(node, "ensembl_gene")
            self._xref_edge(stable_id, node, "base_ensembl_gene", list(range(first, last + 1)), None)
            if previous is None:
                born = first if first == self.first else first - 1
                self.graph.add_edge(void, node, weight=NAN, old_release=born, new_release=first)
            elif previous[2] + 1 == first:
                weight = scores.get(previous[0], NAN)
                self.graph.add_edge(
                    f"{stable_id}.{previous[0]}", node, weight=weight, old_release=previous[2], new_release=first
                )
            else:
                if retired not in self.graph:
                    self._ensembl_node(retired, "ensembl_gene")
                self.graph.add_edge(
                    f"{stable_id}.{previous[0]}",
                    retired,
                    weight=NAN,
                    old_release=previous[2],
                    new_release=previous[2] + 1,
                )
                self.graph.add_edge(retired, node, weight=NAN, old_release=first - 1, new_release=first)
            previous = (version, first, last)
        assert previous is not None
        node = f"{stable_id}.{previous[0]}"
        if previous[2] == self.last:
            self.graph.add_edge(node, node, weight=1.0, old_release=previous[1], new_release=INF)
        else:
            if retired not in self.graph:
                self._ensembl_node(retired, "ensembl_gene")
            self.graph.add_edge(node, retired, weight=NAN, old_release=previous[2], new_release=previous[2] + 1)
        return self

    def event(self, source: str, target: str, old_release: int, weight: float) -> GraphBuilder:
        """Add a history event between two gene nodes (a merge, split or change of id; a scored self loop)."""
        self.graph.add_edge(source, target, weight=weight, old_release=old_release, new_release=old_release + 1)
        return self

    def xref(
        self, external: str, target: str, database: str, releases: range | list[int], assembly: int | None = None
    ) -> GraphBuilder:
        """Attach an external id to a gene, transcript or translation at the given releases."""
        if external not in self.graph:
            self.graph.add_node(external, node_type="external")
        self._xref_edge(external, target, database, list(releases), assembly)
        return self

    def transcript(self, name: str, gene: str, releases: range | list[int]) -> GraphBuilder:
        """Add a transcript of a gene."""
        if name not in self.graph:
            self._ensembl_node(name, "ensembl_transcript")
        self._xref_edge(name, gene, "ensembl_transcript", list(releases), None)
        return self

    def translation(self, name: str, transcript: str, releases: range | list[int]) -> GraphBuilder:
        """Add a translation of a transcript."""
        if name not in self.graph:
            self._ensembl_node(name, "ensembl_translation")
        self._xref_edge(name, transcript, "ensembl_translation", list(releases), None)
        return self

    def track(self) -> Track:
        """Return a ``Track`` on the graph, without a database manager."""
        return Track.from_graph(self.graph)


def targets(result: dict | None) -> list[str]:
    """Return the target ids of a conversion in the order of the result."""
    if result is None:
        return []
    return [target for gene in result for target in result[gene]["final_conversion"]["final_elements"]]


def genes(result: dict | None) -> list[str]:
    """Return the genes of a conversion in the order of the result."""
    return [] if result is None else list(result)


@pytest.fixture
def moved_symbol() -> Track:
    """Symbol S on gene A in releases 1-3 and on gene B in releases 5-9; both genes alive throughout."""
    builder = GraphBuilder(range(1, 10))
    builder.gene("A", [(1, 1, 9)]).gene("B", [(1, 1, 9)])
    builder.xref("S", "A.1", "HGNC Symbol", range(1, 4)).xref("S", "B.1", "HGNC Symbol", range(5, 10))
    return builder.track()


class TestStartingRelease:
    """The release a query is read at (tracking-2, tracking-6, tracking-new-2, accuracy-2)."""

    def test_external_id_active_at_target_is_read_there(self, moved_symbol):
        """External id active at target is read there."""
        assert genes(moved_symbol.convert("S", to_release=9)) == ["B.1"]
        assert genes(moved_symbol.convert("S", to_release=7)) == ["B.1"]
        assert genes(moved_symbol.convert("S", to_release=2)) == ["A.1"]

    def test_external_id_not_active_at_target_is_read_at_its_latest_release(self, moved_symbol):
        """External id not active at target is read at its latest release."""
        result = moved_symbol.convert("S", to_release=4)
        assert genes(result) == ["B.1"]
        assert result["B.1"]["from_release"] == 9
        assert result["B.1"]["from_release_inferred"] is True

    def test_default_equals_from_release_at_target(self, moved_symbol):
        """Default equals from release at target."""
        for release in (1, 2, 3, 5, 6, 7, 8, 9):
            default = moved_symbol.convert("S", to_release=release)
            explicit = moved_symbol.convert("S", from_release=release, to_release=release)
            assert genes(default) == genes(explicit)

    def test_given_release_is_used(self, moved_symbol):
        """Given release is used."""
        result = moved_symbol.convert("S", from_release=2, to_release=9)
        assert genes(result) == ["A.1"]
        assert result["A.1"]["from_release"] == 2
        assert result["A.1"]["from_release_inferred"] is False

    def test_inactive_from_release_is_an_error_for_external_ids(self, moved_symbol):
        """Inactive from release is an error for external ids."""
        with pytest.raises(InactiveAtReleaseError) as error:
            moved_symbol.convert("S", from_release=4, to_release=9)
        assert error.value.active_ranges == [[1, 3], [5, 9]]
        assert "1-3" in str(error.value) and "5-9" in str(error.value)

    def test_inactive_from_release_is_an_error_for_gene_ids_in_both_directions(self):
        """Inactive from release is an error for gene ids in both directions."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 3, 5)]).gene("B", [(1, 1, 9)])
        track = builder.track()
        for from_release, to_release in ((8, 7), (8, 9), (1, 9), (7, 2)):
            with pytest.raises(InactiveAtReleaseError):
                track.convert("A.1", from_release=from_release, to_release=to_release)
            with pytest.raises(InactiveAtReleaseError):
                track.convert("A", from_release=from_release, to_release=to_release)
        assert genes(track.convert("A.1", from_release=4, to_release=5)) == ["A.1"]

    def test_gene_not_at_target_is_read_at_its_latest_release(self):
        """Gene not at target is read at its latest release."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 3), (2, 4, 9)], scores={1: 0.9})
        track = builder.track()
        result = track.convert("A.1", to_release=9)
        assert genes(result) == ["A.2"]
        assert result["A.2"]["from_release"] == 3
        assert result["A.2"]["score"] == pytest.approx(0.9)
        back = track.convert("A.2", to_release=2)
        assert genes(back) == ["A.1"]
        assert back["A.1"]["from_release"] == 9


class TestUnversionedGeneId:
    """A gene id without version stands for its own versions only (accuracy-1, api_harmonize-44, -45)."""

    @pytest.fixture
    def track(self) -> Track:
        """Return the graph of this class as a ``Track``."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4), (2, 5, 9)], scores={1: 0.8})
        builder.gene("B", [(1, 1, 9)])
        # A and B share an alias at every release, and B is the better annotated gene.
        builder.xref("ALIAS", "A.1", "HGNC Symbol", range(1, 5)).xref("ALIAS", "A.2", "HGNC Symbol", range(5, 10))
        builder.xref("ALIAS", "B.1", "HGNC Symbol", range(1, 10))
        for name in ("X1", "X2", "X3"):
            builder.xref(name, "B.1", "EntrezGene", range(1, 10))
        return builder.track()

    def test_current_id_converts_to_its_own_version(self, track):
        """Current id converts to its own version."""
        for strategy in (True, False):
            assert genes(track.convert("A", to_release=9, prioritize_to_one_filter=strategy)) == ["A.2"]
            assert genes(track.convert("A", to_release=3, prioritize_to_one_filter=strategy)) == ["A.1"]

    def test_with_release_equals_versioned_id(self, track):
        """With release equals versioned id."""
        assert genes(track.convert("A", from_release=3, to_release=9)) == ["A.2"]
        assert genes(track.convert("A.1", from_release=3, to_release=9)) == ["A.2"]
        assert genes(track.convert("A", from_release=9, to_release=2)) == ["A.1"]

    @pytest.fixture
    def transcripts(self) -> Track:
        """Gene A with versions 1 (releases 1-4) and 2 (5-9); transcript AT likewise; one translation AP.1."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4), (2, 5, 9)], scores={1: 0.8})
        builder.transcript("AT.1", "A.1", range(1, 5)).transcript("AT.2", "A.2", range(5, 10))
        builder.translation("AP.1", "AT.1", range(1, 5)).translation("AP.1", "AT.2", range(5, 10))
        builder.gene("B", [(1, 1, 9)]).transcript("BT.7", "B.1", range(1, 4))
        return builder.track()

    def test_transcript_and_translation_ids_without_version_stand_for_their_version(self, transcripts):
        """A transcript or translation id without version has no node; it is read as its version (finding 11)."""
        assert "AT" not in transcripts.graph and "AP" not in transcripts.graph
        result = transcripts.convert("AT")
        assert genes(result) == ["A.2"]
        assert result["A.2"]["from_id"] == "AT.2" and result["A.2"]["from_release"] == 9
        assert result["A.2"]["from_release_inferred"] is True
        result = transcripts.convert("AT", to_release=3)
        assert genes(result) == ["A.1"]
        assert result["A.1"]["from_id"] == "AT.1" and result["A.1"]["from_release"] == 3
        result = transcripts.convert("AP", from_release=2, to_release=9)
        assert genes(result) == ["A.2"]
        assert result["A.2"]["from_id"] == "AP.1" and result["A.2"]["from_release_inferred"] is False

    def test_id_without_version_equals_its_version_of_the_starting_release(self, transcripts):
        """The answer is the one of the versioned id read at the same release."""
        for start, target in ((2, 9), (7, 2), (9, 9), (1, 1)):
            versioned = "AT.1" if start <= 4 else "AT.2"
            assert transcripts.convert("AT", from_release=start, to_release=target) == transcripts.convert(
                versioned, from_release=start, to_release=target
            )

    def test_id_without_version_not_at_the_target_is_read_at_its_latest_release(self, transcripts):
        """Decision 1 for an id without version: the latest release at which one of its versions is active."""
        result = transcripts.convert("BT", to_release=9)
        assert genes(result) == ["B.1"]
        assert result["B.1"]["from_id"] == "BT.7" and result["B.1"]["from_release"] == 3

    def test_inactive_from_release_of_an_id_without_version_names_the_releases_of_all_versions(self, transcripts):
        """The error names the id as given and the releases of every version."""
        with pytest.raises(InactiveAtReleaseError) as caught:
            transcripts.convert("BT", from_release=6, to_release=9)
        assert caught.value.identifier == "BT" and caught.value.active_ranges == [[1, 3]]

    def test_unknown_id_without_version_and_external_reading_of_an_ensembl_id(self, transcripts):
        """A prefix of an id is not an id; an Ensembl id is not an id of an external database."""
        assert transcripts.convert("A.") is None
        assert transcripts.convert("ATX") is None
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)]).transcript("AT.1", "A.1", range(1, 10))
        builder.xref("S", "A.1", "HGNC Symbol", range(1, 10))
        assert builder.track().convert("AT", from_database="HGNC Symbol") is None

    def test_archive_assembly_does_not_keep_a_retired_id_active(self):
        """accuracy-new-2: the id's gene is retired on the primary line, an archive assembly still lists it."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)])
        builder.event("A.1", "B.1", 4, 0.95)
        builder.graph["A"]["A.1"][0]["connection"]["base_ensembl_gene"][37] = set(range(5, 10))
        builder.graph["A"]["A.1"][0]["available_releases"] = set(range(1, 10))
        builder.xref("OTHER", "A.1", "HGNC Symbol", range(5, 10), assembly=37)
        builder.gene("C", [(1, 1, 9)]).xref("OTHER", "C.1", "HGNC Symbol", range(5, 10))
        track = builder.track()
        result = track.convert("A", to_release=9)
        assert genes(result) == ["B.1"]
        assert result["B.1"]["from_release"] == 4
        assert result["B.1"]["external_jump"] == 0


class TestHistory:
    """The walk along Ensembl's history."""

    def test_reborn_id_is_not_reported_in_its_gap(self):
        """tracking-23."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("X", [(1, 1, 3), (1, 7, 9)])
        track = builder.track()
        assert track.convert("X.1", from_release=2, to_release=5) is None
        assert track.convert("X.1", to_release=5) is None
        assert track.convert("X.1", from_release=8, to_release=5) is None
        assert genes(track.convert("X.1", from_release=2, to_release=8)) == ["X.1"]
        assert genes(track.convert("X.1", from_release=8, to_release=2)) == ["X.1"]

    def test_split_returns_both_and_best_takes_the_higher_score(self):
        """tracking-9, accuracy-8: the mapping score ranks before the richness of annotation."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
        builder.event("A.1", "B.1", 4, 0.9).event("A.1", "C.1", 4, 0.1)
        for name in ("X1", "X2", "X3", "X4"):
            builder.xref(name, "C.1", "EntrezGene", range(5, 10))
        track = builder.track()
        assert genes(track.convert("A.1", to_release=9)) == ["B.1", "C.1"]
        best = track.convert("A.1", to_release=9, prioritize_to_one_filter=True)
        assert genes(best) == ["B.1"]
        assert best["B.1"]["score"] == pytest.approx(0.9)
        assert best["B.1"]["tie"] is False

    def test_merge_backwards_returns_all_ancestors(self):
        """Merge backwards returns all ancestors."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 1, 4)]).gene("C", [(1, 5, 9)])
        builder.event("A.1", "C.1", 4, 0.6).event("B.1", "C.1", 4, 0.4)
        track = builder.track()
        assert genes(track.convert("C.1", to_release=2)) == ["A.1", "B.1"]
        assert genes(track.convert("C.1", to_release=2, prioritize_to_one_filter=True)) == ["A.1"]

    def test_score_is_the_product_along_the_path(self):
        """Score is the product along the path."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 2), (2, 3, 5), (3, 6, 9)], scores={1: 0.5, 2: 0.8})
        track = builder.track()
        result = track.convert("A.1", to_release=9, return_path=True)
        assert genes(result) == ["A.3"]
        assert result["A.3"]["score"] == pytest.approx(0.4)
        assert result["A.3"]["ensembl_step"] == 2
        assert result["A.3"]["the_path"] == (("A.1", "A.2", 0), ("A.2", "A.3", 0))

    def test_gene_alive_at_target_is_its_own_counterpart(self):
        """Gene alive at target is its own counterpart."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)])
        result = builder.track().convert("A.1", from_release=2, to_release=8, return_path=True)
        assert genes(result) == ["A.1"]
        assert result["A.1"]["the_path"] == ((None, "A.1", None),)
        assert result["A.1"]["score"] == 1.0

    def test_retired_gene_without_successor_gives_nothing(self):
        """Retired gene without successor gives nothing."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)])
        assert builder.track().convert("A.1", to_release=9) is None


class TestBridges:
    """Passing from a gene to another through a shared external id (tracking-22, accuracy-3)."""

    def test_bridge_is_used_only_without_history_and_is_reported(self):
        """Bridge is used only without history and is reported."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)])
        builder.xref("123", "A.1", "EntrezGene", range(1, 5)).xref("123", "B.1", "EntrezGene", range(5, 10))
        track = builder.track()
        result = track.convert("A.1", to_release=9, return_path=True)
        assert genes(result) == ["B.1"]
        assert result["B.1"]["external_jump"] == 1
        assert result["B.1"]["external_gap"] == 1
        assert result["B.1"]["the_path"] == (("A.1", "123", 0, 4), ("123", "B.1", 0, 5))
        assert track.convert("A.1", to_release=9, go_external=False) is None

    def test_external_id_bridges_from_the_gene_it_stands_for_as_that_gene_does(self):
        """Decision 17: one rule for ids and genes. A symbol of a gene without successor bridges as the gene does."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 1, 9)])
        builder.xref("OLDSYM", "A.1", "HGNC Symbol", range(1, 5))
        builder.xref("123", "A.1", "EntrezGene", range(1, 5)).xref("123", "B.1", "EntrezGene", range(5, 10))
        track = builder.track()
        for query in ("A.1", "A", "OLDSYM"):
            for start in (None, 3):
                result = track.convert(query, from_release=start, to_release=9, return_path=True)
                assert genes(result) == ["B.1"], (query, start)
                assert result["B.1"]["external_jump"] == 1 and result["B.1"]["entry_gene"] == "A.1"
                assert result["B.1"]["the_path"][-2:] == (("A.1", "123", 0, 4), ("123", "B.1", 0, 5))
            assert track.convert(query, to_release=9, go_external=False) is None
        # the bridge starts at the gene the id stands for, never at the id itself: OLDSYM is not on C
        assert genes(track.convert("OLDSYM", to_release=9, final_database="HGNC Symbol")) == ["B.1"]

    def test_history_precedes_any_bridge(self):
        """History precedes any bridge."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
        builder.event("A.1", "C.1", 4, 0.3)
        builder.xref("123", "A.1", "EntrezGene", range(1, 5)).xref("123", "B.1", "EntrezGene", range(5, 10))
        result = builder.track().convert("A.1", to_release=9)
        assert genes(result) == ["C.1"]
        assert result["C.1"]["external_jump"] == 0

    def test_bridge_backwards_in_time_is_not_followed(self):
        """Gene A (releases 5-8) and gene B share an id that was on B in release 1 and on A in release 8."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 5, 8)]).gene("B", [(1, 1, 9)])
        builder.xref("123", "B.1", "EntrezGene", [1]).xref("123", "A.1", "EntrezGene", [8])
        assert builder.track().convert("A.1", from_release=8, to_release=9) is None

    def test_clone_based_name_is_not_a_bridge(self):
        """Clone based name is not a bridge."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)])
        builder.xref("AC023055.1", "A.1", "Clone_based_ensembl_gene", range(1, 5))
        builder.xref("AC023055.1", "B.1", "Clone_based_ensembl_gene", range(5, 10))
        assert builder.track().convert("A.1", to_release=9) is None

    def test_nearest_bridge_in_time_is_preferred(self):
        """Nearest bridge in time is preferred."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 1, 9)])
        builder.xref("123", "A.1", "EntrezGene", range(1, 5)).xref("123", "B.1", "EntrezGene", range(5, 10))
        builder.xref("SYM", "A.1", "HGNC Symbol", range(1, 5)).xref("SYM", "C.1", "HGNC Symbol", range(8, 10))
        track = builder.track()
        assert genes(track.convert("A.1", to_release=9)) == ["B.1", "C.1"]
        assert genes(track.convert("A.1", to_release=9, prioritize_to_one_filter=True)) == ["B.1"]

    def test_bridge_through_a_transcript_level_id(self):
        """Bridge through a transcript level id."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)])
        builder.transcript("AT.1", "A.1", range(1, 5)).xref("NM_1", "AT.1", "RefSeq_mRNA", range(1, 5))
        builder.xref("NM_1", "B.1", "RefSeq_mRNA", range(5, 10))
        result = builder.track().convert("A.1", to_release=9)
        assert genes(result) == ["B.1"]
        assert result["B.1"]["external_step"] == 3

    def test_hyperconnective_id_is_not_a_bridge(self):
        """Hyperconnective id is not a bridge."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)])
        builder.xref("U6", "A.1", "RFAM", range(1, 5)).xref("U6", "B.1", "RFAM", range(5, 10))
        for index in range(TheGraph.genes_at_once_threshold + 1):
            builder.gene(f"F{index}", [(1, 1, 9)]).xref("U6", f"F{index}.1", "RFAM", range(1, 10))
        assert builder.track().convert("A.1", to_release=9) is None

    def test_hyperconnective_id_is_still_a_final_target_and_a_query(self):
        """An id attached to many genes is no evidence for a bridge, but it is a gene's id and can be asked for."""
        builder = GraphBuilder(range(1, 10))
        for index in range(TheGraph.genes_at_once_threshold + 2):
            builder.gene(f"F{index:02d}", [(1, 1, 9)]).xref("U6", f"F{index:02d}.1", "RFAM", range(1, 10))
        track = builder.track()
        assert targets(track.convert("F00.1", final_database="RFAM")) == ["U6"]
        assert len(genes(track.convert("U6"))) == TheGraph.genes_at_once_threshold + 2
        assert track.calculate_node_scores("F00.1", 9) == [0, 0, 0]


class TestFinalConversion:
    """Converting the genes found into another database (tracking-3, tracking-13, tracking-17)."""

    @pytest.fixture
    def two_assemblies(self) -> Track:
        """Assembly 37 serves releases 1-5 and assembly 38 releases 6-9; gene G has symbol S throughout."""
        assembly_of = {r: (37 if r <= 5 else 38) for r in range(1, 10)}
        builder = GraphBuilder(range(1, 10), assembly_of)
        builder.gene("G", [(1, 1, 9)])
        builder.xref("S", "G.1", "HGNC Symbol", range(1, 10))
        # An archive copy of assembly 37 keeps another symbol on the gene at release 7.
        builder.xref("OLDNAME", "G.1", "HGNC Symbol", [3, 7], assembly=37)
        builder.xref("OLDNAME", "G.1", "HGNC Symbol", [3])
        return builder.track()

    def test_symbol_at_a_release_of_the_earlier_assembly(self, two_assemblies):
        """Symbol at a release of the earlier assembly."""
        result = two_assemblies.convert("G.1", from_release=3, to_release=3, final_database="HGNC Symbol")
        assert targets(result) == ["OLDNAME", "S"]
        assert result["G.1"]["final_conversion"]["final_database"] == "HGNC Symbol"
        assert result["G.1"]["final_conversion"]["no_target"] is False

    def test_parallel_assembly_is_not_used_at_a_release_of_the_primary_one(self, two_assemblies):
        """Parallel assembly is not used at a release of the primary one."""
        result = two_assemblies.convert("G.1", to_release=7, final_database="HGNC Symbol")
        assert targets(result) == ["S"]

    def test_base_id_as_final_database(self, two_assemblies):
        """Base id as final database."""
        assert targets(two_assemblies.convert("G.1", to_release=3, final_database="base_ensembl_gene")) == ["G"]

    @pytest.fixture
    def split_one_named(self) -> Track:
        """A splits into B (which has a symbol) and C (which has none); D becomes E, which has no symbol."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
        builder.event("A.1", "B.1", 4, 0.5).event("A.1", "C.1", 4, 0.5)
        builder.xref("SYM", "B.1", "HGNC Symbol", range(5, 10))
        builder.gene("D", [(1, 1, 4)]).gene("E", [(1, 5, 9)]).event("D.1", "E.1", 4, 1.0)
        return builder.track()

    def test_one_result_has_one_final_database(self, split_one_named):
        """One result has one final database."""
        result = split_one_named.convert("A.1", to_release=9, final_database="HGNC Symbol")
        assert genes(result) == ["B.1"]
        assert targets(result) == ["SYM"]
        assert result["B.1"]["final_conversion"]["genes_without_target"] == ["C.1"]  # reached, but it has no symbol
        assert result["B.1"]["final_conversion"]["final_conversion_confidence"] == 0
        assert split_one_named.convert("A.1", to_release=9)["B.1"]["final_conversion"]["genes_without_target"] == []
        same = split_one_named.convert(
            "A.1", to_release=9, final_database="HGNC Symbol", return_ensembl_alternative=False
        )
        assert targets(same) == ["SYM"]

    def test_no_gene_has_a_target(self, split_one_named):
        """No gene has a target."""
        result = split_one_named.convert("D.1", to_release=9, final_database="HGNC Symbol")
        assert targets(result) == ["E.1"]
        final = result["E.1"]["final_conversion"]
        assert final["final_database"] == "ensembl_gene"
        assert final["no_target"] is True
        assert final["final_conversion_confidence"] is None
        assert final["genes_without_target"] == []
        json.dumps(result, allow_nan=False)  # strict JSON: no infinity stands for "no confidence"
        assert (
            split_one_named.convert("D.1", to_release=9, final_database="HGNC Symbol", return_ensembl_alternative=False)
            is None
        )

    def test_ids_on_transcripts_and_translations_are_found(self):
        """Ids on transcripts and translations are found."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)])
        builder.transcript("AT.1", "A.1", range(1, 10)).translation("AP.1", "AT.1", range(1, 10))
        builder.xref("NM_1", "AT.1", "RefSeq_mRNA", range(1, 10)).xref("P1", "AP.1", "Uniprot/SWISSPROT", range(1, 10))
        builder.xref("P2", "AP.1", "Uniprot/SWISSPROT", range(1, 10))
        track = builder.track()
        assert targets(track.convert("A.1", final_database="RefSeq_mRNA")) == ["NM_1"]
        assert targets(track.convert("A.1", final_database="Uniprot/SWISSPROT")) == ["P1", "P2"]
        best = track.convert("A.1", final_database="Uniprot/SWISSPROT", prioritize_to_one_filter=True)
        assert targets(best) == ["P1", "P2"]

    def test_name_of_a_gene_without_a_conversion(self):
        """The ids of a gene in a database are the same whatever query led to the gene, and need no search."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4), (2, 5, 9)], scores={1: 0.9})
        builder.xref("S2", "A.2", "HGNC Symbol", range(5, 10)).xref("S1", "A.2", "HGNC Symbol", range(5, 10))
        builder.xref("OLD", "A.1", "HGNC Symbol", range(1, 5))
        track = builder.track()
        assert track.gene_to_database("A.2", "HGNC Symbol") == (["S1", "S2"], 0)
        assert track.gene_to_database("A.1", "HGNC Symbol", 3) == (["OLD"], 0)
        assert track.gene_to_database("A.2", "base_ensembl_gene", 9) == (["A"], 0)
        assert track.gene_to_database("A.2", "ensembl_gene") == (["A.2"], 0)
        for query in ("A.1", "A", "OLD", "S1"):
            assert targets(track.convert(query, to_release=9, final_database="HGNC Symbol")) == ["S1", "S2"]
        with pytest.raises(ValueError, match="not a gene id with version that is active"):
            track.gene_to_database("A.1", "HGNC Symbol", 9)
        with pytest.raises(ValueError, match="final_database"):
            track.gene_to_database("A.2", "NOT_A_DB")

    def test_id_of_a_foreign_gene_is_not_returned(self):
        """Two genes share an external id; the other gene's protein id is not a target of the first."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)]).gene("B", [(1, 1, 9)])
        builder.xref("SHARED", "A.1", "HGNC Symbol", range(1, 10)).xref("SHARED", "B.1", "HGNC Symbol", range(1, 10))
        builder.transcript("BT.1", "B.1", range(1, 10)).translation("BP.1", "BT.1", range(1, 10))
        builder.xref("P9", "BP.1", "Uniprot/SWISSPROT", range(1, 10))
        result = builder.track().convert("A.1", final_database="Uniprot/SWISSPROT")
        assert targets(result) == ["A.1"]
        assert result["A.1"]["final_conversion"]["no_target"] is True


class TestBest:
    """``prioritize_to_one_filter`` (tracking-5, tracking-9, tracking-14, tracking-new-1, decision 8)."""

    def test_external_id_listed_as_versionless_gene_id_does_not_break_the_tie_break(self):
        """tracking-5: an external database lists a string that is also a gene id without version."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
        builder.event("A.1", "B.1", 4, 0.5).event("A.1", "C.1", 4, 0.5)
        builder.graph.nodes["B"]["node_type"] = "external"
        builder.xref("B", "B.1", "MGI Symbol", [5])
        result = builder.track().convert("A.1", to_release=9, prioritize_to_one_filter=True)
        assert genes(result) == ["B.1", "C.1"]

    def test_tie_is_broken_by_annotation_and_counts_own_forms_only(self):
        """Tie is broken by annotation and counts own forms only."""

        def split_in_two(extra_id_on_c: bool) -> Track:
            builder = GraphBuilder(range(1, 10))
            builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
            builder.event("A.1", "B.1", 4, 0.5).event("A.1", "C.1", 4, 0.5)
            for stable_id in ("B", "C"):
                builder.transcript(f"{stable_id}T.1", f"{stable_id}.1", range(5, 10))
                builder.translation(f"{stable_id}P.1", f"{stable_id}T.1", range(5, 10))
                builder.xref("SHARED", f"{stable_id}P.1", "Uniprot/SPTREMBL", range(5, 10))
            if extra_id_on_c:
                builder.xref("EXTRA", "C.1", "EntrezGene", range(5, 10))
            return builder.track()

        track = split_in_two(extra_id_on_c=False)
        assert track.calculate_node_scores("B.1", 9) == [-1, -1, -1]
        assert track.calculate_node_scores("C.1", 9) == [-1, -1, -1]
        assert genes(track.convert("A.1", to_release=9, prioritize_to_one_filter=True)) == ["B.1", "C.1"]
        track = split_in_two(extra_id_on_c=True)
        assert track.calculate_node_scores("C.1", 9) == [-2, -1, -1]
        assert genes(track.convert("A.1", to_release=9, prioritize_to_one_filter=True)) == ["C.1"]

    @pytest.mark.parametrize("forms", [["gene"], ["gene", "transcript"]])
    def test_graph_with_fewer_forms(self, forms):
        """tracking-14."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)])
        builder.event("A.1", "B.1", 4, 0.5).event("A.1", "C.1", 4, 0.5)
        builder.graph.available_forms = forms
        assert genes(builder.track().convert("A.1", to_release=9, prioritize_to_one_filter=True)) == ["B.1", "C.1"]

    def test_same_as_query_is_preferred(self):
        """Same as query is preferred."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)])
        builder.xref("S", "A.1", "HGNC Symbol", range(1, 10)).xref("T", "A.1", "HGNC Symbol", range(1, 10))
        track = builder.track()
        assert targets(track.convert("S", final_database="HGNC Symbol")) == ["S", "T"]
        assert targets(track.convert("T", final_database="HGNC Symbol", prioritize_to_one_filter=True)) == ["T"]

    def test_result_order_does_not_depend_on_insertion_order(self):
        """Result order does not depend on insertion order."""

        def build(order: list[str]) -> Track:
            builder = GraphBuilder(range(1, 10))
            builder.gene("A", [(1, 1, 4)])
            for stable_id in order:
                builder.gene(stable_id, [(1, 5, 9)]).event("A.1", f"{stable_id}.1", 4, 0.5)
            return builder.track()

        first = build(["B", "C", "D"]).convert("A.1", to_release=9, prioritize_to_one_filter=True)
        second = build(["D", "C", "B"]).convert("A.1", to_release=9, prioritize_to_one_filter=True)
        assert genes(first) == genes(second) == ["B.1", "C.1", "D.1"]
        assert all(entry["tie"] is True for entry in first.values())
        every = build(["C", "D", "B"]).convert("A.1", to_release=9)
        assert genes(every) == ["B.1", "C.1", "D.1"]
        assert all(entry["tie"] is True for entry in every.values())


class TestArguments:
    """Arguments are checked before any search, the same for every identifier (tracking-11, -12, tests-17)."""

    @pytest.fixture
    def track(self, moved_symbol) -> Track:
        """Return the graph of this class as a ``Track``."""
        return moved_symbol

    def test_numpy_integers_are_accepted(self, track):
        """Numpy integers are accepted."""
        plain = track.convert("S", from_release=2, to_release=9, final_database="HGNC Symbol")
        numpy = track.convert("S", from_release=np.int64(2), to_release=np.int32(9), final_database="HGNC Symbol")
        assert plain == numpy
        assert type(numpy["A.1"]["from_release"]) is int

    @pytest.mark.parametrize("identifier", ["S", "A.1", "not_in_the_graph"])
    @pytest.mark.parametrize("release", [0, 10, 5000, -1])
    def test_release_outside_the_graph(self, track, identifier, release):
        """Release outside the graph."""
        with pytest.raises(ValueError, match="not a release of this graph"):
            track.convert(identifier, to_release=release)
        with pytest.raises(ValueError, match="not a release of this graph"):
            track.convert(identifier, from_release=release, to_release=9)

    @pytest.mark.parametrize("identifier", ["S", "A.1", "not_in_the_graph"])
    @pytest.mark.parametrize(
        "release",
        [
            pytest.param(5.5, marks=pytest.mark.wrong_argument_type),
            pytest.param(5.0, marks=pytest.mark.wrong_argument_type),
            pytest.param("5", marks=pytest.mark.wrong_argument_type),
            True,
            None,
        ],
    )
    def test_release_of_wrong_type(self, track, identifier, release):
        """Release of wrong type."""
        if release is None:
            assert track.convert(identifier, to_release=None) == track.convert(identifier, to_release=9)
            return
        with pytest.raises(TypeError, match="to_release"):
            track.convert(identifier, to_release=release)

    @pytest.mark.parametrize("identifier", ["S", "A.1", "not_in_the_graph"])
    def test_unknown_final_database(self, track, identifier):
        """Unknown final database."""
        with pytest.raises(ValueError, match="final_database"):
            track.convert(identifier, final_database="NOT_A_DB")

    def test_unknown_identifier(self, track):
        """Unknown identifier."""
        assert track.convert("not_in_the_graph") is None

    @pytest.mark.wrong_argument_type
    def test_identifier_that_is_not_a_string(self, track):
        """Identifier that is not a string."""
        with pytest.raises(TypeError, match="from_id"):
            track.convert(12345)  # type: ignore[arg-type]

    def test_reverse_flag_must_agree_with_the_releases(self, track):
        """Reverse flag must agree with the releases."""
        assert track.get_possible_paths("A.1", 2, 9, reverse=False) == (((None, "A.1", None),),)
        with pytest.raises(ValueError, match="contradicts"):
            track.get_possible_paths("A.1", 2, 9, reverse=True)


class TestBudget:
    """The search is bounded and has a budget (tracking-1, tracking-7, accuracy-9)."""

    @staticmethod
    def chain(n_genes: int, n_shared: int) -> GraphBuilder:
        """A chain of retired genes that pairwise share external ids: the worst case of the old enumeration."""
        builder = GraphBuilder(range(1, 10))
        for index in range(n_genes):
            builder.gene(f"G{index:02d}", [(1, 1, 4), (2, 5, 8)], scores={1: 0.9})
        for index in range(n_genes):
            for other in range(n_genes):
                if index < other:
                    for k in range(n_shared):
                        name = f"X{index:02d}_{other:02d}_{k}"
                        builder.xref(name, f"G{index:02d}.1", "EntrezGene", range(1, 5))
                        builder.xref(name, f"G{other:02d}.2", "EntrezGene", range(5, 9))
        return builder

    def test_work_grows_with_the_graph_not_with_the_number_of_paths(self):
        """Work grows with the graph not with the number of paths."""
        track = self.chain(12, 3).track()
        calls = {"history": 0}
        reader = track._reader
        original = reader.history_edges

        def counting(node, reverse):
            calls["history"] += 1
            return original(node, reverse)

        reader.history_edges = counting
        assert track.convert("G00.1", to_release=9) is None
        # Each gene node is read at most once per direction, however many paths lead to it.
        assert calls["history"] <= 2 * 4 * 12

    def test_over_budget_raises_a_named_error(self):
        """Over budget raises a named error."""
        track = self.chain(12, 3).track()
        track.search_budget_states = 5
        with pytest.raises(SearchBudgetExceededError) as error:
            track.convert("G00.1", to_release=9)
        assert error.value.identifier == "G00.1"
        assert error.value.states > 5

    def test_budget_of_one_call(self):
        """The budget can be given per call and must be positive."""
        track = self.chain(12, 3).track()
        with pytest.raises(SearchBudgetExceededError):
            track.convert("G00.1", to_release=9, budget_states=5)
        assert track.convert("G00.1", to_release=9, budget_states=100000, budget_seconds=60) is None
        with pytest.raises(ValueError, match="budget"):
            track.convert("G00.1", to_release=9, budget_seconds=0)

    def test_equal_bridges_lead_to_one_result(self):
        """tracking-7: three ids shared by the same two genes are three ways to one destination."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)])
        for name in ("X1", "X2", "X3"):
            builder.xref(name, "A.1", "EntrezGene", range(1, 5)).xref(name, "B.1", "EntrezGene", range(5, 10))
        result = builder.track().convert("A.1", to_release=9, return_path=True)
        assert genes(result) == ["B.1"]
        assert result["B.1"]["the_path"] == (("A.1", "X1", 0, 4), ("X1", "B.1", 0, 5))


class TestIdentifySource:
    """Where a list of ids comes from (tracking-15)."""

    def test_ties_are_ordered_latest_release_first_and_no_graph_wide_table_is_built(self):
        """Ties are ordered latest release first and no graph wide table is built."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)]).gene("B", [(1, 3, 7)])
        builder.xref("S", "A.1", "HGNC Symbol", range(1, 10))
        track = builder.track()
        counted = track.identify_source(["A.1", "B.1"], mode="ensembl_release")
        assert counted[:5] == [(7, 2), (6, 2), (5, 2), (4, 2), (3, 2)]
        assert counted[5:] == [(9, 1), (8, 1), (2, 1), (1, 1)]
        assert track.identify_source(["A.1"], mode="assembly") == [(38, 1)]
        assert track.identify_source(["S"], mode="complete")[0] == (("HGNC Symbol", 38, 9), 1)
        assert track.identify_source(["A.1"], mode="assembly_ensembl_release")[0] == ((38, 9), 1)
        assert not [k for k, v in track.graph.__dict__.items() if isinstance(v, dict) and k != "graph" and len(v) >= 3]
        with pytest.raises(ValueError, match="Unknown mode"):
            track.identify_source(["A.1"], mode="bogus")


class TestPlainResult:
    """Results are plain data (decision 8)."""

    def test_result_is_json_serialisable(self, moved_symbol):
        """Result is json serialisable."""
        result = moved_symbol.convert(
            "S", to_release=4, final_database="HGNC Symbol", return_path=True, prioritize_to_one_filter=True
        )
        assert json.loads(json.dumps(result, allow_nan=False)) is not None
        alternative = moved_symbol.convert("S", to_release=4, final_database="base_ensembl_gene", return_path=True)
        assert targets(alternative) == ["B"]


class TestQueryDatabase:
    """An id that belongs to two databases at one release (accuracy-13)."""

    @pytest.fixture
    def track(self) -> Track:
        """Return the graph of this class as a ``Track``."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)]).gene("B", [(1, 1, 9)])
        builder.xref("10123", "A.1", "EntrezGene", range(1, 10))
        builder.xref("10123", "B.1", "Havana gene", range(1, 10))
        return builder.track()

    def test_ambiguity_is_reported(self, track):
        """Ambiguity is reported."""
        result = track.convert("10123")
        assert genes(result) == ["A.1", "B.1"]
        assert result["A.1"]["from_databases"] == ["EntrezGene", "Havana gene"]

    def test_named_database_selects_its_gene(self, track):
        """Named database selects its gene."""
        assert genes(track.convert("10123", from_database="EntrezGene")) == ["A.1"]
        result = track.convert("10123", from_database="Havana gene")
        assert genes(result) == ["B.1"]
        assert result["B.1"]["from_databases"] == ["Havana gene"]

    def test_invalid_from_database(self, track):
        """Invalid from database."""
        with pytest.raises(ValueError, match="from_database"):
            track.convert("10123", from_database="NOT_A_DB")
        assert track.convert("A.1", from_database="EntrezGene") is None
        assert track.convert("A", from_database="EntrezGene") is None


class TestSeveralEntryGenes:
    """An external id that several genes carry at the starting release (decisions 8 and 12)."""

    def test_better_supported_gene_is_the_best_answer(self):
        """Better supported gene is the best answer."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 9)]).gene("B", [(1, 1, 9)])
        for database in ("HGNC Symbol", "EntrezGene", "UniProtKB Gene Name"):
            builder.xref("CKB", "A.1", database, range(1, 10))
        builder.xref("CKB", "B.1", "UniProtKB Gene Name", range(1, 10))
        for name in ("X1", "X2", "X3", "X4", "X5"):
            builder.xref(name, "B.1", "RefSeq_mRNA", range(1, 10))
        track = builder.track()
        every = track.convert("CKB")
        assert genes(every) == ["A.1", "B.1"]
        assert every["A.1"]["entry_support"] == 3 and every["B.1"]["entry_support"] == 1
        best = track.convert("CKB", prioritize_to_one_filter=True)
        assert genes(best) == ["A.1"]
        assert best["A.1"]["tie"] is False

    def test_score_is_compared_within_a_lineage_and_nothing_between_equal_lineages(self):
        """U8 is on A and on B with equal support; B split into C (0.9) and D (0.2)."""

        def two_lineages(ids_on_c: tuple = ()) -> Track:
            builder = GraphBuilder(range(1, 10))
            builder.gene("A", [(1, 1, 9)])
            builder.gene("B", [(1, 1, 4)]).gene("C", [(1, 5, 9)]).gene("D", [(1, 5, 9)])
            builder.event("B.1", "C.1", 4, 0.9).event("B.1", "D.1", 4, 0.2)
            builder.xref("U8", "A.1", "RFAM", range(1, 5)).xref("U8", "B.1", "RFAM", range(1, 5))
            for name in ids_on_c:
                builder.xref(name, "C.1", "EntrezGene", range(5, 10))
            return builder.track()

        track = two_lineages()
        assert genes(track.convert("U8", from_release=3, to_release=9)) == ["A.1", "C.1", "D.1"]
        # A (score 1.0) does not beat C (0.9): they descend from different genes; D loses to C within its lineage.
        best = track.convert("U8", from_release=3, to_release=9, prioritize_to_one_filter=True)
        assert genes(best) == ["A.1", "C.1"]
        assert best["A.1"]["tie"] is True
        assert best["C.1"]["entry_gene"] == "B.1"
        # Nor does a richer annotation: which of two genes a name stands for is not a matter of counting ids.
        track = two_lineages(ids_on_c=("X1", "X2", "X3"))
        best = track.convert("U8", from_release=3, to_release=9, prioritize_to_one_filter=True)
        assert genes(best) == ["A.1", "C.1"]
        assert best["A.1"]["tie"] is True and best["C.1"]["tie"] is True

    def test_every_answer_leads_with_the_best_answer(self):
        """S is on A and on B; A split into C (0.9) and D (0.8), B became E (0.7): C and E are the best answer.

        By the evidence alone D (0.8) would come before E (0.7), but the mapping score compares the successors of
        one entry gene only: with every answer asked, the best answer leads in its own order and D follows.
        """
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("C", [(1, 5, 9)]).gene("D", [(1, 5, 9)])
        builder.gene("B", [(1, 1, 4)]).gene("E", [(1, 5, 9)])
        builder.event("A.1", "C.1", 4, 0.9).event("A.1", "D.1", 4, 0.8).event("B.1", "E.1", 4, 0.7)
        builder.xref("S", "A.1", "HGNC Symbol", range(1, 5)).xref("S", "B.1", "HGNC Symbol", range(1, 5))
        track = builder.track()
        best = track.convert("S", from_release=3, to_release=9, prioritize_to_one_filter=True)
        assert genes(best) == ["C.1", "E.1"] and best["C.1"]["tie"] is True
        every = track.convert("S", from_release=3, to_release=9)
        assert genes(every) == ["C.1", "E.1", "D.1"]
        assert [every[gene]["score"] for gene in every] == [0.9, 0.7, 0.8]
        assert every["C.1"]["tie"] is True

    def test_annotation_decides_between_equal_successors_of_each_entry_gene_only(self):
        """S is on A and on B; A split into C and D equally, D is better annotated: D and B are the answer."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("A", [(1, 1, 4)]).gene("C", [(1, 5, 9)]).gene("D", [(1, 5, 9)])
        builder.gene("B", [(1, 1, 9)])
        builder.event("A.1", "C.1", 4, 0.5).event("A.1", "D.1", 4, 0.5)
        builder.xref("S", "A.1", "HGNC Symbol", range(1, 5)).xref("S", "B.1", "HGNC Symbol", range(1, 5))
        for name in ("X1", "X2"):
            builder.xref(name, "D.1", "EntrezGene", range(5, 10))
        for name in ("Y1", "Y2", "Y3", "Y4"):
            builder.xref(name, "C.1", "RefSeq_mRNA", range(1, 4))  # not at the target release: it does not count
        track = builder.track()
        # every answer comes best first: the best answer (B and D) leads, then C, which lost to D by annotation
        assert genes(track.convert("S", from_release=3, to_release=9)) == ["B.1", "D.1", "C.1"]
        best = track.convert("S", from_release=3, to_release=9, prioritize_to_one_filter=True)
        assert genes(best) == ["B.1", "D.1"]
        assert best["B.1"]["tie"] is True
        scores = best["D.1"]["final_conversion"]["final_elements"]["D.1"]["filter_scores"]
        assert scores["node_importance_filter"] == [-2, 0, 0]
        assert set(scores) == {"initial_filter", "same_as_input_filter", "node_importance_filter"}
        # B has no rival among the successors of its own entry gene: its annotation was never counted
        assert (
            best["B.1"]["final_conversion"]["final_elements"]["B.1"]["filter_scores"]["node_importance_filter"] is None
        )

    def test_graph_without_marks_returns_the_genes_that_carry_an_id_equally_as_a_tie(self):
        """A graph made by hand marks no gene as a copy beside the reference sequence: every carrier is an answer."""
        builder = GraphBuilder(range(1, 10))
        builder.gene("PRIMARY", [(1, 1, 9)]).gene("PATCH", [(1, 1, 9)])
        builder.xref("10000", "PRIMARY.1", "EntrezGene", range(1, 10)).xref(
            "10000", "PATCH.1", "EntrezGene", range(1, 10)
        )
        builder.xref("10000", "PRIMARY.1", "EntrezGene", range(1, 10), assembly=37)
        for name in ("X1", "X2", "X3"):
            builder.xref(name, "PATCH.1", "RefSeq_mRNA", range(1, 10))
        track = builder.track()
        result = track.convert("10000", prioritize_to_one_filter=True)
        assert genes(result) == ["PATCH.1", "PRIMARY.1"]
        assert result["PRIMARY.1"]["tie"] is True and result["PRIMARY.1"]["alternative_locus_genes"] == []
        assert genes(track.convert("10000")) == ["PATCH.1", "PRIMARY.1"]
