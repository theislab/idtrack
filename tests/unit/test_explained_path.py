#!/usr/bin/env python3
"""The explained path of a conversion (decision 46 e), as data, on graphs that every test run holds.

Two graphs: one written by hand (:py:func:`hand_made_graph`), in which every kind of step, both sides of a bridge, a
split, a merge and both at once, a version change with and without Ensembl's score, a scored self loop, a synonym and
a hand-off from one genome assembly to the next stand at known places; and the graph of pig built from the cut real
tables of ``tests/data/real`` (Sscrofa10.2 serves releases up to 89, Sscrofa11.1 from 90 on), whose explained steps
are compared with what Ensembl's own tables say. Every field of :py:class:`idtrack.results.PathNode` and
:py:class:`idtrack.results.PathStep` is asserted with its exact value.
"""

from __future__ import annotations

import json
import math

import pytest

from idtrack import API, ids
from idtrack.graph import TheGraph
from idtrack.results import STEP_REASONS, ConversionResult, ExplainedPath, PathNode, PathStep

NAN = float("nan")
INF = float("inf")
RELEASES = range(1, 10)
#: Assembly 37 serves releases 1 to 4, assembly 38 releases 5 to 9 (the hand-off is between 4 and 5).
ASSEMBLY_OF = {r: (37 if r <= 4 else 38) for r in RELEASES}


class _Writer:
    """Write a graph by hand in the layout of a built one (as ``tests/unit/test_track.py`` does)."""

    def __init__(self) -> None:
        self.graph = TheGraph()
        self.graph.graph.update(
            {
                "organism": "homo_sapiens",
                "genome_assembly": 38,
                "ensembl_release": RELEASES[-1],
                "confident_for_release": list(RELEASES),
                "narrow": True,
                "narrow_external": True,
                "misplaced_external_entry": set(),
            }
        )
        self.graph.available_forms = ["gene", "transcript", "translation"]

    def _xref(self, source: str, target: str, database: str, releases, assembly: int | None = None) -> None:
        per_assembly: dict[int, set] = {}
        for r in releases:
            per_assembly.setdefault(ASSEMBLY_OF[r] if assembly is None else assembly, set()).add(r)
        if self.graph.has_edge(source, target, 0):
            data = self.graph[source][target][0]
            for key, found in per_assembly.items():
                data["connection"].setdefault(database, {}).setdefault(key, set()).update(found)
            data["available_releases"] = set(data["available_releases"]) | set(releases)
        else:
            self.graph.add_edge(
                source, target, 0, connection={database: per_assembly}, available_releases=set(releases)
            )

    def _node(self, name: str, node_type: str) -> None:
        stable_id, version = ids.split(name)
        self.graph.add_node(name, node_type=node_type, ID=stable_id, Version=version, is_latest=False)

    def gene(self, stable_id: str, lives: list[tuple[int, int, int]], scores: dict | None = None) -> _Writer:
        """Add a gene id: ``(version, first release, last release)`` per version, consecutive."""
        scores = scores or {}
        void, retired = f"{stable_id}.Void", f"{stable_id}.Retired"
        self._node(void, "ensembl_gene")
        self.graph.add_node(stable_id, node_type="base_ensembl_gene")
        if not lives:
            raise ValueError("a gene needs at least one version")
        previous: tuple[int, int, int] | None = None
        for version, first, last in lives:
            node = f"{stable_id}.{version}"
            self._node(node, "ensembl_gene")
            self._xref(stable_id, node, "base_ensembl_gene", range(first, last + 1))
            if previous is None:
                born = first if first == RELEASES[0] else first - 1
                self.graph.add_edge(void, node, weight=NAN, old_release=born, new_release=first)
            else:
                self.graph.add_edge(
                    f"{stable_id}.{previous[0]}",
                    node,
                    weight=scores.get(previous[0], NAN),
                    old_release=previous[2],
                    new_release=first,
                )
            previous = (version, first, last)
        assert previous is not None  # `lives` is not empty
        node = f"{stable_id}.{previous[0]}"
        if previous[2] == RELEASES[-1]:
            self.graph.add_edge(node, node, weight=1.0, old_release=previous[1], new_release=INF)
        else:
            self._node(retired, "ensembl_gene")
            self.graph.add_edge(node, retired, weight=NAN, old_release=previous[2], new_release=previous[2] + 1)
        return self

    def event(self, source: str, target: str, old_release: int, weight: float) -> _Writer:
        """Add an event of Ensembl's id history (a split, merge, replacement or a scored self loop)."""
        self.graph.add_edge(source, target, weight=weight, old_release=old_release, new_release=old_release + 1)
        return self

    def xref(self, external: str, target: str, database: str, releases) -> _Writer:
        """Attach an id of an external database to a gene or transcript at releases."""
        if external not in self.graph:
            self.graph.add_node(external, node_type="external")
        self._xref(external, target, database, releases)
        return self

    def transcript(self, name: str, gene: str, releases) -> _Writer:
        """Add a transcript of a gene at releases."""
        self._node(name, "ensembl_transcript")
        self._xref(name, gene, "ensembl_transcript", releases)
        return self


def hand_made_graph() -> TheGraph:
    """Return the hand-made graph of this module (releases 1 to 9; assembly 37 serves 1 to 4, 38 serves 5 to 9).

    - ``A.1`` splits at 4 into ``B.1`` (score 0.9) and ``C.1`` (0.1); ``D.1`` merges into ``B.1`` (0.5); ``B.1`` has
      the symbol ``SYM`` (HGNC Symbol, 5 to 9), ``C.1`` none.
    - ``X`` changes version at 3 (score 0.7) and at 6 (no score).
    - ``G.1`` lives throughout with a scored self loop at 6 (0.95); its transcript ``T.1`` belongs to it at 1 to 3
      only and carries ``NM_1`` (RefSeq_mRNA) throughout.
    - ``Y.1`` is replaced by ``Z.1`` at 4 (0.8); ``Z.1`` has the synonym ``OLDSYM`` (``synonym_id::HGNC Symbol``).
    - ``E.1`` (1 to 4) and ``F.1`` (6 to 9) share the EntrezGene id ``123``: the only way from one to the other is a
      bridge.

    Returns:
        TheGraph: The graph (not frozen).
    """
    w = _Writer()
    w.gene("A", [(1, 1, 4)]).gene("B", [(1, 5, 9)]).gene("C", [(1, 5, 9)]).gene("D", [(1, 1, 4)])
    w.event("A.1", "B.1", 4, 0.9).event("A.1", "C.1", 4, 0.1).event("D.1", "B.1", 4, 0.5)
    w.xref("SYM", "B.1", "HGNC Symbol", range(5, 10))
    w.gene("X", [(1, 1, 3), (2, 4, 6), (3, 7, 9)], scores={1: 0.7})
    w.gene("G", [(1, 1, 9)]).event("G.1", "G.1", 6, 0.95)
    w.transcript("T.1", "G.1", range(1, 4)).xref("NM_1", "T.1", "RefSeq_mRNA", range(1, 10))
    w.gene("Y", [(1, 1, 4)]).gene("Z", [(1, 5, 9)]).event("Y.1", "Z.1", 4, 0.8)
    w.xref("synonym_id::OLDSYM", "Z.1", "synonym_id::HGNC Symbol", range(5, 10))
    w.gene("E", [(1, 1, 4)]).gene("F", [(1, 6, 9)])
    w.xref("123", "E.1", "EntrezGene", range(1, 5)).xref("123", "F.1", "EntrezGene", range(6, 10))
    return w.graph


@pytest.fixture(scope="module")
def api() -> API:
    """An API over the hand-made graph."""
    return API(hand_made_graph())


def gene_node(name: str, first: int, last: int | None) -> PathNode:
    """The node of a gene with version that lives from *first* to *last* (``None``: alive at the last release)."""
    served = {ASSEMBLY_OF[r] for r in range(first, (RELEASES[-1] if last is None else last) + 1)}
    return PathNode(id=name, kind="ensembl_gene", label=name, active=((first, last),), assemblies=tuple(sorted(served)))


def history(source, target, old, new, weight, splits, merges, reason, kind="history", key=0) -> PathStep:
    """A history step; the assemblies are those that serve its two releases."""
    return PathStep(
        kind=kind,
        source=source,
        target=target,
        key=key,
        old_release=old,
        new_release=new,
        weight=weight,
        splits=splits,
        merges=merges,
        assembly=ASSEMBLY_OF[new],
        old_assembly=ASSEMBLY_OF[old],
        reason=reason,
    )


def xref_step(kind, source, target, release, databases, reason=None) -> PathStep:
    """A cross-reference step at a release (``None``: a final hop of confidence 1)."""
    return PathStep(
        kind=kind,
        source=source,
        target=target,
        release=release,
        databases=databases,
        assembly=None if release is None else ASSEMBLY_OF[release],
        reason=reason or kind,
    )


IDENTITY = {"kind": "identity", "source": None, "key": None, "reason": "identity"}


def explained(api: API, identifier: str, **arguments) -> ConversionResult:
    """Convert with ``explain=True`` and check what every explanation must satisfy.

    The steps and nodes are each held once, the nodes in the order in which the steps first use them; every index
    of a candidate points at a step; the explanation is strict JSON and reads back equal.
    """
    result = api.convert(identifier, explain=True, **arguments)
    path = result.path
    assert isinstance(path, ExplainedPath)
    assert len(set(path.steps)) == len(path.steps)
    first_use = []
    for step in path.steps:
        for name in (step.source, step.target):
            if name is not None and name not in first_use:
                first_use.append(name)
    assert [n.id for n in path.nodes] == first_use
    for candidate in result.candidates:
        assert all(0 <= i < len(path.steps) for i in candidate.path + candidate.final_path)
        if candidate.path:
            assert path.steps[candidate.path[-1]].target == candidate.gene
        if candidate.final_path:
            assert path.steps[candidate.final_path[-1]].target == candidate.target
    text = json.dumps(result.to_dict(), allow_nan=False)
    assert ConversionResult.from_dict(json.loads(text)) == result
    assert ExplainedPath.from_dict(path.to_dict()) == path
    return result


# ---------------------------------------------------------------------------------------------- the hand-made graph


def test_split_and_merge_are_counted_on_each_step_and_the_assembly_hand_off_is_shown(api):
    """``A.1 -> B.1`` is a split (A also became C) and a merge (D also became B); ``A.1 -> C.1`` a split only."""
    result = explained(api, "A.1")
    assert result.path.steps == (
        history("A.1", "B.1", 4, 5, 0.9, 1, 1, "split_and_merge"),
        history("A.1", "C.1", 4, 5, 0.1, 1, 0, "split"),
    )
    assert result.path.steps[0].old_assembly == 37 and result.path.steps[0].assembly == 38
    assert result.path.nodes == (gene_node("A.1", 1, 4), gene_node("B.1", 5, None), gene_node("C.1", 5, None))
    assert [(c.gene, c.path, c.final_path) for c in result.candidates] == [("B.1", (0,), ()), ("C.1", (1,), ())]


def test_a_merge_alone(api):
    """``D.1 -> B.1``: B has another predecessor (a merge), D no other successor."""
    result = explained(api, "D.1")
    assert result.path.steps == (history("D.1", "B.1", 4, 5, 0.5, 0, 1, "merge"),)
    assert result.path.nodes == (gene_node("D.1", 1, 4), gene_node("B.1", 5, None))


def test_a_step_back_in_time_keeps_the_releases_and_counts_of_its_edge(api):
    """Walking ``B.1`` back to release 2: the steps go from B to A and D; releases and counts are the edges'."""
    result = explained(api, "B.1", to_release=2)
    assert result.path.steps == (
        history("B.1", "A.1", 4, 5, 0.9, 1, 1, "split_and_merge"),
        history("B.1", "D.1", 4, 5, 0.5, 0, 1, "merge"),
    )
    assert [c.gene for c in result.candidates] == ["A.1", "D.1"]


def test_versions_with_and_without_a_score(api):
    """A new version with Ensembl's score keeps it; one without has no weight (``None``, never NaN)."""
    result = explained(api, "X.1")
    assert result.path.steps == (
        history("X.1", "X.2", 3, 4, 0.7, 0, 0, "new_version"),
        history("X.2", "X.3", 6, 7, None, 0, 0, "new_version"),
    )
    assert result.path.nodes == (gene_node("X.1", 1, 3), gene_node("X.2", 4, 6), gene_node("X.3", 7, None))
    assert result.path.nodes[1].assemblies == (37, 38)
    assert result.candidates[0].path == (0, 1)


def test_an_id_replaced_by_another(api):
    """``Y.1 -> Z.1``: one event between two ids, no split, no merge."""
    result = explained(api, "Y.1")
    assert result.path.steps == (history("Y.1", "Z.1", 4, 5, 0.8, 0, 0, "replaced"),)


def test_a_bridge_has_two_sides_in_the_direction_of_the_walk(api):
    """Forward, the walk leaves E through ``123`` (near side, at E's last release) and enters F (far side)."""
    result = explained(api, "E.1")
    assert result.path.steps == (
        xref_step("bridge_near", "E.1", "123", 4, ("EntrezGene",)),
        xref_step("bridge_far", "123", "F.1", 6, ("EntrezGene",)),
    )
    assert result.path.nodes == (
        gene_node("E.1", 1, 4),
        PathNode(id="123", kind="external", label="123", databases=("EntrezGene",)),
        gene_node("F.1", 6, None),
    )
    candidate = result.candidates[0]
    assert (candidate.gene, candidate.path, candidate.bridges, candidate.bridge_gap) == ("F.1", (0, 1), 1, 2)
    # Backward, F is left (at its first release, on assembly 38) and E entered (at its last, on assembly 37).
    back = explained(api, "F.1", to_release=2)
    assert back.path.steps == (
        xref_step("bridge_near", "F.1", "123", 6, ("EntrezGene",)),
        xref_step("bridge_far", "123", "E.1", 4, ("EntrezGene",)),
    )
    assert (back.path.steps[0].assembly, back.path.steps[1].assembly) == (38, 37)


def test_a_synonym_keeps_the_names_of_the_graph(api):
    """A synonym is read through its node ``synonym_id::OLDSYM``; the path shows the graph's names."""
    result = explained(api, "OLDSYM")
    assert (result.graph_id, result.match, result.from_databases) == (
        "synonym_id::OLDSYM",
        "synonym:exact",
        ("synonym_id::HGNC Symbol",),
    )
    assert result.path.steps == (xref_step("entry", "synonym_id::OLDSYM", "Z.1", 9, ("synonym_id::HGNC Symbol",)),)
    assert result.path.nodes == (
        PathNode(
            id="synonym_id::OLDSYM",
            kind="synonym",
            label="synonym_id::OLDSYM",
            databases=("synonym_id::HGNC Symbol",),
        ),
        gene_node("Z.1", 5, None),
    )


def test_a_symbol_enters_its_gene(api):
    """``SYM`` is attached to ``B.1`` at release 9: one entry step."""
    result = explained(api, "SYM")
    assert result.path.steps == (xref_step("entry", "SYM", "B.1", 9, ("HGNC Symbol",)),)
    assert result.path.nodes[0] == PathNode(id="SYM", kind="external", label="SYM", databases=("HGNC Symbol",))


def test_identity_and_a_scored_self_loop(api):
    """A gene alive at both releases is its own answer; crossing its scored self loop is a step of its own."""
    identity = explained(api, "Y.1", from_release=3, to_release=2)
    assert identity.path.steps == (PathStep(target="Y.1", **IDENTITY),)
    loop = explained(api, "G.1", from_release=2, to_release=8)
    assert loop.path.steps == (history("G.1", "G.1", 6, 7, 0.95, 0, 0, "self_loop", kind="self_loop", key=1),)
    assert loop.path.nodes == (gene_node("G.1", 1, None),)
    assert loop.path.nodes[0].assemblies == (37, 38)


def test_final_steps_at_the_target_release_and_through_another_release(api):
    """The gene's id in the final database: attached at the target release, or through a transcript of before."""
    named = explained(api, "A.1", final_database="HGNC Symbol")
    assert named.path.steps == (
        history("A.1", "B.1", 4, 5, 0.9, 1, 1, "split_and_merge"),
        xref_step("final", "B.1", "SYM", 9, ("HGNC Symbol",)),
    )
    assert [(c.gene, c.target, c.path, c.final_path, c.final_confidence) for c in named.candidates] == [
        ("B.1", "SYM", (0,), (1,), 0)
    ]
    through = explained(api, "G.1", final_database="RefSeq_mRNA")
    assert through.path.steps == (
        PathStep(target="G.1", **IDENTITY),
        xref_step("final", "G.1", "T.1", None, ("ensembl_transcript",), reason="final_other_release"),
        xref_step("final", "T.1", "NM_1", 9, ("RefSeq_mRNA",)),
    )
    assert through.path.nodes == (
        gene_node("G.1", 1, None),
        PathNode(id="T.1", kind="ensembl_transcript", label="T.1"),
        PathNode(id="NM_1", kind="external", label="NM_1", databases=("RefSeq_mRNA",)),
    )
    assert [(c.target, c.path, c.final_path, c.final_confidence) for c in through.candidates] == [
        ("NM_1", (0,), (1, 2), 1)
    ]


def test_every_reason_code_is_given_and_has_a_text(api):
    """The conversions above give every code of ``STEP_REASONS``, and nothing else."""
    calls = [
        ("A.1", {}),
        ("D.1", {}),
        ("X.1", {}),
        ("Y.1", {}),
        ("E.1", {}),
        ("OLDSYM", {}),
        ("Y.1", {"from_release": 3, "to_release": 2}),
        ("G.1", {"from_release": 2, "to_release": 8}),
        ("G.1", {"final_database": "RefSeq_mRNA"}),
    ]
    seen = {
        step.reason for identifier, arguments in calls for step in explained(api, identifier, **arguments).path.steps
    }
    assert seen == set(STEP_REASONS)
    assert all(isinstance(text, str) and text for text in STEP_REASONS.values())


def test_without_explain_there_is_no_path(api):
    """The path is computed only when asked for; the candidates then hold no step indices."""
    result = api.convert("A.1")
    assert result.path is None
    assert all(c.path == () and c.final_path == () for c in result.candidates)


# ------------------------------------------------------------------------------------- the graphs of the cut real tables


def _ranges(releases) -> list[tuple[int, int]]:
    """Return the sorted releases as inclusive ranges of consecutive releases."""
    out: list[list[int]] = []
    for r in sorted(releases):
        if out and out[-1][1] == r - 1:
            out[-1][1] = r
        else:
            out.append([r, r])
    return [(a, b) for a, b in out]


def _events(world) -> dict[tuple[str, int, str, int], tuple[float, int]]:
    """Return Ensembl's gene events of the snapshot's core database.

    Returns:
        ``{(old id, old version, new id, new version): (score, mapping session)}``; events from or to no gene are left out.
    """
    (key,) = (k[:2] for k in world.tables.tables if k[2] == "stable_id_event")
    events = world.ensembl.events(*key)
    return {
        (a, int(va), b, int(vb)): (float(score), int(session))
        for a, va, b, vb, score, session in zip(
            events["old_stable_id"],
            events["old_version"],
            events["new_stable_id"],
            events["new_version"],
            events["score"],
            events["mapping_session_id"],
            strict=True,
        )
        if isinstance(b, str) and not math.isnan(va) and not math.isnan(vb)
    }


def _lives(world, name: str) -> list[int]:
    """Return the releases at which Ensembl's id tables hold a gene id with this version."""
    base, version = ids.split(name)
    return [r for r in sorted(world.backbone) if world.genes_at(r).get(base) == int(version)]


def _disagreements(world, result: ConversionResult, events: dict) -> list:
    """Compare the steps and gene nodes of a conversion forward in time with Ensembl's own tables.

    A history step leaves its source at a release the source has and enters its target at a release the target has;
    its two assemblies are those that serve the two releases (Ensembl's index of core databases); its weight is the
    score Ensembl gives the event, where Ensembl gives one; ``new_version`` is a change of version of one id. A gene
    node is active, served and outside the reference sequence exactly where Ensembl's tables say so.
    """
    from tests._real_tables import SNAPSHOT

    assert result.path is not None, "the conversion was asked to explain"
    found: list[tuple] = []
    for step in result.path.steps:
        if step.kind not in ("history", "self_loop"):
            continue
        assert step.source is not None, "a history step leaves a gene"
        source, target = ids.split(step.source), ids.split(step.target)
        if step.kind == "history" and step.old_release not in _lives(world, step.source):
            found.append(("old_release", step))
        if step.kind == "history" and step.new_release not in _lives(world, step.target):
            found.append(("new_release", step))
        if (step.old_assembly, step.assembly) != (world.backbone[step.old_release], world.backbone[step.new_release]):
            found.append(("assemblies", step))
        score = events.get((source[0], int(source[1]), target[0], int(target[1])), (0.0, None))[0]
        if score and step.weight != score:
            found.append(("weight", step, score))
        if step.reason == "new_version" and source[0] != target[0]:
            found.append(("reason", step))
    for node in result.path.nodes:
        if node.kind != "ensembl_gene":
            continue
        life = _lives(world, node.id)
        active = tuple((a, None if b == SNAPSHOT else b) for a, b in _ranges(life))
        served = tuple(sorted({world.backbone[r] for r in life}))
        base = ids.split(node.id)[0]
        beside = _ranges(r for r in life if base in world.ensembl.non_reference_genes(world.backbone[r], r))
        if (node.active, node.assemblies, node.non_reference) != (active, served, tuple(beside)):
            found.append(("node", node, active, served, beside))
    return found


@pytest.fixture(scope="module")
def pig(real_world):
    """The graph of pig built from its cut tables (shared with the other tests of the session)."""
    return real_world("sus_scrofa")


@pytest.fixture(scope="module")
def human(real_world):
    """The graph of human built from its cut tables (shared with the other tests of the session)."""
    return real_world("homo_sapiens")


def test_pig_gene_through_two_assembly_hand_offs(pig):
    """``ENSSSCG00000000058`` goes from Sscrofa9 (key 9) to 10.2 (102) at 66/67 and to 11.1 (111) at 89/90."""
    result = explained(pig.api, "ENSSSCG00000000058.1")
    name = "ENSSSCG00000000058.{}".format
    expected = [(1, 2, 66, 67, 0.729437, 9, 102), (2, 3, 89, 90, 0.942198, 102, 111)]
    expected += [(3, 4, 97, 98, 0.867324, 111, 111), (4, 5, 106, 107, 0.995732, 111, 111)]
    assert result.path.steps == tuple(
        PathStep(
            kind="history",
            source=name(a),
            target=name(b),
            key=0,
            old_release=old,
            new_release=new,
            weight=weight,
            assembly=assembly,
            old_assembly=old_assembly,
            reason="new_version",
        )
        for a, b, old, new, weight, old_assembly, assembly in expected
    )
    lives = [(1, 56, 66, 9), (2, 67, 89, 102), (3, 90, 97, 111), (4, 98, 106, 111), (5, 107, None, 111)]
    assert result.path.nodes == tuple(
        PathNode(id=name(v), kind="ensembl_gene", label=name(v), active=((a, b),), assemblies=(asm,))
        for v, a, b, asm in lives
    )
    assert [(c.gene, c.path) for c in result.candidates] == [(name(5), (0, 1, 2, 3))]
    # What Ensembl says: the scores of the events, and the assemblies that serve the releases.
    events = _events(pig)
    for step in result.path.steps:
        event = ("ENSSSCG00000000058", int(step.source[-1]), "ENSSSCG00000000058", int(step.target[-1]))
        assert step.weight == events[event][0]
        assert (step.old_assembly, step.assembly) == (pig.backbone[step.old_release], pig.backbone[step.new_release])
    assert _disagreements(pig, result, events) == []


def test_human_split_onto_alternative_loci(human):
    """``ENSG00000007816.4`` (NCBI36, outside the reference sequence) split at 54/55 into three ids on GRCh37."""
    result = explained(human.api, "ENSG00000007816.4", to_release=60)
    assert result.path.steps == (
        PathStep(
            kind="history",
            source="ENSG00000007816.4",
            target="ENSG00000227057.1",
            key=0,
            old_release=54,
            new_release=55,
            weight=0.908942,
            splits=2,
            assembly=37,
            old_assembly=36,
            reason="split",
        ),
        PathStep(
            kind="history",
            source="ENSG00000227057.1",
            target="ENSG00000227057.2",
            key=0,
            old_release=55,
            new_release=56,
            weight=0.725212,
            assembly=37,
            old_assembly=37,
            reason="new_version",
        ),
    )
    assert result.path.nodes == (
        PathNode(
            id="ENSG00000007816.4",
            kind="ensembl_gene",
            label="ENSG00000007816.4",
            active=((48, 54),),
            assemblies=(36,),
            non_reference=((48, 54),),
        ),
        PathNode(
            id="ENSG00000227057.1",
            kind="ensembl_gene",
            label="ENSG00000227057.1",
            active=((55, 55),),
            assemblies=(37,),
        ),
        PathNode(
            id="ENSG00000227057.2",
            kind="ensembl_gene",
            label="ENSG00000227057.2",
            active=((56, 70),),
            assemblies=(37,),
        ),
    )
    assert result.alternative_locus_genes == ("ENSG00000226916.2", "ENSG00000236222.2")
    # Ensembl's session of the split names the two other ids the old one passed to.
    events = _events(human)
    score, session = events[("ENSG00000007816", 4, "ENSG00000227057", 1)]
    others = {
        (b, vb) for (a, va, b, vb), (_, s) in events.items() if (a, va) == ("ENSG00000007816", 4) and s == session
    }
    assert score == 0.908942 and len(others - {("ENSG00000227057", 1)}) == result.path.steps[0].splits
    assert _disagreements(human, result, events) == []


@pytest.mark.parametrize("organism", ["sus_scrofa", "homo_sapiens"])
def test_real_paths_agree_with_ensembl(real_world, organism):
    """Genes of the first release, each converted to the snapshot: every step and gene node as Ensembl says."""
    world = real_world(organism)
    graph = world.api.graph
    first = min(world.backbone)
    genes = sorted(
        n
        for n in graph.nodes
        if graph.node_type(n) == "ensembl_gene"
        and graph.version(n) not in ("Void", "Retired")
        and graph.active_ranges(n)[0][0] == first
    )[:60]
    events = _events(world)
    checked = 0
    for gene in genes:
        result = explained(world.api, gene, from_release=first)
        assert _disagreements(world, result, events) == [], gene
        checked += bool(result.path.steps)
    assert checked >= 40
