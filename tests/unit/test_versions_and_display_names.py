"""The versions of accessions and the display names of genes: what the graph keeps of them and how they are read.

A world (:py:class:`tests.unit.test_graph_truth.World`, pig, releases 110 to 115 on assembly 111) is built by the real
builder. Each test names the plausible wrong version it fails under.
"""

from __future__ import annotations

import os

import pytest

from idtrack import API
from idtrack.build import builder as builder_module
from idtrack.build.builder import GraphMaker
from idtrack.engine.track import Track
from idtrack.errors import UnknownVersionError
from idtrack.graph import TheGraph
from tests.unit.test_graph_truth import World, lifetimes

RELEASES = range(110, 116)


def world() -> World:
    """Genes G1, G3, G4 to G8 at 110 to 115 (version 1), their transcripts ENSSSCT<n>.1, and their external ids.

    - ``XM_9`` (RefSeq) attached with version 3 to G3's transcript at 110 to 112 and with version 4 to G1's at 113 and
      114, at no release after (pig ``XM_003129161``: version 3 on ENSSSCG00000026015 at 67 to 77, version 4 on
      ENSSSCG00000009034 at 78 to 89);
    - ``P9`` (UniProt) attached with version 2 at 110 to 113 and 5 at 114 and 115 to G4's translation;
    - ``SLA`` (a symbol of VGNC, which gives no version) on G4, ``EMBL9`` of a database that gives no version on G4;
    - ``RPL31`` at 115: G5's display name and its VGNC symbol, and named for G6 by NCBI and UniProt's gene names (rhesus
      ``RPL31`` at 89); ``TIE1`` at 115: the display name of G7 and of G8; ``OLDDISPLAY`` at 110 to 115: G5's display
      name at 110 only, named for G6 by two databases.
    - ``2633``: a Havana transcript number of G5's transcript with version 1 at 110 and 111, and NCBI's gene id of G6 at
      every release (no version): one number of two numbering schemes (human ``2633``: a numeric Havana translation
      id of the NCBI36 releases and the NCBI gene id of GBP1).
    """
    genes = lifetimes({f"ENSSSCG{n}": [(110, 115, 1)] for n in (1, 3, 4, 5, 6, 7, 8)})
    externals: dict[int, list] = {}
    for r in RELEASES:
        rows = [
            ("ENSSSCG4.1", "SLA", "VGNC Symbol", 111),
            ("ENSSSCG4.1", "EMBL9", "EMBL", 111),
            ("ENSSSCP4.1", "P9", "Uniprot/SWISSPROT", 111, "translation", {"version": 2 if r <= 113 else 5}),
            ("ENSSSCG5.1", "OLDDISPLAY", "VGNC Symbol", 111, {"display": r == 110}),
            ("ENSSSCG6.1", "OLDDISPLAY", "NCBI gene (formerly Entrezgene)", 111),
            ("ENSSSCG6.1", "OLDDISPLAY", "UniProtKB Gene Name", 111),
            ("ENSSSCG6.1", "2633", "NCBI gene (formerly Entrezgene)", 111),
        ]
        if r <= 111:
            rows.append(("ENSSSCT5.1", "2633", "Havana transcript", 111, "transcript", {"version": 1}))
        if r <= 112:
            rows.append(("ENSSSCT3.1", "XM_9", "RefSeq_mRNA_predicted", 111, "transcript", {"version": 3}))
        elif r <= 114:
            rows.append(("ENSSSCT1.1", "XM_9", "RefSeq_mRNA_predicted", 111, "transcript", {"version": 4}))
        if r == 115:
            rows += [
                ("ENSSSCG5.1", "RPL31", "VGNC Symbol", 111, {"display": True}),
                ("ENSSSCG6.1", "RPL31", "NCBI gene (formerly Entrezgene)", 111),
                ("ENSSSCG6.1", "RPL31", "UniProtKB Gene Name", 111),
                ("ENSSSCG7.1", "TIE1", "VGNC Symbol", 111, {"display": True}),
                ("ENSSSCG8.1", "TIE1", "NCBI gene (formerly Entrezgene)", 111, {"display": True}),
            ]
        externals[r] = rows
    return World("sus_scrofa", {111: genes}, externals=externals)


@pytest.fixture
def graph(monkeypatch, tmp_path) -> TheGraph:
    """The built graph of the world."""
    os.makedirs(tmp_path, exist_ok=True)
    w = world()
    w.install(monkeypatch)
    return GraphMaker(w.manager(tmp_path, 111)).construct_graph()


def best(result) -> list[str]:
    """The genes of the best answer of an API result."""
    return sorted({c.gene for c in result.best})


def engine_best(conversion) -> list[str]:
    """The genes of the best answer of an engine conversion."""
    return sorted({c.gene for c in conversion.candidates if c.best})


# ------------------------------------------------------------------------------- the versions of accessions (I140)


def test_the_graph_keeps_the_versions_each_link_attached(graph):
    """The graph keeps per link and release the version Ensembl's cross reference gave the accession.

    Fails if the version is dropped at build (the rule before: ``XM_9`` was one accession without versions).
    """
    year = sum(1 << r for r in (110, 111, 112))
    assert graph.accession_versions("XM_9") == {3: year, 4: (1 << 113) | (1 << 114)}
    assert graph.accession_versions("P9") == {2: sum(1 << r for r in range(110, 114)), 5: (1 << 114) | (1 << 115)}
    assert graph.version_releases("XM_9", "ENSSSCT3.1", "RefSeq_mRNA_predicted", -1, 3) == year
    assert graph.version_releases("XM_9", "ENSSSCT1.1", "RefSeq_mRNA_predicted", -1, 3) == 0
    assert graph.accession_versions("SLA") == {} and graph.accession_versions("EMBL9") == {}


def test_a_versioned_accession_is_read_where_that_version_was_attached(graph):
    """Decision 14's rule for accessions: without a release, at the latest release that attached that version.

    ``XM_9.3`` is G3's (read at 112), ``XM_9.4`` G1's (read at 114); with a target release that holds the version
    there it is read at the target. Fails if the version is cut off and the accession read where it last is (the rule
    before: ``XM_9.3`` read at 114 answered G1).
    """
    assert graph.resolve_name("XM_9.3") == ("XM_9", "version")
    track = Track(graph)
    old = track.convert("XM_9.3")
    assert engine_best(old) == ["ENSSSCG3.1"] and old.from_release == 112
    assert engine_best(track.convert("XM_9.4")) == ["ENSSSCG1.1"]
    at_target = track.convert("XM_9.3", to_release=111)
    assert engine_best(at_target) == ["ENSSSCG3.1"] and at_target.from_release == 111
    # without a version the accession is read where it last is, as every identifier
    assert engine_best(track.convert("XM_9")) == ["ENSSSCG1.1"]
    # with a release that attached the version, there; at a release that did not, not read there
    assert engine_best(track.convert("XM_9.3", from_release=111)) == ["ENSSSCG3.1"]


def test_a_version_above_every_version_is_read_as_the_latest_and_flagged(graph):
    """A version higher than every version the links attached is read as the latest, and the result says so.

    ``P9`` is attached at 115, the graph's last release, with version 5: ``P9.6`` (the latest plus one) and ``P9.9`` may
    be newer than the graph. Fails if it is read as the accession whatever its version (no flag), refused, or if only a
    version two above the latest is read so.
    """
    for written in ("P9.6", "P9.9"):
        assert graph.resolve_name(written) == ("P9", "version_newer_than_graph"), written
        assert graph.reading_version(written, "P9", "version_newer_than_graph") == 5
    assert engine_best(Track(graph).convert("P9.6")) == ["ENSSSCG4.1"]


def test_a_version_above_every_version_of_an_accession_no_longer_attached_is_unknown(graph):
    """I150: decision 14's condition holds for accessions: newer than the graph only where attached at its last release.

    ``XM_9`` is attached at 110 to 114 only (versions 3 and 4): Ensembl no longer imports it at 115, so ``XM_9.5`` cannot
    be newer than Ensembl's import; it is ``unknown_version`` with the versions there are, as ``ENSSSCG...9`` of a
    retired id is. Fails if a version above every version is read as the latest whatever the accession's last release
    (the rule before: ``XM_9.5`` answered G1, the gene of version 4).
    """
    for written in ("XM_9.5", "XM_9.7"):
        assert graph.resolve_name(written) == (None, None), written
        assert graph.known_versions(written) == ("XM_9.3", "XM_9.4"), written
    with pytest.raises(UnknownVersionError):
        Track(graph).convert("XM_9.5")


def test_a_number_of_two_numbering_schemes_is_never_one_reading_silently(graph, monkeypatch, tmp_path):
    """I150: ``2633.1`` is read by the database whose ids have that form with that version at the release, else both.

    At 111 the Havana number has version 1 there: it is read so, alone (G5). At 115 (no release) no database has
    ``2633`` with a version: the version reading of 110 to 111 is taken by the rule of the latest release of a reading,
    and NCBI's gene id ``2633`` of G6, which the string also is with a suffix, is reported beside it
    (``unversioned_database``). ``2633.2``, a version no link attached, is ``unknown_version``, the other reading
    reported. Fails if the string is read as one database's id without the other being reported (the rule before: G5
    alone at 115), or if the other reading is reported where the version holds at the release.
    """
    readings = graph.readings("2633.1")
    assert readings.taken == (("2633", "version"),) and not readings.here
    assert readings.set_aside == (("2633", "exact", "unversioned_database"),)
    there = graph.readings("2633.1", from_release=111)
    assert there.taken == (("2633", "version"),) and there.here and there.set_aside == ()
    os.makedirs(tmp_path, exist_ok=True)
    world().install(monkeypatch)
    api = API.build("sus_scrofa", 115, str(tmp_path))
    result = api.convert("2633.1")
    assert best(result) == ["ENSSSCG5.1"] and result.from_release == 111
    assert [(o.graph_id, o.reason, o.genes) for o in result.other_readings] == [
        ("2633", "unversioned_database", ("ENSSSCG6.1",))
    ]
    assert api.convert("2633.1", from_release=111).other_readings == ()
    never = api.convert("2633.2")
    assert never.status == "unknown_version" and [o.reason for o in never.other_readings] == ["unversioned_database"]
    assert best(api.convert("2633")) == ["ENSSSCG6.1"]


def test_a_version_the_accession_never_had_has_no_reading(graph):
    """A version below or between the versions attached is ``unknown_version``, never another version's answer.

    Fails if a version never attached is read as the accession (the rule before: 93 to 100 % of family C's never-had
    lines were answered with the gene).
    """
    for written in ("XM_9.1", "XM_9.2", "P9.3", "P9.1"):
        assert graph.resolve_name(written) == (None, None), written
    assert graph.known_versions("XM_9.2") == ("XM_9.3", "XM_9.4")
    assert graph.known_versions("P9.4") == ("P9.2", "P9.5")
    assert graph.known_versions("XM_9.3") == () and graph.known_versions("P9.9") == ()
    with pytest.raises(UnknownVersionError) as caught:
        Track(graph).convert("XM_9.1")
    assert caught.value.versions == ("XM_9.3", "XM_9.4")


def test_which_databases_carry_versions_is_read_from_ensembl(graph):
    """A database whose cross references give no version is no versioned accession: nothing is cut off its ids.

    ``SLA.7`` (a symbol) and ``EMBL9.2`` (a database whose rows carry version 0) match nothing. Fails if a list of
    database names says which carry versions (the rule before: ``EMBL`` was one, so ``EMBL9.2`` answered G4).
    """
    assert graph.resolve_name("SLA.7") == (None, None) and graph.known_versions("SLA.7") == ()
    assert graph.resolve_name("EMBL9.2") == (None, None) and graph.known_versions("EMBL9.2") == ()


def test_the_api_reports_the_version_reading(monkeypatch, tmp_path):
    """The API reads a versioned accession by the same rule and reports the match, a fallback and an unknown version.

    Fails if the API and the engine read the string at different releases, or a batch takes ``XM_9.3`` as a copy of
    ``XM_9`` (decision 82: it would then answer G1).
    """
    os.makedirs(tmp_path, exist_ok=True)
    world().install(monkeypatch)
    api = API.build("sus_scrofa", 115, str(tmp_path))
    old = api.convert("XM_9.3")
    assert (old.match, old.graph_id, best(old), old.from_release) == ("version", "XM_9", ["ENSSSCG3.1"], 112)
    newer = api.convert("P9.8")
    assert (newer.match, best(newer)) == ("version_newer_than_graph", ["ENSSSCG4.1"])
    never = api.convert("XM_9.2")
    assert (never.status, never.reason, never.known_versions) == (
        "unknown_version",
        "unknown_version",
        ("XM_9.3", "XM_9.4"),
    )
    moved = api.convert("XM_9.3", from_release=114)  # version 3 was not attached at 114: read by the rule without it
    assert moved.from_release_fallback and moved.from_release == 112 and best(moved) == ["ENSSSCG3.1"]
    batch = api.convert_many(["XM_9", "XM_9.3", "XM_9.2"])
    assert [best(r) for r in batch[:2]] == [["ENSSSCG1.1"], ["ENSSSCG3.1"]]
    assert batch[1].match == "version" and batch[2].status == "unknown_version"


def test_a_saved_graph_keeps_the_versions_and_the_display_names(graph, tmp_path):
    """The graph file holds the versions and the display names; the loaded graph reads them as the built one.

    Fails if either is lost or changed by the file (the content digest covers both).
    """
    path = os.path.join(tmp_path, "saved.npz")
    graph.save(path)
    loaded = TheGraph.load(path)
    for node in ("XM_9", "P9", "SLA"):
        assert loaded.accession_versions(node) == graph.accession_versions(node)
    assert loaded.display_releases("RPL31", "ENSSSCG5.1") == graph.display_releases("RPL31", "ENSSSCG5.1") == 1 << 115
    assert loaded.record.counts["accession_versions"] == graph.record.counts["accession_versions"] > 0


# ------------------------------------------------------------------------------- Ensembl's display name first (I144)


def test_the_gene_whose_display_name_it_is_ranks_first(graph):
    """Among the genes that carry a name at one release, the gene Ensembl displays it for is the best answer.

    ``RPL31`` at 115: G5's display name (one database) and a name of G6 by two databases. Fails if support decides
    first (the rule before: G6, decided by ``entry_support``; rhesus 713 probe lines were a tie or the other gene).
    """
    conversion = Track(graph).convert("RPL31", from_release=115)
    assert engine_best(conversion) == ["ENSSSCG5.1"] and conversion.decided_by == "display_name"
    shown = {c.gene: c.display_name for c in conversion.candidates}
    assert shown == {"ENSSSCG5.1": True, "ENSSSCG6.1": False}
    assert graph.display_releases("RPL31", "ENSSSCG6.1") == 0


def test_two_genes_that_both_display_a_name_stay_a_tie(graph):
    """Two genes that both display the name at the release are equal in it: still a tie.

    Fails if the mark ranks one display before another (by database, by order).
    """
    conversion = Track(graph).convert("TIE1", from_release=115)
    assert conversion.tie and engine_best(conversion) == ["ENSSSCG7.1", "ENSSSCG8.1"]


def test_a_display_name_of_another_release_does_not_rank(graph):
    """The display name counts at the release it is read at only: ``OLDDISPLAY`` was G5's at 110, not at 115.

    At 115 support decides (G6, two databases); at 110 the display name does (G5). Fails if the mark is read over
    every release of the link.
    """
    track = Track(graph)
    now = track.convert("OLDDISPLAY", from_release=115)
    assert engine_best(now) == ["ENSSSCG6.1"] and now.decided_by == "entry_support"
    then = track.convert("OLDDISPLAY", from_release=110, to_release=110)
    assert engine_best(then) == ["ENSSSCG5.1"] and then.decided_by == "display_name"


def test_a_gene_merged_from_several_entry_genes_has_the_display_name_of_any_of_them(monkeypatch, tmp_path):
    """The display mark and the support of a gene are the best of every entry gene whose history reaches it.

    ``MERGED1`` at 110 is the display name of G11 and G14 and a name of G12; G11 and G12 merge into G13 at 111 (G12 with
    the better score, so the one path the search keeps to G13 starts from G12); G14 lives on (pig ``C/EBP`` from 78 to
    115: ENSSSCG00000028739 and ENSSSCG00000024816 merged into ENSSSCG00000034207, ENSSSCG00000002866 displays it
    too). G13 and G14 are equal in the evidence: a tie. Fails if the mark is read from the kept path's entry gene
    alone (G14 alone, decided by ``display_name``).
    """
    os.makedirs(tmp_path, exist_ok=True)
    genes = lifetimes(
        {
            "ENSSSCG11": [(110, 110, 1)],
            "ENSSSCG12": [(110, 110, 1)],
            "ENSSSCG13": [(111, 115, 1)],
            "ENSSSCG14": [(110, 115, 1)],
        }
    )
    rows = [
        ("ENSSSCG11.1", "MERGED1", "UniProtKB Gene Name", 111, {"display": True}),
        ("ENSSSCG12.1", "MERGED1", "UniProtKB Gene Name", 111),
        ("ENSSSCG14.1", "MERGED1", "UniProtKB Gene Name", 111, {"display": True}),
    ]
    w = World(
        "sus_scrofa",
        {111: genes},
        events=[(1, "ENSSSCG11", 1, "ENSSSCG13", 1, 0.5), (1, "ENSSSCG12", 1, "ENSSSCG13", 1, 0.9)],
        sessions=[(1, 110, 111)],
        externals={r: rows if r == 110 else [] for r in RELEASES},
    )
    w.install(monkeypatch)
    g = GraphMaker(w.manager(tmp_path, 111)).construct_graph()
    conversion = Track(g).convert("MERGED1", from_release=110, to_release=115)
    assert conversion.tie and engine_best(conversion) == ["ENSSSCG13.1", "ENSSSCG14.1"]
    merged = next(c for c in conversion.candidates if c.gene == "ENSSSCG13.1")
    assert merged.entry_gene == "ENSSSCG12.1" and merged.display_name and merged.display_rank == 0


# ----------------------------------------------------------------------------- the facts of a gene in one place (I145)


def test_the_builder_and_the_graph_assemble_equal_facts(monkeypatch, tmp_path):
    """The facts of the one rule of a gene are assembled in one place; the builder and the built graph get equal ones.

    Fails if either assembles them another way (the builder once gave versions without places no releases while the
    graph gave them those of their history).
    """
    seen = []
    real = builder_module.facts_by_stable_id

    def kept(*args, **kwargs):
        seen.append(real(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(builder_module, "facts_by_stable_id", kept)
    os.makedirs(tmp_path, exist_ok=True)
    # ENSSSCG1 becomes ENSSSCG2 between 112 and 113 (a step of Ensembl's history between two ids); ENSSSCG4 lies on a
    # patch beside the reference sequence at 113 to 115
    genes = lifetimes(
        {
            "ENSSSCG1": [(110, 112, 1)],
            "ENSSSCG2": [(113, 115, 1)],
            "ENSSSCG3": [(110, 115, 1)],
            "ENSSSCG4": [(110, 115, 1)],
        }
    )
    w = World(
        "sus_scrofa",
        {111: genes},
        events=[(7, "ENSSSCG1", 1, "ENSSSCG2", 1, 0.9)],
        sessions=[(7, 112, 113)],
        non_reference={111: {release: ["ENSSSCG4"] for release in (113, 114, 115)}},
    )
    w.install(monkeypatch)
    built = GraphMaker(w.manager(tmp_path, 111)).construct_graph()
    assert len(seen) == 1
    facts, joined = built._compute_genes()
    assert seen[0] == (facts, joined)
    assert joined["ENSSSCG1"] == frozenset({"ENSSSCG2"}) and facts["ENSSSCG4"].outside_at


# ------------------------------------------------------------------- an edit counter is no version (decision 101)


def counters_world() -> World:
    """Three databases of versioned accessions on transcripts of three genes, at 110 to 115.

    - ``SEQ1`` of ``RefSeq_mRNA``: version 3 at every release (a version of the sequence: 0 of 5 pairs differ);
    - ``EDIT1`` of ``Uniprot/SPTREMBL``: version 20 + the release's offset at every release (UniProt's entry version:
      5 of 5 pairs differ);
    - ``HALF1`` of ``CCDS``: at 110 to 114 only, versions 1, 1, 2, 2, 3 (2 of 4 pairs differ: exactly one half, the
      bound, which counts as a version of the sequence);
    - ``ABOVE1`` of ``Uniprot/SWISSPROT``: versions 1, 2, 2, 3, 3, 4 (3 of 5 pairs differ: just above one half, an edit
      counter) and ``BELOW1`` of ``RefSeq_ncRNA``: versions 1, 1, 1, 2, 2, 3 (2 of 5: just below, a version);
    - ``EDIT2`` of ``Uniprot/SPTREMBL`` at 113 to 115 only, its version the release's offset (2 of 2 pairs differ).
    """
    genes = lifetimes({f"ENSSSCG{n}": [(110, 115, 1)] for n in (1, 2, 3)})
    externals: dict[int, list] = {}
    for r in RELEASES:
        rows = [
            ("ENSSSCT1.1", "SEQ1", "RefSeq_mRNA", 111, "transcript", {"version": 3}),
            ("ENSSSCT2.1", "EDIT1", "Uniprot/SPTREMBL", 111, "transcript", {"version": 20 + r - 110}),
            ("ENSSSCT1.1", "ABOVE1", "Uniprot/SWISSPROT", 111, "transcript", {"version": [1, 2, 2, 3, 3, 4][r - 110]}),
            ("ENSSSCT3.1", "BELOW1", "RefSeq_ncRNA", 111, "transcript", {"version": [1, 1, 1, 2, 2, 3][r - 110]}),
        ]
        if r >= 113:
            rows.append(("ENSSSCT1.1", "EDIT2", "Uniprot/SPTREMBL", 111, "transcript", {"version": r - 110}))
        if r <= 114:
            rows.append(
                (
                    "ENSSSCT3.1",
                    "HALF1",
                    "CCDS",
                    111,
                    "transcript",
                    {"version": {110: 1, 111: 1, 112: 2, 113: 2, 114: 3}[r]},
                )
            )
        externals[r] = rows
    return World("sus_scrofa", {111: genes}, externals=externals)


def test_a_number_that_changes_with_nearly_every_release_is_no_version(monkeypatch, tmp_path):
    """Decision 101: the share of consecutive-release pairs of one link whose version differs, measured from the graph.

    Above one half (``EDIT1`` and ``EDIT2``: 7 of 7, ``ABOVE1``: 3 of 5) the database's number is an edit counter; at
    the bound (``HALF1``: 2 of 4) and below (``BELOW1``: 2 of 5, ``SEQ1``: 0 of 5) it is a version of the sequence.
    Fails if the property is dropped (every database's number dated the accession), if the bound itself falls on the
    other side, or if the bound is set elsewhere (at 0.95, ``ABOVE1``'s 0.6 would date it; at 0.3, ``BELOW1``'s 0.4
    would not).
    """
    os.makedirs(tmp_path, exist_ok=True)
    w = counters_world()
    w.install(monkeypatch)
    g = GraphMaker(w.manager(tmp_path, 111)).construct_graph()
    shares = g.version_change_shares()
    assert shares["RefSeq_mRNA"] == (5, 0) and shares["Uniprot/SPTREMBL"] == (7, 7) and shares["CCDS"] == (4, 2)
    assert shares["Uniprot/SWISSPROT"] == (5, 3) and shares["RefSeq_ncRNA"] == (5, 2)
    assert g.edit_counter_databases() == frozenset({"Uniprot/SPTREMBL", "Uniprot/SWISSPROT"})
    assert g.resolve_name("ABOVE1.1") == ("ABOVE1", "version_not_judged")
    assert g.resolve_name("BELOW1.2") == ("BELOW1", "version") and g.known_versions("BELOW1.4") == ()
    # an edit counter: the version is not judged, whatever it is, and the accession is read without it
    for written in ("EDIT1.20", "EDIT1.23", "EDIT1.1", "EDIT1.999"):
        assert g.resolve_name(written) == ("EDIT1", "version_not_judged"), written
        assert g.known_versions(written) == (), written
    track = Track(g)
    plain = track.convert("EDIT1")
    for written in ("EDIT1.20", "EDIT1.1"):
        conversion = track.convert(written)
        assert engine_best(conversion) == engine_best(plain) == ["ENSSSCG2.1"]
        assert conversion.from_release == plain.from_release == 115  # not where version 20 was attached (110)
    # versions of the sequence, the bound included: judged as before
    assert g.resolve_name("HALF1.2") == ("HALF1", "version") and track.convert("HALF1.2").from_release == 113
    # the latest version plus one of an accession attached at the last release is read as the latest (M18 of the
    # audit's mutants: a rule that reads only higher versions so is caught); HALF1, no longer attached at 115, is not
    assert g.resolve_name("SEQ1.4") == ("SEQ1", "version_newer_than_graph")
    assert g.resolve_name("BELOW1.4") == ("BELOW1", "version_newer_than_graph")
    assert g.resolve_name("HALF1.9") == (None, None) and g.known_versions("HALF1.9") == (
        "HALF1.1",
        "HALF1.2",
        "HALF1.3",
    )
    assert g.resolve_name("SEQ1.1") == (None, None) and g.known_versions("SEQ1.1") == ("SEQ1.3",)


def test_the_api_says_the_version_was_not_judged(monkeypatch, tmp_path):
    """The result of an edit counter's accession says so by a reason of its own, never ``unknown_version``.

    Fails if the version is cut off silently (no reason) or the string is refused as a version never had.
    """
    os.makedirs(tmp_path, exist_ok=True)
    counters_world().install(monkeypatch)
    api = API.build("sus_scrofa", 115, str(tmp_path))
    result = api.convert("EDIT1.1")
    assert (result.status, result.reason, result.match, result.graph_id) == (
        "converted",
        "version_not_judged",
        "version_not_judged",
        "EDIT1",
    )
    assert best(result) == ["ENSSSCG2.1"] and result.known_versions == ()
    batch = api.convert_many(["EDIT1", "EDIT1.1"])
    assert batch[1].match == "version_not_judged"  # a version of the accession, not decision 82's copy
    assert result.reasons == ("version_not_judged",)


def test_a_result_carries_every_reason_that_applies(monkeypatch, tmp_path):
    """I154: a result read at another release whose version was not judged says both, the fallback first.

    ``EDIT2`` is attached at 113 to 115 only: asked at 110 it is read at 115 (``read_at_another_release``), and its
    version is an edit counter's (``version_not_judged``). Fails if one reason overwrites the other (the rule before:
    ``reason`` was ``version_not_judged`` alone, and the fallback was lost to readers of ``reason``).
    """
    os.makedirs(tmp_path, exist_ok=True)
    counters_world().install(monkeypatch)
    api = API.build("sus_scrofa", 115, str(tmp_path))
    result = api.convert("EDIT2.9", from_release=110)
    assert result.from_release_fallback and result.from_release == 115 and best(result) == ["ENSSSCG1.1"]
    assert result.reasons == ("read_at_another_release", "version_not_judged")
    assert result.reason == "read_at_another_release"
    plain = api.convert("EDIT2", from_release=110)
    assert plain.reasons == ("read_at_another_release",) and api.convert("EDIT2").reasons == ()
    missing = api.convert("SEQ1.1")
    assert missing.reasons == ("unknown_version",) == (missing.reason,)
    assert type(result).from_dict(result.to_dict()) == result


# -------------------------------------------------------------- the display name holds for the spelling typed (I152)


def spelling_world() -> World:
    """Mouse genes G1 to G4 at 110 to 115 (assembly 39) whose display names differ from other names in letter case.

    - ``Foxl1``: G1's MGI symbol and display name; for G2 NCBI's label ``Foxl1`` and UniProt's gene name ``FOXL1``, which
      Ensembl displays for G2 (mouse ``Foxl1`` at 70: ENSMUSG00000043867 and ENSMUSG00000097084);
    - ``Gapd``: G3's MGI symbol and display name, UniProt's ``GAPD`` for G3 too; for G4 UniProt's gene name ``GAPD``
      (its display name) and NCBI's label ``GAPD`` (mouse ``GAPDH`` at 70: ENSMUSG00000057666 and ENSMUSG00000097148);
    - ``Mgi1`` to ``Mgi3``: further MGI symbols of G1, so that most of the organism's symbols have a small letter.
    """
    genes = lifetimes({f"ENSMUSG{n}": [(110, 115, 1)] for n in (1, 2, 3, 4)})
    rows = [
        ("ENSMUSG1.1", "Foxl1", "MGI Symbol", 39, {"display": True}),
        ("ENSMUSG2.1", "Foxl1", "NCBI gene (formerly Entrezgene)", 39),
        ("ENSMUSG2.1", "FOXL1", "UniProtKB Gene Name", 39, {"display": True}),
        ("ENSMUSG3.1", "Gapd", "MGI Symbol", 39, {"display": True}),
        ("ENSMUSG3.1", "GAPD", "UniProtKB Gene Name", 39),
        ("ENSMUSG4.1", "GAPD", "UniProtKB Gene Name", 39, {"display": True}),
        ("ENSMUSG4.1", "GAPD", "NCBI gene (formerly Entrezgene)", 39),
        # MGI's other symbols, written in mixed case as MGI writes them (the organism's convention, read from the graph)
        *(("ENSMUSG1.1", f"Mgi{n}", "MGI Symbol", 39) for n in range(1, 4)),
    ]
    return World("mus_musculus", {39: genes}, externals={r: list(rows) for r in RELEASES})


def with_display_databases(g: TheGraph, shown: dict[tuple[str, str], str]) -> TheGraph:
    """Set the databases of the graph's display names by hand (``""``: a graph that does not keep them)."""
    from idtrack.graph.store import _LinkMarks

    store = g._frozen()
    every = sum(1 << r for r in RELEASES)
    store.displays = _LinkMarks.of(
        {
            (store.index[str(g.resolve_name(name)[0])], store.index[gene], database, 0): every
            for (name, gene), database in shown.items()
        }
    )
    return g


@pytest.fixture
def spelt(monkeypatch, tmp_path) -> TheGraph:
    """The built graph of the spelling world: the builder keeps the database of each display name."""
    os.makedirs(tmp_path, exist_ok=True)
    w = spelling_world()
    w.install(monkeypatch)
    return GraphMaker(w.manager(tmp_path, 39)).construct_graph()


def test_the_builder_keeps_the_database_of_each_display_name(spelt):
    """The display marks of a built graph name the database of the display name, so its spelling is known.

    Fails if the builder keeps the mark without its database (``""``: no spelling to compare).
    """
    node = spelt.resolve_name("Foxl1")[0]
    every = sum(1 << r for r in RELEASES)
    assert spelt.display_links(node, "ENSMUSG1.1") == {"MGI Symbol": every}
    assert spelt.display_links(node, "ENSMUSG2.1") == {"UniProtKB Gene Name": every}


def test_the_display_name_holds_for_the_spelling_typed(spelt):
    """Decision 102: a string with letters of both cases is the display name only where its database spelt it so.

    ``Foxl1`` is G1's display name (MGI) and, for G2, NCBI's label; G2 displays ``FOXL1``. G1 is the answer by the
    display name. Fails if the mark is read over every spelling of the node (the rule before: both genes displayed it,
    and support decided or tied).
    """
    assert spelt.symbol_case() == "mixed"
    conversion = Track(spelt).convert("Foxl1")
    assert engine_best(conversion) == ["ENSMUSG1.1"] and conversion.decided_by == "display_name"
    ranks = {c.gene: (c.display_name, c.display_rank) for c in conversion.candidates}
    assert ranks == {"ENSMUSG1.1": (True, 0), "ENSMUSG2.1": (False, 2)}


def test_a_string_in_one_case_ranks_the_authority_display_first(spelt):
    """Decision 102: a string in one letter case carries no evidence of case where symbols are written in mixed case.

    ``GAPD`` is spelt so for G3 (UniProt) and G4 (UniProt, NCBI: more support); G4 displays ``GAPD`` (UniProt), G3
    displays ``Gapd`` (MGI, the authority). The authority's display name, read without case, ranks first: G3, G4 reported
    after it. Fails if the display name is taken by the spelling typed (G4, the UniProt-named gene) or ignored (G4, by
    support).
    """
    conversion = Track(spelt).convert("GAPD")
    assert engine_best(conversion) == ["ENSMUSG3.1"] and conversion.decided_by == "display_name"
    ranks = {c.gene: c.display_rank for c in conversion.candidates}
    assert ranks == {"ENSMUSG3.1": 0, "ENSMUSG4.1": 1}
    node = spelt.resolve_name("GAPD")[0]
    assert [spelt.display_rank(node, gene, 115, "gapd") for gene in ("ENSMUSG3.1", "ENSMUSG4.1")] == [0, 1]


def test_a_graph_without_the_databases_of_its_display_names_keeps_the_mark(monkeypatch, tmp_path):
    """Where the graph does not say which database displays a name, the mark holds as before (no spelling to compare).

    Fails if such a graph loses every display name (``""`` read as no database at all).
    """
    os.makedirs(tmp_path, exist_ok=True)
    w = spelling_world()
    w.install(monkeypatch)
    g = GraphMaker(w.manager(tmp_path, 39)).construct_graph()
    with_display_databases(g, {("Foxl1", "ENSMUSG1.1"): "", ("Foxl1", "ENSMUSG2.1"): ""})
    node = g.resolve_name("Foxl1")[0]
    assert set().union(*(g.display_links(node, gene) for gene in ("ENSMUSG1.1", "ENSMUSG2.1"))) == {""}
    assert g.display_rank(node, "ENSMUSG1.1", 115, "Foxl1") == g.display_rank(node, "ENSMUSG2.1", 115, "Foxl1") == 0


def test_in_an_organism_of_capitals_a_string_in_one_case_is_read_without_case(graph):
    """Pig writes symbols in capitals: ``TIE1``, displayed for G7 (VGNC) and G8 (NCBI), stays a tie by the display name.

    Fails if a one-case string is ranked by the authority's display where the organism's symbols are in capitals.
    """
    node = graph.resolve_name("TIE1")[0]
    assert graph.display_links(node, "ENSSSCG7.1") == {"VGNC Symbol": 1 << 115}
    assert graph.symbol_case() == "capitals"
    conversion = Track(graph).convert("TIE1")
    assert engine_best(conversion) == ["ENSSSCG7.1", "ENSSSCG8.1"] and conversion.tie
