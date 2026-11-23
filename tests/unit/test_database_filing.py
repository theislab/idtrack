"""Decision 104: which of Ensembl's databases a selected database is at each release (:py:mod:`idtrack.build.continuity`).

The links are hand-made: per release, rows ``(name, identifier, gene)`` as :py:meth:`DatabaseManager._release_links`
gives them (accessions by their database's name, labels by the name of its labels, every form together).
"""

from __future__ import annotations

import pandas as pd
import pytest

from idtrack.build import continuity
from idtrack.build.continuity import database_filing, identifier_form, link_hashes


def _links(table: dict[int, list[tuple]]):
    """Return ``links_of(release)`` over hand-made rows ``{release: [(name, identifier, gene[, form]), ...]}``."""

    def links_of(release: int) -> pd.DataFrame:
        rows = [(*row, "gene") if len(row) == 3 else row for row in table.get(release, [])]
        return pd.DataFrame(rows, columns=["name", "identifier", "gene", "form"])

    return links_of


def _on(form: str, *names: str) -> set[tuple[str, str]]:
    """The selection of some names on one form."""
    return {(form, name) for name in names}


def _genes(n: int, start: int = 1) -> list[str]:
    return [f"ENSG{start + k:011d}" for k in range(n)]


class TestIdentifierForm:
    """The form of an identifier (:py:func:`idtrack.build.continuity.identifier_form`)."""

    def test_forms(self):
        """The characters before the first digit, then ``#`` for a number (with a version) and ``*`` for any other rest."""
        assert [identifier_form(x) for x in ("NM_000546", "NM_000546.6", "NR_1", "OTTMUSG00000003802")] == [
            "NM_#",
            "NM_#",
            "NR_#",
            "OTTMUSG#",
        ]
        assert [identifier_form(x) for x in ("2633", "SMURF1", "5S_rRNA", "C9orf72", "abc")] == [
            "#",
            "SMURF#",
            "*",
            "C*",
            "abc",
        ]


class TestRenamedDatabase:
    """Rule 2: a database whose links move into a selected one between adjacent releases is that database."""

    def test_refseq_dna_is_refseq_mrna_and_ncrna_by_the_identifiers_form(self):
        """Pig and rhesus ``RefSeq_dna`` (to 63) holds ``NM_`` and ``NR_`` accessions; from 64 they are mRNA and ncRNA.

        Its ``NM_`` accessions continue RefSeq_mRNA, its ``NR_`` ones RefSeq_ncRNA, at every release it has (62, 63 here).
        Fails if the database is taken whole by its main continuation (``NR_`` as mRNA), or not taken (the rule off), or
        if an identifier of another form than the main continuation's scheme is taken for it.
        """
        genes = _genes(12)
        mrna = [(f"NM_{k:06d}", g) for k, g in enumerate(genes[:10])]
        ncrna = [(f"NR_{k:06d}", g) for k, g in enumerate(genes[10:])]
        table = {
            62: [("RefSeq_dna", i, g) for i, g in mrna + ncrna],
            63: [("RefSeq_dna", i, g) for i, g in mrna + ncrna],
            64: [("RefSeq_mRNA", i, g) for i, g in mrna] + [("RefSeq_ncRNA", i, g) for i, g in ncrna],
        }
        selected = _on("gene", "RefSeq_mRNA", "RefSeq_ncRNA")
        filing = database_filing(_links(table), [62, 63, 64], selected, {})
        assert filing.continued("RefSeq_dna", "NM_#") == "RefSeq_mRNA"
        assert filing.continued("RefSeq_dna", "NR_#") == "RefSeq_ncRNA"
        # a form that never moved, other than the scheme of the main continuation (NM_#): no RefSeq mRNA accession
        assert filing.schemes["RefSeq_mRNA"] == "NM_#" and filing.continued("RefSeq_dna", "XM_#") is None
        assert filing.database_names({"RefSeq_mRNA"})["RefSeq_mRNA"] == {"RefSeq_mRNA": (64,), "RefSeq_dna": (62, 63)}

    def test_a_database_beside_the_selected_one_is_not_it(self):
        """A mirror of NCBI's ids (WikiGene, on the same genes at the same releases) moves no link: it is not EntrezGene.

        At the boundary where EntrezGene moves from translations to genes (one name on every form) the links are
        EntrezGene's on both sides. Fails if links shared with a selected database are taken as moved.
        """
        genes = _genes(10)
        table = {
            r: [("EntrezGene", str(1000 + k), g) for k, g in enumerate(genes)]
            + [("WikiGene", str(1000 + k), g) for k, g in enumerate(genes)]
            for r in (62, 63)
        }
        filing = database_filing(_links(table), [62, 63], _on("gene", "EntrezGene"), {})
        assert not filing.renamed

    def test_the_share_that_must_move(self):
        """A database half of whose links move is taken (the bound); one that moves less is not.

        The renames measured move 0.62 to 1.00 of a database's links, the databases beside a selected one at most
        0.40 (docs/changes.rst). Fails if the bound is not the share of the candidate's own links at its side.
        """
        genes = _genes(10)
        names = [f"SYM{k}" for k in range(10)]

        def world(moving: int) -> dict:
            return {
                60: [("HGNC_curated_gene label", n, g) for n, g in zip(names, genes)],
                61: [("HGNC Symbol", n, g) for n, g in zip(names[:moving], genes[:moving])]
                + [("HGNC_curated_gene label", n, g) for n, g in zip(names[moving:], genes[moving:])],
            }

        assert (
            database_filing(_links(world(5)), [60, 61], _on("gene", "HGNC Symbol"), {}).continued(
                "HGNC_curated_gene label", "SYM#"
            )
            == "HGNC Symbol"
        )
        assert not database_filing(_links(world(4)), [60, 61], _on("gene", "HGNC Symbol"), {}).renamed

    def test_a_database_that_ends_hands_over_to_the_one_its_links_make(self):
        """Decision 107: a database that ends at a boundary continues the selected one its moved links are half of.

        Human ``RefSeq_dna_predicted`` (to 63) moves 31 % of its links at 63 to 64 (Ensembl dropped the rest), and they
        are 59 % of ``RefSeq_mRNA_predicted``'s links at 64: 3 of 10 links move here, the receiver has 4 at 64. Fails if
        only the leaving side's share decides (not taken), if a database that still holds links after the boundary is
        taken so (it is beside the receiver, not its earlier name), or if the receiver's share is not bounded. A database
        that ends at the boundary and holds a few links again later (human ``RefSeq_dna_predicted`` at 76) still ends
        there.
        """
        genes = _genes(12)
        accessions = [(f"XM_{k:06d}", g) for k, g in enumerate(genes[:10])]
        selected = _on("transcript", "RefSeq_mRNA_predicted")

        def world(received_too: int, stays: bool) -> dict:
            later = [("RefSeq_mRNA_predicted", i, g, "transcript") for i, g in accessions[:3]]
            later += [("RefSeq_mRNA_predicted", f"XM_9{k:05d}", g, "transcript") for k, g in enumerate(genes[10:])][
                :received_too
            ]
            if stays:
                later += [("RefSeq_dna_predicted", i, g, "transcript") for i, g in accessions[3:]]
            return {
                63: [("RefSeq_dna_predicted", i, g, "transcript") for i, g in accessions],
                64: later,
                65: later + [("RefSeq_dna_predicted", "XM_7", genes[0], "transcript")],  # a remnant, later
            }

        def filed(table: dict) -> str | None:
            filing = database_filing(_links(table), [63, 64, 65], selected, {})
            return filing.continued("RefSeq_dna_predicted", "XM_#")

        assert filed(world(1, stays=False)) == "RefSeq_mRNA_predicted"  # 3 of the receiver's 4
        assert filed(world(1, stays=True)) is None  # 3 of its 10 move and it goes on: a database beside the receiver
        assert filed(world(2, stays=False)) == "RefSeq_mRNA_predicted"  # 3 of 5: still at least half
        many = world(2, stays=False)
        many[64] += [("RefSeq_mRNA_predicted", f"XM_8{k:05d}", f"ENSG9{k:010d}", "transcript") for k in range(2)]
        assert filed(many) is None  # 3 of 7: below half of the receiver's links

    def test_a_database_with_a_selected_name_is_never_another_ones(self):
        """The accessions of a database whose labels are selected are not taken for another selected database.

        Pig ``HGNC`` accessions (the projected symbols) move to VGNC Symbol at 101; the selection holds HGNC's labels
        (HGNC Symbol), not its accessions. Fails if every unselected name is a candidate.
        """
        genes = _genes(10)
        table = {
            100: [("HGNC", f"S{k}", g) for k, g in enumerate(genes)]
            + [("HGNC Symbol", f"S{k}", g) for k, g in enumerate(genes)],
            101: [("VGNC Symbol", f"S{k}", g) for k, g in enumerate(genes)],
        }
        database_of = {"HGNC": {"HGNC"}, "HGNC Symbol": {"HGNC"}, "VGNC Symbol": {"VGNC"}}
        filing = database_filing(_links(table), [100, 101], _on("gene", "HGNC Symbol", "VGNC Symbol"), database_of)
        assert not filing.renamed


class TestWhatMoves:
    """What counts as a link that moves, and where a candidate's identifiers go."""

    def test_identifiers_are_compared_as_written(self):
        """Mouse: human symbols that Ensembl displayed on a few mouse genes (``HGNC Symbol``, ``ABCA1``) are not MGI's.

        From the next release MGI gives those genes ``Abca1``. Fails if identifiers are compared without letter case
        (HGNC Symbol would be read as an earlier name of MGI Symbol).
        """
        genes = _genes(6)
        table = {
            75: [("HGNC Symbol", f"ABCA{k}", g) for k, g in enumerate(genes)],
            76: [("MGI Symbol", f"Abca{k}", g) for k, g in enumerate(genes)],
        }
        assert not database_filing(_links(table), [75, 76], _on("gene", "MGI Symbol"), {}).renamed

    def test_a_form_goes_elsewhere_only_as_that_databases_scheme(self):
        """Rhesus HGNC_curated_gene symbols: a few of a form move to NCBI's symbols at the boundary; they stay HGNC's.

        Fails if a candidate's identifiers of a form go to whichever selected database most of their moved links went
        to, also where that form is no identifier scheme of it (symbols of HGNC filed as NCBI's by chance).
        """
        names = ["ACTB", "GAPDH", "TP53", "MYC", "SOX2", "PAX6", "CDK1", "BRCA1", "KRAS", "EGFR"]
        genes = _genes(len(names) + 3)
        ncbi = [("NCBI gene", n, g) for n, g in zip(["LOC1", "Mt-co1", "ZNF2a"], genes[len(names) :])]
        table = {
            60: [("HGNC_curated_gene label", n, g) for n, g in zip(names, genes)]
            + [("HGNC_curated_gene label", "ZAK", genes[0])]
            + ncbi,
            61: [("HGNC Symbol", n, g) for n, g in zip(names, genes)] + [("NCBI gene", "ZAK", genes[0])] + ncbi,
        }
        filing = database_filing(_links(table), [60, 61], _on("gene", "HGNC Symbol", "NCBI gene"), {})
        assert filing.continued("HGNC_curated_gene label", "ACTB") == "HGNC Symbol"
        assert filing.continued("HGNC_curated_gene label", "ZAK") == "HGNC Symbol"


class TestLaterName:
    """A database that continues a selected one after Ensembl renamed the selected one."""

    def test_a_selected_database_renamed_later_is_continued_by_its_new_name(self):
        """A selected name that Ensembl retires, its links filed under a new name from the next release, is that name.

        The new name is the selected database (its later name), at its first release made of the selected database's
        links. Fails if only earlier names are found (a selection by an old name loses the database from the rename on).
        """
        genes = _genes(10)
        table = {
            62: [("Uniprot_genename label", f"GENE{k}", g) for k, g in enumerate(genes)],
            63: [("UniProtKB Gene Name", f"GENE{k}", g) for k, g in enumerate(genes)],
        }
        filing = database_filing(_links(table), [62, 63], _on("gene", "Uniprot_genename label"), {})
        assert filing.continued("UniProtKB Gene Name", "GENE#") == "Uniprot_genename label"


class TestAnotherForm:
    """A selected database's name on a form the selection does not hold is judged by its links like another database."""

    def test_ncbi_ids_on_translations_continue_ncbi_ids_on_genes(self):
        """Ensembl attached NCBI's ids to translations up to 62 and to genes from 63: the translations' are NCBI's.

        Fails if a selected database is read only on the forms the selection names (round 1 listed NCBI on every form
        to hold them; the same holds for Rfam and miRBase on transcripts up to 62).
        """
        genes = _genes(10)
        table = {
            62: [("EntrezGene", str(100 + k), g, "translation") for k, g in enumerate(genes)],
            63: [("EntrezGene", str(100 + k), g, "gene") for k, g in enumerate(genes)],
        }
        filing = database_filing(_links(table), [62, 63], _on("gene", "EntrezGene"), {})
        assert filing.continued("EntrezGene", "#", "translation") == "EntrezGene"
        assert filing.continued("EntrezGene", "#", "gene") is None

    def test_ensembl_names_filed_under_a_database_are_not_it(self):
        """Rhesus 70 to 72: Ensembl's transcript names filed under EntrezGene on transcripts are not EntrezGene's.

        From 73 they are under a name of their own; they leave few of EntrezGene's links (it keeps its genes'), so
        their own database does not continue EntrezGene. Fails if the share is taken of the database that receives the
        links (1.0 here).
        """
        genes = _genes(20)
        own = [("EntrezGene", str(100 + k), g, "gene") for k, g in enumerate(genes)]
        names = [(f"LOC{100 + k}-201", g) for k, g in enumerate(genes[:3])]
        table = {
            72: own + [("EntrezGene", n, g, "transcript") for n, g in names],
            73: own + [("EntrezGene_trans_name", n, g, "transcript") for n, g in names],
        }
        filing = database_filing(
            _links(table), [72, 73], _on("gene", "EntrezGene") | _on("transcript", "EntrezGene"), {}
        )
        assert not filing.renamed


class TestTheSameNameOnAnotherForm:
    """A selected name on another form can continue its own database only."""

    def test_symbols_on_transcripts_are_not_another_databases_names(self):
        """Human GRCh37 HGNC symbols on transcripts (to 75); from 76 Vega's gene names hold the same symbols on genes.

        HGNC's own symbols on genes held them all along, so they move to no HGNC database. Fails if a selected name on
        another form may continue any selected database (the symbols would be filed as Vega's names at 48 to 75).
        """
        genes = _genes(8)
        symbols = [f"SYM{k}" for k in range(8)]
        hgnc = [("HGNC Symbol", s, g, "gene") for s, g in zip(symbols, genes)]
        table = {
            75: hgnc + [("HGNC Symbol", s, g, "transcript") for s, g in zip(symbols, genes)],
            76: hgnc + [("Vega gene", s, g, "gene") for s, g in zip(symbols, genes)],
        }
        filing = database_filing(_links(table), [75, 76], _on("gene", "HGNC Symbol", "Vega gene"), {})
        assert not filing.renamed


class TestScheme:
    """Rule 3: an identifier of another selected database's scheme, whose link that database holds, is its."""

    def test_a_havana_id_given_as_a_symbol_is_a_havana_id(self):
        """R2-6: MGI gave OTTMUSG00000003802 as a mouse gene's symbol at 48 to 53; Havana holds it on the gene at 57.

        Fails if an identifier keeps the database it was filed under whatever its form and link (the version of the
        Havana id is then not read: the id counts as a symbol).
        """
        genes = _genes(5)
        havana = [(f"OTTMUSG{k:011d}", g) for k, g in enumerate(genes)]
        table = {
            48: [("MGI Symbol", "Xkr4", genes[0]), ("MGI Symbol", havana[1][0], genes[1])],
            57: [("Havana gene", i, g) for i, g in havana] + [("MGI Symbol", "Xkr4", genes[0])],
        }
        filing = database_filing(_links(table), [48, 57], _on("gene", "MGI Symbol", "Havana gene"), {})
        assert filing.schemes == {"Havana gene": "OTTMUSG#"}
        identifier, gene = (int(v[0]) for v in link_hashes([havana[1][0]], [genes[1]]))
        assert filing.scheme_of("MGI Symbol", "OTTMUSG#", identifier, gene, 48) == "Havana gene"
        other = (int(v[0]) for v in link_hashes([havana[1][0]], [genes[2]]))
        assert filing.scheme_of("MGI Symbol", "OTTMUSG#", *other, 48) is None  # not Havana's link: kept

    def test_not_between_the_names_of_one_database(self):
        """A label that repeats its database's accession (``RFAM label`` RF00017) stays a label of that database.

        Fails if rule 3 moves identifiers between the accessions and the labels of one Ensembl database.
        """
        genes = _genes(3)
        table = {
            90: [("RFAM", f"RF{k:05d}", g) for k, g in enumerate(genes)]
            + [("RFAM label", "RF00001", genes[1]), ("RFAM label", "5S_rRNA", genes[0])]
        }
        database_of = {"RFAM": {"RFAM"}, "RFAM label": {"RFAM"}}
        filing = database_filing(_links(table), [90], _on("gene", "RFAM", "RFAM label"), database_of)
        identifier, gene = (int(v[0]) for v in link_hashes(["RF00001"], [genes[1]]))
        assert filing.schemes.get("RFAM") == "RF#"
        assert filing.scheme_of("RFAM label", "RF#", identifier, gene, 90) is None


@pytest.mark.parametrize("bound", [continuity.CONTINUITY_SHARE])
def test_the_bound_lies_in_the_measured_gap(bound):
    """The renames measured move at least 0.62 of a database's links; databases beside a selected one at most 0.40."""
    assert 0.40 < bound <= 0.62


def test_rows_are_filed_under_the_database_they_are():
    """``DatabaseManager._filed``: a continuing database's rows by form, unfiled rows dropped but display names.

    RefSeq_dna's ``NM_`` row is RefSeq_mRNA's, its ``NG_`` row (no form that moved, not RefSeq_mRNA's scheme) is no row; a
    display name of a database that continues nothing is kept under its own name; Rfam on transcripts outside the run of
    releases that continued Rfam is no row; an MGI row with a Havana id that Havana holds is Havana's. Fails if the rows
    keep Ensembl's names, or if rows that continue nothing are kept (``RefSeq_dna`` would be a database of the graph).
    """
    from idtrack.build.database_manager import DatabaseManager

    gene = "ENSMUSG00000069588"
    identifier, gene_hash = (int(v[0]) for v in link_hashes(["OTTMUSG00000003802"], [gene]))
    filing = continuity.Filing(
        renamed={
            "RefSeq_dna": {"*": "RefSeq_mRNA", "NR_#": "RefSeq_ncRNA"},
            "RFAM@transcript": {"*": "RFAM"},
            "HGNC (curated)": {"*": "HGNC Symbol"},
        },
        runs={"RFAM@transcript": ((56, 62),)},
        schemes={"RefSeq_mRNA": "NM_#", "RFAM": "RF#", "Havana gene": "OTTMUSG#"},
        scheme_links={"RefSeq_mRNA": {}, "RFAM": {}, "Havana gene": {(identifier, gene_hash): (57, 75)}},
        databases={"MGI Symbol": frozenset({"MGI"}), "Havana gene": frozenset({"OTTG"})},
    )

    class External:
        @staticmethod
        def selected_units():
            return {("transcript", "RefSeq_mRNA"), ("gene", "RFAM"), ("gene", "MGI Symbol"), ("gene", "Havana gene")}

    def manager(form: str, release: int) -> DatabaseManager:
        made = DatabaseManager.__new__(DatabaseManager)
        made.form, made.release = form, release
        made.external_inst = External()  # type: ignore[assignment]
        if form != "gene":  # the gene of a transcript (the relation table of a release)
            made._genes_of_rows = lambda graph_ids: ["ENSMUSG00000000001" for _ in graph_ids]  # type: ignore[method-assign]
        return made

    def rows(*found: tuple[str, str, str, bool]) -> pd.DataFrame:
        return pd.DataFrame(
            [(release, g, i, n, False, d) for g, i, n, d in found for release in (0,)],
            columns=["release", "graph_id", "id_db", "name_db", "is_label", "is_display"],
        ).assign(ensembl_identity=0.0, xref_identity=0.0, version=0)

    transcripts = manager("transcript", 63)._filed(
        rows(("ENST1.1", "NM_000001", "RefSeq_dna", False), ("ENST1.1", "NG_000001", "RefSeq_dna", False)), filing
    )
    assert set(zip(transcripts["id_db"], transcripts["name_db"])) == {("NM_000001", "RefSeq_mRNA")}
    assert not len(manager("transcript", 70)._filed(rows(("ENST1.1", "RF00017-201", "RFAM", False)), filing))
    genes = manager("gene", 50)._filed(
        rows(
            (f"{gene}.1", "OTTMUSG00000003802", "MGI Symbol", False),
            (f"{gene}.1", "Smurf1", "HGNC (curated)", True),
            (f"{gene}.1", "Xkr4", "MGI Symbol", False),
        ),
        filing,
    )
    assert set(zip(genes["id_db"], genes["name_db"])) == {
        ("OTTMUSG00000003802", "Havana gene"),
        ("Smurf1", "HGNC Symbol"),
        ("Xkr4", "MGI Symbol"),
    }
