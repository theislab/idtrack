#!/usr/bin/env python3
"""Constants and configuration for the _external_mappers module.

This module provides:
- Database alias mappings to canonical database names
- Species aliases and mappings for various backends
- Backend-specific configuration for MyGene, pybiomart, g:Profiler, and gget
- Ensembl archive host mappings by release number
"""

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

# ----------------------------- Canonical DBs -------------------------------- #

_DB_ALIASES: dict[str, str] = {
    # Ensembl gene/transcript/protein
    "ensembl_gene": "ensembl_gene",
    "ensembl.gene": "ensembl_gene",
    "ensemblgene": "ensembl_gene",
    "ensg": "ensembl_gene",
    "ensg_id": "ensembl_gene",
    "ensembl_gene_id": "ensembl_gene",
    "ensembl_transcript": "ensembl_transcript",
    "ensembl.transcript": "ensembl_transcript",
    "enst": "ensembl_transcript",
    "enst_id": "ensembl_transcript",
    "ensembl_transcript_id": "ensembl_transcript",
    "ensembl_protein": "ensembl_protein",
    "ensembl.peptide": "ensembl_protein",
    "ensp": "ensembl_protein",
    "ensp_id": "ensembl_protein",
    "ensembl_protein_id": "ensembl_protein",
    "ensembl_peptide_id": "ensembl_protein",
    # Symbols / HGNC (we treat this as "gene symbol" in general)
    "hgnc_symbol": "hgnc_symbol",
    "hgnc": "hgnc_symbol",
    # IDTrack-native external DB label (common in notebooks / configs)
    "hgnc symbol": "hgnc_symbol",
    "symbol": "hgnc_symbol",
    "gene_symbol": "hgnc_symbol",
    "genename": "hgnc_symbol",
    "name": "hgnc_symbol",
    "external_gene_name": "hgnc_symbol",
    "external_gene_id": "hgnc_symbol",
    "gene_name": "hgnc_symbol",
    # HGNC numeric ID (e.g., HGNC:5)
    "hgnc_id": "hgnc_id",
    "hgnc_numeric": "hgnc_id",
    # Entrez Gene
    "entrez_gene": "entrez_gene",
    "entrez": "entrez_gene",
    "entrezgene": "entrez_gene",
    "ncbi_gene": "entrez_gene",
    "geneid": "entrez_gene",
    "entrez_id": "entrez_gene",
    # UniProt
    "uniprot": "uniprot",
    "uniprot_acc": "uniprot",
    "uniprotkb": "uniprot",
    "uniprotkb_acc": "uniprot",
    # Common UniProt labels used in IDTrack graphs
    "uniprotkb/swiss-prot": "uniprot",
    "uniprotkb/swissprot": "uniprot",
    "uniprotkb_swiss-prot": "uniprot",
    "uniprotkb_swissprot": "uniprot",
    "swissprot": "uniprot",
    # RefSeq
    "refseq_mrna": "refseq_mrna",
    "refseq_rna": "refseq_mrna",
    "refseq_transcript": "refseq_mrna",
    "nm": "refseq_mrna",
    "refseq_protein": "refseq_protein",
    "np": "refseq_protein",
    # Misc popular namespaces
    "wormbase": "wormbase",
    "wb": "wormbase",
    "flybase": "flybase",
    "fb": "flybase",
}

SUPPORTED_DBS: set[str] = set(sorted(set(_DB_ALIASES.values())))

# -------------------------- Species normalization --------------------------- #

_SPECIES_ALIASES = {
    # Human
    "human": "hsapiens",
    "homo_sapiens": "hsapiens",
    "homo sapiens": "hsapiens",
    "hsapiens": "hsapiens",
    # Mouse
    "mouse": "mmusculus",
    "mus_musculus": "mmusculus",
    "mus musculus": "mmusculus",
    "mmusculus": "mmusculus",
    # Pig
    "pig": "sscrofa",
    "sus_scrofa": "sscrofa",
    "sus scrofa": "sscrofa",
    "sscrofa": "sscrofa",
}

# Map canonical short codes -> how they appear in Bgee's 'genus' / 'species'
_SPECIES_CANONICAL_TO_BGEENAMES: dict[str, tuple[str, str]] = {
    "hsapiens": ("Homo", "sapiens"),
    "mmusculus": ("Mus", "musculus"),
    "sscrofa": ("Sus", "scrofa"),
    # Extend this as you add more species
}

# --------------------------- Capability registry ---------------------------- #

SUPPORTED_METHODS = ("pybiomart", "mygene", "gprofiler", "gget")

_ENSEMBL_INPUT_DB: set[str] = {
    "ensembl_gene",
    "ensembl_transcript",
    "ensembl_protein",
}

# ---------------------------- Backend: MyGene ------------------------------- #

_MG_SCOPES = {
    "ensembl_gene": "ensembl.gene",
    "ensembl_transcript": "ensembl.transcript",
    "ensembl_protein": "ensembl.protein",
    "hgnc_symbol": "symbol",
    "hgnc_id": "hgnc",
    "entrez_gene": "entrezgene",
    "uniprot": "uniprot",
    "refseq_mrna": "refseq",
    "refseq_protein": "refseq",
    "wormbase": "wormbase",
    "flybase": "flybase",
}

_MG_FIELDS_SUPERSET = [
    "symbol",
    "name",
    "entrezgene",
    "hgnc",
    "HGNC",
    "ensembl.gene",
    "ensembl.transcript",
    "ensembl.protein",
    "uniprot",
    "refseq.rna",
    "refseq.protein",
    "taxid",
    "alias",
    "other_names",
    "mapLocation",
]

# --------------------------- Backend: pybiomart ----------------------------- #

_BM_ATTR_CANDIDATES: dict[str, list[str]] = {
    "ensembl_gene": ["ensembl_gene_id"],
    "ensembl_transcript": ["ensembl_transcript_id"],
    "ensembl_protein": ["ensembl_peptide_id", "ensembl_protein_id"],
    # "Gene symbol" across species. For human, 'hgnc_symbol' exists and is
    # preferred; for other organisms we fall back to 'external_gene_name'.
    "hgnc_symbol": ["hgnc_symbol", "external_gene_name", "external_gene_id", "gene_name"],
    "hgnc_id": ["hgnc_id"],
    # Entrez
    "entrez_gene": ["entrezgene_id", "entrezgene", "entrez_gene_id"],
    # UniProt accessions
    "uniprot": ["uniprotswissprot", "uniprot_swissprot"],
    # RefSeq
    "refseq_mrna": ["refseq_mrna"],
    "refseq_protein": ["refseq_peptide"],
}

_BM_FILTER_CANDIDATES: dict[str, list[str]] = {
    "ensembl_gene": ["ensembl_gene_id"],
    "ensembl_transcript": ["ensembl_transcript_id"],
    "ensembl_protein": ["ensembl_peptide_id"],
    "hgnc_symbol": ["hgnc_symbol", "external_gene_name"],
    "hgnc_id": ["hgnc_id"],
    "entrez_gene": ["entrezgene_id", "entrez_gene_id"],
    "uniprot": ["uniprotswissprot", "uniprot_swissprot"],
    "refseq_mrna": ["refseq_mrna"],
    "refseq_protein": ["refseq_peptide"],
}

# Known Ensembl archive hosts keyed by release number.
_ENSEMBL_ARCHIVE_BY_RELEASE: dict[int, str] = {
    54: "may2009.archive.ensembl.org",
    67: "may2012.archive.ensembl.org",
    74: "dec2013.archive.ensembl.org",
    75: "feb2014.archive.ensembl.org",
    76: "aug2014.archive.ensembl.org",
    77: "oct2014.archive.ensembl.org",
    78: "dec2014.archive.ensembl.org",
    79: "mar2015.archive.ensembl.org",
    80: "may2015.archive.ensembl.org",
    81: "jul2015.archive.ensembl.org",
    82: "sep2015.archive.ensembl.org",
    83: "dec2015.archive.ensembl.org",
    84: "mar2016.archive.ensembl.org",
    85: "jul2016.archive.ensembl.org",
    86: "oct2016.archive.ensembl.org",
    87: "dec2016.archive.ensembl.org",
    88: "mar2017.archive.ensembl.org",
    89: "may2017.archive.ensembl.org",
    90: "aug2017.archive.ensembl.org",
    91: "dec2017.archive.ensembl.org",
    92: "apr2018.archive.ensembl.org",
    93: "jul2018.archive.ensembl.org",
    94: "oct2018.archive.ensembl.org",
    95: "jan2019.archive.ensembl.org",
    96: "apr2019.archive.ensembl.org",
    97: "jul2019.archive.ensembl.org",
    98: "sep2019.archive.ensembl.org",
    99: "jan2020.archive.ensembl.org",
    100: "apr2020.archive.ensembl.org",
    101: "aug2020.archive.ensembl.org",
    102: "nov2020.archive.ensembl.org",
    103: "feb2021.archive.ensembl.org",
    104: "may2021.archive.ensembl.org",
    105: "dec2021.archive.ensembl.org",
    106: "apr2022.archive.ensembl.org",
    107: "jul2022.archive.ensembl.org",
    108: "oct2022.archive.ensembl.org",
    109: "feb2023.archive.ensembl.org",
    110: "jul2023.archive.ensembl.org",
    111: "jan2024.archive.ensembl.org",
    112: "may2024.archive.ensembl.org",
    113: "oct2024.archive.ensembl.org",
    114: "may2025.archive.ensembl.org",
}

_ENSEMBL_SPECIAL_RELEASE_HOSTS: dict[str, str] = {
    # Ensembl GRCh37 archive
    "grch37": "grch37.ensembl.org",
}

# --------------------------- Backend: g:Profiler ---------------------------- #

_GP_NS = {
    "ensembl_gene": "ENSG",
    "ensembl_transcript": "ENST",
    "ensembl_protein": "ENSP",
    # HGNC symbol vs accession
    "hgnc_symbol": "HGNC",  # e.g. TP53
    "hgnc_id": "HGNC_ACC",  # e.g. HGNC:11998
    # Entrez numeric ID
    "entrez_gene": "ENTREZGENE_ACC",
    # UniProtKB/Swiss-Prot accessions
    "uniprot": "UNIPROTSWISSPROT_ACC",
    # RefSeq
    "refseq_mrna": "REFSEQ_MRNA",
    "refseq_protein": "REFSEQ_PEPTIDE",
    "wormbase": "WORMBASE",
    "flybase": "FLYBASE",
}

_GP_INPUT_NAMESPACES: dict[str, set[str]] = {
    "ensembl_gene": {"ENSG"},
    "ensembl_transcript": {"ENST"},
    "ensembl_protein": {"ENSP"},
    "hgnc_symbol": {"HGNC"},
    "hgnc_id": {"HGNC_ACC"},
    "entrez_gene": {"ENTREZGENE_ACC", "ENTREZGENE"},
    "uniprot": {
        "UNIPROTSPTREMBL_ACC",
        "UNIPROTSWISSPROT_ACC",
        "UNIPROTSPTREMBL",
        "UNIPROTSWISSPROT",
        "UNIPROT",
        "UNIPROT_ACC",
        "UNIPROTKB",
        "UNIPROTKB_ACC",
    },
    "refseq_mrna": {"REFSEQ_MRNA"},
    "refseq_protein": {"REFSEQ_PEPTIDE"},
    "wormbase": {"WORMBASE"},
    "flybase": {"FLYBASE"},
}
