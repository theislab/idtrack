#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Allowlists of the externals YAML files shipped in ``idtrack/default_config``.

A shipped ``<organism>_externals_modified.yml`` is the organism's template (made by
:py:meth:`idtrack._database_manager.DatabaseManager.create_database_content` and
:py:meth:`idtrack._external_databases.ExternalDatabases.create_template_yaml` over every release and form) with
``Include: true`` set for the ``(form, database, assembly)`` entries of the organism's allowlist below, by
:py:func:`apply_allowlist`. Nothing else differs between template and modified file. The lists follow one
conceptual set for every organism, with the species' own symbol databases:

* **gene**: clone-based, Havana and Vega names, EntrezGene / NCBI gene, RFAM, UniProtKB gene names, the species'
  symbol source(s), and the ``synonym_id::`` forms of the identifier and symbol databases;
* **transcript**: CCDS, Havana transcripts and RefSeq mRNA / ncRNA (curated and predicted);
* **translation**: Havana translations, RefSeq peptides (curated and predicted) and UniProt (Swiss-Prot, TrEMBL).

Names that an organism's template does not contain are skipped. Tutorial 02
(``docs/_notebooks/02_prepare_new_external_yaml.ipynb``) shows the same lists as editable dictionaries.
"""

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Allowlist type: {form: {database name: [assembly keys]}}
Allowlist = dict[str, dict[str, list[int]]]


def _allowlist(assemblies: Sequence[int], **forms: Iterable[str]) -> Allowlist:
    """Build ``{form: {database: assemblies}}`` with the same assemblies for every database."""
    return {form: {db: list(assemblies) for db in databases} for form, databases in forms.items()}


DEFAULT_EXTERNALS: dict[str, Allowlist] = {
    # Human: HGNC symbols.
    "homo_sapiens": _allowlist(
        [36, 37, 38],
        gene=(
            "Clone_based_ensembl_gene",
            "Clone_based_vega_gene",
            "EntrezGene",
            "HGNC Symbol",
            "Havana gene",
            "NCBI gene",
            "NCBI gene (formerly Entrezgene)",
            "RFAM",
            "UniProtKB Gene Name",
            "Vega gene",
            "Vega_gene",
            "synonym_id::EntrezGene",
            "synonym_id::HGNC Symbol",
            "synonym_id::NCBI gene",
            "synonym_id::NCBI gene (formerly Entrezgene)",
            "synonym_id::UniProtKB Gene Name",
        ),
        transcript=(
            "CCDS",
            "Havana transcript",
            "RefSeq_mRNA",
            "RefSeq_mRNA_predicted",
            "RefSeq_ncRNA",
            "RefSeq_ncRNA_predicted",
        ),
        translation=(
            "Havana translation",
            "RefSeq_peptide",
            "RefSeq_peptide_predicted",
            "Uniprot/SPTREMBL",
            "Uniprot/SWISSPROT",
        ),
    ),
    # Mouse: MGI symbols.
    "mus_musculus": _allowlist(
        [37, 38, 39],
        gene=(
            "Clone_based_ensembl_gene",
            "Clone_based_vega_gene",
            "EntrezGene",
            "MGI Symbol",
            "Havana gene",
            "NCBI gene",
            "NCBI gene (formerly Entrezgene)",
            "RFAM",
            "UniProtKB Gene Name",
            "Vega gene",
            "Vega_gene",
            "synonym_id::EntrezGene",
            "synonym_id::MGI Symbol",
            "synonym_id::NCBI gene",
            "synonym_id::NCBI gene (formerly Entrezgene)",
            "synonym_id::UniProtKB Gene Name",
        ),
        transcript=(
            "CCDS",
            "Havana transcript",
            "RefSeq_mRNA",
            "RefSeq_mRNA_predicted",
            "RefSeq_ncRNA",
            "RefSeq_ncRNA_predicted",
        ),
        translation=(
            "Havana translation",
            "RefSeq_peptide",
            "RefSeq_peptide_predicted",
            "Uniprot/SPTREMBL",
            "Uniprot/SWISSPROT",
        ),
    ),
    # Pig: HGNC symbols on Sscrofa10.2, VGNC symbols on Sscrofa11.1.
    "sus_scrofa": _allowlist(
        [9, 102, 111],
        gene=(
            "Clone_based_ensembl_gene",
            "Clone_based_vega_gene",
            "EntrezGene",
            "HGNC Symbol",
            "VGNC Symbol",
            "Havana gene",
            "NCBI gene",
            "NCBI gene (formerly Entrezgene)",
            "RFAM",
            "UniProtKB Gene Name",
            "synonym_id::EntrezGene",
            "synonym_id::HGNC Symbol",
            "synonym_id::NCBI gene",
            "synonym_id::NCBI gene (formerly Entrezgene)",
            "synonym_id::UniProtKB Gene Name",
        ),
        transcript=(
            "Havana transcript",
            "RefSeq_mRNA",
            "RefSeq_mRNA_predicted",
            "RefSeq_ncRNA",
            "RefSeq_ncRNA_predicted",
        ),
        translation=(
            "RefSeq_peptide",
            "RefSeq_peptide_predicted",
            "Uniprot/SPTREMBL",
            "Uniprot/SWISSPROT",
        ),
    ),
    # Rhesus macaque: HGNC symbols on MMUL_1 and Mmul_8.0.1, VGNC symbols on Mmul_10. Names a release does not
    # have (e.g. CCDS, Vega) are skipped when the allowlist is applied.
    "macaca_mulatta": _allowlist(
        [1, 801, 10],
        gene=(
            "Clone_based_ensembl_gene",
            "Clone_based_vega_gene",
            "EntrezGene",
            "HGNC Symbol",
            "VGNC Symbol",
            "Havana gene",
            "NCBI gene",
            "NCBI gene (formerly Entrezgene)",
            "RFAM",
            "UniProtKB Gene Name",
            "Vega gene",
            "Vega_gene",
            "synonym_id::EntrezGene",
            "synonym_id::HGNC Symbol",
            "synonym_id::VGNC Symbol",
            "synonym_id::NCBI gene",
            "synonym_id::NCBI gene (formerly Entrezgene)",
            "synonym_id::UniProtKB Gene Name",
        ),
        transcript=(
            "CCDS",
            "Havana transcript",
            "RefSeq_mRNA",
            "RefSeq_mRNA_predicted",
            "RefSeq_ncRNA",
            "RefSeq_ncRNA_predicted",
        ),
        translation=(
            "Havana translation",
            "RefSeq_peptide",
            "RefSeq_peptide_predicted",
            "Uniprot/SPTREMBL",
            "Uniprot/SWISSPROT",
        ),
    ),
    # Cynomolgus macaque: HGNC symbols on 5.0, VGNC (and few HGNC) symbols on 6.0.
    "macaca_fascicularis": _allowlist(
        [5, 6],
        gene=(
            "Clone_based_ensembl_gene",
            "Clone_based_vega_gene",
            "EntrezGene",
            "HGNC Symbol",
            "VGNC Symbol",
            "Havana gene",
            "NCBI gene",
            "NCBI gene (formerly Entrezgene)",
            "RFAM",
            "UniProtKB Gene Name",
            "Vega gene",
            "Vega_gene",
            "synonym_id::EntrezGene",
            "synonym_id::HGNC Symbol",
            "synonym_id::VGNC Symbol",
            "synonym_id::NCBI gene",
            "synonym_id::NCBI gene (formerly Entrezgene)",
            "synonym_id::UniProtKB Gene Name",
        ),
        transcript=(
            "CCDS",
            "Havana transcript",
            "RefSeq_mRNA",
            "RefSeq_mRNA_predicted",
            "RefSeq_ncRNA",
            "RefSeq_ncRNA_predicted",
        ),
        translation=(
            "Havana translation",
            "RefSeq_peptide",
            "RefSeq_peptide_predicted",
            "Uniprot/SPTREMBL",
            "Uniprot/SWISSPROT",
        ),
    ),
}


def apply_allowlist(
    template: Mapping[str, Any], organism: str, allowlist: Mapping[str, Mapping[str, Sequence[int]]]
) -> tuple[dict[str, Any], int, list[tuple[str, str]]]:
    """Return a copy of an externals template with ``Include: true`` for the allowlisted entries.

    Args:
        template: Parsed ``<organism>_externals_template.yml`` (``{organism: {form: {database: {...}}}}``).
        organism: Organism whose section is edited.
        allowlist: ``{form: {database: [assembly keys]}}`` (e.g. ``DEFAULT_EXTERNALS[organism]``).

    Returns:
        tuple[dict, int, list[tuple[str, str]]]: The modified copy, the number of ``(form, database, assembly)``
        entries switched on, and the allowlisted ``(form, database)`` pairs that the template does not contain.

    Raises:
        KeyError: If the template has no section for *organism*.
    """
    modified = copy.deepcopy(dict(template))
    section = modified[organism]
    enabled = 0
    absent: list[tuple[str, str]] = []
    for form, databases in allowlist.items():
        for db_name, assemblies in databases.items():
            entry = (section.get(form) or {}).get(db_name)
            if entry is None:
                absent.append((form, db_name))
                continue
            for assembly in assemblies:
                item = entry["Assembly"].get(str(assembly))
                if item is not None:
                    item["Include"] = True
                    enabled += 1
    return modified, enabled, absent
