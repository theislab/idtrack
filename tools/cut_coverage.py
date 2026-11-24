#!/usr/bin/env python3
"""Check that the gene list of the cut test tables carries every gene-level database that only some genes carry.

Usage (from the root of the repository, with the package importable)::

    IDTRACK_ENSEMBL_ACCESS=https python tools/cut_coverage.py --organism sus_scrofa \
        --repository <local repository> [--snapshot 115]

Three databases of the shipped selection (``idtrack/default_config/<organism>_externals_modified.yml``) are on a few
genes only: ``RFAM label`` (the genes whose display name is an Rfam entry) and ``miRBase`` with ``miRBase label`` (the
miRNA genes with a miRBase entry). The test build of the cut tables (``tools/cut_tables.py``) stops on a database
that the selection enables at an assembly and release and that no gene of ``tests/data/real/<organism>.genes.txt``
carries there. For every such assembly and release, this tool reads which genes of the complete tables carry the
database, prints the gaps the gene list leaves (assembly, release, database, number of genes carrying it), and a few
genes that close them (each with the number of gaps it closes, chosen greedily). Run it before ``cut_tables.py``
when the selection or the gene list changes; add the genes it names to the gene list.

Reads: the tables of ``--repository`` (a local repository of idtrack; a table it lacks is downloaded into it), the
shipped selection and the gene list. Writes nothing but the tables it downloads. The exit status is 1 when a gap is
left that no gene closes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from idtrack.build.database_manager import DatabaseManager  # noqa: E402

#: The gene-level databases of the selection that only some genes carry.
PARTIAL_DATABASES = ("RFAM label", "miRBase", "miRBase label")

Gap = tuple[int, int, str]


def wanted(organism: str, snapshot: int) -> dict[tuple[int, int], set[str]]:
    """Return ``{(assembly, release): databases}``: where the shipped selection enables a partial database on genes."""
    path = ROOT / "idtrack" / "default_config" / f"{organism}_externals_modified.yml"
    with open(path, encoding="utf-8") as handle:
        genes = (yaml.safe_load(handle)[organism].get("gene")) or {}
    found: dict[tuple[int, int], set[str]] = defaultdict(set)
    for name in PARTIAL_DATABASES:
        entry = genes.get(name)
        for assembly, item in (entry or {}).get("Assembly", {}).items():
            if item["Include"]:
                for release in str(item["Ensembl release"]).split(","):
                    if int(release) <= snapshot:
                        found[(int(assembly), int(release))].add(name)
    return found


def carriers_at(organism: str, repository: str, assembly: int, release: int) -> dict[str, set[str]]:
    """Return ``{database: gene ids}``: the genes of one assembly and release that carry each partial database."""
    manager = DatabaseManager(
        organism=organism,
        form="gene",
        local_repository=repository,
        release=release,
        first_release=release,
        snapshot_release=release,
        assembly=assembly,
    )
    genes = manager.get_table("gene", usecols=["gene_id", "biotype", "display_xref_id"])
    genes = genes.merge(manager.get_db("idsraw_gene")[["gene_id", "gene_stable_id"]], on="gene_id")
    databases = manager.get_table("external_db", usecols=["external_db_id", "db_name"])
    xref = manager.get_table("xref", usecols=["xref_id", "external_db_id"]).merge(databases, on="external_db_id")
    links = manager.get_table("object_xref", usecols=["ensembl_id", "ensembl_object_type", "xref_id"])
    links = links[links["ensembl_object_type"].str.lower() == "gene"]
    rfam = set(xref.loc[xref["db_name"] == "RFAM", "xref_id"].astype(int))
    display = pd.to_numeric(genes["display_xref_id"], errors="coerce")
    mirbase = set(xref.loc[xref["db_name"] == "miRBase", "xref_id"].astype(int))
    with_mirbase = set(links.loc[links["xref_id"].astype(int).isin(mirbase), "ensembl_id"].astype(int))
    mirna = genes[genes["gene_id"].astype(int).isin(with_mirbase) & (genes["biotype"] == "miRNA")]
    return {
        "RFAM label": set(genes.loc[display.isin(rfam), "gene_stable_id"]),
        "miRBase": set(mirna["gene_stable_id"]),
        "miRBase label": set(mirna["gene_stable_id"]),
    }


def coverage(
    organism: str, repository: str, snapshot: int, genes: set[str]
) -> tuple[list[tuple[int, int, str, int]], list[tuple[str, int]], set[Gap]]:
    """Return the gaps a gene list leaves, the genes that close them, and the gaps no gene closes.

    Args:
        organism: The organism.
        repository: The local repository of the complete tables.
        snapshot: The snapshot release of the cut tables.
        genes: The gene ids (without version) of the gene list.

    Returns:
        tuple: The gaps ``(assembly, release, database, genes carrying it)``, the genes to add ``(gene, gaps it
        closes)`` in the order chosen, and the gaps left that no gene of the complete tables closes.
    """
    carriers: dict[str, set[Gap]] = defaultdict(set)
    gaps = []
    for (assembly, release), names in sorted(wanted(organism, snapshot).items()):
        have = carriers_at(organism, repository, assembly, release)
        for name in sorted(names):
            for gene in have[name]:
                carriers[gene].add((assembly, release, name))
            if not have[name] & genes:
                gaps.append((assembly, release, name, len(have[name])))
    need = {(a, r, n) for a, r, n, _ in gaps}
    chosen: list[tuple[str, int]] = []
    while need and carriers:
        best = max(sorted(carriers), key=lambda gene: len(carriers[gene] & need))
        covered = carriers[best] & need
        if not covered:
            break
        chosen.append((best, len(covered)))
        need -= covered
    return gaps, chosen, need


def read_gene_list(organism: str) -> set[str]:
    """Return the gene ids (without version) of ``tests/data/real/<organism>.genes.txt``."""
    path = ROOT / "tests" / "data" / "real" / f"{organism}.genes.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip().split(".")[0] for line in lines if line.strip() and not line.startswith("#")}


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--organism", required=True, help="Ensembl's species name, e.g. sus_scrofa")
    parser.add_argument("--repository", required=True, help="the local repository of the complete tables")
    parser.add_argument("--snapshot", type=int, default=115, help="the snapshot release of the cut tables")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
    if not os.path.isdir(arguments.repository):
        parser.error(f"the repository {arguments.repository!r} is not a folder")
    genes = read_gene_list(arguments.organism)
    gaps, chosen, left = coverage(arguments.organism, arguments.repository, arguments.snapshot, genes)
    print(f"gaps (assembly, release, database, genes carrying it): {gaps}")
    print(f"genes to add (gene, gaps it closes): {chosen}")
    if left:
        print(f"gaps no gene closes: {sorted(left)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
