#!/usr/bin/env python3
"""External ID mapping backends for idtrack.

This module provides interfaces to external ID mapping services:
- g:Profiler (gprofiler-official)
- MyGene.info (mygene)
- Ensembl BioMart (pybiomart)
- gget (Ensembl REST API)

Additionally, ortholog utilities are available (require gget + biopython).

**Note:** This module requires optional dependencies that are not installed
with the core idtrack package. Install them with:

    pip install gget mygene pybiomart gprofiler-official biopython

Or install only the backends you need.
"""

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

from idtrack._external_mappers._convert import convert_ids
from idtrack._external_mappers._utils import check_optional_dependencies

__all__ = [
    "convert_ids",
    "check_optional_dependencies",
    # Ortholog utilities (require optional dependencies: gget, biopython)
    "get_ortholog_table",
    "get_ortholog_ids_for_species",
    "pick_ortholog_for_species",
    "align_ortholog_pair_with_features",
    "fetch_aa_sequence",
    "run_muscle_pairwise",
    "compute_alignment_scores",
    "AlignmentScores",
]

# Check for missing dependencies at import time and warn the user
# This helps users understand what's available without hitting runtime errors
_dependency_status = check_optional_dependencies(warn=True)


def __getattr__(name: str):
    """Lazy import for ortholog utilities to avoid loading heavy dependencies."""
    ortholog_exports = {
        "get_ortholog_table",
        "get_ortholog_ids_for_species",
        "pick_ortholog_for_species",
        "align_ortholog_pair_with_features",
        "fetch_aa_sequence",
        "run_muscle_pairwise",
        "compute_alignment_scores",
        "AlignmentScores",
    }
    if name in ortholog_exports:
        from idtrack._external_mappers import _ortholog

        return getattr(_ortholog, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
