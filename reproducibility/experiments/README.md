# Reproducibility Experiments (Manuscript-Facing)

This folder contains experiment notebooks and small helper modules used to generate figures/tables for the IDTrack manuscript.

## Notebook Index

-   `hlca_manuscript_figures.ipynb`
    -   Generates Table 1 (`idtrack-manuscript/tables/hlca_harmonization.tex`) and optional HLCA figures.
    -   Uses the curated HLCA study→files mapping aligned with `idtrack/docs/_notebooks/05_tutorial_harmonization.ipynb`.
-   `comparison.ipynb`
    -   Capability-focused tool comparison (Figure 4d) using `idtrack/idtrack/_external_mappers`.
-   `comparison_external_mappers_deep_dive.ipynb`
    -   Marketing appendix: empirical variability + agreement heatmaps across external mapper backends (HGNC + UniProt targets).
-   `random_data.ipynb`
    -   Randomized stress tests used for Figure 4a/b (cache-first; writes to `idtrack/docs/_notebooks/idtrack_cache/experiments/random_data/`).
-   `experiment_cellranger_idtrack/create_data.ipynb`
    -   Gold-standard pipeline scaffold (do not edit code cells up to `## Running IDTrack`).
-   `experiment_cellranger_idtrack/analysis_gold_standard_fig4c.ipynb`
    -   Figure 4c analysis/figure companion for the gold-standard experiment.
-   `experiment_cellranger_idtrack/analysis_gold_standard_expression_consistency.ipynb`
    -   Marketing appendix: pseudo-bulk expression consistency after 1→1 mapping into a target release.
-   `other_organisms_showcase.ipynb`
    -   Optional non-human showcase (mouse + pig). Cache-first.
-   `other_organisms_deep_dive.ipynb`
    -   Marketing appendix: mouse + pig time-axis sweep + multi-panel figure.
-   `identifier_drift_case_studies.ipynb`
    -   Mines explainable “drift” case studies (audit paths) and exports a manuscript-ready multi-panel figure.
-   `external_bridges_tradeoff.ipynb`
    -   Quantifies the coverage/ambiguity/runtime trade-off between curated externals vs “all externals”.
-   `example_manual_running.ipynb`
    -   Manual end-to-end demo notebook (legacy outputs) with an additional export cell for an alternative HLCA LaTeX/CSV table.

## Environment Variables

-   `IDTRACK_LOCAL_REPO`
    -   Where IDTrack graph snapshots / caches live.
    -   Recommended for this repo: `idtrack/docs/_notebooks/idtrack_cache`.
-   `HLCA_BASE_PATH`
    -   HLCA data root used by `hlca_manuscript_figures.ipynb`.
-   `GOLD_STANDARD_ANNDATA_DIR`
    -   Directory containing `.h5ad` files produced by the Cell Ranger pipeline (see `anndata_generator.py`).

## Outputs

-   Experiment caches (heavy / time-consuming): `idtrack/docs/_notebooks/idtrack_cache/experiments/`
-   Manuscript artefacts:
    -   Figures: `idtrack-manuscript/figures/`
    -   Tables: `idtrack-manuscript/tables/`

## Notes on Memory / Runtime

Some steps (graph loading, conversions) can be memory-intensive. The notebooks are **cache-first**: if a required cache is missing, the notebook computes it and writes it under `idtrack/docs/_notebooks/idtrack_cache/experiments/`.
