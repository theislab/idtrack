# Reproducibility Experiments (Manuscript-Facing)

This folder contains experiment notebooks and small helper modules used to generate figures/tables for the IDTrack manuscript.

## Notebook Index

-   `experiment_hlca/00_hlca_manuscript_table1_and_figures.ipynb`
    -   Generates Table 1 plus additional publication-ready tables/figures (HGNC overlap, feature-space summaries).
    -   Uses the curated HLCA study→files mapping aligned with `idtrack/docs/_notebooks/05_tutorial_harmonization.ipynb`.
-   `experiment_hlca/01_hlca_extended_experiments.ipynb`
    -   Extended HLCA analysis notebook (large-scale harmonization + QA; tutorial-style).
-   `experiment_tool_comparison/00_tool_comparison_capability_matrix_fig4d.ipynb`
    -   Capability-focused tool comparison (Figure 4d) using `idtrack/idtrack/_external_mappers`.
-   `experiment_tool_comparison/01_external_mapper_variability_deep_dive.ipynb`
    -   Marketing appendix: empirical variability + agreement heatmaps across external mapper backends (HGNC + UniProt targets).
-   `experiment_tool_comparison/02_pybiomart_release_sensitivity.ipynb`
    -   Marketing appendix: demonstrates release-sensitivity pitfalls in point-in-time Ensembl mappers (pybiomart), and how cache/snapshot discipline matters.
-   `experiment_random_stress_tests/00_random_stress_tests_fig4ab.ipynb`
    -   Randomized stress tests used for Figure 4a/b (cache-first; writes to `idtrack/docs/_notebooks/idtrack_cache/experiments/random_data/`).
-   `experiment_time_travel_matrix/00_build_time_travel_matrix_cache.ipynb`
    -   Builds a cache for a square `from_release × to_release` time-travel matrix (human; Ensembl backbone + HGNC + UniProt targets).
-   `experiment_time_travel_matrix/01_analyze_time_travel_matrix.ipynb`
    -   Consumes the cached grid and exports manuscript-ready multi-panel heatmaps + delta-curves (`fig_time_travel_matrix_human.pdf`, `fig_time_travel_delta_curves.pdf`).
-   `experiment_time_travel_matrix/02_roundtrip_consistency.ipynb`
    -   Marketing appendix: A→B→A round-trip recovery heatmap + distance curve (backbone consistency diagnostic).
-   `experiment_time_travel_vs_external_mappers/00_build_time_travel_vs_external_mappers_cache.ipynb`
    -   Builds cache bundles that quantify external tool failure modes on historical Ensembl IDs (naive mapping) and what changes after IDTrack time travel into a fixed `to_release` boundary.
-   `experiment_time_travel_vs_external_mappers/01_analyze_time_travel_vs_external_mappers.ipynb`
    -   Manuscript-ready multi-panel figures + tables: coverage loss vs `from_release`, recovery after time travel, and agreement (Jaccard) vs IDTrack for HGNC + UniProt targets.
-   `experiment_time_travel_vs_external_mappers/02_case_studies_external_mapper_failures.ipynb`
    -   Marketing appendix: exports a concrete, auditable case-study table (external `1:0` vs IDTrack success) and (optionally) small-N `explain=True` audit summaries.
-   `experiment_cellranger_idtrack/create_data.ipynb`
    -   Gold-standard pipeline scaffold (do not edit code cells up to `## Running IDTrack`).
-   `experiment_cellranger_idtrack/analysis_gold_standard_fig4c.ipynb`
    -   Figure 4c analysis/figure companion for the gold-standard experiment (+ optional extended figure + per-release LaTeX table exports).
-   `experiment_cellranger_idtrack/analysis_gold_standard_expression_consistency.ipynb`
    -   Marketing appendix: pseudo-bulk expression consistency after 1→1 mapping into a target release (+ optional suite figure + distance-summary table).
-   `experiment_other_organisms/00_other_organisms_showcase.ipynb`
    -   Optional non-human showcase (mouse + pig). Cache-first.
-   `experiment_other_organisms/01_other_organisms_deep_dive_time_axis.ipynb`
    -   Marketing appendix: mouse + pig time-axis sweep + multi-panel figure.
-   `experiment_identifier_drift/00_identifier_drift_case_studies.ipynb`
    -   Mines explainable “drift” case studies (audit paths) and exports a manuscript-ready multi-panel figure.
-   `experiment_external_bridges/00_external_bridges_tradeoff.ipynb`
    -   Quantifies the coverage/ambiguity/runtime trade-off between curated externals vs “all externals”.
-   `experiment_marketing_overview/00_marketing_overview_dashboard.ipynb`
    -   Marketing dashboard: builds a single multi-panel overview figure by *reading cached artifacts* (no graph build).
-   `example_manual_running.ipynb`
    -   Manual end-to-end demo notebook (legacy outputs) with an additional export cell for an alternative HLCA LaTeX/CSV table.

## Environment Variables

-   `IDTRACK_LOCAL_REPO`
    -   Where IDTrack graph snapshots / caches live.
    -   Recommended for this repo: `idtrack/docs/_notebooks/idtrack_cache`.
-   `HLCA_BASE_PATH`
    -   HLCA data root used by `experiment_hlca/00_hlca_manuscript_table1_and_figures.ipynb`.
-   `GOLD_STANDARD_ANNDATA_DIR`
    -   Directory containing `.h5ad` files produced by the Cell Ranger pipeline (see `anndata_generator.py`).

## Outputs

-   Experiment caches (heavy / time-consuming): `IDTRACK_LOCAL_REPO/experiments/` (default: `idtrack/docs/_notebooks/idtrack_cache/experiments/`)
-   Publication-ready exports (single location across notebooks):
    -   Figures: `_outputs/_publication/figures/`
    -   Tables: `_outputs/_publication/tables/`
-   Experiment-local copies (per notebook/experiment):
    -   `_outputs/<experiment>/figures/`
    -   `_outputs/<experiment>/tables/`

## Batch Execution (Slurm)

For friction-free, parallel execution on Slurm (with dependency staging and per-notebook logs):

-   Submit both stages (recommended): `reproducibility/scripts/submit_experiment_notebooks_slurm.sh` (umbrella checkout: prefix with `idtrack/`)
-   Notebook manifests:
    -   Stage 0: `notebooks_manifest_stage0.txt`
    -   Stage 1: `notebooks_manifest_stage1.txt`
    -   Stage 2: `notebooks_manifest_stage2.txt` (dashboards / final summaries)
    -   Manifest entries are paths relative to the `.../reproducibility/experiments/` folder (layout-agnostic).
-   Logs:
    -   Slurm stdout/stderr: `_logs/`
    -   Notebook execution logs: `_logs/nbconvert/`
    -   If Slurm logs are empty, check job state/exit codes with `sacct` (jobs may have failed before running the script).

Common overrides:

    -   `CONDA_ENV=idtrack_dev_env ./reproducibility/scripts/submit_experiment_notebooks_slurm.sh`
    -   `IDTRACK_LOCAL_REPO=/path/to/idtrack_cache ./reproducibility/scripts/submit_experiment_notebooks_slurm.sh`

## Notes on Memory / Runtime

Some steps (graph loading, conversions) can be memory-intensive. The notebooks are **cache-first**: if a required cache is missing, the notebook computes it and writes it under `idtrack/docs/_notebooks/idtrack_cache/experiments/`.
