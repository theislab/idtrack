#!/usr/bin/env python3

import argparse
import copy
import os
import pickle

import anndata as ad

# Rely on properly installed package (no sys.path hacks)
import idtrack

# Configure cache path via environment variable or use a sensible default
IDTRACK_LOCAL_REPO = os.environ.get("IDTRACK_LOCAL_REPO", "./idtrack_cache")


def log(message: str) -> None:
    """Print a log message and flush output immediately (useful for SLURM logs)."""
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="IDTrack argument parser")

    parser.add_argument("--anndata-path", required=True, dest="anndata_path")
    parser.add_argument("--dataset-key", required=True, dest="dataset_key")
    parser.add_argument("--assembly-label", required=True, dest="assembly_label")
    parser.add_argument("--from-release", required=True, dest="from_release")
    parser.add_argument("--to-release-str-list", required=True, dest="to_release_str_list")
    parser.add_argument("--from-database", required=True, dest="from_database")
    parser.add_argument("--to-database", required=True, dest="to_database")
    parser.add_argument("--slurm-job-name", required=True, dest="slurm_job_name")
    parser.add_argument("--job-name", required=True, dest="job_name")
    parser.add_argument("--job-dir", required=True, dest="job_dir")

    return parser.parse_args()


def main():
    log("[INFO] Parsing command line arguments...")
    args = parse_args()
    log("[INFO] Arguments parsed successfully.")

    anndata_path = args.anndata_path
    dataset_key = args.dataset_key
    assembly_label = args.assembly_label
    from_release = args.from_release
    to_release_str_list = args.to_release_str_list
    from_database = args.from_database
    to_database = args.to_database
    slurm_job_name = args.slurm_job_name
    job_name = args.job_name
    job_dir = args.job_dir

    # Print all parameters
    log("\n===== Parsed Parameters =====")
    log(
        {
            "anndata_path": anndata_path,
            "dataset_key": dataset_key,
            "assembly_label": assembly_label,
            "from_release": from_release,
            "to_release_str_list": to_release_str_list,
            "from_database": from_database,
            "to_database": to_database,
            "slurm_job_name": slurm_job_name,
            "job_name": job_name,
            "job_dir": job_dir,
        }
    )
    log("===== End Parameters =====\n")

    # ------------------------------------------------------------------
    # STEP 1: Prepare IDTrack API
    # ------------------------------------------------------------------

    log("[STEP 1/5] Setting up IDTrack API...")
    log(f"[INFO] Using IDTrack local repository: {IDTRACK_LOCAL_REPO}")
    idt = idtrack.API(local_repository=IDTRACK_LOCAL_REPO)
    log("[INFO] Configuring IDTrack logger...")
    idt.configure_logger()
    log("[INFO] Resolving organism 'homo sapiens'...")
    organism_formal_name, _ = idt.resolve_organism("homo sapiens")
    log(f"[INFO] Resolved organism name: {organism_formal_name}")
    snapshot_release = 114  # if needed, can be made configurable later
    log(f"[INFO] Building graph for organism={organism_formal_name}, snapshot_release={snapshot_release}...")
    idt.build_graph(
        organism_name=organism_formal_name,
        snapshot_release=snapshot_release,
        calculate_caches=False,
    )
    log("[INFO] Calculating graph caches (this may take a while)...")
    idt.calculate_graph_caches()

    log("[STEP 1/5] IDTrack API setup completed.\n")

    # ------------------------------------------------------------------
    # STEP 2: Load AnnData
    # ------------------------------------------------------------------
    log("[STEP 2/5] Loading AnnData object...")
    if not os.path.exists(anndata_path):
        raise FileNotFoundError(f"AnnData file not found: {anndata_path}")
    log(f"[INFO] Reading AnnData from: {anndata_path}")
    adata = ad.read_h5ad(anndata_path)
    log(f"[INFO] AnnData loaded with shape: {adata.n_obs} cells x {adata.n_vars} features")

    if from_database not in adata.var.columns:
        raise KeyError(
            f"Column '{from_database}' not found in adata.var. " f"Available columns: {list(adata.var.columns)}"
        )

    log(f"[INFO] Extracting query symbols from adata.var['{from_database}']...")
    query_symbols = adata.var[from_database].tolist()
    log(f"[INFO] Extracted {len(query_symbols)} query symbols.\n")

    # ------------------------------------------------------------------
    # STEP 3: Prepare releases
    # ------------------------------------------------------------------
    log("[STEP 3/5] Preparing release list...")
    to_release_list = list(map(int, to_release_str_list.split("|")))[::-1]
    log(f"[INFO] From release: {from_release}")
    log(f"[INFO] Target releases (reversed order): {to_release_list}\n")

    # ------------------------------------------------------------------
    # STEP 4: Run conversions
    # ------------------------------------------------------------------
    log("[STEP 4/5] Running identifier conversions...")
    result_dict = {}

    total_releases = len(to_release_list)
    for idx, to_release in enumerate(to_release_list, start=1):
        log(f"\n========== Release {to_release} ({idx}/{total_releases}) ==========")
        log("[INFO] Running convert_identifier_multiple...")
        matched_symbols = idt.convert_identifier_multiple(
            copy.deepcopy(query_symbols),
            to_release=to_release,
            final_database=to_database,
        )

        log("[INFO] Classifying conversion results...")
        matched_symbols_classified = idt.classify_multiple_conversion(matched_symbols)

        log("[INFO] Printing binned conversion summary:")
        idt.print_binned_conversion(matched_symbols_classified)

        result_dict[to_release] = matched_symbols
        log(f"[INFO] Finished processing release {to_release}.")

    log("\n[STEP 4/5] Identifier conversions completed.\n")

    # ------------------------------------------------------------------
    # STEP 5: Save results
    # ------------------------------------------------------------------
    log("[STEP 5/5] Saving results to pickle file...")
    os.makedirs(job_dir, exist_ok=True)
    result_pickle_path = os.path.join(job_dir, f"{job_name}.pickle")

    with open(result_pickle_path, "wb") as fh:
        pickle.dump(result_dict, fh)

    log(f"[INFO] Results successfully written to: {result_pickle_path}")
    log("\nAll done. Exiting cleanly.")


if __name__ == "__main__":
    main()
