#!/usr/bin/env python3
"""
anndata_generator.py

Create AnnData (.h5ad) files from Cell Ranger outputs.

Additions:
- keep_obs_if_it_is_filtered_by_cellranger (default True): when True, only the
  filtered barcodes are loaded and written to adata.X. When False, original
  behavior (adata.X=raw, layers['filtered']=aligned filtered) is used.

Other behavior unchanged from previous version.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

# ---- Suppress/resolve warnings requested ----
# We also remove dtype= usage in AnnData(...) to resolve that FutureWarning at the source.
warnings.filterwarnings("ignore", message=".*dtype argument is deprecated.*", category=FutureWarning)
try:
    from anndata._core.aligned_df import ImplicitModificationWarning

    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
except Exception:
    warnings.filterwarnings("ignore", message="Transforming to str index.", category=UserWarning)


# --------------------------- small utils ---------------------------


def _log(silent: bool, msg: str) -> None:
    if not silent:
        print(msg)


def _is_finished_outs(outs: Path) -> tuple[bool, float]:
    sentinels = [
        outs / "metrics_summary.csv",
        outs / "web_summary.html",
        outs / "molecule_info.h5",
        outs / "raw_feature_bc_matrix.h5",
        outs / "raw_feature_bc_matrix" / "matrix.mtx.gz",
    ]
    latest = -1.0
    found_any = False
    for p in sentinels:
        try:
            if p.exists() and (p.is_file() and p.stat().st_size > 0 or p.is_dir()):
                found_any = True
                latest = max(latest, p.stat().st_mtime)
        except Exception:
            pass
    return found_any, latest


def _find_best_outs(tag_dir: Path) -> tuple[Path, str] | None:
    candidates: list[tuple[Path, str, float]] = []
    for run in tag_dir.iterdir():
        if not run.is_dir():
            continue
        outs = run / "outs"
        if outs.is_dir():
            ok, mtime = _is_finished_outs(outs)
            if ok:
                candidates.append((outs, run.name, mtime))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[2], reverse=True)
    outs_dir, run_id, _ = candidates[0]
    return outs_dir, run_id


def _decode_ascii(a: Iterable[bytes]) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in a]


@dataclass
class TenxMatrix:
    X: sparse.csr_matrix
    barcodes: list[str]
    gene_id: list[str]
    gene_symbol: list[str]
    feature_type: list[str] | None = None
    genome: list[str] | None = None


def _read_10x_h5_full(h5_path: Path) -> TenxMatrix:
    with h5py.File(str(h5_path), "r") as f:
        g = f["matrix"]
        data = g["data"][()]
        indices = g["indices"][()]
        indptr = g["indptr"][()]
        shape = tuple(int(x) for x in g["shape"][()])

        barcodes = _decode_ascii(g["barcodes"][()])
        fg = g["features"] if "features" in g else g["gene"]

        gene_id = _decode_ascii(fg["id"][()])
        gene_symbol = _decode_ascii(fg["name"][()])

        feature_type = _decode_ascii(fg["feature_type"][()]) if "feature_type" in fg else None
        genome = None
        for key in ("genome", "genomes", "genome_ids"):
            if key in fg:
                genome = _decode_ascii(fg[key][()])
                break

        n_f = len(gene_id)
        n_b = len(barcodes)
        if shape == (n_f, n_b):
            M = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
        elif shape == (n_b, n_f):
            M = sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
        else:
            if shape[0] == n_f:
                M = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
            else:
                M = sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()

        if M.dtype.kind != "i":
            M = M.astype(np.int64, copy=False)

    return TenxMatrix(
        X=M, barcodes=barcodes, gene_id=gene_id, gene_symbol=gene_symbol, feature_type=feature_type, genome=genome
    )


def _read_10x_raw_or_filtered(outs_dir: Path, kind: str) -> TenxMatrix | None:
    assert kind in {"raw", "filtered"}
    h5 = outs_dir / f"{kind}_feature_bc_matrix.h5"
    if h5.exists():
        return _read_10x_h5_full(h5)

    mtx_dir = outs_dir / f"{kind}_feature_bc_matrix"
    if (mtx_dir / "matrix.mtx.gz").exists() or (mtx_dir / "matrix.mtx").exists():
        try:
            import scanpy as sc

            adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols", make_unique=False)
            features_tsv = (mtx_dir / "features.tsv.gz", mtx_dir / "features.tsv")
            fpath = None
            for cand in features_tsv:
                if Path(cand).exists():
                    fpath = Path(cand)
                    break
            if fpath is not None:
                df = pd.read_csv(fpath, sep="\t", header=None, names=["gene_id", "gene_symbol", "feature_type"])
                gene_id = df["gene_id"].astype(str).tolist()
                gene_symbol = df["gene_symbol"].astype(str).tolist()
                feature_type = df["feature_type"].astype(str).tolist()
            else:
                gene_symbol = adata.var_names.astype(str).tolist()
                if "gene_ids" in adata.var:
                    gene_id = adata.var["gene_ids"].astype(str).tolist()
                elif "gene_id" in adata.var:
                    gene_id = adata.var["gene_id"].astype(str).tolist()
                else:
                    gene_id = gene_symbol
                feature_type = adata.var.get("feature_types")
                feature_type = feature_type.astype(str).tolist() if feature_type is not None else None

            M = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
            if M.dtype.kind != "i":
                M = M.astype(np.int64, copy=False)
            return TenxMatrix(
                X=M,
                barcodes=adata.obs_names.astype(str).tolist(),
                gene_id=gene_id,
                gene_symbol=gene_symbol,
                feature_type=feature_type,
                genome=None,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read 10x mtx at {mtx_dir}: {e}")

    return None


def _tenx_to_adata(t: TenxMatrix) -> ad.AnnData:
    n_cells, n_genes = t.X.shape
    var = pd.DataFrame(
        {
            "gene_id": pd.Index(t.gene_id, name="gene_id"),
            "gene_symbol": pd.Index(t.gene_symbol, name="gene_symbol"),
        }
    )
    if t.feature_type is not None:
        var["feature_type"] = t.feature_type
    if t.genome is not None:
        var["genome"] = t.genome

    # integer index for var (AnnData may coerce to str internally; warning suppressed above)
    var.index = pd.RangeIndex(n_genes, name="feature_index")

    obs = pd.DataFrame(index=pd.Index(t.barcodes, name="barcode"))

    # NOTE: removed dtype= to avoid FutureWarning; X already has integer dtype.
    adata = ad.AnnData(X=t.X, obs=obs, var=var)
    return adata


def _align_layer_to_raw(raw: ad.AnnData, subset: ad.AnnData) -> sparse.csr_matrix:
    if "gene_id" not in raw.var or "gene_id" not in subset.var:
        raise ValueError("Both raw and subset must have var['gene_id'] to align.")

    want = pd.Series(raw.var["gene_id"].astype(str).values)
    have = pd.Series(subset.var["gene_id"].astype(str).values)

    if not want.equals(have):
        idx = pd.Index(have)
        try:
            reord = idx.get_indexer(want)
            if (reord < 0).any():
                return sparse.csr_matrix(raw.shape, dtype=raw.X.dtype)
            subset = subset[:, reord]
        except Exception:
            return sparse.csr_matrix(raw.shape, dtype=raw.X.dtype)

    pos = pd.Series(range(raw.n_obs), index=raw.obs_names)
    keep_mask = subset.obs_names.isin(raw.obs_names)
    if not np.all(keep_mask):
        subset = subset[keep_mask].copy()

    row_idx = pos.loc[subset.obs_names].astype(np.int64).values

    S = subset.X.tocoo() if sparse.issparse(subset.X) else sparse.coo_matrix(subset.X)
    new_rows = row_idx[S.row]
    layer = sparse.csr_matrix((S.data, (new_rows, S.col)), shape=raw.shape, dtype=raw.X.dtype)
    return layer


def _read_metrics_csv(csv_path: Path) -> dict[str, Any]:
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 1:
            return {k: (v.item() if hasattr(v, "item") else v) for k, v in df.iloc[0].to_dict().items()}
        else:
            if set(df.columns) >= {"Metric", "Value"}:
                return dict(zip(df["Metric"].astype(str), df["Value"]))
            return {"raw_csv": df.to_dict(orient="list")}
    except Exception as e:
        return {"_error": f"failed_to_parse: {e}"}


def _gather_single_outs(
    outs_dir: Path,
    dataset: str,
    assembly: str,
    release: int,
    silent: bool,
    keep_only_filtered: bool,
) -> ad.AnnData:
    """
    Single 'count' run.
    - keep_only_filtered=True: X := filtered counts, only filtered barcodes present.
    - keep_only_filtered=False: X := raw counts, layer['filtered'] aligned (if available).
    """
    if keep_only_filtered:
        filt = _read_10x_raw_or_filtered(outs_dir, "filtered")
        if filt is None:
            raise FileNotFoundError(f"Requested filtered-only but no filtered_feature_bc_matrix found under {outs_dir}")
        adata = _tenx_to_adata(filt)
        adata.obs["is_filtered_by_cellranger"] = True

    else:
        raw = _read_10x_raw_or_filtered(outs_dir, "raw")
        if raw is None:
            raise FileNotFoundError(f"No raw feature-barcode matrix found under {outs_dir}")
        adata = _tenx_to_adata(raw)
        filt = _read_10x_raw_or_filtered(outs_dir, "filtered")
        if filt is not None:
            filt_adata = _tenx_to_adata(filt)
            adata.layers["filtered"] = _align_layer_to_raw(adata, filt_adata)
            is_filtered = pd.Series(False, index=adata.obs_names)
            is_filtered.loc[filt_adata.obs_names] = True
            adata.obs["is_filtered_by_cellranger"] = is_filtered.values

    cr_uns: dict[str, Any] = {
        "dataset": dataset,
        "assembly_label": assembly,
        "ensembl_release": int(release),
        "outs_dir": str(outs_dir.resolve()),
        "web_summary_html": (
            str((outs_dir / "web_summary.html").resolve()) if (outs_dir / "web_summary.html").exists() else None
        ),
        "cloupe_file": str((outs_dir / "cloupe.cloupe").resolve()) if (outs_dir / "cloupe.cloupe").exists() else None,
        "molecule_info_h5": (
            str((outs_dir / "molecule_info.h5").resolve()) if (outs_dir / "molecule_info.h5").exists() else None
        ),
    }
    m_csv = outs_dir / "metrics_summary.csv"
    if m_csv.exists():
        cr_uns["metrics_summary"] = _read_metrics_csv(m_csv)
    adata.uns["cellranger"] = cr_uns
    return adata


def _gather_multi_outs(
    outs_dir: Path,
    dataset: str,
    assembly: str,
    release: int,
    silent: bool,
    keep_only_filtered: bool,
) -> ad.AnnData:
    """
    'multi' run.
    - keep_only_filtered=True: concatenate filtered per-sample counts; only filtered barcodes present.
    - keep_only_filtered=False: original behavior (raw X, optional filtered layer).
    """
    per_sample = list((outs_dir / "per_sample_outs").glob("*"))
    per_sample = [p for p in per_sample if (p / "count").is_dir()]
    if not per_sample:
        return _gather_single_outs(outs_dir, dataset, assembly, release, silent, keep_only_filtered)

    if keep_only_filtered:
        filt_adatas: list[ad.AnnData] = []
        for sdir in sorted(per_sample):
            sid = sdir.name
            s_count = sdir / "count"
            f = _read_10x_raw_or_filtered(s_count, "filtered")
            if f is None:
                _log(silent, f"[WARN] No filtered matrix for sample {sid} under {s_count}; skipping sample.")
                continue
            af = _tenx_to_adata(f)
            af.obs["sample"] = sid
            af.obs["is_filtered_by_cellranger"] = True
            filt_adatas.append(af)

        if not filt_adatas:
            raise FileNotFoundError(f"Requested filtered-only but no per-sample filtered matrices under {outs_dir}")

        base = filt_adatas[0]
        same_order = True
        for af in filt_adatas[1:]:
            if not np.array_equal(af.var["gene_id"].values, base.var["gene_id"].values):
                same_order = False
                break

        if same_order:
            X = sparse.vstack([af.X for af in filt_adatas], format="csr")
            obs = pd.concat([af.obs for af in filt_adatas], axis=0)
            var = base.var.copy()
            adata = ad.AnnData(X=X, obs=obs, var=var)
        else:
            tmp = []
            for af in filt_adatas:
                af = af.copy()
                af.var_names = af.var["gene_id"].astype(str)
                tmp.append(af)
            adata = ad.concat(tmp, axis=0, join="outer", merge="same", label=None, index_unique=None, fill_value=0)
            if sparse.issparse(adata.X):
                adata.X = adata.X.astype(np.int64, copy=False)
            else:
                if np.all(np.mod(adata.X, 1) == 0):
                    adata.X = adata.X.astype(np.int64)
            if "gene_id" not in adata.var:
                adata.var["gene_id"] = adata.var_names.astype(str).values
            if "gene_symbol" not in adata.var:
                adata.var["gene_symbol"] = adata.var_names.astype(str).values
            adata.var.index = pd.RangeIndex(adata.n_vars, name="feature_index")

    else:
        # Original behavior (raw X + optional filtered layer)
        raw_adatas: list[ad.AnnData] = []
        filt_adatas_opt: list[ad.AnnData | None] = []

        for sdir in sorted(per_sample):
            sid = sdir.name
            s_count = sdir / "count"
            raw = _read_10x_raw_or_filtered(s_count, "raw")
            if raw is None:
                _log(silent, f"[WARN] No raw matrix for sample {sid} under {s_count}; skipping sample.")
                continue
            ar = _tenx_to_adata(raw)
            ar.obs["sample"] = sid
            raw_adatas.append(ar)

            f = _read_10x_raw_or_filtered(s_count, "filtered")
            if f is not None:
                af = _tenx_to_adata(f)
                af.obs["sample"] = sid
                filt_adatas_opt.append(af)
            else:
                filt_adatas_opt.append(None)

        if not raw_adatas:
            raise FileNotFoundError(f"No per-sample raw matrices found under {outs_dir}/per_sample_outs/*/count/")

        base = raw_adatas[0]
        same_order = True
        for ar in raw_adatas[1:]:
            if not np.array_equal(ar.var["gene_id"].values, base.var["gene_id"].values):
                same_order = False
                break

        if same_order:
            X = sparse.vstack([ar.X for ar in raw_adatas], format="csr")
            obs = pd.concat([ar.obs for ar in raw_adatas], axis=0)
            var = base.var.copy()
            adata = ad.AnnData(X=X, obs=obs, var=var)
        else:
            tmp = []
            for ar in raw_adatas:
                ar = ar.copy()
                ar.var_names = ar.var["gene_id"].astype(str)
                tmp.append(ar)
            adata = ad.concat(tmp, axis=0, join="outer", merge="same", label=None, index_unique=None, fill_value=0)
            if sparse.issparse(adata.X):
                adata.X = adata.X.astype(np.int64, copy=False)
            else:
                if np.all(np.mod(adata.X, 1) == 0):
                    adata.X = adata.X.astype(np.int64)
            if "gene_id" not in adata.var:
                adata.var["gene_id"] = adata.var_names.astype(str).values
            if "gene_symbol" not in adata.var:
                adata.var["gene_symbol"] = adata.var_names.astype(str).values
            adata.var.index = pd.RangeIndex(adata.n_vars, name="feature_index")

        has_any_filt = any(af is not None for af in filt_adatas_opt)
        if has_any_filt:
            parts: list[sparse.csr_matrix] = []
            mem_flags = []
            for ar, af in zip(raw_adatas, filt_adatas_opt):
                if af is None:
                    parts.append(sparse.csr_matrix(ar.shape, dtype=adata.X.dtype))
                    s = pd.Series(False, index=ar.obs_names)
                else:
                    parts.append(_align_layer_to_raw(ar, af))
                    s = pd.Series(False, index=ar.obs_names)
                    s.loc[af.obs_names] = True
                mem_flags.append(s)
            adata.layers["filtered"] = sparse.vstack(parts, format="csr")
            adata.obs["is_filtered_by_cellranger"] = pd.concat(mem_flags).reindex(adata.obs_names).values

    # Attach metadata
    cr_uns: dict[str, Any] = {
        "dataset": dataset,
        "assembly_label": assembly,
        "ensembl_release": int(release),
        "outs_dir": str(outs_dir.resolve()),
        "web_summary_html": (
            str((outs_dir / "web_summary.html").resolve()) if (outs_dir / "web_summary.html").exists() else None
        ),
        "cloupe_file": str((outs_dir / "cloupe.cloupe").resolve()) if (outs_dir / "cloupe.cloupe").exists() else None,
    }
    metrics: dict[str, Any] = {}
    for sdir in sorted(per_sample):
        m_csv = sdir / "count" / "metrics_summary.csv"
        if m_csv.exists():
            metrics[sdir.name] = _read_metrics_csv(m_csv)
    if metrics:
        cr_uns["metrics_summary_per_sample"] = metrics

    adata.uns["cellranger"] = cr_uns
    return adata


def _is_multi_outs(outs_dir: Path) -> bool:
    return (outs_dir / "per_sample_outs").is_dir()


def _write_h5ad(adata: ad.AnnData, out_path: Path, silent: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(out_path), compression="gzip")
    _log(silent, f"[OK] Wrote {out_path.name}  shape={adata.n_obs}x{adata.n_vars}")


# --------------------------- public API ---------------------------


def anndata_generator(
    alignments_root: Path | str,
    out_root: Path | str,
    *,
    keep_obs_if_it_is_filtered_by_cellranger: bool = True,
    silent: bool = False,
) -> dict[str, dict[str, dict[str, str]]]:
    """
    Generate one .h5ad per dataset/assembly/release.

    keep_obs_if_it_is_filtered_by_cellranger:
        True  -> only filtered barcodes are kept and written to adata.X
        False -> adata.X is raw counts; filtered (if present) goes to layers['filtered']
    """
    align_root = Path(alignments_root).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict[str, dict[str, str]]] = {}

    if not align_root.is_dir():
        raise NotADirectoryError(f"alignments_root not found: {align_root}")

    for dataset_dir in sorted([p for p in align_root.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        for tag_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            m = re.match(r"^(.+?)_(\d+)$", tag_dir.name)
            if not m:
                continue
            assembly, release_str = m.group(1), m.group(2)
            release = int(release_str)

            outs_info = _find_best_outs(tag_dir)
            if not outs_info:
                _log(silent, f"[WARN] No finished outs/ found under {tag_dir}")
                continue
            outs_dir, run_id = outs_info

            out_path = out_root / f"{dataset}_{assembly}_{release}.h5ad"
            result.setdefault(dataset, {}).setdefault(assembly, {})

            if out_path.exists():
                _log(silent, f"[SKIP] Exists: {out_path.name}")
                result[dataset][assembly][release_str] = str(out_path)
                continue

            try:
                _log(
                    silent,
                    f"[INFO] Building AnnData from {outs_dir} (run={run_id}) "
                    f"[filtered_only={keep_obs_if_it_is_filtered_by_cellranger}]",
                )
                if _is_multi_outs(outs_dir):
                    adata = _gather_multi_outs(
                        outs_dir, dataset, assembly, release, silent, keep_obs_if_it_is_filtered_by_cellranger
                    )
                else:
                    adata = _gather_single_outs(
                        outs_dir, dataset, assembly, release, silent, keep_obs_if_it_is_filtered_by_cellranger
                    )

                adata.uns.setdefault("provenance", {})
                adata.uns["provenance"].update(
                    dict(dataset=dataset, assembly_label=assembly, ensembl_release=release, outs_dir=str(outs_dir))
                )

                _write_h5ad(adata, out_path, silent)
                result[dataset][assembly][release_str] = str(out_path)

            except Exception as e:
                _log(silent, f"[ERROR] Failed for {dataset}/{assembly}_{release}: {e}")

    return result


# --------------------------- CLI ---------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Generate .h5ad files from Cell Ranger outputs.")
    ap.add_argument("--alignments-root", required=True, help="Root of alignments (dataset/assembly_release/run/outs)")
    ap.add_argument("--out-root", required=True, help="Destination folder for .h5ad files")
    # keep filtered defaults to True; expose only if you want to disable it via CLI
    ap.add_argument(
        "--use-raw",
        action="store_true",
        help="If set, keep raw cells (equivalent to keep_obs_if_it_is_filtered_by_cellranger=False)",
    )
    ap.add_argument("--silent", action="store_true", help="Suppress progress messages")
    args = ap.parse_args(argv)

    try:
        paths = anndata_generator(
            args.alignments_root,
            args.out_root,
            keep_obs_if_it_is_filtered_by_cellranger=(not args.use_raw),
            silent=args.silent,
        )
        if not args.silent:
            for ds, asms in paths.items():
                for asm, rels in asms.items():
                    for rel, p in rels.items():
                        print(f"{ds}/{asm}_{rel} -> {p}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_main())
