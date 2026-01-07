#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

import logging
import math
import time
import typing as _t

import pandas as pd
from tqdm import tqdm

from idtrack._external_mappers._constants import _MG_FIELDS_SUPERSET, _MG_SCOPES
from idtrack._external_mappers._utils import (
    _as_list,
    _chunker,
    _empty_result,
    _ensure_all_inputs,
    _json,
    _suppress_stdout_stderr,
    _species_for_mygene,
    _unique_not_null,
    canonical_db,
    logger,
    strip_version,
)


def _mg_extract(rec: dict, target: str) -> list[str]:
    target = canonical_db(target)
    if target == "hgnc_symbol":
        return _unique_not_null([rec.get("symbol")])
    if target == "entrez_gene":
        return _unique_not_null([rec.get("entrezgene")])
    if target == "hgnc_id":
        vals = []
        hg = rec.get("hgnc")
        if hg is not None:
            vals.extend(_as_list(hg))
        hg2 = rec.get("HGNC")
        if hg2 is not None:
            vals.extend(_as_list(hg2))
        return _unique_not_null(
            [f"HGNC:{v}".replace("HGNC:HGNC:", "HGNC:") if str(v).isdigit() else str(v) for v in vals]
        )

    if target in ("ensembl_gene", "ensembl_transcript", "ensembl_protein"):
        ens = rec.get("ensembl")
        vals = []
        if isinstance(ens, dict):
            if target == "ensembl_gene":
                vals.extend(_as_list(ens.get("gene")))
            elif target == "ensembl_transcript":
                vals.extend(_as_list(ens.get("transcript")))
            else:
                vals.extend(_as_list(ens.get("protein")))
        elif isinstance(ens, list):
            for e in ens:
                if not isinstance(e, dict):
                    continue
                if target == "ensembl_gene" and "gene" in e:
                    vals.append(e["gene"])
                if target == "ensembl_transcript" and "transcript" in e:
                    vals.append(e["transcript"])
                if target == "ensembl_protein" and "protein" in e:
                    vals.append(e["protein"])
        return _unique_not_null(vals)

    if target == "uniprot":
        up = rec.get("uniprot")
        vals = []
        if isinstance(up, dict):
            for k in ("Swiss-Prot", "TrEMBL", "SWISS-PROT", "trembl", "SwissProt"):
                if k in up:
                    vals.extend(_as_list(up[k]))
        elif up is not None:
            vals.extend(_as_list(up))
        return _unique_not_null(vals)

    if target == "refseq_mrna":
        rf = rec.get("refseq")
        vals = []
        if isinstance(rf, dict):
            vals.extend(_as_list(rf.get("rna")))
            vals.extend(_as_list(rf.get("mRNA")))
        return _unique_not_null(vals)

    if target == "refseq_protein":
        rf = rec.get("refseq")
        vals = []
        if isinstance(rf, dict):
            vals.extend(_as_list(rf.get("protein")))
        return _unique_not_null(vals)

    if target in {"wormbase", "flybase"}:
        return _unique_not_null([rec.get(target)])

    return []


def map_with_mygene(
    ids: _t.Iterable[str],
    input_db: str,
    output_db: str,
    *,
    species: str = "hsapiens",
    chunk_size: int = 1000,
    pause: float = 0.2,
    max_retries: int = 3,
    strip_versions: bool = True,
    show_progress: bool = True,
    suppress_method_verbosity: bool = True,
) -> pd.DataFrame:
    """Map IDs via MyGene.info."""
    try:
        import mygene  # type: ignore
    except Exception as e:
        raise RuntimeError("mygene is not installed. Try: pip install mygene") from e

    species = _species_for_mygene(species)
    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    clean_ids = [strip_version(i) if strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)
    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "mygene", release_used=None)

    mg = mygene.MyGeneInfo()

    scope = _MG_SCOPES.get(inp)
    if not scope:
        raise ValueError(f"MyGene: unsupported input scope for {inp!r}")

    fields = ",".join(sorted(set(_MG_FIELDS_SUPERSET)))
    frames: list[pd.DataFrame] = []

    n_batches = math.ceil(len(uniq_ids) / chunk_size)
    with tqdm(
        total=len(uniq_ids),
        desc="mygene",
        mininterval=0.25,
        disable=not show_progress,
        ncols=100,
        unit="ids",
    ) as progress:
        for b_idx, chunk in enumerate(_chunker(uniq_ids, chunk_size), start=1):
            logger.debug(f"MyGene: querying batch {b_idx}/{n_batches} (n={len(chunk)})")
            attempt = 0
            while True:
                attempt += 1
                try:
                    with _suppress_stdout_stderr(suppress_method_verbosity):
                        res = mg.querymany(
                            chunk,
                            scopes=scope,
                            fields=fields,
                            species=species,
                            returnall=False,
                            as_dataframe=False,
                            batch_size=min(chunk_size, 1000),
                        )
                    rows = []
                    for r in res:
                        q = r.get("query")
                        if q is None or str(q).strip() == "":
                            logger.warning(f"MyGene returned a result without a valid 'query' field: {r}")
                            continue
                        notfound = bool(r.get("notfound", False))
                        meta = {k: r.get(k) for k in ("_score", "taxid", "notfound") if k in r}
                        outs = [] if notfound else _mg_extract(r, outp)
                        if strip_versions:
                            outs = [strip_version(x) for x in outs]
                        if outs:
                            for ov in outs:
                                rows.append(
                                    {
                                        "input_id": q,
                                        "input_db": inp,
                                        "output_id": ov,
                                        "output_db": outp,
                                        "method": "mygene",
                                        "release_used": None,
                                        "metadata_json": _json(meta),
                                    }
                                )
                        else:
                            rows.append(
                                {
                                    "input_id": q,
                                    "input_db": inp,
                                    "output_id": None,
                                    "output_db": outp,
                                    "method": "mygene",
                                    "release_used": None,
                                    "metadata_json": _json(meta),
                                }
                            )
                    frames.append(pd.DataFrame(rows))
                    break
                except Exception as e:
                    logger.warning(f"MyGene batch failed (attempt {attempt}): {e}")
                    if attempt >= max_retries:
                        meta_err = {"error": str(e)}
                        frames.append(
                            pd.DataFrame(
                                {
                                    "input_id": chunk,
                                    "input_db": inp,
                                    "output_id": [None] * len(chunk),
                                    "output_db": outp,
                                    "method": "mygene",
                                    "release_used": None,
                                    "metadata_json": _json(meta_err),
                                }
                            )
                        )
                        break
                    time.sleep(pause * attempt)

            progress.update(len(chunk))
            time.sleep(pause)

    if not frames:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "mygene", release_used=None)

    out = pd.concat(frames, ignore_index=True)
    out = _ensure_all_inputs(out, clean_ids, inp, outp, "mygene", release_used=None)
    return out[
        [
            "input_id",
            "input_db",
            "mapping",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "metadata_json",
        ]
    ]
