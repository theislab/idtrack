"""
id_mapper.py

Flexible gene/protein ID conversion via MyGene.info, Ensembl BioMart
(pybiomart), g:Profiler (gprofiler-official) and gget.

All converters return the same normalized pandas.DataFrame with columns
input_id | input_db | output_id | output_db | method | release_used | mapping | metadata_json

"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import typing as _t
from dataclasses import dataclass
from urllib.parse import urlparse

import pandas as pd

__all__ = [
    "convert_ids",
    "map_with_mygene",
    "map_with_pybiomart",
    "map_with_gprofiler",
    "map_with_gget",
    "canonical_db",
    "SUPPORTED_DBS",
    "SUPPORTED_METHODS",
    "ConvertOptions",
    "selftest_basic",
    "selftest_detailed",
]

# ------------------------------- Logging ------------------------------------ #

logger = logging.getLogger("id_mapper")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

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


def canonical_db(db: str) -> str:
    """Return canonical DB key given a user-friendly/alias string."""
    if not isinstance(db, str) or not db.strip():
        raise ValueError("db must be a non-empty string")
    db_norm = db.strip().lower()
    if db_norm in _DB_ALIASES:
        return _DB_ALIASES[db_norm]
    if db_norm.startswith("ensg"):
        return "ensembl_gene"
    if db_norm.startswith("enst"):
        return "ensembl_transcript"
    if db_norm.startswith("ensp"):
        return "ensembl_protein"
    raise ValueError(f"Unsupported or unknown db alias: {db!r}. " f"Supported canonical DBs: {sorted(SUPPORTED_DBS)}")


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


def canonical_species(species: str | None) -> str:
    """
    Canonical organism code (g:Profiler / Ensembl style).

    Supported out-of-the-box:
        - human: hsapiens
        - mouse: mmusculus
        - pig:   sscrofa
    """
    if not species:
        return "hsapiens"
    s = species.strip().lower()
    return _SPECIES_ALIASES.get(s, s)


def _species_for_mygene(species: str | None) -> str:
    """MyGene expects common names like 'human' / 'mouse' / 'pig'."""
    cs = canonical_species(species)
    if cs == "hsapiens":
        return "human"
    if cs == "mmusculus":
        return "mouse"
    if cs == "sscrofa":
        return "pig"
    return cs


# ----------------------------- Helper utils -------------------------------- #

_ENS_RE = re.compile(r"^(ENS[A-Z]*\d+)")
_REFSEQ_VER_RE = re.compile(r"^([NX][MRP]_\d+)")
_WS = re.compile(r"\s+")


def strip_version(id_: str, db: str) -> str:
    """Strip version suffixes for Ensembl/RefSeq when appropriate."""
    if not isinstance(id_, str):
        return id_
    x = id_.strip()
    cdb = canonical_db(db)
    if cdb.startswith("ensembl_"):
        m = _ENS_RE.match(x)
        if m:
            return m.group(1)
        return x
    if cdb in {"refseq_mrna", "refseq_protein"}:
        m = _REFSEQ_VER_RE.match(x)
        if m:
            return m.group(1)
        return x
    return _WS.sub(" ", x)


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _unique_not_null(seq: _t.Iterable[_t.Any]) -> list[str]:
    seen, out = set(), []
    for v in seq:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _chunker(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _is_bare_numeric(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(s).strip()))


# --------------------------- Capability registry ---------------------------- #

SUPPORTED_METHODS = ("auto", "pybiomart", "mygene", "gprofiler", "gget")

_METHOD_HINTS: dict[str, set[str]] = {
    "pybiomart": {
        "ensembl_gene",
        "ensembl_transcript",
        "ensembl_protein",
    },
    "mygene": {
        "ensembl_gene",
        "ensembl_transcript",
        "ensembl_protein",
        "hgnc_symbol",
        "hgnc_id",
        "entrez_gene",
        "uniprot",
        "refseq_mrna",
        "refseq_protein",
        "wormbase",
        "flybase",
    },
    "gprofiler": {
        "ensembl_gene",
        "ensembl_transcript",
        "ensembl_protein",
        "hgnc_symbol",
        "entrez_gene",
        "uniprot",
        "refseq_mrna",
        "refseq_protein",
        "wormbase",
        "flybase",
    },
    # gget.info: Ensembl IDs in, several common namespaces out
    "gget": {
        "ensembl_gene",
        "hgnc_symbol",
        "entrez_gene",
        "uniprot",
    },
}


def _ordered_methods_for_pair(inp: str, outp: str) -> list[str]:
    """
    For a given (input_db, output_db) return methods in a sensible order.

    Special-case:
        - gget.info only accepts Ensembl IDs as input, so we only
          consider gget when input_db == 'ensembl_gene'.
    """
    inp = canonical_db(inp)
    outp = canonical_db(outp)

    preferred = ["pybiomart", "mygene", "gprofiler", "gget"]
    methods: list[str] = []

    for m in preferred:
        if m == "gget":
            # Our gget backend wraps gget.info — Ensembl gene IDs in, other
            # namespaces out.
            if inp == "ensembl_gene" and outp in _METHOD_HINTS["gget"]:
                methods.append("gget")
            continue

        if inp in _METHOD_HINTS[m] and outp in _METHOD_HINTS[m]:
            methods.append(m)

    # If nothing obvious, fall back to MyGene and g:Profiler as generalists.
    if not methods:
        for m in ("mygene", "gprofiler"):
            if m not in methods:
                methods.append(m)

    return methods


# ------------------------------- Public API -------------------------------- #


@dataclass
class ConvertOptions:
    species: str = "hsapiens"
    chunk_size: int = 1000
    pause: float = 0.2
    max_retries: int = 3
    strip_versions: bool = True
    # pybiomart archive/time
    ensembl_host: str | None = None  # e.g. "https://nov2020.archive.ensembl.org"
    dataset: str | None = None  # e.g. "hsapiens_gene_ensembl"
    # backend-specific time/release
    as_of_date: str | None = None  # g:Profiler version / date (if supported)
    release: str | int | None = None  # reserved for future backends
    verbose: bool = False


def convert_ids(
    ids: _t.Iterable[str],
    input_db: str,
    output_db: str,
    method: str = "auto",
    options: ConvertOptions | None = None,
) -> pd.DataFrame:
    """
    High-level flexible converter.

    method="auto" tries (in order) pybiomart → MyGene → g:Profiler → gget
    and stops at the first backend that returns at least one non-null
    output_id. Explicit method=... uses only that backend.

    All returned DataFrames contain an additional column `mapping`
    indicating one of:
        - "1:1"  : exactly one unique non-null output_id for this input_id
        - "1:n"  : multiple unique non-null output_ids for this input_id
        - "1:0"  : no mapped output_id for this input_id
    """
    if options is None:
        options = ConvertOptions()
    if options.verbose:
        logger.setLevel(logging.DEBUG)

    id_list = [str(x) for x in ids]
    if not id_list:
        return _empty_result()

    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}, got {method!r}")

    if method == "auto":
        methods_to_try = _ordered_methods_for_pair(inp, outp)
    else:
        methods_to_try = [method]

    last_df: pd.DataFrame | None = None
    last_err: Exception | None = None

    for m in methods_to_try:
        logger.debug(f"convert_ids: trying backend {m!r} for {inp}->{outp}")
        try:
            if m == "pybiomart":
                df = map_with_pybiomart(id_list, inp, outp, options)
            elif m == "mygene":
                df = map_with_mygene(id_list, inp, outp, options)
            elif m == "gprofiler":
                df = map_with_gprofiler(id_list, inp, outp, options)
            elif m == "gget":
                df = map_with_gget(id_list, inp, outp, options)
            else:
                raise RuntimeError(f"Unsupported method {m!r}")
        except Exception as e:
            logger.warning("%s backend failed: %s", m, e)
            last_err = e
            continue

        last_df = df

        # For explicit method, stop immediately.
        if method != "auto":
            break

        # For auto: accept the first backend that returns any mapped output.
        if not df.empty and df["output_id"].notna().any():
            break
        else:
            logger.debug(
                "convert_ids: backend %s returned no mappings, trying next backend",
                m,
            )

    if last_df is None:
        if last_err is not None:
            raise last_err
        return _empty_result()

    return last_df.reset_index(drop=True)


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
    options: ConvertOptions | None = None,
) -> pd.DataFrame:
    """Map IDs via MyGene.info."""
    try:
        import mygene  # type: ignore
    except Exception as e:
        raise RuntimeError("mygene is not installed. Try: pip install mygene") from e

    if options is None:
        options = ConvertOptions()

    species = _species_for_mygene(options.species)
    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    clean_ids = [strip_version(i, inp) if options.strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)
    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "mygene", release_used=None)

    mg = mygene.MyGeneInfo()

    scope = _MG_SCOPES.get(inp)
    if not scope:
        raise ValueError(f"MyGene: unsupported input scope for {inp!r}")

    fields = ",".join(sorted(set(_MG_FIELDS_SUPERSET)))
    frames: list[pd.DataFrame] = []

    n_batches = math.ceil(len(uniq_ids) / options.chunk_size)
    for b_idx, chunk in enumerate(_chunker(uniq_ids, options.chunk_size), start=1):
        logger.debug(f"MyGene: querying batch {b_idx}/{n_batches} (n={len(chunk)})")
        attempt = 0
        while True:
            attempt += 1
            try:
                res = mg.querymany(
                    chunk,
                    scopes=scope,
                    fields=fields,
                    species=species,
                    returnall=False,
                    as_dataframe=False,
                    batch_size=min(options.chunk_size, 1000),
                )
                rows = []
                for r in res:
                    q = r.get("query")
                    notfound = bool(r.get("notfound", False))
                    meta = {k: r.get(k) for k in ("_score", "taxid", "notfound") if k in r}
                    outs = [] if notfound else _mg_extract(r, outp)
                    if options.strip_versions:
                        outs = [strip_version(x, outp) for x in outs]
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
                if attempt >= options.max_retries:
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
                time.sleep(options.pause * attempt)

        time.sleep(options.pause)

    if not frames:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "mygene", release_used=None)

    out = pd.concat(frames, ignore_index=True)
    out = _ensure_all_inputs(out, clean_ids, inp, outp, "mygene", release_used=None)
    return out[
        [
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        ]
    ]


# --------------------------- Backend: pybiomart ----------------------------- #

# Per-canonical-DB candidate attribute names in Ensembl BioMart.
# We will choose the first one that actually exists in the dataset.
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

# Per-canonical-DB candidate filter names. Again, we choose the first one
# that actually exists in the dataset.
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
# Source: biomaRt `listEnsemblArchives()` output and Ensembl docs. :contentReference[oaicite:1]{index=1}
_ENSEMBL_ARCHIVE_BY_RELEASE: dict[int, str] = {
    # Older but still commonly referenced
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
    104: "may2021.archive.ensembl.org",  # Ensembl 104 == May 2021
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

# Non‑numeric special cases, e.g. GRCh37.
_ENSEMBL_SPECIAL_RELEASE_HOSTS: dict[str, str] = {
    # Ensembl GRCh37 archive
    "grch37": "grch37.ensembl.org",
}


def _ensembl_archive_host_for_release(
    release: int | str | None,
) -> str | None:
    """
    Resolve an Ensembl release (e.g. 104) or string key (e.g. 'GRCh37')
    to an archive host like 'may2021.archive.ensembl.org'.

    Returns None if the release is unknown, in which case callers
    should fall back to the default 'www.ensembl.org'.
    """
    if release is None:
        return None

    # String releases can be things like "104", "v104", "GRCh37", ...
    if isinstance(release, str):
        s = release.strip()
        if not s:
            return None

        key = s.lower()
        # Special non‑numeric keys first
        if key in _ENSEMBL_SPECIAL_RELEASE_HOSTS:
            return _ENSEMBL_SPECIAL_RELEASE_HOSTS[key]

        # Strip a leading "v" or "r" if present (e.g. "v104")
        import re as _re  # local import to avoid polluting global namespace

        m = _re.match(r"^[vr]?(\d+)$", key)
        if not m:
            return None
        try:
            rel_int = int(m.group(1))
        except ValueError:
            return None
    else:
        try:
            rel_int = int(release)
        except (TypeError, ValueError):
            return None

    return _ENSEMBL_ARCHIVE_BY_RELEASE.get(rel_int)


def _biomart_dataset_for_species(species: str, explicit: str | None = None) -> str:
    """
    Return the Ensembl BioMart dataset name for the given species.

    Follows the Ensembl / g:Profiler / pyensembl convention of
    concatenating first letter of genus and full species name:
        - Homo sapiens   -> hsapiens_gene_ensembl
        - Mus musculus   -> mmusculus_gene_ensembl
        - Sus scrofa     -> sscrofa_gene_ensembl

    If `explicit` is given, it is returned verbatim. This allows the caller
    to use strain-specific or non-standard datasets directly.
    """
    if explicit:
        return explicit
    s = canonical_species(species)
    return f"{s}_gene_ensembl"


def _normalize_biomart_host(host: str | None) -> str:
    """
    Normalize Ensembl BioMart host for pybiomart.

    Examples of valid outputs:
        "http://www.ensembl.org"
        "http://nov2020.archive.ensembl.org"
        "http://grch37.ensembl.org"
    """
    if not host:
        return "http://www.ensembl.org"

    host = host.strip()
    parsed = urlparse(host if "://" in host else "http://" + host)
    netloc = parsed.netloc or parsed.path.split("/")[0]
    return "http://" + netloc


def _bm_list_attribute_names(ds) -> list[str]:
    """Return a list of attribute names for a pybiomart Dataset."""
    try:
        attrs = ds.list_attributes()
    except Exception:
        attrs = getattr(ds, "attributes", None)

    if attrs is None:
        return []

    try:
        # biomaRt-style DataFrame
        if hasattr(attrs, "columns"):
            if "name" in attrs.columns:
                return [str(x) for x in attrs["name"].tolist()]
            else:
                return [str(x) for x in attrs.iloc[:, 0].tolist()]
        # dict or list-like
        if isinstance(attrs, dict):
            return [str(k) for k in attrs.keys()]
        return [str(x) for x in list(attrs)]
    except Exception:
        try:
            return list(attrs)
        except Exception:
            return []


def _bm_list_filter_names(ds) -> list[str]:
    """Return a list of filter names for a pybiomart Dataset."""
    try:
        filts = ds.list_filters()
    except Exception:
        filts = getattr(ds, "filters", None)

    if filts is None:
        return []

    try:
        if hasattr(filts, "columns"):
            if "name" in filts.columns:
                return [str(x) for x in filts["name"].tolist()]
            else:
                return [str(x) for x in filts.iloc[:, 0].tolist()]
        if isinstance(filts, dict):
            return [str(k) for k in filts.keys()]
        return [str(x) for x in list(filts)]
    except Exception:
        try:
            return list(filts)
        except Exception:
            return []


def _bm_pick_attribute(canonical_db_name: str, available_attrs: list[str]) -> str:
    """
    Choose a BioMart attribute name for a given canonical DB, based on:
      1. Explicit candidates in _BM_ATTR_CANDIDATES
      2. Fuzzy matching on common substrings if needed
    """
    cdb = canonical_db(canonical_db_name)
    attrs = list(dict.fromkeys(available_attrs))  # dedupe, preserve order
    attr_set = set(attrs)

    # 1) Try explicit candidates in preferred order
    candidates = _BM_ATTR_CANDIDATES.get(cdb, [])
    for cand in candidates:
        if cand in attr_set:
            return cand

    # 2) Fuzzy fallback based on typical naming patterns
    if cdb.startswith("ensembl_"):
        pattern = "ensembl_" + cdb.split("_", 1)[1]
    elif cdb == "hgnc_symbol":
        pattern = "external_gene"
    elif cdb == "hgnc_id":
        pattern = "hgnc"
    elif cdb == "entrez_gene":
        pattern = "entrez"
    elif cdb == "uniprot":
        pattern = "uniprot"
    elif cdb.startswith("refseq_"):
        pattern = "refseq"
    else:
        pattern = cdb

    prefix_hits = [a for a in attrs if a.startswith(pattern)]
    if prefix_hits:
        return prefix_hits[0]

    contains_hits = [a for a in attrs if pattern in a]
    if contains_hits:
        return contains_hits[0]

    raise RuntimeError(
        f"pybiomart: dataset does not provide any attribute compatible with "
        f"{cdb!r}; inspect `dataset.list_attributes()` for valid names."
    )


def _bm_pick_filter(
    canonical_db_name: str,
    attr_name: str,
    available_filters: list[str],
) -> str:
    """
    Choose a BioMart filter name for a given canonical DB and chosen attribute.
    We try, in order:

      1. DB-specific candidates from _BM_FILTER_CANDIDATES
      2. The attribute name itself
      3. Minor variants of the attribute name
      4. Fuzzy matching on common substrings
    """
    cdb = canonical_db(canonical_db_name)
    filt_list = list(dict.fromkeys(available_filters))  # dedupe, preserve order
    filt_set = set(filt_list)

    candidates: list[str] = []

    # 1) DB-specific candidates
    candidates.extend(_BM_FILTER_CANDIDATES.get(cdb, []))

    # 2) The attribute name itself
    candidates.append(attr_name)

    # 3) Some small variations
    if attr_name.endswith("_id"):
        candidates.append(attr_name[:-3])
    if attr_name.endswith("_accession"):
        candidates.append(attr_name[:-10])

    # Deduplicate candidates but preserve order
    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered_candidates.append(c)

    for c in ordered_candidates:
        if c in filt_set:
            return c

    # 4) Fuzzy search
    if cdb == "hgnc_symbol":
        pattern = "external_gene"
    elif cdb.startswith("ensembl_"):
        pattern = "ensembl_" + cdb.split("_", 1)[1]
    elif cdb == "entrez_gene":
        pattern = "entrez"
    elif cdb == "uniprot":
        pattern = "uniprot"
    elif cdb.startswith("refseq_"):
        pattern = "refseq"
    else:
        pattern = cdb

    prefix_hits = [f for f in filt_list if f.startswith(pattern)]
    if prefix_hits:
        return prefix_hits[0]

    contains_hits = [f for f in filt_list if pattern in f]
    if contains_hits:
        return contains_hits[0]

    raise RuntimeError(
        f"pybiomart: dataset does not provide any filter compatible with "
        f"{cdb!r}; inspect `dataset.list_filters()` for valid names."
    )


def map_with_pybiomart(
    ids: _t.Iterable[str],
    input_db: str,
    output_db: str,
    options: ConvertOptions | None = None,
) -> pd.DataFrame:
    """
    Map IDs via Ensembl BioMart (pybiomart).

    Time / release component:
        - Use `options.ensembl_host` to point to a specific Ensembl
          archive, e.g. "https://nov2020.archive.ensembl.org".
        - `release_used` in the output is set to the normalized host
          (this indirectly encodes the Ensembl release).
    """
    try:
        from pybiomart import Dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("pybiomart is not installed. Try: pip install pybiomart") from e

    if options is None:
        options = ConvertOptions()

    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    clean_ids = [strip_version(i, inp) if options.strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)

    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "pybiomart", release_used=None)

    # Resolve Ensembl host:
    #   1. If options.ensembl_host is set, use that verbatim.
    #   2. Else, if options.release is set, map release -> archive host.
    #   3. Else, fall back to the live www.ensembl.org.
    raw_host: str | None = options.ensembl_host

    resolved_release_host: str | None = None
    resolved_release_int: int | None = None

    if raw_host is None and options.release is not None:
        resolved_release_host = _ensembl_archive_host_for_release(options.release)
        if resolved_release_host:
            raw_host = resolved_release_host
            try:
                resolved_release_int = int(str(options.release).strip().lstrip("vr"))
            except Exception:
                resolved_release_int = None
        else:
            # Unknown release; harmlessly fall back to main Ensembl.
            logger.debug(
                "pybiomart: no known archive host for Ensembl release %r; " "falling back to www.ensembl.org",
                options.release,
            )

    if raw_host is None:
        raw_host = "http://www.ensembl.org"

    host = _normalize_biomart_host(raw_host)

    dataset_name = _biomart_dataset_for_species(options.species, options.dataset)

    try:
        ds = Dataset(name=dataset_name, host=host)
    except Exception as e:
        raise RuntimeError(
            f"pybiomart: failed to connect to Ensembl BioMart " f"(dataset={dataset_name!r}, host={host!r}): {e}"
        ) from e

    # Discover attributes and filters that actually exist for this dataset
    attr_names = _bm_list_attribute_names(ds)
    filter_names = _bm_list_filter_names(ds)

    if not attr_names:
        raise RuntimeError(f"pybiomart: could not retrieve attributes for dataset {dataset_name!r}")
    if not filter_names:
        raise RuntimeError(f"pybiomart: could not retrieve filters for dataset {dataset_name!r}")

    # Choose valid attribute + filter names for the requested mapping
    in_attr = _bm_pick_attribute(inp, attr_names)
    out_attr = _bm_pick_attribute(outp, attr_names)
    filter_name = _bm_pick_filter(inp, in_attr, filter_names)

    logger.debug(
        "pybiomart: using dataset=%r host=%r in_attr=%r out_attr=%r filter=%r",
        dataset_name,
        host,
        in_attr,
        out_attr,
        filter_name,
    )

    frames: list[pd.DataFrame] = []
    n_chunks = math.ceil(len(uniq_ids) / options.chunk_size)

    for i, chunk in enumerate(_chunker(uniq_ids, options.chunk_size), start=1):
        logger.debug(
            "pybiomart: querying chunk %d/%d (n=%d)",
            i,
            n_chunks,
            len(chunk),
        )
        try:
            df = ds.query(
                attributes=[in_attr, out_attr],
                filters={filter_name: chunk},
                use_attr_names=True,  # keep internal attribute names as headers
            )

            # If pybiomart returns an empty DataFrame, create a skeleton
            if df is None or df.empty:
                frames.append(
                    pd.DataFrame(
                        {
                            "input_id": chunk,
                            "output_id": [None] * len(chunk),
                        }
                    )
                )
            else:
                df = df.rename(columns={in_attr: "input_id", out_attr: "output_id"})
                keep_cols = [c for c in ("input_id", "output_id") if c in df.columns]
                if not keep_cols:
                    frames.append(
                        pd.DataFrame(
                            {
                                "input_id": chunk,
                                "output_id": [None] * len(chunk),
                            }
                        )
                    )
                else:
                    frames.append(df[keep_cols].drop_duplicates())

        except Exception as e:
            logger.warning(f"pybiomart chunk failed: {e}")
            meta = {"error": str(e)}
            frames.append(
                pd.DataFrame(
                    {
                        "input_id": chunk,
                        "output_id": [None] * len(chunk),
                        "metadata_json": [_json(meta)] * len(chunk),
                    }
                )
            )

        time.sleep(options.pause)

    if not frames:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "pybiomart", release_used=host)

    out = pd.concat(frames, ignore_index=True)

    out["input_db"] = inp
    out["output_db"] = outp
    out["method"] = "pybiomart"
    out["release_used"] = host
    if "metadata_json" not in out.columns:
        out["metadata_json"] = _json({})

    out = _ensure_all_inputs(out, clean_ids, inp, outp, "pybiomart", release_used=host)
    out = out.drop_duplicates(["input_id", "output_id", "input_db", "output_db", "method", "release_used"])

    return out[
        [
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        ]
    ]


# --------------------------- Backend: g:Profiler ---------------------------- #

# Mapping from our canonical DB names -> g:Profiler namespace codes
# See: https://biit.cs.ut.ee/gprofiler/page/namespaces-list
# - HGNC        : HGNC gene symbol
# - HGNC_ACC    : HGNC accession (e.g. HGNC:11998)
# - ENTREZGENE_ACC : Entrez Gene numeric ID
# - UNIPROTSWISSPROT_ACC : UniProtKB/Swiss-Prot accessions (e.g. P04637)
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


def _gp_target_candidates(outp: str) -> list[str]:
    """
    Return an ordered list of g:Profiler target_namespace candidates
    for a given canonical output DB.

    For most DBs this is a single value from _GP_NS.

    For UniProt we try several closely related namespaces, preferring
    accession-based namespaces with broad coverage (Swiss-Prot + TrEMBL).
    """
    outp = canonical_db(outp)

    if outp == "uniprot":
        # Ordered by: broad accession coverage -> Swiss-Prot-only accession
        #            -> entry-name namespaces as a last resort.
        candidates = [
            "UNIPROTSPTREMBL_ACC",  # UniProtKB (Swiss-Prot + TrEMBL) accessions
            "UNIPROTSWISSPROT_ACC",  # Swiss-Prot accessions
            "UNIPROTSPTREMBL",  # UniProt entry names
            "UNIPROTSWISSPROT",  # Swiss-Prot entry names
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for ns in candidates:
            if ns and ns not in seen:
                seen.add(ns)
                ordered.append(ns)
        return ordered

    base = _GP_NS.get(outp)
    return [base] if base else []


def map_with_gprofiler(
    ids: _t.Iterable[str],
    input_db: str,
    output_db: str,
    options: ConvertOptions | None = None,
) -> pd.DataFrame:
    """Map IDs via g:Profiler (gprofiler-official)."""
    try:
        from gprofiler import GProfiler  # type: ignore
    except Exception as e:
        raise RuntimeError("gprofiler-official is not installed. Try: pip install gprofiler-official") from e

    if options is None:
        options = ConvertOptions()

    species = canonical_species(options.species)
    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    target_candidates = _gp_target_candidates(outp)
    if not target_candidates:
        raise ValueError(f"g:Profiler: unsupported target namespace for {outp!r}")

    clean_ids = [strip_version(i, inp) if options.strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)
    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gprofiler", release_used=None)

    gp = GProfiler(return_dataframe=True)

    # Introspect signature so we work with both 0.3.x and 1.x
    try:
        from inspect import signature

        sig_params = set(signature(gp.convert).parameters)
    except Exception:
        sig_params = set()

    # target parameter name: target_namespace (1.x) vs target (0.3.x)
    if "target_namespace" in sig_params:
        target_key = "target_namespace"
    elif "target" in sig_params:
        target_key = "target"
    else:
        target_key = "target_namespace"  # best guess, will raise at call time if wrong

    # Numeric-namespace parameter name: numeric_namespace vs numeric_ns
    numeric_param: str | None = None
    if "numeric_namespace" in sig_params:
        numeric_param = "numeric_namespace"
    elif "numeric_ns" in sig_params:
        numeric_param = "numeric_ns"

    # Base kwargs that are independent of query / target_namespace
    base_kwargs: dict[str, _t.Any] = {"organism": species}

    # Optional as-of/date parameter (for archived releases, if exposed)
    if options.as_of_date:
        if "date" in sig_params:
            base_kwargs["date"] = options.as_of_date
        elif "as_of" in sig_params:
            base_kwargs["as_of"] = options.as_of_date

    # Help g:Profiler interpret fully numeric Entrez IDs correctly
    # by explicitly telling it they are ENTREZGENE_ACC.
    if numeric_param is not None and inp == "entrez_gene" and all(_is_bare_numeric(x) for x in uniq_ids):
        base_kwargs[numeric_param] = "ENTREZGENE_ACC"

    last_error: Exception | None = None
    selected_frames: list[pd.DataFrame] | None = None

    for target_ns in target_candidates:
        logger.debug("g:Profiler: trying target namespace %r", target_ns)
        frames: list[pd.DataFrame] = []
        any_non_null = False
        n_chunks = math.ceil(len(uniq_ids) / options.chunk_size)

        for i, chunk in enumerate(_chunker(uniq_ids, options.chunk_size), start=1):
            logger.debug(
                "g:Profiler: querying chunk %d/%d (n=%d, target_ns=%s)",
                i,
                n_chunks,
                len(chunk),
                target_ns,
            )
            attempt = 0
            while True:
                attempt += 1
                try:
                    kwargs = dict(base_kwargs)
                    kwargs[target_key] = target_ns
                    kwargs["query"] = chunk
                    df = gp.convert(**kwargs)

                    if df is None or df.empty:
                        # No rows for this chunk and target_ns
                        frames.append(pd.DataFrame(columns=["input_id", "output_id", "metadata_json"]))
                    else:
                        keep_cols = ["incoming", "converted"]
                        extra_cols = [c for c in df.columns if c not in keep_cols]
                        keep = df[keep_cols].rename(columns={"incoming": "input_id", "converted": "output_id"})
                        keep = keep.drop_duplicates()

                        if extra_cols:
                            # Pack the remaining columns into metadata_json per input_id
                            meta_records: list[tuple[_t.Any, dict[str, _t.Any]]] = []
                            for _, row in df[["incoming"] + extra_cols].drop_duplicates().iterrows():
                                q = row["incoming"]
                                meta = {c: row[c] for c in extra_cols}
                                meta_records.append((q, meta))
                            meta_df = pd.DataFrame(meta_records, columns=["input_id", "meta"])
                            keep = keep.merge(meta_df, on="input_id", how="left")
                            keep["metadata_json"] = keep["meta"].apply(
                                lambda m: _json(m if isinstance(m, dict) else {})
                            )
                            keep = keep.drop(columns=["meta"])
                        else:
                            keep["metadata_json"] = _json({})

                        frames.append(keep)
                        if "output_id" in keep.columns and keep["output_id"].notna().any():
                            any_non_null = True

                    break  # success, move to next chunk

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "g:Profiler batch failed for target %s (attempt %d): %s",
                        target_ns,
                        attempt,
                        e,
                    )
                    if attempt >= options.max_retries:
                        err_meta = _json({"error": str(e), "target_namespace": target_ns})
                        frames.append(
                            pd.DataFrame(
                                {
                                    "input_id": chunk,
                                    "output_id": [None] * len(chunk),
                                    "metadata_json": [err_meta] * len(chunk),
                                }
                            )
                        )
                        break
                    time.sleep(options.pause * attempt)

            time.sleep(options.pause)

        if frames and any_non_null:
            selected_frames = frames
            break
        # Otherwise, try the next candidate namespace

    if selected_frames is None:
        # No candidate produced any mapped outputs.
        if last_error is not None:
            base = pd.DataFrame(
                {
                    "input_id": [str(x) for x in clean_ids],
                    "input_db": inp,
                    "output_id": [None] * len(clean_ids),
                    "output_db": outp,
                    "method": "gprofiler",
                    "release_used": options.as_of_date,
                    "metadata_json": _json({"error": str(last_error)}),
                }
            )
            base = _add_mapping_column(base)
            return base[
                [
                    "input_id",
                    "input_db",
                    "output_id",
                    "output_db",
                    "method",
                    "release_used",
                    "mapping",
                    "metadata_json",
                ]
            ]

        return _ensure_all_inputs(
            _empty_result(),
            clean_ids,
            inp,
            outp,
            "gprofiler",
            release_used=options.as_of_date,
        )

    out = pd.concat(selected_frames, ignore_index=True)

    if "metadata_json" not in out.columns:
        out["metadata_json"] = _json({})

    out["input_db"] = inp
    out["output_db"] = outp
    out["method"] = "gprofiler"
    out["release_used"] = options.as_of_date

    out = _ensure_all_inputs(out, clean_ids, inp, outp, "gprofiler", release_used=options.as_of_date)
    out = out.drop_duplicates(["input_id", "output_id", "input_db", "output_db", "method", "release_used"])

    return out[
        [
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        ]
    ]


# ------------------------------ Backend: gget ------------------------------- #


def _gget_extract(df: pd.DataFrame, outp: str) -> pd.DataFrame:
    """
    Normalize the output from gget.info() into a simple (input_id, output_id)
    DataFrame for a requested target DB.

    gget.info (v0.3.x+) returns columns such as:
        ensembl_id, uniprot_id, ncbi_gene_id, primary_gene_name, ...
    """
    # First, standardize column names into a small internal vocabulary
    rename = {
        # Ensembl / gene IDs
        "id": "gene_id",
        "gene": "gene_id",
        "ensembl_id": "gene_id",
        # Gene symbols / names
        "name": "gene_name",
        "display_name": "gene_name",
        "symbol": "gene_name",
        "gene_symbol": "gene_name",
        "primary_gene_name": "gene_name",
        # Entrez / NCBI
        "entrez": "entrez_id",
        "entrezgene": "entrez_id",
        "entrez_gene": "entrez_id",
        "ncbi_gene_id": "entrez_id",
        # UniProt
        "uniprot": "uniprot_id",
        "uniprot_acc": "uniprot_id",
        "uniprot_id": "uniprot_id",
        # Misc
        "protein_id": "protein_id",
        "transcript_id": "transcript_id",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    outp = canonical_db(outp)
    col_map = {
        "hgnc_symbol": "gene_name",
        "ensembl_gene": "gene_id",
        # gget.info exposes `canonical_transcript` rather than a generic transcript_id
        "ensembl_transcript": "canonical_transcript",
        "ensembl_protein": "protein_id",
        "entrez_gene": "entrez_id",
        "uniprot": "uniprot_id",
    }
    out_col = col_map.get(outp)
    if out_col is None:
        return pd.DataFrame(columns=["input_id", "output_id"])

    # What should we treat as the "input" ID?
    # gget.info currently does NOT return a separate "query" column,
    # but it does return `ensembl_id` which is what we passed in.
    if "query" in df.columns:
        src = df["query"].astype(str).tolist()
    elif "gene_id" in df.columns:
        src = df["gene_id"].astype(str).tolist()
    else:
        src = df.iloc[:, 0].astype(str).tolist()

    if out_col in df.columns:
        outs = df[out_col].astype(object).tolist()
    else:
        outs = [None] * len(src)

    have_in, have_out = [], []
    for s, o in zip(src, outs):
        have_in.append(s)
        if o is None or str(o).lower() in {"nan", "none", "null"}:
            have_out.append(None)
        else:
            have_out.append(str(o))
    return pd.DataFrame({"input_id": have_in, "output_id": have_out})


def map_with_gget(
    ids: _t.Iterable[str],
    input_db: str,
    output_db: str,
    options: ConvertOptions | None = None,
) -> pd.DataFrame:
    """
    Map IDs using gget.info (Ensembl REST-backed).

    Important: gget.info takes Ensembl (gene) IDs as input (plus some model
    organism IDs) and returns rich metadata including UniProt and NCBI IDs.
    Here we support ONLY Ensembl gene IDs as input.
    """
    try:
        from gget import info as gget_info  # type: ignore
    except Exception as e:
        raise RuntimeError("gget is not installed. Try: pip install gget") from e

    if options is None:
        options = ConvertOptions()

    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    if inp != "ensembl_gene":
        raise ValueError(f"gget backend currently supports only Ensembl gene IDs as input, " f"got {inp!r}")

    clean_ids = [strip_version(i, inp) if options.strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)
    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gget", release_used=None)

    # gget.info derives species from the Ensembl IDs; most versions do NOT
    # accept a `species` or `translate` keyword. We only pass well-supported
    # kwargs based on runtime introspection of the signature.
    try:
        from inspect import signature

        sig_params = set(signature(gget_info).parameters)
    except Exception:
        sig_params = set()

    kwargs: dict[str, _t.Any] = {}
    # From gget 0.3.x docs:
    #   info(ens_ids, wrap_text=False, pdb=False, ensembl_only=False,
    #        json=False, verbose=True, save=False, expand=False)
    if "wrap_text" in sig_params:
        kwargs["wrap_text"] = False
    if "pdb" in sig_params:
        kwargs["pdb"] = False
    if "ensembl_only" in sig_params:
        kwargs["ensembl_only"] = False
    if "json" in sig_params:
        kwargs["json"] = False
    if "verbose" in sig_params:
        kwargs["verbose"] = False
    if "save" in sig_params:
        kwargs["save"] = False
    if "expand" in sig_params:
        kwargs["expand"] = False
    # DO NOT pass a `species` argument unless it actually exists in the
    # signature (most published versions do not support it).
    species_code = canonical_species(options.species)
    gget_species = {
        "hsapiens": "homo_sapiens",
        "mmusculus": "mus_musculus",
        "sscrofa": "sus_scrofa",
    }.get(species_code, species_code)
    if "species" in sig_params:
        kwargs["species"] = gget_species

    frames: list[pd.DataFrame] = []
    n_chunks = math.ceil(len(uniq_ids) / options.chunk_size)

    for i, chunk in enumerate(_chunker(uniq_ids, options.chunk_size), start=1):
        logger.debug(f"gget: querying chunk {i}/{n_chunks} (n={len(chunk)})")
        attempt = 0
        while True:
            attempt += 1
            try:
                df_raw = gget_info(chunk, **kwargs)
                if not isinstance(df_raw, pd.DataFrame):
                    df_raw = pd.DataFrame(df_raw)
                part = _gget_extract(df_raw, outp)
                frames.append(part)
                break
            except Exception as e:
                logger.warning(f"gget batch failed (attempt {attempt}): {e}")
                if attempt >= options.max_retries:
                    frames.append(
                        pd.DataFrame(
                            {
                                "input_id": chunk,
                                "output_id": [None] * len(chunk),
                            }
                        )
                    )
                    break
                time.sleep(options.pause * attempt)
        time.sleep(options.pause)

    if not frames:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gget", release_used=None)

    out = pd.concat(frames, ignore_index=True)

    # NORMALIZATION STEP
    if "input_id" not in out.columns:
        out["input_id"] = None
    out["input_id"] = out["input_id"].astype(str)
    if options.strip_versions:
        # For Ensembl / RefSeq inputs this removes the version suffix so that
        # gget’s versioned IDs (e.g. ENSG00000141510.20) align with the
        # versionless IDs we used as queries (e.g. ENSG00000141510).
        out["input_id"] = [strip_version(x, inp) for x in out["input_id"].tolist()]
    else:
        out["input_id"] = [x.strip() for x in out["input_id"].tolist()]

    out["input_db"] = inp
    out["output_db"] = outp
    out["method"] = "gget"
    out["release_used"] = None
    out["metadata_json"] = _json({})
    out = _ensure_all_inputs(out, clean_ids, inp, outp, "gget", release_used=None)
    out = out.drop_duplicates(["input_id", "output_id", "input_db", "output_db", "method", "release_used"])
    return out[
        [
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        ]
    ]


# ---------------------------- Utilities/Finalizers -------------------------- #


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        ]
    )


def _add_mapping_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add / recompute the `mapping` column on a DataFrame with at least
    `input_id` and `output_id` columns.

    mapping is per-input cardinality of the mapping:
        1:0  -> no non-null output_id
        1:1  -> exactly one unique non-null output_id
        1:n  -> more than one unique non-null output_id
    """
    if df is None or df.empty:
        if "mapping" not in df.columns:
            df["mapping"] = pd.Series(dtype=object)
        return df

    if "output_id" not in df.columns or "input_id" not in df.columns:
        # Fallback: mark everything as unmapped
        df["mapping"] = "1:0"
        return df

    # Drop any existing mapping column; we'll recompute it from scratch.
    if "mapping" in df.columns:
        df = df.drop(columns=["mapping"])

    inputs = df["input_id"].astype(str)
    outputs = df["output_id"]

    out_str = outputs.astype(str)
    valid_mask = ~outputs.isna() & out_str.str.strip().ne("") & ~out_str.str.lower().isin({"nan", "none", "null"})

    # Number of unique valid outputs per input
    counts_by_input = df[valid_mask].groupby(inputs[valid_mask])["output_id"].nunique(dropna=True)

    mapping_by_input: dict[str, str] = {}
    for inp_val in inputs.unique():
        key = str(inp_val)
        n = int(counts_by_input.get(key, 0) or 0)
        if n == 0:
            mapping_by_input[key] = "1:0"
        elif n == 1:
            mapping_by_input[key] = "1:1"
        else:
            mapping_by_input[key] = "1:n"

    df["mapping"] = inputs.map(mapping_by_input)

    return df


def _ensure_all_inputs(
    df: pd.DataFrame,
    original_inputs: list[str],
    inp: str,
    outp: str,
    method: str,
    release_used: str | None,
) -> pd.DataFrame:
    """
    Guarantee that each input appears at least once in the output
    (with output_id=None if unmapped). Preserve input order and add
    the `mapping` column.

    mapping semantics:
        "1:0" -> no mapped outputs for this input
        "1:1" -> exactly one unique mapped output
        "1:n" -> multiple mapped outputs
    """
    if df is None or df.empty:
        base = pd.DataFrame(
            {
                "input_id": [str(x) for x in original_inputs],
                "input_db": inp,
                "output_id": [None] * len(original_inputs),
                "output_db": outp,
                "method": method,
                "release_used": release_used,
                "metadata_json": _json({}),
            }
        )
        base = _add_mapping_column(base)
        return base

    for col in (
        "input_id",
        "input_db",
        "output_id",
        "output_db",
        "method",
        "release_used",
        "metadata_json",
    ):
        if col not in df.columns:
            if col in {"input_db", "output_db", "method"}:
                default_map = {"input_db": inp, "output_db": outp, "method": method}
                df[col] = default_map[col]
            elif col == "release_used":
                df[col] = release_used
            elif col == "metadata_json":
                df[col] = _json({})
            else:
                df[col] = None

    present = set(df["input_id"].astype(str))
    missing = [x for x in original_inputs if str(x) not in present]
    if missing:
        tail = pd.DataFrame(
            {
                "input_id": missing,
                "input_db": inp,
                "output_id": [None] * len(missing),
                "output_db": outp,
                "method": method,
                "release_used": release_used,
                "metadata_json": _json({}),
            }
        )
        df = pd.concat([df, tail], ignore_index=True)

    order_map = {str(x): i for i, x in enumerate(original_inputs)}
    df = df.copy()
    df["__ord"] = df["input_id"].astype(str).map(order_map)
    df = df.sort_values(["__ord", "output_id"], na_position="last").drop(columns="__ord")
    df = df.reset_index(drop=True)

    df = _add_mapping_column(df)
    return df
