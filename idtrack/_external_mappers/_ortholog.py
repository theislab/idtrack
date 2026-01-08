from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from idtrack._external_mappers._constants import _SPECIES_ALIASES, _SPECIES_CANONICAL_TO_BGEENAMES
from idtrack._external_mappers._utils import raise_missing_dependency

# Lazy-loaded optional dependencies (gget, biopython)
_gget = None
_substitution_matrices = None
_SeqIO = None
_BLOSUM62 = None


def _require_ortholog_deps():
    """Lazily import gget and biopython; raise helpful error if missing."""
    global _gget, _substitution_matrices, _SeqIO
    if _gget is None:
        # Check gget first
        try:
            import gget as _gget_mod
        except ImportError as e:
            raise_missing_dependency("gget", feature="ortholog utilities", original_error=e)

        # Check biopython
        try:
            from Bio import SeqIO
            from Bio.Align import substitution_matrices as _sm
        except ImportError as e:
            raise_missing_dependency("biopython", feature="ortholog utilities", original_error=e)

        _gget = _gget_mod
        _substitution_matrices = _sm
        _SeqIO = SeqIO
    return _gget, _substitution_matrices, _SeqIO


def _get_blosum62():
    """Lazily load BLOSUM62 matrix."""
    global _BLOSUM62
    if _BLOSUM62 is None:
        _, substitution_matrices, _ = _require_ortholog_deps()
        _BLOSUM62 = substitution_matrices.load("BLOSUM62")
    return _BLOSUM62


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ------------------------ species helpers ------------------------ #


def _canonical_from_alias(name: str) -> str:
    """Return canonical short-code for a species alias.

    Maps strings like ``"human"``, ``"Homo sapiens"``, ``"sus_scrofa"`` to canonical
    short codes (``hsapiens``, ``mmusculus``, ``sscrofa``). Unknown inputs are
    returned unchanged (lowercased/stripped).

    Args:
        name: Species name or alias.

    Returns:
        str: Canonical short code (or a normalized fallback if unknown).
    """
    raw = name.strip().lower()
    if raw in _SPECIES_ALIASES:
        return _SPECIES_ALIASES[raw]

    raw_us = raw.replace(" ", "_")
    if raw_us in _SPECIES_ALIASES:
        return _SPECIES_ALIASES[raw_us]

    # maybe already 'hsapiens', or something like 'bos taurus'
    return raw


def _species_to_genus_species(name: str) -> tuple[str, str, str]:
    """Convert a species string to ``(canonical_code, genus, species)`` for Bgee.

    Inputs may be any key in :py:data:`_SPECIES_ALIASES` or any canonical key in
    :py:data:`_SPECIES_CANONICAL_TO_BGEENAMES`. Unknown canonical codes raise a
    helpful error.

    Args:
        name: Species name or alias.

    Returns:
        tuple[str, str, str]: ``(canonical_code, genus, species)`` for Bgee.

    Raises:
        ValueError: If the species cannot be resolved to a supported canonical code.
    """
    canonical = _canonical_from_alias(name)

    if canonical not in _SPECIES_CANONICAL_TO_BGEENAMES:
        known_canonical = ", ".join(sorted(_SPECIES_CANONICAL_TO_BGEENAMES.keys()))
        known_aliases = ", ".join(sorted(_SPECIES_ALIASES.keys()))
        raise ValueError(
            f"Species {name!r} could not be resolved to a canonical code in "
            f"_SPECIES_CANONICAL_TO_BGEENAMES.\n"
            f"Known canonical codes: {known_canonical}\n"
            f"Known aliases: {known_aliases}\n"
            f"If you need another organism, add it to "
            f"_SPECIES_CANONICAL_TO_BGEENAMES (and optionally _SPECIES_ALIASES)."
        )

    genus, species = _SPECIES_CANONICAL_TO_BGEENAMES[canonical]
    return canonical, genus, species


# ------------------------ dataclasses ------------------------ #


@dataclass
class AlignmentScores:
    """Alignment-derived scalar metrics for a pairwise protein alignment."""

    alignment_length: int
    identity_fraction: float
    positive_fraction: float
    very_negative_fraction: float
    gap_fraction_query: float
    gap_fraction_target: float
    gap_openings_query: int
    gap_openings_target: int
    seq1_coverage: float
    seq2_coverage: float
    blosum62_sum: float
    blosum62_mean: float
    composition_l2_distance: float


@dataclass
class EmbeddingFeatures:
    """Embedding-derived similarity features for two protein sequences."""

    model_name: str
    dim: int
    cosine_similarity: float
    euclidean_distance: float
    diff_embedding: np.ndarray


# ------------------------ Bgee / IDs ------------------------ #


def get_ortholog_ids_for_species(
    ortholog_df: pd.DataFrame,
    target_species: str,
) -> list[str]:
    """Return all ortholog Ensembl IDs in ``ortholog_df`` for a target species."""
    _, genus, species = _species_to_genus_species(target_species)

    mask = ortholog_df["genus"].str.lower().eq(genus.lower()) & ortholog_df["species"].str.lower().eq(species.lower())
    subset = ortholog_df[mask]

    if subset.empty:
        return []

    # unique gene_ids, converted to str just in case
    gene_ids = subset["gene_id"].dropna().astype(str).unique().tolist()
    return gene_ids


def pick_ortholog_for_species(
    ortholog_df: pd.DataFrame,
    target_species: str,
) -> str | None:
    """Backward-compatible helper that returns only the first ortholog ID."""
    gene_ids = get_ortholog_ids_for_species(ortholog_df, target_species)
    return gene_ids[0] if gene_ids else None


def get_ortholog_table(
    query_ensembl_id: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return Bgee ortholog table for a query Ensembl gene ID via gget.bgee."""
    gget, _, _ = _require_ortholog_deps()
    df = gget.bgee(
        query_ensembl_id,
        type="orthologs",
        json=False,
        verbose=verbose,
    )
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("gget.bgee did not return a pandas DataFrame.")

    expected = {"gene_id", "genus", "species"}
    if not expected.issubset(df.columns):
        raise RuntimeError(f"Unexpected Bgee columns {df.columns}. " f"Expected at least {expected}.")
    return df


# ------------------------ sequences & MUSCLE ------------------------ #


def fetch_aa_sequence(ensembl_id: str) -> str:
    """Fetch amino-acid sequence for a gene using gget.seq(translate=True)."""
    gget, _, _ = _require_ortholog_deps()
    header, seq = gget.seq(ensembl_id, translate=True)
    seq = str(seq)
    if not seq:
        raise RuntimeError(f"Empty sequence returned for {ensembl_id!r}.")
    return seq


def run_muscle_pairwise(
    seq1: str,
    seq2: str,
    *,
    name1: str,
    name2: str,
    super5: bool = False,
) -> tuple[str, str, str]:
    """Align two AA sequences via gget.muscle.

    Args:
        seq1: Amino-acid sequence for the first protein.
        seq2: Amino-acid sequence for the second protein.
        name1: Sequence name for the first protein (used in FASTA headers).
        name2: Sequence name for the second protein (used in FASTA headers).
        super5: Whether to enable MUSCLE super5 mode.

    Returns:
        tuple[str, str, str]: ``(aligned_seq1, aligned_seq2, raw_clustal_text)``.

    Raises:
        RuntimeError: If MUSCLE output cannot be parsed or does not contain the expected sequence names.
    """
    fasta_text = f">{name1}\n{seq1}\n>{name2}\n{seq2}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
        path = fh.name
        fh.write(fasta_text)

    gget, _, _ = _require_ortholog_deps()
    try:
        clustal = gget.muscle(path, super5=super5)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    aligned = parse_clustal_alignment(clustal)
    if name1 not in aligned or name2 not in aligned:
        raise RuntimeError(f"Could not find {name1}/{name2} sequences in MUSCLE output.")

    return aligned[name1], aligned[name2], clustal


def parse_clustal_alignment(clustal_text: str) -> dict[str, str]:
    """Parse a minimal ClustalW/MUSCLE text alignment.

    Args:
        clustal_text: Raw ClustalW/MUSCLE text output.

    Returns:
        Mapping of ``{sequence_name: aligned_sequence}``.
    """
    seqs: dict[str, list[str]] = {}

    for line in clustal_text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("CLUSTAL") or line.startswith("MUSCLE"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        name, frag = parts[0], parts[1]

        # Skip consensus lines of just *:. etc.
        if all(c in "*:." for c in name):
            continue

        seqs.setdefault(name, []).append(frag)

    return {name: "".join(frags) for name, frags in seqs.items()}


# ------------------------ scoring ------------------------ #


def _aa_composition_vector(seq: str) -> np.ndarray:
    seq = seq.replace("-", "")
    length = len(seq) or 1
    return np.array([seq.count(a) / length for a in AA_ALPHABET], dtype=float)


def compute_alignment_scores(
    seq1: str,
    seq2: str,
    aligned1: str,
    aligned2: str,
) -> tuple[AlignmentScores, np.ndarray]:
    """Compute alignment-derived scores and the AA composition-difference vector."""
    if len(aligned1) != len(aligned2):
        raise ValueError("Aligned sequences must have the same length.")

    alignment_length = len(aligned1)
    if alignment_length == 0:
        raise ValueError("Empty alignment.")

    blosum62 = _get_blosum62()

    matches = positives = very_negative = 0
    gaps_q = gaps_t = 0
    gap_open_q = gap_open_t = 0
    prev_gap_q = prev_gap_t = False
    blosum_sum = 0.0

    for a, b in zip(aligned1, aligned2):
        gap_q = a == "-"
        gap_t = b == "-"

        if gap_q:
            gaps_q += 1
            if not prev_gap_q:
                gap_open_q += 1
        if gap_t:
            gaps_t += 1
            if not prev_gap_t:
                gap_open_t += 1

        prev_gap_q = gap_q
        prev_gap_t = gap_t

        if gap_q or gap_t:
            continue

        if a == b:
            matches += 1

        pair = (a, b)
        if pair not in blosum62:
            pair = (b, a)
        score = blosum62.get(pair, 0)
        blosum_sum += score

        if score > 0:
            positives += 1
        if score <= -3:
            very_negative += 1

    identity_fraction = matches / alignment_length
    positive_fraction = positives / alignment_length
    very_negative_fraction = very_negative / alignment_length
    blosum_mean = blosum_sum / alignment_length
    gap_fraction_query = gaps_q / alignment_length
    gap_fraction_target = gaps_t / alignment_length

    cov1 = len(aligned1.replace("-", "")) / (len(seq1) or 1)
    cov2 = len(aligned2.replace("-", "")) / (len(seq2) or 1)

    comp1 = _aa_composition_vector(seq1)
    comp2 = _aa_composition_vector(seq2)
    comp_diff = comp1 - comp2
    comp_l2 = float(np.linalg.norm(comp_diff))

    scores = AlignmentScores(
        alignment_length=alignment_length,
        identity_fraction=float(identity_fraction),
        positive_fraction=float(positive_fraction),
        very_negative_fraction=float(very_negative_fraction),
        gap_fraction_query=float(gap_fraction_query),
        gap_fraction_target=float(gap_fraction_target),
        gap_openings_query=int(gap_open_q),
        gap_openings_target=int(gap_open_t),
        seq1_coverage=float(cov1),
        seq2_coverage=float(cov2),
        blosum62_sum=float(blosum_sum),
        blosum62_mean=float(blosum_mean),
        composition_l2_distance=comp_l2,
    )

    return scores, comp_diff


# ------------------------ embeddings (optional) ------------------------ #

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}


def _load_protein_model(model_name: str, device: str, *, revision: str = "main"):
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers / torch not installed. "
            "Install them or call align_ortholog_pair_with_features(..., "
            "embedding_model_name=None)."
        )
    key = (model_name, device, revision)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, revision=revision)  # noqa: S615
    model = AutoModel.from_pretrained(model_name, revision=revision)  # noqa: S615
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def get_protein_embedding(
    sequence: str,
    *,
    model_name: str,
    device: str = "cpu",
    revision: str = "main",
    max_len: int = 1022,
) -> np.ndarray:
    """Return a pooled protein embedding for a sequence via a transformer model.

    The embedding is obtained by mean pooling the per-residue representations.

    Args:
        sequence: Protein sequence (amino-acid string).
        model_name: Hugging Face model name for the embedding model.
        device: Device for the transformer model (e.g. ``"cpu"``, ``"cuda"``).
        revision: Model revision for ``from_pretrained``.
        max_len: Maximum tokenized sequence length (including special tokens).

    Returns:
        np.ndarray: 1D embedding vector.
    """
    tokenizer, model = _load_protein_model(model_name, device, revision=revision)

    tokens = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        out = model(**tokens)

    hidden = out.last_hidden_state[0]  # [L, D]
    if hidden.size(0) > 2:
        residue_embs = hidden[1:-1]
    else:
        residue_embs = hidden

    emb = residue_embs.mean(dim=0)
    return emb.cpu().numpy()


def compute_embedding_features(
    seq1: str,
    seq2: str,
    *,
    model_name: str,
    device: str = "cpu",
    revision: str = "main",
) -> EmbeddingFeatures:
    """Compute embedding-based similarity features for two sequences.

    Args:
        seq1: First protein sequence.
        seq2: Second protein sequence.
        model_name: Hugging Face model name for embeddings.
        device: Device for the transformer model (e.g. ``"cpu"``, ``"cuda"``).
        revision: Model revision for ``from_pretrained``.

    Returns:
        EmbeddingFeatures: Scalar similarities plus the difference vector.
    """
    emb1 = get_protein_embedding(seq1, model_name=model_name, device=device, revision=revision)
    emb2 = get_protein_embedding(seq2, model_name=model_name, device=device, revision=revision)

    diff = emb1 - emb2
    dot = float(np.dot(emb1, emb2))
    n1 = float(np.linalg.norm(emb1) or 1.0)
    n2 = float(np.linalg.norm(emb2) or 1.0)
    cosine = dot / (n1 * n2)
    euclid = float(np.linalg.norm(diff))

    return EmbeddingFeatures(
        model_name=model_name,
        dim=int(diff.shape[0]),
        cosine_similarity=cosine,
        euclidean_distance=euclid,
        diff_embedding=diff,
    )


# ------------------------ main entrypoint (string species only) ------------------------ #


def align_ortholog_pair_with_features(
    query_ensembl_id: str,
    target_species: str,
    *,
    use_super5: bool = False,
    embedding_model_name: str | None = "facebook/esm2_t12_35M_UR50D",
    embedding_device: str = "cpu",
    embedding_revision: str = "main",
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute ortholog alignment features for a query gene and target species.

    Args:
        query_ensembl_id: Ensembl gene ID in the query organism.
        target_species: Target species alias, interpreted via :py:data:`_SPECIES_ALIASES`.
        use_super5: Passed through to ``gget.muscle(super5=...)``.
        embedding_model_name: Hugging Face model name for embeddings. Set to ``None`` to disable embeddings.
        embedding_device: Device for the transformer model (e.g. ``"cpu"``, ``"cuda"``).
        embedding_revision: Model revision for ``from_pretrained`` (pin to a commit hash for reproducibility).
        verbose: Print progress/warnings.

    Returns:
        Mapping from target ortholog Ensembl IDs to feature dictionaries.

    Raises:
        ValueError: If no orthologs are available for the query/target species pair.
    """
    # 1) Get the ortholog table from Bgee
    ortho_df = get_ortholog_table(query_ensembl_id, verbose=verbose)

    # 2) Resolve target species and get all its ortholog IDs
    #    This will raise a clear error if the species is unknown.
    canonical_target, genus, species = _species_to_genus_species(target_species)
    target_gene_ids = get_ortholog_ids_for_species(ortho_df, target_species)

    if len(target_gene_ids) == 0:
        raise ValueError(
            f"No orthologs found in Bgee for query gene {query_ensembl_id!r} and target species {target_species!r} "
            f"(canonical: {canonical_target!r}, genus: {genus!r}, species: {species!r})."
        )

    if verbose:
        print(
            f"[INFO] Found {len(target_gene_ids)} ortholog(s) for "
            f"{query_ensembl_id!r} in target species {target_species!r} "
            f"(canonical={canonical_target!r}, genus={genus!r}, species={species!r})."
        )

    # 3) Fetch query AA sequence once
    query_seq_aa = fetch_aa_sequence(query_ensembl_id)

    # 4) For each ortholog gene_id in the target species, compute features
    results: dict[str, dict[str, Any]] = {}

    for target_gene_id in target_gene_ids:
        if verbose:
            print(f"[INFO] Aligning {query_ensembl_id!r} -> {target_gene_id!r} ({target_species!r})")

        try:
            target_seq_aa = fetch_aa_sequence(target_gene_id)
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not fetch sequence for {target_gene_id}: {e}")
            continue

        # MUSCLE alignment
        try:
            aligned_q, aligned_t, clustal = run_muscle_pairwise(
                query_seq_aa,
                target_seq_aa,
                name1=query_ensembl_id,
                name2=target_gene_id,
                super5=use_super5,
            )
        except Exception as e:
            if verbose:
                print(f"[WARN] MUSCLE failed for {target_gene_id}: {e}")
            continue

        # Alignment-based scores
        try:
            scores, comp_diff = compute_alignment_scores(
                query_seq_aa,
                target_seq_aa,
                aligned_q,
                aligned_t,
            )
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to compute alignment scores for {target_gene_id}: {e}")
            continue

        # Embedding-based features (optional)
        if embedding_model_name is not None:
            try:
                emb_feats = compute_embedding_features(
                    query_seq_aa,
                    target_seq_aa,
                    model_name=embedding_model_name,
                    device=embedding_device,
                    revision=embedding_revision,
                )
                diff_vec = emb_feats.diff_embedding.astype(float)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to compute embeddings for {target_gene_id}: {e}")
                emb_feats = None
                diff_vec = None
        else:
            emb_feats = None
            diff_vec = None

        # Build feature dict for this ortholog
        feat: dict[str, Any] = {
            "query_id": query_ensembl_id,
            "target_id": target_gene_id,
            "target_species": target_species,
            "target_species_canonical": canonical_target,
            "target_species_genus": genus,
            "target_species_species": species,
            "alignment_aligned_query": aligned_q,
            "alignment_aligned_target": aligned_t,
            "alignment_raw_clustal": clustal,
        }

        # Scalar alignment scores
        for k, v in asdict(scores).items():
            feat[f"score_{k}"] = float(v)

        # 20-D AA composition difference (fixed size)
        comp_diff = comp_diff.astype(float)
        feat["composition_diff_vector"] = comp_diff.tolist()
        for i, aa in enumerate(AA_ALPHABET):
            feat[f"compdiff_{aa}"] = float(comp_diff[i])

        # Embedding features (fixed embedding dim for a given model)
        if emb_feats is not None and diff_vec is not None:
            feat["emb_model"] = emb_feats.model_name
            feat["emb_dim"] = int(emb_feats.dim)
            feat["emb_cosine"] = float(emb_feats.cosine_similarity)
            feat["emb_euclid"] = float(emb_feats.euclidean_distance)
            feat["emb_diff"] = diff_vec.tolist()
        else:
            feat["emb_model"] = None
            feat["emb_dim"] = None
            feat["emb_cosine"] = None
            feat["emb_euclid"] = None
            feat["emb_diff"] = None

        # Use the ortholog Ensembl ID as the key in the outer dict
        results[target_gene_id] = feat

    return results
