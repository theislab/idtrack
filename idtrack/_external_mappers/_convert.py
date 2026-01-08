#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

import logging
import typing

import pandas as pd

from idtrack._external_mappers._backend_gget import map_with_gget
from idtrack._external_mappers._backend_gprofiler import map_with_gprofiler
from idtrack._external_mappers._backend_mygene import map_with_mygene
from idtrack._external_mappers._backend_pybiomart import map_with_pybiomart
from idtrack._external_mappers._constants import SUPPORTED_METHODS
from idtrack._external_mappers._utils import _empty_result, canonical_db, logger

_VERBOSE_LEVELS = {
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG,
}

_VERBOSE_NAMES = {
    "warn": 1,
    "warning": 1,
    "error": 1,
    "info": 2,
    "debug": 3,
}


def _normalize_verbose_level(value: int | str | bool) -> int:
    if isinstance(value, bool):
        return 3 if value else 2
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _VERBOSE_NAMES:
            raise ValueError(f"Unknown verbose level {value!r}. Use 1, 2, 3 or 'error', 'warning', 'info', 'debug'.")
        return _VERBOSE_NAMES[key]
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unknown verbose level {value!r}. Use 1, 2, 3 or 'error', 'warning', 'info', 'debug'."
        ) from exc
    if level not in _VERBOSE_LEVELS:
        raise ValueError(f"Unknown verbose level {value!r}. Use 1, 2, 3 or 'error', 'warning', 'info', 'debug'.")
    return level


def convert_ids(
    ids: typing.Iterable[str],
    input_db: str,
    output_db: str,
    method: str,
    species: str,
    drop_metadata_json_column: bool = True,
    chunk_size: int = 1000,
    pause: float = 0.2,
    max_retries: int = 3,
    strip_versions: bool = True,
    release_for_pybiomart: str | int | None = None,
    strict_input_db_gprofiler: bool = True,
    suppress_method_verbosity: bool = True,
    verbose: int | str | bool = 2,
) -> pd.DataFrame:
    """Convert identifiers using an external mapper backend.

    Args:
        ids: Input identifiers to map.
        input_db: Source database type.
        output_db: Target database type.
        method: Backend method name (one of :py:data:`~idtrack._external_mappers._constants.SUPPORTED_METHODS`).
        species: Species code (e.g. ``"hsapiens"``).
        drop_metadata_json_column: If ``True``, drop the ``metadata_json`` column from the returned DataFrame.
        chunk_size: Number of IDs per API request.
        pause: Pause in seconds between requests.
        max_retries: Maximum retry attempts per chunk on failure (for backends that support it).
        strip_versions: Strip version suffixes from Ensembl/RefSeq IDs.
        release_for_pybiomart: Ensembl release/key for the pybiomart backend. Must be ``None`` unless
            ``method="pybiomart"``.
        strict_input_db_gprofiler: If ``True``, enforce strict input-db filtering in the gprofiler backend.
        suppress_method_verbosity: Suppress stdout/stderr from the underlying backend library.
        verbose: Verbosity level (``1``/``2``/``3``) or string alias (``"error"``, ``"warning"``, ``"info"``,
            ``"debug"``).

    Returns:
        pd.DataFrame: Standardized mapping DataFrame.

    Raises:
        ValueError: If ``method``/``verbose`` is invalid, or if ``release_for_pybiomart`` is used with a
            non-pybiomart backend.
    """
    verbose_level = _normalize_verbose_level(verbose)
    logger.setLevel(_VERBOSE_LEVELS[verbose_level])
    show_progress = verbose_level >= 2

    id_list = [str(x) for x in ids]
    if not id_list:
        return _empty_result()

    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")

    method_key = method.strip().lower()

    if method_key not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}, got {method!r}")

    if release_for_pybiomart is not None and method_key != "pybiomart":
        raise ValueError("release parameter can only be used with method='pybiomart'")

    logger.debug(f"convert_ids: using backend {method_key!r} for {inp}->{outp}")

    common_kwargs = {
        "species": species,
        "chunk_size": chunk_size,
        "pause": pause,
        "strip_versions": strip_versions,
        "show_progress": show_progress,
        "suppress_method_verbosity": suppress_method_verbosity,
    }

    backend_configs: dict[str, tuple[typing.Callable[..., pd.DataFrame], dict[str, typing.Any]]] = {
        "pybiomart": (
            map_with_pybiomart,
            {**common_kwargs, "release": release_for_pybiomart},
        ),
        "mygene": (
            map_with_mygene,
            {**common_kwargs, "max_retries": max_retries},
        ),
        "gprofiler": (
            map_with_gprofiler,
            {**common_kwargs, "max_retries": max_retries, "strict_input_db": strict_input_db_gprofiler},
        ),
        "gget": (
            map_with_gget,
            {**common_kwargs, "max_retries": max_retries},
        ),
    }

    func, backend_kwargs = backend_configs[method_key]

    df = func(id_list, inp, outp, **backend_kwargs)

    if drop_metadata_json_column and "metadata_json" in df.columns:
        del df["metadata_json"]

    return df.reset_index(drop=True)
