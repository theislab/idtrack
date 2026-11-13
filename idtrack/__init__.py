#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Top-level package of **idtrack**: the public classes, the errors a caller may want to catch, and the metadata.

The names are imported on first access (``idtrack.API`` imports the conversion core only; ``idtrack.HarmonizeFeatures``
also imports anndata and scipy), so that ``import idtrack`` itself is cheap.
"""

import importlib
from importlib import metadata as _metadata
from typing import Any

__author__: str = "Kemal Inecik"
__email__: str = "k.inecik@gmail.com"

# Prefer the installed package version, but fall back to the source value.
try:  # pragma: no cover
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # Local/editable install
    __version__ = "0.0.5"

_EXPORTS: dict[str, str] = {
    "API": "idtrack._api",
    "ConnectionBridge": "idtrack._connection_bridge",
    "DatabaseManager": "idtrack._database_manager",
    "DB": "idtrack._db",
    "EnsemblDownloadError": "idtrack._ensembl_access",
    "EnsemblRestError": "idtrack._db",
    "ExternalDatabases": "idtrack._external_databases",
    "GraphMaker": "idtrack._graph_maker",
    "GraphMismatchError": "idtrack._the_graph",
    "HarmonizeFeatures": "idtrack._harmonize_features",
    "HistoryValidationError": "idtrack._graph_maker",
    "InactiveAtReleaseError": "idtrack._track",
    "SearchBudgetExceededError": "idtrack._track",
    "TheGraph": "idtrack._the_graph",
    "Track": "idtrack._track",
    "TrackTests": "idtrack._track_tests",
    "UnknownVersionError": "idtrack._track",
    "VerifyOrganism": "idtrack._verify_organism",
}

__all__: list[str] = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import a public name on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the module's names, including the public names not imported yet."""
    return sorted(set(globals()) | set(_EXPORTS))
