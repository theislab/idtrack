#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Top-level package initialisation for **idtrack**.

This module only exposes the public API (re-exporting key classes) and
stores package metadata.  No runtime behaviour has been changed.
"""

from __future__ import annotations

from importlib import metadata as _metadata

__author__: str = "Kemal Inecik"
__email__: str = "k.inecik@gmail.com"

# Prefer the installed package version, but fall back to the source value.
try:  # pragma: no cover
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # Local/editable install
    __version__ = "0.0.4"


# Public re-exports
from idtrack._api import API as API  # noqa: F401
from idtrack._database_manager import DatabaseManager as DatabaseManager  # noqa: F401
from idtrack._db import DB as DB  # noqa: F401
from idtrack._external_databases import ExternalDatabases as ExternalDatabases  # noqa: F401
from idtrack._graph_maker import GraphMaker as GraphMaker  # noqa: F401
from idtrack._harmonize_features import HarmonizeFeatures as HarmonizeFeatures  # noqa: F401
from idtrack._the_graph import TheGraph as TheGraph  # noqa: F401
from idtrack._track import Track as Track  # noqa: F401
from idtrack._track_tests import TrackTests as TrackTests  # noqa: F401
from idtrack._verify_organism import VerifyOrganism as VerifyOrganism  # noqa: F401

__all__: list[str] = [
    "API",
    "DatabaseManager",
    "DB",
    "ExternalDatabases",
    "HarmonizeFeatures",
    "TheGraph",
    "GraphMaker",
    "Track",
    "VerifyOrganism",
    "TrackTests",
]
