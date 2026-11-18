"""Building graphs.

Ensembl access (MySQL and HTTPS dumps, REST), the table store, the derived tables, the externals configuration and the
graph builder. This is the only part of idtrack that needs pandas, h5py, PyMySQL and PyYAML.

The names are imported on first access.
"""

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "DatabaseManager": "idtrack.build.database_manager",
    "ExternalDatabases": "idtrack.build.externals",
    "GraphMaker": "idtrack.build.builder",
    "VerifyOrganism": "idtrack.build.species",
}

__all__: list[str] = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import a public name on first access (PEP 562).

    Args:
        name: The name.

    Returns:
        Any: The object.

    Raises:
        AttributeError: For a name the package does not export.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
