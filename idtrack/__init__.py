"""idtrack: tracks gene identifiers through Ensembl releases, genome assemblies and external databases.

The names below are imported on first access (``idtrack.API`` imports the query side only: the graph, the engine
and the results; building a graph imports the build stack; ``idtrack.HarmonizeFeatures`` needs the ``harmonize``
extra), so that ``import idtrack`` itself is cheap. ``__all__``, and so ``from idtrack import *``, holds the names
that every install has; ``HarmonizeFeatures`` is imported by its name only.

Modules: :py:mod:`idtrack.api` (the API), :py:mod:`idtrack.results` (typed results), :py:mod:`idtrack.errors`
(the error hierarchy), :py:mod:`idtrack.settings`, :py:mod:`idtrack.organisms` (the organism table),
:py:mod:`idtrack.ids` (the graph's vocabulary), :py:mod:`idtrack.graph` (the graph and its file),
:py:mod:`idtrack.engine` (the search and its ranking), :py:mod:`idtrack.checks` (the self-tests of a graph),
:py:mod:`idtrack.build` (Ensembl access, tables and the graph builder) and :py:mod:`idtrack.harmonize`.
"""

import importlib
from typing import Any

from idtrack._version import package_version

__version__: str = package_version()

_EXPORTS: dict[str, str] = {
    "API": "idtrack.api",
    # results
    "AssemblyDescription": "idtrack.results",
    "Candidate": "idtrack.results",
    "ConversionResult": "idtrack.results",
    "ErrorInfo": "idtrack.results",
    "ExplainedPath": "idtrack.results",
    "GraphDescription": "idtrack.results",
    "NameMatch": "idtrack.results",
    "PathNode": "idtrack.results",
    "PathStep": "idtrack.results",
    # settings
    "GraphSettings": "idtrack.settings",
    "SearchSettings": "idtrack.settings",
    # the graph, the self-tests, building
    "TheGraph": "idtrack.graph",
    "CheckResult": "idtrack.checks",
    "GraphChecks": "idtrack.checks",
    "DatabaseManager": "idtrack.build",
    "ExternalDatabases": "idtrack.build",
    "GraphMaker": "idtrack.build",
    # errors
    "ArgumentTypeError": "idtrack.errors",
    "BuildError": "idtrack.errors",
    "EnsemblAccessError": "idtrack.errors",
    "EnsemblDownloadError": "idtrack.errors",
    "EnsemblRestError": "idtrack.errors",
    "EnsemblSchemaError": "idtrack.errors",
    "ExternalsError": "idtrack.errors",
    "ExternalsReleaseNotCoveredError": "idtrack.errors",
    "GraphFileError": "idtrack.errors",
    "GraphMismatchError": "idtrack.errors",
    "HistoryValidationError": "idtrack.errors",
    "IdtrackError": "idtrack.errors",
    "InactiveAtReleaseError": "idtrack.errors",
    "InputError": "idtrack.errors",
    "IntegrityError": "idtrack.errors",
    "InvalidArgumentError": "idtrack.errors",
    "MissingValueError": "idtrack.errors",
    "SearchBudgetExceededError": "idtrack.errors",
    "TableFormatError": "idtrack.errors",
    "UnknownAssemblyError": "idtrack.errors",
    "UnknownDatabaseError": "idtrack.errors",
    "UnknownIdentifierError": "idtrack.errors",
    "UnknownOrganismError": "idtrack.errors",
    "UnknownReleaseError": "idtrack.errors",
    "UnknownVersionError": "idtrack.errors",
}

# Names that need the `harmonize` extra: reachable by name, left out of `__all__` so that a star import works on the
# plain install (decision 48: the extra is optional).
_EXTRA_EXPORTS: dict[str, str] = {"HarmonizeFeatures": "idtrack.harmonize"}

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
    module = _EXPORTS.get(name) or _EXTRA_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the module's names, including the public names not imported yet.

    Returns:
        list[str]: The names.
    """
    return sorted(set(globals()) | set(_EXPORTS) | set(_EXTRA_EXPORTS))
