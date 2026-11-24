"""idtrack: tracks gene identifiers through Ensembl releases, genome assemblies and external databases.

A graph of one organism (:py:class:`idtrack.TheGraph`) holds Ensembl's gene, transcript and translation ids of the
releases it covers, Ensembl's history between the gene ids and the identifiers of external databases that Ensembl
attaches to them. :py:class:`idtrack.API` opens a graph file (:py:meth:`idtrack.API.load`) or builds one from
Ensembl (:py:meth:`idtrack.API.build`) and answers on it: :py:meth:`idtrack.API.convert` and
:py:meth:`idtrack.API.convert_many` convert identifiers from one release and database to another and return the
typed results of :py:mod:`idtrack.results`.

The names of the package, each ``idtrack.<name>``:

- the entry point: :py:class:`~idtrack.API`;
- the results: :py:class:`~idtrack.ConversionResult` (with :py:class:`~idtrack.Candidate`,
  :py:class:`~idtrack.OtherReading`, :py:class:`~idtrack.ErrorInfo`, :py:class:`~idtrack.ExplainedPath`,
  :py:class:`~idtrack.PathNode` and :py:class:`~idtrack.PathStep`), :py:class:`~idtrack.NameMatch`,
  :py:class:`~idtrack.SourceInference`, :py:class:`~idtrack.GraphDescription` and
  :py:class:`~idtrack.AssemblyDescription`;
- the settings: :py:class:`~idtrack.SearchSettings` (how a search runs) and :py:class:`~idtrack.GraphSettings`
  (the graph's caches, and when an external id is too common to identify a gene);
- the graph, its self-tests and building: :py:class:`~idtrack.TheGraph`, :py:class:`~idtrack.GraphChecks` and
  :py:class:`~idtrack.CheckResult`, :py:class:`~idtrack.GraphMaker`, :py:class:`~idtrack.DatabaseManager` and
  :py:class:`~idtrack.ExternalDatabases`;
- the errors: :py:class:`~idtrack.IdtrackError` and the classes derived from it (:py:mod:`idtrack.errors`);
- :py:class:`~idtrack.HarmonizeFeatures`, which converts the features of several AnnData files to one list of genes
  (it needs the ``harmonize`` extra).

The names are imported on first access (``idtrack.API`` imports the query side only: the graph, the engine
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
    "OtherReading": "idtrack.results",
    "SourceInference": "idtrack.results",
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
# plain install (the extra is optional).
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
