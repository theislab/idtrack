"""The errors of idtrack, in one hierarchy rooted in :py:class:`IdtrackError`.

A caller (a script, a notebook, a web server) can tell from the class alone whose fault an error is:

- :py:class:`InputError`: the caller asked for something the package or the loaded graph does not have, or passed an
  argument of the wrong kind. The subclasses name the argument: an unknown organism, release, assembly, database or
  identifier, a missing value, an Ensembl id at a release where it is not active or with a version that does not
  exist, an argument of the wrong type or value. Each subclass also derives from the built-in error that it replaces
  (``ValueError``, or ``TypeError`` for a value of the wrong type), so ``except ValueError`` keeps working.
- :py:class:`GraphFileError`: a graph file is missing, unreadable, of another format, or holds another graph than the
  one asked for (:py:class:`GraphMismatchError`).
- :py:class:`SearchBudgetExceededError`: one conversion needed more work than its budget; an outcome of that one
  identifier, which a batch records and goes on.
- :py:class:`BuildError`: building tables or a graph failed: Ensembl could not be reached or answered something
  unusable (:py:class:`EnsemblAccessError`, also a ``ConnectionError``), a table or a schema is not what the builder
  needs, the history does not validate, the externals configuration does not cover a release.
- :py:class:`IntegrityError`: the package or a graph contradicts itself; a bug or a damaged file, never the caller's
  input.

This module imports nothing of idtrack.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class IdtrackError(Exception):
    """The root of every error idtrack raises on purpose."""


# ---------------------------------------------------------------------------------------------------------------------
# The caller's input


class InputError(IdtrackError):
    """The caller passed a value the package or the loaded graph cannot use.

    Attributes:
        argument (str | None): The name of the argument, where one is concerned.
        value (Any): The value given.
        allowed (tuple | None): The accepted values, where they are few enough to list.
    """

    def __init__(
        self, message: str, *, argument: str | None = None, value: Any = None, allowed: Sequence | None = None
    ):
        """Keep the message and what it is about.

        Args:
            message: The text of the error.
            argument: The name of the argument, if one is concerned.
            value: The value given.
            allowed: The accepted values, if they can be listed.
        """
        super().__init__(message)
        self.message = message
        self.argument = argument
        self.value = value
        self.allowed = None if allowed is None else tuple(allowed)

    def __str__(self) -> str:
        """Return the message alone.

        Returns:
            str: The message.
        """
        return self.message

    def __reduce__(self):
        """Rebuild the error with its attributes (pickling, copying)."""
        return (type(self), (self.message,), dict(self.__dict__))


class ArgumentTypeError(InputError, TypeError):
    """An argument is not of the type the call takes (a number for a string, a float for a release)."""


class InvalidArgumentError(InputError, ValueError):
    """An argument has a value the call does not take (a strategy, a mode, a budget that is not positive)."""


class MissingValueError(InputError, ValueError):
    """A value that stands for nothing was given where an identifier is needed (an empty string)."""


class UnknownOrganismError(InputError, ValueError):
    """The organism is not in idtrack's organism table (:py:mod:`idtrack.organisms`)."""


class UnknownReleaseError(InputError, ValueError):
    """The Ensembl release is not one of the loaded graph (or of the organism)."""


class UnknownAssemblyError(InputError, ValueError):
    """The genome assembly is not one of the organism or of the loaded graph."""


class UnknownDatabaseError(InputError, ValueError):
    """The database is not one of the loaded graph (at the release asked about)."""


class UnknownIdentifierError(InputError, ValueError):
    """The identifier is not one the call can work on (not in the graph, or not of the kind the call needs)."""


class InactiveAtReleaseError(InputError, ValueError):
    """The identifier is not active at the release the caller gave as ``from_release``.

    Attributes:
        identifier (str): The graph node of the query.
        release (int): The release that was given.
        active_ranges (list[list[int]]): Inclusive ``[first, last]`` release ranges in which the identifier is active.
    """

    def __init__(self, identifier: str, release: int, active_ranges: list[list[int]]):  # noqa: B042 - __reduce__
        """Keep the query, the release and the releases at which the identifier is active.

        Args:
            identifier: The graph node of the query.
            release: The release that was given.
            active_ranges: Inclusive release ranges in which the identifier is active.
        """
        ranges = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in active_ranges) or "none"
        super().__init__(
            f"{identifier!r} is not active at Ensembl release {release}; it is active at the releases: "
            f"{ranges}. Give one of these as `from_release`, or leave it out to start from the target release when "
            "the identifier is active there and from its latest active release otherwise.",
            argument="from_release",
            value=release,
        )
        self.identifier = identifier
        self.release = release
        self.active_ranges = active_ranges

    def __reduce__(self):
        """Rebuild the error from its own arguments (pickling, copying)."""
        return (type(self), (self.identifier, self.release, self.active_ranges))


class UnknownVersionError(InputError, ValueError):
    """The identifier is an Ensembl id, or a sequence accession, with a version that the graph does not hold.

    The version never existed, or the graph cannot know it (a version lower than the latest, or of an id that is no
    longer active; of an accession, a version below the latest that Ensembl's cross references never attached). It is
    not answered with another version's gene.

    Attributes:
        identifier (str): The identifier as given.
        versions (tuple[str, ...]): The ``<id>.<version>`` ids (or ``<accession>.<version>``) that the graph holds, by
            ascending version.
    """

    def __init__(self, identifier: str, versions: tuple[str, ...]):  # noqa: B042 - __reduce__
        """Keep the query and the versions that exist.

        Args:
            identifier: The identifier as given.
            versions: The versions of the id that the graph holds.
        """
        super().__init__(
            f"{identifier!r} names a version that this graph does not hold; the versions that exist are: "
            f"{', '.join(versions)}. Give one of them, or the id without version.",
            argument="identifier",
            value=identifier,
            allowed=versions,
        )
        self.identifier = identifier
        self.versions = tuple(versions)

    def __reduce__(self):
        """Rebuild the error from its own arguments (pickling, copying)."""
        return (type(self), (self.identifier, self.versions))


# ---------------------------------------------------------------------------------------------------------------------
# Graph files


class GraphFileError(IdtrackError):
    """A graph file is missing, cannot be read, or is of a format this version of idtrack does not read.

    Attributes:
        path (str): The file.
    """

    def __init__(self, message: str, path: str):
        """Keep the message and the file.

        Args:
            message: The text of the error.
            path: The graph file.
        """
        super().__init__(message, path)
        self.message = message
        self.path = path

    def __str__(self) -> str:
        """Return the message alone.

        Returns:
            str: The message.
        """
        return self.message


class GraphMismatchError(GraphFileError):
    """A graph file holds another graph than the one asked for (organism, window, selection, format, ...)."""


# ---------------------------------------------------------------------------------------------------------------------
# The search


class SearchBudgetExceededError(IdtrackError):
    """A conversion needed more search states or more time than its budget allows.

    Attributes:
        identifier (str): The graph node of the query.
        states (int): Number of search states expanded when the search stopped.
        seconds (float): Seconds spent when the search stopped.
        budget_states (int): The budget of states of the call.
        budget_seconds (float): The budget of seconds of the call.
    """

    def __init__(self, identifier: str, states: int, seconds: float, budget_states: int, budget_seconds: float):
        """Keep the work done and the budget.

        Args:
            identifier: The graph node of the query.
            states: Number of search states expanded when the search stopped.
            seconds: Seconds spent when the search stopped.
            budget_states: The budget of states of the call.
            budget_seconds: The budget of seconds of the call.
        """
        super().__init__(identifier, states, seconds, budget_states, budget_seconds)
        self.identifier = identifier
        self.states = states
        self.seconds = seconds
        self.budget_states = budget_states
        self.budget_seconds = budget_seconds

    def __str__(self) -> str:
        """Say how far the search got and what its budget was.

        Returns:
            str: The message of the error.
        """
        return (
            f"The conversion of {self.identifier!r} was stopped after {self.states} search states and "
            f"{self.seconds:.1f} s (budget: {self.budget_states} states, {self.budget_seconds:.1f} s). Give the call "
            "a larger `budget_states` or `budget_seconds` (or a `SearchSettings` with them) to let it run longer."
        )


# ---------------------------------------------------------------------------------------------------------------------
# Building tables and graphs


class BuildError(IdtrackError):
    """Building the tables or the graph of an organism failed."""


class EnsemblAccessError(BuildError, ConnectionError):
    """Ensembl could not be reached, or did not give a usable answer after every attempt."""


class EnsemblRestError(EnsemblAccessError):
    """Ensembl REST could not be reached or did not give a usable answer after every attempt.

    The message names the URL, the attempts, the timeouts in force and how to change them
    (:py:func:`idtrack.settings.configure_ensembl`).
    """


class EnsemblDownloadError(EnsemblAccessError):
    """A file of Ensembl's HTTPS dumps could not be downloaded after every attempt.

    A file that Ensembl does not have (HTTP 404 or 410) is not this error: the caller is told that it is absent.
    """


class EnsemblSchemaError(BuildError, LookupError):
    """An Ensembl core database has no such table, or the table lacks requested columns.

    It states a fact about the schema of one release (older releases keep stable ids in ``<form>_stable_id``
    tables, and some columns were renamed), so callers may fall back to the older layout when they catch it. A
    failed download, a connection error or a failure to cache a table never raises it; those keep their own
    exception, so that a fallback cannot hide them.

    Attributes:
        table (str): Name of the table.
        missing_columns (tuple[str, ...]): Requested columns the table lacks; empty when the table itself is absent.
    """

    def __init__(self, message: str, table: str, missing_columns: tuple = ()):
        """Record the table and the missing columns next to the message.

        Args:
            message: The text of the error.
            table: Name of the table.
            missing_columns: Requested columns the table lacks.
        """
        # All arguments go to the base class, so that pickling and copying rebuild the same exception.
        super().__init__(message, table, tuple(missing_columns))
        self.message = message
        self.table = table
        self.missing_columns = tuple(missing_columns)

    def __str__(self) -> str:
        """Return the message alone (the base class would show all arguments).

        Returns:
            str: The message.
        """
        return self.message


class TableFormatError(BuildError, ValueError):
    """A stored table cannot be read by this version of idtrack, or a table cannot be stored."""


class HistoryValidationError(BuildError, ValueError):
    """Ensembl's id history of an organism does not validate (a session that no release pair explains, ...)."""


class ExternalsError(BuildError, ValueError):
    """An externals configuration file (the selection of external databases) is malformed."""


class ExternalsReleaseNotCoveredError(ExternalsError):
    """The externals configuration file does not list a release that a graph needs."""


# ---------------------------------------------------------------------------------------------------------------------
# Integrity


class IntegrityError(IdtrackError):
    """The package or a graph contradicts itself: a bug or a damaged file, never the caller's input."""


__all__ = [
    "ArgumentTypeError",
    "BuildError",
    "EnsemblAccessError",
    "EnsemblDownloadError",
    "EnsemblRestError",
    "EnsemblSchemaError",
    "ExternalsError",
    "ExternalsReleaseNotCoveredError",
    "GraphFileError",
    "GraphMismatchError",
    "HistoryValidationError",
    "IdtrackError",
    "InactiveAtReleaseError",
    "InputError",
    "IntegrityError",
    "InvalidArgumentError",
    "MissingValueError",
    "SearchBudgetExceededError",
    "TableFormatError",
    "UnknownAssemblyError",
    "UnknownDatabaseError",
    "UnknownIdentifierError",
    "UnknownOrganismError",
    "UnknownReleaseError",
    "UnknownVersionError",
]
