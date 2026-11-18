"""Settings of idtrack: how Ensembl is reached (for building), how a search is bounded, and how a graph is cached.

Three kinds, each one object:

- :py:class:`SearchSettings` (frozen): the budget of one conversion and the rules of bridges. An
  :py:class:`idtrack.API` holds one (``API.load(path, search=...)``), and every conversion call can override the
  budget for itself (``budget_states``, ``budget_seconds``). Nothing of it is process-wide.
- :py:class:`GraphSettings` (frozen): the hyper-connectivity threshold and the bounds of the per-node caches of one
  loaded graph (``API.load(path, graph_settings=...)``, :py:meth:`idtrack.graph.TheGraph.load`).
- :py:data:`ENSEMBL` (:py:class:`EnsemblSettings`): hosts, timeouts, retries and the access mode of the build stack
  (MySQL or Ensembl's HTTPS dumps, and Ensembl REST). Network settings are a property of the process: set them once
  at start with :py:func:`configure_ensembl`, which checks every value. A loaded graph never uses the network.

This module imports only :py:mod:`idtrack.errors`.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from idtrack.errors import ArgumentTypeError, InvalidArgumentError

#: External databases whose ids name a clone or a region rather than a gene; never a bridge between two genes.
NON_BRIDGING_DATABASES: frozenset[str] = frozenset(
    {
        "Clone_based_ensembl_gene",
        "Clone_based_vega_gene",
        "Clone_based_ensembl_transcript",
        "Clone_based_vega_transcript",
    }
)


def _positive_number(value: Any, name: str, integer: bool = False) -> None:
    """Refuse anything but a positive (integer) number, naming the setting."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral if integer else numbers.Real):
        raise ArgumentTypeError(
            f"`{name}` must be a positive {'integer' if integer else 'number'}, got {value!r}.",
            argument=name,
            value=value,
        )
    # `cast`: a `numbers.Real` compares with numbers at run time, which the type of `numbers.Real` does not say.
    if not cast(float, value) > 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise InvalidArgumentError(
            f"`{name}` must be a positive finite number, got {value!r}.", argument=name, value=value
        )


@dataclass(frozen=True)
class SearchSettings:
    """How one conversion searches: its budget and the rules of bridges through external ids.

    Attributes:
        budget_states (int): Largest number of search states one conversion may expand before
            :py:class:`idtrack.errors.SearchBudgetExceededError` is raised.
        budget_seconds (float): Longest time one conversion may take before the same error is raised.
        bridge_limit (int): Most bridges through external ids (or shared transcripts) one path may take.
        non_bridging_databases (frozenset[str]): External databases whose ids are names of a clone or region rather
            than of a gene; they are never used as a bridge between two genes.
    """

    budget_states: int = 200_000
    budget_seconds: float = 30.0
    bridge_limit: int = 2
    non_bridging_databases: frozenset[str] = NON_BRIDGING_DATABASES

    def __post_init__(self) -> None:
        """Check every value."""
        _positive_number(self.budget_states, "budget_states", integer=True)
        _positive_number(self.budget_seconds, "budget_seconds")
        if isinstance(self.bridge_limit, bool) or not isinstance(self.bridge_limit, numbers.Integral):
            raise ArgumentTypeError(
                f"`bridge_limit` must be an integer, got {self.bridge_limit!r}.", argument="bridge_limit"
            )
        if self.bridge_limit < 0:
            raise InvalidArgumentError(
                f"`bridge_limit` must be 0 or more, got {self.bridge_limit!r}.", argument="bridge_limit"
            )
        if isinstance(self.non_bridging_databases, str) or not all(
            isinstance(d, str) for d in self.non_bridging_databases
        ):
            raise ArgumentTypeError("`non_bridging_databases` must be a set of database names.")
        object.__setattr__(self, "non_bridging_databases", frozenset(self.non_bridging_databases))

    def replace(self, **changes: Any) -> SearchSettings:
        """Return a copy with some values changed (checked like the original).

        Args:
            changes: Values of the fields to change.

        Returns:
            SearchSettings: The new settings.

        Raises:
            InvalidArgumentError: For a name that is not a field of the settings.
        """
        names = [f.name for f in dataclasses.fields(self)]
        unknown = sorted(set(changes) - set(names))
        if unknown:
            raise InvalidArgumentError(
                f"SearchSettings has no setting {unknown[0]!r}; the settings are {names}.",
                argument=unknown[0],
                allowed=names,
            )
        return dataclasses.replace(self, **changes)


_CACHE_KINDS = ("adjacency", "node", "edges", "summary", "trios", "names")


@dataclass(frozen=True)
class GraphSettings:
    """How a loaded graph caches what it computes per node, and when an external id is too common to identify a gene.

    Attributes:
        hyperconnective_threshold (int): An external identifier linked to more genes than this in one release is
            hyper-connective (a family name, not an identity): it is never a bridge and is not counted as annotation
            (:py:attr:`idtrack.graph.TheGraph.hyperconnective_nodes`).
        cache_sizes (Mapping[str, int]): Upper bounds of the per-node caches (entries), per kind: ``adjacency``,
            ``node``, ``edges``, ``summary``, ``trios``, ``names``.
    """

    hyperconnective_threshold: int = 5
    cache_sizes: Mapping[str, int] = field(
        default_factory=lambda: {
            "adjacency": 200_000,
            "node": 200_000,
            "edges": 200_000,
            "summary": 200_000,
            "trios": 20_000,
            "names": 20_000,
        }
    )

    def __post_init__(self) -> None:
        """Check every value."""
        _positive_number(self.hyperconnective_threshold, "hyperconnective_threshold", integer=True)
        sizes = dict(self.cache_sizes)
        if set(sizes) != set(_CACHE_KINDS):
            raise InvalidArgumentError(
                f"`cache_sizes` must give exactly the kinds {list(_CACHE_KINDS)}, got {sorted(sizes)}.",
                argument="cache_sizes",
                allowed=_CACHE_KINDS,
            )
        for kind, size in sizes.items():
            _positive_number(size, f"cache_sizes[{kind!r}]", integer=True)
        object.__setattr__(self, "cache_sizes", dict(sizes))

    def __hash__(self) -> int:
        """Hash by value (the cache sizes are a dict)."""
        return hash((self.hyperconnective_threshold, tuple(sorted(self.cache_sizes.items()))))


def check_search(value: Any) -> SearchSettings | None:
    """Return a ``search`` argument: a :py:class:`SearchSettings` or ``None``.

    Raises:
        ArgumentTypeError: For anything else.
    """
    if value is not None and not isinstance(value, SearchSettings):
        raise ArgumentTypeError(
            f"`search` must be a SearchSettings, got {type(value).__name__}.", argument="search", value=value
        )
    return value


def check_graph_settings(value: Any) -> GraphSettings | None:
    """Return a ``graph_settings`` argument: a :py:class:`GraphSettings` or ``None``.

    Raises:
        ArgumentTypeError: For anything else.
    """
    if value is not None and not isinstance(value, GraphSettings):
        raise ArgumentTypeError(
            f"`graph_settings` must be a GraphSettings, got {type(value).__name__}.",
            argument="graph_settings",
            value=value,
        )
    return value


@dataclass
class EnsemblSettings:
    """How the build stack reaches Ensembl. One instance, :py:data:`ENSEMBL`, set with :py:func:`configure_ensembl`.

    Attributes:
        access (str | None): ``"auto"`` uses the public MySQL service where its ports answer and the HTTPS dumps
            otherwise; ``"https"`` never touches MySQL. ``None`` reads the environment variable
            :py:attr:`access_env_var` and falls back to ``"auto"``.
        access_env_var (str): The environment variable read when :py:attr:`access` is ``None``.
        connection_timeout (float): TCP connect timeout in seconds of HTTPS and REST requests.
        reading_timeout (float): Socket read timeout in seconds of HTTPS and REST requests.
        rest_attempts (int): Attempts of every request to Ensembl (REST and HTTPS dumps): timeouts, connection
            errors and the HTTP statuses 429, 500, 502, 503 and 504 are retried.
        rest_retry_wait (float): Seconds before the second attempt; doubled before each further one (an HTTP
            ``Retry-After`` header is honoured when longer, up to 60 s).
        rest_species_cache_seconds (float): How long a process keeps Ensembl's species list.
        https_host (str): Host of Ensembl's HTTPS dumps.
        rest_server (str): Root URL of Ensembl REST.
        mysql_host (str): Host of Ensembl's public MySQL service.
        mysql_user (str): User of the anonymous MySQL access.
        mysql_password (str): Password of the anonymous MySQL access (empty: the public service needs none).
        mysql_port_min_release (dict[int, int]): The first Ensembl release each public MySQL port serves.
        mysql_port_assembly (dict[int, tuple[str, int]]): Ports that serve one assembly only,
            ``{port: (organism, assembly key)}``: 3337, Ensembl's archive, is read for human GRCh37 only (its frozen
            copies of other assemblies under later release numbers are not annotation of those releases).
        mysql_probe_timeout (float): TCP connect timeout of the one probe of a MySQL port per process.
    """

    access: str | None = None
    access_env_var: str = "IDTRACK_ENSEMBL_ACCESS"
    connection_timeout: float = 15
    reading_timeout: float = 60
    rest_attempts: int = 3
    rest_retry_wait: float = 2.0
    rest_species_cache_seconds: float = 86400.0
    https_host: str = "ftp.ensembl.org"
    rest_server: str = "https://rest.ensembl.org"
    mysql_host: str = "ensembldb.ensembl.org"
    mysql_user: str = "anonymous"
    mysql_password: str = ""
    mysql_port_min_release: dict[int, int] = field(default_factory=lambda: {3306: 48, 5306: 48, 3337: 79})
    mysql_port_assembly: dict[int, tuple[str, int]] = field(default_factory=lambda: {3337: ("homo_sapiens", 37)})
    mysql_probe_timeout: float = 5.0


#: The access modes of :py:attr:`EnsemblSettings.access`.
ACCESS_MODES: tuple[str, ...] = ("auto", "https")

#: How the build stack reaches Ensembl, for the whole process (see :py:func:`configure_ensembl`).
ENSEMBL = EnsemblSettings()


def configure_ensembl(**values: Any) -> EnsemblSettings:
    """Change how the build stack reaches Ensembl, checking every value first (nothing changes if one is wrong).

    Example:
        >>> import idtrack.settings
        >>> _ = idtrack.settings.configure_ensembl(access="https", reading_timeout=120)

    Args:
        values: Fields of :py:class:`EnsemblSettings` and their new values.

    Returns:
        EnsemblSettings: :py:data:`ENSEMBL`, changed.

    Raises:
        idtrack.errors.InvalidArgumentError: For an unknown field, an unknown access mode or a value out of range.
        idtrack.errors.ArgumentTypeError: For a value of the wrong type.
    """
    names = {f.name for f in dataclasses.fields(EnsemblSettings)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise InvalidArgumentError(
            f"Unknown Ensembl settings {unknown}; known: {sorted(names)}.", allowed=sorted(names)
        )
    for name, value in values.items():
        if name == "access":
            if value is not None and value not in ACCESS_MODES:
                raise InvalidArgumentError(
                    f"`access` must be one of {ACCESS_MODES} or None, got {value!r}.",
                    argument=name,
                    allowed=ACCESS_MODES,
                )
        elif name == "rest_retry_wait":
            # `cast`: a `numbers.Real` compares with numbers at run time, which its type does not say.
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0 <= cast(float, value) < math.inf:
                raise InvalidArgumentError(
                    f"`rest_retry_wait` must be a number of seconds, 0 or more, got {value!r}.",
                    argument=name,
                    value=value,
                )
        elif name in (
            "connection_timeout",
            "reading_timeout",
            "rest_species_cache_seconds",
            "mysql_probe_timeout",
        ):
            _positive_number(value, name)
        elif name == "rest_attempts":
            _positive_number(value, name, integer=True)
        elif name in ("access_env_var", "https_host", "rest_server", "mysql_host", "mysql_user", "mysql_password"):
            if not isinstance(value, str):
                raise ArgumentTypeError(f"`{name}` must be a string, got {value!r}.", argument=name, value=value)
        elif not isinstance(value, dict):
            raise ArgumentTypeError(f"`{name}` must be a dict, got {value!r}.", argument=name, value=value)
    for name, value in values.items():
        setattr(ENSEMBL, name, value)
    return ENSEMBL


__all__ = [
    "ACCESS_MODES",
    "ENSEMBL",
    "NON_BRIDGING_DATABASES",
    "EnsemblSettings",
    "GraphSettings",
    "SearchSettings",
    "check_graph_settings",
    "check_search",
    "configure_ensembl",
]
