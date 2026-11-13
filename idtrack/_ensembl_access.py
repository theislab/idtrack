#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""How idtrack reaches Ensembl: the access mode and the per-process reachability of TCP ports.

Ensembl serves the same core databases two ways: the public MySQL service (``ensembldb.ensembl.org``, ports 3306,
5306 and 3337) and the MySQL dumps on ``https://ftp.ensembl.org/pub/``. Many networks block the MySQL ports, and a
blocked port costs a full connect timeout every time it is tried. This module makes that cost a one-off:

* :py:func:`ensembl_access_mode` returns the access mode, ``"auto"`` (MySQL where its port answers, the HTTPS dumps
  otherwise) or ``"https"`` (MySQL is never touched), from :py:attr:`idtrack.DB.ensembl_access` or the
  ``IDTRACK_ENSEMBL_ACCESS`` environment variable.
* :py:data:`reachability` (a :py:class:`PortReachability`) probes a ``(host, port)`` once per process with a short
  timeout and remembers the answer. Every code path that would open a MySQL connection asks it first, and a
  connection that fails after a successful probe marks the port unreachable for the rest of the process.
* :py:func:`rest_get_json` is the one way idtrack asks Ensembl REST, and :py:func:`https_get` the one way it reads
  the HTTPS dumps: both with the timeouts of :py:attr:`idtrack.DB.connection_timeout` and
  :py:attr:`idtrack.DB.reading_timeout`, the same retries, and an error that says what failed and what to change
  (:py:class:`idtrack.EnsemblRestError`, :py:class:`EnsemblDownloadError`). :py:func:`https_get` tells "absent"
  (HTTP 404 or 410 from a server that answered) apart from "unknown" (time-out, connection error, 429, 5xx), which is
  retried and, if it stays unknown, raised: a failed request is never read as "does not exist".

The results do not depend on the mode: both paths read the same Ensembl tables, with the same column types.
"""

import logging
import os
import socket
import threading
import time
from typing import Any

import requests

from idtrack._db import DB, EnsemblRestError

# HTTP statuses after which a REST request is tried again (rate limit and transient server errors).
_REST_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# HTTP statuses that mean "this resource does not exist" from a server that answered.
_ABSENT_STATUSES = frozenset({404, 410})


class EnsemblDownloadError(ConnectionError):
    """A file of Ensembl's HTTPS dumps could not be read: the server did not answer, or answered with an error.

    Raised after :py:attr:`idtrack.DB.rest_attempts` attempts; never raised for a file that does not exist
    (:py:func:`https_get` returns ``None`` then).
    """


_sessions = threading.local()


def _session() -> requests.Session:
    """Return the :py:class:`requests.Session` of the calling thread (connections to Ensembl are reused)."""
    session = getattr(_sessions, "session", None)
    if session is None:
        session = requests.Session()
        _sessions.session = session
    return session


def https_get(
    url: str, *, what: str, stream: bool = False, headers: dict[str, str] | None = None
) -> requests.Response | None:
    """Return the answer to ``GET url``, ``None`` if the file does not exist, retrying transient failures.

    The policy is that of :py:func:`rest_get_json`: timeouts ``(DB.connection_timeout, DB.reading_timeout)``;
    timeouts, connection errors and the statuses 429, 500, 502, 503 and 504 are retried up to
    :py:attr:`idtrack.DB.rest_attempts` attempts with doubling waits (or the server's ``Retry-After``, at most
    60 s). HTTP 404 and 410 mean the file does not exist. The connection is reused within a thread.

    Args:
        url: The URL.
        what: What is asked, for the log and the error message.
        stream: Return before the body is read (the caller reads and closes the response); otherwise the body is
            read and the connection released.
        headers: Extra request headers.

    Returns:
        Optional[requests.Response]: The successful response, or ``None`` for an absent file.

    Raises:
        EnsemblDownloadError: If no attempt gives an answer, or the server answers with another error status.
    """
    attempts = max(1, int(DB.rest_attempts))
    timeout = (DB.connection_timeout, DB.reading_timeout)
    log = logging.getLogger(__name__)
    failure = ""
    last_exc: BaseException | None = None
    session = _session()
    for attempt in range(1, attempts + 1):
        wait = float(DB.rest_retry_wait) * 2 ** (attempt - 1)
        try:
            response = session.get(url, timeout=timeout, stream=stream, headers=headers)
            status = int(response.status_code)
            if response.ok:
                if not stream:
                    _ = response.content
                return response
            response.close()
            if status in _ABSENT_STATUSES:
                return None
            failure = f"HTTP {status} {response.reason}"
            if status not in _REST_RETRY_STATUSES:
                raise EnsemblDownloadError(f"Ensembl refused {what} at {url}: {failure}.")
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None and str(retry_after).strip().isdigit():
                wait = max(wait, min(float(retry_after), 60.0))
        except requests.RequestException as exc:
            last_exc = exc
            failure = f"{exc.__class__.__name__}: {exc}"
        if attempt < attempts:
            log.warning(
                "Ensembl did not answer %s (attempt %d/%d: %s); retrying in %.1f s.",
                what,
                attempt,
                attempts,
                failure,
                wait,
            )
            time.sleep(wait)
    raise EnsemblDownloadError(
        f"Ensembl did not answer {what} at {url} after {attempts} attempt(s) with connect timeout {timeout[0]} s and "
        f"read timeout {timeout[1]} s; last failure: {failure}. On a slow network raise the timeouts, e.g. "
        "`idtrack.DB.reading_timeout = 120` (and `idtrack.DB.rest_attempts`); behind a proxy set HTTPS_PROXY."
    ) from last_exc


def rest_get_json(path: str, *, what: str) -> Any:
    """Return the decoded JSON of ``GET {DB.rest_server_api}{path}``, retrying transient failures.

    Every attempt uses the timeouts ``(DB.connection_timeout, DB.reading_timeout)``. Timeouts, connection errors
    (including SSL and proxy errors) and the HTTP statuses 429, 500, 502, 503 and 504 are retried up to
    :py:attr:`idtrack.DB.rest_attempts` attempts, waiting :py:attr:`idtrack.DB.rest_retry_wait` seconds,
    doubled each time (or the server's ``Retry-After``, at most 60 s). Any other HTTP status is not retried.

    Args:
        path: Path and query of the REST resource, e.g. ``"/info/species?"``.
        what: What is asked, for the error message (e.g. ``"the list of species"``).

    Returns:
        Any: The decoded JSON payload.

    Raises:
        EnsemblRestError: If no attempt gives a successful answer with valid JSON.
    """
    url = f"{DB.rest_server_api}{path}"
    attempts = max(1, int(DB.rest_attempts))
    timeout = (DB.connection_timeout, DB.reading_timeout)
    log = logging.getLogger(__name__)
    failure = ""
    last_exc: BaseException | None = None
    with requests.Session() as session:
        for attempt in range(1, attempts + 1):
            wait = float(DB.rest_retry_wait) * 2 ** (attempt - 1)
            try:
                with session.get(url, headers={"Content-Type": "application/json"}, timeout=timeout) as response:
                    if response.ok:
                        try:
                            return response.json()
                        except ValueError as exc:
                            raise EnsemblRestError(f"Ensembl REST answered {what} at {url} with invalid JSON.") from exc
                    status = int(response.status_code)
                    failure = f"HTTP {status} {response.reason}"
                    if status not in _REST_RETRY_STATUSES:
                        raise EnsemblRestError(f"Ensembl REST refused {what} at {url}: {failure}.")
                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None and str(retry_after).strip().isdigit():
                        wait = max(wait, min(float(retry_after), 60.0))
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                failure = f"{exc.__class__.__name__}: {exc}"
            if attempt < attempts:
                log.warning(
                    "Ensembl REST did not answer %s (attempt %d/%d: %s); retrying in %.1f s.",
                    what,
                    attempt,
                    attempts,
                    failure,
                    wait,
                )
                time.sleep(wait)
    raise EnsemblRestError(
        f"Ensembl REST did not answer {what} at {url} after {attempts} attempt(s) with connect timeout "
        f"{timeout[0]} s and read timeout {timeout[1]} s; last failure: {failure}. On a slow network raise the "
        "timeouts, e.g. `idtrack.DB.reading_timeout = 120` (and `idtrack.DB.rest_attempts`); behind a proxy set "
        "HTTPS_PROXY. Check https://rest.ensembl.org/info/ping from this machine."
    ) from last_exc


def ensembl_access_mode() -> str:
    """Return the active Ensembl access mode.

    :py:attr:`idtrack.DB.ensembl_access` wins when it is set; otherwise the environment variable named by
    :py:attr:`idtrack.DB.ensembl_access_env_var` (``IDTRACK_ENSEMBL_ACCESS``) is read; otherwise ``"auto"``.

    Returns:
        str: ``"auto"`` or ``"https"``.

    Raises:
        ValueError: If the configured value is not one of :py:attr:`idtrack.DB.ensembl_access_modes`.
    """
    value = DB.ensembl_access
    source = "DB.ensembl_access"
    if value is None:
        value = os.environ.get(DB.ensembl_access_env_var)
        source = DB.ensembl_access_env_var
    if value is None or str(value).strip() == "":
        return "auto"
    mode = str(value).strip().lower()
    if mode not in DB.ensembl_access_modes:
        raise ValueError(
            f"{source}={value!r} is not a valid Ensembl access mode; use one of {DB.ensembl_access_modes}."
        )
    return mode


class PortReachability:
    """Remember, per process, whether a TCP ``(host, port)`` accepts connections.

    The first question about a ``(host, port)`` opens one TCP connection with a short timeout
    (:py:attr:`idtrack.DB.mysql_probe_timeout` unless given); later questions are answered from memory.
    :py:meth:`mark_unreachable` records a failure observed elsewhere: a MySQL connection that fails after a
    successful probe (refused, dropped or lost during a query) marks the port unreachable. A "reachable" answer is
    kept for the life of the process; an "unreachable" one expires, so that a long-running process does not stay on
    the slower HTTPS dumps for days after one network blip: after :py:attr:`retry_after_seconds` (10 minutes) the
    port is probed again, and every further failure doubles the wait, up to :py:attr:`max_retry_after_seconds` (one
    day). The memory is process-wide, so parallel requests do not each wait for a probe. Unreachable ports only cost
    speed: the tables are then read from the HTTPS dumps, which hold the same rows. :py:meth:`reset` forgets
    everything, which is needed when the network route changes inside a process (as
    :py:class:`idtrack.ConnectionBridge` does when it starts or stops).
    """

    #: Seconds after a first failure before the port is probed again.
    retry_after_seconds: float = 600.0
    #: Upper bound of the doubling wait between probes of a port that keeps failing.
    max_retry_after_seconds: float = 86400.0

    def __init__(self) -> None:
        """Create an empty reachability memory."""
        # (host, port) -> (answer, time of the answer, number of consecutive failures)
        self._known: dict[tuple[str, int], tuple[bool, float, int]] = {}
        self._lock = threading.Lock()
        self.log = logging.getLogger(__name__)

    def _remembered(self, key: tuple[str, int]) -> bool | None:
        """Return the remembered answer for *key*, or ``None`` when there is none or an "unreachable" one expired."""
        entry = self._known.get(key)
        if entry is None:
            return None
        answer, when, failures = entry
        if answer:
            return True
        wait = min(self.retry_after_seconds * 2 ** max(failures - 1, 0), self.max_retry_after_seconds)
        return False if time.monotonic() - when < wait else None

    def _record(self, key: tuple[str, int], answer: bool) -> None:
        """Store an answer (the caller holds the lock)."""
        failures = 0 if answer else self._known.get(key, (True, 0.0, 0))[2] + 1
        self._known[key] = (answer, time.monotonic(), failures)

    @staticmethod
    def _probe(host: str, port: int, timeout_s: float) -> bool:
        """Open and close one TCP connection; return whether it succeeded."""
        try:
            with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
                return True
        except OSError:
            return False

    def can_connect(self, host: str, port: int, timeout_s: float | None = None) -> bool:
        """Return whether ``(host, port)`` accepts TCP connections, probing it only the first time.

        Args:
            host: Host name.
            port: TCP port.
            timeout_s: Connect timeout of the probe in seconds; defaults to
                :py:attr:`idtrack.DB.mysql_probe_timeout`.

        Returns:
            bool: The remembered (or freshly probed) answer.
        """
        key = (str(host), int(port))
        with self._lock:
            known = self._remembered(key)
        if known is not None:
            return known
        timeout = float(DB.mysql_probe_timeout if timeout_s is None else timeout_s)
        result = self._probe(key[0], key[1], timeout)
        with self._lock:
            # Another thread may have answered meanwhile; the first answer stands.
            known = self._remembered(key)
            if known is None:
                self._record(key, result)
            else:
                result = known
        if not result:
            self.log.info(
                "%s:%d does not accept connections (probe timeout %.1f s); it is not tried again for a while.",
                key[0],
                key[1],
                timeout,
            )
        return result

    def mysql_port_usable(self, port: int) -> bool:
        """Return whether the Ensembl MySQL service may be used on ``port``.

        In ``"https"`` mode this is always ``False`` and nothing is probed; in ``"auto"`` mode it is
        :py:meth:`can_connect` for :py:attr:`idtrack.DB.mysql_host`.

        Args:
            port: MySQL port of :py:attr:`idtrack.DB.mysql_host`.

        Returns:
            bool: ``True`` if a MySQL connection should be attempted.
        """
        if ensembl_access_mode() == "https":
            return False
        return self.can_connect(DB.mysql_host, int(port))

    def mark_unreachable(self, host: str, port: int, reason: str | None = None) -> None:
        """Record that ``(host, port)`` failed, so that it is not tried again for a while (see the class docstring).

        A transient failure therefore sends later tables to the HTTPS dumps for that while, which costs speed but not
        correctness.

        Args:
            host: Host name.
            port: TCP port.
            reason: Optional text for the log message.
        """
        key = (str(host), int(port))
        with self._lock:
            was = self._remembered(key)
            self._record(key, False)
        if was is not False:
            self.log.warning(
                "%s:%d failed%s; using the HTTPS dumps instead for a while.",
                key[0],
                key[1],
                f" ({reason})" if reason else "",
            )

    def known(self) -> dict[tuple[str, int], bool]:
        """Return a copy of the remembered answers (for inspection and tests)."""
        with self._lock:
            return {key: answer for key, (answer, _, _) in self._known.items()}

    def reset(self) -> None:
        """Forget every remembered answer (the next question probes again)."""
        with self._lock:
            self._known.clear()


#: The process-wide reachability memory used by every idtrack code path that would open a MySQL connection.
reachability = PortReachability()
