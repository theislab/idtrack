#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""How idtrack reaches Ensembl: the access mode and the per-process reachability of TCP ports.

Ensembl serves the same core databases two ways: the public MySQL service (``ensembldb.ensembl.org``, ports 3306,
5306 and 3337) and the MySQL dumps on ``https://ftp.ensembl.org/pub/``. Many networks block the MySQL ports, and a
blocked port costs a full connect timeout every time it is tried. This module makes that cost a one-off:

* :py:func:`ensembl_access_mode` returns the access mode, ``"auto"`` (MySQL where its port answers, the HTTPS dumps
  otherwise) or ``"https"`` (MySQL is never touched), from :py:attr:`idtrack._db.DB.ensembl_access` or the
  ``IDTRACK_ENSEMBL_ACCESS`` environment variable.
* :py:data:`reachability` (a :py:class:`PortReachability`) probes a ``(host, port)`` once per process with a short
  timeout and remembers the answer. Every code path that would open a MySQL connection asks it first, and a
  connection that fails after a successful probe marks the port unreachable for the rest of the process.
* :py:func:`rest_get_json` is the one way idtrack asks Ensembl REST: with the timeouts of
  :py:attr:`idtrack._db.DB.connection_timeout` and :py:attr:`idtrack._db.DB.reading_timeout`, retries, and an
  :py:class:`idtrack._db.EnsemblRestError` that says what failed and what to change.

The results do not depend on the mode: both paths read the same Ensembl tables.
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


def rest_get_json(path: str, *, what: str) -> Any:
    """Return the decoded JSON of ``GET {DB.rest_server_api}{path}``, retrying transient failures.

    Every attempt uses the timeouts ``(DB.connection_timeout, DB.reading_timeout)``. Timeouts, connection errors
    (including SSL and proxy errors) and the HTTP statuses 429, 500, 502, 503 and 504 are retried up to
    :py:attr:`idtrack._db.DB.rest_attempts` attempts, waiting :py:attr:`idtrack._db.DB.rest_retry_wait` seconds,
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
    log = logging.getLogger("ensembl_access")
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

    :py:attr:`idtrack._db.DB.ensembl_access` wins when it is set; otherwise the environment variable named by
    :py:attr:`idtrack._db.DB.ensembl_access_env_var` (``IDTRACK_ENSEMBL_ACCESS``) is read; otherwise ``"auto"``.

    Returns:
        str: ``"auto"`` or ``"https"``.

    Raises:
        ValueError: If the configured value is not one of :py:attr:`idtrack._db.DB.ensembl_access_modes`.
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
    """Remember, for the life of the process, whether a TCP ``(host, port)`` accepts connections.

    The first question about a ``(host, port)`` opens one TCP connection with a short timeout
    (:py:attr:`idtrack._db.DB.mysql_probe_timeout` unless given); every later question is answered from memory.
    :py:meth:`mark_unreachable` records a failure observed elsewhere: a MySQL connection that fails after a
    successful probe (refused, dropped or lost during a query) marks the port unreachable for the rest of the
    process, even if the failure was transient. This only costs speed: the tables are then read from the HTTPS dumps,
    which hold the same rows. :py:meth:`reset` forgets everything, which is needed only when the network route changes
    inside a process (as :py:class:`idtrack.ConnectionBridge` does when it starts or stops).
    """

    def __init__(self) -> None:
        """Create an empty reachability memory."""
        self._known: dict[tuple[str, int], bool] = {}
        self._lock = threading.Lock()
        self.log = logging.getLogger("ensembl_access")

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
                :py:attr:`idtrack._db.DB.mysql_probe_timeout`.

        Returns:
            bool: The remembered (or freshly probed) answer.
        """
        key = (str(host), int(port))
        with self._lock:
            known = self._known.get(key)
        if known is not None:
            return known
        timeout = float(DB.mysql_probe_timeout if timeout_s is None else timeout_s)
        result = self._probe(key[0], key[1], timeout)
        with self._lock:
            # Another thread may have answered meanwhile; the first answer stands.
            result = self._known.setdefault(key, result)
        if not result:
            self.log.info(
                "%s:%d does not accept connections (probe timeout %.1f s); it is not tried again in this process.",
                key[0],
                key[1],
                timeout,
            )
        return result

    def mysql_port_usable(self, port: int) -> bool:
        """Return whether the Ensembl MySQL service may be used on ``port``.

        In ``"https"`` mode this is always ``False`` and nothing is probed; in ``"auto"`` mode it is
        :py:meth:`can_connect` for :py:attr:`idtrack._db.DB.mysql_host`.

        Args:
            port: MySQL port of :py:attr:`idtrack._db.DB.mysql_host`.

        Returns:
            bool: ``True`` if a MySQL connection should be attempted.
        """
        if ensembl_access_mode() == "https":
            return False
        return self.can_connect(DB.mysql_host, int(port))

    def mark_unreachable(self, host: str, port: int, reason: str | None = None) -> None:
        """Record that ``(host, port)`` failed, so that it is not tried again in this process.

        The mark lasts until :py:meth:`reset`; a transient failure therefore also sends later tables to the HTTPS
        dumps, which costs speed but not correctness.

        Args:
            host: Host name.
            port: TCP port.
            reason: Optional text for the log message.
        """
        key = (str(host), int(port))
        with self._lock:
            was = self._known.get(key)
            self._known[key] = False
        if was is not False:
            self.log.warning(
                "%s:%d failed%s; using the HTTPS dumps instead for the rest of this process.",
                key[0],
                key[1],
                f" ({reason})" if reason else "",
            )

    def known(self) -> dict[tuple[str, int], bool]:
        """Return a copy of the remembered answers (for inspection and tests)."""
        with self._lock:
            return dict(self._known)

    def reset(self) -> None:
        """Forget every remembered answer (the next question probes again)."""
        with self._lock:
            self._known.clear()


#: The process-wide reachability memory used by every idtrack code path that would open a MySQL connection.
reachability = PortReachability()
