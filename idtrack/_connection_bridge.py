#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Process-scoped SOCKS bridging for restricted servers.

The primary entry point is :py:class:`idtrack.ConnectionBridge`. It enables IDTrack to run on servers without direct
internet access (e.g. HPC clusters) by routing the **current Python process** through a SOCKS5 proxy provided by an
SSH reverse tunnel such as ``ssh -R 1080 user@server``.

The bridge is intentionally lightweight and process-scoped:

- It does **not** modify system-wide proxy configuration.
- It only affects the current interpreter (one Python process / one Jupyter kernel).
- It is reversible via :py:meth:`idtrack.ConnectionBridge.stop` and also cleaned up best-effort at interpreter exit.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import threading
from dataclasses import dataclass
from typing import Any


class ConnectionBridge:
    """Route this Python process' outgoing TCP connections through an SSH-provided SOCKS proxy.

    Many restricted environments block outbound internet access from compute nodes. IDTrack needs outbound access to
    Ensembl services (REST/HTTPS, FTP over HTTPS, and sometimes public MySQL). If you can SSH into the server from a
    machine with internet access, you can expose a SOCKS5 proxy on the server via OpenSSH **remote dynamic
    forwarding**:

    .. code-block:: bash

        ssh -R 1080 user@server

    Then, inside Python on the server (or inside a Jupyter notebook kernel running on the server), enable the bridge:

    .. code-block:: python

        import idtrack

        b = idtrack.ConnectionBridge(proxy_port=1080)
        b.start()  # applies process-scoped networking changes

        # ... run IDTrack ...

        b.stop()   # restores the previous networking configuration

    Internals (for maintainers / power users)
    -----------------------------------------
    ``start()`` monkeypatches :py:data:`socket.socket` to :py:class:`socks.socksocket` (PySocks) and optionally sets the
    environment variables ``ALL_PROXY`` and ``all_proxy`` so subprocesses spawned from this process inherit the proxy.

    A private, process-wide :py:class:`~idtrack._connection_bridge.ConnectionBridge._BridgeState` singleton stores the
    original socket class, environment variables, and PySocks default proxy to ensure :py:meth:`stop` can restore the
    prior state precisely. The singleton also implements a simple reference counter so multiple
    :py:class:`~idtrack.ConnectionBridge` instances can share the same active bridge.

    Notes:
        - The bridge affects only the current Python process (one Jupyter kernel). Closing the Python process/kernel
          automatically removes the monkeypatch.
        - To avoid surprises, call :py:meth:`start` **before** the first network access in your program.
        - Status messages are emitted via the logger named ``"connection_bridge"`` and, when ``verbose=True``, printed
          to stdout for immediate visibility in notebooks.

    Args:
        proxy_host: SOCKS proxy host on the server. With ``ssh -R 1080 ...`` this is typically ``"127.0.0.1"``.
        proxy_port: SOCKS proxy port on the server. Must match the port used in the SSH command.
        set_env_proxy: If ``True`` (default), set ``ALL_PROXY``/``all_proxy`` while active so subprocesses inherit the
            proxy configuration.
    """

    _ENV_PROXY_KEYS: tuple[str, ...] = ("ALL_PROXY", "all_proxy")

    @dataclass
    class _BridgeState:
        """Internal, process-wide bridge state (not part of the public API)."""

        active_count: int = 0
        proxy_host: str = "127.0.0.1"
        proxy_port: int = 1080
        original_socket_cls: type[socket.socket] | None = None
        original_env: dict[str, str | None] | None = None
        original_socks_proxy: Any = None
        atexit_registered: bool = False

    _STATE = _BridgeState()
    _LOCK = threading.RLock()

    def __init__(
        self,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 1080,
        *,
        set_env_proxy: bool = True,
    ) -> None:
        """Create a new bridge controller without applying any network changes.

        Args:
            proxy_host: SOCKS proxy host on the server (default ``127.0.0.1``).
            proxy_port: SOCKS proxy port on the server (default ``1080``).
            set_env_proxy: If ``True``, set ``ALL_PROXY``/``all_proxy`` while active so subprocesses inherit the proxy.

        Attributes:
            log: Logger named ``"connection_bridge"`` for structured diagnostics.
            proxy_host: Effective proxy host for this instance.
            proxy_port: Effective proxy port for this instance.
            set_env_proxy: Whether this instance sets proxy environment variables when activating the bridge.
        """
        self.log = logging.getLogger("connection_bridge")
        self.proxy_host = str(proxy_host)
        self.proxy_port = int(proxy_port)
        self.set_env_proxy = bool(set_env_proxy)
        self._started = False

    @property
    def is_active(self) -> bool:
        """Return ``True`` if this instance currently holds an active bridge reference."""
        return bool(self._started)

    @staticmethod
    def _require_pysocks() -> Any:
        """Import and return the PySocks module (import name: ``socks``).

        Returns:
            Any: Imported ``socks`` module.

        Raises:
            ImportError: If PySocks is not installed.
        """
        try:
            import socks  # type: ignore
        except ImportError as exc:
            raise ImportError("ConnectionBridge requires PySocks. Install with `pip install PySocks`.") from exc
        return socks

    @staticmethod
    def _format_proxy_url(host: str, port: int) -> str:
        """Return a SOCKS proxy URL suitable for environment variables."""
        return f"socks5h://{host}:{int(port)}"

    @classmethod
    def _emit_global(cls, message: str, *, verbose: bool, level: int = logging.INFO) -> None:
        """Emit a message without requiring an instance (used by ``atexit`` cleanup)."""
        try:
            logging.getLogger("connection_bridge").log(level, message)
        except Exception:
            if verbose:
                print(message)
            return
        if verbose:
            print(message)

    def _emit(self, message: str, *, verbose: bool, level: int = logging.INFO) -> None:
        """Emit a status message via the instance logger and (optionally) stdout."""
        try:
            self.log.log(level, message)
        except Exception:
            if verbose:
                print(message)
            return
        if verbose:
            print(message)

    @staticmethod
    def _restore_socks_default_proxy(socks_module: Any, original_proxy: Any) -> None:
        """Restore the PySocks default proxy configuration (best effort)."""
        if original_proxy is None:
            socks_module.set_default_proxy()
            return
        if isinstance(original_proxy, tuple):
            socks_module.set_default_proxy(*original_proxy)
            return
        socks_module.set_default_proxy()

    @staticmethod
    def _atexit_cleanup() -> None:
        """Best-effort cleanup hook registered via :py:mod:`atexit`."""
        try:
            ConnectionBridge._force_disable_bridge(verbose=False)
        except Exception:
            # Never block interpreter shutdown.
            return

    @classmethod
    def _force_disable_bridge(cls, *, verbose: bool) -> None:
        """Disable the bridge regardless of which instance started it (best-effort).

        This method is used by the ``atexit`` hook and by unit tests to ensure a clean process state. It intentionally
        bypasses instance-level bookkeeping (e.g. ``self._started`` flags).

        Args:
            verbose: If ``True``, print a status message to stdout.
        """
        socks = None
        try:
            socks = cls._require_pysocks()
        except Exception:
            socks = None

        with cls._LOCK:
            state = cls._STATE
            if state.active_count <= 0:
                return

            state.active_count = 0

            if state.original_socket_cls is not None:
                socket.socket = state.original_socket_cls  # type: ignore[misc]

            if state.original_env is not None:
                for key, value in state.original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            if socks is not None:
                cls._restore_socks_default_proxy(socks, state.original_socks_proxy)

            state.original_socket_cls = None
            state.original_env = None
            state.original_socks_proxy = None

        cls._emit_global("[idtrack] ConnectionBridge disabled (atexit/test cleanup).", verbose=verbose)

    def start(self, *, test: bool = True, verbose: bool = True) -> bool:
        """Enable the bridge for the current Python process.

        The bridge is reference-counted across instances in the current interpreter. If another
        :py:class:`~idtrack.ConnectionBridge` already enabled the bridge with the *same* proxy host/port, calling
        :py:meth:`start` will simply increment the internal counter and return.

        Args:
            test: If ``True`` (default), run :py:meth:`test_connection` after enabling the bridge. If the test fails,
                the bridge is automatically disabled again and the method returns ``False``.
            verbose: If ``True`` (default), print status messages to stdout.

        Returns:
            bool: ``True`` if the bridge is enabled (and the optional test succeeds), otherwise ``False``.

        Raises:
            RuntimeError: If a bridge is already active in this process but configured with a different proxy host/port.
        """
        socks = self._require_pysocks()

        with self.__class__._LOCK:
            state = self.__class__._STATE

            if self._started:
                self._emit(
                    f"[idtrack] ConnectionBridge already active (proxy {self.proxy_host}:{self.proxy_port}).",
                    verbose=verbose,
                )
                return True

            if state.active_count > 0:
                if (self.proxy_host, self.proxy_port) != (state.proxy_host, state.proxy_port):
                    raise RuntimeError(
                        "ConnectionBridge is already active with a different proxy "
                        f"({state.proxy_host}:{state.proxy_port}). Call `stop()` before switching."
                    )
                state.active_count += 1
                self._started = True
                self._emit(f"[idtrack] ConnectionBridge re-used (refcount={state.active_count}).", verbose=verbose)
                return True

            state.proxy_host = self.proxy_host
            state.proxy_port = self.proxy_port
            state.active_count = 1
            state.original_socket_cls = socket.socket
            state.original_env = {key: os.environ.get(key) for key in self.__class__._ENV_PROXY_KEYS}
            state.original_socks_proxy = socks.get_default_proxy()

            socks.set_default_proxy(socks.SOCKS5, self.proxy_host, int(self.proxy_port), rdns=True)
            socket.socket = socks.socksocket  # type: ignore[misc]

            if self.set_env_proxy:
                proxy_url = self.__class__._format_proxy_url(self.proxy_host, int(self.proxy_port))
                for key in self.__class__._ENV_PROXY_KEYS:
                    os.environ[key] = proxy_url

            if not state.atexit_registered:
                atexit.register(self.__class__._atexit_cleanup)
                state.atexit_registered = True

            self._started = True

        self._emit(
            "[idtrack] ConnectionBridge enabled: all TCP sockets in this Python process are routed through "
            f"{self.proxy_host}:{self.proxy_port}. Call `b.stop()` to restore normal networking.",
            verbose=verbose,
        )

        if not test:
            return True

        ok = self.test_connection(verbose=verbose)
        if not ok:
            self._emit(
                "[idtrack] ConnectionBridge test failed; disabling bridge.", verbose=verbose, level=logging.WARNING
            )
            self.stop(verbose=verbose)
        return ok

    def test_connection(self, *, verbose: bool = True, timeout_s: float = 15.0) -> bool:
        """Verify connectivity to Ensembl services through the active bridge.

        The Ensembl REST ping is treated as the authoritative signal for success. MySQL connectivity checks are
        reported as warnings because IDTrack can fall back to HTTPS/FTP in some workflows.

        Args:
            verbose: If ``True`` (default), print status messages to stdout.
            timeout_s: Timeout (seconds) for the REST request.

        Returns:
            bool: ``True`` if Ensembl REST is reachable, otherwise ``False``.

        Raises:
            RuntimeError: If the bridge is not active in this process.
        """
        if self.__class__._STATE.active_count <= 0:
            raise RuntimeError("ConnectionBridge is not active. Call `ConnectionBridge.start()` first.")

        try:
            import requests

            resp = requests.get(
                "https://rest.ensembl.org/info/ping",
                headers={"Content-Type": "application/json"},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            payload = resp.text.strip()
        except Exception as exc:
            self._emit(
                f"[idtrack] ConnectionBridge check failed: {exc}\n"
                "  Troubleshooting:\n"
                "  - Ensure the SSH session is established from your local machine: `ssh -R 1080 user@server`\n"
                "  - On the server, check the SOCKS port is listening: `ss -tlnp | grep 1080`",
                verbose=verbose,
                level=logging.ERROR,
            )
            return False

        self._emit(f"[idtrack] ConnectionBridge check OK (Ensembl REST): {payload}", verbose=verbose)

        for port, label in ((3306, "3306"), (3337, "3337/GRCh37")):
            try:
                with socket.create_connection(("ensembldb.ensembl.org", int(port)), timeout=10.0):
                    pass
                self._emit(f"[idtrack] ConnectionBridge check OK (Ensembl MySQL {label}): reachable", verbose=verbose)
            except Exception as exc:
                self._emit(
                    f"[idtrack] ConnectionBridge check WARN (Ensembl MySQL {label}): {exc} "
                    "(IDTrack may still work via HTTPS/FTP/REST).",
                    verbose=verbose,
                    level=logging.WARNING,
                )

        return True

    def stop(self, *, verbose: bool = True) -> None:
        """Disable the bridge and restore normal networking for this process.

        If multiple :py:class:`~idtrack.ConnectionBridge` instances are active, the bridge is only fully disabled once
        the last instance calls :py:meth:`stop`.

        Args:
            verbose: If ``True`` (default), print status messages to stdout.
        """
        socks = None
        try:
            socks = self._require_pysocks()
        except Exception:
            socks = None

        with self.__class__._LOCK:
            state = self.__class__._STATE

            if not self._started:
                self._emit("[idtrack] ConnectionBridge already stopped.", verbose=verbose)
                return

            self._started = False
            state.active_count = max(0, int(state.active_count) - 1)
            if state.active_count > 0:
                self._emit(f"[idtrack] ConnectionBridge still active (refcount={state.active_count}).", verbose=verbose)
                return

            if state.original_socket_cls is not None:
                socket.socket = state.original_socket_cls  # type: ignore[misc]

            if state.original_env is not None:
                for key, value in state.original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            if socks is not None:
                self.__class__._restore_socks_default_proxy(socks, state.original_socks_proxy)

            state.original_socket_cls = None
            state.original_env = None
            state.original_socks_proxy = None

        self._emit("[idtrack] ConnectionBridge disabled: normal networking restored.", verbose=verbose)

    def __enter__(self) -> ConnectionBridge:
        """Enter a context manager that keeps the bridge enabled for the enclosed block."""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager and disable the bridge (best-effort)."""
        self.stop()

    def __del__(self) -> None:  # pragma: no cover
        """Best-effort safety net: stop the bridge when this instance is garbage-collected."""
        try:
            if getattr(self, "_started", False):
                self.stop(verbose=False)
        except Exception:
            return
