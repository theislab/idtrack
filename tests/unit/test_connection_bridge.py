#!/usr/bin/env python3
"""Unit tests for idtrack._connection_bridge."""

from __future__ import annotations

import os
import socket

import pytest

import idtrack
from idtrack import _connection_bridge as connection_bridge


@pytest.fixture(autouse=True)
def _bridge_cleanup():
    """Ensure the process-wide bridge state is reset between tests."""
    yield
    connection_bridge.ConnectionBridge._force_disable_bridge(verbose=False)


def test_connection_bridge_patches_and_restores_socket_and_env(monkeypatch):
    """start() patches sockets/env and stop() restores the prior state."""
    import socks

    original_socket_cls = socket.socket

    monkeypatch.setenv("ALL_PROXY", "pre-existing")
    monkeypatch.delenv("all_proxy", raising=False)

    bridge = idtrack.ConnectionBridge(proxy_port=1080)
    assert bridge.start(test=False, verbose=False) is True
    assert socket.socket is socks.socksocket
    assert os.environ.get("ALL_PROXY") == "socks5h://127.0.0.1:1080"
    assert os.environ.get("all_proxy") == "socks5h://127.0.0.1:1080"

    bridge.stop(verbose=False)
    assert socket.socket is original_socket_cls
    assert os.environ.get("ALL_PROXY") == "pre-existing"
    assert os.environ.get("all_proxy") is None


def test_connection_bridge_refcounts_multiple_instances():
    """Multiple instances share a single process-wide bridge via reference counting."""
    import socks

    original_socket_cls = socket.socket

    b1 = idtrack.ConnectionBridge()
    b2 = idtrack.ConnectionBridge()

    assert b1.start(test=False, verbose=False) is True
    assert b2.start(test=False, verbose=False) is True
    assert socket.socket is socks.socksocket

    b1.stop(verbose=False)
    assert socket.socket is socks.socksocket

    b2.stop(verbose=False)
    assert socket.socket is original_socket_cls


def test_connection_bridge_test_failure_disables_bridge(monkeypatch):
    """When the post-start test fails, start() rolls back and returns False."""
    original_socket_cls = socket.socket

    def _fail_test_connection(self, **kwargs) -> bool:
        return False

    monkeypatch.setattr(idtrack.ConnectionBridge, "test_connection", _fail_test_connection)

    bridge = idtrack.ConnectionBridge()
    assert bridge.start(test=True, verbose=False) is False
    assert socket.socket is original_socket_cls
    assert bridge.is_active is False

    # stop() remains safe after a failed start().
    bridge.stop(verbose=False)
    assert socket.socket is original_socket_cls


def test_connection_bridge_rejects_proxy_switch_while_active():
    """A second bridge cannot change proxy host/port while the first is active."""
    bridge = idtrack.ConnectionBridge(proxy_port=1080)
    assert bridge.start(test=False, verbose=False) is True

    other = idtrack.ConnectionBridge(proxy_port=1081)
    with pytest.raises(RuntimeError, match="already active with a different proxy"):
        other.start(test=False, verbose=False)

    bridge.stop(verbose=False)
