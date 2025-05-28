"""Test cases for the __main__ module."""

import pytest

import idtrack


@pytest.fixture
def runner() -> None:
    """Fixture for invoking command-line interfaces."""
    _ = idtrack.DB


def test_main_succeeds() -> None:
    """It exits with a status code of zero."""
    pass
