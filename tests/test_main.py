"""Test cases for the __main__ module."""

import pytest

import idtrack
import idtrack.__main__ as idtrack_main


@pytest.fixture
def runner() -> None:
    """Fixture for invoking command-line interfaces."""
    _ = idtrack.DB


def test_cli_main_returns_zero_on_empty_args(monkeypatch, capsys) -> None:
    """CLI returns 0 and prints banner when invoked with no arguments."""
    monkeypatch.setattr("sys.argv", ["idtrack"])
    rc = idtrack_main.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Python Package" in out


def test_cli_version_flag_exits(monkeypatch, capsys) -> None:
    """CLI --version flag prints version and exits with code 0."""
    monkeypatch.setattr("sys.argv", ["idtrack", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        idtrack_main.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert out.lower().startswith("idtrack ")


def test_cli_main_rejects_unknown_args(monkeypatch, capsys) -> None:
    """CLI rejects unrecognized arguments with proper exit code."""
    monkeypatch.setattr("sys.argv", ["idtrack", "somearg"])
    with pytest.raises(SystemExit) as exc_info:
        idtrack_main.main(["somearg"])
    # argparse exits with code 2 for unrecognized arguments
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err
