"""The commit of a source checkout and the version a graph records.

The commit is read from the checkout's ``.git`` folder without git (:py:func:`idtrack._version._git_commit`); the
version a graph records is :py:func:`idtrack._version.build_version`.
"""

from __future__ import annotations

from pathlib import Path

from idtrack import _version
from idtrack._version import SOURCE_FOLDER_BUILD, _git_commit, build_version

COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _repository(root: Path, *, loose: bool) -> Path:
    """Make the ``.git`` folder of a checkout on branch ``main`` (a loose or a packed ref)."""
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    if loose:
        (git / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    else:
        (git / "packed-refs").write_text(f"# pack-refs with: peeled fully-peeled sorted\n{COMMIT} refs/heads/main\n")
    return git


def test_a_branch_is_read_loose_or_packed(tmp_path):
    """The branch of HEAD resolves through its loose ref, or through packed-refs when it has none."""
    _repository(tmp_path / "loose", loose=True)
    _repository(tmp_path / "packed", loose=False)
    assert _git_commit(str(tmp_path / "loose")) == COMMIT
    assert _git_commit(str(tmp_path / "packed")) == COMMIT


def test_a_detached_head_is_the_commit(tmp_path):
    """A detached HEAD holds the commit itself."""
    git = _repository(tmp_path, loose=True)
    (git / "HEAD").write_text(OTHER + "\n")
    assert _git_commit(str(tmp_path)) == OTHER


def test_a_worktree_reads_the_branches_of_its_repository(tmp_path):
    """A worktree's ``.git`` file names its folder, whose ``commondir`` names the repository's; packed refs too."""
    for loose in (True, False):
        main = tmp_path / ("loose" if loose else "packed")
        git = _repository(main, loose=loose)
        own = git / "worktrees" / "feature"
        own.mkdir(parents=True)
        (own / "HEAD").write_text("ref: refs/heads/main\n")
        (own / "commondir").write_text("../..\n")
        checkout = tmp_path / f"checkout_{main.name}"
        checkout.mkdir()
        (checkout / ".git").write_text(f"gitdir: {own}\n")
        assert _git_commit(str(checkout)) == COMMIT, main.name


def test_no_checkout_has_no_commit(tmp_path):
    """Without ``.git``, with a ``.git`` file that names no folder, or a branch that does not exist: ``None``."""
    assert _git_commit(str(tmp_path)) is None
    (tmp_path / ".git").write_text("not a gitdir line\n")
    assert _git_commit(str(tmp_path)) is None
    other = tmp_path / "other"
    git = _repository(other, loose=True)
    (git / "HEAD").write_text("ref: refs/heads/missing\n")
    assert _git_commit(str(other)) is None


def test_a_graph_built_from_a_source_folder_says_so(monkeypatch):
    """I138: code that is not the installed idtrack has no version; a graph records that plainly, not ``unknown``.

    Fails if a source-folder build records ``unknown`` (a graph that cannot tell what built it), or if an installed
    version is not recorded as it is.
    """
    monkeypatch.setattr(_version, "_distribution", lambda: None)
    assert build_version() == SOURCE_FOLDER_BUILD and "source folder" in SOURCE_FOLDER_BUILD
    assert _version.package_version() == "unknown"  # idtrack.__version__ of such code is unchanged

    class Installed:
        version = "0.1.0"

    monkeypatch.setattr(_version, "_distribution", lambda: Installed())
    assert build_version() == "0.1.0"
