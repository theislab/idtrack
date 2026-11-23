"""The version of the running idtrack and, where it can be known, its git commit.

The version has one source: ``version`` in ``pyproject.toml``, read at run time from the installed distribution's
metadata. When the running code is not the installed distribution (a source tree put on ``sys.path`` beside an
installed idtrack of another place, or no installation at all), the version is ``"unknown"`` rather than the number
of some other code.
"""

from __future__ import annotations

import json
import os
from importlib import metadata

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _distribution() -> metadata.Distribution | None:
    """Return the installed distribution of idtrack if it is the code that runs, else ``None``."""
    try:
        dist = metadata.distribution("idtrack")
    except metadata.PackageNotFoundError:
        return None
    try:
        installed = os.path.dirname(os.path.abspath(str(dist.locate_file("idtrack/__init__.py"))))
    except (OSError, TypeError, ValueError):
        installed = None
    if installed is not None and os.path.realpath(installed) == os.path.realpath(_PACKAGE_DIR):
        return dist
    url = _direct_url(dist).get("url", "")
    if url.startswith("file://") and _direct_url(dist).get("dir_info", {}).get("editable"):
        source = url.removeprefix("file://")
        if os.path.realpath(os.path.join(source, "idtrack")) == os.path.realpath(_PACKAGE_DIR):
            return dist
    return None


def _direct_url(dist: metadata.Distribution) -> dict:
    """Return the distribution's ``direct_url.json`` (PEP 610), or an empty dict."""
    try:
        return json.loads(dist.read_text("direct_url.json") or "{}")
    except (OSError, ValueError):
        return {}


#: What a graph records as the version that built it when the code ran from a source folder that is not an installed
#: idtrack: such code has no released version (the version written in its ``pyproject.toml`` names a release, not
#: the state of the folder), and only the git commit, where it is recorded, names the code that built the graph.
SOURCE_FOLDER_BUILD = "none: built from a source folder, not an installed idtrack"


def package_version() -> str:
    """Return the version of the running idtrack (``"unknown"`` when it is not the installed distribution).

    Returns:
        str: The version.
    """
    dist = _distribution()
    return "unknown" if dist is None else str(dist.version)


def build_version() -> str:
    """Return the version a graph records as the one that built it.

    Returns:
        str: The version of the installed idtrack that runs, or :py:data:`SOURCE_FOLDER_BUILD` when the code runs from
        a source folder that is not installed (the graph then names its code by the commit alone).
    """
    dist = _distribution()
    return SOURCE_FOLDER_BUILD if dist is None else str(dist.version)


def _git_commit(root: str) -> str | None:
    """Return the commit checked out in the git work tree whose root is *root*, read from ``.git`` without git.

    Only the folder that holds the package is looked at (a source checkout of idtrack), never a folder above it,
    so that an installation inside another project's work tree is not given that project's commit.
    """
    git = os.path.join(root, ".git")
    if os.path.isfile(git):  # a worktree: "gitdir: <path>"
        try:
            with open(git) as handle:
                line = handle.read().strip()
        except OSError:
            return None
        if not line.startswith("gitdir:"):
            return None
        git = os.path.join(root, line.removeprefix("gitdir:").strip())
    if not os.path.isdir(git):
        return None
    try:
        with open(os.path.join(git, "HEAD")) as handle:
            head = handle.read().strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.removeprefix("ref:").strip()
        # A worktree's branches live in the repository's common folder, named in its file `commondir`.
        common = git
        commondir = os.path.join(git, "commondir")
        if os.path.isfile(commondir):
            with open(commondir) as handle:
                common = os.path.normpath(os.path.join(git, handle.read().strip()))
        for folder in dict.fromkeys((git, common)):
            path = os.path.join(folder, ref)
            if os.path.isfile(path):
                with open(path) as handle:
                    return handle.read().strip() or None
        for folder in dict.fromkeys((git, common)):
            packed = os.path.join(folder, "packed-refs")
            if os.path.isfile(packed):
                with open(packed) as handle:
                    for line in handle:
                        parts = line.split()
                        if len(parts) == 2 and parts[1] == ref:
                            return parts[0]
    except OSError:
        return None
    return None


def package_commit() -> str | None:
    """Return the git commit of the running idtrack, where it is known.

    Known for an installation from a git URL (pip records the commit, PEP 610) and for code that runs from a git
    checkout (read from its ``.git`` folder); ``None`` otherwise (a wheel from an index, a source archive).

    Returns:
        str | None: The full commit hash.
    """
    dist = _distribution()
    if dist is not None:
        commit = _direct_url(dist).get("vcs_info", {}).get("commit_id")
        if commit:
            return str(commit)
    return _git_commit(os.path.dirname(_PACKAGE_DIR))
