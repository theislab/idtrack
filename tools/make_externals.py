#!/usr/bin/env python3
"""Write an organism's externals YAML files: ``idtrack/default_config/<organism>_externals_{template,modified}.yml``.

Usage (from the root of the repository, with the package importable)::

    IDTRACK_ENSEMBL_ACCESS=https python tools/make_externals.py --organism sus_scrofa \
        --repository <local repository> --out <folder> [--snapshot 115] [--last-release 116] [--check]

What it does: :py:meth:`idtrack.build.DatabaseManager.create_database_content` over every assembly, release and form
of the organism up to the snapshot release, and over each later release up to ``--last-release`` (a manager of that
release alone), then :py:meth:`idtrack.build.ExternalDatabases.create_template_yaml` writes the template, and
:py:func:`idtrack.build.default_externals.apply_allowlist` with the organism's allowlist
(:py:data:`idtrack.build.default_externals.DEFAULT_EXTERNALS`) writes the modified file. The two files are written to
``--out``; copy them into ``idtrack/default_config/`` to ship them.

Reads: the tables of ``--repository``, a local repository of idtrack; what it lacks is downloaded from Ensembl into it
(a repository in which the organism's graph was built holds every table but the last release's, if that is later
than the graph's snapshot). Writes: the two YAML files in ``--out``, and the tables it computes or downloads in the
repository.

With ``--check``, the written files are compared with the shipped ones: their sha256, and for each form the databases
only in one of the two templates and those whose releases differ; the exit status is 1 when a file differs.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from idtrack import API, organisms  # noqa: E402
from idtrack.build.default_externals import DEFAULT_EXTERNALS, apply_allowlist  # noqa: E402
from idtrack.build.externals import ExternalDatabases, load_externals_yaml, write_externals_yaml  # noqa: E402

SHIPPED = ROOT / "idtrack" / "default_config"
KINDS = ("template", "modified")


def file_name(organism: str, kind: str) -> str:
    """Return the name of an organism's externals file of a kind (``template`` or ``modified``)."""
    return f"{organism}_externals_{kind}.yml"


def make(organism: str, repository: str, out: str, snapshot: int, last_release: int) -> dict[str, str]:
    """Write the template and the modified externals YAML of an organism; return their paths by kind.

    Args:
        organism: The organism (Ensembl's species name).
        repository: The local repository of the tables (it must exist).
        out: The folder the files are written to (made if missing).
        snapshot: The last release of the graphs; every release up to it is read with one manager.
        last_release: The last release the files list; each release after *snapshot* up to it is read alone.

    Returns:
        dict[str, str]: ``{"template": path, "modified": path}``.
    """
    os.makedirs(out, exist_ok=True)
    contents = [API.database_manager(organism, snapshot, repository).create_database_content()]
    for release in range(snapshot + 1, last_release + 1):
        manager = API.database_manager(organism, release, repository, first_release=release)
        contents.append(manager.create_database_content())
    content = pd.concat(contents, ignore_index=True)
    primary = organisms.primary_assembly(organism, snapshot)
    ExternalDatabases(organism, snapshot, "gene", out, primary).create_template_yaml(content)
    paths = {kind: os.path.join(out, file_name(organism, kind)) for kind in KINDS}
    template = load_externals_yaml(paths["template"], shipped=False)
    modified, _enabled, _absent = apply_allowlist(template, organism, DEFAULT_EXTERNALS[organism])
    write_externals_yaml(modified, paths["modified"])
    return paths


def sha256(path: str | os.PathLike) -> str:
    """Return the sha256 of a file."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _releases(entry: dict[str, Any]) -> dict[str, list[int]]:
    return {str(a): sorted(int(r) for r in str(v["Ensembl release"]).split(",")) for a, v in entry["Assembly"].items()}


def differences(organism: str, written: str, shipped: str) -> dict[str, dict[str, Any]]:
    """Return, per form, the databases only in one of two externals files and those whose releases differ."""
    new = load_externals_yaml(written, shipped=False)[organism]
    old = load_externals_yaml(shipped, shipped=True)[organism]
    report = {}
    for form in sorted(set(new) | set(old)):
        a, b = old.get(form) or {}, new.get(form) or {}
        report[form] = {
            "only_shipped": sorted(set(a) - set(b)),
            "only_written": sorted(set(b) - set(a)),
            "other_releases": sorted(db for db in set(a) & set(b) if _releases(a[db]) != _releases(b[db])),
        }
    return report


def check(organism: str, paths: dict[str, str]) -> bool:
    """Print the sha256 of the written and the shipped files and their differences; return whether all are equal."""
    equal = True
    for kind, path in paths.items():
        shipped = SHIPPED / file_name(organism, kind)
        written, expected = sha256(path), sha256(shipped)
        print(f"{kind}: written {written}, shipped {expected}: {'equal' if written == expected else 'DIFFERENT'}")
        if written != expected:
            equal = False
            for form, found in differences(organism, path, str(shipped)).items():
                if any(found.values()):
                    print(f"  {form}: {found}")
    return equal


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--organism", required=True, help="Ensembl's species name, e.g. sus_scrofa")
    parser.add_argument("--repository", required=True, help="the local repository of the tables (read; filled)")
    parser.add_argument("--out", required=True, help="the folder the two YAML files are written to")
    parser.add_argument("--snapshot", type=int, default=115, help="the graphs' snapshot release (default 115)")
    parser.add_argument(
        "--last-release", type=int, default=116, help="the last release the files list (default 116, as shipped)"
    )
    parser.add_argument("--check", action="store_true", help="compare the written files with the shipped ones")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
    if not os.path.isdir(arguments.repository):
        parser.error(f"the repository {arguments.repository!r} is not a folder")
    paths = make(arguments.organism, arguments.repository, arguments.out, arguments.snapshot, arguments.last_release)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    if arguments.check:
        return 0 if check(arguments.organism, paths) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
