#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""idtrack command-line interface.

This module exposes a minimal *console-scripts* entry-point for the `idtrack` package so that it
can be executed directly from the command

line::

    python -m idtrack [OPTIONS]

Run ``python -m idtrack --help`` for usage information.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from importlib import metadata
from typing import Any


def _resolve_version() -> str:
    """Return a best-effort version string for *idtrack*.

    The function attempts three mechanisms - in order - to figure out the
    package version at runtime:

    1. **Installed distribution metadata** obtained via
       :py:func:`importlib.metadata.version`.
    2. The ``__version__`` attribute of the *imported* ``idtrack`` package,
       useful while running directly from a source checkout.
    3. Fallback to the literal string ``"unknown"`` when neither of the
       above strategies succeeds.

    Returns:
        str: A semantic-version-like string if detection succeeds; otherwise ``"unknown"``.
    """
    try:
        return metadata.version("idtrack")
    except metadata.PackageNotFoundError:
        # Package not installed; try importing from the source tree.
        try:
            pkg = importlib.import_module("idtrack")
            return getattr(pkg, "__version__", "unknown")
        except ModuleNotFoundError:
            return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the *idtrack* CLI.

    The returned parser currently supports a single ``--version`` option but
    is structured to be extensible with additional options or sub-commands in
    the future.

    Returns:
        argparse.ArgumentParser: A fully configured parser instance ready for argument parsing.
    """
    parser = argparse.ArgumentParser(
        prog="idtrack",
        description="Lightweight command-line entry point for the idtrack package.",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"idtrack {_resolve_version()}",
        help="Show package version and exit.",
    )

    return parser


def _print_banner() -> None:
    """Print basic *idtrack* package information to *stdout*.

    The banner is shown when the CLI is executed without any options. It
    provides users with quick context about the package version, author and
    documentation URL.
    """
    print(
        f"Python Package : idtrack{os.linesep}"
        f"Version        : {_resolve_version()}{os.linesep}"
        f"Author         : Kemal Inecik <k.inecik@gmail.com>{os.linesep}"
        f"Docs           : https://idtrack.readthedocs.io/"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry-point function for the *idtrack* command-line interface.

    The function is designed to be called via ``python -m idtrack`` or as the
    ``console_scripts`` entry defined in the package's *setup.cfg*/pyproject.
    It delegates most of the actual work to helper functions for clarity and
    testability.

    Args:
        argv (list[str] | None, optional): Custom argument vector *excluding*
            the program name (``sys.argv[0]``). If *None* (the default),
            ``sys.argv[1:]`` is used.

    Returns:
        int: An exit status suitable for :py:func:`sys.exit`. ``0`` indicates successful execution.
    """
    parser = _build_parser()
    args: Any = parser.parse_args(argv)  # noqa: F841  (kept for future use)

    # No positional/sub-command yet → if nothing but the interpreter
    # called us, show the banner.
    if len(sys.argv) == 1:
        _print_banner()

    return 0


if __name__ == "__main__":
    sys.exit(main())
