#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Command-line interface for **idtrack**: ``python -m idtrack [--version]``.

Without arguments it prints the package version, author and documentation URL; ``--version`` prints the version.
"""

import argparse
import os
import sys

from idtrack import __version__


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command line (one option, ``--version``)."""
    parser = argparse.ArgumentParser(
        prog="idtrack",
        description="Lightweight command-line entry point for the idtrack package.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"idtrack {__version__}",
        help="Show package version and exit.",
    )
    return parser


def _print_banner() -> None:
    """Print the package name, version, author and documentation URL to standard output."""
    print(
        f"Python Package : idtrack{os.linesep}"
        f"Version        : {__version__}{os.linesep}"
        f"Author         : Kemal Inecik <k.inecik@gmail.com>{os.linesep}"
        f"Docs           : https://idtrack.readthedocs.io/"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command line (``python -m idtrack``, or the ``idtrack`` console script).

    Args:
        argv: Arguments without the program name; ``None`` means ``sys.argv[1:]``.

    Returns:
        int: The exit status, ``0``. Unknown arguments make :py:mod:`argparse` exit with status 2.
    """
    if argv is None:
        argv = sys.argv[1:]
    _build_parser().parse_args(argv)
    if not argv:
        _print_banner()
    return 0


if __name__ == "__main__":
    sys.exit(main())
