#!/usr/bin/env python

# Kemal Inecik
# k.inecik@gmail.com


"""Command-line interface."""

import os

import click
from rich import traceback


@click.command()
@click.version_option(version="0.0.3", message=click.style("idtrack Version: 0.0.3"))
def main() -> None:
    """idtrack."""
    print(
        f"Python Package: `idtrack`{os.linesep}"
        f"Author: `Kemal Inecik <k.inecik@gmail.com>`{os.linesep}"
        f"Description: `ID mapping between different times, databases, genome assemblies.`{os.linesep}"
        f"Documentation: `https://idtrack.readthedocs.io/en/latest/`"
    )


if __name__ == "__main__":
    traceback.install()
    main(prog_name="idtrack")  # pragma: no cover
