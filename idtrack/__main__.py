#!/usr/bin/env python
"""Command-line interface."""
import click
from rich import traceback


@click.command()
@click.version_option(version="0.0.1", message=click.style("idtrack Version: 0.0.1"))
def main() -> None:
    """idtrack."""


if __name__ == "__main__":
    traceback.install()
    main(prog_name="idtrack")  # pragma: no cover
