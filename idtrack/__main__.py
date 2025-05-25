#!/usr/bin/env python3

"""Module Description:
    main.py - A command-line interface for the idtrack package.

Author:
    Kemal Inecik (k.inecik@gmail.com)
"""

import os
from typing import NoReturn


def main() -> NoReturn:
    """Entry point for the script that prints package information."""
    try:
        package_info = (
            f"Python Package: `idtrack`{os.linesep}"
            f"Author: `Kemal Inecik <k.inecik@gmail.com>`{os.linesep}"
            f"Documentation: `https://idtrack.readthedocs.io/en/latest/`"
        )
        print(package_info)
    except Exception as e:
        raise SystemExit(f"Failed to start the application due to: {e}")


if __name__ == "__main__":
    main()
