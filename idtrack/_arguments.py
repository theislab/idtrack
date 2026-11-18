"""The checks of a caller's arguments that need no graph: one rule and one message for each kind of value.

The API, the engine, the graph, the builder and the harmonizer check their arguments here. The checks that need a
graph (a release of the graph, a database of a release) are the engine's (:py:class:`idtrack.engine.Track`), which
the API calls as well.
"""

from __future__ import annotations

import operator
import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from idtrack.errors import ArgumentTypeError


def release(value: Any, name: str) -> int:
    """Return an argument that names an Ensembl release as ``int``.

    Args:
        value: Any integer type (``int``, ``numpy.int64``); never a boolean, a float or a string.
        name: The name of the argument, for the message.

    Returns:
        int: The release.

    Raises:
        ArgumentTypeError: If the value is not an integer.
    """
    if not isinstance(value, (bool, np.bool_)):
        try:
            return int(operator.index(value))
        except TypeError:
            pass
    raise ArgumentTypeError(f"`{name}` must be an integer Ensembl release, got {value!r}.", argument=name, value=value)


def optional_release(value: Any, name: str) -> int | None:
    """Return a release argument that may be ``None`` (see :py:func:`release`)."""
    return None if value is None else release(value, name)


def boolean(value: Any, name: str) -> bool:
    """Return an argument that must be ``True`` or ``False`` (NumPy's booleans too).

    Raises:
        ArgumentTypeError: For anything else (``0``, ``1``, ``"yes"``, ``None``).
    """
    if not isinstance(value, (bool, np.bool_)):
        raise ArgumentTypeError(f"`{name}` must be True or False, got {value!r}.", argument=name, value=value)
    return bool(value)


def string(value: Any, name: str) -> str:
    """Return an argument that must be a string.

    Raises:
        ArgumentTypeError: For anything else.
    """
    if not isinstance(value, str):
        raise ArgumentTypeError(
            f"`{name}` must be a string, got {type(value).__name__} {value!r}.", argument=name, value=value
        )
    return value


def strings(values: Any, name: str) -> list[str]:
    """Return a batch of strings as a list, refusing a single string and items that are not strings.

    Args:
        values: A list, a tuple, a NumPy array, a pandas Index or Series of strings.
        name: The name of the argument, for the message.

    Returns:
        list[str]: The items.

    Raises:
        ArgumentTypeError: For a string, a value that is not iterable, or items that are not strings (their positions
            are named).
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ArgumentTypeError(
            f"`{name}` must be a sequence of strings (a list, a tuple, a NumPy array, a pandas Index), got "
            f"{type(values).__name__}.",
            argument=name,
            value=type(values).__name__,
        )
    items = list(values)
    not_strings = [i for i, x in enumerate(items) if not isinstance(x, str)]
    if not_strings:
        raise ArgumentTypeError(
            f"`{name}` holds {len(not_strings)} non-string items, at positions {not_strings[:20]} "
            f"(e.g. {items[not_strings[0]]!r}).",
            argument=name,
            value=not_strings[:20],
        )
    return items


def path(value: Any, name: str) -> str:
    """Return a file or folder argument (a string or a path-like object) as a string.

    Raises:
        ArgumentTypeError: For anything else.
    """
    if not isinstance(value, (str, os.PathLike)):
        raise ArgumentTypeError(f"`{name}` must be a file path, got {value!r}.", argument=name, value=value)
    return os.fspath(value)
