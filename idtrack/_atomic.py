"""The one rule for writing a file that a reader may open at any time: never show a part of it.

A file is written under a temporary name in its own folder (``<name>.tmp-<pid>-<random>``, unique per process and
call), flushed to disk, and renamed over the target in one step; on any failure the temporary file is removed and the
target is left as it was. Every file idtrack writes (tables, graph files, the externals YAML, Ensembl's index, the
harmonizer's result cache) goes through :py:func:`replacing`.

This module imports nothing of idtrack.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator

#: The part of a temporary file's name that marks it as one (``<name>.tmp-<pid>-<random>``).
TMP_MARKER = ".tmp-"


def temporary_name(path: str) -> str:
    """Return a temporary name beside *path*, unique to this process and call.

    Args:
        path: The file to be written.

    Returns:
        str: The temporary path.
    """
    return f"{path}{TMP_MARKER}{os.getpid()}-{uuid.uuid4().hex[:12]}"


@contextlib.contextmanager
def replacing(path: str) -> Iterator[str]:
    """Give a temporary path to write; on success, flush it to disk and rename it over *path*.

    Example:
        >>> with replacing(target) as tmp:  # doctest: +SKIP
        ...     write_somehow(tmp)

    Args:
        path: The file to be written.

    Yields:
        str: The temporary path, in the same folder as *path*.
    """
    tmp = temporary_name(path)
    try:
        yield tmp
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def write_text(path: str, text: str) -> None:
    """Write a text file (UTF-8) without ever showing a part of it.

    Args:
        path: The file.
        text: Its content.
    """
    with replacing(path) as tmp, open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
