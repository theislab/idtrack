#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Storage of idtrack's cached tables: one immutable HDF5 file per table, written atomically.

Every table that :py:class:`idtrack.DatabaseManager` downloads or computes is kept in a file of
its own (``<directory>/<table key>.h5``). A table is written completely to a private temporary file in the same
directory, flushed to disk and only then renamed to its final name (:py:func:`os.replace`, atomic on POSIX file
systems), so:

* a table that is visible is complete: a process killed in the middle of a write (time limit, out of memory, closed
  kernel) leaves the repository as it was, apart from an orphaned temporary file that no reader ever opens;
* finished files are never opened for writing again, so readers need no lock and are never blocked by writers, and
  two processes that compute the same table at the same time both succeed (the last rename wins, with a complete
  table);
* replacing a table frees the space of the old one (there is no in-file deletion that HDF5 cannot reclaim).

Each file records the storage format version and the parameters the table was computed with
(:py:func:`write_table`); :py:func:`load_table` returns a table only if both match what the caller asks for, so a
table made under other parameters is never served.

The format stores every column by position (labels and dtypes are metadata), text as dictionary codes plus the
distinct values as one compressed UTF-8 byte buffer with offsets (missing values are code ``-1``, never a sentinel
string), categoricals as codes plus categories, datetimes and timedeltas as int64 with their exact unit, and numbers
and booleans as they are, all chunked with the shuffle filter and gzip level 4. A table read back equals the table
written (``DataFrame.equals`` and identical dtypes; text comes back in the default text dtype of the running pandas,
``object`` under pandas 2 and ``str`` under pandas 3, with ``NaN`` for missing values, which is what pandas produces
for freshly computed text). Files are written with the HDF5 format bounds ``("earliest", "v110")``, so a file written
with any HDF5 library from 1.10 on is readable by every other.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Protocol, cast

import h5py
import numpy as np
import pandas as pd

#: Version of the on-disk table format; a file of another version is not read (the table is recomputed).
TABLE_FORMAT_VERSION = 2

#: HDF5 format bounds of every file written: readable by HDF5 1.10 and later, whatever library writes it.
_LIBVER = ("earliest", "v110")

#: Compression of every non-empty dataset.
_COMPRESSION: dict[str, Any] = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

#: Temporary files older than this are left over from killed writers and are removed by the next writer.
_STALE_TEMPORARY_SECONDS = 24 * 3600

_TMP_MARKER = ".tmp-"

_NUMERIC_DTYPES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool",
}

_log = logging.getLogger(__name__)


class TableFormatError(ValueError):
    """A table cannot be stored in, or read from, idtrack's table format."""


def _default_text_dtype():
    """Return the dtype the running pandas infers for text (``object`` in pandas 2, ``str`` in pandas 3).

    Returns:
        The dtype of ``pd.Series(["a"])``.
    """
    return pd.Series(["a"]).dtype


def _column_tag(series: pd.Series) -> str:
    """Return the storage tag of a column, or raise :py:class:`TableFormatError` for an unsupported one.

    Tags: a numeric or boolean NumPy dtype name; ``"text"`` for the default text of pandas (``object`` holding only
    strings and missing values, or pandas 3's ``str``, i.e. ``StringDtype(na_value=nan)``); ``"string"`` for
    ``StringDtype`` with ``pd.NA``; ``"category"`` for categoricals with text categories; ``"datetime64[<unit>]"`` and
    ``"timedelta64[<unit>]"`` for tz-naive temporal columns.

    Args:
        series: The column.

    Returns:
        str: The tag.

    Raises:
        TableFormatError: For any other dtype, an ``object`` column holding something else than strings and missing
            values, or a categorical with non-text categories.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        kind = pd.api.types.infer_dtype(dtype.categories, skipna=True)
        if kind not in ("string", "empty"):
            raise TableFormatError(f"Categorical column {series.name!r} has non-text categories ({kind}).")
        return "category"
    if isinstance(dtype, pd.StringDtype):
        return "string" if dtype.na_value is pd.NA else "text"
    if pd.api.types.is_object_dtype(dtype):
        kind = pd.api.types.infer_dtype(series, skipna=True)
        if kind not in ("string", "empty"):
            raise TableFormatError(f"Object column {series.name!r} holds non-text values ({kind}).")
        return "text"
    if isinstance(dtype, np.dtype) and dtype.kind in "mM":
        return str(dtype)
    if isinstance(dtype, np.dtype) and str(dtype) in _NUMERIC_DTYPES:
        return str(dtype)
    raise TableFormatError(f"Unsupported dtype {dtype!r} for column {series.name!r}.")


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Check that *df* can be stored loss-lessly and return the storage tag of each column.

    Requirements: column labels are unique strings; every column has a supported dtype (see :py:func:`_column_tag`);
    the index is a :py:class:`pandas.RangeIndex` (every table idtrack caches has one: tables are built with
    ``reset_index``/``ignore_index``) and unnamed, and the columns index is unnamed.

    Args:
        df: The table.

    Returns:
        list[str]: The storage tag of every column, in column order.

    Raises:
        TableFormatError: If *df* is not a DataFrame or breaks one of the requirements; nothing has been written then.
    """
    if not isinstance(df, pd.DataFrame):
        raise TableFormatError(f"Only DataFrames are stored, got {type(df).__name__}.")
    labels = list(df.columns)
    if not all(isinstance(c, str) for c in labels):
        raise TableFormatError(f"Column labels must be strings: {labels}")
    if len(set(labels)) != len(labels):
        raise TableFormatError(f"Column labels must be unique: {labels}")
    if not isinstance(df.index, pd.RangeIndex) or df.index.name is not None:
        raise TableFormatError("The index must be an unnamed RangeIndex (reset it before storing).")
    if df.columns.name is not None:
        raise TableFormatError("The columns index must be unnamed.")
    return [_column_tag(df.iloc[:, i]) for i in range(len(labels))]


class _Digest(Protocol):
    """What the writer needs of a hash object (``hashlib.sha256()`` and the like)."""

    def update(self, data: bytes, /) -> None:
        """Feed bytes to the hash."""


def _create(group: h5py.Group, name: str, data: np.ndarray, digest: _Digest | None = None) -> None:
    """Create a dataset, compressed when it is not empty (and feed its bytes to *digest*)."""
    if digest is not None:
        digest.update(name.encode("utf-8"))
        digest.update(str(data.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(data).tobytes())
    if data.size:
        group.create_dataset(name, data=data, **_COMPRESSION)
    else:
        group.create_dataset(name, data=data)


def _write_strings(group: h5py.Group, prefix: str, strings: list[str], digest: _Digest | None = None) -> None:
    """Store distinct strings as one UTF-8 byte buffer with offsets (exact, NUL characters included)."""
    encoded = [s.encode("utf-8") for s in strings]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    if encoded:
        np.cumsum([len(b) for b in encoded], out=offsets[1:])
    _create(group, f"{prefix}.blob", np.frombuffer(b"".join(encoded), dtype=np.uint8), digest)
    _create(group, f"{prefix}.offsets", offsets, digest)


def _read_strings(group: h5py.Group, prefix: str) -> list[str]:
    """Read strings written by :py:func:`_write_strings`."""
    blob = group[f"{prefix}.blob"][()].tobytes()
    offsets = group[f"{prefix}.offsets"][()]
    return [blob[offsets[k] : offsets[k + 1]].decode("utf-8") for k in range(len(offsets) - 1)]


def _write_column(group: h5py.Group, i: int, series: pd.Series, tag: str, digest: _Digest | None = None) -> None:
    """Write column number *i* under the dataset names ``c<i>[.<part>]``."""
    name = f"c{i}"
    if tag in ("text", "string"):
        codes, uniques = pd.factorize(series.to_numpy(dtype=object), use_na_sentinel=True)
        _create(group, f"{name}.codes", codes.astype(np.int64 if len(uniques) >= 2**31 else np.int32), digest)
        _write_strings(group, name, [str(u) for u in uniques], digest)
    elif tag == "category":
        _create(group, f"{name}.codes", series.cat.codes.to_numpy().astype(np.int32), digest)
        _write_strings(group, name, [str(c) for c in series.cat.categories], digest)
    elif tag.startswith(("datetime64", "timedelta64")):
        _create(group, name, series.to_numpy().view(np.int64), digest)
    else:
        _create(group, name, series.to_numpy(), digest)


def _read_column(group: h5py.Group, i: int, tag: str, ordered: bool) -> Any:
    """Read column number *i* written by :py:func:`_write_column` in its final dtype."""
    name = f"c{i}"
    if tag in ("text", "string"):
        codes = group[f"{name}.codes"][()]
        missing = pd.NA if tag == "string" else np.nan
        lookup = np.array([*_read_strings(group, name), missing], dtype=object)
        values = lookup[codes]  # code -1 picks the missing value at the end
        if tag == "string":
            return pd.array(values, dtype="string")
        text_dtype = _default_text_dtype()
        return values if pd.api.types.is_object_dtype(text_dtype) else pd.array(values, dtype=text_dtype)
    if tag == "category":
        categories = _read_strings(group, name)
        return pd.Categorical.from_codes(group[f"{name}.codes"][()], categories=pd.Index(categories), ordered=ordered)
    if tag.startswith(("datetime64", "timedelta64")):
        return group[name][()].view(np.dtype(tag))
    return group[name][()].astype(np.dtype(tag), copy=False)


def _remove_stale_temporaries(directory: str) -> None:
    """Remove temporary files that killed writers left behind (older than :py:data:`_STALE_TEMPORARY_SECONDS`)."""
    now = time.time()
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if _TMP_MARKER not in name:
            continue
        path = os.path.join(directory, name)
        try:
            if now - os.path.getmtime(path) > _STALE_TEMPORARY_SECONDS:
                os.remove(path)
        except OSError:
            continue


def write_table(path: str, df: pd.DataFrame, params: dict[str, Any]) -> None:
    """Write *df* to *path* atomically, recording the format version and the parameters it was computed with.

    The table is written to a private temporary file next to *path*, flushed to disk, and renamed to *path*; an
    existing file is replaced. Nothing is written if *df* cannot be stored (see :py:func:`validate_dataframe`).

    Args:
        path: Final file of the table (its directory is created if needed).
        df: The table.
        params: JSON-serialisable parameters the table depends on; :py:func:`load_table` compares them.

    Raises:
        TableFormatError: If *df* cannot be stored or *params* is not JSON-serialisable.

    Notes:
        If the directory is not writable or the disk is full, the :py:class:`OSError` propagates and the repository is
        unchanged.
    """
    tags = validate_dataframe(df)
    try:
        params_json = json.dumps(params, sort_keys=True)
    except TypeError as exc:
        raise TableFormatError(f"Table parameters must be JSON-serialisable: {params!r}") from exc
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    _remove_stale_temporaries(directory)
    tmp_path = f"{path}{_TMP_MARKER}{os.getpid()}-{uuid.uuid4().hex[:12]}"
    try:
        with h5py.File(tmp_path, "w", libver=_LIBVER, locking=False) as f:
            f.attrs["idtrack_table_format"] = TABLE_FORMAT_VERSION
            f.attrs["params"] = params_json
            f.attrs["columns"] = json.dumps(list(df.columns))
            f.attrs["tags"] = json.dumps(tags)
            f.attrs["ordered"] = json.dumps(
                [bool(df.iloc[:, i].cat.ordered) if t == "category" else False for i, t in enumerate(tags)]
            )
            index = cast(pd.RangeIndex, df.index)
            f.attrs["index"] = json.dumps([int(index.start), int(index.stop), int(index.step)])
            digest = hashlib.sha256(f.attrs["columns"].encode("utf-8") + f.attrs["tags"].encode("utf-8"))
            for i, tag in enumerate(tags):
                _write_column(f, i, df.iloc[:, i], tag, digest)
            f.attrs["content_sha256"] = digest.hexdigest()
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_table_digest(path: str) -> str | None:
    """Return the SHA-256 of a table's content (columns, dtypes and values as stored), or ``None`` if unreadable.

    Two files with the same digest hold the same table; a graph can record the digests of the tables it was built
    from.

    Args:
        path: The table file.

    Returns:
        Optional[str]: The hexadecimal digest recorded by :py:func:`write_table`.
    """
    try:
        with h5py.File(path, "r", locking=False) as f:
            if int(f.attrs.get("idtrack_table_format", -1)) != TABLE_FORMAT_VERSION:
                return None
            return str(f.attrs["content_sha256"])
    except (OSError, KeyError, ValueError):
        return None


def read_table_params(path: str) -> dict[str, Any] | None:
    """Return the parameters recorded in a table file, or ``None`` if it is absent, unreadable or of another format.

    Args:
        path: The table file.

    Returns:
        Optional[dict[str, Any]]: The parameters given to :py:func:`write_table`.
    """
    if not os.path.isfile(path):
        return None
    try:
        with h5py.File(path, "r", locking=False) as f:
            if int(f.attrs.get("idtrack_table_format", -1)) != TABLE_FORMAT_VERSION:
                return None
            return json.loads(f.attrs["params"])
    except (OSError, KeyError, ValueError):
        return None


def read_table(path: str) -> pd.DataFrame:
    """Read a table written by :py:func:`write_table`.

    Args:
        path: The table file.

    Returns:
        pandas.DataFrame: The table as written (see the module docstring for text dtypes).

    Raises:
        FileNotFoundError: If *path* does not exist.
        TableFormatError: If the file is of another format version.

    Notes:
        A file that cannot be read as HDF5 raises h5py's :py:class:`OSError`.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r", locking=False) as f:
        version = int(f.attrs.get("idtrack_table_format", -1))
        if version != TABLE_FORMAT_VERSION:
            raise TableFormatError(f"{path!r} has table format {version}, expected {TABLE_FORMAT_VERSION}.")
        columns = json.loads(f.attrs["columns"])
        tags = json.loads(f.attrs["tags"])
        ordered = json.loads(f.attrs["ordered"])
        start, stop, step = json.loads(f.attrs["index"])
        data = {col: _read_column(f, i, tags[i], ordered[i]) for i, col in enumerate(columns)}
    index = pd.RangeIndex(start, stop, step)
    if not columns:
        return pd.DataFrame(index=index)
    return pd.DataFrame(data, index=index, columns=columns, copy=False)


def load_table(path: str, params: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame | None:
    """Return the table stored at *path* if it was computed with exactly *params*, else ``None``.

    ``None`` means "compute the table": the file is absent, of another format version, made with other parameters, or
    unreadable (a damaged file is reported once in the log and then replaced by the caller's next write).

    Args:
        path: The table file.
        params: The parameters the caller needs (compared with those recorded by :py:func:`write_table`).
        logger: Where to report a damaged or mismatching file; the module logger by default.

    Returns:
        Optional[pandas.DataFrame]: The table, or ``None``.
    """
    log = logger or _log
    if not os.path.isfile(path):
        return None
    try:
        with h5py.File(path, "r", locking=False) as f:
            version = int(f.attrs.get("idtrack_table_format", -1))
            stored = json.loads(f.attrs["params"]) if version == TABLE_FORMAT_VERSION else None
    except (OSError, KeyError, ValueError) as exc:
        log.warning("Cached table %r cannot be read (%s); it is computed again.", path, exc)
        return None
    if stored is None:
        log.info("Cached table %r has another storage format; it is computed again.", path)
        return None
    expected = json.loads(json.dumps(params, sort_keys=True))
    if stored != expected:
        log.warning(
            "Cached table %r was computed with %s, but %s is asked for; it is computed again.", path, stored, expected
        )
        return None
    try:
        return read_table(path)
    except (OSError, KeyError, ValueError) as exc:
        log.warning("Cached table %r cannot be read (%s); it is computed again.", path, exc)
        return None
