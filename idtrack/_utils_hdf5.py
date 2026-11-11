#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import ast
import logging
import os
import tempfile
import types
from collections.abc import Iterator, Sequence
from typing import cast

import h5py
import numpy as np
import pandas as pd

from idtrack._db import DB


def _is_missing_scalar(value) -> bool:
    """Return whether *value* is a missing scalar (``None``, ``NaN``, ``pd.NA`` or ``NaT``).

    Args:
        value: Any object.

    Returns:
        bool: ``True`` for a missing scalar.
    """
    return bool(pd.api.types.is_scalar(value) and pd.isna(value))


def _text_kind(dtype) -> str | None:
    """Classify a dtype as stored text, independently of the pandas version.

    Text is stored under one of two tags, which are also the tags older versions of this module wrote:

    * ``"object"``: the default text dtype of pandas, i.e. ``object`` (pandas 2) or ``str``
      (``StringDtype(na_value=nan)``, the default of pandas 3); missing values are ``NaN``.
    * ``"string"``: the opt-in ``StringDtype`` with ``pd.NA`` as missing value.

    Args:
        dtype: A column or index dtype.

    Returns:
        Optional[str]: ``"object"``, ``"string"``, or ``None`` when *dtype* is not a text dtype.
    """
    if isinstance(dtype, pd.StringDtype):
        return "string" if dtype.na_value is pd.NA else "object"
    if pd.api.types.is_object_dtype(dtype):
        return "object"
    return None


def _default_text_dtype():
    """Return the dtype the running pandas infers for text (``object`` in pandas 2, ``str`` in pandas 3).

    Returns:
        The dtype of ``pd.Series(["a"])``.
    """
    return pd.Series(["a"]).dtype


def _encode_text(values) -> np.ndarray:
    """Encode text values (strings or missing values) for a UTF-8 dataset, missing values as the placeholder.

    Args:
        values: Strings or missing values.

    Returns:
        numpy.ndarray: The values as UTF-8 strings, :py:data:`DB.placeholder_na` for missing values.
    """
    arr = np.asarray(values, dtype=object)
    encoded = arr.copy()
    encoded[pd.isna(arr)] = DB.placeholder_na
    return np.asarray(encoded, dtype=DB.UTF8_STR)


def _decode_text(dataset: h5py.Dataset, kind: str):
    """Decode a UTF-8 dataset written by :py:func:`_encode_text` into text of the given kind (see :py:func:`_text_kind`).

    ``"object"`` text is returned in the default text dtype of the running pandas with ``NaN`` for missing values,
    which is what pandas itself produces for freshly read text (e.g. by :py:func:`pandas.read_csv`), so a table read
    back from the cache equals the table computed afresh under the same pandas. ``"string"`` text is returned as
    ``StringDtype`` with ``pd.NA``.

    Args:
        dataset: The UTF-8 dataset.
        kind: ``"object"`` or ``"string"``.

    Returns:
        numpy.ndarray | pandas.api.extensions.ExtensionArray: The decoded values.
    """
    arr = np.asarray(dataset.asstr()[()], dtype=object)
    missing = arr == DB.placeholder_na
    if kind == "string":
        arr[missing] = pd.NA
        return pd.array(cast(Sequence[object], arr), dtype="string")
    arr[missing] = np.nan
    text_dtype = _default_text_dtype()
    if pd.api.types.is_object_dtype(text_dtype):
        return arr
    return pd.array(cast(Sequence[object], arr), dtype=text_dtype)


def _temporal_as_text(values: pd.Series | pd.Index) -> np.ndarray:
    """Render datetime or timedelta values as text for a UTF-8 dataset; missing values become ``"NaT"``.

    ``astype(str)`` keeps missing values missing under pandas 3 (and turns them into ``"NaT"`` under pandas 2), so
    they are filled explicitly; :py:func:`pandas.to_datetime` and :py:func:`pandas.to_timedelta` read ``"NaT"`` back
    as missing.

    Args:
        values: Datetime or timedelta values.

    Returns:
        numpy.ndarray: The values as UTF-8 strings.
    """
    text = pd.Series(values).astype(str).to_numpy(dtype=object)
    text[pd.isna(text)] = "NaT"
    return np.asarray(text, dtype=DB.UTF8_STR)


def to_hdf(path: str, key: str, df: pd.DataFrame, mode: str = "a", compression: str | int | None = 9):
    """Persist a :py:class:`pandas.DataFrame` to an HDF5 file while retaining rich pandas metadata.

    This helper exists because :py:meth:`pandas.DataFrame.to_hdf` loses information for complex dtypes
    (categoricals, strings with ``pd.NA``, multi-level indexes, etc.) across Pandas/HDF5 versions.  The
    implementation serialises every logical layer—column *labels*, *dtypes*, and the actual *data*—into
    separate datasets or attributes so that :py:func:`read_hdf` can perform a loss-less round-trip, even on
    machines with mismatched library versions.

    Args:
        path (str): Filesystem location of the HDF5 container.  The file is created if it does
            not yet exist and *mode* permits writing.
        key (str): Name of the group that will hold the DataFrame inside *path*.  Leading ``"/"`` is stripped
            for consistency with :py:class:`pandas.HDFStore` behaviour.
        df (pandas.DataFrame): Table to serialise.  Columns may be numeric, boolean, string/object, datetime,
            timedelta, or :py:class:`~pandas.CategoricalDtype`.  Unsupported dtypes raise
            :py:class:`ValueError <builtins.ValueError>`.
        mode (str): File access flag forwarded to :py:class:`pandas.HDFStore`.  Must be one of
            ``"a"``, ``"w"``, or ``"r+"``; defaults to ``"a"`` (create if missing, otherwise append/replace).
        compression (Optional[Union[str, int]]): Compression setting forwarded to :py:class:`h5py.File`.
            Provide an integer (0-9) for gzip-like deflate levels, a string such as ``"lzf"`` or ``"gzip"``,
            or ``None`` to disable compression.  The default ``9`` maximises file-size reduction.
    """
    validate_dataframe(df)  # Validate DataFrame structure

    with HDFStore(path, mode=mode) as store:
        if key in store:
            store.remove(key)

        # Clean key (remove leading slash if present)
        clean_key = key.lstrip("/")
        grp = store.f.create_group(clean_key)

        # Save column metadata
        _save_column_metadata(grp, df)

        # Save index metadata and data
        _save_index_data(grp, df, compression)

        # Save column data with proper dtype preservation
        _save_column_data(grp, df, compression)


def read_hdf(path: str, key: str, mode: str = "r") -> pd.DataFrame:
    """Load a DataFrame previously written by :py:func:`to_hdf`, restoring all metadata faithfully.

    The function reverses the layered serialisation scheme used by :py:func:`to_hdf`.  It reconstructs
    column labels, dtypes (including categoricals), index names, and the original ``columns.name`` so that
    the returned DataFrame is equivalent to the object that was saved—modulo unavoidable floating-point
    representation differences. Text columns (and a text index) are the exception by design: ``object`` and
    pandas 3's default ``str`` text both come back in the default text dtype of the *running* pandas, with ``NaN``
    for missing values (``string`` text comes back as ``string`` with ``pd.NA``), so that a cached table equals the
    table computed afresh under the same pandas, whichever pandas wrote the file.

    Args:
        path (str): HDF5 file that contains the stored DataFrame.
        key (str): Group name under which the DataFrame was stored.  Leading ``"/"`` is ignored.
        mode (str): File access flag forwarded to :py:class:`pandas.HDFStore`.  Defaults to ``"r"``
            (read-only).  Supply ``"r+"`` for concurrent reads and writes.

    Returns:
        pandas.DataFrame: The deserialised table with its original index, column names, dtypes, and metadata
            fully restored.

    Raises:
        KeyError: If *key* is not present inside *path*.
    """
    with HDFStore(path, mode=mode) as store:
        clean_key = key.lstrip("/")
        if clean_key not in store.f:
            raise KeyError(f"Key {key!r} not found in HDF5 file")

        grp = store.f[clean_key]

        # Load column metadata
        columns, dtypes, columns_name = _load_column_metadata(grp)

        # Load index
        index, index_names = _load_index_data(grp)

        # Load column data
        df = _load_column_data(grp, columns, dtypes)

        # Set index and metadata
        df.index = index
        df.index.names = index_names
        df.columns.name = columns_name

        return df


def validate_dataframe(df: pd.DataFrame):
    """Validate a :py:class:`pandas.DataFrame` before HDF5 serialization.

    The helper enforces a strictly controlled schema so that data written by
    :py:func:`idtrack._utils_hdf5.to_hdf` can be loss-lessly restored by its
    companion :py:func:`idtrack._utils_hdf5.read_hdf`.  It checks every column
    and index for:

    * allowable dtypes (numeric, boolean, text — ``object``, ``str`` or ``string`` —, various ``datetime64`` /
      ``timedelta64`` resolutions, or properly formed categoricals);
    * object columns and an object index containing only strings or missing values;
    * categorical columns whose *categories* are exclusively strings or NA;
    * index dtype restricted to integer, unsigned integer, float, datetime, or text;
    * valid column and index *names* (‶str‶, ‶int‶, ‶float‶, or ``None`` only).

    Any deviation raises a descriptive :py:class:`ValueError`, preventing silent
    corruption at write time and making data contracts explicit.

    Args:
        df (pandas.DataFrame): DataFrame to inspect—must already be *in-memory*.
            All columns and the index are validated against the constraints
            listed above.

    Raises:
        ValueError: If *any* of the following conditions are met:

            * A column's dtype is outside the supported set.
            * A categorical column contains categories that are neither strings
              nor NA.
            * An object-dtype column contains non-string, non-missing scalars.
            * The index dtype is unsupported.
            * A column name is not ``str`` or ``int``.
            * ``df.columns.name`` or ``df.index.name`` is not ``None``, ``str``,
              ``int``, or ``float``.
    """
    # Exactly supported non-text, non-categorical dtypes (full matching).
    supported_dtypes = {
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
        "float128",
        "bool",
        "datetime64[ns]",
        "datetime64[us]",
        "datetime64[ms]",
        "datetime64[s]",
        "timedelta64[ns]",
        "timedelta64[us]",
        "timedelta64[ms]",
        "timedelta64[s]",
    }

    for col, dtype in df.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            # Values in categorical are backed by codes, so checking the categories is sufficient.
            categories = df[col].cat.categories
            bad = [cat for cat in categories if not (pd.isna(cat) or isinstance(cat, str))]
            if bad:
                raise ValueError(f"Categorical column {col!r} has non-string categories: {bad}")

        elif _text_kind(dtype) is not None:
            # pandas' string dtypes hold only strings or missing values by construction; object columns are text
            # columns and must hold only strings or missing values.
            if pd.api.types.is_object_dtype(dtype):
                col_series = df[col]
                mask = col_series.map(lambda x: isinstance(x, str) or _is_missing_scalar(x)).to_numpy(dtype=bool)
                if not mask.all():
                    bad_values = col_series[~mask].unique()
                    raise ValueError(f"Object column {col!r} contains non-string non-NA values: {bad_values}")

        elif str(dtype) not in supported_dtypes:
            raise ValueError(f"Unsupported dtype {dtype!r} for column {col!r}")

    # Check index dtype
    if not (df.index.dtype.kind in ["i", "u", "f", "M"] or _text_kind(df.index.dtype) is not None):
        raise ValueError(f"Unsupported index dtype {df.index.dtype!r}")
    if pd.api.types.is_object_dtype(df.index.dtype):
        bad_index = [x for x in df.index if not (isinstance(x, str) or _is_missing_scalar(x))]
        if bad_index:
            raise ValueError(f"Object index contains non-string non-NA values: {bad_index[:10]}")

    # Check column names
    for col_name in df.columns:
        if not isinstance(col_name, (str, int)):
            raise ValueError(f"Column name must be string or int, got {type(col_name)}")

    # Check columns.name and index.name
    for name in [df.columns.name, df.index.name]:
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(f"Index/columns name must be None, string, int, or float, got {type(name)}")


def _save_column_metadata(grp: h5py.Group, df: pd.DataFrame):
    """Persist column metadata—names, logical dtypes, and :py:attr:`~pandas.Index.name`—to an HDF5 group.

    This helper is part of the private DataFrame ↔ HDF5 round-trip layer used by the IDTrack persistence
    stack.  It serialises *structural* information about the columns without touching the data values
    themselves, allowing :py:meth:`_load_column_metadata` and :py:meth:`_load_column_data` to later restore a
    faithful `pandas.DataFrame`.

    Column names are stored as UTF-8 strings in a dataset called ``"column_names"``.  Logical dtypes are
    flattened into human-readable strings—primitive NumPy dtypes remain unchanged, while categoricals are
    encoded as ``"category|<categories>|<ordered>"``—and recorded in ``"column_dtypes"``.  The
    attribute ``"columns_name"`` preserves the higher-level `DataFrame.columns.name` label if one was set.

    Args:
        grp (h5py.Group): Writable HDF5 group in which the metadata datasets will be created.  The caller is
            responsible for opening and closing the parent file or group.
        df (pandas.DataFrame): Data frame whose *column* metadata (not values) should be saved.  Each column
            must have a unique, hashable name.
    """
    # Save column names
    col_names = [str(col) for col in df.columns]
    grp.create_dataset("column_names", data=np.array(col_names, dtype=DB.UTF8_STR), dtype=DB.UTF8_STR)

    # Save column dtypes
    dtypes_info = []
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            # Store category information
            categories = dtype.categories.tolist()
            ordered = dtype.ordered
            dtypes_info.append(f"category|{categories}|{ordered}")
        elif _text_kind(dtype) is not None:
            dtypes_info.append(str(_text_kind(dtype)))
        else:
            dtypes_info.append(str(dtype))

    grp.create_dataset("column_dtypes", data=np.array(dtypes_info, dtype=DB.UTF8_STR), dtype=DB.UTF8_STR)

    # Save columns.name
    columns_name = str(df.columns.name) if df.columns.name is not None else ""
    grp.attrs["columns_name"] = columns_name


def _save_index_data(grp: h5py.Group, df: pd.DataFrame, compression):
    """Persist the DataFrame index values and associated metadata to an HDF5 group.

    The function serialises both single- and multi-level indexes, converting specialised dtypes into
    interoperable on-disk representations:

    * **DatetimeIndex** - converted to ISO-8601 strings to avoid platform-specific endianness issues.
    * **CategoricalIndex** - stored as integer codes plus a UTF-8 encoded ``"index_categories"`` dataset and an
      ``"index_dtype"`` attribute such as ``"category|True"`` (where the flag indicates ordering).
    * **Other dtypes** - written verbatim.

    Index names are always written to ``"index_names"``; missing names become empty strings so that the original
    `None`/string distinction can be restored later.

    Args:
        grp (h5py.Group): Writable HDF5 group that will receive ``"index_*"`` datasets and attributes.
        df (pandas.DataFrame): The frame whose index and index-level names are to be serialised.
        compression: Value forwarded directly to :py:meth:`h5py.Group.create_dataset`.  Accepts anything the
            HDF5 backend does (e.g., ``"gzip"``, ``"lzf"``, an integer compression level, or *None*).
    """
    # Save index names
    index_names = [str(name) if name is not None else "" for name in df.index.names]
    grp.create_dataset("index_names", data=np.array(index_names, dtype=DB.UTF8_STR), dtype=DB.UTF8_STR)

    # Save index data
    if pd.api.types.is_datetime64_any_dtype(df.index):
        # Convert datetime to string for reliable storage
        grp.create_dataset("index_data", data=_temporal_as_text(df.index), dtype=DB.UTF8_STR, compression=compression)
        grp.attrs["index_dtype"] = str(df.index.dtype)
    elif isinstance(df.index.dtype, pd.CategoricalDtype):
        # Store categorical index
        codes = df.index.codes
        cats = [str(x) for x in df.index.categories.tolist()]
        grp.create_dataset("index_data", data=codes, compression=compression)
        grp.create_dataset("index_categories", data=cats, dtype=DB.UTF8_STR)
        grp.attrs["index_dtype"] = f"category|{df.index.ordered}"
    elif _text_kind(df.index.dtype) is not None:
        # Text index (object, pandas 3's default ``str`` or ``string``), tagged like text columns.
        grp.create_dataset(
            "index_data",
            data=_encode_text(df.index.to_numpy(dtype=object)),
            dtype=DB.UTF8_STR,
            compression=compression,
        )
        grp.attrs["index_dtype"] = str(_text_kind(df.index.dtype))
    else:
        grp.create_dataset("index_data", data=df.index.values, compression=compression)
        grp.attrs["index_dtype"] = str(df.index.dtype)


def _save_column_data(grp: h5py.Group, df: pd.DataFrame, compression: str | int | None):
    """Serialise each column's actual values to an HDF5 sub-group while preserving logical dtypes.

    A child group named ``"data"`` is created beneath *grp*, and one or more datasets are generated per column
    depending on its dtype:

    * **Categorical** → ``"<col>_codes"`` (int32) + ``"<col>_categories"`` (UTF-8 strings).
    * **Datetime64** → stored as ISO-8601 strings in ``"<col>"``.
    * **String / object** → strings with :py:data:`DB.placeholder_na` in place of missing values.
    * **Numeric / boolean** → 1-to-1 binary copy into ``"<col>"``.

    The routine deliberately leaves *grp*'s metadata untouched—see :py:meth:`_save_column_metadata` for that.

    Args:
        grp (h5py.Group): Parent group that **must not** already contain a child called ``"data"``.
        df (pandas.DataFrame): Source data frame.  Its columns should match the metadata written earlier.
        compression (Optional[Union[str, int]]): Compression option to pass straight through to HDF5 dataset creation.
    """
    data_grp = grp.create_group("data")

    for col in df.columns:
        col_data = df[col]
        col_key = f"col_{col}"

        if isinstance(col_data.dtype, pd.CategoricalDtype):
            # Store categorical data as codes + categories
            codes = col_data.cat.codes.values
            categories = col_data.cat.categories.tolist()
            data_grp.create_dataset(f"{col_key}_codes", data=codes, compression=compression)
            data_grp.create_dataset(
                f"{col_key}_categories",
                data=np.array([str(cat) for cat in categories], dtype=DB.UTF8_STR),
                dtype=DB.UTF8_STR,
            )

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # Convert datetime to string for reliable storage
            data_grp.create_dataset(
                col_key, data=_temporal_as_text(col_data), dtype=DB.UTF8_STR, compression=compression
            )

        elif pd.api.types.is_timedelta64_dtype(col_data):
            # Convert timedeltas to string for reliable storage (HDF5 has no native timedelta dtype)
            data_grp.create_dataset(
                col_key, data=_temporal_as_text(col_data), dtype=DB.UTF8_STR, compression=compression
            )

        elif _text_kind(col_data.dtype) is not None:
            # Text (object, pandas 3's default ``str``, or ``string``): missing values become the placeholder.
            data_grp.create_dataset(
                col_key, data=_encode_text(col_data.to_numpy(dtype=object)), dtype=DB.UTF8_STR, compression=compression
            )

        else:
            # Numeric, boolean, or other simple types
            data_grp.create_dataset(col_key, data=col_data.values, compression=compression)


def _load_column_metadata(grp: h5py.Group) -> tuple[list[str], list[str | pd.CategoricalDtype], str | None]:
    """Load column names, logical dtypes, and `DataFrame.columns.name` from an HDF5 group.

    This is the inverse of :py:meth:`_save_column_metadata`.  It reconstructs high-level dtype objects—
    converting categorical encodings back into :py:class:`pandas.CategoricalDtype` instances—so that
    :py:meth:`_load_column_data` can re-cast raw numpy arrays into their original pandas dtypes.

    Args:
        grp (h5py.Group): Group previously populated by :py:meth:`_save_column_metadata`.

    Returns:
        tuple[list[str], list[Union[str, pandas.CategoricalDtype]], Optional[str]]:
            * **column_names** - List of column labels in original order.
            * **dtypes** - Parallel list whose elements are either primitive dtype strings or
              :py:class:`pandas.CategoricalDtype` objects.
            * **columns_name** - The ``DataFrame.columns.name`` label, or *None* if it was originally unset.
    """
    # Load column names
    col_names = [name.decode(DB.UTF8) for name in grp["column_names"][()]]

    # Load column dtypes
    dtype_strs = [dt.decode(DB.UTF8) for dt in grp["column_dtypes"][()]]

    # Parse dtypes
    dtypes = []
    for dtype_str in dtype_strs:
        if dtype_str.startswith("category|"):
            # Parse category information
            parts = dtype_str.split("|", 2)
            categories = ast.literal_eval(parts[1])
            ordered = parts[2] == "True"
            dtypes.append(pd.CategoricalDtype(categories=categories, ordered=ordered))
        else:
            dtypes.append(dtype_str)

    # Load columns.name
    columns_name = grp.attrs.get("columns_name", "")
    columns_name = columns_name if columns_name != "" else None

    return col_names, dtypes, columns_name


def _load_index_data(grp: h5py.Group) -> tuple[pd.Index, list[str | None]]:
    """Reconstruct a pandas Index (and its names) from HDF5-serialised form.

    The routine reads the artefacts created by :py:meth:`_save_index_data`, automatically selecting the correct
    deserialisation path based on the stored ``"index_dtype"`` attribute.

    Args:
        grp (h5py.Group): Group containing ``"index_*"`` datasets and attributes.

    Returns:
        tuple[pandas.Index, list[Optional[str]]]:
            * **index** - The restored :py:class:`pandas.Index`, :py:class:`pandas.DatetimeIndex`, or
              :py:class:`pandas.CategoricalIndex`, depending on the original type.
            * **index_names** - List of level names (items may be *None*).
    """
    # Load index names
    index_names = [name.decode(DB.UTF8) for name in grp["index_names"][()]]
    index_names = [name if name != "" else None for name in index_names]

    # Load index data based on dtype
    index_dtype = grp.attrs.get("index_dtype", "object")
    index_ds = grp["index_data"]

    if index_dtype.startswith("datetime64"):
        # Convert string back to datetime, with the recorded unit.
        index = pd.DatetimeIndex(pd.to_datetime(list(index_ds.asstr()[()]), format="ISO8601")).astype(index_dtype)
    elif index_dtype.startswith("category"):
        # Reconstruct categorical index
        ordered = index_dtype.split("|")[1] == "True"
        categories = list(grp["index_categories"].asstr()[()])
        index = pd.CategoricalIndex.from_codes(index_ds[()], categories=categories, ordered=ordered)
    elif index_dtype in {"object", "string"}:
        # Text index (see `_text_kind`).
        values = _decode_text(index_ds, index_dtype)
        index = pd.Index(values, dtype=values.dtype)
    else:
        index = pd.Index(index_ds[()], dtype=index_dtype)

    return index, index_names


def _load_column_data(
    grp: h5py.Group, columns: Sequence[str], dtypes: Sequence[str | pd.CategoricalDtype]
) -> pd.DataFrame:
    """Load column datasets and rebuild a DataFrame, honouring the original logical dtypes.

    This complements :py:meth:`_load_column_metadata`.  It expects that *columns* and *dtypes* come straight from
    that function and therefore align perfectly with the datasets under the ``"data"`` sub-group of *grp*.

    Special-case handling mirrors the rules in :py:meth:`_save_column_data`:

    * **Categorical** - combines stored codes and categories.
    * **Datetime64** - parses ISO-8601 strings back into pandas nanosecond-resolution timestamps.
    * **Text** - restores the dtype as described in :py:func:`_decode_text`; :py:data:`DB.placeholder_na` tokens
      become missing values (``NaN``, or ``pd.NA`` for ``string``).

    Args:
        grp (h5py.Group): Parent group whose ``"data"`` subgroup contains the column datasets.
        columns (Sequence[str]): Column labels in the order they should appear in the output frame.
        dtypes (Sequence[Union[str, pandas.CategoricalDtype]]): Dtype descriptors corresponding 1-to-1 with *columns*.

    Returns:
        pandas.DataFrame: Reconstructed data frame with all logical dtypes restored.
    """
    data_grp = grp["data"]
    data_dict: dict[str, pd.Series] = {}

    for col, dtype in zip(columns, dtypes):
        col_key = f"col_{col}"

        if isinstance(dtype, pd.CategoricalDtype):
            # Reconstruct categorical data
            codes = data_grp[f"{col_key}_codes"][()]
            categories = list(data_grp[f"{col_key}_categories"].asstr()[()])
            data_dict[col] = pd.Series(pd.Categorical.from_codes(codes, categories=categories, ordered=dtype.ordered))

        elif str(dtype).startswith("datetime64"):
            # Convert string back to datetime, with the recorded unit.
            values = pd.to_datetime(list(data_grp[col_key].asstr()[()]), format="ISO8601")
            data_dict[col] = pd.Series(values).astype(np.dtype(str(dtype)))

        elif str(dtype).startswith("timedelta64"):
            # Convert string back to timedelta, with the recorded unit.
            values = pd.to_timedelta(list(data_grp[col_key].asstr()[()]))
            data_dict[col] = pd.Series(values).astype(np.dtype(str(dtype)))

        elif str(dtype) in ("object", "string"):
            # Text (see `_text_kind`): the placeholder becomes the missing value of the restored dtype.
            values = _decode_text(data_grp[col_key], str(dtype))
            data_dict[col] = pd.Series(values, dtype=values.dtype)

        else:
            # Numeric, boolean, or other simple types
            data_dict[col] = pd.Series(data_grp[col_key][()], dtype=dtype)

    # Every column already has its final dtype; building the frame from Series keeps them.
    return pd.DataFrame(data_dict)


class HDFStore:
    """Provide a context-manager wrapper around :py:class:`h5py.File` that emulates ``pandas.HDFStore``.

    This helper delivers a lightweight, PyTables-free alternative to :py:class:`pandas.HDFStore` while preserving
    its familiar, dictionary-like API.  It offers keyed access to *top-level* groups or datasets, convenient
    helpers such as :py:meth:`~HDFStore.keys`, membership tests, and safe resource handling via the context-
    manager protocol.  Internally it delegates all I/O to a single :py:class:`h5py.File` instance that becomes
    available through the public :py:attr:`~HDFStore.f` attribute after entering the context manager.

    The class is primarily intended for *idtrack* utilities that need to read or write HDF5 data but cannot rely
    on PyTables.  Because only the first level of the hierarchy is exposed, traversal of nested groups must be
    performed manually through the exposed :py:attr:`~HDFStore.f` handle.

    Attributes:
        path (str): Absolute or relative path to the ``.h5`` container that will be opened.
        mode (str): File mode passed verbatim to :py:class:`h5py.File`.  One of ``'r'``, ``'r+'``, ``'w'``,
            ``'x'``, or ``'a'`` (see *h5py* documentation for semantics).
        f (Optional[h5py.File]): Live file handle once the context has been
            entered; ``None`` before entry and after exit.
    """

    path: str
    mode: str
    f: h5py.File

    def __init__(self, path: str, mode: str = "r") -> None:
        """Create an unopened :py:class:`~HDFStore` descriptor.

        The constructor records *path* and *mode* only; the actual :py:class:`h5py.File` is opened in
        :py:meth:`__enter__` to support ``with``-statement usage.

        Args:
            path (str): Filesystem location of the HDF5 container.
            mode (str): File mode accepted by :py:class:`h5py.File`.  Defaults to ``'r'``.
        """
        self.path = path
        self.mode = mode
        self.f: h5py.File | None = None

    def __enter__(self) -> "HDFStore":
        """Open the file and return *self* for use inside a ``with`` block.

        Returns:
            HDFStore: The identical instance, now carrying an open :py:class:`h5py.File`
                in :py:attr:`~HDFStore.f`.
        """
        # Default format bounds (no `libver="latest"`): files stay readable by older HDF5 libraries (HDF5 2.x with
        # "latest" writes layout messages HDF5 1.14 cannot read), so a cache moves between environments.
        self.f = h5py.File(self.path, self.mode)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Flush and close :py:attr:`~HDFStore.f`, guaranteeing resource release.

        Any exception raised inside the ``with`` block is propagated unchanged; this method performs no additional
        error handling beyond the unconditional close.

        Args:
            exc_type (Optional[type]): Exception class if an exception occurred, else ``None``.
            exc_val (Optional[BaseException]): Raised exception instance, or ``None``.
            exc_tb (Optional[types.TracebackType]): Traceback of the exception, or ``None``.
        """
        if self.f is not None:
            self.f.flush()
            self.f.close()

    def __contains__(self, key: str) -> bool:
        """Return ``True`` when *key* exists as a top-level group or dataset.

        Args:
            key (str): Name to probe.  A leading ``'/'`` is ignored to match ``pandas.HDFStore``.

        Returns:
            bool: ``True`` if the node exists, ``False`` otherwise.
        """
        return key.lstrip("/") in self.f

    def __iter__(self) -> Iterator[str]:
        """Yield every top-level key, each prefixed with ``'/'`` for ``pandas``-style parity.

        Yields:
            str: Next key in the order provided by :py:class:`h5py.File`.
        """
        yield from self.keys()

    def keys(self) -> list[str]:
        """Return a list of all top-level keys currently present in the container.

        Returns:
            list[str]: Sorted list of fully qualified key names, each starting with ``'/'``.
        """
        return [f"/{name}" for name in self.f.keys()]

    def remove(self, key: str) -> None:
        """Delete a group or dataset referenced by *key* from the file.

        The operation is performed immediately and is irreversible.  Attempting to delete a missing key triggers
        a :py:class:`KeyError` rather than silently failing, mirroring standard dictionary semantics.

        Args:
            key (str): Identifier of the node to remove.  A leading ``'/'`` is stripped before lookup.

        Raises:
            KeyError: If *key* does not exist in the open file.
        """
        name = key.lstrip("/")
        if name not in self.f:
            raise KeyError(f"No such key in HDF5 file: {key}")
        del self.f[name]


def check_h5_key(file_path: str, key: str) -> bool:
    """Verify that *key* exists in an HDF5 file.

    This lightweight helper is a safe, read-only probe used throughout the package to decide whether an HDF5
    group/dataset should be (re)written or removed.  It first checks basic file accessibility with
    :py:data:`os.R_OK`; if the file cannot be read it short-circuits to ``False`` instead of raising, making it
    suitable for “does it exist?” flows during automatic cache housekeeping.

    Args:
        file_path (str): Absolute or relative path to the ``.h5`` file to inspect.
        key (str): HDF5 node path (e.g. ``"/analysis/results"``) whose presence should be tested.

    Returns:
        bool: ``True`` when *file_path* is readable **and** contains *key*; ``False`` otherwise.

    Notes:
        The function never raises—failure to read the file or open the store is converted into a ``False``
        return so that callers can decide how to proceed without a try/except wrapper.
    """
    if not os.access(file_path, os.R_OK):
        return False
    with HDFStore(file_path, mode="r") as f:
        return key in f


def repack_hdf5(path: str) -> None:
    """Repack an HDF5 file in place to reclaim fragmented space.

    The HDF5 “append” write pattern used by pandas and h5py can leave gaps over time.  This utility copies the
    complete object graph (all groups, datasets, and attributes) into a brand-new temporary file created in the
    same directory, then atomically replaces the original file.  The operation is lossless; object names,
    hard/soft links, compression filters, and chunk settings are preserved by relying on
    :py:meth:`h5py.Group.copy`.

    Args:
        path (str): Path to an existing, writable ``.h5`` file that should be compacted.

    Notes:
        The function creates a file named ``<basename>XXXXXX.repack.h5`` (where ``XXXXXX`` is a random suffix)
        next to *path*.  The temporary file is removed automatically on success **or** on most errors, but
        callers running in restricted environments might want to perform an extra cleanup pass.
    """
    base_dir, base_name = os.path.split(path)
    fd, tmp_path = tempfile.mkstemp(prefix=base_name, suffix=".repack.h5", dir=base_dir)
    os.close(fd)

    try:
        with h5py.File(path, "r") as src, h5py.File(tmp_path, "w") as dst:
            # Copy all root-level attributes
            for attr_name, attr_val in src.attrs.items():
                dst.attrs[attr_name] = attr_val

            # Recursively copy all objects in the file (groups, datasets, references, etc.)
            for name in src:
                src.copy(
                    source=name,
                    dest=dst,
                    name=name,
                    shallow=False,
                    expand_soft=True,
                    expand_external=True,
                    expand_refs=True,
                )

            dst.flush()

        # Atomically replace the old file
        os.replace(tmp_path, path)

    finally:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def export_disk(df: pd.DataFrame | pd.Series, hierarchy: str, file_path: str, overwrite: bool, logger: logging.Logger):
    """Persist a pandas object to an HDF5 store under a hierarchical key.

    The helper centralises all on-disk writes so that every table goes through the same “check → optionally
    delete → write” procedure.  By removing an existing node before rewriting, it prevents the HDF5 file from
    ballooning due to internal free-space fragmentation—a common issue when tables are frequently refreshed.

    Args:
        df (Union[pandas.DataFrame, pandas.Series]): The table or one-dimensional array to store.
        hierarchy (str): HDF5 key that uniquely identifies *df* inside *file_path* (e.g. ``"/scores/latest"``).
        file_path (str): Destination ``.h5`` file.  Created automatically if it does not yet exist.
        overwrite (bool): If ``True`` the existing node (when present) is dropped before writing.  If ``False``
            and *hierarchy* already exists, nothing is written.
        logger (logging.Logger): Application-level logger used to report deletions and writes.

    Notes:
        The function deliberately opens the store in *append* mode so that other keys remain untouched.  All
        writes use pandas :py:meth:`pandas.DataFrame.to_hdf` / :py:meth:`pandas.Series.to_hdf` with default
        settings (tables format, no compression); adjust upstream if different storage parameters are needed.
    """
    base_file_path = os.path.basename(file_path)

    if not os.access(file_path, os.R_OK) or overwrite or (not check_h5_key(file_path, hierarchy)):
        # Remove the file first to prevent hdf5 file to go arbitrarily larger after writing.
        if check_h5_key(file_path, hierarchy) or overwrite:
            with HDFStore(file_path, mode="a") as f:
                if hierarchy in f:
                    logger.info(
                        f"Following file is being removed: `{os.path.basename(file_path)}` "
                        f"with key `{hierarchy}`. This could cause hdf5 file to not reclaim the "
                        f"newly emptied disk space."
                    )
                    f.remove(hierarchy)
        # Then save the dataframe under the root.
        logger.info(f"Exporting to the following file `{base_file_path}` with key `{hierarchy}`")
        to_hdf(df=df, path=file_path, key=hierarchy, mode="a")


def read_exported(hierarchy: str, file_path: str) -> pd.DataFrame | pd.Series:
    """Load a previously exported pandas object from an HDF5 store.

    This read side-kick to :py:func:`export_disk` hides the repetitive boilerplate of access checks and
    key-existence validation.  By separating the “does it exist?” logic into :py:func:`check_h5_key`, the
    function ensures a clean, one-line API for callers who only need the data back.

    Args:
        hierarchy (str): HDF5 key that identifies the stored table to fetch.
        file_path (str): Path to the ``.h5`` file created via :py:func:`export_disk`.

    Returns:
        Union[pandas.DataFrame, pandas.Series]: The table referenced by *hierarchy* exactly as written.

    Raises:
        FileNotFoundError: If *file_path* cannot be read.
        KeyError: If the requested *hierarchy* is absent in the store.

    Notes:
        The call opens the file in *read-only* mode and therefore never modifies timestamps or causes locking
        conflicts with writers operating on the same file.
    """
    if not os.access(file_path, os.R_OK):
        raise FileNotFoundError(f"The file {file_path!r} does not exist or is not readable.")

    if not check_h5_key(file_path, hierarchy):
        raise KeyError(f"Key {hierarchy!r} not found in HDF5 file {file_path!r}.")

    df = read_hdf(path=file_path, key=hierarchy, mode="r")
    return df
