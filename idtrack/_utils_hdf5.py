#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import logging
import os
import tempfile
from collections.abc import Iterator
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd

from idtrack._db import DB


def to_hdf(path: str, key: str, df: pd.DataFrame, mode: str = "a", compression=9):
    """Write a DataFrame to an HDF5 file under the group `key`.

    Args:
        path: Path to HDF5 file.
        key: Group name under which to store the DataFrame.
        df: DataFrame to store.
        mode: File mode.  {'a','w','r+'}, default 'a'.
        compression: Compression to use for datasets. None or str
    """
    _validate_dataframe(df)  # Validate DataFrame structure

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
    """Read a DataFrame stored with `to_hdf`.

    Args:
        path : Path to HDF5 file.
        key: Group name under which DataFrame is stored.
        mode: File mode.

    Returns:
        A pandas dataframe object.

    Raises:
        KeyError: When key is not found in the hdf5 file.
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


def _validate_dataframe(df: pd.DataFrame):
    """Validate DataFrame structure and dtypes."""
    # Define exactly supported dtypes (full matching; no 'object' here)
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
        "string",
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
        dtype_str = str(dtype)
        # Strict match for numeric, datetime, string, or category (non-categorical handled below)
        if dtype_str in supported_dtypes and not pd.api.types.is_categorical_dtype(dtype):
            # Non-categorical allowed types pass
            pass

        # For categorical dtype, ensure all categories are strings
        elif pd.api.types.is_categorical_dtype(dtype):
            categories = df[col].cat.categories
            bad = [cat for cat in categories if not (pd.isna(cat) or isinstance(cat, str))]
            if bad:
                raise ValueError(f"Categorical column {col!r} has non-string categories: {bad}")
            # Values in categorical are backed by codes, categories check is sufficient

        # Special handling for object dtype: only allow strings or NA
        elif dtype_str == "object":
            col_series = df[col]
            mask = col_series.apply(lambda x: pd.isna(x) or isinstance(x, str))
            if not mask.all():
                bad = col_series[~mask].unique()
                raise ValueError(f"Object column {col!r} contains non-string non-NA values: {bad}")
        else:
            raise ValueError(f"Unsupported dtype {dtype!r} for column {col!r}")

    # Check index dtype
    index_dtype = str(df.index.dtype)
    if not (df.index.dtype.kind in ["i", "u", "f", "M"] or index_dtype == "string"):
        raise ValueError(f"Unsupported index dtype {df.index.dtype!r}")

    # Check column names
    for col_name in df.columns:
        if not isinstance(col_name, (str, int)):
            raise ValueError(f"Column name must be string or int, got {type(col_name)}")

    # Check columns.name and index.name
    for name in [df.columns.name, df.index.name]:
        if name is not None and not isinstance(name, (str, int, float)):
            raise ValueError(f"Index/columns name must be None, string, int, or float, got {type(name)}")


def _save_column_metadata(grp: h5py.Group, df: pd.DataFrame):
    """Save column names, dtypes, and columns.name."""
    # Save column names
    col_names = [str(col) for col in df.columns]
    grp.create_dataset("column_names", data=np.array(col_names, dtype=DB.UTF8_STR))

    # Save column dtypes
    dtypes_info = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_categorical_dtype(dtype):
            # Store category information
            categories = dtype.categories.tolist()
            ordered = dtype.ordered
            dtypes_info.append(f"category|{categories}|{ordered}")
        else:
            dtypes_info.append(str(dtype))

    grp.create_dataset("column_dtypes", data=np.array(dtypes_info, dtype=DB.UTF8_STR))

    # Save columns.name
    columns_name = str(df.columns.name) if df.columns.name is not None else ""
    grp.attrs["columns_name"] = columns_name


def _save_index_data(grp: h5py.Group, df: pd.DataFrame, compression):
    """Save index data and metadata."""
    # Save index names
    index_names = [str(name) if name is not None else "" for name in df.index.names]
    grp.create_dataset("index_names", data=np.array(index_names, dtype=DB.UTF8_STR))

    # Save index data
    if pd.api.types.is_datetime64_any_dtype(df.index):
        # Convert datetime to string for reliable storage
        idx_data = df.index.astype(str).values
        grp.create_dataset("index_data", data=np.array(idx_data, dtype=DB.UTF8_STR), compression=compression)
        grp.attrs["index_dtype"] = "datetime64[ns]"
    elif pd.api.types.is_categorical_dtype(df.index):
        # Store categorical index
        codes = df.index.codes
        cats = [str(x) for x in df.index.categories.tolist()]
        grp.create_dataset("index_data", data=codes, compression=compression)
        grp.create_dataset("index_categories", data=cats, dtype=DB.UTF8_STR)
        grp.attrs["index_dtype"] = f"category|{df.index.ordered}"
    else:
        # Regular index
        grp.create_dataset("index_data", data=df.index.values, compression=compression)
        grp.attrs["index_dtype"] = str(df.index.dtype)


def _save_column_data(grp: h5py.Group, df: pd.DataFrame, compression):
    """Save column data with proper dtype preservation."""
    data_grp = grp.create_group("data")

    for col in df.columns:
        col_data = df[col]
        col_key = f"col_{col}"

        if pd.api.types.is_categorical_dtype(col_data):
            # Store categorical data as codes + categories
            codes = col_data.cat.codes.values
            categories = col_data.cat.categories.tolist()
            data_grp.create_dataset(f"{col_key}_codes", data=codes, compression=compression)
            data_grp.create_dataset(
                f"{col_key}_categories", data=np.array([str(cat) for cat in categories], dtype=DB.UTF8_STR)
            )

        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # Convert datetime to string for reliable storage
            dt_strings = col_data.astype(str).values
            data_grp.create_dataset(col_key, data=np.array(dt_strings, dtype=DB.UTF8_STR), compression=compression)

        elif pd.api.types.is_string_dtype(col_data) or col_data.dtype == "object":
            # Handle string/object data, replacing NA with a placeholder
            series = col_data.copy()
            na_mask = series.isna()
            series = series.astype(str)
            series[na_mask] = DB.placeholder_na
            str_data = series.values
            data_grp.create_dataset(col_key, data=np.array(str_data, dtype=DB.UTF8_STR), compression=compression)

        else:
            # Numeric, boolean, or other simple types
            data_grp.create_dataset(col_key, data=col_data.values, compression=compression)


def _load_column_metadata(grp: h5py.Group):
    """Load column names, dtypes, and columns.name."""
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
            categories = eval(parts[1])  # Safe here as we control the format
            ordered = parts[2] == "True"
            dtypes.append(pd.CategoricalDtype(categories=categories, ordered=ordered))
        else:
            dtypes.append(dtype_str)

    # Load columns.name
    columns_name = grp.attrs.get("columns_name", "")
    columns_name = columns_name if columns_name != "" else None

    return col_names, dtypes, columns_name


def _load_index_data(grp: h5py.Group):
    """Load index data and names."""
    # Load index names
    index_names = [name.decode(DB.UTF8) for name in grp["index_names"][()]]
    index_names = [name if name != "" else None for name in index_names]

    # Load index data based on dtype
    index_dtype = grp.attrs.get("index_dtype", "object")
    index_data = grp["index_data"][()]

    if index_dtype.startswith("datetime64"):
        # Convert string back to datetime
        index_strs = [x.decode(DB.UTF8) if isinstance(x, bytes) else str(x) for x in index_data]
        index = pd.to_datetime(index_strs)
    elif index_dtype.startswith("category"):
        # Reconstruct categorical index
        ordered = index_dtype.split("|")[1] == "True"
        categories = [cat.decode(DB.UTF8) for cat in grp["index_categories"][()]]
        index = pd.CategoricalIndex.from_codes(index_data, categories=categories, ordered=ordered)
    else:
        # Regular index
        if isinstance(index_data[0], bytes):
            index_data = [x.decode(DB.UTF8) for x in index_data]
        index = pd.Index(index_data, dtype=index_dtype)

    return index, index_names


def _load_column_data(grp: h5py.Group, columns, dtypes):
    """Load column data with proper dtype restoration."""
    data_grp = grp["data"]
    data_dict = {}

    for col, dtype in zip(columns, dtypes):
        col_key = f"col_{col}"

        if isinstance(dtype, pd.CategoricalDtype):
            # Reconstruct categorical data
            codes = data_grp[f"{col_key}_codes"][()]
            categories = [cat.decode(DB.UTF8) for cat in data_grp[f"{col_key}_categories"][()]]
            data_dict[col] = pd.Categorical.from_codes(codes, categories=categories, ordered=dtype.ordered)

        elif str(dtype).startswith("datetime64"):
            # Convert string back to datetime
            dt_strs = [x.decode(DB.UTF8) if isinstance(x, bytes) else str(x) for x in data_grp[col_key][()]]
            data_dict[col] = pd.to_datetime(dt_strs)

        elif str(dtype) in ["object", "string"]:
            # Handle string/object data, restoring placeholder back to NA
            raw = data_grp[col_key][()]
            if isinstance(raw[0], bytes):
                raw = [x.decode(DB.UTF8) for x in raw]
            restored = [pd.NA if x == DB.placeholder_na else x for x in raw]
            data_dict[col] = restored

        else:
            # Numeric, boolean, or other simple types
            data_dict[col] = data_grp[col_key][()]

    # Create DataFrame and restore dtypes
    df = pd.DataFrame(data_dict)
    for col, dtype in zip(columns, dtypes):
        if not isinstance(dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(dtype)

    return df


class HDFStore:
    """Context-manager replacement for pandas.HDFStore using h5py.

    This class provides a context manager API for reading and writing HDF5 files
    via h5py, mimicking pandas.HDFStore. It supports listing keys, removing
    groups or datasets by key, and gives direct access to the underlying
    h5py.File via the `.f` attribute.
    """

    path: str
    mode: str
    f: h5py.File

    def __init__(self, path: str, mode: str = "r"):
        """Initialize an HDFStore.

        Args:
            path (str): Path to the HDF5 file.
            mode (str): File mode, one of:
                - 'r'  : Readonly, file must exist.
                - 'r+' : Read/write, file must exist.
                - 'w'  : Create file, truncate if exists.
                - 'x'  : Create file, fail if exists.
                - 'a'  : Read/write if exists, create otherwise.
        """
        self.path = path
        self.mode = mode
        self.f: Optional[h5py.File] = None

    def __enter__(self) -> "HDFStore":
        """Open the HDF5 file and return this HDFStore instance.

        Returns:
            HDFStore: This instance with an open `h5py.File` in `.f`.
        """
        self.f = h5py.File(self.path, self.mode, libver="latest")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Flush and close the underlying HDF5 file on exiting the context.

        Args:
            exc_type (type): Exception type if raised within the context.
            exc_val (Exception): Exception instance if raised.
            exc_tb (traceback): Traceback object if an exception occurred.
        """
        if self.f is not None:
            self.f.flush()
            self.f.close()

    def __contains__(self, key: str) -> bool:
        """Check for the existence of a key (group or dataset) in the store.

        Args:
            key (str): The key to test for. May be prefixed with '/'.

        Returns:
            bool: True if the key exists in the file, False otherwise.
        """
        return key.lstrip("/") in self.f

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over all keys in the store.

        Yields:
            str: Next key name, always prefixed with '/'.
        """
        yield from self.keys()

    def keys(self) -> list[str]:
        """List all top-level group or dataset names in the file.

        Returns:
            List[str]: List of keys, each starting with '/'.
        """
        return [f"/{name}" for name in self.f.keys()]

    def remove(self, key: str) -> None:
        """Remove a group or dataset at the given key from the file.

        Args:
            key (str): The key to remove, may be prefixed with '/'.

        Raises:
            KeyError: If the specified key does not exist in the file.
        """
        name = key.lstrip("/")
        if name not in self.f:
            raise KeyError(f"No such key in HDF5 file: {key}")
        del self.f[name]


def check_h5_key(file_path: str, key: str) -> bool:
    """Check whether the given key is in the h5 file.

    Args:
        file_path: Absolute path for h5 file.
        key: The key to retrieve the associated table in h5 file.

    Returns:
        If there is such  a key ``True``, else ``False``.
    """
    if not os.access(file_path, os.R_OK):
        return False
    with HDFStore(file_path, mode="r") as f:
        return key in f


def repack_hdf5(path: str) -> None:
    """Repack the HDF5 file.

    Repack the HDF5 file at `path` to reclaim fragmented space by copying
    all groups/datasets (including nested ones) into a new file, then
    atomically replacing the original.

    Args:
        path: Path to the existing .h5 file. Must be writable.
    """
    base_dir, base_name = os.path.split(path)
    fd, tmp_path = tempfile.mkstemp(prefix=base_name, suffix=".repack.h5", dir=base_dir)
    os.close(fd)

    try:
        with h5py.File(path, "r") as src, h5py.File(tmp_path, "w", libver="latest") as dst:
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


def export_disk(
    df: Union[pd.DataFrame, pd.Series], hierarchy: str, file_path: str, overwrite: bool, logger: logging.Logger
):
    """Stored the pandas object into the given h5 file with specified key.

    The method is not expected to be used by the user.

    Args:
        df: The table to stor into the disk.
        hierarchy: The key to retrieve the associated table in h5 file.
        file_path: Absolute path for h5 file.
        overwrite: If ``True``, regardless of whether it is already saved in the disk, the program
            re-saves removing the previous table with the same name.
        logger: A logger instance used for recording file operations and warnings.
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


def read_exported(hierarchy: str, file_path: str) -> Union[pd.DataFrame, pd.Series]:
    """Read the data souces saved previously given h5 file path and the 'h5 key'.

    The method is not expected to be used by the user.

    Args:
        hierarchy: The key to retrieve the associated table in h5 file.
        file_path: Absolute path for h5 file.

    Returns:
        The table of interest as a pandas object.

    Raises:
        FileNotFoundError: If the file indicated by `file_path` is not exist or not readable.
        KeyError: If there is no key `hierarchy` in the ``h5`` file defined by `file_path`.
    """
    if not os.access(file_path, os.R_OK):
        raise FileNotFoundError("The file is not exist or not readable.")

    if not check_h5_key(file_path, hierarchy):
        raise KeyError

    df = read_hdf(path=file_path, key=hierarchy, mode="r")
    return df
