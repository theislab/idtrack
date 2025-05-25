#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import os
import tempfile
from typing import Optional, Union

import h5py
import pandas as pd


class HDFStore:
    """Context-manager replacement for pandas.HDFStore using h5py.

    Supports .keys() and .remove(key), and exposes the raw File as `.f`.
    """

    def __init__(self, path: str, mode: str = "r"):
        # mode: one of 'r','r+','w','x','a'
        self.path = path
        self.mode = mode
        self.f: Optional[h5py.File] = None

    def __enter__(self):
        # libver='latest' enables the newest features & better performance on big files
        self.f = h5py.File(self.path, self.mode, libver="latest")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.f is not None:
            self.f.flush()
            self.f.close()

    def __contains__(self, key: str) -> bool:
        return key.lstrip("/") in self.f

    def __iter__(self):
        yield from self.keys()

    def keys(self) -> list[str]:
        """List top-level group names, prefixed with '/'.

        Matches pd.HDFStore.keys().
        """
        return [f"/{name}" for name in self.f.keys()]

    def remove(self, key: str) -> None:
        """Remove a dataset or group at `key`.

        Matches pd.HDFStore.remove().
        """
        name = key.lstrip("/")  # drop leading slash if present
        if name not in self.f:
            raise KeyError(f"No such key in HDF5 file: {key}")
        del self.f[name]


def read_hdf(
    path: str,
    key: str,
    mode: str = "r",
) -> Union[pd.DataFrame, pd.Series]:
    """Read a pandas object previously written with to_hdf.

    Mirrors pd.read_hdf(path, key=key, mode=mode).
    """
    grp_name = key.lstrip("/")
    with HDFStore(path, mode=mode) as store:
        f = store.f
        if grp_name not in f:
            raise KeyError(f"No key {key!r} in HDF5 file {path!r}")
        grp = f[grp_name]

        # reconstruct index
        idx = grp["index"][()]  # numpy array
        idx_names = grp.attrs.get("index_names", [None])
        index = (
            pd.Index(idx, name=idx_names[0])
            if len(idx_names) == 1
            else pd.MultiIndex.from_arrays([idx], names=idx_names)
        )

        # get column list
        cols: list[str] = grp.attrs["columns"]
        data = {}
        for col in cols:
            arr = grp[f"col__{col}"][()]
            data[col] = arr

        if len(cols) == 1:
            # reconstruct Series
            return pd.Series(data[cols[0]], index=index, name=cols[0])
        else:
            return pd.DataFrame(data, index=index)


def to_hdf(
    path: str,
    key: str,
    df: Union[pd.DataFrame, pd.Series],
    mode: str = "a",
    compression: str = "gzip",
    compression_opts: int = 4,
    shuffle: bool = True,
) -> None:
    """Write a pandas object into HDF5 under group `key` using h5py.

    Mirrors df.to_hdf(path, key=key, mode=mode).
    """
    grp_name = key.lstrip("/")
    with HDFStore(path, mode=mode) as store:
        f = store.f
        # delete old group if overwriting
        if grp_name in f and mode in ("a", "r+"):
            del f[grp_name]

        grp = f.create_group(grp_name)

        # store index
        idx = df.index.to_numpy()
        grp.create_dataset(
            "index",
            data=idx,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        grp.attrs["index_names"] = list(df.index.names)

        # store columns (for Series it'll be an unnamed single column)
        if isinstance(df, pd.Series):
            columns = [df.name]  # allowing None
            arrays = {columns[0]: df.to_numpy()}
        else:
            columns = list(df.columns)
            arrays = {col: df[col].to_numpy() for col in columns}

        grp.attrs["columns"] = columns

        # write each column as its own dataset
        for col, arr in arrays.items():
            grp.create_dataset(
                f"col__{col}",
                data=arr,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
            )


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
    """Repack the HDF5 file at `path` to reclaim fragmented space. Copies all root-level groups & attributes into a new
    file using h5py.File.copy, then atomically replaces the original file.

    assumes no nested structures.

    Parameters
    ----------
    path
        Path to the existing .h5 file. Must be writable.
    """
    # Prepare a temp file in the same directory for atomic replacement
    base_dir, base_name = os.path.split(path)
    fd, tmp_path = tempfile.mkstemp(prefix=base_name, suffix=".repack.h5", dir=base_dir)
    os.close(fd)

    try:
        # Open source as read-only, target as write (latest libver for best performance)
        with h5py.File(path, "r") as src, h5py.File(tmp_path, "w", libver="latest") as dst:
            # Copy root-level attributes
            for attr_name, attr_val in src.attrs.items():
                dst.attrs[attr_name] = attr_val

            # Copy each top-level group/dataset verbatim
            for name in src:
                # this preserves dataset creation properties (compression, chunks, etc.)
                src.copy(name, dst, name)

            dst.flush()

        # Atomically replace the old file
        os.replace(tmp_path, path)

    finally:
        # Clean up stray temp file on error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def export_disk(df: Union[pd.DataFrame, pd.Series], hierarchy: str, file_path: str, overwrite: bool, logger):
    """Stored the pandas object into the given h5 file with specified key.

    The method is not expected to be used by the user.

    Args:
        df: The table to stor into the disk.
        hierarchy: The key to retrieve the associated table in h5 file.
        file_path: Absolute path for h5 file.
        overwrite: If ``True``, regardless of whether it is already saved in the disk, the program
            re-saves removing the previous table with the same name.
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
