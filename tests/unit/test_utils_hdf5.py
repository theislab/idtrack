#!/usr/bin/env python3
"""Unit tests for idtrack._utils_hdf5.

These tests focus on the custom, PyTables-free HDF5 round-trip helpers used by the cache layer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from idtrack._utils_hdf5 import (
    check_h5_key,
    export_disk,
    read_exported,
    read_hdf,
    repack_hdf5,
    to_hdf,
    validate_dataframe,
)


class TestHdf5RoundTrip:
    """Round-trip and validation tests for the HDF5 helpers."""

    def test_to_hdf_and_read_hdf_roundtrip_preserves_metadata(self, tmp_path):
        """Round-trip a DataFrame and preserve metadata/dtypes."""
        path = tmp_path / "roundtrip.h5"

        index = pd.Index(["row1", "row2", "row3"], dtype="string", name="idx")
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "bool_col": [True, False, True],
                "string_col": ["a", pd.NA, "c"],
                "object_col": ["x", None, "z"],
                "cat_col": pd.Categorical(["low", "mid", "low"], categories=["low", "mid", "high"], ordered=True),
                "dt_col": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "td_col": pd.to_timedelta([1, 2, 3], unit="D"),
            },
            index=index,
        )
        df["int_col"] = df["int_col"].astype("int64")
        df["float_col"] = df["float_col"].astype("float64")
        df["bool_col"] = df["bool_col"].astype("bool")
        df["string_col"] = df["string_col"].astype("string")
        df["object_col"] = df["object_col"].astype("object")
        df.columns.name = "cols"

        to_hdf(path=str(path), key="/df", df=df, mode="a")
        loaded = read_hdf(path=str(path), key="/df", mode="r")

        expected = df.copy()
        expected.loc[expected["object_col"].isna(), "object_col"] = pd.NA

        pd.testing.assert_frame_equal(loaded, expected, check_dtype=True)
        assert loaded.columns.name == "cols"
        assert loaded.index.name == "idx"

    def test_validate_dataframe_rejects_bad_object_column(self):
        """validate_dataframe should reject mixed-type object columns."""
        df = pd.DataFrame({"bad": pd.Series(["ok", 1], dtype="object")})
        with pytest.raises(ValueError, match="Object column"):
            validate_dataframe(df)


class TestHdf5CacheUtilities:
    """Tests for cache-oriented HDF5 convenience functions."""

    def test_check_h5_key_missing_file_is_false(self, tmp_path):
        """check_h5_key should be False for missing files."""
        assert check_h5_key(str(tmp_path / "missing.h5"), "/df") is False

    def test_export_disk_and_read_exported_roundtrip(self, tmp_path):
        """export_disk and read_exported should round-trip DataFrames."""
        path = tmp_path / "cache.h5"
        logger = logging.getLogger("test_utils_hdf5")

        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 11, 12], name="idx"))
        export_disk(df=df1, hierarchy="/df", file_path=str(path), overwrite=False, logger=logger)
        assert check_h5_key(str(path), "/df") is True

        loaded = read_exported(hierarchy="/df", file_path=str(path))
        pd.testing.assert_frame_equal(loaded, df1)

        # No overwrite → keep the original.
        df2 = pd.DataFrame({"a": [4, 5, 6]}, index=pd.Index([10, 11, 12], name="idx"))
        export_disk(df=df2, hierarchy="/df", file_path=str(path), overwrite=False, logger=logger)
        loaded_again = read_exported(hierarchy="/df", file_path=str(path))
        pd.testing.assert_frame_equal(loaded_again, df1)

        # Overwrite replaces the key.
        export_disk(df=df2, hierarchy="/df", file_path=str(path), overwrite=True, logger=logger)
        loaded_overwritten = read_exported(hierarchy="/df", file_path=str(path))
        pd.testing.assert_frame_equal(loaded_overwritten, df2)

    def test_repack_hdf5_preserves_keys(self, tmp_path):
        """repack_hdf5 should keep existing keys readable."""
        path = tmp_path / "repack.h5"
        logger = logging.getLogger("test_utils_hdf5")

        df = pd.DataFrame({"x": np.arange(5)})
        export_disk(df=df, hierarchy="/df", file_path=str(path), overwrite=True, logger=logger)
        repack_hdf5(str(path))

        loaded = read_exported(hierarchy="/df", file_path=str(path))
        pd.testing.assert_frame_equal(loaded, df)
