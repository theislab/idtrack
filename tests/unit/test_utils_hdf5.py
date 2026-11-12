#!/usr/bin/env python3
"""Unit tests for idtrack._utils_hdf5.

These tests focus on the custom, PyTables-free HDF5 round-trip helpers used by the cache layer.
"""

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

        # Text of the default kind comes back in the default text dtype of the running pandas with NaN for missing
        # values (what pandas itself gives for fresh text); `string` stays `string` with pd.NA.
        expected = df.copy()
        expected["object_col"] = pd.Series(["x", np.nan, "z"], index=index, dtype=pd.Series(["a"]).dtype)

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


def test_zero_row_frame_with_string_columns_roundtrips(tmp_path):
    """A frame with no rows and string columns (e.g. an empty `idhistory_narrow`) can be saved and read back."""
    df = pd.DataFrame(
        {
            "old_stable_id": pd.Series([], dtype=object),
            "new_stable_id": pd.Series([], dtype=object),
            "score": pd.Series([], dtype=float),
            "old_release": pd.Series([], dtype=float),
        }
    )
    path = tmp_path / "zero.h5"
    export_disk(df, "ens116_processed_idhistory_narrow_gene", str(path), False, logger=logging.getLogger("t"))
    back = read_exported("ens116_processed_idhistory_narrow_gene", str(path))
    assert list(back.columns) == list(df.columns)
    assert len(back) == 0
    assert back["old_stable_id"].dtype == pd.Series(["a"]).dtype  # default text dtype of the running pandas
    assert back["score"].dtype == float

    categorical = pd.DataFrame({"c": pd.Categorical([], categories=[])})
    to_hdf(str(tmp_path / "zero_cat.h5"), "k", categorical, mode="w")
    assert len(read_hdf(str(tmp_path / "zero_cat.h5"), "k")) == 0


def _default_text_dtype():
    return pd.Series(["a"]).dtype


def _text_dtypes() -> list:
    """The three pandas text dtypes: `object`, `string` (pd.NA) and `str` (NaN; the default of pandas 3)."""
    return [
        pytest.param(object, id="object"),
        pytest.param("string", id="string"),
        pytest.param(pd.StringDtype(na_value=np.nan), id="str"),
    ]


@pytest.mark.parametrize("dtype", _text_dtypes())
def test_every_text_dtype_roundtrips_with_its_missing_values(tmp_path, dtype):
    """Every pandas text dtype is accepted; `object`/`str` come back as the default text dtype with NaN."""
    df = pd.DataFrame({"t": pd.Series(["ENSG1", None, "", "ENSG2"], dtype=dtype), "n": [1, 2, 3, 4]})
    to_hdf(str(tmp_path / "t.h5"), "k", df, mode="w")
    back = read_hdf(str(tmp_path / "t.h5"), "k")
    assert back["t"].isna().tolist() == [False, True, False, False]
    assert back["t"].dropna().tolist() == ["ENSG1", "", "ENSG2"]
    if isinstance(dtype, str):
        assert back["t"].dtype == pd.StringDtype()
        assert back["t"].iloc[1] is pd.NA
    else:
        assert back["t"].dtype == _default_text_dtype()
        assert isinstance(back["t"].iloc[1], float) and np.isnan(back["t"].iloc[1])


def test_cached_text_table_equals_the_fresh_table(tmp_path):
    """A text table as pandas reads it from a dump (here: read_csv) is unchanged by a cache round trip."""
    import io

    fresh = pd.read_csv(
        io.StringIO("a\tb\tc\nx\t1\t\\N\ny\t2\tz\n"), sep="\t", na_values=["\\N"], keep_default_na=False
    )
    to_hdf(str(tmp_path / "f.h5"), "k", fresh, mode="w")
    pd.testing.assert_frame_equal(read_hdf(str(tmp_path / "f.h5"), "k"), fresh, check_dtype=True)


@pytest.mark.parametrize("dtype", _text_dtypes())
def test_text_index_roundtrips(tmp_path, dtype):
    """A text index (object, str or string) is accepted and restored like a text column."""
    df = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["r1", "r2"], dtype=dtype, name="rows"))
    to_hdf(str(tmp_path / "i.h5"), "k", df, mode="w")
    back = read_hdf(str(tmp_path / "i.h5"), "k")
    assert back.index.tolist() == ["r1", "r2"]
    assert back.index.name == "rows"
    expected = pd.StringDtype() if isinstance(dtype, str) else _default_text_dtype()
    assert back.index.dtype == expected


def test_datetime_and_timedelta_with_missing_values_roundtrip(tmp_path):
    """Missing datetimes (NaT) survive a round trip, and the recorded resolution is restored."""
    df = pd.DataFrame(
        {
            "dt": pd.Series([pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2021-06-30 12:00:00")]).astype(
                "datetime64[s]"
            ),
            "td": pd.to_timedelta(["1 days", None, "2 hours"]),
        }
    )
    to_hdf(str(tmp_path / "d.h5"), "k", df, mode="w")
    back = read_hdf(str(tmp_path / "d.h5"), "k")
    pd.testing.assert_frame_equal(back, df, check_dtype=True)


# The HDF5 library line that can read the cache files of idtrack 0.0.5: its writer opened them with
# `libver="latest"`, so the HDF5 1.14 of its h5py wheels wrote datatype messages of version 4, which HDF5 1.10 does not
# know ("bad version number for datatype message"); HDF5 1.12 introduced them.
HDF5_READING_IDTRACK_0_0_5_CACHES = (1, 12, 0)


def test_cache_written_by_idtrack_0_0_5_reads_back_unchanged():
    """A table cached by idtrack 0.0.5 reads back with the same values and missing values.

    The fixture `tests/data/legacy_cache_v0.0.5.h5` is a real cache of that version: written by
    `tests/data/make_legacy_cache_v0_0_5.py` with the writer of the `v0.0.5` tag, under pandas 2.2.3 and h5py 3.13.0
    with HDF5 1.14.6, the environment that wrote the owner's caches. Text comes back in the default text dtype of the
    running pandas (`object` under pandas 2, `str` under pandas 3) with NaN for missing values, as pandas itself reads
    text; every other dtype is the recorded one. An HDF5 older than 1.12 cannot open such a file at all, whatever
    idtrack does, so the test is skipped there, with that reason, instead of failing.
    """
    from pathlib import Path

    import h5py

    if h5py.version.hdf5_version_tuple < HDF5_READING_IDTRACK_0_0_5_CACHES:
        pytest.skip(
            f"HDF5 {h5py.version.hdf5_version} cannot read files written by HDF5 1.14 with libver='latest' (datatype "
            "messages of version 4 need HDF5 >= 1.12), as the caches of idtrack 0.0.5 are."
        )
    path = str(Path(__file__).parents[1] / "data" / "legacy_cache_v0.0.5.h5")
    text = _default_text_dtype()
    expected = pd.DataFrame(
        {
            "old_stable_id": pd.Series(["ENSSSCG00000000001", np.nan, "ENSSSCG00000000003", np.nan], dtype=text),
            "old_version": [1.0, np.nan, 2.0, 1.0],
            "new_stable_id": pd.Series(
                ["ENSSSCG00000000001", "ENSSSCG00000000002", "", "ENSSSCG00000000004"], dtype=text
            ),
            "mapping_session_id": np.array([10, 11, 12, 13], dtype=np.int64),
            "type": pd.Categorical(["gene", "gene", "transcript", "gene"]),
            "is_current": [True, False, True, True],
        }
    )
    back = read_hdf(path, "ens115_mysql_stable_id_event")
    pd.testing.assert_frame_equal(back, expected, check_dtype=True, check_categorical=True)
    assert back["old_stable_id"].isna().tolist() == [False, True, False, True]


def test_validate_dataframe_rejects_non_text_object_index():
    """An object index must hold strings, like an object column."""
    df = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["a", 1], dtype=object))
    with pytest.raises(ValueError, match="Object index"):
        validate_dataframe(df)
