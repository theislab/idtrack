#!/usr/bin/env python3
"""Unit tests for idtrack._utils_hdf5: one immutable, atomically written HDF5 file per cached table."""

import multiprocessing
import os
import signal
import time

import h5py
import numpy as np
import pandas as pd
import pytest

import idtrack._utils_hdf5 as hs
from idtrack._utils_hdf5 import TableFormatError, load_table, read_table, read_table_params, write_table


def _default_text_dtype():
    return pd.Series(["a"]).dtype


def _assert_same(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns)
    assert list(a.dtypes) == list(b.dtypes)
    assert type(a.index) is type(b.index) and a.index.equals(b.index)
    assert a.equals(b)


def _mixed_frame(n: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    text = pd.Series([np.nan if i == 1 else f"id{i % 3}" for i in range(n)], dtype=object)
    return pd.DataFrame(
        {
            "text": text,
            "int64": np.arange(n, dtype=np.int64),
            "uint8": np.arange(n, dtype=np.uint8),
            "float16": rng.random(n).astype(np.float16),
            "float64": np.where(np.arange(n) % 2 == 0, np.nan, 1.5),
            "bool": np.arange(n) % 2 == 0,
            "cat": pd.Categorical([["a|b", "c"][i % 2] for i in range(n)], categories=["c", "a|b"], ordered=True),
            "when": pd.to_datetime(["2020-01-01"] * n).astype("datetime64[us]"),
            "a/b": [f"x{i}" for i in range(n)],
        }
    )


def test_roundtrip_is_exact_for_every_supported_dtype(tmp_path):
    """A table read back equals the table written, dtypes and index included (storage_net-7, -8)."""
    df = _mixed_frame()
    df.loc[2, "when"] = pd.NaT
    df["text"] = df["text"].astype(_default_text_dtype())
    df["string"] = pd.array(["s", None, "t", "u", "v", "w", "x"], dtype="string")
    path = str(tmp_path / "t.h5")
    write_table(path, df, {"k": 1})
    back = read_table(path)
    _assert_same(df, back)
    assert pd.isna(back.loc[1, "text"]) and back.loc[1, "text"] is not pd.NA
    assert back.loc[1, "string"] is pd.NA


def test_values_that_look_like_missing_values_stay_values(tmp_path):
    """The old sentinel text, ``nan``, ``None``, NUL and surrounding blanks are values, not missing (storage_net-7b)."""
    values = ["_PLACEHOLDER_NA_", "nan", "None", "a\x00b", " x ", "", "ß€😀", None]
    df = pd.DataFrame({"v": pd.Series(values, dtype=object)})
    path = str(tmp_path / "t.h5")
    write_table(path, df, {})
    back = read_table(path)
    assert back["v"].iloc[:7].tolist() == values[:7]
    assert pd.isna(back["v"].iloc[7])


def test_zero_rows_and_zero_columns_roundtrip(tmp_path):
    """Empty tables are stored and read back with their dtypes."""
    df = _mixed_frame().iloc[0:0].reset_index(drop=True)
    df["text"] = df["text"].astype(_default_text_dtype())
    path = str(tmp_path / "t.h5")
    write_table(path, df, {})
    _assert_same(df, read_table(path))
    write_table(path, pd.DataFrame(), {})
    assert read_table(path).shape == (0, 0)


def test_range_index_is_kept(tmp_path):
    """A RangeIndex other than 0..n comes back as the same RangeIndex (storage_net-7d)."""
    df = pd.DataFrame({"a": [1, 2]}, index=pd.RangeIndex(5, 7))
    path = str(tmp_path / "t.h5")
    write_table(path, df, {})
    back = read_table(path)
    assert isinstance(back.index, pd.RangeIndex) and list(back.index) == [5, 6]


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({0: [1]}),  # integer label
        pd.DataFrame([[1, 2]], columns=["a", "a"]),  # duplicate labels
        pd.DataFrame({"a": [1]}, index=["x"]),  # not a RangeIndex
        pd.DataFrame({"a": pd.Series([1, "x"], dtype=object)}),  # mixed object column
        pd.DataFrame({"a": pd.array([1, None], dtype="Int64")}),  # unsupported dtype
        pd.DataFrame({"a": pd.Categorical([1, 2])}),  # non-text categories
        pytest.param(pd.Series([1, 2]), marks=pytest.mark.wrong_argument_type),  # not a DataFrame (storage_net-9)
    ],
)
def test_unstorable_tables_are_rejected_before_anything_is_written(tmp_path, bad):
    """Labels, dtypes and indexes that would not round-trip raise before a file appears (storage_net-8, -9)."""
    path = str(tmp_path / "t.h5")
    with pytest.raises(TableFormatError):
        write_table(path, bad, {})
    assert os.listdir(tmp_path) == []


def test_exception_during_a_write_leaves_no_table_and_no_temporary(tmp_path, monkeypatch):
    """A write interrupted by an exception (e.g. Ctrl-C) leaves the old state (storage_net-2)."""
    path = str(tmp_path / "t.h5")
    old = pd.DataFrame({"a": ["old"]})
    write_table(path, old, {})
    calls = {"n": 0}
    original = hs._write_column

    def failing(group, i, series, tag, digest=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(group, i, series, tag, digest)

    monkeypatch.setattr(hs, "_write_column", failing)
    with pytest.raises(KeyboardInterrupt):
        write_table(path, pd.DataFrame({"a": ["new"], "b": ["new"]}), {})
    assert os.listdir(tmp_path) == ["t.h5"]
    _assert_same(old, read_table(path))


def _slow_writer(path: str, ready) -> None:
    original = hs._write_column

    def slow(group, i, series, tag, digest=None):
        ready.set()
        time.sleep(60)
        return original(group, i, series, tag, digest)

    hs._write_column = slow
    write_table(path, pd.DataFrame({"a": np.arange(10)}), {})


def test_a_killed_writer_leaves_every_table_readable(tmp_path):
    """SIGKILL in the middle of a write: older tables stay readable, the killed one is absent (storage_net-1, access-6)."""
    keep = str(tmp_path / "keep.h5")
    target = str(tmp_path / "target.h5")
    df = pd.DataFrame({"a": ["x", "y"]})
    write_table(keep, df, {})
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Event()
    proc = ctx.Process(target=_slow_writer, args=(target, ready))
    proc.start()
    assert ready.wait(30)
    os.kill(proc.pid, signal.SIGKILL)
    proc.join()
    _assert_same(df, read_table(keep))
    assert load_table(target, {}) is None
    write_table(target, df, {})
    _assert_same(df, read_table(target))


def test_stale_temporaries_are_removed_by_the_next_writer(tmp_path):
    """Leftovers of killed writers do not accumulate."""
    stale = tmp_path / "t.h5.tmp-1-abc"
    stale.write_bytes(b"x")
    old = time.time() - 2 * hs._STALE_TEMPORARY_SECONDS
    os.utime(stale, (old, old))
    fresh = tmp_path / "u.h5.tmp-2-def"
    fresh.write_bytes(b"x")
    write_table(str(tmp_path / "t.h5"), pd.DataFrame({"a": [1]}), {})
    assert sorted(os.listdir(tmp_path)) == ["t.h5", "u.h5.tmp-2-def"]


def test_load_table_requires_the_same_parameters(tmp_path):
    """A table made with other parameters is not served (decision 5)."""
    path = str(tmp_path / "t.h5")
    write_table(path, pd.DataFrame({"a": [1]}), {"window": [48, 115], "dbs": ["A", "B"]})
    assert load_table(path, {"window": [48, 115], "dbs": ["A", "B"]}) is not None
    assert load_table(path, {"window": [76, 115], "dbs": ["A", "B"]}) is None
    assert load_table(path, {"window": [48, 115], "dbs": ["A"]}) is None
    assert read_table_params(path) == {"dbs": ["A", "B"], "window": [48, 115]}


def test_damaged_or_foreign_files_are_reported_absent(tmp_path):
    """A file that is not a table of this format is not read as one."""
    damaged = tmp_path / "d.h5"
    damaged.write_bytes(b"not hdf5")
    assert load_table(str(damaged), {}) is None
    foreign = str(tmp_path / "f.h5")
    with h5py.File(foreign, "w") as f:
        f.attrs["idtrack_table_format"] = 1
    assert load_table(foreign, {}) is None
    with pytest.raises(TableFormatError):
        read_table(foreign)


def test_replacing_a_table_frees_the_old_one(tmp_path):
    """Ten overwrites of one table use the space of one copy (storage_net-5)."""
    path = str(tmp_path / "t.h5")
    df = pd.DataFrame({"a": [f"value-{i}" for i in range(50_000)], "b": np.arange(50_000)})
    write_table(path, df, {})
    size_one = os.path.getsize(path)
    for _ in range(10):
        write_table(path, df, {})
    assert sum(os.path.getsize(tmp_path / n) for n in os.listdir(tmp_path)) <= 1.1 * size_one


def test_text_is_compressed(tmp_path):
    """Repetitive text costs a small fraction of its raw size on disk (storage_net-4)."""
    n = 200_000
    df = pd.DataFrame(
        {"name_db": ["Uniprot/SPTREMBL", "EntrezGene"] * (n // 2), "id": [f"ENSG{i:011d}" for i in range(n)]}
    )
    path = str(tmp_path / "t.h5")
    write_table(path, df, {})
    assert os.path.getsize(path) < 0.25 * int(df.memory_usage(deep=True).sum())


def test_files_use_the_oldest_hdf5_format(tmp_path):
    """Files are written with the format bounds ("earliest", "v110"), never "latest" (storage_net-new-1)."""
    # The superblock is then version 0, which HDF5 1.10 reads and which has no "open for write" flag that a killed
    # writer could leave set.
    path = str(tmp_path / "t.h5")
    write_table(path, pd.DataFrame({"a": ["x"], "b": [1]}), {})
    with open(path, "rb") as handle:
        head = handle.read(9)
    assert head[:8] == b"\x89HDF\r\n\x1a\n"
    assert head[8] == 0


def _reader_loop(path: str, frames: list, reads) -> None:
    ok = 0
    deadline = time.time() + 3
    while time.time() < deadline:
        back = load_table(path, {})
        assert back is not None
        assert any(back.equals(f) for f in frames)
        ok += 1
    # The count goes back through shared memory, which is written before the function returns. A
    # `multiprocessing.Queue` hands the value to a feeder thread instead, and on Python 3.13 to 3.15 the value of this
    # process, which ends right after `put`, never arrived when the whole test tree had run before (exit code 0, the
    # pipe intact and empty), although it arrives when the test runs alone and on Python 3.12.
    reads.value = ok


def test_readers_never_see_a_partial_table_while_a_writer_replaces_it(tmp_path):
    """Readers are never blocked or fed a partial table by a concurrent writer (storage_net-6)."""
    path = str(tmp_path / "t.h5")
    frames = [pd.DataFrame({"a": [f"{k}-{i}" for i in range(20_000)]}) for k in range(3)]
    write_table(path, frames[0], {})
    ctx = multiprocessing.get_context("fork")
    reads = ctx.Value("q", 0)
    reader = ctx.Process(target=_reader_loop, args=(path, frames, reads))
    reader.start()
    deadline = time.time() + 3
    k = 0
    while time.time() < deadline:
        write_table(path, frames[k % 3], {})
        k += 1
    reader.join(30)
    assert reader.exitcode == 0
    assert reads.value > 0
    assert k > 0


def test_content_digest_identifies_the_table(tmp_path):
    """Equal tables have equal digests whoever wrote them; a changed value changes it (decision 5, for graph files)."""
    from idtrack._utils_hdf5 import read_table_digest

    df = _mixed_frame()
    write_table(str(tmp_path / "a.h5"), df, {"x": 1})
    write_table(str(tmp_path / "b.h5"), df.copy(), {"x": 2})
    other = df.copy()
    other.loc[0, "int64"] = 99
    write_table(str(tmp_path / "c.h5"), other, {})
    a, b, c = (read_table_digest(str(tmp_path / f"{n}.h5")) for n in "abc")
    assert a == b and a != c and len(a) == 64
