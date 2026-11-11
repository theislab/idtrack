#!/usr/bin/env python3
"""Unit tests for `idtrack._mysql_dump.read_mysql_dump`: MySQL dump escapes are decoded as `LOAD DATA` would."""

from __future__ import annotations

import io

import pandas as pd

from idtrack._mysql_dump import read_mysql_dump

COLUMNS = ["id", "name", "description"]

# Written as `SELECT ... INTO OUTFILE` does: a tab or newline inside a field is a backslash followed by the raw
# character, a backslash is doubled, NULL is `\N`, NUL is `\0`; a raw carriage return is ordinary data.
FIXTURE = (
    "1\tplain\t\\N\n"
    "2\twith\\\ttab\tline one\\\nline two\n"
    "3\tback\\\\slash\t\\\\N\n"
    '4\t"quoted" name\tsays "hi"\n'
    "5\tcr\rinside\tnul\\0char\n"
    "6\t\t\\n and \\t as two-character escapes\n"
    "7\t\\N\tlast row without newline"
)


def _read(text: str, usecols=None, chunk_size=None) -> pd.DataFrame:
    stream = io.StringIO(text)
    if chunk_size is not None:
        # Force record boundaries to fall inside a chunk.
        original_read = stream.read
        stream.read = lambda n=-1: original_read(chunk_size)  # type: ignore[method-assign]
    return read_mysql_dump(stream, columns=COLUMNS, usecols=usecols)


def _expected_rows():
    return [
        (1, "plain", None),
        (2, "with\ttab", "line one\nline two"),
        (3, "back\\slash", "\\N"),
        (4, '"quoted" name', 'says "hi"'),
        (5, "cr\rinside", "nul\x00char"),
        (6, "", "\n and \t as two-character escapes"),
        (7, None, "last row without newline"),
    ]


def _rows(df: pd.DataFrame):
    return [tuple(None if pd.isna(v) else v for v in row) for row in df.itertuples(index=False)]


def test_escaped_tab_newline_and_backslash():
    """Escaped tab newline and backslash."""
    df = _read(FIXTURE)
    assert len(df) == 7
    assert df["id"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert _rows(df) == _expected_rows()


def test_null_and_empty_string_differ():
    """Null and empty string differ."""
    df = _read(FIXTURE)
    assert pd.isna(df.loc[0, "description"])
    assert df.loc[5, "name"] == ""
    # `\\N` is a literal backslash-N, not NULL.
    assert df.loc[2, "description"] == "\\N"


def test_small_chunks_split_records_anywhere():
    """Small chunks split records anywhere."""
    for chunk_size in (1, 2, 3, 7, 16):
        assert _rows(_read(FIXTURE, chunk_size=chunk_size)) == _expected_rows()


def test_usecols_keeps_order_and_restored_values():
    """Usecols keeps order and restored values."""
    df = _read(FIXTURE, usecols=["description", "id"])
    assert list(df.columns) == ["description", "id"]
    assert df.loc[2, "description"] == "\\N"
    assert df.loc[4, "description"] == "nul\x00char"


def test_plain_dump_matches_tab_separated_parse():
    """Plain dump matches tab separated parse."""
    text = "1\ta\t\\N\n2\tb\tc\n"
    df = _read(text)
    expected = pd.read_csv(
        io.StringIO(text), sep="\t", header=None, names=COLUMNS, na_values=["\\N"], keep_default_na=False
    )
    pd.testing.assert_frame_equal(df, expected)


def test_empty_dump_gives_empty_frame():
    """Empty dump gives empty frame."""
    df = _read("")
    assert list(df.columns) == COLUMNS
    assert len(df) == 0
