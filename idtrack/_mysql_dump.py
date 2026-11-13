#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

r"""Read Ensembl's MySQL table dumps (``<table>.txt.gz``) exactly as MySQL would load them.

The dumps are written by ``SELECT ... INTO OUTFILE`` with the default ``FIELDS TERMINATED BY '\t' ESCAPED BY
'\\' LINES TERMINATED BY '\n'``. In that format a backslash escapes the next character: a field that holds a tab
or a newline is written as a backslash followed by the raw tab or newline, a backslash as ``\\``, NUL as
``\0``, and SQL ``NULL`` as a field consisting of ``\N``. ``LOAD DATA`` also reads ``\b``, ``\n``, ``\r``,
``\t`` and ``\Z`` as backspace, newline, carriage return, tab and Ctrl-Z. Any other escaped character stands for
itself. Raw carriage returns are ordinary data (only ``\n`` ends a record).

Reading such a file with a plain tab-separated parser splits a record at every escaped newline, keeps the escaping
backslashes and treats a leading double quote as CSV quoting. :py:func:`read_mysql_dump` instead decodes each record
the way ``LOAD DATA`` does and hands pandas an unambiguous CSV stream, so the frame matches what the MySQL server
returns for the same table (``NULL`` becomes ``NaN``; pandas infers the column dtypes as it does for any text
input). Records without escapes or quotes, the vast majority, pass through untouched.
"""

import csv
import io
from collections.abc import Collection
from typing import Any, cast

import pandas as pd

# MySQL `LOAD DATA` escape sequences; any other escaped character stands for itself.
_ESCAPES = {"0": "\x00", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}
_NULL_TOKEN = "\\N"  # noqa: S105 - the MySQL dump token for NULL, not a password
# Values that pandas cannot carry through its parser verbatim: they are emitted as an empty field and restored
# after parsing. A literal "\N" would otherwise read as NULL, and NUL ends a field in pandas' C tokenizer.
_RESTORE_CHARS = ("\x00",)


def _ends_with_odd_backslashes(text: str) -> bool:
    """Return whether *text* ends with an odd number of backslashes (i.e. its last character is escaped)."""
    count = len(text) - len(text.rstrip("\\"))
    return count % 2 == 1


def _only_null_escapes(record: str) -> bool:
    r"""Return whether every backslash of *record* belongs to a whole-field ``\N`` (SQL NULL)."""
    padded = "\t" + record + "\t"
    # Two passes: one pass of `str.replace` skips every second member of adjacent `\N` fields.
    reduced = padded.replace("\t\\N\t", "\t\t").replace("\t\\N\t", "\t\t")
    return "\\" not in reduced


def _decode_fields(record: str) -> list[str | None]:
    """Split one logical record into decoded field values (``None`` for SQL NULL)."""
    fields: list[str | None] = []
    current: list[str] = []
    raw_is_null_token = True  # whether the raw text of the current field is exactly `\N` so far
    raw_length = 0
    i = 0
    n = len(record)
    while i < n:
        char = record[i]
        if char == "\\" and i + 1 < n:
            nxt = record[i + 1]
            current.append(_ESCAPES.get(nxt, nxt))
            raw_is_null_token = raw_is_null_token and raw_length == 0 and nxt == "N"
            raw_length += 2
            i += 2
            continue
        if char == "\t":
            fields.append(None if (raw_is_null_token and raw_length == 2) else "".join(current))
            current, raw_is_null_token, raw_length = [], True, 0
            i += 1
            continue
        # A trailing lone backslash (malformed input) is kept as a literal character.
        current.append(char)
        raw_is_null_token = False
        raw_length += 1
        i += 1
    fields.append(None if (raw_is_null_token and raw_length == 2) else "".join(current))
    return fields


def _quote_csv(value: str) -> str:
    """Quote a value for pandas' default CSV dialect (double quotes, doubled inside)."""
    return '"' + value.replace('"', '""') + '"'


class MySQLDumpCSVStream(io.TextIOBase):
    r"""Text stream that turns a MySQL dump into CSV that pandas parses unambiguously.

    Records are read from *source* (a text stream of the dump) and written out as tab-separated CSV records:
    records without backslash escapes or double quotes are passed through; the others are decoded field by field,
    and fields containing a tab, newline or double quote are quoted. SQL ``NULL`` stays ``\N`` (read by pandas as
    ``NaN`` through ``na_values``). Values that pandas cannot reproduce verbatim are emitted as empty fields and
    recorded in :py:attr:`restore` as ``(record index, field index, value)`` for :py:func:`read_mysql_dump` to put back.

    Args:
        source: Readable text stream of the dump (already decompressed).
        chunk_size: Number of characters read from *source* at a time.
    """

    def __init__(self, source: Any, chunk_size: int = 1 << 20) -> None:
        """Wrap *source*; nothing is read until :py:meth:`read` is called."""
        super().__init__()
        self._source = source
        self._chunk_size = int(chunk_size)
        self._tail = ""  # text after the last newline seen
        self._partial: list[str] = []  # physical lines of a record whose newline was escaped
        self._out = ""
        self._eof = False
        self.records = 0
        self.restore: list[tuple[int, int, str]] = []

    def readable(self) -> bool:
        """Return ``True``: the stream supports :py:meth:`read`."""
        return True

    def _emit_record(self, record: str, pieces: list[str]) -> None:
        if ('"' not in record) and ("\\" not in record or _only_null_escapes(record)):
            pieces.append(record)
        else:
            out_fields: list[str] = []
            for field_index, value in enumerate(_decode_fields(record)):
                if value is None:
                    out_fields.append(_NULL_TOKEN)
                elif value == _NULL_TOKEN or any(c in value for c in _RESTORE_CHARS):
                    self.restore.append((self.records, field_index, value))
                    out_fields.append("")
                elif "\t" in value or "\n" in value or '"' in value:
                    out_fields.append(_quote_csv(value))
                else:
                    out_fields.append(value)
            pieces.append("\t".join(out_fields))
        pieces.append("\n")
        self.records += 1

    def _consume_line(self, line: str, pieces: list[str]) -> None:
        """Handle one physical line (without its newline)."""
        if _ends_with_odd_backslashes(line):
            # The newline that ended this physical line is escaped: it is part of the field.
            self._partial.append(line[:-1] + "\\\n")
            return
        if self._partial:
            self._partial.append(line)
            record = "".join(self._partial)
            self._partial = []
        else:
            record = line
        self._emit_record(record, pieces)

    def _fill(self) -> None:
        data = self._source.read(self._chunk_size)
        pieces: list[str] = []
        if not data:
            self._eof = True
            if self._tail or self._partial:
                # Last record without a terminating newline.
                if self._partial:
                    self._partial.append(self._tail)
                    record = "".join(self._partial)
                    self._partial = []
                    self._emit_record(record, pieces)
                else:
                    self._emit_record(self._tail, pieces)
                self._tail = ""
        else:
            lines = (self._tail + data).split("\n")
            self._tail = lines.pop()
            for line in lines:
                self._consume_line(line, pieces)
        if pieces:
            self._out += "".join(pieces)

    def read(self, size: int | None = -1) -> str:
        """Return up to *size* characters of CSV (everything that is left when *size* is negative or ``None``)."""
        if size is None or size < 0:
            while not self._eof:
                self._fill()
            out, self._out = self._out, ""
            return out
        while len(self._out) < size and not self._eof:
            self._fill()
        out, self._out = self._out[:size], self._out[size:]
        return out


def _parse_csv(
    stream: "MySQLDumpCSVStream", columns: list[str], usecols: list[str] | None, text_columns: Collection[str]
) -> pd.DataFrame:
    r"""Parse the CSV produced by :py:class:`MySQLDumpCSVStream` (tab-separated, double-quote quoting, ``\N`` NULL).

    Columns in *text_columns* are read as text whatever they hold (an all-digit ``varchar`` stays text, leading zeros
    included); the types of the other columns are inferred from the whole column.

    Args:
        stream: The CSV stream.
        columns: Column names of the table, in schema order.
        usecols: Columns to keep, or ``None``.
        text_columns: Columns read as text.

    Returns:
        pandas.DataFrame: The parsed table.
    """
    kept = set(columns if usecols is None else usecols)
    return pd.read_csv(
        cast(Any, stream),
        sep="\t",
        header=None,
        names=columns,
        na_values=[_NULL_TOKEN],
        keep_default_na=False,
        low_memory=False,
        usecols=usecols,
        dtype=cast(Any, {c: str for c in text_columns if c in kept} or None),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        doublequote=True,
        skip_blank_lines=False,
    )


def read_mysql_dump(
    source: Any,
    columns: list[str],
    usecols: list[str] | None = None,
    text_columns: Collection[str] = (),
) -> pd.DataFrame:
    """Parse a MySQL table dump into a DataFrame with the values MySQL would return.

    Args:
        source: Readable text stream of the decompressed ``<table>.txt`` dump.
        columns: Column names of the table, in schema order.
        usecols: Optional subset of *columns* to keep (in this order).
        text_columns: Columns that the schema declares as text (character, enumeration, date and time types); they
            are read as text in the default text dtype of pandas, so their type does not depend on their content.

    Returns:
        pandas.DataFrame: One row per record; SQL ``NULL`` is ``NaN``; empty strings stay empty strings.
    """
    stream = MySQLDumpCSVStream(source)
    try:
        df = _parse_csv(stream, columns, usecols, text_columns)
    except pd.errors.EmptyDataError:
        # An empty table: the MySQL server returns no rows, with the table's columns.
        return pd.DataFrame(columns=list(usecols) if usecols is not None else list(columns))
    if stream.restore:
        kept = set(df.columns)
        for record_index, field_index, value in stream.restore:
            column = columns[field_index] if field_index < len(columns) else None
            if column in kept:
                dtype = df[column].dtype
                if not (isinstance(dtype, pd.StringDtype) or pd.api.types.is_object_dtype(dtype)):
                    df[column] = df[column].astype(object)
                df.at[record_index, column] = value
    if usecols is not None:
        df = df[usecols]
    return df
