"""How a user's identifier is read before it is looked up (:py:mod:`idtrack.reading`; decisions 52, 56, 81, 83 to 85, 93).

Each test names the plausible wrong version it fails under.
"""

from __future__ import annotations

import pytest

from idtrack import reading
from idtrack.errors import ArgumentTypeError, InputError, InvalidArgumentError, MissingValueError
from idtrack.reading import Reading, read_identifier


@pytest.mark.parametrize(
    ("typed", "text", "kinds"),
    [
        ("TP53", "TP53", ()),
        ("  TP53\t\r\n", "TP53", ()),  # white space is no reading
        ("﻿TP53", "TP53", ("unicode",)),  # a byte-order mark at the start of a file
        ("TP53​", "TP53", ("unicode",)),  # zero-width space
        ("TP‍53", "TP53", ("unicode",)),  # zero-width joiner inside
        ("‎TP53", "TP53", ("unicode",)),  # left-to-right mark
        ("‮TP53‬", "TP53", ("unicode",)),  # bidi override
        ("TP­53", "TP53", ("unicode",)),  # soft hyphen
        ("TP53\x00", "TP53", ("unicode",)),  # control character
        ("\x07TP53\x7f", "TP53", ("unicode",)),
        ("ＴＰ５３", "TP53", ("unicode",)),  # full width (NFKC)
        ("HLA–A", "HLA-A", ("unicode",)),  # en dash (category Pd)
        ("HLA‑A", "HLA-A", ("unicode",)),  # non-breaking hyphen (Pd)
        ("HLA⸺A", "HLA-A", ("unicode",)),  # two-em dash (Pd)
        ("HLA−A", "HLA-A", ("unicode",)),  # minus sign (Sm; Unicode's property Dash)
        ("HLA⁻A", "HLA-A", ("unicode",)),  # superscript minus (NFKC: the minus sign)
        ("HLA－A", "HLA-A", ("unicode",)),  # full-width hyphen-minus
        ("'TP53'", "TP53", ("punctuation",)),
        ('"TP53"', "TP53", ("punctuation",)),
        ("“TP53”", "TP53", ("punctuation",)),
        ("«TP53»", "TP53", ("punctuation",)),  # categories Pi and Pf
        ("‹TP53›", "TP53", ("punctuation",)),
        ("＂TP53＂", "TP53", ("unicode", "punctuation")),  # full width: NFKC gives the ASCII quotation mark
        ("TP53;", "TP53", ("punctuation",)),
        ("TP53,", "TP53", ("punctuation",)),
        ("TP53:", "TP53", ("punctuation",)),  # Unicode's Terminal_Punctuation
        ("TP53!?", "TP53", ("punctuation",)),
        ("TP53；", "TP53", ("unicode", "punctuation")),  # full width: NFKC gives the ASCII semicolon
        ("CCDC39.", "CCDC39", ("punctuation",)),
        ("'Agt';", "Agt", ("punctuation",)),
        ("“TP53;”​", "TP53", ("unicode", "punctuation")),
        ("ENSG00000182378_PAR_Y", "ENSG00000182378", ("par_y",)),
        ("ENSG00000182378.14_PAR_Y", "ENSG00000182378.14", ("par_y",)),
        ("ensg00000182378.14_par_y", "ensg00000182378.14", ("par_y",)),
        ("ENST00000381192.10_PAR_Y", "ENST00000381192.10", ("par_y",)),  # GENCODE writes transcripts so too
    ],
)
def test_a_string_is_read_and_the_reading_is_named(typed, text, kinds):
    """Fails if a reading is not applied, or applied without being named (the match would claim ``exact``)."""
    assert read_identifier(typed) == Reading(text, kinds)
    assert Reading(text, kinds).prefix == "".join(f"{k}:" for k in kinds)


@pytest.mark.parametrize(
    "typed",
    [
        "HSP70B'",
        "CK2alpha'",
        "'(P)RR",
        "A;B",
        "TP53.1",
        "NM_000546.6",
        "7157.0",
        "(TP53)",
        "TP 53",
        "S6Kβ",
        "-",
        ".",
        "?",
        "IGH@",
        "hsa-miR-17*",
        "„TP53“",
        "`TP53`",
        "SRF_1",
    ],
)
def test_what_is_part_of_an_identifier_stays(typed):
    """A quote at one end (a prime), inner punctuation, versions, brackets, inner spaces and Greek letters stay.

    So do the punctuation that is no Terminal_Punctuation (``@`` of HGNC's old locus symbols, ``*`` of a miRNA's
    star strand), a low-9 quotation mark (category Ps, with the brackets) and a backtick. Fails if the quotes are
    stripped one by one, or every non-alphanumeric character at the ends is removed.
    """
    assert read_identifier(typed) == Reading(typed, ())


@pytest.mark.parametrize(
    "typed",
    [
        "",
        "   ",
        "\t\n",
        "nan",
        "NaN",
        "NAN",
        "na",
        "NA",
        "N/A",
        "n/a",
        "None",
        "NONE",
        "null",
        "NULL",
        " NA ",
        "NaT",
        "nat",
        "unknown",
        "UNKNOWN",
        "'NA'",
        '"nan"',
        "NA;",
        "﻿nan",
    ],
)
def test_missing_values_are_refused(typed):
    """Decisions 52 and 81: refused before any lookup, naming the value. Fails if a spelling or a reading is missed."""
    with pytest.raises(MissingValueError) as caught:
        read_identifier(typed)
    assert caught.value.argument == "identifier"
    assert caught.value.value == typed


@pytest.mark.parametrize("typed", ["nan", "NA", "unknown", "NaT"])
def test_missing_values_are_looked_up_in_a_named_database(typed):
    """Decision 52: a caller who names the source database has the string looked up. Fails without the exemption."""
    assert read_identifier(typed, from_database="HGNC Symbol") == Reading(typed, ())


def test_an_empty_string_is_missing_even_with_a_database():
    """An empty string is nothing in any database. Fails if the exemption of decision 52 reaches it."""
    with pytest.raises(MissingValueError):
        read_identifier("  ", from_database="HGNC Symbol")


@pytest.mark.parametrize(
    "typed", ["nil", "missing", "-", "?", "inf", "NA1", "NAT1", "NANOG", "unknowns", "NA_character_"]
)
def test_other_strings_are_not_missing_values(typed):
    """Only the spellings of pandas' list and decision 81, whole.

    R's ``NA_character_`` is in neither. Fails if a prefix, a substring or another spelling is taken.
    """
    assert read_identifier(typed).text == typed


def test_the_missing_values_are_pandas_list_and_decision_81s_two():
    """The table is pandas' default list of missing values (case-folded) and ``NaT`` and ``unknown``.

    Fails if a spelling is picked by hand from pandas' list (as ``<NA>`` and ``#N/A`` were left out), or if pandas
    changes its list and the copy does not follow.
    """
    from pandas._libs.parsers import STR_NA_VALUES

    from idtrack import reading

    assert reading.PANDAS_MISSING_VALUES == frozenset(STR_NA_VALUES)
    assert reading.MISSING_VALUES == {value.casefold() for value in STR_NA_VALUES if value} | {"nat", "unknown"}
    for typed in ("<NA>", "#N/A", "-NaN", "1.#QNAN"):
        with pytest.raises(MissingValueError):
            read_identifier(typed)


@pytest.mark.parametrize(
    "typed",
    [
        "synonym_id::TP53",
        "SYNONYM_ID::TP53",
        "ENSG00000141510.Void",
        "ENSG00000141510.void",
        "ENSG00000141510_Void",
        "ENSG00000141510-RETIRED",
        "ENSMUSG00000059552.Retired",
        "LRG_1.Void",
        "'synonym_id::TP53'",
    ],
)
def test_internal_names_are_refused(typed):
    """Decision 56: the graph's bookkeeping is no identifier. Fails if a spelling or separator variant gets through."""
    with pytest.raises(InvalidArgumentError) as caught:
        read_identifier(typed)
    assert caught.value.argument == "identifier"


@pytest.mark.parametrize("typed", ["Void", "AVOID.1", "Retired", "TP53.void"])
def test_names_that_end_like_a_mark_are_read(typed):
    """Only an Ensembl or LRG id before the mark is refused. Fails if every `.void` ending is refused."""
    assert read_identifier(typed).text == typed


def test_a_string_longer_than_any_identifier_is_refused():
    """Decision 85: bounded where it is read, after surrounding white space. Fails without the bound."""
    longest = "A" * reading.MAX_IDENTIFIER_LENGTH
    assert read_identifier(longest).text == longest
    assert read_identifier(" " * 100_000 + "TP53").text == "TP53"
    with pytest.raises(InvalidArgumentError) as caught:
        read_identifier(longest + "A")
    assert caught.value.argument == "identifier"
    assert len(caught.value.value) == reading.MAX_IDENTIFIER_LENGTH


def test_a_version_longer_than_any_ensembl_version_is_refused():
    """Decision 85: five digits at most. Fails if the bound is off by one, or if no bound is kept."""
    reading.version_digits_refused("ENSG00000141510.65535", "65535")
    with pytest.raises(InvalidArgumentError):
        reading.version_digits_refused("ENSG00000141510.999999", "999999")


@pytest.mark.parametrize("value", [5, None, b"TP53", ["TP53"], 1.5])
def test_a_value_that_is_not_a_string_is_a_type_error(value):
    """The type is checked first. Fails if a non-string is coerced."""
    with pytest.raises(ArgumentTypeError):
        read_identifier(value)


def test_every_refusal_is_an_input_error():
    """A server maps every refusal to the caller's error."""
    for typed in ("", "nan", "synonym_id::X", "A" * 5000):
        with pytest.raises(InputError):
            read_identifier(typed)


def test_the_readings_are_documented():
    """Every kind a reading names has its text among the prefixes of a match."""
    from idtrack.results import MATCH_PREFIXES

    assert {"unicode", "punctuation", "par_y"} <= set(MATCH_PREFIXES)
    assert all(MATCH_PREFIXES.values())
