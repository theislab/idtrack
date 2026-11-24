"""How an identifier a user typed or pasted is read before it is looked up in a graph.

What people paste carries more than the identifier: white space, invisible characters (a byte-order mark at the start
of a file, zero-width spaces, bidirectional marks, soft hyphens, control characters), full-width letters and Unicode
dashes, quotation marks and the punctuation that ends a sentence or an item of a list (``A1BG;``, ``CCDC39.``), the
placeholder of a missing value (``nan``, ``NA``), and sometimes the form GENCODE gives the Y copies of the genes and
transcripts of the pseudoautosomal regions (``ENSG00000182378.14_PAR_Y``).

Characters are classed by the Unicode standard (:py:mod:`unicodedata`): by their general category where one says what
they are, and, where the standard defines the class by a property that :py:mod:`unicodedata` does not expose, by the
few members of that property that remain after NFKC (each such constant below says so).
:py:func:`read_identifier` is the one place that turns such a string into the text that is looked up, and says what it
changed, so that a conversion reports it (:py:attr:`idtrack.results.ConversionResult.match`, whose prefixes
``unicode``, ``punctuation`` and ``par_y`` :py:data:`idtrack.results.MATCH_PREFIXES` describes).

Every rule here is a rule of the string alone, the same in every graph. This module imports nothing of idtrack except
:py:mod:`idtrack.errors` and :py:mod:`idtrack.ids`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from idtrack import ids
from idtrack.errors import ArgumentTypeError, InvalidArgumentError, MissingValueError

#: The longest identifier that is read (after surrounding white space is removed). A bound on the work a string can
#: cause, not a rule of identifiers: it is set where it is read, before any graph is consulted, so that one string
#: gets one answer in every graph; the longest name in the graphs of the five organisms of idtrack's table has about
#: a hundred characters, and every bound above that gives the same answers.
MAX_IDENTIFIER_LENGTH = 1000

#: The largest version of an Ensembl id: Ensembl's core schema stores a version as ``SMALLINT UNSIGNED`` (16 bits).
MAX_ENSEMBL_VERSION = 2**16 - 1
#: The most digits a version of an Ensembl id can have (:py:data:`MAX_ENSEMBL_VERSION`).
MAX_VERSION_DIGITS = len(str(MAX_ENSEMBL_VERSION))

#: The strings that pandas reads as a missing value by default (``pandas._libs.parsers.STR_NA_VALUES`` of pandas 3.1):
#: the lists and tables people paste are mostly written by pandas or read with it. Copied here, not read,
#: because the query path imports no pandas (it needs numpy and networkx only); a unit test compares the copy with
#: pandas' own list, so that a change of the source fails a test instead of drifting.
PANDAS_MISSING_VALUES: frozenset[str] = frozenset(
    {
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)
#: The spellings of a missing value, compared without letter case after the reading below: pandas' missing values
#: (:py:data:`PANDAS_MISSING_VALUES`) and two that idtrack adds, pandas' text of a missing date or time
#: (``NaT``) and UniProt's placeholder name (``unknown``). They stand for nothing far more often than for a gene
#: (``NaN`` is an old alias of SCN11A, ``NA`` a synonym of a few genes); a caller who names the source database
#: (``from_database``) has them looked up as identifiers of that database.
MISSING_VALUES: frozenset[str] = frozenset({value.casefold() for value in PANDAS_MISSING_VALUES if value}) | {
    "nat",
    "unknown",
}

# Removed before matching (:py:func:`_visible`): every format character (Unicode category Cf: zero-width space,
# joiners, word joiner, byte-order mark, bidirectional marks and isolates, soft hyphen, ...) and every control
# character (Cc) that is not white space (white space is stripped at the ends; inside, it is no part of an identifier).
_REMOVED_CATEGORIES = ("Cf", "Cc")
# Read as the hyphen-minus after NFKC: the dashes (Unicode category Pd: hyphen, non-breaking hyphen, en and em dash,
# ...), and the minus sign U+2212, which Unicode's property Dash counts among the dashes while its category is that of
# mathematical symbols (Sm); :py:mod:`unicodedata` exposes categories, not that property, and NFKC turns the superscript
# and subscript minus into this one sign.
_DASH_CATEGORY = "Pd"
_MINUS_SIGN = "\u2212"
# Quotation marks (Unicode's property Quotation_Mark): the typographic ones are the categories of initial and final
# quotes (Pi, Pf: ``‘ ’ “ ” « »``); the two of ASCII, apostrophe and quotation mark, belong to category Po with all other
# ASCII punctuation, so they are named (NFKC turns their full-width forms into them). The low-9 marks (category Ps,
# with the opening brackets) are not read as quotes.
_QUOTE_CATEGORIES = ("Pi", "Pf")
_ASCII_QUOTES = "'\""
# The punctuation that ends a sentence, a clause or an item of a list (Unicode's property Terminal_Punctuation): its
# members of ASCII, which NFKC gives for their full-width forms; the property is not exposed by :py:mod:`unicodedata`.
# No identifier of the graphs ends with ``!`` or ``:``; those that end with ``;``, ``,``, ``.`` or ``?`` are names that
# databases wrote with the punctuation of the text they came from (``A1BG;``, ``14-3-3?``).
_TERMINAL_PUNCTUATION = "!,.:;?"
# The graph's marks of a gene id's birth and retirement after an Ensembl or LRG id, with any separator (decision 56).
_SENTINEL = re.compile(
    rf"(?i)^({ids.STABLE_ID_FORM})[{re.escape(ids.SEPARATORS)}](?:{'|'.join(sorted(ids.SENTINEL_VERSIONS))})$"
)
# GENCODE's Y copy of a gene or transcript of a pseudoautosomal region: <Ensembl id>[.<version>]_PAR_Y.
_PAR_Y = re.compile(rf"(?i)^({ids.STABLE_ID_FORM}(?:\.\d+)?)_PAR_Y$")


@dataclass(frozen=True)
class Reading:
    """A string as idtrack reads it.

    Attributes:
        text (str): The text that is looked up.
        kinds (tuple[str, ...]): What the reading changed, keys of :py:data:`idtrack.results.MATCH_PREFIXES` in the order
            applied (``unicode``, ``punctuation``, ``par_y``); empty when the string was used as given (but for
            surrounding white space).
    """

    text: str
    kinds: tuple[str, ...] = ()

    @property
    def prefix(self) -> str:
        """The prefix of the match kind that says what the reading changed (``"unicode:punctuation:"``, or ``""``)."""
        return "".join(f"{kind}:" for kind in self.kinds)


def normalise_unicode(text: str) -> str:
    """Return a string with invisible and control characters removed, NFKC applied and Unicode dashes as ``-``.

    Args:
        text: The string.

    Returns:
        str: The string as it is matched; the string itself when nothing changes.
    """
    if text.isascii() and text.isprintable():
        return text
    normal = unicodedata.normalize("NFKC", _visible(text))
    return "".join("-" if c == _MINUS_SIGN or unicodedata.category(c) == _DASH_CATEGORY else c for c in normal).strip()


def _visible(text: str) -> str:
    """Return a string without its format characters and its control characters that are not white space."""
    return "".join(c for c in text if c.isspace() or unicodedata.category(c) not in _REMOVED_CATEGORIES)


def _is_quote(character: str) -> bool:
    """Tell whether a character is a quotation mark (:py:data:`_QUOTE_CATEGORIES`, :py:data:`_ASCII_QUOTES`)."""
    return character in _ASCII_QUOTES or unicodedata.category(character) in _QUOTE_CATEGORIES


def strip_punctuation(text: str) -> str:
    """Return a string without the quotation marks around it and the punctuation that ends a sentence at its end.

    Repeatedly: the characters of Unicode's Terminal_Punctuation at the end (``;``, ``,``, ``.``, ``:``, ``!``,
    ``?``), and a pair of quotation marks that encloses the string (``'AGT'``, ``“TP53”``). A quotation mark at one end
    only stays (``HSP70B'`` is a name), as does punctuation inside a string.

    Args:
        text: The string.

    Returns:
        str: The string as it is matched; the string itself when nothing changes (or nothing would be left).
    """
    current = text
    while True:
        before = current
        current = current.rstrip(_TERMINAL_PUNCTUATION).rstrip()
        if len(current) > 2 and _is_quote(current[0]) and _is_quote(current[-1]):
            current = current[1:-1].strip()
        if current == before:
            break
    return current if current else text


def is_missing_value(text: str) -> bool:
    """Tell whether a string (as read) is a spelling of a missing value (:py:data:`MISSING_VALUES`, or empty).

    Args:
        text: The string.

    Returns:
        bool: Whether it stands for nothing.
    """
    return not text.strip() or text.strip().casefold() in MISSING_VALUES


def internal_name(text: str) -> str | None:
    """Return why a string is one of the graph's internal names, or ``None`` when it is not one.

    The graph keeps synonyms under the prefix ``synonym_id::`` and marks the birth and retirement of a gene id with the
    versions ``Void`` and ``Retired`` (``ENSG00000141510.Void``); these are its bookkeeping, never identifiers.

    Args:
        text: The string (as read).

    Returns:
        str | None: The reason, for the error message.
    """
    if text.casefold().startswith(ids.SYNONYM_PREFIX.casefold()):
        return f"it begins with the graph's internal prefix {ids.SYNONYM_PREFIX!r}, which no identifier has"
    if _SENTINEL.match(text) is not None:
        return "`Void` and `Retired` are the graph's marks of a gene id's birth and retirement, not versions"
    return None


def read_identifier(value: Any, argument: str = "identifier", from_database: str | None = None) -> Reading:
    """Read an identifier as idtrack looks it up, or refuse it.

    In this order: surrounding white space is removed; the string is refused when it is empty, longer than
    :py:data:`MAX_IDENTIFIER_LENGTH`, a spelling of a missing value (unless ``from_database`` is named) or one of the
    graph's internal names; Unicode is normalised (:py:func:`normalise_unicode`, kind ``"unicode"``); quotes around it
    and stray punctuation at its end are removed (:py:func:`strip_punctuation`, kind ``"punctuation"``); GENCODE's
    ``<id>_PAR_Y`` is read as ``<id>`` (kind ``"par_y"``). The checks of a missing value and of an internal name see
    the string after the Unicode and punctuation readings too (``'NA'`` in quotes is ``NA``).

    Args:
        value: The identifier given.
        argument: The name of the argument, for the error message.
        from_database: The database the caller names as the identifier's source (``None``: none).

    Returns:
        Reading: The text to look up and what the reading changed.

    Raises:
        ArgumentTypeError: If the value is not a string.
        MissingValueError: If it is empty or a spelling of a missing value (and no source database is named).
        InvalidArgumentError: If it is longer than any identifier, or one of the graph's internal names.
    """
    if not isinstance(value, str):
        raise ArgumentTypeError(
            f"`{argument}` must be a string, got {type(value).__name__} {value!r}.", argument=argument, value=value
        )
    text = value.strip()
    if not text:
        raise MissingValueError(f"`{argument}` is empty: {value!r}.", argument=argument, value=value)
    if len(text) > MAX_IDENTIFIER_LENGTH:
        raise InvalidArgumentError(
            f"`{argument}` has {len(text)} characters, more than any identifier ({MAX_IDENTIFIER_LENGTH} at most): "
            f"{text[:40]!r}...",
            argument=argument,
            value=text[:MAX_IDENTIFIER_LENGTH],
        )
    kinds: list[str] = []
    normal = normalise_unicode(text)
    if normal != text and normal:
        kinds.append("unicode")
        text = normal
    bare = strip_punctuation(text)
    if bare != text:
        kinds.append("punctuation")
        text = bare
    if from_database is None and is_missing_value(text):
        raise MissingValueError(
            f"`{argument}`={value!r} stands for a missing value, not an identifier; name its source database "
            "(`from_database`) to look it up as an identifier of that database.",
            argument=argument,
            value=value,
        )
    why = internal_name(text)
    if why is not None:
        raise InvalidArgumentError(
            f"`{argument}`={value!r} is not an identifier: {why}.", argument=argument, value=value
        )
    par_y = _PAR_Y.match(text)
    if par_y is not None:
        kinds.append("par_y")
        text = par_y.group(1)
    return Reading(text, tuple(kinds))


def version_digits_refused(identifier: str, digits: str) -> None:
    """Refuse a version with more digits than any Ensembl version (:py:data:`MAX_VERSION_DIGITS`).

    Args:
        identifier: The identifier, for the message.
        digits: The digits of its version.

    Raises:
        InvalidArgumentError: If the version is longer.
    """
    if len(digits) > MAX_VERSION_DIGITS:
        raise InvalidArgumentError(
            f"`identifier`={identifier[:60]!r}... has a version of {len(digits)} digits; an Ensembl version has at "
            f"most {MAX_VERSION_DIGITS}.",
            argument="identifier",
            value=identifier[:MAX_IDENTIFIER_LENGTH],
        )


__all__ = [
    "MAX_ENSEMBL_VERSION",
    "MAX_IDENTIFIER_LENGTH",
    "MAX_VERSION_DIGITS",
    "MISSING_VALUES",
    "PANDAS_MISSING_VALUES",
    "Reading",
    "internal_name",
    "is_missing_value",
    "normalise_unicode",
    "read_identifier",
    "strip_punctuation",
    "version_digits_refused",
]
