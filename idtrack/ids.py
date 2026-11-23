"""The vocabulary of idtrack's graph and the one rule of Ensembl ids with versions (``<id>.<version>``).

Node types, the marks of a gene id's birth and retirement (``<id>.Void``, ``<id>.Retired``), the prefix under which
the graph keeps synonyms, and the functions that join and split an Ensembl id and its version. Every module that
needs one of these takes it from here.

This module imports nothing of idtrack except :py:mod:`idtrack.errors`.
"""

from __future__ import annotations

import math
import numbers
import re
from types import MappingProxyType
from typing import Any

from idtrack.errors import IntegrityError, TableFormatError

#: The character between an Ensembl stable id and its version (``ENSG00000141510.18``).
ID_VERSION_DELIMITER = "."

#: The prefix of the graph's synonym nodes (``synonym_id::TP53``): the graph's bookkeeping, never part of an id.
SYNONYM_PREFIX = "synonym_id::"

#: The characters that join the parts of a name (``MIR138-2``, ``H1_0``, ``NKX2.1``): the step ``"separator"`` of
#: the name resolution reads them as one (decision 87). A version (``NM_000546.6``, ``ENSG00000141510.18``) follows
#: one of them, and so does the number a single-cell tool appends to a repeated name (``SRF.1``, ``SRF-1``,
#: ``SRF_1``: decision 82).
SEPARATORS = "-_."

#: The form of an Ensembl stable id without its version, by Ensembl's scheme of identifiers: ``ENS``, the species'
#: letters and the letter of the feature, then eleven digits (``ENSG00000141510``, ``ENSMUST00000000001``), or an LRG
#: record's id (``LRG_<number>``). A pattern to embed (no anchors, no flags).
STABLE_ID_FORM = r"(?:ENS[A-Z]*\d{11}|LRG_\d+)"
#: An Ensembl stable id or an LRG id without version, in any letter case (:py:data:`STABLE_ID_FORM`).
STABLE_ID = re.compile(rf"(?i)^{STABLE_ID_FORM}$")
#: An LRG record's id, with or without version.
LRG_ID = re.compile(r"^LRG_\d+(?:\.\d+)?$")
#: A string that ends in one of the :py:data:`SEPARATORS` and digits: ``(string before, separator, digits)``.
NUMBERED_SUFFIX = re.compile(rf"^(.+)([{re.escape(SEPARATORS)}])([0-9]+)$")

#: The version of the node before the first version of a gene id (its birth).
VOID = "Void"
#: The version of the node after the last version of a gene id (its retirement).
RETIRED = "Retired"
#: Both marks: versions that no release holds.
SENTINEL_VERSIONS: frozenset[str] = frozenset({VOID, RETIRED})

#: The key of a node's type among its attributes.
NODE_TYPE_KEY = "node_type"
#: The key of an edge's cross-reference record among its attributes.
CONNECTION_KEY = "connection"

#: The node type of an identifier of an external database (a symbol, an accession, ...).
EXTERNAL = "external"
#: Ensembl's forms, in the order in which one leads to the next (a translation belongs to a transcript, which
#: belongs to a gene).
FORMS: tuple[str, ...] = ("gene", "transcript", "translation")
#: The form whose history is the backbone of the graph.
BACKBONE_FORM = "gene"

#: Form to the node type of its versioned ids (``gene`` -> ``ensembl_gene``).
ENSEMBL_TYPE = MappingProxyType({form: f"ensembl_{form}" for form in FORMS})
#: Node type of versioned ids to its form (``ensembl_gene`` -> ``gene``).
FORM_OF_ENSEMBL_TYPE = MappingProxyType({node_type: form for form, node_type in ENSEMBL_TYPE.items()})
#: Form to the node type of its ids without version (``gene`` -> ``base_ensembl_gene``).
BASE_ENSEMBL_TYPE = MappingProxyType({form: f"base_ensembl_{form}" for form in FORMS})

#: The node type of versioned gene ids, the backbone.
GENE = ENSEMBL_TYPE[BACKBONE_FORM]
#: The node type of gene ids without version.
BASE_GENE = BASE_ENSEMBL_TYPE[BACKBONE_FORM]


def _is_missing(value: Any) -> bool:
    """Tell whether a version cell holds nothing (``None``, NaN, ``pandas.NA``, an empty string)."""
    if value is None or type(value).__name__ == "NAType":
        return True
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return math.isnan(float(value))
    return isinstance(value, str) and value == ""


def join(stable_id: str, version: Any) -> str:
    """Return the node name of an Ensembl id and its version: ``<id>.<version>``.

    A whole number (also a float such as ``3.0`` read from a table) is written as an integer; ``Void`` and
    ``Retired`` are written as they are; a missing version (``None``, NaN, an empty string: the empty id of a
    transcript without a translation) gives the id alone.

    Args:
        stable_id: The stable id.
        version: The version.

    Returns:
        str: The node name.

    Raises:
        TableFormatError: If the version is a number with a fractional part (a malformed table).
    """
    if _is_missing(version):
        return stable_id
    if version in SENTINEL_VERSIONS:
        return f"{stable_id}{ID_VERSION_DELIMITER}{version}"
    if int(version) != float(version):
        raise TableFormatError(f"Version is floating: {(stable_id, version)}")
    return f"{stable_id}{ID_VERSION_DELIMITER}{int(version)}"


def join_columns(stable_ids: Any, versions: Any) -> list[str]:
    """Return the node names of two columns of a table (stable ids and versions), as :py:func:`join` writes them.

    Numeric versions (missing ones as NaN) are joined column-wise; any other column element by element.

    Args:
        stable_ids: The stable ids (a sequence or array of strings).
        versions: Their versions, in the same order (a sequence or array).

    Returns:
        list[str]: One node name per row.

    Raises:
        TableFormatError: If a version is a number with a fractional part.
    """
    import numpy as np

    names = np.asarray(stable_ids, dtype=object).copy()  # never write into the caller's column
    values = np.asarray(versions)
    if not np.issubdtype(values.dtype, np.number):
        return [join(i, v) for i, v in zip(names.tolist(), values.tolist())]
    has = ~np.isnan(values.astype(float))
    if has.any():
        numbers = values[has]
        whole = numbers.astype(np.int64)
        if np.any(whole != numbers):
            bad = int(np.flatnonzero(whole != numbers)[0])
            raise TableFormatError(f"Version is floating: {(names[has][bad], numbers[bad])}")
        names[has] = names[has] + ID_VERSION_DELIMITER + whole.astype(str).astype(object)
    return names.tolist()


def is_versioned_name(name: str) -> bool:
    """Tell whether a name has the form of an Ensembl node name with version: exactly one delimiter.

    Args:
        name: A node name.

    Returns:
        bool: Whether :py:func:`split` reads it.
    """
    return name.count(ID_VERSION_DELIMITER) == 1


def split(name: str) -> tuple[str, str]:
    """Return the stable id and the version of an Ensembl node name ``<id>.<version>``.

    Args:
        name: The node name.

    Returns:
        tuple[str, str]: ``(stable id, version)``; the version as written (digits, ``Void`` or ``Retired``).

    Raises:
        IntegrityError: If the name does not hold exactly one delimiter.
    """
    if not is_versioned_name(name):
        raise IntegrityError(
            f"An Ensembl node name must be `<id>{ID_VERSION_DELIMITER}<version>`; {name!r} has "
            f"{name.count(ID_VERSION_DELIMITER)} delimiters ({ID_VERSION_DELIMITER!r})."
        )
    stable_id, version = name.split(ID_VERSION_DELIMITER)
    return stable_id, version


def versions_prefix(stable_id: str) -> str:
    """Return the beginning shared by the node names of every version of a stable id (``<id>.``).

    Args:
        stable_id: The stable id.

    Returns:
        str: The prefix; a name that starts with it and continues with digits only is a version of the id.
    """
    return f"{stable_id}{ID_VERSION_DELIMITER}"


#: An Ensembl or LRG id (:py:data:`STABLE_ID_FORM`) with a version.
_VERSIONED_STABLE_ID = re.compile(rf"(?i)^({STABLE_ID_FORM}){re.escape(ID_VERSION_DELIMITER)}(\d+)$")


def stable_id_of(identifier: str) -> str:
    """Return an identifier without its version when it is an Ensembl or LRG id with a version.

    This is how a conversion result is compared with the identifier it started from ("did the identifier change?"),
    so that a query without version, or in another letter case, that maps to itself counts as unchanged. An Ensembl id
    is recognised by its whole form (a prefix of Ensembl and eleven digits, or ``LRG_<number>``), so that a symbol
    such as ``ENSA`` is never cut.

    Args:
        identifier: An identifier.

    Returns:
        str: The identifier to compare.
    """
    found = _VERSIONED_STABLE_ID.match(identifier)
    return found.group(1) if found is not None else identifier


def public_name(name: str) -> str:
    """Return a node name or database name as results show it: without the graph's synonym prefix (decision 56).

    The graph keeps a synonym under ``synonym_id::<synonym>`` and its database as ``synonym_id::<database>``; a
    result says that a match is a synonym in its own words (``match="synonym:exact"``, :py:attr:`NameMatch.synonym`,
    a path node of kind ``synonym``) and names the identifier and the database as they are.

    Args:
        name: A node name or a database name of the graph.

    Returns:
        str: The name without the prefix.
    """
    return name.removeprefix(SYNONYM_PREFIX)


def is_synonym_database(database: str) -> bool:
    """Tell whether a database name of the graph is the synonyms of a database (``synonym_id::<database>``)."""
    return database.startswith(SYNONYM_PREFIX)


__all__ = [
    "BACKBONE_FORM",
    "BASE_ENSEMBL_TYPE",
    "BASE_GENE",
    "CONNECTION_KEY",
    "ENSEMBL_TYPE",
    "EXTERNAL",
    "FORMS",
    "FORM_OF_ENSEMBL_TYPE",
    "GENE",
    "ID_VERSION_DELIMITER",
    "NODE_TYPE_KEY",
    "RETIRED",
    "LRG_ID",
    "NUMBERED_SUFFIX",
    "SENTINEL_VERSIONS",
    "SEPARATORS",
    "STABLE_ID",
    "STABLE_ID_FORM",
    "SYNONYM_PREFIX",
    "VOID",
    "is_synonym_database",
    "is_versioned_name",
    "join",
    "join_columns",
    "public_name",
    "split",
    "stable_id_of",
    "versions_prefix",
]
