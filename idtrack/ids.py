"""The vocabulary of idtrack's graph and the one rule of Ensembl ids with versions (``<id>.<version>``).

Node types, the marks of a gene id's birth and retirement (``<id>.Void``, ``<id>.Retired``), the prefix under which
the graph keeps synonyms, and the functions that join and split an Ensembl id and its version. Every module that
needs one of these takes it from here.

This module imports nothing of idtrack except :py:mod:`idtrack.errors`.
"""

from __future__ import annotations

import math
import numbers
from types import MappingProxyType
from typing import Any

from idtrack.errors import IntegrityError, TableFormatError

#: The character between an Ensembl stable id and its version (``ENSG00000141510.18``).
ID_VERSION_DELIMITER = "."

#: The prefix of the graph's synonym nodes (``synonym_id::TP53``): the graph's bookkeeping, never part of an id.
SYNONYM_PREFIX = "synonym_id::"

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

#: Beginnings of the names of the external databases whose ids are sequence accessions, which their sources write
#: with a version (``NM_000546.6``, ``P04637.4``, ``CCDS11118.1``). Only an id of such a database is looked up
#: without a version suffix that the graph does not hold; a gene symbol never is.
ACCESSION_DATABASE_PREFIXES: tuple[str, ...] = ("RefSeq_", "Uniprot/", "CCDS", "EMBL", "protein_id")


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


def stable_id_of(identifier: str) -> str:
    """Return an identifier without the synonym prefix and, for an Ensembl or LRG id, without its version.

    This is how a conversion result is compared with the identifier it started from ("did the identifier change?"):
    an Ensembl id is recognised by its prefix (``ENS``, ``LRG_``), so that a query without version, or in another
    letter case, that maps to itself counts as unchanged.

    Args:
        identifier: An identifier or node name.

    Returns:
        str: The identifier to compare.
    """
    identifier = identifier.removeprefix(SYNONYM_PREFIX)
    if identifier.upper().startswith(("ENS", "LRG_")):
        head, sep, tail = identifier.rpartition(ID_VERSION_DELIMITER)
        if sep and tail.isdigit():
            return head
    return identifier


__all__ = [
    "ACCESSION_DATABASE_PREFIXES",
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
    "SENTINEL_VERSIONS",
    "SYNONYM_PREFIX",
    "VOID",
    "is_versioned_name",
    "join",
    "join_columns",
    "split",
    "stable_id_of",
    "versions_prefix",
]
