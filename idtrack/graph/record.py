"""The record of a graph file: what the graph is and what built it, one typed record, each fact once.

A graph file (:py:meth:`idtrack.graph.TheGraph.save`) holds its arrays and one :py:class:`GraphRecord`. The record
says which graph the file holds (organism, primary assembly, release window, forms, the selection of external
databases and the organism table it was built with: :py:meth:`GraphRecord.identity`), what the graph covers (its
releases, the assembly that serves each, the external databases per assembly), what built it (package version and
commit, date, the tables read and their fingerprints, the counts of the build and its history report) and the digest
of its content. This module imports nothing beyond the standard library and :py:mod:`idtrack.errors`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from idtrack.errors import IntegrityError

#: Version of the graph file layout and of the build rules behind it. It is raised whenever a graph built by an
#: earlier version of the code must not be loaded any more (a change of the file layout, of what an edge means, of
#: how names are chosen or of how the history is validated). There is no reader of an earlier version.
GRAPH_FORMAT_VERSION = 6

#: The fields of a record that say which graph a file holds (what a caller asks for when it loads a file by name).
IDENTITY_FIELDS = (
    "organism",
    "primary_assembly",
    "first_release",
    "window_end",
    "snapshot_release",
    "forms",
    "session_details",
    "all_external",
    "externals_fingerprint",
    "organism_table_fingerprint",
)


@dataclasses.dataclass(frozen=True)
class GraphRecord:
    """What a graph is and what built it.

    Attributes:
        organism (str): The organism.
        primary_assembly (int): The assembly of the snapshot release, the organism's default primary assembly there.
        first_release (int): The first release of the window the graph was built for.
        window_end (int | None): The last release of that window (``None``: no limit); the snapshot is the newest
            release of the primary assembly in the window.
        snapshot_release (int): The snapshot: the last release of the graph.
        releases (tuple[int, ...]): Every release the graph covers, ascending.
        serving_assembly (dict[int, int]): Release to the assembly that serves it in the backbone.
        releases_per_assembly (dict[int, tuple[int, ...]]): Assembly to the releases its cross-references carry.
        forms (tuple[str, ...]): The Ensembl forms of the graph.
        session_details (bool): The history report lists Ensembl's mapping sessions with their database names,
            assemblies and dates (the graph itself is the same).
        all_external (bool): Every external database of Ensembl is in the graph, not only the selected ones.
        externals_fingerprint (str): Digest of the selection of external databases that took effect.
        organism_table_fingerprint (str): Digest of the organism's assemblies in the organism table.
        externals_settings (dict[int, int]): Release to the release of the externals configuration whose settings
            it took (itself where the configuration lists it; the last listed release for a later one).
        external_databases (dict[int, tuple[str, ...]]): Assembly to the external databases whose ids it has.
        connection_form (dict[str, str]): External database to the Ensembl form its ids are attached to.
        label_names (dict[str, str]): The external databases whose ids are labels (display names, symbols) rather
            than accessions, by Ensembl's database name (``external_db.db_name``): the one name a database's label rows
            take at every release of the graph (decision 53), for the databases whose labels the graph holds.
        renamed_databases (dict[str, str]): Names that Ensembl gives a database's labels at some release and the
            graph does not, with the name the graph gives them (one name at every release).
        misplaced_external_entries (tuple[str, ...]): External ids left out because they are Ensembl ids.
        entries_of_several_genes (tuple[str, ...]): Entries of the databases of
            :py:data:`idtrack.build.database_manager.ONE_GENE_ONLY` left out because Ensembl gives them to genes at
            several loci over the releases of the graph (lower case).
        database_names (dict[str, dict[str, tuple[tuple[int, int], ...]]]): Per database of the graph's selection, the
            names of Ensembl's databases it holds and the releases (inclusive ranges) at which they hold links: its own
            name on every form, and the names whose links continue it (decision 104,
            :py:mod:`idtrack.build.continuity`).
        assembly_mappings (dict[str, dict[str, Any]]): Ensembl's mappings of one assembly onto a later one that the
            graph carried places by (decision 103): per chain file, ``from`` and ``to`` (assembly keys), the
            ``release`` whose ``assembly_chain`` folder held it and its ``sha256``; empty where Ensembl publishes none.
        package_version (str | None): The idtrack version that built the graph; for a graph built from a source folder
            that is not an installed idtrack, the statement :py:data:`idtrack._version.SOURCE_FOLDER_BUILD` (no
            released version names that code; the commit does, where it was known).
        package_commit (str | None): The git commit of idtrack that built the graph, where it was known.
        build_date (str | None): When the graph was built (UTC, ISO 8601).
        tables (dict[str, Any]): The kinds of tables read and their row counts.
        source_tables (dict[str, Any]): Per assembly, the number of table files and a fingerprint over them.
        counts (dict[str, int]): Counts of the build (nodes, edges, merged spellings, places, ...).
        history (dict[str, Any]): The history report of the build (sessions, corrections, the events placed
            otherwise than Ensembl recorded them, the events left out).
        content_digest (str): Digest of the graph's content (nodes, edges, connections, spellings, places, marks).
    """

    organism: str
    primary_assembly: int
    first_release: int
    window_end: int | None
    snapshot_release: int
    releases: tuple[int, ...]
    serving_assembly: dict[int, int]
    releases_per_assembly: dict[int, tuple[int, ...]]
    forms: tuple[str, ...]
    session_details: bool
    all_external: bool
    externals_fingerprint: str
    organism_table_fingerprint: str
    externals_settings: dict[int, int]
    external_databases: dict[int, tuple[str, ...]]
    connection_form: dict[str, str]
    label_names: dict[str, str]
    renamed_databases: dict[str, str]
    misplaced_external_entries: tuple[str, ...]
    entries_of_several_genes: tuple[str, ...]
    database_names: dict[str, dict[str, tuple[tuple[int, int], ...]]]
    assembly_mappings: dict[str, dict[str, Any]]
    package_version: str | None
    package_commit: str | None
    build_date: str | None
    tables: dict[str, Any]
    source_tables: dict[str, Any]
    counts: dict[str, int]
    history: dict[str, Any]
    content_digest: str

    def identity(self) -> dict[str, Any]:
        """Return the fields that say which graph this is (:py:data:`IDENTITY_FIELDS`), as JSON values."""
        document = self.to_json()
        return {name: document[name] for name in IDENTITY_FIELDS}

    def to_json(self) -> dict[str, Any]:
        """Return the record as a JSON document (integer keys become strings, tuples lists)."""
        return {field.name: _json(getattr(self, field.name)) for field in dataclasses.fields(self)}

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> GraphRecord:
        """Read a record written by :py:meth:`to_json`.

        Raises:
            IntegrityError: If a field is missing, unknown or of the wrong type.
        """
        names = {field.name for field in dataclasses.fields(cls)}
        missing, unknown = sorted(names - set(document)), sorted(set(document) - names)
        if missing or unknown:
            raise IntegrityError(f"A graph record lacks the fields {missing} or has unknown fields {unknown}.")
        try:
            return cls(
                organism=_text(document["organism"]),
                primary_assembly=_integer(document["primary_assembly"]),
                first_release=_integer(document["first_release"]),
                window_end=None if document["window_end"] is None else _integer(document["window_end"]),
                snapshot_release=_integer(document["snapshot_release"]),
                releases=tuple(_integer(r) for r in document["releases"]),
                serving_assembly={_integer(r): _integer(a) for r, a in document["serving_assembly"].items()},
                releases_per_assembly={
                    _integer(a): tuple(_integer(r) for r in rs) for a, rs in document["releases_per_assembly"].items()
                },
                forms=tuple(_text(f) for f in document["forms"]),
                session_details=_flag(document["session_details"]),
                all_external=_flag(document["all_external"]),
                externals_fingerprint=_text(document["externals_fingerprint"]),
                organism_table_fingerprint=_text(document["organism_table_fingerprint"]),
                externals_settings={_integer(r): _integer(s) for r, s in document["externals_settings"].items()},
                external_databases={
                    _integer(a): tuple(_text(d) for d in dbs) for a, dbs in document["external_databases"].items()
                },
                connection_form={_text(d): _text(f) for d, f in document["connection_form"].items()},
                label_names={_text(d): _text(n) for d, n in document["label_names"].items()},
                renamed_databases={_text(o): _text(n) for o, n in document["renamed_databases"].items()},
                misplaced_external_entries=tuple(_text(e) for e in document["misplaced_external_entries"]),
                entries_of_several_genes=tuple(_text(e) for e in document["entries_of_several_genes"]),
                database_names={
                    _text(d): {
                        _text(n): tuple((_integer(a), _integer(b)) for a, b in ranges) for n, ranges in held.items()
                    }
                    for d, held in document["database_names"].items()
                },
                assembly_mappings={
                    _text(name): {
                        "from": _integer(m["from"]),
                        "to": _integer(m["to"]),
                        "release": _integer(m["release"]),
                        "sha256": _text(m["sha256"]),
                    }
                    for name, m in document["assembly_mappings"].items()
                },
                package_version=_optional_text(document["package_version"]),
                package_commit=_optional_text(document["package_commit"]),
                build_date=_optional_text(document["build_date"]),
                tables=dict(document["tables"]),
                source_tables=dict(document["source_tables"]),
                counts={_text(k): _integer(v) for k, v in document["counts"].items()},
                history=dict(document["history"]),
                content_digest=_text(document["content_digest"]),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise IntegrityError(f"A graph record holds a value of the wrong type ({error}).") from error


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"expected an integer, got {value!r}")
    if isinstance(value, str):
        return int(value)
    if isinstance(value, int):
        return value
    raise TypeError(f"expected an integer, got {value!r}")


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {value!r}")
    return value


def _optional_text(value: Any) -> str | None:
    return None if value is None else _text(value)


def _flag(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected true or false, got {value!r}")
    return value


__all__ = ["GRAPH_FORMAT_VERSION", "IDENTITY_FIELDS", "GraphRecord"]
