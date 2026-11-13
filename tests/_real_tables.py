"""Real Ensembl tables, cut down to a few hundred genes, served to ``DatabaseManager`` in place of Ensembl.

The files ``tests/data/real/<organism>.tables.json.xz`` hold, for every genome assembly and release of an organism,
the raw tables of Ensembl's core database that a graph build reads (``gene``, ``transcript``, ``translation``,
``xref``, ``object_xref``, ``external_synonym``, ``identity_xref``, ``external_db``, ``seq_region_attrib``,
``attrib_type``, ``stable_id_event``, ``mapping_session``), restricted to the genes of
``tests/data/real/<organism>.genes.txt``. They are written by ``tests/data/real/build_real_tables.py`` from the
tables that ``DatabaseManager`` downloads over HTTPS; nothing in them is made up.

:py:class:`RealTables` replays them at the two places where ``DatabaseManager`` reaches Ensembl (the download of a
raw table and the index of the core databases), so a graph of the real organism is built by the production code,
from the id history to the external identifiers, without the network.
"""

from __future__ import annotations

import itertools
import json
import lzma
import os
from pathlib import Path
from typing import Any

import pandas as pd

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB, EnsemblSchemaError

DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "real"
FORMAT = 1
#: The surrogate keys of a core database: whole numbers that the file stores as differences between rows.
KEY_COLUMNS = frozenset(
    {"gene_id", "transcript_id", "translation_id", "xref_id", "object_xref_id", "ensembl_id", "seq_region_id"}
)


def tables_path(organism: str) -> Path:
    """Return the file of an organism's cut tables."""
    return DATA_DIRECTORY / f"{organism}.tables.json.xz"


def genes_path(organism: str) -> Path:
    """Return the file that lists an organism's genes (the input of the build script)."""
    return DATA_DIRECTORY / f"{organism}.genes.txt"


def truth_path(organism: str) -> Path:
    """Return the file of the queries and their expected answers."""
    return DATA_DIRECTORY / f"{organism}.truth.jsonl"


def read_genes(organism: str) -> list[str]:
    """Return the gene ids (without version) the tables of an organism are cut down to."""
    lines = genes_path(organism).read_text(encoding="utf-8").splitlines()
    return sorted({line.strip() for line in lines if line.strip() and not line.startswith("#")})


def read_truth(organism: str) -> list[dict[str, Any]]:
    """Return the queries of an organism with their expected answers, in the order of the file."""
    with open(truth_path(organism), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RealTables:
    """The cut tables of one organism, and the replay of Ensembl from them.

    Attributes:
        organism: Ensembl species name.
        snapshot: The release the tables end at (the snapshot release of the graph to build).
        genes: The gene ids the tables are cut down to.
        indices: Per assembly, the saved index of Ensembl's core databases (as ``DatabaseManager`` writes it).
        tables: Per ``(assembly, release, table)`` a dict with ``columns`` (pairs of a column's name and ``"text"``
            or ``"number"``), ``data`` (one list of values per column) and ``absent`` (the columns that Ensembl's
            schema of that release does not have: asking for one raises ``EnsemblSchemaError``, as Ensembl does).
            In the file a key column (:py:data:`KEY_COLUMNS`) is of the kind ``"differences"``: each row's value
            minus the row's before it.
    """

    def __init__(self, organism: str, snapshot: int, genes: list[str], indices: dict, tables: dict):
        """Keep the parts of a set of cut tables."""
        self.organism = organism
        self.snapshot = int(snapshot)
        self.genes = list(genes)
        self.indices = indices
        self.tables = tables
        self.served = 0

    # ------------------------------------------------------------------ the file

    @classmethod
    def load(cls, organism: str, path: str | os.PathLike | None = None) -> RealTables:
        """Read the cut tables of an organism from the repository's test data."""
        with lzma.open(path or tables_path(organism), "rt", encoding="utf-8") as handle:
            document = json.load(handle)
        if document["format"] != FORMAT or document["organism"] != organism:
            raise ValueError(f"{path or tables_path(organism)} is not a table file of {organism} in format {FORMAT}.")
        tables = {}
        for key, table in document["tables"]:
            name, assembly, release = key.split("/")
            for position, (column, kind) in enumerate(table["columns"]):
                if kind == "differences":
                    table["columns"][position] = [column, "number"]
                    table["data"][position] = list(itertools.accumulate(table["data"][position]))
            tables[(int(assembly), int(release), name)] = table
        return cls(organism, document["snapshot"], document["genes"], document["indices"], tables)

    def save(self, path: str | os.PathLike) -> None:
        """Write the cut tables as one compressed JSON document (in a fixed order: equal tables give equal files)."""
        # The tables of one kind follow each other, release after release: consecutive releases of a table differ
        # little, and the compression finds what they share when they are neighbours.
        stored = []
        for (assembly, release, name), table in sorted(self.tables.items(), key=lambda item: (item[0][2], *item[0])):
            columns, data = [], []
            for (column, kind), values in zip(table["columns"], table["data"]):
                if column in KEY_COLUMNS and kind == "number" and all(isinstance(value, int) for value in values):
                    # a key column as the differences between consecutive rows (the first row as it is)
                    kind, values = "differences", [b - a for a, b in zip([0] + values[:-1], values)]
                columns.append([column, kind])
                data.append(values)
            stored.append(
                [f"{name}/{assembly}/{release}", {"columns": columns, "data": data, "absent": table["absent"]}]
            )
        document = {
            "format": FORMAT,
            "organism": self.organism,
            "snapshot": self.snapshot,
            "genes": self.genes,
            "indices": self.indices,
            "tables": stored,
        }
        text = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with lzma.open(path, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as handle:
            handle.write(text)

    # ------------------------------------------------------------------ the replay

    def table(self, assembly: int, release: int, table_key: str, usecols: list | None) -> pd.DataFrame:
        """Return a raw table as ``DatabaseManager.download_table`` returns it from Ensembl.

        Raises:
            EnsemblSchemaError: If Ensembl's schema of the release has no such table or lacks a column asked for.
            AssertionError: If the table or a column is not part of the cut tables (they are then out of date:
                run ``tests/data/real/build_real_tables.py``).
        """
        key = (int(assembly), int(release), table_key)
        entry = self.tables.get(key)
        if entry is None:
            raise AssertionError(
                f"The cut tables of {self.organism} have no table {table_key!r} of assembly {assembly}, release "
                f"{release}: rebuild them with tests/data/real/build_real_tables.py."
            )
        names = [name for name, _kind in entry["columns"]]
        if not names:
            raise EnsemblSchemaError(f"Table {table_key!r} does not exist.", table=table_key)
        wanted = names if usecols is None else list(usecols)
        missing = [column for column in wanted if column not in names]
        if missing:
            unknown = [column for column in missing if column not in entry["absent"]]
            if unknown:
                raise AssertionError(
                    f"The cut tables of {self.organism} lack the columns {unknown} of {table_key!r} (assembly "
                    f"{assembly}, release {release}): rebuild them with tests/data/real/build_real_tables.py."
                )
            raise EnsemblSchemaError(
                f"Missing columns in {table_key!r}: {missing}", table=table_key, missing_columns=tuple(missing)
            )
        position = {name: i for i, name in enumerate(names)}
        typed = [
            (column, "varchar(255)" if entry["columns"][position[column]][1] == "text" else "double")
            for column in wanted
        ]
        rows = list(zip(*(entry["data"][position[column]] for column in wanted), strict=True))
        self.served += 1
        return DatabaseManager._typed_mysql_frame(rows, typed, table_key)

    def write_indices(self, repository: str | os.PathLike) -> None:
        """Put the saved indices of Ensembl's core databases into a local repository, as a build leaves them."""
        for assembly, payload in self.indices.items():
            path = DatabaseManager._https_index_path(str(repository), self.organism, int(assembly))
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)

    def install(self, monkeypatch, repository: str | os.PathLike) -> None:
        """Make ``DatabaseManager`` read these tables instead of Ensembl, for managers of ``repository``.

        The saved indices are written into the repository (a manager whose window ends at the snapshot release then
        needs no scan of Ensembl's dump listings), Ensembl is reached over the HTTPS route only, and the download of
        a raw table is answered from the cut tables.

        Args:
            monkeypatch: pytest's ``monkeypatch`` (or a ``pytest.MonkeyPatch``).
            repository: The local repository of the managers that read the cut tables.
        """
        self.write_indices(repository)
        monkeypatch.setitem(_SERVED, os.path.realpath(str(repository)), self)
        monkeypatch.setattr(DB, "ensembl_access", "https")
        monkeypatch.setattr(DatabaseManager, "download_table", _replayed_download)


#: The repositories whose managers read cut tables: ``{real path: RealTables}`` (several organisms at a time).
_SERVED: dict[str, RealTables] = {}


def _replayed_download(manager, table_key, usecols=None):
    """Answer ``DatabaseManager.download_table`` from the cut tables installed for the manager's repository."""
    tables = _SERVED.get(os.path.realpath(str(manager.local_repository)))
    if tables is None or manager.organism != tables.organism:
        raise AssertionError(f"Unexpected download of {table_key!r} by {manager.organism} outside the cut tables.")
    return tables.table(manager.genome_assembly, manager.ensembl_release, table_key, usecols or None)


def fold(identifier: str) -> str:
    """Return an identifier in lower case with ``-``, ``_`` and ``.`` taken as one character."""
    return identifier.lower().replace("_", "-").replace(".", "-")


class Ensembl:
    """What Ensembl's tables say, read from the cut tables by code that shares nothing with the package.

    The tests take their expectations about a release from here (which genes exist with which version, which gene
    an external identifier belongs to in which database, which genes lie outside the reference sequence), so that
    a test never asserts what the package itself computed.

    A database is named as the graph names it: an accession of Ensembl's ``external_db.db_name`` (``EntrezGene``,
    ``Uniprot/SWISSPROT``), or a label of its ``db_display_name`` (``HGNC Symbol``, ``NCBI gene (formerly
    Entrezgene)``); the synonyms of a cross reference are reported apart.
    """

    def __init__(self, tables: RealTables):
        """Read from a set of cut tables."""
        self.tables = tables
        self._links: dict[tuple[int, int], list[tuple[str, str, str, str, str]]] = {}
        self._genes: dict[tuple[int, int], dict[str, int]] = {}
        self._selection: dict[tuple[str, str, int], set[int]] | None = None
        self._written: dict[int | None, dict[tuple[str, str], set[tuple[str, str]]]] = {}
        self._serving: tuple[Any, dict[tuple[str, str], dict[int, set[str]]]] = (None, {})
        #: ``{release: assembly}``: the assembly that serves each release (set by :py:class:`RealWorld`).
        self.serving: dict[int, int] = {}
        self._beside: dict[tuple[int, int], set[str]] = {}

    def selected(self, form: str, database: str, assembly: int, release: int) -> bool:
        """Tell whether the shipped externals selection includes a database for a form, assembly and release.

        The selection is the package's configuration file ``<organism>_externals_modified.yml``, read here as
        plain YAML. A synonym's database is named with the prefix ``synonym_id::``.
        """
        if self._selection is None:
            import yaml

            import idtrack

            path = Path(idtrack.__file__).parent / "default_config" / f"{self.tables.organism}_externals_modified.yml"
            with open(path, encoding="utf-8") as handle:
                content = yaml.safe_load(handle)[self.tables.organism]
            self._selection = {}
            for the_form, per_form in content.items():
                for name, entry in per_form.items():
                    for key, per_assembly in entry["Assembly"].items():
                        if per_assembly["Include"]:
                            releases = {int(r) for r in str(per_assembly["Ensembl release"]).split(",")}
                            self._selection[(the_form, name, int(key))] = releases
        return int(release) in self._selection.get((form, database, int(assembly)), ())

    def releases(self, assembly: int) -> list[int]:
        """Return the releases of an assembly in the cut tables."""
        return sorted({r for a, r, name in self.tables.tables if a == int(assembly) and name == "xref"})

    def _frame(self, assembly: int, release: int, name: str) -> pd.DataFrame:
        entry = self.tables.tables[(int(assembly), int(release), name)]
        return pd.DataFrame({column: values for (column, _kind), values in zip(entry["columns"], entry["data"])})

    def _ids(self, assembly: int, release: int, form: str) -> pd.DataFrame:
        """Return the id table of a form with ``stable_id`` and ``version``, whichever schema the release has."""
        table = self._frame(assembly, release, form)
        if "stable_id" not in table.columns:
            table = table.merge(self._frame(assembly, release, f"{form}_stable_id"), on=f"{form}_id")
        return table

    def genes(self, assembly: int, release: int) -> dict[str, int]:
        """Return ``{gene id: version}`` of an assembly at a release."""
        where = (int(assembly), int(release))
        if where not in self._genes:
            table = self._ids(assembly, release, "gene")
            self._genes[where] = {i: int(v) for i, v in zip(table["stable_id"], table["version"])}
        return self._genes[where]

    def products(self, assembly: int, release: int) -> dict[str, dict[str, tuple[int, str]]]:
        """Return the transcripts and translations of a release: ``{form: {stable id: (version, gene id)}}``."""
        gene = self._ids(assembly, release, "gene")[["gene_id", "stable_id"]].rename(columns={"stable_id": "gene"})
        transcript = self._ids(assembly, release, "transcript").merge(gene, on="gene_id")
        translation = self._ids(assembly, release, "translation").merge(
            transcript[["transcript_id", "gene"]], on="transcript_id"
        )
        return {
            form: {i: (int(v), g) for i, v, g in zip(table["stable_id"], table["version"], table["gene"])}
            for form, table in (("transcript", transcript), ("translation", translation))
        }

    def events(self, assembly: int, release: int) -> pd.DataFrame:
        """Return Ensembl's gene events as the core database of a release records them, with their sessions.

        Columns: ``old_stable_id``, ``old_version``, ``new_stable_id``, ``new_version``, ``score`` and the
        ``old_release`` and ``new_release`` of the event's mapping session, as numbers (Ensembl records a few
        sessions with wrong releases; they are returned as recorded).
        """
        events = self._frame(assembly, release, "stable_id_event")
        events = events[events["type"] == "gene"]
        sessions = self._frame(assembly, release, "mapping_session")[
            ["mapping_session_id", "old_release", "new_release"]
        ]
        sessions = sessions.assign(
            old_release=pd.to_numeric(sessions["old_release"]), new_release=pd.to_numeric(sessions["new_release"])
        )
        return events.merge(sessions, on="mapping_session_id")

    def non_reference_genes(self, assembly: int, release: int) -> set[str]:
        """Return the genes on sequence regions that Ensembl marks ``non_ref`` (patches, haplotypes, LRG records)."""
        attributes = self._frame(assembly, release, "attrib_type")
        codes = set(attributes.loc[attributes["code"] == "non_ref", "attrib_type_id"])
        marks = self._frame(assembly, release, "seq_region_attrib")
        beside = set(marks.loc[marks["attrib_type_id"].isin(codes), "seq_region_id"])
        table = self._ids(assembly, release, "gene")
        return set(table.loc[table["seq_region_id"].isin(beside), "stable_id"])

    def links(self, assembly: int, release: int) -> list[tuple[str, str, str, str, str]]:
        """Return ``(gene id, database, external id, "id" | "synonym", form)`` for every cross reference of a release.

        A cross reference of a transcript or translation is reported for its gene; ``form`` says what it is
        attached to in Ensembl (``gene``, ``transcript`` or ``translation``). The synonyms of a cross reference
        are reported under the database of its label.
        """
        where = (int(assembly), int(release))
        if where in self._links:
            return self._links[where]
        gene = self._ids(assembly, release, "gene")[["gene_id", "stable_id"]].rename(columns={"stable_id": "gene"})
        transcript = self._ids(assembly, release, "transcript")[["transcript_id", "gene_id"]].merge(gene, on="gene_id")
        translation = self._ids(assembly, release, "translation")[["translation_id", "transcript_id"]].merge(
            transcript, on="transcript_id"
        )
        owner = {
            "gene": dict(zip(gene["gene_id"], gene["gene"])),
            "transcript": dict(zip(transcript["transcript_id"], transcript["gene"])),
            "translation": dict(zip(translation["translation_id"], translation["gene"])),
        }

        def plain(table: str, key: str) -> tuple[dict[int, list], list[str]]:
            """Return the rows of a table as ``{key: [other columns ...]}`` of plain values, and those columns."""
            entry = self.tables.tables[(int(assembly), int(release), table)]
            columns = {column: values for (column, _kind), values in zip(entry["columns"], entry["data"])}
            others = [column for column in columns if column != key]
            return {int(k): row for k, *row in zip(columns[key], *(columns[column] for column in others))}, others

        database_rows, database_columns = plain("external_db", "external_db_id")
        names_at = (database_columns.index("db_name"), database_columns.index("db_display_name"))
        xref_rows, xref_columns = plain("xref", "xref_id")
        at = [xref_columns.index(column) for column in ("external_db_id", "dbprimary_acc", "display_label")]
        synonyms: dict[int, list[str]] = {}
        entry = self.tables.tables[(int(assembly), int(release), "external_synonym")]
        columns = {column: values for (column, _kind), values in zip(entry["columns"], entry["data"])}
        for xref_id, synonym in zip(columns.get("xref_id", ()), columns.get("synonym", ())):
            synonyms.setdefault(int(xref_id), []).append(str(synonym).strip())
        rows = []
        entry = self.tables.tables[(int(assembly), int(release), "object_xref")]
        columns = {column: values for (column, _kind), values in zip(entry["columns"], entry["data"])}
        for kind, ensembl_id, xref_id in zip(columns["ensembl_object_type"], columns["ensembl_id"], columns["xref_id"]):
            form = str(kind).lower()
            gene_id = owner.get(form, {}).get(int(ensembl_id))
            if gene_id is None:
                continue
            record = xref_rows[int(xref_id)]
            database = database_rows[int(record[at[0]])]
            name, label_name = database[names_at[0]], database[names_at[1]]
            for value, of in ((record[at[1]], name), (record[at[2]], label_name)):
                if value is not None and of is not None and str(value).strip():
                    rows.append((gene_id, str(of), str(value).strip(), "id", form))
            if label_name is not None:
                # Ensembl lists synonyms per cross reference; they belong to the label's database
                for synonym in synonyms.get(int(xref_id), ()):
                    if synonym:
                        rows.append((gene_id, str(label_name), synonym, "synonym", form))
        self._links[where] = sorted(set(rows))
        return self._links[where]

    def genes_of(
        self, external_id: str, assembly: int, release: int, database: str | None = None, synonyms: bool = False
    ) -> set[str]:
        """Return the genes that carry an external identifier at a release (in one database, or in any)."""
        kinds = ("id", "synonym") if synonyms else ("id",)
        return {
            gene
            for gene, name, value, kind, _form in self.links(assembly, release)
            if value == external_id and kind in kinds and (database is None or name == database)
        }

    def written(self, release: int | None = None) -> dict[tuple[str, str], set[tuple[str, str]]]:
        """Return how the selected databases write their identifiers: ``{(kind, folded): {(spelling, database)}}``.

        ``kind`` is ``"id"`` or ``"synonym"``; ``folded`` is the identifier in lower case with ``-``, ``_`` and ``.``
        taken as one character. Only what the externals selection includes for the form, assembly and release is
        counted. At a release: on the assembly that serves it (:py:attr:`serving`; on every assembly that has the
        release when that is not known). ``None``: at any release of any assembly.
        """
        if release not in self._written:
            found: dict[tuple[str, str], set] = {}
            assemblies = sorted({a for a, _r, name in self.tables.tables if name == "xref"})
            if release is not None and self.serving:
                assemblies = [self.serving[release]] if release in self.serving else []
            for assembly in assemblies:
                for at in self.releases(assembly) if release is None else [release]:
                    if at not in self.releases(assembly):
                        continue
                    for _gene, name, value, kind, form in self.links(assembly, at):
                        database = name if kind == "id" else f"{DB.synonym_id_nodes_prefix}{name}"
                        if self.selected(form, database, assembly, at):
                            found.setdefault((kind, fold(value)), set()).add((value, database))
            self._written[release] = found
        return self._written[release]

    def spellings_on(self, backbone: dict[int, int], text: str, kind: str = "id") -> dict[int, set[str]]:
        """Return how an identifier is written at each release at which it exists, letter case ignored.

        Read on the assembly that serves each release (``backbone``: ``{release: assembly}``), from the selected
        databases; ``kind`` is ``"id"`` or ``"synonym"``.
        """
        if backbone is not self._serving[0]:
            index: dict[tuple[str, str], dict[int, set[str]]] = {}
            for release, assembly in sorted(backbone.items()):
                for _gene, name, value, the_kind, form in self.links(assembly, release):
                    database = name if the_kind == "id" else f"{DB.synonym_id_nodes_prefix}{name}"
                    if self.selected(form, database, assembly, release):
                        index.setdefault((the_kind, value.lower()), {}).setdefault(release, set()).add(value)
            self._serving = (backbone, index)
        return self._serving[1].get((kind, text.lower()), {})

    def only_beside(self, text: str, release: int) -> bool:
        """Tell whether a string is an identifier at a release only on an assembly that does not serve the release.

        Human GRCh37 is still listed at the releases that GRCh38 serves, with its identifiers of release 75. ``True``
        when such an assembly writes the string (letter case ignored) as an identifier at the release and the
        serving assembly does not.
        """
        if not self.serving or release not in self.serving:
            return False
        lowered = text.lower()
        for assembly in sorted({a for a, _r, name in self.tables.tables if name == "xref"}):
            if assembly == self.serving[release] or release not in self.releases(assembly):
                continue
            if (release, assembly) not in self._beside:
                self._beside[(release, assembly)] = {
                    value.lower()
                    for _gene, name, value, kind, form in self.links(assembly, release)
                    if kind == "id" and self.selected(form, name, assembly, release)
                }
            if lowered in self._beside[(release, assembly)]:
                on_serving = self.written(release).get(("id", fold(text)), ())
                return not any(spelling.lower() == lowered for spelling, _database in on_serving)
        return False

    def meaning(self, text: str, release: int) -> tuple[str | None, bool, set[str] | None]:
        """Say how a typed string is to be matched at a release, from Ensembl's tables and the rules of matching.

        The rules: between the identifiers a string can mean, one that exists at the release is taken before one
        that does not; among those, exact before letter case before separators (``-``, ``_``, ``.``), each on
        names and then on synonyms.

        Returns:
            tuple: The expected ``match`` (``None`` where the tables alone do not decide it: the graph keeps one
            spelling per identifier, so an identifier that Ensembl wrote in several letter cases over the releases
            matches exactly in one of them only), whether a meaning of the string exists at the release, and the
            databases that write that meaning at the release (``None`` with an undecided match).
        """
        anywhere, here = self.written(None), self.written(release)
        folded = fold(text)
        steps = []
        for step in ("exact", "case", "separator"):
            for kind, prefix in (("id", ""), ("synonym", "synonym:")):
                spellings = {spelling for spelling, _database in anywhere.get((kind, folded), ())}
                if step == "exact":
                    candidates = {spelling for spelling in spellings if spelling == text}
                    decided = len({spelling for spelling in spellings if spelling.lower() == text.lower()}) == 1
                elif step == "case":
                    candidates = {spelling for spelling in spellings if spelling.lower() == text.lower()}
                    decided = len(candidates) == 1
                else:
                    candidates = spellings if any(separator in text for separator in "-_.") else set()
                    decided = len({spelling.lower() for spelling in candidates}) == 1
                if candidates:
                    lowered = {spelling.lower() for spelling in candidates}
                    at_release = {
                        database for spelling, database in here.get((kind, folded), ()) if spelling.lower() in lowered
                    }
                    steps.append((f"{prefix}{step}", decided, at_release))
        exists = any(databases for _match, _decided, databases in steps)
        if not all(decided for _match, decided, _databases in steps):
            return None, exists, None
        for match, _decided, databases in steps:
            if databases:
                return match, True, databases
        return (steps[0][0] if steps else None), False, set()

    def ids_of(self, gene: str, assembly: int, release: int, database: str) -> set[str]:
        """Return the identifiers that a database gives a gene at a release."""
        return {
            value
            for owner, name, value, kind, _form in self.links(assembly, release)
            if owner == gene and name == database and kind == "id"
        }


# --------------------------------------------------------------------------- the expected answers


def without_version(identifier: str) -> str:
    """Return an Ensembl id without its version (``ENSG00000141510.18`` -> ``ENSG00000141510``)."""
    head, dot, tail = identifier.rpartition(".")
    return head if dot and tail.isdigit() else identifier


def answer_of(result: dict[str, Any], final_database: str | None) -> list[str]:
    """Return the ids that a result of ``API.convert_identifier`` gives as its answer in the database asked for.

    Nothing when the identifier was not found or has no counterpart, and, for a conversion into an external
    database, nothing when the genes reached have no id in it (the Ensembl ids returned then are not an answer in
    that database).
    """
    if result["no_corresponding"] or result["no_conversion"]:
        return []
    if final_database and (result["no_target"] or result["final_database"] != final_database):
        return []
    return list(result["target_id"])


def verdict(query: dict[str, Any], result: dict[str, Any]) -> str | None:
    """Compare a result with the expected answer of a query; return ``None`` when right, else what is wrong.

    The rules are those of the truth the queries come from. ``expect`` is the set of right targets. With
    ``strategy="all"``, or with the rule ``set_exact`` (the right answer is several genes), the answer must be
    exactly that set; otherwise (``best`` with the rule ``set``) it must be a non-empty part of it, or nothing when
    nothing is expected. ``cmp`` says how ids are compared: Ensembl gene ids with their version (``versioned``),
    without it (``base``: the release has no versions in Ensembl's files), external ids as written (``exact``). The
    genes of ``apart`` (copies on patches and alternative loci) must be reported in ``alternative_locus_genes``.
    """
    got = answer_of(result, query["final_database"])
    if len(set(got)) != len(got):
        return f"a target is returned twice: {got}"
    reported = set(got)
    apart = set(result["alternative_locus_genes"])
    if query["cmp"] == "base":
        reported = {without_version(i) for i in reported}
        apart = {without_version(i) for i in apart}
    expected = set(query["expect"])
    if query["strategy"] == "all" or query["rule"] == "set_exact":
        if reported != expected:
            return f"expected exactly {sorted(expected)}, got {sorted(reported)}"
    elif expected and not (reported and reported <= expected):
        return f"expected one or more of {sorted(expected)}, got {sorted(reported)}"
    elif not expected and reported:
        return f"expected nothing, got {sorted(reported)}"
    lacking = set(query.get("apart") or ()) - apart
    if lacking:
        return f"copies not reported apart: {sorted(lacking)} (reported: {sorted(apart)})"
    return None


# --------------------------------------------------------------------------- the graph of a session

ORGANISMS = ("sus_scrofa", "homo_sapiens")
SNAPSHOT = 115


class RealWorld:
    """The graph of a real organism built from its cut tables, with what the tests compare it against.

    Attributes:
        organism: Ensembl species name.
        tables: The cut tables (:py:class:`RealTables`).
        ensembl: What Ensembl's tables say (:py:class:`Ensembl`).
        repository: The local repository the graph was built in; it holds the graph file.
        built: The :py:class:`idtrack.API` whose ``build_graph`` built the graph from the tables.
        api: A second :py:class:`idtrack.API`, whose ``build_graph`` loaded the saved file.
        backbone: ``{release: assembly}``: the assembly that serves each release of the graph (the organism's
            preferred assembly among those that have the release), read from Ensembl's index.
    """

    def __init__(self, organism: str, repository: str | os.PathLike):
        """Build the graph through the public API with every socket refused, then load it again from its file."""
        import socket

        import pytest

        import idtrack

        def refuse(*args, **kwargs):
            raise AssertionError("A build from the cut tables, or the load of its graph, opened a socket.")

        self.organism = organism
        self.repository = str(repository)
        self.tables = RealTables.load(organism)
        self.ensembl = Ensembl(self.tables)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
            monkeypatch.setattr(socket, "socket", refuse)
            self.tables.install(monkeypatch, self.repository)
            self.built = idtrack.API(local_repository=self.repository)
            self.built.build_graph(organism, SNAPSHOT)
            self.tables_served = self.tables.served

            def no_manager(*args, **kwargs):
                raise AssertionError("Loading a graph file needed a DatabaseManager.")

            monkeypatch.setattr(DatabaseManager, "__init__", no_manager)
            self.api = idtrack.API(local_repository=self.repository)
            self.api.build_graph(organism, SNAPSHOT)
        releases_of = {int(a): set(payload["releases"]) for a, payload in self.tables.indices.items()}
        self.backbone = {}
        for release in sorted(set().union(*releases_of.values())):
            if release <= SNAPSHOT:
                self.backbone[release] = next(
                    int(a) for a in DB.assemblies_by_priority(organism) if release in releases_of.get(int(a), ())
                )
        self.ensembl.serving = self.backbone

    def genes_at(self, release: int) -> dict[str, int]:
        """Return ``{gene id: version}`` of the assembly that serves a release."""
        return self.ensembl.genes(self.backbone[release], release)

    def releases_of_gene(self, identifier: str) -> list[int]:
        """Return the releases at which Ensembl has a gene id (with its version, when written with one)."""
        base = without_version(identifier)
        version = identifier[len(base) + 1 :]
        return [
            release
            for release in sorted(self.backbone)
            if base in self.genes_at(release) and (not version or self.genes_at(release)[base] == int(version))
        ]
