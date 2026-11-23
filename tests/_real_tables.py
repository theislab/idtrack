"""Real Ensembl tables, cut down to a few hundred genes, served to ``DatabaseManager`` in place of Ensembl.

The files ``tests/data/real/<organism>.tables.json.xz`` hold, for every genome assembly and release of an organism,
the raw tables of Ensembl's core database that a graph build reads (``gene``, ``transcript``, ``translation``,
``xref``, ``object_xref``, ``external_synonym``, ``identity_xref``, ``external_db``, ``seq_region``,
``coord_system``, ``seq_region_attrib``, ``attrib_type``, ``stable_id_event``, ``mapping_session``), restricted to the
genes of
``tests/data/real/<organism>.genes.txt``. They are written by ``tools/cut_tables.py`` from the
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
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from idtrack import ids, organisms, settings
from idtrack.build.database_manager import DatabaseManager
from idtrack.errors import EnsemblSchemaError

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
                run ``tools/cut_tables.py``).
        """
        key = (int(assembly), int(release), table_key)
        entry = self.tables.get(key)
        if entry is None:
            raise AssertionError(
                f"The cut tables of {self.organism} have no table {table_key!r} of assembly {assembly}, release "
                f"{release}: rebuild them with tools/cut_tables.py."
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
                    f"{assembly}, release {release}): rebuild them with tools/cut_tables.py."
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
        monkeypatch.setattr(settings.ENSEMBL, "access", "https")
        monkeypatch.setattr(DatabaseManager, "download_table", _replayed_download)
        monkeypatch.setattr(DatabaseManager, "assembly_chain", _replayed_chain)
        # the tests' default (tests/conftest.py) reads no filing; the cut tables hold every table it reads
        monkeypatch.setattr(DatabaseManager, "database_filing", REAL_METHODS["database_filing"])

    def chain(self, earlier: int, later: int) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        """Return the cut blocks of Ensembl's mapping of one assembly onto another, as ``assembly_chain`` returns them."""
        blocks = self.tables.get((int(earlier), int(later), CHAIN_TABLE))
        if blocks is None:
            return None
        columns = {name: values for (name, _kind), values in zip(blocks["columns"], blocks["data"])}
        frame = pd.DataFrame({name: columns[name] for name in CHAIN_COLUMNS})
        info = self.tables[(int(earlier), int(later), CHAIN_INFO_TABLE)]
        about = {name: values[0] for (name, _kind), values in zip(info["columns"], info["data"])}
        return frame.astype({c: "int64" for c in ("start", "end", "onto_start", "strand")}), {
            "file": str(about["file"]),
            "from": int(earlier),
            "to": int(later),
            "release": int(about["release"]),
            "sha256": str(about["sha256"]),
        }


#: The cut blocks of Ensembl's mapping of an assembly onto a later one are kept under ``(earlier, later, CHAIN_TABLE)``
#: (the later assembly in the place of a release), with the file's name, release and digest under
#: ``CHAIN_INFO_TABLE``.
CHAIN_TABLE = "assembly_chain"
CHAIN_INFO_TABLE = "assembly_chain_file"
CHAIN_COLUMNS = ["region", "start", "end", "onto", "onto_start", "strand"]
#: The package's own methods that ``tests/conftest.py`` replaces for the synthetic tables of the unit tests, as defined.
REAL_METHODS = {"database_filing": DatabaseManager.database_filing, "assembly_chain": DatabaseManager.assembly_chain}

#: The repositories whose managers read cut tables: ``{real path: RealTables}`` (several organisms at a time).
_SERVED: dict[str, RealTables] = {}


def _replayed_chain(manager, earlier, later):
    """Answer ``DatabaseManager.assembly_chain`` from the cut tables installed for the manager's repository."""
    tables = _SERVED.get(os.path.realpath(str(manager.local_repository)))
    if tables is None or manager.organism != tables.organism:
        raise AssertionError(f"Unexpected assembly mapping of {manager.organism} outside the cut tables.")
    return tables.chain(earlier, later)


def _replayed_download(manager, table_key, usecols=None):
    """Answer ``DatabaseManager.download_table`` from the cut tables installed for the manager's repository."""
    tables = _SERVED.get(os.path.realpath(str(manager.local_repository)))
    if tables is None or manager.organism != tables.organism:
        raise AssertionError(f"Unexpected download of {table_key!r} by {manager.organism} outside the cut tables.")
    return tables.table(manager.assembly, manager.release, table_key, usecols or None)


def units_names(units: set[tuple[str, str]]) -> set[str]:
    """Return the names of a set of ``(form, name)`` pairs."""
    return {name for _form, name in units}


def _form_of(identifier: str) -> str:
    """Return an identifier's form: its characters before the first digit, ``#`` for a number rest, ``*`` for another."""
    import re

    head = re.match(r"^[^0-9]*", identifier).group(0)  # type: ignore[union-attr]
    rest = identifier[len(head) :]
    return head + ("#" if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", rest) else "*" if rest else "")


def read_as_typed(text: str) -> str:
    """Return an identifier of the tables as decisions 83 and 84 read every identifier, written apart from the package.

    Format and control characters (Unicode categories Cf, Cc) go, NFKC applies, dashes (category Pd) and the minus sign
    read as ``-``; then, repeatedly, the punctuation that ends a sentence (``;`` ``,`` ``.`` ``:`` ``!`` ``?``) goes from
    the end and a pair of quotation marks (ASCII ``'`` ``"``, categories Pi and Pf) around the whole; a string that
    would be left empty stays as it is.
    """
    if not (text.isascii() and text.isprintable()):
        visible = "".join(c for c in text if c.isspace() or unicodedata.category(c) not in ("Cf", "Cc"))
        normal = unicodedata.normalize("NFKC", visible)
        text = "".join("-" if c == "\u2212" or unicodedata.category(c) == "Pd" else c for c in normal).strip()

    def quote(character: str) -> bool:
        return character in "'\"" or unicodedata.category(character) in ("Pi", "Pf")

    current = text
    while True:
        before = current
        current = current.rstrip("!,.:;?").rstrip()
        if len(current) > 2 and quote(current[0]) and quote(current[-1]):
            current = current[1:-1].strip()
        if current == before:
            return current or text


def fold(identifier: str) -> str:
    """Return an identifier in lower case with ``-``, ``_`` and ``.`` taken as one character."""
    return identifier.lower().replace("_", "-").replace(".", "-")


class Ensembl:
    """What Ensembl's tables say, read from the cut tables by code that shares nothing with the package.

    The tests take their expectations about a release from here (which genes exist with which version, which gene
    an external identifier belongs to in which database, which genes lie outside the reference sequence), so that
    a test never asserts what the package itself computed.

    A database is named as the graph names it: an accession of Ensembl's ``external_db.db_name`` (``EntrezGene``,
    ``Uniprot/SWISSPROT``), or a label of its ``db_display_name`` (``HGNC Symbol``); where Ensembl gives labels and
    accessions one name, the labels of a database take one name at every release: its display name at the newest
    release at which that name is not the name of accessions, else ``<db_name> label``; the synonyms of a cross
    reference are reported apart, under the name of its labels. What is not an identifier is left out: a value without
    a letter or a digit, a synonym that is UniProt's evidence text, a label equal to its accession where Ensembl names
    the labels like the accessions (one row with it), a miRBase entry of a
    gene that is not a miRNA or that names genes at several loci over the releases (:py:meth:`one_locus_count`), and an
    RFAM label that is not its gene's display name (the name Ensembl gives the genes of a family that have no better
    one).
    """

    def __init__(self, tables: RealTables):
        """Read from a set of cut tables."""
        self.tables = tables
        self._links: dict[tuple[int, int], list[tuple[str, str, str, str, str]]] = {}
        self._raw_links: dict[tuple[int, int], list[tuple[str, str, str, str, str]]] = {}
        self._unfiled_rows: dict[tuple[int, int], tuple[list, list, set]] = {}
        self._filing: tuple | None = None
        self._databases_of: dict[str, set[str]] = {}
        self._value_of: dict[tuple[str, str], str] = {}
        self._runs: dict[str, set[tuple[int, int]]] = {}
        self._filed_keys: set[tuple[str, str, int, int]] = set()  # (form, database, assembly, release) filed so
        self._display: dict[tuple[int, int], list[tuple[str, str, str]]] = {}
        self._display_dbs: dict[tuple[int, int], set[str]] = {}
        self._reserved: set[str] | None = None
        self._read_changed: dict[tuple[int, int], set[tuple[str, str, str, str, str]]] = {}
        self._refiled: dict[tuple[int, int], set[tuple[str, str, str, str, str]]] = {}
        self._label_names: dict[str, str] | None = None
        self._mirbase: set[str] | None = None
        self._loci: (
            tuple[
                dict[str, dict[int, set[tuple[int, str, int, int, int, str]]]],
                dict[str, set[int]],
                set[tuple[str, str]],
            ]
            | None
        ) = None
        self._genes: dict[tuple[int, int], dict[str, int]] = {}
        self._selection: dict[tuple[str, str, int], set[int]] | None = None
        self._written: dict[int | None, dict[tuple[str, str], set[tuple[str, str]]]] = {}
        self._serving: tuple[Any, dict[tuple[str, str], dict[int, set[str]]]] = (None, {})
        #: ``{release: assembly}``: the assembly that serves each release (set by :py:class:`RealWorld`).
        self.serving: dict[int, int] = {}
        self._beside: dict[tuple[int, int], set[str]] = {}
        self._versions: dict[str, dict[int, dict[int, set[str]]]] | None = None
        self._link_versions: dict[tuple[str, str, str | None], dict[int, set[int]]] = {}

    def selected(self, form: str, database: str, assembly: int, release: int) -> bool:
        """Tell whether the shipped externals selection includes a database for a form, assembly and release.

        The selection is the package's configuration file ``<organism>_externals_modified.yml``, read here as
        plain YAML. A synonym's database is named with the prefix ``synonym_id::``. A gene's display name is held
        whatever the selection (I129): the database of a display name that no selected database gives its gene is
        selected for the genes at that release (:py:meth:`links` gives that database's rows there as display names
        only).
        """
        if form == "gene" and database in self._display_databases(assembly, release):
            return True
        return self._selected_by_yaml(form, database, assembly, release)

    def _selected_by_yaml(self, form: str, database: str, assembly: int, release: int) -> bool:
        """Tell whether the shipped externals YAML includes a database for a form, assembly and release.

        Or whether the rows are filed under a database it includes (:py:meth:`filed_as`, decision 104): a row the graph
        holds under a selected database is reported under that database, which this then names.
        """
        if self._selection is None:
            import yaml

            import idtrack

            path = Path(idtrack.__file__).parent / "default_config" / f"{self.tables.organism}_externals_modified.yml"
            with open(path, encoding="utf-8") as handle:
                content = yaml.safe_load(handle)[self.tables.organism]
            self._selection = {}  # {(form, name, assembly): releases} of the entries the YAML enables
            for the_form, per_form in content.items():
                for name, entry in per_form.items():
                    for key, per_assembly in entry["Assembly"].items():
                        if per_assembly["Include"]:
                            releases = {int(r) for r in str(per_assembly["Ensembl release"]).split(",")}
                            self._selection[(the_form, name, int(key))] = releases
        if int(release) in self._selection.get((form, database, int(assembly)), ()):
            return True
        # decision 104: a row filed under a selected database by the continuity of its links (:py:meth:`filed_as`) is
        # that database's wherever it lies
        return (form, database, int(assembly), int(release)) in self._filed_keys

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

    def regions(self, assembly: int, release: int) -> dict[str, tuple[str, int, int]]:
        """Return ``{gene id: (sequence region, first base, last base)}`` of an assembly at a release."""
        genes = self._ids(assembly, release, "gene").merge(
            self._frame(assembly, release, "seq_region"), on="seq_region_id"
        )
        return {
            i: (str(n), int(a), int(b))
            for i, n, a, b in zip(genes["stable_id"], genes["name"], genes["seq_region_start"], genes["seq_region_end"])
        }

    def y_twins(self, assembly: int, release: int) -> dict[str, set[str]]:
        """Return the genes in a pseudoautosomal region of Y with the genes of X at their place (decision 95's copies).

        A gene lies in such a region when its place lies inside it; its place read on the region of X the region repeats
        (the same offset from the region's first base) is at the place of an X gene when the two share a base on one
        strand. The regions are the package's (:py:data:`idtrack.organisms.PSEUDOAUTOSOMAL_REGIONS`, by the assembly's
        name); the places are the tables'.
        """
        from idtrack import organisms

        name = organisms.assembly_name(self.tables.organism, int(assembly)) or ""
        places = self.places(assembly, release)
        twins: dict[str, set[str]] = {}
        for (y_region, y_first, y_last), (x_region, x_first, _x_last) in organisms.PSEUDOAUTOSOMAL_REGIONS.get(
            name, ()
        ):
            for gene, (region, start, end, strand) in places.items():
                if region == y_region and y_first <= start and end <= y_last:
                    low, high = start - y_first + x_first, end - y_first + x_first
                    twins[gene] = {
                        other
                        for other, (r, a, b, s) in places.items()
                        if r == x_region and s == strand and a <= high and low <= b
                    }
        return twins

    def _links_of_release(self, assembly: int, release: int) -> list[tuple[str, str, str, str, str]]:
        """Return ``(gene id, database, external id, "id" | "synonym", form)`` for every cross reference of a release.

        A cross reference of a transcript or translation is reported for its gene; ``form`` says what it is
        attached to in Ensembl (``gene``, ``transcript`` or ``translation``). The synonyms of a cross reference
        are reported under the database of its label. A row is reported under the selected database it is
        (decision 104, :py:meth:`filed_as`).
        """
        where = (int(assembly), int(release))
        if where not in self._raw_links:
            rows, shown, changed = self._unfiled(assembly, release)
            filed = {self._file(row, release) for row in rows}
            refiled = self._refiled[where] = set()
            for (_g, before, _v, kind, form), row in zip(rows, (self._file(r, release) for r in rows)):
                after = row[1]
                if "@" in after:
                    continue  # filed under no database of the graph
                if after != before or self._continued_on_form(before, form, release):
                    refiled.add(row)
                    self._filed_keys.add(
                        (form, ("synonym_id::" if kind == "synonym" else "") + after, int(assembly), int(release))
                    )
            self._raw_links[where] = sorted(filed)
            self._display[where] = sorted(
                {(gene, self._shown_as(gene, database, value, release), value) for gene, database, value in shown}
            )
            self._read_changed[where] = {self._file(row, release) for row in changed}
        return self._raw_links[where]

    def _continued_on_form(self, database: str, form: str, release: int) -> bool:
        """Tell whether a selected name's rows on a form the YAML does not hold it on continue it at a release."""
        name = database.removeprefix("synonym_id::")
        if (
            not self.serving
            or (form, name) in self._selected_units()
            or name not in units_names(self._selected_units())
        ):
            return False
        renamed, _schemes, _held = self.filing()
        unit = f"{name}@{form}"
        return unit in renamed and any(a <= release <= b for a, b in self._runs.get(unit, ()))

    def _cross_references(self, assembly: int, release: int) -> list[tuple[str, str, str, str]]:
        """Return ``(gene, name, identifier, form)`` of every cross reference of a release as Ensembl's tables write it.

        The links decision 104 reads: every cross reference's accession under its database's name and its label under
        the name of the database's labels, the identifier as written (but for surrounding white space), on every form
        (a transcript's or a translation's for its gene); no synonym, nothing left out.
        """
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
        databases = self._frame(assembly, release, "external_db")
        name_of = dict(zip(pd.to_numeric(databases["external_db_id"]).astype(int), databases["db_name"].astype(str)))
        xref = self._frame(assembly, release, "xref")
        entry = {
            int(k): (int(d), a, b)
            for k, d, a, b in zip(xref["xref_id"], xref["external_db_id"], xref["dbprimary_acc"], xref["display_label"])
        }
        label_names = self.label_names()
        objects = self._frame(assembly, release, "object_xref")
        found = []
        for kind, ensembl_id, xref_id in zip(objects["ensembl_object_type"], objects["ensembl_id"], objects["xref_id"]):
            form = str(kind).lower()
            gene_id = owner.get(form, {}).get(int(ensembl_id))
            record = entry.get(int(xref_id))
            if gene_id is None or record is None or record[0] not in name_of:
                continue
            database = name_of[record[0]]
            for name, value in ((database, record[1]), (label_names[database], record[2])):
                text = "" if value is None or value != value else str(value).strip()
                if text:
                    found.append((gene_id, name, text, form))
        return found

    def _shown_as(self, gene: str, database: str, value: str, release: int) -> str:
        """Return the database a display name is filed under: as any row, but never none (a name of its gene)."""
        filed = self._file((gene, database, value, "id", "gene"), release)[1]
        return database if "@" in filed else filed

    def _file(self, row: tuple[str, str, str, str, str], release: int) -> tuple[str, str, str, str, str]:
        """Return a row under the database it is filed under (:py:meth:`filed_as`)."""
        gene, database, value, kind, form = row
        return (gene, self.filed_as(database, value, gene, int(release), kind, form), value, kind, form)

    def _unfiled(self, assembly: int, release: int) -> tuple[list, list, set]:
        """Return the rows of a release under Ensembl's names, the display names, and the rows read otherwise."""
        where = (int(assembly), int(release))
        if where in self._unfiled_rows:
            return self._unfiled_rows[where]
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
        genes = self._ids(assembly, release, "gene")
        biotype = dict(zip(genes["stable_id"], genes["biotype"])) if "biotype" in genes.columns else {}
        display = (
            {
                stable_id: int(xref)
                for stable_id, xref in zip(
                    genes["stable_id"], pd.to_numeric(genes.get("display_xref_id"), errors="coerce")
                )
                if xref == xref  # not NaN
            }
            if "display_xref_id" in genes.columns
            else {}
        )

        label_names = self.label_names()

        def identifier(value: Any) -> str | None:
            text = read_as_typed(str(value).strip()) if value is not None else ""
            return text if any(c.isascii() and c.isalnum() for c in text) else None

        xref_rows, xref_columns = plain("xref", "xref_id")
        at = [xref_columns.index(column) for column in ("external_db_id", "dbprimary_acc", "display_label")]
        synonyms: dict[int, list[str]] = {}
        entry = self.tables.tables[(int(assembly), int(release), "external_synonym")]
        columns = {column: values for (column, _kind), values in zip(entry["columns"], entry["data"])}
        for xref_id, synonym in zip(columns.get("xref_id", ()), columns.get("synonym", ())):
            synonyms.setdefault(int(xref_id), []).append(str(synonym).strip())
        rows = []
        changed: set[tuple[str, str, str, str, str]] = set()
        entry = self.tables.tables[(int(assembly), int(release), "object_xref")]
        columns = {column: values for (column, _kind), values in zip(entry["columns"], entry["data"])}
        for kind, ensembl_id, xref_id in zip(columns["ensembl_object_type"], columns["ensembl_id"], columns["xref_id"]):
            form = str(kind).lower()
            gene_id = owner.get(form, {}).get(int(ensembl_id))
            if gene_id is None:
                continue
            record = xref_rows[int(xref_id)]
            database = database_rows[int(record[at[0]])]
            name = str(database[names_at[0]])
            label_name = label_names[name]
            if name.removesuffix("_gene_name") == "miRBase" and biotype.get(gene_id) != "miRNA":
                continue  # a miRBase name is that of the miRNA gene, not of the gene that hosts it
            accession, label = identifier(record[at[1]]), identifier(record[at[2]])
            if accession is not None:
                rows.append((gene_id, name, accession, "id", form))
                if accession != str(record[at[1]]).strip():
                    changed.add(rows[-1])
            if name == "RFAM" and display.get(gene_id) != int(xref_id):
                continue  # a family's name names only the genes it is the display name of
            # a label equal to its accession was one row with it only where Ensembl named both alike at the release
            if label is not None and not (label == accession and database[names_at[1]] == name):
                rows.append((gene_id, str(label_name), label, "id", form))
                if label != str(record[at[2]]).strip():
                    changed.add(rows[-1])
            if label_name is not None:
                # Ensembl lists synonyms per cross reference; they belong to the label's database
                for synonym in synonyms.get(int(xref_id), ()):
                    read = identifier(synonym)
                    if read is not None and "{ECO:" not in synonym:
                        rows.append((gene_id, str(label_name), read, "synonym", form))
                        if read != synonym.strip():
                            changed.add(rows[-1])
        shown = []
        for gene_id, xref_id in display.items():
            shown_record = xref_rows.get(int(xref_id))
            if shown_record is None:
                continue
            shown_database = database_rows.get(int(shown_record[at[0]]))
            shown_label = identifier(shown_record[at[2]])
            if shown_database is not None and str(shown_database[names_at[0]]).removesuffix("_gene_name") == "miRBase":
                if biotype.get(gene_id) != "miRNA":
                    continue  # the miRNA's name shown on its host gene is not the host's name
            if shown_database is not None and shown_label is not None:
                shown.append((gene_id, str(label_names[str(shown_database[names_at[0]])]), shown_label))
        self._unfiled_rows[where] = (sorted(set(rows)), sorted(set(shown)), changed)
        return self._unfiled_rows[where]

    # ------------------------------------------------------------------ decision 104, read here on its own

    def _selected_units(self) -> set[tuple[str, str]]:
        """Return the ``(form, name)`` the externals YAML enables anywhere, a synonym's by its entry's name."""
        if self._selection is None:
            self._selected_by_yaml("gene", "", 0, 0)
        assert self._selection is not None
        return {(form, name.removeprefix("synonym_id::")) for form, name, _assembly in self._selection}

    def filing(
        self,
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[tuple[str, str], tuple[int, int]]]]:
        """Return which of Ensembl's names each selected database holds (decision 104), read from the cut tables.

        The selected databases are the ``(form, name)`` pairs the YAML enables (every selected form of a name
        together). The candidates: a name none of whose Ensembl databases has a selected name, and a selected name on a
        form the YAML does not hold it on (``name@form``). Between two releases next to each other (on the assembly that
        serves each), a link (identifier as written, gene) moves from one to the other where the one holds it
        at the first release and not at the second and the other at the second and not at the first. A candidate
        continues a selected database where the links (identifiers as written) that move between them are at least
        ``CONTINUITY_SHARE`` of the links of the one that leaves (the older side), or, for a candidate that holds no link
        at the next release, of the selected database's links there; its identifiers go by their form
        (the characters before the first digit, then ``#`` for a number with an optional version, ``*`` for any other
        rest) to the selected database most of that form's moved links went with where the form is that database's
        scheme, other forms to the one most of all went with. An
        identifier of a selected database whose form is the form of at least 99 % of another selected database's
        identifiers (of another Ensembl database), on a link that database holds at some release, is that database's
        (the one holding it nearest the release).

        Returns:
            tuple: ``{candidate: {form: selected}}`` (``"*"``: any other form), ``{selected: its scheme}``, and per scheme
            database ``{(identifier, gene): (first release, last release)}``.
        """
        if self._filing is not None:
            return self._filing
        from idtrack.build.continuity import CONTINUITY_SHARE

        units = self._selected_units()
        names = {name for _form, name in units}
        label_names = self.label_names()
        databases_of: dict[str, set[str]] = {}
        for database, label in label_names.items():
            databases_of.setdefault(database, set()).add(database)
            databases_of.setdefault(label, set()).add(database)
        taken = {d for name in names for d in databases_of.get(name, {name})}

        def unit_of(form: str, name: str) -> str | None:
            if (form, name) in units:
                return "selected::" + name
            if name in names:
                return f"{name}@{form}"
            return None if databases_of.get(name, {name}) & taken else name

        def links_at(release: int) -> dict[str, set[tuple[str, str]]]:
            found: dict[str, set[tuple[str, str]]] = {}
            for gene, name, value, form in self._cross_references(self.serving[release], release):
                unit = unit_of(form, name)
                if unit is not None:
                    found.setdefault(unit, set()).add((value, gene))
                    self._value_of[(value, unit)] = value
            return found

        moved: dict[tuple[str, str, int, str, int], dict[str, int]] = {}
        sizes: dict[tuple[str, int], int] = {}
        forms_of: dict[str, dict[str, set[str]]] = {}
        spans: dict[str, dict[tuple[str, str], list[int]]] = {}
        previous: dict[str, set[tuple[str, str]]] | None = None
        previous_release = 0
        for release in sorted(self.serving):
            current = links_at(release)
            for unit, links in current.items():
                sizes[(unit, release)] = len(links)
                if unit.startswith("selected::"):
                    name = unit.removeprefix("selected::")
                    per_form = forms_of.setdefault(name, {})
                    for link in links:
                        per_form.setdefault(_form_of(self._value_of[(link[0], unit)]), set()).add(link[0])
                        span = spans.setdefault(name, {}).setdefault(link, [release, release])
                        span[1] = release
            if previous is not None:
                chosen = [u for u in set(previous) | set(current) if u.startswith("selected::")]
                others = [u for u in set(previous) | set(current) if not u.startswith("selected::")]
                for candidate in others:
                    for target in chosen:
                        name = target.removeprefix("selected::")
                        if "@" in candidate and candidate.partition("@")[0] != name:
                            continue  # a selected name on another form can be that database only
                        # the candidate leaves for the selected database, or the selected database for the candidate
                        for leaving, arriving, base, there in (
                            (candidate, target, candidate, previous_release),
                            (target, candidate, target, release),
                        ):
                            gone = (previous.get(leaving, set()) - current.get(leaving, set())) & current.get(
                                arriving, set()
                            )
                            gone -= previous.get(arriving, set())
                            for link in gone:
                                form = _form_of(self._value_of[(link[0], candidate)])
                                bucket = moved.setdefault((candidate, form, previous_release, base, there), {})
                                bucket[name] = bucket.get(name, 0) + 1
            previous, previous_release = current, release
        order = sorted(self.serving)
        held_at = {unit: {r for (u, r) in sizes if u == unit} for unit, _r in sizes}

        def run_of(unit: str, release: int) -> tuple[int, int]:
            i = j = order.index(release)
            while i > 0 and order[i - 1] in held_at[unit]:
                i -= 1
            while j + 1 < len(order) and order[j + 1] in held_at[unit]:
                j += 1
            return order[i], order[j]

        continued: dict[str, dict[str, int]] = {}
        totals: dict[tuple[str, int, str, str, int], int] = {}
        for (candidate, _form, at, base, there), targets in moved.items():
            for target, count in targets.items():
                key = (candidate, at, base, target, there)
                totals[key] = totals.get(key, 0) + count
        deciding: set[tuple[str, int, str]] = set()
        self._runs = {}
        for (candidate, at, base, target, there), count in totals.items():
            after = order[order.index(at) + 1]
            # decision 107: a candidate that holds no link at the next release hands its links over to the selected
            # database whose links at that release they are at least the share of
            ending = base == candidate and after not in held_at.get(candidate, set())
            if ending and count >= CONTINUITY_SHARE * sizes.get(("selected::" + target, after), 0) > 0:
                pass
            elif count < CONTINUITY_SHARE * sizes[(base, at)]:
                continue
            elif base != candidate and count < CONTINUITY_SHARE * sizes.get((candidate, there), 0):
                continue  # a later name is, at its first release, mostly the selected database's links
            continued.setdefault(candidate, {})
            continued[candidate][target] = continued[candidate].get(target, 0) + count
            deciding.add((candidate, at, base))
            self._runs.setdefault(candidate, set()).add(run_of(candidate, there))
        by_form: dict[tuple[str, str], dict[str, int]] = {}
        for (candidate, form, at, base, _there), targets in moved.items():
            if (candidate, at, base) in deciding:
                for target, count in targets.items():
                    by_form.setdefault((candidate, form), {})
                    by_form[(candidate, form)][target] = by_form[(candidate, form)].get(target, 0) + count
        schemes = {}
        for name, per_form in forms_of.items():
            total = sum(len(v) for v in per_form.values())
            form, ids_of_form = max(sorted(per_form.items()), key=lambda item: len(item[1]))
            if len(ids_of_form) >= 0.99 * total:
                schemes[name] = form
        renamed: dict[str, dict[str, str]] = {}
        for candidate, into in continued.items():
            renamed[candidate] = {"*": max(sorted(into), key=lambda t: into[t])}
            for (other, form), targets in by_form.items():
                if other == candidate:
                    target = max(sorted(targets), key=lambda t: targets[t])
                    if schemes.get(target) == form:  # another database's identifier scheme only
                        renamed[candidate][form] = target
        held = {name: {link: (span[0], span[1]) for link, span in spans[name].items()} for name in schemes}
        self._filing = (renamed, schemes, held)
        self._databases_of = databases_of
        return self._filing

    def filed_as(
        self, database: str, value: str, gene: str, release: int, kind: str = "id", eform: str = "gene"
    ) -> str:
        """Return the database a row of Ensembl's name ``database``, attached to ``eform``, is filed under (:py:meth:`filing`)."""
        if not self.serving:
            return database
        renamed, schemes, held = self.filing()
        prefix = "synonym_id::" if database.startswith("synonym_id::") else ""
        name = database.removeprefix("synonym_id::")
        form = _form_of(value)
        units = self._selected_units()
        filed = (eform, name) in units  # a row the selection holds (rule 3 judges those only)
        if name in {n for _f, n in units} and (eform, name) not in units:
            unit = f"{name}@{eform}"  # a selected name on a form the YAML does not hold it on
            in_run = any(a <= release <= b for a, b in self._runs.get(unit, ()))
            if unit not in renamed or not in_run:
                return prefix + unit  # continues nothing here: no database of the graph
        elif name in units_names(units) or name not in renamed:
            unit = ""
        else:
            unit = name
        if unit and unit in renamed:
            main = renamed[unit]["*"]
            target = renamed[unit].get(form, main if schemes.get(main) in (None, form) else None)
            if target is None:
                return prefix + unit + "@other"  # a form the continuation's identifiers never have: none
            name, filed = target, True
        if filed and kind == "id" and not prefix and schemes.get(name) != form:
            mine = self._databases_of.get(name, {name})
            found = []
            for other, scheme in schemes.items():
                if other == name or scheme != form or self._databases_of.get(other, {other}) & mine:
                    continue
                span = held[other].get((value, gene))
                if span is not None:
                    found.append(
                        (
                            0 if span[0] <= release <= span[1] else min(abs(release - span[0]), abs(release - span[1])),
                            other,
                        )
                    )
            if found:
                name = min(found)[1]
        return prefix + name

    def accession_versions(self) -> dict[str, dict[int, dict[int, set[str]]]]:
        """Return the versions Ensembl's cross references attach to the accessions, with their releases and genes.

        ``{accession: {version: {release: genes}}}``, read from ``xref.version`` (digits other than ``0``) of the
        accession rows of the selected databases (and of the databases filed under them, :py:meth:`filed_as`), at each
        release on the assembly that serves it (:py:attr:`serving`);
        a cross reference of a transcript or translation counts for its gene.
        """
        if self._versions is not None:
            return self._versions
        found: dict[str, dict[int, dict[int, set[str]]]] = {}
        for release, assembly in sorted(self.serving.items()):
            if (int(assembly), int(release), "xref") not in self.tables.tables:
                continue
            gene = self._ids(assembly, release, "gene")[["gene_id", "stable_id"]].rename(columns={"stable_id": "gene"})
            transcript = self._ids(assembly, release, "transcript")[["transcript_id", "gene_id"]].merge(
                gene, on="gene_id"
            )
            translation = self._ids(assembly, release, "translation")[["translation_id", "transcript_id"]].merge(
                transcript, on="transcript_id"
            )
            owner = {
                "gene": dict(zip(gene["gene_id"], gene["gene"])),
                "transcript": dict(zip(transcript["transcript_id"], transcript["gene"])),
                "translation": dict(zip(translation["translation_id"], translation["gene"])),
            }
            versioned = self.genes(assembly, release)
            # every Ensembl object as its node is named (<stable id>.<version>), for the links of decision 101
            node = {}
            for form in ("gene", "transcript", "translation"):
                table = self._ids(assembly, release, form)
                node[form] = dict(
                    zip(table[f"{form}_id"], [f"{i}.{v}" for i, v in zip(table["stable_id"], table["version"])])
                )
            xref = self._frame(assembly, release, "xref")
            names = self._frame(assembly, release, "external_db")
            name_of = dict(zip(names["external_db_id"].astype(int), names["db_name"].astype(str)))
            record = {
                int(x): (name_of.get(int(d)), str(a).strip(), str(v).strip())
                for x, d, a, v in zip(xref["xref_id"], xref["external_db_id"], xref["dbprimary_acc"], xref["version"])
            }
            links = self._frame(assembly, release, "object_xref")
            for kind, ensembl_id, xref_id in zip(links["ensembl_object_type"], links["ensembl_id"], links["xref_id"]):
                form = str(kind).lower()
                stable_id = owner.get(form, {}).get(int(ensembl_id))
                database, accession, version = record.get(int(xref_id), (None, "", "0"))
                if stable_id is None or database is None or not version.isdigit() or int(version) == 0:
                    continue
                # the database the row is filed under (decision 104: RefSeq_dna's NM_ accessions are RefSeq_mRNA's)
                database = self.filed_as(database, accession, stable_id, int(release), "id", form)
                if not (
                    self._selected_by_yaml(form, database, assembly, release)
                    or (form, database) in self._selected_units()
                ):
                    continue
                gene_version = f"{stable_id}.{versioned[stable_id]}" if stable_id in versioned else stable_id
                per = found.setdefault(accession, {}).setdefault(int(version), {})
                per.setdefault(int(release), set()).add(gene_version)
                link = (database, accession, node[form].get(int(ensembl_id)))
                self._link_versions.setdefault(link, {}).setdefault(int(release), set()).add(int(version))
        self._versions = found
        return found

    def edit_counter_databases(self, bound: float) -> dict[str, tuple[int, int]]:
        """Return the databases whose number of an accession is an edit counter, with the pairs counted (decision 101).

        A database's number differs in more than ``bound`` of the pairs of consecutive releases of one link (accession,
        Ensembl object with version, database); read from the tables: ``{database: (pairs, pairs in which the version
        differs)}`` of those above the bound.
        """
        self.accession_versions()
        counts: dict[str, list[int]] = {}
        for (database, _accession, _node), per in self._link_versions.items():
            count = counts.setdefault(database, [0, 0])
            for release in per:
                if release + 1 in per:
                    count[0] += 1
                    count[1] += per[release] != per[release + 1]
        return {database: (p, d) for database, (p, d) in counts.items() if p and d / p > bound}

    def display_names(self, assembly: int, release: int) -> list[tuple[str, str, str]]:
        """Return ``(gene id, database of its labels, display name)`` of every gene with a display name at a release.

        A gene's display name (``gene.display_xref_id``) read as every value is read, under the name of the labels of
        the database of its cross reference, whichever object the cross reference is attached to.
        """
        self._links_of_release(assembly, release)
        return self._display[(int(assembly), int(release))]

    def refiled(self, assembly: int, release: int) -> set[tuple[str, str, str, str, str]]:
        """Return the rows of :py:meth:`_links_of_release` that Ensembl wrote under another database or form.

        The rows of a database that continues a selected one (decision 104, :py:meth:`filed_as`): pig HGNC projections
        filed as ``HGNC_curated_gene`` up to release 60 are rows of ``HGNC Symbol``.
        """
        self._links_of_release(assembly, release)
        return self._refiled[(int(assembly), int(release))]

    def read_changed(self, assembly: int, release: int) -> set[tuple[str, str, str, str, str]]:
        """Return the rows of :py:meth:`_links_of_release` whose value the tables write otherwise (:py:func:`read_as_typed`).

        UniProt's ``A1BG;`` is the row of ``A1BG``: the rows a reading of decisions 83 and 84 made.
        """
        self._links_of_release(assembly, release)
        return self._read_changed[(int(assembly), int(release))]

    def label_names(self) -> dict[str, str]:
        """Return the one name of every database's labels.

        It is the database's display name at the newest release (of any assembly) at which that name is not the name
        of accessions, else ``<db_name> label``.
        """
        if self._label_names is None:
            newest: dict[str, tuple[int, str]] = {}
            names: set[str] = set()
            for assembly, release, name in self.tables.tables:
                if name != "external_db" or int(release) > self.tables.snapshot:
                    continue
                table = self._frame(assembly, release, "external_db")
                of_release = set(table["db_name"].astype(str))
                for db, display in zip(table["db_name"].astype(str), table["db_display_name"]):
                    names.add(db)
                    if display is not None and display == display and str(display) not in of_release:
                        if db not in newest or int(release) >= newest[db][0]:
                            newest[db] = (int(release), str(display))
            self._label_names = {db: newest[db][1] if db in newest else f"{db} label" for db in names}
        return self._label_names

    def _mirbase_names(self) -> set[str]:
        """Return the names of miRBase's accessions and labels, and of the databases of names derived from it.

        Ensembl files miRBase's names that it displays on genes at some releases under ``miRBase_gene_name``; they are
        miRBase's names (I146).
        """
        names = self.label_names()
        databases = {"miRBase"} | {name for name in names if name.removesuffix("_gene_name") == "miRBase"}
        return databases | {names.get(database, f"{database} label") for database in databases}

    def _mirbase_shared(self) -> set[str]:
        """Return the miRBase entries (any letter case) that name genes at several loci over all releases and assemblies.

        The loci are counted by :py:meth:`one_locus_count`, the oracle's own reading of the tables.
        """
        if self._mirbase is None:
            genes: dict[str, set[str]] = {}
            for assembly, release, name in self.tables.tables:
                if name == "xref" and int(release) <= self.tables.snapshot:
                    for gene, database, value, _kind, _form in self._links_of_release(assembly, release):
                        if database in self._mirbase_names():
                            genes.setdefault(value.lower(), set()).add(gene)
                    for gene, database, value in self.display_names(assembly, release):
                        if database in self._mirbase_names():
                            genes.setdefault(value.lower(), set()).add(gene)
            self._mirbase = {value for value, of in genes.items() if len(of) > 1 and self.one_locus_count(of) > 1}
        return self._mirbase

    def places(self, assembly: int, release: int) -> dict[str, tuple[str, int, int, int]]:
        """Return ``{gene id: (sequence region, start, end, strand)}`` of an assembly at a release (the gene table)."""
        table = self._ids(assembly, release, "gene")
        names = self._frame(assembly, release, "seq_region")
        name_of = dict(zip(names["seq_region_id"], names["name"]))
        return {
            gene: (str(name_of[region]), int(start), int(end), int(strand))
            for gene, region, start, end, strand in zip(
                table["stable_id"],
                table["seq_region_id"],
                table["seq_region_start"],
                table["seq_region_end"],
                table["seq_region_strand"],
                strict=True,
            )
        }

    def region_kinds(self, assembly: int, release: int) -> dict[str, str]:
        """Return ``{region name: kind}`` of the regions of the kept genes at a release, read from the tables alone.

        ``mitochondrial`` for a region with the attribute ``codon_table``; ``chromosome`` for a region of the
        coordinate system ``chromosome`` or with the attribute ``karyotype_rank``; ``scaffold`` otherwise.
        """
        names = self._frame(assembly, release, "seq_region")
        systems = self._frame(assembly, release, "coord_system")
        chromosome_systems = set(systems.loc[systems["name"] == "chromosome", "coord_system_id"])
        attributes = self._frame(assembly, release, "attrib_type")
        marks = self._frame(assembly, release, "seq_region_attrib")

        def with_code(code: str) -> set:
            return set(
                marks.loc[
                    marks["attrib_type_id"].isin(attributes.loc[attributes["code"] == code, "attrib_type_id"]),
                    "seq_region_id",
                ]
            )

        mitochondrial, ranked = with_code("codon_table"), with_code("karyotype_rank")
        return {
            str(name): (
                "mitochondrial"
                if region in mitochondrial
                else "chromosome" if region in ranked or system in chromosome_systems else "scaffold"
            )
            for region, name, system in zip(names["seq_region_id"], names["name"], names["coord_system_id"])
        }

    def _locus_facts(
        self,
    ) -> tuple[
        dict[str, dict[int, set[tuple[int, str, int, int, int, str]]]], dict[str, set[int]], set[tuple[str, str]]
    ]:
        """Return every gene's places on the reference sequence by release, the releases it lives, and the history's pairs.

        Places are ``(assembly, region, start, end, strand, kind)`` at releases at which the gene does not lie on a
        ``non_ref`` region, on the assembly that serves the release where that is known (else on every assembly that
        has it), read where the tables hold the release's regions, a place in a pseudoautosomal region of Y read on
        the region of X it repeats (the regions are the package's, :py:data:`idtrack.organisms.PSEUDOAUTOSOMAL_REGIONS`);
        a gene lives at the releases at which the gene tables list it (on such an assembly); the pairs are ``(old id,
        new id)`` of every gene event of every core database of the tables.
        """
        if self._loci is None:
            places: dict[str, dict[int, set[tuple[int, str, int, int, int, str]]]] = {}
            alive: dict[str, set[int]] = {}
            events: set[tuple[str, str]] = set()
            tables = self.tables.tables
            for assembly, release, name in tables:
                if name != "gene" or int(release) > self.tables.snapshot:
                    continue
                if self.serving and self.serving.get(int(release)) != int(assembly):
                    continue  # a release counts on the assembly that serves it
                if any(
                    (assembly, release, other) not in tables
                    for other in ("seq_region", "seq_region_attrib", "coord_system")
                ):
                    continue  # a release of an assembly the tables list without its regions (GRCh37 after 75)
                beside = self.non_reference_genes(assembly, release)
                kinds = self.region_kinds(assembly, release)
                name = organisms.assembly_name(self.tables.organism, int(assembly)) or ""
                repeats = organisms.PSEUDOAUTOSOMAL_REGIONS.get(name, ())
                for gene, (region, start, end, strand) in self.places(assembly, release).items():
                    alive.setdefault(gene, set()).add(int(release))
                    if gene in beside:
                        continue
                    kind = kinds[region]
                    for (y_region, y_first, y_last), (x_region, x_first, _x_last) in repeats:
                        if region == y_region and y_first <= start and end <= y_last:  # read on the X it repeats
                            region, start, end = x_region, start - y_first + x_first, end - y_first + x_first
                    places.setdefault(gene, {}).setdefault(int(release), set()).add(
                        (int(assembly), region, start, end, strand, kind)
                    )
            for assembly, release, name in tables:
                if name == "stable_id_event":
                    table = self._frame(assembly, release, "stable_id_event")
                    table = table[table["type"] == "gene"]
                    events.update(zip(table["old_stable_id"].astype(str), table["new_stable_id"].astype(str)))
            self._loci = (places, alive, events)
        return self._loci

    @staticmethod
    def _whole(places: set[tuple[int, str, int, int, int, str]]) -> set[str] | None:
        """The names of the sequences of some places when all are chromosomes or the mitochondrial genome (it by its kind)."""
        if not places or any(kind not in ("chromosome", "mitochondrial") for *_rest, kind in places):
            return None
        return {"mitochondrial" if kind == "mitochondrial" else region for _a, region, *_rest, kind in places}

    def apart(self, one: str, other: str) -> bool:
        """Tell whether two genes lie apart, on one assembly or, across assemblies, on whole sequences of other names.

        Apart on a common assembly: no places overlap there; else apart on whole sequences of different names (two
        chromosomes, or a chromosome and the mitochondrial genome).

        Read from the tables alone (:py:meth:`_locus_facts`): one region, one strand and a base in common overlap; a
        gene of a ``non_ref`` region at some release is never compared by names (the package's relation says
        ``outside_reference`` first).
        """
        places, _alive, _events = self._locus_facts()
        mine = set().union(*places.get(one, {}).values()) if one in places else set()
        theirs = set().union(*places.get(other, {}).values()) if other in places else set()
        common = {p[0] for p in mine} & {p[0] for p in theirs}
        if common:
            return not any(
                a == a2 and r == r2 and strand == strand2 and start <= end2 and start2 <= end
                for a, r, start, end, strand, _k in mine
                for a2, r2, start2, end2, strand2, _k2 in theirs
            )
        if self.outside_somewhere(one) or self.outside_somewhere(other):
            return False
        names, names_2 = self._whole(mine), self._whole(theirs)
        return names is not None and names_2 is not None and not names & names_2

    def outside_somewhere(self, gene: str) -> bool:
        """Tell whether the tables list a gene on a ``non_ref`` region at some release (a patch, a haplotype, an LRG)."""
        tables = self.tables.tables
        return any(
            gene in self.non_reference_genes(assembly, release)
            for assembly, release, name in tables
            if name == "gene"
            and int(release) <= self.tables.snapshot
            and all((assembly, release, other) in tables for other in ("seq_region", "seq_region_attrib"))
        )

    def holders(self, identifier: str) -> dict[str, set[int]]:
        """Return the releases at which the tables give an identifier to each gene, each release on its serving assembly.

        An external identifier (any database, as id or synonym, any letter case) or a transcript (its stable id: the
        releases at which it belongs to the gene).
        """
        found: dict[str, set[int]] = {}
        text = identifier.lower()
        for assembly, release, name in self.tables.tables:
            if name != "xref" or int(release) > self.tables.snapshot:
                continue
            if self.serving and self.serving.get(int(release)) != int(assembly):
                continue  # a release counts on the assembly that serves it
            for gene, _database, value, _kind, _form in self._links_of_release(assembly, release):
                if value.lower() == text:
                    found.setdefault(gene, set()).add(int(release))
            for transcript, (_version, gene) in self.products(assembly, release)["transcript"].items():
                if transcript.lower() == text:
                    found.setdefault(gene, set()).add(int(release))
        return found

    def is_synonym(self, identifier: str, genes: set[str]) -> bool:
        """Tell whether the tables list an identifier as a synonym of one of some genes at some release."""
        text = identifier.lower()
        return any(
            gene in genes and kind == "synonym" and value.lower() == text
            for assembly, release, name in self.tables.tables
            if name == "xref" and int(release) <= self.tables.snapshot
            for gene, _database, value, kind, _form in self._links_of_release(assembly, release)
        )

    def one_locus_count(self, genes: set[str]) -> int:
        """Return how many loci of the reference sequence some genes are shown to be, read from the tables alone.

        Written apart from the package's rule (:py:func:`idtrack.graph.genes.count_genes`) and to the same definition:
        a gene that the tables list only on ``non_ref`` regions is a copy and adds no locus (copies alone are one).
        The others are compared in frames: each release (genes listed then and on the reference sequence then; genes
        whose places share a base on one region and one strand are one locus, through a chain of such genes of the
        set), each assembly (genes placed on it: one locus where places overlap at any of its releases, or where the
        id history links them through events between genes never listed at one release), and the names of whole
        sequences across assemblies (genes placed only on chromosomes or the mitochondrial genome: one where they
        share a name, the mitochondrial genome by its kind, or are linked so by history). Two genes are told apart when
        a frame that holds both puts them at two loci. The count starts from the loci of the frame with the most of
        them (the first such frame, in the order: releases, assemblies, names) and takes in, gene by gene in sorted
        order, each gene told apart from some gene of every locus so far; at least one.
        """
        if len(genes) < 2:
            return len(genes)
        places, alive, events = self._locus_facts()
        reference = sorted(gene for gene in genes if gene in places or gene not in alive)
        if len(reference) < 2:
            return 1
        root = {gene: gene for gene in reference}

        def find(gene: str) -> str:
            while root[gene] != gene:
                gene = root[gene]
            return gene

        for one in reference:
            for other in reference:
                linked = (one, other) in events or (other, one) in events
                if linked and not (alive.get(one, set()) & alive.get(other, set())):
                    a, b = find(one), find(other)
                    if a != b:
                        root[max(a, b)] = min(a, b)

        def frame(members: list[str], same) -> dict[str, int]:
            locus = {gene: index for index, gene in enumerate(members)}
            changed = True
            while changed:
                changed = False
                for one in members:
                    for other in members:
                        if locus[one] != locus[other] and same(one, other):
                            locus[one] = locus[other] = min(locus[one], locus[other])
                            changed = True
            return locus

        def overlap(mine, theirs) -> bool:
            return any(
                a == a2 and r == r2 and strand == strand2 and start <= end2 and start2 <= end
                for a, r, start, end, strand, _k in mine
                for a2, r2, start2, end2, strand2, _k2 in theirs
            )

        frames: list[dict[str, int]] = []
        for release in sorted(set().union(*(alive.get(gene, set()) for gene in reference))):
            here = {gene: places[gene][release] for gene in reference if release in places.get(gene, {})}
            if len(here) > 1:
                frames.append(frame(sorted(here), lambda a, b, at=here: overlap(at[a], at[b])))
        everywhere = {gene: set().union(*places[gene].values()) for gene in reference if gene in places}
        for assembly in sorted({p[0] for found in everywhere.values() for p in found}):
            on = {gene: {p for p in found if p[0] == assembly} for gene, found in everywhere.items()}
            members = [gene for gene in reference if on.get(gene)]
            if len(members) > 1:
                frames.append(frame(members, lambda a, b, at=on: overlap(at[a], at[b]) or find(a) == find(b)))
        whole = {
            gene: names for gene, names in ((g, self._whole(f)) for g, f in everywhere.items()) if names is not None
        }
        members = [gene for gene in reference if gene in whole]
        if len(members) > 1:
            frames.append(frame(members, lambda a, b: bool(whole[a] & whole[b]) or find(a) == find(b)))
        if not frames:
            return 1
        best = frames[0]
        for candidate in frames[1:]:
            if len(set(candidate.values())) > len(set(best.values())):
                best = candidate
        groups: dict[int, list[str]] = {}
        for gene in sorted(best):
            groups.setdefault(best[gene], []).append(gene)
        loci = list(groups.values())

        def apart(one: str, other: str) -> bool:
            return any(one in f and other in f and f[one] != f[other] for f in frames)

        for gene in reference:
            if gene not in best and all(any(apart(gene, other) for other in locus) for locus in loci):
                loci.append([gene])
        return max(1, len(loci))

    def links(self, assembly: int, release: int) -> list[tuple[str, str, str, str, str]]:
        """Return ``(gene id, database, external id, "id" | "synonym", form)`` for every cross reference of a release.

        A cross reference of a transcript or translation is reported for its gene; ``form`` says what it is
        attached to in Ensembl (``gene``, ``transcript`` or ``translation``). The synonyms of a cross reference are
        reported under the database of its labels. A gene's display name that no selected database gives the gene
        itself is reported as a cross reference of the gene in the database of its labels (I129: held whatever the
        selection); where that database is not selected for the genes at the release, its other rows of the genes
        there are not reported (the graph holds that database's display names only).
        """
        where = (int(assembly), int(release))
        if where not in self._links:
            shared = self._mirbase_shared()
            rows = self._links_of_release(assembly, release)
            shown = self._shown(assembly, release)
            only_shown = self._display_databases(assembly, release)
            rows = [row for row in rows if not (row[4] == "gene" and row[1] in only_shown and row[3] == "id")] + [
                (gene, database, value, "id", "gene") for gene, database, value in shown
            ]
            self._links[where] = sorted(
                {row for row in rows if not (row[1] in self._mirbase_names() and row[2].lower() in shared)}
            )
        return self._links[where]

    def _shown(self, assembly: int, release: int) -> list[tuple[str, str, str]]:
        """Return the display names (gene, database of its labels, name) that no selected database gives the gene.

        A display name that is itself the stable id of a gene of the tables (the display name of an LRG record is its
        own id) is no external identifier: the graph keeps it as that Ensembl id.
        """
        rows = self._links_of_release(assembly, release)
        held = {
            (gene, value)
            for gene, name, value, kind, form in rows
            if kind == "id" and form == "gene" and self._selected_by_yaml(form, name, assembly, release)
        }
        reserved = self._gene_stable_ids()
        return [
            (gene, database, value)
            for gene, database, value in self.display_names(assembly, release)
            if (gene, value) not in held and value not in reserved
        ]

    def _gene_stable_ids(self) -> set[str]:
        """Return the stable ids of the genes of every assembly and release of the tables."""
        if self._reserved is None:
            self._reserved = {
                gene
                for assembly, release, name in self.tables.tables
                if name == "xref"
                for gene in self.genes(assembly, release)
            }
        return self._reserved

    def _display_databases(self, assembly: int, release: int) -> set[str]:
        """Return the databases of display names held whatever the selection that it does not select for the genes."""
        key = (int(assembly), int(release))
        if key not in self._display_dbs:
            self._display_dbs[key] = {
                database
                for _gene, database, _value in self._shown(assembly, release)
                if not self._selected_by_yaml("gene", database, assembly, release)
            }
        return self._display_dbs[key]

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
                        database = name if kind == "id" else f"{ids.SYNONYM_PREFIX}{name}"
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
                    database = name if the_kind == "id" else f"{ids.SYNONYM_PREFIX}{name}"
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


def answer_of(result: Any, final_database: str | None, strategy: str = "best") -> list[str]:
    """Return the ids that a result of ``API.convert`` gives as its answer (one view) in the database asked for.

    Nothing when the identifier was not found or has no counterpart, and, for a conversion into an external
    database, nothing when the genes reached have no id in it (the Ensembl ids returned then are not an answer in
    that database).
    """
    if result.graph_id is None or not result.candidates:
        return []
    if final_database and (result.no_target or result.final_database != final_database):
        return []
    return list(result.targets(strategy))


def verdict(query: dict[str, Any], result: Any) -> str | None:
    """Compare a result with the expected answer of a query; return ``None`` when right, else what is wrong.

    The rules are those of the truth the queries come from. ``expect`` is the set of right targets. With
    ``strategy="all"``, or with the rule ``set_exact`` (the right answer is several genes), the answer must be
    exactly that set; otherwise (``best`` with the rule ``set``) it must be a non-empty part of it, or nothing when
    nothing is expected. ``cmp`` says how ids are compared: Ensembl gene ids with their version (``versioned``),
    without it (``base``: the release has no versions in Ensembl's files), external ids as written (``exact``). The
    genes of ``apart`` (copies on patches and alternative loci) must be reported in ``alternative_locus_genes``.
    """
    got = answer_of(result, query["final_database"], query["strategy"])
    if len(set(got)) != len(got):
        return f"a target is returned twice: {got}"
    reported = set(got)
    apart = set(result.alternative_locus_genes)
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


def with_y_copies_apart(world, query):
    """Return a query whose expected Y copies of pseudoautosomal genes are reported apart beside the gene on X.

    Decision 95: a name the X and Y copies of a pseudoautosomal gene share answers the X copy, the Y copy listed
    apart (Ensembl's tables give the name to both, from release 110); a Y copy is one at the place of an X gene expected
    (:py:meth:`Ensembl.y_twins`). The places are read from the tables, the regions from the package.
    """
    if query["final_database"] is not None or query["cmp"] != "versioned" or len(query["expect"]) < 2:
        return query
    assembly = world.backbone.get(query["to_release"])
    if assembly is None:
        return query
    twins = world.ensembl.y_twins(assembly, query["to_release"])
    expected = {e: e.rpartition(".")[0] for e in query["expect"]}
    y = {e for e, gene in expected.items() if twins.get(gene, set()) & set(expected.values())}
    if not y:
        return query
    return {**query, "expect": sorted(set(query["expect"]) - y), "apart": sorted(set(query.get("apart") or ()) | y)}


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
        built: The :py:class:`idtrack.API` that :py:meth:`idtrack.API.build` built from the tables.
        api: A second :py:class:`idtrack.API`, for which :py:meth:`idtrack.API.build` loaded the saved file.
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
            self.built = idtrack.API.build(organism, SNAPSHOT, self.repository)
            self.tables_served = self.tables.served

            def no_manager(*args, **kwargs):
                raise AssertionError("Loading a graph file needed a DatabaseManager.")

            monkeypatch.setattr(DatabaseManager, "__init__", no_manager)
            self.api = idtrack.API.build(organism, SNAPSHOT, self.repository)
        releases_of = {int(a): set(payload["releases"]) for a, payload in self.tables.indices.items()}
        self.backbone = {}
        for release in sorted(set().union(*releases_of.values())):
            if release <= SNAPSHOT:
                self.backbone[release] = next(
                    int(a) for a in organisms.assemblies_by_priority(organism) if release in releases_of.get(int(a), ())
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
