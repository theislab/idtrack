#!/usr/bin/env python3
"""Write the cut Ensembl tables of the tests: ``tests/data/real/<organism>.tables.json.xz``.

Usage (from the root of the repository, with the package importable)::

    IDTRACK_ENSEMBL_ACCESS=https python tests/data/real/build_real_tables.py \
        --organism sus_scrofa --repository <local repository> [--snapshot 115] [--check-only]

What it does:

1. It builds the graph of the organism with the production code (``DatabaseManager``, ``GraphMaker``) in a scratch
   repository, and answers every download of a raw Ensembl table with that table restricted to the genes of
   ``tests/data/real/<organism>.genes.txt``. The complete tables come from ``--repository``: a local repository of
   idtrack, which holds them already or downloads them from Ensembl's HTTPS dumps and keeps them (a build of the
   organism's complete graph in that repository leaves every table there; an empty one works too, with the
   downloads of a complete build: tens of GB for human).
2. It writes what was served (and which columns Ensembl's schema of a release lacks) to the table file.
3. It builds the graph once more from the written file alone, with every socket refused, and checks that the two
   graphs have the same content.

The cut. A gene is kept when its stable id is in the gene list, at every assembly and release; transcripts and
translations are kept with their gene; cross references (``object_xref``, ``xref``, ``external_synonym``,
``identity_xref``, ``external_db``) are kept for the kept genes, transcripts and translations and for the external
databases that the organism's externals selection names at some release, form or assembly; of the id history, the
gene events are kept, and both sides of every kept event must be in the gene list (the script stops and names the
genes to add otherwise, so the list is always closed under Ensembl's id history); ``mapping_session`` is kept
whole; of ``seq_region_attrib`` the rows of the kept genes' sequence regions that say ``toplevel`` or
``non_ref``. The surrogate keys of a core database (``gene_id``, ``transcript_id``, ``translation_id``,
``xref_id``, ``object_xref_id``, ``seq_region_id``) mean nothing outside it and differ in every release; they are
renumbered from 1: genes, transcripts and translations in the order of their stable ids, cross references in the
order in which these objects refer to them, so that the tables of consecutive releases are equal where Ensembl's
data are and the keys are runs of consecutive numbers (the file stores them as differences). Stable ids, versions,
external ids, database names, events and sessions are Ensembl's, untouched.

When to run it: after a change of a gene list, of the externals selection of an organism (a database enabled for a
release that the cut tables have no cross reference of stops the test build with the name of the database), of the
organism table, or of the tables a graph build reads. The snapshot release is fixed (115) because the expected
answers of ``<organism>.truth.jsonl`` are answers for that release; for a later snapshot the truth has to be
derived again from Ensembl's files of that release.

Size: a table file must stay under 500 kB, the limit of the repository's pre-commit hook for an added file
(``check-added-large-files``); the script fails above it, and so does a test. About 700 bytes per pig gene and
release history, more for human: shorten the gene list when the limit is reached (``--check-only`` verifies an
existing file and prints its size).
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from idtrack._database_manager import DatabaseManager  # noqa: E402
from idtrack._db import DB, EnsemblSchemaError  # noqa: E402
from idtrack._graph_maker import GraphMaker  # noqa: E402
from tests._real_tables import RealTables, read_genes, tables_path  # noqa: E402

SIZE_LIMIT = 500 * 1024  # bytes per table file: the limit of the pre-commit hook check-added-large-files
OBJECT_TABLES = {
    "gene": "gene",
    "gene_stable_id": "gene",
    "transcript": "transcript",
    "transcript_stable_id": "transcript",
    "translation": "translation",
    "translation_stable_id": "translation",
}
REGION_CODES = ("toplevel", DB.non_reference_attribute)
FORMS = ["gene", "transcript", "translation"]
ROW_ORDER = {
    "object_xref": ["object_xref_id"],
    "identity_xref": ["object_xref_id"],
    "xref": ["xref_id"],
    "external_synonym": ["xref_id"],
    "gene": ["gene_id"],
    "gene_stable_id": ["gene_id"],
    "transcript": ["transcript_id"],
    "transcript_stable_id": ["transcript_id"],
    "translation": ["translation_id"],
    "translation_stable_id": ["translation_id"],
}


class Cutter:
    """Answers the downloads of a graph build with Ensembl's tables restricted to a list of genes."""

    def __init__(self, organism: str, genes: list[str], source_repository: str, snapshot: int):
        """Remember the organism, its genes and the repository of the complete tables."""
        self.organism = organism
        self.genes = set(genes)
        self.source_repository = os.path.realpath(source_repository)
        self.snapshot = int(snapshot)
        self.original_download = DatabaseManager.download_table
        self.keys: dict[tuple[int, int], dict[str, Any]] = {}
        self.record: dict[tuple[int, int, str], dict] = {}
        self._relevant: set[str] | None = None
        self.open_events: set[str] = set()
        self.order: dict[tuple[int, int, str], list] = {}

    # ------------------------------------------------------------------ the complete tables

    def source(self, manager: DatabaseManager, form: str | None = None) -> DatabaseManager:
        """Return the manager of the complete tables for the assembly, release (and form) of a manager."""
        return DatabaseManager(
            organism=manager.organism,
            form=form or manager.form,
            local_repository=self.source_repository,
            ensembl_release=manager.ensembl_release,
            ignore_before=manager.ignore_before,
            ignore_after=manager.ignore_after,
            genome_assembly=manager.genome_assembly,
        )

    def relevant_databases(self, manager: DatabaseManager) -> set[str]:
        """Return the external databases that the externals selection names at any release, form or assembly."""
        if self._relevant is None:
            content = manager.external_inst.read_modified_yaml()[self.organism]
            names = {
                database
                for per_form in content.values()
                for database, entry in per_form.items()
                if any(per_assembly.get("Include") for per_assembly in entry["Assembly"].values())
            }
            prefix = DB.synonym_id_nodes_prefix
            self._relevant = {name[len(prefix) :] if name.startswith(prefix) else name for name in names}
        return self._relevant

    def keys_of(self, manager: DatabaseManager) -> dict:
        """Return which rows of one core database are kept, and the new number of every surrogate key."""
        where = (int(manager.genome_assembly), int(manager.ensembl_release))
        if where in self.keys:
            return self.keys[where]
        source = self.source(manager)
        genes = source.get_db("idsraw_gene")
        genes = genes[genes["gene_stable_id"].isin(self.genes)].sort_values("gene_stable_id")
        transcripts = source.get_db("idsraw_transcript")
        transcripts = transcripts[transcripts["gene_id"].isin(genes["gene_id"])].sort_values("transcript_stable_id")
        translations = source.get_db("idsraw_translation")
        translations = translations[translations["transcript_id"].isin(transcripts["transcript_id"])]
        translations = translations.sort_values("translation_stable_id")
        number: dict[str, Any] = {
            "gene": {int(k): i for i, k in enumerate(genes["gene_id"], 1)},
            "transcript": {int(k): i for i, k in enumerate(transcripts["transcript_id"], 1)},
            "translation": {int(k): i for i, k in enumerate(translations["translation_id"], 1)},
        }

        regions = source.get_table("gene", usecols=["gene_id", "seq_region_id"])
        regions = regions[regions["gene_id"].isin(list(number["gene"]))]
        number["seq_region"] = {int(k): i for i, k in enumerate(sorted(set(regions["seq_region_id"].astype(int))), 1)}
        attributes = source.get_table("attrib_type", usecols=["attrib_type_id", "code"])
        number["attrib_type"] = {
            int(k) for k in attributes.loc[attributes["code"].isin(REGION_CODES), "attrib_type_id"].tolist()
        }

        external_db = source.get_table("external_db", usecols=["external_db_id", "db_name", "db_display_name"])
        relevant = self.relevant_databases(manager)
        keep_db = external_db["db_name"].isin(relevant) | external_db["db_display_name"].isin(relevant)
        database_ids = set(pd.to_numeric(external_db.loc[keep_db, "external_db_id"]).astype(int))
        xref = source.get_table("xref", usecols=["xref_id", "external_db_id", "dbprimary_acc", "display_label"])
        xref = xref[pd.to_numeric(xref["external_db_id"], errors="coerce").isin(database_ids)]
        object_xref = source.get_table(
            "object_xref", usecols=["ensembl_id", "ensembl_object_type", "xref_id", "object_xref_id"]
        )
        object_xref = object_xref[object_xref["xref_id"].isin(xref["xref_id"])]
        kinds = object_xref["ensembl_object_type"].str.lower()
        kept = np.zeros(len(object_xref), dtype=bool)
        for form in ("gene", "transcript", "translation"):
            kept |= ((kinds == form) & object_xref["ensembl_id"].isin(list(number[form]))).to_numpy()
        object_xref = object_xref[kept].assign(kind=kinds[kept])
        xref = xref[xref["xref_id"].isin(object_xref["xref_id"])]
        number["external_db"] = set(pd.to_numeric(xref["external_db_id"]).astype(int))
        # Cross references are numbered in the order in which the kept objects refer to them (genes, then
        # transcripts, then translations, each in the order of its new number; within an object by database and
        # accession), and the rows of `object_xref` in that same order: both keys are then runs of consecutive
        # numbers, which is what lets the table file stay small.
        described = xref.rename(columns={"xref_id": "old_xref"})
        described = described.assign(external_db_id=pd.to_numeric(described["external_db_id"]))
        references = pd.DataFrame(
            {
                "rank": [FORMS.index(k) for k in object_xref["kind"]],
                "object": [number[k][int(i)] for k, i in zip(object_xref["kind"], object_xref["ensembl_id"])],
                "old_xref": object_xref["xref_id"].astype(int).to_numpy(),
                "key": object_xref["object_xref_id"].astype(int).to_numpy(),
            }
        ).merge(described, on="old_xref", validate="many_to_one")
        references = references.sort_values(
            ["rank", "object", "external_db_id", "dbprimary_acc", "display_label", "old_xref", "key"],
            kind="stable",
            na_position="last",
        )
        number["xref"] = {}
        for old in references["old_xref"].tolist():
            number["xref"].setdefault(int(old), len(number["xref"]) + 1)
        number["object_xref"] = {}
        for key in references["key"].tolist():
            number["object_xref"].setdefault(int(key), len(number["object_xref"]) + 1)
        self.keys[where] = number
        return number

    # ------------------------------------------------------------------ the cut

    @staticmethod
    def _renumber(table: pd.DataFrame, column: str, number: dict[int, int]) -> pd.DataFrame:
        if column not in table.columns:
            return table
        return table.assign(**{column: [number[int(k)] for k in table[column]]})

    def cut(self, manager: DatabaseManager, table_key: str, table: pd.DataFrame) -> pd.DataFrame:
        """Return a complete raw table restricted to the kept rows, surrogate keys renumbered, rows in a fixed order."""
        if table_key == "mapping_session":
            return table
        if table_key == "stable_id_event":
            events = table[table["type"] == "gene"]
            old_in = events["old_stable_id"].isin(self.genes)
            new_in = events["new_stable_id"].isin(self.genes)
            one_side = (old_in & events["new_stable_id"].notna() & ~new_in) | (
                new_in & events["old_stable_id"].notna() & ~old_in
            )
            window = events.loc[one_side]
            if len(window):
                sessions = self.source(manager).get_table("mapping_session", usecols=None)
                first = dict(zip(sessions["mapping_session_id"], sessions["old_release"]))
                in_window = [
                    manager.ignore_before <= float(first[s]) <= manager.ignore_after
                    for s in window["mapping_session_id"]
                ]
                window = window[in_window]
                self.open_events |= (set(window["old_stable_id"]) | set(window["new_stable_id"])) - self.genes
            return events[
                (old_in | events["old_stable_id"].isna())
                & (new_in | events["new_stable_id"].isna())
                & (old_in | new_in)
            ]
        number = self.keys_of(manager)
        if table_key in OBJECT_TABLES:
            form = OBJECT_TABLES[table_key]
            table = table[table[f"{form}_id"].isin(list(number[form]))]
            for column, kind in (
                ("gene_id", "gene"),
                ("transcript_id", "transcript"),
                ("translation_id", "translation"),
            ):
                table = self._renumber(table, column, number[kind])
            return self._renumber(table, "seq_region_id", number["seq_region"])
        if table_key == "object_xref":
            # by the object as well as by the key: some old core databases use a key for more than one row
            kinds = table["ensembl_object_type"].str.lower()
            kept = np.zeros(len(table), dtype=bool)
            for form in ("gene", "transcript", "translation"):
                kept |= ((kinds == form) & table["ensembl_id"].isin(list(number[form]))).to_numpy()
            kept &= table["object_xref_id"].isin(list(number["object_xref"])).to_numpy()
            kept &= table["xref_id"].isin(list(number["xref"])).to_numpy()
            table = table[kept]
            kinds = table["ensembl_object_type"].str.lower()
            table = table.assign(ensembl_id=[number[k][int(i)] for k, i in zip(kinds, table["ensembl_id"])])
            return self._renumber(
                self._renumber(table, "xref_id", number["xref"]), "object_xref_id", number["object_xref"]
            )
        if table_key in ("xref", "external_synonym"):
            table = table[table["xref_id"].isin(list(number["xref"]))]
            return self._renumber(table, "xref_id", number["xref"])
        if table_key == "identity_xref":
            table = table[table["object_xref_id"].isin(list(number["object_xref"]))]
            return self._renumber(table, "object_xref_id", number["object_xref"])
        if table_key == "external_db":
            return table[pd.to_numeric(table["external_db_id"], errors="coerce").isin(number["external_db"])]
        if table_key == "attrib_type":
            return table[table["attrib_type_id"].isin(number["attrib_type"])]
        if table_key == "seq_region_attrib":
            table = table[table["seq_region_id"].isin(list(number["seq_region"]))]
            table = table[table["attrib_type_id"].isin(number["attrib_type"])]
            return self._renumber(table, "seq_region_id", number["seq_region"])
        raise ValueError(f"No rule to cut the table {table_key!r}: add one to {Path(__file__).name}.")

    # ------------------------------------------------------------------ the recording download

    def download(self, manager: DatabaseManager, table_key: str, usecols: list | None = None) -> pd.DataFrame:
        """Answer a download: the complete table for the source repository, the cut one (recorded) otherwise."""
        if os.path.realpath(manager.local_repository) == self.source_repository:
            return self.original_download(manager, table_key, usecols)
        where = (int(manager.genome_assembly), int(manager.ensembl_release), table_key)
        entry = self.record.setdefault(where, {"columns": [], "data": [], "absent": []})
        try:
            complete = self.source(manager).get_table(table_key, usecols=usecols or None)
        except EnsemblSchemaError as error:
            entry["absent"] = sorted(set(entry["absent"]) | set(error.missing_columns))
            raise
        table = self.cut(manager, table_key, complete)
        # Rows in the order of the table's own key (the new numbers), then of their content. A table is asked for
        # more than once with other columns (``gene``): the complete tables of one core database have their rows
        # in one order, so the order found for the first set of columns is the order of the others.
        if where not in self.order:
            first = [column for column in ROW_ORDER.get(table_key, ()) if column in table.columns]
            by = first + [column for column in table.columns if column not in first]
            self.order[where] = list(table.sort_values(by, kind="stable", na_position="last").index)
        table = table.loc[self.order[where]].reset_index(drop=True)
        have = {name for name, _kind in entry["columns"]}
        for column in table.columns:
            if column in have:
                continue
            values = table[column]
            numeric = pd.api.types.is_numeric_dtype(values)
            entry["columns"].append([column, "number" if numeric else "text"])
            entry["data"].append([_plain(value, numeric) for value in values.tolist()])
        lengths = {len(column) for column in entry["data"]}
        if len(lengths) > 1:
            raise ValueError(f"The columns of {where} were cut to different rows: {sorted(lengths)}.")
        return table.copy()

    def install(self, monkeypatch) -> None:
        """Route every download of ``DatabaseManager`` through this cutter."""
        cutter = self
        monkeypatch.setattr(DB, "ensembl_access", "https")
        monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
        monkeypatch.setattr(
            DatabaseManager,
            "download_table",
            lambda self, table_key, usecols=None: cutter.download(self, table_key, usecols),
        )


def _plain(value, numeric: bool):
    """Return a cell as JSON holds it: ``None`` for a missing value, an ``int`` for an integral number."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    if not numeric:
        return str(value)
    return int(value) if float(value).is_integer() else float(value)


def build_graph(organism: str, repository: str, snapshot: int):
    """Build the organism's graph in ``repository`` as ``API.build_graph`` does, without saving it."""
    manager = DatabaseManager(
        organism=organism, form=DB.backbone_form, local_repository=repository, ignore_after=snapshot
    )
    return GraphMaker(manager).construct_graph()


def indices_of(organism: str, repository: str) -> dict[str, dict]:
    """Read the saved indices of Ensembl's core databases that a build left in a repository."""
    import json

    indices = {}
    for assembly in DB.assemblies_by_priority(organism):
        path = DatabaseManager._https_index_path(repository, organism, int(assembly))
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                indices[str(int(assembly))] = json.load(handle)
    return indices


def replay(organism: str, path: str, snapshot: int):
    """Build the graph from a table file alone, with every socket refused."""

    def refuse(*args, **kwargs):
        raise AssertionError("The replay of the cut tables opened a socket.")

    scratch = tempfile.mkdtemp(prefix="idtrack_real_replay_")
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
            RealTables.load(organism, path).install(monkeypatch, scratch)
            monkeypatch.setattr(socket, "socket", refuse)
            return build_graph(organism, scratch, snapshot)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    """Run the script; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--organism", required=True)
    parser.add_argument("--repository", help="local repository with (or for) the complete tables")
    parser.add_argument("--snapshot", type=int, default=115)
    parser.add_argument("--out", help="write here instead of the repository's test data")
    parser.add_argument("--check-only", action="store_true", help="only build from the existing file and report")
    arguments = parser.parse_args()
    organism, snapshot = arguments.organism, arguments.snapshot
    out = arguments.out or str(tables_path(organism))

    if not arguments.check_only:
        if not arguments.repository:
            parser.error("--repository is needed to cut the tables")
        genes = read_genes(organism)
        cutter = Cutter(organism, genes, arguments.repository, snapshot)
        scratch = tempfile.mkdtemp(prefix="idtrack_real_cut_")
        try:
            with pytest.MonkeyPatch.context() as monkeypatch:
                cutter.install(monkeypatch)
                # the index of every assembly: read from (or scanned into) the source repository, then copied
                DatabaseManager(
                    organism=organism,
                    form=DB.backbone_form,
                    local_repository=arguments.repository,
                    ignore_after=snapshot,
                ).available_releases_all_assemblies
                for assembly in indices_of(organism, arguments.repository):
                    shutil.copy(
                        DatabaseManager._https_index_path(arguments.repository, organism, int(assembly)),
                        DatabaseManager._https_index_path(scratch, organism, int(assembly)),
                    )
                monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
                try:
                    graph = build_graph(organism, scratch, snapshot)
                finally:
                    if cutter.open_events:
                        print(
                            f"The gene list is not closed under Ensembl's id history; add: {sorted(cutter.open_events)}",
                            file=sys.stderr,
                        )
                if cutter.open_events:
                    return 1
                indices = indices_of(organism, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        RealTables(organism, snapshot, genes, indices, cutter.record).save(out)
        digest = graph.manifest["content_digest"]
        print(f"cut: {len(genes)} genes, {len(cutter.record)} tables, graph of {graph.manifest['counts']}")
    again = replay(organism, out, snapshot)
    size = os.path.getsize(out)
    print(f"replayed from {out}: {size} bytes, graph content {again.manifest['content_digest']}")
    if not arguments.check_only and again.manifest["content_digest"] != digest:
        print("The graph built from the written file differs from the graph built while cutting.", file=sys.stderr)
        return 1
    if size > SIZE_LIMIT:
        print(f"The file is larger than the limit of {SIZE_LIMIT} bytes.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
