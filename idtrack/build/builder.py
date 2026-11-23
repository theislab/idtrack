#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Build a graph from an organism's Ensembl tables: the id history, the cross-references and the graph file."""

import dataclasses
import datetime
import functools
import hashlib
import json
import logging
import math
import os
import warnings
from collections import Counter
from collections.abc import Callable, Collection
from typing import Any

import numpy as np
import pandas as pd

from idtrack import ids, organisms
from idtrack._version import build_version, package_commit
from idtrack.build.assembly_chain import AssemblyChain
from idtrack.build.database_manager import DatabaseManager, entries_of_several_genes
from idtrack.errors import (
    BuildError,
    HistoryValidationError,
    IntegrityError,
    InvalidArgumentError,
    UnknownReleaseError,
)
from idtrack.graph import Connection, GraphRecord, ReleaseSet, TheGraph, files, serving_line
from idtrack.graph.genes import SCAFFOLD, Exceptions, Repeats, count_genes, facts_by_stable_id
from idtrack.graph.store import _LinkMarks, _Places, _Store
from idtrack.graph.types import _KIND_HISTORY, _KIND_XREF, AssemblyException
from idtrack.settings import GraphSettings


class GraphMaker:
    """Builds the identifier graph of one organism and keeps it in the local repository.

    The graph (:py:class:`idtrack.TheGraph`) holds the history of Ensembl gene ids over the releases of
    the manager's window, the transcripts and translations of every release linked to their genes, versionless base
    gene ids, and the identifiers of the selected external databases linked to the Ensembl ids they annotate.

    The history is not copied from Ensembl's ``mapping_session`` table unchecked: the releases of every event are
    validated against the id tables of the releases (which ids exist where), sessions whose recorded releases
    contradict the tables are corrected, and every correction is logged and recorded in the graph's record.

    A built graph is written as one file whose record says what it is and what built it; :py:meth:`get_graph` loads a
    file only if that record is the graph that is asked for.
    """

    # Note: Example chaotic ID history: ENSG00000263464
    def __init__(self, database_manager: DatabaseManager):
        """Class initialization.

        Args:
            database_manager: Needed to download all necessary tables and data frames.
                It contains the temporary directory to save the resultant graph.

        Raises:
            InvalidArgumentError: If the release of ``database_manager`` is not the newest release of its window (a graph is
                the snapshot of the releases ``first_release`` to ``snapshot_release``; it is built from the newest of
                them), or if its genome assembly is not the organism's default primary assembly of that release
                (graphs are built on the default primary assembly only; the releases of the other assemblies are in
                the graph through it).
        """
        # Instance attributes
        self.log = logging.getLogger(__name__)
        # A graph is the snapshot of the manager's window and is built from the newest release in it.
        newest = max(database_manager.available_releases)
        if database_manager.release != newest:
            release = database_manager.release
            raise InvalidArgumentError(
                f"A graph is built from the newest release of the manager's window ({newest}), but this manager is "
                f"at release {release}. For a graph whose snapshot is release {release}, end the window there: "
                f"`API.build(organism, snapshot_release={release}, local_repository)` or "
                f"`DatabaseManager(..., snapshot_release={release})`. A graph of a later snapshot also converts to and "
                f"from release {release} (`to_release`/`from_release` of `API.convert`)."
            )
        default = organisms.primary_assembly(database_manager.organism, int(database_manager.release))
        if int(database_manager.assembly) != int(default):
            raise InvalidArgumentError(
                f"A graph is built on the organism's default primary assembly of its snapshot release, which is "
                f"{default} for {database_manager.organism} at release {int(database_manager.release)}; this manager reads the "
                f"tables of assembly {int(database_manager.assembly)}. The graph holds the releases of every assembly "
                f"of the organism: build it with `API.build(organism, snapshot_release, local_repository)`."
            )

        self.database_manager = database_manager
        self._managers: dict[tuple, DatabaseManager] = {}
        self._ids_cache: dict[tuple, dict[str, Any]] = {}
        self._table_counts: Counter = Counter()

    # ------------------------------------------------------------------ what the graph covers

    def graph_releases(self) -> list[int]:
        """Return the releases the graph covers: every release of the organism in the window, up to the snapshot.

        The snapshot is the release of the manager, the newest release of the primary assembly in the window.

        Returns:
            list[int]: Releases, ascending.
        """
        snapshot = int(self.database_manager.release)
        return sorted(int(r) for r in self.database_manager.available_releases_all_assemblies if int(r) <= snapshot)

    def _form_manager(self, form: str) -> DatabaseManager:
        key = (form, None, None)
        if key not in self._managers:
            self._managers[key] = self.database_manager.change_form(form)
        return self._managers[key]

    def _backbone_manager(self, form: str, release: int) -> DatabaseManager:
        """Return the manager of the assembly that serves ``release`` in the backbone.

        A release is taken from the primary assembly of the graph where that assembly has it, and otherwise from
        the organism's preferred assembly among those that have it.
        """
        key = (form, int(release), "backbone")
        if key not in self._managers:
            base = self._form_manager(form)
            if int(release) in {int(r) for r in self.database_manager.available_releases}:
                self._managers[key] = base.change_release(int(release))
            else:
                self._managers[key] = base.change_release_auto_assembly(int(release))
        return self._managers[key]

    def _assembly_manager(self, form: str, assembly: int, release: int | None) -> DatabaseManager:
        """Return the manager of one assembly at one of its releases (``None``: its newest release in the window)."""
        key = (form, None if release is None else int(release), int(assembly))
        if key not in self._managers:
            if release is None:
                self._managers[key] = self._form_manager(form).change_assembly(
                    int(assembly), last_possible_release=True
                )
            else:
                self._managers[key] = self._assembly_manager(form, assembly, None).change_release(int(release))
        return self._managers[key]

    def _source_tables_fingerprint(self, assembly: int) -> dict[str, Any]:
        """Return what identifies the tables stored for one assembly when the graph was built.

        The content digest of every table file of the assembly in the local repository
        (:py:meth:`DatabaseManager.table_digests`) is folded into one fingerprint, so that two graphs can be told to
        come from the same tables, or not, without keeping the tables' names in every graph file.

        Args:
            assembly: Assembly key of the organism.

        Returns:
            dict[str, Any]: ``{"tables": number of table files, "fingerprint": SHA-256 over their names and digests}``.
        """
        digests = self._assembly_manager(ids.BACKBONE_FORM, assembly, None).table_digests()
        folded = hashlib.sha256()
        for name, digest in sorted(digests.items()):
            folded.update(f"{name}\t{digest}\n".encode())
        return {"tables": len(digests), "fingerprint": folded.hexdigest()}

    def _assembly_releases(self) -> dict[int, list[int]]:
        """Return, for every assembly of the organism that has releases in the graph, those releases."""
        releases = set(self.graph_releases())
        table = organisms.ORGANISM_TABLE[self.database_manager.organism]
        result: dict[int, list[int]] = {}
        for assembly in organisms.assemblies_by_priority(self.database_manager.organism):
            first, last = table[assembly]["ReleaseRange"]
            if not any(first <= r and (last is None or r <= last) for r in releases):
                continue
            if int(assembly) == int(self.database_manager.assembly):
                found = [int(r) for r in self.database_manager.available_releases]
            else:
                try:
                    found = [
                        int(r) for r in self._assembly_manager(ids.BACKBONE_FORM, assembly, None).available_releases
                    ]
                except UnknownReleaseError as error:  # the one expected case: no core database in the window
                    # Every other error (a damaged table, a history that does not validate, Ensembl not reachable)
                    # rises: skipping the assembly would change the graph without an error.
                    self.log.warning(
                        f"Assembly {assembly} of {self.database_manager.organism} is skipped: no core database was found "
                        f"for it in the window ({error})."
                    )
                    continue
            found = sorted(r for r in found if r in releases)
            if found:
                result[int(assembly)] = found
        return result

    @staticmethod
    def _names(table: pd.DataFrame) -> list[str]:
        """Return the node names ``<id>.<version>`` of a two-column id table."""
        return [ids.join(i, j) for i, j in table.values]

    def _gene_ids(self, manager: DatabaseManager, save: bool) -> dict[str, Any]:
        """Return ``{stable id: version}`` of the gene table of a manager (read once per assembly and release)."""
        key = (manager.form, int(manager.assembly), int(manager.release))
        if key not in self._ids_cache:
            table = manager.get_db("ids") if save else manager.get_db("ids", save_after_calculation=False)
            self._table_counts[("ids", "tables")] += 1
            self._table_counts[("ids", "rows")] += len(table)
            self._ids_cache[key] = {i: j for i, j in table.values}
        return self._ids_cache[key]

    # ------------------------------------------------------------------ the build

    def construct_graph(
        self,
        session_details: bool = False,
        forms: list | None = None,
        all_external: bool = False,
    ) -> TheGraph:
        """Build the graph from the tables of the manager.

        The steps: the ids of every form at every release (:py:meth:`_collect_ids`), the gene history
        (:py:meth:`history`), the Ensembl nodes, the genes that lie outside the reference sequence
        (:py:meth:`_gene_regions`), the relations between the forms (:py:meth:`_add_relation_edges`), the
        identifiers of the external databases (:py:meth:`_add_external_edges`), and the versionless base gene ids
        (:py:meth:`_add_base_id_edges`). External identifiers that
        differ only in letter case are one node, named with the spelling that Ensembl gives at the latest release in
        which the identifier exists; how each database wrote it at each release is kept beside it
        (:py:meth:`_merge_spellings`, :py:meth:`idtrack.TheGraph.spelling`). The nodes and edges are then packed into
        the graph's store (:py:func:`_pack`), and the graph's record says what it is and what built it
        (:py:class:`idtrack.graph.GraphRecord`).

        Args:
            session_details: If ``True``, the history report of the record also lists Ensembl's mapping sessions with
                their database names, assemblies and dates (read from the full id-history table). The graph itself is
                the same.
            forms: The forms to include, a list of 'gene', 'transcript', 'translation' that contains 'gene'. If
                ``None``, all forms of the manager.
            all_external: If ``True``, every external database of Ensembl is included, not only those selected in the
                organism's externals YAML file. The graph becomes much larger and conversions slower.

        Returns:
            The frozen graph, with its record (:py:attr:`idtrack.TheGraph.record`).

        Raises:
            ValueError: If the Ensembl ids of the organism carry no versions in some release (such organisms are
                not supported; raised by the manager when the ids are read, before anything else is done), or if
                the tables contradict each other (an edge to a node that does not exist, an external table that
                names an Ensembl id which the primary assembly does not have).
            HistoryValidationError: If a mapping session cannot be reconciled with the id tables.
        """
        dm = self.database_manager
        forms = list(dm.available_form_of_interests if not forms else forms)
        if ids.BACKBONE_FORM not in forms:
            raise InvalidArgumentError(f"The forms of a graph must include {ids.BACKBONE_FORM!r}, got {forms}.")
        primary = int(dm.assembly)
        releases = self.graph_releases()

        # Check the externals YAML before the (long) history build: a missing file stops here with a message that
        # names the file to create, and releases without an entry are announced once.
        externals_releases = self._releases_with_externals()
        names_of_form = self._collect_ids(forms, releases)

        nodes = _Nodes()
        history_edges, history_report = self.history(
            self._form_manager(ids.BACKBONE_FORM), session_details=session_details
        )
        self._add_ensembl_nodes(nodes, history_edges, names_of_form, forms)
        non_reference, places = self._gene_regions(releases, nodes.number, nodes.types)
        serving_assembly = {int(r): int(self._backbone_manager(ids.BACKBONE_FORM, r).assembly) for r in releases}
        chains, mappings = self._assembly_chains(serving_assembly)
        carried = _carried_places(places, non_reference, chains, serving_assembly)
        exceptions = self._assembly_exceptions(releases, chains, serving_assembly)

        edges = _Edges()
        relation_pairs = _relation_pairs(forms)
        self._add_relation_edges(nodes, edges, relation_pairs, forms, releases)

        # The gene ids of every assembly and release: the base ids they imply are Ensembl ids, never external ones.
        assembly_releases = self._assembly_releases()
        gene_tables: dict[tuple[int, int], dict[str, Any]] = {}
        reserved: set[str] = set()
        for assembly, its_releases in assembly_releases.items():
            for release in its_releases:
                manager = self._assembly_manager(ids.BACKBONE_FORM, assembly, release)
                gene_tables[(assembly, release)] = self._gene_ids(manager, save=True)
                reserved.update(gene_tables[(assembly, release)])

        misplaced, label_names, versions, shown = self._add_external_edges(
            nodes, edges, forms, releases, externals_releases, reserved, all_external
        )
        left_out, unlinked, several = _links_of_one_gene_only(
            nodes,
            edges,
            dm.one_gene_only_names(),
            _gene_counter(
                nodes,
                places,
                non_reference,
                history_edges,
                organisms.pseudoautosomal_repeats(dm.organism),
                carried,
                _exception_regions(exceptions),
            ),
        )
        renamed = dm.renamed_label_names()  # read before the tables are fingerprinted (it reads `external_db`)
        self._add_base_id_edges(nodes, edges, gene_tables, primary)

        # One node per external identifier, whatever its letter case; the canonical spelling is its name.
        merge_into, spellings_by_number = self._merge_spellings(
            nodes.names, nodes.types, edges.groups, serving_assembly
        )
        # The versions of the accessions and the display names of the genes, per link that the graph keeps.
        version_marks = _link_marks(versions, edges, merge_into, serving_assembly)
        display_marks = _link_marks(shown, edges, merge_into, serving_assembly)
        del versions, shown

        self.log.info("Edges are being put together.")
        store, connections, final_of = _pack(nodes, edges, history_edges, merge_into, unlinked)
        store.spellings = {
            (int(final_of[i]), int(final_of[target]), db): entry
            for (i, target, db), entry in spellings_by_number.items()
        }
        store.non_reference = {int(final_of[i]): ReleaseSet.from_mask(mask) for i, mask in non_reference.items()}
        store.places = _Places.of({(int(final_of[key[0]]), *key[1:]): mask for key, mask in places.items()})
        store.carried = _Places.of({(int(final_of[key[0]]), *key[1:]): mask for key, mask in carried.items()})
        store.exceptions = exceptions
        store.versions = _LinkMarks.of(
            {(int(final_of[a]), int(final_of[b]), db, value): mask for (a, b, db, value), mask in version_marks.items()}
        )
        store.displays = _LinkMarks.of(
            {(int(final_of[a]), int(final_of[b]), db, value): mask for (a, b, db, value), mask in display_marks.items()}
        )
        del version_marks, display_marks
        store.build_name_indices()
        collisions = store.lower_collisions()
        if collisions:
            raise IntegrityError(f"Node names that differ only in letter case remain in the graph: {collisions[:5]}.")

        releases_per_assembly = self._check_assembly_ranges(connections, releases)
        counts = {
            "nodes": store.n,
            "history_edges": int(np.count_nonzero(store.e_kind == _KIND_HISTORY)),
            "cross_reference_edges": int(np.count_nonzero(store.e_kind == _KIND_XREF)),
            "connections": len(connections),
            "merged_spellings": len(merge_into),
            "misplaced_external_entries": len(misplaced),
            "non_reference_genes": len(non_reference),
            "places": len(store.places),
            "carried_places": len(store.carried),
            "assembly_exceptions": len(store.exceptions),
            "accession_versions": len(store.versions),
            "display_names": len(store.displays),
            "links_of_entries_of_several_genes": left_out,
        }
        record = GraphRecord(
            organism=dm.organism,
            primary_assembly=primary,
            first_release=int(dm.first_release),
            window_end=None if np.isinf(dm.snapshot_release) else int(dm.snapshot_release),
            snapshot_release=int(dm.release),
            releases=tuple(releases),
            serving_assembly=serving_assembly,
            releases_per_assembly={a: tuple(r) for a, r in releases_per_assembly.items()},
            forms=tuple(forms),
            session_details=bool(session_details),
            all_external=bool(all_external),
            externals_fingerprint=self._externals_fingerprint(forms),
            organism_table_fingerprint=self._organism_table_fingerprint(),
            externals_settings={int(r): int(s) for r, s in externals_releases.items()},
            external_databases={},
            connection_form={},
            label_names={db: name for db, name in sorted(dm.label_names().items()) if name in label_names},
            renamed_databases={},
            misplaced_external_entries=tuple(sorted(misplaced)),
            entries_of_several_genes=tuple(sorted(several)),
            database_names=self._database_names(),
            assembly_mappings=mappings,
            package_version=build_version(),
            package_commit=package_commit(),
            build_date=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            tables={
                kind: {"tables": self._table_counts[(kind, "tables")], "rows": self._table_counts[(kind, "rows")]}
                for kind in ("ids", "relationcurrent", "external", "idhistory", "generegion", "assemblyexception")
            },
            source_tables={
                str(assembly): self._source_tables_fingerprint(assembly) for assembly in sorted(assembly_releases)
            },
            counts=counts,
            history=history_report,
            content_digest=_content_digest(store),
        )
        graph = TheGraph._from_store(store, record)
        graph.calculate_caches()
        # The databases of the graph, per assembly, and the form each attaches its ids to: facts of the record.
        graph._with_record(
            dataclasses.replace(
                record,
                external_databases={
                    int(a): tuple(sorted(names))
                    for a, names in sorted(graph.available_external_databases_assembly.items())
                },
                connection_form=dict(sorted(graph.external_database_connection_form.items())),
                label_names={
                    db: name
                    for db, name in sorted(dm.label_names().items())
                    if name in label_names and name in graph.available_external_databases
                },
                renamed_databases={
                    old: new
                    for old, new in renamed.items()
                    if new in graph.available_external_databases and old not in graph.available_external_databases
                },
            )
        )
        return graph

    def _collect_ids(self, forms: list[str], releases: list[int]) -> dict[str, set[str]]:
        """Read the ids of every form at every release, before anything else is done.

        An organism whose Ensembl ids carry no versions is refused here, before any work: the manager does not serve
        the id table of such a release and says so. The gene ids are kept in memory for the history
        (:py:meth:`_gene_ids`); the ids of the other forms are returned as node names.

        Args:
            forms: The forms of the graph.
            releases: The releases of the graph.

        Returns:
            dict[str, set[str]]: ``{form: node names}`` for every form; empty for the gene form.
        """
        names_of_form: dict[str, set[str]] = {}
        for form in forms:
            self.log.info(f"Ids are being collected: {form}")
            collected: set[str] = set()
            for release in releases:
                manager = self._backbone_manager(form, release)
                if form == ids.BACKBONE_FORM:
                    self._gene_ids(manager, save=False)  # kept in memory for the history
                    continue
                table = manager.get_db("ids", save_after_calculation=False)
                self._table_counts[("ids", "tables")] += 1
                self._table_counts[("ids", "rows")] += len(table)
                collected.update(self._names(table))
            names_of_form[form] = collected
        return names_of_form

    def _add_ensembl_nodes(
        self, nodes: "_Nodes", history_edges: list[tuple], names_of_form: dict[str, set[str]], forms: list[str]
    ) -> None:
        """Add the backbone genes of the history, then the transcripts and translations, each form in sorted order.

        An id that two forms share (Ensembl has protein ids that start with ``ENST``, and some id is a transcript in
        one release and a translation in another, e.g. ``ENST00000515292.1``) becomes a node of the later form; this
        is announced.
        """
        for name in sorted({n for edge in history_edges for n in edge[:2]}):
            nodes.add(name, ids.ENSEMBL_TYPE[ids.BACKBONE_FORM])
        for form in forms:
            if form == ids.BACKBONE_FORM:
                continue
            found = names_of_form.pop(form)
            clash = sorted(n for n in found if n in nodes.number)
            if clash:
                self.log.warning(
                    f"Intersecting Ensembl nodes: {len(clash)} names of earlier forms are also {form} ids and "
                    f"become {form} nodes: {', '.join(clash[:20])!r}."
                )
                for name in clash:
                    nodes.types[nodes.number[name]] = ids.ENSEMBL_TYPE[form]
            for name in sorted(found):
                nodes.add(name, ids.ENSEMBL_TYPE[form])
        for name in nodes.names:
            ids.split(name)  # every Ensembl node is `<id>.<version>`

    def _add_relation_edges(
        self,
        nodes: "_Nodes",
        edges: "_Edges",
        relation_pairs: list[tuple[str, str]],
        forms: list[str],
        releases: list[int],
    ) -> None:
        """Add the relations between the forms of every release: transcript to gene, translation to transcript.

        Raises:
            IntegrityError: If a relation names an id that is not a node.
        """
        if not relation_pairs:
            return
        self.log.info("Establishing connection between different forms.")
        for release in releases:
            manager = self._backbone_manager(forms[0], release)
            table = manager.get_db("relationcurrent", save_after_calculation=manager.store_raw_always)
            self._table_counts[("relationcurrent", "tables")] += 1
            self._table_counts[("relationcurrent", "rows")] += len(table)
            for lower, upper in relation_pairs:
                pairs = self._pairs(table, lower, upper)
                sources, targets = [], []
                for first, second in pairs:
                    source, target = nodes.number.get(first), nodes.number.get(second)
                    if source is None or target is None:
                        raise IntegrityError(
                            f"Missing graph node(s) while adding edge: {lower}={first!r} → {upper}={second!r} "
                            f"(release {release})."
                        )
                    sources.append(source)
                    targets.append(target)
                edges.add(sources, targets, ids.ENSEMBL_TYPE[lower], int(manager.assembly), release)

    def _add_external_edges(
        self,
        nodes: "_Nodes",
        edges: "_Edges",
        forms: list[str],
        releases: list[int],
        externals_releases: dict[int, int],
        reserved: set[str],
        all_external: bool,
    ) -> tuple[set[str], set[str], dict[tuple[int, int, str, int], dict[int, int]], dict[tuple, dict[int, int]]]:
        """Add the identifiers of the external databases of every form and release, with their edges.

        An external id that is an Ensembl node, or the base id of a gene, stays that Ensembl id: the entry is
        recorded as misplaced and skipped. Two facts of a link are collected beside its edge: the version of the
        accession that the link attaches at each release (``xref.version``, the column ``version`` of the table:
        :py:meth:`idtrack.build.DatabaseManager._external_rows`), and whether the identifier is the gene's display name
        at the release (``is_display``: :py:meth:`idtrack.build.DatabaseManager._with_display_names`).

        Args:
            nodes: The nodes so far.
            edges: The edges so far.
            forms: The forms of the graph.
            releases: The releases of the graph.
            externals_releases: ``{release: the release whose externals settings it takes}``; a release after the last
                one the YAML lists takes the settings of that one.
            reserved: The stable ids of the genes of every assembly and release.
            all_external: See :py:meth:`construct_graph`.

        Returns:
            The misplaced external ids; the names of the databases whose rows are labels (the ``is_label`` rows of
            :py:meth:`idtrack.build.DatabaseManager.create_external_db`); ``{(external node, Ensembl node, database,
            assembly): {version: release mask}}`` of the links that attach a version; and ``{(external node, gene,
            database, assembly): {0: release mask}}`` of the links that are a gene's display name.

        Raises:
            BuildError: If the table names an unknown assembly, or an Ensembl id that no id table of the graph's
                releases has (the graph follows the ids of the assembly that serves each release; a frozen assembly
                that Ensembl keeps beside the primary one holds no id that the primary line lacks).
        """
        dm = self.database_manager
        before_externals = len(nodes.names)
        organism_assemblies = organisms.assemblies_by_priority(dm.organism)
        misplaced: set[str] = set()
        label_names: set[str] = set()
        versions: dict[tuple[int, int, str, int], dict[int, int]] = {}
        shown: dict[tuple, dict[int, int]] = {}
        number = nodes.number
        for form in forms:
            self.log.info(f"Edges between external IDs to Ensembl IDs is being added for {form!r}.")
            for release in releases:
                if release not in externals_releases:
                    # Announced once by `_releases_with_externals`; the Ensembl-only history graph still covers it.
                    continue
                manager = self._backbone_manager(form, release)
                table = manager.create_external_all(all_external=all_external)
                if "is_label" in table.columns:
                    named = table.loc[table["is_label"].to_numpy(dtype=bool), "name_db"].astype(str)
                    label_names.update(name for name in named.unique() if not name.startswith(ids.SYNONYM_PREFIX))
                self._table_counts[("external", "tables")] += 1
                self._table_counts[("external", "rows")] += len(table)
                if not len(table):
                    continue
                assemblies = [int(a) for a in table["assembly"].tolist()]
                unknown = sorted(set(assemblies) - set(organism_assemblies))
                if unknown:
                    raise BuildError(
                        f"Unknown assembly {unknown[0]!r} in the external table of release {release}. "
                        f"Assemblies of {dm.organism}: {sorted(organism_assemblies)}."
                    )
                valid = np.asarray(assemblies) != 0
                for column in ("graph_id", "id_db", "name_db", "release"):
                    cells = table[column]
                    valid &= (cells.notna() & (cells != "") & (cells != 0)).to_numpy()
                table = table[valid]
                grouped: dict[tuple[str, int, int], tuple[list[int], list[int]]] = {}
                for ensembl_id, external_id, row_release, database, assembly, version, display in zip(
                    table["graph_id"].tolist(),
                    table["id_db"].astype(str).str.strip().tolist(),
                    table["release"].tolist(),
                    table["name_db"].tolist(),
                    [int(a) for a in table["assembly"].tolist()],
                    table["version"].tolist(),
                    table["is_display"].tolist(),
                ):
                    if not external_id:
                        continue
                    target = number.get(ensembl_id)
                    if target is None or target >= before_externals:
                        raise BuildError(
                            f"The external table of release {release} names the Ensembl id {ensembl_id!r} on assembly "
                            f"{assembly}, but no id table of the graph's releases has it: the graph follows the ids of "
                            f"the assembly that serves each release ({dm.organism}, primary assembly "
                            f"{int(dm.assembly)})."
                        )
                    source = number.get(external_id)
                    if (source is not None and source < before_externals) or external_id in reserved:
                        # Some external databases list an Ensembl id as their identifier. A name that is an Ensembl
                        # node, or the base id of a gene, stays that Ensembl id; the entry is recorded and skipped.
                        misplaced.add(external_id)
                        continue
                    if source is None:
                        source = nodes.add(external_id, ids.EXTERNAL)
                    pair = grouped.get((database, assembly, int(row_release)))
                    if pair is None:
                        pair = grouped[(database, assembly, int(row_release))] = ([], [])
                    pair[0].append(source)
                    pair[1].append(target)
                    bit = 1 << int(row_release)
                    if version:
                        per = versions.setdefault((source, target, database, assembly), {})
                        per[int(version)] = per.get(int(version), 0) | bit
                    if display:
                        per = shown.setdefault((source, target, database, assembly), {})
                        per[0] = per.get(0, 0) | bit
                for (database, assembly, row_release), (sources, targets) in grouped.items():
                    edges.add(sources, targets, database, assembly, row_release)
        if misplaced:
            self.log.warning(f"Misplaced external entry: {len(misplaced)}.")
        return misplaced, label_names, versions, shown

    def _add_base_id_edges(
        self,
        nodes: "_Nodes",
        edges: "_Edges",
        gene_tables: dict[tuple[int, int], dict[str, Any]],
        primary: int,
    ) -> None:
        """Link the versionless base gene id to its versioned genes, for every assembly and release of the graph.

        Raises:
            BuildError: If a gene of the primary assembly is not a node, or a base id is the name of another kind of
                node.
        """
        self.log.info("Versionless Ensembl IDs are being connected.")
        base_type = ids.BASE_ENSEMBL_TYPE[ids.BACKBONE_FORM]
        number, types = nodes.number, nodes.types
        for (assembly, release), table in gene_tables.items():
            sources, targets = [], []
            for stable_id, version in table.items():
                name = ids.join(stable_id, version)
                target = number.get(name)
                if target is None:
                    if assembly == primary:
                        raise BuildError(
                            f"The gene {name!r} of the primary assembly {assembly} at release {release} is not a "
                            "node of the graph."
                        )
                    continue  # an id that only another assembly has: not a node of the graph
                source = number.get(stable_id)
                if source is None:
                    source = nodes.add(stable_id, base_type)
                elif types[source] != base_type:
                    raise BuildError(
                        f"The base gene id {stable_id!r} is already a node of type {types[source]!r}; a base id "
                        "must not be the name of another kind of node."
                    )
                sources.append(source)
                targets.append(target)
            edges.add(sources, targets, base_type, assembly, release)

    def _check_assembly_ranges(self, connections: list[Connection], releases: list[int]) -> dict[int, list[int]]:
        """Check that every assembly's edges lie inside the releases the organism table gives the assembly.

        Returns:
            dict[int, list[int]]: ``{assembly: releases}`` found on the edges, ascending.

        Raises:
            BuildError: If an assembly's edges carry releases outside its range or outside the graph.
        """
        table = organisms.ORGANISM_TABLE[self.database_manager.organism]
        releases_on_edges: dict[int, int] = {}
        for connection in connections:
            for _database, assembly, release_set in connection.items:
                releases_on_edges[assembly] = releases_on_edges.get(assembly, 0) | release_set.mask
        releases_per_assembly = {a: [r for r in releases if m >> r & 1] for a, m in sorted(releases_on_edges.items())}
        for assembly, found_releases in releases_per_assembly.items():
            lowest, highest = table[assembly]["ReleaseRange"] if assembly in table else (None, None)
            outside = [
                r for r in found_releases if lowest is None or r < lowest or (highest is not None and r > highest)
            ]
            if outside or releases_on_edges[assembly] != sum(1 << r for r in found_releases):
                raise BuildError(
                    f"Edges of assembly {assembly} carry releases outside the assembly's range in the organism "
                    f"table ({lowest} to {highest}) or outside the graph: {outside}."
                )
        return releases_per_assembly

    def _gene_regions(
        self, releases: list[int], number: dict[str, int], types: list[str]
    ) -> tuple[dict[int, int], dict[tuple[int, int, str, int, int, int, str], int]]:
        """Collect where every gene lies, and the genes that lie outside the reference sequence, release by release.

        For every release the ``generegion`` table of the assembly that serves the release in the backbone is read
        (:py:meth:`idtrack.DatabaseManager.create_gene_region`). It gives the place of every gene (sequence region,
        start, end, strand, and what the region is: :py:attr:`idtrack.graph.types.Place.kind`;
        :py:meth:`idtrack.TheGraph.places`) and marks the genes on alternative loci, on assembly patches and LRG
        records. These are nodes of the backbone like every gene; the mark lets a conversion tell a
        gene of the reference sequence from its copies (:py:meth:`idtrack.TheGraph.is_non_reference`).

        Args:
            releases: The releases of the graph.
            number: Node number by node name; holds the backbone genes.
            types: Node type by node number.

        Returns:
            ``{node number: release bit mask}`` for the marked genes only, and ``{(node number, assembly, region,
            start, end, strand, kind): release bit mask}`` for every gene.

        Raises:
            BuildError: If a gene of a release is not a node of the graph.
        """
        self.log.info("Places of the genes and genes outside the reference sequence are being collected.")
        backbone_type = ids.ENSEMBL_TYPE[ids.BACKBONE_FORM]
        marked: dict[int, int] = {}
        places: dict[tuple[int, int, str, int, int, int, str], int] = {}
        for release in releases:
            manager = self._backbone_manager(ids.BACKBONE_FORM, release)
            assembly = int(manager.assembly)
            table = manager.get_db("generegion")
            self._table_counts[("generegion", "tables")] += 1
            self._table_counts[("generegion", "rows")] += len(table)
            bit = 1 << int(release)
            for name, beside, region, start, end, strand, kind in zip(
                self._names(table[["gene_stable_id", "gene_version"]]),
                table["non_reference"].to_numpy(dtype=bool).tolist(),
                table["seq_region"].tolist(),
                table["start"].tolist(),
                table["end"].tolist(),
                table["strand"].tolist(),
                table["kind"].tolist(),
            ):
                node = number.get(name)
                if node is None:
                    raise BuildError(f"The gene {name!r} of release {release} is not a node of the graph.")
                if types[node] != backbone_type:  # a name that another form took over (announced above)
                    continue
                key = (node, assembly, str(region), int(start), int(end), int(strand), str(kind))
                places[key] = places.get(key, 0) | bit
                if beside:
                    marked[node] = marked.get(node, 0) | bit
        return marked, places

    def _assembly_chains(
        self, serving_assembly: dict[int, int]
    ) -> tuple[dict[tuple[int, int], AssemblyChain], dict[str, dict[str, Any]]]:
        """Read Ensembl's mappings between the assemblies that serve the graph's releases, where it publishes them.

        For every ordered pair of those assemblies (the earlier one by the releases it serves first), Ensembl's chain
        file of the pair (:py:meth:`idtrack.DatabaseManager.assembly_chain`: downloaded once into the local repository
        and kept).

        Args:
            serving_assembly: ``{release: assembly that serves it}``.

        Returns:
            tuple: ``{(earlier assembly, later assembly): chain}`` for the pairs Ensembl maps, and what the graph's
            record says of them (``{file name: {"from", "to", "release", "sha256"}}``).
        """
        dm = self.database_manager
        order = _assembly_order(serving_assembly)
        chains: dict[tuple[int, int], AssemblyChain] = {}
        mappings: dict[str, dict[str, Any]] = {}
        for i, earlier in enumerate(order):
            for later in order[i + 1 :]:
                found = dm.assembly_chain(earlier, later)
                if found is None:
                    continue
                blocks, info = found
                chains[(earlier, later)] = AssemblyChain(blocks)
                mappings[info["file"]] = {k: info[k] for k in ("from", "to", "release", "sha256")}
        self.log.info(f"Ensembl's assembly mappings of {dm.organism}: {sorted(mappings) or 'none'}.")
        return chains, mappings

    def _database_names(self) -> dict[str, dict[str, tuple[tuple[int, int], ...]]]:
        """Return, per selected database, the names of Ensembl it holds with their releases (decision 104)."""
        dm = self.database_manager
        prefix = ids.SYNONYM_PREFIX
        selected = {n[len(prefix) :] if n.startswith(prefix) else n for n in dm.external_inst.selected_names()}
        held = dm.database_filing().database_names(selected)
        return {
            database: {name: ReleaseSet.of(releases).ranges() for name, releases in names.items()}
            for database, names in held.items()
        }

    def _assembly_exceptions(
        self, releases: list[int], chains: dict[tuple[int, int], AssemblyChain], serving_assembly: dict[int, int]
    ) -> tuple[AssemblyException, ...]:
        """Collect Ensembl's assembly exceptions of every release, on the assembly that serves it.

        Each row of :py:meth:`idtrack.DatabaseManager.create_assembly_exceptions` with the releases that list it, and
        its reference region carried onto the later assemblies that Ensembl maps it onto.

        Args:
            releases: The releases of the graph.
            chains: Ensembl's mappings of the assemblies (:py:meth:`_assembly_chains`).
            serving_assembly: ``{release: assembly that serves it}``.

        Returns:
            tuple[AssemblyException, ...]: The rows, sorted.
        """
        self.log.info("Ensembl's assembly exceptions are being collected.")
        found: dict[tuple, int] = {}
        for release in releases:
            manager = self._backbone_manager(ids.BACKBONE_FORM, release)
            assembly = int(manager.assembly)
            table = manager.get_db("assemblyexception")
            self._table_counts[("assemblyexception", "tables")] += 1
            self._table_counts[("assemblyexception", "rows")] += len(table)
            for row in table.itertuples(index=False):
                key = (
                    assembly,
                    str(row.seq_region),
                    int(row.start),
                    int(row.end),
                    str(row.type),
                    str(row.ref_region),
                    int(row.ref_start),
                    int(row.ref_end),
                    str(row.ref_kind),
                )
                found[key] = found.get(key, 0) | (1 << int(release))
        rows = []
        for key, mask in sorted(found.items()):
            assembly, ref_region, ref_start, ref_end = key[0], key[5], key[6], key[7]
            onto = tuple(
                sorted(
                    {
                        (later, region, start, end)
                        for later in _later_assemblies(assembly, serving_assembly)
                        for region, start, end, _ in _carry(
                            chains, (ref_region, ref_start, ref_end, 1), assembly, later, serving_assembly
                        )
                    }
                )
            )
            rows.append(AssemblyException(*key, ReleaseSet.from_mask(mask), onto))
        return tuple(rows)

    @staticmethod
    def _pairs(table: pd.DataFrame, lower: str, upper: str) -> list[tuple[str, str]]:
        """Return the distinct ``(lower form id, upper form id)`` pairs of a relation table (both present)."""
        return sorted(
            {(a, b) for a, b in zip(table[lower].tolist(), table[upper].tolist()) if _present(a) and _present(b)}
        )

    def _merge_spellings(
        self,
        names: list[str],
        types: list[str],
        edges: dict[tuple[str, int], dict[int, int]],
        serving_assembly: dict[int, int],
    ) -> tuple[dict[int, int], dict[tuple[int, int, str], tuple[str, tuple[tuple[int, str], ...]]]]:
        """Choose one node for the external identifiers that are equal in lower case, and keep how each link spells it.

        A release belongs to the assembly that serves it in the backbone (``serving_assembly``). What another
        assembly says at that release does not count while a serving assembly says anything: human GRCh37 is
        frozen and still listed at every release that GRCh38 serves, with the spellings of the time it was frozen.
        Only an identifier (or a link of it) that no serving assembly has at any release is judged by all its
        assemblies.

        The node keeps the spelling that Ensembl gives at the latest release in which the identifier exists. Where
        several spellings exist at that release, the one carried by more (database, Ensembl id) links at that
        release wins, then the one with more upper-case letters, then the first in sort order.

        The spelling belongs to the link: for every (Ensembl id, database) that the identifier is linked to, the
        spelling of each release is kept (``Tc``, a synonym of one gene, and ``tc``, a synonym of Noto, are one node
        whose links say which gene carries which spelling), so that a conversion reads the identifier as typed only
        through the links spelt so (:py:meth:`idtrack.TheGraph.spelling`). Two spellings of one value that one
        database gives one Ensembl id at one release are one value: the one with more upper-case letters is kept,
        then the first in sort order.

        Args:
            names: Node names by provisional number.
            types: Node types by provisional number.
            edges: The cross-reference edges collected so far.
            serving_assembly: The assembly that serves each release of the graph.

        Returns:
            ``{number of a merged spelling: number of the node that stays}`` and, only for the links that spell the
            identifier otherwise than the node is named at some release, ``{(number of the node that stays, number of
            the Ensembl id, database): (spelling at the link's latest release, ((release mask, spelling), ...))}``;
            the pairs name the releases at which the link spells it otherwise than at its latest release.

        Raises:
            BuildError: If a name that differs only in case from another one is not an external identifier.
        """
        self.log.info("Synonymous external nodes are being merged into one.")
        by_lower: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            by_lower.setdefault(name.lower(), []).append(i)
        groups = [group for group in by_lower.values() if len(group) > 1]
        del by_lower
        if not groups:
            return {}, {}
        colliding = {i for group in groups for i in group}
        for i in sorted(colliding):
            if types[i] != ids.EXTERNAL:
                raise BuildError(
                    f"The node {names[i]!r} of type {types[i]!r} differs only in letter case from another node; "
                    "only external identifiers can be merged."
                )
        served = serving_line(serving_assembly)

        # Per spelling, (Ensembl id, database) and assembly, the release mask of the link.
        links: dict[int, dict[tuple[int, str], list[tuple[int, int]]]] = {i: {} for i in colliding}
        for (database, assembly), group in edges.items():
            for key, mask in group.items():
                source = key >> 32
                if source in colliding:
                    links[source].setdefault((key & 0xFFFFFFFF, database), []).append((assembly, mask))

        def on_line(pairs: list[tuple[int, int]]) -> int:
            union = 0
            for assembly, mask in pairs:
                if assembly in served:
                    union |= mask & served[assembly].mask
            return union

        def anywhere(pairs: list[tuple[int, int]]) -> int:
            union = 0
            for _assembly, mask in pairs:
                union |= mask
            return union

        def upper(i: int) -> int:
            return sum(1 for c in names[i] if c.isupper())

        merge_into: dict[int, int] = {}
        spellings: dict[tuple[int, int, str], tuple[str, tuple[tuple[int, str], ...]]] = {}
        for same_lower in groups:
            ordered = sorted(same_lower, key=lambda i: names[i])
            judged = on_line if any(on_line(p) for i in ordered for p in links[i].values()) else anywhere
            # every (Ensembl id, database, assembly) link counts once, as it holds on the releases judged
            masks = {i: [judged([pair]) for p in links[i].values() for pair in p] for i in ordered}
            masks = {i: [m for m in ms if m] for i, ms in masks.items()}
            ranks = {i: (*_latest_and_count(masks[i]), upper(i)) for i in ordered}
            winner = max(ordered, key=ranks.__getitem__)  # the first of equals: first in sort order
            for i in ordered:
                if i != winner:
                    merge_into[i] = winner
            for link in sorted({key for i in ordered for key in links[i]}):
                per_spelling = {i: judged(links[i][link]) for i in ordered if link in links[i]}
                if not any(per_spelling.values()):  # the link holds on no serving assembly: judged where it is listed
                    per_spelling = {i: anywhere(links[i][link]) for i in ordered if link in links[i]}
                per_spelling = {i: mask for i, mask in per_spelling.items() if mask}
                union = 0
                for mask in per_spelling.values():
                    union |= mask
                chosen_of: dict[int, int] = {}
                for release in ReleaseSet.from_mask(union):
                    chosen = max((i for i, mask in per_spelling.items() if mask >> release & 1), key=upper)
                    chosen_of[chosen] = chosen_of.get(chosen, 0) | (1 << release)
                latest = max(chosen_of, key=lambda i: chosen_of[i].bit_length())
                otherwise = {i: mask for i, mask in chosen_of.items() if i != latest}
                if latest != winner or otherwise:
                    spellings[(winner, link[0], link[1])] = (
                        names[latest],
                        tuple(
                            (mask, names[i]) for i, mask in sorted(otherwise.items(), key=lambda item: names[item[0]])
                        ),
                    )
        self.log.info(f"Number of removed nodes in the process of merging synonymous nodes: {len(merge_into)}")
        return merge_into, spellings

    def _releases_with_externals(self) -> dict[int, int]:
        """Return the releases of the graph that the organism's externals YAML covers, warning once about the rest.

        A release that the YAML lists takes its own settings; a release after the last one it lists takes the settings
        of that last release (:py:meth:`idtrack.ExternalDatabases.settings_release`), announced once in the log and
        recorded in the graph's record. Releases of :py:meth:`graph_releases` before or between the listed ones get
        no external identifiers in the graph. That is announced here once, as a :py:class:`UserWarning` and in the
        log, with the way to fix it (:py:meth:`idtrack.ExternalDatabases.extend_modified_yaml`), instead of being
        skipped quietly release by release.

        Returns:
            dict[int, int]: ``{release whose external identifiers are added: release whose settings it takes}``.

        Raises:
            ExternalsError: If the organism has no externals YAML file at all (the message names the file to
                create and how).
        """
        external_inst = self.database_manager.external_inst
        covered = external_inst.covered_releases()
        releases = self.graph_releases()
        settings = {r: r for r in releases if r in covered}
        if covered:
            settings.update({r: max(covered) for r in releases if r > max(covered)})
            later = sorted(r for r in releases if r > max(covered))
            if later:
                self.log.warning(
                    f"The externals YAML of {self.database_manager.organism} lists releases up to {max(covered)}; releases "
                    f"{later} take its settings of release {max(covered)} (recorded in the graph's record)."
                )
        missing = [r for r in releases if r not in settings]
        if missing:
            path = external_inst._modified_yaml_path()
            if covered:
                msg = (
                    f"The externals YAML `{path}` has no entry for {self.database_manager.organism} Ensembl releases "
                    f"{missing}: the graph gets no external identifiers (symbols, NCBI gene, UniProt, ...) for these "
                    "releases. Extend the file with `ExternalDatabases.extend_modified_yaml` (tutorial 02) and "
                    "rebuild the graph."
                )
            else:
                msg = (
                    f"The externals YAML `{path}` has no entry for {self.database_manager.organism}: the graph is built "
                    "without external identifiers."
                )
            warnings.warn(msg, UserWarning, stacklevel=3)
            self.log.warning(msg)
        return dict(sorted(settings.items()))

    # ------------------------------------------------------------------ the gene history

    def history(
        self, database_manager: DatabaseManager, session_details: bool = False
    ) -> tuple[list[tuple], dict[str, Any]]:
        """Build the history edges of one form from the id tables of the releases and Ensembl's id-history table.

        The id tables say which ``<id>.<version>`` exists in which release (each release from the assembly that
        serves it in the backbone; :py:meth:`_timeline`). From them come the edges of an id with itself
        (:py:meth:`_self_history`): ``<id>.Void → <id>.<v>`` at its first appearance, ``<id>.<v> → <id>.<w>`` at a
        change of version, ``<id>.<v> → <id>.Retired`` when it disappears, ``<id>.Retired → <id>.<v>`` when it comes
        back, and a self-loop to infinity on every id of the last release (:py:meth:`_snapshot_loops`).

        Ensembl's ``stable_id_event`` table says which id passed into which other id (and which id was re-mapped
        onto itself with a score below 1, or onto itself in a session that also maps it to or from another id), per
        mapping session (:py:meth:`_events_and_weights`). The releases of a session are not taken from
        ``mapping_session``: an event from ``A`` to ``B`` belongs to the pair of consecutive releases at which ``A``
        exists in the first and ``B`` in the second. For every session the pair that most of its events fit is
        determined; it replaces the recorded pair where the two differ (Ensembl's pig session 106 to 107 is recorded
        as 106 to 104; sessions over a gap such as rhesus 50 to 52 belong to 51 to 52). An event whose old version did
        not exist at the session's old release, while its stable id did, is placed with the version the id had
        there; an event that does not fit its session's pair otherwise is placed at the nearest pair it fits; an
        event that fits no pair, or whose ids are in no id table, contradicts the tables and is left out
        (:py:meth:`_place_sessions`). All of it is counted in the report. The edges are put in a deterministic order,
        keyed and checked against the id tables (:py:func:`_key_and_validate`).

        Args:
            database_manager: The manager of the form, at the snapshot release.
            session_details: Read the full id-history table, whose session metadata then goes into the report
                (``True``), or the narrow one.

        Returns:
            The edges as ``(source, target, old_release, new_release, weight, key)`` tuples (``new_release`` is
            infinity on the self-loops of the last release), sorted, and the report for the record: the sessions
            with recorded and used release pairs and their counts, and the corrections.

        Raises:
            HistoryValidationError: If the events of a session fit no pair of consecutive releases, fit several
                pairs equally well without the recorded one among them, if an event fits two pairs equally near its
                session's pair, or if the id-history table does not reach back to the first release of the graph
                although Ensembl's ``mapping_session`` does (a table computed for another release window).
            ValueError: If an id of the last release is missing from the backbone, or if the ids of a release
                carry no versions (raised by the manager, see :py:meth:`construct_graph`).
        """
        self.log.info(f"History is being built: {database_manager.form}")
        form = database_manager.form
        releases = self.graph_releases()
        timeline, alive = self._timeline(form, releases)

        table = database_manager.get_db(table_key="idhistory" if session_details else "idhistory_narrow").copy()
        self._table_counts[("idhistory", "tables")] += 1
        self._table_counts[("idhistory", "rows")] += len(table)
        events, weights, session_columns = self._events_and_weights(table, releases, session_details)
        sessions = _sessions_of(events, alive, session_columns, session_details, table.columns)
        edges, loop_release, report_sessions, corrections, versions_placed = self._place_sessions(
            sessions, releases, form, timeline, alive
        )
        if form == ids.BACKBONE_FORM:
            self._check_history_reaches_back(database_manager, releases, table["old_release"])

        edges.extend(_self_history(timeline, releases, weights))
        edges.extend(self._snapshot_loops(database_manager, alive, loop_release, releases))
        keyed = _key_and_validate(edges, alive, {r: i for i, r in enumerate(releases)})
        report = {
            "sessions": report_sessions,
            "corrections": corrections,
            "events": int(sum(s["events"] for s in report_sessions)),
            "event_edges": int(sum(s["edges"] for s in report_sessions)),
            "events_of_ids_in_no_table": int(sum(s["absent"] for s in report_sessions)),
            "events_moved_to_another_pair": int(sum(s["moved"] for s in report_sessions)),
            "events_contradicting_tables": int(sum(s["contradicting"] for s in report_sessions)),
            "events_placed_with_the_version_of_the_release": versions_placed,
        }
        return keyed, report

    def _timeline(self, form: str, releases: list[int]) -> tuple[dict[str, list[tuple[int, Any]]], dict[str, int]]:
        """Return which id has which version in which release, and in which releases each ``<id>.<version>`` exists.

        Args:
            form: The form.
            releases: The releases of the graph.

        Returns:
            ``{stable id: [(position of the release, version), ...]}`` in release order, and ``{<id>.<version>: bit
            mask of the positions of the releases at which it exists}``.
        """
        timeline: dict[str, list[tuple[int, Any]]] = {}
        alive: dict[str, int] = {}
        for i, release in enumerate(releases):
            manager = self._backbone_manager(form, release)
            table = (
                self._gene_ids(manager, save=False)
                if form == ids.BACKBONE_FORM
                else {a: b for a, b in manager.get_db("ids", save_after_calculation=False).values}
            )
            for stable_id, version in table.items():
                timeline.setdefault(stable_id, []).append((i, version))
                name = ids.join(stable_id, version)
                alive[name] = alive.get(name, 0) | (1 << i)
        return timeline, alive

    def _events_and_weights(
        self, table: pd.DataFrame, releases: list[int], session_details: bool
    ) -> tuple[pd.DataFrame, dict[tuple, float], list[str]]:
        """Choose the rows of Ensembl's id-history table that are events, and the scores of version changes.

        An id that a session maps onto itself is an event of the history in two cases: the mapping is not perfect
        (a score below 1), or the same session also maps the id to or from another id. In the second case the self
        event is what says that the id lived on beside the other one: an id that exists on both sides of a session
        and is mapped only to other ids did not, and without the self event the two cannot be told apart. Scores of
        version changes are kept as weights; pairs that Ensembl scores twice are ignored (an annotation mistake).

        Args:
            table: The id-history table (changed in place: empty ids become missing, numbers become numeric).
            releases: The releases of the graph.
            session_details: Whether it is the full table (otherwise sessions are told apart by their release pair).

        Returns:
            The event rows, the weights ``{(old id, old version, new id, new version): score}``, and the columns that
            name a session.
        """
        for column in ("old_stable_id", "new_stable_id"):
            table[column] = table[column].replace("", np.nan)
        for column in ("old_version", "new_version", "score", "old_release", "new_release"):
            table[column] = pd.to_numeric(table[column], errors="coerce")
        usable = (
            table["old_stable_id"].notna()
            & table["new_stable_id"].notna()
            & (table["new_version"] != 0)  # remove 0 versions: like the ones ASMPATCHG00000000170.0
            & (table["old_version"] != 0)
            & (table["old_release"] >= releases[0])
        )
        same_id = table["new_stable_id"] == table["old_stable_id"]
        same_version = table["new_version"] == table["old_version"]
        session_columns = (
            ["mapping_session_id"]
            if session_details and "mapping_session_id" in table
            else ["old_release", "new_release"]
        )

        to_other_id = usable & ~same_id
        onto_itself = usable & same_id & same_version
        session_of_row = table[session_columns].fillna(-1)
        beside_others: set[tuple] = set()
        for id_column, version_column in (("old_stable_id", "old_version"), ("new_stable_id", "new_version")):
            beside_others.update(
                zip(
                    *(session_of_row.loc[to_other_id, c].tolist() for c in session_columns),
                    table.loc[to_other_id, id_column].tolist(),
                    table.loc[to_other_id, version_column].tolist(),
                )
            )
        with_other_events = pd.Series(
            [
                key in beside_others
                for key in zip(
                    *(session_of_row[c].tolist() for c in session_columns),
                    table["old_stable_id"].tolist(),
                    table["old_version"].tolist(),
                )
            ],
            index=table.index,
            dtype=bool,
        )
        is_event = to_other_id | (onto_itself & ((table["score"] < 1.0) | with_other_events))
        is_weight = usable & same_id & ~same_version & table["score"].notna()

        weights: dict[tuple, float] = {}
        twice: set[tuple] = set()
        weight_columns = ["old_stable_id", "old_version", "new_stable_id", "new_version", "score"]
        for a, av, b, bv, score in table.loc[is_weight, weight_columns].itertuples(index=False, name=None):
            key = (a, int(av), b, int(bv))
            if key in weights or key in twice:
                twice.add(key)
                weights.pop(key, None)
            else:
                weights[key] = float(score)
        if twice:
            self.log.warning(f"Edge weights ignored due to duplicate entries: {len(twice)}.")
        return table.loc[is_event], weights, session_columns

    def _place_sessions(
        self,
        sessions: dict[tuple, dict[str, Any]],
        releases: list[int],
        form: str,
        timeline: dict[str, list[tuple[int, Any]]],
        alive: dict[str, int],
    ) -> tuple[
        list[tuple], dict[str, tuple[int, int]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
    ]:
        """Give every mapping session the pair of consecutive releases its events fit, and make their edges.

        A session takes the pair that most of its events between different ids fit; where it ties, the recorded
        pair if it is among the best, and otherwise the history cannot be read from the tables (no such session
        exists in the graphs of the organisms idtrack ships, so no rule that would choose between the pairs is
        guessed). An event takes its session's pair. An event whose old version did not exist at the session's
        old release, while its stable id did and its new id exists at the session's new release, is placed with the
        version the id had at the old release (Ensembl's event names the gene; the version field is its slip); it is
        counted and listed. An event that fits only other pairs is placed at the nearest of them
        (none does in the five graphs; two equally near ones are refused as above). An event that fits no pair is
        left out and counted.

        Args:
            sessions: The sessions of :py:func:`_sessions_of`.
            releases: The releases of the graph.
            form: The form, for the messages.
            timeline: ``{stable id: [(position of the release, version), ...]}`` (:py:meth:`_timeline`).
            alive: ``{<id>.<version>: bit mask of the positions of the releases at which it exists}``.

        Returns:
            The event edges ``(old, new, old release, new release, score)``; for every id that a session re-mapped
            onto itself, the latest release pair of such a re-mapping; the report of every session; the corrections
            of recorded release pairs; the events placed with the version of the session's old release.

        Raises:
            HistoryValidationError: If the events of a session fit no pair of consecutive releases, if they fit
                several pairs equally well and the recorded pair is not among them, or if an event fits two other
                pairs equally near its session's pair.
        """
        position = {r: i for i, r in enumerate(releases)}
        last = len(releases) - 1
        edges: list[tuple] = []
        loop_release: dict[str, tuple[int, int]] = {}
        report_sessions: list[dict[str, Any]] = []
        corrections: list[dict[str, Any]] = []
        versions_placed: list[dict[str, Any]] = []
        for session_key in sorted(sessions, key=lambda k: tuple((isinstance(v, str), v) for v in k)):
            session = sessions[session_key]
            recorded = session["recorded"]
            used, best, voters = self._session_pair(session, releases, position, form)
            made = moved = contradicting = 0
            for old, new, fits, score in session["rows"]:
                if used is not None and fits >> used & 1:
                    pair = used
                elif used is not None and (placed := _version_of_release(old, new, used, timeline, alive)) is not None:
                    versions_placed.append(
                        {
                            "event": [old, new],
                            "placed": [placed, new],
                            "releases": [releases[used], releases[used + 1]],
                        }
                    )
                    self.log.warning(
                        f"History ({form}): Ensembl's event {old} → {new} of the session of releases "
                        f"{releases[used]} to {releases[used + 1]} names a version that did not exist at "
                        f"{releases[used]}; it is placed with the version the id had there: {placed} → {new}."
                    )
                    old, pair = placed, used
                elif fits:
                    candidates = [i for i in range(last) if fits >> i & 1]
                    pair = _nearest_pair(candidates, used, releases, old, new, recorded, form)
                    moved += 1
                else:
                    contradicting += 1
                    continue
                edges.append((old, new, releases[pair], releases[pair + 1], score))
                made += 1
                if old == new and (old not in loop_release or loop_release[old][0] < releases[pair]):
                    loop_release[old] = (releases[pair], releases[pair + 1])
            entry = {k: v for k, v in session.items() if k not in ("votes", "rows")}
            entry.update(
                {
                    "used": None if used is None else [releases[used], releases[used + 1]],
                    "edges": made,
                    "moved": moved,
                    "contradicting": contradicting,
                }
            )
            report_sessions.append(entry)
            if entry["used"] is not None and entry["used"] != recorded:
                corrections.append(
                    {"recorded": recorded, "used": entry["used"], "events": session["events"], "edges": made}
                )
                self.log.warning(
                    f"History corrected ({form}): Ensembl's mapping session recorded as releases {recorded[0]} to "
                    f"{recorded[1]} belongs to releases {entry['used'][0]} to {entry['used'][1]} according to the "
                    f"id tables ({best} of {voters} events between different ids fit that pair); {made} edges."
                )
        return edges, loop_release, report_sessions, corrections, versions_placed

    @staticmethod
    def _session_pair(
        session: dict[str, Any], releases: list[int], position: dict[int, int], form: str
    ) -> tuple[int | None, int, int]:
        """Return the pair of a session (its position, or ``None``), the votes of the best pair, and the voters.

        Raises:
            HistoryValidationError: If the session's events fit no pair, or several pairs equally without the
                recorded one among them.
        """
        recorded = session["recorded"]
        last = len(releases) - 1
        recorded_pair = None
        if recorded[0] in position and recorded[1] in position and position[recorded[1]] == position[recorded[0]] + 1:
            recorded_pair = position[recorded[0]]
        tally = [0] * max(last, 1)
        for fits, count in session["votes"].items():
            for i in range(last):
                if fits >> i & 1:
                    tally[i] += count
        voters = sum(session["votes"].values())
        best = max(tally) if voters else 0
        if voters and best == 0:
            raise HistoryValidationError(
                f"The {form} events of Ensembl's mapping session recorded as releases {recorded[0]} to "
                f"{recorded[1]} fit no pair of consecutive releases of the id tables ({voters} events between "
                f"ids that exist in the tables, e.g. {session['rows'][0][0]} to {session['rows'][0][1]}). The "
                "history cannot be reconciled with the id tables."
            )
        winners = [i for i in range(last) if voters and tally[i] == best]
        if not voters or recorded_pair in winners:
            return recorded_pair, best, voters
        if len(winners) > 1:
            pairs = ", ".join(f"{releases[i]} to {releases[i + 1]}" for i in winners)
            raise HistoryValidationError(
                f"The {form} events of Ensembl's mapping session recorded as releases {recorded[0]} to "
                f"{recorded[1]} fit the pairs of releases {pairs} equally well ({best} of {voters} events each), and "
                "the recorded pair is not among them: the id tables do not say where the session belongs."
            )
        return winners[0], best, voters

    def _snapshot_loops(
        self,
        database_manager: DatabaseManager,
        alive: dict[str, int],
        loop_release: dict[str, tuple[int, int]],
        releases: list[int],
    ) -> list[tuple]:
        """Return the self-loops to infinity of the ids of the last release.

        A loop starts at the start of the id's last stay, or at the release of its latest re-mapping onto itself if
        that is later.

        Raises:
            BuildError: If an id of the snapshot release is not in the backbone at that release.
        """
        form = database_manager.form
        last = len(releases) - 1
        edges: list[tuple] = []
        latest_table = database_manager.get_db("ids", save_after_calculation=False)
        for name in self._names(latest_table):
            mask = alive.get(name, 0)
            if not mask >> last & 1:
                raise BuildError(
                    f"The {form} id {name!r} of the snapshot release {releases[-1]} is not in the backbone of the "
                    "graph at that release."
                )
            start = last
            while start > 0 and mask >> (start - 1) & 1:
                start -= 1
            since = releases[start]
            if name in loop_release and loop_release[name][1] > since:
                since = loop_release[name][1]
            edges.append((name, name, since, math.inf, 1.0))
        return edges

    def _check_history_reaches_back(self, database_manager: DatabaseManager, releases: list[int], old_releases) -> None:
        """Stop if the id-history table starts later than Ensembl's history does within the window of the graph.

        The id-history table is kept in the local repository and is computed for a release window. A table that was
        computed for a later window lacks the early sessions, and a graph built from it silently has no history
        there. Ensembl's ``mapping_session`` table of the snapshot release lists all sessions; when it has sessions
        in the window before the first one of the id-history table, the events of those sessions are counted in
        Ensembl's ``stable_id_event`` table, and events of this form there mean that the stored table is not the
        table of this window.

        Args:
            database_manager: The manager of the form at the snapshot release.
            releases: The releases of the graph.
            old_releases: The ``old_release`` column of the id-history table.

        Raises:
            HistoryValidationError: If Ensembl has events of the form in sessions that the table lacks.
        """
        have = sorted({int(r) for r in pd.to_numeric(old_releases, errors="coerce").dropna().tolist()})
        sessions = database_manager.get_table(
            "mapping_session", save_after_calculation=database_manager.store_raw_always
        )
        if sessions is None or not len(sessions) or "old_release" not in sessions:
            return
        old = pd.to_numeric(sessions["old_release"], errors="coerce")
        early = sessions[(old >= releases[0]) & (old <= releases[-1]) & (old < (min(have) if have else math.inf))]
        if not len(early):
            return
        events = database_manager.get_table("stable_id_event", save_after_calculation=database_manager.store_raw_always)
        lost = events[
            events["mapping_session_id"].isin(early["mapping_session_id"]) & (events["type"] == database_manager.form)
        ]
        if len(lost):
            raise HistoryValidationError(
                f"The id-history table of {database_manager.organism} in the local repository starts at release "
                f"{min(have) if have else 'none'}, but Ensembl has {len(lost)} {database_manager.form} events in mapping "
                f"sessions from release {int(old[early.index].min())} on, inside the window of the graph. The stored "
                "table was computed for another release window: remove it (or use an empty local repository) and "
                "build again."
            )

    # ------------------------------------------------------------------ files

    def graph_identity(
        self, session_details: bool = False, forms: list[str] | None = None, all_external: bool = False
    ) -> dict[str, Any]:
        """Return the fields of a record that say which graph this maker builds; a loaded file must have the same.

        Computed from the manager, the organism table and the externals YAML file only (no table is read).

        Args:
            session_details: See :py:meth:`construct_graph`.
            forms: See :py:meth:`construct_graph`.
            all_external: See :py:meth:`construct_graph`.

        Returns:
            dict: The fields :py:data:`idtrack.graph.record.IDENTITY_FIELDS` as JSON values: ``organism``,
            ``primary_assembly``, ``first_release`` and ``window_end`` (the window; ``None`` for no limit),
            ``snapshot_release``, ``forms``, ``session_details``, ``all_external``, ``externals_fingerprint`` (the
            databases selected per form, assembly and release, as far as the selection takes effect: inside the
            window and inside the releases the organism table gives the assembly) and ``organism_table_fingerprint``
            (the assemblies of the organism with their release ranges and order).
        """
        dm = self.database_manager
        forms = list(dm.available_form_of_interests if not forms else forms)
        return {
            "organism": dm.organism,
            "primary_assembly": int(dm.assembly),
            "first_release": int(dm.first_release),
            "window_end": None if np.isinf(dm.snapshot_release) else int(dm.snapshot_release),
            "snapshot_release": int(dm.release),
            "forms": forms,
            "session_details": bool(session_details),
            "all_external": bool(all_external),
            "externals_fingerprint": self._externals_fingerprint(forms),
            "organism_table_fingerprint": self._organism_table_fingerprint(),
        }

    def _externals_fingerprint(self, forms: list[str]) -> str:
        """Return a digest of the selection of external databases that takes effect in this maker's graph."""
        dm = self.database_manager
        return files.selection_fingerprint(
            dm.external_inst.read_modified_yaml(), dm.organism, forms, dm.first_release, dm.snapshot_release
        )

    def _organism_table_fingerprint(self) -> str:
        """Return a digest of the organism's assemblies: keys, release ranges, order and database suffixes."""
        return files.table_fingerprint(self.database_manager.organism)

    @staticmethod
    def load_graph(
        local_repository: str,
        organism: str,
        snapshot_release: int,
        first_release: int | None = None,
        *,
        session_details: bool = False,
        forms: list[str] | None = None,
        all_external: bool = False,
        graph_settings: GraphSettings | None = None,
    ) -> TheGraph:
        """Load a stored graph without a :py:class:`DatabaseManager` and without any network access.

        The file is found by its name (:py:func:`idtrack.graph.files.graph_file_path`) and checked against what is
        asked for, as in :py:meth:`get_graph`: organism, primary assembly, release window, forms, the selection of
        external databases (the externals YAML of the local repository, else the one shipped with the package) and the
        organism table.

        Args:
            local_repository: Folder that holds the graph file.
            organism: Organism name, e.g. ``homo_sapiens``.
            snapshot_release: Last release of the graph's window (``snapshot_release`` of ``API.build``).
            first_release: First release of the window; by default the first release of the organism in the
                organism table (:py:meth:`idtrack.organisms.first_release`), as ``API.build``.
            session_details: See :py:meth:`construct_graph`.
            forms: See :py:meth:`construct_graph`.
            all_external: See :py:meth:`construct_graph`.
            graph_settings: The caches and hyper-connectivity threshold of the graph.

        Returns:
            The frozen graph.

        Raises:
            GraphFileError: If the local repository has no such graph file (build it with :py:meth:`get_graph`).
            GraphMismatchError: If the file holds another graph than the one asked for.
            UnknownReleaseError: If no assembly of the organism has ``snapshot_release``.
        """
        from idtrack.build.externals import ExternalDatabases

        primary = organisms.primary_assembly(organism, snapshot_release)  # an organism that is not configured is named
        if first_release is None:
            first_release = organisms.first_release(organism)
        forms = list(ids.FORMS if not forms else forms)
        path = files.graph_file_path(
            local_repository,
            organism,
            primary,
            first_release,
            snapshot_release,
            session_details=session_details,
            forms=None if forms == list(ids.FORMS) else forms,
            all_external=all_external,
        )
        externals = ExternalDatabases(organism, int(snapshot_release), ids.BACKBONE_FORM, local_repository, primary)
        expect = {
            "organism": organism,
            "primary_assembly": int(primary),
            "first_release": int(first_release),
            "window_end": int(snapshot_release),
            "forms": forms,
            "session_details": bool(session_details),
            "all_external": bool(all_external),
            "externals_fingerprint": files.selection_fingerprint(
                externals.read_modified_yaml(), organism, forms, first_release, snapshot_release
            ),
            "organism_table_fingerprint": files.table_fingerprint(organism),
        }
        return TheGraph.load(path, expect=expect, graph_settings=graph_settings)

    def get_graph(
        self,
        session_details: bool = False,
        create_even_if_exist: bool = False,
        save_after_calculation: bool = True,
        *,
        forms: list[str] | None = None,
        all_external: bool = False,
    ) -> TheGraph:
        """Load the graph from the local repository, or build it.

        A graph file is loaded only if its record is the graph that is asked for: organism, primary assembly, release
        window, snapshot release, forms, the selection of external databases and the organism table. A file that
        holds another graph under the expected name is an error, not a silent substitute and not a silent rebuild of
        several hours.

        Args:
            session_details: See :py:meth:`construct_graph`.
            create_even_if_exist: Build the graph even if a file exists. The new graph replaces the file (unless
                ``save_after_calculation`` is ``False``).
            save_after_calculation: Write a newly built graph to the local repository.
            forms: See :py:meth:`construct_graph`.
            all_external: See :py:meth:`construct_graph`.

        Returns:
            The frozen graph.

        Raises:
            GraphMismatchError: If the file in the local repository is not the graph asked for, or unreadable. The
                message names the differences; ``create_even_if_exist=True`` replaces the file.
        """
        options: dict[str, Any] = {"session_details": session_details, "forms": forms, "all_external": all_external}
        file_path = self.create_file_name(**options)
        if os.path.isfile(file_path) and not create_even_if_exist:
            self.log.info(f"The graph is being read: {file_path}")
            return TheGraph.load(file_path, expect=self.graph_identity(**options))

        self.log.info(f"The graph is being constructed: {file_path}")
        g = self.construct_graph(**options)
        if save_after_calculation:
            self.log.info(f"The graph is being exported as {file_path!r}.")
            g.save(file_path)
        return g

    def create_file_name(
        self, session_details: bool = False, forms: list[str] | None = None, all_external: bool = False
    ) -> str:
        """Return the path of the graph file in the manager's local repository.

        Args:
            session_details: See :py:meth:`construct_graph`.
            forms: See :py:meth:`construct_graph`.
            all_external: See :py:meth:`construct_graph`.

        Returns:
            Absolute file path; see :py:func:`idtrack.graph.files.graph_file_path`.
        """
        dm = self.database_manager
        return files.graph_file_path(
            dm.local_repository,
            dm.organism,
            int(dm.assembly),
            dm.first_release,
            dm.snapshot_release,
            session_details=session_details,
            forms=(None if forms is None or list(forms) == list(dm.available_form_of_interests) else forms),
            all_external=all_external,
        )


def _present(value: Any) -> bool:
    """Tell whether a table cell holds a value (not empty, not missing, not zero)."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return bool(value)


def _plain(value: Any) -> Any:
    """Turn a table cell into a JSON value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, (int, str, bool)):
        return value
    return str(value)


def _release_json(value: Any) -> Any:
    """Return a release as an int where it is one (Ensembl has releases such as 18.2), ``None`` if missing."""
    if value is None or pd.isna(value):
        return None
    return int(value) if float(value).is_integer() else float(value)


def _int_or(value: Any) -> Any:
    """Return an integral number as ``int`` (the key type of the weight table), anything else unchanged."""
    try:
        return int(value) if float(value).is_integer() else value
    except (TypeError, ValueError):
        return value


class _Nodes:
    """The nodes of a graph being built: a number per name, in the order they are added, and the type of each."""

    def __init__(self) -> None:
        self.number: dict[str, int] = {}
        self.names: list[str] = []
        self.types: list[str] = []

    def add(self, name: str, node_type: str) -> int:
        """Return the number of a node, adding it with ``node_type`` if it is new (a known node keeps its type)."""
        found = self.number.get(name)
        if found is None:
            found = self.number[name] = len(self.names)
            self.names.append(name)
            self.types.append(node_type)
        return found


class _Edges:
    """The cross-reference edges of a graph being built: ``{(database, assembly): {source << 32 | target: mask}}``."""

    def __init__(self) -> None:
        self.groups: dict[tuple[str, int], dict[int, int]] = {}

    def add(self, sources: list[int], targets: list[int], database: str, assembly: int, release: int) -> None:
        """Record that ``database`` links every source to its target on ``assembly`` at ``release``."""
        if not sources:
            return
        bit = 1 << int(release)
        keys = np.unique((np.asarray(sources, dtype=np.int64) << 32) | np.asarray(targets, dtype=np.int64))
        group = self.groups.setdefault((str(database), int(assembly)), {})
        for key in keys.tolist():
            group[key] = group.get(key, 0) | bit


def _latest_and_count(masks: list[int]) -> tuple[int, int]:
    """Return the latest release of some links (``-1`` for none) and the number of them that hold at it."""
    union = 0
    for mask in masks:
        union |= mask
    latest = union.bit_length() - 1
    return latest, sum(1 for mask in masks if mask >> latest & 1) if latest >= 0 else 0


def _relation_pairs(forms: list[str]) -> list[tuple[str, str]]:
    """Return the relations between the forms of a graph: ``(lower form, upper form)``."""
    pairs: list[tuple[str, str]] = []
    if "transcript" in forms and "gene" in forms:
        pairs.append(("transcript", "gene"))
    if "translation" in forms and "transcript" in forms:
        pairs.append(("translation", "transcript"))
    return pairs


def _assembly_order(serving_assembly: dict[int, int]) -> list[int]:
    """Return the assemblies that serve releases, by the first release each serves."""
    first: dict[int, int] = {}
    for release, assembly in serving_assembly.items():
        first[int(assembly)] = min(first.get(int(assembly), int(release)), int(release))
    return sorted(first, key=lambda a: first[a])


def _later_assemblies(assembly: int, serving_assembly: dict[int, int]) -> list[int]:
    """Return the assemblies that serve releases after those an assembly serves first, in their order."""
    order = _assembly_order(serving_assembly)
    return order[order.index(assembly) + 1 :] if assembly in order else []


def _carry(
    chains: dict[tuple[int, int], AssemblyChain],
    segment: tuple[str, int, int, int],
    earlier: int,
    later: int,
    serving_assembly: dict[int, int],
) -> list[tuple[str, int, int, int]]:
    """Carry a place of one assembly onto a later one by Ensembl's mapping (decision 103).

    By the chain of the pair where Ensembl publishes one; else through the assembly that follows the earlier one,
    where Ensembl maps the earlier onto it and that one onto the later (a published mapping at every step); else none.

    Returns:
        list[tuple[str, int, int, int]]: ``(region, first base, last base, strand)`` on the later assembly.
    """
    chain = chains.get((earlier, later))
    if chain is not None:
        return chain.carry(*segment)
    following = _later_assemblies(earlier, serving_assembly)
    if not following or following[0] == later or (earlier, following[0]) not in chains:
        return []
    found: list[tuple[str, int, int, int]] = []
    for step in chains[(earlier, following[0])].carry(*segment):
        found.extend(_carry(chains, step, following[0], later, serving_assembly))
    return sorted(set(found))


def _carried_places(
    places: dict[tuple[int, int, str, int, int, int, str], int],
    non_reference: dict[int, int],
    chains: dict[tuple[int, int], AssemblyChain],
    serving_assembly: dict[int, int],
) -> dict[tuple[int, int, str, int, int, int, str], int]:
    """Return every place on the reference sequence carried onto the later assemblies by Ensembl's mapping.

    Args:
        places: ``{(node number, assembly, region, start, end, strand, kind): release mask}`` (:py:meth:`GraphMaker._gene_regions`).
        non_reference: ``{node number: release mask}`` of the genes outside the reference sequence.
        chains: Ensembl's mappings of the assemblies.
        serving_assembly: ``{release: assembly that serves it}``.

    Returns:
        dict: ``{(node number, later assembly, region, start, end, strand, kind): release mask of the place carried}``;
        the kind of a region of the later assembly is the kind its places give it (a scaffold where none lies there).
    """
    if not chains:
        return {}
    kind_of = {(key[1], key[2]): key[6] for key in places}
    carried: dict[tuple[int, int, str, int, int, int, str], int] = {}
    known: dict[tuple, list[tuple[str, int, int, int]]] = {}
    for (node, assembly, region, start, end, strand, _kind), mask in places.items():
        if mask & non_reference.get(node, 0):
            continue
        for later in _later_assemblies(assembly, serving_assembly):
            key = (assembly, later, region, start, end, strand)
            found = known.get(key)
            if found is None:
                found = known[key] = _carry(chains, (region, start, end, strand), assembly, later, serving_assembly)
            for there, first, last, sign in found:
                place = (node, later, there, first, last, sign, kind_of.get((later, there), SCAFFOLD))
                carried[place] = carried.get(place, 0) | mask
    return carried


def _exception_regions(exceptions: tuple[AssemblyException, ...]) -> dict:
    """Return the assembly exceptions by assembly and region (:py:data:`idtrack.graph.genes.Exceptions`)."""
    table: dict[tuple[int, str], list] = {}
    for row in exceptions:
        table.setdefault((row.assembly, row.region), []).append(
            (
                row.start,
                row.end,
                row.releases.mask,
                row.ref_region,
                row.ref_start,
                row.ref_end,
                row.ref_kind,
                row.carried,
            )
        )
    return {key: tuple(rows) for key, rows in table.items()}


def _gene_counter(
    nodes: _Nodes,
    places: dict[tuple[int, int, str, int, int, int, str], int],
    non_reference: dict[int, int],
    history_edges: list[tuple],
    repeats: Repeats,
    carried: dict[tuple[int, int, str, int, int, int, str], int] | None = None,
    exceptions: Exceptions | None = None,
) -> Callable[[Collection[str]], int]:
    """Return the number of genes some stable gene ids name, by the package's one rule, from the facts of the build.

    The facts are those a built graph keeps (:py:meth:`idtrack.TheGraph.count_genes`): every gene version's places,
    with the kinds of their regions and their releases (a version lives at the releases of its places), the releases
    at which it lies outside the reference sequence (:py:meth:`GraphMaker._gene_regions`), and Ensembl's history
    between two stable ids.

    Args:
        nodes: The nodes so far.
        places: ``{(node number, assembly, region, start, end, strand, kind): release mask}``.
        non_reference: ``{node number: release mask}`` of the genes outside the reference sequence.
        history_edges: The keyed history edges (:py:meth:`GraphMaker.history`): source and target names first.
        repeats: The regions of the organism that repeat another (:py:func:`idtrack.organisms.pseudoautosomal_repeats`).
        carried: The places carried onto later assemblies (:py:func:`_carried_places`).
        exceptions: Ensembl's assembly exceptions (:py:func:`_exception_regions`).

    Returns:
        The function of :py:func:`idtrack.graph.genes.count_genes` over these facts.
    """
    names = nodes.names
    per_version: dict[str, list[tuple[int, str, int, int, int, int, str]]] = {}
    for (node, assembly, region, start, end, strand, kind), mask in places.items():
        per_version.setdefault(names[node], []).append((assembly, region, start, end, strand, mask, kind))
    outside = {names[node]: mask for node, mask in non_reference.items()}
    history = (
        (source, target)
        for source, target, *_ in history_edges
        if ids.is_versioned_name(source) and ids.is_versioned_name(target)
    )
    # Every gene version of the build holds a place (the places of every release are read, `_gene_regions`), so none
    # needs the releases of its history: the facts are those the built graph gives (:py:meth:`TheGraph.count_genes`).
    moved: dict[str, list[tuple[int, str, int, int, int]]] = {}
    for node, assembly, region, start, end, strand, _kind in carried or {}:
        moved.setdefault(names[node], []).append((assembly, region, start, end, strand))
    facts, joined = facts_by_stable_id(per_version, outside, history, repeats, None, (), moved, exceptions)
    return functools.partial(count_genes, facts=facts, joined=joined)


def _links_of_one_gene_only(
    nodes: _Nodes, edges: _Edges, databases: set[str], gene_count: Callable[[Collection[str]], int]
) -> tuple[int, set[int], set[str]]:
    """Leave out the links of every entry of :py:data:`idtrack.build.database_manager.ONE_GENE_ONLY` of several genes.

    An entry (an accession, a label or a synonym of such a database, whatever its letter case) is judged over every
    release and assembly of the graph: it keeps its links unless the genes they reach are shown at two loci by the
    package's rule (``gene_count``, :py:func:`idtrack.graph.genes.count_genes`: two genes that live at one release at
    places that do not overlap, or genes one after the other at places apart on one assembly or on chromosomes of
    different names; a copy outside the reference sequence adds none, and holders on two assemblies that nothing
    compares are no evidence of two loci). A link to a transcript or a translation counts for the gene that holds it
    at the releases of the link (a transcript that Ensembl moved to another gene at other releases, such as a miRNA's
    transcript later placed in its host gene, is not evidence for that gene). An external node left without any link
    is no node of the graph.

    Args:
        nodes: The nodes so far.
        edges: The edges so far (the relations between the forms among them).
        databases: The names of those databases' accessions and labels (synonyms carry the prefix).
        gene_count: The number of genes some stable gene ids name (:py:func:`_gene_counter`).

    Returns:
        The number of links left out, the external nodes left without a link, and the entries left out (lower case).
    """

    def judged(database: str) -> bool:
        return database.removeprefix(ids.SYNONYM_PREFIX) in databases

    names, types = nodes.names, nodes.types
    keyed = [group for (database, _assembly), group in edges.groups.items() if judged(database)]
    # a transcript's genes, a translation's transcripts, each with the releases of the relation (read where needed)
    upper: dict[int, dict[int, int]] = {}

    def genes_of(target: int, mask: int) -> set[str]:
        """The genes that hold a node at some release of ``mask`` (a transcript's gene at those releases)."""
        form = ids.FORM_OF_ENSEMBL_TYPE.get(types[target])
        if form == ids.BACKBONE_FORM:
            return {ids.split(names[target])[0]}
        if not upper:
            for (database, _assembly), group in edges.groups.items():
                if database in (ids.ENSEMBL_TYPE["transcript"], ids.ENSEMBL_TYPE["translation"]):
                    for key, held in group.items():
                        parents = upper.setdefault(key >> 32, {})
                        parents[key & 0xFFFFFFFF] = parents.get(key & 0xFFFFFFFF, 0) | held
        found: set[str] = set()
        for parent, held in upper.get(target, {}).items():
            if mask & held:
                found |= genes_of(parent, mask & held)
        return found

    shared = entries_of_several_genes(
        (
            (names[key >> 32], gene)
            for group in keyed
            for key, mask in group.items()
            for gene in genes_of(key & 0xFFFFFFFF, mask)
        ),
        gene_count,
    )
    left_out, touched = 0, set()
    for group in keyed:
        for key in [key for key in group if names[key >> 32].lower() in shared]:
            del group[key]
            left_out += 1
            touched.add(key >> 32)
    linked = {key >> 32 for group in edges.groups.values() for key in group} if touched else set()
    return left_out, {number for number in touched if number not in linked}, shared


def _link_marks(
    marks: dict[tuple, dict[int, int]],
    edges: "_Edges",
    merge_into: dict[int, int],
    serving_assembly: dict[int, int],
    by_database: bool = True,
) -> dict[tuple[int, int, str, int], int]:
    """Return the values that the links of the graph carry per release, as the graph keeps them.

    A value counts only where its link is an edge of the graph (an entry that the one-gene rule left out keeps none:
    :py:func:`_links_of_one_gene_only`), on the node that stays after the merge of spellings, and at a release as the
    assembly that serves it has it, as a spelling is judged (:py:meth:`GraphMaker._merge_spellings`): human GRCh37,
    frozen and listed at every release that GRCh38 serves, does not give the version or the display name of such a
    release. A link that no serving assembly has is judged where it is listed.

    Args:
        marks: ``{(external node, Ensembl node, database, assembly): {value: release mask}}``.
        edges: The cross-reference edges of the graph.
        merge_into: ``{number of a merged spelling: number of the node that stays}``.
        serving_assembly: The assembly that serves each release of the graph.
        by_database: Keep the database of a value (the versions of an accession, the database a display name is
            filed under); ``False``: the value of a link whatever database gives it.

    Returns:
        ``{(external node, Ensembl node, database or "", value): release mask}``, in the provisional numbers of the
        build (the external node the one that stays).
    """
    served = {assembly: releases.mask for assembly, releases in serving_line(serving_assembly).items()}
    per_link: dict[tuple[int, int, str], dict[int, list[tuple[int, int]]]] = {}
    for (source, target, database, assembly), values in marks.items():
        held = edges.groups.get((database, assembly), {}).get((source << 32) | target, 0)
        link = (merge_into.get(source, source), target, database if by_database else "")
        for value, mask in values.items():
            if mask & held:
                per_link.setdefault(link, {}).setdefault(value, []).append((assembly, mask & held))
    result: dict[tuple[int, int, str, int], int] = {}
    for (source, target, database), by_value in per_link.items():
        judged: dict[int, int] = {}
        for value, pairs in by_value.items():
            judged[value] = 0
            for assembly, mask in pairs:
                judged[value] |= mask & served.get(assembly, 0)
        if not any(judged.values()):  # a link that no serving assembly has: judged where it is listed
            for value, pairs in by_value.items():
                for _assembly, mask in pairs:
                    judged[value] |= mask
        for value, mask in judged.items():
            if mask:
                result[(source, target, database, value)] = mask
    return result


def _pack(
    nodes: _Nodes,
    edges: _Edges,
    history_edges: list[tuple],
    merge_into: dict[int, int],
    unlinked: set[int] = frozenset(),  # type: ignore[assignment]
) -> tuple[_Store, list[Connection], np.ndarray]:
    """Pack the nodes and edges of a graph being built into its store.

    The final node numbers follow the sorted names of the nodes that remain after the merge of spellings; the
    cross-reference edges get their connections, databases and assemblies in sorted order. The edge groups are
    emptied on the way (their memory is released).

    Args:
        nodes: The nodes.
        edges: The cross-reference edges.
        history_edges: The keyed history edges of :py:meth:`GraphMaker.history`.
        merge_into: ``{number of a merged spelling: number of the node that stays}``.
        unlinked: External nodes left without a link, which are no nodes of the graph.

    Returns:
        The store, its distinct connections, and the final number of every provisional node number.
    """
    names, types = nodes.names, nodes.types
    keep_numbers = [i for i in range(len(names)) if i not in merge_into and i not in unlinked]
    keep_numbers.sort(key=lambda i: names[i])
    final_of = np.full(len(names), -1, dtype=np.int64)
    final_of[keep_numbers] = np.arange(len(keep_numbers))
    for old, new in merge_into.items():
        final_of[old] = final_of[new]
    final_names = [names[i] for i in keep_numbers]
    node_types = sorted({types[i] for i in keep_numbers})
    type_code = {t: c for c, t in enumerate(node_types)}
    ntype = np.asarray([type_code[types[i]] for i in keep_numbers], dtype=np.int16)

    groups = edges.groups
    items_of_edge: dict[int, tuple] = {}
    for database, assembly in sorted(groups):
        for key, mask in groups[(database, assembly)].items():
            source = key >> 32
            if source in merge_into:
                key = (merge_into[source] << 32) | (key & 0xFFFFFFFF)
            have = items_of_edge.get(key)
            if have is None:
                items_of_edge[key] = ((database, assembly, mask),)
            elif have[-1][0] == database and have[-1][1] == assembly:
                items_of_edge[key] = have[:-1] + ((database, assembly, have[-1][2] | mask),)
            else:
                items_of_edge[key] = have + ((database, assembly, mask),)
        groups[(database, assembly)] = {}  # release the memory
    connections: list[Connection] = []
    connection_code: dict[tuple, int] = {}
    x_keys = np.fromiter(items_of_edge.keys(), dtype=np.int64, count=len(items_of_edge))
    x_conn = np.empty(len(items_of_edge), dtype=np.int32)
    for position, items in enumerate(items_of_edge.values()):
        code = connection_code.get(items)
        if code is None:
            code = connection_code[items] = len(connections)
            connections.append(Connection.of(items))
        x_conn[position] = code
    del items_of_edge
    x_src = final_of[x_keys >> 32]
    x_dst = final_of[x_keys & 0xFFFFFFFF]

    number = nodes.number
    h_src = final_of[np.asarray([number[e[0]] for e in history_edges], dtype=np.int64)]
    h_dst = final_of[np.asarray([number[e[1]] for e in history_edges], dtype=np.int64)]
    h_key = np.asarray([e[5] for e in history_edges], dtype=np.int32)
    h_old = np.asarray([e[2] for e in history_edges], dtype=np.int32)
    h_new = np.asarray([-1 if e[3] == math.inf else e[3] for e in history_edges], dtype=np.int32)
    h_weight = np.asarray([e[4] for e in history_edges], dtype=np.float64)

    store = _Store.from_parts(
        final_names, node_types, ntype, (h_src, h_dst, h_key, h_old, h_new, h_weight), (x_src, x_dst, x_conn),
        connections,
    )  # fmt: skip
    return store, connections, final_of


def _sessions_of(
    events: pd.DataFrame, alive: dict[str, int], session_columns: list[str], session_details: bool, columns: Any
) -> dict[tuple, dict[str, Any]]:
    """Group the events by mapping session, with the release pairs each event fits.

    Args:
        events: The event rows of :py:meth:`GraphMaker._events_and_weights`.
        alive: ``{<id>.<version>: bit mask of the positions of the releases at which it exists}``.
        session_columns: The columns that name a session.
        session_details: Whether the table is the full one (the session's metadata is then kept for the report).
        columns: The columns of the id-history table.

    Returns:
        ``{session key: {"recorded", "events", "absent", "votes", "rows", ...}}``: the recorded release pair, the
        number of events and of events between ids in no id table, the votes of the events between different ids
        for the pairs they fit (``{fits mask: count}``), and the rows ``(old, new, fits mask, score)``.
    """
    sessions: dict[tuple, dict[str, Any]] = {}
    for row in events.itertuples(index=False):
        session_key = tuple(getattr(row, c) for c in session_columns)
        session = sessions.get(session_key)
        if session is None:
            session = sessions[session_key] = {
                "recorded": [_release_json(row.old_release), _release_json(row.new_release)],
                "events": 0, "absent": 0, "votes": Counter(), "rows": [],
            }  # fmt: skip
            if session_details:
                for column in (
                    "mapping_session_id",
                    "old_db_name",
                    "new_db_name",
                    "old_assembly",
                    "new_assembly",
                    "created",
                ):
                    if column in columns:
                        session[column] = _plain(getattr(row, column))
        session["events"] += 1
        old = ids.join(row.old_stable_id, int(row.old_version))
        new = ids.join(row.new_stable_id, int(row.new_version))
        if old not in alive or new not in alive:
            # For example, 'ENSG00000289022' is defined in release 105 according to the history but is in no
            # gene table (neither 104, 105, 106). Such ids are not nodes.
            session["absent"] += 1
            continue
        fits = alive[old] & (alive[new] >> 1)  # bit i: `old` exists at releases[i] and `new` at releases[i + 1]
        session["rows"].append((old, new, fits, float(row.score) if not pd.isna(row.score) else math.nan))
        if old != new:
            session["votes"][fits] += 1
    return sessions


def _version_of_release(
    old: str, new: str, pair: int, timeline: dict[str, list[tuple[int, Any]]], alive: dict[str, int]
) -> str | None:
    """Return the old id with the version it had at the old release of the event's pair, where the event names another.

    That is when the event's old version did not exist at the old release, the stable id did (with another version),
    and the new id exists at the new release; otherwise ``None``.
    """
    if alive.get(old, 0) >> pair & 1 or not alive.get(new, 0) >> (pair + 1) & 1:
        return None
    stable_id = ids.split(old)[0]
    for position, version in timeline.get(stable_id, ()):
        if position == pair:
            return ids.join(stable_id, version)
    return None


def _nearest_pair(
    candidates: list[int], used: int | None, releases: list[int], old: str, new: str, recorded: list, form: str
) -> int:
    """Return the pair of an event that does not fit its session's pair: the nearest of those it fits.

    Raises:
        HistoryValidationError: If two pairs are equally near, or the session has no pair to measure from and the
            event fits several (none does in the graphs of the organisms idtrack ships; no choice is guessed).
    """
    if used is None:
        if len(candidates) > 1:
            raise HistoryValidationError(
                f"The {form} event {old} → {new} of Ensembl's mapping session recorded as releases {recorded[0]} to "
                f"{recorded[1]} fits the pairs of releases "
                f"{', '.join(f'{releases[i]} to {releases[i + 1]}' for i in candidates)}, and its session has no pair."
            )
        return candidates[0]
    ranked = sorted(candidates, key=lambda i: (abs(i - used), i))
    if len(ranked) > 1 and abs(ranked[0] - used) == abs(ranked[1] - used):
        raise HistoryValidationError(
            f"The {form} event {old} → {new} does not fit the pair of its session ({releases[used]} to "
            f"{releases[used + 1]}) and fits the pairs {releases[ranked[0]]} to {releases[ranked[0] + 1]} and "
            f"{releases[ranked[1]]} to {releases[ranked[1] + 1]} equally near it."
        )
    return ranked[0]


def _self_history(
    timeline: dict[str, list[tuple[int, Any]]], releases: list[int], weights: dict[tuple, float]
) -> list[tuple]:
    """Return the history of every id with itself, from the id tables: births, version changes, retirements, returns.

    Args:
        timeline: ``{stable id: [(position of the release, version), ...]}`` (:py:meth:`GraphMaker._timeline`).
        releases: The releases of the graph.
        weights: Ensembl's scores of version changes.

    Returns:
        The edges ``(old, new, old release, new release, weight)``.
    """
    void, retired = ids.VOID, ids.RETIRED
    last = len(releases) - 1
    edges: list[tuple] = []
    for stable_id, seen in timeline.items():
        previous_index: int | None = None
        previous_version: Any = None
        first_run = True
        for i, version in seen:
            if previous_index is None or i != previous_index + 1:
                if previous_index is not None:  # it had disappeared after `previous_index`
                    edges.append(
                        (
                            ids.join(stable_id, previous_version),
                            ids.join(stable_id, retired),
                            releases[previous_index],
                            releases[previous_index + 1],
                            math.nan,
                        )
                    )
                origin = void if first_run else retired
                old_release = releases[i - 1] if i > 0 else releases[0]
                edges.append(
                    (ids.join(stable_id, origin), ids.join(stable_id, version), old_release, releases[i], math.nan)
                )
                first_run = False
            elif version != previous_version:
                weight = weights.get((stable_id, _int_or(previous_version), stable_id, _int_or(version)), math.nan)
                edges.append(
                    (
                        ids.join(stable_id, previous_version),
                        ids.join(stable_id, version),
                        releases[i - 1],
                        releases[i],
                        weight,
                    )
                )
            previous_index, previous_version = i, version
        if previous_index is not None and previous_index != last:
            edges.append(
                (
                    ids.join(stable_id, previous_version),
                    ids.join(stable_id, retired),
                    releases[previous_index],
                    releases[previous_index + 1],
                    math.nan,
                )
            )
    return edges


def _key_and_validate(edges: list[tuple], alive: dict[str, int], position: dict[int, int]) -> list[tuple]:
    """Put the history edges in a deterministic order, key parallel edges, and check them against the id tables.

    Args:
        edges: ``(old, new, old release, new release, weight)``.
        alive: ``{<id>.<version>: bit mask of the positions of the releases at which it exists}``.
        position: ``{release: its position among the releases of the graph}``.

    Returns:
        ``(old, new, old release, new release, weight, key)``, sorted.

    Raises:
        HistoryValidationError: If an edge between two versions does not join consecutive releases at which its ends
            exist.
    """
    void, retired = ids.VOID, ids.RETIRED
    edges.sort(key=lambda e: (e[0], e[1], e[2], e[3], math.inf if math.isnan(e[4]) else e[4]))
    keyed: list[tuple] = []
    previous: tuple[str, str] | None = None
    parallel = 0
    for old, new, old_release, new_release, weight in edges:
        parallel = parallel + 1 if previous == (old, new) else 0
        previous = (old, new)
        keyed.append((old, new, old_release, new_release, weight, parallel))
        if (
            new_release == math.inf
            or old.endswith(f".{void}")
            or old.endswith(f".{retired}")
            or new.endswith(f".{retired}")
        ):
            continue
        a, b = position[old_release], position[new_release]
        if not (b == a + 1 and alive[old] >> a & 1 and alive[new] >> b & 1):
            raise HistoryValidationError(
                f"History edge {old} → {new} ({old_release} to {new_release}) contradicts the id tables."
            )
    return keyed


def _content_digest(store: _Store) -> str:
    """Return a digest of the content of a graph.

    It covers the node names and types, the edges, the connections, the spellings of the external ids, the places
    of the genes, the marks of the genes outside the reference sequence, the versions of the accessions per link,
    the display names of the genes, the places carried onto later assemblies and Ensembl's assembly exceptions.

    Two builds from the same tables give the same digest, whatever the machine, the Python version or the hash
    seed; any difference in the tables that reaches the graph changes it.
    """
    digest = hashlib.sha256()
    digest.update("\x00".join(store.names).encode("utf-8"))
    digest.update(json.dumps(store.node_types).encode("utf-8"))
    for array in (store.ntype, store.e_src, store.e_dst, store.e_key, store.e_kind, store.h_old, store.h_new):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(np.nan_to_num(store.h_weight, nan=-1.0).tobytes())
    codes: dict[int, int] = {}
    texts: list[str] = []
    for connection in store.connections:
        codes[id(connection)] = len(texts)
        texts.append(json.dumps([[db, asm, rs.mask] for db, asm, rs in connection.items]))
    order = sorted(range(len(texts)), key=lambda i: texts[i])
    rank = np.empty(len(texts), dtype=np.int64)
    rank[order] = np.arange(len(texts))
    digest.update("\x00".join(texts[i] for i in order).encode("utf-8"))
    xref = store.e_kind == 1
    digest.update(rank[store.e_ref[xref]].tobytes())
    digest.update(store.e_ref[~xref].tobytes())
    digest.update(json.dumps([[i, store.non_reference[i].mask] for i in sorted(store.non_reference)]).encode("utf-8"))
    spellings = [
        [*key, latest, [[mask, text] for mask, text in otherwise]]
        for key, (latest, otherwise) in store.spellings.items()
    ]
    digest.update(json.dumps(sorted(spellings)).encode("utf-8"))
    digest.update(store.places.digest_bytes())
    digest.update(store.versions.digest_bytes())
    digest.update(store.displays.digest_bytes())
    digest.update(store.carried.digest_bytes())
    digest.update(
        json.dumps([[*row[:9], row.releases.mask, [list(c) for c in row.carried]] for row in store.exceptions]).encode()
    )
    return digest.hexdigest()


__all__ = ["GraphMaker"]
