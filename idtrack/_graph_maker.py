#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import datetime
import hashlib
import itertools
import json
import logging
import math
import os
import re
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._the_graph import (
    GRAPH_FORMAT_VERSION,
    Connection,
    GraphMismatchError,
    ReleaseSet,
    TheGraph,
    _Store,
    serving_line,
)

_GRAPH_FILE = re.compile(
    r"^graph_(?P<organism>.+)_assembly(?P<genome_assembly>-?[0-9]+)(?:_min(?P<ignore_before>[0-9]+))?"
    r"(?:_max(?P<ignore_after>[0-9]+))?(?P<narrow>_narrow)?(?:_forms(?P<forms>[a-z-]+))?(?P<extall>_extall)?\.npz$"
)


class HistoryValidationError(ValueError):
    """Raised when Ensembl's id history cannot be reconciled with the id tables of the releases."""


class GraphMaker:
    """Builds the identifier graph of one organism and keeps it in the local repository.

    The graph (:py:class:`idtrack.TheGraph`) holds the history of Ensembl gene ids over the releases of
    the manager's window, the transcripts and translations of every release linked to their genes, versionless base
    gene ids, and the identifiers of the selected external databases linked to the Ensembl ids they annotate.

    The history is not copied from Ensembl's ``mapping_session`` table unchecked: the releases of every event are
    validated against the id tables of the releases (which ids exist where), sessions whose recorded releases
    contradict the tables are corrected, and every correction is logged and recorded in the graph's manifest.

    A built graph is written as one file whose manifest says what it was built from; :py:meth:`get_graph` loads a
    file only if that manifest is what is asked for.
    """

    # Note: Example chaotic ID history: ENSG00000263464
    def __init__(self, db_manager: DatabaseManager):
        """Class initialization.

        Args:
            db_manager: Needed to download all necessary tables and data frames.
                It contains the temporary directory to save the resultant graph.

        Raises:
            ValueError: If the release of ``db_manager`` is not the newest release of its window. A graph is the
                snapshot of the releases ``ignore_before`` to ``ignore_after``; it is built from the newest of them.
        """
        # Instance attributes
        self.log = logging.getLogger(__name__)

        # A graph is the snapshot of the manager's window and is built from the newest release in it.
        newest = max(db_manager.available_releases)
        if db_manager.ensembl_release != newest:
            release = db_manager.ensembl_release
            raise ValueError(
                f"A graph is built from the newest release of the manager's window ({newest}), but this manager is "
                f"at release {release}. For a graph whose snapshot is release {release}, end the window there: "
                f"`API.build_graph(organism_name, snapshot_release={release})` or "
                f"`DatabaseManager(..., ignore_after={release})`. A graph of a later snapshot also converts to and "
                f"from release {release} (`to_release`/`from_release` of `API.convert_identifier`)."
            )
        self.db_manager = db_manager
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
        snapshot = int(self.db_manager.ensembl_release)
        return sorted(int(r) for r in self.db_manager.available_releases_all_assemblies if int(r) <= snapshot)

    def _form_manager(self, form: str) -> DatabaseManager:
        key = (form, None, None)
        if key not in self._managers:
            self._managers[key] = self.db_manager.change_form(form)
        return self._managers[key]

    def _backbone_manager(self, form: str, release: int) -> DatabaseManager:
        """Return the manager of the assembly that serves ``release`` in the backbone.

        A release is taken from the primary assembly of the graph where that assembly has it, and otherwise from
        the organism's preferred assembly among those that have it.
        """
        key = (form, int(release), "backbone")
        if key not in self._managers:
            base = self._form_manager(form)
            if int(release) in {int(r) for r in self.db_manager.available_releases}:
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
                    int(assembly), last_possible_ensembl_release=True
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
        digests = self._assembly_manager(DB.backbone_form, assembly, None).table_digests()
        folded = hashlib.sha256()
        for name, digest in sorted(digests.items()):
            folded.update(f"{name}\t{digest}\n".encode())
        return {"tables": len(digests), "fingerprint": folded.hexdigest()}

    def _assembly_releases(self) -> dict[int, list[int]]:
        """Return, for every assembly of the organism that has releases in the graph, those releases."""
        releases = set(self.graph_releases())
        table = DB.assembly_mysqlport_priority[self.db_manager.organism]
        result: dict[int, list[int]] = {}
        for assembly in DB.assemblies_by_priority(self.db_manager.organism):
            first, last = table[assembly]["ReleaseRange"]
            if not any(first <= r and (last is None or r <= last) for r in releases):
                continue
            if int(assembly) == int(self.db_manager.genome_assembly):
                found = [int(r) for r in self.db_manager.available_releases]
            else:
                try:
                    found = [
                        int(r) for r in self._assembly_manager(DB.backbone_form, assembly, None).available_releases
                    ]
                except ValueError as error:
                    self.log.warning(
                        f"Assembly {assembly} of {self.db_manager.organism} is skipped: no core database was found "
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
        return [DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(i, j)) for i, j in table.values]

    def _gene_ids(self, manager: DatabaseManager, save: bool) -> dict[str, Any]:
        """Return ``{stable id: version}`` of the gene table of a manager (read once per assembly and release)."""
        key = (manager.form, int(manager.genome_assembly), int(manager.ensembl_release))
        if key not in self._ids_cache:
            table = manager.get_db("ids") if save else manager.get_db("ids", save_after_calculation=False)
            self._table_counts[("ids", "tables")] += 1
            self._table_counts[("ids", "rows")] += len(table)
            self._ids_cache[key] = {i: j for i, j in table.values}
        return self._ids_cache[key]

    # ------------------------------------------------------------------ the build

    # flake8: noqa: C901
    def construct_graph(
        self,
        narrow: bool = True,
        form_list: list | None = None,
        narrow_external: bool = True,
    ) -> TheGraph:
        """Build the graph from the tables of the manager.

        The steps: the gene history (:py:meth:`history`), the transcripts and translations of every release, the
        genes that lie outside the reference sequence (:py:meth:`_non_reference_genes`), the relations between the
        forms, the identifiers of the external databases, the Ensembl ids that exist on another assembly only, and
        the versionless base gene ids. External identifiers that differ only in letter
        case are one node, named with the spelling that Ensembl gives at the latest release in which the identifier
        exists; how each database wrote it at each release is kept beside it (:py:meth:`_merge_spellings`,
        :py:meth:`idtrack.TheGraph.spelling`).

        Args:
            narrow: If ``False``, the manifest of the graph also lists Ensembl's mapping sessions with their
                database names, assemblies and dates (read from the full id-history table). The graph itself is the
                same.
            form_list: The forms to include, a list of 'gene', 'transcript', 'translation' that contains 'gene'. If
                ``None``, all forms of the manager.
            narrow_external: If ``False``, every external database of Ensembl is included, not only those selected
                in the organism's externals YAML file. The graph becomes much larger and conversions slower.

        Returns:
            The frozen graph, with its manifest in ``graph.graph["manifest"]``.

        Raises:
            ValueError: If the Ensembl ids of the organism carry no versions in some release (such organisms are
                not supported; raised by the manager when the ids are read, before anything else is done), or if
                the tables contradict each other (an edge to a node that does not exist, an external table that
                names an Ensembl id which the primary assembly does not have).
            HistoryValidationError: If a mapping session cannot be reconciled with the id tables.
        """
        dm = self.db_manager
        form_list = list(dm.available_form_of_interests if not form_list else form_list)
        if DB.backbone_form not in form_list:
            raise ValueError(f"The forms of a graph must include {DB.backbone_form!r}, got {form_list}.")
        primary = int(dm.genome_assembly)
        releases = self.graph_releases()

        # Check the externals YAML before the (long) history build: a missing file stops here with a message that
        # names the file to create, and releases without an entry are announced once.
        externals_releases = self._releases_with_externals()

        # The ids of every form at every release are read first. An organism whose Ensembl ids carry no versions is
        # refused here, before any work: the manager does not serve the id table of such a release and says so.
        names_of_form: dict[str, set[str]] = {}
        for form in form_list:
            self.log.info(f"Ids are being collected: {form}")
            collected: set[str] = set()
            for release in releases:
                manager = self._backbone_manager(form, release)
                if form == DB.backbone_form:
                    self._gene_ids(manager, save=False)  # kept in memory for the history
                    continue
                table = manager.get_db("ids", save_after_calculation=False)
                self._table_counts[("ids", "tables")] += 1
                self._table_counts[("ids", "rows")] += len(table)
                collected.update(self._names(table))
            names_of_form[form] = collected

        # Nodes. A provisional number per node; the final numbers follow the sorted names.
        number: dict[str, int] = {}
        names: list[str] = []
        types: list[str] = []

        def add_node(name: str, node_type: str) -> int:
            found = number.get(name)
            if found is None:
                found = number[name] = len(names)
                names.append(name)
                types.append(node_type)
            return found

        history_edges, history_report = self.history(self._form_manager(DB.backbone_form), narrow=narrow)
        gene_nodes = {n for edge in history_edges for n in edge[:2]}
        for name in sorted(gene_nodes):
            add_node(name, DB.nts_ensembl[DB.backbone_form])
        for form in form_list:
            if form == DB.backbone_form:
                continue
            found = names_of_form.pop(form)
            clash = sorted(n for n in found if n in number)
            if clash:
                # Fun fact: There are exceptional Ensembl protein IDs that starts with 'ENST', and sometimes the
                # same ID is defined in two forms (in different releases), e.g. "ENST00000515292.1".
                self.log.warning(
                    f"Intersecting Ensembl nodes: {len(clash)} names of earlier forms are also {form} ids and "
                    f"become {form} nodes: '{', '.join(clash[:20])}'."
                )
                for name in clash:
                    types[number[name]] = DB.nts_ensembl[form]
            for name in sorted(found):
                add_node(name, DB.nts_ensembl[form])
        for name in names:
            GraphMaker.split_id(name, "ID")  # every Ensembl node is `<id>.<version>`

        # Genes outside the reference sequence of the assembly that serves each release: {provisional number: mask}.
        non_reference = self._non_reference_genes(releases, number, types)

        # Cross-reference edges: {(database, assembly): {source << 32 | target: release mask}}.
        edges: dict[tuple[str, int], dict[int, int]] = {}

        def add_edges(sources: list[int], targets: list[int], database: str, assembly: int, release: int) -> None:
            if not sources:
                return
            bit = 1 << int(release)
            keys = np.unique((np.asarray(sources, dtype=np.int64) << 32) | np.asarray(targets, dtype=np.int64))
            group = edges.setdefault((str(database), int(assembly)), {})
            for key in keys.tolist():
                group[key] = group.get(key, 0) | bit

        # Relations between the forms: transcript -> gene, translation -> transcript.
        relation_pairs: list[tuple[str, str]] = []
        if "transcript" in form_list and "gene" in form_list:
            relation_pairs.append(("transcript", "gene"))
        if "translation" in form_list and "transcript" in form_list:
            relation_pairs.append(("translation", "transcript"))
        if relation_pairs:
            self.log.info("Establishing connection between different forms.")
            for release in releases:
                manager = self._backbone_manager(form_list[0], release)
                table = manager.get_db("relationcurrent", save_after_calculation=manager.store_raw_always)
                self._table_counts[("relationcurrent", "tables")] += 1
                self._table_counts[("relationcurrent", "rows")] += len(table)
                for lower, upper in relation_pairs:
                    pairs = self._pairs(table, lower, upper)
                    sources, targets = [], []
                    for first, second in pairs:
                        source, target = number.get(first), number.get(second)
                        if source is None or target is None:
                            raise ValueError(
                                f"Missing graph node(s) while adding edge: {lower}={first!r} → {upper}={second!r} "
                                f"(release {release})."
                            )
                        sources.append(source)
                        targets.append(target)
                    add_edges(sources, targets, DB.nts_ensembl[lower], int(manager.genome_assembly), release)

        # The gene ids of every assembly and release: the base ids they imply are Ensembl ids, never external ones.
        assembly_releases = self._assembly_releases()
        gene_tables: dict[tuple[int, int], dict[str, Any]] = {}
        reserved: set[str] = set()
        for assembly, its_releases in assembly_releases.items():
            for release in its_releases:
                manager = self._assembly_manager(DB.backbone_form, assembly, release)
                gene_tables[(assembly, release)] = self._gene_ids(manager, save=True)
                reserved.update(gene_tables[(assembly, release)])

        # External identifiers.
        before_externals = len(names)
        organism_assemblies = DB.assemblies_by_priority(dm.organism)
        added_by_assembly: dict[int, set[str]] = {a: set() for a in organism_assemblies if a != primary}
        first_release_of_new_nodes: dict[int, int] = {}
        new_assembly_nodes: list[tuple[str, str, int]] = []
        misplaced: set[str] = set()
        for form in form_list:
            self.log.info(f"Edges between external IDs to Ensembl IDs is being added for '{form}'.")
            new_nodes_count = 0
            for release in releases:
                if release not in externals_releases:
                    # Announced once by `_releases_with_externals`; the Ensembl-only history graph still covers it.
                    continue
                manager = self._backbone_manager(form, release)
                table = manager.create_external_all(return_mode="all", narrow_external=narrow_external)
                self._table_counts[("external", "tables")] += 1
                self._table_counts[("external", "rows")] += len(table)
                if not len(table):
                    continue
                assemblies = [int(a) for a in table["assembly"].tolist()]
                unknown = sorted(set(assemblies) - set(organism_assemblies))
                if unknown:
                    raise ValueError(
                        f"Unknown assembly {unknown[0]!r} in the external table of release {release}. "
                        f"Assemblies of {dm.organism}: {sorted(organism_assemblies)}."
                    )
                valid = np.asarray(assemblies) != 0
                for column in ("graph_id", "id_db", "name_db", "release"):
                    cells = table[column]
                    valid &= (cells.notna() & (cells != "") & (cells != 0)).to_numpy()
                table = table[valid]
                grouped: dict[tuple[str, int, int], tuple[list[int], list[int]]] = {}
                for ensembl_id, external_id, row_release, database, assembly in zip(
                    table["graph_id"].tolist(),
                    table["id_db"].astype(str).str.strip().tolist(),
                    table["release"].tolist(),
                    table["name_db"].tolist(),
                    [int(a) for a in table["assembly"].tolist()],
                ):
                    if not external_id:
                        continue
                    target = number.get(ensembl_id)
                    if target is None or target >= before_externals:
                        # An Ensembl id that the backbone does not have: it exists on another assembly only. For
                        # Homo sapiens these are GRCh37 ids that were retired before GRCh38 or changed there. They
                        # are added once, as nodes of that assembly: an external id may bridge them to the backbone.
                        # For example, 'ENSG00000148828.5', 'ENSG00000167765.3'.
                        if assembly == primary:
                            raise ValueError(
                                f"The external table of release {release} names the Ensembl id {ensembl_id!r} on "
                                f"the primary assembly {primary}, but the id tables of the graph do not have it."
                            )
                        if any(ensembl_id in added_by_assembly[a] for a in added_by_assembly if a != assembly):
                            raise ValueError(
                                f"Node {ensembl_id!r} should have been already added by a higher priority assembly."
                            )
                        if ensembl_id not in added_by_assembly[assembly]:
                            if first_release_of_new_nodes.setdefault(assembly, release) != release:
                                raise ValueError(
                                    f"Assembly {assembly} brings the Ensembl id {ensembl_id!r} at release {release}, "
                                    f"which the backbone of the graph does not have, after its first release with "
                                    f"such ids ({first_release_of_new_nodes[assembly]}). A graph whose primary "
                                    f"assembly is {primary} cannot hold the later history of assembly {assembly}: "
                                    "build the graph with the organism's default primary assembly."
                                )
                            added_by_assembly[assembly].add(ensembl_id)
                            GraphMaker.split_id(ensembl_id, "ID")
                            target = add_node(ensembl_id, DB.nts_assembly[assembly][form])
                            types[target] = DB.nts_assembly[assembly][form]
                            new_assembly_nodes.append((ensembl_id, form, assembly))
                            new_nodes_count += 1
                        target = number[ensembl_id]
                    source = number.get(external_id)
                    if (source is not None and source < before_externals) or external_id in reserved:
                        # Some external databases list an Ensembl id as their identifier. A name that is an Ensembl
                        # node, or the base id of a gene, stays that Ensembl id; the entry is recorded and skipped.
                        misplaced.add(external_id)
                        continue
                    if source is None:
                        source = add_node(external_id, DB.nts_external)
                    pair = grouped.get((database, assembly, int(row_release)))
                    if pair is None:
                        pair = grouped[(database, assembly, int(row_release))] = ([], [])
                    pair[0].append(source)
                    pair[1].append(target)
                for (database, assembly, row_release), (sources, targets) in grouped.items():
                    add_edges(sources, targets, database, assembly, row_release)
            if new_nodes_count > 0:
                self.log.warning(f"New nodes added as assembly nodes: {new_nodes_count}")
        if misplaced:
            self.log.warning(f"Misplaced external entry: {len(misplaced)}.")

        # The other forms of the Ensembl ids that exist on another assembly only, and their relations.
        if new_assembly_nodes:
            self.log.info("Different forms of assembly-Ensembl-nodes are being connecting.")
            added_edges = 0
            for assembly in sorted({a for _, _, a in new_assembly_nodes}):
                wanted = {f: {n for n, form, a in new_assembly_nodes if a == assembly and form == f} for f in form_list}
                for release in sorted(assembly_releases.get(assembly, []), reverse=True):
                    table = self._assembly_manager(form_list[0], assembly, release).get_db("relationcurrent")
                    self._table_counts[("relationcurrent", "tables")] += 1
                    self._table_counts[("relationcurrent", "rows")] += len(table)
                    keep = np.zeros(len(table), dtype=bool)
                    for form in form_list:
                        keep |= table[form].isin(wanted[form]).to_numpy()
                    table = table[keep]
                    for form in form_list:
                        for name in table[form].tolist():
                            if _present(name) and name not in number:
                                GraphMaker.split_id(name, "ID")
                                add_node(name, DB.nts_assembly[assembly][form])
                    for lower, upper in relation_pairs:
                        pairs = self._pairs(table, lower, upper)
                        known = edges.get((DB.nts_assembly[assembly][lower], assembly), {})
                        added_edges += sum(1 for a, b in pairs if (number[a] << 32 | number[b]) not in known)
                        add_edges(
                            [number[a] for a, _ in pairs],
                            [number[b] for _, b in pairs],
                            DB.nts_assembly[assembly][lower],
                            assembly,
                            release,
                        )
            if added_edges > 0:
                self.log.warning(f"New edges are added between assembly Ensembl nodes: {added_edges}")

        # Versionless base gene ids, for every assembly and release of the graph.
        self.log.info("Versionless Ensembl IDs are being connected.")
        base_type = DB.nts_base_ensembl[DB.backbone_form]
        for (assembly, release), table in gene_tables.items():
            sources, targets = [], []
            for stable_id, version in table.items():
                name = DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(stable_id, version))
                target = number.get(name)
                if target is None:
                    if assembly == primary:
                        raise ValueError(
                            f"The gene {name!r} of the primary assembly {assembly} at release {release} is not a "
                            "node of the graph."
                        )
                    continue  # an id of another assembly that no external id refers to
                source = number.get(stable_id)
                if source is None:
                    source = add_node(stable_id, base_type)
                elif types[source] != base_type:
                    raise ValueError(
                        f"The base gene id {stable_id!r} is already a node of type {types[source]!r}; a base id "
                        "must not be the name of another kind of node."
                    )
                sources.append(source)
                targets.append(target)
            add_edges(sources, targets, base_type, assembly, release)

        # One node per external identifier, whatever its letter case; the canonical spelling is its name.
        backbone_assembly = {int(r): int(self._backbone_manager(DB.backbone_form, r).genome_assembly) for r in releases}
        merge_into, spellings_by_number = self._merge_spellings(names, types, edges, backbone_assembly)

        # Final node numbers: sorted names of the nodes that remain.
        keep_numbers = [i for i in range(len(names)) if i not in merge_into]
        keep_numbers.sort(key=lambda i: names[i])
        final_of = np.full(len(names), -1, dtype=np.int64)
        final_of[keep_numbers] = np.arange(len(keep_numbers))
        for old, new in merge_into.items():
            final_of[old] = final_of[new]
        final_names = [names[i] for i in keep_numbers]
        node_types = sorted({types[i] for i in keep_numbers})
        type_code = {t: c for c, t in enumerate(node_types)}
        ntype = np.asarray([type_code[types[i]] for i in keep_numbers], dtype=np.int16)

        # Cross-reference edges with their connections; databases and assemblies in sorted order.
        self.log.info("Edges are being put together.")
        items_of_edge: dict[int, tuple] = {}
        for database, assembly in sorted(edges):
            for key, mask in edges[(database, assembly)].items():
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
            edges[(database, assembly)] = {}  # release the memory
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
        store.spellings = {(int(final_of[i]), db): entry for (i, db), entry in spellings_by_number.items()}
        store.non_reference = {int(final_of[i]): ReleaseSet.from_mask(mask) for i, mask in non_reference.items()}
        store.build_name_indices()
        collisions = store.lower_collisions()
        if collisions:
            raise ValueError(f"Node names that differ only in letter case remain in the graph: {collisions[:5]}.")

        new_form = "-".join(form_list)
        attributes: dict[str, Any] = {
            "name": f"{dm.organism}_{dm.ensembl_release}_{new_form}",
            "type": new_form,
            "ensembl_release": int(dm.ensembl_release),
            "genome_assembly": primary,
            "organism": dm.organism,
            "confident_for_release": releases,
            "narrow": narrow,
            "narrow_external": narrow_external,
            "misplaced_external_entry": misplaced,
            "backbone_assembly": backbone_assembly,
        }
        graph = TheGraph._from_store(store, attributes, form_list)

        # What every assembly's edges say must lie inside the releases that the organism table gives the assembly.
        table = DB.assembly_mysqlport_priority[dm.organism]
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
                raise ValueError(
                    f"Edges of assembly {assembly} carry releases outside the assembly's range in the organism "
                    f"table ({lowest} to {highest}) or outside the graph: {outside}."
                )

        manifest = self.graph_identity(narrow=narrow, form_list=form_list, narrow_external=narrow_external)
        manifest.update(
            {
                "format_version": GRAPH_FORMAT_VERSION,
                "package_version": _package_version(),
                "build_date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "releases": releases,
                "releases_per_assembly": {str(a): r for a, r in releases_per_assembly.items()},
                "backbone_assembly": {str(r): a for r, a in backbone_assembly.items()},
                "tables": {
                    kind: {"tables": self._table_counts[(kind, "tables")], "rows": self._table_counts[(kind, "rows")]}
                    for kind in ("ids", "relationcurrent", "external", "idhistory", "generegion")
                },
                "source_tables": {
                    str(assembly): self._source_tables_fingerprint(assembly) for assembly in sorted(assembly_releases)
                },
                "counts": {
                    "nodes": len(final_names),
                    "history_edges": int(len(h_src)),
                    "cross_reference_edges": int(len(x_src)),
                    "connections": len(connections),
                    "merged_spellings": len(merge_into),
                    "misplaced_external_entries": len(misplaced),
                    "non_reference_genes": len(non_reference),
                },
                "history": history_report,
                "content_digest": _content_digest(store),
            }
        )
        graph.graph["manifest"] = manifest
        graph.calculate_caches(for_test=False)
        _ = graph.external_database_connection_form
        return graph

    def _non_reference_genes(self, releases: list[int], number: dict[str, int], types: list[str]) -> dict[int, int]:
        """Collect the genes that lie outside the reference sequence, with the releases at which they do.

        For every release the ``generegion`` table of the assembly that serves the release in the backbone is read
        (:py:meth:`idtrack.DatabaseManager.create_gene_region`): genes on alternative loci, on
        assembly patches and LRG records. They are nodes of the backbone like every gene; the mark lets a conversion
        tell a gene of the reference sequence from its copies (:py:meth:`idtrack.TheGraph.is_non_reference`).

        Args:
            releases: The releases of the graph.
            number: Node number by node name; holds the backbone genes.
            types: Node type by node number.

        Returns:
            dict[int, int]: ``{node number: release bit mask}`` for the marked genes only.

        Raises:
            ValueError: If a marked gene of a release is not a node of the graph.
        """
        self.log.info("Genes outside the reference sequence are being collected.")
        backbone_type = DB.nts_ensembl[DB.backbone_form]
        marked: dict[int, int] = {}
        for release in releases:
            table = self._backbone_manager(DB.backbone_form, release).get_db("generegion")
            self._table_counts[("generegion", "tables")] += 1
            self._table_counts[("generegion", "rows")] += len(table)
            table = table[table["non_reference"].to_numpy(dtype=bool)]
            for name in self._names(table[["gene_stable_id", "gene_version"]]):
                node = number.get(name)
                if node is None:
                    raise ValueError(
                        f"The gene {name!r}, which release {release} has outside the reference sequence, is not a "
                        "node of the graph."
                    )
                if types[node] == backbone_type:  # not a name that another form took over (announced above)
                    marked[node] = marked.get(node, 0) | (1 << int(release))
        return marked

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
        backbone_assembly: dict[int, int],
    ) -> tuple[dict[int, int], dict[tuple[int, str], tuple[str, tuple[tuple[int, str], ...]]]]:
        """Choose one node for the external identifiers that are equal in lower case, and keep how each is written.

        A release belongs to the assembly that serves it in the backbone (``backbone_assembly``). What another
        assembly says at that release does not count while a serving assembly says anything: human GRCh37 is
        frozen and still listed at every release that GRCh38 serves, with the spellings of the time it was frozen.
        Only an identifier (or a database of it) that no serving assembly has at any release is judged by all its
        assemblies.

        The node keeps the spelling that Ensembl gives at the latest release in which the identifier exists. Where
        several spellings exist at that release, the one carried by more (database, Ensembl id) links at that
        release wins, then the one with more upper-case letters, then the first in sort order.

        For every database of the identifier the spelling of each release is kept as well (the same rule within the
        database: more links of the database at the release, more upper-case letters, sort order), so that a
        conversion to a release returns the identifier as the database wrote it at that release
        (:py:meth:`idtrack.TheGraph.spelling`).

        Args:
            names: Node names by provisional number.
            types: Node types by provisional number.
            edges: The cross-reference edges collected so far.
            backbone_assembly: The assembly that serves each release of the graph.

        Returns:
            ``{number of a merged spelling: number of the node that stays}`` and, only for the databases that write
            the identifier otherwise than the node is named at some release, ``{(number of the node that stays,
            database): (spelling of the database at its latest release, ((release mask, spelling), ...))}``; the
            pairs name the releases at which the database wrote it otherwise than at its latest release.

        Raises:
            ValueError: If a name that differs only in case from another one is not an external identifier.
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
            if types[i] != DB.nts_external:
                raise ValueError(
                    f"The node {names[i]!r} of type {types[i]!r} differs only in letter case from another node; "
                    "only external identifiers can be merged."
                )
        served = serving_line(backbone_assembly)

        # Per spelling and database, the release mask of every link: on the serving assemblies, and on any assembly.
        on_line: dict[int, dict[str, list[int]]] = {i: {} for i in colliding}
        anywhere: dict[int, dict[str, list[int]]] = {i: {} for i in colliding}
        for (database, assembly), group in edges.items():
            line = served[assembly].mask if assembly in served else 0
            for key, mask in group.items():
                source = key >> 32
                if source in colliding:
                    anywhere[source].setdefault(database, []).append(mask)
                    if mask & line:
                        on_line[source].setdefault(database, []).append(mask & line)

        def upper(i: int) -> int:
            return sum(1 for c in names[i] if c.isupper())

        def rank(masks: list[int], i: int) -> tuple[int, int, int]:
            """Latest release of the links, number of links at it, upper-case letters of the spelling."""
            union = 0
            for mask in masks:
                union |= mask
            latest = union.bit_length() - 1
            return latest, sum(1 for mask in masks if mask >> latest & 1) if latest >= 0 else 0, upper(i)

        merge_into: dict[int, int] = {}
        spellings: dict[tuple[int, str], tuple[str, tuple[tuple[int, str], ...]]] = {}
        for same_lower in groups:
            ordered = sorted(same_lower, key=lambda i: names[i])
            counted = on_line if any(on_line[i] for i in ordered) else anywhere
            all_links = {i: [mask for masks in counted[i].values() for mask in masks] for i in ordered}
            winner = max(ordered, key=lambda i: rank(all_links[i], i))  # the first of equals: first in sort order
            for i in ordered:
                if i != winner:
                    merge_into[i] = winner
            for database in sorted({db for i in ordered for db in anywhere[i]}):
                links = {i: counted[i][database] for i in ordered if database in counted[i]}
                if not links:  # a database that has the identifier on no serving assembly
                    links = {i: anywhere[i][database] for i in ordered if database in anywhere[i]}
                at_latest = max(links, key=lambda i: rank(links[i], i))
                # The spelling of every release: among the spellings the database has there, by its links there.
                union = 0
                for masks in links.values():
                    for mask in masks:
                        union |= mask
                otherwise: dict[int, int] = {}
                for release in ReleaseSet.from_mask(union):
                    counts = {i: sum(1 for mask in masks if mask >> release & 1) for i, masks in links.items()}
                    chosen = max((i for i in links if counts[i]), key=lambda i: (counts[i], upper(i)))
                    if chosen != at_latest:
                        otherwise[chosen] = otherwise.get(chosen, 0) | (1 << release)
                if at_latest != winner or otherwise:
                    spellings[(winner, database)] = (
                        names[at_latest],
                        tuple(
                            (mask, names[i]) for i, mask in sorted(otherwise.items(), key=lambda item: names[item[0]])
                        ),
                    )
        self.log.info(f"Number of removed nodes in the process of merging synonymous nodes: {len(merge_into)}")
        return merge_into, spellings

    def _releases_with_externals(self) -> set[int]:
        """Return the releases of the graph that the organism's externals YAML covers, warning once about the rest.

        Releases of :py:meth:`graph_releases` that the YAML does not list get no external identifiers in the graph.
        That is announced here once, as a :py:class:`UserWarning` and in the log, with the way to fix it
        (:py:meth:`idtrack.ExternalDatabases.extend_modified_yaml`), instead of being skipped
        quietly release by release.

        Returns:
            set[int]: Releases whose external identifiers are added to the graph.

        Raises:
            FileNotFoundError: If the organism has no externals YAML file at all (the message names the file to
                create and how).
        """
        external_inst = self.db_manager.external_inst
        covered = external_inst.covered_releases()
        releases = self.graph_releases()
        missing = [r for r in releases if r not in covered]
        if missing:
            path = external_inst._modified_yaml_path()
            if covered:
                msg = (
                    f"The externals YAML `{path}` has no entry for {self.db_manager.organism} Ensembl releases "
                    f"{missing}: the graph gets no external identifiers (symbols, NCBI gene, UniProt, ...) for these "
                    "releases. Extend the file with `ExternalDatabases.extend_modified_yaml` (tutorial 02) and "
                    "rebuild the graph."
                )
            else:
                msg = (
                    f"The externals YAML `{path}` has no entry for {self.db_manager.organism}: the graph is built "
                    "without external identifiers."
                )
            warnings.warn(msg, UserWarning, stacklevel=3)
            self.log.warning(msg)
        return {r for r in releases if r in covered}

    # ------------------------------------------------------------------ the gene history

    def history(self, db_manager: DatabaseManager, narrow: bool = True) -> tuple[list[tuple], dict[str, Any]]:
        """Build the history edges of one form from the id tables of the releases and Ensembl's id-history table.

        The id tables say which ``<id>.<version>`` exists in which release (each release from the assembly that
        serves it in the backbone). From them come the edges of an id with itself: ``<id>.Void → <id>.<v>`` at its
        first appearance, ``<id>.<v> → <id>.<w>`` at a change of version, ``<id>.<v> → <id>.Retired`` when it
        disappears, ``<id>.Retired → <id>.<v>`` when it comes back, and a self-loop to infinity on every id of the
        last release.

        Ensembl's ``stable_id_event`` table says which id passed into which other id (and which id was re-mapped
        onto itself with a score below 1, or onto itself in a session that also maps it to or from another id), per
        mapping session. The releases of a session are not taken from
        ``mapping_session``: an event from ``A`` to ``B`` belongs to the pair of consecutive releases at which ``A``
        exists in the first and ``B`` in the second. For every session the pair that most of its events fit is
        determined; it replaces the recorded pair where the two differ (Ensembl's pig session 106 to 107 is recorded
        as 106 to 104; sessions over a gap such as rhesus 50 to 52 belong to 51 to 52). An event that does not fit
        its session's pair is placed at the nearest pair it fits; an event that fits no pair, or whose ids are in no
        id table, contradicts the tables and is left out. All of it is counted in the report.

        Args:
            db_manager: The manager of the form, at the snapshot release.
            narrow: Read the narrow id-history table (``True``) or the full one, whose session metadata then goes
                into the report.

        Returns:
            The edges as ``(source, target, old_release, new_release, weight, key)`` tuples (``new_release`` is
            infinity on the self-loops of the last release), sorted, and the report for the manifest: the sessions
            with recorded and used release pairs and their counts, and the corrections.

        Raises:
            HistoryValidationError: If the events of a session fit no pair of consecutive releases, fit several
                pairs equally well without the recorded one among them, or if the id-history table does not reach
                back to the first release of the graph although Ensembl's ``mapping_session`` does (a table computed
                for another release window).
            ValueError: If an id of the last release is missing from the backbone, or if the ids of a release
                carry no versions (raised by the manager, see :py:meth:`construct_graph`).
        """
        self.log.info(f"History is being built: {db_manager.form}")
        form = db_manager.form
        releases = self.graph_releases()
        position = {r: i for i, r in enumerate(releases)}
        last = len(releases) - 1

        # Which id has which version in which release, and in which releases each `<id>.<version>` exists.
        timeline: dict[str, list[tuple[int, Any]]] = {}
        alive: dict[str, int] = {}
        for i, release in enumerate(releases):
            manager = self._backbone_manager(form, release)
            table = (
                self._gene_ids(manager, save=False)
                if form == DB.backbone_form
                else {a: b for a, b in manager.get_db("ids", save_after_calculation=False).values}
            )
            for stable_id, version in table.items():
                timeline.setdefault(stable_id, []).append((i, version))
                name = DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(stable_id, version))
                alive[name] = alive.get(name, 0) | (1 << i)

        def node(stable_id: str, version: Any) -> str:
            return DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(stable_id, version))

        # Ensembl's events.
        table = db_manager.get_db(df_indicator="idhistory_narrow" if narrow else "idhistory").copy()
        self._table_counts[("idhistory", "tables")] += 1
        self._table_counts[("idhistory", "rows")] += len(table)
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
        session_columns = ["mapping_session_id"] if not narrow and "mapping_session_id" in table else ["old_release", "new_release"]  # fmt: skip

        # An id that a session maps onto itself is an event of the history in two cases: the mapping is not perfect
        # (a score below 1), or the same session also maps the id to or from another id. In the second case the
        # self event is what says that the id lived on beside the other one: an id that exists on both sides of a
        # session and is mapped only to other ids did not, and without the self event the two cannot be told apart.
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

        # Scores of version changes; pairs that Ensembl scores twice are ignored (an annotation mistake).
        weights: dict[tuple, float] = {}
        twice: set[tuple] = set()
        for a, av, b, bv, score in table.loc[is_weight, ["old_stable_id", "old_version", "new_stable_id", "new_version", "score"]].itertuples(index=False, name=None):  # fmt: skip
            key = (a, int(av), b, int(bv))
            if key in weights or key in twice:
                twice.add(key)
                weights.pop(key, None)
            else:
                weights[key] = float(score)
        if twice:
            self.log.warning(f"Edge weights ignored due to duplicate entries: {len(twice)}.")

        # Sessions: the release pair of each from the id tables.
        events = table.loc[is_event]
        sessions: dict[tuple, dict[str, Any]] = {}
        placed: list[tuple[str, str, tuple, float]] = []
        for row in events.itertuples(index=False):
            session_key = tuple(getattr(row, c) for c in session_columns)
            session = sessions.get(session_key)
            if session is None:
                session = sessions[session_key] = {
                    "recorded": [_release_json(row.old_release), _release_json(row.new_release)],
                    "events": 0, "absent": 0, "votes": Counter(), "rows": [],
                }  # fmt: skip
                if not narrow:
                    for column in ("mapping_session_id", "old_db_name", "new_db_name", "old_assembly", "new_assembly", "created"):  # fmt: skip
                        if column in table:
                            session[column] = _plain(getattr(row, column))
            session["events"] += 1
            old = node(row.old_stable_id, int(row.old_version))
            new = node(row.new_stable_id, int(row.new_version))
            if old not in alive or new not in alive:
                # For example, 'ENSG00000289022' is defined in release 105 according to the history but is in no
                # gene table (neither 104, 105, 106). Such ids are not nodes.
                session["absent"] += 1
                continue
            fits = alive[old] & (alive[new] >> 1)  # bit i: `old` exists at releases[i] and `new` at releases[i + 1]
            session["rows"].append((old, new, fits, float(row.score) if not pd.isna(row.score) else math.nan))
            if old != new:
                session["votes"][fits] += 1

        edges: list[tuple] = []
        loop_release: dict[str, tuple[int, int]] = {}
        report_sessions: list[dict[str, Any]] = []
        corrections: list[dict[str, Any]] = []
        for session_key in sorted(sessions, key=lambda k: tuple((isinstance(v, str), v) for v in k)):
            session = sessions[session_key]
            recorded = session["recorded"]
            recorded_pair = None
            if (
                recorded[0] in position
                and recorded[1] in position
                and position[recorded[1]] == position[recorded[0]] + 1
            ):
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
            if not voters:
                used = recorded_pair
            elif recorded_pair in winners:
                used = recorded_pair
            else:
                # Several pairs explain the session equally well (its ids live through them): the pair nearest to
                # the recorded old release, the earlier one of two at the same distance.
                anchor = recorded[0] if recorded[0] is not None else releases[winners[0]]
                used = min(winners, key=lambda i: (abs(releases[i] - anchor), i))
            made = moved = contradicting = 0
            for old, new, fits, score in session["rows"]:
                if used is not None and fits >> used & 1:
                    pair = used
                elif fits:
                    candidates = [i for i in range(last) if fits >> i & 1]
                    pair = min(candidates, key=lambda i: (abs(i - used), i)) if used is not None else candidates[-1]
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
        if form == DB.backbone_form:
            self._check_history_reaches_back(db_manager, releases, table["old_release"])

        # The history of every id with itself, from the id tables.
        void, retired = DB.no_old_node_id, DB.no_new_node_id
        for stable_id, seen in timeline.items():
            previous_index: int | None = None
            previous_version: Any = None
            first_run = True
            for i, version in seen:
                if previous_index is None or i != previous_index + 1:
                    if previous_index is not None:  # it had disappeared after `previous_index`
                        edges.append(
                            (node(stable_id, previous_version), node(stable_id, retired), releases[previous_index], releases[previous_index + 1], math.nan)  # fmt: skip
                        )
                    origin = void if first_run else retired
                    old_release = releases[i - 1] if i > 0 else releases[0]
                    edges.append(
                        (node(stable_id, origin), node(stable_id, version), old_release, releases[i], math.nan)
                    )
                    first_run = False
                elif version != previous_version:
                    weight = weights.get((stable_id, _int_or(previous_version), stable_id, _int_or(version)), math.nan)
                    edges.append(
                        (
                            node(stable_id, previous_version),
                            node(stable_id, version),
                            releases[i - 1],
                            releases[i],
                            weight,
                        )
                    )
                previous_index, previous_version = i, version
            if previous_index is not None and previous_index != last:
                edges.append(
                    (node(stable_id, previous_version), node(stable_id, retired), releases[previous_index], releases[previous_index + 1], math.nan)  # fmt: skip
                )

        # Self-loops to infinity for the ids of the last release: from the start of their last stay, or from the
        # release of their latest re-mapping if that is later.
        latest_table = db_manager.get_db("ids", save_after_calculation=False)
        for name in self._names(latest_table):
            mask = alive.get(name, 0)
            if not mask >> last & 1:
                raise ValueError(
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

        # Deterministic order and keys of parallel edges; the invariant of the history.
        edges.sort(key=lambda e: (e[0], e[1], e[2], e[3], math.inf if math.isnan(e[4]) else e[4]))
        keyed: list[tuple] = []
        previous: tuple[str, str] | None = None
        parallel = 0
        for old, new, old_release, new_release, weight in edges:
            parallel = parallel + 1 if previous == (old, new) else 0
            previous = (old, new)
            keyed.append((old, new, old_release, new_release, weight, parallel))
            if new_release == math.inf or old.endswith(f".{void}") or old.endswith(f".{retired}") or new.endswith(f".{retired}"):  # fmt: skip
                continue
            a, b = position[old_release], position[new_release]
            if not (b == a + 1 and alive[old] >> a & 1 and alive[new] >> b & 1):
                raise HistoryValidationError(
                    f"History edge {old} → {new} ({old_release} to {new_release}) contradicts the id tables."
                )
        report = {
            "sessions": report_sessions,
            "corrections": corrections,
            "events": int(sum(s["events"] for s in report_sessions)),
            "event_edges": int(sum(s["edges"] for s in report_sessions)),
            "events_of_ids_in_no_table": int(sum(s["absent"] for s in report_sessions)),
            "events_moved_to_another_pair": int(sum(s["moved"] for s in report_sessions)),
            "events_contradicting_tables": int(sum(s["contradicting"] for s in report_sessions)),
        }
        return keyed, report

    def _check_history_reaches_back(self, db_manager: DatabaseManager, releases: list[int], old_releases) -> None:
        """Stop if the id-history table starts later than Ensembl's history does within the window of the graph.

        The id-history table is kept in the local repository and is computed for a release window. A table that was
        computed for a later window lacks the early sessions, and a graph built from it silently has no history
        there. Ensembl's ``mapping_session`` table of the snapshot release lists all sessions; when it has sessions
        in the window before the first one of the id-history table, the events of those sessions are counted in
        Ensembl's ``stable_id_event`` table, and events of this form there mean that the stored table is not the
        table of this window.

        Args:
            db_manager: The manager of the form at the snapshot release.
            releases: The releases of the graph.
            old_releases: The ``old_release`` column of the id-history table.

        Raises:
            HistoryValidationError: If Ensembl has events of the form in sessions that the table lacks.
        """
        have = sorted({int(r) for r in pd.to_numeric(old_releases, errors="coerce").dropna().tolist()})
        sessions = db_manager.get_table("mapping_session", save_after_calculation=db_manager.store_raw_always)
        if sessions is None or not len(sessions) or "old_release" not in sessions:
            return
        old = pd.to_numeric(sessions["old_release"], errors="coerce")
        early = sessions[(old >= releases[0]) & (old <= releases[-1]) & (old < (min(have) if have else math.inf))]
        if not len(early):
            return
        events = db_manager.get_table("stable_id_event", save_after_calculation=db_manager.store_raw_always)
        lost = events[
            events["mapping_session_id"].isin(early["mapping_session_id"]) & (events["type"] == db_manager.form)
        ]
        if len(lost):
            raise HistoryValidationError(
                f"The id-history table of {db_manager.organism} in the local repository starts at release "
                f"{min(have) if have else 'none'}, but Ensembl has {len(lost)} {db_manager.form} events in mapping "
                f"sessions from release {int(old[early.index].min())} on, inside the window of the graph. The stored "
                "table was computed for another release window: remove it (or use an empty local repository) and "
                "build again."
            )

    def construct_graph_form(self, narrow: bool, db_manager: DatabaseManager) -> TheGraph:
        """Return the history of one form as a graph of its own (not frozen); see :py:meth:`history`.

        Args:
            narrow: See :py:meth:`construct_graph`.
            db_manager: The manager of the form, at the snapshot release.

        Returns:
            A graph with the history edges (``weight``, ``old_release``, ``new_release``) and the node attributes
            ``node_type``, ``ID`` and ``Version``; the report of the validation is ``graph.graph["history"]``.
        """
        edges, report = self.history(db_manager, narrow=narrow)
        g = TheGraph(
            name=f"{db_manager.organism}_{db_manager.ensembl_release}_{db_manager.form}",
            type=db_manager.form,
            ensembl_release=int(db_manager.ensembl_release),
            genome_assembly=int(db_manager.genome_assembly),
            organism=db_manager.organism,
            confident_for_release=self.graph_releases(),
            narrow=narrow,
            history=report,
        )
        node_type = DB.nts_ensembl[db_manager.form]
        for name in sorted({n for edge in edges for n in edge[:2]}):
            g.add_node(
                name,
                **{
                    DB.node_type_str: node_type,
                    "ID": self.split_id(name, "ID"),
                    "Version": self.split_id(name, "Version"),
                },
            )
        for old, new, old_release, new_release, weight, key in edges:
            g.add_edge(old, new, key, weight=weight, old_release=old_release, new_release=new_release)
        return g

    @staticmethod
    def split_id(id_to_split: str, which_part: str) -> str | float:
        """Return the id or the version of an Ensembl node name ``<id>.<version>``.

        Args:
            id_to_split: The node name.
            which_part: Either 'Version' or 'ID'.

        Returns:
            The requested part of the node name.

        Raises:
            ValueError: If the name does not hold exactly one delimiter, or ``which_part`` is neither 'Version'
                nor 'ID'.
        """
        count = id_to_split.count(DB.id_ver_delimiter)
        if count != 1:
            raise ValueError(
                f"An Ensembl node name must be `<id>{DB.id_ver_delimiter}<version>`; {id_to_split!r} has {count} "
                f"delimiters ('{DB.id_ver_delimiter}')."
            )
        if which_part == "ID":
            return id_to_split.split(DB.id_ver_delimiter)[0]
        elif which_part == "Version":
            return id_to_split.split(DB.id_ver_delimiter)[1]
        else:
            raise ValueError(f"which_part must be 'ID' or 'Version', got {which_part!r}.")

    # ------------------------------------------------------------------ files

    def graph_identity(
        self, narrow: bool = True, form_list: list[str] | None = None, narrow_external: bool = True
    ) -> dict[str, Any]:
        """Return the manifest fields that say which graph this maker builds; a loaded file must have the same.

        Computed from the manager, the organism table and the externals YAML file only (no table is read).

        Args:
            narrow: See :py:meth:`construct_graph`.
            form_list: See :py:meth:`construct_graph`.
            narrow_external: See :py:meth:`construct_graph`.

        Returns:
            dict: ``organism``, ``genome_assembly`` (the primary assembly), ``ignore_before`` and ``ignore_after``
            (the window; ``None`` for no limit), ``snapshot_release``, ``forms``, ``narrow``, ``narrow_external``,
            ``externals_fingerprint`` (the databases selected per form, assembly and release, as far as the
            selection takes effect: inside the window and inside the releases the organism table gives the assembly) and
            ``organism_table_fingerprint`` (the assemblies of the organism with their release ranges and order).
        """
        dm = self.db_manager
        forms = list(dm.available_form_of_interests if not form_list else form_list)
        return {
            "organism": dm.organism,
            "genome_assembly": int(dm.genome_assembly),
            "ignore_before": None if np.isinf(dm.ignore_before) else int(dm.ignore_before),
            "ignore_after": None if np.isinf(dm.ignore_after) else int(dm.ignore_after),
            "snapshot_release": int(dm.ensembl_release),
            "forms": forms,
            "narrow": bool(narrow),
            "narrow_external": bool(narrow_external),
            "externals_fingerprint": self._externals_fingerprint(forms),
            "organism_table_fingerprint": self._organism_table_fingerprint(),
        }

    def _externals_fingerprint(self, forms: list[str]) -> str:
        """Return a digest of the selection of external databases that takes effect in this maker's graph."""
        dm = self.db_manager
        return GraphMaker._selection_fingerprint(
            dm.external_inst.read_modified_yaml(), dm.organism, forms, dm.ignore_before, dm.ignore_after
        )

    @staticmethod
    def _selection_fingerprint(content: dict | None, organism: str, forms: list[str], first: Any, last: Any) -> str:
        """Return a digest of the selection of external databases that takes effect in a graph.

        Of a parsed externals YAML this is, per form, database and assembly, the included releases that lie in the
        window of the graph and in the release range the organism table gives the assembly. An entry that can never
        produce an edge (a database that is not included, a release outside the window, an assembly at releases it
        does not have, an assembly the organism does not have) is not part of the identity of a graph: tidying such
        entries in a YAML file leaves the graphs built with it readable.

        Args:
            content: The parsed YAML, or ``None``.
            organism: Organism of the graph.
            forms: The forms of the graph.
            first: First release of the window (may be infinite).
            last: Last release of the window (may be infinite).

        Returns:
            str: The digest.
        """
        table = DB.assembly_mysqlport_priority[organism]
        selected = (content or {}).get(organism) or {}
        selection = []
        for form in sorted(forms):
            for database in sorted(selected.get(form) or {}):
                per_assembly = ((selected[form][database] or {}).get("Assembly")) or {}
                for assembly in sorted(per_assembly, key=lambda a: int(a)):
                    item = per_assembly[assembly]
                    if not item.get("Include") or int(assembly) not in table:
                        continue
                    lowest, highest = table[int(assembly)]["ReleaseRange"]
                    releases = sorted(
                        r
                        for r in {int(x) for x in str(item["Ensembl release"]).split(",")}
                        if first <= r <= last and lowest <= r and (highest is None or r <= highest)
                    )
                    if releases:
                        selection.append([form, str(database), int(assembly), releases])
        return hashlib.sha256(json.dumps(selection, sort_keys=True).encode("utf-8")).hexdigest()[:20]

    def _organism_table_fingerprint(self) -> str:
        """Return a digest of the organism's assemblies: keys, release ranges, order and database suffixes."""
        return GraphMaker._table_fingerprint(self.db_manager.organism)

    @staticmethod
    def _table_fingerprint(organism: str) -> str:
        """Return a digest of an organism's entry in the organism table (what shapes a graph)."""
        table = DB.assembly_mysqlport_priority[organism]
        content = [
            [int(a), list(table[a]["ReleaseRange"]), int(table[a]["Priority"]), str(table[a].get("DbSuffix", a))]
            for a in sorted(table)
        ]
        return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def load_graph(
        local_repository: str,
        organism: str,
        snapshot_release: int,
        genome_assembly: int | None = None,
        ignore_before: int | None = None,
        *,
        narrow: bool = True,
        form_list: list[str] | None = None,
        narrow_external: bool = True,
    ) -> TheGraph:
        """Load a stored graph without a :py:class:`DatabaseManager` and without any network access.

        The file is found by its name (:py:meth:`graph_file_path`) and checked against what is asked for, as in
        :py:meth:`get_graph`: organism, primary assembly, release window, forms, the selection of external
        databases (the externals YAML of the local repository, else the one shipped with the package) and the
        organism table.

        Args:
            local_repository: Folder that holds the graph file.
            organism: Organism name, e.g. ``homo_sapiens``.
            snapshot_release: Last release of the graph's window (``snapshot_release`` of ``API.build_graph``).
            genome_assembly: Primary assembly; by default the organism's preferred assembly among those whose
                release range holds ``snapshot_release``.
            ignore_before: First release of the window; by default the first release of the organism in the
                organism table (:py:meth:`idtrack.DB.first_release`), as ``API.build_graph``.
            narrow: See :py:meth:`construct_graph`.
            form_list: See :py:meth:`construct_graph`.
            narrow_external: See :py:meth:`construct_graph`.

        Returns:
            The frozen graph.

        Raises:
            FileNotFoundError: If the local repository has no such graph file (build it with :py:meth:`get_graph`).
            GraphMismatchError: If the file holds another graph than the one asked for.
            ValueError: If no assembly of the organism has ``snapshot_release``.
        """
        from idtrack._external_databases import ExternalDatabases

        by_priority = DB.assemblies_by_priority(organism)  # an organism that is not configured is named here
        table = DB.assembly_mysqlport_priority[organism]
        if genome_assembly is None:
            for assembly in by_priority:
                first, last = table[assembly]["ReleaseRange"]
                if first <= snapshot_release and (last is None or snapshot_release <= last):
                    genome_assembly = int(assembly)
                    break
            else:
                raise ValueError(f"No assembly of {organism} has release {snapshot_release} in the organism table.")
        if ignore_before is None:
            ignore_before = DB.first_release(organism)
        forms = list(DB.forms_in_order if not form_list else form_list)
        path = GraphMaker.graph_file_path(
            local_repository,
            organism,
            genome_assembly,
            ignore_before,
            snapshot_release,
            narrow=narrow,
            form_list=None if forms == list(DB.forms_in_order) else forms,
            narrow_external=narrow_external,
        )
        externals = ExternalDatabases(
            organism, int(snapshot_release), DB.backbone_form, local_repository, genome_assembly
        )
        expect = {
            "organism": organism,
            "genome_assembly": int(genome_assembly),
            "ignore_before": int(ignore_before),
            "ignore_after": int(snapshot_release),
            "forms": forms,
            "narrow": bool(narrow),
            "narrow_external": bool(narrow_external),
            "externals_fingerprint": GraphMaker._selection_fingerprint(
                externals.read_modified_yaml(), organism, forms, ignore_before, snapshot_release
            ),
            "organism_table_fingerprint": GraphMaker._table_fingerprint(organism),
        }
        return TheGraph.load(path, expect=expect)

    def get_graph(
        self,
        narrow: bool = True,
        create_even_if_exist: bool = False,
        save_after_calculation: bool = True,
        *,
        form_list: list[str] | None = None,
        narrow_external: bool = True,
    ) -> TheGraph:
        """Load the graph from the local repository, or build it.

        A graph file is loaded only if its manifest is the graph that is asked for: organism, primary assembly,
        release window, snapshot release, forms, the selection of external databases and the organism table. A file
        that holds another graph under the expected name is an error, not a silent substitute and not a silent
        rebuild of several hours.

        Args:
            narrow: See :py:meth:`construct_graph`.
            create_even_if_exist: Build the graph even if a file exists. The new graph replaces the file (unless
                ``save_after_calculation`` is ``False``).
            save_after_calculation: Write a newly built graph to the local repository.
            form_list: See :py:meth:`construct_graph`.
            narrow_external: See :py:meth:`construct_graph`.

        Returns:
            The frozen graph.

        Raises:
            GraphMismatchError: If the file in the local repository is not the graph asked for, or unreadable. The
                message names the differences; ``create_even_if_exist=True`` replaces the file.
        """
        file_path = self.create_file_name(narrow, form_list=form_list, narrow_external=narrow_external)
        if os.path.isfile(file_path) and not create_even_if_exist:
            self.log.info(f"The graph is being read: {file_path}")
            expect = self.graph_identity(narrow=narrow, form_list=form_list, narrow_external=narrow_external)
            return TheGraph.load(file_path, expect=expect)

        self.log.info(f"The graph is being constructed: {file_path}")
        g = self.construct_graph(narrow=narrow, form_list=form_list, narrow_external=narrow_external)
        if save_after_calculation:
            self.log.info(f"The graph is being exported as '{file_path}'.")
            g.save(file_path)
        return g

    def create_file_name(
        self, narrow: bool = True, form_list: list[str] | None = None, narrow_external: bool = True
    ) -> str:
        """Return the path of the graph file in the manager's local repository.

        Args:
            narrow: See :py:meth:`construct_graph`.
            form_list: See :py:meth:`construct_graph`.
            narrow_external: See :py:meth:`construct_graph`.

        Returns:
            Absolute file path; see :py:meth:`graph_file_path`.
        """
        dm = self.db_manager
        return GraphMaker.graph_file_path(
            dm.local_repository,
            dm.organism,
            int(dm.genome_assembly),
            dm.ignore_before,
            dm.ignore_after,
            narrow=narrow,
            form_list=(
                None if form_list is None or list(form_list) == list(dm.available_form_of_interests) else form_list
            ),
            narrow_external=narrow_external,
        )

    @staticmethod
    def graph_file_path(
        local_repository: str,
        organism: str,
        genome_assembly: int,
        ignore_before: int | float,
        ignore_after: int | float,
        narrow: bool = True,
        form_list: list[str] | None = None,
        narrow_external: bool = True,
    ) -> str:
        """Return the path of a graph file; needs no manager and no network.

        The name tells graphs apart that can sit in one folder:
        ``graph_<organism>_assembly<primary assembly>_min<first>_max<last>[_narrow][_forms<a-b>][_extall].npz``.
        What a name does not say (the selection of external databases, the organism table, the format version) is in
        the manifest inside the file and is checked when the file is loaded.

        Args:
            local_repository: Folder of the graph files.
            organism: Organism name, e.g. ``homo_sapiens``.
            genome_assembly: Primary assembly of the graph (key of the organism table).
            ignore_before: First release of the window (infinity or ``None`` for no limit).
            ignore_after: Last release of the window (infinity or ``None`` for no limit).
            narrow: See :py:meth:`construct_graph`.
            form_list: The forms if they are not all forms; ``None`` for all.
            narrow_external: See :py:meth:`construct_graph`.

        Returns:
            Absolute file path.
        """
        min_ext = "" if ignore_before is None or np.isinf(ignore_before) else f"_min{int(ignore_before)}"
        max_ext = "" if ignore_after is None or np.isinf(ignore_after) else f"_max{int(ignore_after)}"
        narrow_ext = "_narrow" if narrow else ""
        forms_ext = f"_forms{'-'.join(form_list)}" if form_list else ""
        external_ext = "" if narrow_external else "_extall"
        name = f"graph_{organism}_assembly{int(genome_assembly)}{min_ext}{max_ext}{narrow_ext}{forms_ext}{external_ext}.npz"
        return os.path.join(local_repository, name)

    @staticmethod
    def list_graph_files(local_repository: str, organism: str | None = None) -> list[dict[str, Any]]:
        """List the graph files of a folder with what their names say.

        Args:
            local_repository: Folder to look in.
            organism: Only graphs of this organism; ``None`` for all.

        Returns:
            list[dict]: One entry per file, sorted by name: ``path``, ``organism``, ``genome_assembly``,
            ``ignore_before``, ``ignore_after`` (``None`` where the name has no limit), ``narrow``, ``forms``
            (``None`` for all forms) and ``narrow_external``.
        """
        found: list[dict[str, Any]] = []
        if not os.path.isdir(local_repository):
            return found
        for name in sorted(os.listdir(local_repository)):
            match = _GRAPH_FILE.match(name)
            if not match or (organism is not None and match.group("organism") != organism):
                continue
            found.append(
                {
                    "path": os.path.join(local_repository, name),
                    "organism": match.group("organism"),
                    "genome_assembly": int(match.group("genome_assembly")),
                    "ignore_before": (
                        None if match.group("ignore_before") is None else int(match.group("ignore_before"))
                    ),
                    "ignore_after": None if match.group("ignore_after") is None else int(match.group("ignore_after")),
                    "narrow": match.group("narrow") is not None,
                    "forms": None if match.group("forms") is None else match.group("forms").split("-"),
                    "narrow_external": match.group("extall") is None,
                }
            )
        return found


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


def _package_version() -> str:
    """Return the version of the installed idtrack package."""
    import idtrack

    return str(getattr(idtrack, "__version__", "unknown"))


def _content_digest(store: _Store) -> str:
    """Return a digest of the content of a graph: node names and types, edges, connections and gene marks.

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
    return digest.hexdigest()


__all__ = ["GraphMaker", "GraphMismatchError", "HistoryValidationError"]
