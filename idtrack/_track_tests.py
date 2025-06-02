#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import itertools
import logging
import os
import random
import time
import traceback
from abc import ABC
from math import ceil
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from tqdm import tqdm

from idtrack._db import DB, EmptyConversionMetricsError
from idtrack._track import Track


class TrackTests(Track, ABC):
    """Developer-facing integrity-test harness for :py:class:`~idtrack.Track`.

    This module defines :py:class:`TrackTests`, a mix-in that adds an extensive
    white-box test suite to a populated :py:class:`idtrack.Track` instance.  The
    class is **for developers only**; it should never be used in production
    pipelines.  Every public method beginning with ``is_`` returns a boolean
    that tells whether a specific invariant holds.  Methods beginning with
    ``history_`` execute heavier, end-to-end conversions and collect rich
    statistics. The class is intended to be *mixin-ed* into a concrete Track
    subclass—or instantiated standalone—**after** the underlying graph and
    lookup tables have been fully built.  It performs purely read-only
    operations and therefore imposes no risk of mutating state.

    All test methods share the following contract:

    * They never raise on failure—*return-value only*—so they can be run in
      bulk without interrupting your session.
    * A return value of ``True`` means the invariant holds; ``False`` means
      a violation was detected.
    * Where useful, a *verbose* flag gives a tqdm progress bar so long-
      running checks remain user-friendly.

    Typical use::

        tests = TrackTests(...)
        tests.is_id_functions_consistent_ensembl()  # Raises if inconsistent.

    Note:
        The class is *not* designed for production; instantiate it only in test
        suites or interactive debugging sessions.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the test harness.

        All positional and keyword arguments are forwarded verbatim to
        :py:class:`~idtrack.Track.__init__`.  Besides constructing the underlying
        graph, the initializer sets up a dedicated :py:data:`logging.Logger`
        named ``"track_tests"`` so individual test routines can emit structured
        diagnostics without polluting the main application log.

        Args:
            args: Positional arguments accepted by :py:class:`~idtrack.Track.__init__`.
            kwargs: Keyword arguments accepted by :py:class:`~idtrack.Track.__init__`.
        """
        super().__init__(*args, **kwargs)  # Sub-class initialization
        self.log = logging.getLogger("track_tests")

    def is_id_functions_consistent_ensembl(self, verbose: bool = True):
        """Ensure Ensembl ID-list helpers agree with the SQL back-end.

        For every release listed in ``graph.graph["confident_for_release"]``
        the test compares two independent sources of Ensembl-gene IDs for the
        *current genome assembly*:

        1. IDs retrieved directly from MySQL via
           :py:class:`~idtrack.DatabaseManager`.
        2. IDs returned by :py:meth:`idtrack.Track.get_id_list` from the graph.

        A mismatch means either the graph was built incompletely or the
        helper functions drift out of sync with the database schema.

        Args:
            verbose: If ``True`` (default) show a tqdm progress-bar while
                iterating through the releases.

        Returns:
            bool: ``True`` when **all** releases produce identical sets; else
            ``False`` (a descriptive warning is logged).
        """
        assembly = self.graph.graph["genome_assembly"]
        # Others should not have consistent as only the main assembly is added fully.
        # In other assemblies, the Ensembl IDs are added if there is an external connection.
        switch = True

        with tqdm(
            self.graph.graph["confident_for_release"], mininterval=0.25, disable=not verbose, ncols=100
        ) as loop_obj:
            for ens_rel in loop_obj:
                loop_obj.set_postfix_str(f"Item:{ens_rel}", refresh=False)

                if self.db_manager.check_if_change_assembly_works(
                    db_manager=self.db_manager.change_release(ens_rel), target_assembly=assembly
                ):
                    db_from = self.db_manager.change_release(ens_rel).change_assembly(assembly)
                    ids_from = set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False)))

                    ids_from_graph = set(
                        self.graph.get_id_list(DB.nts_assembly[assembly][DB.backbone_form], assembly, ens_rel)
                    )
                    if ids_from != ids_from_graph:
                        switch = False
                        self.log.warning(
                            f"Inconsistent results for ID list functions (ensembl), at release `{ens_rel}`."
                        )

        return switch

    def is_id_functions_consistent_ensembl_2(self, verbose: bool = True):
        """Cross-check Ensembl ID *range* helpers against raw edge data.

        For every **backbone** Ensembl-gene node the routine computes the list
        of active release ranges in two distinct ways:

        1. **Raw computation** - by flattening
           ``graph.combined_edges_genes`` and compacting the releases with
           :py:meth:`idtrack.Track.list_to_ranges`.
        2. **Cached lookup** - via the lazily built dictionary
           ``graph.get_active_ranges_of_id``.

        The two lists must match exactly.  A divergence would indicate that
        the cached helper is out of sync with the authoritative edge
        structure.

        Args:
            verbose: If ``True`` (default) wrap the iteration in a tqdm bar.

        Returns:
            bool: ``True`` if *all* gene nodes pass; ``False`` after the first
            failure (a warning is emitted).
        """
        the_ids = list()
        for i in self.graph.nodes:
            if (
                self.graph.nodes[i]["node_type"] == DB.nts_ensembl[DB.backbone_form]
                and self.graph.nodes[i]["Version"] not in DB.alternative_versions
            ):
                the_ids.append(i)

        assembly = self.graph.graph["genome_assembly"]
        with tqdm(the_ids, mininterval=0.25, disable=not verbose, ncols=100) as loop_obj:
            for ti in loop_obj:
                loop_obj.set_postfix_str(f"Item:{ti}", refresh=False)
                # (1) raw recomputation
                data1_raw = self.graph.combined_edges_genes[ti]
                data1_ = sorted({k for i in data1_raw for j in data1_raw[i] for k in data1_raw[i][j] if j == assembly})
                data1 = self.graph.list_to_ranges(data1_)
                # (2) cached view
                data2 = [
                    [i, j if not np.isinf(j) else max(self.graph.graph["confident_for_release"])]
                    for i, j in self.graph.get_active_ranges_of_id[ti]
                ]

                if data1 != data2:
                    self.log.warning(f"Inconsistency between ID range functions: `{ti, data1, data2, data1_raw}`.")
                    return False

        return True

    def is_range_functions_robust(self, verbose: bool = True):
        """Detect overlapping release ranges among sibling Ensembl IDs.

        A *base* Ensembl-gene ID is the stable identifier that groups multiple
        versioned Ensembl-gene records (siblings).  The gene-history model
        requires that the release ranges of sibling IDs **never overlap** -
        each release must be covered by *exactly one* child stable ID.

        This method traverses every base-gene node and checks that condition.

        Args:
            verbose: If ``True`` (default) display a tqdm progress-bar.

        Returns:
            bool: ``True`` when no overlaps are found; ``False`` otherwise.
            Each offending base ID triggers a warning with the conflicting
            ranges.
        """
        base_ids = set()
        for i in self.graph.nodes:
            if self.graph.nodes[i]["node_type"] == DB.nts_base_ensembl[DB.backbone_form]:
                base_ids.add(i)

        switch = True
        with tqdm(base_ids, mininterval=0.25, disable=not verbose, ncols=100) as loop_obj:
            for i in loop_obj:
                loop_obj.set_postfix_str(f"Item:{i}", refresh=False)
                id_family = list(self.graph.neighbors(i))
                id_ranges = [self.graph.get_active_ranges_of_id[j] for j in id_family]

                for r1, r2 in itertools.combinations(id_ranges, 2):
                    r12 = self.graph.get_intersecting_ranges(r1, r2, compact=False)
                    if len(r12) > 1:
                        self.log.warning(f"For Base Ensembl ID {i}: Two associated Ensembl IDs cover the same area")
                        switch = False

        return switch

    def is_base_is_range_correct(self, verbose: bool = True):
        """Verify consistency of *base-gene* active-range calculations.

        Each "base Ensembl gene" node (``node_type == 'base_ensembl_gene'``)
        has an *active release range*—the list of Ensembl releases during
        which descendants of the gene were present.  There are two
        independent ways to obtain this information:

        1. **High-level helper**
           ``graph.get_active_ranges_of_base_id_alternative`` - a cached
           convenience wrapper.
        2. **Low-level reconstruction** by aggregating the *combined_edges*
           table and converting the set of releases into compact
           ``[start, end]`` slices via ``graph.list_to_ranges``.

        This test iterates through **all** base-gene nodes and asserts that
        the two methods deliver byte-identical results.

        Args:
            verbose (bool): If *True* (default) show a tqdm progress
                bar that updates with the current node under inspection.

        Returns:
            bool: ``True`` if *every* base-gene has matching ranges; ``False``
            as soon as a single mismatch is encountered.
        """
        base_ids = set()
        # Gather all base-gene identifiers up-front to avoid repeated lookups.
        for i in self.graph.nodes:
            if self.graph.nodes[i]["node_type"] == "base_ensembl_gene":
                base_ids.add(i)

        switch = True
        with tqdm(base_ids, mininterval=0.25, disable=not verbose, ncols=100) as loop_obj:
            for bi in loop_obj:
                loop_obj.set_postfix_str(f"Item:{bi}", refresh=False)

                # 1. High-level cached helper. get_base_id_range form function
                bi_fun = self.graph.get_active_ranges_of_base_id_alternative(bi)
                # 2. Low-level reconstruction from edge metadata. get range directly from graph
                rd = self.graph.combined_edges[bi]
                bi_dir = self.graph.list_to_ranges(sorted({s for p in rd for r in rd[p] for s in rd[p][r]}))
                # Replace "inf" sentinel with highest known release.  The
                # helper already performs this normalisation so we mimic it
                # here for a fair comparison.
                bi_fun = [
                    [i, j if not np.isinf(j) else max(self.graph.graph["confident_for_release"])] for i, j in bi_fun
                ]

                if bi_dir != bi_fun:
                    switch = False

        return switch

    def is_combined_edges_dicts_overlapping_and_complete(self):
        """Check edge-dictionary partitioning invariants.

        The Track graph materialises three *edge caches*—``combined_edges``
        and its two specialised siblings—each storing adjacency and release
        metadata for a different subset of nodes:

        * ``combined_edges`` - all nodes, including backbone genes.
        * ``combined_edges_genes`` - stable Ensembl genes (non-assembly-specific).
        * ``combined_edges_assembly_specific_genes`` - genes that exist only
          on a single assembly.

        The design contract says:

        1. **Disjointness** - No node key may appear in more than one
           dictionary.
        2. **Completeness** - The *union* of the dictionaries must cover all
           graph nodes **except** those that represent *alternative database
           versions* (e.g. "EnsemblMetazoa") which are intentionally kept
           separate.

        This routine enforces both rules.

        Returns:
            bool: ``True`` if the dictionaries are pair-wise disjoint **and**
            collectively cover every eligible node; ``False`` otherwise.
        """
        the_dicts = [
            self.graph.combined_edges,
            self.graph.combined_edges_genes,
            self.graph.combined_edges_assembly_specific_genes,
        ]
        # 1. Disjointness - if any intersection is non-empty the invariant is violated.
        for d1, d2 in itertools.permutations(iterable=the_dicts, r=2):
            if len(d1.keys() & d2.keys()) != 0:
                return False  # There are some overlapping identifiers.

        # 2. Completeness - every node (minus alternative DB versions) must be represented in exactly one cache.
        covered_nodes = set.union(*map(set, the_dicts))
        uncovered_nodes = self.graph.nodes - covered_nodes
        for un in uncovered_nodes:
            node_data = self.graph.nodes[un]
            if "Version" not in node_data or node_data["Version"] not in DB.alternative_versions:
                return False  # There are identifiers that is not covered with these dicts.

        return True

    def is_edge_with_same_nts_only_at_backbone_nodes(self):
        """Assert *same-node-type* edges exist **only** between backbone genes.

        The graph is a *multilayer* network where nodes of different

        This method traverses every base-gene node and checks that condition.

        Returns:
            bool: ``True`` when no overlaps are found; ``False`` otherwise.
            Each offending base ID triggers a warning with the conflicting
            ranges.
        """
        for n1 in self.graph.nodes:
            nts1 = self.graph.nodes[n1][DB.node_type_str]

            for n2 in self.graph.neighbors(n1):
                nts2 = self.graph.nodes[n2][DB.node_type_str]

                if nts1 == nts2 and nts1 != DB.external_search_settings["nts_backbone"]:
                    return False

        return True

    def is_id_functions_consistent_external(self, verbose: bool = True):
        """Check external-ID list helpers against the raw MySQL tables.

        For **every** combination of *assembly*, *Ensembl release* (limited to
        :py:attr:`graph.graph["confident_for_release"]`) and *external
        database* this test performs the following steps:

        1. Query the authoritative list of external IDs directly from the
           MySQL snapshot via :py:class:`~idtrack.database_manager.DatabaseManager`.
        2. Ask the in-memory graph for the same list via
           :py:meth:`idtrack.Track.get_id_list`.
        3. Normalise node names through
           :py:meth:`idtrack.Track.node_name_alternatives` to cope with the
           occasional “_1” suffix.
        4. Compare the two sets.  A mismatch is logged and the method returns
           ``False`` immediately.

        The exhaustive traversal is expensive (minutes for large genomes) but
        ensures the graph`s indexing helpers never drift from the actual
        database content.

        Args:
            verbose: If *True* (default) display a tqdm progress bar and emit
                log messages at INFO level.  When *False* the method runs
                silently.

        Returns:
            bool: *True* when every single comparison matched, *False* as soon
            as an inconsistency is encountered.
        """
        narrow_external = self.graph.graph["narrow_external"]
        misplace_entries = self.graph.graph["misplaced_external_entry"]

        for assembly in self.graph.available_genome_assemblies:
            with tqdm(
                self.graph.graph["confident_for_release"],
                mininterval=0.25,
                desc=f"Assembly {assembly}",
                disable=not verbose,
                ncols=100,
            ) as loop_obj:
                for release in loop_obj:
                    loop_obj.set_postfix_str(f"Item:{release}", refresh=False)

                    if self.db_manager.check_if_change_assembly_works(
                        db_manager=self.db_manager.change_release(release), target_assembly=assembly
                    ):
                        dm = self.db_manager.change_release(release).change_assembly(assembly)
                        ex_rel_d = {
                            f: dm.change_form(f).get_db("external_relevant" if narrow_external else "external")
                            for f in self.graph.available_forms
                        }

                        for database in self.graph.available_external_databases_assembly[assembly]:
                            form = self.graph.external_database_connection_form[database]
                            ex_rel = ex_rel_d[form]

                            from_dm_ = set(ex_rel["id_db"][ex_rel["name_db"] == database])
                            from_dm = set()
                            for nd in from_dm_:
                                l1, l2 = self.graph.node_name_alternatives(nd)
                                from_dm.add(l1 if l2 else nd)
                            from_gr = set(self.graph.get_id_list(database, assembly, release))

                            if from_gr != from_dm and not all([i in misplace_entries for i in (from_dm - from_gr)]):
                                self.log.warning(
                                    f"Inconsistent results for ID list functions (external) "
                                    f"for: database, assembl, ensembl release: {(database, assembly, release)}"
                                )
                                return False

        return True

    def how_many_corresponding_path_ensembl(
        self, from_release: int, from_assembly: int, to_release: int, go_external: bool, verbose: bool = True
    ):
        """Count history paths between two Ensembl releases.

        The method iterates over **all** Ensembl-gene stable IDs that exist in
        *from_release*/*from_assembly*.  For every ID that is present in the
        graph it calls :py:meth:`idtrack.Track.get_possible_paths` and records
        how many distinct paths the searcher finds to *to_release*.

        The routine is **non-destructive**; it merely provides a quick way to
        gauge the `density` of the history graph or to spot releases where
        path-finding was unexpectedly difficult.

        Args:
            from_release: Source Ensembl release number.
            from_assembly: Source genome assembly code.
            to_release: Target Ensembl release number.
            go_external: If *True* history paths are allowed to temporarily
                leave the Ensembl lineage via external databases.
            verbose: Show a tqdm progress bar (default *True*).

        Returns:
            list[list[Union[str, int, None]]]: A list of two-element sub-lists
            ``[stable_id, n_paths]`` where ``n_paths`` is

                *   an *int* ≥ 0 when the ID was in the graph, or
                *   *None* when the source ID was absent.
        """
        db_from = self.db_manager.change_release(from_release).change_assembly(from_assembly)
        ids_from = sorted(set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False))))
        to_reverse = from_release > to_release

        result = list()
        with tqdm(ids_from, mininterval=0.25, disable=not verbose, ncols=100) as loop_obj:
            for from_id in loop_obj:
                loop_obj.set_postfix_str(f"Item:{from_id}", refresh=False)

                if from_id in self.graph.nodes:
                    lfr = len(
                        self.get_possible_paths(
                            from_id, from_release, to_release, reverse=to_reverse, go_external=go_external
                        )
                    )
                else:
                    lfr = None
                result.append([from_id, lfr])

        return result

    def history_travel_testing(
        self,
        from_release: int,
        from_assembly: int,
        from_database: str,
        to_release: int,
        to_database: str,
        go_external: bool,
        prioritize_to_one_filter: bool,
        convert_using_release: bool,
        from_fraction: float = 1.0,
        verbose: bool = True,
        verbose_detailed: bool = False,
        return_ensembl_alternative: bool = False,
    ):
        """Run an end-to-end Ensembl-history conversion and collect granular QA metrics.

        The routine samples identifiers from *from_database*/*from_release*
        (optionally down-sampling via *from_fraction*) and converts each one to
        *to_database*/*to_release* using :py:meth:`idtrack.Track.convert`.  It
        is intentionally **non-fatal**: every failure mode is caught, logged
        and tallied so that large regression suites can run unattended.  All
        results are returned in a single nested *metrics* dictionary whose
        structure mirrors the printable report produced by
        :py:meth:`format_history_travel_testing_report`.

        The statistics fall into **four conceptual groups**; each counter not only records an absolute event
        count but also serves as a red-flag indicator for specific classes of mapping pathology.  Use the
        guidelines below to interpret the numbers and decide whether a run is *healthy*, *questionable*, or
        *action-required*.

        * **Failure / anomaly counters**

          * ``history_voyage_failed_gracefully`` - the converter raised :py:data:`EmptyConversionMetricsError`.
          * ``history_voyage_failed_unknown`` - any *other* unexpected exception.
          * ``query_not_in_the_graph`` - source ID absent from the graph.
          * ``lost_item`` - traversal finished but produced **no** final IDs.
          * ``lost_item_but_the_same_id_exists`` - special case of *lost_item*
            when the target and source DB are both Ensembl-gene and the target
            ID still exists in the graph.
          * ``found_ids_not_accurate`` - at least one returned target ID is not
            part of the authoritative *ids_to* reference set.

        * **Mapping quality**

          * ``one_to_one_ids`` - queries that resolved to *exactly* one target ID.
          * ``one_to_multiple_ids`` - queries with > 1 admissible targets.
          * ``one_to_multiple_final_conversion`` - subset of the above where
            *exactly one* traversal path was found (heuristics eliminated alternatives).

        * **Collision analysis**

          The list ``clashing_id_type == [clash_one_one, clash_multi_multi,
          clash_multi_one]`` classifies *target IDs* that were reached by
          more than one query:

          * ``clash_one_one`` - every colliding query was 1→1.
          * ``clash_multi_multi`` - every colliding query was 1→many.
          * ``clash_multi_one`` - mixture of 1→1 and 1→many queries (most
            alarming category).

        * **Timings & book-keeping**

          * ``time`` - wall-clock runtime in seconds.
          * ``conversion`` - per-query mapping result for *all* successful traversals.
          * ``converted_item_dict`` / ``converted_item_dict_reversed`` - raw
            per-ID caches used to derive the higher-level counters above.
          * ``parameters`` - echo of the function arguments.
          * ``ids`` - the sampled *from* and reference *to* ID sets.

        Args:
            from_release (int): Ensembl release number of the *source* IDs.
            from_assembly (int): Genome assembly code of the source IDs.
            from_database (str): Node-type / database of the source IDs.
            to_release (int): Ensembl release number of the *target* IDs.
            to_database (str): Node-type / database to convert **into**.
            go_external (bool): Permit temporary detours through external IDs when native Ensembl history edges break.
            prioritize_to_one_filter (bool): Prefer 1→1 mappings over 1→many when multiple paths exist.
            convert_using_release (bool): Pass *from_release* straight into
                :py:meth:`idtrack.Track.convert` instead of letting it infer the starting point.
            from_fraction (float): Fraction (0 < x ≤ 1) of the
                *ids_from* population to sample; speeds up smoke tests.
            verbose (bool): Show tqdm progress bar (coarse).
            verbose_detailed (bool): Embed live metric counters in the tqdm postfix.
            return_ensembl_alternative (bool): Forwarded to :py:meth:`idtrack.Track.convert`.

        Raises:
            ValueError: If either database argument refers to an Ensembl
                node-type (must use backbone helpers instead) or if
                *from_fraction* is outside the open interval (0, 1].

        Returns:
            dict: Nested *metrics* dictionary with the layout described above.
                Use :py:meth:`format_history_travel_testing_report` for a
                human-readable summary.

        Notes:
            * All counters are **absolute counts** - divide by
              ``len(metrics['ids']['from'])`` to obtain rates.
            * The collision analysis is inspired by the *clash statistics*
              logic implemented at the end of the function and helps spot
              discrepant “unique” IDs that suddenly become ambiguous. Keeping
              all three clash counters at zero is the gold standard for a
              healthy build.
        """
        if from_database in DB.nts_ensembl or to_database in DB.nts_ensembl:
            raise ValueError

        ids_from = sorted(self.graph.get_id_list(from_database, from_assembly, from_release))
        ids_to = set(self.graph.get_id_list(to_database, self.graph.graph["genome_assembly"], to_release))
        ids_to_s = {self.graph.nodes[i]["ID"] for i in ids_to} if to_database in DB.nts_assembly_reverse else set()

        # Input sampling
        if from_fraction == 1.0:
            pass
        elif 0.0 < from_fraction < 1.0:
            from_faction_count = ceil(len(ids_from) * from_fraction)
            ids_from = sorted(random.sample(ids_from, from_faction_count))
        else:
            raise ValueError

        # Metric scaffold
        parameters: dict[str, Union[bool, str, int, float]] = {
            "from_release": from_release,
            "from_assembly": from_assembly,
            "from_database": from_database,
            "to_release": to_release,
            "to_database": to_database,
            "go_external": go_external,
            "from_fraction": from_fraction,
            "prioritize_to_one_filter": prioritize_to_one_filter,
        }

        metrics: dict[str, Any] = {
            "parameters": parameters,
            "ids": {"from": ids_from, "to": ids_to},
            "lost_item": [],
            "one_to_one_ids": {},
            "query_not_in_the_graph": [],
            "history_voyage_failed_gracefully": [],
            "history_voyage_failed_unknown": [],
            "lost_item_but_the_same_id_exists": [],
            "found_ids_not_accurate": {},
            "conversion": {},
            "one_to_multiple_ids": {},
            "one_to_multiple_final_conversion": [],
            "converted_item_dict": {},
            "converted_item_dict_reversed": {},
        }

        t1 = time.time()

        with tqdm(ids_from, mininterval=0.25, disable=not verbose, desc="Mapping", ncols=100) as loop_obj:
            for the_id in loop_obj:
                if not verbose_detailed:
                    loop_obj.set_postfix_str(f"Item:{the_id}", refresh=False)
                else:
                    suffix = (
                        f"Item:{the_id}, "
                        f"{len(metrics['one_to_one_ids'])},"
                        f"{len(metrics['one_to_multiple_ids'])},"
                        f"{len(metrics['lost_item'])},"
                        f"{len(metrics['found_ids_not_accurate'])},"
                        f"{len(metrics['query_not_in_the_graph']) + len(metrics['history_voyage_failed_unknown'])}"
                    )
                    loop_obj.set_postfix_str(suffix, refresh=False)

                # Conversion attempt
                try:
                    if convert_using_release:
                        converted_item = self.convert(
                            from_id=the_id,
                            from_release=from_release,
                            to_release=to_release,
                            final_database=to_database,
                            go_external=go_external,
                            prioritize_to_one_filter=prioritize_to_one_filter,
                            return_ensembl_alternative=return_ensembl_alternative,
                        )
                    else:
                        converted_item = self.convert(
                            from_id=the_id,
                            from_release=None,
                            to_release=to_release,
                            final_database=to_database,
                            go_external=go_external,
                            prioritize_to_one_filter=prioritize_to_one_filter,
                            return_ensembl_alternative=return_ensembl_alternative,
                        )
                    metrics["converted_item_dict"][the_id] = converted_item

                except nx.exception.NetworkXError:
                    metrics["query_not_in_the_graph"].append(the_id)
                    continue
                except EmptyConversionMetricsError:
                    full_tb = traceback.format_exc()
                    metrics["history_voyage_failed_gracefully"].append((the_id, full_tb))
                    continue
                except Exception:
                    full_tb = traceback.format_exc()
                    metrics["history_voyage_failed_unknown"].append((the_id, full_tb))
                    continue

                # Metrics aggregation
                if converted_item is None:
                    if to_database == from_database == "ensembl_gene" and self.graph.nodes[the_id]["ID"] in ids_to_s:
                        metrics["lost_item_but_the_same_id_exists"].append(the_id)
                    metrics["lost_item"].append(the_id)
                else:
                    covr = {j for i in converted_item for j in converted_item[i]["final_conversion"]["final_elements"]}
                    metrics["conversion"][the_id] = list(covr)
                    if len(covr) == 1:
                        metrics["one_to_one_ids"][the_id] = list(covr)
                    elif len(covr) > 1:
                        metrics["one_to_multiple_ids"][the_id] = list(covr)
                        if len(converted_item) == 1:
                            metrics["one_to_multiple_final_conversion"].append(the_id)
                    else:
                        raise ValueError

                    # How much of the converted IDs show the same ID
                    for c in covr:
                        if c in metrics["converted_item_dict_reversed"]:
                            metrics["converted_item_dict_reversed"][c].append(the_id)
                        else:
                            metrics["converted_item_dict_reversed"][c] = [the_id]

                    for c in covr:
                        if c not in ids_to:
                            if the_id not in metrics["found_ids_not_accurate"]:
                                metrics["found_ids_not_accurate"][the_id] = list()
                            metrics["found_ids_not_accurate"][the_id].append(c)

        # ── Clash statistics:
        # Each counter tracks how many *target* IDs were reached by more than
        # one *source* ID, broken down by the **type of mapping** that led to
        # the collision:
        clash_multi_multi: int = 0  # all colliding sources were 1→many mappings
        clash_multi_one: int = 0  # mix of 1→many and 1→1 sources
        clash_one_one: int = 0  # all colliding sources were clean 1→1

        # `converted_item_dict_reversed` maps every target ID to the list of
        # source IDs that converted to it (built earlier inside the big mapping
        # loop).  We inspect only those targets hit by ≥ 2 sources - a “clash”.
        for _, cidr_val in metrics["converted_item_dict_reversed"].items():

            # Skip non-clashing targets (exactly one source mapped here).
            if len(cidr_val) > 1:
                # Does *any* of the sources belong to the 1→many bucket?
                s1 = any([cv in metrics["one_to_multiple_ids"] for cv in cidr_val])
                # Does *any* of the sources belong to the 1→1 bucket?
                s2 = any([cv in metrics["one_to_one_ids"] for cv in cidr_val])
                # Categorise clash by the mixture of mapping types.
                if s1 and s2:  # at least one 1→many **and** one 1→1
                    clash_multi_one += 1
                elif s1:  # only 1→many sources present
                    clash_multi_multi += 1
                elif s2:  # only 1→1 sources present
                    clash_one_one += 1
                else:
                    raise ValueError
        metrics["clashing_id_type"] = [clash_one_one, clash_multi_multi, clash_multi_one]

        t2 = time.time()
        metrics["time"] = t2 - t1
        return metrics

    def history_travel_testing_random_arguments_generator(self, strict_forward: bool, include_exclude_list: list):
        """Generate a plausible random parameter set for :py:meth:`history_travel_testing`.

        The helper picks **compatible** source/target assemblies, releases and
        databases so the subsequent conversion test has a realistic chance to
        succeed.  When *strict_forward* is *True* the target release is
        guaranteed to be **≥** the source release (no time-travel back).

        Args:
            strict_forward: Enforce a non-decreasing release direction.
            include_exclude_list: Todo.

        Returns:
            dict: Keys ``from_assembly``, ``from_release``, ``to_release``,
            ``from_database``, ``to_database`` ready to be splatted into
            :py:meth:`history_travel_testing`.
        """
        include_ensembl_1, include_external_1, include_ensembl_2, include_external_2 = include_exclude_list
        only_backbone_tests_1 = True if not any([include_ensembl_1, include_external_1]) else False
        only_backbone_tests_2 = True if not any([include_ensembl_2, include_external_2]) else False

        from_assembly = (
            random.choice(list(DB.assembly_mysqlport_priority.keys()))
            if not only_backbone_tests_1
            else DB.main_assembly
        )
        # as the final release should be always the main assembly
        to_assembly = self.graph.graph["genome_assembly"] if not only_backbone_tests_2 else DB.main_assembly
        the_key1, the_key2 = None, None
        while the_key1 is None or the_key2 is None:
            the_key1 = self.random_dataset_source_generator(
                from_assembly,
                include_ensembl=include_ensembl_1,
                include_external=include_external_1,
                only_backbone_tests=only_backbone_tests_1,
                for_final_database=False,
            )
            if the_key1 != ("", -1, -1):
                the_key2 = self.random_dataset_source_generator(
                    to_assembly,
                    include_ensembl=include_ensembl_2,
                    include_external=include_external_2,
                    for_final_database=True,
                    only_backbone_tests=only_backbone_tests_2,
                    release_lower_limit=None if not strict_forward else the_key1[2],
                )
            if the_key1 == ("", -1, -1) or the_key2 == ("", -1, -1):
                self.log.warning("Recalculating parameters.")

        return {
            "from_assembly": from_assembly,
            "from_release": the_key1[2],
            "to_release": the_key2[2],
            "from_database": the_key1[0],
            "to_database": the_key2[0],
        }

    def history_travel_testing_random(
        self,
        from_fraction: float,
        include_ensembl_source=True,
        include_external_source=True,
        include_ensembl_destination=True,
        include_external_destination=True,
        verbose: bool = True,
        verbose_detailed: bool = False,
        strict_forward: bool = False,
        convert_using_release: bool = False,
        prioritize_to_one_filter: bool = True,
        return_result: bool = False,
    ):
        """Convenience wrapper around :py:meth:`history_travel_testing`.

        The routine generates a *random* but internally consistent test case
        via :py:meth:`history_travel_testing_random_arguments_generator`, logs
        the chosen parameters (unless *verbose* is *False*) and delegates the
        heavy lifting to :py:meth:`history_travel_testing`.

        Args:
            from_fraction: Fraction of IDs to sample from the source set.
            strict_forward: Forwarded to the argument generator.
            convert_using_release: Forwarded to
                :py:meth:`history_travel_testing`.
            prioritize_to_one_filter: Forwarded to
                :py:meth:`history_travel_testing`.
            verbose: Show coarse progress information.
            verbose_detailed: Include extended per-ID counters in the progress bar.
            return_result: Todo.
            include_ensembl_source: Todo.
            include_external_source: Todo.
            include_ensembl_destination: Todo.
            include_external_destination: Todo.

        Returns:
            dict: The *metrics* dictionary returned by
            :py:meth:`history_travel_testing`.
        """
        include_exclude_list = [
            include_ensembl_source,
            include_external_source,
            include_ensembl_destination,
            include_external_destination,
        ]

        parameters = self.history_travel_testing_random_arguments_generator(
            strict_forward=strict_forward, include_exclude_list=include_exclude_list
        )

        if verbose:
            printable1 = os.linesep + os.linesep.join(self.format_history_travel_testing_report_header(parameters))
            self.log.info(printable1)

        res = self.history_travel_testing(
            **parameters,
            go_external=True,
            prioritize_to_one_filter=prioritize_to_one_filter,
            convert_using_release=convert_using_release,
            from_fraction=from_fraction,
            verbose=verbose,
            verbose_detailed=verbose_detailed,
            return_ensembl_alternative=False,
        )

        if verbose:
            printable2 = os.linesep + os.linesep.join(
                self.format_history_travel_testing_report(res, include_header=False)
            )
            self.log.info(printable2)

        if return_result:
            return res

    def is_final_external_conversion_robust(
        self,
        convert_using_release: bool = False,
        database: Optional[str] = None,
        ens_rel: Optional[int] = None,
        verbose: bool = True,
        from_fraction: float = 1.0,
        prioritize_to_one_filter: bool = False,
    ):
        """Validate Ensembl→external conversion against MySQL ground truth.

        A random external database is chosen for **every** genome assembly. For
        the selected combination the method grabs the authoritative mapping
        table (graph-ID → external ID set) from MySQL and converts the same
        graph-IDs with :py:meth:`idtrack.Track.convert`.

        Args:
            convert_using_release: Whether to pin the *from_release* when
                calling the converter.  Keeping this *True* usually speeds up
                the search and mimics user-facing behaviour.
            verbose: Print the current assembly/database/release being tested.
            prioritize_to_one_filter: Todo.
            ens_rel: Todo.
            from_fraction: Todo.
            database: Todo.

        Returns:
            bool: *True* if every converted set equals the MySQL reference,
                *False* upon the first deviation.

        Raises:
            ValueError: Todo.
        """
        issues_t1 = []
        issues_t2 = []
        issues_t3 = []
        assembly = DB.main_assembly

        if database is None or ens_rel is None:
            self.log.info("Random database and Ensembl release.")
            database, _, ens_rel = self.random_dataset_source_generator(
                include_ensembl=True,
                include_external=True,
                only_backbone_tests=False,
                for_final_database=False,
                assembly=assembly,
                form=DB.backbone_form,
            )

        if not self.db_manager.check_if_change_assembly_works(
            db_manager=self.db_manager.change_release(ens_rel), target_assembly=assembly
        ):
            raise ValueError

        dm = self.db_manager.change_release(ens_rel).change_assembly(assembly)

        df = dm.get_db("external_relevant")
        df = df[df["name_db"] == database]
        base_dict: dict[str, set] = dict()
        for _, item in df.iterrows():
            if item["graph_id"] not in base_dict:
                base_dict[item["graph_id"]] = set()
            base_dict[item["graph_id"]].add(item["id_db"])

        if verbose:
            self.log.info(f"Assembly: {assembly}, Database: {database}, Release: {ens_rel}")

        res = self.history_travel_testing(
            from_fraction=from_fraction,
            from_release=ens_rel,
            from_assembly=assembly,
            from_database=DB.nts_assembly[assembly][DB.backbone_form],
            to_release=ens_rel,
            to_database=database,
            go_external=True,
            prioritize_to_one_filter=prioritize_to_one_filter,
            convert_using_release=convert_using_release,
            verbose=verbose,
            verbose_detailed=False,
        )

        for from_id in res["ids"]["from"]:
            if from_id not in res["conversion"]:
                issues_t3.append(from_id)
            else:
                issue_dict = {
                    "database": database,
                    "asym": assembly,
                    "ens_rel": ens_rel,
                    "id": from_id,
                    "converted": res["conversion"][from_id],
                }
                if from_id not in base_dict:
                    issues_t2.append(issue_dict)
                elif {i.lower() for i in res["conversion"][from_id]} != {i.lower() for i in base_dict[from_id]}:
                    issue_dict["base_expectation"] = base_dict[from_id]
                    issues_t1.append(issue_dict)

        # make sure issues_t3 is not found in base_dict_from_id

        if len(issues_t1) == 0 and len(issues_t2) == 0:
            return True, (issues_t1, issues_t2, issues_t3, res)
        else:
            return False, (issues_t1, issues_t2, issues_t3, res)

    def random_dataset_source_generator(
        self,
        assembly: int,
        include_external: bool,
        include_ensembl: bool,
        for_final_database: bool,
        only_backbone_tests: bool,
        release_lower_limit: Optional[int] = None,
        form: Optional[str] = None,
    ) -> tuple[str, int, int]:
        """Pick a random (<database>, <assembly>, <release>) tuple.

        The function guarantees that the triple actually exists in the graph
        and - if *release_lower_limit* is provided - honours the minimum
        release constraint.

        Args:
            assembly: NCBI assembly code (integer).
            include_ensembl: Whether Ensembl backbone databases may be returned
                as *database*.
            release_lower_limit: Smallest permissible Ensembl release number
                for the returned triple.  *None* disables the filter.
            form: Restrict the draw to a particular connection *form*
                (protein/coding/gene).  *None* means no restriction.
            only_backbone_tests: Todo.
            include_external: Todo.
            for_final_database: Todo.

        Returns:
            tuple | None: ``(<database>, <assembly>, <release>)`` or *None* when
            no matching release exists.

        Raises:
            ValueError: Todo.
        """
        if not only_backbone_tests:
            if include_external:
                all_possible_sources = copy.deepcopy(list(self.graph.available_external_databases_assembly[assembly]))
                all_possible_sources = [i for i in all_possible_sources if not i.startswith("synonym_id")]
            else:
                all_possible_sources = []

            if form is not None:
                all_possible_sources = [
                    i for i in all_possible_sources if self.graph.external_database_connection_form[i] == form
                ]

            if include_ensembl and not for_final_database:
                if form is None:
                    all_possible_sources.extend(list(DB.nts_assembly[assembly].values()))
                else:
                    all_possible_sources.append(DB.nts_assembly[assembly][form])

            if include_ensembl and (form == DB.backbone_form or form is None):
                all_possible_sources.append(DB.nts_ensembl[DB.backbone_form])
                all_possible_sources.append(DB.nts_base_ensembl[DB.backbone_form])

            if not all_possible_sources:
                raise ValueError("There is nothing as possible sources.")
        else:
            all_possible_sources = [DB.nts_ensembl[DB.backbone_form]]

        selected_database = random.choice(all_possible_sources)
        possible_releases = self.graph.available_releases_given_database_assembly[(selected_database, assembly)]
        if release_lower_limit is not None:
            possible_releases = {i for i in possible_releases if i >= release_lower_limit}

        if len(possible_releases) > 1:
            selected_release = random.choice(list(possible_releases))
            the_key = (selected_database, assembly, selected_release)
            return the_key
        else:
            return ("", -1, -1)

    def is_node_consistency_robust(self, verbose: bool = True):
        """Check for illegal neighbour relationships and multi-edges.

        The graph may contain **exactly one** edge between nodes of *different*
        node-types.  Nodes of the *same* node-type are only allowed when that
        type is the Ensembl backbone (``ensembl_gene``).  Any deviation - a
        lateral same-type connection or >1 multi-edge - is logged and aborts
        the test.

        Args:
            verbose: Print offending nodes when a violation is detected.

        Returns:
            bool: *True* when the graph satisfies the topology rules, *False*
            otherwise.
        """
        with tqdm(self.graph.nodes, mininterval=0.25, disable=not verbose, ncols=100) as loop_obj:
            for i in loop_obj:
                for j in self.graph.neighbors(i):
                    ni = self.graph.nodes[i][DB.node_type_str]
                    nj = self.graph.nodes[j][DB.node_type_str]

                    if ni == nj and ni != DB.nts_ensembl[DB.backbone_form]:
                        if verbose:
                            self.log.warning(f"Neighbor nodes (not backbone) has similar node type: `{i}`, `{j}`")
                        return False

                    elif ni != nj and len(self.graph[i][j]) != 1:
                        if verbose:
                            self.log.warning(f"Multiple edges between `{i, ni}` and `{j, nj}`")
                        return False

        return True

    def format_history_travel_testing_report_header(self, p: dict[str, Any]) -> list[str]:
        """Todo.

        Args:
            p (dict[str, Any]): Todo.

        Returns:
            str: Todo.
        """
        header = [
            "╔═ History-Travel-Testing Report ═╗",
            f"Source  : {p.get('from_database')} "
            f"(Assembly {p.get('from_assembly')}, Release {p.get('from_release')})",
            f"Target  : {p.get('to_database')} " f"(Release {p.get('to_release')})",
        ]

        return header

    def format_history_travel_testing_report(
        self, res: dict[str, Any], include_header=False, line_separation_at_end=True
    ) -> list[str]:
        """Todo.

        Args:
            res (dict[str, Any]): Todo.
            include_header (bool): Todo. Defaults to False.
            line_separation_at_end (bool): Todo.. Defaults to True.

        Returns:
            str: Todo.
        """

        def cnt(key: str) -> int:
            """Return length of list/dict at res[key] or zero if absent."""
            return len(res.get(key, []))

        def block(title: str, rows: list[tuple]) -> list[str]:
            pad = max(len(k) for k, _ in rows)
            return [f"{title}:"] + [f"  - {k.ljust(pad)} : {v:,}" for k, v in rows]

        if include_header:
            p = res.get("parameters", {})
            header = self.format_history_travel_testing_report_header(p)
            header_extension = [
                f"External: {p.get('go_external')}   " f"1→1-pref.: {p.get('prioritize_to_one_filter')}",
                f"Sample  : {p.get('from_fraction'):g} of source IDs",
            ]
            header.extend(header_extension)
        else:
            header = []

        failure_rows = [
            ("Voyage failed (graceful)      ", cnt("history_voyage_failed_gracefully")),
            ("Voyage failed (unknown)       ", cnt("history_voyage_failed_unknown")),
            ("Query not in graph            ", cnt("query_not_in_the_graph")),
            ("Lost item                     ", cnt("lost_item")),
            ("Lost item, but ID exists      ", cnt("lost_item_but_the_same_id_exists")),
            ("Found IDs not accurate        ", cnt("found_ids_not_accurate")),
        ]

        clashes = res.get("clashing_id_type", [0, 0, 0])
        mapping_rows = [
            ("One→one IDs                   ", cnt("one_to_one_ids")),
            ("One→many IDs                  ", cnt("one_to_multiple_ids")),
            ("One→many (single conv.)       ", cnt("one_to_multiple_final_conversion")),
            ("Successfully converted IDs    ", cnt("conversion")),
            ("Clash one→one                 ", clashes[0]),
            ("Clash many→many               ", clashes[1]),
            ("Clash mixed                   ", clashes[2]),
        ]

        report_lines: list[str] = (
            header
            + block("Failure / Anomaly Counts", failure_rows)
            + block("Mapping Statistics", mapping_rows)
            + [f"Total runtime: {res.get('time', 0):.2f} s"]
        )

        if line_separation_at_end:
            report_lines.append("")

        return report_lines
