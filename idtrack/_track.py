#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import bisect
import copy
import itertools
import logging
import warnings
from collections import Counter
from collections.abc import Iterable
from functools import cached_property
from typing import Any, Callable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB, EmptyConversionMetricsError
from idtrack._graph_maker import GraphMaker
from idtrack._the_graph import TheGraph


class Track:
    """Bidirectional path-finding resolver for biological identifiers.

    `Track` builds and queries a *bio-ID* multigraph that stitches together
    Ensembl history edges (genes, transcripts, proteins) and cross-reference
    edges to external databases (UniProt, RefSeq, …).  Given a **source
    identifier**, a **target Ensembl release**, and/or a **target database**,
    the class:

    1. **Normalises** the source to an Ensembl *gene* node when necessary.
    2. **Time-travels** through historical edges—forward or backward—until it
       reaches the requested release, optionally "beaming-up" through external
       IDs when the backbone is disconnected.
    3. **Converts** the resolved Ensembl gene into the requested external
       database (or returns the gene itself) while annotating the result with
       confidence scores and the full traversal path.


    Two mutually-recursive engines power the search:

    * `_recursive_function`   — depth-first search along temporal edges.
    * `_recursive_synonymous` — search for synonymous nodes at a single
      release to enable the external "beam-up".

    Attributes
    ----------
    graph : networkx.MultiDiGraph
        The pre-computed bio-ID graph produced by
        :py:class:`_graph_maker.GraphMaker`.
    version_info : dict
        Metadata about the graph build (Ensembl releases included, build date,
        Git commit, etc.).
    _external_entrance_placeholder : dict[bool, int]
        Sentinel node IDs that mark artificial edges used when an external ID
        is pulled onto the Ensembl backbone (`False → -1`, `True → 10001`).
    _external_entrance_placeholders : list[int]
        Sorted list of the sentinel values above.
    """

    # ENSG00000263464
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        """Create a `Track` resolver and load (or build) its graph.

        Args:
            db_manager (DatabaseManager): Connection manager that knows how to fetch Ensembl and
                cross-reference tables from a local cache or a live MySQL mirror.
                The same instance is forwarded to :py:class:`_graph_maker.GraphMaker`.
            kwargs: Additional keyword arguments forwarded verbatim to
                :py:meth:`_graph_maker.GraphMaker.get_graph`.  Common flags include
                `force_rebuild` (recompute the graph from scratch), `species`
                (restrict to one taxon), and `cache_dir` (override on-disk cache location).
        """
        self.log = logging.getLogger("track")
        self.db_manager = db_manager
        graph_creator = GraphMaker(self.db_manager)

        # Calculate/Load the graph
        self.graph = graph_creator.get_graph(**kwargs)
        self.version_info = self.graph.graph["version_info"]
        self._external_entrance_placeholder = {False: -1, True: 10001}
        self._external_entrance_placeholders = sorted(self._external_entrance_placeholder.values())

    def _recursive_synonymous(
        self,
        _the_id: str,
        synonymous_ones: list,
        synonymous_ones_db: list,
        filter_node_type: set[str],
        the_path: Optional[list] = None,
        the_path_db: Optional[list] = None,
        depth_max: int = 0,
        from_release: Optional[int] = None,
        ensembl_backbone_shallow_search: bool = False,
        account_for_hyperconnected_nodes: bool = True,
    ):
        """Helper method to be used in :py:meth:`_graph.Track.synonymous_nodes`.

        Recursively explore the bio-ID graph to collect **synonymous paths**
        starting at `_the_id` and ending on a node whose *type* is a member of `filter_node_type`.

        A *path* is a list of node identifiers (`_the_path`) together with a
        parallel list of their node-type strings (`_the_path_db`).
        The search is **breadth-limited**: the *depth* of a path is defined as
        the **maximum count of any single node-type** it contains
        (e.g. a path with three `external` nodes has depth 3).  Recursion
        stops when that depth would exceed `depth_max`.

        Additional pruning rules:

        - The walk never visits the same node twice (no cycles).
        - It never traverses two consecutive edges whose source and target
            share the **same node-type**—this prevents "time-travel" within the
            Ensembl history backbone.
        - When `ensembl_backbone_shallow_search` is *True*, the search is
            restricted to the **reverse** direction except for node-types listed
            in :py:attr:`DB.nts_bidirectional_synonymous_search`.

        On reaching a terminating node the method *appends* the discovered
        paths to `synonymous_ones` and `synonymous_ones_db`. It does not
        return anything. Results are accumulated in `synonymous_ones` and `synonymous_ones_db`.

        Args:
            _the_id (str): Identifier of the starting node (Ensembl or
                external).
            synonymous_ones (list): Mutable list that will receive
                each successful identifier path.
            synonymous_ones_db (list): Mutable list that will receive
                the corresponding node-type paths.
            filter_node_type (set[str]): Allowed node-types for the **final**
                node of a path (e.g. `{'ensembl_gene'}`).
            the_path (list | None): Current path leading to
                `_the_id`; *None* for the root invocation.
            the_path_db (list | None): Node-type counterpart of
                `the_path`; *None* for the root invocation.
            depth_max (int): Maximum allowed depth as defined above.
            from_release (int | None): If given, only keep terminal
                nodes that are *active* in this Ensembl release.
            ensembl_backbone_shallow_search (bool): Activate the
                shallow, mostly-reverse search mode described above.
            account_for_hyperconnected_nodes: Todo.
        """

        def decide_terminate_externals():
            if input_node_type == DB.nts_external:  # last node nts in the path
                if len(_the_path) > 1:
                    edge_key = self.edge_key_orientor(*_the_path[-2:], 0)
                    the_data = self.graph.get_edge_data(*edge_key)[DB.connection_dict]  # edge_data
                else:
                    the_data = self.graph.combined_edges[_the_id]  # last node is _the_id, node_data

                for _ed1 in the_data:  # database
                    if _ed1 in filter_node_type:
                        for _ed2 in the_data[_ed1]:  # assembly
                            if from_release is None or from_release in the_data[_ed1][_ed2]:
                                return True
            return False

        def decide_terminate_others():
            if input_node_type != DB.nts_external:  # last node nts in the path
                if input_node_type in filter_node_type and (
                    from_release is None
                    or TheGraph.is_point_in_range(self.graph.get_active_ranges_of_id[_the_id], from_release)
                ):
                    return True
            return False

        input_node_type = self.graph.nodes[_the_id][DB.node_type_str]
        _the_path = [_the_id] if the_path is None else the_path
        _the_path_db = [input_node_type] if the_path_db is None else the_path_db

        if decide_terminate_others() or decide_terminate_externals():
            synonymous_ones.append(_the_path)
            synonymous_ones_db.append(_the_path_db)

        if depth_max > max(Counter(_the_path_db).values()):  # The depth is all node_type.
            for graph in (
                (self.graph, self.graph.rev)
                # if ensembl_backbone_shallow_search activated, follow only reverse direction. This should typically
                # start from the ensembl_gene nodes.
                # Allow bidirectional search for certain nts when ensembl_backbone_shallow_search activated, this is to
                # make sure assembl_nodes are bridging correctly.
                if not ensembl_backbone_shallow_search or input_node_type in DB.nts_bidirectional_synonymous_search
                else (self.graph.rev,)
            ):
                for _next_neighbour in graph.neighbors(_the_id):
                    if _next_neighbour in _the_path:
                        # prevent bouncing.
                        continue

                    if account_for_hyperconnected_nodes and _next_neighbour in self.graph.hyperconnective_nodes:
                        continue

                    gnt = self.graph.nodes[_next_neighbour][DB.node_type_str]

                    if gnt == input_node_type:
                        # prevent history travel. Note that only backbone node type (ensembl_gene) actually
                        # has connection to the same node type. 'combined_edges' in TheGraph class checks that.
                        continue

                    if len(_the_path) >= 2:
                        # prevent bouncing between transcript and gene id.versions
                        # [..., l1, l2, gnt]
                        if (
                            # if the addition of new element is a bouncing event (like A-B-A)
                            _the_path_db[-2] == gnt
                            # Allow bouncing if it bridges two 'external' nodes.
                            and gnt != DB.nts_external
                            # Allow bouncing if it bridges two 'ensembl base' nodes.
                            and gnt != DB.nts_base_ensembl[DB.backbone_form]  # transcript/translation base depracated.
                            # Check if 1st and 3rd element are the same 'ID' (could have different 'Versions'.)
                            and self.graph.nodes[_the_path[-2]]["ID"] == self.graph.nodes[_next_neighbour]["ID"]
                        ):
                            continue

                    if from_release is not None:
                        # assert len(graph[_the_id][_next_neighbour]) == 1
                        # To use of '0' below, this is verified by TheGraph.is_node_consistency_robust

                        if from_release not in graph.get_edge_data(_the_id, _next_neighbour, 0)["available_releases"]:
                            # Do not care which assembly or which database.
                            continue

                    self._recursive_synonymous(
                        _next_neighbour,
                        synonymous_ones,
                        synonymous_ones_db,
                        filter_node_type,
                        the_path=_the_path + [_next_neighbour],
                        the_path_db=_the_path_db + [gnt],
                        depth_max=depth_max,
                        from_release=from_release,
                        ensembl_backbone_shallow_search=ensembl_backbone_shallow_search,
                        account_for_hyperconnected_nodes=account_for_hyperconnected_nodes,
                    )

    def synonymous_nodes(
        self,
        the_id: str,
        depth_max: int,
        filter_node_type: set[str],
        from_release: Optional[int] = None,
        ensembl_backbone_shallow_search: bool = False,
        account_for_hyperconnected_nodes: bool = True,
    ):
        """Public wrapper around :py:meth:`_recursive_synonymous`.

        The method returns **all minimal-length synonym paths** emanating from `the_id`.

        The function first runs a *default* depth search determined by
        `DB.external_search_settings['synonymous_max_depth']`.
        If no synonym is found and `depth_max` is greater than that default,
        a second, deeper search is attempted.

        For every distinct *target* node the shortest path is kept; longer
        paths to the same target are discarded.

        Args:
            the_id (str): Source identifier.
            depth_max (int): Maximum search depth to try if the default search fails.
            filter_node_type (set[str]): Node-types that are acceptable for the
                *target* node(s).  Must *not* include the generic
                `'external'` type—specify the concrete external DB instead.
            from_release (int | None): Constrain targets to those active in this Ensembl release.
            ensembl_backbone_shallow_search (bool): If *True*,
                restricts the graph traversal as explained in :py:meth:`_recursive_synonymous`.
            account_for_hyperconnected_nodes: Todo.

        Returns:
            list[list[list[str]]]: A list whose elements are `[identifier_path, node_type_path]` pairs,
                each representing the minimal route to one synonymous node.

        Raises:
            ValueError: If `filter_node_type` improperly contains the generic
                external type, or if `depth_max` is incompatible with `ensembl_backbone_shallow_search`.
        """
        if DB.nts_external in filter_node_type:
            raise ValueError(f"Define which external database: `{filter_node_type}`.")

        if ensembl_backbone_shallow_search:
            if depth_max != 2:
                raise ValueError("Does not allowed.")

        # Use default depth max if there is at least one item in the syn path.
        synonymous_ones: list = []
        synonymous_ones_db: list = []
        self._recursive_synonymous(
            the_id,
            synonymous_ones,
            synonymous_ones_db,
            filter_node_type,
            depth_max=DB.external_search_settings["synonymous_max_depth"],
            from_release=from_release,
            ensembl_backbone_shallow_search=ensembl_backbone_shallow_search,
            account_for_hyperconnected_nodes=account_for_hyperconnected_nodes,
        )

        # Otherwise use supplemented depth_max, which is generally 1, 2 higher than default.
        if len(synonymous_ones) == 0 and depth_max != DB.external_search_settings["synonymous_max_depth"]:
            self._recursive_synonymous(
                the_id,
                synonymous_ones,
                synonymous_ones_db,
                filter_node_type,
                depth_max=depth_max,
                from_release=from_release,
                ensembl_backbone_shallow_search=ensembl_backbone_shallow_search,
                account_for_hyperconnected_nodes=account_for_hyperconnected_nodes,
            )

        remove_set: set = set()
        the_ends_min: dict = dict()

        # Below is to choose the minimum path to the same target
        for p in synonymous_ones:
            e = p[-1]  # get the the target of each path
            lp = len(p)
            am = the_ends_min.get(e, None)
            if am is None or am > lp:
                the_ends_min[e] = lp
            # Create a dictionary of lengths to process in the next step

        for ind in range(len(synonymous_ones)):
            # Determine which elements will be deleted
            e = synonymous_ones[ind][-1]
            am = the_ends_min[e]
            lp = len(synonymous_ones[ind])
            if lp > am:
                remove_set.add(ind)

        return [  # Remove and return the result. Zip two list together while returning.
            [synonymous_ones[ind], synonymous_ones_db[ind]]
            for ind in range(len(synonymous_ones))
            if ind not in remove_set
        ]

    @staticmethod
    def get_from_release_and_reverse_vars(lor: list, p: int, mode: str):
        """Derive a list of `(release, reverse)` tuples.

        Derive a list of `(release, reverse)` tuples that indicate *which*
        Ensembl release to start the graph walk from and *whether* that walk
        should move **backwards in time**.

        Given a collection of active-range intervals `lor` and a pivot
        release `p`, the algorithm selects one or two release points per
        interval depending on `mode`:

        * 'closest' - choose the release **nearest** to `p` within or at
            the ends of the interval.
        * 'distant' - choose the release **farthest** from `p` within the
            interval.

        The boolean in each tuple is *True* when the walk should start **after**
        the selected release and move backwards (i.e. "reverse mode"), and
        *False* when it should move forwards.

        Args:
            lor (list): List of inclusive
                `(first_release, last_release)` intervals in ascending order.
            p (int): Pivot release around which "closest" or "distant" is
                evaluated.
            mode (str): Either `'closest'` or `'distant'`.

        Returns:
            list[tuple[int, bool]]: Release / reverse-flag pairs, ordered in
            the sequence they should be tried by the path-finder.

        Raises:
            ValueError: If an interval in `lor` is malformed, `mode` is
                not recognised, or internal consistency checks fail.
        """
        result = list()
        # Note that there is no need to consider assembly here because always this function is used in the context of
        # graph backbone nodes (ensembl gene), and the backbone is always the latest (currently 38).
        for l1, l2 in lor:
            if l1 > l2:
                raise ValueError("l1 > l2")

            elif mode == "closest" and p == l1:
                result.append((l1, False))

            elif mode == "closest" and p < l1:
                result.append((l1, True))

            elif mode == "closest" and l2 <= p:
                result.append((l2, False))

            elif mode == "closest" and l1 < p < l2:
                result.append((l1, True))
                result.append((l2, False))

            elif mode == "distant" and p <= l1:
                result.append((l2, True))

            elif mode == "distant" and l2 <= p:
                result.append((l1, False))

            elif mode == "distant" and l1 < p < l2:
                result.append((l2, True))
                result.append((l1, False))

            else:
                raise ValueError("Else?")

        return result

    def _choose_relevant_synonym_helper(
        self, from_id, synonym_ids, to_release: int, from_release: Optional[int], mode: str
    ):
        """Select the **most temporally relevant** synonym(s) for an Ensembl gene-ID family.

        The method evaluates each candidate in `synonym_ids` against the
        *target* release `to_release` and, when applicable, the *source*
        release `from_release`.  Its job is to decide **where** the path
        should *enter* the Ensembl backbone and **whether** the remainder of
        the traversal must run in *reverse* (new → old) order.

        Selection strategy
        ------------------
        1. **Fixed `from_release`** - If the caller already knows the release
           of the starting node, every candidate is paired with that same
           release and the correct *reverse* flag is derived trivially.
        2. **Non-backbone start** - When the starting node is not an
           Ensembl-gene backbone ID, the synonym whose **active range edge** is
           *closest* (or *farthest*, per `mode`) to `to_release` is chosen.
        3. **Backbone start** - If the query is itself an Ensembl-gene, the
           algorithm first looks for *overlapping* ranges between the query
           and each synonym; if none overlap, it falls back to the distance
           rule described in step 2.

        Args:
            from_id (str): Identifier from which the path search will start.
            synonym_ids (Sequence[str]): Ensembl IDs considered synonyms of
                `from_id` (typically the *same* gene with different version
                numbers).
            to_release (int): Target Ensembl release that the overall conversion
                aims for.
            from_release (int | None): Release in which `from_id` is known to
                be active.  If *None*, the method infers a suitable release for
                each candidate.
            mode (str): Either `'closest'` or `'distant'`—controls whether
                the synonym chosen should minimise or maximise its distance to
                `to_release`.

        Returns:
            list[list[Union[str, int, bool]]]: One or more triplets of the form
            `[synonym_id, entry_release, reverse]` where:

            * `synonym_id` - the chosen synonym,
            * `entry_release` - release at which to join the backbone, and
            * `reverse` - *True* if the subsequent history walk must run
              backwards in time.

        Raises:
            ValueError: If no synonym satisfies the distance/overlap criteria
                or if `mode` is invalid.
        """
        distance_to_target = list()
        candidate_ranges = list()
        # synonym_ids should be ensembl of the same id (different versions)

        if from_release is not None:
            candidate_ranges.extend([[i, from_release, from_release > to_release] for i in synonym_ids])
            return candidate_ranges

        # If the queried ID is not 'ensembl_gene':
        # find the synonym_ID with the closest distance to to_release
        if self.graph.nodes[from_id][DB.node_type_str] != DB.external_search_settings["nts_backbone"]:
            for syn_id in synonym_ids:
                n = self.graph.get_active_ranges_of_id[syn_id]
                m = Track.get_from_release_and_reverse_vars(n, to_release, mode=mode)
                # Find the ranges of syn_id and find the reve
                for m1, m2 in m:
                    min_distance_of_range = abs(m1 - to_release)
                    distance_to_target.append(min_distance_of_range)
                    candidate_ranges.append([syn_id, m1, m2])

        else:
            # If the queried ID and synonyms has some overlapping ranges:
            # find the synonym_ID which has coinciding release with from_id and closest to the to_release.
            for syn_id in synonym_ids:
                n = self.graph.get_two_nodes_coinciding_releases(from_id, syn_id)
                m = Track.get_from_release_and_reverse_vars(n, to_release, mode=mode)

                for m1, m2 in m:
                    min_distance_of_range = abs(m1 - to_release)
                    distance_to_target.append(min_distance_of_range)
                    candidate_ranges.append([syn_id, m1, m2])
            # This can output multiple ID and/or multiple range.

            # If the queried ID and synonyms has no overlapping ranges:
            # find the synonym_ID with the closest distance to from_id
            if len(distance_to_target) == 0:
                # Find the closest point (1 or 2 exist due to reverse orientation thing)
                # of from_id range to the to_release, if it does not contain it.
                from_id_range = self.graph.get_active_ranges_of_id[from_id]
                if TheGraph.is_point_in_range(from_id_range, to_release):
                    new_to_release = [to_release]
                else:
                    flattened_fir = [i for j in from_id_range for i in j]
                    distances_to_rel = [abs(to_release - i) for i in flattened_fir]
                    minimum_distance = min(distances_to_rel)
                    new_to_release = [
                        edge_rel
                        for ind, edge_rel in enumerate(flattened_fir)
                        if minimum_distance == distances_to_rel[ind]
                    ]

                for syn_id in synonym_ids:
                    for ntr in new_to_release:
                        n = self.graph.get_active_ranges_of_id[syn_id]
                        # Find correct from_release, the closest range edge to the from_id
                        m = Track.get_from_release_and_reverse_vars(n, ntr, mode=mode)
                        for m1, _ in m:
                            # Find correct reverse_info
                            m2 = to_release <= m1
                            min_distance_of_range = abs(m1 - ntr)
                            distance_to_target.append(min_distance_of_range)
                            candidate_ranges.append([syn_id, m1, m2])

        if len(distance_to_target) == 0:
            raise ValueError("len(distance_to_target) == 0")

        global_min_distance = min(distance_to_target)
        result = [item for ind, item in enumerate(candidate_ranges) if global_min_distance == distance_to_target[ind]]

        return result  # [[id, new_from_id, new_reverse], [id, new_from_id, new_reverse], ]

        # a->x->z3,6,9 ise
        # given final release
        # given from release

    def choose_relevant_synonym(
        self, the_id: str, depth_max: int, to_release: int, filter_node_type: set[str], from_release: Optional[int]
    ):
        """Wrapper that **discovers, clusters, and ranks** synonymous Ensembl candidates for a given identifier.

        The function performs three steps:

        1. **Discover** paths to *all* Ensembl-gene nodes that share the same
           biological identity (`synonymous_nodes`).
        2. **Cluster** those paths by **gene ID** (ignoring version).
        3. **Rank** each cluster with
           :py:meth:`_choose_relevant_synonym_helper`, selecting the entry release
           (and direction) that best suits `to_release`.

        Args:
            the_id (str): Source identifier (Ensembl or external).
            depth_max (int): Maximum depth passed to
                :py:meth:`synonymous_nodes`; governs how far the synonym search is
                allowed to roam through external nodes.
            to_release (int): Target Ensembl release required by the overall
                conversion.
            filter_node_type (set[str]): Node-types that the synonym search
                must terminate on (usually `{'ensembl_gene'}`).
            from_release (int | None): Known active release of `the_id`.  If
                *None*, the helper will infer one.

        Returns:
            list[list[Any]]: A list whose elements are

            `[synonym_id, entry_release, reverse, identifier_path, node_type_path]`

            where the last two items reproduce the path returned by
            :py:meth:`synonymous_nodes`.

        Notes:
            *The method purposefully keeps **all** equally-ranked candidates;
            further tie-breaking is deferred to the main path-scoring routine.*
        """
        syn = self.synonymous_nodes(the_id, depth_max, filter_node_type, from_release=from_release)
        # help to choose z for a->x->z3,6,9
        # filter_node_type == 'ensembl_gene'
        # it also returns itself, which is important

        syn_ids: dict = dict()
        for syn_p, syn_db in syn:
            si = syn_p[-1]
            s = self.graph.nodes[si]["ID"]
            if s not in syn_ids:
                syn_ids[s] = []
            syn_ids[s].append([si, syn_p, syn_db])

        # remove ens_gene -> ens-transcript -> ens_gene

        result = list()  # si, from_rel, reverse, syn_p, syn_db
        for s in syn_ids:  # the one with the same id
            si_list, syn_p_list, syn_db_list = map(list, zip(*syn_ids[s]))
            # could give route to same id from multiple routes

            best_ids_best_ranges = self._choose_relevant_synonym_helper(
                the_id, si_list, to_release, from_release, mode="closest"
            )

            for a1, a2, a3 in best_ids_best_ranges:
                for_filtering = [i == a1 for i in si_list]

                for b1, b2 in zip(
                    itertools.compress(syn_p_list, for_filtering), itertools.compress(syn_db_list, for_filtering)
                ):
                    result.append([a1, a2, a3, b1, b2])

        return result  # new_from_id, new_from_rel, new_reverse, path, path_db

    def get_next_edges(self, from_id: str, from_release: int, reverse: bool, debugging: bool = False):
        """Enumerate **chronologically admissible** history edges from a node.

        Starting at `from_id` and release `from_release`, the method scans
        outgoing (or incoming, when `reverse` is *True*) edges whose
        timestamps allow the path to **advance** in the desired temporal
        direction.  It collapses duplicate "same-ID" transitions and flags
        self-loops so that later heuristics can treat branch points and tips
        differently.

        Args:
            from_id (str): Current node from which the search will step.
            from_release (int): Release at which the current node is known to
                exist.
            reverse (bool): *False* to walk **forward** in history
                (old → new), *True* to walk **backward** (new → old).
            debugging (bool): If set, disables the duplicate-edge
                collapse so that unit tests can inspect the raw edge set.

        Returns:
            list[list[Union[int, bool, str, int]]]: Sorted list of edge
            descriptors, each of which is

            `[edge_release, is_self_loop, src_node, dst_node, multiedge_key]`.

        Raises:
            ValueError: If inconsistent multi-edges (same nodes, same release)
                are detected—this signals a corrupted graph build.
        """
        edges: list = list()
        more_than_one_edges: dict = dict()
        index_counter: int = 0
        for node_after in nx.neighbors(self.graph if not reverse else self.graph.rev, from_id):
            # This forces to follow the same form tree during the recursion
            if self.graph.nodes[node_after][DB.node_type_str] == self.graph.nodes[from_id][DB.node_type_str]:
                for multi_edge_id, an_edge in (
                    (self.graph if not reverse else self.graph.rev).get_edge_data(from_id, node_after).items()
                ):
                    self_loop = node_after == from_id
                    edge_release = an_edge["old_release"] if not reverse else an_edge["new_release"]

                    if (
                        (not reverse and edge_release >= from_release)
                        or (reverse and edge_release <= from_release)
                        or (not reverse and np.isinf(an_edge["new_release"]))
                    ):  # keep last node
                        list_to_add = [edge_release, self_loop, from_id, node_after, multi_edge_id]

                        node_after_id = self.graph.nodes[node_after]["ID"]
                        # node_after_ver = self.graph.nodes[node_after]["Version"]
                        from_id_id = self.graph.nodes[from_id]["ID"]
                        bool_check = (
                            not debugging
                            and node_after_id == from_id_id  # for the same ID transitions
                            and from_id != node_after
                        )  # if it is not self loop

                        if bool_check and node_after_id in more_than_one_edges:  # if this happened one before
                            prev_edge_release_index = more_than_one_edges[node_after_id]
                            prev_edge_release = edges[prev_edge_release_index][0]
                            if prev_edge_release == edge_release:
                                raise ValueError("prev_edge_release == edge_release")
                            if (not reverse and prev_edge_release > edge_release) or (
                                reverse and prev_edge_release < edge_release
                            ):  # keep only the first possible edge!
                                edges[prev_edge_release_index] = list_to_add
                        else:
                            if bool_check:
                                more_than_one_edges[node_after_id] = index_counter
                            edges.append(list_to_add)
                            index_counter += 1

        simultaneous = [e[0] for e in edges]
        for ind, edge in enumerate(edges):
            if edge[1]:
                if simultaneous.count(edge[0]) > 1:  # it is not only_self_loop, it is branched there.
                    edges[ind][1] = False

        return sorted(edges, reverse=reverse)  # sort based on history

    def should_graph_reversed(self, from_id, to_release):
        """Determine the **temporal orientation** of the graph walk.

        Given an identifier that is active in one or more release intervals,
        the routine decides whether the subsequent path-finder must move
        **forward in time**, **backward in time**, or explore *both* directions
        in parallel in order to reach the target release.

        The decision is based on the *closest* boundary of every active
        interval returned by
        :py:meth:`Track.get_from_release_and_reverse_vars` (`mode='closest'`).

        Args:
            from_id (str): The starting identifier (Ensembl gene, transcript,
                protein, or external ID).
            to_release (int): The Ensembl release the user wishes to convert
                *to*.

        Returns:
            tuple[str, Union[int, tuple[int, int]]]:

                * `'forward'` - walk old → new, starting at the **earliest**
                    release in which *from_id* is active
                    &nbsp;&nbsp;&nbsp;→ return `('forward', start_release)`
                * `'reverse'` - walk new → old, starting at the **latest**
                    active release
                    &nbsp;&nbsp;&nbsp;→ return `('reverse', start_release)`
                * `'both'` - split search: one forward walk and one reverse
                    walk
                    &nbsp;&nbsp;&nbsp;→ return `('both', (forward_start,
                    reverse_start))`

        Raises:
            ValueError: If *from_id* is never active in or around
                `to_release` (i.e. no viable starting release can be found).
        """
        n = self.graph.get_active_ranges_of_id[from_id]  # calculates ensembl_gene nodes and also any other node.
        m = Track.get_from_release_and_reverse_vars(n, to_release, mode="closest")

        forward_from_ids = [i for i, j in m if not j]
        reverse_from_ids = [i for i, j in m if j]

        lffi = len(forward_from_ids)
        lrfi = len(reverse_from_ids)

        if lrfi and lffi:
            return "both", (min(forward_from_ids), max(reverse_from_ids))
        elif lrfi:
            return "reverse", max(reverse_from_ids)
        elif lffi:
            return "forward", min(forward_from_ids)
        else:
            raise ValueError("no other choice")

    def path_search(
        self,
        from_id: str,
        from_release: int,
        to_release: int,
        reverse: bool,
        external_settings: dict,
        external_jump: Optional[float] = None,
        multiple_ensembl_transition: bool = False,
    ) -> set:
        """Enumerate every admissible **history path** from *from_id* at *from_release* to *to_release*.

        The algorithm performs a depth-first traversal of the Ensembl history
        edges.  Whenever it becomes stranded on a non-backbone node it may
        "beam-up" via a *synonym path* through an external database, subject to
        the constraints in `external_settings`:

        * `synonymous_max_depth` - maximum depth of a synonym search.
        * `jump_limit` - maximum number of external "beam-up" jumps allowed.
        * `nts_backbone` - canonical node-type of the Ensembl backbone.

        Additional flags control the initial conditions:

        * Setting `external_jump` to *np.inf* **disables** external jumps.
            Setting it to *None* **enables** them with the counter reset to `0`.
        * `multiple_ensembl_transition` allows the algorithm to time-travel to
            a *different* release while still on an external node; this is useful
            when *from_release* was inferred and might not actually connect.

        Args:
            from_id (str): Identifier to start the search from.
            from_release (int): Release number where *from_id* is considered
                active.
            to_release (int): Target Ensembl release.
            reverse (bool): If *True*, traverse the graph **backwards** in time;
                otherwise forwards.
            external_settings (dict): Copy of
                :py:attr:`DB.external_search_settings` that governs depth,
                jump limits, and backbone node-type.
            external_jump (float | None): Current external-jump count
                (*None* starts from zero, *np.inf* forbids any jump).
            multiple_ensembl_transition (bool): Permit the synonym
                engine to select a *different* release for an external node when
                no path exists at *from_release*.

        Returns:
            set[tuple[tuple[str, str, int]]]:
                A set of **edge-lists**.  Each edge is stored as
                `(src, dst, key)`; an empty walk that terminates immediately
                is represented by `((None, from_id, None),)`.

        Notes:
            *The method is intentionally side-effect free; it constructs all
            intermediate data on the stack and returns a fresh set.*
        """

        def _recursive_function(
            __from_id: str,
            __from_release: int,
            __reverse: bool,
            __beamed_up: bool,
            __external_jump: Optional[float],
            __edge_hist: Optional[list],
            __edge_hist_non_backbone: Optional[set],
        ) -> None:
            def _external_path_maker(a_from_id, a_ens_rel, a_syn_pth, free_from_release: bool, reverse: bool):
                a_edge_hist_alt = list()
                a_from_id_ext_path = copy.deepcopy(a_from_id)
                for a_path_ind, a_next_node in enumerate(a_syn_pth):
                    if a_path_ind == 0:
                        continue
                    # 0 is tested TheGraph.is_node_consistency_robust
                    a_edge_hist_alt.append(
                        (
                            a_from_id_ext_path,
                            a_next_node,
                            0,
                            a_ens_rel if not free_from_release else self._external_entrance_placeholder[reverse],
                        )
                    )
                    a_from_id_ext_path = copy.deepcopy(a_next_node)
                return a_edge_hist_alt

            def _non_backbone_finder(_syn_path_):
                return {
                    i
                    for i in _syn_path_
                    if self.graph.nodes[i][DB.node_type_str] != DB.external_search_settings["nts_backbone"]
                }

            def _recurring_element(_syn_path_):
                for _syn_path_i in _syn_path_:
                    if _syn_path_i in _edge_hist_non_backbone:
                        return False

                return True

            _edge_hist = list() if __edge_hist is None else __edge_hist
            _edge_hist_non_backbone = set() if __edge_hist_non_backbone is None else __edge_hist_non_backbone
            _external_jump = 0 if __external_jump is None else __external_jump
            next_edges = self.get_next_edges(__from_id, __from_release, __reverse)

            if (
                len(_edge_hist) == 0
                and len(next_edges) == 0
                and self.graph.nodes[__from_id][DB.node_type_str] != DB.external_search_settings["nts_backbone"]
                # the step input is actually external
            ):
                # get syn only for given release
                if not multiple_ensembl_transition:
                    s = self.choose_relevant_synonym(
                        __from_id,
                        depth_max=external_settings["synonymous_max_depth"],
                        to_release=to_release,
                        filter_node_type={external_settings["nts_backbone"]},
                        # do not time travel here, go with the same release
                        from_release=__from_release,
                    )

                # If from release is infered by the program (not provided by the user), then run pathfinder from all
                # possible ones. This will increase the computational demand but yield more robust results.
                switch_met = False
                if multiple_ensembl_transition or len(s) == 0:
                    s = self.choose_relevant_synonym(
                        __from_id,
                        depth_max=external_settings["synonymous_max_depth"],
                        to_release=to_release,
                        filter_node_type={external_settings["nts_backbone"]},
                        # if there is way to find with the same release, go just find it in other releases.
                        from_release=None,
                    )
                    switch_met = True

                for s1, s2, s3, s4, _s5 in s:  # new_from_id, new_from_rel, new_reverse, path, path_db
                    # TODO: remove path_db (s5) from the function and pathfinder.

                    if _recurring_element(s4):
                        # with synonym route, can't go synonym finding in the next iteration
                        _recursive_function(
                            s1,  # __from_id
                            s2,  # __from_release
                            s3,  # __reverse
                            True,  # __beamed_up
                            _external_jump,  # __external_jump. It does not count as it is starting point
                            _edge_hist + _external_path_maker(__from_id, s2, s4, switch_met, s3),  # __edge_hist
                            _edge_hist_non_backbone | _non_backbone_finder(s4),  # __edge_hist_non_backbone
                        )
                        # Add parallel path finding searches

            else:
                for _edge_release, _only_self_loop, _from_id, _node_after, _multi_edge_id in next_edges:
                    # Synonymous genes of the gene of interest until the next node in the history travel.

                    if not __beamed_up and _external_jump < external_settings["jump_limit"]:
                        s = self.choose_relevant_synonym(
                            _from_id,
                            depth_max=external_settings["synonymous_max_depth"],
                            to_release=to_release,
                            filter_node_type={external_settings["nts_backbone"]},
                            from_release=None,
                        )

                        for s1, s2, s3, s4, _s5 in s:  # new_from_id, new_from_rel, new_reverse, path, path_db
                            # TODO: remove path_db (s5) from the function and pathfinder.

                            if _recurring_element(s4):
                                # with synonym route, don't go synonym finding in the next iteration
                                _recursive_function(
                                    s1,  # __from_id
                                    s2,  # __from_release
                                    s3,  # __reverse
                                    True,  # __beamed_up
                                    _external_jump + 1.0,  # __external_jump
                                    _edge_hist + _external_path_maker(_from_id, s2, s4, True, s3),  # __edge_hist
                                    _edge_hist_non_backbone | _non_backbone_finder(s4),  # __edge_hist_non_backbone
                                )
                                # Add parallel path finding searches

                    # History travel

                    dict_key = (_from_id, _node_after, _multi_edge_id)

                    if dict_key not in _edge_hist:
                        # self loops and, o yoldan daha önce geçmiş mi. extinction event'i var mı
                        _from_id_ver = self.graph.nodes[_from_id]["Version"]

                        if __reverse:
                            if _edge_release <= to_release:
                                if _from_id_ver not in DB.alternative_versions:
                                    all_paths.add(
                                        tuple(_edge_hist) if len(_edge_hist) > 0 else ((None, __from_id, None),)
                                    )
                            elif _only_self_loop:
                                _edge_other = self.graph.rev.get_edge_data(*dict_key)["old_release"]
                                if _edge_other <= to_release:  # and _edge_other != np.inf
                                    all_paths.add(tuple(_edge_hist + [dict_key]))
                                else:
                                    _edge_hist.append(dict_key)
                            else:
                                _recursive_function(
                                    _node_after,  # __from_id
                                    _edge_release,  # __from_release
                                    __reverse,  # __reverse
                                    False,  # __beamed_up
                                    _external_jump,  # __external_jump
                                    _edge_hist + [dict_key],  # __edge_hist
                                    _edge_hist_non_backbone,  # __edge_hist_non_backbone
                                )
                        else:  # if not reverse
                            if _edge_release >= to_release:
                                if _from_id_ver not in DB.alternative_versions:
                                    all_paths.add(
                                        tuple(_edge_hist) if len(_edge_hist) > 0 else ((None, __from_id, None),)
                                    )
                            elif _only_self_loop:  # latest also goes here
                                _edge_other = self.graph.get_edge_data(*dict_key)["new_release"]
                                _is_latest_loop = np.isinf(_edge_other)
                                if _edge_other >= to_release and not _is_latest_loop:
                                    all_paths.add(tuple(_edge_hist + [dict_key]))
                                elif _is_latest_loop:
                                    all_paths.add(
                                        tuple(_edge_hist) if len(_edge_hist) > 0 else ((None, __from_id, None),)
                                    )
                                else:
                                    # Do not parallelize (recursive) at that point. keep loop for all.
                                    _edge_hist.append(dict_key)
                            else:
                                _recursive_function(
                                    _node_after,  # __from_id
                                    _edge_release,  # __from_release
                                    __reverse,  # __reverse
                                    False,  # __beamed_up
                                    _external_jump,  # __external_jump
                                    _edge_hist + [dict_key],  # __edge_hist
                                    _edge_hist_non_backbone,  # __edge_hist_non_backbone
                                )

        all_paths: set = set()
        _recursive_function(
            from_id,  # __from_id
            from_release,  # __from_release
            reverse,  # __reverse
            False,  # __beamed_up
            external_jump,  # __external_jump
            None,  # __edge_hist
            None,  # __edge_hist_non_backbone
        )

        return all_paths

    def get_possible_paths(
        self,
        from_id: str,
        from_release: int,
        to_release: int,
        reverse: bool,
        go_external: bool = True,
        increase_depth_until: int = 2,
        increase_jump_until: int = 0,
        from_release_inferred: bool = False,
    ) -> tuple:
        """Run :py:meth:`path_search` under **progressively relaxed settings**.

        Run :py:meth:`path_search` under **progressively relaxed settings** until at
        least one viable path is found—or every relaxation level is exhausted.

        Four search stages are attempted in order:

        1. **Backbone-only** - external jumps disabled.
        2. **External enabled** - allow external jumps; increment synonym depth
            and jump limit after each failure up to
            `increase_depth_until`/`increase_jump_until`.
        3. **Backbone with multiple-Ensembl transition** - external disabled but
            permit starting release to shift on external nodes.
        4. **External + multiple-transition** - most permissive search, with
            iterative depth/jump relaxation as in stage 2.

        Args:
            from_id (str): Identifier to convert.
            from_release (int): Release at which the search begins.
            to_release (int): Desired target release.
            reverse (bool): Traverse the Ensembl history backwards if *True*,
                forwards otherwise.
            go_external (bool): If *False*, skip any stage that
                requires external jumps.
            increase_depth_until (int): Additional synonym-search
                depth to allow beyond the default.
            increase_jump_until (int): Additional external-jump count
                to allow beyond the default.
            from_release_inferred (bool): *Reserved for future use.*
                Indicates that *from_release* was chosen automatically rather
                than provided by the user.

        Returns:
            tuple[tuple[tuple[str, str, int]]]:
                All paths discovered by the most restrictive stage that yielded
                **at least one** result, returned as an immutable tuple.

        Notes:
            The function copies and mutates
            :py:attr:`DB.external_search_settings` internally; the caller’s
            copy is not modified.
        """
        es: dict = copy.deepcopy(DB.external_search_settings)
        idu = increase_depth_until + es["synonymous_max_depth"]
        iju = increase_jump_until + es["jump_limit"]

        # Try first with no external jump route
        es = copy.deepcopy(DB.external_search_settings)
        all_paths = self.path_search(
            from_id=from_id,
            from_release=from_release,
            to_release=to_release,
            reverse=reverse,
            external_settings=es,
            external_jump=np.inf,
            multiple_ensembl_transition=False,
        )

        # Activate external jump and increase the depth of search at each step
        es = copy.deepcopy(DB.external_search_settings)
        while go_external and len(all_paths) < 1:
            all_paths = self.path_search(
                from_id=from_id,
                from_release=from_release,
                to_release=to_release,
                reverse=reverse,
                external_settings=es,
                external_jump=None,
                multiple_ensembl_transition=False,
            )
            if es["synonymous_max_depth"] < idu:
                es["synonymous_max_depth"] = es["synonymous_max_depth"] + 1
            elif es["jump_limit"] < iju:
                es["jump_limit"] = es["jump_limit"] + 1
            else:
                break

        # If none found, make relaxed search in terms of ensembl transition.
        es = copy.deepcopy(DB.external_search_settings)
        if len(all_paths) < 1:
            all_paths = self.path_search(
                from_id=from_id,
                from_release=from_release,
                to_release=to_release,
                reverse=reverse,
                external_settings=es,
                external_jump=np.inf,
                multiple_ensembl_transition=True,
            )

        # The same as above except this time with relaxed transition.
        es = copy.deepcopy(DB.external_search_settings)
        while go_external and len(all_paths) < 1:
            all_paths = self.path_search(
                from_id=from_id,
                from_release=from_release,
                to_release=to_release,
                reverse=reverse,
                external_settings=es,
                external_jump=None,
                multiple_ensembl_transition=True,
            )
            if es["synonymous_max_depth"] < idu:
                es["synonymous_max_depth"] = es["synonymous_max_depth"] + 1
            elif es["jump_limit"] < iju:
                es["jump_limit"] = es["jump_limit"] + 1
            else:
                break

        return tuple(all_paths)

    @cached_property
    def _calculate_node_scores_helper(self):
        """Build and cache helper look-ups for node-scoring.

        The property constructs two complementary data structures:

        * **filter_set** - the union of
            (1) every external-database node-type present in the graph, and (2) every Ensembl-specific
            node-type (gene, transcript, translation, …) across *all* assemblies.
            This set can therefore be passed unmodified to
            :py:meth:`synonymous_nodes` to ask for "anything that is not an assembly-less backbone gene".
        * **ensembl_include** - a mapping
            `{form → set(node_type_str)}` where each value lists the node-types
            that should be considered equivalent to that *form* (e.g. *gene*, *transcript*, *translation*) when
            computing richness metrics.

        Returns:
            tuple[set[str], dict[str, set[str]]]:
                `(filter_set, ensembl_include)` exactly as described above.
        """
        form_list = [i for i in self.graph.available_forms if i != DB.backbone_form]
        ensembl_include = dict()
        filter_set = copy.deepcopy(self.graph.available_external_databases)
        for i in form_list:
            ensembl_include_form = set([DB.nts_ensembl[i]] + [DB.nts_assembly[j][i] for j in DB.nts_assembly])
            ensembl_include[i] = ensembl_include_form
            filter_set.update(ensembl_include_form)
        return filter_set, ensembl_include

    def calculate_node_scores(self, the_id, ens_release) -> list:
        """Rank competing Ensembl targets by the "richness" of their synonyms.

        The method counts, within a radius of *two* synonym hops, how many unique
        identifiers of various categories point to each candidate and returns the
        counts as **negative integers** so that *smaller* is *better* for the
        up-stream sorter.

        Args:
            the_id (str): Identifier that is being converted.
            ens_release (int): Target Ensembl release; only synonyms active in this
                release are considered.

        Returns:
            list: `[-ext, -form₁, -form₂]` where
                * `ext`  - number of distinct **external-database** synonyms.
                * `form₁` - number of distinct synonyms of the most important Ensembl form (typically *gene*).
                * `form₂` - number of distinct synonyms of the second form (typically *transcript* or *translation*).

        Raises:
            ValueError: If the graph does not expose exactly the two expected
                non-backbone forms, or if a synonym node's type cannot be mapped to
                *external*, *form₁*, or *form₂*.
        """
        form_list = [i for i in self.graph.available_forms if i != DB.backbone_form]
        form_importance_order = ["gene", "transcript", "translation"]
        form_importance_order = [i for i in form_importance_order if i in form_list]
        if len(form_importance_order) != 2:
            raise ValueError(form_list, form_importance_order)

        filter_set, ensembl_include = self._calculate_node_scores_helper
        _temp = [
            i[-1]
            for i, _ in self.synonymous_nodes(
                the_id,
                depth_max=2,
                filter_node_type=filter_set,
                from_release=ens_release,
                ensembl_backbone_shallow_search=True,
                account_for_hyperconnected_nodes=True,
            )
        ]
        imp1, imp2, imp3 = set(), set(), set()
        for i in _temp:
            nt = self.graph.nodes[i][DB.node_type_str]
            if nt in DB.nts_external:
                imp1.add(i)
            elif nt in ensembl_include[form_importance_order[0]]:
                imp2.add(i)
            elif nt in ensembl_include[form_importance_order[1]]:
                imp3.add(i)
            else:
                raise ValueError("unexpected!2")

        # Importance order is as following
        return [-len(imp1), -len(imp2), -len(imp3)]  # minus is added to minimize in the method used.

    def calculate_score_and_select(
        self,
        all_possible_paths: tuple,
        reduction: Callable,
        remove_na: str,
        from_releases: Iterable[int],
        to_release: int,
        score_of_the_queried_item: float,
        return_path: bool,
        from_id: str,
    ) -> dict:
        """Collapse a set of candidate paths into the single best path per target.

        For each path produced by the search engine the function:

        1. Computes an **edge-score aggregate** using `reduction` while handling
        missing values as directed by `remove_na`.
        2. Tallies *external* statistics (steps, jumps, initial conversion
        confidence) and *assembly* statistics (number of priority drops, final
        priority).
        3. Packs all metrics into a dictionary and stores it under the key of the
        path's final destination node.
        4. Keeps only the lexicographically "smallest" dictionary per destination
        via :py:meth:`_path_score_sorter_single_target`.

        Args:
            all_possible_paths (tuple): Sequence of edge-lists representing
                every admissible walk returned by the path-finder.
            reduction (Callable): Function such as
                `np.mean` or `sum` used to collapse edge weights into one number.
            remove_na (str): How to treat *NaN* edge weights - one of
                `'omit'`, `'to_1'`, `'to_0'`.
            from_releases (Iterable[int]): Release that each path starts from;
                must align with `all_possible_paths`.
            to_release (int): Target release - needed to know whether an edge is
                traversed forward or reverse.
            score_of_the_queried_item (float): Fallback weight for the implicit edge
                that represents the *query ID* itself.
            return_path (bool): If *True*, embed the full edge-list inside each
                score dict under the key `'the_path'`.
            from_id (str): Original identifier being converted - echoed back in the
                score dict for traceability.

        Returns:
            dict: Mapping
            `{destination_id → best_score_dict}`.
            Each *score dict* contains (inter alia)
            `assembly_jump`, `external_jump`, `external_step`,
            `edge_scores_reduced`, and `ensembl_step`.

        Raises:
            ValueError: If an unexpected edge encoding is encountered, if an edge
                score is invalid/∞, or if `remove_na` is set to an unknown mode.
        """

        def er_maker_for_initial_conversion(n1, n2, n3, n4):
            edge_key = self.edge_key_orientor(n1, n2, n3)
            return self.graph.get_edge_data(*edge_key)["available_releases"]

        scores: dict = dict()

        for the_path, from_release in zip(all_possible_paths, from_releases):
            edge_scores = list()
            external_step = 0
            external_jump = 0
            in_external = False

            for the_edge in the_path:
                if len(the_edge) == 3:
                    reverse = from_release > to_release

                    if not (the_edge[0] is None and the_edge[2] is None):
                        w = (self.graph if not reverse else self.graph.rev).get_edge_data(*the_edge)["weight"]
                    else:
                        w = score_of_the_queried_item

                    edge_scores.append(w)
                    in_external = False

                elif len(the_edge) == 4:  # External path is followed.
                    external_step += 1
                    if not in_external:  # independent of length of the jump, it should be counted as 1.
                        external_jump += 1
                    in_external = True
                    from_release = the_edge[3]  # _external_path_maker

                else:
                    raise ValueError("len(edge)")

            # longest continous ens_gene path'ine sahip olan one cikmali?

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 0 weight means weight is not defined (np.nan in the graph)
                if any([not isinstance(s, (int, float)) or np.isinf(s) for s in edge_scores]):
                    raise ValueError(f"Unexpected edge score: {the_path, edge_scores}")

                if remove_na == "omit":
                    edge_scores_r = reduction([s for s in edge_scores if not pd.isna(0)])
                elif remove_na == "to_1":
                    edge_scores_r = reduction([s if not pd.isna(0) else 1 for s in edge_scores])
                elif remove_na == "to_0":
                    edge_scores_r = reduction([s if not pd.isna(0) else 0 for s in edge_scores])
                else:
                    raise ValueError(f"Undefined parameter for 'remove_na': {remove_na}")

            final_destination = the_path[-1][1]
            if final_destination not in scores:
                scores[final_destination] = list()

            if len(the_path) == 1 and (the_path[0][0] is None or the_path[0][2] is None):
                # Happens when the path is like following: ((None, 'ENSG00000170558.10', None),)
                # Just check whether this Ensembl ID exist in both assemblies
                assert the_path[0][1] is not None and the_path[0][0] is None and the_path[0][2] is None
                assemblies = list(
                    {
                        j
                        for i in self.graph.combined_edges_genes[the_path[0][1]]
                        for j in self.graph.combined_edges_genes[the_path[0][1]][i]
                    }
                )
                step_pri = sorted(DB.assembly_mysqlport_priority[i]["Priority"] for i in assemblies)
                assembly_jump, current_priority = 0, max(step_pri)
            else:
                the_path = tuple(
                    (
                        tuple(list(the_step[:3]) + [er_maker_for_initial_conversion(*the_step)])
                        if len(the_step) == 4 and the_step[-1] in self._external_entrance_placeholders
                        # -1, 10001 is when you have external to graph conversion and from_release is None.
                        else the_step
                    )
                    for the_step in the_path
                )
                assembly_jump, step_pri, current_priority = self.minimum_assembly_jumps(the_path)
            initial_conversion_conf = len(the_path[0]) == 4 and the_path[0][-1] in self._external_entrance_placeholders
            to_add = {
                # explain each
                "from_id": from_id,
                "assembly_jump": assembly_jump,  # a.k.a assembly penalty
                "external_jump": external_jump,
                "external_step": external_step,
                "initial_conversion_conf": int(not initial_conversion_conf),
                "edge_scores_reduced": -edge_scores_r,  # 'minus' is added so that all variables should be minimized.
                "ensembl_step": len(the_path) - external_step,
                "final_assembly_priority": (step_pri, current_priority),
            }
            if return_path:
                to_add["the_path"] = the_path
            scores[final_destination].append(to_add)

        # choose the best route to each target, ignore others. Do it for each target separately.
        max_score = {i: Track._path_score_sorter_single_target(scores[i]) for i in scores}

        return max_score  # dict of dict

    def edge_key_orientor(self, n1: str, n2: str, n3: int):
        """Return the stored orientation of a multigraph edge.

        For multigraphs every logical edge is stored *once*, but the caller may
        hold `(u, v, k)` or `(v, u, k)`.
        This helper resolves the ambiguity so that subsequent attribute look-ups
        succeed.

        Args:
            n1 (str): One endpoint of the edge.
            n2 (str): The other endpoint.
            n3 (int): Edge key (index) within the `networkx` multi-edge.

        Returns:
            tuple[str, str, int]: A triple that is guaranteed to exist as written
            in `self.graph`.

        Raises:
            AssertionError: If neither orientation is present in the graph.
        """
        if self.graph.has_edge(n1, n2, n3):
            edge_key = (n1, n2, n3)
        else:
            edge_key = (n2, n1, n3)
            assert self.graph.has_edge(*edge_key), edge_key

        return edge_key

    def path_step_possible_assembly_jumps(self, n1, n2, n3, n4=None):
        """Return the genome assemblies that can legally be used for a single edge.

        The helper inspects the edge that connects `n1` → `n2` and filters the
        assemblies recorded on that edge against the *release* constraint `n4`:

        * **None** - the edge is treated as backbone history; the result is the
            graph-wide default assembly (usually the build on which the backbone was
            constructed).
        * **int** - keep only assemblies whose *release set* contains that single release.
        * **set[int]** - keep assemblies whose release set intersects the provided set.

        Args:
            n1 (str): Source node identifier.
            n2 (str): Destination node identifier.
            n3 (int): Edge key within the NetworkX multigraph.
            n4 (int | set[int] | None, optional): Release filter as described above.

        Returns:
            list[str]: Sorted list of assembly names (e.g. `["GRCh37", "GRCh38"]`).

        Raises:
            ValueError: If `n4` is of an unsupported type.
        """
        edge_key = self.edge_key_orientor(n1, n2, n3)
        if n4 is None:
            return [self.graph.graph["genome_assembly"]]
        elif isinstance(n4, int):
            edge_data = self.graph.get_edge_data(*edge_key)[DB.connection_dict]
            return sorted({j for i in edge_data for j in edge_data[i] if n4 in edge_data[i][j]})
        elif isinstance(n4, set):
            edge_data = self.graph.get_edge_data(*edge_key)[DB.connection_dict]
            return sorted({j for i in edge_data for j in edge_data[i] if len(n4 & edge_data[i][j]) > 0})
        else:
            raise ValueError("isinstance(n4, ...)")

    def minimum_assembly_jumps(self, the_path, step_pri=None, current_priority=None) -> tuple:
        """Compute the penalty incurred by assembly downgrades along a path.

        Each path step may be annotated with one or more candidate assemblies.
        These are translated into priority values via
        `DB.assembly_mysqlport_priority`.  The algorithm walks the path,
        tracking the current priority and counting how many times it must drop
        to a lower priority value—each drop constitutes an "assembly jump"
        penalty.

        Args:
            the_path (Iterable[tuple]): Sequence of edge descriptors; each element is
                either `(n1, n2, k)` or `(n1, n2, k, release)`.
            step_pri (list[int] | None, optional): Priority list for the first edge.
                If None, it is derived from `the_path`.
            current_priority (int | None, optional): Starting priority.  If None,
                initialised to `max(step_pri)`.

        Returns:
            tuple[int, list[int], int]:
                - `assembly_jump` - total number of priority drops.
                - `step_pri` - priority list of the last processed edge.
                - `current_priority` - priority value after the final edge.
        """
        assemblies = [self.path_step_possible_assembly_jumps(*i) for i in the_path]
        # should be sorted for bisect function to work properly in '_minimum_assembly_jumps_helper'
        priorities = [[DB.assembly_mysqlport_priority[j]["Priority"] for j in i] for i in assemblies]

        if step_pri is None:
            step_pri = priorities.pop(0)
        if current_priority is None:
            current_priority = max(step_pri)  # The logic is basically follow the lowest priority if possible.

        return Track._minimum_assembly_jumps_helper(
            step_pri=step_pri, current_priority=current_priority, priorities=priorities
        )

    @staticmethod
    def _minimum_assembly_jumps_helper(step_pri, current_priority, priorities):
        """Internal worker for :py:meth:`minimum_assembly_jumps`.

        Given the priority sets for the remaining edges, iterate until all have
        been consumed while updating the *current* assembly priority and counting
        how often it must drop.

        Args:
            step_pri (list[int]): Priority values of the edge currently under
                consideration.
            current_priority (int): Priority value inherited from previous steps.
            priorities (list[list[int]]): Priority lists for the *rest* of the path,
                **already sorted** for correct bisecting.

        Returns:
            tuple[int, list[int], int]: Same three-tuple as documented in
            :py:meth:`minimum_assembly_jumps`.

        Raises:
            ValueError: If the priority lattice enters an impossible state.
        """

        def next_priority(p):
            p_ind = DB.assembly_priority.index(p)
            # cannot return boundaries due to where the function is called.
            return DB.assembly_priority[p_ind + 1]

        def next_priority_2(p, p_list):
            p_ind = bisect.bisect(p_list, p) - 1
            # bisect cannot return boundaries due to where the function is called.
            return p_list[p_ind]

        penalty = 0
        while len(priorities) > 0:
            if max(step_pri) < current_priority:
                # Increase in priority if all assemblies of this step has higher priority than previous.
                current_priority = next_priority(current_priority)
            elif current_priority < min(step_pri):
                current_priority = min(step_pri)
                step_pri = priorities.pop(0)
                penalty += 1  # lower to higher is fine, higher to lower is not.
            elif min(step_pri) <= current_priority <= max(step_pri):
                if current_priority not in step_pri:
                    current_priority = next_priority_2(current_priority, step_pri)
                step_pri = priorities.pop(0)
            else:
                raise ValueError("while len(priorities) > 0")

        return penalty, step_pri, current_priority

    @staticmethod
    def _path_score_sorter_single_target(lst_of_dict: list[dict]) -> dict:
        """Select the best score dictionary for *one* conversion target.

        The input is a list of dictionaries produced by
        :py:meth:`calculate_score_and_select`.  Each dictionary is converted into a
        tuple according to the *lexicographic importance order*

        `("assembly_jump", "external_jump", "external_step",
        "edge_scores_reduced", "ensembl_step")`

        and the dictionary with the **smallest** tuple is returned.

        Args:
            lst_of_dict (list[dict]): Candidate score dictionaries for this target.

        Returns:
            dict: The chosen "winner" score dictionary.

        Raises:
            ValueError: If the input list is empty.
        """
        importance_order = (  # the variables from to_add in 'calculate_score_and_select'.
            "assembly_jump",
            "external_jump",  # e.g. uniprot bridge and hlca bridge becomes equivalent
            "external_step",  # e.g. uniprot bridge and hlca bridge is different
            "edge_scores_reduced",
            "ensembl_step",
        )  # they all are needed to be minimized

        if not len(lst_of_dict) > 0:
            raise ValueError("not len(lst_of_dict) > 0")

        minimum_scores = [[dct[i] for i in importance_order] + [ind] for ind, dct in enumerate(lst_of_dict)]
        minimum_scores = sorted(minimum_scores, reverse=False)
        best_score_index = minimum_scores[0][-1]
        return lst_of_dict[best_score_index]  # choose the best & shortest

    def _path_score_sorter_all_targets(self, dict_of_dict: dict, from_id: str, to_release: int) -> dict:
        """Select the overall best target(s).

        Select the **overall best target(s)** once every candidate Ensembl node
        has itself been reduced to its single best path.

        The method linearises several per-path metrics into an *importance
        order* (see the tuple at the top of the function), then:

        1. Computes that ordered score for **each** pair `(ensembl_gene, final_target)`.
        2. Finds the global minimum; if multiple pairs tie:
            * Prefer the target whose identifier is identical to `from_id`.
            * If more than one Ensembl gene still tie, fall back on
                :py:meth:`calculate_node_scores` to favour the "richer" node.
        3. Returns a *pruned* copy of `dict_of_dict` that contains only the
           surviving Ensembl genes, each with only the winning `final_elements` entry.
           Additional provenance is written to `filter_scores`.

        Args:
            dict_of_dict (dict): Nested result of
                :py:meth:`calculate_score_and_select`.  Keys are candidate
                Ensembl genes; values are dictionaries that already contain
                *one* best path per final target.
            from_id (str): Original query identifier; used to break ties in
                favour of "same as input".
            to_release (int): Target Ensembl release; forwarded to
                :py:meth:`calculate_node_scores` during tie-breaking.

        Returns:
            dict: A reduced version of `dict_of_dict` holding only the winner(s) and enriched with a
                `final_elements[*]['filter_scores']` sub-dict that records the filters applied.

        Raises:
            ValueError: If `dict_of_dict` is empty.
        """
        importance_order = (
            # the variables from to_add in 'calculate_score_and_select'.
            "final_conversion_conf",
            # "initial_conversion_conf", it should be the same for all possible routes.
            "final_conv_asy_min_prior",  # return the one with latest assembly
            "assembly_jump",
            "external_jump",
            # "external_step",  # TODO: calculcate "external_jump_depth" and put here.
            "final_asy_min_prior",
            "final_asy_min_prior_count",
            "final_conv_asy_min_prior_count",
            # "edge_scores_reduced",
            # "ensembl_step",  # Not really relevant
            # "ens_node_importance", # Handled separately at the end due to computational cost
        )  # they all are needed to be minimized

        # max(external_count), max(protein_count), max(transcript_count)

        if not len(dict_of_dict) > 0:
            raise ValueError("not len(dict_of_dict) > 0")

        minimum_scores: dict[tuple, list] = dict()

        for dct in dict_of_dict:
            for target in dict_of_dict[dct]["final_conversion"]["final_elements"]:
                _temp = dict_of_dict[dct]["final_conversion"]["final_elements"]
                ordered_score = {i: dict_of_dict[dct][i] for i in importance_order if i in dict_of_dict[dct]}
                ordered_score["final_asy_min_prior"] = min(dict_of_dict[dct]["final_assembly_priority"][0])
                ordered_score["assembly_jump"] += _temp[target]["additional_assembly_jump"]
                fcc = dict_of_dict[dct]["final_conversion"]["final_conversion_confidence"]
                ordered_score["final_conversion_conf"] = fcc
                ordered_score["final_conv_asy_min_prior"] = _temp[target]["final_assembly_min_priority"]
                ordered_score["final_conv_asy_min_prior_count"] = -_temp[target]["final_assembly_priority_count"]
                ordered_score["final_asy_min_prior_count"] = -len(dict_of_dict[dct]["final_assembly_priority"][0])

                minimum_scores[(dct, target)] = [ordered_score[i] for i in importance_order]

        the_min_key: Callable = minimum_scores.get
        best_score_key = min(minimum_scores, key=the_min_key)
        best_score_value = minimum_scores[best_score_key]
        best_scoring_targets = [i for i in minimum_scores if best_score_value == minimum_scores[i]]
        the_same_switch, node_score_switch = False, False

        # If multiple target's passed here, choose the target that is the same as from_id.
        final_targets = {j for _, j in best_scoring_targets}
        if from_id in final_targets:
            best_scoring_targets = [(i, j) for i, j in best_scoring_targets if j == from_id]
            the_same_switch = True

        # If multiple dct's passed here, calculate the node score importance as an additional metric
        final_ensembl_targets = {i for i, _ in best_scoring_targets}
        if len(final_ensembl_targets) > 1:
            minimum_scores_ns = {et: self.calculate_node_scores(et, to_release) for et in final_ensembl_targets}
            the_min_key_ns: Callable = minimum_scores_ns.get
            best_score_key_ns = min(minimum_scores_ns, key=the_min_key_ns)
            best_score_value_ns = minimum_scores_ns[best_score_key_ns]
            best_scoring_targets_ns = {i for i in minimum_scores_ns if best_score_value_ns == minimum_scores_ns[i]}
            assert len(best_scoring_targets_ns) > 0
            best_scoring_targets = [(i, j) for i, j in best_scoring_targets if i in best_scoring_targets_ns]
            node_score_switch = True

        # Narrow down the results based on the findings and return.
        output = dict()
        for bst, _ in best_scoring_targets:
            output[bst] = copy.deepcopy(dict_of_dict[bst])
            output[bst]["final_conversion"]["final_elements"] = dict()
        for bst, trgt in best_scoring_targets:
            output[bst]["final_conversion"]["final_elements"][trgt] = dict_of_dict[bst]["final_conversion"][
                "final_elements"
            ][trgt]
            output[bst]["final_conversion"]["final_elements"][trgt]["filter_scores"] = {
                "initial_filter": minimum_scores[(bst, trgt)],
                "same_as_input_filter": the_same_switch,
                "node_importance_filter": minimum_scores_ns[bst] if node_score_switch else None,
            }

        return output  # choose the best & shortest

    def convert(
        self,
        from_id: str,
        from_release: Optional[int] = None,
        to_release: Optional[int] = None,
        final_database: Optional[str] = None,
        reduction: Callable = np.mean,
        remove_na="omit",
        score_of_the_queried_item: float = np.nan,
        go_external: bool = True,
        prioritize_to_one_filter: bool = False,
        return_path: bool = False,
        deprioritize_lrg_genes: bool = True,
        return_ensembl_alternative: bool = True,
    ):
        """End-to-end ID conversion workflow.

        Starting from `from_id` the routine

        1. Determines the correct **time-travel direction** if
           `from_release` is unspecified.
        2. Enumerates *all* admissible paths with
           :py:meth:`get_possible_paths` (forward and/or reverse).
        3. Collapses those paths with :py:meth:`calculate_score_and_select`.
        4. Optionally converts the surviving Ensembl gene(s) into
           `final_database` via :py:meth:`_final_conversion`.
        5. Optionally applies a final global selection with
           :py:meth:`_path_score_sorter_all_targets`.

        The output structure mirrors this decision tree and, when
        `return_path` is *True*, embeds the full edge list so that callers
        can audit every hop.

        Args:
            from_id (str): Source identifier (Ensembl, UniProt, RefSeq, …).
            from_release (int | None): Starting Ensembl release.
                *None* → infer from the graph.
            to_release (int | None): Target Ensembl release.
                Defaults to the newest release contained in the graph.
            final_database (str | None): External database to
                convert into.  *None* → stay on the Ensembl gene.
            reduction (Callable): Function (e.g. `numpy.mean`)
                used to collapse per-edge weights.  Must accept an iterable of
                floats and return a float.
            remove_na (str): Strategy for *NaN* edge weights -
                `'omit'`, `'to_1'`, or `'to_0'`.
            score_of_the_queried_item (float): Weight assigned to
                the implicit edge that represents `from_id` itself.
            go_external (bool): Allow jumps through external
                databases when the backbone is disconnected.
            prioritize_to_one_filter (bool): After all scoring,
                keep only the single globally best target.
            return_path (bool): Embed the full edge list(s) in the
                returned dictionary.
            deprioritize_lrg_genes (bool): If *True* and other
                results exist, drop LRG_* genomic regions from the final set.
            return_ensembl_alternative (bool): When converting to an
                external database, also return the Ensembl gene as a fallback.

        Returns:
            dict | None:
                * **dict** - Structured result as described above.
                * **None** - No admissible path was found.

        Raises:
            ValueError: For non-callable `reduction`, unsupported
                `remove_na` modes, unknown `final_database` values, or
                logical inconsistencies detected during processing.
        """
        if not callable(reduction):
            raise ValueError("not callable(reduction)")
        to_release = to_release if to_release is not None else self.graph.graph["ensembl_release"]

        if from_release is None:
            should_reversed, fr = self.should_graph_reversed(from_id, to_release)
            fri = True
        else:
            should_reversed = "forward" if from_release <= to_release else "reverse"
            fr = copy.deepcopy(from_release)
            fri = False

        # fri stands for from_release_infered. When from release is inferred rather than provided by the user,
        # get_possible_paths method first strictly looks at ensembl genes defined the inferred release in
        # external/transcript etc to ensembl gene transition. For ensembl gene query ID, it does not matter.
        # Starting travel history from the first possible ensembl gene ID sometimes yield lost IDs. When lost, the
        # program looks at all possible ensembl gene ID transition possible.

        if should_reversed == "both":
            possible_paths_forward = self.get_possible_paths(
                from_id,
                fr[0],
                to_release,
                go_external=go_external,
                reverse=False,
                from_release_inferred=fri,
            )
            possible_paths_reverse = self.get_possible_paths(
                from_id,
                fr[1],
                to_release,
                go_external=go_external,
                reverse=True,
                from_release_inferred=fri,
            )
            poss_paths = tuple(list(itertools.chain(possible_paths_forward, possible_paths_reverse)))
            ff = itertools.chain(
                itertools.repeat(fr[0], len(possible_paths_forward)),
                itertools.repeat(fr[1], len(possible_paths_reverse)),
            )
        elif should_reversed == "forward":
            poss_paths = self.get_possible_paths(
                from_id, fr, to_release, go_external=go_external, reverse=False, from_release_inferred=fri
            )
            ff = itertools.chain(itertools.repeat(fr, len(poss_paths)))
        elif should_reversed == "reverse":
            poss_paths = self.get_possible_paths(
                from_id, fr, to_release, go_external=go_external, reverse=True, from_release_inferred=fri
            )
            ff = itertools.chain(itertools.repeat(fr, len(poss_paths)))
        else:
            raise ValueError("should_reversed")

        if len(poss_paths) == 0:
            return None
        else:
            converted = self.calculate_score_and_select(
                poss_paths, reduction, remove_na, ff, to_release, score_of_the_queried_item, return_path, from_id
            )  # chooses one path with best score for a given target.

            # They are actually not genes as we understand, they are genomic regions.
            if deprioritize_lrg_genes:
                # Not robust coding here.
                new_converted = {i: converted[i] for i in converted if not i.lower().startswith("lrg")}
                if len(new_converted) > 0:
                    converted = new_converted

            allowed_ones = self.graph.available_external_databases_assembly[DB.main_assembly].union(
                {DB.nts_ensembl[DB.backbone_form], DB.nts_base_ensembl[DB.backbone_form]}
            )
            for cnvt in converted:
                if final_database not in allowed_ones:
                    raise ValueError(f"Final database (`final_database`) is not among allowed ones: {allowed_ones}")
                elif final_database is None or final_database == DB.nts_ensembl[DB.backbone_form]:
                    prio_list = self._create_priority_list_ensembl(cnvt, to_release)
                    converted[cnvt]["final_conversion"] = Track._final_conversion_dict_prepare(
                        confidence=0,
                        sysns=[cnvt],
                        paths=[[]] if return_path else None,
                        add_ass_jump_list=[0],
                        min_priority_list=[min(prio_list)],
                        len_priority_list=[len(prio_list)],
                        final_database=DB.nts_ensembl[DB.backbone_form],
                    )
                elif (
                    final_database in self.graph.available_external_databases
                    or final_database == DB.nts_base_ensembl[DB.backbone_form]
                ):
                    converted = self._final_conversion(
                        converted, cnvt, final_database, to_release, return_path, return_ensembl_alternative
                    )
                else:
                    raise ValueError("This should not be raised.")

            # if there is no conversable entry, remove the conversion
            converted = {
                i: converted[i] for i in converted if len(converted[i]["final_conversion"]["final_elements"]) > 0
            }

            if len(converted) == 0:
                return None
            elif not prioritize_to_one_filter:
                return converted
            else:
                return self._path_score_sorter_all_targets(converted, from_id, to_release)

    def _create_priority_list_ensembl(self, from_id: str, to_release: int):
        """Build a priority list of assemblies in which `from_id` is active.

        The priorities are the numeric **assembly rankings** defined in
        :py:attr:`DB.assembly_mysqlport_priority` (smaller numbers mean
        higher priority).

        Args:
            from_id (str): Ensembl gene identifier.
            to_release (int): Target Ensembl release; only assemblies that
                contain this release are considered.

        Returns:
            list[int]: Sorted list of priority values (ascending).
        """
        ceg = self.graph.combined_edges_genes[from_id]
        ceg_assembly_list = sorted({j for i in ceg for j in ceg[i] if to_release in ceg[i][j]})
        if len(ceg_assembly_list) == 0:
            self.log.warning(f"A form of rare event found for {from_id!r}.")
            return [np.iinfo(np.int32).max]  # placeholder large integer
        return [DB.assembly_mysqlport_priority[i]["Priority"] for i in ceg_assembly_list]

    @staticmethod
    def _final_conversion_dict_prepare(
        confidence: Union[int, float],
        sysns: list,
        paths: Optional[list[list]],
        min_priority_list: list,
        len_priority_list: list,
        add_ass_jump_list: list,
        final_database: str,
    ):
        """Assemble the final-conversion section that will be attached to a candidate path.

        The section contains a global conversion-confidence flag plus one
        entry per synonym that survived the path-finding stage.  When
        `paths` is None the structure is identical but omits the
        `'the_path'` member to save memory.

        Args:
            confidence (int | float): Heuristic confidence for the *whole*
                conversion step - `0` for "perfect", larger values for
                fallback scenarios, `np.inf` when no conversion was
                possible.
            sysns (list): List of synonym identifiers *in the same order*
                as the metric lists below.
            paths (list[list] | None): One walk (edge list) per synonym, or
                *None* if the caller does not want to expose paths.
            min_priority_list (list): Minimum assembly priority reached
                by each walk.
            len_priority_list (list): Number of distinct assembly
                priorities encountered by each walk.
            add_ass_jump_list (list): Additional assembly-jump penalty
                incurred during the synonym hop itself.
            final_database (str): Name of the database these synonyms belong
                to (e.g. `'uniprot'` or `DB.nts_ensembl[DB.backbone_form]`).

        Returns:
            dict: Nested dictionary ready to be stored under the key
            `'final_conversion'`.
        """
        if paths is not None:
            return {
                "final_conversion_confidence": confidence,
                "final_database": final_database,
                "final_elements": {
                    s: {
                        "final_assembly_priority_count": len_priority_list[ind],
                        "final_assembly_min_priority": min_priority_list[ind],
                        "additional_assembly_jump": add_ass_jump_list[ind],
                        "the_path": tuple(tuple(i) for i in paths[ind]),
                    }
                    for ind, s in enumerate(sysns)
                },
            }
        else:
            return {
                "final_conversion_confidence": confidence,
                "final_database": final_database,
                "final_elements": {
                    s: {
                        "final_assembly_priority_count": len_priority_list[ind],
                        "final_assembly_min_priority": min_priority_list[ind],
                        "additional_assembly_jump": add_ass_jump_list[ind],
                    }
                    for ind, s in enumerate(sysns)
                },
            }

    def _final_conversion(
        self,
        converted: dict,
        cnvt: str,
        final_database: str,
        ens_release: int,
        return_path: bool,
        return_ensembl_alternative: bool,
        prevent_assembly_jumps: bool = True,
        account_for_hyperconnected_nodes: bool = False,
    ):
        """Convert an Ensembl gene node to the requested external database.

        Convert an Ensembl gene node to the requested external database and
        merge the result back into `converted`.

        The routine:

        1. Builds every legal synonym path from `cnvt` to
           `final_database` that is active in `ens_release` (or in any
           release as a fallback).
        2. Computes assembly-jump penalties for each path.
        3. Calls :py:meth:`_final_conversion_dict_prepare` to create the
           conversion sub-dict.
        4. Optionally falls back to returning the Ensembl gene itself when no
           synonym exists and `return_ensembl_alternative` is True.

        Args:
            converted (dict): The current accumulator being built by
                :py:meth:`convert`.
            cnvt (str): Ensembl gene identifier that is undergoing final
                conversion.
            final_database (str): Target external database.
            ens_release (int): Target Ensembl release.
            return_path (bool): If True, embed the path(s) that lead to each
                synonym.
            return_ensembl_alternative (bool): When no synonym can be found,
                add a fallback entry that keeps the Ensembl gene.
            prevent_assembly_jumps: Todo.
            account_for_hyperconnected_nodes: Todo.

        Returns:
            dict: The same `converted` dict, updated in place (and also
            returned for convenience).

        Raises:
            EmptyConversionMetricsError: Todo.
        """

        def _final_conversion_path(gene_id: str, target_db: str, from_release: int):
            """Produce every synonym path from an Ensembl gene to *one* external database.

            The search is attempted twice:
            1. Release-restricted - only synonyms active in
               `from_release`; confidence = 0.
            2. Release-agnostic - synonyms active in any release;
               confidence = 1.

            Args:
                gene_id (str): Ensembl gene identifier acting as the source.
                target_db (str): Desired external database.
                from_release (int): Ensembl release that paths must reach
                    (first attempt).

            Returns:
                tuple[list[list], int]:
                    - `syn_ids_path` - list of edge-lists, one per synonym.
                    - `confidence`   - 0 for strict, 1 for fallback search.
            """

            def _final_conversion_path_helper(res_syn_mth, er: Optional[int]):
                add_to_step = [0] if er is None else [0, er]
                return [[list(i) + add_to_step for i in zip(m1, m1[1:])] for m1, _ in res_syn_mth]

            def _final_conversion_path_helper_2(one_path):
                result = list()
                for k in one_path:
                    edge_key = self.edge_key_orientor(*k)
                    ens_rels_avail = self.graph.get_edge_data(*edge_key)["available_releases"]
                    result.append(ens_rels_avail)
                return result

            def _contains_assembly(lst, asyml: int, er_for_jumps: Optional[int], er_for_last_node: Optional[int]):

                # prevent assembly jumps {some ensembl-gene's are also shared, and they are not called as assembly_37..}
                # just for each returned synonymous, check whether the connection is only possible on assembly 38,
                # if not just remove this one. check graph ans see whether the nodes returned share 38 assembly.

                def _get_edge_data(u, v):
                    e1 = self.graph.get_edge_data(u, v)
                    e2 = self.graph.get_edge_data(v, u)
                    if e1 and e2 is None:
                        return e1
                    elif e2 and e1 is None:
                        return e2
                    elif e1 is None and e2 is None:
                        raise ValueError(f"No edge between {u!r} and {v!r}.")
                    else:
                        raise ValueError(f"Two directional edge between {u!r} and {v!r}.")

                # these are for external jumps to make it allowed in only assembly defined
                switch_1 = False
                for u, v in zip(lst, lst[1:]):
                    edges = _get_edge_data(u, v)
                    for _, edge in edges.items():
                        for _, ae in edge["connection"].items():
                            for assembly, ensembl_release in ae.items():
                                if er_for_jumps is None:
                                    if asyml == assembly:
                                        switch_1 = True
                                else:
                                    if er_for_jumps in ensembl_release and asyml == assembly:
                                        switch_1 = True
                switch_2 = False
                # last node should be in the database requested within assembly defined. See the line:
                # idt.convert_identifier("ENSG00000199293.1", to_release=94, from_release=94, final_database="RFAM")
                edges = _get_edge_data(u, v)
                for _, edge in edges.items():
                    for assembly, ensembl_release in edge["connection"][target_db].items():
                        if er_for_last_node is None:
                            if asyml == assembly:
                                switch_2 = True
                        else:
                            if er_for_last_node in ensembl_release and asyml == assembly:
                                switch_2 = True

                return all([switch_1, switch_2])

            def _prevent_assembly_jumps(synonymous_nodes_output, er_for_jumps, er_for_last_node):
                if prevent_assembly_jumps:
                    return [
                        i
                        for i in synonymous_nodes_output
                        if _contains_assembly(
                            lst=i[0],  # where node ids are stored.
                            asyml=DB.main_assembly,
                            er_for_jumps=er_for_jumps,
                            er_for_last_node=er_for_last_node,
                        )
                    ]
                else:
                    return synonymous_nodes_output

            a = _final_conversion_path_helper(
                _prevent_assembly_jumps(
                    self.synonymous_nodes(
                        the_id=gene_id,
                        depth_max=2,
                        filter_node_type={target_db},
                        from_release=from_release,
                        ensembl_backbone_shallow_search=True,
                        account_for_hyperconnected_nodes=account_for_hyperconnected_nodes,
                    ),
                    er_for_jumps=from_release,
                    er_for_last_node=from_release,
                ),
                er=from_release,
            )

            if len(a) > 0:
                confidence = 0
                return a, confidence  # syns, confidence (lower better)
            else:
                the_paths_no_ens_rel = _final_conversion_path_helper(
                    _prevent_assembly_jumps(
                        self.synonymous_nodes(
                            the_id=gene_id,
                            depth_max=2,
                            filter_node_type={target_db},
                            from_release=None,
                            ensembl_backbone_shallow_search=True,
                            account_for_hyperconnected_nodes=account_for_hyperconnected_nodes,
                        ),
                        er_for_jumps=None,
                        er_for_last_node=from_release,
                    ),
                    er=None,
                )

                for ind1, pt in enumerate(the_paths_no_ens_rel):
                    ens_rels = _final_conversion_path_helper_2(pt)  # add ens_rel sets
                    the_paths_no_ens_rel[ind1] = [i + [ens_rels[ind2]] for ind2, i in enumerate(pt)]
                confidence = 1
                return the_paths_no_ens_rel, confidence

        def _final_conversion_helper(conv_dict: dict, conv_dict_key: str, a_path: list):
            """Re-compute assembly-jump metrics after appending the external conversion path.

            Args:
                conv_dict (dict): Parent entry in `converted` holding the
                    pre-conversion metrics.
                conv_dict_key (str): Key of that entry (the Ensembl gene).
                a_path (list): Edge list representing one complete walk
                    (history + synonym hop).

            Returns:
                list[int | float]: `[assembly_jump, min_priority, n_priorities]`.
            """
            # the path will continue with final conversion so assembly penalty should be calculated again.
            _s1, _s2 = conv_dict[conv_dict_key]["final_assembly_priority"]
            penalty, step_pri, _ = self.minimum_assembly_jumps(a_path, _s1, _s2)
            return [penalty, min(step_pri), len(step_pri)]

        syn_ids_path, conf = _final_conversion_path(cnvt, final_database, from_release=ens_release)

        syn_ids = [i[-1][1] for i in syn_ids_path]

        if len(syn_ids) == 0 and return_ensembl_alternative:
            # Alternativelu return ensembl itself, return confidence np.inf [which means unsuccessful]
            prio_list = self._create_priority_list_ensembl(cnvt, ens_release)

            converted[cnvt]["final_conversion"] = Track._final_conversion_dict_prepare(
                confidence=np.inf,  # as it is not the main target.
                sysns=[cnvt],
                paths=[[]] if return_path else None,
                add_ass_jump_list=[0],
                min_priority_list=[min(prio_list)],
                len_priority_list=[len(prio_list)],
                final_database=DB.nts_ensembl[DB.backbone_form],
            )
        else:
            conversion_metrics = [_final_conversion_helper(converted, cnvt, i) for i in syn_ids_path]
            if not conversion_metrics:
                raise EmptyConversionMetricsError(
                    "The `conversion_metrics` is emtpy. It happens when `return_ensembl_alternative` "
                    f"is `false` and no corresponding final conversion possible: {cnvt}\n{converted}."
                )
            sl1, sl2, sl3 = list(map(list, zip(*conversion_metrics)))
            converted[cnvt]["final_conversion"] = Track._final_conversion_dict_prepare(
                confidence=conf,
                sysns=syn_ids,
                paths=syn_ids_path if return_path else None,
                add_ass_jump_list=sl1,
                min_priority_list=sl2,
                len_priority_list=sl3,
                final_database=final_database,
            )

        return converted

    def identify_source(self, dataset_ids: list[str], mode: str):
        """Infer the most likely origin (assembly and/or Ensembl release) of a heterogeneous identifier list.

        The function tallies how often each origin triple appears among
        `dataset_ids` and returns the counts sorted in descending order.

        Args:
            dataset_ids (list[str]): Collection of identifiers to analyse.
            mode (str): Granularity of the origin to extract - one of
                - 'complete'                  → (assembly, db, release)
                - 'ensembl_release'           → release only
                - 'assembly'                  → assembly only
                - 'assembly_ensembl_release'  → (assembly, release)

        Returns:
            list[tuple[Any, int]]: Pairs `(origin, count)` sorted by
            frequency.

        Raises:
            ValueError: If `mode` is not one of the recognised values.
        """
        possible_trios: list[Any] = list()
        for di in dataset_ids:
            # assembly = self.get_external_assembly()
            if mode == "complete":
                possible_trios.extend(self.graph.node_trios[di])
            elif mode == "ensembl_release":
                possible_trios.extend({i[2] for i in self.graph.node_trios[di]})
            elif mode == "assembly":
                possible_trios.extend({i[1] for i in self.graph.node_trios[di]})
            elif mode == "assembly_ensembl_release":
                possible_trios.extend({i[1:] for i in self.graph.node_trios[di]})
            else:
                raise ValueError("no 1234")

        return list(Counter(possible_trios).most_common())

    def convert_optimized_multiple(self):
        """Placeholder for a **batch-optimised** converter.

        The intended behaviour is to accept *multiple* query IDs and choose a
        conversion target for each such that cross-sample clashes (e.g.
        duplicate loci) are minimised.

        Raises:
            NotImplementedError: Always - the optimisation strategy is not yet implemented.
        """
        raise NotImplementedError  # TODO
