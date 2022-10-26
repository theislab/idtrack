#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import bisect
import copy
import itertools
import logging
import warnings
from collections import Counter
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from ._database_manager import DatabaseManager
from ._db import DB
from ._graph_maker import GraphMaker
from ._the_graph import TheGraph


class Track:
    """Pathfinding algorithm in prepared bio-ID graph.

    Uses :py:class:`_the_graph.TheGraph` in order to calculate the matching ID in queried Ensembl release and queried
    database. It calculates the corresponding IDs from a given ID, by first converting into Ensembl gene ID, then time
    travelling via connected nodes until requested Ensembl release, finally converting into the requested database.
    The class uses two important recursive path finding functions: the one is for time travelling and the other is
    finding the synonymous nodes.
    """

    # ENSG00000263464
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        """Class initialization.

        Args:
            db_manager: See parameter in :py:attr:`_graph.Graph.__init__.db_manager`.
            kwargs: Keyword arguments to be passed to :py:attr:`_the_graph.TheGraph.get_graph`.
        """
        self.log = logging.getLogger("track")
        self.db_manager = db_manager
        graph_creator = GraphMaker(self.db_manager)

        # Calculate/Load the graph
        self.graph = graph_creator.get_graph(**kwargs)
        self.version_info = self.graph.graph["version_info"]
        self._external_entrance_placeholder = -1

    def _recursive_synonymous(
        self,
        _the_id: str,
        synonymous_ones: list,
        synonymous_ones_db: list,
        filter_node_type: set,
        the_path: list = None,
        the_path_db: list = None,
        depth_max: int = 0,
        from_release: int = None,
        ensembl_backbone_shallow_search: bool = False,
    ):
        """Helper method to be used in :py:meth:`_graph.Track.synonymous_nodes`.

        It recursively looks at the graphs and returns a synonymous IDs under two main constrains: the depth
        and the final node type. The depth is simply defined as the total max number of certain database jump to
        be included in the final path. The recursive search does not walk across two nodes that has the same node type
        to prevent this method to travel through time.

        Args:
            _the_id: The name of the node such as Ensembl ID or external ID.
            synonymous_ones: Todo.
            synonymous_ones_db: Todo.
            the_path: Todo.
            the_path_db: Todo.
            depth_max: Todo.
            filter_node_type: Todo.
            from_release: Todo.
            ensembl_backbone_shallow_search: Todo
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

                    if _next_neighbour in self.graph.hyperconnective_nodes:
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
                            and gnt != DB.nts_base_ensembl["gene"]  # transcript or translation base depracated.
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
                    )

    def synonymous_nodes(
        self,
        the_id: str,
        depth_max: int,
        filter_node_type: set,
        from_release: int = None,
        ensembl_backbone_shallow_search: bool = False,
    ):
        """Todo.

        Args:
            the_id: Todo.
            depth_max: Todo.
            filter_node_type: Todo.
            from_release: Todo.
            ensembl_backbone_shallow_search: Todo. assemly bridge'i bozar cunku direction iki tarafli olmali

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if DB.nts_external in filter_node_type:
            raise ValueError(f"Define which external database: '{filter_node_type}'.")

        if ensembl_backbone_shallow_search:
            if depth_max != 2:
                raise ValueError

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
        """Todo.

        Args:
            lor: Todo.
            p: Todo.
            mode: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        result = list()
        # Note that there is no need to consider assembly here because always this function is used in the context of
        # graph backbone nodes (ensembl gene), and the backbone is always the latest (currently 38).
        for l1, l2 in lor:

            if l1 > l2:
                raise ValueError

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
                raise ValueError

        return result

    def _choose_relevant_synonym_helper(
        self, from_id, synonym_ids, to_release: int, from_release: Optional[int], mode: str
    ):
        """Todo.

        Args:
            from_id: Todo.
            synonym_ids: Todo.
            to_release: Todo.
            from_release: Todo.
            mode: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # synonym_ids should be ensembl of the same id (different versions)
        distance_to_target = list()
        candidate_ranges = list()

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
            raise ValueError

        global_min_distance = min(distance_to_target)
        result = [item for ind, item in enumerate(candidate_ranges) if global_min_distance == distance_to_target[ind]]

        return result  # [[id, new_from_id, new_reverse], [id, new_from_id, new_reverse], ]

        # a->x->z3,6,9 ise
        # given final release
        # given from release

    def choose_relevant_synonym(
        self, the_id: str, depth_max: int, to_release: int, filter_node_type: set, from_release: Optional[int]
    ):
        """Todo.

        Args:
            the_id: Todo.
            depth_max: Todo.
            to_release: Todo.
            filter_node_type: Todo.
            from_release: Todo

        Returns:
            Todo.
        """
        # help to choose z for a->x->z3,6,9

        # filter_node_type == 'ensembl_gene'
        syn = self.synonymous_nodes(the_id, depth_max, filter_node_type, from_release=from_release)
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
        """Todo.

        Args:
            from_id: Todo.
            from_release: Todo.
            reverse: Todo.
            debugging: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
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
                                raise ValueError
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
        """Todo.

        Args:
            from_id: Todo.
            to_release: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
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
            raise ValueError

    def path_search(
        self,
        from_id: str,
        from_release: int,
        to_release: int,
        reverse: bool,
        external_settings: dict,
        external_jump: float = None,
        multiple_ensembl_transition: bool = False,
    ) -> set:
        """Todo.

        Args:
            from_id: Todo.
            from_release: Todo.
            to_release: Todo.
            reverse: Todo.
            external_settings: Todo.
            external_jump: Todo.
            multiple_ensembl_transition: Todo.

        Returns:
            _description_
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
            def _external_path_maker(a_from_id, a_ens_rel, a_syn_pth, free_from_release: bool):
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
                            a_ens_rel if not free_from_release else self._external_entrance_placeholder,
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
                            _edge_hist + _external_path_maker(__from_id, s2, s4, switch_met),  # __edge_hist
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
                                    _edge_hist + _external_path_maker(_from_id, s2, s4, True),  # __edge_hist
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
        """Todo.

        Args:
            from_id: Todo.
            from_release: Todo.
            to_release: Todo.
            reverse: Todo.
            go_external: Todo.
            increase_depth_until: Todo.
            increase_jump_until: Todo.
            from_release_inferred: Todo.

        Returns:
            Todo.
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
        form_list = [i for i in self.graph.available_forms if i != DB.backbone_form]
        ensembl_include = dict()
        filter_set = copy.deepcopy(self.graph.available_external_databases)
        for i in form_list:
            ensembl_include_form: set = set([DB.nts_ensembl[i]] + [DB.nts_assembly[j][i] for j in DB.nts_assembly])
            ensembl_include[i] = ensembl_include_form
            filter_set.update(ensembl_include_form)
        return filter_set, ensembl_include

    def calculate_node_scores(self, the_id, ens_release) -> list:
        """A metric to choose from multiple targets.

        Args:
            the_id: Todo.
            ens_release: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
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
                raise ValueError

        # Importance order is as following
        return [-len(imp1), -len(imp2), -len(imp3)]  # minus is added to minimize in the method used.

    def calculate_score_and_select(
        self,
        all_possible_paths: tuple,
        reduction: Callable,
        remove_na: str,
        from_releases: Iterable,
        to_release: int,
        score_of_the_queried_item: float,
        return_path: bool,
        from_id: str,
    ) -> dict:
        """Todo.

        Args:
            all_possible_paths: Todo.
            reduction: Todo.
            remove_na: Todo.
            from_releases: Todo.
            to_release: Todo.
            score_of_the_queried_item: Todo.
            return_path: Todo.
            from_id: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
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
                    raise ValueError

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
                initial_conversion_conf = (
                    len(the_path[0]) == 4 and the_path[0][-1] == self._external_entrance_placeholder
                )
                the_path = tuple(
                    tuple(list(the_step[:3]) + [er_maker_for_initial_conversion(*the_step)])
                    if len(the_step) == 4 and the_step[-1] == self._external_entrance_placeholder
                    # -1 is when you have external to graph conversion and from_release is None.
                    else the_step
                    for the_step in the_path
                )

                assembly_jump, step_pri, current_priority = self.minimum_assembly_jumps(the_path)
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
        """Todo.

        Args:
            n1: Todo.
            n2: Todo.
            n3: Todo.

        Returns:
            Todo.
        """
        if self.graph.has_edge(n1, n2, n3):
            edge_key = (n1, n2, n3)
        else:
            edge_key = (n2, n1, n3)
            assert self.graph.has_edge(*edge_key), edge_key

        return edge_key

    def path_step_possible_assembly_jumps(self, n1, n2, n3, n4=None):
        """Todo.

        Args:
            n1: Todo.
            n2: Todo.
            n3: Todo.
            n4: Todo.

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
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
            raise ValueError

    def minimum_assembly_jumps(self, the_path, step_pri=None, current_priority=None) -> tuple:
        """Todo.

        Args:
            the_path: Todo.
            step_pri: Todo.
            current_priority: Todo.

        Returns:
            Todo.
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
        """Todo.

        Args:
            step_pri: Todo.
            current_priority: Todo.
            priorities: Todo.

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
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
                raise ValueError

        return penalty, step_pri, current_priority

    @staticmethod
    def _path_score_sorter_single_target(lst_of_dict: list) -> dict:
        """Todo.

        Args:
            lst_of_dict: Todo.

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
        """
        importance_order = (  # the variables from to_add in 'calculate_score_and_select'.
            "assembly_jump",
            "external_jump",  # e.g. uniprot bridge and hlca bridge becomes equivalent
            "external_step",  # e.g. uniprot bridge and hlca bridge is different
            "edge_scores_reduced",
            "ensembl_step",
        )  # they all are needed to be minimized

        if not len(lst_of_dict) > 0:
            raise ValueError

        minimum_scores = [[dct[i] for i in importance_order] + [ind] for ind, dct in enumerate(lst_of_dict)]
        minimum_scores = sorted(minimum_scores, reverse=False)
        best_score_index = minimum_scores[0][-1]
        return lst_of_dict[best_score_index]  # choose the best & shortest

    def _path_score_sorter_all_targets(self, dict_of_dict: dict, from_id: str, to_release: int) -> dict:
        """Todo.

        Args:
            dict_of_dict: Todo.
            from_id: Todo.
            to_release: Todo.

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
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
            raise ValueError

        minimum_scores: Dict[tuple, list] = dict()

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
        """Todo.

        Args:
            from_id: Todo.
            from_release: Todo.
            to_release: Todo.
            final_database: Todo.
            reduction: Todo.
            remove_na: Todo.
            score_of_the_queried_item: Todo.
            go_external: Todo.
            prioritize_to_one_filter: Todo.
            return_path: Todo.
            deprioritize_lrg_genes: Todo.
            return_ensembl_alternative: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if not callable(reduction):
            raise ValueError
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
            raise ValueError

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

            for cnvt in converted:
                if final_database is None or final_database == DB.nts_ensembl["gene"]:
                    prio_list = self._create_priority_list_ensembl(cnvt, to_release)
                    converted[cnvt]["final_conversion"] = Track._final_conversion_dict_prepare(
                        confidence=0,
                        sysns=[cnvt],
                        paths=[[]] if return_path else None,
                        add_ass_jump_list=[0],
                        min_priority_list=[min(prio_list)],
                        len_priority_list=[len(prio_list)],
                        final_database=DB.nts_ensembl["gene"],
                    )
                elif (
                    final_database in self.graph.available_external_databases
                    or final_database == DB.nts_base_ensembl["gene"]
                ):
                    converted = self._final_conversion(
                        converted, cnvt, final_database, to_release, return_path, return_ensembl_alternative
                    )
                else:
                    raise ValueError

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
        ceg = self.graph.combined_edges_genes[from_id]
        ceg_assembly_list = sorted({j for i in ceg for j in ceg[i] if to_release in ceg[i][j]})
        if len(ceg_assembly_list) == 0:
            raise ValueError
        return [DB.assembly_mysqlport_priority[i]["Priority"] for i in ceg_assembly_list]

    @staticmethod
    def _final_conversion_dict_prepare(
        confidence: Union[int, float],
        sysns: list,
        paths: Optional[List[List]],
        min_priority_list: list,
        len_priority_list: list,
        add_ass_jump_list: list,
        final_database: str,
    ):
        """Todo.

        Args:
            confidence: Todo.
            sysns: Todo.
            paths: Todo.
            min_priority_list: Todo.
            len_priority_list: Todo
            add_ass_jump_list: Todo.
            final_database: Todo.

        Returns:
            Todo.
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
    ):
        """Todo.

        Args:
            converted: Todo.
            cnvt: Todo.
            final_database: Todo.
            ens_release: Todo.
            return_path: Todo.
            return_ensembl_alternative: Todo.

        Returns:
            Todo.
        """

        def _final_conversion_path(gene_id: str, target_db: str, from_release: int):
            """Todo.

            Args:
                gene_id: Todo.
                target_db: Todo.
                from_release: Todo.

            Returns:
                Todo.
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

            a = _final_conversion_path_helper(
                self.synonymous_nodes(
                    the_id=gene_id,
                    depth_max=2,
                    filter_node_type={target_db},
                    from_release=from_release,
                    ensembl_backbone_shallow_search=True,
                ),
                er=from_release,
            )

            if len(a) > 0:
                confidence = 0
                return a, confidence  # syns, confidence (lower better)
            else:
                the_paths_no_ens_rel = _final_conversion_path_helper(
                    self.synonymous_nodes(
                        the_id=gene_id,
                        depth_max=2,
                        filter_node_type={target_db},
                        from_release=None,
                        ensembl_backbone_shallow_search=True,
                    ),
                    er=None,
                )

                for ind1, pt in enumerate(the_paths_no_ens_rel):
                    ens_rels = _final_conversion_path_helper_2(pt)  # add ens_rel sets
                    the_paths_no_ens_rel[ind1] = [i + [ens_rels[ind2]] for ind2, i in enumerate(pt)]
                confidence = 1
                return the_paths_no_ens_rel, confidence

        def _final_conversion_helper(conv_dict, conv_dict_key, a_path):
            """Todo.

            Args:
                conv_dict: Todo.
                conv_dict_key: Todo.
                a_path: Todo.

            Returns:
                Todo.
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
                final_database=DB.nts_ensembl["gene"],
            )
        else:
            conversion_metrics = [_final_conversion_helper(converted, cnvt, i) for i in syn_ids_path]
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

    def identify_source(self, dataset_ids: list, mode: str):
        """Todo.

        Args:
            dataset_ids: Todo.
            mode: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        possible_trios: List[Any] = list()
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
                raise ValueError

        return list(Counter(possible_trios).most_common())

    def convert_optimized_multiple(self):
        """Accept multiple ID list and return the most optimal set of IDs, minimizing the clashes.

        Raises:
            NotImplementedError: Todo.
        """
        raise NotImplementedError  # TODO
