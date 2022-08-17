#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import itertools
import logging
import re
import warnings
from collections import Counter
from functools import cached_property
from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from ._database_manager import DatabaseManager
from ._db import DB
from ._graph import Graph
from ._verbose import progress_bar


class Track:
    """Pathfinding algorithm in prepared bio-ID graph.

    Uses :py:class:`_graph.Graph` in order to calculate the matching ID in queried Ensembl release and queried
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
            kwargs: Keyword arguments to be used in :py:attr:`_graph.Graph.get_graph.get_graph`.
        """
        self.log = logging.getLogger("track")
        self.db_manager = db_manager
        graph_creator = Graph(self.db_manager)

        # Calculate/Load the graph
        self.graph = graph_creator.get_graph(**kwargs)
        self.reverse_graph = self.graph.reverse(copy=False)
        self.version_info = self.graph.graph["version_info"]

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

        Raises:
            ValueError: Todo.
        """
        input_node_type = self.graph.nodes[_the_id]["node_type"]
        _the_path = [_the_id] if the_path is None else the_path
        _the_path_db = [input_node_type] if the_path_db is None else the_path_db

        counted_elements = Counter(_the_path_db)
        early_termination = [counted_elements[i] for i in filter_node_type if i in counted_elements]
        if depth_max > max(counted_elements.values()):  # The depth is all node_type.

            if len(_the_path) > 0 and (
                self.graph.nodes[_the_path[-1]]["node_type"] in filter_node_type
                or (
                    self.graph.nodes[_the_path[-1]]["node_type"] == "external"
                    and any([i in self.graph.nodes[_the_path[-1]]["release_dict"] for i in filter_node_type])
                )
            ):
                synonymous_ones.append(_the_path)
                synonymous_ones_db.append(_the_path_db)

            elif len(early_termination) == 0 or depth_max > max(early_termination) + 1:
                # no need to add one as it will violate depth_max

                for _direction, graph in (("forward", self.graph), ("reverse", self.reverse_graph)):

                    for _next_neighbour in graph.neighbors(_the_id):

                        gnt = graph.nodes[_next_neighbour]["node_type"]

                        if len(_the_path) >= 2:
                            # prevent bouncing between transcript and gene id.versions
                            l1, l2 = _the_path[-2:]  # [..., l1, l2, gnt]
                            if (
                                self.graph.nodes[l1]["node_type"] == gnt
                                and gnt != "external"
                                and gnt != "base_ensembl_gene"  # transcript, gene or translation
                                and self.graph.nodes[l1]["ID"] == self.graph.nodes[_next_neighbour]["ID"]
                            ):
                                if self.graph.nodes[l1]["node_type"] != "base_ensembl_gene":
                                    # if all the above satisfies, then make sure the below statement.
                                    continue

                        if gnt == "external" or gnt != input_node_type:  # prevent history travel

                            if len(graph[_the_id][_next_neighbour]) > 1:
                                raise ValueError

                            if _next_neighbour not in _the_path:

                                if from_release is not None:
                                    the_edge = graph[_the_id][_next_neighbour][0]  # 0 is verified above
                                    if from_release not in the_edge["releases"]:
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
                                )

    def synonymous_nodes(self, the_id: str, depth_max: int, filter_node_type: set, from_release: int = None):
        """Todo.

        Args:
            the_id: Todo.
            depth_max: Todo.
            filter_node_type: Todo.
            from_release: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        synonymous_ones: list = []
        synonymous_ones_db: list = []
        if "external" in filter_node_type:
            raise ValueError(f"Define which external database: '{filter_node_type}'.")

        self._recursive_synonymous(
            the_id,
            synonymous_ones,
            synonymous_ones_db,
            filter_node_type,
            depth_max=depth_max,
            from_release=from_release,
        )

        remove_set: set = set()
        the_ends_min: dict = dict()

        for p in synonymous_ones:
            e = p[-1]
            lp = len(p)
            am = the_ends_min.get(e, None)
            if am is None or am > lp:
                the_ends_min[e] = lp

        for ind in range(len(synonymous_ones)):
            # choose the minimum path to the same target
            e = synonymous_ones[ind][-1]
            am = the_ends_min[e]
            lp = len(synonymous_ones[ind])
            if lp > am:
                remove_set.add(ind)

        return [
            [synonymous_ones[ind], synonymous_ones_db[ind]]
            for ind in range(len(synonymous_ones))
            if ind not in remove_set
        ]

    def get_active_ranges_of_id_others(self, the_id):
        """Todo.

        Args:
            the_id: Todo.

        Returns:
            Todo.
        """
        pairs = self.memorized_node_database_release_pairs[the_id]
        rels = sorted({p2 for p1, p2 in pairs})
        return Track.list_to_ranges(rels)

    @staticmethod
    def list_to_ranges(i: list):
        """Todo.

        Args:
            i: Todo.

        Returns:
            Todo.
        """
        res = list()
        for _, a in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
            b = list(a)
            res.append([b[0][1], b[-1][1]])
        return res

    def get_active_ranges_of_id(self, the_id):
        """Todo.

        Args:
            the_id: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if self.graph.nodes[the_id]["node_type"] != DB.external_search_settings["backbone_node_type"]:
            raise ValueError

        t_outs = self.get_next_edge_releases(from_id=the_id, reverse=True)
        t_ins = self.get_next_edge_releases(from_id=the_id, reverse=False)

        if len(t_outs) == 0 and len(t_ins) == 0:
            raise ValueError
        elif len(t_outs) == 0:
            assert self.graph.nodes[the_id]["Version"] == DB.no_old_node_id, the_id
            t_outs = [min(self.graph.graph["confident_for_release"])]
        elif len(t_ins) == 0:
            assert self.graph.nodes[the_id]["Version"] == DB.no_new_node_id, the_id
            t_ins = [max(self.graph.graph["confident_for_release"])]

        inout_edges = sorted(
            itertools.chain(zip(t_outs, itertools.repeat(True)), zip(t_ins, itertools.repeat(False))),
            reverse=False,
            key=lambda k: (k[0], -k[1]),
        )

        narrowed = []
        active_state = False
        for ind, (ens_rel, inout) in enumerate(inout_edges):

            if ind == 0:
                assert inout

            if not active_state:
                if inout:
                    narrowed.append(ens_rel)
                    active_state = True
                else:
                    narrowed[-1] = ens_rel
                    active_state = False
            else:
                if inout:
                    pass
                else:
                    narrowed.append(ens_rel)
                    active_state = False
        narrowed = [narrowed[i : i + 2] for i in range(0, len(narrowed), 2)]
        # outputs always increasing, inclusive ranges, for get_intersecting_ranges
        return narrowed

    def get_base_id_range(self, base_id):
        """Todo.

        Args:
            base_id: Todo.

        Returns:
            Todo.
        """
        associated_ids = self.graph.neighbors(base_id)
        all_ranges = sorted(r for ai in associated_ids for r in self.get_active_ranges_of_id(ai))
        return Track.compact_ranges(all_ranges)

    @staticmethod
    def compact_ranges(lor):
        """Todo.

        Args:
            lor: Todo.

        Returns:
            Todo.
        """
        # lot = list of ranges (increasing, inclusive ranges) output of get_active_ranges_of_id
        #  O(n) time and space complexity: a forward in place compaction and copying back the elements,
        #  as then each inner step is O(1) (get/set instead of del)
        next_index = 0  # Keeps track of the last used index in our result
        for index in range(len(lor) - 1):
            if lor[next_index][1] + 1 >= lor[index + 1][0]:
                lor[next_index][1] = lor[index + 1][1]
            else:
                next_index += 1
                lor[next_index] = lor[index + 1]
        return lor[: next_index + 1]

    @staticmethod
    def get_intersecting_ranges(lor1, lor2, compact: bool = True):
        """Todo.

        Args:
            lor1: Todo.
            lor2: Todo.
            compact: Todo.

        Returns:
            Todo.
        """
        # a and b is sorted,
        # Each list will contain lists of length 2, which represent a range (inclusive)
        # the ranges will always increase and never overlap

        result = [
            [max(first[0], second[0]), min(first[1], second[1])]
            for first in lor1
            for second in lor2
            if max(first[0], second[0]) <= min(first[1], second[1])
        ]

        return Track.compact_ranges(result) if compact else result

    def get_two_nodes_coinciding_releases(self, id1, id2, compact: bool = True):
        """Todo.

        Args:
            id1: Todo.
            id2: Todo.
            compact: Todo.

        Returns:
            Todo.
        """
        r1 = self.get_active_ranges_of_id(id1)
        r2 = self.get_active_ranges_of_id(id2)

        r = Track.get_intersecting_ranges(r1, r2, compact=compact)

        return r

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

    @staticmethod
    def is_point_in_range(lor, p):
        """Todo.

        Args:
            lor: Todo.
            p: Todo.

        Returns:
            Todo.
        """
        for l1, l2 in lor:
            if l1 <= p <= l2:
                return True
        return False

    def _choose_relevant_synonym_helper(self, from_id, synonym_ids, to_release):
        """Todo.

        Args:
            from_id: Todo.
            synonym_ids: Todo.
            to_release: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # synonym_ids should be ensembl of the same id (different versions)
        distance_to_target = list()
        candidate_ranges = list()

        # If the queried ID is not 'ensembl_gene':
        # find the synonym_ID with the closest distance to to_release
        if self.graph.nodes[from_id]["node_type"] != DB.external_search_settings["backbone_node_type"]:

            for syn_id in synonym_ids:
                n = self.get_active_ranges_of_id(syn_id)
                m = Track.get_from_release_and_reverse_vars(n, to_release, mode="closest")
                # Find the ranges of syn_id and find the reve
                for m1, m2 in m:
                    min_distance_of_range = abs(m1 - to_release)
                    distance_to_target.append(min_distance_of_range)
                    candidate_ranges.append([syn_id, m1, m2])

        else:
            # If the queried ID and synonyms has some overlapping ranges:
            # find the synonym_ID which has coinciding release with from_id and closest to the to_release.
            for syn_id in synonym_ids:

                n = self.get_two_nodes_coinciding_releases(from_id, syn_id)
                m = Track.get_from_release_and_reverse_vars(n, to_release, mode="closest")

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
                from_id_range = self.get_active_ranges_of_id(from_id)
                if Track.is_point_in_range(from_id_range, to_release):
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
                        n = self.get_active_ranges_of_id(syn_id)
                        # Find correct from_release, the closest range edge to the from_id
                        m = Track.get_from_release_and_reverse_vars(n, ntr, mode="closest")
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
        self, the_id: str, depth_max: int, to_release: int, filter_node_type: set, from_release: int = None
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
        # it returns itself, which is important

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

            best_ids_best_ranges = self._choose_relevant_synonym_helper(the_id, si_list, to_release)

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
        for node_after in nx.neighbors(self.graph if not reverse else self.reverse_graph, from_id):

            # This forces to follow the same form tree during the recursion
            if self.graph.nodes[node_after]["node_type"] == self.graph.nodes[from_id]["node_type"]:

                for multi_edge_id, an_edge in (
                    (self.graph if not reverse else self.reverse_graph).get_edge_data(from_id, node_after).items()
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

    def get_next_edge_releases(self, from_id: str, reverse: bool):
        """Todo.

        Args:
            from_id: Todo.
            reverse: Todo.

        Returns:
            Todo.
        """
        return list(
            {
                an_edge["old_release"]
                if (not np.isinf(an_edge["new_release"]) and not reverse)
                else an_edge["new_release"]
                for node_after in nx.neighbors(self.graph if not reverse else self.reverse_graph, from_id)
                for mei, an_edge in (self.graph if not reverse else self.reverse_graph)
                .get_edge_data(from_id, node_after)
                .items()
                if (
                    self.graph.nodes[node_after]["node_type"] == self.graph.nodes[from_id]["node_type"]
                    and (
                        node_after != from_id or (np.isinf(an_edge["new_release"]) and not reverse)
                    )  # keep inf self-loop for forward'
                )
            }
        )

    def get_next_edge_releases_deprecated(self, from_id: str, reverse: bool):
        """Todo.

        Args:
            from_id: Todo.
            reverse: Todo.

        Returns:
            Todo.
        """
        return list(
            {
                an_edge["old_release"] if not reverse else an_edge["new_release"]
                for node_after in nx.neighbors(self.graph if not reverse else self.reverse_graph, from_id)
                for mei, an_edge in (self.graph if not reverse else self.reverse_graph)
                .get_edge_data(from_id, node_after)
                .items()
                if self.graph.nodes[node_after]["node_type"] == self.graph.nodes[from_id]["node_type"]
            }
        )

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
        if self.graph.nodes[from_id]["node_type"] == DB.external_search_settings["backbone_node_type"]:
            n = self.get_active_ranges_of_id(from_id)
        else:
            n = self.get_active_ranges_of_id_others(from_id)
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

    def _recursive_path_search(
        self,
        from_id: str,
        from_release: int,
        to_release: int,
        all_paths: set,
        reverse: bool,
        external_settings: dict,
        beamed_up: bool = False,
        external_jump: float = None,
        edge_hist: list = None,
        multiple_ensembl_transition: bool = False,
    ):
        """Todo.

        Args:
            from_id: Todo.
            from_release: Todo.
            to_release: Todo.
            all_paths: Todo.
            reverse: Todo.
            external_settings: Todo.
            beamed_up: Todo.
            external_jump: Todo.
            edge_hist: Todo.
            multiple_ensembl_transition: Todo
        """

        def _external_path_maker(a_from_id, a_ens_rel, a_syn_pth, a_syn_dbp):
            a_edge_hist_alt = list()
            a_from_id_ext_path = copy.copy(a_from_id)
            for a_path_ind, a_next_node in enumerate(a_syn_pth):
                if a_path_ind == 0:
                    continue
                multi_edge_id_syn = 0  # as tested in the self._recursive_synonymous
                a_dict_key = (a_from_id_ext_path, a_next_node, multi_edge_id_syn, (a_syn_dbp[a_path_ind], a_ens_rel))
                a_edge_hist_alt.append(a_dict_key)
                a_from_id_ext_path = copy.copy(a_next_node)
            return a_edge_hist_alt

        _edge_hist = list() if edge_hist is None else edge_hist
        _external_jump = 0 if external_jump is None else external_jump
        next_edges = self.get_next_edges(from_id, from_release, reverse)

        if (
            len(_edge_hist) == 0
            and len(next_edges) == 0
            and self.graph.nodes[from_id]["node_type"] != DB.external_search_settings["backbone_node_type"]
            # the step input is actually external
        ):
            # get syn only for given release
            if not multiple_ensembl_transition:
                s = self.choose_relevant_synonym(
                    from_id,
                    depth_max=external_settings["synonymous_max_depth"],
                    to_release=to_release,
                    filter_node_type={external_settings["backbone_node_type"]},
                    # do not time travel here, go with the same release
                    from_release=from_release,
                )

            # If from release is infered by the program (not provided by the user), then run pathfinder from all
            # possible ones. This will increase the computational demand but yield more robust results.
            if multiple_ensembl_transition or len(s) == 0:
                s = self.choose_relevant_synonym(
                    from_id,
                    depth_max=external_settings["synonymous_max_depth"],
                    to_release=to_release,
                    filter_node_type={external_settings["backbone_node_type"]},
                    # if there is way to find with the same release, go just find it in other releases.
                    from_release=None,
                )

            for s1, s2, s3, s4, s5 in s:
                alt_external_path = _external_path_maker(from_id, s2, s4, s5)

                self._recursive_path_search(
                    # with synonym route, don't go synonym finding in the next iteration
                    s1,
                    s2,
                    to_release,
                    all_paths,
                    s3,
                    external_settings,
                    beamed_up=True,
                    external_jump=_external_jump,  # It does not count as it is starting point
                    edge_hist=_edge_hist + alt_external_path,
                )  # Add parallel path finding searches

        else:
            for _edge_id, (_edge_release, _only_self_loop, _from_id, _node_after, _multi_edge_id) in enumerate(
                next_edges
            ):

                # Synonymous genes of the gene of interest until the next node in the history travel.

                if not beamed_up and _external_jump < external_settings["jump_limit"]:

                    s = self.choose_relevant_synonym(
                        _from_id,
                        depth_max=external_settings["synonymous_max_depth"],
                        to_release=to_release,
                        filter_node_type={external_settings["backbone_node_type"]},
                    )

                    for s1, s2, s3, s4, s5 in s:
                        alt_external_path = _external_path_maker(_from_id, s2, s4, s5)

                        if all([eha not in _edge_hist for eha in alt_external_path]):  # prevent loops
                            self._recursive_path_search(
                                # with synonym route, don't go synonym finding in the next iteration
                                s1,
                                s2,
                                to_release,
                                all_paths,
                                s3,
                                external_settings,
                                beamed_up=True,
                                external_jump=_external_jump + 1.0,
                                edge_hist=_edge_hist + alt_external_path,
                            )  # Add parallel path finding searches

                # History travel

                dict_key = (_from_id, _node_after, _multi_edge_id)

                if dict_key not in _edge_hist:
                    # self loops and, o yoldan daha önce geçmiş mi. extinction event'i var mı
                    _from_id_ver = self.graph.nodes[_from_id]["Version"]

                    if reverse:
                        if _edge_release <= to_release:
                            if _from_id_ver not in DB.alternative_versions:
                                all_paths.add(tuple(_edge_hist) if len(_edge_hist) > 0 else ((None, from_id, None),))
                        elif _only_self_loop:
                            _edge_other = self.reverse_graph.get_edge_data(*dict_key)["old_release"]
                            if _edge_other <= to_release:  # and _edge_other != np.inf
                                all_paths.add(tuple(_edge_hist + [dict_key]))
                            else:
                                _edge_hist.append(dict_key)
                        else:
                            self._recursive_path_search(
                                _node_after,
                                _edge_release,
                                to_release,
                                all_paths,
                                reverse,
                                external_settings,
                                beamed_up=False,
                                external_jump=_external_jump,
                                edge_hist=_edge_hist + [dict_key],
                            )
                    else:  # if not reverse
                        if _edge_release >= to_release:
                            if _from_id_ver not in DB.alternative_versions:
                                all_paths.add(tuple(_edge_hist) if len(_edge_hist) > 0 else ((None, from_id, None),))
                        elif _only_self_loop:  # latest also goes here
                            _edge_other = self.graph.get_edge_data(*dict_key)["new_release"]
                            _is_latest_loop = np.isinf(_edge_other)
                            if _edge_other >= to_release and not _is_latest_loop:
                                all_paths.add(tuple(_edge_hist + [dict_key]))
                            elif _is_latest_loop:
                                all_paths.add(tuple(_edge_hist) if len(_edge_hist) > 0 else ((None, from_id, None),))
                            else:
                                # Do not parallelize (recursive) at that point. keep loop for all.
                                _edge_hist.append(dict_key)
                        else:
                            self._recursive_path_search(
                                _node_after,
                                _edge_release,
                                to_release,
                                all_paths,
                                reverse,
                                external_settings,
                                beamed_up=False,
                                external_jump=_external_jump,
                                edge_hist=_edge_hist + [dict_key],
                            )

    def get_possible_paths(
        self,
        from_id: str,
        from_release: int,
        to_release: int,
        reverse: bool,
        go_external: bool = True,
        increase_depth_until: int = 1,
        increase_jump_until: int = 0,
        from_release_infered: bool = False,
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
            from_release_infered: Todo.

        Returns:
            Todo.
        """
        es: dict = copy.copy(DB.external_search_settings)
        idu = increase_depth_until + es["synonymous_max_depth"]
        iju = increase_jump_until + es["jump_limit"]

        # Todo: check if from_id exist in from_release
        #   if from_id in self.graph.nodes:

        # Try first with no external jump route
        all_paths: set = set()
        self._recursive_path_search(
            from_id,
            from_release,
            to_release,
            all_paths,
            reverse,
            es,
            external_jump=np.inf,
            multiple_ensembl_transition=False,
        )

        if len(all_paths) < 1 and from_release_infered:
            # If none found, make relaxed search in terms of ensembl transition.
            all_paths = set()
            self._recursive_path_search(
                from_id,
                from_release,
                to_release,
                all_paths,
                reverse,
                es,
                external_jump=np.inf,
                multiple_ensembl_transition=True,
            )

        while go_external and len(all_paths) < 1:
            # Activate external jump and increase the depth of search at each step
            all_paths = set()
            self._recursive_path_search(
                from_id,
                from_release,
                to_release,
                all_paths,
                reverse,
                es,
                external_jump=None,
                multiple_ensembl_transition=False,
            )
            if es["synonymous_max_depth"] < idu:
                es["synonymous_max_depth"] = es["synonymous_max_depth"] + 1
            elif es["jump_limit"] < iju:
                es["jump_limit"] = es["jump_limit"] + 1
            else:
                break

        es = copy.copy(DB.external_search_settings)
        while go_external and len(all_paths) < 1 and from_release_infered:
            # The same as above except this time with relaxed transition.
            all_paths = set()
            self._recursive_path_search(
                from_id,
                from_release,
                to_release,
                all_paths,
                reverse,
                es,
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
        ensembl_form_to_include = ["transcript", "translation"]
        ensembl_include = dict()
        filter_set = self.available_external_databases
        for i in ensembl_form_to_include:
            ensembl_include_form: set = set([DB.nts_ensembl[i]] + [DB.nts_assembly[j][i] for j in DB.nts_assembly])
            ensembl_include[i] = ensembl_include_form
            filter_set = filter_set | ensembl_include_form

        return filter_set, ensembl_include

    def calculate_node_scores(self, gene_id) -> list:
        """A metric to choose from multiple targets.

        Args:
            gene_id: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        filter_set, ensembl_include = self._calculate_node_scores_helper
        _temp = [i[-1] for i, _ in self.synonymous_nodes(gene_id, 2, filter_node_type=filter_set)]
        the_tran, the_prot, the_ext = [], [], []
        for i in _temp:
            nt = self.graph.nodes[i][DB.node_type_str]
            if nt in DB.nts_external:
                the_ext.append(i)
            elif nt in ensembl_include["transcript"]:
                the_tran.append(i)
            elif nt in ensembl_include["translation"]:
                the_prot.append(i)
            else:
                raise ValueError

        # Importance order is as following
        return [len(set(the_ext)), len(set(the_prot)), len(set(the_tran))]

    def find_external(self, gene_id, target_db, from_release):
        """Todo.

        Args:
            gene_id: Todo.
            target_db: Todo.
            from_release: Todo.

        Returns:
            Todo.
        """
        a = {
            i[-1]
            for i, j in self.synonymous_nodes(
                the_id=gene_id, depth_max=2, filter_node_type={target_db}, from_release=from_release
            )
        }
        if len(a) > 0 or from_release is None:
            return a
        else:
            return {
                i[-1]
                for i, j in self.synonymous_nodes(
                    the_id=gene_id, depth_max=2, filter_node_type={target_db}, from_release=None
                )
            }

    def find_ensembl_gene(self, external_id):
        """Todo.

        Args:
            external_id: Todo.

        Returns:
            Todo.
        """
        return {i[0][-1] for i in self.synonymous_nodes(external_id, 2, {"ensembl_gene"})}

    def get_external_database_nodes(self, database_name):
        """Todo.

        Args:
            database_name: Todo.

        Returns:
            Todo.
        """
        return {
            i
            for i in self.graph.nodes
            if self.graph.nodes[i]["node_type"] == "external"
            and database_name in self.graph.nodes[i]["release_dict"].keys()
        }

    @cached_property
    def available_external_databases(self):
        """Todo.

        Returns:
            Todo.
        """
        return {
            j
            for i in self.graph.nodes
            if self.graph.nodes[i]["node_type"] == "external"
            for j in self.graph.nodes[i]["release_dict"].keys()
        }

    @cached_property
    def external_database_connection_form(self):
        """Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        aed = self.available_external_databases
        res = dict()

        for e in aed:
            ra = list()
            nodes = self.get_external_database_nodes(e)

            for node in nodes:
                r = [self.graph.nodes[nei]["node_type"] for nei in self.graph.neighbors(node)]
                a = [i.split("_")[1] for i in r if i.startswith("ensembl")]

                if any([i not in self.db_manager.available_form_of_interests for i in a]) and len(a) > 0:
                    raise ValueError(a, e, node)

                ra.extend(a)
            res[e] = Counter(ra).most_common(1)[0][0]
        return res

    def database_bins(self, anchor_database_name, verbose: bool = True):
        """Todo.

        Args:
            anchor_database_name: Todo.
            verbose: Todo.

        Returns:
            Todo.
        """
        self.log.info(f"Database bin dictionary is being created for '{anchor_database_name}'.")
        external_nodes = self.get_external_database_nodes(anchor_database_name)
        bins = dict()
        for ind, en in enumerate(external_nodes):
            if verbose:
                progress_bar(ind, len(external_nodes) - 1)
            a_bin = {ene: self.calculate_node_scores(ene) for ene in self.find_ensembl_gene(en)}
            bins[en] = a_bin

        return bins

    def calculate_score_and_select(
        self,
        all_possible_paths,
        reduction,
        remove_na,
        from_releases,
        to_release,
        score_of_the_queried_item,
        return_path: bool = False,
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

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
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
                        w = (self.graph if not reverse else self.reverse_graph).get_edge_data(*the_edge)["weight"]
                    else:
                        w = score_of_the_queried_item
                    edge_scores.append(w)
                    in_external = False

                else:  # External path is followed.
                    external_step += 1
                    if not in_external:
                        external_jump += 1
                    in_external = True
                    from_release = the_edge[3][1]  # _external_path_maker

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
            to_add = [external_jump, external_step, edge_scores_r, len(the_path) - external_step]
            if return_path:
                to_add.append(the_path)
            scores[final_destination].append(to_add)

        max_score = {
            i: sorted(
                scores[i],  # min external_jump, external_step of max edge_scores
                key=lambda k: (k[0], k[1], -k[2], k[3]),  # k[4] is the path, do not include that.
                reverse=False,
            )[
                0
            ]  # choose the best & shortest
            for i in scores
        }  # choose the best route to a target, and report all targets

        return max_score

    # def convert_dataset(self, id_list, return_scores, *args, **kwargs):

    #     for the_id in id_list:
    #         res = self.convert(the_id, *args, **kwargs)
    #         if return_scores:
    #             pass

    def scorer_for_1_to_n(self, ids: dict, return_path: bool, return_best: bool):
        """Todo.

        Args:
            ids: Todo.
            return_path: Todo.
            return_best: Todo.

        Returns:
            Todo.
        """
        key_lst = list(ids.keys())

        get_scores_list = [0, 1, 2]  # Do not get path, and ensembl_step. Get external these scores.
        get_path_index = 4

        # remove ensembl path scores based on edge weights as they are used already.
        prev_path_scores = {ik: [ids[ik][mana] for mana in get_scores_list] for ik in key_lst}
        if return_path:
            prev_path_path = {ik: ids[ik][get_path_index] for ik in key_lst}  # Get the path

        dot_product = [-1, -1, 1]

        scores = [
            [x * dot_product[ind] for ind, x in enumerate(prev_path_scores[i])]
            # minimum of first two elements, maximum of third element.
            + self.calculate_node_scores(i)  # maximum of all elements
            for i in key_lst
        ]

        if return_best:
            best_score = sorted(scores, reverse=True)[0]  # get all ones with the best score
            updated_ids = {
                k: best_score if not return_path else best_score + [prev_path_path[k]]
                for ind, k in enumerate(key_lst)
                if scores[ind] == best_score
            }
            return updated_ids

        else:
            return {i: scores[ind] for ind, i in enumerate(key_lst)}

    def convert(
        self,
        from_id: str,
        from_release: Optional[int],
        to_release: Optional[int],
        final_database: Optional[str] = None,
        reduction: Callable = np.mean,
        remove_na="omit",
        score_of_the_queried_item: float = np.nan,
        go_external: bool = True,
        prioritize_to_one_filter: bool = False,
        return_path: bool = False,
        deprioritize_lrg_genes: bool = True,
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
            fr = copy.copy(from_release)
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
                from_release_infered=fri,
            )
            possible_paths_reverse = self.get_possible_paths(
                from_id,
                fr[1],
                to_release,
                go_external=go_external,
                reverse=True,
                from_release_infered=fri,
            )
            poss_paths = tuple(list(itertools.chain(possible_paths_forward, possible_paths_reverse)))
            ff = itertools.chain(
                itertools.repeat(fr[0], len(possible_paths_forward)),
                itertools.repeat(fr[1], len(possible_paths_reverse)),
            )
        elif should_reversed == "forward":
            poss_paths = self.get_possible_paths(
                from_id, fr, to_release, go_external=go_external, reverse=False, from_release_infered=fri
            )
            ff = itertools.chain(itertools.repeat(fr, len(poss_paths)))
        elif should_reversed == "reverse":
            poss_paths = self.get_possible_paths(
                from_id, fr, to_release, go_external=go_external, reverse=True, from_release_infered=fri
            )
            ff = itertools.chain(itertools.repeat(fr, len(poss_paths)))
        else:
            raise ValueError

        if len(poss_paths) == 0:
            return None
        else:
            converted = self.calculate_score_and_select(
                poss_paths, reduction, remove_na, ff, to_release, score_of_the_queried_item, return_path
            )

            # They are actually not genes as we understand, they are genomic regions.
            if deprioritize_lrg_genes:
                # Not robust coding here, correct it asap.
                new_converted = {i: converted[i] for i in converted if not i.lower().startswith("lrg")}
                if len(new_converted) > 0:
                    converted = new_converted

            # min(external_jump), min(external_step), max(edge_scores), min(ensembl_step), path: Optional
            converted = (
                self.scorer_for_1_to_n(converted, return_path=return_path, return_best=True)
                if prioritize_to_one_filter and len(converted) > 0
                else converted
            )
            # min(external_jump), min(external_step), min(ensembl_step),
            # max(external_count), max(protein_count), max(transcript_count), path: Optional
            if final_database is None or final_database == "ensembl_gene":
                return converted
            elif final_database in self.available_external_databases:
                converted = {
                    (i, j): converted[i]
                    for i in converted.keys()
                    for j in self.find_external(i, final_database, from_release=to_release)
                }
                return None if len(converted) == 0 else converted
            elif final_database == "base_ensembl_gene":
                converted = {
                    (i, j): converted[i]
                    for i in converted.keys()
                    for j in [a[-1] for a, _ in self.synonymous_nodes(i, 2, {"base_ensembl_gene"})]
                }
                return converted
            else:
                raise ValueError

    def get_tree_with_id(self, the_id):
        """Todo.

        Args:
            the_id: Todo.

        Returns:
            Todo.
        """
        res = list()
        for components in nx.weakly_connected_components(self.graph):
            if the_id in components:
                res.append(self.graph.subgraph(components))
        return res

    # def update_graph_with_the_new_release(self):
    #     # When new release arrive, just add new nodes.
    #     pass

    def node_database_release_pairs(self, the_id):
        """Todo.

        Args:
            the_id: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
            KeyError: Todo.
        """

        def non_inf_range(l1: int, l2: Union[float, int]):

            if not 0 < l1 <= l2:
                raise ValueError

            return range(l1, (l2 if not np.isinf(l2) else max(self.graph.graph["confident_for_release"])) + 1)

        if the_id in self.graph.nodes:
            # external ise database ismi digerleriyse node_type
            nt = self.graph.nodes[the_id][DB.node_type_str]
            if nt == DB.nts_external:
                rd = self.graph.nodes[the_id]["release_dict"]
                return {(r, p) for r in rd for p in rd[r]}
            elif nt == DB.nts_ensembl["gene"]:
                return {(nt, k) for i, j in self.get_active_ranges_of_id(the_id) for k in non_inf_range(i, j)}
            elif nt in DB.nts_non_external_ensembl:  # ensembl_gene is used up above
                _available = {
                    r
                    for ne in self.graph.neighbors(the_id)
                    for _, s in self.graph[the_id][ne].items()
                    for r in s["releases"]
                }
                return {(nt, av) for av in _available}
            else:
                raise ValueError
        else:
            raise KeyError

    @cached_property
    def lower_chars_graph(self):
        """Todo.

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
        """
        result = dict()
        for i in self.graph.nodes:
            j = i.lower()
            if j not in result:
                result[j] = i
            else:
                raise ValueError
        return result

    def unfound_node_solutions(self, the_id: str) -> tuple:
        """Todo.

        Args:
            the_id: Todo.

        Returns:
            Todo.
        """

        def compare_lowers(id_to_find):
            lower_id_find = id_to_find.lower()
            if lower_id_find in self.lower_chars_graph:
                return self.lower_chars_graph[lower_id_find], True
            else:
                return None, False

        if the_id in self.graph.nodes:
            return the_id, False

        lower_id, is_lower_found = compare_lowers(the_id)
        if is_lower_found:
            return lower_id, True

        regex_pattern = re.compile(r"^(.+)(_|-|\.)[0-9]+$")
        regex_found = regex_pattern.match(the_id)
        if regex_found:
            new_id = regex_found.groups()[0]

            if new_id in self.graph.nodes:
                return new_id, True

            lower_id, is_lower_found = compare_lowers(new_id)
            if is_lower_found:
                return lower_id, True

        char_indices = [ind for ind, i in enumerate(the_id) if i in ["-", "_"]]
        possible_alternatives = list()
        if len(char_indices) > 0:
            for comb in range(len(char_indices) + 1):
                for replace_indices in itertools.combinations(char_indices, comb):
                    replace_indices_other = [i for i in char_indices if i not in replace_indices]
                    new_id_l = list(the_id)
                    for ri in replace_indices:
                        new_id_l[ri] = "_"
                    for rio in replace_indices_other:
                        new_id_l[rio] = "-"
                    possible_alternatives.append("".join(new_id_l))

        for pa in possible_alternatives:

            if pa in self.graph.nodes:
                return pa, True

            lower_id, is_lower_found = compare_lowers(pa)
            if is_lower_found:
                return lower_id, True

            regex_pattern = re.compile(r"^(.+)(_|-|\.)[0-9]+$")
            regex_found = regex_pattern.match(pa)
            if regex_found:
                new_id = regex_found.groups()[0]

                if new_id in self.graph.nodes:
                    return new_id, True

                lower_id, is_lower_found = compare_lowers(new_id)
                if is_lower_found:
                    return lower_id, True

        return None, False

    def unfound_correct(self, gene_list: Union[list, set, tuple], verbose: bool = False) -> tuple:
        """Todo.

        Args:
            gene_list: Todo.
            verbose: Todo.

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
        """
        lost_ids = list()
        converted_ids = list()
        result = list()

        for ind, gl in enumerate(gene_list):
            if verbose and (ind % 100 == 0 or ind > len(gene_list) - 5):
                progress_bar(ind, len(gene_list) - 1)
            new_gl, is_converted = self.unfound_node_solutions(gl)

            # As the last resort, try to look at among synonyms.
            if new_gl is None:
                new_gl, is_converted = self.unfound_node_solutions(f"{DB.synonym_id_nodes_prefix}{gl}")
                is_converted = True if new_gl is not None else False

            if new_gl is None:
                lost_ids.append(gl)
            elif new_gl and is_converted:
                converted_ids.append(gl)
                result.append(new_gl)
            elif new_gl:
                result.append(gl)
            else:
                raise ValueError

        if len(converted_ids) > 0:
            self.log.warning(f"Number of converted IDs with small modifications: {len(converted_ids)}")

        if len(lost_ids) > 0:
            self.log.warning(f"Number of IDs not found in the graph: {len(lost_ids)}")

        return result, converted_ids, lost_ids

    def identify_source(self, dataset_ids: list):
        """Todo.

        Args:
            dataset_ids: Todo.

        Returns:
            Todo.
        """
        possible_pairs = list()
        for di in dataset_ids:
            # assembly = self.get_external_assembly()
            possible_pairs.extend(self.memorized_node_database_release_pairs[di])

        return list(Counter(possible_pairs).most_common())

    def get_external_assembly(self):
        """Todo."""
        raise NotImplementedError

    @cached_property
    def memorized_node_database_release_pairs(self):  # Uses so much uncess
        """Todo."""
        return {n: self.node_database_release_pairs(n) for n in self.graph.nodes}

    def get_id_list(self, database: str, release: int):
        """Todo.

        Args:
            database: Todo.
            release: Todo.

        Returns:
            Todo.
        """
        the_key = (database, release)
        final_list = list()
        for n in self.graph.nodes:

            if (
                self.graph.nodes[n]["node_type"] == "ensembl_gene"
                and self.graph.nodes[n]["Version"] in DB.alternative_versions
            ):
                continue

            pairs = self.memorized_node_database_release_pairs[n]
            if the_key in pairs:
                final_list.append(n)
        return final_list

    def convert_from_anndata(self, anndata_path, obs_column):
        """Problem is anndata package will be a package requirement.

        Args:
            anndata_path: Todo.
            obs_column: Todo.

        Raises:
            NotImplementedError: Todo.
        """
        raise NotImplementedError

    def convert_optimized_multiple(self):
        """Accept multiple ID list and return the most optimal set of IDs, minimizing the clashes.

        Raises:
            NotImplementedError: Todo.
        """
        raise NotImplementedError
