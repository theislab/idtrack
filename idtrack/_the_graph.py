#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import itertools
import re
import numpy as np
from collections import Counter

import networkx as nx
from functools import cached_property
from typing import Union

from ._db import DB


class TheGraph(nx.MultiDiGraph):

    def __init__(self, *args, **kwargs):
        """Todo."""
        # SubClass initialization
        super().__init__(*args, **kwargs)

        # Other variables
        self.regex_pattern = re.compile(r"^(.+)(_|-|\.)[0-9]+$")
        self.log = logging.getLogger("the_graph")
        self.available_forms = None

    def attach_included_forms(self, available_forms):
        self.available_forms = available_forms

    @cached_property
    def rev(self):
        return self.reverse(copy=False)

    def test_node_consistency(self, verbose=False):
        for i in self.nodes:
            for j in self.neighbors(i):
                if self.nodes[i][DB.node_type_str] == self.nodes[j][DB.node_type_str] \
                        and self.nodes[i][DB.node_type_str] != DB.nts_ensembl["gene"]:
                    if verbose:
                        self.log.warning(f"Neighbor nodes (not backbone) has similar node type: \'{i}\', \'{j}\'")
                    return False

                elif len(self[i][j]) != 1:
                    if verbose:
                        self.log.warning(f"Multiple edges between \'{i}\' and \'{j}\'")
                    return False
        return True

    @cached_property
    def combined_edges(self):
        self.log.info(f"Cached properties being calculated: {'combined_edges'}")
        # does not have assembly_specific_genes and ensembl_gene
        result = dict()
        result = TheGraph._combined_edges(self.nodes, self, result)
        return result
    
    @cached_property
    def combined_edges_genes(self):
        self.log.info(f"Cached properties being calculated: {'_combined_edges_genes'}")
        non_captured_genes = [nd for nd in self.nodes if self.nodes[nd][DB.node_type_str] == DB.nts_ensembl["gene"]]
        assert all([n not in self.combined_edges and n not in self.combined_edges_assembly_specific_genes 
                    for n in non_captured_genes]) 
        result = dict()
        result = TheGraph._combined_edges(non_captured_genes, self.rev, result)
        return result
    
    @cached_property
    def combined_edges_assembly_specific_genes(self):
        self.log.info(f"Cached properties being calculated: {'_combined_edges_assembly_specific_genes'}")
        non_captured_genes = [nd for nd in self.nodes 
                              if nd not in self.combined_edges
                                  and self.nodes[nd][DB.node_type_str] != DB.nts_ensembl["gene"]
                                  ]
        if not all([self.nodes[ncg][DB.node_type_str] in DB.nts_assembly_gene for ncg in non_captured_genes]):
            raise ValueError       
        result = dict()
        result = TheGraph._combined_edges(non_captured_genes, self.rev, result)
        return result

    @staticmethod
    def _combined_edges(node_list: Union[nx.classes.reportviews.NodeView, list, set], 
                        the_graph: nx.MultiDiGraph, 
                        result: dict):
        for i in node_list:
            for j in the_graph.neighbors(i):

                if the_graph.nodes[i][DB.node_type_str] != the_graph.nodes[j][DB.node_type_str]:

                    if i not in result:
                        result[i] = dict()
                    edge_info = the_graph[i][j][0][DB.connection_dict]

                    for db_name in edge_info:

                        if db_name not in result[i]:
                            result[i][db_name] = dict()

                        for assembly_name in edge_info[db_name]:

                            if assembly_name not in result[i][db_name]:
                                result[i][db_name][assembly_name] = set()

                            release_set = edge_info[db_name][assembly_name]
                            result[i][db_name][assembly_name].update(release_set)
        return result

    @cached_property
    def lower_chars_graph(self):
        """Todo.

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
        """
        self.log.info(f"Cached properties being calculated: {'lower_chars_graph'}")
        result = dict()
        for i in self.nodes:
            j = i.lower()
            if j not in result:
                result[j] = i
            else:
                raise ValueError
        return result

    def node_name_alternatives(self, the_id: str) -> tuple:
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

        def check_variation(id_str):

            lower_id, is_lower_found = compare_lowers(id_str)
            if is_lower_found:
                return lower_id, True

            regex_found = self.regex_pattern.match(id_str)
            if regex_found:
                new_id = regex_found.groups()[0]

                if new_id in self.nodes:
                    return new_id, True

                lower_id, is_lower_found = compare_lowers(new_id)
                if is_lower_found:
                    return lower_id, True

            return None, False

        if the_id in self.nodes:
            return the_id, False

        mti1, mti2 = check_variation(the_id)
        if mti2:
            return mti1, mti2

        for pa in TheGraph._possible_alternatives(the_id):

            if pa in self.nodes:
                return pa, True

            mpa1, mpa2 = check_variation(pa)
            if mpa2:
                return mpa1, mpa2

        return None, False

    @staticmethod
    def _possible_alternatives(the_id):
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
        return possible_alternatives

    @cached_property
    def get_active_ranges_of_id(self):
        self.log.info(f"Cached properties being calculated: {'get_active_ranges_of_id'}")
        return {n: self._get_active_ranges_of_id(n) for n in self.nodes}

    def _get_active_ranges_of_id(self, input_id):

        def _get_active_ranges_of_id_nonbackbone(the_id):  # HELLO #
            """Todo.

            Args:
                the_id: Todo.

            Returns:
                Todo.
            """
            
            the_node_type = self.nodes[the_id][DB.node_type_str]
            if the_node_type in DB.nts_assembly_gene:
                rd = self.combined_edges_assembly_specific_genes[the_id]
            else:
                rd = self.combined_edges[the_id]
            rels =  sorted({s for p in rd for r in rd[p] for s in rd[p][r]})
            return TheGraph.list_to_ranges(rels)

        def _get_active_ranges_of_id_backbone(the_id):
            """Todo.

            Args:
                the_id: Todo.

            Returns:
                Todo.

            Raises:
                ValueError: Todo.
            """
            t_outs = self.get_next_edge_releases(from_id=the_id, reverse=True)
            t_ins = self.get_next_edge_releases(from_id=the_id, reverse=False)

            if len(t_outs) == 0 and len(t_ins) == 0:
                raise ValueError
            elif len(t_outs) == 0:
                assert self.nodes[the_id]["Version"] == DB.no_old_node_id, the_id
                t_outs = [min(self.graph["confident_for_release"])]
            elif len(t_ins) == 0:
                assert self.nodes[the_id]["Version"] == DB.no_new_node_id, the_id
                t_ins = [max(self.graph["confident_for_release"])]

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
            narrowed = [narrowed[i: i + 2] for i in range(0, len(narrowed), 2)]
            # outputs always increasing, inclusive ranges, for get_intersecting_ranges
            
            return narrowed

        if self.nodes[input_id][DB.node_type_str] == DB.external_search_settings["nts_backbone"]:
            return _get_active_ranges_of_id_backbone(input_id)
        else:
            return _get_active_ranges_of_id_nonbackbone(input_id)

    def get_active_ranges_of_id_ensembl_all_inclusive(self, the_id):
        
        ndt = self.nodes[the_id][DB.node_type_str]
        if ndt == DB.external_search_settings["nts_backbone"]:
            narrowed = self.get_active_ranges_of_id[the_id]
            # sanity check with its connections
            comb_result = self.combined_edges_genes[the_id]
            comb_reduced = dict()
            for i in comb_result:
                for j in comb_result[i]:
                    if j not in comb_reduced:
                        comb_reduced[j] = set()
                    comb_reduced[j].update(comb_result[i][j])
            # Sanity check with externals/other forms etc.
            if not all([TheGraph.is_point_in_range(narrowed, i) for i in comb_reduced[self.graph["genome_assembly"]]]):
                raise ValueError
            
            other_assemblies = [j for i in comb_reduced for j in comb_reduced[i] 
                                if i != self.graph["genome_assembly"]]
            other_assemblies = TheGraph.list_to_ranges(other_assemblies)
            
            return TheGraph.compact_ranges(narrowed + other_assemblies)
        elif ndt in DB.nts_assembly_gene:
            return self.get_active_ranges_of_id[the_id]
        else:
            raise ValueError

    def get_next_edge_releases(self, from_id: str, reverse: bool):
        """Todo.

        Args:
            from_id: Todo.
            reverse: Todo.

        Returns:
            Todo.
        """
        if self.nodes[from_id][DB.node_type_str] != DB.external_search_settings["nts_backbone"]:
            raise ValueError
        
        return list(
            {
                an_edge["old_release"]
                if (not np.isinf(an_edge["new_release"]) and not reverse)
                else an_edge["new_release"]
                for node_after in nx.neighbors(self if not reverse else self.rev, from_id)
                for mei, an_edge in (self if not reverse else self.rev)
                .get_edge_data(from_id, node_after)
                .items()
                if (
                    self.nodes[node_after][DB.node_type_str] == self.nodes[from_id][DB.node_type_str]
                    and (
                        node_after != from_id or (np.isinf(an_edge["new_release"]) and not reverse)
                    )  # keep inf self-loop for forward'
                )
            }
        )

    def get_active_ranges_of_base_id_alternative(self, base_id):
        """Todo.

        Args:
            base_id: Todo.

        Returns:
            Todo.
        """
        associated_ids = self.neighbors(base_id)
        all_ranges = sorted(r for ai in associated_ids for r in self.get_active_ranges_of_id_ensembl_all_inclusive(ai))
        return TheGraph.compact_ranges(all_ranges)

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

    @cached_property
    def node_trios(self):  # Uses so much unnecessary memory
        """Todo."""
        self.log.info(f"Cached properties being calculated: {'node_trios'}")
        return {n: self._node_trios(n) for n in self.nodes}

    def _node_trios(self, the_id):
        """Todo.

        Args:
            the_id: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """

        def non_inf_range(l1: int, l2: Union[float, int]):

            if not 0 < l1 <= l2:
                raise ValueError

            return range(l1, (l2 if not np.isinf(l2) else max(self.graph["confident_for_release"])) + 1)

        # external ise database ismi digerleriyse node_type
        the_node_type = self.nodes[the_id][DB.node_type_str]
        if the_node_type == DB.nts_ensembl["gene"]:
            return {(the_node_type, self.graph["genome_assembly"], k)
                    for i, j in self.get_active_ranges_of_id[the_id] for k in non_inf_range(i, j)}
        elif the_node_type in DB.nts_assembly_gene:
            rd = self.combined_edges_assembly_specific_genes[the_id]
            return {(p, r, s) for p in rd for r in rd[p] for s in rd[p][r]} 
        else:
            rd = self.combined_edges[the_id]
            return {(p, r, s) for p in rd for r in rd[p] for s in rd[p][r]}

    @staticmethod
    def compact_ranges(lor):
        """Todo.

        Args:
            lor: Todo.

        Returns:
            Todo.
        """
        # lot = list of ranges (increasing, inclusive ranges) output of get_active_ranges_of_id_backbone
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

        return TheGraph.compact_ranges(result) if compact else result

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

    def get_two_nodes_coinciding_releases(self, id1, id2, compact: bool = True):
        """Todo.

        Args:
            id1: Todo.
            id2: Todo.
            compact: Todo.

        Returns:
            Todo.
        """
        r1 = self.get_active_ranges_of_id[id1]
        r2 = self.get_active_ranges_of_id[id2]

        r = TheGraph.get_intersecting_ranges(r1, r2, compact=compact)

        return r

    @cached_property
    def available_external_databases(self):
        """Todo.

        Returns:
            Todo.
        """
        self.log.info(f"Cached properties being calculated: {'available_external_databases'}")
        return {
            j
            for i in self.nodes
            if self.nodes[i][DB.node_type_str] == DB.nts_external
            for j in self.combined_edges[i]
        }
        
    @cached_property
    def available_external_databases_assembly(self):
        """Todo.

        Returns:
            Todo.
        """
        self.log.info(f"Cached properties being calculated: {'available_external_databases_assembly'}")
        result = {i: set() for i in self.available_genome_assemblies}
        for i in self.combined_edges:
            d = self.combined_edges[i]
            for i in d:
                for j in d[i]:
                    if i in self.available_external_databases:
                        result[j].add(i)
        return result

    @cached_property
    def external_database_connection_form(self) -> dict:
        """Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        self.log.info(f"Cached properties being calculated: {'external_database_connection_form'}")
        aed = self.available_external_databases
        res = dict()

        for e in aed:
            ra = list()
            nodes = self.get_external_database_nodes(e)

            for node in nodes:
                r = [self.nodes[nei][DB.node_type_str] for nei in self.neighbors(node)]
                a = [i.split("_")[1] for i in r if i.startswith("ensembl")]

                if any([i not in self.available_forms for i in a]) and len(a) > 0:
                    raise ValueError(a, e, node)

                ra.extend(a)
            res[e] = Counter(ra).most_common(1)[0][0]
        return res

    @cached_property
    def available_genome_assemblies(self):
        self.log.info(f"Cached properties being calculated: {'available_genome_assemblies'}")
        output = set()
        for td in (self.combined_edges, self.combined_edges_genes, self.combined_edges_assembly_specific_genes):
            output.update({k for i in td for j in td[i] for k in td[i][j]})
        
        return output

    def get_id_list(self, database: str, assembly: int, release: int) -> list:
        """Todo.

        Args:
            database: Todo.
            assembly: Todo.
            release: Todo.

        Returns:
            Todo.
        """
        the_key = (database, assembly, release)
        final_list = list()
        for n in self.nodes:

            if (
                self.nodes[n][DB.node_type_str] == DB.nts_ensembl["gene"]
                and self.nodes[n]["Version"] in DB.alternative_versions
            ):
                continue

            trios = self.node_trios[n]
            if the_key in trios:
                final_list.append(n)
        return final_list

    def get_external_database_nodes(self, database_name) -> set:
        """Todo.

        Args:
            database_name: Todo.

        Returns:
            Todo.
        """
        return {
            i
            for i in self.nodes
            if self.nodes[i][DB.node_type_str] == DB.nts_external
            and database_name in self.combined_edges[i]
        }
