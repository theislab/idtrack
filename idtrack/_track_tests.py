#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import itertools
import networkx as nx

from ._verbose import progress_bar
from ._track import Track


class TrackTests(Track):

    def __init__(self, *args, **kwargs):
        super(TrackTests, self).__init__(*args, **kwargs)
        # SubClass initialization

    def range_functions_2(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        for ind, ens_rel in enumerate(self.graph.graph['confident_for_release']):
            if verbose:
                progress_bar(ind, len(self.graph.graph['confident_for_release']) - 1)

            db_from = self.db_manager.change_release(ens_rel)
            ids_from = set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False)))

            ids_from_graph = self.get_id_list("ensembl_gene", ens_rel)
            ids_from_graph = set(ids_from_graph)

            if ids_from != ids_from_graph:
                return False
        return True

    def range_functions(self):
        """Todo.

        Returns:
             Todo.
        """
        base_ids = set()
        for i in self.graph.nodes:
            if self.graph.nodes[i]["node_type"] == "base_ensembl_gene":
                base_ids.add(i)

        for i in base_ids:
            id_family = list(self.graph.neighbors(i))
            id_ranges = [self.get_active_ranges_of_id(j) for j in id_family]

            for r1, r2 in itertools.combinations(id_ranges, 2):
                r12 = Track.get_intersecting_ranges(r1, r2, compact=False)
                if len(r12) > 1:
                    self.log.warning(f"For Base Ensembl ID {i}: Two associated Ensembl IDs cover the same area")
                    return False
        return True

    def id_get_accuracy(self, database: str, release: int):

        return 123

    def how_many_corresponding_path(
        self, from_release: int, to_release: int, go_external: bool, verbose: bool = True
    ):
        """Todo.

        Args:
            from_release: Todo.
            to_release: Todo.
            go_external: Todo.
            verbose: Todo.

        Returns:
             Todo.
        """
        db_from = self.db_manager.change_release(from_release)
        ids_from = sorted(set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False))))
        to_reverse = from_release > to_release

        result = list()
        for ind, from_id in enumerate(ids_from):
            if verbose:
                progress_bar(ind, len(ids_from) - 1)
            lfr = len(
                self.get_possible_paths(from_id, from_release, to_release, reverse=to_reverse, go_external=go_external)
            )
            result.append([from_id, lfr])

        return result

    def history_travel_with_externals(self, from_release, to_release, from_database, to_database,
                                      prioritize_to_one_filter, verbose=True):
        return 123

    def history_travel_ensembl_to_ensembl(
        self, from_release, to_release, go_external, prioritize_to_one_filter, verbose=True
    ):
        """Todo.

        Args:
            from_release: Todo.
            to_release: Todo.
            go_external: Todo.
            prioritize_to_one_filter: Todo.
            verbose: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        db_from = self.db_manager.change_release(from_release)
        db_to = self.db_manager.change_release(to_release)

        ids_from = set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False)))
        ids_to = set(db_to.id_ver_from_df(db_to.get_db("ids", save_after_calculation=False)))

        ids_to_s = set(db_to.get_db("ids", save_after_calculation=False)["gene_stable_id"])

        lost_item = list()
        one_to_one_ids = dict()
        query_not_in_the_graph = list()
        history_voyage_failed = list()
        lost_item_but_the_same_id_exists = list()
        found_ids_not_accurate = dict()
        one_to_multiple_count = dict()

        converted_item_dict = dict()
        converted_item_dict2 = dict()

        for ind, i in enumerate(ids_from):
            if verbose:
                progress_bar(ind, len(ids_from) - 1)

            try:
                converted_item = self.convert(
                    i,
                    from_release,
                    to_release,
                    final_database="ensembl_gene",
                    go_external=go_external,
                    prioritize_to_one_filter=prioritize_to_one_filter,
                )
                converted_item_dict[i] = converted_item

            except nx.exception.NetworkXError:
                query_not_in_the_graph.append(i)
                continue
            except Exception as e:
                history_voyage_failed.append((i, e))
                continue

            if converted_item is None:
                i_id = self.graph.nodes[i]["ID"]
                if i_id in ids_to_s:
                    lost_item_but_the_same_id_exists.append(i)
                lost_item.append(i)
            else:
                if len(converted_item) == 1:
                    one_to_one_ids[i] = list(converted_item.keys())
                elif len(converted_item) > 1:
                    one_to_multiple_count[i] = list(converted_item.keys())
                else:
                    raise ValueError

                # How much of the converted IDs show the same ID
                for c in converted_item.keys():
                    if c in converted_item_dict2:
                        converted_item_dict2[c].append(i)
                    else:
                        converted_item_dict2[c] = [i]

                for c in converted_item.keys():
                    if c not in ids_to:
                        if i not in found_ids_not_accurate:
                            found_ids_not_accurate[i] = list()
                        found_ids_not_accurate[i].append(c)

        # multilerin ne kadarı unique ne kadarı clash içinde
        # ne kadar ID in the destination not mapped to origin

        # Todo: converted_item'ı et ve ne kadar intersect etmiş, ne kadar 1:n ne kadar n:1 n:n var onları keşfet

        results = {
            "Origin IDs": ids_from,
            "Destination IDs": ids_to,
            "Converted IDs": converted_item_dict,
            "Lost Item": lost_item,
            "Lost Item but the same ID Exists": lost_item_but_the_same_id_exists,
            "One-to-One": one_to_one_ids,
            "One-to-Multiple": one_to_multiple_count,
            "Query not in the Graph Error": query_not_in_the_graph,
            "Conversion Failed due to Program Error": history_voyage_failed,
            "Found IDs are not accurate": found_ids_not_accurate,
            "Converted ID Clashes": converted_item_dict2,
        }

        return results

    # def test(self):
    # self.get_base_id_range() is equal to release in the adjacent nodes
    # pass
