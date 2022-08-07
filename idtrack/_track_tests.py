#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import itertools
import multiprocessing
import random
import time
from abc import ABC

import networkx as nx
import numpy as np

from ._track import Track
from ._verbose import progress_bar


class TrackTests(Track, ABC):
    """Tests for its parent class."""

    def __init__(self, *args, **kwargs):
        """Todo."""
        super().__init__(*args, **kwargs)
        # SubClass initialization

    def is_id_functions_consistent_ensembl(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        switch = True
        for ind, ens_rel in enumerate(self.graph.graph["confident_for_release"]):
            if verbose:
                progress_bar(ind, len(self.graph.graph["confident_for_release"]) - 1)

            db_from = self.db_manager.change_release(ens_rel)
            ids_from = set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False)))

            ids_from_graph = self.get_id_list("ensembl_gene", ens_rel)
            ids_from_graph = set(ids_from_graph)

            if ids_from != ids_from_graph:
                switch = False
                self.log.warning(f"Inconsistent results for ID list functions (ensembl), at release '{ens_rel}'.")
        return switch

    def is_range_functions_robust(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        base_ids = set()
        for i in self.graph.nodes:
            if self.graph.nodes[i]["node_type"] == "base_ensembl_gene":
                base_ids.add(i)

        switch = True
        for ind, i in enumerate(base_ids):
            if verbose:
                progress_bar(ind, len(base_ids) - 1)

            id_family = list(self.graph.neighbors(i))
            id_ranges = [self.get_active_ranges_of_id(j) for j in id_family]

            for r1, r2 in itertools.combinations(id_ranges, 2):
                r12 = Track.get_intersecting_ranges(r1, r2, compact=False)
                if len(r12) > 1:
                    self.log.warning(f"For Base Ensembl ID {i}: Two associated Ensembl IDs cover the same area")
                    switch = False
        return switch

    def is_base_is_range_correct(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        base_ids = set()
        for i in self.graph.nodes:
            if self.graph.nodes[i]["node_type"] == "base_ensembl_gene":
                base_ids.add(i)

        switch = True
        for ind, bi in enumerate(base_ids):
            if verbose:
                progress_bar(ind, len(base_ids) - 1)

            # get_base_id_range form function
            bi_fun = self.get_base_id_range(bi)
            # get range directly from graph
            bi_dir = self.list_to_ranges(sorted(l2 for l1, l2 in self.memorized_node_database_release_pairs[bi]))
            bi_fun = [[i, j if not np.isinf(j) else max(self.graph.graph["confident_for_release"])] for i, j in bi_fun]

            if bi_dir != bi_fun:
                switch = False

        return switch

    def is_id_functions_consistent_external(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        switch = True
        narrow_external = self.graph.graph["narrow_external"]
        total_iteration = len(self.graph.graph["confident_for_release"]) * len(self.available_external_databases)
        misplace_entries = self.graph.graph["misplaced_external_entry"]
        current_iteration = 0
        for release in self.graph.graph["confident_for_release"]:
            dm = self.db_manager.change_release(release)
            ex_rel_d = {
                f: dm.change_form(f).get_db("external_relevant" if narrow_external else "external")
                for f in self.db_manager.available_form_of_interests
            }

            for database in self.available_external_databases:
                if verbose:
                    progress_bar(current_iteration, total_iteration - 1)

                form = self.external_database_connection_form[database]
                ex_rel = ex_rel_d[form]

                from_dm = set(ex_rel["id_db"][ex_rel["name_db"] == database])
                from_gr = set(self.get_id_list(database, release))

                if from_gr != from_dm and not all([i in misplace_entries for i in (from_dm - from_gr)]):
                    self.log.warning(
                        f"Inconsistent results for ID list functions (external) "
                        f"for: database '{database}', ensembl release '{release}'."
                    )
                    switch = False
                current_iteration += 1

        return switch

    def dataset_identification_metric(self):
        """Todo."""
        raise NotImplementedError

    def how_many_corresponding_path_ensembl(
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

    def history_travel_testing_random(self, reverse: bool, verbose: bool = True, return_metrics: bool = True):
        """Todo.

        Args:
            reverse: Todo.
            verbose: Todo.
            return_metrics: Todo.

        Returns:
            Todo.
        """
        fr1 = random.choice(self.graph.graph["confident_for_release"])
        if not reverse:
            fr2 = random.choice([i for i in self.graph.graph["confident_for_release"] if i >= fr1])
        else:
            fr2 = random.choice([i for i in self.graph.graph["confident_for_release"] if i < fr1])

        dbs = ["ensembl_gene", "HGNC Symbol", "Uniprot/SWISSPROT", "base_ensembl_gene"]
        db1 = random.choice(dbs)
        db2 = random.choice(dbs)

        if verbose:
            print(f"From={fr1}, To={fr2}, From={db1}, To={db2}")
        res = self.history_travel_testing(
            fr1,
            fr2,
            db1,
            db2,
            go_external=True,
            prioritize_to_one_filter=False,
            verbose=verbose,
            return_metrics=return_metrics,
        )
        return res

    def history_travel_testing_stream_multiprocessing(self, repeat: int, reverse: bool, verbose: bool = True):
        """Todo.

        Args:
            repeat: Todo.
            reverse: Todo.
            verbose: Todo.
        """
        mcc = multiprocessing.cpu_count()
        print(f"Number of CPU: {mcc}")

        def func(_):
            return self.history_travel_testing_random(reverse=reverse, verbose=False)

        with multiprocessing.Pool(processes=mcc) as pool:
            for res in pool.imap_unordered(func, range(mcc * repeat)):
                print(res)

    def history_travel_testing(
        self,
        from_release: int,
        to_release: int,
        from_database: str,
        to_database: str,
        go_external: bool,
        prioritize_to_one_filter: bool,
        verbose: bool = True,
        return_metrics: bool = False,
    ):
        """Todo.

        Args:
            from_release: Todo.
            to_release: Todo.
            from_database: Todo.
            to_database: Todo.
            go_external: Todo.
            prioritize_to_one_filter: Todo.
            verbose: Todo.
            return_metrics: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        ids_from = sorted(set(self.get_id_list(from_database, from_release)))
        ids_to = set(self.get_id_list(to_database, to_release))
        ids_to_s = {self.graph.nodes[i]["ID"] for i in ids_to} if to_database == "ensembl_gene" else set()

        lost_item: list = list()
        one_to_one_ids: dict = dict()
        query_not_in_the_graph: list = list()
        history_voyage_failed: list = list()
        lost_item_but_the_same_id_exists: list = list()
        found_ids_not_accurate: dict = dict()
        one_to_multiple_ids: dict = dict()

        converted_item_dict: dict = dict()
        converted_item_dict_reversed: dict = dict()

        t1 = time.time()
        for ind, i in enumerate(ids_from):
            t2 = time.time()
            if verbose:
                progress_bar(
                    ind,
                    len(ids_from) - 1,
                    suffix=f"{ind+1}/{len(ids_from)}"
                    f" {i}"
                    f" ["
                    f"{len(one_to_one_ids)},"
                    f"{len(one_to_multiple_ids)},"
                    f"{len(lost_item)},"
                    f"{len(found_ids_not_accurate)},"
                    f"{len(query_not_in_the_graph) + len(history_voyage_failed)}"
                    f"] "
                    f"time: {round(t2 - t1, 1)} sec",
                )

            try:
                converted_item = self.convert(
                    i,
                    from_release,
                    to_release,
                    final_database=to_database,
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
                if to_database == from_database == "ensembl_gene" and self.graph.nodes[i]["ID"] in ids_to_s:
                    lost_item_but_the_same_id_exists.append(i)
                lost_item.append(i)
            else:
                if len(converted_item) == 1:
                    one_to_one_ids[i] = list(converted_item.keys())
                elif len(converted_item) > 1:
                    one_to_multiple_ids[i] = list(converted_item.keys())
                else:
                    raise ValueError

                # How much of the converted IDs show the same ID
                for c in converted_item.keys():
                    if c in converted_item_dict_reversed:
                        converted_item_dict_reversed[c].append(i)
                    else:
                        converted_item_dict_reversed[c] = [i]

                for c in converted_item.keys():
                    if (c if to_database == "ensembl_gene" else c[1]) not in ids_to:
                        if i not in found_ids_not_accurate:
                            found_ids_not_accurate[i] = list()
                        found_ids_not_accurate[i].append(c)

        clash_multi_multi: int = 0
        clash_multi_one: int = 0
        clash_one_one: int = 0

        for cidr in converted_item_dict_reversed:
            cidr_val = converted_item_dict_reversed[cidr]
            if len(cidr_val) > 1:
                s1 = any([cv in one_to_multiple_ids for cv in cidr_val])
                s2 = any([cv in one_to_one_ids for cv in cidr_val])
                if s1 and s2:
                    clash_multi_one += 1
                elif s1:
                    clash_multi_multi += 1
                elif s2:
                    clash_one_one += 1
                else:
                    raise ValueError

        # multilerin ne kadarı unique ne kadarı clash içinde
        # ne kadar ID in the destination not mapped to origin

        t2 = time.time()
        func_args = {
            "ARG_from_release": from_release,
            "ARG_to_release": to_release,
            "ARG_from_database": from_database,
            "ARG_to_database": to_database,
            "ARG_go_external": go_external,
            "ARG_prioritize_to_one_filter": prioritize_to_one_filter,
            "ARG_verbose": verbose,
            "ARG_return_metrics": return_metrics,
            "Time": t2 - t1,
        }

        if not return_metrics:
            res = {
                "Origin IDs": ids_from,
                "Destination IDs": ids_to,
                "Converted IDs": converted_item_dict,
                "Lost Item": lost_item,
                "Lost Item but the same ID Exists": lost_item_but_the_same_id_exists,
                "One-to-One": one_to_one_ids,
                "One-to-Multiple": one_to_multiple_ids,
                "Query not in the Graph Error": query_not_in_the_graph,
                "Conversion Failed due to Program Error": history_voyage_failed,
                "Inaccurate ID Conversion": found_ids_not_accurate,
                "Converted ID Clashes": converted_item_dict_reversed,
                "Clashing ID Type": (clash_one_one, clash_multi_multi, clash_multi_one),
            }
        else:
            res = {
                "Origin ID Count": len(ids_from),
                "Destination ID Count": len(ids_from),
                "Lost Item Count": len(lost_item),
                "Lost Item but the same ID Count": len(lost_item_but_the_same_id_exists),
                "One-to-One Count": len(one_to_one_ids),
                "One-to-Multiple Count": len(one_to_multiple_ids),
                "Program Error": len(query_not_in_the_graph) + len(history_voyage_failed),
                "Inaccurate ID Conversion Count": len(found_ids_not_accurate),
                "Clashing ID Type": (clash_one_one, clash_multi_multi, clash_multi_one),
            }
        res.update(func_args)

        return res

    # def test(self):
    # self.get_base_id_range() is equal to release in the adjacent nodes
    # pass
