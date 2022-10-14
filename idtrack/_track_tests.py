#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import itertools
import copy
import logging
import random
import time
from abc import ABC

import networkx as nx
import numpy as np
import pandas as pd
from typing import Optional

from ._track import Track
from ._verbose import progress_bar
from ._db import DB
from ._dataset import Dataset


class TrackTests(Track, ABC):
    """Tests for its parent class."""

    def __init__(self, *args, **kwargs):
        """Todo."""
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger("track_tests")
        # SubClass initialization

    def is_id_functions_consistent_ensembl(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        assembly = self.graph.graph["genome_assembly"]
        # Others should not have consistent as only the main assembly is added fully. 
        # In other assemblies, the Ensembl IDs are added if there is an external connection.
        switch = True
        for ind, ens_rel in enumerate(self.graph.graph["confident_for_release"]):
            progress_bar(ind, len(self.graph.graph["confident_for_release"]) - 1, frequency=0.01, verbose=verbose)

            db_from = self.db_manager.change_release(ens_rel).change_assembly(assembly)
            ids_from = set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False)))

            ids_from_graph = self.graph.get_id_list("ensembl_gene", assembly, ens_rel)
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
            if self.graph.nodes[i]["node_type"] == DB.nts_base_ensembl["gene"]:
                base_ids.add(i)

        switch = True
        for ind, i in enumerate(base_ids):
            progress_bar(ind, len(base_ids) - 1, frequency=0.01, verbose=verbose)

            id_family = list(self.graph.neighbors(i))
            id_ranges = [self.graph.get_active_ranges_of_id[j] for j in id_family]

            for r1, r2 in itertools.combinations(id_ranges, 2):
                r12 = self.graph.get_intersecting_ranges(r1, r2, compact=False)
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
            progress_bar(ind, len(base_ids) - 1, frequency=0.01, verbose=verbose)

            # get_base_id_range form function
            bi_fun = self.graph.get_active_ranges_of_base_id_alternative(bi)
            # get range directly from graph
            rd = self.graph.combined_edges[bi]
            bi_dir = self.graph.list_to_ranges(sorted({s for p in rd for r in rd[p] for s in rd[p][r]}))
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
        
        narrow_external = self.graph.graph["narrow_external"]
        misplace_entries = self.graph.graph["misplaced_external_entry"]
        problematic_list = list()
        for assembly in self.graph.available_genome_assemblies:
            current_iteration = 0
            total_database = len(self.graph.available_external_databases_assembly[assembly])
            total_iteration = len(self.graph.graph["confident_for_release"]) * total_database
            for release in self.graph.graph["confident_for_release"]:
                dm = self.db_manager.change_release(release).change_assembly(assembly)
                ex_rel_d = {
                    f: dm.change_form(f).get_db("external_relevant" if narrow_external else "external")
                    for f in self.graph.available_forms
                }

                for database in self.graph.available_external_databases_assembly[assembly]:
                    progress_bar(current_iteration, total_iteration - 1, frequency=0.01, verbose=verbose)

                    form = self.graph.external_database_connection_form[database]
                    ex_rel = ex_rel_d[form]

                    from_dm_ = set(ex_rel["id_db"][ex_rel["name_db"] == database])
                    from_dm = set()
                    for nd in from_dm_:
                        l1, l2 = self.graph.node_name_alternatives(nd)
                        from_dm.add(l1 if l2 else nd)
                    from_gr = set(self.graph.get_id_list(database, assembly, release))

                    if from_gr != from_dm and not all([i in misplace_entries for i in (from_dm - from_gr)]):
                        
                        problematic_list.append((database, assembly, release))
                    current_iteration += 1
        if len(problematic_list) != 0:
            self.log.warning(
                            f"Inconsistent results for ID list functions (external) "
                            f"for: database, assembl, ensembl release: {problematic_list}"
                        )
        return len(problematic_list) == 0

    def dataset_identification_metric(self):
        """Todo."""
        raise NotImplementedError

    def how_many_corresponding_path_ensembl(
        self, from_release: int, from_assembly: int, to_release: int, go_external: bool, verbose: bool = True
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
        db_from = self.db_manager.change_release(from_release).change_assembly(from_assembly)
        ids_from = sorted(set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False))))
        to_reverse = from_release > to_release

        result = list()
        for ind, from_id in enumerate(ids_from):
            progress_bar(ind, len(ids_from) - 1, frequency=0.01, verbose=verbose)
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
        convert_without_release: bool,
        verbose: bool = True
    ):
        
        ids_from = sorted(self.graph.get_id_list(from_database, from_assembly, from_release))
        ids_to = set(self.graph.get_id_list(to_database, self.graph.graph["genome_assembly"], to_release))
        ids_to_s = {self.graph.nodes[i]["ID"] for i in ids_to} if to_database == "ensembl_gene" else set()

        metrics = {
            "parameters": {
                "from_release": from_release,
                "from_assembly": from_assembly,
                "from_database": from_database,
                "to_release": to_release,
                "to_database": to_database,
                "go_external": go_external,
                "prioritize_to_one_filter": prioritize_to_one_filter,
            },
            "id_list": {
                "ids_from": ids_from,
                "ids_to": ids_to
            },
            "lost_item": list(),
            "one_to_one_ids": dict(),
            "query_not_in_the_graph": list(),
            "history_voyage_failed": list(),
            "lost_item_but_the_same_id_exists": list(),
            "found_ids_not_accurate": dict(),
            "conversion": dict(),
            "one_to_multiple_ids": dict(),
            "one_to_multiple_final_conversion": list(),
            "converted_item_dict": dict(),
            "converted_item_dict_reversed": dict()
        }

        t1 = time.time()
        for ind, the_id in enumerate(ids_from):
            progress_bar(ind, len(ids_from) - 1, frequency=0.01, verbose=verbose)

            try:
                if not convert_without_release:
                    converted_item = self.convert(
                        from_id=the_id,
                        from_release=from_release,
                        to_release=to_release,
                        final_database=to_database,
                        go_external=go_external,
                        prioritize_to_one_filter=prioritize_to_one_filter,
                    )
                else:
                    converted_item = self.convert(
                        from_id=the_id,
                        from_release=None,
                        to_release=None,
                        final_database=to_database,
                        go_external=go_external,
                        prioritize_to_one_filter=prioritize_to_one_filter,
                    )
                metrics["converted_item_dict"][the_id] = converted_item

            except nx.exception.NetworkXError:
                metrics["query_not_in_the_graph"].append(the_id)
                continue
            except Exception as err:
                metrics["history_voyage_failed"].append((the_id, err))
                continue

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

        clash_multi_multi: int = 0
        clash_multi_one: int = 0
        clash_one_one: int = 0

        for cidr in metrics["converted_item_dict_reversed"]:
            cidr_val = metrics["converted_item_dict_reversed"][cidr]
            if len(cidr_val) > 1:
                s1 = any([cv in metrics["one_to_multiple_ids"] for cv in cidr_val])
                s2 = any([cv in metrics["one_to_one_ids"] for cv in cidr_val])
                if s1 and s2:
                    clash_multi_one += 1
                elif s1:
                    clash_multi_multi += 1
                elif s2:
                    clash_one_one += 1
                else:
                    raise ValueError
        metrics["clashing_id_type"] = (clash_one_one, clash_multi_multi, clash_multi_one)
        
        t2 = time.time()
        metrics["time"] = t2 - t1
        return metrics

    def history_travel_testing_random_arguments_generator(self, strict_forward: bool):
        """Todo.

        Args:
            strict_forward: Todo.

        Returns:
            Todo.
        """
        
        from_assembly = random.choice(list(DB.assembly_mysqlport_priority.keys()))
        # as the final release should be always the main assembly
        to_assembly = self.graph.graph["genome_assembly"]
        the_key1, the_key2 = None, None
        while the_key1 is None or the_key2 is None:
            the_key1 = self.random_dataset_source_generator(from_assembly, include_ensembl=True) 
            if the_key1 is not None:
                the_key2 = self.random_dataset_source_generator(
                    to_assembly, include_ensembl=True, release_lower_limit=None if strict_forward else the_key1[2]) 
            self.log.warning("Recalculating arguments.")

        return {
            "from_assembly": from_assembly,
            "from_release": the_key1[2],
            "to_release": the_key2[2],
            "from_database": the_key1[0],
            "to_database": the_key1[0]
        }
        
    def history_travel_testing_random(self, 
                                      strict_forward: bool, 
                                      convert_without_release: bool, 
                                      prioritize_to_one_filter: bool,
                                      verbose: bool = True):
        parameters = self.history_travel_testing_random_arguments_generator(strict_forward=strict_forward)
        return self.history_travel_testing(**parameters, 
                                           go_external=True, 
                                           prioritize_to_one_filter=prioritize_to_one_filter, 
                                           convert_without_release=convert_without_release,
                                           verbose=verbose)

    def is_external_conversion_robust(self, convert_without_release: bool, verbose: bool):
        
        for asym in DB.assembly_mysqlport_priority:
            database, _, ens_rel = self.random_dataset_source_generator(assembly=asym, include_ensembl=False)
            dm = self.db_manager.change_assembly(asym).change_release(ens_rel)
            
            df = dm.get_db("external_relevant")
            df = df[df["name_db"] == database]
            base_dict = dict()
            for _, item in df.iterrows():
                if item["graph_id"] not in base_dict:
                    base_dict[item["graph_id"]] = set()
                base_dict[item["graph_id"]].add(item["id_db"])
            
            res = self.history_travel_testing(from_release=ens_rel, from_assembly=asym, 
                                        from_database=DB.nts_ensembl[DB.backbone_form], 
                                        to_release=ens_rel, to_database=database, prioritize_to_one_filter=True,
                                        convert_without_release=convert_without_release, verbose=verbose)
            converts = res["conversion"]
            for from_id in converts:
                if set(converts[from_id]) == base_dict[from_id]:
                    self.log.warning(f"Inconsistent external conversion for \'{(database, asym, ens_rel)}\':\n"
                                     f"ID: {from_id},\n"
                                     f"converted: {converts[from_id]},\n"
                                     f"base expectation: {base_dict[from_id]}")
                    return False
        return True

    def random_dataset_source_generator(self, assembly: int, include_ensembl: bool, release_lower_limit: int = None):
        
        all_possible_sources = copy.deepcopy(list(self.graph.available_external_databases_assembly[assembly]))
        if include_ensembl:
            all_possible_sources.append(DB.nts_base_ensembl[DB.backbone_form])
            if assembly != self.graph.graph["genome_assembly"]:
                all_possible_sources.append(DB.nts_assembly[assembly][DB.backbone_form])
            else:
                all_possible_sources.append(DB.nts_ensembl[DB.backbone_form])

        selected_database = random.choice(all_possible_sources)
        possible_releases = self.graph.available_releases_given_database_assembly(selected_database, assembly)
        if release_lower_limit is not None:
            possible_releases = {i for i in possible_releases if i >= release_lower_limit}
        
        if len(possible_releases) > 1:
            selected_release = random.choice(possible_releases)
            the_key = (selected_database, assembly, selected_release)
            return the_key
        else:
            return None
        
    def is_node_consistency_robust(self, verbose: bool):
        self.graph.is_node_consistency_robust(verbose=verbose)

    def is_dataset_source_inference_robust(self):
        raise NotImplementedError('method is from functions')  # TODO
