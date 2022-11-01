#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import itertools
import logging
import random
import time
from abc import ABC
from math import ceil
from typing import Any, Dict, Union

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from ._db import DB
from ._track import Track


class TrackTests(Track, ABC):
    """Tests for its parent class."""

    def __init__(self, *args, **kwargs):
        """Todo."""
        super().__init__(*args, **kwargs)  # SubClass initialization
        self.log = logging.getLogger("track_tests")

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

        with tqdm(self.graph.graph["confident_for_release"], mininterval=0.25, disable=not verbose) as loop_obj:
            for ens_rel in loop_obj:
                loop_obj.set_postfix_str(f"Item:{ens_rel}", refresh=False)

                db_from = self.db_manager.change_release(ens_rel).change_assembly(assembly)
                ids_from = set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False)))

                ids_from_graph = set(self.graph.get_id_list(DB.nts_assembly[assembly]["gene"], assembly, ens_rel))
                if ids_from != ids_from_graph:
                    switch = False
                    self.log.warning(f"Inconsistent results for ID list functions (ensembl), at release '{ens_rel}'.")

        return switch

    def is_id_functions_consistent_ensembl_2(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        the_ids = list()
        for i in self.graph.nodes:
            if (
                self.graph.nodes[i]["node_type"] == DB.nts_ensembl["gene"]
                and self.graph.nodes[i]["Version"] not in DB.alternative_versions
            ):
                the_ids.append(i)

        assembly = self.graph.graph["genome_assembly"]
        with tqdm(the_ids, mininterval=0.25, disable=not verbose) as loop_obj:
            for ti in loop_obj:
                loop_obj.set_postfix_str(f"Item:{ti}", refresh=False)

                data1_raw = self.graph.combined_edges_genes[ti]
                data1_ = sorted({k for i in data1_raw for j in data1_raw[i] for k in data1_raw[i][j] if j == assembly})
                data1 = self.graph.list_to_ranges(data1_)

                data2 = [
                    [i, j if not np.isinf(j) else max(self.graph.graph["confident_for_release"])]
                    for i, j in self.graph.get_active_ranges_of_id[ti]
                ]

                if data1 != data2:
                    self.log.warning(f"Inconsistency between ID range functions: '{ti, data1, data2, data1_raw}'.")
                    return False

        return True

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
        with tqdm(base_ids, mininterval=0.25, disable=not verbose) as loop_obj:
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
        with tqdm(base_ids, mininterval=0.25, disable=not verbose) as loop_obj:
            for bi in loop_obj:
                loop_obj.set_postfix_str(f"Item:{bi}", refresh=False)

                # get_base_id_range form function
                bi_fun = self.graph.get_active_ranges_of_base_id_alternative(bi)
                # get range directly from graph
                rd = self.graph.combined_edges[bi]
                bi_dir = self.graph.list_to_ranges(sorted({s for p in rd for r in rd[p] for s in rd[p][r]}))
                bi_fun = [
                    [i, j if not np.isinf(j) else max(self.graph.graph["confident_for_release"])] for i, j in bi_fun
                ]

                if bi_dir != bi_fun:
                    switch = False

        return switch

    def is_combined_edges_dicts_overlapping_and_complete(self):
        """Todo.

        Returns:
            Todo.
        """
        the_dicts = [
            self.graph.combined_edges,
            self.graph.combined_edges_genes,
            self.graph.combined_edges_assembly_specific_genes,
        ]
        for d1, d2 in itertools.permutations(iterable=the_dicts, r=2):
            if len(d1.keys() & d2.keys()) != 0:
                return False  # There are some overlapping identifiers.

        covered_nodes = set.union(*map(set, the_dicts))
        uncovered_nodes = self.graph.nodes - covered_nodes
        for un in uncovered_nodes:
            node_data = self.graph.nodes[un]
            if "Version" not in node_data or node_data["Version"] not in DB.alternative_versions:
                return False  # There are identifiers that is not covered with these dicts.

        return True

    def is_edge_with_same_nts_only_at_backbone_nodes(self):
        """Todo.

        Returns:
            Todo.
        """
        for n1 in self.graph.nodes:
            nts1 = self.graph.nodes[n1][DB.node_type_str]

            for n2 in self.graph.neighbors(n1):
                nts2 = self.graph.nodes[n2][DB.node_type_str]

                if nts1 == nts2 and nts1 != DB.external_search_settings["nts_backbone"]:
                    return False

        return True

    def is_id_functions_consistent_external(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        narrow_external = self.graph.graph["narrow_external"]
        misplace_entries = self.graph.graph["misplaced_external_entry"]

        for assembly in self.graph.available_genome_assemblies:

            with tqdm(
                self.graph.graph["confident_for_release"],
                mininterval=0.25,
                desc=f"Assembly {assembly}",
                disable=not verbose,
            ) as loop_obj:

                for release in loop_obj:
                    loop_obj.set_postfix_str(f"Item:{release}", refresh=False)

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
        """Todo.

        Args:
            from_release: Todo.
            from_assembly: Todo.
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
        with tqdm(ids_from, mininterval=0.25, disable=not verbose) as loop_obj:
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
        """Todo.

        Args:
            from_release: Todo.
            from_assembly: Todo.
            from_database: Todo.
            to_release: Todo.
            to_database: Todo.
            go_external: Todo.
            prioritize_to_one_filter: Todo.
            convert_using_release: Todo.
            from_fraction: Todo.
            verbose: Todo.
            verbose_detailed: Todo.
            return_ensembl_alternative: Todo

        Raises:
            ValueError: Todo.

        Returns:
            Todo.
        """
        if from_database in DB.nts_ensembl or to_database in DB.nts_ensembl:
            raise ValueError

        ids_from = sorted(self.graph.get_id_list(from_database, from_assembly, from_release))
        ids_to = set(self.graph.get_id_list(to_database, self.graph.graph["genome_assembly"], to_release))
        ids_to_s = {self.graph.nodes[i]["ID"] for i in ids_to} if to_database in DB.nts_assembly_reverse else set()

        if from_fraction == 1.0:
            pass
        elif 0.0 < from_fraction < 1.0:
            from_faction_count = ceil(len(ids_from) * from_fraction)
            ids_from = sorted(random.sample(ids_from, from_faction_count))
        else:
            raise ValueError

        parameters: Dict[str, Union[bool, str, int, float]] = {
            "from_release": from_release,
            "from_assembly": from_assembly,
            "from_database": from_database,
            "to_release": to_release,
            "to_database": to_database,
            "go_external": go_external,
            "from_fraction": from_fraction,
            "prioritize_to_one_filter": prioritize_to_one_filter,
        }

        metrics: Dict[str, Any] = {
            "parameters": parameters,
            "ids": {"from": ids_from, "to": ids_to},
            "lost_item": [],
            "one_to_one_ids": {},
            "query_not_in_the_graph": [],
            "history_voyage_failed": [],
            "lost_item_but_the_same_id_exists": [],
            "found_ids_not_accurate": {},
            "conversion": {},
            "one_to_multiple_ids": {},
            "one_to_multiple_final_conversion": [],
            "converted_item_dict": {},
            "converted_item_dict_reversed": {},
        }

        t1 = time.time()

        with tqdm(ids_from, mininterval=0.25, disable=not verbose) as loop_obj:
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
                        f"{len(metrics['query_not_in_the_graph']) + len(metrics['history_voyage_failed'])}"
                    )
                    loop_obj.set_postfix_str(suffix, refresh=False)

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
        metrics["clashing_id_type"] = [clash_one_one, clash_multi_multi, clash_multi_one]

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
                    to_assembly, include_ensembl=True, release_lower_limit=None if not strict_forward else the_key1[2]
                )
            if the_key1 is None or the_key2 is None:
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
        strict_forward: bool,
        convert_using_release: bool,
        prioritize_to_one_filter: bool,
        verbose: bool,
        verbose_detailed: bool,
    ):
        """Todo.

        Args:
            from_fraction: Todo.
            strict_forward: Todo.
            convert_using_release: Todo.
            prioritize_to_one_filter: Todo.
            verbose: Todo.
            verbose_detailed: Todo.

        Returns:
            Todo.
        """
        parameters = self.history_travel_testing_random_arguments_generator(strict_forward=strict_forward)
        if verbose:
            print(parameters)
        return self.history_travel_testing(
            **parameters,
            go_external=True,
            prioritize_to_one_filter=prioritize_to_one_filter,
            convert_using_release=convert_using_release,
            from_fraction=from_fraction,
            verbose=verbose,
            verbose_detailed=verbose_detailed,
            return_ensembl_alternative=False,
        )

    def is_external_conversion_robust(self, convert_using_release: bool, verbose: bool = True):
        """Todo.

        Args:
            convert_using_release: Todo.
            verbose: Todo.

        Returns:
            Todo.
        """
        for asym in DB.assembly_mysqlport_priority:
            database, _, ens_rel = self.random_dataset_source_generator(
                assembly=asym, form=DB.backbone_form, include_ensembl=False
            )
            dm = self.db_manager.change_assembly(asym).change_release(ens_rel)

            df = dm.get_db("external_relevant")
            df = df[df["name_db"] == database]
            base_dict: Dict[str, set] = dict()
            for _, item in df.iterrows():
                if item["graph_id"] not in base_dict:
                    base_dict[item["graph_id"]] = set()
                base_dict[item["graph_id"]].add(item["id_db"])

            if verbose:
                print(f"Assembly: {asym}, Database: {database}, Release: {ens_rel}")

            res = self.history_travel_testing(
                from_release=ens_rel,
                from_assembly=asym,
                from_database=DB.nts_assembly[asym][DB.backbone_form],
                to_release=ens_rel,
                to_database=database,
                go_external=True,
                prioritize_to_one_filter=True,
                convert_using_release=convert_using_release,
                verbose=verbose,
                verbose_detailed=False,
            )
            converts = res["conversion"]
            for from_id in converts:
                if set(converts[from_id]) != base_dict[from_id]:
                    self.log.warning(
                        f"Inconsistent external conversion for '{(database, asym, ens_rel)}':\n"
                        f"ID: {from_id},\n"
                        f"Converted: {converts[from_id]},\n"
                        f"Base expectation: {base_dict[from_id]}"
                    )
                    return False
        return True

    def random_dataset_source_generator(
        self, assembly: int, include_ensembl: bool, release_lower_limit: int = None, form: str = None
    ):
        """Todo.

        Args:
            assembly: Todo.
            include_ensembl: Todo.
            release_lower_limit: Todo.
            form: Todo.

        Returns:
            Todo.
        """
        all_possible_sources = copy.deepcopy(list(self.graph.available_external_databases_assembly[assembly]))
        if form is not None:
            all_possible_sources = [
                i for i in all_possible_sources if self.graph.external_database_connection_form[i] == form
            ]

        if include_ensembl:
            if form is None:
                all_possible_sources.extend(list(DB.nts_assembly[assembly].values()))
            else:
                all_possible_sources.append(DB.nts_assembly[assembly][form])

        if form is None or form == DB.backbone_form:
            all_possible_sources.append(DB.nts_base_ensembl[DB.backbone_form])

        selected_database = random.choice(all_possible_sources)
        possible_releases = self.graph.available_releases_given_database_assembly(selected_database, assembly)
        if release_lower_limit is not None:
            possible_releases = {i for i in possible_releases if i >= release_lower_limit}

        if len(possible_releases) > 1:
            selected_release = random.choice(list(possible_releases))
            the_key = (selected_database, assembly, selected_release)
            return the_key
        else:
            return None

    def is_node_consistency_robust(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        with tqdm(self.graph.nodes, mininterval=0.25, disable=not verbose) as loop_obj:

            for i in loop_obj:

                for j in self.graph.neighbors(i):

                    ni = self.graph.nodes[i][DB.node_type_str]
                    nj = self.graph.nodes[j][DB.node_type_str]

                    if ni == nj and ni != DB.nts_ensembl["gene"]:
                        if verbose:
                            self.log.warning(f"Neighbor nodes (not backbone) has similar node type: '{i}', '{j}'")
                        return False

                    elif ni != nj and len(self.graph[i][j]) != 1:
                        if verbose:
                            self.log.warning(f"Multiple edges between '{i, ni}' and '{j, nj}'")
                        return False

        return True
