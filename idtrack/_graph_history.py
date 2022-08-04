#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import itertools
import logging
import os
import warnings
from collections import Counter
from functools import cached_property
from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from ._database_manager import DatabaseManager
from ._dataset import Dataset
from ._db import DB
from ._verbose import progress_bar


class GraphHistory:
    """Creates ID history graph."""

    # ENSG00000263464
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        """Todo.

        Args:
            db_manager: Todo.
            kwargs: Todo.

        Raises:
            ValueError: Todo.
        """
        # Instance attributes
        self.db_manager = db_manager
        self.log = logging.getLogger("graph")
        self.confident_for_release = self.db_manager.available_releases

        # Protected attributes
        self._no_old_node_id = DB.no_old_node_id
        self._no_new_node_id = DB.no_new_node_id
        self._alternative_versions = {self._no_old_node_id, self._no_new_node_id}

        # Make sure the graph is constructed from the latest release available.
        if self.db_manager.ensembl_release != max(self.db_manager.available_releases):
            raise ValueError

        # Calculate/Load the graph
        self.graph = self.get_graph(**kwargs)
        self.reverse_graph = self.graph.reverse(copy=False)
        self.version_info = self.graph.graph["version_info"]

    def construct_graph(self, narrow: bool, form_list: list = None, narrow_external: bool = True) -> nx.MultiDiGraph:
        """Todo.

        Args:
            narrow: Todo.
            form_list: Todo.
            narrow_external: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """

        def add_edge(n1, n2, ens_rel):
            if not g.has_edge(n1, n2):
                n_edge_att = {"releases": {ens_rel}}
                g.add_edge(n1, n2, **n_edge_att)
            else:
                if len(g.get_edge_data(n1, n2)) != 1:
                    raise ValueError
                g[n1][n2][0]["releases"].add(ens_rel)

        form_list = self.db_manager.available_form_of_interests if not form_list else form_list
        dbman_s = {f: self.db_manager.change_form(f) for f in form_list}
        graph_s = {f: self._remove_nongene_tree(self.construct_graph_form(narrow, dbman_s[f])) for f in form_list}
        # Fun fact: There are Ensembl protein IDs that starts with 'ENST', and sometimes there are clash of IDs.
        # Example clash: "ENST00000515292.1". It does not clash in time, they are defined in different ensembl releases.
        # Remove_non_gene_tree before compose_all.

        g = nx.compose_all(list(graph_s.values()))

        # Establish connection between different forms
        self.log.info("Establishing connection between different forms.")
        for ensembl_release in self.db_manager.available_releases:
            db_manager = dbman_s["transcript"].change_release(ensembl_release)  # Does not matter which form.
            rc = db_manager.get_db("relationcurrent", save_after_calculation=db_manager.store_raw_always)

            for _ind, entry in rc.iterrows():
                for e1_str, e2_str in (("transcript", "gene"), ("translation", "transcript")):

                    e1 = entry[e1_str]
                    e2 = entry[e2_str]

                    if e1 and e2:
                        if e1 not in g.nodes or e2 not in g.nodes:
                            raise ValueError
                        add_edge(e1, e2, ensembl_release)

        # Add versionless versions as well
        if g.graph["version_info"] != "without_version":
            self.log.info("Versionless Ensembl IDs are being connected.")
            for f in ["gene"]:  # transcript and translation does not have base
                for er in self.db_manager.available_releases:
                    db_manager = self.db_manager.change_form(f).change_release(er)  # Does not matter which form.
                    ids_db = db_manager.get_db("ids", save_after_calculation=False)
                    ids = db_manager.id_ver_from_df(ids_db)
                    for n in ids:

                        if n not in g.nodes:
                            raise ValueError

                        m = g.nodes[n]["ID"]
                        if m not in g.nodes:
                            node_attributes = {"node_type": f"base_ensembl_{f}"}
                            g.add_node(m, **node_attributes)

                        add_edge(m, n, er)  # Versionless Base -> EnsID.EnsVer
                self.log.info(f"Edges between versionless ID to version ID has been added for '{f}'.")

        # Establish connection between different databases
        graph_nodes_before_external = set(g.nodes)
        misplaced_external_entry = 0
        for f in form_list:
            db_manager = dbman_s[f].change_release(max(self.db_manager.available_releases))
            st = Dataset(db_manager, narrow_search=narrow_external)
            rc = st.initialize_external_conversion()
            self.log.info(f"Edges between external IDs to Ensembl IDs is being added for '{f}'.")
            for _ind, entry in rc.iterrows():
                e1 = entry["graph_id"]
                e2 = entry["id_db"]
                er = entry["release"]
                edb = entry["name_db"]

                if e1 and e2 and er and edb:

                    if e1 not in graph_nodes_before_external:
                        raise ValueError

                    if e2 in graph_nodes_before_external:
                        misplaced_external_entry += 1
                        # Todo: Have a look and decide whether they are a feature or a bug.
                        #   Decide whether it can be useful for our purposes or not.
                    else:
                        if e2 not in g.nodes:
                            node_attributes_2 = {"release_dict": {edb: {er}}, "node_type": "external"}
                            g.add_node(e2, **node_attributes_2)
                        elif edb not in g.nodes[e2]["release_dict"]:
                            g.nodes[e2]["release_dict"][edb] = {er}
                        elif er not in g.nodes[e2]["release_dict"][edb]:
                            g.nodes[e2]["release_dict"][edb].add(er)

                        add_edge(e2, e1, er)  # External -> gene/transcript/translation

        if misplaced_external_entry > 0:
            self.log.warning(f"Misplaced external entry: {misplaced_external_entry}.")

        return g

    def construct_graph_form(self, narrow: bool, db_manager: DatabaseManager) -> nx.MultiDiGraph:
        """Todo.

        Args:
            narrow: Todo.
            db_manager: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """

        def ms_creator():
            # Download the mapping session table from MYSQL server
            df_ms = db_manager.get_table("mapping_session", save_after_calculation=True)  # not a raw indeed

            # The downloaded column type for old and new release is string, convert them into float.
            # Note that due to some early versions like 18.2, converting to int here is not recommended.
            df_ms["new_release"] = df_ms["new_release"].astype(float)
            df_ms["old_release"] = df_ms["old_release"].astype(float)
            # Make sure there is only one to many
            if np.any(df_ms["new_release"].duplicated(keep=False)):
                raise ValueError("Multiple rows in 'mapping_session' for one 'new_release'.")
            return df_ms

        def edge_attribute_maker(the_edge_old_rel, the_edge_new_rel, weight):
            # Initialize the dictionary for edge attributes.
            the_ea = {
                "weight": float(weight),
                "old_release": int(the_edge_old_rel) if not np.isinf(the_edge_old_rel) else np.inf,
                "new_release": int(the_edge_new_rel) if not np.isinf(the_edge_new_rel) else np.inf,
            }

            # If some additional information is requested for the edge attributes, get them from ms dataframe.
            if not narrow:
                mss = ms[(ms["new_release"] == the_edge_new_rel) & (ms["old_release"] == the_edge_old_rel)]

                # If there is one row corresponding to new_release and old_release.
                if mss.shape[0] == 1:
                    mss = mss.iloc[0]
                    the_ea.update(
                        {
                            "mapping_session_id": int(mss["mapping_session_id"]),
                            "created": mss["created"],
                            "old_db_name": str(mss["old_db_name"]),
                            "old_assembly": str(mss["old_assembly"]),
                            "new_db_name": str(mss["new_db_name"]),
                            "new_assembly": str(mss["new_assembly"]),
                        }
                    )

                # If there is no row corresponding to new_release and old_release.
                elif mss.shape[0] == 0:
                    the_ea.update(
                        {
                            _ea: np.nan
                            for _ea in [
                                "mapping_session_id",
                                "created",
                                "old_db_name",
                                "old_assembly",
                                "new_db_name",
                                "new_assembly",
                            ]
                        }
                    )

                # Otherwise, just raise an error.
                else:
                    raise ValueError(f"Multiple rows in 'mapping_session' for 'new_release' of {the_edge_new_rel}.")
            return the_ea

        def find_edge_score(lookup_cols_tuple):
            try:  # try to find row corresponding to input multiindex, lookup_cols_tuple.
                return df_w.loc[lookup_cols_tuple]["score"]
            except KeyError:  # If no row is found, then the edge score is not known.
                return np.nan

        def find_last_loop_release(lookup_cols_tuple):
            try:  # try to find row corresponding to input multiindex, lookup_cols_tuple.
                return df.loc[lookup_cols_tuple]["new_release"]
            except KeyError:  # Otherwise, just put None
                return None

        def edge_maker_pipe(id_1, ver_1, id_2, ver_2, rel_1, rel_2, the_weight):
            # First create a dictionary for the node attributes
            node_att_1 = DatabaseManager.node_dict_maker(id_1, ver_1)
            node_att_2 = DatabaseManager.node_dict_maker(id_2, ver_2)
            # Create a node name based on the node attributes.
            node_1 = DatabaseManager.node_name_maker(node_att_1)
            node_2 = DatabaseManager.node_name_maker(node_att_2)
            # Create edge attributes using the
            edge_a = edge_attribute_maker(rel_1, rel_2, the_weight)  # old==new
            g.add_edge(node_1, node_2, **edge_a)

        def split_id(id_to_split: str, which_part: str):
            if which_part == "ID":
                return id_to_split.split(DB.id_ver_delimiter)[0]
            elif which_part == "Version":
                return (
                    id_to_split.split(DB.id_ver_delimiter)[1] if id_to_split.count(DB.id_ver_delimiter) == 1 else np.nan
                )  # there are max 1 as checked previously
            else:
                raise ValueError

        # Initialize important variables
        ms = ms_creator()
        version_info = db_manager.check_version_info()
        _available_set = set(db_manager.available_releases)
        # Create the ID history information from ensembl sources
        df = db_manager.get_db(df_indicator="idhistory_narrow" if narrow else "idhistory")
        df["old_stable_id"] = df["old_stable_id"].replace("", np.nan)  # Convert back to np.nan
        df["new_stable_id"] = df["new_stable_id"].replace("", np.nan)
        # No need to check whether DB.id_ver_delimiter is in the version or ID name. As the methods used to fetch the
        # data already controls that.

        # Split the created edge connection data info two: df, df_w
        min_available = min(db_manager.available_releases)
        if version_info == "without_version":
            graph_down_bool = (
                (
                    (df["new_stable_id"] != df["old_stable_id"])  # keep branches or
                    | (
                        (df["new_stable_id"] == df["old_stable_id"])  # keep self loops
                        & (df["score"].astype(float) < 1.0)
                    )  # as it would not change anything
                )
                & ~pd.isna(df["old_stable_id"])  # no void entry
                & ~pd.isna(df["new_stable_id"])  # no retirement entry
                & (df["old_release"].astype(float) >= min_available)
            )  # ignore before available.
            weight_down_bool = np.full(len(df), False)  # Zero the same id event exists
        else:
            common_down_bool = (
                ~pd.isna(df["old_stable_id"])  # no void entry
                & ~pd.isna(df["new_stable_id"])  # no retirement entry
                & (df["new_version"] != 0)  # remove 0 versions: like the ones ASMPATCHG00000000170.0
                & (df["old_version"] != 0)
                # ignore before available.
                & (df["old_release"].astype(float) >= min_available)
            )
            graph_down_bool = (
                (df["new_stable_id"] != df["old_stable_id"])  # keep branches or
                | (
                    (df["new_stable_id"] == df["old_stable_id"])
                    & (df["new_version"] == df["old_version"])  # keep self loops
                    & (df["score"].astype(float) < 1.0)
                )  # as it would not change anything
            ) & common_down_bool
            weight_down_bool = (
                (
                    (df["new_stable_id"] == df["old_stable_id"]) & (df["new_version"] != df["old_version"])
                )  # keep the same id events, no self loops
                & (~pd.isna(df["score"]))  # no undefined score
                & common_down_bool
            )

        # First dataframe will be a one that will be used to fetch edge weight later in the process.
        df_w = df[weight_down_bool].copy()
        lookup_columns = ["old_stable_id", "old_version", "new_stable_id", "new_version"]
        duplicated_dw_w_bool = df_w.duplicated(lookup_columns, keep=False)
        duplicated_dw_w = df_w[duplicated_dw_w_bool]  # ignore those! this is actually a silly mistake by annotators.
        df_w = df_w[~duplicated_dw_w_bool]
        df_w.set_index(lookup_columns, inplace=True, verify_integrity=True)
        df_w.sort_index(inplace=True)
        if len(duplicated_dw_w) > 0:
            self.log.warning(f"Edge weights ignored due to duplicate entries: {len(duplicated_dw_w)}.")
        # Second dataframe will be directly used to add nodes.
        df = df[graph_down_bool]

        g = nx.MultiDiGraph(  # because of multiple self loops of some nodes
            name=f"{db_manager.organism}_{db_manager.ensembl_release}_{db_manager.form}",
            type=db_manager.form,
            ensembl_release=db_manager.ensembl_release,
            organism=db_manager.organism,
            confident_for_release=self.confident_for_release,
            version_info=version_info,
        )

        self.log.info("Edges between across different IDs and self loops are being added.")
        # Add each row in the filtered dataframe as an edge between two nodes.
        for _, e in df.iterrows():
            # Make sure old and new releases are integers (available releases can be only integer), and also one of the
            # defined available release for the given organism.
            _or, _nr = int(e["old_release"]), int(e["new_release"])
            if _or not in _available_set or _nr not in _available_set:
                raise ValueError

            # Create the edge using the pipe function.
            edge_maker_pipe(
                id_1=e["old_stable_id"],
                ver_1=int(e["old_version"]),
                id_2=e["new_stable_id"],
                ver_2=int(e["new_version"]),
                rel_1=_or,
                rel_2=_nr,
                the_weight=e["score"],
            )

        # As we do not need df dataframe anymore, transform it into a new format to be used for other purpose.
        # It will be used in find_last_loop_release function later in the process.
        df.sort_values(by="old_release", inplace=True, ascending=False)
        # Sort the df so that the highest old_release is at the top
        df.drop_duplicates(subset=lookup_columns, keep="first", inplace=True, ignore_index=True)
        # Remove everything except the highest old release.
        df.set_index(lookup_columns, inplace=True, verify_integrity=True)
        df.sort_index(inplace=True)  # Convert the df into multiindex

        self.log.info("Edges between the same IDs are being added.")
        # Initialize some temporary variables
        re_d_prev: dict = dict()
        re_prev_rel: Optional[int] = None
        ll_re: zip = zip(set(), itertools.repeat(True))
        # Get the latest release IDs to create following variables and also to label some nodes as latest at the end.
        latest_rel_ids_df = db_manager.get_db("ids", save_after_calculation=False)
        latest_release_ids = set(db_manager.id_ver_from_df(latest_rel_ids_df))
        # These will be used to get the latest possible ID.Versions' latest definition in the database.
        latest_nodes_last_rel_from_db = {
            DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(i_id, i_ver)): find_last_loop_release(
                (i_id, i_ver, i_id, i_ver)
            )
            for ind, (i_id, i_ver) in latest_rel_ids_df.iterrows()
        }
        # Get the information from the df, created above. Basically because sometimes, the last ID.Version redefined,
        # so self-loops are created. This is just to keep track of these self-loops when finding the latest
        # redefinition of the latest release IDs.
        latest_nodes_last_rel = {i: -1 for i in latest_release_ids}
        latest_nodes_last_rel_locked = {i: False for i in latest_release_ids}
        # Some entities are retired and then refined in a later release, these are to keep track of them.
        void_added: dict = dict()
        reassignment_retirement: int = 0

        # First, follow each release in reversed direction.
        # This loop aims to add edges between 'the-same-ID different-version' couples.
        # Also, finds the first appearance of an ID, and adds self._no_old_node_id to ID.
        # When an ID disappears and then reappears the graph structure should be consistent. In these cases, we
        # add an edge between Retired-to-ID. This loop also aims to find this re-appearance issues.
        for ind_re, rel_re in enumerate(reversed(db_manager.available_releases)):
            # Create a DatabaseManager object with the release of interest.
            rel_db_re = db_manager.change_release(rel_re)

            # Get the IDs and create a dictionary from ID to Version.
            ids_re = rel_db_re.get_db("ids", save_after_calculation=False)
            re_d = {i: j for i, j in ids_re.values}

            # This block should be inspected by the reader together with the last lines of this for loop.
            # Get the IDs which was present in later release but not here, so there is birth of an ID.
            new_void_id_candidates = re_d_prev.keys() - re_d.keys()  # void-3
            # If ID is common in two consecutive releases..
            intersecting_ids = re_d_prev.keys() & re_d.keys()
            # then, get the IDs with different versions to add an edge in between.
            new_version_edge_candidates = {ii for ii in intersecting_ids if re_d_prev[ii] != re_d[ii]}
            # Ff the versions are also the same, keep them as additional elements in the next loops 're_d_prev'
            extend_backwards_candidates = {ii: re_d[ii] for ii in intersecting_ids if re_d_prev[ii] == re_d[ii]}

            # In this block, the ID.Versions at latest release is checked. The aim is to find the first appearance
            # of an ID.Versions at latest release. The dictionary created here will be used later in the process.
            ids_re_set = set(rel_db_re.id_ver_from_df(ids_re))
            for lat_id in latest_nodes_last_rel:  # Run the loop for latest release IDs only, not of this iteration.
                if not latest_nodes_last_rel_locked[lat_id] and lat_id in ids_re_set:
                    from_db = latest_nodes_last_rel_from_db[lat_id]
                    if from_db and from_db > rel_re:
                        latest_nodes_last_rel_locked[lat_id] = True
                        # Self loops can be further defined after the first appearance of the latest_id, so keep the
                        # last possible old_release to use afterwards.
                        if int(from_db) != from_db:
                            raise ValueError
                        latest_nodes_last_rel[lat_id] = int(from_db)
                    else:
                        latest_nodes_last_rel[lat_id] = rel_re
                # If 'lat_id not in ids_re_set' but a release is previously associated, then the ID-Version is edge is
                # here, so ignore this ID afterwards
                elif latest_nodes_last_rel[lat_id] != -1:  # and lat_id not in ids_re_set
                    latest_nodes_last_rel_locked[lat_id] = True

            # When an ID is redefined after with a new version, then add this information as a new edge.
            for nvec in new_version_edge_candidates:
                # Get the edge score using the dataframe created before.
                check_edge_score = (nvec, re_d[nvec], nvec, re_d_prev[nvec])
                # Create the edge using the pipe function.
                edge_maker_pipe(
                    id_1=nvec,
                    ver_1=re_d[nvec],
                    id_2=nvec,
                    ver_2=re_d_prev[nvec],
                    rel_1=rel_re,
                    rel_2=re_prev_rel,
                    the_weight=find_edge_score(check_edge_score),
                )

            # In the last iteration of this loop, dump every ID in current rel, including the ones possibly
            # postponed adding for a long time as a member of 'extend_backwards_candidates'.
            if ind_re == len(db_manager.available_releases) - 1:
                ll_re = zip(re_d.keys(), itertools.repeat(True))  # dump everything

            for nvic, is_ll_re in itertools.chain(zip(new_void_id_candidates, itertools.repeat(False)), ll_re):
                # Note that if an ID branched out (in reverse direction) while losing itself, this loop also
                # treats them as a birth.

                # Since last iteration dumping event has to have a different old and new release (which is the same)
                version_this_case = re_d_prev[nvic] if not is_ll_re else re_d[nvic]
                rel1_this_case = rel_re if not is_ll_re else rel_re
                rel2_this_case = re_prev_rel if not is_ll_re else rel_re

                # Sometimes (for example, for homo sapiens at release 105 for following genes: "LRG_1170", "LRG_170",
                # "LRG_131", "LRG_143", "LRG_166", "LRG_167"), two birth is associated with the same ID. However, this
                # breaks the graphs edge structure. This block following kind of issues:
                # from-[Void-1-2-Retired, Void-1-2-3] fixes-as-[Void-1-2-Retired-1-2-3].
                # This help us to follow the edges any problem in the recursive path finder method of this class.
                if nvic in void_added:  # If there is a birth defined for this ID
                    # Get the data of it and remove from the dictionary as it will be fixed here.
                    rel1_come_alive, rel2_come_alive, version_for_come_alive = void_added.pop(nvic)
                    # Remove the previously added birth edge
                    g.remove_edge(
                        key=0,  # as there is only one edge
                        u=DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(nvic, self._no_old_node_id)),
                        v=DatabaseManager.node_name_maker(
                            DatabaseManager.node_dict_maker(nvic, version_for_come_alive)
                        ),
                    )
                    # Instead, add an edge from Retired to ID with the same edge attributes
                    edge_maker_pipe(
                        id_1=nvic,
                        ver_1=self._no_new_node_id,
                        id_2=nvic,
                        ver_2=version_for_come_alive,
                        rel_1=rel1_come_alive,
                        rel_2=rel2_come_alive,
                        the_weight=np.nan,
                    )
                    reassignment_retirement += 1  # Count to report at the end.
                # As birth is associated with the ID, then add it into the dictionary for above block.
                void_added[nvic] = (rel1_this_case, rel2_this_case, version_this_case)

                # Create the edge using the pipe function.
                edge_maker_pipe(
                    id_1=nvic,
                    ver_1=self._no_old_node_id,
                    id_2=nvic,
                    ver_2=version_this_case,
                    rel_1=rel1_this_case,
                    rel_2=rel2_this_case,
                    the_weight=np.nan,
                )

            # Prepare variables for the next iteration.
            re_d_prev = re_d.copy()
            re_d_prev.update(extend_backwards_candidates)
            re_prev_rel = copy.copy(rel_re)

        if reassignment_retirement > 0:
            self.log.warning(f"Retired ID come alive again: {reassignment_retirement}.")
        # Make sure all latest releases IDs are visited at least once.
        if not np.all([isinstance(latest_nodes_last_rel[i], int) for i in latest_nodes_last_rel]):
            raise ValueError

        # Then, very similar to above reverse loop, but in forward direction.
        # Main aim of this loop is to add Retired information to the nodes.
        self.log.info("Edges showing the retirement of IDs are being added.")
        fo_d_prev: dict = dict()  # Initialize some variables
        fo_prev_rel: Optional[int] = None
        for _, rel_fo in enumerate(db_manager.available_releases):
            # Create a DatabaseManager object with the release of interest.
            rel_db_fo = db_manager.change_release(rel_fo)

            # Get the IDs and create a dictionary from ID to Version.
            ids_fo = rel_db_fo.get_db("ids", save_after_calculation=False)
            fo_d = {i: j for i, j in ids_fo.values}

            # Similar to above for loop in reverse direction. Only difference is the aim is to keep
            # track of retired IDs only. There is no dumping all IDs in the last iteration as above, since not being
            # retired is actually important information, showing it exist in the latest release.
            new_retired_id_candidates = fo_d_prev.keys() - fo_d.keys()
            intersecting_ids = fo_d_prev.keys() & fo_d.keys()
            extend_forwards_candidates = {ii: fo_d[ii] for ii in intersecting_ids if fo_d_prev[ii] == fo_d[ii]}

            for nric in new_retired_id_candidates:
                # Create the edge using the pipe function.
                edge_maker_pipe(
                    id_1=nric,
                    ver_1=fo_d_prev[nric],
                    id_2=nric,
                    ver_2=self._no_new_node_id,
                    rel_1=fo_prev_rel,
                    rel_2=rel_fo,
                    the_weight=np.nan,
                )

            # Prepare variables for the next iteration.
            fo_d_prev = fo_d.copy()
            fo_d_prev.update(extend_forwards_candidates)
            fo_prev_rel = copy.copy(rel_fo)

        # In some cases, the table from `dm_manager.get('idhistory_narrow')` has some edges, that is completely
        # problematic. For example, 'ENSG00000289022' gene is defined in release_105, but it does not seem to
        # exist in the release gene id lists (neither 104, 105, 106 and also online sources).
        # Delete the edge if there are other edges from the previous node.
        self.log.info("Problematic nodes due of Ensembl annotations are being removed.")
        ids_amc = set()
        problematic_nodes = list()
        for amc_rel in db_manager.available_releases:
            # Create a DatabaseManager object with the release of interest.
            amc_dm = db_manager.change_release(amc_rel)

            # Get the IDs and create a dictionary from ID to Version.
            ids_amc_df = amc_dm.get_db("ids", save_after_calculation=False)
            ids_amc.update(set(amc_dm.id_ver_from_df(ids_amc_df)))
        for node in g.nodes:
            if node not in ids_amc and split_id(node, "Version") not in self._alternative_versions:
                problematic_nodes.append(node)
        if len(problematic_nodes) > 0:
            self.log.warning(f"Nodes are deleted due to Ensembl annotation error: {len(problematic_nodes)}.")
            for node in problematic_nodes:
                g.remove_node(node)

        # In reverse loop, we got the latest redefinition (let's say 'x') of latest release IDs. Here, an edge is
        # added to them from x-to-inf. This is very essential for the recursive pathfinder to work robustly.
        self.log.info("Self-loops for latest release entries are being added.")
        for _, (lrc_id, lrc_ver) in latest_rel_ids_df.iterrows():
            # Create the last node name using the conventional class methods.
            last_node_name = DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(lrc_id, lrc_ver))

            # Create the edge using the pipe function.
            edge_maker_pipe(
                id_1=lrc_id,
                ver_1=lrc_ver,
                id_2=lrc_id,
                ver_2=lrc_ver,
                # Get the saved release for the old_release of the edge.
                rel_1=latest_nodes_last_rel[last_node_name],
                rel_2=np.inf,
                the_weight=1.0,
            )

        self.log.info("Node attributes are being added.")
        # Add some node features as node attributes.
        nx.set_node_attributes(g, {n: f"ensembl_{db_manager.form}" for n in g.nodes}, "node_type")
        nx.set_node_attributes(g, {n: n in latest_release_ids for n in g.nodes}, "is_latest")
        nx.set_node_attributes(g, {n: split_id(n, "ID") for n in g.nodes}, "ID")
        nx.set_node_attributes(g, {n: split_id(n, "Version") for n in g.nodes}, "Version")
        self.log.info("Graph is successfully created.")
        return g

    def _remove_nongene_tree(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:

        forms_remove = ["ensembl_transcript", "ensembl_translation"]

        node_to_remove = []
        edge_to_remove = []

        for n in graph.nodes:

            the_node = graph.nodes[n]
            nt = the_node["node_type"]

            if nt in forms_remove:

                if the_node["Version"] in self._alternative_versions:
                    node_to_remove.append(n)  # we only remove Void or retired
                else:
                    for m in graph.neighbors(n):
                        mt = graph.nodes[m]["node_type"]
                        if nt == mt:
                            kmn = [k for k in graph[n][m]]
                            for k in kmn:
                                edge_to_remove.append((n, m, k))

        for c in edge_to_remove:
            graph.remove_edge(*c)
        for c in node_to_remove:
            graph.remove_node(c)

        return graph

    def get_graph(
        self,
        narrow: bool = True,
        create_even_if_exist: bool = False,
        save_after_calculation: bool = True,
        overwrite_even_if_exist: bool = False,
    ) -> nx.MultiDiGraph:
        """Todo.

        Args:
            narrow: Todo.
            create_even_if_exist: Todo.
            save_after_calculation: Todo.
            overwrite_even_if_exist: Todo.

        Returns:
            Todo.
        """
        # Get the file name and narrow parameter.
        file_path = self.create_file_name(narrow)

        # If the file name is not accessible for reading, or explicitly prompt to do so, then create the graph.
        if not os.access(file_path, os.R_OK) or create_even_if_exist:
            self.log.info("The graph is being constructed.")
            g = self.construct_graph(narrow)
        else:  # Otherwise, just read the file that is already in the directory.
            self.log.info("The graph is being read.")
            g = self.read_exported(file_path)

        # If prompt, save the dataframe in requested format.
        if save_after_calculation:
            self.export_disk(g, file_path, overwrite_even_if_exist)

        return g

    def read_exported(self, file_path: str) -> nx.MultiDiGraph:
        """Todo.

        Args:
            file_path: Todo.

        Returns:
            Todo.

        Raises:
            FileNotFoundError: Todo.
        """
        if not os.access(file_path, os.R_OK):
            raise FileNotFoundError

        return nx.read_gpickle(file_path)

    def create_file_name(self, narrow: bool) -> str:
        """Todo.

        Args:
            narrow: Todo.

        Returns:
            Todo.
        """
        narrow_ext = "_narrow" if narrow else ""
        min_ext = f"_min{self.db_manager.ignore_before}" if not np.isinf(self.db_manager.ignore_before) else ""
        max_ext = f"_max{self.db_manager.ignore_after}" if not np.isinf(self.db_manager.ignore_after) else ""
        ext = f"ens{self.db_manager.ensembl_release}{min_ext}{max_ext}{narrow_ext}"
        return os.path.join(self.db_manager.local_repository, f"graph_{self.db_manager.organism}_{ext}.pickle")

    def export_disk(self, g: nx.MultiDiGraph, file_path: str, overwrite: bool):
        """Todo.

        Args:
            g: Todo.
            file_path: Todo.
            overwrite: Todo.
        """
        if not os.access(file_path, os.R_OK) or overwrite:
            self.log.info(f"The graph is being exported as '{file_path}'.")
            nx.write_gpickle(g, file_path)

    def _recursive_synonymous(
        self,
        _the_id,
        synonymous_ones,
        synonymous_ones_db,
        filter_node_type: list,
        the_path: list = None,
        the_path_db: list = None,
        depth_max: int = 0,
    ):
        """Todo.

        Args:
            _the_id: Todo.
            synonymous_ones: Todo.
            synonymous_ones_db: Todo.
            the_path: Todo.
            the_path_db: Todo.
            depth_max: Todo.
            filter_node_type: Todo.

        Raises:
            ValueError: Todo.
        """
        input_node_type = self.graph.nodes[_the_id]["node_type"]
        _the_path = [_the_id] if the_path is None else the_path
        _the_path_db = [input_node_type] if the_path_db is None else the_path_db

        counted_elements = Counter(_the_path_db)
        if depth_max > max(counted_elements.values()):  # The depth is all node_type.

            if len(_the_path) > 0 and self.graph.nodes[_the_path[-1]]["node_type"] in filter_node_type:
                synonymous_ones.append(_the_path)
                synonymous_ones_db.append(_the_path_db)

            for _direction, graph in (("forward", self.graph), ("reverse", self.reverse_graph)):

                _neighbours = list(graph.neighbors(_the_id))
                for _next_neighbour in _neighbours:

                    gnt = graph.nodes[_next_neighbour]["node_type"]

                    if len(_the_path) >= 2:
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
                            self._recursive_synonymous(
                                _next_neighbour,
                                synonymous_ones,
                                synonymous_ones_db,
                                filter_node_type,
                                the_path=_the_path + [_next_neighbour],
                                the_path_db=_the_path_db + [gnt],
                                depth_max=depth_max,
                            )

    def synonymous_nodes(self, the_id: str, depth_max: int, filter_node_type: list):
        """Todo.

        Args:
            the_id: Todo.
            depth_max: Todo.
            filter_node_type: Todo.

        Returns:
            Todo.
        """
        synonymous_ones: list = []
        synonymous_ones_db: list = []
        self._recursive_synonymous(the_id, synonymous_ones, synonymous_ones_db, filter_node_type, depth_max=depth_max)

        remove_set: set = set()
        the_ends_min: dict = dict()

        for p in synonymous_ones:
            e = p[-1]
            lp = len(p)
            am = the_ends_min.get(e, None)
            if am is None or am > lp:
                the_ends_min[e] = lp

        for ind in range(len(synonymous_ones)):
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
            t_outs = [min(self.confident_for_release)]
        elif len(t_ins) == 0:
            assert self.graph.nodes[the_id]["Version"] == DB.no_new_node_id, the_id
            t_ins = [max(self.confident_for_release)]

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
        return GraphHistory.compact_ranges(all_ranges)

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

        return GraphHistory.compact_ranges(result) if compact else result

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

        r = GraphHistory.get_intersecting_ranges(r1, r2, compact=compact)

        return r

    @staticmethod
    def get_from_release_and_reverse_vars(lor, p):
        """Todo.

        Args:
            lor: Todo.
            p: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        result = list()

        for l1, l2 in lor:

            if l1 > l2:
                raise ValueError
            elif p == l1:
                result.append((l1, False))  # target'e yakin uc
            elif p < l1:
                result.append((l1, True))  # target'e yakin uc
            elif l2 <= p:
                result.append((l2, False))  # target'e yakin uc
            elif l1 < p < l2:
                result.append((l1, True))
                result.append((l2, False))
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
                m = GraphHistory.get_from_release_and_reverse_vars(n, to_release)
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
                m = GraphHistory.get_from_release_and_reverse_vars(n, to_release)

                for m1, m2 in m:
                    min_distance_of_range = abs(m1 - to_release)
                    distance_to_target.append(min_distance_of_range)
                    candidate_ranges.append([syn_id, m1, m2])
            # multiple id ve/veya multiple range output verebilir

            # If the queried ID and synonyms has no overlapping ranges:
            # find the synonym_ID with the closest distance to from_id
            if len(distance_to_target) == 0:

                # Find the closest point (1 or 2 exist due to reverse orientation thing)
                # of from_id range to the to_release, if it does not contain it.
                from_id_range = self.get_active_ranges_of_id(from_id)
                if GraphHistory.is_point_in_range(from_id_range, to_release):
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
                        m = GraphHistory.get_from_release_and_reverse_vars(n, ntr)
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

    def choose_relevant_synonym(self, the_id: str, depth_max: int, to_release: int, filter_node_type: list):
        """Todo.

        Args:
            the_id: Todo.
            depth_max: Todo.
            to_release: Todo.
            filter_node_type: Todo.

        Returns:
            Todo.
        """
        # help to choose z for a->x->z3,6,9

        # filter_node_type == 'ensembl_gene'
        syn = self.synonymous_nodes(the_id, depth_max, filter_node_type)  # it returns itself, which is important

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
        n = self.get_active_ranges_of_id(from_id)
        m = GraphHistory.get_from_release_and_reverse_vars(n, to_release)

        forward_from_ids = [i for i, j in m if not j]
        reverse_from_ids = [i for i, j in m if j]

        lffi = len(forward_from_ids)
        lrfi = len(reverse_from_ids)

        if lrfi and lffi:
            return "both", (max(forward_from_ids), min(reverse_from_ids))
        elif lrfi:
            return "reverse", min(reverse_from_ids)
        elif lffi:
            return "forward", max(forward_from_ids)
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
    ):
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
            and self.graph.nodes[from_id]["node_type"]  # the step input is actually external
            != DB.external_search_settings["backbone_node_type"]
        ):
            # get syn only for given release
            s = self.choose_relevant_synonym(
                from_id,
                depth_max=external_settings["synonymous_max_depth"],
                to_release=to_release,
                filter_node_type=external_settings["backbone_node_type"],
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
                        filter_node_type=external_settings["backbone_node_type"],
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
                            if _from_id_ver not in self._alternative_versions:
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
                            if _from_id_ver not in self._alternative_versions:
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

        Returns:
            Todo.
        """
        es: dict = copy.copy(DB.external_search_settings)
        idu = increase_depth_until + es["synonymous_max_depth"]
        iju = increase_jump_until + es["jump_limit"]

        # Todo: check if from_id exist in from_release
        #   if from_id in self.graph.nodes:

        all_paths: set = set()
        self._recursive_path_search(from_id, from_release, to_release, all_paths, reverse, es, external_jump=np.inf)

        while go_external and len(all_paths) < 1:
            all_paths = set()
            self._recursive_path_search(from_id, from_release, to_release, all_paths, reverse, es, external_jump=None)
            if es["synonymous_max_depth"] < idu:
                es["synonymous_max_depth"] = es["synonymous_max_depth"] + 1
            elif es["jump_limit"] < iju:
                es["jump_limit"] = es["jump_limit"] + 1
            else:
                break

        return tuple(all_paths)

    def calculate_node_scores(self, gene_id):
        """Todo.

        Args:
            gene_id: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # a metric to choose from multiple targets
        _temp = [
            i[-1]
            for i, j in self.synonymous_nodes(gene_id, 2, ["external", "ensembl_transcript", "ensembl_translation"])
        ]
        the_tran, the_prot, the_ext = [], [], []
        for i in _temp:
            nt = self.graph.nodes[i]["node_type"]
            if nt == "external":
                the_ext.append(i)
            elif nt == "ensembl_transcript":
                the_tran.append(i)
            elif nt == "ensembl_translation":
                the_prot.append(i)
            else:
                raise ValueError

        return [len(set(the_ext)), len(set(the_tran)), len(set(the_prot))]

    def find_external(self, gene_id, target_db):
        """Todo.

        Args:
            gene_id: Todo.
            target_db: Todo.

        Returns:
            Todo.
        """
        return [
            i[-1]
            for i, j in self.synonymous_nodes(gene_id, 2, ["external"])
            if target_db in self.graph.nodes[i[-1]]["release_dict"].keys()
        ]

    def find_ensembl_gene(self, external_id):
        """Todo.

        Args:
            external_id: Todo.

        Returns:
            Todo.
        """
        return [i[0][-1] for i in self.synonymous_nodes(external_id, 2, ["ensembl_gene"])]

    def get_database_nodes(self, database_name):
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

    def database_bins(self, anchor_database_name, verbose: bool = True):
        """Todo.

        Args:
            anchor_database_name: Todo.
            verbose: Todo.

        Returns:
            Todo.
        """
        self.log.info(f"Database bin dictionary is being created for '{anchor_database_name}'.")
        external_nodes = self.get_database_nodes(anchor_database_name)
        bins = dict()
        for ind, en in enumerate(external_nodes):
            if verbose:
                progress_bar(ind, len(external_nodes) - 1)
            a_bin = {ene: self.calculate_node_scores(ene) for ene in self.find_ensembl_gene(en)}
            bins[en] = a_bin

        return bins

    def calculate_score_and_select(
        self, all_possible_paths, reduction, remove_na, from_releases, to_release, score_of_the_queried_item
    ) -> dict:
        """Todo.

        Args:
            all_possible_paths: Todo.
            reduction: Todo.
            remove_na: Todo.
            from_releases: Todo.
            to_release: Todo.
            score_of_the_queried_item: Todo.

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
                if remove_na == "omit":
                    edge_scores = reduction([s for s in edge_scores if s != 0])
                elif remove_na == "to_1":
                    edge_scores = reduction([s if s != 0 else 1 for s in edge_scores])
                elif remove_na == "to_0":
                    edge_scores = reduction(edge_scores)
                else:
                    raise ValueError(f"Undefined parameter for 'remove_na': {remove_na}")

            final_destination = the_path[-1][1]
            if final_destination not in scores:
                scores[final_destination] = list()
            scores[final_destination].append(
                [
                    external_jump,
                    external_step,
                    edge_scores,
                    len(the_path) - external_step,
                    # the_path
                ]
            )

        max_score = {
            i: sorted(
                scores[i],  # min external_jump, external_step of max edge_scores
                key=lambda k: (k[0], k[1], -k[2], k[3]),
                reverse=False,
            )[
                0
            ]  # choose the best & shortest
            for i in scores
        }  # choose the best route to a target, and report all targets

        return max_score

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
        prioritize_to_one_filter: bool = True,
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

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """

        def prioritize_one_in_1_to_n(ids: dict):
            key_lst = list(ids.keys())
            scores = [
                [1 if pd.isna(ids[i][2]) or ids[i][2] > 0.8 else 0] + self.calculate_node_scores(i) for i in key_lst
            ]
            best_score = sorted(scores, reverse=True)[0]
            return {k: ids[k] + best_score for ind, k in enumerate(key_lst) if scores[ind] == best_score}

        if not callable(reduction):
            raise ValueError
        to_release = to_release if to_release is not None else self.graph.graph["ensembl_release"]

        if from_release is None:
            # self.log.warning(f"Auto direction finding is not recommended: {from_id}.")
            should_reversed, fr = self.should_graph_reversed(from_id, to_release)
        else:
            should_reversed = "forward" if from_release <= to_release else "reverse"
            fr = copy.copy(from_release)

        if should_reversed == "both":
            possible_paths_forward = self.get_possible_paths(
                from_id, fr[0], to_release, go_external=go_external, reverse=False
            )
            possible_paths_reverse = self.get_possible_paths(
                from_id, fr[1], to_release, go_external=go_external, reverse=True
            )
            poss_paths = tuple(list(itertools.chain(possible_paths_forward, possible_paths_reverse)))
            ff = itertools.chain(
                itertools.repeat(fr[0], len(possible_paths_forward)),
                itertools.repeat(fr[1], len(possible_paths_reverse)),
            )
        elif should_reversed == "forward":
            poss_paths = self.get_possible_paths(from_id, fr, to_release, go_external=go_external, reverse=False)
            ff = itertools.chain(itertools.repeat(fr, len(poss_paths)))
        elif should_reversed == "reverse":
            poss_paths = self.get_possible_paths(from_id, fr, to_release, go_external=go_external, reverse=True)
            ff = itertools.chain(itertools.repeat(fr, len(poss_paths)))
        else:
            raise ValueError

        if len(poss_paths) == 0:
            return None
        else:
            converted = self.calculate_score_and_select(
                poss_paths, reduction, remove_na, ff, to_release, score_of_the_queried_item
            )
            converted = (
                prioritize_one_in_1_to_n(converted) if prioritize_to_one_filter and len(converted) > 0 else converted
            )
            if final_database is None or final_database == "ensembl_gene":
                return converted
            elif final_database in self.available_external_databases:
                converted = {
                    (i, j): converted[i] for i in converted.keys() for j in self.find_external(i, final_database)
                }
                return None if len(converted) == 0 else converted
            elif final_database == "base_ensembl_gene":
                converted = {
                    (i, j): converted[i]
                    for i in converted.keys()
                    for j in [a[-1] for a, _ in self.synonymous_nodes(i, 2, ["base_ensembl_gene"])]
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
        """

        def non_inf_range(l1: int, l2: Union[float, int]):

            if not 0 < l1 <= l2:
                raise ValueError

            return range(l1, (l2 if not np.isinf(l2) else max(self.confident_for_release)) + 1)

        if the_id in self.graph.nodes:
            nt = self.graph.nodes[the_id]["node_type"]
            if nt == "external":
                rd = self.graph.nodes[the_id]["release_dict"]
                return [(r, p) for r in rd for p in rd[r]]
            elif nt == "ensembl_gene":
                return [(nt, k) for i, j in self.get_active_ranges_of_id(the_id) for k in non_inf_range(i, j)]
            elif nt in ["ensembl_transcript", "ensembl_translation", "base_ensembl_gene"]:
                _available = {
                    r
                    for ne in self.graph.neighbors(the_id)
                    for _, s in self.graph[the_id][ne].items()
                    for r in s["releases"]
                }
                return [(nt, av) for av in _available]
            else:
                raise ValueError
        else:
            return [(None, None)]

    def identify_source(self, dataset_ids: list):
        """Todo.

        Args:
            dataset_ids: Todo.

        Returns:
            Todo.
        """
        possible_pairs = list()
        for di in dataset_ids:
            possible_pairs.extend(self.node_database_release_pairs(di))

        return list(Counter(possible_pairs).most_common())

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
                and self.graph.nodes[n]["Version"] in self._alternative_versions
            ):
                continue

            pairs = self.node_database_release_pairs(n)
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

    # def test(self):
    # self.get_base_id_range() is equal to release in the adjacent nodes
    # pass

    def test_range_functions_2(self, verbose: bool = True):
        """Todo.

        Args:
            verbose: Todo.

        Returns:
            Todo.
        """
        for ind, ens_rel in enumerate(self.confident_for_release):
            if verbose:
                progress_bar(ind, len(self.confident_for_release) - 1)

            db_from = self.db_manager.change_release(ens_rel)
            ids_from = set(db_from.id_ver_from_df(db_from.get_db("ids", save_after_calculation=False)))

            ids_from_graph = self.get_id_list("ensembl_gene", ens_rel)
            ids_from_graph = set(ids_from_graph)

            if ids_from != ids_from_graph:
                return False
        return True

    def test_range_functions(self):
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
                r12 = GraphHistory.get_intersecting_ranges(r1, r2, compact=False)
                if len(r12) > 1:
                    self.log.warning(f"For Base Ensembl ID {i}: Two associated Ensembl IDs cover the same area")
                    return False
        return True

    def test_how_many_corresponding_path(
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

    def test_history_travel_ensembl_to_ensembl(
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
