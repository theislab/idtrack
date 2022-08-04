#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import itertools
import logging
import os
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from ._database_manager import DatabaseManager
from ._dataset import Dataset
from ._db import DB


class Graph:
    """Creates ID history graph."""

    # ENSG00000263464
    def __init__(self, db_manager: DatabaseManager):
        """Todo.

        Args:
            db_manager: Todo.
            kwargs: Todo.

        Raises:
            ValueError: Todo.
        """
        # Instance attributes
        self.log = logging.getLogger("graph")

        # Make sure the graph is constructed from the latest release available.
        if db_manager.ensembl_release != max(db_manager.available_releases):
            raise ValueError
        self.db_manager = db_manager

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

        # The order is important for form_list in compose_all, due to some clashing ensembl IDs.
        form_list = self.db_manager.available_form_of_interests if not form_list else form_list
        dbman_s = {f: self.db_manager.change_form(f) for f in form_list}
        graph_s = {f: self._remove_non_gene_trees(self.construct_graph_form(narrow, dbman_s[f])) for f in form_list}
        # Fun fact: There are Ensembl protein IDs that starts with 'ENST', and sometimes there are clash of IDs.
        # Example clash: "ENST00000515292.1". It does not clash in time, they are defined in different ensembl releases.
        # Remove_non_gene_tree before compose_all.

        for m, n in itertools.combinations(form_list, 2):
            if m in graph_s and n in graph_s:
                gm = set(graph_s[m].nodes)
                gn = set(graph_s[n].nodes)
                intersection = (gm & gn)
                if len(intersection):
                    self.log.warning(f"Intersecting Ensembl nodes in two different forms: '{m}'-'{n}'.")
                    self.log.warning(f"Nodes in '{m}' will be replaced by '{n}': '{', '.join(intersection)}'.")

        g = nx.compose_all([graph_s[f] for f in form_list])

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

        new_form = '-'.join(form_list)
        g.graph['name'] = f"{self.db_manager.organism}_{self.db_manager.ensembl_release}_{new_form}",
        g.graph.graph['type'] = new_form

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
            confident_for_release=self.db_manager.available_releases,
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
        # Also, finds the first appearance of an ID, and adds DB.no_old_node_id to ID.
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
                        u=DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(nvic, DB.no_old_node_id)),
                        v=DatabaseManager.node_name_maker(
                            DatabaseManager.node_dict_maker(nvic, version_for_come_alive)
                        ),
                    )
                    # Instead, add an edge from Retired to ID with the same edge attributes
                    edge_maker_pipe(
                        id_1=nvic,
                        ver_1=DB.no_new_node_id,
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
                    ver_1=DB.no_old_node_id,
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
                    ver_2=DB.no_new_node_id,
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
            if node not in ids_amc and split_id(node, "Version") not in DB.alternative_versions:
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

    def _remove_non_gene_trees(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:

        forms_remove = ["ensembl_transcript", "ensembl_translation"]

        node_to_remove = []
        edge_to_remove = []

        for n in graph.nodes:

            the_node = graph.nodes[n]
            nt = the_node["node_type"]

            if nt in forms_remove:

                if the_node["Version"] in DB.alternative_versions:
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
