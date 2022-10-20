#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import itertools
import logging
import os
import string
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from ._database_manager import DatabaseManager
from ._db import DB
from ._the_graph import TheGraph


class GraphMaker:
    """Creates ID history graph.

    It includes Ensembl gene ID history. Ensembl ID history is obtained from Ensembl
    resources, which shows the connection between different Ensembl base IDs or different versions of the same Ensembl
    base ID. Ensembl transcripts (with base IDs and versions) are connected to gene, and Ensembl proteins are
    connected to transcripts. Additionally, a selected set of external databases are connected to the related Ensembl
    IDs: for example UniProt IDs are associated with proteins, while RefSeq transcript IDs are associated with
    transcripts. The ``Graph`` class also saves the resulting graph into the defined temporary directory for later
    calculations.
    """

    # Note: Example chaotic ID history: ENSG00000263464
    def __init__(self, db_manager: DatabaseManager):
        """Class initialization.

        Args:
            db_manager: Needed to download all necessary tables and data frames.
                It contains the temporary directory to save the resultant graph.

        Raises:
            ValueError: ``Graph`` has to be created with the latest release possible defined in ``db_manager``. If not,
                the exception is raised.
        """
        # Instance attributes
        self.log = logging.getLogger("graph_maker")

        # Make sure the graph is constructed from the latest release available.
        if db_manager.ensembl_release != max(db_manager.available_releases):
            raise ValueError("'Graph' has to be created with the latest release possible defined in 'db_manager'.")
        self.db_manager = db_manager

    def initialize_downloads(self):
        """Initialize the external database downloads.

        Raises:
            NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError

    def update_graph_with_the_new_release(self):
        """When new release arrive, just add new nodes.

        Raises:
            NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError

    # flake8: noqa: C901
    def construct_graph(
        self,
        narrow: bool = False,
        form_list: list = None,
        narrow_external: bool = True,
    ) -> TheGraph:
        """Main method to construct the graph.

        It creates the graph with Ensembl gene, transcript and protein information. It also adds
        ``DB.nts_base_ensembl[f]`` nodes into the graph, which has only base Ensembl gene ID. External database entries
        described in ``DatabaseManager`` will be part of the graph. Normally, user is not expected to use this method,
        as the method is utilized in ``get_graph`` method.

        Args:
            narrow: Determine whether a some more information should be added between Ensembl gene IDs. For example,
                which genome assembly is used, or when was the connection is established. For usual uses, no need to
                set it ``True``.
            form_list: Determine which forms (transcript, translation, gene) should be included. If ``None``, then
                include all possible forms defined in ``DatabaseManager``.
                It has to be list composed of following strings: 'gene', 'transcript', 'translation'.
            narrow_external: If set ``False``, all possible external databases defined in Ensembl *MySQL* server will be
                included into the graph. The graph will be immensely larger, and the ID history travel calculation will
                be very slow. Additionally, the success of ID conversion under such a setting it has not been
                tested yet.

        Returns:
            Resultant multi edge directed graph.

        Raises:
            ValueError: Unexpected error.
        """

        def add_edge(n1: str, n2: str, db12: str, a12: int, er12: int):
            """A simple function to create edges between provided nodes. Edits the graph that is under the construction.

            Args:
                n1: The first node of the edge. The edge is taken from this node.
                n2: The second node of the edge. The edge is taken to this node.
                db12: Database for such a connection
                a12: Ensembl release for such a connection.
                er12: Assembly for such a connection.

            Raises:
                ValueError: If there are more than one edge between the source and the target node.
            """
            if n1 not in g.nodes or n2 not in g.nodes:
                raise ValueError(n1, n2, db12, a12, er12)

            if not g.has_edge(n1, n2):
                n_edge_att = {DB.connection_dict: {db12: {a12: {er12}}}}
                g.add_edge(n1, n2, **n_edge_att)
            else:
                if len(g.get_edge_data(n1, n2)) != 1:
                    raise ValueError  # test it to use '0' below
                elif db12 not in g[n1][n2][0][DB.connection_dict]:
                    g[n1][n2][0][DB.connection_dict][db12] = {a12: {er12}}
                elif a12 not in g[n1][n2][0][DB.connection_dict][db12]:
                    g[n1][n2][0][DB.connection_dict][db12][a12] = {er12}
                elif er12 not in g[n1][n2][0][DB.connection_dict][db12][a12]:
                    g[n1][n2][0][DB.connection_dict][db12][a12].add(er12)
                else:
                    raise ValueError(n1, n2, db12, a12, er12, g[n1][n2][0][DB.connection_dict])

        # The order is important for form_list in compose_all, due to some clashing ensembl IDs.
        form_list = self.db_manager.available_form_of_interests if not form_list else form_list
        dbman_s = {f: self.db_manager.change_form(f) for f in form_list}
        graph_s = {
            f: GraphMaker.remove_non_gene_trees(  # Remove_non_gene_tree before compose_all.
                # The time travel will be between gene IDs, so no need to have such edges.
                self.construct_graph_form(narrow, dbman_s[f])
            )
            for f in form_list
        }

        # Fun fact: There are exceptional Ensembl protein IDs that starts with 'ENST', and sometimes there are
        # clash of IDs (the same ID is defined in Ensembl proteins and Ensembl transcripts). For example,
        # "ENST00000515292.1". It does not clash in time, that is, they are defined in different ensembl releases.
        # Report all possible conflicts.
        for m, n in itertools.combinations(form_list, 2):
            if m in graph_s and n in graph_s:
                gm = set(graph_s[m].nodes)
                gn = set(graph_s[n].nodes)
                intersection = gm & gn
                if len(intersection):
                    self.log.warning(
                        f"Intersecting Ensembl nodes: Nodes in '{m}' will "
                        f"be replaced by '{n}': '{', '.join(intersection)}'."
                    )

        # Compose all graphs into one. If there is a
        g = nx.compose_all([graph_s[f] for f in form_list])
        # To make the form_list cached in the object.
        g.attach_included_forms(form_list)

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

                        # Edges are from transcript to gene, from translation to transcript
                        sly = self.db_manager.genome_assembly
                        add_edge(e1, e2, DB.nts_ensembl[e1_str], sly, ensembl_release)

        # Establish connection between different databases
        graph_nodes_before_external = set(g.nodes)
        graph_nodes_added_assembly: dict = {
            i: set() for i in DB.assembly_mysqlport_priority.keys() if i != self.db_manager.genome_assembly
        }
        misplaced_external_entry = list()
        establish_form_connection = list()
        min_ens_release = dict()
        added_assemblies = set()
        for f in form_list:

            self.log.info(f"Edges between external IDs to Ensembl IDs is being added for '{f}'.")
            nodes_from_previous_release = 0
            for ens_rel in sorted(self.db_manager.available_releases):
                # the order is important in adding new nodes into the core graph.
                # it is important to capture correct ens_release in min_ens_release dictionary

                db_manager = dbman_s[f].change_release(ens_rel)
                rc = db_manager.create_external_all(return_mode="all")

                for _ind, entry in rc.iterrows():
                    # Note that the `rc` dataframe have higher priority assembly entries at the top.

                    e1, e2 = entry["graph_id"], entry["id_db"]
                    er, edb = entry["release"], entry["name_db"]
                    sly = int(entry["assembly"])

                    if sly not in DB.assembly_mysqlport_priority:
                        raise ValueError

                    if e1 and e2 and er and edb and sly:

                        if e1 not in graph_nodes_before_external:
                            # Here, Only add the node once and retire. For Homo sapiens these nodes are added at
                            # previous assemblies, but retired before the last assembly and the Ensembl MySQL server,
                            # minimum release (For 37, it is 79). They are nevertheless important to add, as maybe an
                            # external ID bridge them to the active ones.
                            # For example, 'ENSG00000148828.5', 'ENSG00000167765.3'

                            if sly == self.db_manager.genome_assembly:
                                raise ValueError(
                                    "The main assembly created for the graph should contain all the "
                                    "nodes in the external table."
                                )

                            elif any(
                                [e1 in graph_nodes_added_assembly[i] for i in graph_nodes_added_assembly if i != sly]
                            ):
                                # Hypothetical statement for now, as there are only two assembly in Ensembl.
                                # 'create_external_all' method should have resolve the issue.
                                raise ValueError("Node should have been already added by a higher priority assembly.")

                            elif e1 not in graph_nodes_added_assembly[sly]:

                                if sly not in min_ens_release:
                                    min_ens_release[sly] = ens_rel
                                elif ens_rel != min_ens_release[sly]:
                                    # The duplicate entries were already removed by `create_external_all` method.
                                    # IDs retired before `min(db_manager.available_releases)` will be missing only.
                                    # In the first possible release, all should have been already added.

                                    # Here, the if-elif statement is written in case of third assembly in the future.
                                    raise ValueError(
                                        "The new nodes here should have been added in "
                                        "the first Ensembl release possible."
                                    )
                                graph_nodes_added_assembly[sly].add(e1)

                                node_attributes_3 = {
                                    DB.node_type_str: DB.nts_assembly[sly][f],
                                    "ID": GraphMaker.split_id(e1, "ID"),
                                    "Version": GraphMaker.split_id(e1, "Version"),
                                }
                                g.add_node(e1, **node_attributes_3)

                                nodes_from_previous_release += 1
                                establish_form_connection.append([e1, f, sly])
                                added_assemblies.add(sly)
                            else:
                                # Do nothing. The hope is a second external database is bridging to a known Ensembl ID.
                                pass

                        if e2 in graph_nodes_before_external:
                            misplaced_external_entry.append(e2)
                            # Some external database entries contains an Ensembl ID as an external ID. If such an item
                            # is a part of the graph before externals, store them in the graph attributes at the end.
                        else:
                            # Create a node with external ID, store relevant database and ensembl release information.
                            if e2 not in g.nodes:
                                node_attributes_2 = {DB.node_type_str: DB.nts_external}
                                g.add_node(e2, **node_attributes_2)

                            if e1 not in g.nodes:
                                raise ValueError(f"Node '{e1}' should have been added.")
                            # Edges are from external ID to Ensembl ID.
                            add_edge(e2, e1, edb, sly, er)
                            added_assemblies.add(sly)

            if nodes_from_previous_release > 0:
                self.log.warning(f"New nodes added as assembly nodes: {nodes_from_previous_release}")

        if len(misplaced_external_entry) > 0:
            self.log.warning(f"Misplaced external entry: {len(misplaced_external_entry)}.")

        if len(establish_form_connection) > 0:

            added_edge = 0
            self.log.info("Different forms of assembly-Ensembl-nodes are being connecting.")
            new_nodes = pd.DataFrame(establish_form_connection, columns=["node", "form", "assembly"])
            new_nodes.drop_duplicates(keep="first", inplace=True, ignore_index=True)
            avail_assemblies = np.unique(new_nodes["assembly"])

            for aa in avail_assemblies:
                nn_aa = new_nodes[new_nodes["assembly"] == aa]
                dm_aa = self.db_manager.change_assembly(aa)
                for the_er in sorted(dm_aa.available_releases, reverse=True):
                    dm_aa_er = dm_aa.change_release(the_er)
                    df_aa = dm_aa_er.get_db("relationcurrent")

                    # Attach nodes that has found in the graph and there are form relationships.
                    bool_filter = pd.DataFrame()
                    for f in form_list:
                        fids = np.unique(nn_aa["node"][nn_aa["form"] == f])
                        bool_filter = pd.concat([bool_filter, df_aa[f].isin(fids)], axis=1)
                    bool_filter = np.sum(bool_filter, axis=1) > 0
                    df_aa = df_aa[bool_filter]

                    for _, item in df_aa.iterrows():

                        for f in form_list:
                            anid = item[f]

                            if anid and anid not in g.nodes:
                                node_attributes_4 = {
                                    DB.node_type_str: DB.nts_assembly[aa][f],
                                    "ID": GraphMaker.split_id(anid, "ID"),
                                    "Version": GraphMaker.split_id(anid, "Version"),
                                }
                                g.add_node(anid, **node_attributes_4)

                        for e1_str, e2_str in (("transcript", "gene"), ("translation", "transcript")):

                            new1, new2 = item[e1_str], item[e2_str]
                            if new1 and new2:
                                if not g.has_edge(new1, new2):
                                    added_edge += 1
                                add_edge(new1, new2, DB.nts_assembly[aa][e1_str], aa, the_er)

            if added_edge > 0:
                self.log.warning(f"New edges are added between assembly Ensembl nodes: {added_edge}")

        # Add versionless versions as well
        if g.graph["version_info"] != "without_version":
            self.log.info("Versionless Ensembl IDs are being connected.")

            added_assemblies.add(self.db_manager.genome_assembly)
            avail_assemblies = sorted(added_assemblies)

            for aa in avail_assemblies:
                for f in ["gene"]:
                    # transcript and translation does not have base.
                    # It causes the tracking algorithm unnecessarily process too many possibilities.
                    for er in self.db_manager.change_assembly(aa).available_releases:

                        db_manager = self.db_manager.change_form(f).change_assembly(aa).change_release(er)

                        ids_db = db_manager.get_db("ids")
                        ids = db_manager.id_ver_from_df(ids_db)

                        for n in ids:

                            if n not in g.nodes and aa == self.db_manager.genome_assembly:
                                raise ValueError(aa, er, n)
                            elif n not in g.nodes:
                                pass
                            else:
                                m = g.nodes[n]["ID"]
                                if m not in g.nodes:
                                    node_attributes = {DB.node_type_str: DB.nts_base_ensembl[f]}
                                    g.add_node(m, **node_attributes)

                                # Edges are from versionless base ID to ID (with version).
                                add_edge(m, n, DB.nts_base_ensembl[f], aa, er)

                self.log.info(f"Edges between versionless ID to version ID has been added for '{f}', assembly {aa}.")
        else:
            self.log.info("The graph will be constructed with 'versionless' IDs. It has not been extensively tested.")

        new_form = "-".join(form_list)
        g.graph["name"] = (f"{self.db_manager.organism}_{self.db_manager.ensembl_release}_{new_form}",)
        g.graph["type"] = new_form  # Need to update this, as 'construct_graph_form' puts the form here previously.
        g.graph["narrow_external"] = narrow_external
        g.graph["misplaced_external_entry"] = set(misplaced_external_entry)

        # Merge the nodes that are the same when they are convert into lowercase/uppercase.
        g = self._merge_nodes_with_the_same_in_lower_case(g)

        for e1, e2, e3 in g.edges:
            edge_data = g.get_edge_data(e1, e2, e3)
            if DB.connection_dict in edge_data:
                thed = edge_data[DB.connection_dict]
                available_releases = {k for i in thed for j in thed[i] for k in thed[i][j]}
                g[e1][e2][e3]["available_releases"] = available_releases

        # Run cached_property functions to save them into the disk.
        # _ = g.combined_edges
        # _ = g.combined_edges_assembly_specific_genes
        # _ = g.combined_edges_genes
        # _ = g.lower_chars_graph
        # _ = g.get_active_ranges_of_id
        # _ = g.available_external_databases
        # _ = g.external_database_connection_form
        # _ = g.available_genome_assemblies
        # _ = g.available_external_databases_assembly
        # _ = g.node_trios

        return g

    def _merge_nodes_with_the_same_in_lower_case(self, g: TheGraph):

        self.log.info("Synonymous external nodes are being merged into one.")
        before_node_count = len(g.nodes)
        # Get the problematic nodes
        merge_dict = dict()
        for intm in g.nodes:
            jntm = intm.lower()
            if jntm not in merge_dict:
                merge_dict[jntm] = [intm]
            else:
                merge_dict[jntm].append(intm)
        merge_dict = {i: j for i, j in merge_dict.items() if len(j) > 1}
        # Merge them separately

        reverse_g = g.reverse(copy=False)

        for _lower_name, merge_list in merge_dict.items():

            if any([g.nodes[m][DB.node_type_str] != DB.nts_external for m in merge_list]):
                raise NotImplementedError("The method is only for 'external' nodes.")

            all_out_edges = list()
            for m in merge_list:
                for n in g.neighbors(m):
                    if len(g[m][n]) != 1:
                        raise ValueError  # Make sure it is allowed to use '0' below.
                    else:
                        all_out_edges.append((m, n, 0))

            # Merge the edge attributes, if the target is the same
            distiled_out = dict()
            for edge_key in all_out_edges:
                target_node = edge_key[1]
                edge_data = g.edges[edge_key]

                if target_node not in distiled_out:
                    distiled_out[target_node] = edge_data
                else:
                    for i in distiled_out[target_node]:  # Go over attributes to update

                        if i != DB.connection_dict:
                            raise NotImplementedError(f"The method is only for certain edge attributes'{i}'.")

                        else:
                            for edi_db in edge_data[i]:  # {edi_db: {edi_ass: set()}}
                                if edi_db not in distiled_out[target_node][i]:
                                    distiled_out[target_node][i][edi_db] = edge_data[i][edi_db]
                                else:
                                    for edi_ass in edge_data[i][edi_db]:
                                        if edi_ass not in distiled_out[target_node][i][edi_db]:
                                            distiled_out[target_node][i][edi_db][edi_ass] = edge_data[i][edi_db][
                                                edi_ass
                                            ]
                                        else:  # Then, merge the releases
                                            distiled_out[target_node][i][edi_db][edi_ass] = (
                                                distiled_out[target_node][i][edi_db][edi_ass]
                                                | edge_data[i][edi_db][edi_ass]
                                            )

            for m in merge_list:  # all_in_edges = list()
                for n in reverse_g.neighbors(m):
                    if len(reverse_g[n][m]) != 1:
                        raise ValueError  # Make sure it is allowed to use '0' below.
                    else:
                        raise NotImplementedError("External nodes should always have edges going out, not going in.")
                        # all_in_edges.append((n, m, 0))  # edge key for forward graph (not reverse)

            merged_node_attributes = g.nodes[merge_list[0]].copy()
            for m in merge_list[1:]:
                node_att = g.nodes[m]
                for na in node_att:
                    if na not in merged_node_attributes:
                        raise ValueError
                    elif na == DB.node_type_str and merged_node_attributes[na] == node_att[na]:
                        pass
                    else:
                        raise NotImplementedError

            # Find new correct name, the one with most edges or most upper case elements is winner,
            correct_name_scorer = sorted(
                (
                    len(list(g.neighbors(m))),
                    len(list(reverse_g.neighbors(m))),
                    sum(1 for i in m if i in string.ascii_uppercase),
                    m,
                )
                for m in merge_list
            )
            correct_name = correct_name_scorer[0][-1]  # the best one, '-1' is as the last element is ID

            # remove nodes and all associated edges
            for m in merge_list:
                g.remove_node(m)

            g.add_node(correct_name, **merged_node_attributes)
            for target in distiled_out:
                g.add_edge(correct_name, target, 0, **distiled_out[target])

        removed_node_count = before_node_count - len(g.nodes)
        if removed_node_count > 0:
            self.log.info(f"Number of removed nodes in the process of merging synonymous nodes: {removed_node_count}")

        return g

    def construct_graph_form(self, narrow: bool, db_manager: DatabaseManager) -> TheGraph:
        """Creates a graph with connected nodes based on historical relationships between each Ensembl IDs.

        Args:
            narrow: See parameter in :py:attr:`_graph.Graph.construct_graph.narrow`
            db_manager: The method reads ID history dataframe, and Ensembl IDs lists at each Ensembl release,
                provided by ``DatabaseManager``.

        Returns:
            Resultant multi edge directed graph.

        Raises:
            ValueError: Unexpected error.
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

        self.log.info(f"Graph is being created: {db_manager.form}")

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

        g = TheGraph(  # because of multiple self loops of some nodes
            name=f"{db_manager.organism}_{db_manager.ensembl_release}_{db_manager.form}",
            type=db_manager.form,
            ensembl_release=db_manager.ensembl_release,
            genome_assembly=db_manager.genome_assembly,
            organism=db_manager.organism,
            confident_for_release=self.db_manager.available_releases,
            version_info=version_info,
            narrow=narrow,
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
            re_prev_rel = copy.deepcopy(rel_re)

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
            fo_prev_rel = copy.deepcopy(rel_fo)

        # In some cases, the table from `dm_manager.get('idhistory_narrow')` has some edges, that is completely
        # problematic. For example, 'ENSG00000289022' gene is defined in release_105, but it does not seem to
        # exist in the release gene id lists (neither 104, 105, 106 and also online sources).
        # Delete the edge if there are other edges from the previous node.
        self.log.info("Problematic nodes in Ensembl ID history are being removed.")
        ids_amc = set()
        problematic_nodes = list()
        for amc_rel in db_manager.available_releases:
            # Create a DatabaseManager object with the release of interest.
            amc_dm = db_manager.change_release(amc_rel)

            # Get the IDs and create a dictionary from ID to Version.
            ids_amc_df = amc_dm.get_db("ids", save_after_calculation=False)
            ids_amc.update(set(amc_dm.id_ver_from_df(ids_amc_df)))
        for node in g.nodes:
            if node not in ids_amc and GraphMaker.split_id(node, "Version") not in DB.alternative_versions:
                problematic_nodes.append(node)
        if len(problematic_nodes) > 0:
            self.log.warning(f"Nodes are deleted due to Ensembl ID history mistake: {len(problematic_nodes)}.")
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
        nx.set_node_attributes(g, {n: DB.nts_ensembl[db_manager.form] for n in g.nodes}, DB.node_type_str)
        nx.set_node_attributes(g, {n: n in latest_release_ids for n in g.nodes}, "is_latest")
        nx.set_node_attributes(g, {n: GraphMaker.split_id(n, "ID") for n in g.nodes}, "ID")
        nx.set_node_attributes(g, {n: GraphMaker.split_id(n, "Version") for n in g.nodes}, "Version")

        return g

    @staticmethod
    def split_id(id_to_split: str, which_part: str):
        """Todo.

        Args:
            id_to_split: Todo.
            which_part: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if id_to_split.count(DB.id_ver_delimiter) != 1:
            raise ValueError(f"Ensembl node has more than one delimiter. {id_to_split}")

        if which_part == "ID":
            return id_to_split.split(DB.id_ver_delimiter)[0]
        elif which_part == "Version":
            return (
                id_to_split.split(DB.id_ver_delimiter)[1] if id_to_split.count(DB.id_ver_delimiter) == 1 else np.nan
            )  # there are max 1 as checked previously
        else:
            raise ValueError

    @staticmethod
    def remove_non_gene_trees(graph: TheGraph, forms_remove: list = None) -> TheGraph:
        """Removes the edges between the nodes with the same `node type` and removes abstract nodes (Void and Retired).

        The nodes between two the same ``DB.node_type_str`` will be removed. Also, the nodes with versions
        ``DB.no_new_node_id`` and ``DB.no_old_node_id`` will be also removed.

        Args:
            graph: The output of :py:attr:`_graph.Graph.construct_graph` or
                :py:attr:`_graph.Graph.construct_graph_form`.
            forms_remove: Determine which `node type` are of interest.

        Returns:
            Resultant multi edge directed graph.
        """
        forms_remove = (
            [DB.nts_ensembl[i] for i in ["transcript", "translation"]] if forms_remove is None else forms_remove
        )

        node_to_remove = []
        edge_to_remove = []

        for n in graph.nodes:

            the_node = graph.nodes[n]
            nt = the_node[DB.node_type_str]

            if nt in forms_remove:

                if the_node["Version"] in DB.alternative_versions:
                    node_to_remove.append(n)  # we only remove Void or retired
                else:
                    for m in graph.neighbors(n):
                        mt = graph.nodes[m][DB.node_type_str]
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
    ) -> TheGraph:
        """Simplifies the graph construction process.

        Args:
            narrow: See parameter in :py:attr:`_graph.Graph.construct_graph.narrow`
            create_even_if_exist: Determine whether create the graph even if it exists. If there is no graph in the
                provided temporary directory, the graph will be created regardless.
            save_after_calculation: Determine whether resultant graph will be saved or not.
            overwrite_even_if_exist: If the graph will be saved, determine whether the program should overwrite.
                If ``False``, it does not re-saves the calculated (or loaded) graph.

        Returns:
            Resultant multi edge directed graph, which can be used in all future calculations.
        """
        # Get the file name and narrow parameter.
        file_path = self.create_file_name(narrow)

        # If the file name is not accessible for reading, or explicitly prompt to do so, then create the graph.
        if not os.access(file_path, os.R_OK) or create_even_if_exist:
            self.log.info("The graph is being constructed.")
            g = self.construct_graph(narrow)
        else:  # Otherwise, just read the file that is already in the directory.
            self.log.info("The graph is being read.")
            g = GraphMaker.read_exported(file_path)

        # If prompt, save the dataframe in requested format.
        if save_after_calculation:
            self.export_disk(g, file_path, overwrite_even_if_exist)

        return g

    @staticmethod
    def read_exported(file_path: str) -> TheGraph:
        """Read the `pickle` file in the provided file path, which contains the graph.

        Args:
            file_path: Absolute path of the file of interest.

        Returns:
            Resultant multi edge directed graph.

        Raises:
            FileNotFoundError: When there is no file in the provided directory.
        """
        if not os.access(file_path, os.R_OK):
            raise FileNotFoundError

        return nx.read_gpickle(file_path)

    def create_file_name(self, narrow: bool) -> str:
        """File name creator which includes some information regarding the construction process.

        Facilitates to recognize the graph based on file name.

        Args:
            narrow: See parameter in :py:attr:`_graph.Graph.construct_graph.narrow`

        Returns:
            Absolute file path in the temporary directory provided by ``DatabaseManager``.
        """
        narrow_ext = "_narrow" if narrow else ""
        min_ext = f"_min{self.db_manager.ignore_before}" if not np.isinf(self.db_manager.ignore_before) else ""
        max_ext = f"_max{self.db_manager.ignore_after}" if not np.isinf(self.db_manager.ignore_after) else ""
        ext = f"ens{self.db_manager.ensembl_release}{min_ext}{max_ext}{narrow_ext}"
        return os.path.join(self.db_manager.local_repository, f"graph_{self.db_manager.organism}_{ext}.pickle")

    def export_disk(self, g: TheGraph, file_path: str, overwrite: bool):
        """Write the `pickle` file in the provided file path, which contains the graph.

        Args:
            g: Multi edge directed graph object to stor in the disk.
            file_path: Absolute target path, provided by :py:meth:`_graph.Graph.create_file_name`
            overwrite: See parameter in :py:attr:`_graph.Graph.get_graph.overwrite_even_if_exist`
        """
        if not os.access(file_path, os.R_OK) or overwrite:
            self.log.info(f"The graph is being exported as '{file_path}'.")
            nx.write_gpickle(g, file_path)
