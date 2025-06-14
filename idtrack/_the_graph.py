#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import itertools
import logging
import re
from collections import Counter
from functools import cached_property
from typing import Optional, Union

import networkx as nx
import numpy as np

from idtrack._db import DB


class TheGraph(nx.MultiDiGraph):
    """Represent a bio-identifier multigraph with IDTrack-specific helpers.

    The class extends :py:class:`networkx.MultiDiGraph` to model historical and
    cross-reference relationships between Ensembl identifiers (genes,
    transcripts, translations) and third-party database accessions
    (UniProt, RefSeq, …).  It is built by
    :py:class:`idtrack._graph_maker.GraphMaker`, then queried by
    :py:class:`idtrack.Track` for high-performance path-finding across Ensembl
    releases and external resources.

    Additional cached properties (e.g. :py:attr:`rev`,
    :py:attr:`combined_edges`, and :py:attr:`hyperconnective_nodes`) collapse
    expensive aggregate calculations into single attribute look-ups, while
    helpers such as :py:meth:`attach_included_forms` record which biological
    forms were merged into a particular instance.  Together these conveniences
    allow downstream algorithms to traverse millions of edges without the
    memory overhead of duplicating graphs or recomputing summaries.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate the multigraph and configure package logging.

        All positional and keyword arguments are forwarded verbatim to
        :py:class:`networkx.MultiDiGraph`, allowing callers to pre-seed the graph
        with nodes, edges, or name/metadata attributes exactly as they would with a
        vanilla NetworkX constructor.  After delegating to ``super().__init__``,
        the method initialises two convenience attributes:

        - :py:data:`log` — a dedicated ``logging.Logger`` named ``"the_graph"`` for
            structured, per-instance diagnostics.
        - :py:data:`available_forms` — a placeholder set to ``None`` until
            :py:meth:`attach_included_forms` is called by :py:class:`idtrack._graph_maker.GraphMaker`.

        Args:
            args (Any): Positional arguments accepted by :py:meth:`networkx.MultiDiGraph.__init__`.
            kwargs (Any): Keyword arguments accepted by :py:meth:`networkx.MultiDiGraph.__init__`.
        """
        super().__init__(*args, **kwargs)  # SubClass initialization

        # Other variables
        self.log = logging.getLogger("the_graph")
        self.available_forms = None

    def _attach_included_forms(self, available_forms: list[str]) -> None:
        """Record which Ensembl forms are present in the merged graph.

        Graphs for *gene*, *transcript*, and *protein* are first built
        independently by :py:class:`~idtrack._graph_maker.GraphMaker` and then
        merged into a single :py:class:`TheGraph` instance.  This helper runs
        **after** that merge to store the subset of forms that actually made it
        into the final graph—information required by several cached properties
        (e.g. :py:data:`available_external_databases`) for consistency checks
        and downstream analyses.  Calling the method **before** the merge would
        mis-report available forms and corrupt those caches.

        Args:
            available_forms (list[str]): Exact list of included forms
                (typically ``["gene", "transcript", "protein"]``).  Order is
                preserved so callers can rely on a deterministic iteration
                sequence.
        """
        self.available_forms = available_forms

    def calculate_caches(self, for_test: bool = False) -> None:
        """Eagerly materialise every ``@cached_property`` to prime the cache.

        Accessing a cached property for the first time triggers an expensive
        computation. Batch-loading all of them up-front improves latency for
        subsequent graph queries and simplifies unit-test expectations because
        no additional properties are computed lazily in the background.

        The optional *for_test* flag activates a few heavyweight diagnostics
        that are normally skipped in production but useful for test suites and
        profiling.

        Args:
            for_test (bool): If ``True`` (default), also compute
                caches that exist solely for testing or sanity-check purposes
                (e.g. :py:data:`external_database_connection_form`).  Set to
                ``False`` to warm only the properties required at run-time.
        """
        _ = self.combined_edges
        _ = self.combined_edges_assembly_specific_genes
        _ = self.combined_edges_genes
        _ = self.lower_chars_graph
        _ = self.get_active_ranges_of_id
        _ = self.available_external_databases
        _ = self.available_genome_assemblies
        _ = self.available_external_databases_assembly
        _ = self.node_trios
        _ = self.hyperconnective_nodes
        if for_test:
            _ = self.external_database_connection_form
            _ = self.available_releases_given_database_assembly

    @cached_property
    def rev(self) -> "TheGraph":
        """Return a view of the same graph with all edge directions reversed.

        The call delegates to :py:meth:`networkx.MultiDiGraph.reverse` with
        ``copy=False``, meaning the returned object re-uses the underlying data
        structures and therefore consumes **no additional memory**.  Use this
        property whenever a temporal walk must proceed *backwards* in history
        (e.g. when resolving identifiers from a newer to an older Ensembl
        release).

        Returns:
            TheGraph: A non-copying, lazily constructed reverse-orientation
                view that honours every node and edge attribute of the original graph.
        """
        return self.reverse(copy=False)

    @cached_property
    def hyperconnective_nodes(self) -> dict[str, int]:
        """Return hyper-connective external nodes and their out-degree counts.

        Hyper-connective nodes are external identifiers whose out-degree (number of outgoing edges) exceeds
        :py:attr:`idtrack._db.DB.hyperconnecting_threshold`. Because such nodes may participate in tens of thousands
        of mappings, they explode the breadth-first frontier of the synonym *pathfinder* algorithm and become a major
        performance bottleneck.  The algorithm therefore **ignores** these nodes, sacrificing a small amount of
        theoretical precision for a substantial speed-up.

        In practice the precision penalty is negligible: hyper-connective nodes tend to be coarse-grained identifiers
        that already suffer from low mapping specificity (for example, generic protein or transcript accessions
        re-used across many unrelated biological entities).  Meaningful, one-to-one synonym relationships are almost
        always reachable through alternative external identifiers.  Consequently, ignoring hyper-connective nodes both
        accelerates the search and often improves the overall relevance of the results.

        The value is computed lazily on first access and memoised via :py:meth:`functools.cached_property`, so the
        underlying query runs at most once per :py:class:`~idtrack._the_graph.TheGraph` instance.

        Returns:
            dict[str, int]: Mapping from external node identifier to its out-degree, limited to nodes whose
                out-degree is greater than :py:attr:`idtrack._db.DB.hyperconnecting_threshold` and whose
                :py:data:`idtrack._db.DB.node_type_str` equals :py:data:`idtrack._db.DB.nts_external`.
        """
        self.log.info(f"Cached properties being calculated: {'hyperconnective_nodes'}")

        hcn_dict = dict()  # Initialize a dictionary

        for hcn in self.nodes:
            od = self.out_degree[hcn]
            # Note that external ids are always has edges going out, so in_degree is not relevant here.

            if self.nodes[hcn][DB.node_type_str] == DB.nts_external and od > DB.hyperconnecting_threshold:
                hcn_dict[hcn] = od

        return hcn_dict

    @cached_property
    def combined_edges(self) -> dict:
        """Aggregate outgoing-edge metadata for every non-gene node in the graph.

        This cached view pre-computes, for each backbone or external identifier, which external databases,
        genome assemblies, and Ensembl releases are reachable through *outgoing* edges—while purposely
        excluding Ensembl gene and assembly-specific gene nodes.  The summary accelerates synonym search
        and other traversal routines in :py:meth:`idtrack.track.Track.pathfinder` because consumers can
        consult a compact dictionary instead of repeatedly iterating raw NetworkX edges and attributes.

        Returns:
            dict: Nested mapping of the form ``{node_name: {database_name: {assembly: set[int]}}}``, where

                * **node_name** (*str*) - Identifier of the start node whose edges were inspected.
                * **database_name** (*str*) - Canonical name of the external database or Ensembl sub-type
                    (e.g. ``uniprot``, ``refseq_rna``, ``assembly_x_ensembl_gene``).
                * **assembly** (*str*) - UCSC-style assembly label (e.g. ``GRCh38``); ``None`` when the edge
                    is not assembly-scoped.
                * **set[int]** - Collection of Ensembl release numbers in which the connection is valid.

        Notes:
            *Edges that link two nodes of the **same** node-type are ignored,* ensuring the dictionary
            focuses on cross-type relationships that matter for ID translation.
        """
        self.log.info(f"Cached properties being calculated: {'combined_edges'}")

        # Note that the `TheGraph._combined_edges` method does not return assembly_specific_genes and ensembl_gene.
        result = TheGraph._combined_edges(self.nodes, self)
        return result

    @cached_property
    def combined_edges_genes(self) -> dict:
        """Aggregate incoming-edge metadata for Ensembl gene nodes.

        Gene nodes only possess **incoming** edges (toward the gene); therefore the calculation traverses
        the graph in reverse (``self.rev``) to collect equivalent information to
        :py:meth:`~TheGraph.combined_edges`, but restricted solely to nodes whose
        :py:data:`idtrack._db.DB.node_type_str` is ``DB.nts_ensembl["gene"]``.  The result merges edge
        data from *all* contributing external databases so that downstream callers receive one consolidated
        view per gene.

        Returns:
            dict: Nested mapping ``{gene_id: {database_name: {assembly: set[int]}}}``.
                A single gene may appear under multiple assemblies when reference genomes share that transcript locus.
        """
        self.log.info(f"Cached properties being calculated: {'combined_edges_genes'}")

        # Choose only the nodes that has a node type of 'ensembl_gene'
        non_captured_genes = [nd for nd in self.nodes if self.nodes[nd][DB.node_type_str] == DB.nts_ensembl["gene"]]
        # Note that TrackTest.is_combined_edges_dicts_overlapping_and_complete checks whether there is an error
        # in terms of missing nodes, or repeated nodes in all combined_edges properties.

        result = TheGraph._combined_edges(non_captured_genes, self.rev)
        return {i: TheGraph._combined_edges_genes_helper(result[i]) for i in result}

    @cached_property
    def combined_edges_assembly_specific_genes(self) -> dict:
        """Aggregate incoming‐edge metadata for assembly-specific Ensembl gene nodes.

        Assembly-specific gene identifiers (e.g. ``GRCh37:ENSG00000123456``) represent loci that differ
        between reference builds.  This property mirrors the logic of
        :py:meth:`~TheGraph.combined_edges_genes` but targets nodes *not* captured by that property,
        ensuring the three cached dictionaries are mutually exclusive and collectively exhaustive.
        Because each such gene belongs to exactly **one** assembly, the returned structure always contains
        a single assembly key per outer node.

        Returns:
            dict: Mapping ``{assembly_specific_gene_id: {database_name: {assembly: set[int]}}}`` where the sole
                *assembly* key matches the assembly implied by the node's own identifier.
        """
        self.log.info(f"Cached properties being calculated: {'combined_edges_assembly_specific_genes'}")

        # Choose only the nodes that has a node type of 'assembly_x_ensembl_gene'.
        non_captured_genes = [
            nd
            for nd in self.nodes
            if nd not in self.combined_edges and self.nodes[nd][DB.node_type_str] != DB.nts_ensembl["gene"]
        ]
        # Note that TrackTest.is_combined_edges_dicts_overlapping_and_complete checks whether there is an error
        # in terms of missing nodes, or repeated nodes in all combined_edges properties.

        result = TheGraph._combined_edges(non_captured_genes, self.rev)
        return {i: TheGraph._combined_edges_genes_helper(result[i]) for i in result}

    @staticmethod
    def _combined_edges_genes_helper(the_result) -> dict:
        """Merge per-neighbour edge metadata for gene-centric queries.

        This helper is used exclusively by :py:meth:`TheGraph.combined_edges_genes` and
        :py:meth:`TheGraph.combined_edges_assembly_specific_genes` to post-process
        the dictionaries returned by :py:meth:`TheGraph._combined_edges`.
        Because backbone *gene* nodes have no outgoing edges except to other gene
        nodes, the caller invokes :py:meth:`TheGraph._combined_edges` on a
        **reversed** graph and receives one nested dictionary per neighbour.
        The present routine

        1. Flattens those per-neighbour sub-dicts so that information from
            multiple neighbours of the same external database and assembly is unified.
        2. Re-labels the generic ``ensembl_gene`` key to the assembly-qualified
            form ``assembly_<N>_ensembl_gene`` so that the provenance of every
            entry remains explicit and consistent with the rest of the code base.

        Args:
            the_result (dict): Nested mapping produced by
                :py:meth:`TheGraph._combined_edges` for a *single* gene node.
                The structure is  ``{neighbour: {database: {assembly: set[int]}}}``.

        Returns:
            dict: Collapsed mapping  ``{database: {assembly: set[int]}}`` where all neighbour-level
                dictionaries have been merged and database names have been renamed
                to their assembly-specific counterparts when appropriate.
        """
        output: dict = dict()  # Initialize a dict
        for i in the_result:
            for j in the_result[i]:
                if j not in output:
                    output[j] = set()
                output[j].update(the_result[i][j])  # Combine the info from databases

        # Rename the database_name as mentioned in the docstring. Also, separate assemblies accordingly.
        output = {
            DB.nts_assembly[i][DB.backbone_form]: {j: copy.deepcopy(output)[j] for j in output if j == i}
            for i in output
        }
        return output

    @staticmethod
    def _combined_edges(node_list: Union[nx.classes.reportviews.NodeView, list], the_graph: nx.MultiDiGraph) -> dict:
        """Aggregate database/assembly/release metadata for the edges of *node_list*.

        The routine is the work-horse behind the
        :py:attr:`TheGraph.combined_edges` family of cached properties.
        It iterates over every node in *node_list*, inspects each outgoing
        (or, when *the_graph* is a reversed view, incoming) edge, and builds a
        deterministic description of which external database, genome assembly,
        and Ensembl release the connection originates from.

        Edges that link two nodes of the **same** node-type are ignored so that
        backbone history links (gene ↔ gene, transcript ↔ transcript, …) do not
        pollute the output (as tested in
        :py:meth:`idtrack._track_tests.TrackTest.is_edge_with_same_nts_only_at_backbone_nodes`).
        For edges whose database key is one of the generic Ensembl forms
        (``ensembl_gene``, ``ensembl_transcript``, …) the key is rewritten to the
        assembly-specific variant (e.g. ``assembly_38_ensembl_gene``) to keep
        assemblies logically separate in downstream analyses.

        Args:
            node_list (NodeView | list[str]): Nodes whose edge metadata will be
                consolidated.  Accepts either a plain list or the
                :py:class:`networkx` view returned by ``graph.nodes``.
            the_graph (nx.MultiDiGraph): Graph to inspect.  Pass ``self`` for the
                native orientation or ``self.rev`` when a reverse walk is
                required.

        Returns:
            dict: Mapping  ``{node: {database: {assembly: set[int]}}}`` that summarises every
                admissible edge attached to the requested nodes.
        """
        result: dict = dict()
        for i in node_list:
            for j in the_graph.neighbors(i):
                if the_graph.nodes[i][DB.node_type_str] != the_graph.nodes[j][DB.node_type_str]:
                    # Exclude the edges connecting the nodes from the same node type (exclude backbone connections).

                    if i not in result:
                        result[i] = dict()
                    edge_info = the_graph[i][j][0][DB.connection_dict]

                    for db_name in edge_info:
                        if db_name in DB.nts_ensembl_reverse:  # if db_name is a ensembl_x
                            _db_name_form = DB.nts_ensembl_reverse[db_name]  # get the form
                            # Get the corresponding node type in assembly specific dictionary. This is to make
                            # sure assemblies are intuitively separated into different db_names and make
                            # all subsequent calculation consistent.
                            _db_name = DB.nts_assembly[the_graph.graph["genome_assembly"]][_db_name_form]
                        else:
                            _db_name = copy.deepcopy(db_name)

                        if _db_name not in result[i]:
                            result[i][_db_name] = dict()

                        for assembly_name in edge_info[db_name]:
                            if assembly_name not in result[i][_db_name]:
                                result[i][_db_name][assembly_name] = set()

                            release_set = edge_info[db_name][assembly_name]
                            result[i][_db_name][assembly_name].update(release_set)
        return result

    @cached_property
    def lower_chars_graph(self) -> dict:
        """A simple dictionary mapping the node name with its lower characters into the original node name.

        Raises:
            ValueError: If there are multiple nodes which becomes the same after lower character conversion.

        Returns:
            A dictionary with following format
            ``{lower_char_id: id}``
        """
        self.log.info(f"Cached properties being calculated: {'lower_chars_graph'}")

        result = dict()
        for i in self.nodes:  # Iterate through all nodes.
            j = i.lower()

            if j not in result:  # Make sure there is only one and only one after lower conversion.
                result[j] = i
            else:
                raise ValueError(f"The node {i} and {result[j]} has the same lower character formats.")

        return result

    def node_name_alternatives(self, identifier: str) -> tuple[Optional[str], bool]:
        """Matching a query ID into the ID found in the graph based on some criteria and priorities.

        A query ID is sometimes not found exactly in the graph due to the format it has. However, very slight
        modifications of the string of query ID could help the pathfinder locate the ID of interest. For example,
        `actb` can be queried for the pathfinder, but nothing is found as upper character version of the query,
        `ACTB`, is found in the graph instead.

        Priority is as follows: (1) try to find directly in the graph (2) look for lower-char version (3) look for
        initial substring before separators (4) check all possible variations of dash and underscore for each of which
        follow the priority list above.

        Args:
            identifier: A bio-ID of interest. This could be an ID in the graph or some other ID to be matched
                with the corresponding ID in the graph.

        Returns:
            A tuple with the first element is the bio-ID in the graph if the query is found somehow, else ``None``.
                The second element is to show whether the query ID is found without any modifications or not.
        """

        def _node_name_alternatives(the_id: str) -> tuple[Optional[str], bool]:
            """Helper function for :py:meth:`TheGraph.node_name_alternatives`.

            Args:
                the_id: A bio-ID of interest as defined in :py:attr:`TheGraph.node_name_alternatives.identifier`.

            Returns:
                The same as the parental method.
            """

            def compare_lowers(id_to_find: str) -> tuple[Optional[str], bool]:
                """Check whether lower-character ID is found.

                Args:
                    id_to_find: Query ID.

                Returns:
                    Tuple of found ID (None if unfound) as the first element, and whether it is found as the second.
                """
                lower_id_find = id_to_find.lower()

                if lower_id_find in self.lower_chars_graph:
                    return self.lower_chars_graph[lower_id_find], True

                else:
                    return None, False  # If cannot return anything, just return None (unfound).

            def check_variation(id_str: str) -> tuple[Optional[str], bool]:
                """Search ID in the graph without flanking substring.

                Args:
                    id_str: Query ID.

                Returns:
                    Tuple of found ID (None if unfound) as the first element, and whether it is found as the second.
                """
                # First, try to find the lower character version of the querry.
                lower_id, is_lower_found = compare_lowers(id_str)
                if is_lower_found:  # If something is found, just return it.
                    return lower_id, True

                # Them, try to match with the regular-expression pattern. The pattern is basically to remove any flanking
                # numbers (possibly versions) separated with following characters '-', '_', or '.'. In order to match with
                # the regex patter, the query has to have these separators, but the subsequent integer is optional. Note
                # that the last separator is of interest only.
                regex_found = regex_pattern.match(id_str)
                if regex_found:
                    new_id = regex_found.groups()[
                        0
                    ]  # If found, just get the first section (e.g. the ID without version)

                    if new_id in self.nodes:
                        return new_id, True  # If the substring is in the graph, just return it.

                    # If not, check whether the lower-character version of the substring is in the graph.
                    lower_id, is_lower_found = compare_lowers(new_id)
                    if is_lower_found:
                        return lower_id, True

                return None, False  # If cannot return anything, just return None (unfound).

            def possible_alternatives(the_id_pa: str) -> list:
                """Search a query ID with all possible substitutions of '_' and '-'.

                Sometimes the query ID has a '-' in somewhere but the corresponding ID in the graph has '_', or vice versa.
                The method here creates all possible combinations of query ID where '_' and '-' characters are replaced.
                For example, if query ID is "AC-TB_1", the method returns: ["AC_TB_1", "AC-TB-1", "AC-TB_1", "AC_TB-1"].

                Args:
                    the_id_pa: Query ID.

                Returns:
                    All possible versions of query ID.
                """
                # Get the indexes of the characters of interests ('_' and '-') in the query ID.
                char_indices = [ind for ind, i in enumerate(the_id_pa) if i in ["-", "_"]]

                possible_alternatives = list()
                if len(char_indices) > 0:  # If there are more than 1 of such.
                    for comb in range(len(char_indices) + 1):
                        for replace_indices in itertools.combinations(char_indices, comb):
                            # Create the every combination of such ID and append to the list
                            replace_indices_other = [i for i in char_indices if i not in replace_indices]
                            new_id_l = list(the_id_pa)
                            for ri in replace_indices:
                                new_id_l[ri] = "_"
                            for rio in replace_indices_other:
                                new_id_l[rio] = "-"
                            possible_alternatives.append("".join(new_id_l))

                return possible_alternatives

            # To be used on check_variation.
            regex_pattern = re.compile(r"^(.+)(_|-|\.)[0-9]+$")

            if the_id in self.nodes:  # If the ID is already found in the graph, just return it.
                return the_id, False

            mti1, mti2 = check_variation(the_id)
            if mti2:  # If a variation is found
                return mti1, mti2  # Return the found ID and indicate it is a variation.

            for pa in possible_alternatives(the_id):
                if pa in self.nodes:
                    return pa, True

                mpa1, mpa2 = check_variation(pa)
                if mpa2:  # If a variation is found
                    return mpa1, mpa2  # Return the found ID and indicate it is a variation.

            return None, False  # If cannot return anything, just return None (unfound).

        new_ident, is_conv = _node_name_alternatives(identifier)  # Check with the base function

        if new_ident is None:  # If not found, check with the synonym prefix added version.
            new_ident, is_conv = _node_name_alternatives(f"{DB.synonym_id_nodes_prefix}{identifier}")

        return new_ident, is_conv

    @cached_property
    def get_active_ranges_of_id(self) -> dict[str, list[list]]:
        """Returns the range of active ensembl releases of nodes, ignoring which assembly the release is coming from.

        Returns:
            A dictionary with following format
            ``{id: list_of_ranges``.
        """
        self.log.info(f"Cached properties being calculated: {'get_active_ranges_of_id'}")
        return {n: self._get_active_ranges_of_id(n) for n in self.nodes}

    def _get_active_ranges_of_id(self, input_id: str) -> list[list]:
        """Calculating the range of active ensembl releases of nodes separately for backbone nodes and others.

        Args:
            input_id: Query ID.

        Returns:
            Ranges of the ID as list of lists. Outputs should always be increasing, inclusive ranges.
        """

        def _get_active_ranges_of_id_nonbackbone(the_id: str) -> list[list]:
            """For the non-backbone nodes, calculates the ranges of IDs using `combined_edges` dictionaries.

            Args:
                the_id: Query ID.

            Returns:
                Ranges of the ID as list of lists. Outputs should always be increasing, inclusive ranges.
            """
            the_node_type = self.nodes[the_id][DB.node_type_str]

            # Get the node info from the relevant combined edge dictionary
            if the_node_type in DB.nts_assembly_gene:
                rd = self.combined_edges_assembly_specific_genes[the_id]
            else:
                rd = self.combined_edges[the_id]

            # Create the sorted list of all ensembl releases defined for the given node.
            rels = sorted({s for p in rd for r in rd[p] for s in rd[p][r]})
            return TheGraph.list_to_ranges(rels)  # Convert the list of ensembl releases into range.

        def _get_active_ranges_of_id_backbone(the_id: str) -> list[list]:
            """For the backbone nodes, calculates the ranges of IDs.

            Args:
                the_id: Query ID.

            Returns:
                Ranges of the ID as list of lists. Outputs should always be increasing, inclusive ranges.

            Raises:
                ValueError: If there is no ID going out and going in to the query ID.
            """
            # Get the in- and out-nodes via 'get_next_edge_releases' method.
            t_outs = self.get_next_edge_releases(from_id=the_id, reverse=True)
            t_ins = self.get_next_edge_releases(from_id=the_id, reverse=False)

            if len(t_outs) == 0 and len(t_ins) == 0:
                raise ValueError(f"No out and in edges for the given ID: {the_id}.")

            elif len(t_outs) == 0:
                if self.nodes[the_id]["Version"] != DB.no_old_node_id:
                    # Make sure the graph is constructed as it should be.
                    raise ValueError(f"If no-out node, the version should be `{DB.no_old_node_id}`: {the_id}.")
                # The t_outs is then the first possible ensembl release. Note that t_ins is not empty.
                t_outs = [min(self.graph["confident_for_release"])]

            elif len(t_ins) == 0:
                if self.nodes[the_id]["Version"] != DB.no_new_node_id:
                    # Make sure the graph is constructed as it should be.
                    raise ValueError(f"If no in-node, the version should be `{DB.no_new_node_id}`: {the_id}.")
                # The t_ins is then the last possible ensembl release. Note that t_outs is not empty.
                t_ins = [max(self.graph["confident_for_release"])]

            # Sort in- and out-releases in increasing order.
            # If one in-edge and out-edge has the same ensembl release, put in-edges before.
            inout_edges = sorted(
                itertools.chain(zip(t_outs, itertools.repeat(True)), zip(t_ins, itertools.repeat(False))),
                reverse=False,
                key=lambda k: (k[0], -k[1]),
            )  # Create a dict with ensembl releases and information of being in- or out-edge.

            narrowed = []
            active_state = False
            # Start from the lowest ensembl release and go up at each iteration.
            # Assume in the beginning, the ID is not active.
            for ind, (ens_rel, inout) in enumerate(inout_edges):
                if ind == 0:
                    assert inout, "The range building should start with in-edge."

                if not active_state:
                    # If ID is not active and there is in-node, there is a beginning of new active range.
                    if inout:
                        narrowed.append(ens_rel)
                        active_state = True  # Set the ID active.

                    # If ID is not active and there is out-node, modify the end of last active range, because to have
                    # a out-node, the ID must be active (so last element is possibly a branching event).
                    else:
                        narrowed[-1] = ens_rel
                        active_state = False

                else:
                    # If ID is active and there is in-node, do nothing.
                    if inout:
                        pass

                    # If ID is active and there is out-node, end the range.
                    else:
                        narrowed.append(ens_rel)
                        active_state = False  # Set the ID not active.

            # Group the results as list of list.
            narrowed = [narrowed[i : i + 2] for i in range(0, len(narrowed), 2)]
            return narrowed

        # Use associated function to create the ranges of a node.
        if self.nodes[input_id][DB.node_type_str] == DB.external_search_settings["nts_backbone"]:
            return _get_active_ranges_of_id_backbone(input_id)
        else:
            return _get_active_ranges_of_id_nonbackbone(input_id)

    def get_active_ranges_of_id_ensembl_all_inclusive(self, the_id: str) -> list[list]:
        """Generate active ranges of Ensembl gene nodes with all assemblies.

        Note that :py:meth:`TheGraph.get_active_ranges_of_id` method provided the range for main assembly
        that the graph is build on. This method combines the other assemblies together. Also, it verifies whether
        the :py:meth:`TheGraph.combined_edges` and :py:meth:`TheGraph.get_active_ranges_of_id`
        methods provides consistent results.

        Args:
            the_id: Query ID. Should be either Ensembl gene or assembly specific Ensembl gene IDs.

        Raises:
            ValueError: If there is inconsistency between the outputs of these two functions.
                If the query is not one of the specified node type.

        Returns:
            Ranges of the ID as list of lists. Outputs should always be increasing, inclusive ranges.
        """
        # Use associated function to create the ranges of a node.
        ndt = self.nodes[the_id][DB.node_type_str]
        main_assembly = self.graph["genome_assembly"]

        if ndt == DB.nts_ensembl["gene"]:
            narrowed = self.get_active_ranges_of_id[the_id]  # Get the range of main assembly.
            comb_result = self.combined_edges_genes[the_id]  # Get the range of main assembly and also others.
            comb_reduced: dict[int, set] = dict()  # Create a dict that flattens all ensembl releases on assemblies.
            for i in comb_result:
                for j in comb_result[i]:
                    if j not in comb_reduced:
                        comb_reduced[j] = set()
                    comb_reduced[j].update(comb_result[i][j])

            # Sanity check with externals/other forms etc.

            if not all([TheGraph.is_point_in_range(narrowed, i) for i in comb_reduced[main_assembly]]):
                raise ValueError(f"Inconsistency in ID range vs combined edges: {the_id}, {narrowed}, {comb_reduced}.")

            # Note that ``narrowed == self.list_to_ranges(comb_reduced[main_assembly])`` may not give always True.
            # It is basically because the ID may be defined even though there is no external ID pointing towards it.

            other_assemblies = [j for i in comb_reduced for j in comb_reduced[i] if i != main_assembly]
            result_list = self.ranges_to_list(narrowed) + other_assemblies

            # Cannot use compact_ranges method as it necessitates non-overlapping increasing ranges.
            return TheGraph.list_to_ranges(sorted(set(result_list)))

        elif ndt in DB.nts_assembly_gene:
            return self.get_active_ranges_of_id[the_id]

        else:
            raise ValueError(f"Query `{the_id}` isn't `{DB.nts_ensembl['gene']}` or in `{DB.nts_assembly_gene}`.")

    def get_next_edge_releases(self, from_id: str, reverse: bool) -> list:
        """Retrieves the next edge releases from a node, depending on the directionality of the graph.

        Args:
            from_id: Query ID. Should be with node type of Ensembl gene.
            reverse: The direction of desired next edges.
                If ``True``, previous edges are returned. If ``False``, next edges are returned.

        Returns:
            The Ensembl releases of next (or previous if reverse is ``True``) edges.

        Raises:
            ValueError: If the query ID is not with the node type of graph backbone.
        """
        if self.nodes[from_id][DB.node_type_str] != DB.external_search_settings["nts_backbone"]:
            raise ValueError(f"The method should be called only for backbone nodes: `{from_id}`.")

        return list(
            {
                (
                    an_edge["old_release"]  # (8) Get the 'old release' attribute of the edge.
                    if (not np.isinf(an_edge["new_release"]) and not reverse)  # (7) In forward dir and non-retired edge
                    else an_edge["new_release"]
                )  # (9) Else (reverse dir or non-retired edge), get the 'new release' attr.
                # (1) Get the node_after based on the direction of interest
                for node_after in nx.neighbors(self if not reverse else self.rev, from_id)
                # (2) Get the edge data between each node_after and from_id.
                # (3) For every multi-edge-ind (mei).
                for mei, an_edge in (self if not reverse else self.rev).get_edge_data(from_id, node_after).items()
                if (
                    # (4) Check if the connection is in the backbone
                    self.nodes[node_after][DB.node_type_str] == self.nodes[from_id][DB.node_type_str]
                    and (
                        node_after != from_id  # (5) Check if this is not a self-loop.
                        or (
                            np.isinf(an_edge["new_release"])
                            # (6.1) or self-loop but an_edge["new_release"] is np.inf (happens when not retired at all)
                            and not reverse
                        )  # (6.2) but if we are at forward.
                    )  # (6.3) That is, keep inf self-loop for forward.
                )
            }  # (10) Create a set out of those to remove the duplicates.
        )  # (11) Convert into a list at the end.

    def get_active_ranges_of_base_id_alternative(self, base_id: str) -> list[list]:
        """Get the range of an base ID based on the child IDs it is connected to.

        Args:
            base_id: Query ID. Should be with node type of Ensembl base ID.

        Returns:
            Ranges of the ID as list of lists. Outputs should always be increasing, inclusive ranges.
        """
        associated_ids = self.neighbors(base_id)
        all_ranges = [r for ai in associated_ids for r in self.get_active_ranges_of_id_ensembl_all_inclusive(ai)]
        return self.list_to_ranges(self.ranges_to_list(all_ranges))

    @staticmethod
    def list_to_ranges(lst: list[int]) -> list[list]:
        """Convert sorted non-repeating list of integers into list of inclusive non-overlapping ranges.

        Args:
            lst: List of integers. It should be sorted in increasing order. Repeating element is not allowed.
                The output of :py:meth:`TheGraph.ranges_to_list` is a perfect input here.

        Returns:
            Ranges as list of lists. Outputs should always be increasing, inclusive ranges. With positive integers.
        """
        res = list()
        for _, a in itertools.groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
            b = list(a)
            res.append([b[0][1], b[-1][1]])
        return res

    def ranges_to_list(self, lor: list[list]) -> list[int]:
        """Convert list of inclusive non-overlapping ranges into sorted sorted non-repeating list of integers.

        Args:
            lor: Ranges as list of lists. Should always be increasing, inclusive ranges. With positive integers.
                The output of :py:meth:`TheGraph.list_to_ranges` is a perfect input here.

        Returns:
            Sorted non-repeating list of integers.
        """
        return sorted(
            {
                k
                for i, j in lor
                for k in range(i, j + 1 if not np.isinf(j) else max(self.graph["confident_for_release"]) + 1)
            }
        )

    @cached_property
    def node_trios(self) -> dict[str, set[tuple]]:
        """Creates a dict for all nodes with `node_trios` calculated by :py:meth:`TheGraph._node_trios`.

        Returns:
            A memory intensive dictionary with node name as the key and calculated `node_trios` as the value.
        """
        self.log.info(f"Cached properties being calculated: {'node_trios'}")
        return {n: self._node_trios(n) for n in self.nodes}

    def _node_trios(self, the_id: str) -> set[tuple]:
        """Calculates the unique tuple called `trios` (database, assembly, Ensembl release) for a given ID.

        Args:
            the_id: Query ID.

        Returns:
            Set of trios.
        """

        def non_inf_range(l1: int, l2: Union[float, int]) -> range:
            """Convert the np.inf range element into a Ensembl release.

            Args:
                l1: Left hand side of a range.
                l2: Right hand side of a range.
                    This item is converted into the max Ensembl release of the graph if this is `np.inf`.

            Raises:
                ValueError: If not ``0 < l1 <= l2``.

            Returns:
                The range instance which iterates from `l1` to `l2`, including both.
            """
            if not 0 < l1 <= l2:
                raise ValueError

            if not np.isinf(l2) and isinstance(l2, int):
                right_l2 = l2
            elif not np.isinf(l2):
                raise ValueError(f"Unexpected error: {l2!r} should be either np.inf or integer.")
            else:
                right_l2 = max(self.graph["confident_for_release"])

            return range(l1, right_l2 + 1)

        # Use associated function to create the ranges of a node.
        the_node_type = self.nodes[the_id][DB.node_type_str]
        # If it is external 'database' is external database name (not node type). For others it is node type.

        if the_node_type == DB.nts_ensembl["gene"] and self.nodes[the_id]["Version"] in DB.alternative_versions:
            ass = self.graph["genome_assembly"]
            return {
                (DB.nts_assembly[ass]["gene"], ass, k)
                for i, j in self.get_active_ranges_of_id[the_id]
                for k in non_inf_range(i, j)
            }
        elif the_node_type == DB.nts_ensembl["gene"]:
            rd = self.combined_edges_genes[the_id]
        elif the_node_type in DB.nts_assembly_gene:
            rd = self.combined_edges_assembly_specific_genes[the_id]
        else:
            rd = self.combined_edges[the_id]

        return {(p, r, s) for p in rd for r in rd[p] for s in rd[p][r]}

    @staticmethod
    def compact_ranges(list_of_ranges: list[list]) -> list[list]:
        """Reduce the list of ranges into least possible number of ranges.

        O(n) time and space complexity: a forward in place compaction and copying back the elements,
        as then each inner step is O(1) (get/set instead of del).

        Args:
            list_of_ranges: List of increasing, inclusive ranges. The elements are positive integers. Note that there
                should be no overlapping elements.
                The output of :py:meth:`TheGraph.list_to_ranges` or
                :py:meth:`TheGraph._get_active_ranges_of_id` are a perfect input here.

        Returns:
            List of ranges as defined in :py:attr:`TheGraph.compact_ranges.list_of_ranges`.
        """
        next_index = 0  # Keeps track of the last used index in our result
        for index in range(len(list_of_ranges) - 1):
            if list_of_ranges[next_index][1] + 1 >= list_of_ranges[index + 1][0]:
                list_of_ranges[next_index][1] = list_of_ranges[index + 1][1]

            else:
                next_index += 1
                list_of_ranges[next_index] = list_of_ranges[index + 1]

        return list_of_ranges[: next_index + 1]

    @staticmethod
    def get_intersecting_ranges(lor1: list[list], lor2: list[list], compact: bool = True) -> list[list]:
        """As the name suggest, calculates the intersecting ranges of two list of ranges.

        Args:
            lor1: List of ranges as defined in :py:attr:`TheGraph.compact_ranges.list_of_ranges`.
            lor2: List of ranges as defined in :py:attr:`TheGraph.compact_ranges.list_of_ranges`.
            compact: If set `True`, returns the reduced list of ranges via
                :py:meth:`TheGraph.compact_ranges` method.

        Returns:
            List of ranges as defined in :py:attr:`TheGraph.compact_ranges.list_of_ranges`.
        """
        result = [
            [max(first[0], second[0]), min(first[1], second[1])]
            for first in lor1
            for second in lor2
            if max(first[0], second[0]) <= min(first[1], second[1])
        ]

        return TheGraph.compact_ranges(result) if compact else result

    @staticmethod
    def is_point_in_range(lor: list[list], p: int) -> bool:
        """Simple method to determine whether a given integer is covered by list of ranges.

        Args:
            lor: List of ranges as defined in :py:attr:`TheGraph.compact_ranges.list_of_ranges`.
            p: A positive integer.

        Returns:
            ``True`` if it is in the range, ``False`` otherwise.
        """
        for l1, l2 in lor:
            if l1 <= p <= l2:
                return True
        return False

    def get_two_nodes_coinciding_releases(self, id1: str, id2: str, compact: bool = True) -> list[list]:
        """Find the intersecting range of two nodes in the graph.

        Args:
            id1: First Query ID.
            id2: Second Query ID.
            compact: Parameter to pass into :py:meth:`TheGraph.get_intersecting_ranges` method.

        Returns:
            List of ranges as defined in :py:attr:`TheGraph.compact_ranges.list_of_ranges`.
        """
        r1 = self.get_active_ranges_of_id[id1]
        r2 = self.get_active_ranges_of_id[id2]

        r = TheGraph.get_intersecting_ranges(r1, r2, compact=compact)

        return r

    @cached_property
    def available_external_databases(self) -> set:
        """Find the available external databases found in the graph.

        Returns:
            Set of all external databases in the graph.
        """
        self.log.info(f"Cached properties being calculated: {'available_external_databases'}")
        return {
            j for i in self.nodes if self.nodes[i][DB.node_type_str] == DB.nts_external for j in self.combined_edges[i]
        }

    @cached_property
    def available_external_databases_assembly(self) -> dict[int, set]:
        """Find the available external databases found in the graph for each assembly separately.

        Returns:
            Dict of all external databases in the graph, assemblies as keys.
        """
        self.log.info(f"Cached properties being calculated: {'available_external_databases_assembly'}")
        result: dict[int, set] = {i: set() for i in self.available_genome_assemblies}
        for i in self.combined_edges:
            d = self.combined_edges[i]
            for i in d:
                for j in d[i]:
                    if i in self.available_external_databases:
                        result[j].add(i)
        return result

    @cached_property
    def external_database_connection_form(self) -> dict[str, str]:
        """Finds which form of Ensembl ID the external database identifiers are connected to.

        Each external database connects to a specific form (gene, transcript, translation) Ensembl ID. The relevant
        form is chosen by :py:class:`ExternalDatabases` class.

        Returns:
            Dictionary mapping external database name into the associated form (gene, transcript, translation).

        Raises:
            ValueError: If non-Ensembl node is connected.
        """
        self.log.info(f"Cached properties being calculated (for tests): {'external_database_connection_form'}")

        # Get the available databases to be matched
        aed = self.available_external_databases
        res = dict()

        for e in aed:  # For each database
            ra: list[str] = list()
            # Get the identifiers (node names) from all Ensembl releases and assemblies for a given external database.
            nodes = self.get_external_database_nodes(e)

            for node in nodes:
                for nei in self.neighbors(node):
                    # Look at the node type of each neighbour
                    nei_nts = self.nodes[nei][DB.node_type_str]
                    # Convert to nts_ensembl if it is nts_assembly, else keep at it is.
                    nei_nts = DB.nts_assembly_reverse.get(nei_nts, nei_nts)

                    if nei_nts not in DB.nts_ensembl_reverse:  # The result should be a nts_ensembl always.
                        raise ValueError(f"Not connected only to Ensembl ID. DB:{e}, From:{node}, To:{nei}, {nei_nts}")

                    nei_form = DB.nts_ensembl_reverse[nei_nts]  # Find the form

                    # Due to some weird annotations like: PRDX3P2 being in 'RefSeq_mRNA' and also 'HGNC Symbol'
                    # Not all the elements in 'ra' is the same. The rare ones are just exceptions, or misannotations.
                    # Counter resolves the issue although not ideal.

                    ra.append(nei_form)

            res[e] = Counter(ra).most_common(1)[0][0]
        return res

    @cached_property
    def available_genome_assemblies(self) -> set[int]:
        """Find the genome assemblies found in the graph by iterating through all nodes.

        Returns:
            Genome assemblies, which is also found in :py:attr:`DB.assembly_mysqlport_priority`.
        """
        self.log.info(f"Cached properties being calculated: {'available_genome_assemblies'}")

        output = set()  # Initialize a set and iterate through all 'combined_edges' dictionaries.
        for td in (self.combined_edges, self.combined_edges_genes, self.combined_edges_assembly_specific_genes):
            output.update({k for i in td for j in td[i] for k in td[i][j]})

        return output

    @cached_property
    def available_releases_given_database_assembly(self) -> dict[tuple[str, int], set]:
        """Todo."""
        self.log.info(f"Cached properties being calculated (for tests): {'available_releases_given_database_assembly'}")

        # Inline logic from _available_releases_given_database_assembly
        def _inline_available_releases(database_name: str, assembly: int) -> set[int]:
            """Possible Ensembl releases defined for a given database and assembly.

            The method uses `node_trios` unnecessarily method, which consumes a lot of memory and hinders high
            computational efficiency. However, this method is used only in testing purposes, when the speed and memory
            is not of a concern.

            It is important to note that not all databases are defined in all Ensembl release. To see for more
            information, have a look at the :py:class:`ExternalDatabases`.

            Args:
                database_name: External database or node type (except `external`) it should be one of the item from
                    :py:meth:`TheGraph.available_external_databases`. The method also works with `node types`
                    (except `external`), since they are also defined in `node_trios`. Important to note that the
                    `node type` for Ensembl should follow :py:attr:`DB.nts_assembly` or :py:attr:`DB.nts_base_ensembl`.
                assembly: Assembly, it should be one of the item from
                    :py:meth:`TheGraph.available_genome_assemblies`.

            Returns:
                Available Ensembl releases as set of integers.
            """
            return {
                j3
                for key in self.node_trios
                for j1, j2, j3 in self.node_trios[key]
                if j1 == database_name and j2 == assembly
            }

        r = dict()
        for assembly, database_names in self.available_external_databases_assembly.items():
            for database_name in database_names:
                r[(database_name, assembly)] = _inline_available_releases(
                    assembly=assembly, database_name=database_name
                )

        for assembly, _inner_dict in DB.nts_assembly.items():
            for _, database_name in _inner_dict.items():
                r[(database_name, assembly)] = _inline_available_releases(
                    assembly=assembly, database_name=database_name
                )

        for database_name in DB.nts_base_ensembl.values():
            r[(database_name, DB.main_assembly)] = _inline_available_releases(
                assembly=DB.main_assembly, database_name=database_name
            )

        available_release = {j for i in r.values() for j in i}
        for database_name in DB.nts_ensembl.values():
            r[(database_name, DB.main_assembly)] = available_release

        return r

    def get_id_list(self, database: str, assembly: int, release: int) -> list[str]:
        """Given a trio (database, assembly, release), generates a list of node names (identifiers).

        Similar to :py:meth:`TheGraph.available_releases_given_database_assembly`, the method uses
        `node_trios` unnecessarily method, which consumes a lot of memory and hinders high
        computational efficiency. However, this method is used only in testing purposes, when the speed and memory is
        not of a concern.

        It is imporant to note that the Ensembl IDs with versions :py:attr:`DB.alternative_versions` will be also
        returned if the database is the Ensembl gene with the main assembly, and the assembly is the main assembly.

        Args:
            database: External database name if the `node type` is `external`, else `node type`. The `node type`
                for Ensembl should follow :py:attr:`DB.nts_assembly` or :py:attr:`DB.nts_base_ensembl`.
            assembly: Assembly, it should be one of the item from
                :py:meth:`TheGraph.available_genome_assemblies`.
            release: Requested Ensembl releases as an integer.

        Returns:
            Node names (identifiers) list.
        """
        if database in DB.nts_ensembl_reverse.keys():
            form = DB.nts_ensembl_reverse[database]
            the_key = (f"assembly_{assembly}_ensembl_{form}", assembly, release)
        else:
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

    def get_external_database_nodes(self, database_name: str) -> set[str]:
        """For a given external database, returns set of all node names defined at least once in that database.

        Args:
            database_name: External database, it should be one of the item from
                :py:meth:`TheGraph.available_external_databases`.

        Returns:
            Node names (identifiers) set.
        """
        return {
            i
            for i in self.nodes
            if self.nodes[i][DB.node_type_str] == DB.nts_external and database_name in self.combined_edges[i]
        }
