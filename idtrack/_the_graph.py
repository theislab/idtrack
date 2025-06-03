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
    def lower_chars_graph(self) -> dict[str, str]:
        """Map lowercase node identifiers to their canonical graph node names.

        The cached mapping enables **case-insensitive** queries against the graph by translating a
        lowercase version of every node into the exact identifier stored in :py:attr:`self.nodes`.
        ID-resolution helpers such as :py:meth:`node_name_alternatives` rely on this cache to
        recover the intended node even when callers supply mixed-case or lowercase strings.

        During construction the method iterates once over *all* nodes, lowers each identifier, and
        asserts that no two distinct nodes collide after lower-casing.  The result is memoised via
        :py:data:`functools.cached_property`, so subsequent accesses are O(1).

        Returns:
            dict[str, str]: ``{lowercase_id: original_id}`` giving a one-to-one mapping from
                lowercase node identifiers to the exact strings used in the graph.

        Raises:
            ValueError: If two or more nodes become identical after converting to lowercase,
                indicating ambiguous casing in the underlying graph.
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
        """Resolve a raw query identifier to the *exact* graph node label that ID-Track expects.

        The routine shields downstream path-finding code from the myriad ways users may spell or format
        biological identifiers.  It walks through a well-defined priority list—direct lookup, case-blind
        match, version-suffix trimming, and dash/underscore substitutions—before finally retrying the whole
        sequence with the ``synonym:`` prefix used by :py:data:`~idtrack._db.DB.synonym_id_nodes_prefix`.
        This makes interactive exploration tolerant to typos such as lower-case gene symbols
        (``actb`` → ``ACTB``) or versioned Ensembl IDs written with underscores
        (``ENSG00000123456_2`` → ``ENSG00000123456.2``).

        Args:
            identifier (str): Raw identifier supplied by the caller.  May be an Ensembl ID, external database
                key, or any variant handled by the heuristics described above.

        Returns:
            tuple[Optional[str], bool]:
                * The canonical node label **or** ``None`` when no match is possible.
                * ``True`` when *identifier* had to be modified (case change, suffix strip, etc.); ``False``
                    when an exact graph hit was found.

        Notes:
            Internally this is a thin wrapper that delegates the heavy lifting to the private
            :py:meth:`_node_name_alternatives` helper, then retries once with the synonym prefix if the first
            pass fails.  The helper itself is further decomposed into specialised sub-functions—see their
            individual docstrings for details.
        """

        def _node_name_alternatives(the_id: str) -> tuple[Optional[str], bool]:
            """Apply the heuristic cascade to *one* identifier candidate.

            The helper encapsulates the common logic needed by both the public wrapper and the synonym-prefixed
            fallback.  Its four-step search order is:

            1. **Exact** match - return immediately if *the_id* is already in :py:attr:`self.nodes`.
            2. **Case-blind** match - consult :py:attr:`~TheGraph.lower_chars_graph`.
            3. **Suffix-trim** - strip a trailing version segment introduced by ``'.'``, ``'_'`` or ``'-'`` and
                test both the trimmed and its lower-case form.
            4. **Dash/underscore permutations** - substitute every combination of ``'-'`` ⇄ ``'_'`` and repeat
                steps 2-3 for each synthetic candidate.

            Args:
                the_id (str): One potential spelling of the identifier.

            Returns:
                tuple[Optional[str], bool]: Same semantics as the public method.
            """

            def compare_lowers(id_to_find: str) -> tuple[Optional[str], bool]:
                """Perform a constant-time, case-insensitive lookup via :py:attr:`self.lower_chars_graph`.

                Args:
                    id_to_find (str): Candidate identifier.

                Returns:
                    tuple[Optional[str], bool]: ``(canonical_id, True)`` when a lower-case hit is found;
                        ``(None, False)`` otherwise.
                """
                lower_id_find = id_to_find.lower()

                if lower_id_find in self.lower_chars_graph:
                    return self.lower_chars_graph[lower_id_find], True

                else:
                    return None, False  # If cannot return anything, just return None (unfound).

            def check_variation(id_str: str) -> tuple[Optional[str], bool]:
                r"""Strip terminal *version-like* segments and retry the exact/CI match sequence.

                The regular expression ``r"^(.+)(_|-|\\.)[0-9]+$"`` removes anything that looks like
                ``<core><sep><digits>``, where *sep* is ``'.'``, ``'_'`` or ``'-'``.  Both the trimmed string and its
                lower-case counterpart are tested against the graph.

                Args:
                    id_str (str): Identifier to normalise.

                Returns:
                    tuple[Optional[str], bool]: Match result using the same convention as :py:meth:`compare_lowers`.
                """
                # First, try to find the lower character version of the querry.
                lower_id, is_lower_found = compare_lowers(id_str)
                if is_lower_found:  # If something is found, just return it.
                    return lower_id, True

                # Them, try to match with the regular-expression pattern. The pattern is basically to remove any
                # flanking numbers (possibly versions) separated with following characters '-', '_', or '.'. In order
                # to match with the regex patter, the query has to have these separators, but the subsequent integer
                # is optional. Note that the last separator is of interest only.
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

            def possible_alternatives(the_id_pa: str) -> list[str]:
                """Enumerate every dash/underscore permutation of *the_id_pa*.

                Each ``'-'`` or ``'_'`` is treated as a binary position; the function returns ``2ⁿ`` synthetic IDs
                where *n* is the number of such positions.  Example: ``"AC-TB_1"`` produces four variants
                (``"AC_TB_1"``, ``"AC-TB-1"``, ``"AC-TB_1"``, ``"AC_TB-1"``).

                Args:
                    the_id_pa (str): Prototype identifier.

                Returns:
                    list[str]: All generated permutations (duplicates are preserved for downstream simplicity).
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
        """Return inclusive Ensembl-release intervals in which every node in the graph is biologically active.

        The convenience wrapper iterates over all nodes currently stored in this
        :py:class:`idtrack.the_graph.TheGraph` instance and delegates the heavy
        lifting to :py:meth:`_get_active_ranges_of_id`.  The latter performs node-type-specific
        logic (backbone vs. assembly-specific) to determine *contiguous* release
        windows—at no point does this method examine which genome assembly the
        release originated from, because for downstream tasks (lifecycle analysis,
        deprecation reports, etc.) only the presence/absence across release numbers
        matters.

        Returns:
            dict[str, list[list[int]]]: Mapping ``{node_id: [[start_rel, end_rel], ...]}``
                where every inner two-element list is an *inclusive* range.  Ranges
                are sorted in ascending order and guaranteed not to overlap.
        """
        self.log.info(f"Cached properties being calculated: {'get_active_ranges_of_id'}")
        return {n: self._get_active_ranges_of_id(n) for n in self.nodes}

    def _get_active_ranges_of_id(self, input_id: str) -> list[list]:
        """Compute Ensembl-release ranges for a single identifier, choosing logic by node type.

        This private helper inspects the ``input_id`` and dispatches to an internal
        routine tailored to the node's role in the graph:

        * :py:meth:`_get_active_ranges_of_id_backbone` - deals with *backbone* nodes
            that form the primary versioned lineage.
        * :py:meth:`_get_active_ranges_of_id_nonbackbone` - handles assembly-specific
            or auxiliary identifiers recorded in one of the *combined-edges* lookup tables.

        Args:
            input_id (str): Identifier whose life-span across Ensembl releases is
                requested.  Must exist in :py:attr:`~TheGraph.nodes`.

        Returns:
            list[list[int]]: Ordered, non-overlapping ``[[start_rel, end_rel], …]``
                where both ends are inclusive.
        """

        def _get_active_ranges_of_id_nonbackbone(the_id: str) -> list[list[int]]:
            """Determine release intervals for non-backbone (assembly-specific) nodes.

            For aliases and assembly-level gene identifiers the graph stores life-span
            information inside one of two *combined-edge* dictionaries:

            * :py:data:`self.combined_edges_assembly_specific_genes` for per-assembly genes.
            * :py:data:`self.combined_edges` for all other non-backbone nodes.

            The routine flattens every inner ``set`` of release numbers, sorts them,
            and converts the resulting sequence into contiguous, inclusive ranges via
            :py:meth:`TheGraph.list_to_ranges`.

            Args:
                the_id (str): Identifier to resolve (must be non-backbone).

            Returns:
                list[list[int]]: Ascending, inclusive release intervals.
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

        def _get_active_ranges_of_id_backbone(the_id: str) -> list[list[int]]:
            """Compute release ranges for backbone nodes via explicit edge traversal.

            A backbone node's activity starts where an *in-edge* appears and stops
            immediately *before* its next *out-edge* (branching or version bump).
            The function therefore:

            1. Collects all releases with an incoming edge
               (``get_next_edge_releases(reverse=False)``).
            2. Collects all releases with an outgoing edge
               (``reverse=True``).
            3. Merges and sorts the two sets, then scans them in ascending order,
               toggling an *active* flag to build start/stop pairs.

            Args:
                the_id (str): Backbone identifier whose version chain is analysed.

            Returns:
                list[list[int]]: One or more inclusive ``[start, end]`` intervals.

            Raises:
                ValueError: If the identifier has neither in-edges nor out-edges,
                    or if graph invariants (e.g., first release must be an in-edge)
                    are violated, implying a malformed backbone.
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
            _narrowed = [narrowed[i : i + 2] for i in range(0, len(narrowed), 2)]
            return _narrowed

        # Use associated function to create the ranges of a node.
        if self.nodes[input_id][DB.node_type_str] == DB.external_search_settings["nts_backbone"]:
            return _get_active_ranges_of_id_backbone(input_id)
        else:
            return _get_active_ranges_of_id_nonbackbone(input_id)

    def get_active_ranges_of_id_ensembl_all_inclusive(self, the_id: str) -> list[list]:
        """Return the inclusive Ensembl-release ranges during which *the_id* is active across **all** assemblies.

        This helper generalises :py:meth:`~TheGraph.get_active_ranges_of_id`, which only reports activity on the
        graph's **main assembly**, by folding in evidence from every other assembly represented in
        :py:data:`~TheGraph.combined_edges_genes`.  The resulting timeline therefore reflects *all* times at which
        the identifier (or any assembly-specific sibling) existed in Ensembl—crucial when downstream analyses
        must ignore assembly boundaries, e.g. when tracking identifier synonymy across genome builds.  After
        merging, the routine validates that the main-assembly slice remains consistent with the authoritative
        backbone cache and aborts with a detailed error if divergence is detected.

        Args:
            the_id (str): Ensembl gene identifier—either backbone (``ENSG…``) or assembly-qualified
                (``assembly_<code>_ensembl_gene``).  The node's :py:data:`DB.node_type_str` **must** be one of
                ``DB.nts_ensembl["gene"]`` or the set in :py:data:`DB.nts_assembly_gene`.

        Returns:
            list[list[int]]: A list of ``[start, end]`` pairs (inclusive, sorted, non-overlapping) covering every
            Ensembl release in which *the_id* was present on **any** assembly.

        Raises:
            ValueError: If (1) activity inferred from :py:data:`~TheGraph.combined_edges_genes` disagrees with
                :py:meth:`~TheGraph.get_active_ranges_of_id` for the main assembly, or (2) *the_id* is not a
                recognised Ensembl-gene node type.
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

    def get_next_edge_releases(self, from_id: str, reverse: bool) -> list[int]:
        """List the Ensembl releases reachable by the **next** (or **previous**) edges from *from_id*.

        The method scans the immediate neighbourhood of a **backbone gene node** and extracts the release
        numbers that mark either the next chronological transition (*reverse* = ``False``) or the previous
        one (*reverse* = ``True``).  It respects graph directionality, skips non-backbone connections,
        collapses duplicate multi-edges, and treats infinite self-loops as “still active” when stepping
        forward in time.  The result is a de-duplicated, easy-to-use list that higher-level path-finding
        algorithms can feed directly into release-oriented traversals.

        Args:
            from_id (str): Ensembl gene identifier that must belong to the backbone
                (``DB.external_search_settings["nts_backbone"]``).
            reverse (bool): If ``False`` return *forward* (old → new) transition releases; if ``True`` return
                *backward* (new → old) releases.

        Returns:
            list[int]: Sorted list of unique Ensembl release numbers adjacent to *from_id* in the chosen temporal
                direction.

        Raises:
            ValueError: If *from_id* is **not** a backbone node—i.e. its ``node_type`` does not match
                :py:data:`DB.external_search_settings["nts_backbone"]`.
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
        """Return the Ensembl-release intervals during which a *base* gene identifier is active.

        The routine unifies child-level history into an easy-to-query representation.  A *base* Ensembl ID
        (e.g. ``ENSG00000123456``) has one or more *versioned* descendants
        (``ENSG00000123456.1``, ``ENSG00000123456.2``, …) whose lifetimes can never overlap.
        By walking the immediate neighbours of *base_id* and unioning every child's
        ``get_active_ranges_of_id_ensembl_all_inclusive`` result, the method derives *exactly* the
        releases in which **any** descendant existed.  This summary read-out is used by higher-level
        diagnostics (for example, *range-overlap* sanity checks) and by algorithms that need to
        reason about the birth and retirement of genes at the stable-ID level.

        Args:
            base_id (str): Stable Ensembl gene identifier **without** version suffix.  The node must have
                ``node_type == DB.nts_base_ensembl["gene"]`` inside the graph.

        Returns:
            list[list[int]]: Sorted, non-overlapping ``[start, end]`` slices **inclusive** at both ends.
                ``end`` may be ``np.inf`` when the gene is still present in the most recent release.
        """
        associated_ids = self.neighbors(base_id)
        all_ranges = [r for ai in associated_ids for r in self.get_active_ranges_of_id_ensembl_all_inclusive(ai)]
        return self.list_to_ranges(self.ranges_to_list(all_ranges))

    @staticmethod
    def list_to_ranges(lst: list[int]) -> list[list]:
        """Compact a sorted list of releases into minimal inclusive ranges.

        The helper converts monotonically increasing, duplicate-free release numbers into a
        *run-length* representation (e.g. ``[1, 2, 3, 5] → [[1, 3], [5, 5]]``).  It is the logical
        inverse of :py:meth:`TheGraph.ranges_to_list` and is frequently used to post-process the raw
        release sets collected from edge metadata.

        Args:
            lst (list[int]): Releases **strictly increasing** with no repetitions.  Supplying an
                unsorted or duplicate-containing list leads to undefined behaviour.

        Returns:
            list[list[int]]: Non-overlapping ``[start, end]`` intervals covering exactly the input
                elements.  Each inner list is inclusive; singleton releases become ``[r, r]``.
        """
        res = list()
        for _, a in itertools.groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
            b = list(a)
            res.append([b[0][1], b[-1][1]])
        return res

    def ranges_to_list(self, lor: list[list]) -> list[int]:
        """Explode inclusive ranges back into a sorted list of releases.

        This is the inverse of :py:meth:`TheGraph.list_to_ranges`.  Each ``[start, end]`` slice is
        expanded **inclusive** of both boundaries; if *end* is ``np.inf`` the interval is closed with
        ``max(self.graph["confident_for_release"])`` so that downstream numeric operations continue to
        work on finite integers.  The union of all expanded ranges is returned in ascending order
        without duplicates.

        Args:
            lor (list[list[int | float]]): List of inclusive, non-overlapping ``[start, end]`` pairs.
                ``start`` must be ``> 0``; ``end`` may be ``np.inf`` to denote open-ended activity.

        Returns:
            list[int]: Strictly increasing sequence of releases represented by *lor*.
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
        """Return a full **node → trio-set** cache.

        Builds the complete mapping *once* and stores it as a :py:data:`functools.cached_property`.
        The mapping is memory-heavy but accelerates downstream helpers that repeatedly need the
        ``(<database>, <assembly>, <release>)`` origin of many nodes.

        Returns:
            dict[str, set[tuple[str, int, int]]]:  Node identifier → the set of unique
                ``(database, assembly, release)`` combinations in which that node is active.

        Notes:
            *The builder simply iterates ``self.nodes`` and delegates the per-node logic to
            :py:meth:`_node_trios`.  Expect a multi-second start-up on large graphs.*
        """
        self.log.info(f"Cached properties being calculated: {'node_trios'}")
        return {n: self._node_trios(n) for n in self.nodes}

    def _node_trios(self, the_id: str) -> set[tuple]:
        """Compute all origin *trios* for a single node.

        The routine identifies the node-type, chooses the appropriate *combined_edges* cache, and
        expands any Ensembl *release ranges* so that every individual release is represented.
        Alternative-assembly backbone genes and assembly-specific genes receive special handling to
        ensure the correct database label is recorded.

        Args:
            the_id (str): Canonical node name used inside the graph.

        Returns:
            set[tuple[str, int, int]]:  Unique triples ``(<database>, <assembly>, <release>)`` describing
                every context in which *the_id* occurs.
        """

        def _non_inf_range(l1: int, l2: Union[float, int]) -> range:
            """Create an inclusive ``range`` while resolving ``np.inf`` upper bounds.

            Args:
                l1 (int): Lower bound; must be **> 0**.
                l2 (Union[float, int]): Upper bound. If ``np.inf``, it is replaced by the highest Ensembl
                    release stored in ``self.graph["confident_for_release"]``.

            Returns:
                range: A Python :py:class:`range` covering **all** releases from *l1* to *l2* (after
                    normalisation), inclusive.

            Raises:
                ValueError: If *l1* ≤ 0, *l1* > *l2*, or *l2* is neither an integer nor ``np.inf``.
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
                for k in _non_inf_range(i, j)
            }
        elif the_node_type == DB.nts_ensembl["gene"]:
            rd = self.combined_edges_genes[the_id]
        elif the_node_type in DB.nts_assembly_gene:
            rd = self.combined_edges_assembly_specific_genes[the_id]
        else:
            rd = self.combined_edges[the_id]

        return {(p, r, s) for p in rd for r in rd[p] for s in rd[p][r]}

    @staticmethod
    def compact_ranges(list_of_ranges: list[list[int]]) -> list[list[int]]:
        """Collapse adjacent or touching integer ranges into the smallest possible set.

        In the IDTrack graph every Ensembl identifier is active for one or more contiguous release intervals.
        Storing those intervals as ``[[start, end], …]`` is convenient but can become redundant when consecutive
        ranges abut each other.  *compact_ranges* performs an in-place, **O(n)** forward sweep that merges any pair
        of ranges where the gap between ``end`` of the first and ``start`` of the next is ≤ 1, returning a new
        list that covers the exact same discrete releases with the fewest possible intervals.  The helper is a
        cornerstone for many caching utilities (e.g. :py:meth:`TheGraph.get_active_ranges_of_id`) and therefore
        optimised for speed and minimal allocations.

        Args:
            list_of_ranges (list[list[int]]): Sorted, non-overlapping, inclusive ranges in the form
                ``[[start, end], …]``.  All numbers must be positive integers and ``start ≤ end`` for every range.

        Returns:
            list[list[int]]: A new list containing the minimal, non-overlapping, inclusive ranges that exactly
                cover the union of *list_of_ranges*.
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
    def get_intersecting_ranges(lor1: list[list[int]], lor2: list[list[int]], compact: bool = True) -> list[list[int]]:
        """Return the set of releases common to **both** input range lists.

        The routine computes the pairwise intersection between every range in *lor1* and every range in *lor2*,
        yielding a list of ranges where the two original lists overlap.  Optionally the result may be passed
        through :py:meth:`TheGraph.compact_ranges` to merge adjacent slices and guarantee a minimal
        representation.  Because the helper is frequently used inside path-finding algorithms it trades clarity
        for raw performance and therefore assumes both inputs are already sorted, non-overlapping, and inclusive
        as produced elsewhere in the library.

        Args:
            lor1 (list[list[int]]): First list of inclusive, ascending, non-overlapping ranges.
            lor2 (list[list[int]]): Second list of ranges with the same invariants as *lor1*.
            compact (bool): When ``True`` (default) the raw intersections are passed to
                :py:meth:`TheGraph.compact_ranges` before being returned.

        Returns:
            list[list[int]]: Inclusive integer ranges where *lor1* and *lor2* overlap.  The list is empty when
                no overlap exists.
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
        """Check whether a single integer lies inside any range in *lor*.

        The helper performs a linear scan over *lor* (assumed sorted and non-overlapping) and returns as soon as
        *p* falls between a ``[start, end]`` pair.  It is intentionally lightweight because it is called inside
        tight loops that filter large identifier sets by Ensembl release.

        Args:
            lor (list[list[int]]): Inclusive, ascending, non-overlapping ranges against which *p* is tested.
            p (int): The release number to evaluate.

        Returns:
            bool: ``True`` when *p* is covered by at least one range in *lor*; ``False`` otherwise.
        """
        for l1, l2 in lor:
            if l1 <= p <= l2:
                return True
        return False

    def get_two_nodes_coinciding_releases(self, id1: str, id2: str, compact: bool = True) -> list[list]:
        """Determine releases in which **both** graph nodes are simultaneously active.

        Graph nodes (Ensembl genes, transcripts, proteins, or external IDs) exist only for defined release
        intervals.  When integrating annotations it is often necessary to know the time span where two nodes
        co-exist—for example, when building an orthogonal mapping table or validating edge chronology.
        The method retrieves each node's active ranges via :py:meth:`TheGraph.get_active_ranges_of_id`, computes
        their intersection with :py:meth:`TheGraph.get_intersecting_ranges`, and optionally compacts the result.
        The returned list therefore represents every Ensembl release in which *id1* **and** *id2* are valid
        simultaneously.

        Args:
            id1 (str): Identifier of the first node (must exist in ``self.nodes``).
            id2 (str): Identifier of the second node (must exist in ``self.nodes``).
            compact (bool): Forwarded to :py:meth:`TheGraph.get_intersecting_ranges`.  When ``True``
                (default) the final ranges are minimised; when ``False`` the raw intersections are returned.

        Returns:
            list[list[int]]: Inclusive release intervals ``[[start, end], …]`` where *id1* and *id2* overlap.
            The list is empty if the nodes never co-occur.
        """
        r1 = self.get_active_ranges_of_id[id1]
        r2 = self.get_active_ranges_of_id[id2]

        r = TheGraph.get_intersecting_ranges(r1, r2, compact=compact)

        return r

    @cached_property
    def available_external_databases(self) -> set[str]:
        """Return the set of external databases represented in the graph.

        This helper inspects every node whose *node-type* flag matches
        :py:data:`idtrack._db.DB.nts_external` and records the *database name* attached
        to the outbound edges.  The resulting set is cached so that downstream
        routines—such as validating user-supplied database names or determining which
        third-party resources must be fetched—can query the information in **O(1)**
        time instead of re-scanning the graph.

        Returns:
            set[str]: Unique names of all third-party (non-Ensembl) databases present
                in the current :py:class:`~TheGraph` instance.
        """
        self.log.info(f"Cached properties being calculated: {'available_external_databases'}")
        return {
            j for i in self.nodes if self.nodes[i][DB.node_type_str] == DB.nts_external for j in self.combined_edges[i]
        }

    @cached_property
    def available_external_databases_assembly(self) -> dict[int, set[str]]:
        """Return external databases available for each genome assembly.

        For every assembly identifier in :py:attr:`~TheGraph.available_genome_assemblies`,
        this method gathers the subset of external databases that are connected—directly
        or indirectly—to nodes annotated with that assembly.  The per-assembly view is
        vital when users need to restrict conversions to genomes with consistent
        annotation coverage (e.g., choosing GRCh38-only resources for a human data
        set).

        Returns:
            dict[int, set[str]]: Mapping from *assembly number* (for example ``37`` or
                ``38``) to the set of external databases that have at least one entry
                linked to that assembly.
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
        """Infer which Ensembl identifier form each external database connects to.

        External databases link to exactly one “form” of Ensembl identifier—gene,
        transcript, or translation—determined upstream by :py:class:`idtrack._external_databases.ExternalDatabases`.
        The method walks the neighborhood of every external-database node, tallies the *node-type* of
        its Ensembl neighbours, and assigns the majority form.  A mis-annotation that
        connects an external node directly to a non-Ensembl node is interpreted as a
        schema violation and aborts with :py:class:`ValueError`.

        Returns:
            dict[str, str]: Dictionary whose keys are external-database names and
                whose values are one of ``"gene"``, ``"transcript"``, or
                ``"translation"``, indicating the form of Ensembl ID to which the
                database links.

        Raises:
            ValueError: If any external-database node is found connected to a node
                that is *not* an Ensembl identifier, indicating an inconsistent graph state.
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
        """Return the set of genome assemblies represented in the current graph.

        The helper scans every identifier edge table cached on the instance (e.g. ``combined_edges``,
        ``combined_edges_genes``, ``combined_edges_assembly_specific_genes``) and extracts the
        assembly component of each edge key.  It therefore answers the question *“Which genome builds
        does this graph actually know about?”*  Several public utilities depend on this information
        when validating user-supplied assembly arguments or iterating across assemblies in reproducible
        order (see :py:attr:`DB.assembly_mysqlport_priority`).

        Returns:
            set[int]: Unique genome assembly identifiers (e.g. ``104``, ``108``) present anywhere in the graph.
        """
        self.log.info(f"Cached properties being calculated: {'available_genome_assemblies'}")

        output = set()  # Initialize a set and iterate through all 'combined_edges' dictionaries.
        for td in (self.combined_edges, self.combined_edges_genes, self.combined_edges_assembly_specific_genes):
            output.update({k for i in td for j in td[i] for k in td[i][j]})

        return output

    @cached_property
    def available_releases_given_database_assembly(self) -> dict[tuple[str, int], set]:
        """Map *(database, assembly)* pairs to the Ensembl releases in which they occur.

        This expensive, cached property lets callers quickly answer *“Which Ensembl releases contain
        at least one node from database **D** on assembly **A**?”*  Internally it delegates the
        per-pair work to the nested
        :py:meth:`available_releases_given_database_assembly._inline_available_releases`
        helper, then augments the mapping with additional information gleaned from several
        :py:data:`idtrack.DB` look-ups (e.g. :py:data:`DB.nts_assembly`,
        :py:data:`DB.nts_base_ensembl`, :py:data:`DB.nts_ensembl`).  Although heavy, the routine is
        indispensable for test suites and diagnostic notebooks that must reason about historical
        coverage across many releases.

        Returns:
            dict[tuple[str, int], set[int]]: A dictionary whose keys are *(database_name, assembly)*
                tuples and whose values are the sets of Ensembl release numbers in which that pair is
                represented.
        """
        self.log.info(f"Cached properties being calculated (for tests): {'available_releases_given_database_assembly'}")

        # Inline logic from _available_releases_given_database_assembly
        def _inline_available_releases(database_name: str, assembly: int) -> set[int]:
            """Return Ensembl releases containing *database_name* for *assembly*.

            The inline helper loops over :py:data:`TheGraph.node_trios`, filtering for rows whose
            *(database, assembly)* fields match the inputs and collecting the corresponding release
            column.  It is purposefully *not* optimised for speed or memory because it is used only in
            low-frequency contexts such as unit tests or exploratory diagnostics.

            Args:
                database_name (str): External database identifier or, for Ensembl data, a node-type
                    string accepted by :py:meth:`TheGraph.available_external_databases`.  Ensembl
                    node-type strings must follow the conventions in
                    :py:data:`DB.nts_assembly` or :py:data:`DB.nts_base_ensembl`.
                assembly (int): Genome assembly to query.  Must be present in
                    :py:meth:`TheGraph.available_genome_assemblies`.

            Returns:
                set[int]: Releases in which *database_name* appears on *assembly*.
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
        """Return node identifiers for a specific (database, assembly, release) slice of the multigraph.

        This helper exists primarily for *unit-testing and exploratory analysis*.  Internally, the graph stores
        node metadata in the memory-intensive :py:attr:`~TheGraph.node_trios` cache, keyed by a triple
        ``(database_or_node_type, assembly, release)``.  :py:meth:`get_id_list` hides that complexity, walking the
        full node set and extracting only those identifiers whose tuple key matches the requested slice.  Because
        the traversal touches every node, the method is slow and scales poorly compared with the vectorised access
        paths used in production code.  It is therefore **not** called in performance-critical workflows; its main
        purpose is to generate deterministic ground-truth lists that test-suites can compare against.

        The method also reproduces legacy Ensembl behaviour: when *database* resolves to the canonical Ensembl gene
        node type on the primary assembly, identifiers whose ``Version`` attribute is one of
        :py:data:`idtrack._db.DB.alternative_versions` are still included, ensuring that versioned and unversioned
        IDs appear together—exactly as they do in public Ensembl MySQL dumps.

        Args:
            database (str): External database name for *external* nodes (e.g. ``"uniprot"``, ``"refseq"``) **or** an
                Ensembl node-type label such as ``"gene"``, ``"transcript"``, or ``"translation"``.  Ensembl labels must
                match the keys defined in :py:data:`idtrack._db.DB.nts_ensembl`.
            assembly (int): Genome assembly identifier (e.g. ``1`` for GRCh38) that must be present in
                :py:meth:`~TheGraph.available_genome_assemblies`.
            release (int): Ensembl release number (e.g. ``111``) corresponding to the graph snapshot of interest.

        Returns:
            list[str]: A list of **unique** node names (identifiers) in insertion order that belong to the requested
            ``(database, assembly, release)`` tuple.  The list may include versioned Ensembl genes as noted above.

        Notes:
            The helper performs a linear scan over :py:data:`networkx.MultiDiGraph.nodes`, so its runtime is
            ``O(|V|)`` and memory footprint equals that of :py:attr:`~TheGraph.node_trios`.  Prefer dedicated graph
            queries for production workloads and reserve this method for tests or ad-hoc inspection.
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
        """Collect identifiers that appear at least once in the specified external database.

        The graph stores one *node* per identifier and attaches metadata—such as its origin
        database—to each node via ``self.nodes[node_name]``.  This helper filters that dictionary,
        returning every node whose metadata marks it as an *external* identifier belonging to
        *database_name*.  The result is often fed into downstream integrity checks or exported so
        that analysts can cross-reference original accession lists.

        Args:
            database_name (str): Name of the external resource (e.g. ``"UniProtKB"``).  Must be one of
                the values returned by :py:meth:`TheGraph.available_external_databases`.

        Returns:
            set[str]: All unique node names (accessions) associated with *database_name*.
        """
        return {
            i
            for i in self.nodes
            if self.nodes[i][DB.node_type_str] == DB.nts_external and database_name in self.combined_edges[i]
        }
