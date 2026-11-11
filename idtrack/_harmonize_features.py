#!/usr/bin/env python3

import gc
import logging
import os
import pickle  # noqa: S403
import re
from functools import cached_property
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import tqdm

import idtrack
from idtrack import DB


def _is_text(series: pd.Series) -> bool:
    """Return whether *series* holds text (``object`` dtype, or a pandas string dtype).

    Args:
        series: A column.

    Returns:
        bool: ``True`` for text.
    """
    return pd.api.types.is_object_dtype(series.dtype) or isinstance(series.dtype, pd.StringDtype)


def _holding_missing(series: pd.Series) -> pd.Series:
    """Return *series* in a dtype that can hold missing values of its kind.

    Integers become the nullable ``Int``/``UInt`` dtype of the same width and booleans ``boolean`` (missing value
    ``pd.NA``); every other dtype (float, text, categorical, datetime, nullable dtypes) already has a missing value
    and is returned unchanged.

    Args:
        series: A column.

    Returns:
        pandas.Series: The column, in a dtype with a missing value.
    """
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "iu":
        return series.astype(pd.api.types.pandas_dtype(f"{'U' if dtype.kind == 'u' else ''}Int{8 * dtype.itemsize}"))
    if isinstance(dtype, np.dtype) and dtype.kind == "b":
        return series.astype("boolean")
    return series


def _missing_like(template: pd.Series, index: pd.Index) -> pd.Series:
    """Return a column of missing values of the kind of *template* (see :py:func:`_holding_missing`).

    Args:
        template: A column of the same name in another dataset.
        index: Index of the new column.

    Returns:
        pandas.Series: All values missing (``NaN``, ``pd.NA``, ``NaT`` or a missing category).
    """
    return pd.Series(index=index, dtype=_holding_missing(template.iloc[:0]).dtype)


class HarmonizeFeatures:
    """Harmonize gene/feature identifiers across multiple single-cell expression datasets.

    This manager streamlines the otherwise error-prone task of bringing heterogeneous gene identifiers
    (Ensembl IDs, gene symbols, etc.) into a single, version-controlled namespace before integrated
    downstream analysis.  Under the hood it leverages :py:class:`idtrack.api.API` to resolve identifier
    mappings through a pre-computed Ensembl graph, handles one-to-many and one-to-zero conversions, logs
    any ambiguous or inconsistent matches, and finally produces harmonised
    :py:class:`anndata.AnnData` objects ready for comparative or joint analysis.

    The public workflow is intentionally simple:

    * :py:meth:`feature_harmonizer` — convert a single dataset and return the filtered ``AnnData`` plus
      before/after feature counts.
    * :py:meth:`unify_multiple_anndatas` — apply harmonisation across *all* supplied datasets and return
      an integrated object.
    * :py:meth:`get_idtrack_matchings_for_all_datasets` — inspect the raw IDTrack matchings used.

    Instances keep several diagnostic attributes (e.g. ``removed_conversion_failed_identifiers``,
    ``multiple_ensembl_dict``) so that users can audit every decision that removed or altered a feature.

    Args:
        project_name (str): Human-readable label used in log messages and derived output file names.
        data_h5ad_dict (dict[str, str]): Mapping *dataset_alias → absolute .h5ad path* of the source
            single-cell expression matrices.
        project_local_repository (str): Writable directory where harmonised outputs, logs, and temporary
            artefacts will be stored.
        idtrack_local_repository (str): Local clone or cache directory understood by
            :py:class:`idtrack.api.API`; used to read the pre-built identifier graph.
        target_ensembl_release (int): Ensembl release that all identifiers will be converted *to*.  Must
            be ≤ *graph_last_ensembl_release*.
        final_database (Optional[str]): Canonical namespace kept after conversion (e.g. ``"HGNC Symbol"``).
            Defaults to the organism's first symbol database (:py:attr:`idtrack.DB.symbol_databases`: HGNC Symbol
            for human, MGI Symbol for mouse, VGNC Symbol for pig and the macaques).
        organism_name (Optional[str]): Organism, as an Ensembl species name (e.g. ``"homo_sapiens"``) or any query
            :py:meth:`idtrack.API.resolve_organism` accepts. If ``None``, it is read from the graph files in
            *idtrack_local_repository*, which must then hold graphs of exactly one organism.
        graph_last_ensembl_release (Optional[int]): Snapshot release of the IDTrack graph. If ``None``, the newest
            graph of the organism in *idtrack_local_repository* (built by :py:meth:`idtrack.API.build_graph`) is
            used, or the latest Ensembl release of the organism when there is none.
        verbose_level (Literal[0, 1, 2]): Logging verbosity; 0 = errors only, 1 = warnings, 2 =
            info.  Defaults to ``2``.
        debugging_variables (bool): Retain heavy intermediate structures for post-mortem
            inspection.  Defaults to ``False``.
        converted_id_column (str): Column name used to store converted identifiers inside the
            resulting ``AnnData.var`` DataFrame.  Defaults to ``"converted_id"``.

    Attributes:
        idt (idtrack.api.API): Lazily initialised IDTrack interface used for all identifier look-ups.
        multiple_ensembl_dict (dict[str, list[str]]): Map of collapsed IDs to *all* Ensembl IDs that were
            originally associated with the same target identifier.
        removed_conversion_failed_identifiers (dict[str, set[str]]): Features that failed conversion and
            were dropped from each dataset.
        kept_conversion_failed_identifiers (dict[str, set[str]]): Non-convertible features kept because
            they were consistently non-convertible across *all* datasets.
        removed_inconsistent_identifier_matching (dict[str, set[str]]): Features whose mappings disagreed
            between datasets and were therefore removed for consistency.
    """

    nonmatching_but_consistent_suffix = ""

    # Internal column name for matched Ensembl gene IDs.
    # Using a distinct name avoids conflicts when final_database="ensembl_gene".
    _ENSEMBL_GENE_COLUMN = "matching_ensembl_gene"

    @staticmethod
    def _get_column_as_series(df: pd.DataFrame, column: str) -> pd.Series:
        """Safely extract a single column from a DataFrame, always returning a Series.

        When a DataFrame has duplicate column names, ``df[column]`` returns a DataFrame
        instead of a Series. This method handles such edge cases by selecting the first
        matching column when duplicates exist.

        Args:
            df (pd.DataFrame): The DataFrame to extract from.
            column (str): The column name to extract.

        Returns:
            pd.Series: The column data as a Series.
        """
        result = df[column]
        if isinstance(result, pd.DataFrame):
            # Duplicate column names exist; take the first one and warn
            result = result.iloc[:, 0]
        return result

    def __init__(
        self,
        project_name: str,
        data_h5ad_dict: dict[str, str],  # custom dataset name to dataset absolute path
        project_local_repository: str,
        idtrack_local_repository: str,
        target_ensembl_release: int,
        final_database: str | None = None,
        organism_name: str | None = None,
        graph_last_ensembl_release: int | None = None,
        verbose_level: Literal[0, 1, 2] = 2,
        debugging_variables: bool = False,
        converted_id_column: str = "converted_id",
    ):
        """Instantiate the harmoniser and perform lightweight validation.

        The constructor merely *prepares* the harmonisation context: it validates input paths, configures
        logging, and primes IDTrack.  Heavy work—graph initialisation, identifier matching, gene-symbol
        resolution—happens lazily when the first harmonisation method is called.

        Args:
            project_name (str): See :py:meth:`HarmonizeFeatures`.
            data_h5ad_dict (dict[str, str]): See :py:meth:`HarmonizeFeatures`.
            project_local_repository (str): See :py:meth:`HarmonizeFeatures`.
            idtrack_local_repository (str): See :py:meth:`HarmonizeFeatures`.
            target_ensembl_release (int): See :py:meth:`HarmonizeFeatures`.
            final_database (Optional[str]): See :py:meth:`HarmonizeFeatures`.
            organism_name (Optional[str]): See :py:meth:`HarmonizeFeatures`.
            graph_last_ensembl_release (Optional[int]): See :py:meth:`HarmonizeFeatures`.
            verbose_level (Literal[0, 1, 2]): See :py:meth:`HarmonizeFeatures`.
            debugging_variables (bool): See :py:meth:`HarmonizeFeatures`.
            converted_id_column (str): See :py:meth:`HarmonizeFeatures`.

        Raises:
            ValueError: If *verbose_level* is not 0, 1, 2 or 3, if *organism_name* is ``None`` and the graphs in
                *idtrack_local_repository* do not name exactly one organism, or if *final_database* is ``None`` and
                the organism declares no symbol database.
        """
        self.project_name = project_name
        self.project_local_repository = project_local_repository
        self.data_h5ad_dict = data_h5ad_dict
        self.converted_id_column = converted_id_column
        self.idtrack_local_repository = idtrack_local_repository
        self.target_ensembl_release = target_ensembl_release
        self.n_datasets = len(self.data_h5ad_dict)
        self.debugging_variables = debugging_variables

        if verbose_level == 0:
            _logging = logging.ERROR
        elif verbose_level == 1:
            _logging = logging.WARNING
        elif verbose_level == 2:
            _logging = logging.INFO
        elif verbose_level == 3:
            _logging = logging.DEBUG
        else:
            raise ValueError(f"verbose_level must be 0, 1, 2, or 3, got {verbose_level!r}.")
        self.verbose_level = verbose_level

        self.idt = idtrack.API(local_repository=self.idtrack_local_repository)
        self.log = logging.getLogger("harmonize_features")
        self.idt.configure_logger(_logging)
        self.idt_initialized = False

        # Organism, final database and snapshot release: explicit values win; nothing defaults to one organism.
        self.organism_name = self._resolve_organism_name(organism_name)
        self.final_database = (
            final_database if final_database is not None else self.default_final_database(self.organism_name)
        )
        self.graph_last_ensembl_release = (
            int(graph_last_ensembl_release)
            if graph_last_ensembl_release is not None
            else self._default_graph_last_release()
        )

        self.dict_n_to_1: dict[str, list[str]] = dict()
        self.multiple_ensembl_list: list[list[tuple[str, str]]] = list()

        # Important dictionaries for debugging:

        # If there is n-to-1 matching in a database and if one of them is query=matching, keep it and remove others
        # for a given group (n), below shows the chosen id (due to its identical as explained), the group
        # and the datasets that has this group
        self.dict_n_to_1_with_query: dict[str, dict[tuple, list]] = dict()
        # this dictionary is ideal to find out the removed ids
        self.dict_n_to_1_with_query_reverse: dict[str, dict[str, list[str]]] = dict()
        # this is a dictary showing group-to-dataset
        self.dict_n_to_1_without_query: dict[tuple, list] = dict()
        # also:
        # cached_property: self.dict_1_to_not_1

        # to include into integrated anndata object.

        # sometimes, final database (e.g. HGNC) id is associated with multiple ensembl ids, the algorithm chooses
        # one of them and creates a list. here you can find the actual possible ensembl ids.
        self.multiple_ensembl_dict: dict[str, list[str]] = dict()
        # removed ids from the anndata by `harmonizer` method. the list and the in which datasets it is deleted is kept.
        self.removed_conversion_failed_identifiers: dict[str, dict[str, list[str]]] = dict()
        # these ids could be removed but kept as they are consistent across all provided datasets.
        self.kept_conversion_failed_identifiers: dict[str, dict[str, list[str]]] = dict()
        # inconsistent matching
        self.removed_inconsistent_identifier_matching: dict[str, dict[str, list[str]]] = dict()
        # removed due to multiple source IDs mapping to same final_database (ensembl_gene override duplicates)
        self.removed_final_database_duplicates: dict[str, dict[str, list[str]]] = dict()

        self._initialize()

    @staticmethod
    def default_final_database(organism: str) -> str:
        """Return the organism's default final database: its first entry in :py:attr:`idtrack.DB.symbol_databases`.

        Args:
            organism: Ensembl species name of a registered organism.

        Returns:
            str: E.g. ``"HGNC Symbol"`` for human, ``"VGNC Symbol"`` for rhesus macaque.

        Raises:
            ValueError: If the organism is not registered or declares no symbol database.
        """
        databases = DB.symbol_databases.get(organism)
        if not databases:
            raise ValueError(
                f"No symbol database is declared for organism {organism!r} (`DB.symbol_databases`); "
                "pass `final_database` explicitly."
            )
        return str(databases[0])

    def _graph_snapshots(self) -> dict[str, list[int]]:
        """Return ``{organism: [snapshot releases]}`` of the graphs :py:meth:`idtrack.API.build_graph` would load.

        Graph files are named by :py:meth:`idtrack._graph_maker.GraphMaker.create_file_name`; the API builds them
        with ``ignore_before`` at idtrack's floor and ``narrow=True``.

        Returns:
            dict[str, list[int]]: Sorted snapshot releases per organism with at least one such graph.
        """
        floor = int(min(DB.mysql_port_min_release.values()))
        found: dict[str, list[int]] = {}
        try:
            names = os.listdir(self.idtrack_local_repository)
        except OSError:
            return found
        for organism in DB.supported_organisms:
            pattern = re.compile(rf"^graph_{re.escape(organism)}_min{floor}_max([0-9]+)_narrow\.pickle$")
            releases = sorted(int(m.group(1)) for m in map(pattern.match, names) if m)
            if releases:
                found[organism] = releases
        return found

    def _resolve_organism_name(self, organism_name: str | None) -> str:
        """Return the Ensembl species name to use (see the ``organism_name`` argument)."""
        if organism_name is None:
            organisms = sorted(self._graph_snapshots())
            if len(organisms) != 1:
                found = (
                    f"holds graphs of several organisms ({', '.join(organisms)})"
                    if organisms
                    else "holds no graph built by `API.build_graph`"
                )
                raise ValueError(
                    f"Pass `organism_name`: the idtrack local repository {self.idtrack_local_repository!r} {found}, "
                    "so the organism cannot be read from it. Use an Ensembl species name of "
                    f"`idtrack.DB.supported_organisms` ({', '.join(DB.supported_organisms)}), e.g. "
                    '`organism_name="homo_sapiens"`, or a taxon ID or name that `API.resolve_organism` accepts.'
                )
            return organisms[0]
        # The formal name of a configured organism resolves without the network; other queries need Ensembl REST.
        return idtrack.VerifyOrganism(organism_name).get_formal_name()

    def _default_graph_last_release(self) -> int:
        """Return the newest graph snapshot of the organism, else the organism's latest Ensembl release."""
        releases = self._graph_snapshots().get(self.organism_name)
        if releases:
            return int(max(releases))
        _, latest_release = self.idt.resolve_organism(self.organism_name)
        return int(latest_release)

    def _initialize(self) -> None:
        """Populate diagnostic structures for failed or ambiguous identifier conversions.

        Called once by :py:meth:`HarmonizeFeatures.__init__`, this routine scans every input dataset and
        updates several reporting attributes (for example :py:attr:`removed_conversion_failed_identifiers`
        or :py:attr:`removed_inconsistent_identifier_matching`).  It also derives
        :py:attr:`multiple_ensembl_dict`, a reverse map of ambiguous *Ensembl ID → source identifiers*,
        enabling downstream inspection of one-to-many relationships.

        Returns ``None``: All results are stored on *self* for later inspection.

        Internally the method:

        1. Extracts raw feature identifiers from each :py:class:`anndata.AnnData` file.
        2. Classifies identifiers into failure or inconsistency categories.
        3. Records per-dataset membership via :py:meth:`reporter_dict_creator`.
        4. Builds the ``multiple_ensembl_list`` used by
           :py:meth:`HarmonizeFeatures.create_multiple_ensembl_dict`.
        5. Touches :py:meth:`datataset_conversion_dataframe_issues` so the cached-property is built
           eagerly.
        """
        for dataset_name, dataset_path in self.data_h5ad_dict.items():
            gene_list = self.extract_source_identifiers_from_anndata(dataset_path)

            self.reporter_dict_creator(
                dataset_name=dataset_name,
                the_dict=self.removed_conversion_failed_identifiers,
                the_set=set(gene_list) & self.conversion_failed_identifiers,
            )

            self.reporter_dict_creator(
                dataset_name=dataset_name,
                the_dict=self.kept_conversion_failed_identifiers,
                the_set=set(gene_list) & self.conversion_failed_but_consistent_identifiers,
            )

            self.reporter_dict_creator(
                dataset_name=dataset_name,
                the_dict=self.removed_inconsistent_identifier_matching,
                the_set=set(gene_list) & self.datataset_conversion_dataframe_issues["gene_names_inconsistency"],
            )

            self.reporter_dict_creator(
                dataset_name=dataset_name,
                the_dict=self.removed_final_database_duplicates,
                the_set=set(gene_list)
                & self.datataset_conversion_dataframe_issues["final_database_duplicate_source_ids"],
            )

            for the_id in gene_list:
                conversion = self.unified_matching_dict[the_id]["matching"]["last_node"]
                if the_id not in self.conversion_failed_but_consistent_identifiers:
                    self.multiple_ensembl_list.append(conversion)

        self.multiple_ensembl_dict = self.create_multiple_ensembl_dict()
        self.datataset_conversion_dataframe_issues

    def _initialize_idt(self) -> None:
        """Instantiate the IDTrack interface on first use.

        The public API defers expensive graph loading until it is actually required.  This helper therefore
        checks whether :py:attr:`idt` is ``None`` and, if so, loads the on-disk identifier graph described by
        *idtrack_local_repository* and *graph_last_ensembl_release*, then configures release filters so that
        subsequent look-ups always target *target_ensembl_release*.  Re-invocations are no-ops.

        Returns ``None``: The :py:attr:`idt` attribute is populated and ready for queries.
        """
        if not self.idt_initialized:
            # `self.organism_name` is already the Ensembl species name (see `_resolve_organism_name`).
            self.idt.build_graph(organism_name=self.organism_name, snapshot_release=self.graph_last_ensembl_release)
            self.idt.calculate_graph_caches()
            self.idt_initialized = True

    def create_multiple_ensembl_dict(self) -> dict[str, list[str]]:
        """Reverse map ambiguous Ensembl target IDs to their originating source identifiers.

        During scanning, :py:meth:`_initialize` collects every *(source_id, target_ensembl_id)* pair that
        falls outside the *consistent one-to-one* category into ``multiple_ensembl_list``.  This helper
        consolidates that list into a dictionary keyed by ``target_ensembl_id`` with a **sorted list** of
        associated ``source_id`` values, allowing auditors to quickly discover all inputs that collapsed
        onto the same Ensembl record.

        Returns:
            dict[str, list[str]]: ``{target_ensembl_id: [source_id₁, source_id₂, …]}`` with duplicates
            removed and values sorted alphanumerically.
        """
        r: dict[str, set[str]] = dict()
        for i in self.multiple_ensembl_list:
            for n, m in i:
                if m not in r:
                    r[m] = set()
                r[m].add(n)

        return {k: sorted(v) for k, v in r.items()}

    def reporter_dict_creator_helper_reason_finder(self, the_id: str) -> str:
        """Infer why a particular identifier failed or produced a non-one-to-one conversion.

        The algorithm inspects :py:attr:`unified_matching_dict` and categorises *the_id* into one or more
        mutually non-exclusive reasons:

        * ``"n-to-1"``   — The identifier was part of an *n → 1* collapse within *at least one* dataset.
        * ``"1-to-0"``   — No target identifier was returned (conversion failure).
        * ``"1-to-n"``   — The conversion yielded multiple targets (ambiguous mapping).

        The final label is a single string where multiple reasons are concatenated with underscores, e.g.
        ``"1-to-0_1-to-n"``.

        Args:
            the_id (str): Source identifier whose conversion outcome needs explanation.

        Returns:
            str: Underscore-delimited reason string describing the failure or ambiguity class.
        """
        reason = []

        matching = self.unified_matching_dict[the_id]
        len_targets = len(matching["matching"]["target_id"])

        if the_id in self.dict_n_to_1:  # in one of the datasets during the integration, this ID was a part of n-to-1.
            reason.append("n-to-1")
        if len_targets == 0:
            reason.append("1-to-0")
        if len_targets > 1:
            reason.append("1-to-n")

        return "_".join(reason)

    def reporter_dict_creator(
        self,
        the_dict: dict[str, dict],
        the_set: set[str],
        dataset_name: str,
    ) -> None:
        """Update or create per-identifier diagnostic entries for a single dataset.

        Each identifier in *the_set* is ensured to exist as a key inside *the_dict*.  The entry's
        ``"reason"`` field is generated exactly once using
        :py:meth:`reporter_dict_creator_helper_reason_finder`; its ``"datasets_containing"`` list is then
        appended with *dataset_name*.  This allows quick aggregation of “where did this problematic
        identifier occur?” across all datasets.

        Returns ``None``: *the_dict* is modified in-place.

        Args:
            the_dict (dict[str, dict]): Target dictionary that stores diagnostic metadata.  Keys are
                source identifiers; values have keys ``"reason"`` (``str``) and
                ``"datasets_containing"`` (``list[str]``).
            the_set (set[str]): Identifiers that belong to the diagnostic category represented by
                *the_dict*.
            dataset_name (str): Human-readable alias of the dataset currently being processed.
        """
        for i in the_set:
            if i not in the_dict:
                the_dict[i] = {"reason": self.reporter_dict_creator_helper_reason_finder(i), "datasets_containing": []}
            the_dict[i]["datasets_containing"].append(dataset_name)

    @cached_property
    def datataset_conversion_dataframe_issues(self) -> pd.DataFrame:
        """Aggregate conversion failures and ambiguities into a tidy diagnostic table.

        The cached DataFrame has one row per *source identifier* encountered across **all** datasets and
        the following columns:

        * ``dataset``                — Dataset alias that triggered the row (duplicates possible).
        * ``reason``                 — Underscore-delimited label from
          :py:meth:`reporter_dict_creator_helper_reason_finder`.
        * ``target_identifier``      — The resolved identifier or ``NaN`` if conversion failed.
        * ``was_removed`` (bool)     — Whether the feature was ultimately dropped from the dataset.

        This compact view is ideal for spreadsheet export or in-notebook inspection because it condenses
        the richer nested structures stored on the class into a flat, analysis-friendly format.

        Returns:
            pandas.DataFrame: Combined diagnostic table sorted lexicographically by *dataset* and *reason*.
        """
        gene_list = set(self.unified_matching_dict.keys()) - self.conversion_failed_identifiers
        df = self.create_dataset_conversion_dataframe(gene_list=gene_list, initialization_run=True)

        # In Shiddar and Wang in HLCA datasets contains FAM231C, and FAM231B. Very weird genes, IDTrack fails:
        # ENSG00000268674.2	FAM231C	FAM231C
        # ENSG00000268674.2	FAM231B	FAM231B
        # Report and remove from all conversion lists.
        ensembl_col = self._ENSEMBL_GENE_COLUMN
        dfa = df.drop_duplicates(subset=[ensembl_col, self.final_database], keep="first")

        # Use safe column accessor to handle potential duplicate column names edge cases
        ensembl_gene_duplicated_mask = self._get_column_as_series(dfa, ensembl_col).duplicated(keep=False)
        dfa_duplicated_ensembl = dfa[ensembl_gene_duplicated_mask]
        _gene_names_inconsistency = set(
            self._get_column_as_series(dfa_duplicated_ensembl, self.final_database).unique()
        )

        final_db_col = self._get_column_as_series(df, self.final_database)
        gene_names_inconsistency = set(
            self._get_column_as_series(df[final_db_col.isin(_gene_names_inconsistency)], "Query ID").unique()
        )

        final_database_chosen_single_ensembl_dict = dict()
        # Track source IDs that should be removed because their ensembl_gene was not chosen
        # (they would cause duplicates after the ensembl_gene override)
        final_database_duplicate_source_ids = set()

        final_db_col_dfa = self._get_column_as_series(dfa, self.final_database)
        dfa_duplicated_final_db = dfa[final_db_col_dfa.duplicated(keep=False)]
        for final_final_database_gene_name, df_subset in dfa_duplicated_final_db.groupby(self.final_database):
            # Due to the path, idtrack may choose different ensembl_gene
            # for corresponding final_database (e.g. HGNC). Here, make a rule for each final_database id.
            # You can see all possible ensembl ids in
            ensembl_gene_col = self._get_column_as_series(df_subset, ensembl_col)
            chosen_ensembl_id = sorted(ensembl_gene_col)[0]  # choose first one
            final_database_chosen_single_ensembl_dict[final_final_database_gene_name] = chosen_ensembl_id
            # Mark source IDs with non-chosen ensembl_genes for removal to prevent duplicates
            non_chosen_rows = df_subset[ensembl_gene_col != chosen_ensembl_id]
            final_database_duplicate_source_ids.update(self._get_column_as_series(non_chosen_rows, "Query ID").tolist())

        return {
            "gene_names_inconsistency": gene_names_inconsistency,
            "final_database_chosen_single_ensembl_dict": final_database_chosen_single_ensembl_dict,
            "final_database_duplicate_source_ids": final_database_duplicate_source_ids,
        }

    def create_dataset_conversion_dataframe(
        self, gene_list: list[str] | pd.Index, initialization_run: bool
    ) -> pd.DataFrame:
        """Build a two-column mapping table for a single dataset's feature identifiers.

        The routine transforms every source identifier in *gene_list* into the target namespace defined by
        ``self.final_database`` and Ensembl gene IDs.  The resulting convertible subset is written into a
        new :py:class:`pandas.DataFrame` with three columns—``_ENSEMBL_GENE_COLUMN``, ``self.final_database``,
        and ``"Query ID"``—while problematic identifiers are annotated or filtered according to the rules
        established during :py:meth:`_initialize`.

        When called by :py:meth:`_initialize` (*initialization_run* ``True``), the method writes provisional
        mappings without inspecting post-initialisation overrides.  In subsequent calls
        (*initialization_run* ``False``) it resolves single-Ensembl ambiguities via
        ``self.datataset_conversion_dataframe_issues["final_database_chosen_single_ensembl_dict"]`` to
        guarantee a one-to-one relation between indices and feature rows.

        Args:
            gene_list (Union[list[str], pd.Index]): Ordered collection of source identifiers to convert for
                the current dataset.
            initialization_run (bool): ``True`` if invoked from :py:meth:`_initialize`; disables the
                single-Ensembl disambiguation step applied in later passes.

        Returns:
            pandas.DataFrame: Mapping table ready to become ``adata.var``.  Columns are
            ``_ENSEMBL_GENE_COLUMN``, ``self.final_database``, and the original ``"Query ID"`` for traceability.

        Raises:
            AssertionError: If diagnostic sets such as :py:attr:`conversion_failed_identifiers` were not
                populated—indicating an incorrect call order—or if unexpected duplicate target IDs remain
                after processing.
            ValueError: When malformed conversion entry.
        """
        # Ensure unified_matching_dict is populated (verifies proper initialization order).
        # The cached properties conversion_failed_identifiers and conversion_failed_but_consistent_identifiers
        # may legitimately be empty when all conversions succeed or when failures are all consistent/inconsistent.
        if len(self.unified_matching_dict) == 0:
            raise AssertionError("Possible function call order issue: unified_matching_dict is empty!")

        new_var_list = list()
        for the_id in gene_list:
            conversion = self.unified_matching_dict[the_id]["matching"]["last_node"]

            if the_id in self.conversion_failed_but_consistent_identifiers:
                modified_id = (f"{the_id}{HarmonizeFeatures.nonmatching_but_consistent_suffix}", the_id)
                new_var_list.append(list(modified_id) + [the_id])

            # 1-to-n's and 1-to-0s are either removed or kept as consistent ones.
            # if there is multiple ensembl id matching, then len(conversion) != 1.
            # it keeps the first one for now, the possible lists are saved self.multiple_ensembl_list in `initialize` method

            elif initialization_run:
                first_match = conversion[0]
                # Validate the conversion tuple structure to prevent DataFrame column misalignment
                if not isinstance(first_match, (tuple, list)) or len(first_match) != 2:
                    raise ValueError(
                        f"Malformed conversion entry for identifier {the_id!r}: expected a 2-tuple "
                        f"(ensembl_gene, {self.final_database}), got {first_match!r} with "
                        f"type={type(first_match).__name__}, len={len(first_match) if hasattr(first_match, '__len__') else 'N/A'}"
                    )
                ensembl_gene, final_db_value = first_match
                new_var_list.append([ensembl_gene, final_db_value, the_id])
            else:
                first_match = conversion[0]
                # Validate the conversion tuple structure
                if not isinstance(first_match, (tuple, list)) or len(first_match) != 2:
                    raise ValueError(
                        f"Malformed conversion entry for identifier {the_id!r}: expected a 2-tuple "
                        f"(ensembl_gene, {self.final_database}), got {first_match!r} with "
                        f"type={type(first_match).__name__}, len={len(first_match) if hasattr(first_match, '__len__') else 'N/A'}"
                    )
                ensembl_gene, final_db_value = first_match
                if (
                    final_db_value
                    in self.datataset_conversion_dataframe_issues["final_database_chosen_single_ensembl_dict"]
                ):
                    ensembl_gene = self.datataset_conversion_dataframe_issues[
                        "final_database_chosen_single_ensembl_dict"
                    ][final_db_value]

                new_var_list.append([ensembl_gene, final_db_value, the_id])

        df = pd.DataFrame(new_var_list, columns=[self._ENSEMBL_GENE_COLUMN, self.final_database, "Query ID"])
        return df

    def feature_harmonizer(self, dataset_name: str) -> tuple[ad.AnnData, int, int]:
        """Convert one dataset's feature space into the unified target namespace.

        The ``.h5ad`` file is read completely, the features removed during :py:meth:`_initialize` are dropped, and
        the feature axis is renamed with the mapping of :py:meth:`create_dataset_conversion_dataframe`. Everything
        else of the dataset is kept: ``X``, ``layers``, ``obs``, ``obsm``, ``obsp`` and ``uns`` as they are, and
        ``varm``/``varp`` subset to the kept features. ``var`` holds the input's own columns (a column whose name is
        one of the conversion columns gets the suffix ``"_input"``) after the conversion columns. ``raw`` is dropped
        (with a warning), because its features are not harmonised. The source file is never altered.

        Args:
            dataset_name (str): Key from :py:attr:`data_h5ad_dict` identifying which dataset to load and
                harmonise.

        Returns:
            tuple:
                * **resulting_adata** (:py:class:`anndata.AnnData`) - The dataset whose ``var`` index is the matched
                  Ensembl gene ID (or the input name for features kept unconverted), with the columns
                  ``self.final_database`` and ``"Query ID"`` (the input name) before the input's ``var`` columns.
                * **t0** (int) - Number of features *before* filtering and harmonisation.
                * **t1** (int) - Number of features *after* the procedure (i.e., retained in
                  *resulting_adata*).

        Raises:
            AssertionError: If duplicate Ensembl or target-database IDs slip past the conversion checks,
                which would break one-to-one mapping assumptions.
        """
        adata = ad.read_h5ad(self.data_h5ad_dict[dataset_name])
        t0 = adata.n_vars
        keep = ~(
            adata.var.index.isin(self.conversion_failed_identifiers)
            | adata.var.index.isin(self.datataset_conversion_dataframe_issues["gene_names_inconsistency"])
            | adata.var.index.isin(self.datataset_conversion_dataframe_issues["final_database_duplicate_source_ids"])
        )
        if adata.raw is not None:
            self.log.warning(
                f"Dataset {dataset_name!r}: `raw` is dropped, because its features are not harmonised; "
                "keep raw counts in a layer to carry them through."
            )
            del adata.raw
        adata = adata[:, keep].copy()

        df = self.create_dataset_conversion_dataframe(gene_list=adata.var.index, initialization_run=False)

        # Use safe column accessor to check for duplicates
        ensembl_col = self._ENSEMBL_GENE_COLUMN
        ensembl_gene_col = self._get_column_as_series(df, ensembl_col)
        final_db_col = self._get_column_as_series(df, self.final_database)
        if ensembl_gene_col.duplicated(keep=False).any() or final_db_col.duplicated(keep=False).any():
            raise AssertionError("Unexpected duplicated entry!")

        conversion = df.set_index(ensembl_col)
        conversion.index = conversion.index.astype(str)
        conversion.index.name = None
        input_var = adata.var.copy()
        input_var.columns = [f"{c}_input" if c in conversion.columns else c for c in input_var.columns]
        input_var.index = conversion.index
        adata.var = pd.concat([conversion, input_var], axis=1)

        return adata, t0, adata.n_vars

    @cached_property
    def dict_1_to_not_1(self) -> dict[str, set[str]]:
        """Collect identifiers involved in one-to-many or one-to-zero conversions.

        This helper scans :py:attr:`unified_matching_dict` and extracts every *source identifier* whose
        conversion to the target namespace is **not** a strict one-to-one mapping.  Two situations are
        considered problematic:

        * **1 → 0 (conversion failure)** — no target identifier could be resolved.
        * **1 → n (ambiguous hit)** — multiple targets share the best score, preventing an unambiguous choice.

        The resulting dictionary is later consumed by :py:meth:`reporter_dict_creator` to populate the
        diagnostic attributes exposed to users and by
        :py:meth:`create_dataset_conversion_dataframe` to decide which features should be dropped or
        flagged in each :py:class:`anndata.AnnData` object.

        Returns:
            dict[str, set[str]]: ``{problem_class: {source_id₁, source_id₂, …}}`` where *problem_class* is
                either ``"1-to-0"`` or ``"1-to-n"``.
        """
        result = dict()
        for query, value in self.unified_matching_dict.items():
            if len(value["matching"]["target_id"]) != 1:
                result[query] = value["datasets_containing"]

        return result

    @cached_property
    def conversion_failed_identifiers(self) -> set[str]:
        """Return identifiers that could not be converted in at least one dataset.

        The property wraps :py:meth:`dict_1_to_not_1` and filters its ``"1-to-0"`` category so that
        downstream code can quickly query *irrecoverable* failures without iterating over the entire
        diagnostic structure.

        Returns:
            set[str]: Identifiers that failed conversion in at least one dataset
                or have inconsistent mappings (1-to-0, 1-to-n, or n-to-1 where not
                all datasets share the same mapping).
        """
        result = set()

        for query, datasets_containing in self.dict_1_to_not_1.items():
            # for 1-to-n and 1-to-0, if it is not shared across all the datasets, add remove list.
            if len(datasets_containing) != self.n_datasets:
                result.add(query)

        for query, datasets_containing_2 in self.dict_n_to_1.items():
            # for one gene in n (among n-to-1), if it is not shared across all the datasets, add remove list.
            if len(datasets_containing_2) != self.n_datasets:
                result.add(query)

        return result

    @cached_property
    def conversion_failed_but_consistent_identifiers(self) -> set[str]:
        """Identify non-convertible identifiers that are consistently absent across *all* datasets.

        An identifier that fails conversion in *every* dataset can be retained (or at least logged once)
        without jeopardising dataset comparability.  This property computes the set intersection of
        :py:attr:`conversion_failed_identifiers` across datasets and makes the result available for
        selective retention or downstream visualisation.

        Returns:
            set[str]: Identifiers that were never convertible but appeared in every dataset examined.
        """
        _result1 = {i for i, _ in self.dict_n_to_1.items() if i not in self.conversion_failed_identifiers}
        _result2 = {i for i, _ in self.dict_1_to_not_1.items() if i not in self.conversion_failed_identifiers}
        result = set.union(_result1, _result2)

        return result

    @cached_property
    def unified_matching_dict(self):
        """Expose the full source-to-target identifier mapping produced by IDTrack.

        The dictionary is created during :py:meth:`_initialize` when the IDTrack graph is first queried.
        Keys are *source identifiers* (as found in input files); values are **all** candidate target IDs
        returned by the graph query, ordered by decreasing score.  A value may therefore be

        * a single-element list (unambiguous one-to-one),
        * a multi-element list (ambiguous one-to-n), or
        * an empty list (1-to-0 conversion failure).

        Public access to this attribute enables advanced users to perform their own diagnostics or to
        reproduce the algorithm's decisions outside the class.

        Returns:
            dict[str, list[str]]: Mapping ``{source_id: [target_id₁, target_id₂, …]}`` in the order
                delivered by the IDTrack query.
        """
        matching_dict = self.get_idtrack_matchings_for_all_datasets()
        unified_dict = dict()

        for dataset_name, matching in matching_dict.items():
            for m in matching:
                if m["query_id"] not in unified_dict:
                    unified_dict[m["query_id"]] = {"matching": m, "datasets_containing": []}
                unified_dict[m["query_id"]]["datasets_containing"].append(dataset_name)

        return unified_dict

    def get_idtrack_matchings_for_all_datasets(self) -> dict[str, pd.DataFrame]:
        """Return raw ID-Track matchings for every dataset in the project.

        This helper exposes the unfiltered mapping tables produced by ID-Track so that users can inspect
        exactly how each *source identifier* was converted (or failed to convert) in every individual
        dataset.  Internally it triggers :py:meth:`run_idtrack_for_single_dataset` for any dataset that
        has not yet been processed, caches the resulting tables in memory, and then assembles a
        ``{dataset_name: dataframe}`` dictionary whose keys align one-to-one with :py:data:`data_h5ad_dict`.

        Each returned :py:class:`pandas.DataFrame` includes at least the following columns:
        ``source_id``, ``target_id``, ``conversion_status``, ``reason``, and any custom
        metadata injected by :py:class:`idtrack.api.API`.

        Returns:
            dict[str, pandas.DataFrame]: Mapping of dataset alias to its full, row-level ID-Track
            matching table.  The dictionary order follows the insertion order of
            :py:data:`data_h5ad_dict`.
        """
        result_dataset = dict()
        for dataset_name, dataset_path in self.data_h5ad_dict.items():
            dataset_pickle = self.result_cache_path(dataset_name)

            switch_inconsistency = None
            if os.path.isfile(dataset_pickle) and os.access(dataset_pickle, os.R_OK):
                with open(dataset_pickle, "rb") as handle:
                    matching_list = pickle.load(handle)  # noqa: S301
                _saved_id_list = sorted([i["query_id"] for i in matching_list])
                _dataset_id_list = sorted(self.extract_source_identifiers_from_anndata(dataset_path=dataset_path))

                # if the id lists are different than the provided one.
                switch_inconsistency = _saved_id_list != _dataset_id_list

            if switch_inconsistency is not False:
                if switch_inconsistency is None and self.verbose_level > 2:
                    self.log.debug(f"Pickle not found: `{dataset_name}`. Calculating IDTrack matchings.")
                elif switch_inconsistency is True and self.verbose_level > 2:
                    self.log.debug(
                        "Inconsistent IDs with pickle object and the "
                        f"dataset: `{dataset_name}`. Calculating IDTrack matchings."
                    )
                matching_list = self.run_idtrack_for_single_dataset(dataset_name, dataset_path)
                with open(dataset_pickle, "wb") as handle:
                    pickle.dump(matching_list, handle)  # noqa: S301

            result_dataset[dataset_name] = matching_list
            if self.verbose_level > 1:
                binned_conversions = self.idt.classify_multiple_conversion(matching_list)
                self.log.info(f"{dataset_name}")
                self.idt.print_binned_conversion(binned_conversions)

            # Do not allow n_to_1 within a certain dataset.
            self.n_to_1_within_individual_dataset(dataset_name=dataset_name, dataset_matching_list=matching_list)

        return result_dataset

    def result_cache_path(self, dataset_name: str) -> str:
        """Return the path of the cached ID-Track matchings of one dataset.

        The file name holds everything the matchings depend on besides the identifiers themselves (which are
        checked when the file is read): project, dataset, organism, final database, target release and graph
        snapshot release, e.g. ``idtrack_result_<project>_<dataset>_homo_sapiens_HGNC-Symbol_rel114_graph115.pickle``.

        Args:
            dataset_name: Key of :py:attr:`data_h5ad_dict`.

        Returns:
            str: Absolute or repository-relative path inside :py:attr:`project_local_repository`.
        """
        final_db_slug = re.sub(r"[^0-9A-Za-z]+", "-", str(self.final_database)).strip("-")
        name = (
            f"idtrack_result_{self.project_name}_{dataset_name}_{self.organism_name}_{final_db_slug}"
            f"_rel{self.target_ensembl_release}_graph{self.graph_last_ensembl_release}.pickle"
        )
        return os.path.join(self.project_local_repository, name)

    def extract_source_identifiers_from_anndata(self, dataset_path: str) -> list[str]:
        """Load an ``.h5ad`` file and harvest the raw feature identifiers.

        To prepare inputs for ID-Track, this routine opens the single-cell expression matrix at
        *dataset_path*, reads the ``.var`` DataFrame, and extracts either the ``"gene_id"`` field (if
        present) or the index itself as the *source identifier*.  Identifiers are returned in file order
        so that downstream procedures can preserve the original feature ordering when reconstructing
        matrices.

        Args:
            dataset_path (str): Absolute or project-relative path to an ``.h5ad`` file containing a valid
                :py:class:`anndata.AnnData` object.

        Returns:
            list[str]: Ordered list of identifier strings exactly as they appear in the source file.
        """
        adata = ad.read_h5ad(dataset_path, backed="r")
        result = list(adata.var_names)
        del adata
        gc.collect()

        return result

    def run_idtrack_for_single_dataset(self, dataset_name: str, dataset_path: str) -> pd.DataFrame:
        """Convert identifiers for one dataset and cache the raw ID-Track output.

        Given a dataset alias and its on-disk location, this method:

        1. Calls :py:meth:`extract_source_identifiers_from_anndata` to obtain the feature list.
        2. Feeds those identifiers to :py:class:`idtrack.api.API` and collects the per-feature match
           results.
        3. Stores the resulting :py:class:`pandas.DataFrame` inside the
           :py:attr:`_idtrack_matchings_per_dataset` cache so repeated calls are O(1).
        4. Updates :py:attr:`unified_matching_dict` so that cross-dataset diagnostics remain consistent.

        Users rarely call this directly—:py:meth:`get_idtrack_matchings_for_all_datasets` handles the
        orchestration—but it remains public for advanced, dataset-by-dataset debugging.

        Args:
            dataset_name (str): Human-friendly alias used as the key inside diagnostic dictionaries.
            dataset_path (str): Absolute or project-relative ``.h5ad`` path passed straight to
                :py:meth:`extract_source_identifiers_from_anndata`.

        Returns:
            pandas.DataFrame: Full ID-Track matching table for *dataset_name* with columns
                ``source_id``, ``target_id``, ``conversion_status``, and any extra metadata returned by the API.
        """
        self._initialize_idt()
        gene_list = self.extract_source_identifiers_from_anndata(dataset_path)

        matching_list = self.idt.convert_identifier_multiple(
            gene_list,
            final_database=self.final_database,
            to_release=self.target_ensembl_release,
            pbar_prefix=dataset_name,
            verbose=self.verbose_level > 0,
        )

        return matching_list

    def n_to_1_within_individual_dataset(self, dataset_name: str, dataset_matching_list: list[dict]) -> None:
        """Detect n-to-1 collapses inside one dataset and populate diagnostic caches.

        In the ID-Track context *n-to-1* means **several source identifiers** (``query_id``) converging on the
        **same target identifier** (``matched_id``).  Such collapses are problematic because they merge
        distinct features when building the harmonised expression matrix.  This helper inspects the raw
        matching rows for a single dataset, discovers all many-to-one events (including those that passed
        through the *alternative target database*), and records the results in a family of per-project
        dictionaries so that later stages—merging, filtering, and reporting—can make informed decisions.

        The routine never returns a value; instead it mutates the following public attributes:

        * :py:attr:`dict_n_to_1` - ``{matched_id: [dataset₁, dataset₂, …]}`` listing every dataset where the
          collapse occurred.
        * :py:attr:`dict_n_to_1_with_query` - ``{matched_id: {(query_id₁,…): [dataset]}}`` for cases where the
          ``matched_id`` also appears in the collapsing query set.
        * :py:attr:`dict_n_to_1_with_query_reverse` - ``{query_id: {matched_id: [dataset]}}`` for a
          query-centric view.
        * :py:attr:`dict_n_to_1_without_query` - collapses where the target never appears in its own query set.

        Returns ``None``: All information is stored on the instance for subsequent pipeline stages.

        Args:
            dataset_name (str): Human-readable alias used throughout the project for *this* dataset.
            dataset_matching_list (list[dict]): Raw per-feature matchings returned by
                :py:class:`idtrack.api.API`.  Each dictionary must provide at least the keys
                ``"query_id"``, ``"last_node"``, and ``"final_database"``.
        """
        reverse_dict_target: dict[str, list[str]] = dict()
        reverse_dict_alternative_target: dict[str, list[str]] = dict()

        for matching_entry in dataset_matching_list:

            # to capture matching without alternative target database.
            # note that final database is `None` for 1-to-0, `ensembl_gene` for alternative target database.

            query_id = matching_entry["query_id"]

            if matching_entry["final_database"] == self.final_database:
                for _, j in matching_entry["last_node"]:  # for loop if it is 1-to-n
                    if j not in reverse_dict_target:
                        reverse_dict_target[j] = []
                    reverse_dict_target[j].append(query_id)

            elif matching_entry["final_database"] == "ensembl_gene":
                for i, _ in matching_entry["last_node"]:  # for loop if it is 1-to-n

                    if i not in reverse_dict_alternative_target:
                        reverse_dict_alternative_target[i] = []
                    reverse_dict_alternative_target[i].append(query_id)

        # simply get the ids that has 1-to-n in reverse orientation, so get n-to-1

        set_n_to_1_in_dataset = set()

        for the_dict_of_interest in [reverse_dict_target, reverse_dict_alternative_target]:
            for matched_id, query_id_list in the_dict_of_interest.items():
                if len(query_id_list) > 1:
                    for query_id in query_id_list:

                        # if there is query and mathed id are the same, exlude this query id from the n-to-1 set.
                        # basically assign it to be the true matching of the conversion for integration tasks
                        if query_id != matched_id:
                            # and matching is 1:1. note that if matching is actually 1-to-n,
                            # it is resolved by other parts of the code.
                            set_n_to_1_in_dataset.add(query_id)

                    # For debugging and reporting
                    if self.debugging_variables:
                        query_id_list_tuple = tuple(sorted(query_id_list))
                        if matched_id in query_id_list_tuple:

                            if matched_id not in self.dict_n_to_1_with_query:
                                self.dict_n_to_1_with_query[matched_id] = dict()
                            if query_id_list_tuple not in self.dict_n_to_1_with_query[matched_id]:
                                self.dict_n_to_1_with_query[matched_id][query_id_list_tuple] = list()
                            self.dict_n_to_1_with_query[matched_id][query_id_list_tuple].append(dataset_name)

                            for query_id in query_id_list_tuple:
                                if query_id not in self.dict_n_to_1_with_query_reverse:
                                    self.dict_n_to_1_with_query_reverse[query_id] = dict()
                                if matched_id not in self.dict_n_to_1_with_query_reverse[query_id]:
                                    self.dict_n_to_1_with_query_reverse[query_id][matched_id] = list()
                                self.dict_n_to_1_with_query_reverse[query_id][matched_id].append(dataset_name)

                        else:
                            if query_id_list_tuple not in self.dict_n_to_1_without_query:
                                self.dict_n_to_1_without_query[query_id_list_tuple] = list()
                            self.dict_n_to_1_without_query[query_id_list_tuple].append(dataset_name)

        for nto1 in set_n_to_1_in_dataset:
            if nto1 not in self.dict_n_to_1:
                self.dict_n_to_1[nto1] = []
            self.dict_n_to_1[nto1].append(dataset_name)

    def unify_multiple_anndatas(
        self,
        mode: Literal["union", "intersect"] = "union",
        obs_join: Literal["outer", "inner"] = "outer",
        obs_columns_to_keep: list[str] | None = None,
        numeric_var_columns: set[str] | None = None,
        numeric_obs_columns: set[str] | None = None,  # for hdca it is {"age"}
        handle_anndata_key: str = "handle_anndata",
    ) -> ad.AnnData:
        """Merge the harmonised datasets of :py:attr:`data_h5ad_dict` into one :py:class:`~anndata.AnnData`.

        Each dataset is harmonised with :py:meth:`feature_harmonizer` and all are concatenated along the cells in
        one :py:func:`anndata.concat`. By default nothing is lost, and values are never rewritten: text stays text,
        numbers stay numbers, and where a dataset has no value the result holds a missing value of the column's
        kind (``NaN`` for floats and text, ``pd.NA`` in a nullable integer or boolean column, ``NaT`` for times, a
        missing category for categoricals), never a placeholder string.

        * **Features.** ``"union"`` (default) keeps every feature of any dataset; a dataset that lacks a feature
          gets zeros for it in ``X`` and in every layer. ``"intersect"`` keeps the features present in every
          dataset.

        * **.obs.** ``obs_join="outer"`` (default) keeps every ``obs`` column of any dataset; in the cells of a dataset
          without the column it is missing (see above; an integer or boolean column becomes the nullable ``Int``/
          ``boolean`` dtype so that it can hold the missing values). ``obs_join="inner"`` keeps only the columns every
          dataset has. *obs_columns_to_keep*, if given, selects among the joined columns (in its order); a name that
          no dataset has, or (with ``"inner"``) that not every dataset has, raises ``ValueError`` naming it. Columns in
          *numeric_obs_columns* are converted to ``float`` (missing values allowed); other text columns become
          categorical (values unchanged). The column *handle_anndata_key* records the dataset of each cell
          (categorical, in the order of :py:attr:`data_h5ad_dict`), and cells are named ``f"{handle}_{cell}"``.

        * **.var.** ``converted_id_column`` holds the final-database name of each feature. In ``"union"`` mode the
          per-dataset columns ``f"{converted_id_column}_{handle}"`` (missing where the dataset lacks the feature)
          and an integer ``"intersection"`` column (1 if the feature is in every dataset) are added. Every ``var``
          column of the inputs is kept. A column is kept as **one** column when, for every feature, the datasets
          that have a value for it agree; it then holds that value (missing only where no dataset has one). When
          two datasets give different values for the same feature, nothing is chosen: the column is kept **per
          dataset** as ``f"{column}_{handle}"`` for each dataset that has it, and no column ``column`` exists.
          Missing values follow the same kinds as in ``obs``. Columns in *numeric_var_columns* are converted to
          ``float``; other text columns become categorical (values unchanged).

        * **layers, obsm, obsp, uns.** Handled by :py:func:`anndata.concat`: a layer or ``obsm`` entry present in
          every dataset is concatenated (``obsp`` block-diagonally), and an ``uns`` entry is kept when it is the same
          in every dataset. ``varm`` and ``varp`` are not carried into the merged object (they describe one
          dataset's features), nor is ``raw`` (see :py:meth:`feature_harmonizer`).

        ``X`` keeps the common dtype of the inputs (e.g. ``float32`` stays ``float32``).

        Args:
            mode (Literal["union", "intersect"]): How to combine the feature sets (see above).
            obs_join (Literal["outer", "inner"]): Which ``obs`` columns to keep: of any dataset (``"outer"``) or
                of every dataset (``"inner"``).
            obs_columns_to_keep (list[str] | None): Restrict ``obs`` to these joined columns. ``None`` keeps all.
            numeric_var_columns (set[str] | None): ``var`` columns to convert to ``float``.
            numeric_obs_columns (set[str] | None): ``obs`` columns to convert to ``float``.
            handle_anndata_key (str): Name of the ``obs`` column that records the dataset of each cell. Defaults to
                ``"handle_anndata"``; it must not be a column of any dataset.

        Returns:
            anndata.AnnData: The merged object; cells in the order of :py:attr:`data_h5ad_dict`, then of each file.

        Raises:
            ValueError: If *mode* or *obs_join* is not one of its values; if there is no dataset; if a name of
                *obs_columns_to_keep* is not among the joined columns, *handle_anndata_key* is already an ``obs``
                column, or a numeric column is absent or holds a value that is not a number; if two cells get the
                same name; if a generated ``var`` column name clashes with an input column; or if a feature gets
                different final-database names in different datasets.
        """
        if mode not in ("union", "intersect"):
            raise ValueError(f"`mode` must be 'union' or 'intersect', got {mode!r}.")
        if obs_join not in ("outer", "inner"):
            raise ValueError(f"`obs_join` must be 'outer' or 'inner', got {obs_join!r}.")
        numeric_obs_columns = set(numeric_obs_columns or set())
        numeric_var_columns = set(numeric_var_columns or set())
        handles = list(self.data_h5ad_dict)
        if not handles:
            raise ValueError("`data_h5ad_dict` holds no dataset to merge.")

        studies: list[ad.AnnData] = []
        for handle in tqdm.tqdm(handles, ncols=140, disable=self.verbose_level == 0):
            adata_study, t0, t1 = self.feature_harmonizer(dataset_name=handle)
            self.log.info(f"Dataset {handle!r}: {t1} of {t0} features kept after harmonisation.")
            studies.append(adata_study)

        obs_columns = self._joined_obs_columns([s.obs for s in studies], obs_join, obs_columns_to_keep)
        clashing = [h for h, s in zip(handles, studies) if handle_anndata_key in s.obs.columns]
        if clashing:
            raise ValueError(
                f"`handle_anndata_key` {handle_anndata_key!r} is already an obs column of {clashing}; pass another name."
            )

        templates = {c: next(s.obs[c] for s in studies if c in s.obs.columns) for c in obs_columns}
        # A column that some dataset lacks must hold missing values: integers and booleans become nullable there.
        partial = {c for c in obs_columns if not all(c in s.obs.columns for s in studies)}
        var_frames: dict[str, pd.DataFrame] = {}
        for handle, study in zip(handles, studies):
            index = pd.Index(f"{handle}_" + study.obs_names.astype(str))
            obs = pd.DataFrame(
                {
                    c: (
                        (_holding_missing(study.obs[c]) if c in partial else study.obs[c]).set_axis(index)
                        if c in study.obs.columns
                        else _missing_like(templates[c], index)
                    )
                    for c in obs_columns
                },
                index=index,
            )
            obs[handle_anndata_key] = handle
            study.obs = obs
            var_frames[handle] = study.var
            study.var = pd.DataFrame(index=study.var_names)

        adata = ad.concat(
            studies,
            join="outer" if mode == "union" else "inner",
            fill_value=0,
            merge=None,
            uns_merge="same",
            pairwise=True,
        )
        del studies

        duplicated_cells = adata.obs_names[adata.obs_names.duplicated()].unique().tolist()
        if duplicated_cells:
            raise ValueError(f"Cell names collide after prefixing with the dataset handle: {duplicated_cells[:10]}.")
        adata.obs[handle_anndata_key] = pd.Categorical(adata.obs[handle_anndata_key], categories=handles)

        adata.var = self._merge_harmonized_var(var_frames, adata.var_names, mode)

        for frame, columns, where in ((adata.obs, numeric_obs_columns, "obs"), (adata.var, numeric_var_columns, "var")):
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"`numeric_{where}_columns` names columns the merged {where} does not have: {missing}."
                )
            for column in columns:
                frame[column] = self._to_float(frame[column], f"adata.{where}[{column!r}]")
        for frame in (adata.obs, adata.var):
            for column in frame.columns:
                if column not in numeric_obs_columns | numeric_var_columns and _is_text(frame[column]):
                    frame[column] = frame[column].astype("category")
        return adata

    @staticmethod
    def _joined_obs_columns(
        frames: list[pd.DataFrame], obs_join: str, obs_columns_to_keep: list[str] | None
    ) -> list[str]:
        """Return the ``obs`` columns of the merged object (see :py:meth:`unify_multiple_anndatas`).

        Args:
            frames: ``obs`` of each harmonised dataset.
            obs_join: ``"outer"`` (columns of any dataset) or ``"inner"`` (columns of every dataset).
            obs_columns_to_keep: Optional selection among the joined columns.

        Returns:
            list[str]: The columns, in order of first appearance (or in the order of *obs_columns_to_keep*).

        Raises:
            ValueError: If a selected name is not among the joined columns.
        """
        seen: list[str] = []
        for frame in frames:
            seen += [c for c in frame.columns if c not in seen]
        joined = seen if obs_join == "outer" else [c for c in seen if all(c in f.columns for f in frames)]
        if obs_columns_to_keep is None:
            return joined
        nowhere = [c for c in obs_columns_to_keep if c not in seen]
        if nowhere:
            raise ValueError(f"`obs_columns_to_keep` names columns that no dataset has: {nowhere}.")
        not_everywhere = [c for c in obs_columns_to_keep if c not in joined]
        if not_everywhere:
            raise ValueError(
                f"`obs_columns_to_keep` names columns that not every dataset has: {not_everywhere}; "
                "they are kept only with obs_join='outer'."
            )
        return list(dict.fromkeys(obs_columns_to_keep))

    @staticmethod
    def _to_float(series: pd.Series, where: str) -> pd.Series:
        """Convert *series* to ``float``; missing values stay missing, any other non-number raises.

        Args:
            series: Values to convert.
            where: Name of the column, for the error message.

        Returns:
            pandas.Series: The values as ``float``.

        Raises:
            ValueError: If a non-missing value is not a number.
        """
        converted = pd.to_numeric(series.astype(object), errors="coerce")
        bad = series[converted.isna() & series.notna()]
        if len(bad):
            raise ValueError(f"Non-numeric value in {where}: {bad.unique().tolist()[:10]}.")
        return converted.astype(float)

    def _merge_harmonized_var(
        self, var_frames: dict[str, pd.DataFrame], var_names: pd.Index, mode: str
    ) -> pd.DataFrame:
        """Build the merged ``var`` of :py:meth:`unify_multiple_anndatas` (see its docstring for the rules).

        Args:
            var_frames: ``var`` of each harmonised dataset (from :py:meth:`feature_harmonizer`), by handle.
            var_names: Features of the merged object, in its order.
            mode: ``"union"`` or ``"intersect"``.

        Returns:
            pandas.DataFrame: The merged ``var``, indexed by *var_names*.

        Raises:
            ValueError: If a feature has different final-database names in different datasets, or a generated
                column name is also an input column.
        """
        # Every column of every dataset, aligned to the merged features; where a dataset lacks features, integers and
        # booleans become nullable first so that those features are missing values of the column's kind.
        aligned = {
            h: {
                c: (v[c] if var_names.isin(v.index).all() else _holding_missing(v[c])).reindex(var_names)
                for c in v.columns
            }
            for h, v in var_frames.items()
        }

        columns: dict[str, pd.Series] = {}

        def _add(name: str, values: pd.Series) -> None:
            if name in columns:
                raise ValueError(f"The merged var would have the column {name!r} twice; rename the input column.")
            columns[name] = values

        names = {h: a[self.final_database] for h, a in aligned.items()}
        agreed = self._agreed_values(names, var_names)
        if agreed is None:
            frame = pd.DataFrame({h: s.astype(object) for h, s in names.items()}, index=var_names)
            examples = frame[frame.nunique(axis=1, dropna=True) > 1].head(10).to_dict(orient="index")
            raise ValueError(f"Features with different {self.final_database} names across datasets: {examples}.")
        _add(self.converted_id_column, agreed)
        if mode == "union":
            for h, s in names.items():
                _add(f"{self.converted_id_column}_{h}", s)
            _add("intersection", pd.concat(list(names.values()), axis=1).notna().all(axis=1).astype(int))

        input_columns: list[str] = []
        for v in var_frames.values():
            input_columns += [c for c in v.columns if c not in (self.final_database, "Query ID", *input_columns)]
        for column in input_columns:
            per_dataset = {h: a[column] for h, a in aligned.items() if column in a}
            merged = self._agreed_values(per_dataset, var_names)
            if merged is not None:
                _add(column, merged)
            else:
                for h, s in per_dataset.items():
                    _add(f"{column}_{h}", s)
        return pd.DataFrame(columns, index=var_names)

    @staticmethod
    def _agreed_values(per_dataset: dict[str, pd.Series], index: pd.Index) -> pd.Series | None:
        """Combine aligned per-dataset values into one column, or return ``None`` when two datasets disagree.

        Args:
            per_dataset: Values of one column per dataset, aligned to *index* (missing where a dataset has none).
            index: The merged features.

        Returns:
            Optional[pandas.Series]: For each feature the value the datasets agree on (missing if none has one), or
            ``None`` if some feature has two different values.
        """
        series = list(per_dataset.values())
        as_objects = pd.DataFrame({i: s.astype(object) for i, s in enumerate(series)}, index=index)
        if (as_objects.nunique(axis=1, dropna=True) > 1).any():
            return None
        if len({str(s.dtype) for s in series}) > 1:
            series = [_holding_missing(s) for s in series]
        merged = series[0]
        for other in series[1:]:
            if merged.dtype == other.dtype:
                merged = merged.combine_first(other)
            else:  # e.g. categoricals with different categories, or int and float: combine as objects
                merged = merged.astype(object).combine_first(other.astype(object)).infer_objects()
        return merged
