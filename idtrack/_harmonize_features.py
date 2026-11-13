#!/usr/bin/env python3

import hashlib
import json
import logging
import os
import re
import tempfile
from typing import Any, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tqdm

import idtrack
from idtrack import DB
from idtrack._api import API, _stable_id
from idtrack._graph_maker import GraphMaker

_CACHE_FORMAT = 2

#: Status of an output feature (``var["idtrack_status"]``); see :py:class:`HarmonizeFeatures`.
STATUSES = (
    "converted",
    "converted_no_final_id",
    "merged",
    "merged_no_final_id",
    "unconverted_merge_partner",
    "unconverted_ambiguous",
    "unconverted_no_conversion",
    "unconverted_not_in_graph",
    "unconverted_error",
)


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


def read_feature_names(path: str) -> list[str]:
    """Return the feature names (``var`` index) of an ``.h5ad`` file without reading anything else.

    Args:
        path: Path of the ``.h5ad`` file.

    Returns:
        list[str]: The names in file order.
    """
    with h5py.File(path, "r") as handle:
        var = handle["var"]
        index_key = var.attrs["_index"]
        index_key = index_key.decode() if isinstance(index_key, bytes) else str(index_key)
        values = ad.io.read_elem(var[index_key])
    return [v.decode() if isinstance(v, bytes) else v for v in np.asarray(values, dtype=object).tolist()]


def _validated_feature_names(dataset_name: str, names: list[Any]) -> list[str]:
    """Return *names* or raise naming the dataset and the offending names (none, not strings, empty, repeated)."""
    if not names:
        raise ValueError(f"Dataset {dataset_name!r} has no features.")
    bad = [n for n in names if not isinstance(n, str) or not n.strip()]
    if bad:
        raise ValueError(f"Dataset {dataset_name!r} has feature names that are empty or not strings: {bad[:10]}.")
    repeated = pd.Index(names)
    repeated = repeated[repeated.duplicated()].unique().tolist()
    if repeated:
        raise ValueError(
            f"Dataset {dataset_name!r} has {len(repeated)} repeated feature names (each feature must be unique), "
            f"e.g. {repeated[:10]}."
        )
    return names


def _slug(text: str) -> str:
    """Return *text* reduced to characters that are safe in file names, with a short hash that keeps it unique."""
    readable = re.sub(r"[^0-9A-Za-z.-]+", "-", text).strip("-.")[:60] or "x"
    return f"{readable}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def _write_json_atomically(path: str, content: Any) -> None:
    """Write *content* as JSON to a temporary file beside *path* and rename it, so a reader never sees a part."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(content, handle)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def feature_table(
    feature_names: list[str],
    matchings: list[dict[str, Any]],
    gene_names: dict[str, list[str]],
) -> pd.DataFrame:
    """Classify each input feature of one dataset by its conversion; a pure function of that dataset alone.

    Args:
        feature_names: The dataset's feature names, in file order.
        matchings: :py:meth:`idtrack.API.convert_identifier` results, one per feature, in the same order.
        gene_names: ``{ensembl_gene: names}``: the final-database names of every gene reached (empty: none).

    Returns:
        pandas.DataFrame: Indexed by the input names, in file order, with the columns ``status`` (one of
        ``"converted"``, ``"converted_no_final_id"``, ``"ambiguous"``, ``"no_conversion"``, ``"not_in_graph"``,
        ``"error"``), ``reason`` (missing for ``"converted"``, unless the feature had to be read at another release
        than the dataset's source release because the graph does not hold it there), ``ensembl_gene`` (the gene at
        the target release
        when there is exactly one), ``final_id`` (the gene's final-database name: the first in sorted order of
        *gene_names*), ``final_id_alternatives`` (the gene's other names, ``";"``-joined) and ``own`` (the input
        names the gene itself: its stable id, or one of its final-database names, compared without case).
    """
    rows = []
    for name, m in zip(feature_names, matchings):
        if m["query_id"] != name:
            raise ValueError(f"Matching of {m['query_id']!r} given for feature {name!r}.")
        genes = sorted({gene for gene, _ in m["last_node"]})
        row: dict[str, Any] = {
            "status": None,
            "reason": None,
            "ensembl_gene": None,
            "final_id": None,
            "final_id_alternatives": None,
            "own": False,
        }
        if m.get("error"):
            row.update(status="error", reason=f"{m['error']['type']}: {m['error']['message']}")
        elif m["no_corresponding"]:
            row.update(status="not_in_graph", reason="not found in the graph")
        elif m["no_conversion"] or not genes:
            row.update(status="no_conversion", reason=f"no gene at release {m['to_release']}")
        elif len(genes) > 1:
            row.update(status="ambiguous", reason=f"maps to {len(genes)} genes: {';'.join(genes)}")
        else:
            gene = genes[0]
            names = sorted(gene_names.get(gene, []))
            resolved = _stable_id(m.get("graph_id") or name).casefold()
            row.update(
                ensembl_gene=gene,
                final_id=names[0] if names else None,
                final_id_alternatives=";".join(names[1:]) or None,
                own=resolved in {_stable_id(gene).casefold(), *(n.casefold() for n in names)},
            )
            if m.get("from_release_fallback"):
                row["reason"] = (
                    f"not in the graph at release {m['from_release_given']}; read at release {m['from_release']}"
                )
            if names:
                row["status"] = "converted"
            else:
                no_name = "the gene has no id in the final database"
                row.update(
                    status="converted_no_final_id", reason=f"{no_name}; {row['reason']}" if row["reason"] else no_name
                )
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(feature_names, dtype=object), columns=list(rows[0]) if rows else None)


def _text_or_none(value: Any) -> str | None:
    """Return a table cell as text, or ``None`` for a missing value (pandas stores a missing text as NaN)."""
    return None if value is None or (isinstance(value, float) and value != value) else str(value)


def output_plan(table: pd.DataFrame, merge: str, keep_unconverted: bool) -> pd.DataFrame:
    """Decide the output features of one dataset from its :py:func:`feature_table` (a pure function of it).

    Rules (also in :py:class:`HarmonizeFeatures`):

    * A gene reached by exactly one input feature is one output feature named by the gene (``"converted"``, or
      ``"converted_no_final_id"`` when the gene has no final-database id).
    * A gene reached by several input features ("n-to-1"): with ``merge="sum"`` one output feature named by the gene
      whose values are the sum of theirs (``"merged"``/``"merged_no_final_id"``); with ``merge="flag"``, if exactly
      one of them names the gene itself (``own``) it becomes the gene's feature and the others stay under their
      input names as ``"unconverted_merge_partner"``; otherwise all stay under their input names as
      ``"unconverted_merge_partner"`` and the gene has no output feature.
    * Every other input feature stays under its input name (``"unconverted_ambiguous"``,
      ``"unconverted_no_conversion"``, ``"unconverted_not_in_graph"``, ``"unconverted_error"``) when
      *keep_unconverted*, and is left out otherwise; merge partners are always kept.
    * An input name kept unconverted that equals the name of a converted output feature is renamed
      ``"unconverted:<name>"``.

    Args:
        table: Output of :py:func:`feature_table`.
        merge: ``"sum"`` or ``"flag"``.
        keep_unconverted: Keep the unconverted features (see above).

    Returns:
        pandas.DataFrame: One row per output feature, in the order of its first input feature, indexed by the output
        name, with the columns ``sources`` (list of input positions), ``query_ids`` (``";"``-joined input names),
        ``status``, ``reason``, ``ensembl_gene``, ``final_id``, ``final_id_alternatives``.
    """
    converted = table["status"].isin(["converted", "converted_no_final_id"])
    groups: dict[str, list[int]] = {}
    for position, gene in enumerate(table["ensembl_gene"].tolist()):
        if converted.iloc[position]:
            groups.setdefault(gene, []).append(position)
    names = table.index.tolist()
    statuses = table["status"].tolist()
    rows: list[dict[str, Any]] = []
    first_of_gene: dict[str, int] = {}

    def _gene_row(gene: str, sources: list[int], status: str, reason: str | None) -> dict[str, Any]:
        first = sources[0]
        return {
            "name": gene,
            "sources": sources,
            "query_ids": ";".join(names[s] for s in sources),
            "status": status,
            "reason": reason,
            "ensembl_gene": gene,
            "final_id": table["final_id"].iloc[first],
            "final_id_alternatives": table["final_id_alternatives"].iloc[first],
        }

    def _kept_row(position: int, status: str, reason: str | None, gene: str | None) -> dict[str, Any]:
        return {
            "name": names[position],
            "sources": [position],
            "query_ids": names[position],
            "status": status,
            "reason": reason,
            "ensembl_gene": gene,
            "final_id": None,
            "final_id_alternatives": None,
        }

    for position in range(len(names)):
        status = statuses[position]
        if status in ("converted", "converted_no_final_id"):
            gene = table["ensembl_gene"].iloc[position]
            sources = groups[gene]
            suffix = "" if status == "converted" else "_no_final_id"
            if len(sources) == 1:
                rows.append(_gene_row(gene, sources, status, _text_or_none(table["reason"].iloc[position])))
                continue
            partners = ";".join(names[s] for s in sources)
            if merge == "sum":
                if gene not in first_of_gene:
                    first_of_gene[gene] = position
                    reason = f"sum of {len(sources)} input features: {partners}"
                    rows.append(_gene_row(gene, sources, f"merged{suffix}", reason))
                continue
            own = [s for s in sources if table["own"].iloc[s]]
            if len(own) == 1 and own[0] == position:
                rows.append(_gene_row(gene, [position], status, f"kept as the gene itself among: {partners}"))
            else:
                reason = f"shares the gene {gene} with: {partners}"
                rows.append(_kept_row(position, "unconverted_merge_partner", reason, gene))
        elif keep_unconverted:
            rows.append(
                _kept_row(position, f"unconverted_{status}", _text_or_none(table["reason"].iloc[position]), None)
            )

    plan = pd.DataFrame(rows, columns=list(_kept_row(0, "", None, None)) if not rows else None)
    if plan.empty:
        return pd.DataFrame(
            columns=["sources", "query_ids", "status", "reason", "ensembl_gene", "final_id", "final_id_alternatives"],
            index=pd.Index([], dtype=object),
        )
    gene_names_out = set(plan.loc[~plan["status"].str.startswith("unconverted"), "name"])
    clash = plan["status"].str.startswith("unconverted") & plan["name"].isin(gene_names_out)
    plan.loc[clash, "name"] = "unconverted:" + plan.loc[clash, "name"]
    plan = plan.set_index("name")
    plan.index.name = None
    if plan.index.duplicated().any():
        raise AssertionError(f"Output features named twice: {plan.index[plan.index.duplicated()].tolist()[:10]}.")
    return plan


def _aggregate(matrix: Any, plan: pd.DataFrame, n_inputs: int) -> Any:
    """Return *matrix* (cells x input features) as cells x output features: columns selected, merged ones summed.

    Args:
        matrix: Dense or sparse matrix with one column per input feature.
        plan: Output of :py:func:`output_plan`.
        n_inputs: Number of input features.

    Returns:
        The matrix of the output features, of the same kind and dtype.
    """
    sources = plan["sources"].tolist()
    if all(len(s) == 1 for s in sources):
        columns = [s[0] for s in sources]
        return matrix[:, columns]
    rows = np.concatenate([np.asarray(s, dtype=np.int64) for s in sources])
    cols = np.repeat(np.arange(len(sources)), [len(s) for s in sources])
    dtype = matrix.dtype if np.issubdtype(matrix.dtype, np.number) else np.float64
    summing = sp.csr_matrix((np.ones(len(rows), dtype=dtype), (rows, cols)), shape=(n_inputs, len(sources)))
    if sp.issparse(matrix):
        out = (sp.csr_matrix(matrix) @ summing).astype(matrix.dtype, copy=False)
        return out if isinstance(matrix, (sp.csr_matrix, sp.csr_array)) else out.asformat(matrix.format)
    return np.asarray(summing.T @ np.asarray(matrix).T).T.astype(matrix.dtype, copy=False)


class HarmonizeFeatures:
    """Convert the features of several single-cell datasets to one Ensembl release and merge the datasets.

    For every dataset, each feature name (Ensembl gene id with or without version, symbol or other external id) is
    converted with :py:meth:`idtrack.API.convert_identifier_multiple` to its Ensembl gene at
    *target_ensembl_release*. The conversion of a dataset depends only on that dataset (its feature names and its
    *source_releases* and *source_databases* entries), never on the other datasets.

    **Output features.** Each converted gene is one feature, named by the gene id at the target release (with
    version); its final-database name (default: the organism's symbol database) is in the column named after the
    final database. That name is a property of the gene, not of the query: the gene's ids in the final database at
    the target release (:py:meth:`idtrack.Track.gene_to_database`: read from the gene, no search), the first in sorted order when
    there are several (the others in ``"<final database>_alternatives"``). A gene without an id in the final database
    keeps the gene id as its feature name and a missing final-database name (status ``"converted_no_final_id"``):
    the final-database column never holds an Ensembl id. Features that are not converted keep their input name; no
    feature is dropped silently. Every output feature has, in ``var``:

    * ``"idtrack_status"``: ``"converted"``, ``"converted_no_final_id"``, ``"merged"``, ``"merged_no_final_id"``
      (with ``merge="sum"``, the sum of several input features), ``"unconverted_merge_partner"`` (with
      ``merge="flag"``, an input feature that shares its gene with others), ``"unconverted_ambiguous"`` (several
      genes), ``"unconverted_no_conversion"`` (in the graph, but no gene at the target release),
      ``"unconverted_not_in_graph"`` or ``"unconverted_error"`` (the conversion of this feature raised an error that
      concerns it alone, a search over its budget);
    * ``"idtrack_reason"``: why (missing for a plain conversion; for a feature that the graph does not hold at its
      dataset's source release and that was therefore read at another release, that release);
    * ``"Query ID"``: the input name(s), ``";"``-joined for a merged feature;
    * ``"matching_ensembl_gene"``: the gene (also for a merge partner; missing for the other unconverted features).

    **Several input features of one dataset on one gene** (an Ensembl merge, or two aliases): with ``merge="sum"``
    (right for counts) they become one feature whose values in ``X`` and every layer are the sum of theirs; with
    ``merge="flag"`` (default, for normalised data, where a sum has no meaning) the feature that names the gene itself
    (its stable id or one of its final-database names, ignoring version and case) becomes the gene's feature, and
    the others keep their input names with status ``"unconverted_merge_partner"`` (if none or several name the gene,
    all keep their names and the gene has no feature). Rows that a merge sums carry a ``var`` value only where all
    their input features agree (missing otherwise), and ``varm``/``varp`` are left out of a dataset in which features
    were summed.

    An unconverted input name that equals a converted feature's name (in the dataset, or in another dataset when
    merging) is renamed ``"unconverted:<name>"``, so that converted and unconverted measurements never share a row.

    **Result cache.** The conversions of each dataset are stored in *project_local_repository* as one JSON file per
    dataset (written to a temporary file and renamed). A stored file is used only if it was written by the same
    idtrack version for the same feature names, organism, final database, target and source release, graph snapshot
    and graph (the graph file's name, size and modification time, or the loaded graph's metadata when *api* is
    given); an unreadable or mismatching file is recomputed and replaced.

    Args:
        project_name (str): Label used in log messages and cache file names.
        data_h5ad_dict (dict[str, str]): ``{dataset name: path of an .h5ad file}``; at least one.
        project_local_repository (str): Writable folder for the result cache (created if missing).
        idtrack_local_repository (str): idtrack's local repository holding the graph.
        target_ensembl_release (int): Ensembl release all features are converted to; a release of the graph.
        final_database (Optional[str]): Database of the feature names in the output (e.g. ``"HGNC Symbol"``).
            Defaults to the organism's first symbol database (:py:attr:`idtrack.DB.symbol_databases`: HGNC Symbol for
            human, MGI Symbol for mouse, VGNC Symbol for pig and the macaques).
        organism_name (Optional[str]): Organism, as an Ensembl species name (e.g. ``"homo_sapiens"``) or any query
            :py:meth:`idtrack.API.resolve_organism` accepts. If ``None``, it is read from the graph files in
            *idtrack_local_repository*, which must then hold graphs of exactly one organism.
        graph_last_ensembl_release (Optional[int]): Snapshot release of the graph. If ``None``, the newest graph of
            the organism in *idtrack_local_repository* (built by :py:meth:`idtrack.API.build_graph`) is used, or the
            latest Ensembl release of the organism when there is none.
        source_releases (Optional[dict[str, int]]): The Ensembl release each dataset was annotated with, by dataset
            name. A dataset with a release is converted with ``from_release``; a feature that the graph does not hold
            at that release is read where the graph holds it (the rule without a release, below), and its
            ``"idtrack_reason"`` says so. Without a release, each feature is read at the target release if it is
            active there, else at the latest release in which it is active.
        source_databases (Optional[dict[str, str]]): The external database the feature names of a dataset are ids
            of, by dataset name (``{"study_a": "HGNC Symbol"}``; one of
            :py:meth:`idtrack.API.list_external_databases`). A dataset with a database is converted with
            ``from_database``: every feature is read as an id of that database, so a symbol that another database
            gives to another gene is not a tie, and a feature that is no id of that database (an Ensembl id among
            symbols) is not converted. Without one, nothing is assumed about where a name comes from: a name that
            two databases give to two genes stands for both (status ``"unconverted_ambiguous"``) when both genes
            have a name in *final_database*; a tie is judged among the genes that can be named there, so when only
            one of the two has a name in it, the feature is converted to that gene. Nothing is inferred
            from the names on the caller's behalf.
        api (Optional[idtrack.API]): An :py:class:`idtrack.API` with the graph already loaded, to use instead of
            loading the graph again. Without it, the harmonizer loads the graph only if some dataset is not in the
            result cache, and releases it when the conversions are done.
        verbose_level (Literal[0, 1, 2, 3]): 0 shows nothing, 1 progress bars, 2 also a summary of each dataset's
            conversions (logged at INFO by the ``"idtrack"`` loggers; see :py:meth:`idtrack.API.configure_logger`),
            3 also debug messages. The harmonizer never configures logging itself.
        converted_id_column (str): Name of the merged final-database column of :py:meth:`unify_multiple_anndatas`.

    Attributes:
        feature_names (dict[str, list[str]]): The feature names of each dataset, in file order.
        matchings (dict[str, list[dict]]): The conversion of each feature of each dataset (results of
            :py:meth:`idtrack.API.convert_identifier`, plain data).
        gene_names (dict[str, dict[str, list[str]]]): Per dataset, the final-database names of each gene reached.
    """

    _ENSEMBL_GENE_COLUMN = "matching_ensembl_gene"
    _QUERY_COLUMN = "Query ID"
    _STATUS_COLUMN = "idtrack_status"
    _REASON_COLUMN = "idtrack_reason"

    def __init__(
        self,
        project_name: str,
        data_h5ad_dict: dict[str, str],
        project_local_repository: str,
        idtrack_local_repository: str,
        target_ensembl_release: int,
        final_database: str | None = None,
        organism_name: str | None = None,
        graph_last_ensembl_release: int | None = None,
        source_releases: dict[str, int] | None = None,
        source_databases: dict[str, str] | None = None,
        api: API | None = None,
        verbose_level: Literal[0, 1, 2, 3] = 2,
        converted_id_column: str = "converted_id",
    ):
        """Validate the arguments, read the feature names and convert every feature of every dataset.

        Everything that can be checked without the graph is checked before any conversion starts (the arguments,
        the files, the feature names, the project folder). The conversions then run, or are read from the result
        cache; this is the long part (minutes for large datasets when nothing is cached).

        Args:
            project_name (str): See :py:class:`HarmonizeFeatures`.
            data_h5ad_dict (dict[str, str]): See :py:class:`HarmonizeFeatures`.
            project_local_repository (str): See :py:class:`HarmonizeFeatures`.
            idtrack_local_repository (str): See :py:class:`HarmonizeFeatures`.
            target_ensembl_release (int): See :py:class:`HarmonizeFeatures`.
            final_database (Optional[str]): See :py:class:`HarmonizeFeatures`.
            organism_name (Optional[str]): See :py:class:`HarmonizeFeatures`.
            graph_last_ensembl_release (Optional[int]): See :py:class:`HarmonizeFeatures`.
            source_releases (Optional[dict[str, int]]): See :py:class:`HarmonizeFeatures`.
            source_databases (Optional[dict[str, str]]): See :py:class:`HarmonizeFeatures`.
            api (Optional[idtrack.API]): See :py:class:`HarmonizeFeatures`.
            verbose_level (Literal[0, 1, 2, 3]): See :py:class:`HarmonizeFeatures`.
            converted_id_column (str): See :py:class:`HarmonizeFeatures`.

        Raises:
            ValueError: If *verbose_level* is not 0 to 3; if *data_h5ad_dict* is empty, names a missing file or a
                dataset whose feature names are empty, not strings or repeated; if *source_releases* or
                *source_databases* names an unknown dataset; if a release is not an integer or a source database not
                a string, or *target_ensembl_release* is after the graph snapshot; if
                *organism_name* is ``None`` and the graphs in *idtrack_local_repository* do not name exactly one
                organism; if *final_database* is ``None`` and the organism declares no symbol database; and the errors
                of :py:meth:`idtrack.API.convert_identifier_multiple` (e.g. a release or database not of the graph).
        """
        if verbose_level not in (0, 1, 2, 3):
            raise ValueError(f"verbose_level must be 0, 1, 2, or 3, got {verbose_level!r}.")
        self.verbose_level = verbose_level
        self.log = logging.getLogger(__name__)
        if not isinstance(data_h5ad_dict, dict) or not data_h5ad_dict:
            raise ValueError("`data_h5ad_dict` must be a non-empty dict {dataset name: .h5ad path}.")
        missing = {k: v for k, v in data_h5ad_dict.items() if not os.path.isfile(v)}
        if missing:
            raise ValueError(f"`data_h5ad_dict` names files that do not exist: {missing}.")
        self.project_name = str(project_name)
        self.data_h5ad_dict = dict(data_h5ad_dict)
        self.project_local_repository = project_local_repository
        self.idtrack_local_repository = idtrack_local_repository
        self.converted_id_column = converted_id_column
        self.target_ensembl_release = self._integer(target_ensembl_release, "target_ensembl_release")
        self.source_releases = {
            k: self._integer(v, f"source_releases[{k!r}]") for k, v in (source_releases or {}).items() if v is not None
        }
        unknown = sorted(set(self.source_releases) - set(self.data_h5ad_dict))
        if unknown:
            raise ValueError(f"`source_releases` names datasets that are not in `data_h5ad_dict`: {unknown}.")
        self.source_databases = {k: v for k, v in (source_databases or {}).items() if v is not None}
        unknown = sorted(set(self.source_databases) - set(self.data_h5ad_dict))
        if unknown:
            raise ValueError(f"`source_databases` names datasets that are not in `data_h5ad_dict`: {unknown}.")
        not_names = {k: v for k, v in self.source_databases.items() if not isinstance(v, str) or not v.strip()}
        if not_names:
            raise ValueError(f"`source_databases` must name one external database per dataset, got {not_names}.")

        if api is not None:
            graph = api._require_track().graph.graph
            self.organism_name = str(graph["organism"])
            self.graph_last_ensembl_release = int(graph["ensembl_release"])
            given = {"organism_name": organism_name, "graph_last_ensembl_release": graph_last_ensembl_release}
            if organism_name is not None and self._resolve_organism_name(organism_name) != self.organism_name:
                raise ValueError(
                    f"`organism_name`={organism_name!r} differs from the graph of `api` ({self.organism_name})."
                )
            if graph_last_ensembl_release is not None and graph_last_ensembl_release != self.graph_last_ensembl_release:
                raise ValueError(
                    f"`graph_last_ensembl_release`={given['graph_last_ensembl_release']!r} differs from the snapshot of "
                    f"the graph of `api` ({self.graph_last_ensembl_release})."
                )
        else:
            self.organism_name = self._resolve_organism_name(organism_name)
            self.graph_last_ensembl_release = (
                self._integer(graph_last_ensembl_release, "graph_last_ensembl_release")
                if graph_last_ensembl_release is not None
                else self._default_graph_last_release()
            )
        if self.target_ensembl_release > self.graph_last_ensembl_release:
            raise ValueError(
                f"`target_ensembl_release`={self.target_ensembl_release} is after the graph snapshot "
                f"{self.graph_last_ensembl_release}."
            )
        self.final_database = (
            final_database if final_database is not None else self.default_final_database(self.organism_name)
        )

        os.makedirs(self.project_local_repository, exist_ok=True)
        if not os.access(self.project_local_repository, os.W_OK):
            raise ValueError(f"`project_local_repository` {self.project_local_repository!r} is not writable.")

        self.feature_names: dict[str, list[str]] = {
            name: _validated_feature_names(name, read_feature_names(path)) for name, path in self.data_h5ad_dict.items()
        }
        self._api = api
        self._api_given = api is not None
        self.matchings: dict[str, list[dict[str, Any]]] = {}
        self.gene_names: dict[str, dict[str, list[str]]] = {}
        try:
            for name in self.data_h5ad_dict:
                self.matchings[name], self.gene_names[name] = self._dataset_conversions(name)
                if self.verbose_level > 1:
                    api_for_summary = self._api or API(self.idtrack_local_repository)
                    summary = api_for_summary.binned_conversion_summary(
                        api_for_summary.classify_multiple_conversion(self.matchings[name])
                    )
                    self.log.info(f"Dataset {name!r}:{summary}")
        finally:
            if api is None:
                self._api = None  # release a graph loaded here
        self._tables = {
            name: feature_table(self.feature_names[name], self.matchings[name], self.gene_names[name])
            for name in self.data_h5ad_dict
        }

    @staticmethod
    def _integer(value: Any, name: str) -> int:
        """Return *value* as an ``int`` release, or raise naming the argument."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"`{name}` must be an integer Ensembl release, got {value!r}.")
        return int(value)

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

    def _graph_files(self, repository: str, organism: str | None = None) -> list[dict[str, Any]]:
        """Return the graph files of *repository* that :py:meth:`idtrack.API.build_graph` would load.

        Those are the files :py:meth:`idtrack.GraphMaker.list_graph_files` lists with the window and the content the
        API builds: from the organism's first release (:py:meth:`idtrack.DB.first_release`) to a snapshot release,
        the narrow history, every form, the selected external databases, and as primary assembly the organism's
        preferred one that has the snapshot release.

        Args:
            repository: Folder to look in.
            organism: Only graphs of this organism; ``None`` for all.

        Returns:
            list[dict]: The entries of :py:meth:`idtrack.GraphMaker.list_graph_files`, sorted by file name.
        """
        found = []
        for entry in GraphMaker.list_graph_files(repository, organism):
            snapshot = entry["ignore_after"]
            if (
                entry["organism"] not in DB.supported_organisms
                or snapshot is None
                or entry["ignore_before"] != DB.first_release(entry["organism"])
                or not entry["narrow"]
                or entry["forms"] is not None
                or not entry["narrow_external"]
            ):
                continue
            for assembly in DB.assemblies_by_priority(entry["organism"]):
                first, last = DB.release_range(entry["organism"], assembly)
                if first <= snapshot and (last is None or snapshot <= last):
                    if assembly == entry["genome_assembly"]:
                        found.append(entry)
                    break
        return found

    def _graph_snapshots(self) -> dict[str, list[int]]:
        """Return ``{organism: [snapshot releases]}`` of the graphs :py:meth:`idtrack.API.build_graph` would load.

        Returns:
            dict[str, list[int]]: Sorted snapshot releases per organism with at least one such graph
            (:py:meth:`_graph_files`).
        """
        found: dict[str, list[int]] = {}
        for entry in self._graph_files(self.idtrack_local_repository):
            found.setdefault(entry["organism"], []).append(int(entry["ignore_after"]))
        return {organism: sorted(releases) for organism, releases in sorted(found.items())}

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
        _, latest_release = API(self.idtrack_local_repository).resolve_organism(self.organism_name)
        return int(latest_release)

    # ----------------------------------------------------------------------------------------------------------------
    # Conversions and their cache

    def _graph_identity(self) -> dict[str, Any]:
        """Return what identifies the graph the conversions come from (see "Result cache" in the class docs).

        The graph file of the organism and snapshot in the repository of the API (given or own): its name, size and
        modification time, so that the same graph identifies the same way whether it is loaded or not. A loaded graph
        without such a file (e.g. built elsewhere) is identified by a digest of its metadata.
        """
        repository = (
            self._api.local_repository if self._api_given and self._api is not None else self.idtrack_local_repository
        )
        for entry in self._graph_files(repository, self.organism_name):
            if entry["ignore_after"] == self.graph_last_ensembl_release:
                stat = os.stat(entry["path"])
                return {
                    "graph_file": os.path.basename(entry["path"]),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        if self._api is not None and self._api.track is not None:
            metadata = {
                k: v
                for k, v in self._api.track.graph.graph.items()
                if isinstance(v, (str, int, float, bool)) or (isinstance(v, (list, tuple)) and len(v) < 1000)
            }
            digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()
            return {"loaded_graph_metadata_sha256": digest}
        return {"graph_file": None}

    def result_cache_path(self, dataset_name: str) -> str:
        """Return the path of the cached conversions of one dataset (a JSON file in the project folder).

        Args:
            dataset_name: Key of :py:attr:`data_h5ad_dict`.

        Returns:
            str: ``<project folder>/idtrack_result_<project>_<dataset>.json`` with both names made safe for file
            names (characters other than letters, digits, ``.`` and ``-`` replaced, and a short hash added).
        """
        return os.path.join(
            self.project_local_repository,
            f"idtrack_result_{_slug(self.project_name)}_{_slug(dataset_name)}.json",
        )

    def _cache_header(self, dataset_name: str) -> dict[str, Any]:
        """Return everything a dataset's conversions depend on (compared with a stored file before it is used)."""
        names = self.feature_names[dataset_name]
        return {
            "format": _CACHE_FORMAT,
            "idtrack_version": idtrack.__version__,
            "organism": self.organism_name,
            "final_database": self.final_database,
            "target_ensembl_release": self.target_ensembl_release,
            "source_release": self.source_releases.get(dataset_name),
            "source_database": self.source_databases.get(dataset_name),
            "graph_snapshot": self.graph_last_ensembl_release,
            "graph": self._graph_identity(),
            "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        }

    def _dataset_conversions(self, dataset_name: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Return the matchings and gene names of a dataset, from a valid result cache or computed."""
        path = self.result_cache_path(dataset_name)
        header = self._cache_header(dataset_name)
        try:
            with open(path) as handle:
                stored = json.load(handle)
            if stored.get("header") == header:
                return stored["matchings"], stored["gene_names"]
            self.log.info(f"Dataset {dataset_name!r}: the cached conversions are of other inputs; recomputing.")
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.log.warning(
                f"Dataset {dataset_name!r}: the result cache {path!r} is unreadable ({error}); recomputing."
            )
        api = self._require_api()
        header = self._cache_header(dataset_name)  # the graph is loaded now: identify it as loaded
        matchings, gene_names = self._convert_dataset(api, dataset_name)
        _write_json_atomically(path, {"header": header, "matchings": matchings, "gene_names": gene_names})
        return matchings, gene_names

    def _require_api(self) -> API:
        """Return the API, loading the graph first if this harmonizer has none yet."""
        if self._api is None:
            api = API(local_repository=self.idtrack_local_repository)
            api.build_graph(organism_name=self.organism_name, snapshot_release=self.graph_last_ensembl_release)
            self._api = api
        return self._api

    def _convert_dataset(self, api: API, dataset_name: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Convert the features of one dataset, then name every gene reached (see the class docs).

        The names of a gene in the final database depend on the gene, the database and the release only, so they are
        read from the gene itself (:py:meth:`idtrack.Track.gene_to_database`: a few look-ups per gene, no search),
        whatever feature led to the gene and whatever name that feature's own conversion returned.

        Args:
            api: The API with the graph loaded.
            dataset_name: Key of :py:attr:`data_h5ad_dict`.

        Returns:
            tuple: The matchings (one per feature, in file order) and ``{gene: final-database names}`` (sorted; empty
            for a gene without an id in the final database).
        """
        matchings = api.convert_identifier_multiple(
            self.feature_names[dataset_name],
            from_release=self.source_releases.get(dataset_name),
            from_database=self.source_databases.get(dataset_name),
            to_release=self.target_ensembl_release,
            final_database=self.final_database,
            strategy="best",
            verbose=self.verbose_level > 0,
            pbar_prefix=dataset_name,
        )
        genes = sorted({gene for m in matchings for gene, _ in m["last_node"]})
        if self.final_database == DB.nts_ensembl[DB.backbone_form]:
            return matchings, {gene: [gene] for gene in genes}
        track = api._require_track()
        gene_names = {
            gene: track.gene_to_database(gene, self.final_database, self.target_ensembl_release)[0] for gene in genes
        }
        return matchings, gene_names

    # ----------------------------------------------------------------------------------------------------------------
    # Reports

    def feature_table(self, dataset_name: str) -> pd.DataFrame:
        """Return how each input feature of a dataset was converted (see :py:func:`feature_table`).

        Args:
            dataset_name: Key of :py:attr:`data_h5ad_dict`.

        Returns:
            pandas.DataFrame: A new frame, one row per input feature in file order.
        """
        return self._tables[dataset_name].copy()

    def conversion_report(self) -> pd.DataFrame:
        """Return how every input feature of every dataset was converted.

        Returns:
            pandas.DataFrame: One row per input feature: ``dataset``, ``query_id`` and the columns of
            :py:func:`feature_table`, in the order of :py:attr:`data_h5ad_dict` and of each file.
        """
        frames = []
        for name, table in self._tables.items():
            frame = table.reset_index(names="query_id")
            frame.insert(0, "dataset", name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def output_plan(self, dataset_name: str, merge: str = "flag", keep_unconverted: bool = True) -> pd.DataFrame:
        """Return the output features of a dataset (see :py:func:`output_plan`).

        Args:
            dataset_name: Key of :py:attr:`data_h5ad_dict`.
            merge: ``"flag"`` (default) or ``"sum"``.
            keep_unconverted: Keep the unconverted features (default).

        Returns:
            pandas.DataFrame: One row per output feature.
        """
        if merge not in ("flag", "sum"):
            raise ValueError(f"`merge` must be 'flag' or 'sum', got {merge!r}.")
        return output_plan(self._tables[dataset_name], merge, keep_unconverted)

    # ----------------------------------------------------------------------------------------------------------------
    # Matrices

    def feature_harmonizer(
        self, dataset_name: str, merge: Literal["flag", "sum"] = "flag", keep_unconverted: bool = True
    ) -> tuple[ad.AnnData, int, int]:
        """Return one dataset with its features converted (see :py:class:`HarmonizeFeatures` for the rules).

        The ``.h5ad`` file is read completely and never altered. ``X`` and every layer get the output features
        (columns selected, merged ones summed); ``obs``, ``obsm``, ``obsp`` and ``uns`` are kept as they are; ``varm``
        and ``varp`` are subset to the output features, or left out when features were summed. ``var`` holds the
        columns ``<final database>``, ``"<final database>_alternatives"``, ``"Query ID"``, ``"matching_ensembl_gene"``,
        ``"idtrack_status"``, ``"idtrack_reason"`` and then the input's own columns (an input column with one of those
        names gets the suffix ``"_input"``; for a merged feature, an input value is kept where all its input features
        agree and is missing otherwise). ``raw`` is dropped with a warning, because its features are not converted.

        Args:
            dataset_name (str): Key of :py:attr:`data_h5ad_dict`.
            merge (Literal["flag", "sum"]): How several input features of one gene are handled (default ``"flag"``).
            keep_unconverted (bool): Keep the features that are not converted, under their input names (default).

        Returns:
            tuple: The harmonised :py:class:`anndata.AnnData`, the number of input features and the number of output
            features.
        """
        plan = self.output_plan(dataset_name, merge, keep_unconverted)
        adata = ad.read_h5ad(self.data_h5ad_dict[dataset_name])
        if list(adata.var_names) != self.feature_names[dataset_name]:
            raise ValueError(f"Dataset {dataset_name!r} changed on disk since the harmonizer read its feature names.")
        t0 = adata.n_vars
        if adata.raw is not None:
            self.log.warning(
                f"Dataset {dataset_name!r}: `raw` is dropped, because its features are not harmonised; "
                "keep raw counts in a layer to carry them through."
            )
        summed = any(len(s) > 1 for s in plan["sources"])
        if summed and (len(adata.varm) or len(adata.varp)):
            self.log.warning(
                f"Dataset {dataset_name!r}: `varm` and `varp` are left out, because input features were summed."
            )
        var = self._harmonized_var(adata.var, plan)
        out = ad.AnnData(
            X=None if adata.X is None else _aggregate(adata.X, plan, t0),
            obs=adata.obs,
            var=var,
            uns=adata.uns,
            obsm=adata.obsm,
            obsp=adata.obsp,
            layers={k: _aggregate(v, plan, t0) for k, v in adata.layers.items() if k is not None},  # None is X
        )
        if not summed:
            columns = [s[0] for s in plan["sources"]]
            for key, value in adata.varm.items():
                out.varm[key] = value[columns]  # type: ignore[index]
            for key, value in adata.varp.items():
                out.varp[key] = value[columns][:, columns]  # type: ignore[index]
        return out, t0, out.n_vars

    def _harmonized_var(self, input_var: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
        """Return the ``var`` of a harmonised dataset (see :py:meth:`feature_harmonizer`)."""
        alternatives = f"{self.final_database}_alternatives"
        generated = pd.DataFrame(
            {
                self.final_database: plan["final_id"].astype(object),
                alternatives: plan["final_id_alternatives"].astype(object),
                self._QUERY_COLUMN: plan["query_ids"].astype(object),
                self._ENSEMBL_GENE_COLUMN: plan["ensembl_gene"].astype(object),
                self._STATUS_COLUMN: pd.Categorical(plan["status"], categories=list(STATUSES)),
                self._REASON_COLUMN: plan["reason"].astype(object),
            },
            index=plan.index,
        )
        sources = plan["sources"].tolist()
        first = [s[0] for s in sources]
        carried = input_var.iloc[first].copy()
        carried.index = plan.index
        for position, group in enumerate(sources):
            if len(group) > 1:
                block = input_var.iloc[group]
                for column in input_var.columns:
                    if block[column].astype(object).nunique(dropna=False) > 1:
                        carried[column] = _holding_missing(carried[column])
                        carried.iloc[position, carried.columns.get_loc(column)] = None
        carried.columns = [f"{c}_input" if c in generated.columns else c for c in carried.columns]
        return pd.concat([generated, carried], axis=1)

    def unify_multiple_anndatas(
        self,
        mode: Literal["union", "intersect"] = "union",
        obs_join: Literal["outer", "inner"] = "outer",
        obs_columns_to_keep: list[str] | None = None,
        numeric_var_columns: set[str] | None = None,
        numeric_obs_columns: set[str] | None = None,
        handle_anndata_key: str = "handle_anndata",
        merge: Literal["flag", "sum"] = "flag",
        keep_unconverted: bool = True,
    ) -> ad.AnnData:
        """Merge the harmonised datasets of :py:attr:`data_h5ad_dict` into one :py:class:`~anndata.AnnData`.

        Each dataset is harmonised with :py:meth:`feature_harmonizer` (with *merge* and *keep_unconverted*) and all
        are concatenated along the cells in one :py:func:`anndata.concat`. A feature is the same feature in two
        datasets when it has the same name: a converted gene id, or the same unconverted input name (an unconverted
        name equal to another dataset's converted gene id is renamed ``"unconverted:<name>"`` first). By default
        nothing is lost, and values are never rewritten: text stays text, numbers stay numbers, and where a dataset
        has no value the result holds a missing value of the column's kind (``NaN`` for floats and text, ``pd.NA``
        in a nullable integer or boolean column, ``NaT`` for times, a missing category for categoricals), never a
        placeholder string.

        * **Features.** ``"union"`` (default) keeps every feature of any dataset; a dataset that lacks a feature gets
          zeros for it in ``X`` and in every layer. ``"intersect"`` keeps the features present in every dataset.

        * **.obs.** ``obs_join="outer"`` (default) keeps every ``obs`` column of any dataset; in the cells of a dataset
          without the column it is missing (see above; an integer or boolean column becomes the nullable ``Int``/
          ``boolean`` dtype so that it can hold the missing values). ``obs_join="inner"`` keeps only the columns every
          dataset has. *obs_columns_to_keep*, if given, selects among the joined columns (in its order); a name that
          no dataset has, or (with ``"inner"``) that not every dataset has, raises ``ValueError`` naming it. Columns in
          *numeric_obs_columns* are converted to ``float`` (missing values allowed); other text columns become
          categorical (values unchanged). The column *handle_anndata_key* records the dataset of each cell
          (categorical, in the order of :py:attr:`data_h5ad_dict`), and cells are named ``f"{handle}_{cell}"``.

        * **.var.** ``converted_id_column`` holds the final-database name of each feature (missing for features
          without one), ``"matching_ensembl_gene"`` its gene. Per dataset: ``f"{column}_{handle}"`` for ``"Query
          ID"``, ``"idtrack_status"`` and ``"idtrack_reason"`` (missing where the dataset lacks the feature). In
          ``"union"`` mode also ``f"{converted_id_column}_{handle}"`` and an integer ``"intersection"`` column (1 if
          the feature is in every dataset, by presence, whatever its name). Every other ``var`` column of the inputs is
          kept: as **one** column when, for every feature, the datasets that have a value for it agree; otherwise
          **per dataset** as ``f"{column}_{handle}"`` for each dataset that has it. Columns in *numeric_var_columns*
          are converted to ``float``; other text columns become categorical (values unchanged).

        * **layers, obsm, obsp, uns.** Handled by :py:func:`anndata.concat`: a layer or ``obsm`` entry present in
          every dataset is concatenated (``obsp`` block-diagonally), and an ``uns`` entry is kept when it is the same
          in every dataset. ``varm``, ``varp`` and ``raw`` are not carried into the merged object.

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
            merge (Literal["flag", "sum"]): See :py:meth:`feature_harmonizer`.
            keep_unconverted (bool): See :py:meth:`feature_harmonizer`.

        Returns:
            anndata.AnnData: The merged object; cells in the order of :py:attr:`data_h5ad_dict`, then of each file.

        Raises:
            ValueError: If *mode*, *obs_join* or *merge* is not one of its values; if a name of *obs_columns_to_keep*
                is not among the joined columns, *handle_anndata_key* is already an ``obs`` column, or a numeric column
                is absent or holds a value that is not a number; if two cells get the same name; if a generated
                ``var`` column name clashes with an input column.
        """
        if mode not in ("union", "intersect"):
            raise ValueError(f"`mode` must be 'union' or 'intersect', got {mode!r}.")
        if obs_join not in ("outer", "inner"):
            raise ValueError(f"`obs_join` must be 'outer' or 'inner', got {obs_join!r}.")
        if merge not in ("flag", "sum"):
            raise ValueError(f"`merge` must be 'flag' or 'sum', got {merge!r}.")
        numeric_obs_columns = set(numeric_obs_columns or set())
        numeric_var_columns = set(numeric_var_columns or set())
        handles = list(self.data_h5ad_dict)

        studies: list[ad.AnnData] = []
        for handle in tqdm.tqdm(handles, ncols=140, disable=self.verbose_level == 0):
            adata_study, t0, t1 = self.feature_harmonizer(handle, merge=merge, keep_unconverted=keep_unconverted)
            self.log.info(f"Dataset {handle!r}: {t1} output features from {t0} input features.")
            studies.append(adata_study)

        converted_anywhere = set()
        for study in studies:
            converted_anywhere |= set(
                study.var_names[~study.var[self._STATUS_COLUMN].astype(str).str.startswith("unconverted")]
            )
        for study in studies:
            unconverted = study.var[self._STATUS_COLUMN].astype(str).str.startswith("unconverted").to_numpy()
            clash = unconverted & study.var_names.isin(converted_anywhere)
            if clash.any():
                study.var_names = pd.Index(
                    np.where(clash, "unconverted:" + study.var_names.astype(str), study.var_names.astype(str))
                )

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
            for key in list(study.varm):
                del study.varm[key]
            for key in list(study.varp):
                del study.varp[key]

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
            ValueError: If a generated column name is also an input column.
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

        def _same_everywhere(column: str) -> pd.Series:
            # A gene's final-database names are a property of the gene (the same in every dataset): they agree.
            merged = self._agreed_values({h: a[column] for h, a in aligned.items()}, var_names)
            if merged is None:
                raise AssertionError(f"The datasets disagree on {column!r} for a feature; this is an internal error.")
            return merged

        names = {h: a[self.final_database] for h, a in aligned.items()}
        alternatives = f"{self.final_database}_alternatives"
        _add(self.converted_id_column, _same_everywhere(self.final_database))
        # The gene of a merge partner (kept under its input name) can differ between datasets of different source
        # releases: then the column is kept per dataset, like an input column.
        genes = {h: a[self._ENSEMBL_GENE_COLUMN] for h, a in aligned.items()}
        merged_genes = self._agreed_values(genes, var_names)
        if merged_genes is not None:
            _add(self._ENSEMBL_GENE_COLUMN, merged_genes)
        else:
            for h, g in genes.items():
                _add(f"{self._ENSEMBL_GENE_COLUMN}_{h}", g)
        _add(alternatives, _same_everywhere(alternatives))
        for column in (self._QUERY_COLUMN, self._STATUS_COLUMN, self._REASON_COLUMN):
            for h, a in aligned.items():
                _add(f"{column}_{h}", a[column])
        if mode == "union":
            for h, s in names.items():
                _add(f"{self.converted_id_column}_{h}", s)
            present = pd.DataFrame({h: var_names.isin(v.index) for h, v in var_frames.items()}, index=var_names)
            _add("intersection", present.all(axis=1).astype(int))

        generated = {
            self.final_database,
            alternatives,
            self._QUERY_COLUMN,
            self._ENSEMBL_GENE_COLUMN,
            self._STATUS_COLUMN,
            self._REASON_COLUMN,
        }
        input_columns: list[str] = []
        for v in var_frames.values():
            input_columns += [c for c in v.columns if c not in generated and c not in input_columns]
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
