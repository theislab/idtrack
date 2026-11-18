"""The pure core of harmonization: how the features of one dataset convert, and which output features they give.

Functions of their arguments alone (no files, no AnnData): :py:func:`convert_features` converts feature names with an
:py:class:`idtrack.API` and names every gene reached; :py:func:`feature_table` classifies each input feature by its
conversion; :py:func:`output_plan` says which output feature each input becomes. A web service can serve them for a
list of names; :py:class:`idtrack.harmonize.HarmonizeFeatures` applies them to ``.h5ad`` files.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from idtrack import ids
from idtrack.errors import IntegrityError
from idtrack.results import ConversionResult

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


def convert_features(
    api: Any,
    feature_names: list[str],
    to_release: int,
    final_database: str,
    from_release: int | None = None,
    from_database: str | None = None,
    progress: bool = False,
) -> tuple[list[ConversionResult], dict[str, list[str]]]:
    """Convert the feature names of one dataset, then name every gene reached in the final database.

    The names of a gene in the final database depend on the gene, the database and the release only, so they are
    read from the gene itself (:py:meth:`idtrack.API.gene_to_database`: a few look-ups per gene, no search), whatever
    feature led to the gene and whatever name that feature's own conversion returned.

    Args:
        api: An :py:class:`idtrack.API` with the graph.
        feature_names: The feature names.
        to_release: The target release.
        final_database: The final database.
        from_release: The release the names come from; ``None``: read each where the graph holds it.
        from_database: The external database the names are meant as; ``None``: any.
        progress: Show a progress bar.

    Returns:
        tuple: The results (one per feature, in order) and ``{gene: final-database names}`` (sorted; empty for a gene
        without an id in the final database).
    """
    results = api.convert_many(
        feature_names,
        from_release=from_release,
        from_database=from_database,
        to_release=to_release,
        final_database=final_database,
        progress=progress,
    )
    genes = sorted({candidate.gene for result in results for candidate in result.view("best")})
    if final_database == ids.GENE:
        return results, {gene: [gene] for gene in genes}
    return results, {gene: api.gene_to_database(gene, final_database, to_release)[0] for gene in genes}


def feature_table(
    feature_names: list[str],
    matchings: list[ConversionResult],
    gene_names: dict[str, list[str]],
) -> pd.DataFrame:
    """Classify each input feature of one dataset by its conversion; a pure function of that dataset alone.

    Args:
        feature_names: The dataset's feature names, in file order.
        matchings: :py:meth:`idtrack.API.convert` results, one per feature, in the same order (their best view
            counts).
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
        if m.query_id != name:
            raise IntegrityError(f"Matching of {m.query_id!r} given for feature {name!r}.")
        genes = sorted({candidate.gene for candidate in m.view("best")})
        row: dict[str, Any] = {
            "status": None,
            "reason": None,
            "ensembl_gene": None,
            "final_id": None,
            "final_id_alternatives": None,
            "own": False,
        }
        if m.error is not None:
            row.update(status="error", reason=f"{m.error.type}: {m.error.message}")
        elif m.graph_id is None:
            row.update(status="not_in_graph", reason="not found in the graph")
        elif not m.candidates or not genes:
            row.update(status="no_conversion", reason=f"no gene at release {m.to_release}")
        elif len(genes) > 1:
            row.update(status="ambiguous", reason=f"maps to {len(genes)} genes: {';'.join(genes)}")
        else:
            gene = genes[0]
            names = sorted(gene_names.get(gene, []))
            resolved = ids.stable_id_of(m.graph_id or name).casefold()
            row.update(
                ensembl_gene=gene,
                final_id=names[0] if names else None,
                final_id_alternatives=";".join(names[1:]) or None,
                own=resolved in {ids.stable_id_of(gene).casefold(), *(n.casefold() for n in names)},
            )
            if m.from_release_fallback:
                row["reason"] = f"not in the graph at release {m.from_release_given}; read at release {m.from_release}"
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
        raise IntegrityError(f"Output features named twice: {plan.index[plan.index.duplicated()].tolist()[:10]}.")
    return plan


__all__ = ["STATUSES", "convert_features", "feature_table", "output_plan"]
