"""Harmonization of ``.h5ad`` datasets: the features of every dataset converted to one release and database.

Needs the ``harmonize`` extra (``pip install 'idtrack[harmonize]'``: anndata and scipy).
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Literal

try:
    import anndata as ad
    import scipy.sparse as sp
except ImportError as error:  # pragma: no cover - exercised by the installed-wheel job without the extra
    raise ImportError(
        "idtrack.harmonize.HarmonizeFeatures needs anndata and scipy, which come with the extra `harmonize`: "
        "pip install 'idtrack[harmonize]'."
    ) from error
import h5py
import numpy as np
import pandas as pd
import tqdm

from idtrack import _arguments, organisms
from idtrack._atomic import write_text
from idtrack.api import API
from idtrack.errors import IntegrityError, InvalidArgumentError
from idtrack.graph import files
from idtrack.graph.files import list_graph_files
from idtrack.harmonize.core import (
    STATUSES,
    _holding_missing,
    _is_text,
    _missing_like,
    convert_features,
    feature_table,
    output_plan,
)
from idtrack.results import ConversionResult
from idtrack.settings import RULES_VERSION

_CACHE_FORMAT = 4


def read_feature_names(path: str) -> list[str]:
    """Return the feature names (the ``var`` index) of an ``.h5ad`` file without reading anything else.

    The file is opened read-only with h5py and only the index of ``var`` is read: neither the matrix nor the other
    annotations are loaded, so a dataset of any size costs the size of its feature names.
    :py:class:`idtrack.harmonize.HarmonizeFeatures` reads each dataset's features so.

    Args:
        path: Path of the ``.h5ad`` file.

    Returns:
        list[str]: The names in file order (bytes decoded).
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
        raise InvalidArgumentError(f"Dataset {dataset_name!r} has no features.")
    bad = [n for n in names if not isinstance(n, str) or not n.strip()]
    if bad:
        raise InvalidArgumentError(
            f"Dataset {dataset_name!r} has feature names that are empty or not strings: {bad[:10]}."
        )
    repeated = pd.Index(names)
    repeated = repeated[repeated.duplicated()].unique().tolist()
    if repeated:
        raise InvalidArgumentError(
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
    write_text(path, json.dumps(content))


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
    converted with :py:meth:`idtrack.API.convert_many` to its Ensembl gene at
    *to_release*. The conversion of a dataset depends only on that dataset (its feature names and its
    *from_release_by_dataset* and *from_database_by_dataset* entries), never on the other datasets.

    **Output features.** Each converted gene is one feature, named by the gene id at the target release (with
    version); its final-database name (default: the organism's symbol database) is in the column named after the
    final database. That name is a property of the gene, not of the query: the gene's ids in the final database at
    the target release (:py:meth:`idtrack.engine.Track.gene_to_database`: read from the gene, no search), the first in
    sorted order when there are several (the others in ``"<final database>_alternatives"``). A gene without an id in
    the final database keeps the gene id as its feature name and a missing final-database name (status ``"converted_no_final_id"``):
    the final-database column never holds an Ensembl id. Which gene a feature is, is decided as Ensembl genes
    (:py:meth:`idtrack.API.convert`: a target in a final database is an id of the gene answered), so a feature whose
    gene has no name there is that gene, never another gene that has one. Features that are not converted keep their
    input name; no feature is dropped silently. Every output feature has, in ``var``:

    * ``"idtrack_status"``: ``"converted"``, ``"converted_no_final_id"``, ``"merged"``, ``"merged_no_final_id"``
      (with ``merge="sum"``, the sum of several input features), ``"unconverted_merge_partner"`` (with
      ``merge="flag"``, an input feature that shares its gene with others), ``"unconverted_ambiguous"`` (several
      genes), ``"unconverted_no_conversion"`` (in the graph, but no gene at the target release),
      ``"unconverted_not_in_graph"``, ``"unconverted_missing_value"`` (a spelling of a missing value, ``nan`` or
      ``NA``, which is never a gene) or ``"unconverted_error"`` (the conversion of this feature raised an
      error that concerns it alone, a search over its budget);
    * ``"idtrack_reason"``: why (missing for a plain conversion; the text of the conversion's reason,
      :py:data:`idtrack.results.REASONS`, for a feature without a conversion; for a feature that the graph does not
      hold at its dataset's source release and that was therefore read at another release, that release; for a
      feature that is another feature of the dataset with a suffix a single-cell tool appends to repeated names, the
      feature it copies);
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
    dataset (written to a temporary file and renamed). A stored file is used only if it was written under the same
    rules of conversion (:py:data:`idtrack.settings.RULES_VERSION`, raised with every change of an answer) for the same
    graph (its content digest, read from the graph file's record or from the loaded graph) and the same feature
    names, organism, final database, target and source release and graph snapshot; an unreadable or mismatching file
    is recomputed and replaced. Conversions with a graph that records no digest (made in memory) are not cached.

    Args:
        project_name (str): Label used in log messages and cache file names.
        data_h5ad_dict (dict[str, str]): ``{dataset name: path of an .h5ad file}``; at least one.
        project_local_repository (str): Writable folder for the result cache (created if missing).
        idtrack_local_repository (str): idtrack's local repository holding the graph.
        to_release (int): Ensembl release all features are converted to; a release of the graph.
        final_database (Optional[str]): Database of the feature names in the output (e.g. ``"HGNC Symbol"``).
            Defaults to the organism's first symbol database (:py:attr:`idtrack.organisms.SYMBOL_DATABASES`: HGNC Symbol for
            human, MGI Symbol for mouse, VGNC Symbol for pig and the macaques).
        organism (Optional[str]): Organism, as an Ensembl species name (e.g. ``"homo_sapiens"``) or any query
            :py:meth:`idtrack.API.ensembl_organism` accepts. If ``None``, it is read from the graph files in
            *idtrack_local_repository*, which must then hold graphs of exactly one organism.
        snapshot_release (Optional[int]): Snapshot release of the graph. If ``None``, the newest graph of
            the organism in *idtrack_local_repository* (built by :py:meth:`idtrack.API.build`) is used, or the
            latest Ensembl release of the organism when there is none.
        from_release_by_dataset (Optional[dict[str, int]]): The Ensembl release each dataset was annotated with, by dataset
            name. A dataset with a release is converted with ``from_release``; a feature that the graph does not hold
            at that release is read where the graph holds it (the rule without a release, below), and its
            ``"idtrack_reason"`` says so. Without a release, each feature is read at the target release if it is
            active there, else at the latest release in which it is active.
        from_database_by_dataset (Optional[dict[str, str]]): The external database the feature names of a dataset are ids
            of, by dataset name (``{"study_a": "HGNC Symbol"}``; one of
            :py:meth:`idtrack.API.external_databases`). A dataset with a database is converted with
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
        verbosity (Literal[0, 1, 2, 3]): 0 shows nothing, 1 progress bars, 2 also a summary of each dataset's
            conversions (logged at INFO by the ``"idtrack"`` loggers; see :py:meth:`idtrack.API.configure_logger`),
            3 also debug messages. The harmonizer never configures logging itself.
        converted_id_column (str): Name of the merged final-database column of :py:meth:`unify_multiple_anndatas`.

    Attributes:
        feature_names (dict[str, list[str]]): The feature names of each dataset, in file order.
        matchings (dict[str, list[idtrack.results.ConversionResult]]): The conversion of each feature of each
            dataset, one result per feature in file order (:py:meth:`idtrack.API.convert_many`, or read back from the
            result cache).
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
        to_release: int,
        final_database: str | None = None,
        organism: str | None = None,
        snapshot_release: int | None = None,
        from_release_by_dataset: dict[str, int] | None = None,
        from_database_by_dataset: dict[str, str] | None = None,
        api: API | None = None,
        verbosity: Literal[0, 1, 2, 3] = 2,
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
            to_release (int): See :py:class:`HarmonizeFeatures`.
            final_database (Optional[str]): See :py:class:`HarmonizeFeatures`.
            organism (Optional[str]): See :py:class:`HarmonizeFeatures`.
            snapshot_release (Optional[int]): See :py:class:`HarmonizeFeatures`.
            from_release_by_dataset (Optional[dict[str, int]]): See :py:class:`HarmonizeFeatures`.
            from_database_by_dataset (Optional[dict[str, str]]): See :py:class:`HarmonizeFeatures`.
            api (Optional[idtrack.API]): See :py:class:`HarmonizeFeatures`.
            verbosity (Literal[0, 1, 2, 3]): See :py:class:`HarmonizeFeatures`.
            converted_id_column (str): See :py:class:`HarmonizeFeatures`.

        Raises:
            idtrack.errors.ArgumentTypeError: If a release is not an integer.
            idtrack.errors.InvalidArgumentError: If *verbosity* is not 0 to 3; if *data_h5ad_dict* is empty, names a
                missing file or a dataset whose feature names are empty, not strings or repeated; if
                *from_release_by_dataset* or *from_database_by_dataset* names an unknown dataset; if a source database
                is not a string, or *to_release* is after the graph snapshot; if *organism* is ``None`` and the graphs
                in *idtrack_local_repository* do not name exactly one organism; if *final_database* is ``None`` and the
                organism declares no symbol database. Also the errors of :py:meth:`idtrack.API.convert_many` (e.g. a
                release or database not of the graph).
        """
        if verbosity not in (0, 1, 2, 3):
            raise InvalidArgumentError(
                f"`verbosity` must be 0, 1, 2 or 3, got {verbosity!r}.", argument="verbosity", value=verbosity
            )
        self.verbosity = verbosity
        self.log = logging.getLogger(__name__)
        if not isinstance(data_h5ad_dict, dict) or not data_h5ad_dict:
            raise InvalidArgumentError(
                "`data_h5ad_dict` must be a non-empty dict {dataset name: .h5ad path}.", argument="data_h5ad_dict"
            )
        missing = {k: v for k, v in data_h5ad_dict.items() if not os.path.isfile(v)}
        if missing:
            raise InvalidArgumentError(
                f"`data_h5ad_dict` names files that do not exist: {missing}.", argument="data_h5ad_dict", value=missing
            )
        self.project_name = str(project_name)
        self.data_h5ad_dict = dict(data_h5ad_dict)
        self.project_local_repository = project_local_repository
        self.idtrack_local_repository = idtrack_local_repository
        self.converted_id_column = converted_id_column
        self.to_release = _arguments.release(to_release, "to_release")
        self.from_release_by_dataset = {
            k: _arguments.release(v, f"from_release_by_dataset[{k!r}]")
            for k, v in (from_release_by_dataset or {}).items()
            if v is not None
        }
        unknown = sorted(set(self.from_release_by_dataset) - set(self.data_h5ad_dict))
        if unknown:
            raise InvalidArgumentError(
                f"`from_release_by_dataset` names datasets that are not in `data_h5ad_dict`: {unknown}.",
                argument="from_release_by_dataset",
                value=unknown,
            )
        self.from_database_by_dataset = {k: v for k, v in (from_database_by_dataset or {}).items() if v is not None}
        unknown = sorted(set(self.from_database_by_dataset) - set(self.data_h5ad_dict))
        if unknown:
            raise InvalidArgumentError(
                f"`from_database_by_dataset` names datasets that are not in `data_h5ad_dict`: {unknown}.",
                argument="from_database_by_dataset",
                value=unknown,
            )
        not_names = {k: v for k, v in self.from_database_by_dataset.items() if not isinstance(v, str) or not v.strip()}
        if not_names:
            raise InvalidArgumentError(
                f"`from_database_by_dataset` must name one external database per dataset, got {not_names}.",
                argument="from_database_by_dataset",
                value=not_names,
            )

        if api is not None:
            self.organism = str(api.graph.organism)
            self.snapshot_release = int(api.graph.snapshot_release)
            given = {"organism": organism, "snapshot_release": snapshot_release}
            if organism is not None and self._resolve_organism(organism) != self.organism:
                raise InvalidArgumentError(
                    f"`organism`={organism!r} differs from the graph of `api` ({self.organism}).",
                    argument="organism",
                    value=organism,
                )
            if snapshot_release is not None and snapshot_release != self.snapshot_release:
                raise InvalidArgumentError(
                    f"`snapshot_release`={given['snapshot_release']!r} differs from the snapshot of "
                    f"the graph of `api` ({self.snapshot_release}).",
                    argument="snapshot_release",
                    value=snapshot_release,
                )
        else:
            self.organism = self._resolve_organism(organism)
            self.snapshot_release = (
                _arguments.release(snapshot_release, "snapshot_release")
                if snapshot_release is not None
                else self._default_snapshot_release()
            )
        if self.to_release > self.snapshot_release:
            raise InvalidArgumentError(
                f"`to_release`={self.to_release} is after the graph snapshot {self.snapshot_release}.",
                argument="to_release",
                value=self.to_release,
            )
        self.final_database = (
            final_database if final_database is not None else self.default_final_database(self.organism)
        )

        os.makedirs(self.project_local_repository, exist_ok=True)
        if not os.access(self.project_local_repository, os.W_OK):
            raise InvalidArgumentError(f"`project_local_repository` {self.project_local_repository!r} is not writable.")

        self.feature_names: dict[str, list[str]] = {
            name: _validated_feature_names(name, read_feature_names(path)) for name, path in self.data_h5ad_dict.items()
        }
        self._api = api
        self.matchings: dict[str, list[ConversionResult]] = {}
        self.gene_names: dict[str, dict[str, list[str]]] = {}
        try:
            for name in self.data_h5ad_dict:
                self.matchings[name], self.gene_names[name] = self._dataset_conversions(name)
                if self.verbosity > 1:
                    summary = API.classification_summary(API.classify(self.matchings[name]))
                    self.log.info(f"Dataset {name!r}:{summary}")
        finally:
            if api is None:
                self._api = None  # release a graph loaded here
        self._tables = {
            name: feature_table(self.feature_names[name], self.matchings[name], self.gene_names[name])
            for name in self.data_h5ad_dict
        }

    @staticmethod
    def default_final_database(organism: str) -> str:
        """Return the organism's default final database: its first entry in :py:attr:`idtrack.organisms.SYMBOL_DATABASES`.

        Args:
            organism: Ensembl species name of a registered organism.

        Returns:
            str: E.g. ``"HGNC Symbol"`` for human, ``"VGNC Symbol"`` for rhesus macaque.

        Raises:
            ValueError: If the organism is not registered or declares no symbol database.
        """
        databases = organisms.SYMBOL_DATABASES.get(organism)
        if not databases:
            raise InvalidArgumentError(
                f"No symbol database is declared for organism {organism!r} (`organisms.SYMBOL_DATABASES`); "
                "pass `final_database` explicitly."
            )
        return str(databases[0])

    def _graph_files(self, repository: str, organism: str | None = None) -> list[dict[str, Any]]:
        """Return the graph files of *repository* that :py:meth:`idtrack.API.build` would load.

        Those are the files :py:func:`idtrack.graph.files.list_graph_files` lists with the window and the content the
        API builds: from the organism's first release (:py:meth:`idtrack.organisms.first_release`) to a snapshot release,
        without session details, every form, the selected external databases, and as primary assembly the organism's
        preferred one that has the snapshot release.

        Args:
            repository: Folder to look in.
            organism: Only graphs of this organism; ``None`` for all.

        Returns:
            list[dict]: The entries of :py:func:`idtrack.graph.files.list_graph_files`, sorted by file name.
        """
        found = []
        for entry in list_graph_files(repository, organism):
            snapshot = entry["snapshot_release"]
            if (
                entry["organism"] not in organisms.supported_organisms()
                or snapshot is None
                or entry["first_release"] != organisms.first_release(entry["organism"])
                or entry["session_details"]
                or entry["forms"] is not None
                or entry["all_external"]
            ):
                continue
            for assembly in organisms.assemblies_by_priority(entry["organism"]):
                first, last = organisms.release_range(entry["organism"], assembly)
                if first <= snapshot and (last is None or snapshot <= last):
                    if assembly == entry["assembly"]:
                        found.append(entry)
                    break
        return found

    def _graph_snapshots(self) -> dict[str, list[int]]:
        """Return ``{organism: [snapshot releases]}`` of the graphs :py:meth:`idtrack.API.build` would load.

        Returns:
            dict[str, list[int]]: Sorted snapshot releases per organism with at least one such graph
            (:py:meth:`_graph_files`).
        """
        found: dict[str, list[int]] = {}
        for entry in self._graph_files(self.idtrack_local_repository):
            found.setdefault(entry["organism"], []).append(int(entry["snapshot_release"]))
        return {organism: sorted(releases) for organism, releases in sorted(found.items())}

    def _resolve_organism(self, organism: str | None) -> str:
        """Return the Ensembl species name to use (see the ``organism`` argument)."""
        if organism is None:
            held = sorted(self._graph_snapshots())
            if len(held) != 1:
                found = (
                    f"holds graphs of several organisms ({', '.join(held)})"
                    if held
                    else "holds no graph built by `API.build`"
                )
                raise InvalidArgumentError(
                    f"Pass `organism`: the idtrack local repository {self.idtrack_local_repository!r} {found}, "
                    "so the organism cannot be read from it. Use an Ensembl species name of "
                    f"`idtrack.organisms.supported_organisms()` ({', '.join(organisms.supported_organisms())}), e.g. "
                    '`organism="homo_sapiens"`, or a taxon ID or name that `API.ensembl_organism` accepts.'
                )
            return held[0]
        # The formal name of a configured organism resolves without the network; other queries need Ensembl REST.
        from idtrack.build.species import VerifyOrganism

        return VerifyOrganism(organism).get_formal_name()

    def _default_snapshot_release(self) -> int:
        """Return the newest graph snapshot of the organism, else the organism's latest Ensembl release."""
        releases = self._graph_snapshots().get(self.organism)
        if releases:
            return int(max(releases))
        _, latest_release = API.ensembl_organism(self.organism)
        return int(latest_release)

    # ----------------------------------------------------------------------------------------------------------------
    # Conversions and their cache

    def _graph_identity(self) -> str | None:
        """Return the content digest of the graph the conversions come from (see "Result cache" in the class docs).

        The loaded graph's (when an API is given), else the one the graph file of the organism and snapshot in
        *idtrack_local_repository* records (read from the file's record without loading the graph); ``None`` for a
        graph without a record (made in memory), whose conversions are never cached.
        """
        if self._api is not None:
            record = self._api.graph.record
            return None if record is None else record.content_digest
        for entry in self._graph_files(self.idtrack_local_repository, self.organism):
            if entry["snapshot_release"] == self.snapshot_release:
                return files.content_digest(entry["path"])
        return None

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
            "rules_version": RULES_VERSION,
            "organism": self.organism,
            "final_database": self.final_database,
            "target_ensembl_release": self.to_release,
            "source_release": self.from_release_by_dataset.get(dataset_name),
            "source_database": self.from_database_by_dataset.get(dataset_name),
            "graph_snapshot": self.snapshot_release,
            "graph_content_digest": self._graph_identity(),
            "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        }

    def _dataset_conversions(self, dataset_name: str) -> tuple[list[ConversionResult], dict[str, list[str]]]:
        """Return the matchings and gene names of a dataset, from a valid result cache or computed."""
        path = self.result_cache_path(dataset_name)
        header = self._cache_header(dataset_name)
        try:
            if header["graph_content_digest"] is None:
                raise FileNotFoundError  # a graph without a digest: nothing cached is known to be of it
            with open(path) as handle:
                stored = json.load(handle)
            if stored.get("header") == header:
                return [ConversionResult.from_dict(m) for m in stored["matchings"]], stored["gene_names"]
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
        if header["graph_content_digest"] is not None:
            _write_json_atomically(
                path, {"header": header, "matchings": [m.to_dict() for m in matchings], "gene_names": gene_names}
            )
        return matchings, gene_names

    def _require_api(self) -> API:
        """Return the API, loading the graph first if this harmonizer has none yet."""
        if self._api is None:
            self._api = API.build(self.organism, self.snapshot_release, self.idtrack_local_repository)
        return self._api

    def _convert_dataset(self, api: API, dataset_name: str) -> tuple[list[ConversionResult], dict[str, list[str]]]:
        """Convert the features of one dataset and name every gene reached (:py:func:`idtrack.harmonize.convert_features`).

        Args:
            api: The API with the graph loaded.
            dataset_name: Key of :py:attr:`data_h5ad_dict`.

        Returns:
            tuple: The results (one per feature, in file order) and ``{gene: final-database names}``.
        """
        return convert_features(
            api,
            self.feature_names[dataset_name],
            to_release=self.to_release,
            final_database=self.final_database,
            from_release=self.from_release_by_dataset.get(dataset_name),
            from_database=self.from_database_by_dataset.get(dataset_name),
            progress=self.verbosity > 0,
        )

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
            raise InvalidArgumentError(f"`merge` must be 'flag' or 'sum', got {merge!r}.")
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
            raise InvalidArgumentError(
                f"Dataset {dataset_name!r} changed on disk since the harmonizer read its feature names."
            )
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
        features: Literal["union", "intersect"] = "union",
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
          ``"union"`` also ``f"{converted_id_column}_{handle}"`` and an integer ``"intersection"`` column (1 if
          the feature is in every dataset, by presence, whatever its name). Every other ``var`` column of the inputs is
          kept: as **one** column when, for every feature, the datasets that have a value for it agree; otherwise
          **per dataset** as ``f"{column}_{handle}"`` for each dataset that has it. Columns in *numeric_var_columns*
          are converted to ``float``; other text columns become categorical (values unchanged).

        * **layers, obsm, obsp, uns.** Handled by :py:func:`anndata.concat`: a layer or ``obsm`` entry present in
          every dataset is concatenated (``obsp`` block-diagonally), and an ``uns`` entry is kept when it is the same
          in every dataset. ``varm``, ``varp`` and ``raw`` are not carried into the merged object.

        ``X`` keeps the common dtype of the inputs (e.g. ``float32`` stays ``float32``).

        Args:
            features (Literal["union", "intersect"]): How to combine the feature sets (see above).
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
            InvalidArgumentError: If *features*, *obs_join* or *merge* is not one of its values; if a name of *obs_columns_to_keep*
                is not among the joined columns, *handle_anndata_key* is already an ``obs`` column, or a numeric column
                is absent or holds a value that is not a number; if two cells get the same name; if a generated
                ``var`` column name clashes with an input column.
        """
        if features not in ("union", "intersect"):
            raise InvalidArgumentError(f"`features` must be 'union' or 'intersect', got {features!r}.")
        if obs_join not in ("outer", "inner"):
            raise InvalidArgumentError(f"`obs_join` must be 'outer' or 'inner', got {obs_join!r}.")
        if merge not in ("flag", "sum"):
            raise InvalidArgumentError(f"`merge` must be 'flag' or 'sum', got {merge!r}.")
        numeric_obs_columns = set(numeric_obs_columns or set())
        numeric_var_columns = set(numeric_var_columns or set())
        handles = list(self.data_h5ad_dict)

        studies: list[ad.AnnData] = []
        for handle in tqdm.tqdm(handles, ncols=140, disable=self.verbosity == 0):
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
            raise InvalidArgumentError(
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
            join="outer" if features == "union" else "inner",
            fill_value=0,
            merge=None,
            uns_merge="same",
            pairwise=True,
        )
        del studies

        duplicated_cells = adata.obs_names[adata.obs_names.duplicated()].unique().tolist()
        if duplicated_cells:
            raise InvalidArgumentError(
                f"Cell names collide after prefixing with the dataset handle: {duplicated_cells[:10]}."
            )
        adata.obs[handle_anndata_key] = pd.Categorical(adata.obs[handle_anndata_key], categories=handles)

        adata.var = self._merge_harmonized_var(var_frames, adata.var_names, features)

        for frame, columns, where in ((adata.obs, numeric_obs_columns, "obs"), (adata.var, numeric_var_columns, "var")):
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise InvalidArgumentError(
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
            raise InvalidArgumentError(f"`obs_columns_to_keep` names columns that no dataset has: {nowhere}.")
        not_everywhere = [c for c in obs_columns_to_keep if c not in joined]
        if not_everywhere:
            raise InvalidArgumentError(
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
            raise InvalidArgumentError(f"Non-numeric value in {where}: {bad.unique().tolist()[:10]}.")
        return converted.astype(float)

    def _merge_harmonized_var(
        self, var_frames: dict[str, pd.DataFrame], var_names: pd.Index, features: str
    ) -> pd.DataFrame:
        """Build the merged ``var`` of :py:meth:`unify_multiple_anndatas` (see its docstring for the rules).

        Args:
            var_frames: ``var`` of each harmonised dataset (from :py:meth:`feature_harmonizer`), by handle.
            var_names: Features of the merged object, in its order.
            features: ``"union"`` or ``"intersect"``.

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
                raise InvalidArgumentError(
                    f"The merged var would have the column {name!r} twice; rename the input column."
                )
            columns[name] = values

        def _same_everywhere(column: str) -> pd.Series:
            # A gene's final-database names are a property of the gene (the same in every dataset): they agree.
            merged = self._agreed_values({h: a[column] for h, a in aligned.items()}, var_names)
            if merged is None:
                raise IntegrityError(f"The datasets disagree on {column!r} for a feature; this is an internal error.")
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
        if features == "union":
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
