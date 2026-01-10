#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import copy
import logging
from typing import Any, Literal, Optional, Union

import numpy as np
from tqdm import tqdm

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._track import Track
from idtrack._track_tests import TrackTests
from idtrack._verify_organism import VerifyOrganism


class API:
    """Provide a high-level façade for building graphs and converting biological identifiers with IDTrack.

    This class centralises common workflows so users can quickly initialise the underlying graph (for a chosen organism
    and Ensembl release), configure logging, and run identifier-related operations. Internally it delegates to
    lower-level components such as :py:class:`idtrack.DatabaseManager` for data access and :py:class:`idtrack.Track` (or
    :py:class:`idtrack.TrackTests`) for graph traversal and matching. It is intended as the primary entry point for
    day-to-day tasks like resolving an organism name, constructing the working graph snapshot, converting identifiers
    between releases or external databases, and inspecting available external data sources.
    """

    def __init__(self, local_repository: str) -> None:
        """Bind the interface to a local repository used for data downloads and on-disk caches.

        This initialiser wires up a dedicated logger for the API layer and records the path where IDTrack will keep
        its working files. The actual graph and tracking objects are created lazily (e.g. by
        :py:meth:`idtrack.API.build_graph`) so that simply constructing :py:class:`idtrack.API` is inexpensive.

        Args:
            local_repository (str): Absolute (recommended) or relative path to a writable directory where the package
                may store downloaded resources and precomputed artefacts. The caller is responsible for ensuring the
                path exists and is accessible.

        Attributes:
            log (logging.Logger): Logger named ``"api"`` for progress messages and diagnostics.
            logger_configured (bool): ``False`` until :py:meth:`idtrack.API.configure_logger` is called.
            local_repository (str): The given repository path.
            track (idtrack.Track | idtrack.TrackTests): Placeholder for the active tracker; populated after
                :py:meth:`idtrack.API.build_graph` is invoked.
        """
        # Instance attributes
        self.log = logging.getLogger("api")
        self.logger_configured = False
        self.local_repository = local_repository
        self.track: Optional[Union[Track, TrackTests]] = None

    def _require_track(self) -> Union[Track, TrackTests]:
        """Return the active tracker or raise a clear error if the graph is not built yet."""
        track = getattr(self, "track", None)
        if track is None:
            raise RuntimeError("No graph is attached to this API instance. Call `API.build_graph(...)` first.")
        return track

    def configure_logger(self, level=None) -> None:
        """Configure process-wide logging with a concise, time-stamped console format.

        This method is idempotent per :py:class:`idtrack.API` instance: the first call sets up a basic configuration
        for the Python logging system (time, level, logger name, and message). Subsequent calls on the same instance
        will not reconfigure logging and instead emit an informational message via :py:attr:`idtrack.API.log`.

        Args:
            level (int | str | None): Desired logging level (e.g. ``logging.INFO``, ``"INFO"``, ``logging.DEBUG``).
                If ``None``, defaults to :py:data:`logging.INFO`.

        Notes:
            The configuration applies to the root logger and therefore affects logging for the entire Python process,
            not only this package. Call this early in your application if you want IDTrack's log output formatted
            consistently with the rest of your program.
        """
        if not self.logger_configured:
            logging.basicConfig(
                level=logging.INFO if level is None else level,
                datefmt="%Y-%m-%d %H:%M:%S",
                format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
            )
            self.logger_configured = True
        else:
            self.log.info("The logger is already configured.")

    def calculate_graph_caches(self, for_test: bool = False) -> None:
        """Prime the working graph by eagerly computing all cached properties.

        This helper reduces first-call latency and makes test runs deterministic by batch-computing every
        ``@cached_property`` exposed by :py:class:`idtrack._the_graph.TheGraph`. Use it after
        :py:meth:`idtrack.API.build_graph` has attached a populated
        :py:class:`idtrack._track.Track` (or :py:class:`idtrack._track_tests.TrackTests`) to
        :py:attr:`self.track`. Internally it forwards to
        :py:meth:`idtrack._the_graph.TheGraph.calculate_caches`.

        Args:
            for_test (bool): If ``True``, also compute heavyweight, test-only caches such as
                :py:attr:`idtrack._the_graph.TheGraph.external_database_connection_form` and
                :py:attr:`idtrack._the_graph.TheGraph.available_releases_given_database_assembly`.
                Defaults to ``False``.
        """
        self._require_track().graph.calculate_caches(for_test=for_test)

    def resolve_organism(self, tentative_organism_name: str) -> tuple[str, int]:
        """Normalize a tentative organism name and fetch the latest supported Ensembl release.

        This shields callers from Ensembl naming quirks by resolving a user-provided synonym (e.g. common name,
        shorthand, taxon ID) to the canonical Ensembl species identifier (e.g. ``"homo_sapiens"``) and to the newest
        Ensembl release that still hosts that species. The lookup delegates to
        :py:class:`idtrack._verify_organism.VerifyOrganism`, ensuring subsequent graph construction and data access use
        a consistent, up-to-date pair.

        Args:
            tentative_organism_name (str): Organism descriptor in any supported synonym form (e.g. ``"human"``,
                ``"hsapiens"``, ``"9606"``, or ``"homo_sapiens"``). Matching is case-insensitive.

        Returns:
            tuple[str, int]: ``(formal_name, latest_release)`` where ``formal_name`` is the canonical Ensembl species
                string and ``latest_release`` is the most recent Ensembl release number known for that species.
        """
        vdf = VerifyOrganism(tentative_organism_name)
        formal_name = vdf.get_formal_name()
        latest_release = vdf.get_latest_release()
        return formal_name, latest_release

    def get_database_manager(
        self,
        organism_name: str,
        snapshot_release: int,
        genome_assembly: Optional[int] = None,
        ignore_before: Optional[int] = None,
    ) -> "DatabaseManager":
        """Create a database manager configured for an organism and a release-bounded snapshot.

        Construct and return :py:class:`idtrack._database_manager.DatabaseManager` bound to ``organism_name`` and
        configured to ignore data newer than ``snapshot_release``. The manager centralizes all download, caching, and
        version logic for graph builds and identifier conversions. The biological *form* is initialised from
        :py:data:`idtrack._db.DB.backbone_form`, and all artefacts are stored
        under :py:attr:`idtrack.API.local_repository`.

        Args:
            organism_name (str): Canonical Ensembl species name (e.g. ``"homo_sapiens"``).
            snapshot_release (int): Most recent Ensembl release to include; later releases are ignored for
                reproducibility.
            genome_assembly (int | None): Genome assembly code used in Ensembl core schema names
                (``<organism>_core_<release>_<assembly>``). This selects the **primary** assembly for the snapshot
                (e.g. ``38`` = human GRCh38, ``37`` = human GRCh37, ``39`` = mouse GRCm39, ``111`` = pig Sscrofa11.1).
                If ``None`` (default), the highest-priority assembly configured for the organism is used. Note that
                the resulting snapshot graph can still include additional assemblies within the snapshot window;
                use :py:meth:`idtrack.API.list_genome_assemblies` to inspect what is present.
            ignore_before (int | None): Earliest Ensembl release to include in the snapshot window. When ``None``
                (default), use the earliest release supported by the public Ensembl MySQL/FTP dumps (see
                :py:data:`idtrack._db.DB.mysql_port_min_release`). This default ensures multi-assembly history is
                retained for clean-handoff species (e.g. mouse) where older assemblies live entirely in earlier
                releases.

        Returns:
            idtrack._database_manager.DatabaseManager: A manager ready for use by graph-building and
                conversion routines.

        Notes:
            Any exceptions raised by :py:class:`idtrack._database_manager.DatabaseManager` propagate unchanged.
        """
        if ignore_before is None:
            ignore_before = int(min(DB.mysql_port_min_release.values()))
        return DatabaseManager(
            organism=organism_name,
            ensembl_release=None,
            ignore_before=ignore_before,
            ignore_after=snapshot_release,
            form=copy.deepcopy(DB.backbone_form),
            local_repository=self.local_repository,
            genome_assembly=genome_assembly,
        )

    def build_graph(
        self,
        organism_name: str,
        snapshot_release: int,
        genome_assembly: Optional[int] = None,
        return_test: bool = False,
        calculate_caches: bool = True,
    ) -> None:
        """Build the bio-ID graph for an organism and prepare the path-finding engine.

        This method wires together the high-level components used throughout IDTrack. It first creates a
        :py:class:`idtrack._database_manager.DatabaseManager` that ignores releases newer than ``snapshot_release``.
        It then instantiates :py:class:`idtrack._track.Track` (or :py:class:`idtrack._track_tests.TrackTests` when
        testing), which loads or builds the underlying :py:class:`idtrack._the_graph.TheGraph` via
        :py:class:`idtrack._graph_maker.GraphMaker`. Optionally, it primes all graph caches to improve query latency.
        The resulting resolver is stored on :py:attr:`self.track` for subsequent conversions and inspections.

        Args:
            organism_name (str): Canonical Ensembl species name, typically the output of
                :py:meth:`idtrack.API.resolve_organism`.
            snapshot_release (int): Ensembl release anchoring this build. Data from later releases are ignored to ensure
                reproducible results.
            genome_assembly (int | None): Genome assembly code used in Ensembl core schema names
                (``<organism>_core_<release>_<assembly>``). This selects the **primary** assembly for the snapshot
                (default: highest-priority/newest configured for the organism). The snapshot graph can still include
                additional assemblies within the snapshot window; use :py:meth:`idtrack.API.list_genome_assemblies`
                to inspect what is present.
            return_test (bool): If ``True``, initialise :py:class:`idtrack._track_tests.TrackTests` instead of the
                standard :py:class:`idtrack._track.Track` to enable test and diagnostics helpers. Defaults to ``False``.
            calculate_caches (bool): If ``True``, eagerly compute the graph’s cached properties. When combined with
                ``return_test=True``, test-only caches are included. Defaults to ``True``.

        See Also:
            :py:meth:`idtrack.API.get_database_manager`,
            :py:meth:`idtrack.API.calculate_graph_caches`,
            :py:class:`idtrack._track.Track`,
            :py:class:`idtrack._track_tests.TrackTests`
        """
        dm = self.get_database_manager(
            organism_name=organism_name, snapshot_release=snapshot_release, genome_assembly=genome_assembly
        )

        if return_test:
            self.track = TrackTests(dm)
        else:
            self.track = Track(dm)

        if calculate_caches and return_test:
            self.calculate_graph_caches(for_test=True)
        elif calculate_caches:
            self.calculate_graph_caches(for_test=False)

    def convert_identifier(
        self,
        identifier: str,
        from_release: Optional[int] = None,
        to_release: Optional[int] = None,
        final_database: Optional[str] = None,
        strategy: Literal["all", "best"] = "best",
        explain: bool = False,
    ) -> dict[str, Any]:
        """Resolve a raw identifier and convert it to a target Ensembl release and (optionally) an external database.

        This high-level helper wraps :py:meth:`idtrack._track.Track.convert` and returns a compact, user-oriented
        summary of the result. It first normalises *identifier* to the canonical graph node label with
        :py:meth:`idtrack._the_graph.TheGraph.node_name_alternatives`, then invokes the path-finding and
        final-conversion pipeline to reach the requested *to_release* and *final_database*. The output is designed for
        interactive use and downstream tooling: it reports whether the query is present in the graph, whether a
        conversion could be computed, and (if requested) the full path(s) followed through the Ensembl backbone and
        the external database hop.

        Args:
            identifier (str): Query identifier to resolve. May be an Ensembl stable ID, gene symbol, or a known synonym;
                case and common punctuation variations are tolerated by the normaliser.
            from_release (int | None): Ensembl release the *identifier* originates from. If ``None``, the direction of
                time travel is inferred automatically. Supplying a value constrains the search to
                forward/reverse travel.
            to_release (int | None): Target Ensembl release. If ``None``, the newest release available in
                the graph is used.
            final_database (str | None): Name of the external database to convert into (e.g. ``"uniprot"``).
                If ``None``, the result remains on the Ensembl gene backbone (reported as
                :py:data:`idtrack._db.DB.nts_ensembl[idtrack._db.DB.backbone_form]`).
            strategy (Literal["all", "best"]): Selection strategy applied *after* scoring all admissible targets.
                ``"best"`` keeps a single globally best target; ``"all"`` keeps all scored
                targets. Defaults to ``"best"``.
            explain (bool): If ``True``, include the concatenated edge list(s) that show how each result was reached.

        Returns:
            dict[str, Any]: Dictionary describing the conversion outcome with the following keys.

                - ``"target_id"`` (list[str]): Unique identifiers in the requested *final_database*. When \
                    ``strategy="best"`` and a target exists, this list contains exactly one element. \
                    If *final_database* is ``None``, the list contains the Ensembl gene ID(s).
                - ``"last_node"`` (list[tuple[str, str]]): Pairs of ``(ensembl_gene_id, target_id)`` for every \
                    surviving candidate. The first element is the final Ensembl node reached by time travel; the \
                    second is the chosen target in *final_database* (or the Ensembl gene itself when staying on \
                    the backbone).
                - ``"final_database"`` (str | None): The database name the *target_id* values come from. \
                    ``None`` only when the query was not found at all; otherwise this is either *final_database* or \
                    the Ensembl backbone label :py:data:`idtrack._db.DB.nts_ensembl[idtrack._db.DB.backbone_form]`.
                - ``"graph_id"`` (str | None): Canonical node label used internally by the graph for *identifier* \
                    (e.g. ``"ACTB"`` for the symbol ``"actb"``). ``None`` when the query has no corresponding graph node.
                - ``"query_id"`` (str): Echo of the original *identifier* argument for bookkeeping.
                - ``"no_corresponding"`` (bool): ``True`` if the query could not be matched to any graph node (nothing \
                    to convert). In this case ``"graph_id"`` is ``None`` and the other fields are empty or ``None``.
                - ``"no_conversion"`` (bool): ``True`` if the query exists in the graph but no admissible path to \
                    *to_release* and/or *final_database* could be constructed (a 1→0 mapping).
                - ``"no_target"`` (bool): ``True`` if an Ensembl gene was reached but the requested *final_database* \
                    yielded no synonym. The result may fall back to returning the Ensembl gene itself; this flag lets \
                    callers distinguish that fallback from a genuine external match.
                - ``"the_path"`` (dict[tuple[str, str], tuple[tuple]]): Present only when *explain* is ``True``. Maps \
                    each ``(target_id, ensembl_gene_id)`` pair to an ordered tuple of edges representing the full \
                    walk: first the Ensembl *history* segment that reaches the gene, then the *final-conversion* hop \
                    into the external database. Each edge is expressed in the internal format used by \
                    :py:class:`idtrack._track.Track` and may include auxiliary fields (e.g. release markers).

        Raises:
            ValueError: If *strategy* is not ``"all"`` or ``"best"``.

        See Also:
            :py:meth:`idtrack._the_graph.TheGraph.node_name_alternatives`,
            :py:meth:`idtrack._track.Track.convert`.

        Notes:
            - Interactions between the boolean flags:

                * ``no_corresponding=True`` ⇒ no conversion is attempted; ``graph_id`` is ``None``; \
                    ``target_id`` is ``[]``.
                * ``no_conversion=True`` ⇒ query exists but path scoring/selecting produced no admissible target.
                * ``no_target=True`` ⇒ Ensembl history succeeded but the external database lacked a synonym; callers \
                    may still receive an Ensembl fallback target.\

            - When ``strategy="best"``, the scoring and tie-breakers are those implemented by \
                :py:meth:`idtrack._track.Track.calculate_score_and_select` and its callers. When ``"all"``, no global \
                tie-break is applied and all scored targets are returned.
        """
        if strategy == "best":
            prioritize_to_one_filter = True
        elif strategy == "all":
            prioritize_to_one_filter = False
        else:
            raise ValueError(f"Invalid strategy={strategy!r}. Valid options are 'best' or 'all'.")

        # Get the graph ID if possible.
        track = self._require_track()
        new_ident, _ = track.graph.node_name_alternatives(identifier)
        no_corresponding, no_conversion = False, False

        if new_ident is not None:
            cnt = track.convert(
                from_id=new_ident,
                from_release=from_release,
                to_release=to_release,
                final_database=final_database,
                prioritize_to_one_filter=prioritize_to_one_filter,
                return_path=explain,
            )
            if cnt is None:
                no_conversion = True
        else:
            no_corresponding = True

        final_ids_together: list[tuple[str, str]] = (
            list({(i, j) for i in cnt for j in cnt[i]["final_conversion"]["final_elements"]})
            if not no_corresponding and not no_conversion
            else []
        )
        target_ids = list({i[1] for i in final_ids_together})

        final_database_conv_: set[Optional[str]] = (
            {cnt[i]["final_conversion"]["final_database"] for i in cnt}
            if not no_corresponding and not no_conversion
            else {None}
        )
        if len(final_database_conv_) != 1:
            raise ValueError(f"Expected exactly one final database, got: {final_database_conv_}")
        final_database_conv = next(iter(final_database_conv_))

        final_conf_: set[Optional[Union[int, float]]] = (
            {cnt[i]["final_conversion"]["final_conversion_confidence"] for i in cnt}
            if not no_corresponding and not no_conversion
            else {None}
        )
        if len(final_conf_) != 1:
            raise ValueError(f"Expected exactly one final confidence, got: {final_conf_}")
        final_conf = next(iter(final_conf_))

        result: dict[str, Any] = {
            "target_id": target_ids,
            "last_node": final_ids_together,
            "final_database": final_database_conv,
            "graph_id": new_ident,
            "query_id": identifier,
            "no_corresponding": no_corresponding,
            "no_conversion": no_conversion,
            "no_target": np.isinf(final_conf) if final_conf is not None else False,
        }

        if explain:
            result["the_path"] = (
                {
                    (j, i): tuple(
                        list(cnt[i]["the_path"]) + list(cnt[i]["final_conversion"]["final_elements"][j]["the_path"])
                    )
                    for i in cnt
                    for j in cnt[i]["final_conversion"]["final_elements"]
                }
                if not no_corresponding and not no_conversion
                else {}
            )

        return result

    def convert_identifier_multiple(
        self, identifier_list: list[str], verbose: bool = True, pbar_prefix: str = "", **kwargs
    ) -> list[dict]:
        """Convert a batch of identifiers and aggregate per-query conversion metadata.

        This is a thin, progress-enabled wrapper around :py:meth:`idtrack.API.convert_identifier`. It iterates over
        *identifier_list* in order, forwards ``**kwargs`` to the single-item converter, and collects each per-identifier
        result. Use this helper for bulk operations where you want progress feedback and a uniform result structure that
        mirrors the single-call API.

        Args:
            identifier_list (list[str]): Input identifiers to resolve and convert. Each element is passed to
                :py:meth:`idtrack.API.convert_identifier` as its *identifier* argument, in the same order.
            verbose (bool): If ``True``, display a :py:mod:`tqdm` progress bar (throttled to avoid excessive redraws).
                Set to ``False`` to disable the progress bar. Defaults to ``True``.
            pbar_prefix (str): Optional label shown before the progress bar text (for distinguishing concurrent runs).
                Defaults to an empty string.
            kwargs: Keyword arguments forwarded verbatim to :py:meth:`idtrack.API.convert_identifier`. Common options
                include:

                    - ``from_release`` (int | None): Origin Ensembl release of the input identifier.
                    - ``to_release`` (int | None): Target Ensembl release to which to time-travel.
                    - ``final_database`` (str | None): Name of the external database to project into (e.g. \
                        ``"uniprot"``). If ``None``, results stay on the Ensembl backbone and are reported as \
                        :py:data:`idtrack._db.DB.nts_ensembl[idtrack._db.DB.backbone_form]`.
                    - ``strategy`` (Literal[``"best"``, ``"all"``]): Selection policy after scoring candidates.
                    - ``explain`` (bool): If ``True``, include full path details in the result (see \
                        ``"the_path"`` below).

        Returns:
            list[dict[str, Any]]: One element per input identifier, preserving input order. Each dictionary matches the
            schema returned by :py:meth:`idtrack.API.convert_identifier`.

        See Also:
            :py:meth:`idtrack.API.convert_identifier`,
            :py:meth:`idtrack.API.classify_multiple_conversion`,
            :py:meth:`idtrack.API.print_binned_conversion`.

        Notes:
            The output list preserves the order of *identifier_list*. Items are independent; failures for one
            query do not prevent processing of the others.
        """
        result = list()
        with tqdm(identifier_list, mininterval=0.25, disable=not verbose, desc=pbar_prefix, ncols=100) as loop_obj:
            for identifier in loop_obj:
                loop_obj.set_postfix_str(f"ID:{identifier}", refresh=False)

                result.append(self.convert_identifier(identifier, **kwargs))
        return result

    def classify_multiple_conversion(self, matchings: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """Group batch-conversion results into semantic bins for downstream reporting.

        This post-processing step takes the per-identifier results produced by
        :py:meth:`idtrack.API.convert_identifier_multiple` (or a compatible list of
        :py:meth:`idtrack.API.convert_identifier` payloads) and organises them into logically meaningful categories.
        The bins distinguish between “no match,” one-to-one vs. one-to-many mappings, whether the output differs from
        the input, and whether a reported target is an Ensembl fallback due to a missing external synonym.

        Args:
            matchings (list[dict[str, Any]]): Collection of dictionaries returned by
                :py:meth:`idtrack.API.convert_identifier_multiple`. Each element must contain, at minimum, the keys
                ``"query_id"``, ``"target_id"``, ``"no_corresponding"``, ``"no_conversion"``, and ``"no_target"`` as
                described in :py:meth:`idtrack.API.convert_identifier`.

        Returns:
            dict[str, list[dict[str, Any]]]: A dictionary of category → list-of-results. Categories are **not**
                mutually exclusive; an item can appear in multiple bins (e.g. a changed 1→1 mapping also appears in the
                general 1→1 bin). Keys are:

                - ``"input_identifiers"``: All input result objects, echoed unchanged (convenient for summary counts).
                - ``"matching_1_to_0"``: Queries that could not be mapped to any target (either \
                    ``no_corresponding=True`` or ``no_conversion=True``). Indicates a 1→0 outcome.
                - ``"matching_1_to_1"``: Queries with exactly one target in ``"target_id"``. Includes both unchanged \
                    and changed outputs, and may overlap with ``"changed_only_1_to_1"`` or \
                    ``"alternative_target_1_to_1"``.
                - ``"matching_1_to_n"``: Queries with more than one target in ``"target_id"`` (n > 1). \
                    May overlap with ``"changed_only_1_to_n"`` or ``"alternative_target_1_to_n"``.
                - ``"changed_only_1_to_1"``: Strict subset of 1→1 where the single ``"target_id"[0]`` is \
                    **different** from ``"query_id"`` (i.e. the identifier changed across releases/databases).
                - ``"changed_only_1_to_n"``: Strict subset of 1→n where **none** of the ``"target_id"`` \
                    entries equal ``"query_id"`` (the original identifier is not present among the alternatives).
                - ``"alternative_target_1_to_1"``: Cases with exactly one ``"target_id"`` and ``no_target=True``. \
                    This flags Ensembl fallbacks where the external database lacked a synonym; the single reported \
                    value is not a genuine external match.
                - ``"alternative_target_1_to_n"``: As above, but with multiple entries in ``"target_id"`` (n > 1) \
                    while ``no_target=True`` (typically multiple Ensembl-side candidates with no external synonym).

        Raises:
            ValueError: If any element in *matchings* has an empty ``"target_id"`` list despite
                ``no_corresponding`` and ``no_conversion`` both being ``False`` (indicates an unexpected upstream state).

        See Also:
            :py:meth:`idtrack.API.convert_identifier_multiple`,
            :py:meth:`idtrack.API.print_binned_conversion`,
            :py:meth:`idtrack.API.convert_identifier`.

        Notes:
            The function does not mutate input dictionaries. The binning logic is intentionally overlapping so that
            “summary” buckets (``matching_*``) can be used alongside “diagnostic” buckets (``changed_only_*``,
            ``alternative_target_*``) without additional passes.
        """
        result: dict[str, list[dict]] = {
            "changed_only_1_to_n": [],
            "changed_only_1_to_1": [],
            "alternative_target_1_to_1": [],
            "alternative_target_1_to_n": [],
            "matching_1_to_0": [],
            "matching_1_to_1": [],
            "matching_1_to_n": [],
            "input_identifiers": [],
        }

        for i in matchings:
            result["input_identifiers"].append(i)

            if i["no_corresponding"]:
                result["matching_1_to_0"].append(i)
                continue

            if i["no_conversion"]:
                result["matching_1_to_0"].append(i)
                continue

            if len(i["target_id"]) == 0:
                raise ValueError(
                    f"Unexpected conversion result: query_id={i.get('query_id')!r} returned an empty target_id list. "
                    "This indicates an internal error in the conversion logic. Please report this issue."
                )

            if i["no_target"]:
                if len(i["target_id"]) == 1:
                    result["alternative_target_1_to_1"].append(i)
                else:
                    result["alternative_target_1_to_n"].append(i)

            else:
                if len(i["target_id"]) == 1 and i["target_id"][0] != i["query_id"]:
                    result["changed_only_1_to_1"].append(i)

                if len(i["target_id"]) > 1 and not any([i["query_id"] == k for k in i["target_id"]]):
                    result["changed_only_1_to_n"].append(i)

                if len(i["target_id"]) == 1:
                    result["matching_1_to_1"].append(i)

                if len(i["target_id"]) > 1:
                    result["matching_1_to_n"].append(i)

        return result

    def print_binned_conversion(self, classified: dict[str, list[dict]]) -> None:
        """Log a structured multi-line summary of binned conversion results with percentages and rest counts.

        Args:
            classified (dict[str, list[dict]]): Output from classify_multiple_conversion.
        """
        total = len(classified.get("input_identifiers", []))
        one_to_zero = len(classified.get("matching_1_to_0", []))

        one_to_one_target = len(classified.get("matching_1_to_1", []))
        one_to_n_target = len(classified.get("matching_1_to_n", []))
        changed_1_to_1 = len(classified.get("changed_only_1_to_1", []))
        changed_1_to_n = len(classified.get("changed_only_1_to_n", []))
        alt_1_to_1 = len(classified.get("alternative_target_1_to_1", []))
        alt_1_to_n = len(classified.get("alternative_target_1_to_n", []))

        one_to_one_total = one_to_one_target + alt_1_to_1
        one_to_n_total = one_to_n_target + alt_1_to_n

        rest_1_to_1 = one_to_one_target - changed_1_to_1

        no_corresp = sum(x["no_corresponding"] for x in classified.get("input_identifiers", []))
        no_conv = sum(x["no_conversion"] for x in classified.get("input_identifiers", []))
        no_target = sum(x["no_target"] for x in classified.get("input_identifiers", []))

        def pct(part, whole):
            return (part / whole * 100) if whole else 0

        self.log.info(
            f"\nIDTrack conversion summary:\n"
            f"  Total processed: {total}\n"
            f"  1→0: {one_to_zero} ({pct(one_to_zero, total):.1f}%)\n"
            f"  1→1: {one_to_one_total} ({pct(one_to_one_total, total):.1f}%)\n"
            f"    Changed only: {changed_1_to_1} ({pct(changed_1_to_1, one_to_one_total):.1f}%)\n"
            f"    Alternative targets: {alt_1_to_1} ({pct(alt_1_to_1, one_to_one_total):.1f}%)\n"
            f"    Rest: {rest_1_to_1} ({pct(rest_1_to_1, one_to_one_total):.1f}%)\n"
            f"  1→n: {one_to_n_total} ({pct(one_to_n_total, total):.1f}%)\n"
            f"    Changed only: {changed_1_to_n} ({pct(changed_1_to_n, one_to_n_total):.1f}%)\n"
            f"    Alternative targets: {alt_1_to_n} ({pct(alt_1_to_n, one_to_n_total):.1f}%)\n"
            f"  Diagnostics:\n"
            f"    no_corresponding: {no_corresp}\n"
            f"    no_conversion:   {no_conv}\n"
            f"    no_target:       {no_target}"
        )

    def infer_identifier_source(
        self, id_list: list[str], mode: str = "assembly_ensembl_release", report_only_winner: bool = True
    ) -> Union[tuple[str, int, int], tuple[int, int], int, list[tuple[Any, int]]]:
        """Infer the most likely source (database/assembly/release) for a heterogeneous identifier list.

        This helper estimates which origin best explains the given IDs so users can pick a sensible graph configuration
        before running conversions at scale. Internally it resolves each input to a canonical node (where possible),
        consults :py:attr:`idtrack._the_graph.TheGraph.node_trios` to recover known origins, and tallies them via
        :py:meth:`idtrack._track.Track.identify_source`. **Under development:** both the public signature and the
        scoring details may change in future releases.

        Args:
            id_list (list[str]): Identifiers to analyse. Each item should be a string; non-existent IDs are safely
                ignored (and logged) during the tally.
            report_only_winner (bool): If ``True``, return the single highest-count origin for the requested *mode*.
                If ``False``, return all candidate origins ranked by descending count.
            mode (str): Granularity of the origin to infer.

                One of:
                - ``"complete"`` → return triples ``(database, assembly, release)``.
                - ``"ensembl_release"`` → return the Ensembl release only (``int``).
                - ``"assembly"`` → return the genome assembly only (``int``).
                - ``"assembly_ensembl_release"`` → return pairs ``(assembly, release)``.

        Returns:
            tuple[str, int, int] | tuple[int, int] | int | list[tuple[object, int]]: The inferred origin(s),
                depending on *report_only_winner*:

                - If *report_only_winner* is ``True``:
                    - ``mode == "complete"`` → ``(database: str, assembly: int, release: int)``
                    - ``mode == "ensembl_release"`` → ``release: int``
                    - ``mode == "assembly"`` → ``assembly: int``
                    - ``mode == "assembly_ensembl_release"`` → ``(assembly: int, release: int)``
                - If *report_only_winner* is ``False``:
                    A list of ``(origin, count)`` pairs where *origin* has the corresponding shape above.
        """
        found_id_list = list()
        none_id_list = list()

        for i in id_list:
            found_id, _ = self._require_track().graph.node_name_alternatives(i)
            if found_id is None:
                none_id_list.append(i)
            else:
                found_id_list.append(found_id)

        if len(none_id_list) > 0:
            self.log.warning(f"Number of unfound IDs: {len(none_id_list)}.")

        identification = self._require_track().identify_source(found_id_list, mode=mode)
        if not report_only_winner:
            return identification
        else:
            return identification[0][0]

    def list_external_databases(self) -> set[str]:
        """Return the set of third-party (non-Ensembl) databases represented in the current graph.

        Returns:
            set[str]: Unique external database names discovered via
                :py:meth:`idtrack._the_graph.TheGraph.available_external_databases`.
        """
        return self._require_track().graph.available_external_databases

    def list_genome_assemblies(self) -> set[int]:
        """List genome assemblies represented in the currently loaded graph.

        Exposes the assembly identifiers discovered when the graph was built. This is a thin wrapper over
        :py:meth:`idtrack._the_graph.TheGraph.available_genome_assemblies` and requires that
        :py:meth:`idtrack.API.build_graph` has been called.

        Returns:
            set[int]: Unique genome assembly identifiers present in the graph (e.g., ``38`` for GRCh38).
        """
        return self._require_track().graph.available_genome_assemblies

    def list_external_databases_by_assembly(self) -> dict[int, set[str]]:
        """Map each genome assembly to the external databases present in that slice of the graph.

        Delegates to :py:meth:`idtrack._the_graph.TheGraph.available_external_databases_assembly` to reveal which
        third-party resources are available per assembly for the loaded organism/release window.

        Returns:
            dict[int, set[str]]: Mapping of assembly → set of external database names.
        """
        return self._require_track().graph.available_external_databases_assembly

    def external_database_forms(self) -> dict[str, str]:
        """Return the Ensembl form each external database connects through.

        Provides a compact view of how third-party databases attach to the Ensembl backbone (``"gene"``,
        ``"transcript"``, or ``"translation"``) via
        :py:meth:`idtrack._the_graph.TheGraph.external_database_connection_form`.

        Returns:
            dict[str, str]: Mapping of external database name → Ensembl form (e.g., ``"gene"``).
        """
        return self._require_track().graph.external_database_connection_form

    def list_ensembl_releases(self) -> list[int]:
        """List Ensembl releases reachable for the configured organism and assembly.

        Wraps :py:meth:`idtrack._database_manager.DatabaseManager.available_releases`, honoring any ignore window
        configured in the manager. The result is sorted in ascending order.

        Returns:
            list[int]: Sorted release numbers that can be queried and cached locally.
        """
        return self._require_track().db_manager.available_releases
