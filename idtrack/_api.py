#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import copy
import logging
import os
from typing import Any, Optional, Union

import numpy as np
from tqdm import tqdm

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._track import Track
from idtrack._track_tests import TrackTests
from idtrack._verify_organism import VerifyOrganism


class API:
    """Application programming interface for simple operations using ``idtrack`` package."""

    def __init__(self, local_repository: str) -> None:
        """Class initialization.

        Args:
            local_repository: An absolute path in local machine to store downloaded and preprocessed content.
        """
        # Instance attributes
        self.log = logging.getLogger("api")
        self.logger_configured = False
        self.local_repository = local_repository
        self.track: Union[Track, TrackTests]

    def configure_logger(self, level=None):
        """Configure logger in a way that shows the logs in a specified format."""
        if not self.logger_configured:
            logging.basicConfig(
                level=logging.INFO if level is None else level,
                datefmt="%Y-%m-%d %H:%M:%S",
                format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
            )
            self.logger_configured = True
        else:
            self.log.info("The logger is already configured.")

    def calculate_graph_caches(self):
        """Calculate cached variables of the graph object using only one method."""
        self.track.graph.calculate_caches()

    def get_ensembl_organism(self, tentative_organism_name: str) -> tuple[str, int]:
        """Make sure the user enters correct organism name and retrieves latest Ensembl release with ease.

        Args:
            tentative_organism_name: Organism of interest.
                This does not have to be formal name. For example, 'human', 'hsapiens' as well as formal name
                'homo_sapiens' is accepted by the program.

        Returns:
            Formal organism name and associated latest Ensembl release.
        """
        vdf = VerifyOrganism(tentative_organism_name)
        formal_name = vdf.get_formal_name()
        latest_release = vdf.get_latest_release()
        return formal_name, latest_release

    def get_database_manager(self, organism_name: str, last_ensembl_release: int):
        """Instantiate and return a DatabaseManager object for a specified organism and Ensembl release.

        This method sets up a DatabaseManager configured to ignore data after a given Ensembl release.
        It uses a deep copy of the default database backbone form and stores data in the local repository
        specified during API initialization.

        Args:
            organism_name: The formal name of the organism (e.g., 'homo_sapiens').
            last_ensembl_release: The most recent Ensembl release to include. Data after this release will be ignored.

        Returns:
            An instance of the DatabaseManager class configured with the provided organism and release settings.
        """
        return DatabaseManager(
            organism=organism_name,
            ensembl_release=None,
            ignore_after=last_ensembl_release,
            form=copy.deepcopy(DB.backbone_form),
            local_repository=self.local_repository,
        )

    def initialize_graph(self, organism_name: str, last_ensembl_release: int, return_test: bool = False):
        """Creates a graph and initializes pathfinder class.

        Args:
            organism_name: Formal organism name as an output of ``get_ensembl_organism`` method.
            last_ensembl_release: Ensembl release of interest. The object will work on only given Ensembl release,
                but some methods does not care which form the DatabaseManager is defined to.
                The latest possible Ensembl release is the best choice for graph building with no drawbacks.
            return_test: If ``True``, return the ``TrackTest`` object instead, which has some functions to test the
                pathfinder performance and graph integrity.
        """
        dm = self.get_database_manager(organism_name=organism_name, last_ensembl_release=last_ensembl_release)

        if return_test:
            self.track = TrackTests(dm)
        else:
            self.track = Track(dm)

    def convert_identifier(
        self,
        identifier: str,
        from_release: Optional[int] = None,
        to_release: Optional[int] = None,
        final_database: Optional[str] = None,
        prioritize_to_one_filter: bool = True,
        return_path: bool = False,
    ) -> dict:
        """Finds corresponding identifier in specified target using the constructed graph and pathfinder algorithm.

        Args:
            identifier: Query ID.
            from_release: Query ID is from which Ensembl release, if provided.
            to_release: Ensembl release for target gene set.
                Which Ensembl release the user wants to convert the ID into. The default is the latest Ensembl release.
            final_database: Database for the target gene set.
                Which database the user wants to convert the ID into. The default is 'ensembl_gene'.
            prioritize_to_one_filter: Decide to use a series of filters to score the possible paths, and ideally choose
                a single target at the end.
            return_path: If ``True``, returns the path from source to query ID.

        Returns:
            A dictionary with following keys

            - ``"target_id"``: Final IDs after conversion.
            - ``"last_node"``: The last node in history travel, so it is an Ensembl gene ID.
            - ``"final_database"``: Final database of the final IDs.
            - ``"graph_id"``: The corresponding ID in the graph for the query ID.
              For example, if the query ID is 'actb', this will be 'ACTB'.
            - ``"query_id"``: The input query ID.
            - ``"no_corresponding"``: If ``True``, there is no such ID in the graph.
            - ``"no_conversion"``: If ``True``, it is not possible to convert into the target. It is 1-to-0 matching.
            - ``"no_target"``: If ``True``, there is no matching in the described final database, but the history
              travel was successful until 'last_node'. In other words, 'final_conversion' failed, no mathing identifier
              is found in the target database but there is mathing identifier in Ensembl.
            - ``"the_path"``: The path from source to query ID. (If `return_path` is set to ``True``.)
        """
        # Get the graph ID if possible.
        new_ident, _ = self.track.graph.node_name_alternatives(identifier)
        no_corresponding, no_conversion = False, False

        if new_ident is not None:
            cnt = self.track.convert(
                from_id=new_ident,
                from_release=from_release,
                to_release=to_release,
                final_database=final_database,
                prioritize_to_one_filter=prioritize_to_one_filter,
                return_path=return_path,
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
        assert len(final_database_conv_) == 1
        final_database_conv = list(final_database_conv_)[0]

        final_conf_: set[Optional[Union[int, float]]] = (
            {cnt[i]["final_conversion"]["final_conversion_confidence"] for i in cnt}
            if not no_corresponding and not no_conversion
            else {None}
        )
        assert len(final_conf_) == 1
        final_conf = list(final_conf_)[0]

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

        if return_path:
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
        self, identifier_list, verbose: bool = True, pbar_prefix: str = "", **kwargs
    ) -> list[dict]:
        """Basically ``convert_identifier`` method for multiple conversion procedure with progress bar.

        Args:
            identifier_list: List of query IDs to feed the ``convert_identifier`` method.
            kwargs: Keyword arguments to pass into ``convert_identifier`` method.
            verbose: If ``True``, shows the progress.
            pbar_prefix: The string to be put before the progress bar.

        Returns:
            List of ``convert_identifier`` method outputs.
        """
        result = list()
        with tqdm(identifier_list, mininterval=0.25, disable=not verbose, desc=pbar_prefix, ncols=100) as loop_obj:
            for identifier in loop_obj:
                loop_obj.set_postfix_str(f"ID:{identifier}", refresh=False)

                result.append(self.convert_identifier(identifier, **kwargs))
        return result

    def classify_multiple_conversion(self, matchings: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """Create a dictionary by classifying the results into different bins.

        Args:
            matchings: The output of ``convert_identifier_multiple`` method.

        Raises:
            ValueError: Unexpected program error

        Returns:
            List of ``convert_identifier`` method outputs.
        """
        r: dict[str, list[dict]] = {
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
            r["input_identifiers"].append(i)

            if i["no_corresponding"]:
                r["matching_1_to_0"].append(i)
                continue

            if i["no_conversion"]:
                r["matching_1_to_0"].append(i)
                continue

            if len(i["target_id"]) == 0:
                raise ValueError("Unexpected error.")

            if i["no_target"]:
                if len(i["target_id"]) == 1:
                    r["alternative_target_1_to_1"].append(i)
                else:
                    r["alternative_target_1_to_n"].append(i)

            else:
                if len(i["target_id"]) == 1 and i["target_id"][0] != i["query_id"]:
                    r["changed_only_1_to_1"].append(i)

                if len(i["target_id"]) > 1 and not any([i["query_id"] == k for k in i["target_id"]]):
                    r["changed_only_1_to_n"].append(i)

                if len(i["target_id"]) == 1:
                    r["matching_1_to_1"].append(i)

                if len(i["target_id"]) > 1:
                    r["matching_1_to_n"].append(i)

        return r

    def print_binned_conversion(self, binned_conversion):
        """Print the output of ``classify_multiple_conversion`` method.

        Args:
            binned_conversion: The output of ``classify_multiple_conversion`` method.
        """
        self.log.info(
            os.linesep + f"{os.linesep}".join([f"{i}: {len(binned_conversion[i])}" for i in binned_conversion])
        )

    def infer_identifier_source(
        self, id_list: list, mode: str = "assembly_ensembl_release", report_only_winner: bool = True
    ):
        """Infer the source of given set of IDs, by comparing which source covers most of the query IDs.

        Args:
            id_list: List of query IDs.
            mode:
                - ``"complete"``: Looks for the best match in terms of database, assembly and Ensembl release.
                - ``"ensembl_release"``: Looks for the best match in terms of Ensembl release only.
                - ``"assembly"``: Looks for the best match in terms of genome assembly only.
                - ``"assembly_ensembl_release"``: Looks for the best match in terms of Ensembl release and assembly.
            report_only_winner: If ``True``, return only the winner. If ``False``, return each
                possible source with corresponding scores.

        Returns:
            Result based on `mode` parameter.
        """
        found_id_list = list()
        none_id_list = list()

        for i in id_list:
            found_id, _ = self.track.graph.node_name_alternatives(i)
            if found_id is None:
                none_id_list.append(i)
            else:
                found_id_list.append(found_id)

        if len(none_id_list) > 0:
            self.log.warning(f"Number of unfound IDs: {len(none_id_list)}.")

        identification = self.track.identify_source(found_id_list, mode=mode)
        if not report_only_winner:
            return identification
        else:
            return identification[0][0]

    def available_available_external_databases(self) -> set[str]:
        """Show which organisms are available to be used in the scope of this package.

        Returns:
            Set of strings saying which external databases are present in the graph.
        """
        return self.track.graph.available_external_databases
