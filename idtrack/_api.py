#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import copy
import logging
from typing import Any, Dict, List, Optional, Set, Union

from tqdm.autonotebook import tqdm

from ._database_manager import DatabaseManager
from ._db import DB
from ._track import Track
from ._track_tests import TrackTests
from ._verify_organism import VerifyOrganism


class API:
    """Todo."""

    def __init__(self, local_repository: str) -> None:
        """Todo.

        Args:
            local_repository: Todo.
        """
        # Instance attributes
        self.log = logging.getLogger("api")
        self.logger_configured = False
        self.local_repository = local_repository
        self.track: Union[Track, TrackTests]

    def configure_logger(self):
        """Todo."""
        if not self.logger_configured:
            logging.basicConfig(
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
                format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
            )
            self.logger_configured = True
        else:
            self.log.info("The logger is already configured.")

    def calculate_graph_caches(self):
        """Calculate cached variables of the graph object using only one method."""
        self.track.graph.calculate_caches()

    def get_ensembl_organism(self, tentative_organism_name: str) -> tuple:
        """Todo.

        Args:
            tentative_organism_name: Todo.

        Returns:
            Todo.
        """
        vdf = VerifyOrganism(tentative_organism_name)
        formal_name = vdf.get_formal_name()
        latest_release = vdf.get_latest_release()
        return formal_name, latest_release

    def initialize_graph(self, organism_name: str, ensembl_release: int, return_test: bool = False):
        """Todo.

        Args:
            organism_name: Todo.
            ensembl_release: Todo.
            return_test: Todo.
        """
        backbone_form = copy.deepcopy(DB.backbone_form)
        dm = DatabaseManager(organism_name, ensembl_release, backbone_form, self.local_repository)

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
        """Todo.

        Args:
            identifier: Todo.
            from_release: Todo.
            to_release: Todo.
            final_database: Todo.
            prioritize_to_one_filter: Todo.
            return_path: Todo.

        Returns:
            Todo.
        """
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

        final_ids: List[str] = (
            list({j for i in cnt for j in cnt[i]["final_conversion"]["final_elements"]})
            if not no_corresponding and not no_conversion
            else []
        )
        final_database_conv_: Set[Optional[str]] = (
            {cnt[i]["final_conversion"]["final_database"] for i in cnt}
            if not no_corresponding and not no_conversion
            else {None}
        )
        assert len(final_database_conv_) == 1
        final_database_conv = list(final_database_conv_)[0]

        result: Dict[str, Any] = {
            "final_ids": final_ids,
            "final_database": final_database_conv,
            "graph_id": new_ident,
            "query_id": identifier,
            "no_corresponding": no_corresponding,
            "no_conversion": no_conversion,
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

    def convert_identifier_multiple(self, identifier_list, verbose: bool = True, **kwargs):
        """Todo.

        Args:
            identifier_list: Todo.
            kwargs: Todo
            verbose: Todo.

        Returns:
            Todo.
        """
        result = list()
        with tqdm(identifier_list, mininterval=0.25, disable=not verbose) as loop_obj:
            for identifier in loop_obj:
                loop_obj.set_postfix_str(f"ID:{identifier}", refresh=False)

                result.append(self.convert_identifier(identifier, **kwargs))
        return result

    def infer_identifier_release(self, id_list: list, mode: str = "ensembl_release", report_winner: bool = True):
        """Todo.

        Args:
            id_list: Todo.
            mode: Todo.
            report_winner: Todo.

        Returns:
            Todo.
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
        if not report_winner:
            return identification
        else:
            return identification[0][0]
