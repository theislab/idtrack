#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import copy
import logging
from typing import Optional

from ._database_manager import DatabaseManager
from ._db import DB
from ._track import Track
from ._verbose import logger_config
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
        self.track: Optional[Track] = None

    def configure_logger(self):
        """Todo."""
        if not self.logger_configured:
            logger_config()
            self.logger_configured = True
        else:
            self.log.info("The logger is already configured.")

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

    def initialize_graph(
        self, organism_name: Optional[str] = None, ensembl_release: Optional[int] = None, return_tracker: bool = False
    ) -> None:
        """Todo.

        Args:
            organism_name: Todo.
            ensembl_release: Todo.
            return_tracker: Todo.
        """
        if organism_name is None:
            formal_name = "homo_sapiens"
        if ensembl_release is None:
            formal_name, ensembl_release = self.get_ensembl_organism(str(organism_name))

        backbone_form = copy.deepcopy(DB.backbone_form)
        dm = DatabaseManager(formal_name, ensembl_release, backbone_form, self.local_repository)
        g = Track(dm)

        if return_tracker:
            self.track = g

    def convert_identifier(self, identifier):
        """Todo.

        Args:
            identifier: Todo.
        """
        pass

    def convert_identifier_multiple(self, identifier_list):
        """Todo.

        Args:
            identifier_list: Todo.
        """
        # Here, the identifiers will be
        pass

    def infer_identifier_source(self, identifier_list):
        """Todo.

        Args:
            identifier_list: Todo.
        """
        pass
