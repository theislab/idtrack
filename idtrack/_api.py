#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


from typing import Optional
import logging
import copy

import numpy as np
import pandas as pd

from ._database_manager import DatabaseManager
from ._db import DB
from ._track import Track
from ._verbose import logger_config
from ._verify_organism import VerifyOrganism


class API():
    
    def __init__(self, local_repository: str) -> None:
        # Instance attributes
        self.log = logging.getLogger("api")
        self.logger_configured = False
        self.local_repository = local_repository
        self.track = None
    
    def configure_logger(self):
        if not self.logger_configured:
            logger_config()
            self.logger_configured = True
        else:
            self.log.info("The logger is already configured.")
    
    def get_ensembl_organism(tentative_organism_name: str) -> tuple:
        vdf = VerifyOrganism(tentative_organism_name)
        formal_name = vdf.get_formal_name()
        latest_release = vdf.get_latest_release()
        return formal_name, latest_release
            
    def initialize_graph(self, formal_name: Optional[str] = None, ensembl_release: Optional[int] = None, 
                         return_tracker: bool = False) -> None:
        
        if formal_name is None:
            formal_name = 'homo_sapiens'
        if ensembl_release is None:
            formal_name, ensembl_release = self.get_ensembl_organism(formal_name)
        
        backbone_form = copy.deepcopy(DB.backbone_form)
        dm = DatabaseManager(formal_name, ensembl_release, backbone_form, self.local_repository)
        g = Track(dm)
        
        if return_tracker:
            self.track = g

    def convert_identifier(self, identifier):
        pass
    
    def convert_identifier_multiple(self, identifier_list):
        # Here, the identifiers will be 
        pass
    
    def infer_identifier_source(self, identifier_list):
        pass
    
