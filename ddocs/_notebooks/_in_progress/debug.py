import sys

sys.path.append("/Users/kemalinecik/git_nosync/idtrack")

import os
import pickle
import time

import idtrack

local_dir = "/Users/kemalinecik/Downloads/idtrack_temp"
idt = idtrack.API(local_repository=local_dir)
idt.configure_logger()
# dm = idt.get_database_manager(organism_name='homo_sapiens', last_ensembl_release=114)

# df=dm.get_db(df_indicator="idhistory_narrow")


idt.initialize_graph(organism_name="homo_sapiens", last_ensembl_release=82, return_test=True)
