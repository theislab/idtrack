#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from idtrack._database_manager import DatabaseManager
from idtrack._dataset import Dataset
from idtrack._dataset import DB
from idtrack._graph_history import GraphHistory
from idtrack._verify_organism import VerifyOrganism

# TODO LIST:
#  -------------------------------------------------Type converter / ensembl converter'ı tamamla, ayır bir class olarak.
#  ------------------------------------------------------------------ Bunları ayrı ayrı tut, bir arada saçma öyle zaten.
#  ------------------------------------------------------------ Version clash'ı çöz bizdeki ve input id list arasındaki.
#  -------------------------------------------------------------------------------------- Initialize dataset'si tamamla.
#  ----------------------------------------------------------------------------------------------- diğer todoo'lara bak!
#  - Basic function'ları yaz
#  - Figure çiz
#  - Örnek kullanım olarak websitesine koy, test et öncesinde
#  - Documentation'u tamamla
#  - Websiteyi tamamla
#  - Test'e birşeyler daha ekle, covarage için
#  - David/Leander/Felix mesaj at.


def logger_config():
    """Todo."""
    logging.basicConfig(
        level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
    )


def initialize_minimal(
    organism,
    form,
    local_repository,
    ignore_before: int = None,
    ignore_after: int = None,
    initialize_datasets: bool = False,
    initialize_graph: bool = False,
    clean_up: bool = True,
    target_format_graph: str = "pickle",
    narrow_search: bool = True,
    narrow_edge_attributes: bool = True,
    compress: bool = True,
    store_raw_always: bool = True,
):
    """Todo.

    Args:
        organism: Todo.
        form: Todo.
        local_repository: Todo.
        ignore_before: Todo.
        ignore_after: Todo.
        initialize_datasets: Todo.
        initialize_graph: Todo.
        clean_up: Todo.
        target_format_graph: Todo.
        narrow_search: Todo.
        narrow_edge_attributes: Todo.
        compress: Todo.
        store_raw_always: Todo.

    Returns:
        Todo.
    """
    logger_config()
    vdf = VerifyOrganism(organism)
    fm = vdf.get_formal_name()
    lr = vdf.get_latest_release()
    dm = DatabaseManager(
        fm, lr, form, local_repository, ignore_before, ignore_after, compress, store_raw_always
    )
    st = Dataset(dm, narrow_search=narrow_search)

    if initialize_datasets:
        st.initialize_external_conversion(to_return=False)
        st.initialize_form_conversion()

    if initialize_graph:
        g: Optional[GraphHistory_Depracated] = GraphHistory_Depracated(dm, target_format_graph, narrow=narrow_edge_attributes)
    else:
        g = None

    if clean_up:
        clean_disk_minimal(dm)

    return dm, vdf, st, g


def clean_disk_minimal(db_manager: DatabaseManager, form_list: Optional[list] = None):
    """Todo.

    Args:
        db_manager: Todo.
        form_list: Todo.
    """
    logger_config()

    if not (form_list is None or isinstance(form_list, list)) or (
        isinstance(form_list, list) and not all([i for i in form_list if i in db_manager.available_form_of_interests])
    ):
        raise ValueError

    form_list = form_list if form_list else db_manager.available_form_of_interests
    either_form = f"(?:{'|'.join(db_manager.available_form_of_interests)})"
    alternative_forms = f"(?:{'|'.join(form_list)})" if len(form_list) > 1 else form_list[0]

    keys = db_manager.tables_in_disk()
    externals = re.compile(f"^/?ens[0-9]+_processed_external_relevant_{alternative_forms}")
    relations = re.compile("^/?ens[0-9]+_common_relationcurrent")
    available = re.compile("^/?ens[0-9]+_common_availabledatabases")
    versioninfo = re.compile(f"^/?ens[0-9]+_processed_versioninfo_{either_form}")

    remove_list = list()
    keep_list = list()
    for k in keys:
        if externals.search(k) or relations.search(k) or available.search(k) or versioninfo.search(k):
            keep_list.append(k)
        else:
            remove_list.append(k)

    if len(keep_list) == 0:
        raise FileNotFoundError("There is no associated files in the directory.")
    elif len(remove_list) > 0:
        db_manager.clean_up(remove_list)


def random_dataset(
    st: Dataset,
    ex: pd.DataFrame,
    d: Optional[str] = None,
    id_count: Optional[int] = None,
    percentage: Optional[float] = None,
):
    """Todo.

    Args:
        st: Todo.
        ex: Todo.
        d: Todo.
        id_count: Todo.
        percentage: Todo.

    Returns:
        Todo.
    """
    if (id_count and percentage) or (not id_count and not percentage):
        raise ValueError

    r = np.random.choice(ex["release"].unique())
    ex = ex[ex["release"] == r]
    if not d:
        dbs = list(ex["name_db"].unique()) + ["graph_id"]
        d = np.random.choice(dbs)

    if d != "graph_id":
        ex = np.unique(ex[ex["name_db"] == d]["id_db"])
    else:
        ex = np.unique(ex["graph_id"])

    if percentage:
        id_count = int(len(ex) / 100 * percentage)

    vv = st.db_manager.check_version_info()
    if d == "graph_id":
        d = st.ensembl_db_no_version if vv == "without_version" else st.ensembl_db
    lst = list(np.random.choice(ex, id_count, replace=False))

    return {"IDs": lst, "Release": r, "Database": d}


def test_db_inference(st, ex, percentage):
    """Todo.

    Args:
        st: Todo.
        ex: Todo.
        percentage: Todo.
    """
    id_dict = random_dataset(st, ex, percentage=percentage)
    res = st.dataset_score_external(ex, id_dict["IDs"], True, True)
    print(f"ID count: {len(id_dict['IDs'])}")
    print(f"Input : {id_dict['Release']}")
    print(f"Result: {[i['Release'] for i in res]}")
    print(f"Input : {id_dict['Database']}")
    print(f"Result: {[i['Database'] for i in res]}")
    print()


def test_external_conversion(st, ex, percentage):
    """Todo.

    Args:
        st: Todo.
        ex: Todo.
        percentage: Todo.

    Returns:
        Todo.
    """
    id_dict = random_dataset(st, ex, percentage=percentage)
    conv = st.convert_external_to_ensembl(id_dict["Release"], id_dict["Database"], id_dict["IDs"])
    return conv


def test_form_conversion(st: Dataset, ex, percentage):
    """Todo.

    Args:
        st: Todo.
        ex: Todo.
        percentage: Todo.

    Returns:
        Todo.
    """
    id_dict = random_dataset(st, ex, d="graph_id", percentage=percentage)
    st.db_manager = st.db_manager.change_release(id_dict["Release"])
    print(f"Input : {id_dict['Release']}")
    conv = st.convert_ensembl_form(id_dict["IDs"], "transcript")
    return conv


# local_repo = "/home/icb/kemal.inecik/temp"
# local_repo = "/Users/kemalinecik/Downloads/idmapping"
# INTEGRATION OF FOLLOWINGS ?: unmapped_object, unmapped_reason, dependent_xref
# TODO: IMPLEMENT that
# https://stackoverflow.com/questions/9522877/pythonic-way-to-have-a-choice-of-2-3-options-as-an-argument-to-a-function/9523071
