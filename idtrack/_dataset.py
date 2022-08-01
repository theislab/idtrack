#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging

import numpy as np
import pandas as pd

from ._database_manager import DatabaseManager
from ._db import DB


class Dataset:
    """Todo."""

    def __init__(self, db_manager: DatabaseManager, narrow_search=True):
        """Todo.

        Args:
            db_manager: Todo.
            narrow_search: Todo.
        """
        self.log = logging.getLogger("dataset")
        self.db_manager = db_manager
        self.narrow_search = narrow_search

        self.ensembl_db = f"ensembl_{self.db_manager.form}_with_version"
        self.ensembl_db_no_version = f"ensembl_{self.db_manager.form}_without_version"

    @staticmethod
    def ensembl_list_check_version(id_lst: list) -> str:
        """Detects naively whether there is ID version or not.

         Args:
            id_lst: List of IDs from Ensembl only.

        Returns:
            "without_version" or "with_version".

        Raises:
            ValueError: When IDs are not string. When the version info is not consistent.
        """
        if not np.all([isinstance(i, str) for i in id_lst]):
            raise ValueError("The IDs in the input list has to be string objects.")
        if not np.all([i.count(DB.id_ver_delimiter) <= 1 for i in id_lst]):
            raise ValueError("The IDs in the input list should not contain more than 1 delimiter.")

        id_vers = [i.find(DB.id_ver_delimiter) == -1 for i in id_lst]
        if np.all(id_vers):
            # If there is no version information associated with stable_ids. For some organisms like S. cerevisiae
            return "without_version"
        elif np.any(id_vers):
            raise ValueError("Inconsistent versions in the IDs in the input list.")
        else:
            return "with_version"

    def ensembl_list_warning_version_consistency(self, id_lst: list):
        """Todo.

        Args:
            id_lst: Todo.
        """
        dbvi = self.db_manager.check_version_info()
        idvi = Dataset.ensembl_list_check_version(id_lst)
        if dbvi != idvi:
            self.log.warning(f"Version info inconsistency: Database is '{dbvi}', but ID list is '{idvi}'.")

    def initialize_external_conversion(self, to_return: bool = True):
        """Todo.

        Args:
            to_return: Todo.

        Returns:
            Todo.
        """
        rel_to_df = sorted(self.db_manager.available_releases, reverse=True)
        self.log.info(f"Comparison data frame is being constructed for releases: {rel_to_df}.")
        ex_all = pd.DataFrame()
        for rel in sorted(rel_to_df, reverse=True):
            db_man_rel = self.db_manager.change_release(rel)
            ex_rel = db_man_rel.get_db("external_relevant" if self.narrow_search else "external")
            if to_return:
                ex_all = pd.concat([ex_all, ex_rel], axis=0)
        if to_return:
            ex_all.reset_index(inplace=True, drop=True)
            return ex_all

    def initialize_external_conversion_uniques(self):
        """Todo.

        Returns:
            Todo.
        """
        ex_all = self.initialize_external_conversion()
        non_uniques_external = ex_all["name_id"].duplicated(keep=False)
        non_uniques_ensembl = ex_all["graph_id"].duplicated(keep=False)
        ex_all_uniques_external = ex_all[~non_uniques_external].copy()
        ex_all_uniques_ensembl = ex_all[~non_uniques_ensembl].copy()
        ex_all = pd.concat([ex_all_uniques_external, ex_all_uniques_ensembl], axis=0)
        ex_all.drop_duplicates(inplace=True, ignore_index=True)
        ex_all.reset_index(inplace=True, drop=True)
        return ex_all

    def initialize_form_conversion(self):
        """Todo."""
        rel_to_df = sorted(self.db_manager.available_releases, reverse=True)
        self.log.info(f"Form conversion data frames are being constructed for releases: {rel_to_df}.")
        for rel in rel_to_df:
            db_man_rel = self.db_manager.change_release(rel)
            _ = db_man_rel.get_db("relationcurrent")

    def dataset_score_external(self, ex_df, id_lst, external: bool, ensembl: bool):
        """Todo.

        Args:
            ex_df: Todo.
            id_lst: Todo.
            external: Todo.
            ensembl: Todo.

        Returns:
            Todo.
        """
        result = []
        # m = exx[(exx["name_db"] == "HGNC Symbol") & (exx["release"] == 96)]["id_db"].unique()
        # ids = list(np.random.choice(m, 5000, replace=False))
        id_lst = list(np.unique(id_lst))
        if external:
            exx = ex_df[["release", "id_db", "name_db"]].drop_duplicates(ignore_index=True)
            ids = pd.Series(id_lst, name="id_db")
            mex = exx.merge(ids, how="right", on="id_db")
            mexg = mex.groupby(["release", "name_db"])
            sme = mexg.size()
            max_sme = sme.max()
            result.extend(
                [(num_ids / len(ids), rel, database) for (rel, database), num_ids in sme.items() if num_ids == max_sme]
            )

        if ensembl:
            exx = ex_df[["release", "graph_id"]].drop_duplicates(ignore_index=True)
            vv = self.db_manager.check_version_info()
            for drop_version in [False, True]:
                vv_switch = vv == "without_version"
                if drop_version and vv_switch:
                    exx["graph_id"] = [i.split(DB.id_ver_delimiter)[0] for i in exx["graph_id"]]
                    vv_switch = not vv_switch
                elif drop_version:
                    continue
                ids = pd.Series(id_lst, name="graph_id")
                mex = exx.merge(ids, how="right", on="graph_id")
                mexg = mex.groupby(["release"])
                sme = mexg.size()
                max_sme = sme.max()
                result.extend(
                    [
                        (num_ids / len(ids), rel, self.ensembl_db if not vv_switch else self.ensembl_db_no_version)
                        for rel, num_ids in sme.items()
                        if num_ids == max_sme
                    ]
                )

        return [
            {"Score": i, "Release": j, "Database": k, "Form": self.db_manager.form}
            for i, j, k in sorted(result, reverse=True)
        ]

    def _convert_external_helper(self, rel, db):
        """Todo.

        Args:
            rel: Todo.
            db: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if db == self.ensembl_db or db == self.ensembl_db_no_version:
            raise ValueError("Ensembl ID to Ensembl ID conversion!")
        else:
            db_man_rel = self.db_manager.change_release(rel)
            ex_rel = db_man_rel.get_db("external_relevant" if self.narrow_search else "external")
            ex_rel = ex_rel[ex_rel["name_db"] == db]  # no need for drop.duplicate as above line already does that
            return ex_rel

    def convert_external_to_ensembl(self, rel, db, id_lst):
        """Todo.

        Args:
            rel: Todo.
            db: Todo.
            id_lst: Todo.

        Returns:
            Todo.
        """
        ex_rel = self._convert_external_helper(rel, db)
        ids = pd.Series(id_lst, name="id_db")
        ex_rel = ex_rel.merge(ids, how="right", on="id_db")
        ex_rel.sort_values(by=["xref_identity", "ensembl_identity"], ascending=False, ignore_index=True)
        return ex_rel[["id_db", "graph_id"]].drop_duplicates(keep="first", ignore_index=True)

    def convert_ensembl_to_external(self, rel, db, id_lst):
        """Todo.

        Args:
            rel: Todo.
            db: Todo.
            id_lst: Todo.

        Returns:
            Todo.
        """
        self.ensembl_list_warning_version_consistency(id_lst)
        ex_rel = self._convert_external_helper(rel, db)
        ids = pd.Series(id_lst, name="graph_id")
        ex_rel = ex_rel.merge(ids, how="right", on="graph_id")
        ex_rel.sort_values(by=["ensembl_identity", "xref_identity"], ascending=False, ignore_index=True)
        return ex_rel[["graph_id", "id_db"]].drop_duplicates(keep="first", ignore_index=True)

    def convert_ensembl_form(self, id_lst, to_form):
        """Todo.

        from release, form of db_manager

        Args:
            id_lst: Todo.
            to_form: Todo.

        Returns:
            Todo.
        """
        self.ensembl_list_warning_version_consistency(id_lst)
        rc = self.db_manager.get_db("relationcurrent", save_after_calculation=self.db_manager.store_raw_always)
        rc.replace(to_replace="", value=np.nan, inplace=True)
        rc = rc[[self.db_manager.form, to_form]]
        rc = rc[~(rc[self.db_manager.form].isna() | rc[to_form].isna())]

        ids = pd.Series(id_lst, name=self.db_manager.form)
        result = rc.merge(ids, how="right", on=self.db_manager.form)
        result.drop_duplicates(inplace=True, ignore_index=True)
        result.reset_index(inplace=True, drop=True)
        return result
