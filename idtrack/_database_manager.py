#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import copy
import os
import re
from collections import Counter
from functools import cached_property
from itertools import repeat

import numpy as np
import pandas as pd
import pymysql.cursors

from ._db import DB
from ._external_databases import ExternalDatabases


class DatabaseManager:
    """Todo."""

    def __init__(
        self,
        organism: str,
        ensembl_release: int,
        form: str,
        local_repository: str,
        ignore_before: int = None,
        ignore_after: int = None,
        compress: bool = True,
        store_raw_always: bool = True,
        genome_assembly: int = None,
    ):
        """Todo.

        Args:
            organism: Todo.
            ensembl_release: Todo.
            form: Todo.
            local_repository: Todo.
            ignore_before: Todo.
            ignore_after: Todo.
            compress: Todo.
            store_raw_always: Todo.
            genome_assembly: Todo.

        Raises:
            ValueError: Todo.
        """
        # MYSQL Settings
        if genome_assembly is None:
            genome_assembly = sorted(
                (DB.assembly_mysqlport_priority[i]['Priority'],i) for i in DB.assembly_mysqlport_priority)[0][1]
        
        self.genome_assembly = genome_assembly
        self.mysql_settings = {
            "host": DB.mysql_host,
            "user": DB.myqsl_user,
            "password": DB.mysql_togo,
            "port": DB.assembly_mysqlport_priority[self.genome_assembly]["Port"],
        }

        # Instance attributes
        self.log = logging.getLogger("database_manager")
        self.local_repository = local_repository
        self.ensembl_release = int(ensembl_release)
        self.organism = organism
        self.form = form
        self.compress = compress
        self.store_raw_always = store_raw_always
        default_min_er = max([DB.assembly_mysqlport_priority[i]["MinRelease"] for i in DB.assembly_mysqlport_priority])
        self.ignore_before = ignore_before if ignore_before else default_min_er
        self.ignore_after = ignore_after if ignore_after else np.inf

        # Protected attributes
        self.available_form_of_interests = copy.deepcopy(DB.forms_in_order)  # Warning: the order is important.
        self._available_version_info = ["add_version", "without_version", "with_version"]
        self._comp_hdf5 = {"complevel": 9, "complib": "blosc:zlib"} if self.compress else dict()
        self._column_sep = "_COL_"
        self._identifiers = [f"{self.form}_stable_id", f"{self.form}_version"]

        # Check if it seems ok.
        checkers = (
            float(ensembl_release) == int(ensembl_release),
            self.ignore_after >= self.ensembl_release >= self.ignore_before,
            self.ensembl_release in self.available_releases,
            self.genome_assembly in DB.assembly_mysqlport_priority,
            self.form in self.available_form_of_interests,
            (
                os.path.isdir(self.local_repository)
                and os.access(self.local_repository, os.W_OK)
                and os.access(self.local_repository, os.R_OK)
            ),
            not (self.ensembl_release < DB.assembly_mysqlport_priority[self.genome_assembly]["MinRelease"])
        )
        if not all(checkers):
            raise ValueError(f"\'DatabaseManager\' could not pass the \'checkers\': {checkers}")

    @cached_property
    def external_inst(self):
        return ExternalDatabases(
            organism=self.organism,
            ensembl_release=self.ensembl_release,
            form=self.form,
            local_repository=self.local_repository,
            genome_assembly=self.genome_assembly
        )

    @cached_property
    def available_releases(self):
        """Todo.

        Returns:
            Todo.
        """
        # Get all possible ensembl releases for a given organism
        dbs = self.get_db("availabledatabases")  # Obtain the databases dataframe
        pattern = re.compile(f"^{self.organism}_core_([0-9]+)_.+$")
        # Search organism name in a specified format. Extract ensembl release number
        releases = [float(pattern.search(i).groups()[0]) for i in dbs if pattern.match(i)]

        # Get rid of floating ensembl releases if exists: In very early releases, there are floating releases
        # like "18.2". This Python package does not support those.
        floating_ensembl = list()
        releases_final = list()
        for r in releases:
            if float(r) == int(r):
                releases_final.append(int(r))
            else:
                floating_ensembl.append(r)
        if len(floating_ensembl) > 0:
            self.log.warning(
                f"Some ensembl releases are included for {self.organism}. "
                f"There are floating ensembl releases: {floating_ensembl}."
            )

        # Sort in ascending order and return the result.
        releases_final = sorted(
            i
            for i in releases_final
            if self.ignore_after >= i >= self.ignore_before  # Filter out the releases that are not of interest.
        )

        return releases_final

    @cached_property
    def mysql_database(self):
        """Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # In MYSQL server, one has to choose a database based on ensembl release and organism name.
        dbs = self.get_db("availabledatabases")  # Obtain the databases dataframe
        # In very early releases, there are floating ensembl releases like "18.2". This package does not support those.
        if not int(self.ensembl_release) == float(self.ensembl_release):
            raise ValueError
        pattern = re.compile(f"^{self.organism}_core_{int(self.ensembl_release)}_.+$")
        located = [i for i in dbs if pattern.match(i)]
        # Search database in a specified format. Extract ensembl release number
        if not len(located) == 1:
            raise ValueError
        # Make sure there are only one database in the server satisfying given ensembl release and organism name.
        return located[0]

    def change_form(self, form):
        """Todo.

        Args:
            form: Todo.

        Returns:
            Todo.
        """
        return DatabaseManager(
            organism=self.organism,
            ensembl_release=self.ensembl_release,
            form=form,
            local_repository=self.local_repository,
            ignore_before=self.ignore_before,
            ignore_after=self.ignore_after,
            compress=self.compress,
            store_raw_always=self.store_raw_always,
            genome_assembly=self.genome_assembly,
        )

    def change_release(self, ensembl_release):
        """Todo.

        Args:
            ensembl_release: Todo.

        Returns:
            Todo.
        """
        return DatabaseManager(
            organism=self.organism,
            ensembl_release=ensembl_release,
            form=self.form,
            local_repository=self.local_repository,
            ignore_before=self.ignore_before,
            ignore_after=self.ignore_after,
            compress=self.compress,
            store_raw_always=self.store_raw_always,
            genome_assembly=self.genome_assembly,
        )

    def change_assembly(self, genome_assembly):
        """Todo.

        Args:
            ensembl_mysql_server: Todo.

        Returns:
            Todo.
        """
        return DatabaseManager(
            organism=self.organism,
            ensembl_release=self.ensembl_release,
            form=self.form,
            local_repository=self.local_repository,
            ignore_before=self.ignore_before,
            ignore_after=self.ignore_after,
            compress=self.compress,
            store_raw_always=self.store_raw_always,
            genome_assembly=genome_assembly,
        )

    def check_exist_as_diff_release(self, df_type, df_indicator):
        """Todo.

        Args:
            df_type: Todo.
            df_indicator: Todo.

        Returns:
            Todo.
        """
        # Get the file name associated with table_key and columns of interest.
        hierarchy, file_path = self.file_name(df_type, df_indicator)

        # The below pattern is based on file_name function with some modifications.
        # Organism name and form is excluded as it does not change the resulting file.
        pattern = re.compile(f"ens([0-9]+)_{df_type}_{df_indicator}")

        if not os.access(file_path, os.R_OK):
            return None, list()

        with pd.HDFStore(file_path, mode="r") as f:
            keys = f.keys()
        downloaded_rels = list(set([int(pattern.search(i).groups()[0]) for i in keys if pattern.search(i)]))

        for dr in downloaded_rels:
            if dr >= self.ensembl_release:
                return dr, downloaded_rels
        return None, downloaded_rels

    def remove_redundant_exist(self, df_type, df_indicator, keep_rel, all_rel_lst):
        """Todo.

        Args:
            df_type: Todo.
            df_indicator: Todo.
            keep_rel: Todo.
            all_rel_lst: Todo.
        """
        for arl in all_rel_lst:
            if arl != keep_rel:
                hi, fi = self.file_name(df_type, df_indicator, ensembl_release=arl)
                with pd.HDFStore(fi, mode="a") as f:
                    self.log.info(
                        f"Following file is being removed: '{os.path.basename(fi)}' with key '{hi}'. "
                        f"This could cause hdf5 file to not reclaim the emptied disk space."
                    )
                    f.remove(hi)

    def create_available_databases(self):
        """Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        self.log.info(
            f"Available MySQL databases for {self.organism} in {self.genome_assembly} "
            f"assembly and {self.ensembl_release} release is being fetched."
        )

        with pymysql.connect(**self.mysql_settings) as connection:
            with connection.cursor() as cur:
                cur.execute("SHOW databases")
                results_query = cur.fetchall()

        if not all([len(i) == 1 and isinstance(i[0], str) for i in results_query]):
            raise ValueError
        results_query = [i[0] for i in results_query]

        pattern = re.compile(f"^{self.organism}_core_[0-9]+_.+$")
        accepted_databases = sorted(i for i in results_query if pattern.match(i))
        results = pd.Series(accepted_databases)

        return results

    def get_table(
        self,
        table_key,
        usecols: list = None,
        create_even_if_exist=False,
        save_after_calculation=True,
        overwrite_even_if_exist=False,
    ):
        """Todo.

        Args:
            table_key: Todo.
            usecols: Todo.
            create_even_if_exist: Todo.
            save_after_calculation: Todo.
            overwrite_even_if_exist: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if not (usecols is None or (isinstance(usecols, list) and len(usecols) > 0)):
            raise ValueError
        # Get the file name associated with table_key and columns of interest.
        hierarchy, file_path = self.file_name("mysql", table_key, usecols)

        # If the file name is not accessible for reading, or if the hdf5 file does not contain the table,
        # or explicitly prompt to do so, then download the table.
        if (
            not os.access(file_path, os.R_OK)
            or create_even_if_exist
            or (not DatabaseManager.check_h5_key(file_path, hierarchy))
        ):
            df = self.download_table(table_key, usecols)
        else:  # Otherwise, just read the file that is already in the directory.
            df = self.read_exported(hierarchy, file_path)

        # If prompt, save the dataframe in requested format.
        if save_after_calculation:
            self.export_disk(df, hierarchy, file_path, overwrite_even_if_exist)

        return df

    def download_table(self, table_key, usecols: list = None):
        """Todo.

        Args:
            table_key: Todo.
            usecols: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # Base settings for MYSQL server.
        which_mysql_server = dict(**self.mysql_settings, **{"database": self.mysql_database})

        # Connect to the MYSQL server, close the connection after the code block
        with pymysql.connect(**which_mysql_server) as connection:

            # Create a cursor to be able to make some queries: First get the associated column names.
            with connection.cursor() as cur1:
                cur1.execute(f"SHOW columns FROM {table_key}")
                column_names = pd.DataFrame(cur1.fetchall())[0]
                if pd.isna(column_names).any():
                    raise ValueError
                # The MYSQL sever before 'DB.mysql_port_min_version' gives the result as bytes, but
                # the one after gives as string.
                if not isinstance(column_names.iloc[0], str):
                    # Convert everything to string to be consistent.
                    column_names = column_names.str.decode("utf-8")
                # Just to make sure conversion is successful, no problem is expected to arise afterwards.
                if not all([isinstance(k, str) and self._column_sep not in k for k in column_names]):
                    raise ValueError

            # Create a cursor to be able to make some queries: Second get the associated table content.
            with connection.cursor() as cur2:
                # Convert the list of column names into string to be used in MYSQL query
                usecol_sql = ", ".join(usecols) if usecols else ""
                cur2.execute(f"SELECT {'*' if usecols is None else usecol_sql} FROM {table_key}")
                # Fetch all the content and save as a tuple file.
                results_content = cur2.fetchall()

            # Create a dataframe using the columns fetched.
            df = pd.DataFrame(results_content, columns=column_names if usecols is None else usecols)
            # Make sure the content does not contain any bytes object.
            if np.any(df.applymap(lambda x: isinstance(x, bytes))):
                raise ValueError

            info_usecols = " for following columns: " + ", ".join(usecols) + "." if usecols else "."
            self.log.info(
                f"Raw table for '{table_key}' on ensembl release '{self.ensembl_release}' "
                f"was downloaded{info_usecols}"
            )

            return df

    # def available_tables_mysql(self):
    #     pass  # todo

    # def get_release_date(self):
    #     pass  # todo

    @staticmethod
    def _determine_usecols_ids(form):
        """Todo.

        Args:
            form: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        stable_id_version = ["stable_id", "version"]
        if form == "gene":
            usecols_core = ["gene_id"] + stable_id_version
            usecols_asso = ["gene_id"]
        elif form == "transcript":
            usecols_core = ["transcript_id"] + stable_id_version
            usecols_asso = ["transcript_id", "gene_id"]
        elif form == "translation":
            usecols_core = ["translation_id"] + stable_id_version
            usecols_asso = ["translation_id", "transcript_id"]
        else:
            raise ValueError()
        return stable_id_version, usecols_core, usecols_asso

    def create_ids(self, form):
        """Todo.

        Args:
            form: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # Determine which columns are interesting for each form.
        if form not in self.available_form_of_interests:
            raise ValueError
        stable_id_version, usecols_core, usecols_asso = DatabaseManager._determine_usecols_ids(form)

        try:
            # In order to have the same column order with below exception code block.
            usecols = usecols_core + [i for i in usecols_asso if i not in usecols_core]
            df = self.get_table(f"{form}", usecols=usecols, save_after_calculation=self.store_raw_always)
            # Earlier versions has different table for stable_id.
            # When there is no associated column for a given table, the following error will be raised.
        except pymysql.err.OperationalError:
            df = self.get_table(f"{form}_stable_id", usecols=usecols_core, save_after_calculation=self.store_raw_always)
            df_2 = (
                self.get_table(f"{form}", usecols=usecols_asso, save_after_calculation=self.store_raw_always)
                if form != "gene"
                else df[usecols_asso]
            )
            df = df.merge(df_2, how="left", on=f"{form}_id", validate="one_to_one")

        # Remove rows with NaN stable_id if exists
        df = df[df["stable_id"].notna()]
        # Convert all IDs to int except stable_id and version.
        for col in [i for i in list(set(usecols_asso + usecols_core)) if i not in stable_id_version]:
            df[col] = df[col].astype(int)
        df["stable_id"] = df["stable_id"].astype(str)  # Convert stable_ids to string
        # Rename to prevent any conflicts in the package
        df.rename(columns={"stable_id": f"{form}_stable_id", "version": f"{form}_version"}, inplace=True)
        df.drop_duplicates(inplace=True, ignore_index=True)  # Remove duplicates if exists
        df.reset_index(inplace=True, drop=True)
        return self.version_uniformize(df, version_str=f"{form}_version")

    def version_uniformize(self, df, version_str):
        """Todo.

        Todo further.

        Args:
            df: Todo.
            version_str: Todo.

        Returns:
            Todo.

        Raises:
            NotImplementedError: Todo.
        """
        contains_na = pd.isna(df[version_str])
        if np.all(contains_na):
            # If there is no version information associated with stable_ids. For some organisms like S. cerevisiae
            df[version_str] = np.nan
            return df
        elif np.any(contains_na):
            raise NotImplementedError("Some versions are NaN, some are not.")
        else:
            df[version_str] = df[version_str].astype(int)
            return df

    def version_fix(self, df, version_str, version_info: str = None):
        """Todo.

        Args:
            df:  Todo.
            version_str: Todo.
            version_info:  Todo.

        Returns:
             Todo.

        Raises:
            ValueError: Todo.
        """
        version_info = version_info if version_info else self.check_version_info()
        if version_info == "add_version":
            df[version_str] = DB.first_version
        elif version_info == "without_version":
            df[version_str] = np.nan
        elif version_info == "with_version":
            df[version_str] = df[version_str].astype(int)
        else:
            raise ValueError
        return df

    def create_relation_current(self):
        """Todo.

        Returns:
            Todo.
        """
        # Get required gene, transcript and translation IDs
        g = self.get_db("idsraw_gene", save_after_calculation=self.store_raw_always)
        t = self.get_db("idsraw_transcript", save_after_calculation=self.store_raw_always)
        p = self.get_db("idsraw_translation", save_after_calculation=self.store_raw_always)

        # Combine them into one
        tgp = t.merge(
            g,
            how="left",
            on="gene_id",
            validate="many_to_one",
        ).merge(p, how="left", on="transcript_id", validate="one_to_one")
        tgp.drop(columns=["gene_id", "transcript_id", "translation_id"], inplace=True, errors="raise")

        return self._create_relation_helper(tgp)

    def create_relation_archive(self):
        """Todo.

        Returns:
            Todo.
        """
        self.log.warning("Not recommended: blablabla")
        # Get the table from the server
        df = self.get_table("gene_archive", usecols=None, save_after_calculation=self.store_raw_always)
        # Remove unnecessary columns and return.
        df.drop(columns=["peptide_archive_id", "mapping_session_id"], inplace=True, errors="raise")

        return self._create_relation_helper(df)

    def _create_relation_helper(self, df):
        """Todo.

        Args:
            df: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # Make sure there are correct number and name of columns
        cols = {
            "gene_stable_id",
            "gene_version",
            "transcript_stable_id",
            "transcript_version",
            "translation_stable_id",
            "translation_version",
        }
        if not (len(df.columns) == len(set(df.columns).intersection(cols)) == 6):
            raise ValueError

        # Process the dataframe
        for col in ["gene", "transcript"]:
            df[f"{col}_version"] = df[f"{col}_version"].astype(int, errors="raise")
            # Transcript may have different version info than gene.
            df = self.change_form(col).version_fix(df, f"{col}_version")

        for col in ["translation"]:
            # Translation may have different version info than gene/transcript.
            df = self.change_form(col).version_fix_incomplete(df, f"{col}_stable_id", f"{col}_version")
            df[f"{col}_version"] = df[f"{col}_version"].astype(float, errors="raise")  # due to np.nans

        for col in ["gene_stable_id", "translation_stable_id", "transcript_stable_id"]:
            df[col] = df[col].fillna("").astype(str)

        res = pd.DataFrame()
        for col in ["gene", "transcript", "translation"]:
            dm = self.change_form(col)
            res[col] = dm.id_ver_from_df(df[[f"{col}_stable_id", f"{col}_version"]])

        # Drop duplicates if exists and return.
        res.drop_duplicates(inplace=True, ignore_index=True)
        res.reset_index(inplace=True, drop=True)
        return res

    def version_fix_incomplete(self, df_fx, id_col_fx, ver_col_fx):
        """Todo.

        Args:
            df_fx: Todo.
            id_col_fx: Todo.
            ver_col_fx: Todo.

        Returns:
            Todo.
        """
        # Get the columns that do not have any id
        na_cols_fx = pd.isna(df_fx[id_col_fx])
        # Split the dataframe to process separately
        df_fm1, df_fm2 = df_fx[na_cols_fx].copy(deep=True), df_fx[~na_cols_fx].copy(deep=True)
        version_info = self.check_version_info()
        df_fm1 = self.version_fix(df_fm1, ver_col_fx, version_info="without_version")
        df_fm2 = self.version_fix(df_fm2, ver_col_fx, version_info=version_info)
        # Concatenate the results and return.
        df_fx = pd.concat([df_fm1, df_fm2], axis=0)
        df_fx.reset_index(inplace=True, drop=True)
        return df_fx

    def create_id_history(self, narrow):
        """Todo.

        Args:
            narrow: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # Get the tables from the server
        s = self.get_table("stable_id_event", usecols=None, save_after_calculation=self.store_raw_always)
        m = self.get_table("mapping_session", usecols=None, save_after_calculation=self.store_raw_always)
        # Combine them into one and filter only the form of interest.
        sm = pd.merge(s, m, how="outer", on="mapping_session_id")
        sm = sm[sm["type"] == self.form]

        if narrow:
            # Remove some unnecessary columns if prompt so.
            sm.drop(
                columns=[
                    "mapping_session_id",
                    "type",
                    "old_db_name",
                    "new_db_name",
                    "old_assembly",
                    "new_assembly",
                    "created",
                ],
                inplace=True,
            )

        # Correct the version based on version_info for each old_stable_id and new_stable_id columns.
        sm = self.version_fix_incomplete(
            self.version_fix_incomplete(sm, "old_stable_id", "old_version"), "new_stable_id", "new_version"
        )
        # Convert np.nan's to "" so that saving to hdf5 file is not a problem.
        for col in ["new_stable_id", "old_stable_id"]:
            sm[col] = sm[col].fillna("")
            sm[col] = sm[col].astype(str)
        for col in ["score", "old_release", "new_release"]:
            sm[col] = sm[col].astype(float)
        sm["score"] = sm["score"].replace(0.0, np.nan)  # 0 means no information according to ensembl.

        # Check the delimiter is not in the ID.
        if not np.all(sm["new_stable_id"].str.find(DB.id_ver_delimiter) == -1):
            raise ValueError
        if not np.all(sm["old_stable_id"].str.find(DB.id_ver_delimiter) == -1):
            raise ValueError
        # No need to check for version as it can be already float or int by fix_stable_events

        sm = sm[(self.ignore_after >= sm["old_release"]) & (sm["old_release"] >= self.ignore_before)]
        sm.drop_duplicates(inplace=True, ignore_index=True)  # Remove duplicates if exists
        sm.reset_index(inplace=True, drop=True)
        return sm

    def create_id_history_fixed(self, narrow, inspect):
        """Todo.

        This method created a fixed version of idhistory. It fixes the problems seen in some exceptional cases like
        Homo sapiens ENSG00000232423 at release 105. The raw version gives the order of versions as follows: 1-2, 2-3,
        1-2, 2-3 and so on. However, after second connection, version 1 should be already lost, the last active version
        should be 3. For this reason, 1-2 connection should be corrected as 3-2. This method does this.

        Args:
            narrow: Todo.
            inspect: Todo.

        Returns:
            Todo.
        """
        # Get the raw version of idhistory first, and sort.
        df = self.get_db("idhistory" if not narrow else "idhistory_narrow")
        df.sort_values(by=["new_release"], inplace=True)
        # Initialize some temp variables
        extinct_version, last_active_version, corrected_entries = dict(), dict(), list()

        for ind, row in df.iterrows():
            changed_old, changed_new = np.nan, np.nan
            # If old_stable_id and new_stable_id is different, then we are not interested in those; because extinction
            # of specific version does not have to exist. For example, it can basically branch.
            if row["old_stable_id"] != row["new_stable_id"]:  # not of interested
                continue

            # Self-loops are also not of interest, because it does not cause a specific version of an ID to be extinct.
            elif row["old_version"] == row["new_version"]:
                continue

            else:
                row_key = row["old_stable_id"]
                # If old_stable_id not seen before in the temp dictionaries, then basically add.
                if row_key not in extinct_version:
                    extinct_version[row_key] = set()
                    last_active_version[row_key] = None

                # Add the version into the set.
                if row["old_version"] not in extinct_version[row_key]:
                    extinct_version[row_key].add(row["old_version"])
                    # Save the last active version
                    last_active_version[row_key] = row
                else:  # If old_version is already seen before.
                    # Replace the value in the database with the last_active_version's associated rows.
                    df.at[ind, "old_version"] = last_active_version[row_key]["new_version"]
                    changed_old, changed_new = row["old_version"], row["new_version"]

                # If new_version is seen, then basically remove it.
                if row["new_version"] in extinct_version[row_key]:
                    extinct_version[row_key] -= {row["new_version"]}

            # If inspect is on, then add the changed parameters.
            if inspect:
                corrected_entries.append(
                    (changed_old, changed_new),
                )
        # If inspect is on, then add these columns as new ones into the existing one.
        if inspect:
            ce = pd.DataFrame(corrected_entries, columns=["unfixed_old_version", "unfixed_new_version"])
            df = pd.concat([df, ce], axis=1)

        df.reset_index(inplace=True, drop=True)
        return df

    def create_external_db(self, filter_mode):
        """Todo.

        Not exactly this but similar to the:
            SELECT g.stable_id, t.stable_id, tr.stable_id, x.dbprimary_acc, edb.db_name, es.synonym, ix.*
            FROM gene g
            JOIN transcript t USING (gene_id)
            JOIN translation tr USING (transcript_id)
            JOIN object_xref ox ON (g.gene_id = ox.ensembl_id AND ox.ensembl_object_type = "Gene")
            JOIN xref x ON (ox.xref_id = x.xref_id) ##
            LEFT JOIN external_db edb ON (x.external_db_id = edb.external_db_id)
            LEFT JOIN identity_xref ix ON (ox.object_xref_id = ix.object_xref_id)
            LEFT JOIN external_synonym es ON (x.xref_id = es.xref_id)
            LIMIT 10;

        alternatively
            FROM coord_system cs
            JOIN seq_region sr USING (coord_system_id)
            JOIN gene g USING (seq_region_id)

        Using the following:
            mysql --user=anonymous --host=ensembldb.ensembl.org -D homo_sapiens_core_105_38 -A

        https://m.ensembl.org/info/docs/api/core/core_schema.html

        Args:
            filter_mode: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # Get the necessary tables from the server
        m = {"save_after_calculation": self.store_raw_always}
        a = self.get_db(f"idsraw_{self.form}", save_after_calculation=self.store_raw_always)
        ox = self.get_table(
            "object_xref", usecols=["ensembl_id", "ensembl_object_type", "xref_id", "object_xref_id"], **m
        )
        x = self.get_table("xref", usecols=["xref_id", "external_db_id", "dbprimary_acc", "display_label"], **m)
        ed = self.get_table("external_db", usecols=["external_db_id", "db_name", "db_display_name"], **m)
        ix = self.get_table("identity_xref", usecols=["ensembl_identity", "xref_identity", "object_xref_id"], **m)
        es = self.get_table("external_synonym", usecols=["xref_id", "synonym"], **m)

        # Make entities in synonyms table appended to the xref file as additional lines
        es = es.merge(x, how="inner", on="xref_id", validate="many_to_one")
        es.drop(columns=["display_label"], inplace=True)
        es.rename(columns={"synonym": "display_label"}, inplace=True)
        es["display_label"] = DB.synonym_id_nodes_prefix + es["display_label"]
        x = pd.concat([x, es], ignore_index=True)

        # Merge the tables as requested
        comb = pd.merge(
            a,
            ox[ox["ensembl_object_type"].str.lower() == self.form.lower()],
            how="left",
            left_on=f"{self.form}_id",
            right_on="ensembl_id",
            validate="one_to_many",
        )
        comb = comb.merge(x, how="left", on="xref_id", validate="many_to_many")
        comb = comb.merge(ed, how="left", on="external_db_id", validate="many_to_one")
        comb = comb.merge(ix, how="left", on="object_xref_id", validate="many_to_one")

        # Remove unnecessary columns and reset the index.
        stable_id_version, usecols_core, usecols_asso = DatabaseManager._determine_usecols_ids(self.form)
        ids_only = list(set(usecols_asso + usecols_core) - set(stable_id_version))
        comb.drop(
            columns=["ensembl_id", "object_xref_id", "ensembl_object_type", "xref_id", "external_db_id"] + ids_only,
            inplace=True,
        )
        comb.reset_index(inplace=True, drop=True)

        # Processing the merged dataframe

        # Constants for the processing
        identities = ["ensembl_identity", "xref_identity"]
        db_id = "id_db"
        db_name = "name_db"
        id_graph = "graph_id"
        count_col = "count"

        def comb_renamer(col_list):
            return {col_list[0]: db_id, col_list[1]: db_name}

        # Create "ID.Version"
        # No need for version_uniformize as the gene_ids are obtained from create_id
        comb_temp = pd.Series(self.id_ver_from_df(comb[self._identifiers]))
        # Basically split below columns as separate rows and rename the columns.
        comb_3_columns = ["display_label", "db_display_name"]
        comb_4_columns = ["dbprimary_acc", "db_name"]
        comb_3, comb_4 = pd.DataFrame(), pd.DataFrame()
        comb_3[id_graph], comb_4[id_graph] = comb_temp, comb_temp
        comb_3[comb_3_columns + identities] = comb[comb_3_columns + identities]
        comb_4[comb_4_columns + identities] = comb[comb_4_columns + identities]
        comb_3.rename(columns=comb_renamer(comb_3_columns), inplace=True, errors="raise")
        comb_4.rename(columns=comb_renamer(comb_4_columns), inplace=True, errors="raise")
        comb_3.reset_index(inplace=True, drop=True)
        comb_4.reset_index(inplace=True, drop=True)
        res = pd.concat([comb_3, comb_4], axis=0, ignore_index=True)
        res = res[~(res[id_graph].isna() | res[db_name].isna() | res[db_id].isna())]
        res.sort_values(by=[db_name, id_graph, db_id], inplace=True)
        res.reset_index(inplace=True, drop=True)
        # Add the release information at the leftmost place
        res.insert(0, "release", pd.Series(repeat(self.ensembl_release, len(res))))

        # Convert some columns type for convenience.
        res[identities[0]] = res[identities[0]].astype(np.float16, errors="raise")
        res[identities[1]] = res[identities[1]].astype(np.float16, errors="raise")
        res["release"] = res["release"].astype(np.uint8, errors="raise")

        # Drop duplicates if exists. Note that it is not trivial, there are many duplicated lines after adding
        # these columns as rows. Because, for some of them, comb_X_columns are actually the same.
        res.drop_duplicates(inplace=True, ignore_index=True)

        # Change the synonym IDs' database name
        to_add = np.array(
            [DB.synonym_id_nodes_prefix if i else "" for i in res[db_id].str.startswith(DB.synonym_id_nodes_prefix)]
        )
        res[db_name] = to_add + res[db_name]
        # Unless you specifically look at synonyms, they will not mean the same thing as the counterparts.
        # They will be used as the bridging point in the pathfinder algorithm only.

        if filter_mode in ["relevant", "relevant-database"]:
            # In order to prevent the search space to be too big and to prevent unnecessary data to be kept in the disk
            # and in the memory.
            isin_list = self.external_inst.give_list_for_case(give_type="db")
            available_databases = set(np.unique(res[db_name]))
            if not all([il in available_databases for il in isin_list]):
                raise ValueError("Inconsistency between external yaml file and current state of DatabaseManager.")
            res = res[res[db_name].isin(isin_list)]

        res.reset_index(inplace=True, drop=True)

        if filter_mode in ["all", "relevant"]:
            return res
        elif filter_mode in ["database", "relevant-database"]:
            databases = pd.DataFrame(Counter(res[db_name]).most_common(), columns=[db_name, count_col])
            return databases
        else:
            raise ValueError

    def get_db(
        self, df_indicator, create_even_if_exist=False, save_after_calculation=True, overwrite_even_if_exist=False
    ):
        """For saving, exporting, and naming convention.

        Args:
            df_indicator: Todo.
            create_even_if_exist: Todo.
            save_after_calculation: Todo.
            overwrite_even_if_exist: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        # Split the df_indicator with "-", to get the extra parameters.
        # Main point of naming and df_indicator is to include the paramters in the file_names
        # for exporting and importing without writing explicit methods of read/write for each case.
        split_ind = df_indicator.split("_")
        main_ind = split_ind[0]
        param1_ind = split_ind[1] if len(split_ind) > 1 else None
        if len(split_ind) > 2:
            raise ValueError

        # For 'availabledatabases' accept different files explained in the below functions.
        if main_ind == "availabledatabases" and param1_ind is None:
            xr1, xr1_a = self.check_exist_as_diff_release("common", df_indicator)
            self.remove_redundant_exist("common", df_indicator, xr1, xr1_a)
            hierarchy, file_path = self.file_name("common", df_indicator, ensembl_release=xr1)
        elif main_ind == "versioninfo" and param1_ind is None:
            xr1, xr1_a = self.check_exist_as_diff_release("processed", df_indicator)
            self.remove_redundant_exist("processed", df_indicator, xr1, xr1_a)
            hierarchy, file_path = self.file_name("processed", df_indicator, ensembl_release=xr1)
        elif param1_ind is None and main_ind in ("relationarchive", "relationcurrent"):
            hierarchy, file_path = self.file_name("common", df_indicator)
        else:
            # Get the file name associated with table_key and columns of interest.
            hierarchy, file_path = self.file_name("processed", df_indicator)

        # If the file name is not accessible for reading, or if the hdf5 file does not contain the table,
        # or explicitly prompt to do so, then download the table.
        if (
            not os.access(file_path, os.R_OK)
            or create_even_if_exist
            or (not DatabaseManager.check_h5_key(file_path, hierarchy))
        ):

            if main_ind == "external" and param1_ind is None:
                df = self.create_external_db(filter_mode="all")

            elif main_ind == "external" and param1_ind in ["relevant", "database", "relevant-database"]:
                df = self.create_external_db(filter_mode=param1_ind)

            elif main_ind == "idsraw":
                if param1_ind not in self.available_form_of_interests:
                    raise ValueError(
                        f"'idsraw' should be used together with one "
                        f"of followings: {','.join(self.available_form_of_interests)}."
                    )
                df = self.create_ids(form=param1_ind)

            elif main_ind == "ids" and param1_ind is None:
                df = self.get_release_id()

            elif main_ind == "externalcontent" and param1_ind is None:
                df = self.create_database_content()

            elif main_ind == "relationcurrent" and param1_ind is None:
                df = self.create_relation_current()

            elif main_ind == "relationarchive" and param1_ind is None:
                df = self.create_relation_archive()

            elif main_ind == "idhistory" and param1_ind is None:
                df = self.create_id_history(narrow=False)

            elif main_ind == "idhistory" and param1_ind == "narrow":
                df = self.create_id_history(narrow=True)

            elif main_ind == "versioninfo" and param1_ind is None:
                df = self.create_version_info()

            elif main_ind == "availabledatabases" and param1_ind is None:
                df = self.create_available_databases()

            else:
                raise ValueError("Unexpected entry for 'df_indicator'.")

        else:  # Otherwise, just read the file that is already in the directory.
            df = self.read_exported(hierarchy, file_path)

        # If prompt, save the dataframe in requested format.
        if save_after_calculation:
            self.export_disk(df, hierarchy, file_path, overwrite_even_if_exist)

        return df

    def read_exported(self, hierarchy, file_path):
        """Todo.

        Args:
            hierarchy: Todo.
            file_path: Todo.

        Returns:
            Todo.

        Raises:
            FileNotFoundError: Todo.
            KeyError: Todo.
        """
        if not os.access(file_path, os.R_OK):
            raise FileNotFoundError

        if not DatabaseManager.check_h5_key(file_path, hierarchy):
            raise KeyError

        df = pd.read_hdf(file_path, key=hierarchy, mode="r")
        return df

    def file_name(self, df_type, *args, ensembl_release: int = None, **kwargs):
        """Todo.

        Args:
            df_type: Todo.
            args: Todo.
            ensembl_release: Todo.
            kwargs: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        ensembl_release = self.ensembl_release if not ensembl_release else ensembl_release

        def file_name_processed(df_indicator: str):
            return f"ens{ensembl_release}_{df_type}_{df_indicator}_{self.form}"

        def file_name_mysql(table_key: str, usecols: list = None):
            col_suffix = f"{self._column_sep}{self._column_sep.join(sorted(usecols))}" if usecols is not None else ""
            return f"ens{ensembl_release}_{df_type}_{table_key}{col_suffix}"

        def file_name_noform(df_indicator: str):
            return f"ens{ensembl_release}_{df_type}_{df_indicator}"

        if df_type not in ["processed", "mysql", "common"]:
            raise ValueError
        if df_type == "processed":
            hierarchy = file_name_processed(*args, **kwargs)
        elif df_type == "common":
            hierarchy = file_name_noform(*args, **kwargs)
        else:
            hierarchy = file_name_mysql(*args, **kwargs)

        return hierarchy, os.path.join(
            self.local_repository, f"{self.organism}_assembly-{self.genome_assembly}.h5"
        )

    def export_disk(self, df, hierarchy, file_path, overwrite: bool):
        """Todo.

        Args:
            df: Todo.
            hierarchy: Todo.
            file_path: Todo.
            overwrite: Todo.
        """
        base_file_path = os.path.basename(file_path)

        if not os.access(file_path, os.R_OK) or overwrite or (not DatabaseManager.check_h5_key(file_path, hierarchy)):

            # Remove the file first to prevent hdf5 file to go arbitrarily larger after writing.
            if DatabaseManager.check_h5_key(file_path, hierarchy) or overwrite:
                with pd.HDFStore(file_path, mode="a") as f:
                    if hierarchy in f:
                        self.log.info(
                            f"Following file is being removed: '{os.path.basename(file_path)}' "
                            f"with key '{hierarchy}'. This could cause hdf5 file to not reclaim the "
                            f"newly emptied disk space."
                        )
                        f.remove(hierarchy)
            # Then save the dataframe under the root, compressed.
            self.log.info(
                f"Exporting to the following file '{base_file_path}' with key '{hierarchy}', "
                f"{'' if self.compress else 'uncompressed'}."
            )
            df.to_hdf(file_path, key=hierarchy, mode="a", **self._comp_hdf5)

    @staticmethod
    def check_h5_key(file_path, key):
        """Todo.

        Args:
            file_path: Todo.
            key: Todo.

        Returns:
            Todo.
        """
        if not os.access(file_path, os.R_OK):
            return False
        with pd.HDFStore(file_path, mode="r") as f:
            return key in f

    def get_release_id(self):
        """Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        dbm_the_ids = self.get_db(f"idsraw_{self.form}")
        dbm_the_ids = self.version_fix(dbm_the_ids, f"{self.form}_version")
        dbm_the_ids = dbm_the_ids[self._identifiers]

        if not np.all(dbm_the_ids[f"{self.form}_stable_id"].str.find(DB.id_ver_delimiter) == -1):
            raise ValueError

        dbm_the_ids.drop_duplicates(keep="first", inplace=True)
        if not dbm_the_ids[f"{self.form}_stable_id"].is_unique:
            raise ValueError

        dbm_the_ids.reset_index(inplace=True, drop=True)
        dbm_the_ids.drop_duplicates(inplace=True)
        return dbm_the_ids

    def repack_hdf5(self, remove_list: list = None):
        """Todo.

        Args:
            remove_list: Todo.
        """
        _, file_name = self.file_name("common", "place_holder")
        old_name = file_name + "_to_delete_temp"
        os.rename(file_name, old_name)

        with pd.HDFStore(old_name, mode="r") as f:
            all_keys = f.keys()

        if not remove_list:
            keys = all_keys
        else:
            keys = [i for i in all_keys if i not in remove_list]

        if len(keys) != 0:
            self.log.disabled = True
            for key in keys:
                df = self.read_exported(key, old_name)
                self.export_disk(df, key, file_name, overwrite=False)
            self.log.disabled = False
        else:
            os.remove(file_name)

        os.remove(old_name)

    def tables_in_disk(self):
        """Todo.

        Returns:
            Todo.
        """
        _, file_name = self.file_name("common", "place_holder")

        if not os.path.isfile(file_name):
            return list()
        else:
            with pd.HDFStore(file_name, mode="r") as f:
                return f.keys()

    def id_ver_from_df(self, dbm_the_ids):
        """Todo.

        Args:
            dbm_the_ids: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if np.all(dbm_the_ids.columns != self._identifiers):
            raise ValueError
        gri_generator = (DatabaseManager.node_dict_maker(i, j) for i, j in dbm_the_ids.values)
        return list(map(DatabaseManager.node_name_maker, gri_generator))

    def clean_up(self, remove_list):
        """Todo.

        Args:
            remove_list: Todo.
        """
        self.repack_hdf5(remove_list)

    @staticmethod
    def node_name_maker(node_dict):
        """This function creates ID-Version.

        If the Version information is not there, it only uses ID, which is necessary for some organisms which
        does not have versioned IDs.

        Args:
            node_dict: Todo.

        Returns:
            Todo.
        """
        if node_dict["Version"] and not pd.isna(node_dict["Version"]):
            return node_dict["ID"] + DB.id_ver_delimiter + str(node_dict["Version"])
        else:
            return node_dict["ID"]

    @staticmethod
    def node_dict_maker(id_entry, version_entry):
        """Create a dict for ID and Version.

        Args:
            id_entry: Todo.
            version_entry: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        if (
            version_entry
            and not pd.isna(version_entry)
            and version_entry != DB.no_old_node_id
            and version_entry != DB.no_new_node_id
        ):
            if int(version_entry) != float(version_entry):
                raise ValueError
            else:
                version_entry = int(version_entry)
        return {"ID": id_entry, "Version": version_entry}

    def check_version_info(self):
        """Todo.

        the same for all the release for a given animal.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
        """
        vi_df = self.get_db("versioninfo")
        narrowed = vi_df["version_info"].unique()
        if narrowed.dtype != np.dtype("bool"):
            raise ValueError
        if len(narrowed) == 1 and narrowed[0]:
            return "without_version"
        elif len(narrowed) == 1:
            return "with_version"
        else:
            return "add_version"

    def create_version_info(self):
        """Todo.

        Returns:
            Todo.

        Raises:
            NotImplementedError: Todo.
        """
        ver = list()
        for i in sorted(self.available_releases, reverse=True):
            db_manager_temp = self.change_release(i)
            df_ids = db_manager_temp.get_db(f"idsraw_{self.form}", save_after_calculation=self.store_raw_always)
            _vv = pd.isna(df_ids[f"{self.form}_version"])
            if np.all(_vv):
                with_version = True
            elif np.any(_vv):
                raise NotImplementedError("Some versions are NaN, some are not.")
            else:
                with_version = False
            ver.append([i, with_version])
        df = pd.DataFrame(ver, columns=["ensembl_release", "version_info"])
        return df

    def create_database_content(self):
        """Todo.

        Returns:
            Todo.
        """
        df = pd.DataFrame()
        for k in DB.assembly_mysqlport_priority.keys():
            for j in self.available_form_of_interests:
                for i in self.available_releases:
                    self.log.info(f"Database content is being created for "
                                  f"\'{self.organism}\', assembly \'{k}\', form \'{j}\', ensembl release \'{i}\'")
                    df_temp = self.change_assembly(k).change_release(i).change_form(j).get_db("external_database")
                    df_temp["assembly"] = k
                    df_temp["release"] = i
                    df_temp["form"] = j
                    df = pd.concat([df, df_temp], axis=0)
        df["organism"] = self.organism
        df.reset_index(inplace=True, drop=True)
        return df

    def create_external_all(self, return_mode):
        """Todo.

        Returns:
            Todo.
        """
        ass = self.external_inst.give_list_for_case(give_type="assembly")
        df = pd.DataFrame()
        assembly_priority = [DB.assembly_mysqlport_priority[i]["Priority"] for i in ass]

        for i in [x for _, x in sorted(zip(assembly_priority, ass))]:  # sort according to priority
            dm = self.change_assembly(i)
            df_temp = dm.get_db("external_relevant")
            df_temp["assembly"] = i
            df = pd.concat([df, df_temp])
        df.reset_index(drop=True, inplace=True)
        compare_columns = [i for i in df.columns
                           if i != "assembly" and not i.endswith("_identity")]  # 'ensembl_identity', 'xref_identity'
        compare_columns_2 = compare_columns + ["assembly"]

        if return_mode == "all":
            df.drop_duplicates(keep="first", inplace=True, ignore_index=True, subset=compare_columns_2)
            return df

        elif return_mode == "unique":
            df.drop_duplicates(keep="first", inplace=True, ignore_index=True, subset=compare_columns)
            # drop duplicates: after transition to new assembly. ensembl does not assign new versions etc to the older
            # keep the most priority one.
            return df

        elif return_mode == "duplicated":
            df = df[df.duplicated(subset=compare_columns, keep=False)]
            dfg = df.groupby(by=compare_columns)
            return dfg

        else:
            raise ValueError
