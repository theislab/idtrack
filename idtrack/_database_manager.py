#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import copy
import logging
import os
import re
from collections import Counter
from functools import cached_property
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymysql

import idtrack._utils_hdf5 as hs
from idtrack._db import DB
from idtrack._external_databases import ExternalDatabases


class DatabaseManager:
    """Downloads, preprocesses, and stores the necessary source files for the program."""

    def __init__(
        self,
        organism: str,
        form: str,
        local_repository: str,
        ensembl_release: Optional[int] = None,
        ignore_before: Optional[int] = None,
        ignore_after: Optional[Union[int, float]] = None,
        store_raw_always: bool = True,
        genome_assembly: Optional[int] = None,
    ):
        """Class initialization.

        Args:
            organism: Formal organism name. The output provided by :py:class:`VerifyOrganism` would be
                a perfect choice.
            ensembl_release: Ensembl release of interest. The object will work on only given Ensembl release, but some
                methods does not care which form the DatabaseManager is defined to.
                The latest possible Ensembl release is the best choice for graph building with no drawbacks.
            form: Either 'gene', 'transcript' or 'translation'. The object will work on only given form, but some
                methods does not care which form the DatabaseManager is defined to.
            local_repository: An absolute path in local machine to store downloaded and preprocessed content.
            ignore_before: Ensembl release as the lower limit to include in the downloaded contents. The object will
                ignore all Ensembl release lower than this integer.
            ignore_after: Similar to 'ignore_before' but as the upper limit as the name suggest.
            store_raw_always: If ``True``, the raw MySQL tables will be also saved in the disk.
            genome_assembly: Genome assembly of interest. The selection should be one of the keys in the
                :py:attr:`DB.assembly_mysqlport_priority` dictionary. The object will work on only given assembly.
                The default is the latest genome assembly (also called highest priority assembly).

        Raises:
            ValueError: When the input parameters are not in the specified format.
            NotImplementedError: Currently only supports 'homo_sapiens' as the organism name.
        """
        if organism not in ["homo_sapiens", "mus_musculus"]:
            raise NotImplementedError(
                "Organisms other than human and mouse is not implemented. In theory, it should work but "
                "no tests have been conducted yet. In the next version of the package, other "
                "organisms will be available. Please note that adding new organisms necessitates "
                "to determine which external databases to include using ExternalDatabase class."
            )

        if genome_assembly is None:  # Set default genome assembly when not specified specifically.
            genome_assembly = sorted(
                (DB.assembly_mysqlport_priority[i]["Priority"], i) for i in DB.assembly_mysqlport_priority
            )[0][
                1
            ]  # Have the most important priority genome assembly as the default value.

        # MYSQL Settings
        self.genome_assembly = genome_assembly
        self.mysql_settings = {
            "host": DB.mysql_host,
            "user": DB.myqsl_user,
            "password": DB.mysql_togo,
            # Port depends on which genome assembly is of interest. Refer to the following link.
            # https://www.ensembl.org/info/data/mysql.html
            "port": DB.assembly_mysqlport_priority[self.genome_assembly]["Port"],
        }

        # The logger for informing the user about the progress.
        self.log = logging.getLogger("database_manager")

        # Instance attributes
        self.local_repository = local_repository
        self.organism = organism
        self.form = form
        self.store_raw_always = store_raw_always
        # If ignore_before is not specified clearly, than use the lowest possible priority defined by the MySQL server.
        default_min_er = max(DB.assembly_mysqlport_priority[i]["MinRelease"] for i in DB.assembly_mysqlport_priority)
        self.ignore_before: int = ignore_before if ignore_before is not None else default_min_er
        # If ignore_after is not specified, than set it to infinite.
        self.ignore_after: Union[int, float] = ignore_after if ignore_after is not None else np.inf

        # Protected attributes
        self.available_form_of_interests = copy.deepcopy(DB.forms_in_order)  # Warning: the order is important.
        self._available_version_info = ["add_version", "without_version", "with_version"]
        self._column_sep = "_COL_"
        self._identifiers = [f"{self.form}_stable_id", f"{self.form}_version"]

        if ensembl_release is None:  # Set last possible Ensembl release for given genome assembly.
            self.ensembl_release = 0  # placeholder value for for file naming method, it is not gonna be saved anyway.
            ensembl_release = sorted(self.available_releases_no_save)[-1]
        self.ensembl_release = int(ensembl_release)

        # Check if it seems ok.
        checkers = (
            # In very early releases, there are floating ensembl releases like "18.2". This package does not support.
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
            not (self.ensembl_release < DB.assembly_mysqlport_priority[self.genome_assembly]["MinRelease"]),
        )
        if not all(checkers):
            raise ValueError(f"'DatabaseManager' could not pass the 'checkers': {checkers}")

    def __str__(self) -> str:
        """Makes the instance status to be inspected by the user easily."""
        return (
            f"DatabaseManager instance:{os.linesep}"
            f"    Organism: {self.organism}{os.linesep}"
            f"    Form: {self.form}{os.linesep}"
            f"    Ensembl Release: {self.ensembl_release}{os.linesep}"
            f"    Genome Assembly: {self.genome_assembly}{os.linesep}"
            f"    Ignore Before: {self.ignore_before}{os.linesep}"
            f"    Ignore After: {self.ignore_after}{os.linesep}"
            f"    Local Repository: {self.local_repository}{os.linesep}"
            f"    Store Raw Always: {self.store_raw_always}{os.linesep}"
        )

    @cached_property
    def external_inst(self) -> ExternalDatabases:
        """Create an instance of :py:class:`ExternalDatabases` and set as a property.

        The parameters of the DatabaseManager instance will be directly passed in the creation of the
        `ExternalDatabases` instance so they will be consistent with each other.

        Returns:
            An ExternalDatabases instance.
        """
        return ExternalDatabases(
            organism=self.organism,
            ensembl_release=self.ensembl_release,
            form=self.form,
            local_repository=self.local_repository,
            genome_assembly=self.genome_assembly,
        )

    @cached_property
    def available_releases(self) -> list[int]:
        return self.available_releases_versions()

    @cached_property
    def available_releases_no_save(self) -> list[int]:
        return self.available_releases_versions(save_after_calculation=False)

    def available_releases_versions(self, **kwargs) -> list[int]:
        """Define available Ensembl releases for the DatabaseManager instance to work on.

        It looks on the MySQL server results to determine which Ensembl releases are available. The method does not
        return directly the Ensembl releases defined by 'ignore_after' and 'ignore_before' parameters, and looks on the
        server as a verification.

        Returns:
            List of integers indicating which Ensembl releases are available.

        Raises:
            ValueError: Unexpected error in regex functions.
        """
        # Get all possible ensembl releases for a given organism
        dbs = self.get_db("availabledatabases", **kwargs)["available_databases"]  # Obtain the databases dataframe
        # print()
        # print("##########################################")
        # print(dbs)
        pattern = re.compile(f"^{self.organism}_core_([0-9]+)_.+$")
        # Search organism name in a specified format. Extract ensembl release number
        releases = list()
        for dbs_i in dbs:
            # print(dbs_i)
            if pattern.match(dbs_i):
                dbs_ps = pattern.search(dbs_i)
                if not dbs_ps:
                    raise ValueError
                releases.append(float(dbs_ps.groups()[0]))

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
    def mysql_database(self) -> str:
        """The program has to choose a 'MySQL database' based on ensembl release and organism name in the MYSQL server.

        Returns:
            The string specifying the 'MySQL database' in the format determined by the regex pattern.

        Raises:
            ValueError: If there is more than one such match.
        """
        dbs = self.get_db("availabledatabases")["available_databases"]  # Obtain the databases dataframe

        pattern = re.compile(f"^{self.organism}_core_{int(self.ensembl_release)}_.+$")
        located = [i for i in dbs if pattern.match(i)]
        # Search database in a specified format. Extract ensembl release number

        if not len(located) == 1:
            raise ValueError(
                "There are more than one 'MySQL database' in the server "
                "satisfying given ensembl release and organism name."
            )
        # Make sure there are only one database in the server satisfying given ensembl release and organism name.

        return located[0]

    def change_form(self, form: str) -> "DatabaseManager":
        """Changes the form of DatabaseManager instance with passing all other variables unchanged.

        Args:
            form: New form of interest. Refer to :py:attr:`DatabaseManager.__init__.form`

        Returns:
            New instance of DatabaseManager with only 'form' is changed.
        """
        return DatabaseManager(
            organism=self.organism,
            ensembl_release=self.ensembl_release,
            form=form,
            local_repository=self.local_repository,
            ignore_before=self.ignore_before,
            ignore_after=self.ignore_after,
            store_raw_always=self.store_raw_always,
            genome_assembly=self.genome_assembly,
        )

    def change_release(self, ensembl_release: int) -> "DatabaseManager":
        """Changes the Ensembl release of DatabaseManager instance with passing all other variables unchanged.

        Args:
            ensembl_release: New Ensembl release of interest.
                Refer to :py:attr:`DatabaseManager.__init__.ensembl_release`

        Returns:
            New instance of DatabaseManager with only 'ensembl_release' is changed.
        """
        return DatabaseManager(
            organism=self.organism,
            ensembl_release=ensembl_release,
            form=self.form,
            local_repository=self.local_repository,
            ignore_before=self.ignore_before,
            ignore_after=self.ignore_after,
            store_raw_always=self.store_raw_always,
            genome_assembly=self.genome_assembly,
        )

    def change_assembly(self, genome_assembly: int, last_possible_ensembl_release: bool = False) -> "DatabaseManager":
        """Changes the genome assembly of DatabaseManager instance with passing all other variables unchanged.

        Args:
            genome_assembly: New genome assembly of interest.
                Refer to :py:attr:`DatabaseManager.__init__.genome_assembly`

        Returns:
            New instance of DatabaseManager with only 'genome_assembly' is changed.
        """
        return DatabaseManager(
            organism=self.organism,
            ensembl_release=self.ensembl_release if not last_possible_ensembl_release else None,
            form=self.form,
            local_repository=self.local_repository,
            ignore_before=self.ignore_before,
            ignore_after=self.ignore_after,
            store_raw_always=self.store_raw_always,
            genome_assembly=genome_assembly,
        )

    def create_available_databases(self) -> pd.Series:
        """Fetches all the databases in the MySQL server of the instance.

        Filters out the query ``SHOW databases`` based on matching to a specific regex string
        ``f"^{self.organism}_core_[0-9]+_.+$"``

        Returns:
            All available 'MySQL databases' in the server for given assembly (which is a parameter in the
            `DatabaseManager` instance).

        Raises:
            ValueError: If the response has unexpected format or length.
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
            raise ValueError("The result is in unexpected format.")
        results_query = [i[0] for i in results_query]  # Get the relevant portion.

        pattern = re.compile(f"^{self.organism}_core_[0-9]+_.+$")
        accepted_databases = sorted(i for i in results_query if pattern.match(i))
        results = pd.DataFrame(accepted_databases, columns=["available_databases"])

        return results

    def get_table(
        self,
        table_key: str,
        usecols: Optional[list] = None,
        create_even_if_exist: bool = False,
        save_after_calculation: bool = True,
        overwrite_even_if_exist: bool = False,
    ) -> pd.DataFrame:
        """Master method for MySQL processes. Downloads, stores, or retrieves the database of raw MySQL results.

        Args:
            table_key: The raw MySQL database (table) of interest. For example `mapping_session`, `xref`, `gene`.
            usecols: The column of interest in the specified table. Short for 'use columns'.
            create_even_if_exist: If ``True``, independent of the status in the disk, the table will be downloaded.
            save_after_calculation: If ``True``, the downloaded table will be stored in the disk, under a `h5` file at
                the local directory specified in init method.
            overwrite_even_if_exist: If ``True``, regardless of whether it is already saved in the disk, the program
                re-saves removing the previous table with the same name.

        Returns:
            Raw table as a pandas DataFrame.

        Raises:
            ValueError: If the 'usecols' is not specified correctly.
        """
        if not (usecols is None or (isinstance(usecols, list) and len(usecols) > 0)):
            raise ValueError("Empty 'usecols' parameter, or 'usecols' is not a list.")

        # Get the file name associated with table_key and columns of interest.
        hierarchy, file_path = self.file_name("mysql", table_key, usecols)

        # If the file name is not accessible for reading, or if the hdf5 file does not contain the table,
        # or explicitly prompt to do so, then download the table.
        if not os.access(file_path, os.R_OK) or create_even_if_exist or (not hs.check_h5_key(file_path, hierarchy)):
            df = self.download_table(table_key, usecols)
        else:  # Otherwise, just read the file that is already in the directory.
            df = hs.read_exported(hierarchy, file_path)

        # If prompt, save the dataframe in requested format.
        if save_after_calculation:
            hs.export_disk(df, hierarchy, file_path, overwrite_even_if_exist, logger=self.log)

        return df

    def tables_in_disk(self) -> list[str]:
        """Retrieves the keys in the h5 file, which is associated with the DatabaseManager instance.

        Returns:
            List of keys (also called hierarchy) in the h5 file.
        """
        _, file_name = self.file_name("common", "place_holder")

        if not os.path.isfile(file_name):
            return list()
        else:
            with hs.HDFStore(file_name, mode="r") as f:
                return list(f.keys())

    def download_table(self, table_key: str, usecols: Optional[list] = None) -> pd.DataFrame:
        """Downloads the raw table from MySQL server and extracts requested columns.

        The method is not generally expected to be used by the user. User is expected to use 'get_table' instead.

        Args:
            table_key: The raw MySQL database (table) of interest. For example `mapping_session`, `xref`, `gene`.
            usecols: The column of interest in the specified table. Short for 'use columns'.

        Returns:
            Raw table as a pandas DataFrame.

        Raises:
            ValueError: If there is no column in the table.
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
                    raise ValueError("There is no column in the table")

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
            if np.any(df.map(lambda x: isinstance(x, bytes))):
                raise ValueError

            info_usecols = " for following columns: " + ", ".join(usecols) + "." if usecols else "."
            self.log.info(
                f"Raw table for `{table_key}` on ensembl release `{self.ensembl_release}` "
                f"was downloaded{info_usecols}"
            )

            return df

    def available_tables_mysql(self):
        """Fetches all the tables in the MySQL database of the instance.

        Raises:
            NotImplementedError: Not implemented.
        """
        raise NotImplementedError

    def get_release_date(self):
        """Get the associated date of release for each Ensembl release.

        Raises:
            NotImplementedError: Note implemented.
        """
        raise NotImplementedError

    @staticmethod
    def _determine_usecols_ids(form: str) -> tuple[list[str], list[str], list[str]]:
        """Helper method to guide which columns are interesting for each form.

        Args:
            form: Form of interest

        Returns:
            Tuple of lists to be used further in the main methods.

        Raises:
            ValueError: If form is not either 'gene', 'transcript', or 'translation'.
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
            raise ValueError(f"Form has to be one of {DB.forms_in_order}. Input form is `{form}`.")

        return stable_id_version, usecols_core, usecols_asso

    def create_ids(self, form: str) -> pd.DataFrame:
        """Retrieves the Ensembl IDs.

        Args:
            form: Form of interest, either 'gene', 'transcript', or 'translation'.

        Returns:
            Dataframe of three columns: `gene_id`, `gene_stable_id`, and `gene_version`. The 'ID' and 'Version' need to
            be concatanated afterwards.
        """
        # Determine which columns are interesting for each form.
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

    def create_relation_current(self) -> pd.DataFrame:
        """Retrieves the relationship between different forms of Ensembl IDs.

        Returns:
            Dataframe of three columns: `gene`, `transcript`, and `translation`. Note that there are some empty cells
            in `translation` column as not all transcripts has translations.
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

    def create_relation_archive(self) -> pd.DataFrame:
        """Retrieves the relationship between different forms of Ensembl IDs for all Ensembl releases.

        It is not recommended as there are some missing rows. Instead use
        :py:meth:`DatabaseManager.create_relation_current` for all Ensembl releases separately,
        and concatanate the resulting data frames.

        Returns:
            Dataframe of three columns: `gene`, `transcript`, and `translation`. Note that there are some empty cells
            in `translation` column as not all transcripts has translations.
        """
        self.log.warning("Not recommended method: Use 'create_relation_current' instead.")
        # Get the table from the server
        df = self.get_table("gene_archive", usecols=None, save_after_calculation=self.store_raw_always)
        # Remove unnecessary columns and return.
        df.drop(columns=["peptide_archive_id", "mapping_session_id"], inplace=True, errors="raise")

        return self._create_relation_helper(df)

    def _create_relation_helper(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper method for creating 'relationship' dataframes for two methods.

        The two methods is :py:meth:`DatabaseManager.create_relation_current` and
        :py:meth:`DatabaseManager.create_relation_archive`. The method is not expected to be used by
        the user.

        Args:
            df: Output of these two methods mentioned.

        Returns:
            Dataframe of three columns `gene`, `transcript`, and `translation`. Note that there are some empty
            cells in `translation` column as not all transcripts has translations.

        Raises:
            ValueError: If the input dataframe is not with the correct number of columns or column names.
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

    def create_id_history(self, narrow: bool) -> pd.DataFrame:
        """Retrieves historical releationship between Ensembl IDs of a given form.

        Args:
            narrow:  Determine whether a some more information should be added between Ensembl gene IDs. For example,
                which genome assembly is used, or when was the connection is established. For usual uses, no need to
                set it ``True``.

        Returns:
            Dataframe of following columns; `old_stable_id`, `old_version`, `new_stable_id`, `new_version`, `score`,
            `old_release`, `new_release`. Note that there are some empty cells in new and old Ensembl IDs, since
            there are 'retirement' events or 'birth' events of IDs.

        Raises:
            ValueError: If the delimiter :py:attr:`DB.id_ver_delimiter` is in Ensembl IDs.
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

    def create_id_history_fixed(self, narrow: bool, inspect: bool) -> pd.DataFrame:
        """Depracated method alternative for 'create_id_history', which also corrects the resulting dataframe.

        This method created a fixed version of idhistory. It fixes the problems seen in some exceptional cases like
        Homo sapiens ``ENSG00000232423`` at release 105. Raw version gives the order of versions as follows: 1-2, 2-3,
        1-2, 2-3 and so on. However, after second connection, version 1 should be already lost, the last active version
        should be 3. For this reason, 1-2 connection should be corrected as 3-2. This method does this.

        Args:
            narrow: The same parameter in :py:attr:`DatabaseManager.create_id_history.narrow`.
            inspect: If inspect is ``True``, then add the newly created columns as new ones into the existing one.

        Returns:
            Dataframe of following columns: `old_stable_id`, `old_version`, `new_stable_id`, `new_version`, `score`,
            `old_release`, `new_release`.
        """
        # Get the raw version of idhistory first, and sort.
        df = self.get_db("idhistory" if not narrow else "idhistory_narrow")
        df.sort_values(by=["new_release"], inplace=True)

        # Initialize some temp variables
        extinct_version: dict[str, set] = dict()
        last_active_version: dict[str, Any] = dict()
        corrected_entries = list()

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

    def create_external_db(self, filter_mode: str) -> pd.DataFrame:
        """Retrieves External identifier relationship. Returns the connections or databases only.

        The method actually based on a rationele to the following MySQL query, not exactly but quite close to the
        following query.

        .. code-block:: sql

            SELECT g.stable_id, t.stable_id, tr.stable_id, x.dbprimary_acc, edb.db_name, es.synonym, ix.*
            FROM gene g
            JOIN transcript t USING (gene_id)
            JOIN translation tr USING (transcript_id)
            JOIN object_xref ox ON (g.gene_id = ox.ensembl_id AND ox.ensembl_object_type = "Gene")
            JOIN xref x ON (ox.xref_id = x.xref_id)
            LEFT JOIN external_db edb ON (x.external_db_id = edb.external_db_id)
            LEFT JOIN identity_xref ix ON (ox.object_xref_id = ix.object_xref_id)
            LEFT JOIN external_synonym es ON (x.xref_id = es.xref_id)
            LIMIT 10;

        Instead of ``FROM gene g``, it is possible to be a bit more specific by replacing the following:

        .. code-block:: sql

            FROM coord_system cs
            JOIN seq_region sr USING (coord_system_id)
            JOIN gene g USING (seq_region_id)

        The MySQL server can be following for the experimentation purpose.

        .. code-block:: bash

            mysql --user=anonymous --host=ensembldb.ensembl.org -D homo_sapiens_core_105_38 -A
            # As written in the following link.
            # https://m.ensembl.org/info/docs/api/core/core_schema.html

        Args:
            filter_mode: Determine what to return after retrieving the data.
                One of the followings: `relevant`, `all`, `database`, `relevant-database`.

                - `relevant` and `all`:
                  Returns the relationship information. 'all' returns all relationships while
                  'relevant' returns only those indicated by ``ExternalDatabases`` class.
                  The result is a dataframe with 'release', 'graph_id', 'id_db', 'name_db', 'ensembl_identity', and
                  'xref_identity'.
                - `database` and `relevant-database`:
                  Returns the a dataframe indicating databases. The resulting dataframe
                  contain two columns: 'name_db' and 'count'.

        Returns:
            Dataframe of specifified type via ``filter_mode``.

        Raises:
            ValueError: If incorrect entry for ``filter_mode`` parameter, or if there is an inconsistency between
                external `yaml` file and current state of ``DatabaseManager``.
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

    def create_database_content(self, just_download: bool = False) -> pd.DataFrame:
        """Retrives all External database information from the server to feed the ``ExternalDatabase`` class.

        It is quite costly operation, potentially takes time to be completed. It helps ``ExternalDatabase`` class to
        create the ``yaml`` file mentioned. It downloads for all assemblies, all Ensembl releases and all forms
        available.

        Returns:
            The output of ``create_external_db`` when `filter_mode` `relevant-database`. Adds `assembly`, `release`,
            and `form` columns to the resulting dataframe.
        """
        df = pd.DataFrame()
        for k in DB.assembly_mysqlport_priority.keys():  # For all assemblies possible.
            dm_assembly = self.change_assembly(k, last_possible_ensembl_release=True)
            for j in dm_assembly.available_form_of_interests:  # For all assemblies possible.
                for i in dm_assembly.available_releases:
                    self.log.info(
                        f"Database content is being created for "
                        f"`{self.organism}`, assembly `{k}`, form `{j}`, ensembl release `{i}`"
                    )
                    df_temp = dm_assembly.change_release(i).change_form(j).get_db("external_database")
                    df_temp["assembly"] = k
                    df_temp["release"] = i
                    df_temp["form"] = j
                    if not just_download:
                        df = pd.concat([df, df_temp], axis=0)
        if not just_download:
            df["organism"] = self.organism
            df.reset_index(inplace=True, drop=True)
            return df
        else:
            return df

    def create_release_id(self) -> pd.DataFrame:
        """Retrieves the Ensembl IDs and applies `version_fix` method to refine it.

        Returns:
            A dataframe with ``f'{form}_stable_id'`` and ``f'{form}_version'``.

        Raises:
            ValueError: If the delimiter :py:attr:`DB.id_ver_delimiter` is in Ensembl IDs, or if the stable IDs are
                not unique.
        """
        dbm_the_ids = self.get_db(f"idsraw_{self.form}")
        dbm_the_ids = self.version_fix(dbm_the_ids, f"{self.form}_version")
        dbm_the_ids = dbm_the_ids[self._identifiers]

        if not np.all(dbm_the_ids[f"{self.form}_stable_id"].str.find(DB.id_ver_delimiter) == -1):
            raise ValueError("The delimiter is in Ensembl IDs.")

        dbm_the_ids.drop_duplicates(keep="first", inplace=True)
        if not dbm_the_ids[f"{self.form}_stable_id"].is_unique:
            raise ValueError("The stable IDs are not unique")

        dbm_the_ids.reset_index(inplace=True, drop=True)
        dbm_the_ids.drop_duplicates(inplace=True)
        return dbm_the_ids

    def check_if_change_assembly_works(self, db_manager, target_assembly):
        try:
            db_manager.change_assembly(target_assembly)
            return True
        except ValueError:
            return False

    def create_external_all(self, return_mode: str) -> pd.DataFrame:
        """Download external databases for all assemblies.

        The method considers the 'priority' of the assemblies indicated in
        :py:attr:`DB.assembly_mysqlport_priority`. The method is not found in 'get_db' so the result is not saved.

        Args:
            return_mode: Either one of three choices `all`, `unique`, `duplicated`. There is currently no use case for
                `unique`, `duplicated` by the program.

        Returns:
            Returns the relationship information indicated by ``ExternalDatabases`` class.
            The result is a dataframe with 'release', 'graph_id', 'id_db', 'name_db', 'ensembl_identity', and
            'xref_identity', and also 'assembly'.

        Raises:
            ValueError: If `return_mode` is not among the possible ones.
        """
        ass = self.external_inst.give_list_for_case(give_type="assembly")
        df = pd.DataFrame()
        assembly_priority = [DB.assembly_mysqlport_priority[i]["Priority"] for i in ass]

        for i in [x for _, x in sorted(zip(assembly_priority, ass))]:  # sort according to priority
            if self.check_if_change_assembly_works(db_manager=self, target_assembly=i):
                dm = self.change_assembly(i)
                df_temp = dm.get_db("external_relevant")
                df_temp["assembly"] = i
                df = pd.concat([df, df_temp])
        df.reset_index(drop=True, inplace=True)

        compare_columns = [
            i for i in df.columns if i != "assembly" and not i.endswith("_identity")
        ]  # 'ensembl_identity', 'xref_identity'
        compare_columns_2 = compare_columns + ["assembly"]

        if return_mode == "all":
            # Drop duplicate rows that have all the columns the same with another row in the dataframe.
            # This also looks for 'assembly' columns so it is possible to say assemlies are evaluated separately.
            df.drop_duplicates(keep="first", inplace=True, ignore_index=True, subset=compare_columns_2)
            return df

        elif return_mode == "unique":
            # Unlike above, this does not also look for 'assembly' columns, so an entry found in the higher priority
            # assembly will be kept but the others will be removed.
            df.drop_duplicates(keep="first", inplace=True, ignore_index=True, subset=compare_columns)
            # Note that: after transition to new assembly. ensembl does not assign new versions etc to the older
            # keep the most priority one.
            return df

        elif return_mode == "duplicated":
            df = df[df.duplicated(subset=compare_columns, keep=False)]
            dfg = df.groupby(by=compare_columns)
            return dfg

        else:
            raise ValueError(f"Undefined parameter for 'return_mode': {return_mode}.")

    def create_version_info(self) -> pd.DataFrame:
        """Check whether all Ensembl release has Ensembl IDs with versions or without versions.

        Some organisms such as S. cerevisiae has Ensembl identifiers that has no 'Version', but only 'ID'. The method
        looks whether this is the case for all the Ensembl releases for a given organism.

        Returns:
            A dataframe with two columns as `ensembl_release`, `version_info`.

        Raises:
            NotImplementedError: If some Ensembl identifiers without versions, some are not."
        """
        ver = list()

        for i in sorted(self.available_releases, reverse=True):
            db_manager_temp = self.change_release(i)
            df_ids = db_manager_temp.get_db(f"idsraw_{self.form}", save_after_calculation=self.store_raw_always)
            _vv = pd.isna(df_ids[f"{self.form}_version"])

            if np.all(_vv):
                with_version = True

            elif np.any(_vv):
                raise NotImplementedError("Some identifiers with versions that are NaN, some are not.")

            else:
                with_version = False

            ver.append([i, with_version])

        df = pd.DataFrame(ver, columns=["ensembl_release", "version_info"])
        return df

    def get_db(
        self,
        df_indicator: str,
        create_even_if_exist: bool = False,
        save_after_calculation: bool = True,
        overwrite_even_if_exist: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """For saving, exporting, and naming convention. Main method to retrieve the data sources.

        Args:
            df_indicator: A string indicating which dataframe the user is requested to access.
                It should contain only one or zero '_' character.
            create_even_if_exist: If ``True``, independent of the status in the disk, the table will be downloaded.
            save_after_calculation: If ``True``, the downloaded table will be stored in the disk, under a `h5` file at
                the local directory specified in init method.
            overwrite_even_if_exist: If ``True``, regardless of whether it is already saved in the disk, the program
                re-saves removing the previous table with the same name.

        Returns:
            The data of interest depending on `df_indicator` parameter.

        Raises:
            ValueError: If `df_indicator` parameter is not in specified format.
        """

        def check_exist_as_diff_release(_df_type, _df_indicator):
            # Get the file name associated with table_key and columns of interest.
            _, _file_path = self.file_name(_df_type, _df_indicator)

            # The below pattern is based on file_name function with some modifications.
            # Organism name and form is excluded as it does not change the resulting file.
            _pattern = re.compile(f"ens([0-9]+)_{_df_type}_{_df_indicator}")

            if not os.access(_file_path, os.R_OK):
                return None, list()

            with hs.HDFStore(_file_path, mode="r") as f:
                _keys = f.keys()
            _downloaded_rels = list({int(_pattern.search(i).groups()[0]) for i in _keys if _pattern.search(i)})

            for _dr in _downloaded_rels:
                if _dr >= self.ensembl_release:
                    return _dr, _downloaded_rels
            return None, _downloaded_rels

        def remove_redundant_exist(_df_type, _df_indicator, _keep_rel, _all_rel_lst):
            for _arl in _all_rel_lst:
                if _arl != _keep_rel:
                    _hi, _fi = self.file_name(_df_type, _df_indicator, ensembl_release=_arl)
                    with hs.HDFStore(_fi, mode="a") as f:
                        f.remove(_hi)
                        self.log.info(
                            f"Following file is being removed: `{os.path.basename(_fi)}` with key `{_hi}`. "
                            f"This could cause hdf5 file to not reclaim the emptied disk space."
                        )

        # Split the df_indicator with "_", to get the extra parameters.
        # Main point of naming and df_indicator is to include the paramters in the file_names
        # for exporting and importing without writing explicit methods of read/write for each case.
        split_ind = df_indicator.split("_")
        main_ind = split_ind[0]
        param1_ind = split_ind[1] if len(split_ind) > 1 else None
        if len(split_ind) > 2:
            raise ValueError

        # For 'availabledatabases' accept different files explained in the below functions.
        if main_ind == "availabledatabases" and param1_ind is None:
            xr1, xr1_a = check_exist_as_diff_release("common", df_indicator)
            remove_redundant_exist("common", df_indicator, xr1, xr1_a)
            hierarchy, file_path = self.file_name("common", df_indicator, ensembl_release=xr1)

        elif main_ind == "versioninfo" and param1_ind is None:
            xr1, xr1_a = check_exist_as_diff_release("processed", df_indicator)
            remove_redundant_exist("processed", df_indicator, xr1, xr1_a)
            hierarchy, file_path = self.file_name("processed", df_indicator, ensembl_release=xr1)

        elif param1_ind is None and main_ind in ("relationarchive", "relationcurrent"):
            hierarchy, file_path = self.file_name("common", df_indicator)

        else:
            # Get the file name associated with table_key and columns of interest.
            hierarchy, file_path = self.file_name("processed", df_indicator)

        # If the file name is not accessible for reading, or if the hdf5 file does not contain the table,
        # or explicitly prompt to do so, then download the table.
        if not os.access(file_path, os.R_OK) or create_even_if_exist or (not hs.check_h5_key(file_path, hierarchy)):
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
                df = self.create_release_id()

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
            df = hs.read_exported(hierarchy, file_path)

        # If prompt, save the dataframe in requested format.
        if save_after_calculation:
            hs.export_disk(df, hierarchy, file_path, overwrite_even_if_exist, logger=self.log)

        return df

    def file_name(self, df_type: str, *args, ensembl_release: Optional[int] = None, **kwargs) -> tuple[str, str]:
        """Determine file name for reading/writing into h5 file based on dataframe type.

        The method is not expected to be used by the user.

        Args:
            df_type: Either `processed`, `mysql`, `common`.
            args: Arguments to pass into the functions, which are specific to each ``df_type``.
            ensembl_release: Associated Ensembl release of the file of interest.
            kwargs: Keyword arguments to pass into the functions, which are specific to each ``df_type``.

        Returns:
            The file name for h5 file and the key to be used inside.

        Raises:
            ValueError: If the parameter `df_type` is not in specified format.
        """
        ensembl_release = self.ensembl_release if not ensembl_release else ensembl_release

        def file_name_processed(df_indicator: str):
            return f"ens{ensembl_release}_{df_type}_{df_indicator}_{self.form}"

        def file_name_mysql(table_key: str, usecols: Optional[list] = None):
            col_suffix = f"{self._column_sep}{self._column_sep.join(sorted(usecols))}" if usecols is not None else ""
            return f"ens{ensembl_release}_{df_type}_{table_key}{col_suffix}"

        def file_name_noform(df_indicator: str):
            return f"ens{ensembl_release}_{df_type}_{df_indicator}"

        if df_type not in ["processed", "mysql", "common"]:
            raise ValueError("The parameter is not in specified format: df_type")
        if df_type == "processed":
            hierarchy = file_name_processed(*args, **kwargs)
        elif df_type == "common":
            hierarchy = file_name_noform(*args, **kwargs)
        else:
            hierarchy = file_name_mysql(*args, **kwargs)

        return hierarchy, os.path.join(self.local_repository, f"{self.organism}_assembly-{self.genome_assembly}.h5")

    def id_ver_from_df(self, dbm_the_ids: pd.DataFrame) -> list[str]:
        """Creates node names given the dataframe of IDs, which has different columns for 'ID' and 'Version'.

        Args:
            dbm_the_ids: Dataframe of IDs, typically the output of ``db_manager.get_db("ids")``.

        Returns:
            Node name as "ID.Version" if there is 'Version', else only "ID".

        Raises:
            ValueError: If column names of the `dbm_the_ids` is not as expected.
        """
        if np.all(dbm_the_ids.columns != self._identifiers):
            raise ValueError(
                f"Column names of the 'dbm_the_ids' is not as expected. "
                f"{dbm_the_ids.columns} vs {self._identifiers}"
            )
        gri_generator = (DatabaseManager.node_dict_maker(i, j) for i, j in dbm_the_ids.values)
        return list(map(DatabaseManager.node_name_maker, gri_generator))

    @staticmethod
    def node_name_maker(node_dict: dict[str, Any]) -> str:
        """This function creates ID-Version.

        If the Version information is not there, it only uses ID, which is necessary for some organisms which
        does not have versioned IDs.

        Args:
            node_dict: The output of :py:meth:`DatabaseManager.node_dict_maker`.

        Returns:
            Node name as "ID.Version" if there is 'Version', else only "ID".
        """
        if node_dict["Version"] and not pd.isna(node_dict["Version"]):
            return node_dict["ID"] + DB.id_ver_delimiter + str(node_dict["Version"])
        else:
            return node_dict["ID"]

    @staticmethod
    def node_dict_maker(id_entry: str, version_entry: Any) -> dict[str, Any]:
        """Create a dict for ID and Version.

        Args:
            id_entry: For example, the first part of (`ENSG00000000001`) in `ENSG00000000001.1` before delimiter.
            version_entry: For example, the second part of (`1`) in `ENSG00000000001.1` before delimiter.

        Returns:
            Dictionary in the format of ``{"ID": id_entry, "Version": version_entry}``

        Raises:
            ValueError: If 'Version' is not floating number (cannot be converted into integer).
        """
        if version_entry and not pd.isna(version_entry) and version_entry not in DB.alternative_versions:
            if int(version_entry) != float(version_entry):
                raise ValueError(f"Version is floating: {(id_entry, version_entry)}")
            else:
                version_entry = int(version_entry)
        return {"ID": id_entry, "Version": version_entry}

    def version_uniformize(self, df: pd.DataFrame, version_str: str) -> pd.DataFrame:
        """Final operation for :py:meth:`DatabaseManager.create_ids`.

        The method is not expected to be used by the user.

        Make the 'Version' column, integer if there is 'Version' information, else, make put ``np.nan`` instead. The
        operation is in line with :py:meth:`DatabaseManager.create_version_info` method.
        Note that ``np.nan`` values will be used by `node_name_maker` and `node_name_maker` methods.

        Args:
            df: Input dataframe, see the `create_ids` method for more detail.
            version_str: Which column contain the 'Version' information

        Returns:
            Corrected version of the dataframe in terms of 'Version' information.

        Raises:
            NotImplementedError: If some Ensembl identifiers without versions, some are not."
        """
        contains_na = pd.isna(df[version_str])
        if np.all(contains_na):
            # If there is no version information associated with stable_ids. For some organisms like S. cerevisiae
            df[version_str] = np.nan
            return df

        elif np.any(contains_na):
            raise NotImplementedError("Some identifiers with versions that are NaN, some are not.")

        else:
            df[version_str] = df[version_str].astype(int)
            return df

    def version_fix_incomplete(self, df_fx: pd.DataFrame, id_col_fx: str, ver_col_fx: str) -> pd.DataFrame:
        """The same logic with ``version_fix`` method, but do different process if associated 'ID' is ``np.nan``.

        Args:
            df_fx: Input dataframe to fix the version information.
            id_col_fx: Which column contain the 'ID' information.
            ver_col_fx: Which column contain the 'Version' information.

        Returns:
            Modified dataframe in terms of version information.
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

    def version_fix(self, df: pd.DataFrame, version_str: str, version_info: Optional[str] = None) -> pd.DataFrame:
        """Depending the version information of the organism, uniformize the identifiers.

        The method is not expected to be used by the user.

        Args:
            df: Input dataframe to fix the version information.
            version_str: Which column contain the 'Version' information.
            version_info: The program parameter obtained somehow by ``check_version_info`` method.

        Returns:
            Modified dataframe in terms of version information.

        Raises:
            ValueError: If `version_info` is not one of the output of `check_version_info` method.
        """
        # If version_info is not entered, just re-calculate.
        version_info = version_info if version_info else self.check_version_info()

        if version_info == "add_version":
            # Set the constant value of DB.first_version as the Version.
            df[version_str] = DB.first_version

        elif version_info == "without_version":
            df[version_str] = np.nan

        elif version_info == "with_version":
            df[version_str] = df[version_str].astype(int)

        else:
            raise ValueError("Undefined choice for 'version_info'.")

        return df

    def check_version_info(self) -> str:
        """Look at across all Ensembl releases and decide the 'version_info' variable to be used in the program.

        Returns:
            Either `without_version` when all Ensembl releases has identifiers without 'Versions',
            `with_version` when all Ensembl releases has identifiers with 'Versions',
            or `add_version` when some Ensembl releases has identifiers with 'Versions'.

        Raises:
            ValueError: If the dataframe obtained by ``self.get_db("versioninfo")`` has a problematic column.
        """
        vi_df = self.get_db("versioninfo")
        narrowed = vi_df["version_info"].unique()

        if narrowed.dtype != np.dtype("bool"):
            raise ValueError("Data type of 'version_info' column must be boolean.")

        if len(narrowed) == 1 and narrowed[0]:
            return "without_version"

        elif len(narrowed) == 1:
            return "with_version"

        else:
            return "add_version"
