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
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pymysql

import idtrack._utils_hdf5 as hs
from idtrack._db import DB
from idtrack._external_databases import ExternalDatabases


class DatabaseManager:
    """Manage retrieval, preprocessing, and storage of Ensembl Core and related external datasets.

    The *DatabaseManager* centralizes all low-level operations required for ID-track analyses, including
    discovering which Ensembl releases are available for a given organism/assembly, downloading the
    corresponding MySQL tables, normalizing column names, persisting raw and processed files under a
    local cache directory, and orchestrating auxiliary look-ups to third-party resources via
    :py:class:`ExternalDatabases`.  By funnelling every data-access path through a single object the
    wider package gains:

    * **Stable, reproducible builds** - every graph, lookup table, or ID-history file is anchored to
      the exact *Ensembl release*, *genome assembly*, and *form* (gene, transcript, translation, …)
      with which the manager was configured.
    * **Transparent caching** - expensive downloads happen once; subsequent requests are served from
      disk, making large iterative analyses feasible on modest hardware.
    * **Unified version logic** - helper methods such as
      :py:meth:`~DatabaseManager.version_uniformize` and
      :py:meth:`~DatabaseManager.check_version_info` guarantee that cross-release identifier changes
      are captured and resolved consistently across the codebase.

    Key public methods/attributes
    -----------------------------
    * :py:meth:`available_releases` — list releases that can be queried *and* saved locally.
    * :py:meth:`change_release` — switch the manager to another Ensembl release in-place.
    * :py:meth:`download_table` — fetch a single MySQL table and write it to ``local_repository``.
    * :py:meth:`create_external_all` — pull every supported external resource (UniProt, RefSeq, …).
    * :py:attr:`organism`, :py:attr:`form`, :py:attr:`ensembl_release`,
      :py:attr:`genome_assembly` — core configuration knobs, surfaced for quick inspection.

    The class is **stateful**: change-mutating helpers update internal cached properties so that the
    instance always reflects its current configuration.  Use the built-in :py:meth:`__str__` for a
    concise, human-readable dump of that state.
    """

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
        """Initialize a :py:class:`DatabaseManager` for a specific organism, release, and assembly.

        Args:
            organism (str): Canonical species name in Ensembl schema (e.g. ``"homo_sapiens"`` or
                ``"mus_musculus"``). Anything else raises :py:class:`NotImplementedError`.
            form (str): Biological entity level of interest—one of ``"gene"``, ``"transcript"``,
                ``"translation"``, …—governing which stable-ID columns will be expected downstream.
            local_repository (str): Absolute or relative path to a writable directory that will hold
                all downloaded MySQL dumps, intermediate parquet/Feather files, and ready-to-use
                artefacts. The directory must already exist and be both readable and writable.
            ensembl_release (Optional[int]): Target Ensembl release number. If ``None`` the most
                recent release available for *genome_assembly* is selected automatically.
            ignore_before (Optional[int]): Earliest release to include when building cross-release ID
                histories. Defaults to the minimum release supported by the selected assembly.
            ignore_after (Optional[int | float]): Latest release to include when building histories.
                ``np.inf`` (the default) disables the upper bound and includes all newer releases.
            store_raw_always (bool): When ``True`` raw MySQL tables are *always* copied to
                ``local_repository`` before conversion; when ``False`` they are kept only in memory.
            genome_assembly (Optional[int]): NCBI assembly version (e.g. ``38`` for GRCh38). If
                omitted, the assembly with the highest priority for *organism* is used.

        Raises:
            ValueError: If *form* is not in the supported list or *local_repository* fails basic
                path/read/write checks.
            NotImplementedError: If *organism* is not yet supported by the package.
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
        # _ = self.external_inst.load_modified_yaml()

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
        """Return a multi-line snapshot of the manager's current configuration.

        The string lists organism, form, Ensembl release, genome assembly, ignore range,
        repository path, and caching mode, making it convenient to embed in logs or console output.

        Returns:
            str: Readable status summary, one attribute per line, ending with a newline.
        """
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
        """Instantiate and cache an :py:class:`ExternalDatabases` helper for this manager.

        The instance mirrors the configuration of the surrounding :py:class:`DatabaseManager`—organism,
        Ensembl release, identifier form, local repository path, and genome assembly—so that all
        interactions with external data sources remain consistent throughout the session.  Because the
        property is backed by :py:data:`functools.cached_property`, the helper is created exactly once
        and reused on subsequent accesses, eliminating redundant network or file-system look-ups.

        Returns:
            ExternalDatabases: A lazily created, configuration-matched helper object.
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
        """Return Ensembl releases that are both reachable and within the ignore window.

        The set is discovered via :py:meth:`available_releases_versions`, filtered against
        ``ignore_before`` / ``ignore_after``, sorted in ascending order, and cached for the lifetime of
        this :py:class:`DatabaseManager` instance.  The resulting list represents releases that can
        safely be queried **and** cached locally, guaranteeing reproducible downstream analyses.

        Returns:
            list[int]: Sorted release numbers satisfying reachability and ignore-window constraints.
        """
        return self.available_releases_versions()

    @cached_property
    def available_releases_no_save(self) -> list[int]:
        """Return reachable Ensembl releases without persisting the discovery to disk.

        Functionally identical to :py:meth:`available_releases`, except that the discovered list is **not**
        written to the on-disk YAML cache. This helper is useful when users want a quick, read-only view
        of server availability—e.g., inside CI pipelines—without contaminating the persistent cache.
        The value is still memoized in memory for the current :py:class:`DatabaseManager` instance.

        Returns:
            list[int]: Sorted release numbers reachable on the remote MySQL server and compliant with the
            ignore window.
        """
        return self.available_releases_versions(save_after_calculation=False)

    def available_releases_versions(self, **kwargs) -> list[int]:
        """Discover valid Ensembl releases for the configured organism and assembly.

        A ``SHOW DATABASES`` query is issued against the configured MySQL mirror.  Results are matched
        against the pattern ``^{organism}_core_<release>_.*$``; the captured ``<release>`` component is
        converted to ``int`` (floating-point labels are rejected), range-checked against the manager's
        ``ignore_before`` / ``ignore_after`` bounds, and finally sorted. Optional keyword arguments are
        forwarded verbatim to :py:meth:`DatabaseManager.get_db`, allowing callers to tweak connection or
        caching behaviour.

        Args:
            kwargs: Arbitrary keyword arguments passed straight through to
                :py:meth:`DatabaseManager.get_db` (e.g., ``mysql_conn``, ``force_refresh``).

        Returns:
            list[int]: Sorted list of release numbers that exist on the mirror and comply with the ignore window.

        Raises:
            ValueError: If a database name does not match the expected regex, if floating-point release
                labels are encountered, or if other inconsistencies arise while parsing server results.
        """
        # Get all possible ensembl releases for a given organism
        dbs = self.get_db("availabledatabases", **kwargs)["available_databases"]  # Obtain the databases dataframe
        pattern = re.compile(f"^{self.organism}_core_([0-9]+)_.+$")
        # Search organism name in a specified format. Extract ensembl release number
        releases = list()
        for dbs_i in dbs:
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
        """Return the canonical Ensembl Core schema name for the current organism, release, and assembly.

        The manager must resolve exactly one MySQL schema on the remote Ensembl server whose name encodes the
        *organism* (e.g. ``homo_sapiens``), *release* (e.g. ``111``), and *genome assembly* (e.g. ``38``).  This
        helper centralises that lookup, guaranteeing that downstream calls—such as :py:meth:`download_table`—always
        talk to the correct database.  The search relies on :py:meth:`get_db` to fetch the server's catalogue and
        then filters it by a strict regular expression; any ambiguity is treated as fatal because it would break
        reproducibility.

        Returns:
            str: A single schema name like ``"homo_sapiens_core_111_38"`` that uniquely matches the manager's
                configuration.

        Raises:
            ValueError: If zero **or** more than one schema satisfies the search criteria, signalling server
                misconfiguration or an invalid combination of *organism* and *ensembl_release*.
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
        """Clone the manager while switching the biological form of interest.

        A “form” denotes the identifier namespace to track—``gene``, ``transcript``, ``translation``, etc.  This
        method preserves every other configuration knob (organism, release, assembly, cache directory, ignore
        windows, …) and returns a brand-new instance so that the original object remains unaffected.

        Args:
            form (str): Target form/namespace recognised by :py:meth:`~DatabaseManager.__init__`.  Typical values
                are ``"gene"``, ``"transcript"``, or ``"translation"``.

        Returns:
            DatabaseManager: An independent manager identical to *self* except for :py:attr:`form`.
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
        """Produce a new manager that targets a different Ensembl release.

        The returned instance inherits organism, form, assembly, and all caching parameters, but points every
        subsequent query (MySQL, FTP, or REST) to *ensembl_release*.  This is the recommended way to traverse
        releases in scripted analyses without mutating objects in-place.

        Args:
            ensembl_release (int): Desired Ensembl release number (e.g. ``111``).  Must be available for the
                current genome assembly or a :py:data:`NotImplementedError` may be raised further down the call
                stack when data retrieval is attempted.

        Returns:
            DatabaseManager: Fresh manager initialised for *ensembl_release*.
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
        """Clone the manager while targeting a new genome assembly (e.g. GRCh38 → GRCh37).

        Genome assemblies are encoded as integers in Ensembl's schema naming (``38`` for *GRCh38*,
        ``37`` for *GRCh37*, ``102`` for *GRCm39*, …).  When *last_possible_ensembl_release* is ``True`` the
        method automatically picks the most recent Ensembl release that **still** provides MySQL dumps for the
        requested assembly, ensuring compatibility.  All other settings are copied verbatim.

        Args:
            genome_assembly (int): Key from :py:data:`DB.assembly_mysqlport_priority` mapping—see Ensembl
                documentation for valid values.
            last_possible_ensembl_release (bool): When ``True`` override *ensembl_release* with the
                newest version available for *genome_assembly*.  Defaults to ``False``.

        Returns:
            DatabaseManager: New manager tied to the requested assembly (and possibly a recalculated release).
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
        """Discover MySQL databases for the configured organism/assembly.

        The manager issues a ``SHOW DATABASES`` query against the Ensembl public MySQL mirror and filters
        names that match ``^{organism}_core_[0-9]+_.*$``.  The resulting list is returned as a single-column
        dataframe so that callers can seamlessly chain further pandas operations or persist the result.

        Returns:
            pandas.DataFrame: One column named ``"available_databases"`` listing all databases that match
                the organism, irrespective of Ensembl release or genome assembly.

        Raises:
            ValueError: If the server response is not a sequence of single-field tuples **or** if any tuple
                element is not a string.
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
        """Download, cache, or read a raw MySQL table for the current release.

        A high-level wrapper that coordinates three steps:

        1. **Path resolution** - determines the HDF5 file and internal key under the local repository that
           belong to *table_key* (and *usecols*, if provided).
        2. **Fetch or reuse** - if the target key is absent, unreadable, or forcibly refreshed, delegates to
           :py:meth:`download_table` to query the MySQL server; otherwise loads the dataframe from disk.
        3. **Persistence** - optionally stores the freshly downloaded dataframe back to disk, shrinking the
           number of future network calls.

        Args:
            table_key (str): Name of the MySQL table (e.g. ``"gene"``, ``"xref"``, ``"mapping_session"``).
            usecols (list[str] | None): Column subset to retrieve. ``None`` (default) selects *all* columns.
            create_even_if_exist (bool): Ignore any on-disk cache and re-download the table unconditionally.
            save_after_calculation (bool): Persist the dataframe to the computed HDF5 path when ``True``.
            overwrite_even_if_exist (bool): Replace an existing HDF5 key even when it is already present.

        Returns:
            pandas.DataFrame: The requested raw table with column order mirroring *usecols* when supplied,
                otherwise the server's natural order.

        Raises:
            ValueError: If *usecols* is an empty list, not a list, or otherwise fails basic validation.
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
        """List all dataframes cached for this manager on local disk.

        The helper inspects the HDF5 file located at the path generated by
        :py:meth:`file_name` (``df_type="common"``) and returns every key it contains.
        When the file does not exist yet, an empty list is returned instead of raising.

        Returns:
            list[str]: Sorted HDF5 keys corresponding to dataframes already materialised for this manager.
        """
        _, file_name = self.file_name("common", "place_holder")

        if not os.path.isfile(file_name):
            return list()
        else:
            with hs.HDFStore(file_name, mode="r") as f:
                return list(f.keys())

    def download_table(self, table_key: str, usecols: Optional[list] = None) -> pd.DataFrame:
        """Download a raw Ensembl MySQL table and return it as a DataFrame.

        The method forms the low-level backbone of all table acquisition in *IDTrackDocs*.  It opens a
        direct connection to the Ensembl Core (or comparable) MySQL schema configured on the current
        :py:class:`DatabaseManager` instance, issues a `SELECT` statement against *table_key*, converts
        the results into a :py:class:`pandas.DataFrame`, and performs a minimal sanitisation pass
        (bytes-to-string decoding, column subset validation, logging).  Public code is expected to call
        :py:meth:`DatabaseManager.get_table`, which wraps this helper with caching and post-processing,
        but keeping this routine separate allows fine-grained testing, mocking, and reuse in advanced
        workflows.

        Args:
            table_key (str): Name of the raw table as it appears in the remote Ensembl database
                (e.g. ``'gene'``, ``'mapping_session'``, ``'xref'``).  Must exist in the schema
                returned by :py:meth:`DatabaseManager.mysql_database`.
            usecols (Optional[list[str]]): Sequence of column names to project; *None* retrieves the
                entire table.  Column order is preserved.  An empty list is treated the same as *None*.

        Returns:
            pandas.DataFrame: A frame containing the requested columns in the exact order supplied via
                *usecols* (or all columns if *usecols* is *None*).  Index is monotonic and zero-based.

        Raises:
            ValueError: If any element of *usecols* is missing from *table_key*, or if the query returns
                binary payloads that cannot be coerced into native Python types.
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
        """Enumerate tables present in the selected Ensembl MySQL schema.

        Intended to complement :py:meth:`available_databases_mysql`: while that
        method lists *databases* (one per organism/release/assembly), this one will
        drill into the active database and return the table names themselves, such
        as ``"gene"``, ``"transcript"``, ``"xref"``, and so on.

        Raises:
            NotImplementedError: Always - the table enumeration logic has not yet been written.
        """
        raise NotImplementedError

    def get_release_date(self):
        """Return a mapping of Ensembl release numbers to their publication dates.

        The future implementation will query the ``meta`` table of each reachable
        release—or fall back to the Ensembl REST API—to build a dictionary such as
        ``{105: date(2022, 11, 1), 106: date(2023, 2, 7), …}``.  Down-stream
        routines can then translate between absolute dates and release numbers,
        enabling chronology-aware analyses and reporting.

        Raises:
            NotImplementedError: Always - date discovery is not yet implemented.
        """
        raise NotImplementedError

    @staticmethod
    def _determine_usecols_ids(form: str) -> tuple[list[str], list[str], list[str]]:
        """Determine column subsets needed to fetch identifier tables for a given Ensembl molecular form.

        The helper translates a user-facing *form* string (``"gene"``, ``"transcript"``, or
        ``"translation"``) into three ordered lists that drive low-level SQL selects throughout
        *ID-track*.  Splitting the information this way lets public routines such as
        :py:meth:`DatabaseManager.create_ids` assemble the minimal column set required for each
        organism/release while still keeping associated keys available for later joins.

        Args:
            form (str): Molecular form whose identifier columns are requested. Must be one of
                :py:data:`idtrack._db.DB.forms_in_order` (``"gene"``, ``"transcript"``, or
                ``"translation"``).

        Returns:
            tuple[list[str], list[str], list[str]]:
                * **stable_id_version** - always ``["stable_id", "version"]``; the canonical ID and its
                  version counter.
                * **usecols_core** - primary-key column for *form* plus ``stable_id_version``.
                * **usecols_asso** - foreign-key columns linking *form* to upstream forms, enabling
                  later joins (e.g., ``["transcript_id", "gene_id"]`` for transcripts).

        Raises:
            ValueError: If *form* is not in ``{"gene", "transcript", "translation"}``.
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
        """Retrieve and normalise raw Ensembl identifier records for the requested molecular form.

        This method pulls the appropriate MySQL table(s) for *form*, copes with schema differences
        across Ensembl releases (e.g. the historical ``*_stable_id`` split tables), coerces data
        types, and standardises column names so that downstream graph-building steps all consume the
        same shape.  It finishes by delegating to :py:meth:`DatabaseManager.version_uniformize` to
        ensure the *Version* field is either a proper integer or ``NaN`` across the entire DataFrame.

        Args:
            form (str): Target molecular form - ``"gene"``, ``"transcript"``, or ``"translation"``.
                Anything else triggers a :py:class:`ValueError`.

        Returns:
            pandas.DataFrame: A de-duplicated, index-reset table whose columns depend on *form*:

                * **gene** - ``gene_id``, ``gene_stable_id``, ``gene_version``
                * **transcript** - ``transcript_id``, ``gene_id``, ``transcript_stable_id``,
                  ``transcript_version``
                * **translation** - ``translation_id``, ``transcript_id``, ``translation_stable_id``,
                  ``translation_version``

                All ID columns are ``int64`` except the ``*_stable_id`` strings; version columns are
                ``int64`` or ``float64`` (with ``NaN`` when absent).
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
        """Build a current-release gene-transcript-translation mapping table.

        The routine fetches the *raw* stable-ID/​version tables for genes,
        transcripts and translations via :py:meth:`DatabaseManager.get_db`, merges
        them into a single wide frame, and then delegates to
        :py:meth:`DatabaseManager._create_relation_helper` to harmonise version
        columns and compress the information into three canonical node labels
        (``"<stable_id>.<version>"``).  The resulting mapping is the authoritative
        per-release link between molecular forms and is consumed by downstream
        graph-building utilities such as :py:meth:`DatabaseManager.create_graph`.

        Returns:
            pandas.DataFrame: Three columns—``gene``, ``transcript``, and
            ``translation``—with one row per transcript.  The ``translation`` column
            may contain empty strings where non-coding transcripts have no
            peptide.  All data are UTF-8 strings; duplicates are removed and the
            index is reset.
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
        """Retrieve a cross-release gene-transcript-translation mapping table.

        This legacy helper pulls the Ensembl ``gene_archive`` table—spanning *all*
        releases for the current organism—via
        :py:meth:`DatabaseManager.get_table`, drops columns unrelated to identifier
        mapping, and passes the result to
        :py:meth:`DatabaseManager._create_relation_helper`.  **Because the archive
        contains known gaps, the preferred workflow is to call**
        :py:meth:`DatabaseManager.create_relation_current` **once per release and
        concatenate the outputs.**

        Returns:
            pandas.DataFrame: Same schema as
                :py:meth:`DatabaseManager.create_relation_current`—``gene``,
                ``transcript``, ``translation``—but potentially with missing rows
                because Ensembl did not always back-populate older releases.
        """
        self.log.warning("Not recommended method: Use 'create_relation_current' instead.")
        # Get the table from the server
        df = self.get_table("gene_archive", usecols=None, save_after_calculation=self.store_raw_always)
        # Remove unnecessary columns and return.
        df.drop(columns=["peptide_archive_id", "mapping_session_id"], inplace=True, errors="raise")

        return self._create_relation_helper(df)

    def _create_relation_helper(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert an ID/version matrix into the canonical three-column relationship table.

        The helper is shared by :py:meth:`DatabaseManager.create_relation_current`
        and :py:meth:`DatabaseManager.create_relation_archive` and is not intended
        for direct use.  It validates the incoming frame, fixes inconsistent
        version numbers (via :py:meth:`DatabaseManager.version_fix` and
        :py:meth:`DatabaseManager.version_fix_incomplete`), converts missing
        translations to ``NaN``-compatible floats, casts all stable-ID columns to
        string, and finally compresses each *ID + version* pair into the compact
        node label used throughout *ID-track* graphs.

        Args:
            df (pandas.DataFrame): A six-column frame with exactly the following
                names (order irrelevant): ``gene_stable_id``, ``gene_version``,
                ``transcript_stable_id``, ``transcript_version``,
                ``translation_stable_id``, ``translation_version``.

        Returns:
            pandas.DataFrame: Three columns—``gene``, ``transcript``,
                ``translation``—deduplicated and index-reset, ready for graph
                construction.

        Raises:
            ValueError: If *df* does not contain the required six columns or if
                version columns cannot be coerced to the expected numeric dtype.
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
        """Retrieve historical relationships between successive Ensembl stable IDs.

        Build a cross-release lineage table mapping every obsolete ID version to its immediate successor for the
        configured *organism*, *form*, and release window.  The information is assembled from the Core tables
        ``stable_id_event`` and ``mapping_session`` and then normalised so that all identifiers follow the canonical
        ``<stable_id>.<version>`` convention.  Downstream graph-construction utilities depend on this table to
        reconstruct how genes, transcripts, or translations evolve across Ensembl releases.

        Args:
            narrow (bool): If ``True`` drop auxiliary columns (mapping session metadata, assembly labels, creation
                timestamps, etc.) to minimise on-disk footprint; otherwise return the full schema for exploratory
                analyses.

        Returns:
            pandas.DataFrame: Seven-column table with the following fields, ordered as listed—

                * ``old_stable_id`` - obsolete identifier (empty string for “birth” events).
                * ``old_version``   - version number paired with *old_stable_id*.
                * ``new_stable_id`` - successor identifier (empty string for “retirement” events).
                * ``new_version``   - version paired with *new_stable_id*.
                * ``score``         - homology score reported by Ensembl (``NaN`` if unavailable).
                * ``old_release``   - Ensembl release where the *old* identifier last appeared.
                * ``new_release``   - release where the *new* identifier first appeared.

        Raises:
            ValueError: If the identifier delimiter
                :py:data:`idtrack._db.DB.id_ver_delimiter` is found inside any ``*_stable_id`` field, indicating
                malformed input.
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
        """Create a corrected ID-history table that repairs cyclic or duplicated version transitions (deprecated).

        Certain edge cases in the raw ``idhistory`` extraction—e.g. *Homo sapiens* ``ENSG00000232423`` at release 105—
        produce sequences like ``1 → 2, 2 → 3, 1 → 2`` where an already retired version resurfaces later on.
        Such cycles violate the monotonic version semantics assumed by graph algorithms.  This helper rewrites the
        offending rows so that once a version is superseded it never reappears, transforming the above sequence into
        the logically consistent ``3 → 2``.  The routine is retained for reproducibility but superseded by
        :py:meth:`DatabaseManager.create_id_history`.

        Args:
            narrow (bool): Propagated to the underlying data fetch—when ``True`` start from the column-reduced
                ``idhistory_narrow`` view instead of the full table.
            inspect (bool): When ``True`` add diagnostic columns (e.g. ``changed_old`` and ``changed_new``) to aid
                manual auditing of the corrections; when ``False`` return only the cleaned canonical schema.

        Returns:
            pandas.DataFrame: Corrected seven-column table ``old_stable_id``, ``old_version``, ``new_stable_id``,
                ``new_version``, ``score``, ``old_release``, ``new_release``—ready for serialization and downstream use.

        Note:
            This function is deprecated and will be removed in a future major release once the core extractor fully
            addresses the ordering anomaly.
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
        """Retrieve Ensembl-external-ID relationships and/or database statistics.

        This consolidates a complex SQL join—spanning Ensembl core tables *gene*, *transcript*, *translation* and
        the cross-reference tables *xref*, *object_xref*, *identity_xref*, *external_db*, and *external_synonym*—into a
        single pandas dataframe. It enables downstream analyses such as mapping Ensembl gene models to
        UniProt, RefSeq, or CCDS identifiers, or summarising which external sources are represented in a given Ensembl
        release.  The result type and granularity are controlled by *filter_mode*, allowing either the raw relationship
        rows or a per-database count to be returned.

        The query executed is conceptually equivalent to the (simplified) MySQL statement below,
        though the actual SQL is constructed programmatically for flexibility and performance:

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

        When tighter genomic scoping is required the *gene* table can be prefixed with *coord_system* and *seq_region*:

        .. code-block:: sql

            FROM coord_system cs
            JOIN seq_region  sr USING (coord_system_id)
            JOIN gene        g  USING (seq_region_id)

        You can experiment interactively against the public Ensembl MySQL mirror:

        .. code-block:: bash

            mysql --user=anonymous --host=ensembldb.ensembl.org -D homo_sapiens_core_105_38 -A
            # Schema reference:
            # https://m.ensembl.org/info/docs/api/core/core_schema.html

        Args:
            filter_mode (str): Controls both the **row subset** and the **output schema**. Must be one of:

                * ``"all"`` - return **every** mapping found in MySQL, no post-filtering applied.
                * ``"relevant"`` - return only mappings whose external database is marked *Include: true* in the
                    :py:meth:`ExternalDatabases.give_list_for_case` YAML configuration.
                * ``"database"`` - return a two-column summary (``name_db``, ``count``) for **all** external databases.
                * ``"relevant-database"`` - as above, but restricted to databases flagged *Include: true*.
                    The special values ``"relevant"`` and ``"relevant-database"`` implicitly consult the cached
                    :py:attr:`external_inst` to honour the user's curated allow-list.

        Returns:
            pandas.DataFrame:
                * For ``"all"`` / ``"relevant"`` - six-column frame
                    ``["release", "graph_id", "id_db", "name_db", "ensembl_identity", "xref_identity"]`` holding one
                    row per Ensembl→external identifier edge. ``graph_id`` is the Ensembl stable ID (+version), while
                    the two *identity* columns store Smith-Waterman percent identities (*float16*) for QC.
                * For ``"database"`` / ``"relevant-database"`` - two-column frame
                    ``["name_db", "count"]`` giving how many distinct ``graph_id`` values each external database
                    touches. ``count`` is an ``int64``.

        Raises:
            ValueError: If *filter_mode* is not one of the accepted literals **or** if the YAML allow-list claims a
                database that is absent from the retrieved mappings—indicating the configuration and MySQL data are
                out of sync.

        Notes:
            *Synonym handling* - any synonym brought in from ``external_synonym`` is prefixed with
            :py:data:`DB.synonym_id_nodes_prefix`, and its ``name_db`` is likewise prefixed so that synonym nodes remain
            distinguishable during graph building.
            *Caching* - the heavy MySQL queries are executed only if the processed frame is not already present in the
            manager's per-organism HDF5 cache; otherwise the cached frame is read from disk, ensuring repeat calls are
            inexpensive.
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
        """Retrieve and optionally cache external-database metadata for every assembly, release, and form.

        The helper iterates over *all* genome assemblies defined in
        :py:data:`idtrack._db.DB.assembly_mysqlport_priority`, every available Ensembl release for each assembly,
        and every identifier *form* supported by the package, downloading the ``external_database`` table for
        each combination.  The resulting frames are concatenated, enriched with ``assembly``, ``release``,
        ``form``, and ``organism`` columns, and returned to the caller.  When ``just_download`` is ``True`` the
        downloads are still performed (ensuring they are cached on disk for future runs) but an **empty**
        dataframe is returned to avoid unnecessary memory use.

        Args:
            just_download (bool):
                * **False** - concatenate intermediate results and return the union dataframe (default).
                * **True** - download and cache each frame but return an empty dataframe.

        Returns:
            pandas.DataFrame: External-database relationships augmented with assembly, release, form, and organism
                columns.  Empty when ``just_download`` is ``True``.
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
        """Return deduplicated stable-identifier/version pairs for the current form and release.

        Raw identifiers are fetched via :py:meth:`DatabaseManager.get_db`, normalised with
        :py:meth:`DatabaseManager.version_fix`, trimmed to the canonical columns, and sanity-checked.  Two
        integrity rules are enforced: (1) the delimiter
        :py:data:`idtrack._db.DB.id_ver_delimiter` must **not** appear inside any stable identifier, and
        (2) every stable identifier must be unique after deduplication.  Violations raise
        :py:class:`ValueError`.

        Returns:
            pandas.DataFrame: Two-column dataframe ``[{form}_stable_id, {form}_version]`` with duplicates removed.

        Raises:
            ValueError: If the delimiter is present inside any stable identifier or if identifiers are not
                unique after deduplication.
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

    def check_if_change_assembly_works(self, db_manager: "DatabaseManager", target_assembly: int) -> bool:
        """Evaluate whether *db_manager* can be cloned to operate on *target_assembly*.

        A lightweight health-check that calls :py:meth:`DatabaseManager.change_assembly` inside a
        ``try/except`` block and converts the outcome to a boolean flag rather than letting the exception
        propagate.  It allows batch workflows to skip assemblies that are unavailable or invalid without
        interrupting processing.

        Args:
            db_manager (DatabaseManager): Manager instance to probe.
            target_assembly (int): Genome-assembly code to test (a key of
                :py:data:`idtrack._db.DB.assembly_mysqlport_priority`).

        Returns:
            bool: ``True`` if the assembly switch succeeds without raising :py:class:`ValueError`; ``False`` otherwise.
        """
        try:
            db_manager.change_assembly(target_assembly)
            return True
        except ValueError:
            return False

    def create_external_all(self, return_mode: str) -> Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]:
        """Download and collate cross-reference mappings from every supported genome assembly.

        The manager cycles through every genome assembly recognised for the current organism (ordered by
        :py:data:`idtrack._db.DB.assembly_mysqlport_priority`), fetches the *external_relevant* mapping table
        for each via :py:meth:`~DatabaseManager.get_db`, labels every row with its source assembly, and finally
        concatenates the tables.  Because this helper is intended for **ad-hoc inspection only**, it bypasses
        the :py:meth:`~DatabaseManager.get_db` caching layer and therefore **never writes** the result to the
        local repository.

        Args:
            return_mode (str): Strategy for handling rows that appear in more than one assembly.

                - ``"all"``
                    Keep one copy of every unique
                    ``(release, graph_id, id_db, name_db, ensembl_identity, xref_identity, assembly)``
                    combination.  Duplicates are resolved *within* each assembly only.

                - ``"unique"``
                    Keep one copy of every unique
                    ``(release, graph_id, id_db, name_db, ensembl_identity, xref_identity)`` combination *across*
                    **all** assemblies, preferring the assembly with the highest
                    priority. *(Currently no downstream use case.)*

                - ``"duplicated"``
                    Return **only** the rows that occur in more than one assembly as a
                    :py:class:`pandas.core.groupby.generic.DataFrameGroupBy`, keyed by the same column set used for
                    ``"unique"``. *(Currently no downstream use case.)*

        Returns:
            Union[pandas.DataFrame, pandas.core.groupby.generic.DataFrameGroupBy]:
                - If *return_mode* is ``"all"`` or ``"unique"``,
                    a de-duplicated cross-reference table with the
                    columns ``release``, ``graph_id``, ``id_db``, ``name_db``, ``ensembl_identity``, ``xref_identity``,
                    and ``assembly``.
                - If *return_mode* is ``"duplicated"``,
                    a group-by view containing only duplicated entries.

        Raises:
            ValueError: If *return_mode* is not ``"all"``, ``"unique"``, or ``"duplicated"``.
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
        """Determine whether each Ensembl release stores identifiers with or without version suffixes.

        Ensembl stable identifiers can appear either *with* a ``.version`` facet (e.g. *ENSG00000139618.17*)
        or *without* it (e.g. *YAL001C* in *S. cerevisiae*).  For robust cross-release tracking the package
        needs to know which convention applies to every release of the current organism.  The method loops
        over :py:attr:`available_releases`, downloads the raw identifier table for *self.form*, and inspects
        the ``<form>_version`` column:

        * **All values NaN** → the release uses *unversioned* identifiers.
        * **No values NaN**  → the release uses *versioned* identifiers.
        * **Mixed NaN / non-NaN** → unsupported; raises :py:class:`NotImplementedError`.

        The outcome is encoded as a Boolean flag per release and later consumed by
        :py:meth:`~DatabaseManager.check_version_info` to decide whether version strings should be kept,
        stripped, or synthesised.

        Returns:
            pandas.DataFrame: Two-column table with:
                * ``ensembl_release`` - integer release number.
                * ``version_info``   - ``True`` if *all* identifiers **lack** a version suffix, ``False`` if
                    *all* identifiers **include** a version suffix.

        Raises:
            NotImplementedError: If any individual release contains a mixture of versioned and unversioned
                identifiers, indicating an inconsistent upstream annotation.
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
        """Retrieve or create a cached data table defined by an indicator string.

        This method is the central gateway for *all* tabular resources managed by
        :py:class:`~DatabaseManager`.  It interprets a compact *indicator* string,
        decides whether the requested table already exists in the local HDF5
        repository, and either loads the cached copy or triggers the appropriate
        builder (``create_*`` helper) to download/assemble it.  A consistent naming
        convention is maintained so that subsequent calls with the same indicator
        transparently reuse the on-disk cache, ensuring reproducible builds and
        minimal network traffic.

        **Supported base indicators**

        * ``external`` — cross-reference database registry; optional qualifier
            ``relevant`` | ``database`` | ``relevant-database`` narrows the view.
        * ``idsraw`` — raw Ensembl identifiers for a given form *(``gene``,
          ``transcript``, ``translation``)*; requires the form as qualifier.
        * ``ids`` — release-specific identifier table (no qualifier).
        * ``externalcontent`` — summary of per-database content.
        * ``relationcurrent`` — current gene/ID relationships.
        * ``relationarchive`` — historical gene/ID relationships across releases.
        * ``idhistory`` — full ID history; qualifier ``narrow`` restricts to current IDs.
        * ``versioninfo`` — version comparison across releases.
        * ``availabledatabases`` — list of locally cacheable resources.

        Additional indicators may be introduced by subclass extensions; consult the
        module documentation for the authoritative list.

        Args:
            df_indicator (str): Compact descriptor of the table to retrieve.  Must
                follow the ``base[qualifier]`` pattern described above.
            create_even_if_exist (bool): Force a rebuild/download even if
                a cached copy is present.  Defaults to ``False``.
            save_after_calculation (bool): Persist a newly created table
                to the local HDF5 store.  Has no effect when the table is merely
                loaded from disk.  Defaults to ``True``.
            overwrite_even_if_exist (bool): When saving, replace an
                existing HDF5 key with the same hierarchy (file-internal path).
                Defaults to ``False``.

        Returns:
            pandas.DataFrame | pandas.Series: The requested dataset.  The exact
            shape, index, and column layout depend on ``df_indicator``; see the
            indicator list above for semantic details.

        Raises:
            ValueError: If *df_indicator* is malformed, references an unsupported
                resource, or its qualifier violates the expected pattern (e.g.,
                missing form for ``idsraw``).
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
        """Resolve HDF5 hierarchy key and absolute file path for a dataframe request.

        This internal helper centralises every rule that :py:class:`DatabaseManager` uses to build HDF5
        *hierarchy* keys and their corresponding on-disk filenames, ensuring that any two call-sites
        confronted with the same combination of organism, genome assembly, Ensembl release, dataframe
        *kind*, and optional column subset produce **identical** results.  By funnelling every I/O
        operation through this method the wider package avoids silent cache misses, duplicate downloads,
        and hard-to-trace inconsistencies in downstream analytics.  Public code is expected to invoke
        higher-level wrappers such as :py:meth:`DatabaseManager.get_db`; use this routine only when
        implementing new caching utilities or in low-level tests.

        Args:
            df_type (str): Category of dataframe whose name is required. Accepted values are
                ``"processed"``, ``"mysql"``, and ``"common"``; any other string triggers :py:class:`ValueError`.
            ensembl_release (int, optional): Ensembl release to encode in the filename.  If *None*, the
                current :py:attr:`DatabaseManager.ensembl_release` is used instead.
            kwargs: Additional keyword arguments forwarded to the helper that handles the selected
                *df_type* (currently only ``usecols`` for the *mysql* path).
            args: Positional arguments interpreted according to *df_type*:

                - **processed** - ``df_indicator`` (str): symbolic label such as ``"idhistory"`` or
                  ``"idsraw_gene"``.  The manager appends :py:attr:`DatabaseManager.form` so that artefacts
                  for different biological forms do not collide.

                - **mysql** - ``table_key`` (str): raw MySQL table name (e.g. ``"gene"``, ``"exon"``).
                  An optional ``usecols`` (list[str]) must then be supplied via *kwargs*; the column list
                  is embedded in the hierarchy using the delimiter held in
                  :py:attr:`DatabaseManager._column_sep`.

                - **common** - ``df_indicator`` (str): same as the processed case **but without** the form
                  suffix, allowing cross-form artefacts (e.g. ``"availabledatabases"``) to share a single key.

        Returns:
            tuple[str, str]: Two-element tuple ``(hierarchy_key, file_path)`` where *hierarchy_key* is
                the internal node path (e.g. ``"ens111_mysql_gene_COL_gene_id"``) and *file_path* is the
                absolute path to ``<local_repository>/<organism>_assembly-<assembly>.h5``.  The path is
                **not** created on disk—callers remain responsible for reading or writing the HDF5 file.

        Raises:
            ValueError: If *df_type* is not one of the accepted categories or if the positional/keyword
                argument combination does not satisfy the expectations for that category (e.g. missing
                ``table_key`` when *df_type* is ``"mysql"``).
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
        """Assemble fully qualified node names from a *stable-ID / version* DataFrame.

        This convenience routine converts a two-column frame—usually produced by
        :py:meth:`DatabaseManager.get_db` with the *ids* form—into the canonical node
        labels used throughout ID-track graphs (e.g. ``ENSG00000000001.1``).  It first
        validates that the input columns match :py:data:`self._identifiers`
        (typically ``["gene_stable_id", "gene_version"]`` or analogous for the
        current ``form``), then delegates per-row processing to
        :py:meth:`DatabaseManager.node_dict_maker` and
        :py:meth:`DatabaseManager.node_name_maker`.  The resulting list may be fed
        directly into downstream graph builders or written to disk for later reuse.

        Args:
            dbm_the_ids (pandas.DataFrame): Two-column frame containing the stable
                identifiers and their Ensembl version numbers.  The column order and
                names **must** exactly match ``self._identifiers``; otherwise an
                exception is raised.

        Returns:
            list[str]: Ordered list where each element is either
                ``"<ID>.<version>"`` when a valid numeric version is present or simply
                ``"<ID>"`` when the version is *None* / *NaN* / an alternative marker
                (see :py:data:`idtrack._db.DB.alternative_versions`).

        Raises:
            ValueError: If ``dbm_the_ids`` does not contain the expected column
                names stored in :py:data:`self._identifiers`.
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
        """Concatenate *ID* and *Version* into a single node label.

        Given the miniature dictionary returned by
        :py:meth:`DatabaseManager.node_dict_maker`, this helper builds the string
        representation that uniquely identifies a biological entity within the graph
        layer.  When a numeric version is available, it appends that value to the
        stable ID using :py:data:`idtrack._db.DB.id_ver_delimiter` (``"."`` by
        default).  For organisms or datasets lacking versioned identifiers, it falls
        back to the bare stable ID to preserve compatibility.

        Args:
            node_dict (dict[str, Any]): Mapping with exactly two keys, ``"ID"`` and
                ``"Version"``, as produced by
                :py:meth:`DatabaseManager.node_dict_maker`.

        Returns:
            str: Either ``"<ID>.<version>"`` or ``"<ID>"`` depending on whether a
                non-null, non-alternative version is present.
        """
        if node_dict["Version"] and not pd.isna(node_dict["Version"]):
            return node_dict["ID"] + DB.id_ver_delimiter + str(node_dict["Version"])
        else:
            return node_dict["ID"]

    @staticmethod
    def node_dict_maker(id_entry: str, version_entry: Any) -> dict[str, Any]:
        """Return a normalized *ID/Version* dictionary from raw column values.

        This helper creates the canonical structure consumed by
        :py:meth:`DatabaseManager.node_name_maker` and higher-level graph utilities,
        ensuring that version numbers are strictly integers whenever possible.  It
        also recognises special placeholders defined in
        :py:data:`idtrack._db.DB.alternative_versions` (e.g. ``"Retired"`` or
        ``"Void"``) and passes them through unchanged so that downstream code can
        handle deprecated or missing entries appropriately.

        Args:
            id_entry (str): Stable identifier portion preceding the delimiter
                (e.g. ``"ENSG00000000001"``).
            version_entry (Any): Raw version value following the delimiter
                (e.g. ``1`` in ``"ENSG00000000001.1"``).  May be *float*, *int*,
                *str*, *None*, *NaN*, or an alternative placeholder such as
                ``"Retired"``.

        Returns:
            dict[str, Any]: ``{"ID": id_entry, "Version": version_entry}`` with
            *Version* coerced to *int* when it represents a whole number.

        Raises:
            ValueError: If ``version_entry`` is numeric but contains a fractional
                component (e.g. ``1.2``), indicating a malformed identifier that cannot
                be represented as an integer version.
        """
        if version_entry and not pd.isna(version_entry) and version_entry not in DB.alternative_versions:
            if int(version_entry) != float(version_entry):
                raise ValueError(f"Version is floating: {(id_entry, version_entry)}")
            else:
                version_entry = int(version_entry)
        return {"ID": id_entry, "Version": version_entry}

    def version_uniformize(self, df: pd.DataFrame, version_str: str) -> pd.DataFrame:
        """Normalise a *Version* column so every entry is either an ``int`` or ``NaN``.

        This post-processing helper finalises the output of :py:meth:`DatabaseManager.create_ids`.
        Ensembl releases differ: some assign an explicit integer version to every stable identifier,
        whereas others omit the suffix entirely.  Downstream code expects a *uniform* dtype, so this
        routine coerces the designated column to a proper integer when *all* entries are present or
        fills the entire column with ``np.nan`` when *none* are.  Mixed presence is forbidden because
        it would break the ID-version pairing logic used by :py:meth:`DatabaseManager.node_name_maker`.

        Args:
            df (pandas.DataFrame): Frame returned by :py:meth:`create_ids`; must already contain a
                column named *version_str*.
            version_str (str): Name of the column that holds version information (e.g. ``"gene_version"``).

        Returns:
            pandas.DataFrame: Same object *df* with *version_str* either cast to ``int64`` or overwritten
                with ``np.nan`` for every row.

        Raises:
            NotImplementedError: If some rows have a version and others do not, indicating an Ensembl
                release with inconsistent schema.  Such a release is currently unsupported.
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
        """Clean up version columns when *some* identifiers are entirely missing.

        Ensembl *translation* tables occasionally encode parent IDs without a version while descendants
        retain one, producing frames where *id_col_fx* is ``NaN`` but *ver_col_fx* contains a number.
        This helper splits the frame, delegates to :py:meth:`version_fix` for each subset, then stitches
        the pieces back together so that every row obeys a single “with/without/add version” policy.

        Args:
            df_fx (pandas.DataFrame): Data to harmonise.  The frame **must** include *id_col_fx* and *ver_col_fx*.
            id_col_fx (str): Column holding the *stable* part of the identifier (e.g. ``"translation_id"``).
            ver_col_fx (str): Column holding the integer version suffix.

        Returns:
            pandas.DataFrame: Frame whose *ver_col_fx* is consistent with the organism-level policy
                determined by :py:meth:`check_version_info`.
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
        """Apply a *global* ID-version policy to a DataFrame.

        Depending on the organism and its historical annotation quirks, identifiers may (1) **never**
        include a version, (2) **always** include a version, or (3) require a *synthetic* version when
        mixing cross-release data.  The *version_info* flag encodes that policy:

        * ``"without_version"`` — strip all versions (set column to ``NaN``).
        * ``"with_version"``    — cast column to ``int64`` (all values must exist).
        * ``"add_version"``     — fill missing entries with :py:data:`DB.first_version`.

        Args:
            df (pandas.DataFrame): Frame whose *version_str* column needs harmonising.
            version_str (str): Name of the column that stores version numbers.
            version_info (Optional[str]): One of ``"add_version"``, ``"without_version"``, or
                ``"with_version"``.  When ``None`` (default) the method calls
                :py:meth:`check_version_info` to determine the correct policy automatically.

        Returns:
            pandas.DataFrame: Same object *df* with *version_str* updated in-place.

        Raises:
            ValueError: If *version_info* is not recognised.
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
        """Infer whether the organism's Ensembl IDs come **with**, **without**, or **mixed** versions.

        The method scans all releases available for the current genome assembly and inspects the boolean
        flag in the ``version_info`` column of a pre-computed table (``get_db("versioninfo")``).  Three
        mutually exclusive scenarios exist:

        * All releases lack version suffixes: ``"without_version"``
        * All releases include suffixes: ``"with_version"``
        * A mixture of both states: ``"add_version"`` (synthetic versions will be injected)

        Returns:
            str: One of ``"without_version"``, ``"with_version"``, or ``"add_version"``.  Callers use the
                string to decide how to standardise identifier columns.

        Raises:
            ValueError: If the *version_info* column in the source table is not strictly boolean, signalling
                a corrupted download or schema drift.
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
