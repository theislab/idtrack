#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import os

import numpy as np
import pandas as pd
import yaml


class ExternalDatabases:
    """Manage third-party metadata for Ensembl entities through YAML side-car files.

    This helper encapsulates everything related to the *external* (i.e. non-Ensembl) databases that
    can be linked to a given organism, genome assembly, release, and biological form
    (*gene* / *transcript* / *translation*).  Examples of such resources include ArrayExpress,
    RefSeq, Uniprot, HGNC, and dozens of smaller annotation providers.  Rather than hard-coding
    those relationships, the wider *ID-Track* toolkit stores them in a human-readable YAML file that
    lives next to the local data cache managed by :py:class:`_database_manager.DatabaseManager`.

    The YAML workflow is:

    1. :py:meth:`create_template_yaml` enumerates every known combination and writes a *template*
       where each entry is marked ``Include: false``.
    2. A user (or an automated post-processing step) reviews the template, toggling ``Include`` to
       ``true`` for the resources they need.
    3. The modified YAML is saved under *local_repository*; subsequent calls to
       :py:meth:`load_modified_yaml` return it as a plain :py:class:`dict` for downstream logic.
    4. :py:meth:`validate_yaml_file_up_to_date` warns if the user file lags behind a newer template
       (e.g. because a later Ensembl release introduced extra tables).
    5. Utility helpers such as :py:meth:`give_list_for_case` expose convenient filtered views—e.g.
       *all databases that should be downloaded for the current form*, or *all releases supported by
       assembly 38*.

    In short, *ExternalDatabases* provides a single, version-controlled “contract” describing which
    third-party tables belong in an ID-track run, while granting users explicit opt-in control over
    optional resources.
    """

    def __init__(
        self,
        organism: str,
        ensembl_release: int,
        form: str,
        local_repository: str,
        genome_assembly: int,
    ):
        """Instantiate a YAML controller tied to a specific organism, release, and assembly.

        The constructor mirrors the core configuration of :py:class:`_database_manager.DatabaseManager`
        so that both objects operate on the exact same coordinate system.  No I/O is performed at
        construction time; paths are merely recorded, and loggers are configured.  Heavy-weight
        actions—such as scanning the cache for existing YAMLs or writing new ones—happen lazily when
        the corresponding methods are called.

        Args:
            organism (str): Canonical Ensembl species identifier in *snake_case*
                (e.g. ``"homo_sapiens"``).  Case-insensitive but must match Ensembl conventions.
            ensembl_release (int): Target Ensembl release number (e.g. ``110``).  Must be ≥ 79 and
                correspond to a release that actually exists for *organism* and *assembly*.
            form (str): Entity level—``"gene"``, ``"transcript"``, or ``"translation"``.  Any other
                value raises :py:class:`ValueError` in higher-level validation.
            local_repository (str): Writable directory where YAML files and downloads are cached.
                The directory need not pre-exist; if missing, most public methods will attempt to create it.
            genome_assembly (int): NCBI genome assembly version (e.g. ``38`` for GRCh38).  Used to
                disambiguate multiple assemblies available for the same *organism*/*release* pair.
        """
        # The logger for informing the user about the progress.
        self.log = logging.getLogger("external_databases")

        # Instance attributes
        self.local_repository = local_repository
        self.organism = organism
        self.form = form
        self.ensembl_release = ensembl_release
        self.genome_assembly = genome_assembly

    def create_template_yaml(self, df: pd.DataFrame):
        """Generate a template YAML enumerating external-database options.

        This helper scans *df*—typically the dataframe returned by
        :py:meth:`idtrack._database_manager.DatabaseManager.create_database_content`—and
        writes a scaffold configuration file to
        :py:meth:`~ExternalDatabases.file_name_template_yaml`.
        The file lists every *organism* → *form* → database combination observed in
        *df*, grouped by genome assembly and Ensembl release.  For each entry the
        template records whether the database should be *included* when building an
        ID-history graph, its integer **Database Index**, and an empty
        **Potential Synonymous** placeholder that future versions may use to flag
        overlapping resources.

        Users are **expected to edit** the generated file—changing ``Include`` from
        ``false`` to ``true`` where appropriate—and **rename** it by appending
        ``_modified`` to the filename before the package will load it.  A warning to
        that effect is emitted via :py:meth:`logging.Logger.warning`.

        The resulting YAML resembles the structure below (truncated for brevity):

        .. code-block:: yaml

            homo_sapiens:
                gene:
                    ArrayExpress:
                        Assembly:
                            "37":
                                Ensembl release: 79,80,81,82,83,84,85,86,87,88,89
                                Include: false
                            "38":
                                Ensembl release: 79,80,81,82,83,84,85,86,87,88,89
                                Include: false
                        Database Index: 0
                        Potential Synonymous: ""
                    Clone-based (Ensembl):
                        Assembly:
                            "37":
                                Ensembl release: 79,80,81,82,83,84,85
                                Include: false
                            "38":
                                Ensembl release: 79,80,81,82,83,84,85
                                Include: false
                        Database Index: 5
                        Potential Synonymous: ""

        **Editing guidelines**

        * Set ``Include`` to *true* for every assembly of the databases you need.
        * Save the edited file with ``_modified`` appended to the base name so that
          downstream routines load the customised version.

        Args:
            df (pandas.DataFrame): Dataframe containing at least the columns
                ``["organism", "form", "name_db", "assembly", "release"]``.  It
                should be produced by
                :py:meth:`idtrack._database_manager.DatabaseManager.create_database_content`
                so that the expected schema is guaranteed.

        Raises:
            ValueError: If *df* contains duplicate *assembly* entries for the same
                organism/form/database triple, causing an internal consistency check
                to fail.

        Notes:
            The **Potential Synonymous** is now all empty. In the following versions, it is aimed to integrate
            a feature that prevent to heve synonymous databases in the list. Likewise, `Database Index` has now no
            use case, in the program.
            It is important to follow the final warning raised by the method. ''Please edit the file based on
            requested external databases and add '_modified' to the file name.''. The editing should be done by
            converting `Include` sections from `false` to `true`. It is recommended to make the change for each
            assembly for a given database.
        """

        def list_to_str(iterable):
            return ",".join(map(str, iterable))

        r: dict[str, dict] = dict()
        database_id = {item: i for i, item in enumerate(sorted(np.unique(df["name_db"])))}
        for a1 in sorted(np.unique(df["organism"])):
            df_a1 = df[df["organism"] == a1]
            for a2 in sorted(np.unique(df_a1["form"])):
                df_a2 = df_a1[df_a1["form"] == a2]
                for a3 in sorted(np.unique(df_a2["name_db"])):
                    df_a3 = df_a2[df_a2["name_db"] == a3]
                    for a4 in sorted(np.unique(df_a3["assembly"])):
                        df_a4 = df_a3[df_a3["assembly"] == a4]
                        a4_str = str(a4)

                        if a1 not in r:
                            r[a1] = dict()
                        if a2 not in r[a1]:
                            r[a1][a2] = dict()
                        if a3 not in r[a1][a2]:
                            r[a1][a2][a3] = {
                                "Database Index": database_id[a3],
                                "Potential Synonymous": "",
                            }
                            r[a1][a2][a3]["Assembly"] = dict()
                        if a4_str not in r[a1][a2][a3]["Assembly"]:
                            r[a1][a2][a3]["Assembly"][a4_str] = {
                                "Ensembl release": list_to_str(sorted(np.unique(df_a4["release"]))),
                                "Include": False,
                            }
                        else:
                            raise ValueError

        with open(self.file_name_template_yaml(), "w") as yaml_file:
            yaml.dump(r, yaml_file)

        self.log.warning(
            f"File created on {self.file_name_template_yaml()}\n"
            f"Please edit the file based on requested external databases "
            f"and add '_modified' to the file name. See package documentation for further detail."
        )

    def file_name_template_yaml(self) -> str:
        """Return absolute path to the *template* YAML configuration file.

        A helper that deterministically builds the filename used by
        :py:meth:`ExternalDatabases.create_template_yaml` when it first scaffolds the
        external-database configuration for *organism*.  Centralising the logic here
        keeps every component of *idtrack* that may need the path (tests, CLI tools,
        future maintenance scripts) in perfect sync with a single implementation.  The
        method performs **no** I/O; it merely concatenates
        :py:attr:`ExternalDatabases.local_repository` and the conventional filename
        pattern ``"<organism>_externals_template.yml"`` so callers can decide whether
        to create, read, or overwrite the file.

        Returns:
            str: Absolute path of ``<organism>_externals_template.yml`` located inside
            :py:attr:`ExternalDatabases.local_repository`.
        """
        return os.path.join(self.local_repository, f"{self.organism}_externals_template.yml")

    def file_name_modified_yaml(self, mode: str) -> str:
        """Resolve the path to a *modified* YAML file customised by the user or shipped with the package.

        The method supports two *modes* that map to different storage locations:

        - ``"configured"`` - the user-edited file living in :py:attr:`ExternalDatabases.local_repository`.
        - ``"default"`` - the read-only fallback bundled under ``<package_root>/default_config`` for quick starts
          and unit tests.

        By funnelling every lookup through this routine, higher-level helpers such as
        :py:meth:`ExternalDatabases.load_modified_yaml` remain agnostic about the
        underlying directory structure and can focus on validation and parsing
        instead.

        Args:
            mode (str): Either ``"configured"`` or ``"default"`` selecting the
                corresponding search location.

        Returns:
            str: Absolute path of the requested YAML file.

        Raises:
            ValueError: If *mode* is not one of the recognised values.
        """
        fb = f"{self.organism}_externals_modified.yml"
        if mode == "configured":
            return os.path.join(self.local_repository, fb)

        elif mode == "default":
            return os.path.join(os.path.dirname(__file__), "default_config", fb)

        else:
            raise ValueError(f"Unknown value for 'mode': {mode}.")

    def load_modified_yaml(self) -> dict:
        """Load the user-edited or default YAML configuration and verify release compatibility.

        This convenience wrapper searches for the *configured* YAML file first; if it
        does not exist or lacks read permissions a warning is logged and the
        *default* YAML file shipped with the package is tried instead.  Failure to
        locate **either** file aborts the process with :py:class:`FileNotFoundError`.
        After loading, the method delegates to
        :py:meth:`ExternalDatabases.validate_yaml_file_up_to_date` to ensure that the
        currently requested Ensembl release is represented in the configuration.

        Returns:
            dict: Parsed YAML content keyed by ``{organism → form → database → Assembly → {...}}``.

        Raises:
            FileNotFoundError: If neither the *configured* nor the *default* YAML file can be accessed.
        """
        file_name = self.file_name_modified_yaml(mode="configured")
        if not os.access(file_name, os.R_OK):
            self.log.warning(f"External database config is not found in provided temp directory: `{file_name}`.")

            file_name = self.file_name_modified_yaml(mode="default")  # Look at the alternative
            if os.access(file_name, os.R_OK):
                self.log.warning(f"The package uses the default config file for {self.organism}.")
            else:
                raise FileNotFoundError(
                    f"No default config file for `{self.organism}` distributed with the package: `{file_name}`. "
                    f"Please see `create_template_yaml' method description "
                    f"to learn how to create an external 'yaml' file."
                )

        with open(file_name) as yaml_file:
            y = yaml.safe_load(yaml_file)
        self.validate_yaml_file_up_to_date(y)
        return y

    def validate_yaml_file_up_to_date(self, read_yaml_file):
        """Assert that the YAML configuration lists the active Ensembl release.

        The external-database mapping evolves with each Ensembl release.  This helper
        extracts the set of releases encoded in *read_yaml_file*—no matter how deeply
        nested—and verifies that :py:attr:`ExternalDatabases.ensembl_release` is
        present.  Triggering an exception here prevents downstream graph-construction
        logic from silently operating on incomplete or outdated metadata, prompting
        users to regenerate or update the YAML file before proceeding.

        Args:
            read_yaml_file (dict): Dictionary produced by
                :py:meth:`ExternalDatabases.load_modified_yaml` containing the loaded YAML structure.

        Raises:
            ValueError: If the current Ensembl release is absent from the YAML configuration.
        """
        ensembl_releases = {
            int(e)
            for _, j1 in read_yaml_file.items()
            for _, j2 in j1.items()
            for _, j3 in j2.items()
            for _, j4 in j3["Assembly"].items()
            for e in j4["Ensembl release"].split(",")
        }
        if self.ensembl_release not in ensembl_releases:
            raise ValueError(
                f"The Ensembl release {self.ensembl_release} is not included in any entry of the YAML config file.\n"
                f"Please update the configuration to include this release, or create the graph/track object using a "
                f"supported release (e.g., {max(ensembl_releases)}).\n"
                "You may also report this issue on GitHub so the default configuration can be updated."
            )

    def give_list_for_case(self, give_type: str) -> list:
        """Return database names or assembly codes extracted from the external-DB YAML file.

        The helper provides a lightweight way for higher-level components (e.g.
        :py:class:`~idtrack._database_manager.DatabaseManager`) to discover which
        external resources—or which genome assemblies—are currently *eligible*
        according to the user-editable YAML configuration created by
        :py:meth:`ExternalDatabases.create_template_yaml`.  Instead of forcing the
        caller to parse the YAML structure manually, the method filters the entries
        for the manager's *organism*, *form*, *Ensembl release* and
        *genome assembly* and returns the requested slice.

        Args:
            give_type (str): Kind of list to return.  Accepted values are

                - ``"db"``
                  external-database **names** (``str``) whose
                  ``Include`` flag is ``true`` for the current organism, form,
                  assembly and Ensembl release.

                - ``"assembly"``
                  genome-assembly **codes** (``int``) that
                  contain *any* external database entry for the current organism
                  and form.  Assemblies are returned even if their ``Include``
                  flags are still ``false``.

        Returns:
            list[str] | list[int]:
                - When *give_type* is ``"db"``, a list of database names.
                - When *give_type* is ``"assembly"``, a list of assembly codes.

        Raises:
            ValueError: If *give_type* is not ``"db"`` nor ``"assembly"`` or if an
                unexpected internal inconsistency is encountered while traversing
                the YAML structure.
        """
        the_dict_loaded = self.load_modified_yaml()
        the_dict = the_dict_loaded[self.organism][self.form]

        result = set()
        for db_name in the_dict:
            for asm in the_dict[db_name]["Assembly"]:
                item = the_dict[db_name]["Assembly"][asm]
                res_ens = map(int, item["Ensembl release"].split(","))

                if self.ensembl_release in res_ens and item["Include"]:
                    if give_type == "db" and int(asm) == self.genome_assembly:
                        result.add(db_name)

                    elif give_type == "db":
                        pass

                    elif give_type == "assembly":
                        result.add(int(asm))

                    else:
                        raise ValueError

        return list(result)
