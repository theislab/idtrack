#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import os

import numpy as np
import pandas as pd
import yaml


class ExternalDatabases:
    """Creates and manages external database `yaml` file."""

    def __init__(
        self,
        organism: str,
        ensembl_release: int,
        form: str,
        local_repository: str,
        genome_assembly: int,
    ):
        """Class initialization.

        Args:
            organism: Refer to :py:attr:`DatabaseManager.__init__.organism`.
            ensembl_release: Refer to :py:attr:`DatabaseManager.__init__.ensembl_release`.
            form: Refer to :py:attr:`DatabaseManager.__init__.form`.
            local_repository: Refer to :py:attr:`DatabaseManager.__init__.local_repository`.
            genome_assembly: Refer to :py:attr:`DatabaseManager.__init__.genome_assembly`.
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
        """Creates the templete `yaml` file, indicating each external database possible.

        Example output file will start as following, of course, with much more database under `gene`.

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

        Note that the `Potential Synonymous` is now all empty. In the following versions, it is aimed to integrate
        a feature that prevent to heve synonymous databases in the list. Likewise, `Database Index` has now no use case,
        in the program.

        It is important to follow the final warning raised by the method. "Please edit the file based on
        requested external databases and add '_modified' to the file name.". The editing should be done by converting
        `Include` sections from `false` to `true`. It is recommended to make the change for each assembly for a given
        database.

        Args:
            df: The output of following operation ``db_manager.create_database_content()``.

        Raises:
            ValueError: Unexpected error.
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
        """File name creator, intended to be used only :py:meth:`ExternalDatabases.create_template_yaml`.

        Returns:
            Absolute path for the template yaml file.
        """
        return os.path.join(self.local_repository, f"{self.organism}_externals_template.yml")

    def file_name_modified_yaml(self, mode: str) -> str:
        """File name creator, intended to be used only :py:meth:`ExternalDatabases.load_modified_yaml`.

        Args:
            mode: Decide to retrieve whether the `default` yaml file for the given organism
                or `configured` yaml file, which is created and modified by the user.

        Raises:
            ValueError: If 'mode' is not either 'configured' or 'default'.

        Returns:
            File name as an absolute path.
        """
        fb = f"{self.organism}_externals_modified.yml"
        if mode == "configured":
            return os.path.join(self.local_repository, fb)

        elif mode == "default":
            return os.path.join(os.path.dirname(__file__), "default_config", fb)

        else:
            raise ValueError(f"Unknown value for 'mode': {mode}.")

    def load_modified_yaml(self) -> dict:
        """Reads the modified `yaml` file if created by the user, else read the default for the given organism.

        Returns:
            Loaded yaml file as the dictionary

        Raises:
            FileNotFoundError: When neither default nor configured `yaml` file exist.
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

    def validate_yaml_file_up_to_date(self, y):
        ensembl_releases = {
            int(e)
            for _, j1 in y.items()
            for _, j2 in j1.items()
            for _, j3 in j2.items()
            for _, j4 in j3["Assembly"].items()
            for e in j4["Ensembl release"].split(",")
        }
        if self.ensembl_release not in ensembl_releases:
            raise ValueError(
                f"The ensembl release of DatabaseManager ({self.ensembl_release}) is not included in ensembl releases "
                "in ExternalDatabase config yaml file. The YAML file needs to be updated with never version. "
                f"If you use the default config, please create graph (or track object) with '{max(ensembl_releases)}', "
                "which should solve the issue. Let us know this also on GitHub issues so that we can update the "
                "default config file."
            )

    def give_list_for_case(self, give_type: str) -> list:
        """Retrieve some simple information from `yaml` file.

        Args:
            give_type: Either ``'db'`` or ``'assembly'``.

                - `db`: the method gives associated external database names in `yaml` file for given
                  organism, assembly, Ensembl release and form.

                - `assembly`: the method gives possible Ensembl assemblies that at least one external database is
                  defined in ``yaml`` config file.

        Returns:
            List of strings of databases if 'db' is `give_type`, or list of integers if 'assembly' is `give_type`.

        Raises:
            ValueError: If `give_type` is not in specified format.
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
