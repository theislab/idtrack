#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import os
from functools import cached_property

import numpy as np
import pandas as pd
import yaml


class ExternalDatabases:
    """Todo."""

    def __init__(self,
                 organism: str,
                 ensembl_release: int,
                 form: str,
                 local_repository: str,
                 genome_assembly: int,
                 ):
        """Todo.

        Args:
            organism: Todo.
            ensembl_release: Todo.
            form: Todo.
            local_repository: Todo.
            genome_assembly: Todo.
        """
        self.local_repository = local_repository
        self.organism = organism
        self.form = form
        self.ensembl_release = ensembl_release
        self.genome_assembly = genome_assembly
        self.log = logging.getLogger("external")

    def create_template_yaml(self, df: pd.DataFrame):
        """Todo.

        Args:
            df: = db_manager.get_db("externalcontent")

        Raises:
            ValueError: Todo.
        """

        def list_to_str(iterable):
            return ",".join(map(str, iterable))

        r = dict()
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

        with open(self.file_name_yaml, "w") as yaml_file:
            yaml.dump(r, yaml_file)

        self.log.info(
            f"File created on {self.file_name_yaml}\n"
            f"Please edit the file based on requested external databases "
            f"and add '_modified' to the file name. See package documentation for further detail."
        )

    @cached_property
    def file_name_yaml(self):
        """Todo.

        Returns:
            Todo.
        """
        return os.path.join(self.local_repository, f"{self.organism}_externals_template.yml")

    @cached_property
    def file_name_modified_yaml(self):
        """Todo.

        Returns:
            Todo.
        """
        return os.path.join(self.local_repository, f"{self.organism}_externals_modified.yml")

    def load_modified_yaml(self) -> dict:
        """Todo.

        Returns:
            Todo.

        Raises:
            FileNotFoundError: Todo.
        """
        if not os.access(self.file_name_modified_yaml, os.R_OK):
            td, tf = os.path.split(self.file_name_modified_yaml)
            raise FileNotFoundError(
                f"External database config '{tf}' is not found in provided temp directory: '{td}'. "
                f"Either download from the GitHub repository, or create a template with "
                f"`create_template_yaml` method and edit accordingly. "
                f"See `create_template_yaml` explanation for details of editing procedure."
            )

        with open(self.file_name_modified_yaml) as yaml_file:
            return yaml.safe_load(yaml_file)

    def give_list_for_case(self, give_type: str) -> list:
        """Todo.

        Args:
            give_type:
                - ``'db'``: the method gives associated external database names of class' ``DatabaseManager``
                  instance which has a certain the Ensembl release and the Ensembl assembly.

                - ``'assembly'``: the method gives possible Ensembl assemblies of class' ``DatabaseManager``
                  instance that has a certain the Ensembl release and has at least one external database
                  defined in ``yaml`` config file.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
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
