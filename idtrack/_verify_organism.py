#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging

import pandas as pd
import requests

from ._db import DB


class VerifyOrganism:
    """Todo."""

    def __init__(self, organism_query: str):
        """The class constructor for VerifyOrganism.

        Args:
            organism_query: Todo.
        """
        self.log = logging.getLogger("verify_organism")
        self._connection_timeout = 10
        self._reading_timeout = 10
        self.organism_query = organism_query.lower()
        self.name_synonyms_dataframe, self.ensembl_release_dataframe = self.fetch_organism_and_latest_release(
            DB.connection_timeout, DB.reading_timeout
        )

    def get_latest_release(self):
        """Todo.

        Returns:
            Todo.
        """
        formal_name = self.get_formal_name()
        # returns int not float!
        return int(self.ensembl_release_dataframe.loc[formal_name]["ensembl_release"])

    def get_formal_name(self):
        """Todo.

        Returns:
            Todo.

        Raises:
            KeyError: Todo.
            ValueError: Todo.
        """
        the_search = self.name_synonyms_dataframe[self.name_synonyms_dataframe["synonym"] == self.organism_query]
        cm = "Please inspect 'name_synonyms_dataframe' variable of this instance for troubleshooting."
        if len(the_search) == 0:
            raise KeyError(f"The query ('{self.organism_query}') is not " f"defined in the Ensembl database. {cm}")
        elif len(the_search) > 1:
            raise ValueError(
                f"There are multiple entries for query ('{self.organism_query}') "
                f" in Ensembl database. The query is hence ambiguous. {cm}"
            )
        else:
            return self.name_synonyms_dataframe[self.name_synonyms_dataframe["synonym"] == self.organism_query][
                "formal_name"
            ].iloc[0]

    def fetch_organism_and_latest_release(self, connect_timeout, read_timeout):
        """Todo.

        Args:
            connect_timeout: Todo.
            read_timeout: Todo.

        Returns:
            Todo.

        Raises:
            ValueError: Todo.
            TimeoutError: Todo.
        """
        self.log.info("Ensembl Rest API query to get the organism names and associated releases.")
        # FTP query to get the name of the organisms defined in the database, but
        # also to get the latest ensembl release to go through.
        try:
            with requests.Session() as session:
                with session.get(
                    DB.rest_server_api + DB.rest_server_ext,
                    headers={"Content-Type": "application/json"},
                    timeout=(connect_timeout, read_timeout),
                ) as r:
                    if not r.ok:
                        r.raise_for_status()
                    decoded_results = r.json()["species"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as esc:
            raise TimeoutError("Internet connection is slow or server does not respond!") from esc
        # Process the resulting response.
        name_synonyms: list = []
        latest_ensembl_releases: dict = {}
        for organism in decoded_results:
            # Make sure the organisms are as they should be.
            if "core" not in organism["groups"]:
                raise ValueError(
                    f"'{organism['name']}' does not contain 'core' group. "
                    f"This causes many errors when downloading databases in this package."
                )
            if organism["name"] in latest_ensembl_releases:
                raise ValueError(f"'{organism['name']}' has defined more than one latest release.")

            # In very early releases, there are floating releases like "18.2". This package does not support those.
            if not float(organism["release"]) == int(organism["release"]):
                raise ValueError
            # Save the associated the latest release
            latest_ensembl_releases[organism["name"].lower()] = int(organism["release"])
            # Fetch all the organism names associated with the organism.
            synonyms = list(
                set(
                    [organism[j] for j in ["common_name", "name", "taxon_id", "accession", "assembly", "display_name"]]
                    + organism["aliases"]
                )
            )
            # For all the names of a given organism: if it is None, ignore.
            organism_name_synonyms = [(i, organism["name"].lower()) for i in synonyms if i]
            # Also add the organism TaxID as integer.
            organism_name_synonyms += [(int(i), j) for i, j in organism_name_synonyms if i.isdecimal()]
            name_synonyms.extend(organism_name_synonyms)
        name_synonyms_df = pd.DataFrame(name_synonyms, columns=["synonym", "formal_name"])
        #  If it is associated with another organism, mark as 'ambiguous'.
        name_synonyms_df["ambiguous"] = name_synonyms_df.duplicated(subset=["synonym"], keep=False)
        # Convert the result into data frame.
        latest_ensembl_releases_df = pd.DataFrame.from_dict(
            latest_ensembl_releases, orient="index", columns=["ensembl_release"]
        )
        return name_synonyms_df, latest_ensembl_releases_df
