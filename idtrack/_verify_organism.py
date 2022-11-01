#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
from typing import Tuple

import pandas as pd
import requests

from ._db import DB


class VerifyOrganism:
    """Finds the formal organism name and associated latest Ensembl release given a tentative organism name.

    This class is designed to aid the user regarding the organism name accepted by the program and latest
    Ensembl release possible without manually looking online resources. Every time of use, the class uses Ensembl
    REST API to get the information regarding the organisms and associated Ensembl releases, so the class
    requires an active Internet connection.
    """

    def __init__(self, organism_query: str):
        """Class initialization.

        Args:
            organism_query: Organism of interest.
                This does not have to be formal name. For example, 'human', 'hsapiens' as well as formal name
                'homo_sapiens' is accepted by the program.
        """
        # Instance attributes
        self.log = logging.getLogger("verify_organism")

        # Fetch the latest information from the Ensembl resources regarding
        self.organism_query = organism_query.lower()
        self.name_synonyms_dataframe, self.ensembl_release_dataframe = self.fetch_organism_and_latest_release(
            DB.connection_timeout, DB.reading_timeout
        )

    def get_latest_release(self) -> int:
        """Using the API response and queried organism name, find the latest Ensembl release to be used for the program.

        Returns:
            Latest Ensembl release.
        """
        formal_name = self.get_formal_name()
        return int(self.ensembl_release_dataframe.loc[formal_name]["ensembl_release"])

    def get_formal_name(self) -> str:
        """Using the API response and queried organism name, find the formal organism name to be used for the program.

        Returns:
            Formal organism name.

        Raises:
            KeyError: If the query is not defined in the pre-created dataframes.
            ValueError: If there are multiple entries for query and hence the query is ambiguous."
        """
        the_search = self.name_synonyms_dataframe[self.name_synonyms_dataframe["synonym"] == self.organism_query]
        cm = "Please inspect 'name_synonyms_dataframe' variable of this instance for troubleshooting."
        if len(the_search) == 0:
            raise KeyError(f"The query ('{self.organism_query}') is not defined in the Ensembl database. {cm}")
        elif len(the_search) > 1:
            raise ValueError(
                f"There are multiple entries for query ('{self.organism_query}') "
                f" in Ensembl database. The query is hence ambiguous. {cm}"
            )
        else:
            return self.name_synonyms_dataframe[self.name_synonyms_dataframe["synonym"] == self.organism_query][
                "formal_name"
            ].iloc[0]

    def fetch_organism_and_latest_release(
        self, connect_timeout: int, read_timeout: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Creates a query and process the data to construct two dataframes for Ensembl release and organism names.

        The query to get the name of the organisms defined in the database, but also to get the latest
        ensembl release to go through. This method is actually main functionality of the class; in general uses, the
        user is not expected to use this method directly.

        Args:
            connect_timeout: The number of seconds the requests will wait to establish a connection to a remote machine.
            read_timeout: The number of seconds the client will wait for the server to send a response

        Returns:
            Two dataframes to be used by the instance methods later on.

        Raises:
            ValueError: Unexpected error.
            TimeoutError: If allowed time is passed.
        """
        self.log.info("Ensembl Rest API query to get the organism names and associated releases.")

        try:
            with requests.Session() as session:  # Create a session
                with session.get(
                    DB.rest_server_api + DB.rest_server_ext,
                    headers={"Content-Type": "application/json"},
                    timeout=(connect_timeout, read_timeout),
                ) as r:  # Connect to the REST API.
                    if not r.ok:
                        # If everything is not fine regarding the response, raise an error.
                        r.raise_for_status()
                    decoded_results = r.json()["species"]  # Otherwise decode the json into a dictionary.

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as esc:
            # Raise an exception if it takes too long to get the data from Ensembl REST API.
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
                # Note that 'core' is used in the MySQL or FTP queries by the idtrack package.

            # Make sure there is one and only one item for a given organism.
            if organism["name"] in latest_ensembl_releases:
                raise ValueError(f"'{organism['name']}' has defined more than one latest release.")

            # In very early releases, there are floating releases like "18.2". This package does not support those.
            if not float(organism["release"]) == int(organism["release"]):
                raise ValueError("Ensembl release is not in integer format. {}")

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
