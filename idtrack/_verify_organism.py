#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging

import pandas as pd
import requests

from idtrack._db import DB


class VerifyOrganism:
    """Resolve a tentative organism identifier to the formal Ensembl species name and its latest supported release.

    The class shields end-users from the quirks of the Ensembl REST payload by converting any synonym—common name,
    scientific name, assembly accession, or NCBI taxon ID—into the canonical Ensembl species identifier
    (e.g. ``homo_sapiens``) and the newest Ensembl release that still hosts that species.  Because the mapping is
    refreshed on every instantiation through a live call to the Ensembl REST API, downstream workflows in *idtrack*
    always rely on up-to-date metadata rather than a possibly stale local cache.

    After construction the instance offers two high-level helpers::

        >>> resolver = VerifyOrganism("human")
        >>> resolver.get_formal_name()      # 'homo_sapiens'
        >>> resolver.get_latest_release()   # 117  (example)

    Both helpers are backed by two public dataframes created during initialisation:

    * :py:attr:`~VerifyOrganism.name_synonyms_dataframe` — maps every synonym returned by the REST service to the chosen
      formal name and flags synonyms that are ambiguous across species.
    * :py:attr:`~VerifyOrganism.ensembl_release_dataframe` — one-row table (indexed by *formal_name*) holding the latest
      Ensembl release number.
    """

    def __init__(self, organism_query: str):
        """Initialise the resolver and pre-fetch synonym/release tables from the Ensembl REST API.

        The constructor immediately invokes :py:meth:`~VerifyOrganism.fetch_organism_and_latest_release`, downloading
        the complete species list from ``{DB.rest_server_api}{DB.rest_server_ext}`` so that all subsequent look-ups run
        entirely in-memory.  Any exceptions raised during that fetch are allowed to propagate unchanged so that callers
        can handle network or data-quality issues explicitly.

        Args:
            organism_query (str): Organism identifier supplied by the user—common name (``"human"``),
                shorthand (``"hsapiens"``), taxon ID (``9606``), or fully qualified Ensembl species name
                (``"homo_sapiens"``).  The value is converted to lower case before processing.
        """
        # Instance attributes
        self.log = logging.getLogger("verify_organism")

        # Fetch the latest information from the Ensembl resources regarding
        self.organism_query = organism_query.lower()
        self.name_synonyms_dataframe, self.ensembl_release_dataframe = self.fetch_organism_and_latest_release(
            connect_timeout=DB.connection_timeout, read_timeout=DB.reading_timeout
        )

    def get_latest_release(self) -> int:
        """Return the latest Ensembl release number associated with the queried organism.

        This helper calls :py:meth:`~VerifyOrganism.get_formal_name` to resolve the user-supplied organism query
        to the canonical Ensembl species name, then looks up that key in the dataframe prepared at instantiation
        time.  Down-stream routines (e.g. database connectors, file download helpers) rely on this value to decide
        which Ensembl release to fetch, ensuring the entire pipeline stays on a single, internally consistent
        genome build.

        Returns:
            int: The most recent Ensembl release available for the resolved organism.
        """
        formal_name = self.get_formal_name()
        return int(self.ensembl_release_dataframe.loc[formal_name]["ensembl_release"])

    def get_formal_name(self) -> str:
        """Resolve the user's organism query to the canonical Ensembl species name.

        The method performs an exact match against the *synonym* column of
        :py:data:`~VerifyOrganism.name_synonyms_dataframe`, which was pre-populated from the Ensembl REST *species*
        endpoint.  Synonyms include scientific names, common names, NCBI TaxIDs, assembly accessions and other
        aliases, allowing flexible user input while guaranteeing that only one formally recognised organism is
        selected before any expensive data retrieval begins.

        Returns:
            str: The canonical Ensembl species identifier (always lower-case, e.g. ``"homo_sapiens"``).

        Raises:
            KeyError: If the query string does not match any synonym in the dataframe.
            ValueError: If the query matches more than one formal name, indicating an ambiguous synonym.
        """
        the_search = self.name_synonyms_dataframe[self.name_synonyms_dataframe["synonym"] == self.organism_query]
        cm = "Please inspect 'name_synonyms_dataframe' variable of this instance for troubleshooting."
        if len(the_search) == 0:
            raise KeyError(f"The query (`{self.organism_query}`) is not defined in the Ensembl database. {cm}")
        elif len(the_search) > 1:
            raise ValueError(
                f"There are multiple entries for query (`{self.organism_query}`) "
                f" in Ensembl database. The query is hence ambiguous. {cm}"
            )
        else:
            return self.name_synonyms_dataframe[self.name_synonyms_dataframe["synonym"] == self.organism_query][
                "formal_name"
            ].iloc[0]

    def fetch_organism_and_latest_release(
        self, connect_timeout: int, read_timeout: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Query the Ensembl REST API once and build lookup tables for species synonyms and latest releases.

        This internal utility performs a single call to ``/info/species`` on the Ensembl REST server, parses the
        returned JSON, and constructs two pandas dataframes:

        * ``name_synonyms_df`` - one row per synonym, with columns ``synonym``, ``formal_name`` and ``ambiguous``
            (``True`` if the synonym belongs to more than one species).
        * ``latest_ensembl_releases_df`` - indexed by ``formal_name`` and holding a single ``ensembl_release``
            integer column.

        Consolidating the REST query in one place avoids repeated network traffic and provides a cache-friendly
        structure for subsequent lookups.

        Args:
            connect_timeout (int): Seconds to wait while establishing the TCP connection to the Ensembl server.
            read_timeout (int): Seconds to wait for the server to send the full response after the connection
                has been established.

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]:
                ``(name_synonyms_df, latest_ensembl_releases_df)`` as described above.

        Raises:
            TimeoutError: If the combined *(connect_timeout, read_timeout)* limit is exceeded.
            ValueError: If the JSON schema differs from the expected ``{"species": [...]}`` structure or required
                keys are missing.
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
                    f"`{organism['name']}` does not contain 'core' group. "
                    f"This causes many errors when downloading databases in this package."
                )
                # Note that 'core' is used in the MySQL or FTP queries by the idtrack package.

            # Make sure there is one and only one item for a given organism.
            if organism["name"] in latest_ensembl_releases:
                raise ValueError(f"`{organism['name']}` has defined more than one latest release.")

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
