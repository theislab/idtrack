#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import re
from typing import Union

import pandas as pd
import requests

from idtrack._db import DB


class VerifyOrganism:
    """Resolve a tentative organism identifier to the formal Ensembl species name and its latest supported release.

    The class shields end-users from the quirks of the Ensembl REST payload by converting any synonym—common name,
    scientific name, assembly name or accession, or NCBI taxon ID—into the canonical Ensembl species identifier
    (e.g. ``homo_sapiens``) and the newest Ensembl release that still hosts that species. Matching is
    case-insensitive on both sides. A query equal to a formal name (also written with spaces, e.g.
    ``"Macaca mulatta"``) or to a taxon ID is never ambiguous: a taxon ID that Ensembl also gives to strains or
    breeds resolves to the species-level entry (``9823`` → ``sus_scrofa``). Any other synonym that several species
    share (``"pig"`` is the common name of every pig breed) raises an error that lists the candidates.  Because the mapping is
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

    def __init__(self, organism_query: Union[str, int]):
        """Initialise the resolver and pre-fetch synonym/release tables from the Ensembl REST API.

        The constructor immediately invokes :py:meth:`~VerifyOrganism.fetch_organism_and_latest_release`, downloading
        the complete species list from ``{DB.rest_server_api}{DB.rest_server_ext}`` so that all subsequent look-ups run
        entirely in-memory.  Any exceptions raised during that fetch are allowed to propagate unchanged so that callers
        can handle network or data-quality issues explicitly.

        Args:
            organism_query (str | int): Organism identifier supplied by the user—common name (``"human"``),
                shorthand (``"hsapiens"``), taxon ID (``9606`` or ``"9606"``), assembly name (``"Mmul_10"``), or
                Ensembl species name (``"homo_sapiens"``, ``"Macaca mulatta"``). The value is converted to a
                stripped, lower-case string before processing.
        """
        # Instance attributes
        self.log = logging.getLogger("verify_organism")

        # Fetch the latest information from the Ensembl resources regarding
        self.organism_query = self.normalize(organism_query)
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

    @staticmethod
    def normalize(value: Union[str, int]) -> str:
        """Return the matching key of a query or synonym: ``str(value)``, stripped and lower-cased."""
        return str(value).strip().lower()

    def get_formal_name(self) -> str:
        """Resolve the user's organism query to the canonical Ensembl species name.

        Resolution, in this order (all comparisons on normalised, lower-case strings):

        1. the query is a formal name (``"homo_sapiens"``), also with spaces or hyphens for underscores
           (``"Macaca mulatta"``);
        2. the query is a taxon ID: the organism with that ID, or—when Ensembl gives the ID to several strains or
           breeds—the species-level entry (the formal name of two words, ``sus_scrofa`` for ``9823``);
        3. any other synonym of :py:data:`~VerifyOrganism.name_synonyms_dataframe` (common names, display names,
           assembly names and accessions, aliases), which must belong to exactly one organism.

        Returns:
            str: The canonical Ensembl species identifier (always lower-case, e.g. ``"homo_sapiens"``).

        Raises:
            KeyError: If the query does not match any synonym.
            ValueError: If the query matches more than one organism (an ambiguous synonym); the message lists the
                candidates.
        """
        query = self.organism_query
        cm = "Please inspect 'name_synonyms_dataframe' variable of this instance for troubleshooting."
        formal_names = self.ensembl_release_dataframe.index
        for candidate in (query, re.sub(r"[\s\-]+", "_", query)):
            if candidate in formal_names:
                return str(candidate)

        df = self.name_synonyms_dataframe
        matches = df[df["synonym"] == query]
        if len(matches) == 0:
            raise KeyError(f"The query (`{query}`) is not defined in the Ensembl database. {cm}")

        candidates = sorted(set(matches["formal_name"]))
        if len(candidates) == 1:
            return str(candidates[0])

        if query.isdecimal() and (matches["kind"] == "taxon_id").any():
            taxon_candidates = sorted(set(matches.loc[matches["kind"] == "taxon_id", "formal_name"]))
            species_level = [c for c in taxon_candidates if len(c.split("_")) == 2]
            if len(species_level) == 1:
                return str(species_level[0])

        shown = ", ".join(candidates[:25]) + (f", ... ({len(candidates)} in total)" if len(candidates) > 25 else "")
        raise ValueError(
            f"There are multiple entries for query (`{query}`) in Ensembl database. The query is hence ambiguous. "
            f"Candidates: {shown}. Use the Ensembl species name (e.g. `{candidates[0]}`) or the taxon ID instead."
        )

    def fetch_organism_and_latest_release(
        self, connect_timeout: int, read_timeout: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Query the Ensembl REST API once and build lookup tables for species synonyms and latest releases.

        This internal utility performs a single call to ``/info/species`` on the Ensembl REST server, parses the
        returned JSON, and constructs two pandas dataframes:

        * ``name_synonyms_df`` - one row per (normalised synonym, species), with columns ``synonym``,
            ``formal_name``, ``kind`` (``"formal_name"``, ``"taxon_id"`` or ``"other"``) and ``ambiguous`` (``True``
            if the synonym belongs to more than one species).
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
                raise ValueError(
                    f"Ensembl release is not in integer format: {organism['release']!r} "
                    f"(organism: {organism['name']!r})"
                )

            # Save the associated the latest release
            latest_ensembl_releases[organism["name"].lower()] = int(organism["release"])

            # Fetch all the organism names associated with the organism; match on normalised strings.
            formal_name = organism["name"].lower()
            kinds: dict[str, str] = {}
            for field in ["common_name", "accession", "assembly", "display_name"]:
                if organism.get(field):
                    kinds.setdefault(self.normalize(organism[field]), "other")
            for alias in organism.get("aliases") or []:
                if alias:
                    kinds.setdefault(self.normalize(alias), "other")
            if organism.get("taxon_id"):
                kinds[self.normalize(organism["taxon_id"])] = "taxon_id"
            kinds[formal_name] = "formal_name"
            name_synonyms.extend((synonym, formal_name, kind) for synonym, kind in kinds.items())

        name_synonyms_df = pd.DataFrame(name_synonyms, columns=["synonym", "formal_name", "kind"])
        #  If it is associated with another organism, mark as 'ambiguous'.
        name_synonyms_df["ambiguous"] = name_synonyms_df.duplicated(subset=["synonym"], keep=False)
        # Convert the result into data frame.
        latest_ensembl_releases_df = pd.DataFrame.from_dict(
            latest_ensembl_releases, orient="index", columns=["ensembl_release"]
        )

        return name_synonyms_df, latest_ensembl_releases_df
