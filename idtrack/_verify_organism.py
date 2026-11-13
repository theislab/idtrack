#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import re
import time

import pandas as pd

from idtrack._db import DB
from idtrack._ensembl_access import rest_get_json


class VerifyOrganism:
    """Resolve a tentative organism identifier to the formal Ensembl species name and its latest supported release.

    The class shields end-users from the quirks of the Ensembl REST payload by converting any synonym—common name,
    scientific name, assembly name or accession, or NCBI taxon ID—into the canonical Ensembl species identifier
    (e.g. ``homo_sapiens``) and the newest Ensembl release that still hosts that species. Matching is
    case-insensitive on both sides. A query equal to a formal name (also written with spaces, e.g.
    ``"Macaca mulatta"``) or to a taxon ID is never ambiguous: a taxon ID that Ensembl also gives to strains or
    breeds resolves to the species-level entry (``9823`` → ``sus_scrofa``). Any other synonym that several species
    share raises an error that lists the candidates, unless idtrack declares it as the alias of one organism
    (:py:attr:`idtrack.DB.organism_aliases`: ``"pig"``, ``"human"``, ``"mouse"``, ``"rhesus"``, ``"cynomolgus"``;
    Ensembl lists ``"pig"`` as the common name of every pig breed).

    The species list comes from one call to the Ensembl REST API per process and day (``/info/species``, see
    :py:func:`idtrack._ensembl_access.rest_get_json` and :py:attr:`idtrack.DB.rest_species_cache_seconds`), made
    only when it is needed: the formal name of an organism
    configured in idtrack (:py:attr:`idtrack.DB.supported_organisms`, also written with spaces) and its declared
    aliases resolve without the network; other synonyms, taxon IDs and the latest release need it.

    After construction the instance offers two high-level helpers::

        >>> resolver = VerifyOrganism("human")
        >>> resolver.get_formal_name()      # 'homo_sapiens'
        >>> resolver.get_latest_release()   # 117  (example)

    Both helpers are backed by two public dataframes, loaded on first use:

    * :py:attr:`~VerifyOrganism.name_synonyms_dataframe` — maps every synonym returned by the REST service to the chosen
      formal name and flags synonyms that are ambiguous across species.
    * :py:attr:`~VerifyOrganism.ensembl_release_dataframe` — one row per Ensembl species (indexed by *formal_name*)
      holding its latest Ensembl release number.

    The names of the assemblies of the organism table (also of earlier assemblies, e.g. ``"GRCh37"``) resolve to their
    organism without the network; they name the organism, not the assembly. A malformed species entry of Ensembl is
    left out with a warning, unless it is one of idtrack's organisms (then it is an error).
    """

    # Species tables per REST URL with the time they were fetched; shared by the instances of a process.
    _species_cache: dict[str, tuple[float, tuple[pd.DataFrame, pd.DataFrame]]] = {}

    def __init__(self, organism_query: str | int):
        """Initialise the resolver; nothing is fetched until a look-up needs the Ensembl species list.

        Args:
            organism_query (str | int): Organism identifier supplied by the user—common name (``"human"``),
                shorthand (``"hsapiens"``), taxon ID (``9606`` or ``"9606"``), assembly name (``"Mmul_10"``), or
                Ensembl species name (``"homo_sapiens"``, ``"Macaca mulatta"``). The value is converted to a
                stripped, lower-case string before processing.
        """
        # Instance attributes
        self.log = logging.getLogger(__name__)
        self.organism_query = self.normalize(organism_query)

    @property
    def name_synonyms_dataframe(self) -> pd.DataFrame:
        """Synonym table of every Ensembl species (see :py:meth:`fetch_organism_and_latest_release`)."""
        return self._species_tables()[0]

    @property
    def ensembl_release_dataframe(self) -> pd.DataFrame:
        """Latest Ensembl release of every species, indexed by formal name."""
        return self._species_tables()[1]

    def _species_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the species tables of this process.

        They are fetched at the first need and again once they are older than
        :py:attr:`idtrack.DB.rest_species_cache_seconds`, so that a long-running process learns of a new release.

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]: See :py:meth:`fetch_organism_and_latest_release`.
        """
        url = DB.rest_server_api + DB.rest_server_ext
        cached = VerifyOrganism._species_cache.get(url)
        if cached is None or time.monotonic() - cached[0] > float(DB.rest_species_cache_seconds):
            cached = (time.monotonic(), self.fetch_organism_and_latest_release())
            VerifyOrganism._species_cache[url] = cached
        return cached[1]

    def _configured_formal_name(self) -> str | None:
        """Return the configured organism the query names without the network, else ``None``.

        The query names a configured organism when it is its formal name (also with spaces or hyphens), one of its
        declared aliases (:py:attr:`idtrack.DB.organism_aliases`: ``"pig"``, ``"human"``; a declared alias wins over
        whatever Ensembl lists under the same word) or the name of one of its assemblies in the organism table
        (``"GRCh37"``, ``"Sscrofa10.2"``, ``"Mmul_8.0.1"``; matched case-insensitively). An assembly name resolves
        the organism only; it does not select that assembly.

        Returns:
            Optional[str]: The organism's formal name, or ``None``.
        """
        for candidate in (self.organism_query, re.sub(r"[\s\-]+", "_", self.organism_query)):
            if candidate in DB.supported_organisms:
                return str(candidate)
        for organism, aliases in DB.organism_aliases.items():
            if self.organism_query in aliases:
                return organism
        owners = sorted(
            organism
            for organism, assemblies in DB.assembly_mysqlport_priority.items()
            if any(self.normalize(entry.get("Name", "")) == self.organism_query for entry in assemblies.values())
        )
        if len(owners) == 1:
            return owners[0]
        return None

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
    def normalize(value: str | int) -> str:
        """Return the matching key of a query or synonym: ``str(value)``, stripped and lower-cased."""
        return str(value).strip().lower()

    def get_formal_name(self) -> str:
        """Resolve the user's organism query to the canonical Ensembl species name.

        Resolution, in this order (all comparisons on normalised, lower-case strings):

        1. the query is a formal name (``"homo_sapiens"``), also with spaces or hyphens for underscores
           (``"Macaca mulatta"``), or an alias that idtrack declares for an organism (``"pig"``); for an organism
           configured in idtrack this needs no network;
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
        configured = self._configured_formal_name()
        if configured is not None:
            return configured

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

    def fetch_organism_and_latest_release(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Query the Ensembl REST API once and build lookup tables for species synonyms and latest releases.

        This internal utility performs a single call to ``/info/species`` on the Ensembl REST server, parses the
        returned JSON, and constructs two pandas dataframes:

        * ``name_synonyms_df`` - one row per (normalised synonym, species), with columns ``synonym``,
            ``formal_name``, ``kind`` (``"formal_name"``, ``"taxon_id"`` or ``"other"``) and ``ambiguous`` (``True``
            if the synonym belongs to more than one species).
        * ``latest_ensembl_releases_df`` - indexed by ``formal_name`` and holding a single ``ensembl_release``
            integer column.

        The request uses the timeouts and retries of :py:func:`idtrack._ensembl_access.rest_get_json`, whose
        :py:class:`idtrack.EnsemblRestError` propagates when Ensembl does not answer
        (:py:attr:`idtrack.DB.connection_timeout`, :py:attr:`idtrack.DB.reading_timeout`,
        :py:attr:`idtrack.DB.rest_attempts`). :py:meth:`_species_tables` keeps the result for the process.

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame]:
                ``(name_synonyms_df, latest_ensembl_releases_df)`` as described above.

        Raises:
            ValueError: If the JSON schema differs from the expected ``{"species": [...]}`` structure or required
                keys are missing.
        """
        self.log.info("Ensembl Rest API query to get the organism names and associated releases.")
        payload = rest_get_json(DB.rest_server_ext, what="the list of species")
        if not isinstance(payload, dict) or not isinstance(payload.get("species"), list):
            raise ValueError(f"Unexpected Ensembl REST payload for {DB.rest_server_ext!r}: no 'species' list.")
        decoded_results = payload["species"]

        # Process the resulting response.
        name_synonyms: list = []
        latest_ensembl_releases: dict = {}

        skipped: list[str] = []
        for organism in decoded_results:
            # An entry that idtrack cannot use: an error for an organism configured in idtrack, otherwise the entry is
            # left out (one odd species among Ensembl's hundreds must not break every look-up).
            name = str(organism.get("name", "")).lower() if isinstance(organism, dict) else ""
            problem = None
            if not name:
                problem = "an entry without a name"
            elif "core" not in (organism.get("groups") or []):
                # 'core' is used in the MySQL or FTP queries by the idtrack package.
                problem = f"`{name}` does not contain 'core' group"
            elif name in latest_ensembl_releases:
                problem = f"`{name}` has defined more than one latest release"
            else:
                try:
                    release_number = float(str(organism.get("release")))
                except ValueError:
                    release_number = float("nan")
                # In very early releases, there are floating releases like "18.2". This package does not support those.
                if not release_number.is_integer():
                    problem = (
                        f"Ensembl release is not in integer format: {organism.get('release')!r} (organism: {name!r})"
                    )
            if problem is not None:
                if name in DB.supported_organisms:
                    raise ValueError(f"Unexpected Ensembl species entry: {problem}.")
                skipped.append(problem)
                continue

            # Save the associated the latest release
            latest_ensembl_releases[name] = int(float(str(organism["release"])))

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

        if skipped:
            self.log.warning(f"{len(skipped)} Ensembl species entries are left out: {'; '.join(skipped[:10])}.")
        name_synonyms_df = pd.DataFrame(name_synonyms, columns=["synonym", "formal_name", "kind"])
        #  If it is associated with another organism, mark as 'ambiguous'.
        name_synonyms_df["ambiguous"] = name_synonyms_df.duplicated(subset=["synonym"], keep=False)
        # Convert the result into data frame.
        latest_ensembl_releases_df = pd.DataFrame.from_dict(
            latest_ensembl_releases, orient="index", columns=["ensembl_release"]
        )

        return name_synonyms_df, latest_ensembl_releases_df
