#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""The manager of an organism's Ensembl tables: download (MySQL or HTTPS), store, derive what a graph needs."""

import bz2
import copy
import gzip
import hashlib
import io
import json
import logging
import os
import re
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from contextlib import contextmanager
from functools import cached_property
from typing import Any, SupportsIndex

import numpy as np
import pandas as pd
import pymysql

import idtrack.build.tables as hs
from idtrack import _arguments, ids, organisms, reading, settings
from idtrack._atomic import write_text
from idtrack.build import assembly_chain, continuity
from idtrack.build.ensembl_access import (
    EnsemblDownloadError,
    ensembl_access_mode,
    https_get,
    reachability,
    rest_get_json,
)
from idtrack.build.externals import ExternalDatabases
from idtrack.build.mysql_dump import read_mysql_dump
from idtrack.errors import (
    BuildError,
    EnsemblRestError,
    EnsemblSchemaError,
    ExternalsError,
    IntegrityError,
    InvalidArgumentError,
    TableFormatError,
    UnknownAssemblyError,
    UnknownOrganismError,
    UnknownReleaseError,
)
from idtrack.graph.genes import CHROMOSOME, MITOCHONDRIAL, SCAFFOLD

#: The code of the attribute by which Ensembl's core databases mark a sequence region that is not part of the
#: reference sequence of an assembly (``attrib_type.code``: an alternative locus, an assembly patch or an LRG record).
NON_REFERENCE_ATTRIBUTE = "non_ref"
#: Ensembl's names that say what a sequence region is (:py:attr:`idtrack.graph.types.Place.kind`): the coordinate
#: system of the chromosomes (``coord_system.name``), the attribute by which Ensembl ranks a region in the organism's
#: karyotype (the chromosomes of an assembly loaded with one level of sequences, whose coordinate system is
#: ``primary_assembly``), and the attribute of a region translated with an alternate genetic code (in a vertebrate
#: core database, the mitochondrial genome alone).
CHROMOSOME_COORD_SYSTEM = "chromosome"
KARYOTYPE_ATTRIBUTE = "karyotype_rank"
MITOCHONDRIAL_ATTRIBUTE = "codon_table"

# What two external databases' entries mean. The cross-reference tables idtrack reads (``external_db``, ``xref``,
# ``object_xref``) say which database an entry belongs to, not whether its labels are family names or whether an entry
# stands for one gene; these are the definitions of the databases themselves (Rfam: families of RNA; miRBase: one
# precursor hairpin per entry), kept here, in one place, for the databases the shipped organisms select. Measured
# effects: docs/changes.rst ("What the graph holds of the external databases"). Any other database is read as Ensembl
# gives it.

#: Databases whose labels name a gene only where the label is the gene's display name (``gene.display_xref_id``): the
#: label of a family (``5S_rRNA``, ``U6``, ``SNORD115``) is Ensembl's name of the genes that have no better name, not a
#: name of every member of the family. They are the databases of families (:py:data:`idtrack.settings.FAMILY_DATABASES`),
#: whose accessions are no bridge between genes either.
DISPLAY_LABEL_ONLY: frozenset[str] = settings.FAMILY_DATABASES

#: Databases whose identifiers are kept only on genes of the given biotypes: a miRBase name is the name of the miRNA
#: gene, not of the protein-coding gene that hosts it.
BIOTYPE_RESTRICTED: dict[str, frozenset[str]] = {"miRBase": frozenset({"miRNA"})}

#: Databases whose entries link a gene only where Ensembl gives the entry to one gene over all the releases of a graph:
#: a miRBase entry that Ensembl attaches to genes at several loci, at once (copies of one precursor at several loci)
#: or one after the other (an accession moved to a gene at another locus), is no evidence of which gene it is; as a
#: link it ties those genes, and carries retired ids and old names to the others. An entry is left out only where the
#: graph shows two loci, by the package's one rule (:py:func:`idtrack.graph.genes.count_genes`): two genes that live
#: at one release at places that do not overlap (a split of one id into two that live side by side is two genes), or
#: genes one after the other at places apart on one assembly or on chromosomes of different names; ids that Ensembl's
#: history carried one to the other where they follow one another, or whose places overlap, are one gene, a copy
#: outside the reference sequence adds none, and holders on two assemblies that nothing compares (no history link, no
#: comparable place) are no evidence of two loci (:py:meth:`idtrack.build.GraphMaker.construct_graph`). A link on a
#: transcript or a translation counts for the gene that holds it at the releases of the link.
ONE_GENE_ONLY: frozenset[str] = frozenset({"miRBase"})

#: Ensembl files the names it derives from an external database for genes under a database of their own, named after
#: it: ``<db_name>_gene_name`` (``miRBase_gene_name``: the names of miRBase that Ensembl displays on genes at releases 61
#: and 62, ``RFAM_gene_name``). Such a name is a name of the database it is derived from, and that database's
#: definitions (:py:data:`BIOTYPE_RESTRICTED`, :py:data:`ONE_GENE_ONLY`) hold for it whatever name it is filed under
#: (:py:func:`rules_database`). Ensembl's names of transcripts (``<db_name>_trans_name``: the gene's name and a number,
#: ``FZD5-201``) are Ensembl's own names, not names of that database.
DERIVED_GENE_NAME_SUFFIX = "_gene_name"


def rules_database(database: str) -> str:
    """Return the database whose definitions hold for a database's rows: itself, or the one its gene names come from.

    Args:
        database: A database's ``db_name``.

    Returns:
        str: ``miRBase`` for ``miRBase_gene_name`` (:py:data:`DERIVED_GENE_NAME_SUFFIX`); the name itself otherwise.
    """
    if database.endswith(DERIVED_GENE_NAME_SUFFIX) and len(database) > len(DERIVED_GENE_NAME_SUFFIX):
        return database[: -len(DERIVED_GENE_NAME_SUFFIX)]
    return database


def entries_of_several_genes(
    links: Iterable[tuple[str, str]], count_genes: Callable[[Collection[str]], int]
) -> set[str]:
    """Return the entries that name more than one gene (the rule of :py:data:`ONE_GENE_ONLY`).

    Args:
        links: ``(entry, Ensembl stable gene id without version)`` for every link of the judged databases over every
            release and assembly of a graph (accessions, labels and synonyms together, as the graph names its nodes).
        count_genes: The number of genes some stable ids name (:py:func:`idtrack.graph.genes.count_genes` with the
            graph's places, marks and history).

    Returns:
        set[str]: The entries, in lower case (the graph keeps one node for spellings that differ in case only), whose
        stable ids are more than one gene.
    """
    genes: dict[str, set[str]] = {}
    for entry, stable_id in links:
        genes.setdefault(entry.lower(), set()).add(stable_id)
    return {entry for entry, of in genes.items() if len(of) > 1 and count_genes(of) > 1}


#: Columns of the ``gene`` table that say what a gene is and where it lies.
GENE_ANNOTATION_COLUMNS = [
    "gene_id",
    "biotype",
    "display_xref_id",
    "seq_region_id",
    "seq_region_start",
    "seq_region_end",
    "seq_region_strand",
]

#: The columns of Ensembl's ``assembly_exception`` table that idtrack reads (:py:meth:`DatabaseManager.create_assembly_exceptions`).
ASSEMBLY_EXCEPTION_COLUMNS = [
    "seq_region_id",
    "seq_region_start",
    "seq_region_end",
    "exc_type",
    "exc_seq_region_id",
    "exc_seq_region_start",
    "exc_seq_region_end",
]

#: The columns of Ensembl's ``xref`` table that idtrack reads: the entry, its database, its accession and label, and the
#: version of the accession that the cross reference attaches (``xref.version``; ``0`` where the database gives none).
XREF_COLUMNS = ["xref_id", "external_db_id", "dbprimary_acc", "display_label", "version"]

#: Version of the rules that derive the tables of cross references from Ensembl's raw tables; a stored table derived
#: under other rules is derived again.
EXTERNAL_RULES_VERSION = 10

_ALPHANUMERIC = r"[0-9A-Za-z]"
# UniProt writes the evidence of a name inside the name field as ``{ECO:<evidence code>|<source>}`` (the Evidence and
# Conclusion Ontology tag of UniProt's flat files); a synonym that carries it is evidence text, not a name.
_EVIDENCE_TEXT = "{ECO:"


def _in_total_order(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of a table of cross references in a total order (database, Ensembl id, identifier, the rest).

    The order does not depend on the order of the input rows; the identities are compared as float64 (pandas cannot
    sort by float16 columns).
    """
    identities = ["ensembl_identity", "xref_identity"]
    keys = ["name_db", "graph_id", "id_db", *identities, "version", "is_label", "is_display"]
    rows = rows.astype({identities[0]: np.float64, identities[1]: np.float64})
    rows = rows.sort_values(by=[k for k in keys if k in rows.columns], kind="stable", na_position="last")
    return rows.astype({identities[0]: np.float16, identities[1]: np.float16}).reset_index(drop=True)


def label_names_of_databases(rows: Iterable[tuple[int, str, str | None, frozenset[str]]]) -> dict[str, str]:
    """Return the one name that every database's label rows take at every release of a window.

    One database name means one kind of identifier at every release. Ensembl names a database's label rows by its
    display name (``external_db.db_display_name``), which changes over the releases for some databases and is, at some
    releases, also the name of a database's accessions (a ``db_name`` of the release: the labels and the accessions of
    ``EntrezGene`` share that name up to release 87, ``RFAM``'s always). The labels of a database are named, at every
    release, by its display name at the newest release of the window at which that display name is not the name of
    accessions; a database whose display name always is takes ``<db_name> label``.

    Args:
        rows: ``(release, db_name, db_display_name, the db_name of every database of that release)`` for every
            database of every release (and assembly) of the window.

    Returns:
        dict[str, str]: ``{db_name: name of its label rows}``.
    """
    newest: dict[str, tuple[int, str]] = {}
    names: set[str] = set()
    for release, name, display, names_of_release in rows:
        names.add(name)
        if display is not None and display not in names_of_release:
            if name not in newest or int(release) >= newest[name][0]:
                newest[name] = (int(release), display)
    return {name: newest[name][1] if name in newest else label_name(name) for name in names}


def label_name(database: str, names: Mapping[str, str] | None = None) -> str:
    """Return the name of a database's label rows: its name in ``names``, else ``<database> label``.

    Args:
        database: The database's ``db_name``.
        names: The names of the label rows of a window (:py:func:`label_names_of_databases`); a database that no
            ``external_db`` of the window has is not in it.

    Returns:
        str: The name of the database's label rows.
    """
    if names is not None and database in names:
        return names[database]
    return f"{database} label"


def _read_values(values: pd.Series) -> pd.Series:
    """Return external values read as a typed identifier is read, a synonym's prefix kept.

    The reading is :py:func:`idtrack.reading.normalise_unicode`, then :py:func:`idtrack.reading.strip_punctuation`.

    Args:
        values: The values, synonyms with :py:data:`idtrack.ids.SYNONYM_PREFIX`.

    Returns:
        pandas.Series: The values read; the same object when none changes.
    """
    prefix = ids.SYNONYM_PREFIX
    changed: dict[str, str] = {}
    for value in pd.unique(values.astype(object)).tolist():
        text = str(value)
        head = prefix if text.startswith(prefix) else ""
        read = head + reading.strip_punctuation(reading.normalise_unicode(text[len(head) :]))
        if read != text:
            changed[text] = read
    if not changed:
        return values
    return values.astype(object).map(lambda text: changed.get(text, text)).astype(values.dtype)


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
    * **Transparent caching** - expensive downloads happen once; every table is kept in a file of its own in
      the local repository (:py:attr:`tables_directory`), written atomically and named by everything it depends on
      (release, form, externals selection), so a table computed for other parameters is never
      served.
    * **Versioned ids only** - idtrack follows ``<id>.<version>`` through the releases. A release whose Ensembl
      ids carry no versions (some organisms, e.g. *S. cerevisiae*) is refused where its id table is read
      (:py:meth:`~DatabaseManager.create_ids`), with an error that says so: no table is made up for it.

    Key public methods/attributes
    -----------------------------
    * :py:meth:`available_releases` — list releases of the assembly within the window.
    * :py:meth:`change_release` — a new manager for another Ensembl release (likewise :py:meth:`change_form`,
      :py:meth:`change_assembly`, :py:meth:`change_release_auto_assembly`).
    * :py:meth:`get_table` — a raw Ensembl table (stored after the first download); :py:meth:`download_table`
      fetches one without storing it.
    * :py:meth:`get_db` — a derived table (ids, history, relations, cross references).
    * :py:meth:`create_external_all` — the cross references of every assembly the externals YAML enables for the
      release.
    * :py:attr:`organism`, :py:attr:`form`, :py:attr:`release`,
      :py:attr:`assembly` — core configuration knobs, surfaced for quick inspection.

    A manager's configuration does not change after construction; the ``change_*`` methods return new managers.
    Use the built-in :py:meth:`__str__` for a concise, human-readable dump of that state.
    """

    # Keyed by (organism, assembly key, organisms.table_version()) so that `organisms.register_organism` invalidates it.
    _core_db_index_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    # Keyed by (organism, first release, snapshot release, local repository): :py:meth:`label_names`.
    _label_names_cache: dict[tuple[str, int, int | None, str], dict[str, str]] = {}
    # The filing of the selected databases (decision 104), per organism, window, repository and selection.
    _filing_cache: dict[tuple, continuity.Filing] = {}
    _ftp_schema_cache: dict[str, dict[str, list[tuple[str, str]]]] = {}
    # (latest release, time.monotonic() of the answer)
    _ensembl_latest_release_cache: tuple[int, float] | None = None
    # Schema-dump URL resolved per dump directory (one directory listing per database at most).
    _ftp_schema_url_cache: dict[str, str] = {}

    # pymysql error codes that mean the server or the route to it failed (not the query): can't connect,
    # server has gone away, lost connection during query, lost connection at handshake.
    _MYSQL_CONNECTION_ERROR_CODES = frozenset({2003, 2006, 2013, 2055})

    @classmethod
    def _is_mysql_connection_failure(cls, exc: BaseException) -> bool:
        """Return whether *exc* (or its cause) means the MySQL server could not be reached or dropped the link."""
        for err in cls._iter_exception_chain(exc):
            if isinstance(err, (pymysql.err.InterfaceError, OSError)):
                return True
            if isinstance(err, pymysql.err.OperationalError) and err.args:
                try:
                    if int(err.args[0]) in cls._MYSQL_CONNECTION_ERROR_CODES:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    @staticmethod
    def _release_window(organism: str, assembly: int) -> tuple[int, int | None]:
        """Return the releases an assembly may have: its ``ReleaseRange`` clipped to idtrack's floor (48).

        Both discovery paths (the MySQL probe and the HTTPS scan) accept only releases inside this window, so they
        agree on what an assembly is even where two assemblies share a core-database suffix.

        Args:
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.

        Returns:
            tuple[int, Optional[int]]: ``(first, last)``; ``last`` is ``None`` for an open-ended assembly.
        """
        first, last = organisms.release_range(organism, int(assembly))
        floor = int(min(settings.ENSEMBL.mysql_port_min_release.values()))
        return max(int(first), floor), last

    @classmethod
    def _ensembl_latest_release(cls) -> int:
        """Return Ensembl's latest release, asked from REST once per :py:attr:`idtrack.settings.ENSEMBL.rest_species_cache_seconds`.

        The answer expires (one day by default) so that a long-running process learns of a new release. If Ensembl
        REST does not answer (after its retries), :py:class:`idtrack.EnsemblRestError` propagates.

        Returns:
            int: The latest release.

        Raises:
            ValueError: If the answer is not the expected payload.
        """
        cached = cls._ensembl_latest_release_cache
        if cached is not None and time.monotonic() - cached[1] < float(settings.ENSEMBL.rest_species_cache_seconds):
            return int(cached[0])

        url = f"{settings.ENSEMBL.rest_server}/info/data"
        payload = rest_get_json("/info/data", what="the latest Ensembl release")
        releases = payload.get("releases") if isinstance(payload, dict) else None
        if not isinstance(releases, list) or not releases:
            raise EnsemblRestError(f"Unexpected Ensembl REST payload at {url!r}: expected non-empty 'releases' list.")

        latest = max(int(r) for r in releases)
        cls._ensembl_latest_release_cache = (int(latest), time.monotonic())
        return int(latest)

    @staticmethod
    def _parse_apache_dir_listing_dirs(html: str) -> list[str]:
        """Extract directory names from an Apache-style directory listing HTML page."""
        # Keep this dependency-free (no BeautifulSoup). Directory links end with '/'.
        dirs = []
        for href in re.findall(r"href=\"([^\"]+/)\"", html, flags=re.IGNORECASE):
            if href.startswith("?") or href.startswith("/"):
                continue
            name = href.rstrip("/")
            if name and name not in (".", ".."):
                dirs.append(name)
        return dirs

    @staticmethod
    def _parse_apache_dir_listing_files(html: str) -> list[str]:
        """Extract file names from an Apache-style directory listing HTML page."""
        files: list[str] = []
        for href in re.findall(r"href=\"([^\"]+)\"", html, flags=re.IGNORECASE):
            if href.startswith("?") or href.startswith("/") or href.endswith("/"):
                continue
            if href and href not in (".", ".."):
                files.append(href)
        return files

    @staticmethod
    def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
        """Return the exception chain (`__cause__`/`__context__`) as a list."""
        chain: list[BaseException] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return chain

    @classmethod
    def _is_retryable_http_read_error(cls, exc: BaseException) -> bool:
        """Heuristically decide whether an HTTP read/decompress error is transient and safe to retry."""
        retryable_names = {
            # urllib3 / http.client
            "ProtocolError",
            "IncompleteRead",
            "ReadTimeoutError",
            "RemoteDisconnected",
            # requests
            "ChunkedEncodingError",
            "ConnectionError",
            "ReadTimeout",
            "ConnectTimeout",
            "SSLError",
            # compression / streaming
            "BadGzipFile",
            "EOFError",
            "OSError",
            "error",  # e.g. zlib.error
        }

        for err in cls._iter_exception_chain(exc):
            if err.__class__.__name__ in retryable_names:
                return True
        return False

    @classmethod
    def _ftp_mysql_root_candidates(cls, *, organism: str, assembly: int, release: int) -> tuple[str, ...]:
        """Return candidate HTTPS roots to search for an Ensembl core DB directory.

        An assembly that Ensembl keeps beside its successor (the organism table gives it a release at which an assembly
        of higher priority is the organism's primary one: GRCh37 from release 76) is served from an archive folder of
        its own, named by the assembly in lower case (``/pub/grch37/release-<r>/mysql/``), which is searched first.
        """
        host = settings.ENSEMBL.https_host
        base = f"https://{host}/pub/release-{int(release)}/mysql/"
        try:
            beside = organisms.primary_assembly(organism, int(release)) != int(assembly)
        except UnknownReleaseError:
            beside = False
        name = organisms.assembly_name(organism, int(assembly)) if beside else None
        if name:
            return (f"https://{host}/pub/{name.lower()}/release-{int(release)}/mysql/", base)
        return (base,)

    @classmethod
    def _ftp_find_core_db_dir(cls, *, organism: str, assembly: int, release: int) -> tuple[str | None, str | None]:
        """Locate the core DB directory name (may include patch letters) and its HTTPS directory URL.

        The exact name is probed first; if Ensembl answers that it does not exist (HTTP 404), the release's
        directory is listed to find a name with patch letters (e.g. ``homo_sapiens_core_54_36p``). If Ensembl does
        not answer (after the retries of :py:func:`idtrack.build.ensembl_access.https_get`), its
        :py:class:`idtrack.EnsemblDownloadError` propagates: "unknown" is never taken for "absent".

        Args:
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.
            release: Ensembl release.

        Returns:
            tuple[Optional[str], Optional[str]]: ``(name, url)``, or ``(None, None)`` when Ensembl answered that the
            release has no such database.
        """
        release = int(release)
        assembly = int(assembly)
        expected = f"{organism}_core_{release}_{organisms.db_suffix(organism, assembly)}"
        pattern = re.compile(rf"^{re.escape(expected)}[a-z]*$")

        for root in cls._ftp_mysql_root_candidates(organism=organism, assembly=assembly, release=release):
            # Fast path: exact directory (no patch letter)
            resp = https_get(f"{root}{expected}/", what=f"the dump directory of {expected}", stream=True)
            if resp is not None:
                resp.close()
                return expected, f"{root}{expected}/"

            # Slow path: list and look for patch-letter variants (e.g. `_36p`).
            listing = https_get(root, what=f"the dump listing of release {release}")
            if listing is None:
                continue  # Ensembl has no such release directory under this root.
            matches = [d for d in cls._parse_apache_dir_listing_dirs(listing.text) if pattern.match(d)]
            if matches:
                db_name = min(matches, key=lambda d: (len(d), d))
                return db_name, f"{root}{db_name}/"

        return None, None

    @classmethod
    def _get_core_db_index_from_ftp(cls, *, organism: str, assembly: int, ports: list[int]) -> dict[str, Any]:
        """Build a core-DB availability index by probing the Ensembl HTTPS/FTP MySQL dumps.

        Every release of the assembly's window (:py:meth:`_release_window`, up to Ensembl's latest release) is looked
        up with :py:meth:`_ftp_find_core_db_dir`. A release is left out only when Ensembl answered that it has no
        core database of this assembly; a request that fails after its retries raises, so an incomplete index is
        never built (and never cached or saved). An assembly that has no release at all gives an index with an empty
        release list. A request that stays unanswered raises
        :py:class:`idtrack.EnsemblDownloadError`; if the latest Ensembl release cannot be learnt,
        :py:class:`idtrack.EnsemblRestError` propagates.

        Args:
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly (releases are assigned to them).

        Returns:
            dict[str, Any]: The index (see :py:meth:`_get_core_db_index`), with ``scanned_up_to``.
        """
        log = logging.getLogger(__name__)
        latest_release = cls._ensembl_latest_release()
        min_release, range_max = cls._release_window(organism, int(assembly))
        max_release = min(int(range_max) if range_max is not None else int(latest_release), int(latest_release))
        log.info(
            f"Scanning the HTTPS dumps of {organism} assembly {assembly} "
            f"(core suffix {organisms.db_suffix(organism, int(assembly))!r}) for releases {min_release}-{max_release}."
        )

        releases: set[int] = set()
        db_for_release: dict[int, str] = {}
        db_dir_url_by_release: dict[int, str] = {}
        for rel in range(min_release, max_release + 1):
            db_name, db_dir_url = cls._ftp_find_core_db_dir(organism=organism, assembly=assembly, release=int(rel))
            if db_name and db_dir_url:
                releases.add(int(rel))
                db_for_release[int(rel)] = str(db_name)
                db_dir_url_by_release[int(rel)] = str(db_dir_url)

        releases_union = sorted(releases)
        if not releases_union:
            log.warning(
                f"Ensembl's HTTPS dumps have no core database of {organism!r} assembly {int(assembly)} "
                f"in releases {min_release}..{max_release}."
            )

        # Assign each release to the first configured port that could plausibly host it (based on min-release rules).
        releases_by_port: dict[int, set[int]] = {int(p): set() for p in ports}
        db_by_port_release: dict[int, dict[int, str]] = {int(p): {} for p in ports}
        port_for_release: dict[int, int] = {}
        for rel in releases_union:
            chosen: int | None = None
            for p in ports:
                min_rel = settings.ENSEMBL.mysql_port_min_release.get(int(p))
                if min_rel is None or int(rel) >= int(min_rel):
                    chosen = int(p)
                    break
            if chosen is None:
                chosen = int(ports[-1])
            port_for_release[int(rel)] = int(chosen)
            releases_by_port[int(chosen)].add(int(rel))
            db_by_port_release[int(chosen)][int(rel)] = db_for_release[int(rel)]

        return {
            "organism": organism,
            "genome_assembly": int(assembly),
            "ports": tuple(int(p) for p in ports),
            "releases_by_port": releases_by_port,
            "db_by_port_release": db_by_port_release,
            "releases": releases_union,
            "port_for_release": port_for_release,
            "db_for_release": db_for_release,
            "db_dir_url_by_release": db_dir_url_by_release,
            "source": "ftp",
            "scanned_up_to": int(max_release),
        }

    @staticmethod
    def _https_index_path(local_repository: str, organism: str, assembly: int) -> str:
        """Return the file in *local_repository* that keeps the HTTPS-dump index of one organism and assembly."""
        return os.path.join(local_repository, f"ensembl_https_index_{organism}_assembly-{int(assembly)}.json")

    @classmethod
    def _index_covers(cls, index: dict[str, Any], organism: str, assembly: int, needed_up_to: int | None) -> bool:
        """Return whether a core-DB index answers for every release up to *needed_up_to*.

        An index answers when its scan reached *needed_up_to* (``None``: the latest Ensembl release), clipped to the
        end of the assembly's release range: releases that existed at the scan cannot disappear, and releases after
        it are not needed. Otherwise only an index scanned from the network in this process, less than
        :py:attr:`idtrack.settings.ENSEMBL.rest_species_cache_seconds` ago, answers (it reached the latest release then; a
        long-running process scans again to learn of a new one).

        Args:
            index: A core-DB index (see :py:meth:`_get_core_db_index`).
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.
            needed_up_to: Last release needed; ``None`` for the latest Ensembl release.

        Returns:
            bool: ``True`` if *index* can be used without scanning again.
        """
        _, range_max = cls._release_window(organism, int(assembly))
        if needed_up_to is None:
            need = range_max
        else:
            need = int(needed_up_to) if range_max is None else min(int(needed_up_to), int(range_max))
        scanned_up_to = index.get("scanned_up_to")
        if need is not None and scanned_up_to is not None and int(scanned_up_to) >= int(need):
            return True
        scanned_at = index.get("scanned_live_at")
        return scanned_at is not None and time.monotonic() - float(scanned_at) < float(
            settings.ENSEMBL.rest_species_cache_seconds
        )

    @classmethod
    def _https_index_signature(cls, organism: str, assembly: int, ports: list[int]) -> dict[str, Any]:
        """Return the organism-table fields an HTTPS-dump index depends on (a saved index must match them).

        Args:
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly (releases are assigned to them).

        Returns:
            dict[str, Any]: Organism, assembly, core-database suffix, release window and ports.
        """
        first, last = cls._release_window(organism, int(assembly))
        return {
            "organism": organism,
            "genome_assembly": int(assembly),
            "db_suffix": organisms.db_suffix(organism, int(assembly)),
            "release_window": [int(first), None if last is None else int(last)],
            "ports": [int(p) for p in ports],
        }

    @classmethod
    def _read_https_index(cls, path: str, organism: str, assembly: int, ports: list[int]) -> dict[str, Any] | None:
        """Read an index saved by :py:meth:`_write_https_index`.

        Args:
            path: The saved file.
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly.

        Returns:
            Optional[dict[str, Any]]: The index, or ``None`` if the file is absent, unreadable, or was written for
            other organism-table settings.
        """
        try:
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(saved, dict) or saved.get("signature") != cls._https_index_signature(
            organism, assembly, ports
        ):
            return None
        try:
            return {
                "organism": organism,
                "genome_assembly": int(assembly),
                "ports": tuple(int(p) for p in ports),
                "releases_by_port": {int(p): {int(r) for r in rs} for p, rs in saved["releases_by_port"].items()},
                "db_by_port_release": {
                    int(p): {int(r): str(n) for r, n in d.items()} for p, d in saved["db_by_port_release"].items()
                },
                "releases": sorted(int(r) for r in saved["releases"]),
                "port_for_release": {int(r): int(p) for r, p in saved["port_for_release"].items()},
                "db_for_release": {int(r): str(n) for r, n in saved["db_for_release"].items()},
                "db_dir_url_by_release": {int(r): str(u) for r, u in saved["db_dir_url_by_release"].items()},
                "source": "ftp",
                "scanned_up_to": int(saved["scanned_up_to"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def _write_https_index(cls, path: str, index: dict[str, Any], ports: list[int]) -> None:
        """Save an HTTPS-dump index next to the cached tables (written to a temporary file, then renamed).

        Args:
            path: Destination file.
            index: Index returned by :py:meth:`_get_core_db_index_from_ftp`.
            ports: Configured MySQL ports of the assembly.
        """
        payload = {
            "signature": cls._https_index_signature(index["organism"], int(index["genome_assembly"]), ports),
            "scanned_up_to": int(index["scanned_up_to"]),
            "releases": sorted(int(r) for r in index["releases"]),
            "releases_by_port": {str(p): sorted(int(r) for r in rs) for p, rs in index["releases_by_port"].items()},
            "db_by_port_release": {
                str(p): {str(r): n for r, n in d.items()} for p, d in index["db_by_port_release"].items()
            },
            "port_for_release": {str(r): int(p) for r, p in index["port_for_release"].items()},
            "db_for_release": {str(r): n for r, n in index["db_for_release"].items()},
            "db_dir_url_by_release": {str(r): u for r, u in index["db_dir_url_by_release"].items()},
        }
        try:  # a cache: a failure to save it costs a scan next time, nothing else
            write_text(path, json.dumps(payload, indent=1, sort_keys=True))
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not save the HTTPS-dump index %r: %s", path, exc)

    @classmethod
    def _https_index(
        cls,
        *,
        organism: str,
        assembly: int,
        ports: list[int],
        local_repository: str | None,
        needed_up_to: int | None,
    ) -> dict[str, Any]:
        """Return the HTTPS-dump index, from *local_repository* when a saved one covers the need, else scanned.

        The scan asks Ensembl REST for the latest release and probes the dump directory of every release, so a saved
        index is what lets a manager whose window ends at a known release (every graph built or loaded through
        :py:meth:`idtrack.API.build`) work without the network once its tables and graph are on disk.

        Args:
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly.
            local_repository: Folder where the index is saved; ``None`` neither reads nor saves one.
            needed_up_to: Last release needed; ``None`` for the latest Ensembl release.

        Returns:
            dict[str, Any]: The HTTPS-dump index; ``scanned_live_at`` (``time.monotonic()``) is set when it was scanned
            in this call.
        """
        path = cls._https_index_path(local_repository, organism, assembly) if local_repository else None
        if path is not None:
            saved = cls._read_https_index(path, organism, assembly, ports)
            if saved is not None and cls._index_covers(saved, organism, assembly, needed_up_to):
                logging.getLogger(__name__).info(
                    "Ensembl releases of %s assembly %s read from %r (scanned up to release %s).",
                    organism,
                    assembly,
                    path,
                    saved["scanned_up_to"],
                )
                return saved
        index = cls._get_core_db_index_from_ftp(organism=organism, assembly=int(assembly), ports=ports)
        if path is not None:
            cls._write_https_index(path, index, ports)
        return {**index, "scanned_live_at": time.monotonic()}

    @classmethod
    def _probe_mysql_core_schemas_by_port(
        cls, *, organism: str, assembly: int, ports: list[int]
    ) -> tuple[dict[int, set[int]], dict[int, dict[int, str]], dict[int, Exception]]:
        """Return per-port core DB releases and schema names from the live Ensembl MySQL service.

        Only names of the assembly's core databases (``DbSuffix``, optional patch letters) whose release lies in the
        assembly's release window (:py:meth:`_release_window`) are kept, exactly as on the HTTPS path. Ports that
        :py:attr:`idtrack.settings.ENSEMBL.mysql_port_assembly` reserves for another assembly are never queried. Ports that the
        process-wide :py:data:`idtrack.build.ensembl_access.reachability` reports unusable (blocked, or ``"https"``
        access mode) are skipped without a connection attempt.

        Args:
            organism: Ensembl species name.
            assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly.

        Returns:
            tuple: ``(releases_by_port, db_by_port_release, connection_errors)``.
        """
        # Some Ensembl releases encode patch builds as a trailing letter, e.g. `homo_sapiens_core_56_37a`.
        # Treat these as part of the same assembly for configuration and graph-level purposes.
        pattern = organisms.core_db_name_pattern(organism, int(assembly))
        first_release, last_release = cls._release_window(organism, int(assembly))
        suffix = organisms.db_suffix(organism, int(assembly))
        # Use an information_schema filter to avoid `SHOW DATABASES` timeouts on busy ports.
        escaped_organism = organism.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = f"{escaped_organism}\\_core\\_%\\_{suffix}%"

        # Keep keys for all configured ports, even if some are unreachable.
        releases_by_port: dict[int, set[int]] = {int(p): set() for p in ports}
        db_by_port_release: dict[int, dict[int, str]] = {int(p): {} for p in ports}
        connection_errors: dict[int, Exception] = {}

        for port in ports:
            owner = settings.ENSEMBL.mysql_port_assembly.get(int(port))
            if owner is not None and owner != (organism, int(assembly)):
                # A port reserved for another assembly (3337, the GRCh37 archive) holds frozen copies of older
                # assemblies under later release numbers (e.g. MMUL_1 as `..._98_10`), which the suffix and release
                # range would attribute to the wrong assembly. The organism table cannot list such a port for another
                # assembly; this guards tables assigned without validation.
                continue
            # Avoid long hangs when outbound MySQL ports are blocked (common on managed networks): the port is
            # probed at most once per process.
            if not reachability.mysql_port_usable(int(port)):
                connection_errors[int(port)] = ConnectionError(
                    f"MySQL {settings.ENSEMBL.mysql_host}:{int(port)} is not used (unreachable, or Ensembl access mode 'https')"
                )
                continue

            results_query = None
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    with pymysql.connect(
                        host=settings.ENSEMBL.mysql_host,
                        user=settings.ENSEMBL.mysql_user,
                        password=settings.ENSEMBL.mysql_password,
                        port=int(port),
                        connect_timeout=max(settings.ENSEMBL.connection_timeout, 30),
                        read_timeout=max(settings.ENSEMBL.reading_timeout, 60),
                    ) as connection:
                        with connection.cursor() as cur:
                            cur.execute(
                                "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                                "WHERE SCHEMA_NAME LIKE %s ESCAPE '\\\\'",
                                (like_pattern,),
                            )
                            results_query = cur.fetchall()
                    break
                except Exception as exc:
                    last_exc = exc
                    results_query = None
                    if attempt == 0:
                        continue

            if results_query is None:
                connection_errors[int(port)] = last_exc or RuntimeError("Unknown MySQL connection error")
                if last_exc is None or cls._is_mysql_connection_failure(last_exc):
                    reachability.mark_unreachable(settings.ENSEMBL.mysql_host, int(port), reason=str(last_exc))
                continue

            rels: set[int] = set()
            db_for_release_on_port: dict[int, str] = {}
            for (db_name,) in results_query:
                if isinstance(db_name, bytes):
                    db_name = db_name.decode("utf-8")
                db_name = str(db_name)
                m = pattern.match(db_name)
                if not m:
                    continue
                rel = int(m.group("release"))
                if rel < first_release or (last_release is not None and rel > last_release):
                    # Another assembly that shares the core-database suffix (e.g. rhesus MMUL_1 vs Mmul_10).
                    continue
                rels.add(rel)
                existing = db_for_release_on_port.get(rel)
                if existing is None or len(db_name) < len(existing):
                    db_for_release_on_port[rel] = db_name

            releases_by_port[int(port)] = rels
            db_by_port_release[int(port)] = db_for_release_on_port

        return releases_by_port, db_by_port_release, connection_errors

    @classmethod
    def _refresh_core_db_index_mysql(cls, index: dict[str, Any], *, organism: str, assembly: int) -> dict[str, Any]:
        """Refresh the MySQL-derived portion of a cached core-index in place."""
        ports = list(index.get("ports") or organisms.ORGANISM_TABLE[organism][int(assembly)]["Ports"])
        ports = [int(p) for p in ports]
        releases_by_port, db_by_port_release, connection_errors = cls._probe_mysql_core_schemas_by_port(
            organism=organism,
            assembly=int(assembly),
            ports=ports,
        )

        mysql_releases = sorted({r for rs in releases_by_port.values() for r in rs})
        mysql_port_for_release: dict[int, int] = {}
        mysql_db_for_release: dict[int, str] = {}
        for rel in mysql_releases:
            for port in ports:
                if rel in releases_by_port.get(int(port), set()):
                    mysql_port_for_release[int(rel)] = int(port)
                    mysql_db_for_release[int(rel)] = db_by_port_release[int(port)][int(rel)]
                    break

        updated = dict(index)
        updated["ports"] = tuple(int(p) for p in ports)
        updated["releases_by_port"] = releases_by_port
        updated["db_by_port_release"] = db_by_port_release
        updated["releases_on_mysql"] = mysql_releases
        updated["mysql_connection_errors"] = {
            int(port): f"{exc.__class__.__name__}: {exc}" for port, exc in sorted(connection_errors.items())
        }

        releases = sorted(set(updated.get("releases", []) or []) | set(mysql_releases))
        updated["releases"] = releases

        db_for_release = dict(updated.get("db_for_release", {}) or {})
        for rel, db_name in mysql_db_for_release.items():
            db_for_release[int(rel)] = str(db_name)
        updated["db_for_release"] = db_for_release

        port_for_release = dict(updated.get("port_for_release", {}) or {})
        for rel, port in mysql_port_for_release.items():
            port_for_release[int(rel)] = int(port)
        for rel in releases:
            if int(rel) not in port_for_release:
                port_for_release[int(rel)] = int(ports[0])
        updated["port_for_release"] = port_for_release

        if mysql_releases:
            if updated.get("db_dir_url_by_release"):
                updated["source"] = "merged"
            else:
                updated["source"] = "mysql"
        else:
            if updated.get("db_dir_url_by_release"):
                updated["source"] = "ftp"

        return updated

    @classmethod
    def _get_core_db_index(
        cls,
        *,
        organism: str,
        assembly: int,
        local_repository: str | None = None,
        needed_up_to: int | None = None,
    ) -> dict[str, Any]:
        """Return cached core-DB availability for an (organism, assembly) pair across all configured ports.

        The Ensembl public MySQL service can host the same assembly on multiple ports depending on release
        (e.g. *homo_sapiens* assembly 37). To support full-history workflows we build a small in-memory index:

        - `ports`: ports probed for this (organism, assembly) in preference order
        - `releases_by_port`: releases available for this assembly on each reachable port
        - `db_by_port_release`: schema name for each available release on each reachable port
        - `releases`: sorted union of all releases across ports
        - `port_for_release`: deterministic choice of port for each release (first configured port that has it)
        - `db_for_release`: chosen schema name for each release (matching `port_for_release`, includes patch-letter suffixes)

        Both sources (the MySQL probe and the HTTPS scan) look for core databases named with the assembly's
        ``DbSuffix`` and accept only releases inside its ``ReleaseRange`` (see :py:meth:`_release_window`), so
        they return the same releases for an assembly. The index is cached per process and per version of the
        organism table. The HTTPS part is also saved in *local_repository* (see :py:meth:`_https_index`) and read
        from there when it covers the releases up to *needed_up_to*, so that no Ensembl REST call and no dump scan
        is needed for a window that ends at a release the saved scan already reached.

        Args:
            organism: Canonical Ensembl organism name (e.g. ``"homo_sapiens"``).
            assembly: Assembly key of the organism table (e.g. ``38`` for GRCh38).
            local_repository: Folder of the caller's cached tables, where the HTTPS index is saved; ``None`` keeps it
                in memory only.
            needed_up_to: Last release the caller needs (its ``snapshot_release``); ``None`` means the latest Ensembl
                release.

        Returns:
            dict[str, Any]: Mapping describing reachable releases/ports for this organism/assembly.

        If the dump scan cannot be completed (a request stays unanswered), its
        :py:class:`idtrack.EnsemblDownloadError` propagates; no partial index is ever returned or
        cached. An unknown organism or assembly raises the organism table's error.
        """
        key = (organism, int(assembly), int(organisms.table_version()))
        cached = cls._core_db_index_cache.get(key)
        if cached is not None and not cls._index_covers(cached, organism, int(assembly), needed_up_to):
            cached = None
        if cached is not None:
            errors = cached.get("mysql_connection_errors") or {}
            if errors:
                # If a previously failing port is usable now (its "unreachable" answer in the reachability memory
                # expired and a new probe succeeded), refresh the MySQL-derived portion so downstream callers don't
                # get stuck with stale partial results. Asking the memory costs nothing otherwise.
                if any(reachability.mysql_port_usable(int(port)) for port in errors):
                    refreshed = cls._refresh_core_db_index_mysql(cached, organism=organism, assembly=int(assembly))
                    cls._core_db_index_cache[key] = refreshed
                    return refreshed
            return cached

        ports = list(organisms._entry(organism, int(assembly))["Ports"])
        releases_by_port, db_by_port_release, connection_errors = cls._probe_mysql_core_schemas_by_port(
            organism=organism,
            assembly=int(assembly),
            ports=[int(p) for p in ports],
        )

        mysql_releases = sorted({r for rs in releases_by_port.values() for r in rs})
        mysql_port_for_release: dict[int, int] = {}
        mysql_db_for_release: dict[int, str] = {}
        for rel in mysql_releases:
            for port in ports:
                if rel in releases_by_port.get(int(port), set()):
                    mysql_port_for_release[int(rel)] = int(port)
                    mysql_db_for_release[int(rel)] = db_by_port_release[int(port)][int(rel)]
                    break

        # The HTTPS dumps hold every release; the live MySQL service only recent ones. An index without the dump scan
        # would silently lack releases, so a failed scan raises (and nothing is cached).
        ftp_index = cls._https_index(
            organism=organism,
            assembly=int(assembly),
            ports=[int(p) for p in ports],
            local_repository=local_repository,
            needed_up_to=needed_up_to,
        )

        # Merge: use FTP for the full release catalogue (historic snapshots), but keep MySQL-derived
        # per-port availability for releases that are on the live service.
        ftp_releases = list(ftp_index["releases"])
        releases = sorted(set(ftp_releases) | set(mysql_releases))

        db_for_release = dict(ftp_index.get("db_for_release", {}))
        # For MySQL-backed releases, prefer the *live* schema name (may include patch-letter suffixes).
        for rel, db_name in mysql_db_for_release.items():
            db_for_release[int(rel)] = str(db_name)

        db_dir_url_by_release = dict(ftp_index.get("db_dir_url_by_release", {}))

        port_for_release = dict(ftp_index.get("port_for_release", {}))
        for rel, port in mysql_port_for_release.items():
            port_for_release[int(rel)] = int(port)

        # Ensure every release in the union has a deterministic port assignment.
        missing_ports = [rel for rel in releases if int(rel) not in port_for_release]
        if missing_ports:
            # Assign unknown ones to the first configured port as a safe default; these should be rare.
            for rel in missing_ports:
                port_for_release[int(rel)] = int(ports[0])

        index = {
            "organism": organism,
            "genome_assembly": int(assembly),
            "ports": tuple(int(p) for p in ports),
            "releases_by_port": releases_by_port,
            "db_by_port_release": db_by_port_release,
            "releases": releases,
            "port_for_release": port_for_release,
            "db_for_release": db_for_release,
            "db_dir_url_by_release": db_dir_url_by_release,
            "releases_on_mysql": mysql_releases,
            "mysql_connection_errors": {
                int(port): f"{exc.__class__.__name__}: {exc}" for port, exc in sorted(connection_errors.items())
            },
            "source": "merged" if mysql_releases else "ftp",
            "scanned_up_to": ftp_index.get("scanned_up_to"),
            "scanned_live_at": ftp_index.get("scanned_live_at"),
        }
        cls._core_db_index_cache[key] = index
        return index

    def __init__(
        self,
        organism: str,
        form: str,
        local_repository: str,
        release: int | None = None,
        first_release: int | None = None,
        snapshot_release: int | float | None = None,
        store_raw_always: bool = True,
        assembly: SupportsIndex | float | str | None = None,
    ):
        """Initialize a :py:class:`DatabaseManager` for a specific organism, release, and assembly.

        Args:
            organism (str): Canonical species name in Ensembl schema (e.g. ``"homo_sapiens"``); one of
                :py:func:`idtrack.organisms.supported_organisms`, otherwise :py:class:`idtrack.errors.UnknownOrganismError`
                is raised.
            form (str): Biological entity level of interest—one of ``"gene"``, ``"transcript"``,
                ``"translation"``, …—governing which stable-ID columns will be expected downstream.
            local_repository (str): Directory of the cached tables (:py:attr:`tables_directory`, one HDF5 file per
                table), the saved index of Ensembl's dumps and the graphs. It must exist and be readable; it must be
                writable only when a table has to be computed and stored (a read-only repository serves what it
                holds).
            release (Optional[int]): Target Ensembl release number. If ``None`` the most
                recent release available for *assembly* is selected automatically.
            first_release (Optional[int]): Earliest release to include when building cross-release ID
                histories. Defaults to the first release of the organism in the organism table
                (:py:meth:`idtrack.organisms.first_release`, over all its assemblies), the same default as
                :py:class:`idtrack.API` uses.
            snapshot_release (Optional[int | float]): Latest release to include when building histories.
                ``None`` (the default) or ``np.inf`` disables the upper bound and includes all newer releases.
            store_raw_always (bool): When ``True`` (default) raw Ensembl tables are stored in the repository after
                their download; when ``False`` they are not, and every derived table that needs them downloads them
                again.
            assembly (Optional[int]): Assembly key of the organism table
                (:py:attr:`idtrack.organisms.ORGANISM_TABLE`), as an integer of any integer type, an integral
                float or a string of digits (:py:meth:`idtrack.organisms.assembly_key`). This selects the
                **primary** assembly used for data access (e.g. ``38`` = human GRCh38, ``39`` = mouse GRCm39,
                ``111`` = pig Sscrofa11.1, ``10`` = rhesus Mmul_10). The key usually equals the code in core schema
                names
                (``<organism>_core_<release>_<code>``); see the table's ``DbSuffix``. If omitted, the highest-priority
                assembly for *organism* is used. If *release* is provided, the selection is restricted to
                assemblies that actually contain that release; otherwise to assemblies with a release inside the
                ``first_release``/``snapshot_release`` window.

        Raises:
            ValueError: If *form* is not in the supported list, *local_repository* fails basic
                path/read/write checks, or the Ensembl access mode (``IDTRACK_ENSEMBL_ACCESS`` /
                :py:attr:`idtrack.settings.ENSEMBL.access`) is invalid.
            RuntimeError: If internal port/release configuration is inconsistent.
            UnknownOrganismError: If *organism* is not in the organism table (:py:attr:`idtrack.organisms.supported_organisms()`).
        """
        if organism not in organisms.supported_organisms():
            raise UnknownOrganismError(
                f"Organism {organism!r} is not configured. Supported organisms: {organisms.supported_organisms()}. "
                "To add an organism, register it with `idtrack.organisms.register_organism` (or add it to "
                "`organisms.ORGANISM_TABLE` in idtrack/organisms.py) and provide its externals YAML file "
                f"`{organism}_externals_modified.yml`; see 'Adding organism support' in the contributing guide."
            )

        # Validate the Ensembl access mode first, so that an invalid `IDTRACK_ENSEMBL_ACCESS` / `settings.ENSEMBL.access`
        # raises its own error here instead of surfacing later inside an assembly probe.
        ensembl_access_mode()

        if not os.path.isdir(local_repository):
            raise InvalidArgumentError(f"`local_repository` must be an existing directory: {local_repository!r}")
        # Write access is needed only when a table is to be stored (see `_store`): a repository on a read-only volume
        # serves the tables and graphs it holds.
        if not os.access(local_repository, os.R_OK):
            raise InvalidArgumentError(f"`local_repository` must be readable: {local_repository!r}")

        if form not in ids.FORMS:
            raise InvalidArgumentError(f"Form {form!r} is not supported. Supported forms: {ids.FORMS}")

        # One rule for a release argument (`idtrack._arguments.release`); `None` (or `np.inf`) as the last release
        # means no upper bound.
        release = _arguments.optional_release(release, "release")
        first_effective = _arguments.optional_release(first_release, "first_release")
        snapshot_effective: int | float = (
            np.inf
            if snapshot_release is None or (isinstance(snapshot_release, float) and np.isinf(snapshot_release))
            else _arguments.release(snapshot_release, "snapshot_release")
        )

        if first_effective is not None and not np.isinf(snapshot_effective):
            if int(snapshot_effective) < int(first_effective):
                raise InvalidArgumentError(
                    f"`snapshot_release` must be >= `first_release` (got first_release={first_effective}, "
                    f"snapshot_release={snapshot_effective}).",
                    argument="snapshot_release",
                    value=snapshot_effective,
                )

        configured_assemblies = organisms.ORGANISM_TABLE[organism]
        # The window's end decides whether a saved HTTPS-dump index can answer without the network.
        index_kwargs: dict[str, Any] = {
            "local_repository": local_repository,
            "needed_up_to": None if np.isinf(snapshot_effective) else int(snapshot_effective),
        }
        if assembly is None:
            # Pick the highest-priority assembly that has the requested release (or, without one, a release inside the
            # window; e.g. a mouse snapshot at release 100 is GRCm38, because GRCm39 starts at 103). An
            # assembly is passed over only when its complete index lacks the release; an index that cannot be built
            # raises, because taking the next assembly would silently give another primary assembly (one failed
            # request would make human GRCh37).
            requested_release = int(release) if release is not None else None

            def _has_release_in_window(index: dict[str, Any]) -> bool:
                return any(
                    (first_effective is None or int(r) >= int(first_effective))
                    and (np.isinf(snapshot_effective) or int(r) <= int(snapshot_effective))
                    for r in index.get("releases", [])
                )

            available_summary: dict[int, tuple[int | None, int | None]] = {}
            for asm in organisms.assemblies_by_priority(organism):
                index = self._get_core_db_index(organism=organism, assembly=int(asm), **index_kwargs)
                releases = index.get("releases") or []
                available_summary[int(asm)] = (min(releases), max(releases)) if releases else (None, None)
                if (requested_release is None and _has_release_in_window(index)) or (
                    requested_release is not None and requested_release in index["port_for_release"]
                ):
                    assembly = int(asm)
                    break

            if assembly is None:
                raise UnknownAssemblyError(
                    f"No assembly of {organism!r} has a core database "
                    + (
                        f"at release {requested_release}"
                        if requested_release is not None
                        else f"in the window {first_effective}..{snapshot_effective}"
                    )
                    + f". Available (min, max) releases by assembly: {available_summary}."
                )
        else:
            assembly = organisms.assembly_key(assembly)
            if assembly not in configured_assemblies:
                raise UnknownAssemblyError(
                    f"Genome assembly {assembly} is not configured for organism {organism!r}. "
                    f"Supported assemblies: {sorted(configured_assemblies)}",
                    argument="assembly",
                    value=assembly,
                    allowed=sorted(configured_assemblies),
                )

        # MYSQL Settings
        self.assembly = assembly
        core_index = self._get_core_db_index(organism=organism, assembly=self.assembly, **index_kwargs)

        if release is None:
            if not core_index["releases"]:
                raise UnknownReleaseError(
                    f"No Ensembl core databases found for {organism!r} on assembly {self.assembly} "
                    f"(tried MySQL ports {core_index['ports']})."
                )
            if not np.isinf(snapshot_effective):
                eligible = [r for r in core_index["releases"] if r <= int(snapshot_effective)]
                if not eligible:
                    raise UnknownReleaseError(
                        f"No Ensembl core database found for {organism!r} on assembly {self.assembly} "
                        f"at or before snapshot_release={snapshot_effective!r}. "
                        f"Available releases: {core_index['releases']}"
                    )
                release = eligible[-1]
            else:
                release = core_index["releases"][-1]

        release = int(release)
        if release not in core_index["port_for_release"]:
            raise UnknownReleaseError(
                f"No Ensembl core database found for {organism!r} release {release} on assembly {self.assembly}. "
                f"Tried MySQL ports {core_index['ports']}. Available releases: {core_index['releases']}"
            )
        selected_port = int(core_index["port_for_release"][release])

        self.mysql_settings: dict[str, Any] = {
            "host": settings.ENSEMBL.mysql_host,
            "user": settings.ENSEMBL.mysql_user,
            "password": settings.ENSEMBL.mysql_password,
            "port": selected_port,
            "connect_timeout": max(settings.ENSEMBL.connection_timeout, 30),
            "read_timeout": max(settings.ENSEMBL.reading_timeout, 60),
        }

        # The logger for informing the user about the progress.
        self.log = logging.getLogger(__name__)

        # Instance attributes
        self.local_repository = local_repository
        self.organism = organism
        self.form = form
        self.store_raw_always = store_raw_always
        if first_effective is None:
            first_effective = organisms.first_release(organism)
        self.first_release = int(first_effective)
        # If snapshot_release is not specified, then set it to infinite.
        self.snapshot_release: int | float = snapshot_effective

        # Protected attributes
        self.available_form_of_interests = copy.deepcopy(ids.FORMS)  # Warning: the order is important.
        self._column_sep = "_COL_"
        self._identifiers = [f"{self.form}_stable_id", f"{self.form}_version"]

        self.release = int(release)

        if not np.isinf(self.snapshot_release) and self.snapshot_release < self.first_release:
            raise InvalidArgumentError(
                f"`snapshot_release` must be >= `first_release` (got first_release={self.first_release}, "
                f"snapshot_release={self.snapshot_release}).",
                argument="snapshot_release",
                value=self.snapshot_release,
            )

        releases_in_window = sorted(
            r for r in core_index["releases"] if self.snapshot_release >= r >= self.first_release
        )
        if not releases_in_window:
            msg = (
                "No Ensembl releases remain after applying "
                f"first_release={self.first_release} and snapshot_release={self.snapshot_release} "
                f"for {self.organism!r} assembly {int(self.assembly)}. "
                f"Available releases: {core_index['releases']}"
            )
            raise UnknownReleaseError(msg)
        if self.release not in releases_in_window:
            raise UnknownReleaseError(
                f"release={self.release} is outside the window "
                f"[{self.first_release}, {self.snapshot_release}] for {self.organism!r} assembly {int(self.assembly)}. "
                f"Releases within window: {releases_in_window}"
            )

        # Only enforce port-level minimum-release rules when we expect to query the live MySQL service.
        # When a release is accessed via the HTTPS/FTP dumps, the configured MySQL port is only a
        # bookkeeping detail and may not support the historical release directly.
        mysql_releases = set(core_index.get("releases_on_mysql", []) or [])
        if mysql_releases and int(self.release) in mysql_releases:
            min_port_release = settings.ENSEMBL.mysql_port_min_release.get(int(self.mysql_settings["port"]))
            if min_port_release is not None and self.release < int(min_port_release):
                raise IntegrityError(
                    f"Internal configuration error: selected port {int(self.mysql_settings['port'])} has min release "
                    f"{int(min_port_release)} but release {int(self.release)} was selected."
                )

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
            f"    Ensembl Release: {self.release}{os.linesep}"
            f"    Genome Assembly: {self.assembly}{os.linesep}"
            f"    Ignore Before: {self.first_release}{os.linesep}"
            f"    Ignore After: {self.snapshot_release}{os.linesep}"
            f"    Local Repository: {self.local_repository}{os.linesep}"
            f"    Store Raw Always: {self.store_raw_always}{os.linesep}"
        )

    @property
    def _core_index_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of :py:meth:`_get_core_db_index` for this manager (its repository and window end)."""
        snapshot_release = getattr(self, "snapshot_release", np.inf)
        return {
            "local_repository": getattr(self, "local_repository", None),
            "needed_up_to": None if np.isinf(snapshot_release) else int(snapshot_release),
        }

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
            release=self.release,
            form=self.form,
            local_repository=self.local_repository,
            assembly=self.assembly,
        )

    @cached_property
    def available_releases(self) -> list[int]:
        """Return Ensembl releases that are both reachable and within the window.

        Availability is read from the cached, multi-port aware core index built by :py:meth:`_get_core_db_index`
        (the union of releases over the ports), filtered against ``first_release`` / ``snapshot_release``, sorted in
        ascending order, and cached for the lifetime of this :py:class:`DatabaseManager` instance. The resulting list
        represents releases that can safely be queried **and** cached locally, guaranteeing reproducible downstream
        analyses.

        Returns:
            list[int]: Sorted release numbers satisfying reachability and window constraints.
        """
        core_index = self._get_core_db_index(organism=self.organism, assembly=self.assembly, **self._core_index_kwargs)
        return sorted(r for r in core_index["releases"] if self.snapshot_release >= r >= self.first_release)

    @cached_property
    def available_releases_all_assemblies(self) -> list[int]:
        """Return all Ensembl releases that have a core database of at least one configured assembly.

        For clean-handoff species (e.g. mouse: 37 → 38 → 39), no single assembly spans the full release
        history. Graph construction and YAML template generation therefore need a release catalogue that is the
        union over *all* assemblies configured for :py:attr:`organism` in
        :py:data:`idtrack.organisms.ORGANISM_TABLE`.

        The list is filtered by the manager's ``first_release`` / ``snapshot_release`` window and cached for the
        lifetime of this :py:class:`DatabaseManager` instance. Every assembly's index must be complete: if one cannot
        be built (Ensembl does not answer), the error is raised instead of leaving that assembly's releases out.

        Returns:
            list[int]: Sorted release numbers available for at least one configured assembly.
        """
        releases: set[int] = set()
        for asm in organisms.assemblies_by_priority(self.organism):
            core_index = self._get_core_db_index(organism=self.organism, assembly=int(asm), **self._core_index_kwargs)
            releases.update(int(r) for r in core_index["releases"] if self.snapshot_release >= r >= self.first_release)
        return sorted(releases)

    @cached_property
    def mysql_database(self) -> str:
        """Return the canonical Ensembl Core schema name for the current organism, release, and assembly.

        The schema naming convention is deterministic:

        ``{organism}_core_{release}_{DbSuffix}[<patch>]``

        where ``DbSuffix`` is the assembly's core-database code (:py:meth:`idtrack.organisms.db_suffix`, usually the
        assembly key itself).

        For multi-port assemblies (e.g. *sus_scrofa* assembly ``102``), the *port* is selected in
        :py:meth:`__init__` using :py:meth:`_get_core_db_index`.  Once the release is validated to exist on that
        chosen port, the schema name itself does not require another server-side discovery query.

        Returns:
            str: Schema name like ``"homo_sapiens_core_111_38"``.

        Raises:
            ValueError: If the current release is not available for this (organism, assembly) pair.
        """
        core_index = self._get_core_db_index(organism=self.organism, assembly=self.assembly, **self._core_index_kwargs)
        if int(self.release) not in core_index["port_for_release"]:
            raise UnknownReleaseError(
                f"No Ensembl core database found for {self.organism!r} release {int(self.release)} "
                f"on assembly {int(self.assembly)}. Available releases: {core_index['releases']}"
            )
        return str(core_index["db_for_release"][int(self.release)])

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
            release=self.release,
            form=form,
            local_repository=self.local_repository,
            first_release=self.first_release,
            snapshot_release=self.snapshot_release,
            store_raw_always=self.store_raw_always,
            assembly=self.assembly,
        )

    def change_release(self, release: int) -> "DatabaseManager":
        """Produce a new manager that targets a different Ensembl release.

        The returned instance inherits organism, form, assembly, and all caching parameters, but points every
        subsequent query (MySQL, FTP, or REST) to *release*.  This is the recommended way to traverse
        releases in scripted analyses without mutating objects in-place.

        Args:
            release (int): Desired Ensembl release number (e.g. ``111``).  Must be available for the
                current genome assembly, otherwise :py:class:`idtrack.errors.UnknownReleaseError` is raised.

        Returns:
            DatabaseManager: Fresh manager initialised for *release*.
        """
        return DatabaseManager(
            organism=self.organism,
            release=release,
            form=self.form,
            local_repository=self.local_repository,
            first_release=self.first_release,
            snapshot_release=self.snapshot_release,
            store_raw_always=self.store_raw_always,
            assembly=self.assembly,
        )

    def change_release_auto_assembly(self, release: int) -> "DatabaseManager":
        """Clone the manager for *release* while inferring a compatible genome assembly.

        Unlike :py:meth:`change_release`, this helper allows :py:attr:`assembly` to change when the
        requested release is not present in the current assembly (common for clean-handoff species).
        Assembly inference follows the same priority rules as :py:meth:`__init__` with ``assembly=None``:
        pick the highest-priority configured assembly that contains the requested release.

        Args:
            release (int): Desired Ensembl release number.

        Returns:
            DatabaseManager: Fresh manager initialised for *release* with an inferred assembly.
        """
        return DatabaseManager(
            organism=self.organism,
            release=int(release),
            form=self.form,
            local_repository=self.local_repository,
            first_release=self.first_release,
            snapshot_release=self.snapshot_release,
            store_raw_always=self.store_raw_always,
            assembly=None,
        )

    def change_assembly(self, assembly: int, last_possible_release: bool = False) -> "DatabaseManager":
        """Clone the manager while targeting a new genome assembly (e.g. GRCh38 → GRCh37).

        Genome assemblies are identified by the integer keys of :py:attr:`idtrack.organisms.ORGANISM_TABLE`
        (``38`` for *GRCh38*, ``37`` for *GRCh37*, ``39`` for *GRCm39*, ``111`` for *Sscrofa11.1*, ``1`` for rhesus
        *MMUL_1*, …).  When
        *last_possible_release* is ``True`` the
        method automatically picks the most recent Ensembl release that **still** provides MySQL dumps for the
        requested assembly, ensuring compatibility.  All other settings are copied verbatim.

        Args:
            assembly (int): Assembly code configured under
                :py:attr:`organisms.ORGANISM_TABLE` for this manager's organism.
            last_possible_release (bool): When ``True`` override *release* with the
                newest version available for *assembly*.  Defaults to ``False``.

        Returns:
            DatabaseManager: New manager tied to the requested assembly (and possibly a recalculated release).
        """
        return DatabaseManager(
            organism=self.organism,
            release=self.release if not last_possible_release else None,
            form=self.form,
            local_repository=self.local_repository,
            first_release=self.first_release,
            snapshot_release=self.snapshot_release,
            store_raw_always=self.store_raw_always,
            assembly=assembly,
        )

    def create_available_databases(self) -> pd.DataFrame:
        """List the core databases of the organism's configured assemblies, each within its release range.

        The names come from the core-database index of every assembly in the organism table
        (:py:meth:`_get_core_db_index`): databases named with the assembly's ``DbSuffix`` (patch letters included,
        e.g. ``homo_sapiens_core_54_36p``) whose release lies in the assembly's ``ReleaseRange`` and at or above
        release 48. The index merges the live MySQL catalogue (where its ports are usable) with the HTTPS dump
        listings, so the list is the same whichever way Ensembl is reached. Core databases of assemblies that the
        table does not declare, or outside their ranges, are not listed. An assembly for which no database is found
        contributes nothing; a failure to build an index raises.

        An invalid Ensembl access mode raises its own ``ValueError`` here (see
        :py:func:`idtrack.build.ensembl_access.ensembl_access_mode`).

        Returns:
            pandas.DataFrame: One column named ``"available_databases"``, sorted.
        """
        ensembl_access_mode()  # an invalid mode must raise here, not be read as "no databases found"
        self.log.info(f"Available core databases for {self.organism} are being listed.")
        names: set[str] = set()
        for assembly in organisms.assemblies_by_priority(self.organism):
            core_index = self._get_core_db_index(
                organism=self.organism, assembly=int(assembly), **self._core_index_kwargs
            )
            names.update(str(v) for v in core_index.get("db_for_release", {}).values() if v)
        return pd.DataFrame(sorted(names), columns=["available_databases"])

    # ------------------------------------------------------------------------------------------------------------
    # Cached tables: one file per table (see idtrack.build.tables); the file name carries every parameter the table
    # depends on, and the file records them again (checked when it is read).

    @property
    def tables_directory(self) -> str:
        """Folder of this organism's and assembly's cached tables: ``<local_repository>/idtrack_tables/<org>_assembly-<key>``."""
        return os.path.join(self.local_repository, "idtrack_tables", f"{self.organism}_assembly-{int(self.assembly)}")

    def table_path(self, name: str, release: int | None = None) -> str:
        """Return the file of a cached table.

        Args:
            name: The table's key without the release, e.g. ``"mysql_gene__cols=gene_id+stable_id+version"`` or
                ``"idhistory-narrow_gene"``; it names every parameter the table depends on.
            release: Release of the table; the manager's release by default.

        Returns:
            str: ``<tables_directory>/ens<release>_<name>.h5``.
        """
        return os.path.join(self.tables_directory, f"ens{int(self.release if release is None else release)}_{name}.h5")

    def table_digests(self) -> dict[str, str]:
        """Return ``{file name: content SHA-256}`` of every table stored for this organism and assembly.

        The digests identify the tables' content (:py:func:`idtrack.build.tables.read_table_digest`), so that a graph
        can record what it was built from.

        Returns:
            dict[str, str]: Sorted by file name; empty if nothing is stored.
        """
        if not os.path.isdir(self.tables_directory):
            return {}
        out = {}
        for name in sorted(os.listdir(self.tables_directory)):
            if name.endswith(".h5"):
                digest = hs.read_table_digest(os.path.join(self.tables_directory, name))
                if digest is not None:
                    out[name] = digest
        return out

    def _table_params(self, name: str, release: int | None = None, **extra: Any) -> dict[str, Any]:
        """Return the parameters recorded with a cached table (checked again when it is read)."""
        params: dict[str, Any] = {
            "organism": self.organism,
            "genome_assembly": int(self.assembly),
            "table": name,
            "ensembl_release": int(self.release if release is None else release),
        }
        params.update(extra)
        return params

    def _store(self, path: str, df: pd.DataFrame, params: dict[str, Any]) -> None:
        """Write a table to the repository (atomically, see :py:func:`idtrack.build.tables.write_table`).

        Args:
            path: The table's file.
            df: The table.
            params: Parameters the table depends on.

        Raises:
            PermissionError: If the repository is not writable; the message names it.
        """
        if not os.access(self.local_repository, os.W_OK):
            raise PermissionError(
                f"Table {os.path.basename(path)!r} has to be computed and stored, but `local_repository` "
                f"{self.local_repository!r} is not writable. A read-only repository serves only the tables and "
                "graphs it already holds."
            )
        self.log.info(f"Storing table `{os.path.basename(path)}` in `{self.tables_directory}`.")
        hs.write_table(path, df, params)

    def _cached(
        self,
        path: str,
        params: dict[str, Any],
        compute: Any,
        create_even_if_exist: bool,
        save_after_calculation: bool,
    ) -> pd.DataFrame:
        """Return the table stored at *path* with *params*, or compute it (and store it when asked).

        Args:
            path: File of the table (:py:meth:`table_path`).
            params: Parameters the table depends on.
            compute: Zero-argument callable that computes the table.
            create_even_if_exist: Compute the table even if it is stored; the stored copy is then replaced (when
                *save_after_calculation*), so the next call reads the new table.
            save_after_calculation: Store a computed table.

        Returns:
            pandas.DataFrame: The table.
        """
        if not create_even_if_exist:
            stored = hs.load_table(path, params, logger=self.log)
            if stored is not None:
                return stored
        df: pd.DataFrame = compute()
        if save_after_calculation:
            df = df.reset_index(drop=True)
            self._store(path, df, params)
        return df

    def get_table(
        self,
        table_key: str,
        usecols: list | None = None,
        create_even_if_exist: bool = False,
        save_after_calculation: bool = True,
    ) -> pd.DataFrame:
        """Return a raw Ensembl table of the manager's release, from the repository or downloaded.

        The table is read from its file in :py:attr:`tables_directory` if one exists (one file per table, release
        and column set); otherwise it is downloaded (:py:meth:`download_table`) and, if *save_after_calculation*,
        stored. A raw table depends only on the release's core database, so it is shared by every manager of the
        assembly whatever its form or window.

        Args:
            table_key (str): Name of the MySQL table (e.g. ``"gene"``, ``"xref"``, ``"mapping_session"``).
            usecols (list[str] | None): Column subset to retrieve. ``None`` (default) selects *all* columns.
            create_even_if_exist (bool): Download the table even if it is stored, and replace the stored copy.
            save_after_calculation (bool): Store a downloaded table.

        Returns:
            pandas.DataFrame: The requested raw table, its columns in the order of *usecols* when given, otherwise in
                the schema's order.

        Raises:
            ValueError: If *usecols* is an empty list or not a list.
        """
        if not (usecols is None or (isinstance(usecols, list) and len(usecols) > 0)):
            raise InvalidArgumentError("Empty 'usecols' parameter, or 'usecols' is not a list.")
        cols = sorted(str(c) for c in usecols) if usecols is not None else None
        name = f"mysql_{table_key}" + (f"__cols={'+'.join(cols)}" if cols else "")
        path = self.table_path(name)
        params = self._table_params(name, columns=cols)
        df = self._cached(
            path,
            params,
            lambda: self.download_table(table_key, usecols),
            create_even_if_exist,
            save_after_calculation,
        )
        if usecols is not None and list(df.columns) != list(usecols):
            df = df[list(usecols)]
        return df

    def _ftp_db_dir_url(self) -> str:
        """Return the HTTPS directory URL for the current core database dump (from the core-database index)."""
        core_index = self._get_core_db_index(organism=self.organism, assembly=self.assembly, **self._core_index_kwargs)
        db_dir = core_index.get("db_dir_url_by_release", {}).get(int(self.release))
        if isinstance(db_dir, str) and db_dir:
            return db_dir

        db_name = self.mysql_database
        release = int(self.release)
        for root in self._ftp_mysql_root_candidates(organism=self.organism, assembly=self.assembly, release=release):
            candidate = f"{root}{db_name}/"
            resp = https_get(candidate, what=f"the dump directory of {db_name}", stream=True)
            if resp is not None:
                resp.close()
                return candidate

        raise FileNotFoundError(
            f"Ensembl has no MySQL dump directory for {self.organism!r} release {release} "
            f"assembly {int(self.assembly)} (expected db {db_name!r})."
        )

    def _ftp_schema_url(self) -> str:
        """Return the schema-dump URL (``*.sql.gz`` or ``*.sql.gz.bz2``) of the current DB dump directory.

        Resolved once per dump directory and process: the conventional name is asked first, a directory listing
        only when Ensembl answers that it does not exist.

        Returns:
            str: The URL.
        """
        db_dir_url = self._ftp_db_dir_url()
        cached = type(self)._ftp_schema_url_cache.get(db_dir_url)
        if cached is not None:
            return cached
        url = self._resolve_ftp_schema_url(db_dir_url)
        type(self)._ftp_schema_url_cache[db_dir_url] = url
        return url

    def _resolve_ftp_schema_url(self, db_dir_url: str) -> str:
        """Find the schema dump in *db_dir_url* (see :py:meth:`_ftp_schema_url`)."""
        db_name = self.mysql_database
        for url in (f"{db_dir_url}{db_name}.sql.gz", f"{db_dir_url}{db_name}.sql.gz.bz2"):
            resp = https_get(url, what=f"the schema dump of {db_name}", stream=True)
            if resp is not None:
                resp.close()
                return url

        # Slow path: directory listing (needed for some releases where Ensembl ships mismatched names).
        listing = https_get(db_dir_url, what=f"the dump listing of {db_name}")
        if listing is None:
            raise FileNotFoundError(f"Ensembl dump directory does not exist: {db_dir_url!r}")
        sql_files = [
            f
            for f in self._parse_apache_dir_listing_files(listing.text)
            if f.endswith(".sql.gz") or f.endswith(".sql.gz.bz2")
        ]
        if not sql_files:
            raise FileNotFoundError(f"No `.sql.gz` schema dump found in Ensembl directory: {db_dir_url!r}")

        # Heuristic: prefer the schema file whose name matches organism + assembly, closest release number.
        m_db = re.match(r"^(?P<org>.+)_core_(?P<rel>[0-9]+)_(?P<asm>[0-9]+)(?P<patch>[a-z]*)$", str(db_name))
        if m_db:
            org = m_db.group("org")
            asm = int(m_db.group("asm"))
            rel_target = int(self.release)
            scored: list[tuple[int, int, str]] = []
            for fname in sql_files:
                base = fname
                for suffix in (".sql.gz.bz2", ".sql.gz"):
                    if base.endswith(suffix):
                        base = base[: -len(suffix)]
                        break
                m = re.match(r"^(?P<org>.+)_core_(?P<rel>[0-9]+)_(?P<asm>[0-9]+)(?P<patch>[a-z]*)$", base)
                if not m or m.group("org") != org or int(m.group("asm")) != asm:
                    continue
                rel = int(m.group("rel"))
                scored.append((abs(rel - rel_target), -rel, fname))
            if scored:
                scored.sort()
                return f"{db_dir_url}{scored[0][2]}"

        # Final fallback: pick the first (stable) in sorted order.
        return f"{db_dir_url}{sorted(sql_files)[0]}"

    #: MySQL column types read as text on both routes (so the type of a column never depends on its content).
    _TEXT_SQL_TYPES = frozenset(
        {
            "char",
            "varchar",
            "tinytext",
            "text",
            "mediumtext",
            "longtext",
            "enum",
            "set",
            "date",
            "datetime",
            "timestamp",
            "time",
            "year",
        }
    )

    @classmethod
    def _is_text_sql_type(cls, sql_type: str) -> bool:
        """Return whether a MySQL column type (e.g. ``"varchar(128)"``, ``"enum('a','b')"``) is read as text."""
        base = re.split(r"[\s(]", str(sql_type).strip().lower(), maxsplit=1)[0]
        return base in cls._TEXT_SQL_TYPES

    @classmethod
    def _ftp_schema_for_sql_url(cls, sql_url: str) -> dict[str, list[tuple[str, str]]]:
        """Return ``{table: [(column, MySQL type), ...]}`` parsed from an Ensembl ``<db>.sql.gz`` schema dump."""
        cached = cls._ftp_schema_cache.get(sql_url)
        if cached is not None:
            return cached

        resp = https_get(sql_url, what="an Ensembl schema dump")
        if resp is None:
            raise FileNotFoundError(f"Ensembl schema dump does not exist: {sql_url!r}")
        data = resp.content
        # Some releases ship nested compression: `.sql.gz.bz2` → bz2(gzip(sql)).
        if sql_url.endswith(".bz2") or data.startswith(b"BZh"):
            data = bz2.decompress(data)
        if sql_url.endswith(".gz") or data.startswith(b"\x1f\x8b"):
            data = gzip.decompress(data)
        sql_text = data.decode("utf-8", errors="replace")

        schema: dict[str, list[tuple[str, str]]] = {}
        create_re = re.compile(
            r"CREATE TABLE(?: IF NOT EXISTS)? `([^`]+)` \((.*?)\)\s*ENGINE",
            flags=re.DOTALL,
        )
        for match in create_re.finditer(sql_text):
            table = match.group(1)
            block = match.group(2)
            columns = re.findall(r"^\s*`([^`]+)`\s+([A-Za-z]+(?:\([^)]*\))?)", block, flags=re.MULTILINE)
            if columns:
                schema[str(table)] = [(str(c), str(t)) for c, t in columns]

        if not schema:
            raise BuildError(f"No CREATE TABLE blocks found in Ensembl schema dump: {sql_url!r}")

        cls._ftp_schema_cache[sql_url] = schema
        return schema

    @staticmethod
    @contextmanager
    def _open_decompressed_http_text(url: str):
        """Yield a text stream for an Ensembl dump file, handling `.gz`, `.bz2`, and nested `.gz.bz2`.

        Args:
            url: The dump file.

        Yields:
            The decompressed text stream, or ``None`` if Ensembl answers that the file does not exist.
        """
        # Some environments/proxies apply HTTP-level content encoding to already-compressed dump files.
        # If urllib3 decodes that content, its Content-Length accounting can break and surface as
        # `IncompleteRead(..., negative_remaining)` during streaming reads. Force identity transfer and
        # disable urllib3 decoding so we always operate on the raw bytes indicated by the URL suffix.
        resp = https_get(
            url,
            what="an Ensembl table dump",
            stream=True,
            headers={"Accept-Encoding": "identity", "Connection": "close"},
        )
        if resp is None:
            yield None
            return
        try:
            if hasattr(resp.raw, "decode_content"):
                resp.raw.decode_content = False
            if hasattr(resp.raw, "enforce_content_length"):
                # Some mirrors/proxies report an incorrect Content-Length for streamed downloads
                # (notably when applying HTTP-level compression). We rely on the decompressor to
                # validate integrity instead of urllib3's length accounting.
                resp.raw.enforce_content_length = False

            stream: Any = resp.raw
            # Outer layer: bzip2 (used for many `*.txt.gz.bz2` and some `*.sql.gz.bz2`).
            if url.endswith(".bz2"):
                stream = bz2.BZ2File(stream)
            # Inner layer: gzip (used by Ensembl for `.gz` and nested `.gz.bz2`).
            if ".gz" in url:
                stream = gzip.GzipFile(fileobj=stream)

            text = io.TextIOWrapper(stream, encoding="utf-8", errors="replace", newline="")
            try:
                yield text
            finally:
                text.close()
        finally:
            resp.close()

    def _schema_of(self, table_key: str) -> list[tuple[str, str]]:
        """Return the ``[(column, MySQL type)]`` of *table_key* in the release's schema dump.

        Args:
            table_key: The table.

        Returns:
            list[tuple[str, str]]: Columns and their MySQL types, in schema order.

        Raises:
            EnsemblSchemaError: If the release has no table *table_key*.
        """
        sql_url = self._ftp_schema_url()
        schema = self._ftp_schema_for_sql_url(sql_url)
        if table_key not in schema:
            raise EnsemblSchemaError(
                f"Table {table_key!r} not found in Ensembl schema dump {sql_url!r}.", table=table_key
            )
        return schema[table_key]

    def _download_table_from_ftp(self, table_key: str, usecols: list[str] | None = None) -> pd.DataFrame:
        """Download a table from Ensembl's HTTPS MySQL dumps (no direct MySQL connection).

        The table file is requested directly (``<table>.txt.gz``, then ``<table>.txt.gz.bz2`` if Ensembl answers that
        the first does not exist). Columns of text types are read as text (see :py:meth:`_is_text_sql_type`).

        Args:
            table_key: The table.
            usecols: Columns to keep (in this order); all by default.

        Returns:
            pandas.DataFrame: The table.

        Raises:
            EnsemblSchemaError: If the release has no such table, or the table lacks an element of *usecols*.
            EnsemblDownloadError: If the dump cannot be read (after the retries).
        """
        db_dir_url = self._ftp_db_dir_url()
        typed = self._schema_of(table_key)
        columns = [c for c, _ in typed]
        text_columns = [c for c, t in typed if self._is_text_sql_type(t)]

        if usecols is not None:
            missing = [c for c in usecols if not isinstance(c, str) or c not in columns]
            if missing:
                raise EnsemblSchemaError(
                    f"Missing columns in {table_key!r}: {missing}", table=table_key, missing_columns=tuple(missing)
                )

        max_attempts = max(1, int(settings.ENSEMBL.rest_attempts))
        last_exc: Exception | None = None
        df: pd.DataFrame | None = None
        table_url: str | None = None
        for attempt in range(max_attempts):
            try:
                for candidate in (f"{db_dir_url}{table_key}.txt.gz", f"{db_dir_url}{table_key}.txt.gz.bz2"):
                    with self._open_decompressed_http_text(candidate) as fh:
                        if fh is None:
                            continue
                        table_url = candidate
                        # Decode MySQL's dump escaping (escaped tabs/newlines/backslashes, NULL as `\N`) so that the
                        # rows are the ones the MySQL server would return.
                        df = read_mysql_dump(fh, columns=columns, usecols=usecols, text_columns=text_columns)
                        break
                last_exc = None
                break
            except EnsemblDownloadError:
                raise  # already retried by `https_get`
            except Exception as exc:  # noqa: BLE001 - retry only on known transient classes
                last_exc = exc
                if attempt < max_attempts - 1 and self._is_retryable_http_read_error(exc):
                    sleep_s = float(settings.ENSEMBL.rest_retry_wait) * (2**attempt)
                    self.log.warning(
                        "Transient error while downloading %r from %s (attempt %d/%d): %s. Retrying in %.1fs.",
                        table_key,
                        db_dir_url,
                        attempt + 1,
                        max_attempts,
                        exc.__class__.__name__,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                break

        if last_exc is not None:
            raise EnsemblDownloadError(
                f"Unable to read table {table_key!r} of {self.mysql_database!r} from Ensembl HTTPS dumps "
                f"({db_dir_url!r})."
            ) from last_exc
        if df is None:
            raise EnsemblDownloadError(
                f"The schema of {self.mysql_database!r} has table {table_key!r}, but Ensembl's dump directory "
                f"{db_dir_url!r} has no file for it."
            )

        info_usecols = " for following columns: " + ", ".join(usecols) + "." if usecols else "."
        self.log.info(
            f"Raw table for `{table_key}` on ensembl release `{self.release}` "
            f"was downloaded via HTTPS/FTP dumps ({table_url}){info_usecols}"
        )
        return df

    @staticmethod
    def _typed_mysql_frame(rows: list[tuple], typed: list[tuple[str, str]], table_key: str) -> pd.DataFrame:
        """Build a frame from MySQL rows with the column types the HTTPS route gives.

        Text-type columns (:py:meth:`_is_text_sql_type`) become text in pandas' default text dtype (bytes decoded as
        UTF-8, dates and times in MySQL's text form ``YYYY-MM-DD[ HH:MM:SS]``, as in the dumps); the other columns are
        numbers (``NULL`` makes an integer column float, as in the dumps).

        Args:
            rows: The rows of the query.
            typed: ``[(column, MySQL type)]`` of the selected columns.
            table_key: The table, for the messages.

        Returns:
            pandas.DataFrame: The table.

        Raises:
            ValueError: If a value of a text column cannot be decoded.
        """
        names = [c for c, _ in typed]
        text_dtype = pd.Series(["a"]).dtype
        data: dict[str, Any] = {}
        for k, (col, sql_type) in enumerate(typed):
            values = np.empty(len(rows), dtype=object)
            values[:] = [row[k] for row in rows]
            if DatabaseManager._is_text_sql_type(sql_type) and pd.api.types.infer_dtype(values, skipna=True) in (
                "string",
                "empty",
            ):
                data[col] = pd.Series(values, dtype=text_dtype).where(pd.notna(values), np.nan)
            elif DatabaseManager._is_text_sql_type(sql_type):
                out = np.empty(len(values), dtype=object)
                for i, v in enumerate(values):
                    if v is None:
                        out[i] = np.nan
                    elif isinstance(v, str):
                        out[i] = v
                    elif isinstance(v, (bytes, bytearray)):
                        try:
                            out[i] = bytes(v).decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise TableFormatError(
                                f"Table {table_key!r} column {col!r} holds undecodable bytes."
                            ) from exc
                    else:
                        out[i] = str(v)  # datetime, date, Decimal...: the text MySQL prints
                data[col] = pd.Series(out, dtype=text_dtype)
            else:
                series = pd.Series([np.nan if v is None else v for v in values], dtype=object)
                data[col] = pd.to_numeric(series)
        return pd.DataFrame(data, columns=names)

    def download_table(self, table_key: str, usecols: list | None = None) -> pd.DataFrame:
        """Download a raw Ensembl MySQL table and return it as a DataFrame.

        When the release is on the live Ensembl MySQL service and its port is usable (see
        :py:mod:`idtrack.build.ensembl_access`: the port answered its probe and the access mode is not ``"https"``), the
        table is selected from the core schema with a server-side cursor (rows are streamed, not held twice);
        otherwise it is read from Ensembl's HTTPS dumps (:py:meth:`_download_table_from_ftp`). Both routes give the
        same rows with the same column types: columns of text types (character, enumeration, date and time) are
        text, the others numbers. Only a failure to reach the MySQL server makes the HTTPS route take over (with a
        warning that names the cause); a query or data error is raised. Public code is expected to call
        :py:meth:`DatabaseManager.get_table`, which wraps this helper with caching.

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
            EnsemblSchemaError: If the release has no table *table_key*, or the table lacks an element of
                *usecols* (:py:class:`idtrack.errors.EnsemblSchemaError`).
            ValueError: If the query returns text that cannot be decoded.

        Notes:
            If the HTTPS dumps cannot be read, :py:class:`idtrack.EnsemblDownloadError` propagates.
        """
        if usecols == []:
            usecols = None

        def _quote_identifier(identifier: str) -> str:
            if not isinstance(identifier, str):
                raise InvalidArgumentError(f"Unsafe SQL identifier type: {type(identifier)!r}")
            if not re.fullmatch(r"[0-9A-Za-z_]+", identifier):
                raise InvalidArgumentError(f"Unsafe SQL identifier: {identifier!r}")
            return f"`{identifier}`"

        core_index = self._get_core_db_index(organism=self.organism, assembly=self.assembly, **self._core_index_kwargs)
        mysql_releases = set(core_index.get("releases_on_mysql", []) or [])
        # If the requested release is not present on the live MySQL service, go straight to HTTPS dumps.
        if int(self.release) not in mysql_releases:
            return self._download_table_from_ftp(table_key, usecols=usecols)

        # If MySQL is unreachable (common when outbound 3306/5306 is blocked) or the access mode is "https", use the
        # HTTPS dumps.
        if not reachability.mysql_port_usable(int(self.mysql_settings["port"])):
            return self._download_table_from_ftp(table_key, usecols=usecols)

        which_mysql_server = dict(**self.mysql_settings, **{"database": self.mysql_database})
        table_sql = _quote_identifier(table_key)
        try:
            with pymysql.connect(**which_mysql_server) as connection:
                with connection.cursor() as cur1:
                    cur1.execute(f"SHOW COLUMNS FROM {table_sql}")  # noqa: S608
                    typed = []
                    for row in cur1.fetchall():
                        name, sql_type = row[0], row[1]
                        name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
                        sql_type = sql_type.decode("utf-8") if isinstance(sql_type, bytes) else str(sql_type)
                        typed.append((name, sql_type))
                if not typed:
                    raise TableFormatError(f"Table {table_key!r} has no columns.")
                if any(self._column_sep in c for c, _ in typed):
                    raise TableFormatError(f"Column names of {table_key!r} contain the separator {self._column_sep!r}.")

                if usecols is not None:
                    names = {c for c, _ in typed}
                    missing = [c for c in usecols if not isinstance(c, str) or c not in names]
                    if missing:
                        raise EnsemblSchemaError(
                            f"Missing columns in {table_key!r}: {missing}",
                            table=table_key,
                            missing_columns=tuple(missing),
                        )
                    by_name = dict(typed)
                    typed = [(c, by_name[c]) for c in usecols]
                select_sql = ", ".join(_quote_identifier(c) for c, _ in typed)

                # A server-side cursor streams the rows instead of materialising the whole result first.
                rows: list[tuple] = []
                with connection.cursor(pymysql.cursors.SSCursor) as cur2:
                    cur2.execute(f"SELECT {select_sql} FROM {table_sql}")  # noqa: S608
                    while True:
                        chunk = cur2.fetchmany(100_000)
                        if not chunk:
                            break
                        rows.extend(chunk)
        except EnsemblSchemaError:
            # The live schema answered: the dumps have the same schema, so asking them cannot help.
            raise
        except Exception as exc:
            if not self._is_mysql_connection_failure(exc):
                raise
            # A connection that worked before and fails now: do not try this port again for a while.
            reachability.mark_unreachable(str(self.mysql_settings["host"]), int(self.mysql_settings["port"]), str(exc))
            self.log.warning(
                "MySQL %s:%s failed while reading %r (%s: %s); reading it from the HTTPS dumps instead.",
                self.mysql_settings["host"],
                self.mysql_settings["port"],
                table_key,
                exc.__class__.__name__,
                exc,
            )
            return self._download_table_from_ftp(table_key, usecols=usecols)

        df = self._typed_mysql_frame(rows, typed, table_key)
        del rows
        info_usecols = " for following columns: " + ", ".join(usecols) + "." if usecols else "."
        self.log.info(f"Raw table for `{table_key}` on ensembl release `{self.release}` was downloaded{info_usecols}")
        return df

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
                :py:data:`idtrack.ids.FORMS` (``"gene"``, ``"transcript"``, or
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
            raise InvalidArgumentError(f"Form has to be one of {ids.FORMS}. Input form is `{form}`.")

        return stable_id_version, usecols_core, usecols_asso

    def create_ids(self, form: str) -> pd.DataFrame:
        """Retrieve and normalise raw Ensembl identifier records for the requested molecular form.

        This method pulls the appropriate MySQL table(s) for *form*, copes with schema differences
        across Ensembl releases (e.g. the historical ``*_stable_id`` split tables), coerces data
        types, and standardises column names so that downstream graph-building steps all consume the
        same shape.  Every id must carry a version (:py:meth:`DatabaseManager._require_versions`): a release
        whose ids have none, or only some, is refused here, so no derived table exists for it.

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

                All ID columns are ``int64`` except the ``*_stable_id`` strings; version columns are ``int64``.

        Raises:
            ValueError: If ids of the release carry no version (organisms without versioned ids are not supported).
        """
        # Determine which columns are interesting for each form.
        stable_id_version, usecols_core, usecols_asso = DatabaseManager._determine_usecols_ids(form)

        try:
            # In order to have the same column order with below exception code block.
            usecols = usecols_core + [i for i in usecols_asso if i not in usecols_core]
            df = self.get_table(f"{form}", usecols=usecols, save_after_calculation=self.store_raw_always)
            # Earlier versions has different table for stable_id.
            # When there is no associated column for a given table, the following error will be raised.

        except EnsemblSchemaError as exc:
            # The table has no stable_id/version columns: in older Ensembl releases they are in separate
            # `<form>_stable_id` tables. Only this schema fact leads to the fallback; a failed download or a failure to
            # cache the table keeps its own error instead of being reported as a missing `<form>_stable_id` table.
            self.log.debug(
                "Columns stable_id/version not found in %r table for release %s; "
                "trying separate %s_stable_id table (older Ensembl schema). Original error: %s",
                form,
                self.release,
                form,
                exc,
            )
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
        df = df.rename(columns={"stable_id": f"{form}_stable_id", "version": f"{form}_version"})
        df = df.drop_duplicates(ignore_index=True)  # Remove duplicates if exists
        df = df.reset_index(drop=True)

        return self._require_versions(df, f"{form}_version", form)

    def create_relation_current(self) -> pd.DataFrame:
        """Build a current-release gene-transcript-translation mapping table.

        The routine fetches the *raw* stable-ID/​version tables for genes,
        transcripts and translations via :py:meth:`DatabaseManager.get_db`, merges
        them into a single wide frame, and then delegates to
        :py:meth:`DatabaseManager._create_relation_helper` to harmonise version
        columns and compress the information into three canonical node labels
        (``"<stable_id>.<version>"``).  The resulting mapping is the authoritative
        per-release link between molecular forms and is consumed by downstream
        graph-building utilities such as :py:meth:`idtrack.GraphMaker.construct_graph`.

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
        tgp = tgp.drop(columns=["gene_id", "transcript_id", "translation_id"], errors="raise")

        return self._create_relation_helper(tgp)

    def _create_relation_helper(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert an ID/version matrix into the canonical three-column relationship table.

        The helper is used by :py:meth:`DatabaseManager.create_relation_current`
        and is not intended for direct use.  It validates the incoming frame, keeps the
        versions of genes and transcripts as integers and those of translations as
        ``NaN``-compatible floats (a transcript without translation has none), casts all
        stable-ID columns to string, and finally compresses each *ID + version* pair into
        the compact node label used throughout *ID-track* graphs.

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
            raise TableFormatError(
                f"Expected 6 columns {sorted(cols)}, got {len(df.columns)} columns: {list(df.columns)}."
            )

        # Every transcript has a gene, both with a version; a transcript without translation has neither a
        # translation id nor a translation version.
        for col in ["gene", "transcript"]:
            df[f"{col}_version"] = df[f"{col}_version"].astype(int)
        df["translation_version"] = (
            df["translation_version"].where(df["translation_stable_id"].notna()).astype(float, errors="raise")
        )

        for col in ["gene_stable_id", "translation_stable_id", "transcript_stable_id"]:
            df[col] = df[col].fillna("").astype(str)

        res = pd.DataFrame()
        for col in ["gene", "transcript", "translation"]:
            dm = self.change_form(col)
            res[col] = dm.id_ver_from_df(df[[f"{col}_stable_id", f"{col}_version"]])

        # Drop duplicates if exists and return.
        res = res.drop_duplicates(ignore_index=True)
        res = res.reset_index(drop=True)
        return res

    def create_id_history(self, narrow: bool) -> pd.DataFrame:
        """Retrieve historical relationships between successive Ensembl stable IDs.

        Build a cross-release lineage table mapping every obsolete ID version to its immediate successor for the
        configured *organism* and *form*, over every mapping session that the release's tables record (all earlier
        releases; :py:meth:`get_db` restricts it to the manager's window).  The information is assembled from the Core tables
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
                :py:data:`idtrack.ids.ID_VERSION_DELIMITER` is found inside any ``*_stable_id`` field, indicating
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
            sm = sm.drop(
                columns=[
                    "mapping_session_id",
                    "type",
                    "old_db_name",
                    "new_db_name",
                    "old_assembly",
                    "new_assembly",
                    "created",
                ],
            )

        # A side of an event without an id (a birth, a retirement) has no version; a side with an id has one.
        sm = sm.reset_index(drop=True)
        for side in ("old", "new"):
            has_id = sm[f"{side}_stable_id"].notna()
            versions = pd.to_numeric(sm[f"{side}_version"], errors="coerce").where(has_id)
            self._require_versions(pd.DataFrame({"version": versions[has_id]}), "version", self.form)
            sm[f"{side}_version"] = versions.astype(float)
        # Convert np.nan's to "" so that saving to hdf5 file is not a problem.
        for col in ["new_stable_id", "old_stable_id"]:
            sm[col] = sm[col].fillna("")
            sm[col] = sm[col].astype(str)
        for col in ["score", "old_release", "new_release"]:
            sm[col] = sm[col].astype(float)
        sm["score"] = sm["score"].replace(0.0, np.nan)  # 0 means no information according to ensembl.

        # Check the delimiter is not in the ID.
        if not np.all(sm["new_stable_id"].str.find(ids.ID_VERSION_DELIMITER) == -1):
            raise TableFormatError(
                f"new_stable_id column contains delimiter {ids.ID_VERSION_DELIMITER!r} which is not allowed."
            )
        if not np.all(sm["old_stable_id"].str.find(ids.ID_VERSION_DELIMITER) == -1):
            raise TableFormatError(
                f"old_stable_id column contains delimiter {ids.ID_VERSION_DELIMITER!r} which is not allowed."
            )

        # Every mapping session the release records is kept: the table does not depend on the manager's window, which
        # `get_db` applies when the table is read (a table computed for one window is then never served to another).
        sm = sm.drop_duplicates(ignore_index=True)  # Remove duplicates if exists
        sm = sm.reset_index(drop=True)
        return sm

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
                * ``"database"`` - return a two-column summary (``name_db``, ``count``) for **all** external databases
                    (the content of the externals template: what a YAML can select).
                * ``"relevant-database"`` - as above, but restricted to databases flagged *Include: true*.
                    The counts are of the databases' cross references: the genes' display names, which a graph holds
                    whatever the YAML selects (:py:meth:`_with_display_names`), are no entry a YAML selects and are
                    not counted.
                    The special values ``"relevant"`` and ``"relevant-database"`` implicitly consult the cached
                    :py:attr:`external_inst` to honour the user's curated allow-list.

        Returns:
            pandas.DataFrame:
                * For ``"all"`` / ``"relevant"`` - seven-column frame
                    ``["release", "graph_id", "id_db", "name_db", "ensembl_identity", "xref_identity", "is_label"]``
                    holding one row per Ensembl→external identifier edge. ``graph_id`` is the Ensembl stable ID
                    (+version), the two *identity* columns store Smith-Waterman percent identities (*float16*) for QC,
                    and ``is_label`` says whether the row is a label (a display name) rather than an accession.
                * For ``"database"`` / ``"relevant-database"`` - two-column frame
                    ``["name_db", "count"]`` giving the number of rows of each external database (most rows first,
                    ties by name). ``count`` is an ``int64``.

        Raises:
            ValueError: If *filter_mode* is not one of the accepted literals **or** if the YAML allow-list claims a
                database that is absent from the retrieved mappings—indicating the configuration and MySQL data are
                out of sync.
            EnsemblSchemaError: If ``identity_xref`` lacks columns other than its identity columns.

        Notes:
            *Synonym handling* - any synonym brought in from ``external_synonym`` is prefixed with
            :py:data:`ids.SYNONYM_PREFIX`, and its ``name_db`` is likewise prefixed so that synonym nodes remain
            distinguishable during graph building.
            *Caching* - this method always computes; :py:meth:`get_db` stores and reuses its results
            (``external``, ``external_database``, and ``external_relevant`` keyed by the enabled databases). The raw
            tables it reads are stored by :py:meth:`get_table`.
            *Restriction first* - with ``relevant`` the raw tables are restricted to the enabled databases before
            the join (identical result, a fraction of the memory). External ids and synonyms are stripped of
            surrounding white space; ids that are empty after it are dropped. The rows are in a total order (database,
            Ensembl id, external id, identities).
        """
        if filter_mode not in ("all", "relevant", "database", "relevant-database"):
            raise InvalidArgumentError(
                f"filter_mode must be 'all', 'relevant', 'database', or 'relevant-database', got {filter_mode!r}."
            )
        relevant = filter_mode in ("relevant", "relevant-database")
        isin_list: list[str] = list(self.external_inst.give_list_for_case(give_type="db")) if relevant else []
        filing = self.database_filing() if relevant else None
        selection = self._selection() if relevant else []
        ed, x, ox, ix = self._cross_reference_tables(selection if relevant else None)
        res = self._external_rows(ed, x, ox, ix)
        if relevant:
            if self.external_inst.settings_release() == int(self.release):
                self._check_selection(res, isin_list)
            else:  # the settings of an earlier release: a database it enables that this release lacks is left out
                absent = sorted(set(isin_list) - set(res["name_db"].unique()))
                if absent:
                    self.log.warning(
                        f"{self.organism} {self.form}, assembly {self.assembly}, release {self.release} takes "
                        f"the externals settings of release {self.external_inst.settings_release()}; it has no cross "
                        f"references of {absent}."
                    )
            res = res[res["name_db"].isin(selection)]
        if filter_mode not in ("all", "relevant"):
            # the template's content (what a YAML selects): display names are held whatever it selects, and a link is
            # one cross reference whatever versions of the accession Ensembl attached to it
            links = res.drop(columns=["version", "is_display"]).drop_duplicates(ignore_index=True)
            return self._count_databases(links)
        if self.form == ids.BACKBONE_FORM:
            res = self._with_display_names(res)
        if filing is not None:
            res = self._filed(res, filing)
        return res.reset_index(drop=True)

    # ------------------------------------------------------------------------------------------------------------
    # Which of Ensembl's databases a selected database is at each release (decision 104, idtrack.build.continuity)

    def _release_links(self) -> pd.DataFrame:
        """Return every cross reference of the release as a link: database name, identifier, gene (every form).

        Read from the raw tables (``xref``, ``object_xref``, ``external_db`` and the ids of the three forms); an
        accession is named by its database's name, a label by the name of its database's labels (:py:meth:`label_names`);
        synonyms are left out (they follow their entry). The gene of a transcript or a translation is its gene.

        Returns:
            pandas.DataFrame: ``name``, ``form`` (the Ensembl form the identifier is attached to), ``identifier``,
            ``gene`` (the gene's stable id).
        """
        m = {"save_after_calculation": self.store_raw_always}
        ed = self._numeric_key(self._external_databases_named(), "external_db_id", "external_db")
        x = self._numeric_key(self.get_table("xref", usecols=list(XREF_COLUMNS), **m), "external_db_id", "xref")
        ox = self.get_table(
            "object_xref", usecols=["ensembl_id", "ensembl_object_type", "xref_id", "object_xref_id"], **m
        )
        genes = self.get_db(f"idsraw_{ids.BACKBONE_FORM}")
        stable_of_gene = dict(zip(genes["gene_id"].astype(np.int64).tolist(), genes["gene_stable_id"].astype(str)))
        gene_of: dict[str, dict[int, str]] = {"gene": stable_of_gene}
        if "transcript" in self.available_form_of_interests:
            t = self.get_db("idsraw_transcript")
            gene_of["transcript"] = {
                int(k): stable_of_gene.get(int(v), "")
                for k, v in zip(t["transcript_id"].tolist(), t["gene_id"].tolist())
            }
            if "translation" in self.available_form_of_interests:
                p = self.get_db("idsraw_translation")
                gene_of["translation"] = {
                    int(k): gene_of["transcript"].get(int(v), "")
                    for k, v in zip(p["translation_id"].tolist(), p["transcript_id"].tolist())
                }
        form = ox["ensembl_object_type"].astype(str).str.lower().to_numpy(dtype=object)
        objects = pd.to_numeric(ox["ensembl_id"], errors="coerce").fillna(-1).astype(np.int64).tolist()
        gene = np.asarray([gene_of.get(f, {}).get(i, "") for f, i in zip(form.tolist(), objects)], dtype=object)
        ox = pd.DataFrame({"xref_id": pd.to_numeric(ox["xref_id"], errors="coerce"), "form": form, "gene": gene})
        ox = ox[ox["gene"] != ""].dropna(subset=["xref_id"])
        x = x.assign(xref_id=pd.to_numeric(x["xref_id"], errors="coerce")).dropna(subset=["xref_id"])
        joined = ox.merge(x, on="xref_id").merge(ed[["external_db_id", "db_name", "label_name"]], on="external_db_id")
        parts = [
            pd.DataFrame(
                {"name": joined[name], "form": joined["form"], "identifier": joined[value], "gene": joined["gene"]}
            )
            for name, value in (("db_name", "dbprimary_acc"), ("label_name", "display_label"))
        ]
        return pd.concat(parts, ignore_index=True)

    def _serving_manager(self, release: int) -> "DatabaseManager":
        """Return the manager of the release on the assembly that serves it (the manager's own where it has the release)."""
        if int(release) in {int(r) for r in self.available_releases}:
            return self.change_release(int(release))
        return self.change_release_auto_assembly(int(release))

    def database_filing(self) -> continuity.Filing:
        """Return which of Ensembl's database names each database the externals YAML selects holds.

        The selected databases are the ``(form, name)`` pairs the YAML enables anywhere (any assembly or release; a
        synonym's name by its entry's). The links of every release of the window are read on the assembly that serves it
        (:py:meth:`_release_links`) and compared between adjacent releases
        (:py:func:`idtrack.build.continuity.database_filing`). Computed once per process and window; a graph holds what
        it found (:py:attr:`idtrack.graph.GraphRecord.database_names`).

        Returns:
            idtrack.build.continuity.Filing: The names and where they go.
        """
        selected = self.external_inst.selected_units()
        snapshot = None if np.isinf(self.snapshot_release) else int(self.snapshot_release)
        key = (
            self.organism,
            int(self.first_release),
            snapshot,
            os.path.realpath(self.local_repository),
            tuple(sorted(selected)),
        )
        found = DatabaseManager._filing_cache.get(key)
        if found is None:
            found = DatabaseManager._filing_cache[key] = self._stored_filing(selected)
        return found

    def _stored_filing(self, selected: set[tuple[str, str]]) -> continuity.Filing:
        """Return the filing from the repository's stored tables, computing and storing it where none is stored.

        Stored with the tables of the window's last release (on the assembly that serves it, whichever manager asks),
        keyed by the window, the selection, the names of the labels and the version and bounds of the rule
        (:py:mod:`idtrack.build.continuity`).
        """
        last = max(int(r) for r in self.available_releases_all_assemblies if int(r) <= self.snapshot_release)
        manager = self.change_form(ids.BACKBONE_FORM)._serving_manager(last)
        digest = hashlib.sha256(json.dumps(sorted(selected)).encode("utf-8")).hexdigest()[:16]
        params: dict[str, Any] = {
            "selection": sorted(f"{f}:{n}" for f, n in selected),
            "first_release": int(self.first_release),
            "snapshot": None if np.isinf(self.snapshot_release) else int(self.snapshot_release),
            "labels": self._label_names_digest(),
            "share": continuity.CONTINUITY_SHARE,
            "scheme_share": continuity.SCHEME_SHARE,
            "rules": continuity.FILING_RULES_VERSION,
        }
        decided_path = manager.table_path(f"databasefiling-decided__sel={digest}")
        links_path = manager.table_path(f"databasefiling-links__sel={digest}")
        decided = hs.load_table(
            decided_path, manager._table_params(f"databasefiling-decided__sel={digest}", **params), logger=self.log
        )
        links = hs.load_table(
            links_path, manager._table_params(f"databasefiling-links__sel={digest}", **params), logger=self.log
        )
        if decided is not None and links is not None:
            return continuity.Filing.from_tables(decided, links)
        found = self._compute_filing(selected)
        decided, links = found.to_tables()
        if os.access(self.local_repository, os.W_OK):
            manager._store(
                decided_path, decided, manager._table_params(f"databasefiling-decided__sel={digest}", **params)
            )
            manager._store(links_path, links, manager._table_params(f"databasefiling-links__sel={digest}", **params))
        return found

    def _compute_filing(self, selected: set[tuple[str, str]]) -> continuity.Filing:
        """Compute the filing (:py:meth:`database_filing`) from the links of every release of the window."""
        names = self.label_names()
        database_of: dict[str, set[str]] = {name: {name} for name in names}
        for database, label in names.items():
            database_of.setdefault(label, set()).add(database)
        releases = sorted(int(r) for r in self.available_releases_all_assemblies if int(r) <= self.snapshot_release)
        self.log.info(
            f"The databases of {self.organism} are being read across its releases, to find the names under which "
            "Ensembl filed each selected database at each release."
        )
        return continuity.database_filing(
            lambda release: self.change_form(ids.BACKBONE_FORM)._serving_manager(release)._release_links(),
            releases,
            selected,
            database_of,
        )

    def _selection(self) -> list[str]:
        """Return the names whose rows the manager's ``external_relevant`` table holds.

        The databases the externals YAML enables for the manager's form, assembly and release, and every database that
        continues one of them (:py:meth:`database_filing`; a selected database's name on a form the YAML does not hold
        it on, where its links there continue it), with its synonyms where the YAML enables the selected database's.

        Returns:
            list[str]: The names, sorted.
        """
        prefix = ids.SYNONYM_PREFIX
        chosen = set(self.external_inst.give_list_for_case(give_type="db"))
        filing = self.database_filing()
        synonyms = self.external_inst.selected_names() if filing.renamed else set()
        for unit, forms in filing.renamed.items():
            name, _, form = unit.partition("@")
            if form and form != self.form:
                continue  # a selected name on another form than the manager's
            chosen.add(name)
            if any(prefix + target in synonyms for target in forms.values()):
                chosen.add(prefix + name)
        return sorted(chosen)

    def _filed(self, rows: pd.DataFrame, filing: continuity.Filing) -> pd.DataFrame:
        """Return rows of cross references filed under the selected database each is.

        A row of a database that continues a selected one is filed under it (by its identifier's form; a selected
        name on a form the YAML does not hold it on, in the runs of releases that continue it), and dropped where it
        continues none (an identifier of another form than the continuation's scheme, a release outside those runs),
        but for a gene's display name, which is a name of its gene whatever database holds it; a row of a selected
        database whose identifier has the scheme of another selected database, and whose link that one holds at some
        release, is filed under that one (:py:mod:`idtrack.build.continuity`). Synonym rows follow their entry.

        Args:
            rows: Rows of :py:meth:`_external_rows` (and :py:meth:`_with_display_names`) of the manager's form.
            filing: :py:meth:`database_filing`.

        Returns:
            pandas.DataFrame: The rows, the column ``name_db`` changed and rows left out where the rule says, without
            duplicates.
        """
        if not len(rows) or not (filing.renamed or filing.schemes):
            return rows
        prefix = ids.SYNONYM_PREFIX
        names = rows["name_db"].astype(str).tolist()
        values = rows["id_db"].astype(str).tolist()
        shown = rows["is_display"].to_numpy(dtype=bool) if "is_display" in rows.columns else np.zeros(len(rows), bool)
        bare_names = [n[len(prefix) :] if n.startswith(prefix) else n for n in names]
        bare_values = [v[len(prefix) :] if v.startswith(prefix) else v for v in values]
        forms = [continuity.identifier_form(v) for v in bare_values]
        release = int(self.release)
        own = {n for f, n in self.external_inst.selected_units() if f == self.form}
        filed: list[str | None] = []
        for name, form, display in zip(bare_names, forms, shown.tolist()):
            if name in own or not (
                name in filing.renamed or continuity.unit_of_form(name, self.form) in filing.renamed
            ):
                filed.append(name)
                continue
            target = filing.continued(name, form, self.form, release)
            filed.append(target if target is not None else name if display else None)
        scheme_forms = set(filing.schemes.values())
        candidates = [
            i
            for i, f in enumerate(forms)
            if f in scheme_forms and filed[i] is not None and not names[i].startswith(prefix)
        ]
        if candidates:
            genes = self._genes_of_rows(rows["graph_id"].astype(str).iloc[candidates].tolist())
            hashed, gene_hashes = continuity.link_hashes([bare_values[i] for i in candidates], genes)
            for j, i in enumerate(candidates):
                other = filing.scheme_of(str(filed[i]), forms[i], int(hashed[j]), int(gene_hashes[j]), release)
                if other is not None:
                    filed[i] = other
        keep = np.fromiter((f is not None for f in filed), dtype=bool, count=len(filed))
        new = [(prefix if n.startswith(prefix) else "") + (f or "") for n, f in zip(names, filed)]
        if new == names:
            return rows
        rows = rows.assign(name_db=pd.Series(new, index=rows.index, dtype=rows["name_db"].dtype))[keep]
        return _in_total_order(rows.drop_duplicates(ignore_index=True))

    def _genes_of_rows(self, graph_ids: list[str]) -> list[str]:
        """Return the gene stable id of each Ensembl id ``<id>.<version>`` of the manager's form (its gene's)."""
        if self.form == ids.BACKBONE_FORM:
            return [ids.split(g)[0] if ids.is_versioned_name(g) else g for g in graph_ids]
        relation = self.get_db("relationcurrent")
        gene_of = dict(zip(relation[self.form].astype(str).tolist(), relation[ids.BACKBONE_FORM].astype(str).tolist()))
        return [
            ids.split(gene_of[g])[0] if g in gene_of and ids.is_versioned_name(gene_of[g]) else "" for g in graph_ids
        ]

    def _with_display_names(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Return the cross references of the genes with every gene's display name of the release among them.

        A gene's display name at a release (``gene.display_xref_id``) is a name of the gene at that release, whatever
        database Ensembl filed it under and whether or not the externals YAML selects that database: Ensembl gave the
        display names of pig and macaque genes before release 61 from HGNC's projections (``HGNC_curated_gene``,
        ``HGNC_automatic_gene``), those of mouse genes before 52 from MGI under other names, and those of genes without
        a better name from UniProt's entry names, RefSeq or Rfam, often through a transcript or a translation. A
        display name that the gene's rows (of the gene form, selected) do not give it already as an identifier of some
        database is added as a label of the gene, under the name of the labels of the database it came from
        (:py:meth:`label_names`), so that the source of every name is kept (a projection is told from the authority's
        own name by its database). It is read as every value is read (:py:func:`_read_values`); a value without a letter
        or a digit is no name. The definitions of the databases hold for display names as for every row
        (:py:data:`BIOTYPE_RESTRICTED`: a miRBase name that Ensembl displays on a gene of another biotype, the host or
        antisense gene of the miRNA, is the name of the miRNA, not of that gene), whatever database Ensembl filed it
        under (:py:func:`rules_database`: ``miRBase_gene_name`` is miRBase's). Every row of a gene whose identifier is
        the gene's display name, added or held, is marked so (``is_display``): Ensembl's own statement of the gene's
        name at that release, which ranks a gene first among the genes that carry the name then
        (:py:meth:`idtrack.TheGraph.displays`).

        Args:
            rows: The cross references of the gene form (:py:meth:`_external_rows`, restricted to the selection).

        Returns:
            pandas.DataFrame: The rows with the display names added and marked, in the total order of
            :py:meth:`_external_rows`.
        """
        genes = self._gene_annotation()
        shown = genes.dropna(subset=["display_xref_id"])
        if not len(shown):
            return rows
        ids_of = self.get_db(f"idsraw_{self.form}")
        names = pd.Series(
            self.id_ver_from_df(ids_of[self._identifiers].reset_index(drop=True)),
            index=ids_of[f"{self.form}_id"].astype(int).to_numpy(),
        )
        m = {"save_after_calculation": self.store_raw_always}
        x = self.get_table("xref", usecols=list(XREF_COLUMNS), **m)
        x = self._numeric_key(x, "external_db_id", "xref")
        x = x[pd.to_numeric(x["xref_id"], errors="coerce").isin(pd.to_numeric(shown["display_xref_id"]))]
        ed = self._external_databases_named()
        ed = self._numeric_key(ed, "external_db_id", "external_db")
        label_of = dict(zip(ed["external_db_id"].tolist(), ed["label_name"].tolist()))
        name_of = dict(zip(ed["external_db_id"].tolist(), ed["db_name"].tolist()))
        biotype_of = dict(zip(shown["gene_id"].astype(int).tolist(), shown["biotype"].tolist()))
        xref_of = {
            int(xref_id): (label_of.get(database), label, name_of.get(database))
            for xref_id, database, label in zip(
                pd.to_numeric(x["xref_id"], errors="coerce").tolist(),
                x["external_db_id"].tolist(),
                x["display_label"].tolist(),
            )
            if xref_id == xref_id  # not NaN
        }
        found = []
        for gene_id, xref_id in zip(
            shown["gene_id"].astype(int).tolist(), pd.to_numeric(shown["display_xref_id"], errors="coerce").tolist()
        ):
            if xref_id != xref_id or gene_id not in names.index:
                continue
            database, label, db_name = xref_of.get(int(xref_id), (None, None, None))
            if database is None or label is None or label != label:
                continue
            rules = None if db_name is None else rules_database(db_name)
            if rules in BIOTYPE_RESTRICTED and biotype_of.get(gene_id) not in BIOTYPE_RESTRICTED[rules]:
                continue
            found.append((names[gene_id], str(label).strip(), str(database)))
        if not found:
            return rows
        display = pd.DataFrame(found, columns=["graph_id", "id_db", "name_db"])
        display["id_db"] = _read_values(display["id_db"].astype(object))
        display = display[display["id_db"].astype(str).str.contains(_ALPHANUMERIC, regex=True)]
        shown_pairs = set(zip(display["graph_id"].astype(str).tolist(), display["id_db"].astype(str).tolist()))
        pairs = list(zip(rows["graph_id"].astype(str).tolist(), rows["id_db"].astype(str).tolist()))
        rows = rows.assign(
            is_display=np.fromiter((pair in shown_pairs for pair in pairs), dtype=bool, count=len(pairs))
        )
        held = set(pairs)
        display = display[[(g, i) not in held for g, i in zip(display["graph_id"], display["id_db"])]]
        if not len(display):
            return rows
        display = display.drop_duplicates(ignore_index=True)
        display = display.assign(
            release=np.full(len(display), self.release, dtype=np.uint8),
            ensembl_identity=np.full(len(display), np.nan, dtype=np.float16),
            xref_identity=np.full(len(display), np.nan, dtype=np.float16),
            is_label=np.ones(len(display), dtype=bool),
            version=np.zeros(len(display), dtype=np.int32),
            is_display=np.ones(len(display), dtype=bool),
        )[list(rows.columns)]
        res = pd.concat([rows, display.astype({"id_db": rows["id_db"].dtype, "name_db": rows["name_db"].dtype})])
        return _in_total_order(res)

    def _cross_reference_tables(
        self, selection: list[str] | None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return the raw cross-reference tables of the release for the manager's form, restricted to a selection.

        With a selection, the raw tables are restricted to the databases it can produce before any join (the result
        of :py:meth:`create_external_db` is the same; the join of all cross references is never built). External ids
        and synonyms are stripped of surrounding white space, and the synonyms are appended to the ``xref`` rows as
        further rows whose label is the synonym with :py:data:`ids.SYNONYM_PREFIX`.

        Args:
            selection: The database names of the externals YAML (``None``: every database).

        Returns:
            ``external_db``, ``xref`` (with the synonym rows), ``object_xref`` of the form, and ``identity_xref`` (with
            the identity columns ``ensembl_identity`` and ``xref_identity``).

        Raises:
            EnsemblSchemaError: If ``identity_xref`` lacks columns other than its identity columns.
        """
        prefix = ids.SYNONYM_PREFIX
        m = {"save_after_calculation": self.store_raw_always}
        ed = self._external_databases_named()
        ed = self._numeric_key(ed, "external_db_id", "external_db")
        if selection is not None:
            # A database is named by `db_name` (accession rows) or by the name of its labels, with the synonym prefix
            # for synonym rows; keep the external_db rows that can produce one of the enabled names.
            bare = {n[len(prefix) :] if n.startswith(prefix) else n for n in selection}
            ed = ed[ed["db_name"].isin(bare) | ed["label_name"].isin(bare)]

        x = self.get_table("xref", usecols=list(XREF_COLUMNS), **m)
        x = self._numeric_key(x, "external_db_id", "xref")
        if selection is not None:
            x = x[x["external_db_id"].isin(ed["external_db_id"])]
        es = self._numeric_key(
            self.get_table("external_synonym", usecols=["xref_id", "synonym"], **m), "xref_id", "synonym"
        )
        if selection is not None:
            es = es[es["xref_id"].isin(x["xref_id"])]
        ox = self.get_table(
            "object_xref", usecols=["ensembl_id", "ensembl_object_type", "xref_id", "object_xref_id"], **m
        )
        ox = ox[ox["ensembl_object_type"].str.lower() == self.form.lower()]
        if selection is not None:
            ox = ox[ox["xref_id"].isin(x["xref_id"])]
        ix = self._identity_table()
        ix = ix[ix["object_xref_id"].isin(ox["object_xref_id"])]

        # External ids and synonyms without surrounding white space (Ensembl stores some synonyms with a trailing
        # blank, which no query written without it would match).
        x = x.assign(
            dbprimary_acc=x["dbprimary_acc"].str.strip(),
            display_label=x["display_label"].str.strip(),
        )
        # Make entities in synonyms table appended to the xref file as additional lines
        es = es.merge(x, how="inner", on="xref_id", validate="many_to_one")
        es = es.drop(columns=["display_label"]).rename(columns={"synonym": "display_label"})
        es["display_label"] = prefix + es["display_label"].str.strip()
        x = pd.concat([x, es], ignore_index=True)
        return ed, x, ox, ix

    def _external_db(self) -> pd.DataFrame:
        """Return the release's ``external_db`` (``external_db_id``, ``db_name``, ``db_display_name``)."""
        return self.get_table(
            "external_db",
            usecols=["external_db_id", "db_name", "db_display_name"],
            save_after_calculation=self.store_raw_always,
        )

    def label_names(self) -> dict[str, str]:
        """Return the name of every database's label rows in the manager's window (:py:func:`label_names_of_databases`).

        Read from the ``external_db`` table of every release of every assembly of the organism in the window (small
        tables, read once per window and process).

        Returns:
            dict[str, str]: ``{db_name: name of its label rows}``.
        """
        snapshot = None if np.isinf(self.snapshot_release) else int(self.snapshot_release)
        key = (self.organism, int(self.first_release), snapshot, os.path.realpath(self.local_repository))
        found = DatabaseManager._label_names_cache.get(key)
        if found is None:
            rows: list[tuple[int, str, str | None, frozenset[str]]] = []
            for assembly in organisms.assemblies_by_priority(self.organism):
                index = self._get_core_db_index(
                    organism=self.organism, assembly=int(assembly), **self._core_index_kwargs
                )
                releases = [int(r) for r in index["releases"] if self.first_release <= int(r) <= self.snapshot_release]
                if not releases:
                    continue
                manager = self.change_assembly(assembly, last_possible_release=True)
                for release in releases:
                    ed = manager.change_release(release)._external_db()
                    names_of_release = frozenset(ed["db_name"].dropna().astype(str))
                    rows.extend(
                        (release, str(name), None if pd.isna(display) else str(display), names_of_release)
                        for name, display in zip(ed["db_name"].tolist(), ed["db_display_name"].tolist())
                        if not pd.isna(name)
                    )
            found = DatabaseManager._label_names_cache[key] = label_names_of_databases(rows)
        return found

    def one_gene_only_names(self) -> set[str]:
        """Return the database names of the accessions and labels of :py:data:`ONE_GENE_ONLY` in the window.

        A database whose gene names are derived from one of them (:py:func:`rules_database`: ``miRBase_gene_name``)
        is judged with it.
        """
        names = self.label_names()
        judged = set(ONE_GENE_ONLY) | {database for database in names if rules_database(database) in ONE_GENE_ONLY}
        return judged | {label_name(database, names) for database in judged}

    def _label_names_digest(self) -> str:
        """Return a short digest of :py:meth:`label_names` (part of the key of a stored table of cross references)."""
        text = json.dumps(sorted(self.label_names().items()))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def renamed_label_names(self) -> dict[str, str]:
        """Return the names that Ensembl gives a database's labels in the window and idtrack does not, with idtrack's.

        The same holds for their synonyms, with the synonym prefix. A name that is also a database's own name
        (``db_name``) is not listed (it names that database's accessions), nor is a name Ensembl gave the labels of
        several databases.

        Returns:
            dict[str, str]: ``{name Ensembl used: name of the label rows}``.
        """
        label_names = self.label_names()
        renamed: dict[str, set[str]] = {}
        for assembly in organisms.assemblies_by_priority(self.organism):
            index = self._get_core_db_index(organism=self.organism, assembly=int(assembly), **self._core_index_kwargs)
            releases = [int(r) for r in index["releases"] if self.first_release <= int(r) <= self.snapshot_release]
            if not releases:
                continue
            manager = self.change_assembly(assembly, last_possible_release=True)
            for release in releases:
                ed = manager.change_release(release)._external_db()
                for name, display in zip(ed["db_name"].tolist(), ed["db_display_name"].tolist()):
                    if pd.isna(name) or pd.isna(display) or str(display) == label_names.get(str(name)):
                        continue
                    renamed.setdefault(str(display), set()).add(label_names[str(name)])
        # A name Ensembl gave the labels of several databases says nothing about which one is meant: not listed.
        unique = {old: next(iter(new)) for old, new in renamed.items() if len(new) == 1}
        own = set(label_names) | set(label_names.values())
        prefix = ids.SYNONYM_PREFIX
        result = {old: new for old, new in unique.items() if old not in own}
        result.update({f"{prefix}{old}": f"{prefix}{new}" for old, new in unique.items()})
        return dict(sorted(result.items()))

    def _external_databases_named(self) -> pd.DataFrame:
        """Return the release's ``external_db`` with the name of every database's label rows.

        The name of a database's label rows is the same at every release of the window (:py:meth:`label_names`).

        Returns:
            pandas.DataFrame: ``external_db_id``, ``db_name``, ``db_display_name``, ``label_name``.
        """
        ed = self._external_db()
        names = self.label_names()
        return ed.assign(label_name=[label_name(str(name), names) for name in ed["db_name"].tolist()])

    def _numeric_key(self, df: pd.DataFrame, col: str, table_name: str) -> pd.DataFrame:
        """Return *df* with a join key as numbers, without the rows whose key is not a number (logged).

        Tables loaded via HTTPS/FTP dumps can contain join-key columns as strings. Some older releases also include
        malformed/continuation lines that shift text into numeric ID columns. Pandas refuses merges when dtypes
        disagree (e.g. int64 vs object), so the key is normalised here.
        """
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
            return df
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = numeric.isna() & df[col].notna()
        if bad.any():
            example = df.loc[bad, col].iloc[0]
            self.log.warning(
                "Dropping %d malformed rows from `%s` where `%s` is not numeric (e.g. %r).",
                int(bad.sum()),
                table_name,
                col,
                example,
            )
            df = df.loc[~bad].copy()
            numeric = numeric.loc[~bad]
        df[col] = numeric
        return df

    def _identity_table(self) -> pd.DataFrame:
        """Return ``identity_xref`` with the columns ``ensembl_identity``, ``xref_identity`` and ``object_xref_id``.

        Old releases (e.g. 48) name the identity columns ``target_identity`` and ``query_identity``; a release whose
        table has neither gets ``NaN`` identities.

        Raises:
            EnsemblSchemaError: If the table lacks columns other than its identity columns.
        """
        m = {"save_after_calculation": self.store_raw_always}
        try:
            return self.get_table("identity_xref", usecols=["ensembl_identity", "xref_identity", "object_xref_id"], **m)
        except EnsemblSchemaError as e:
            if not ({"ensembl_identity", "xref_identity"} & set(e.missing_columns)):
                raise
        self.log.info(
            "Using legacy column names (target_identity, query_identity) for identity_xref table "
            f"in Ensembl release {self.release}."
        )
        try:
            ix = self.get_table("identity_xref", usecols=["target_identity", "query_identity", "object_xref_id"], **m)
            return ix.rename(columns={"target_identity": "ensembl_identity", "query_identity": "xref_identity"})
        except EnsemblSchemaError:
            self.log.warning(
                f"identity_xref table missing identity columns for Ensembl release {self.release}. "
                "Using NaN values for ensembl_identity and xref_identity."
            )
            ix = self.get_table("identity_xref", usecols=["object_xref_id"], **m)
            ix["ensembl_identity"] = np.nan
            ix["xref_identity"] = np.nan
            return ix

    def _external_rows(self, ed: pd.DataFrame, x: pd.DataFrame, ox: pd.DataFrame, ix: pd.DataFrame) -> pd.DataFrame:
        """Join the cross-reference tables to the ids of the form: one row per (Ensembl id, external id, database).

        Accessions are named by the database's name, labels by the one name of its labels in the window
        (:py:meth:`label_names`; a label that repeats its accession is no row of its own where Ensembl names the labels
        like the accessions); a synonym row's database name carries :py:data:`ids.SYNONYM_PREFIX`. The labels of
        :py:data:`DISPLAY_LABEL_ONLY` are kept where they are the gene's display name, the identifiers of
        :py:data:`BIOTYPE_RESTRICTED` on genes of those biotypes (those of :py:data:`ONE_GENE_ONLY` are judged by the
        graph, over all its releases). Every value is read as a typed identifier is read (stray punctuation and quotes
        at its ends removed, Unicode normalised: ``A1BG;`` is ``A1BG``). An identifier without a letter or a digit
        (``.``, ``-``) is no identifier, nor is a synonym that carries UniProt's evidence text (``{ECO:...}``). Rows
        without an Ensembl id, a database or an identifier are dropped. The rows are in a total order (database,
        Ensembl id, external id, identities), without duplicates.

        Args:
            ed: ``external_db``.
            x: ``xref`` with the synonym rows (:py:meth:`_cross_reference_tables`).
            ox: ``object_xref`` of the form.
            ix: ``identity_xref``.

        The version of an accession (``xref.version``, Ensembl's version of the sequence the cross reference attaches)
        is kept on the rows whose identifier is the accession (``version``; a label that is the accession itself
        carries it too); labels, synonyms and the accessions of a database that gives no version carry ``0``.
        ``is_display`` is ``False`` here (:py:meth:`_with_display_names` marks the genes' display names).

        Returns:
            pandas.DataFrame: ``release``, ``graph_id``, ``id_db``, ``name_db``, ``ensembl_identity``,
            ``xref_identity``, ``is_label``, ``version``, ``is_display``.
        """
        prefix = ids.SYNONYM_PREFIX
        identities = ["ensembl_identity", "xref_identity"]
        db_id, db_name, id_graph = "id_db", "name_db", "graph_id"
        a = self.get_db(f"idsraw_{self.form}")
        comb = pd.merge(
            a,
            ox,
            how="left",
            left_on=f"{self.form}_id",
            right_on="ensembl_id",
            validate="one_to_many",
        )
        comb = comb.merge(x, how="left", on="xref_id", validate="many_to_many")
        comb = comb.merge(ed, how="left", on="external_db_id", validate="many_to_one")
        comb = comb.merge(ix, how="left", on="object_xref_id", validate="many_to_one")

        # Node name "ID.Version" once per id of the form, then attached by the internal key.
        names = pd.Series(self.id_ver_from_df(a[self._identifiers].reset_index(drop=True)), index=a[f"{self.form}_id"])
        graph_ids = comb[f"{self.form}_id"].map(names)
        del a
        label_kept, kept = self._cross_reference_masks(comb)
        # The version of the accession the cross reference attaches: digits, ``0`` where the database gives none (a
        # synonym row carries the version of its entry and is no accession).
        written = comb["version"].astype(object).where(comb["version"].notna(), "").astype(str).str.strip()
        version = pd.to_numeric(written.where(written.str.fullmatch(r"[0-9]{1,9}"), "0"), errors="coerce")
        version = np.array(version.fillna(0).astype(np.int32).to_numpy(), dtype=np.int32)
        version[comb["display_label"].astype(object).str.startswith(prefix, na=False).to_numpy(dtype=bool)] = 0
        label_is_accession = (comb["display_label"] == comb["dbprimary_acc"]).to_numpy(dtype=bool)

        # One row per (Ensembl id, external id, database): labels with the name of the labels, accessions with the
        # database's name.
        parts = []
        for id_col, name_col, mask in (("display_label", "label_name", label_kept), ("dbprimary_acc", "db_name", kept)):
            is_label = id_col == "display_label"
            part = pd.DataFrame(
                {
                    id_graph: graph_ids.to_numpy(),
                    db_id: comb[id_col].to_numpy(),
                    db_name: comb[name_col].to_numpy(),
                    identities[0]: comb[identities[0]].to_numpy(),
                    identities[1]: comb[identities[1]].to_numpy(),
                    "is_label": np.full(len(comb), is_label, dtype=bool),
                    "version": np.where(label_is_accession, version, 0) if is_label else version,
                    "is_display": np.zeros(len(comb), dtype=bool),
                }
            )
            parts.append(part[mask])
        del comb, graph_ids
        res = pd.concat(parts, axis=0, ignore_index=True)
        del parts
        bare_id = res[db_id].where(
            ~res[db_id].astype(object).str.startswith(prefix, na=False), res[db_id].str[len(prefix) :]
        )
        res = res[~(res[id_graph].isna() | res[db_name].isna() | res[db_id].isna()) & (bare_id != "")]
        # A value is read as a typed string is read (decisions 83 and 84: Unicode's invisible characters, forms and
        # dashes; the punctuation that ends a sentence, quotation marks around it), so that the graph holds the name a
        # database wrote with the punctuation of the text it came from (UniProt's ``A1BG;``) as the name.
        res[db_id] = _read_values(res[db_id])
        # An identifier is written with a letter or a digit; a synonym is not UniProt's evidence for one.
        bare_id = res[db_id].where(
            ~res[db_id].astype(object).str.startswith(prefix, na=False), res[db_id].str[len(prefix) :]
        )
        is_identifier = bare_id.astype(str).str.contains(_ALPHANUMERIC, regex=True) & ~(
            res[db_id].astype(str).str.startswith(prefix)
            & res[db_id].astype(str).str.contains(_EVIDENCE_TEXT, regex=False)
        )
        res = res[is_identifier.to_numpy(dtype=bool)]

        # Labels and accessions that are equal, and a synonym equal to its label, give the same row twice.
        res = _in_total_order(res).drop_duplicates(ignore_index=True)
        res.insert(0, "release", np.full(len(res), self.release, dtype=np.uint8))

        # A synonym row's database is the database name with the synonym prefix: a synonym is not the identifier it
        # stands beside, and the search uses it as a bridge only.
        is_synonym = res[db_id].astype(object).str.startswith(prefix).to_numpy(dtype=bool)
        res[db_name] = np.where(is_synonym, prefix, "").astype(object) + res[db_name].astype(object)
        res[db_name] = res[db_name].astype(pd.Series(["a"]).dtype)
        return res

    def _cross_reference_masks(self, comb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return which joined rows give a label row and which give an accession row (see :py:meth:`_external_rows`).

        Args:
            comb: The cross references of the form joined to its ids (``xref_id``, ``db_name``,
                ``display_label``, ``dbprimary_acc`` and the form's internal id).

        Returns:
            The masks of the label rows and of the accession rows.
        """
        kept = np.ones(len(comb), dtype=bool)
        # Naming the labels changes the name of rows, never their number: where Ensembl names a database's labels like
        # its accessions (display name equal to the database's name at that release), a label that repeats its
        # accession was one row with it and stays so; elsewhere both rows are kept as Ensembl gives them.
        same_name = (comb["db_display_name"] == comb["db_name"]).to_numpy(dtype=bool)
        label_kept = ~(same_name & (comb["display_label"] == comb["dbprimary_acc"]).to_numpy(dtype=bool))
        # the database whose definitions hold for each row: its own, or the one its gene names are derived from
        rules = comb["db_name"].map(lambda name: rules_database(name) if isinstance(name, str) else name)
        restricted = (rules.isin(set(BIOTYPE_RESTRICTED)) | comb["db_name"].isin(DISPLAY_LABEL_ONLY)).to_numpy(
            dtype=bool
        )
        if not restricted.any():
            return label_kept, kept
        genes = self._gene_annotation()
        gene_of = self._gene_of_form(comb)
        biotype = gene_of.map(dict(zip(genes["gene_id"], genes["biotype"])))
        display = gene_of.map(dict(zip(genes["gene_id"], genes["display_xref_id"])))
        for database, biotypes in BIOTYPE_RESTRICTED.items():
            outside = (rules == database).to_numpy(dtype=bool) & ~biotype.isin(biotypes).to_numpy(dtype=bool)
            kept &= ~outside
            label_kept &= ~outside
        for database in DISPLAY_LABEL_ONLY:
            not_display = (comb["db_name"] == database).to_numpy(dtype=bool) & ~(
                pd.to_numeric(comb["xref_id"], errors="coerce").to_numpy()
                == pd.to_numeric(display, errors="coerce").to_numpy()
            )
            label_kept &= ~not_display
        return label_kept, kept

    def _gene_of_form(self, comb: pd.DataFrame) -> pd.Series:
        """Return the internal gene id of every joined row of the manager's form (a transcript's or translation's gene)."""
        if self.form in ("gene", "transcript"):
            return comb["gene_id"]
        transcripts = self.get_db("idsraw_transcript")
        return comb["transcript_id"].map(dict(zip(transcripts["transcript_id"], transcripts["gene_id"])))

    def _gene_annotation(self) -> pd.DataFrame:
        """Return what every gene of the release is and where it lies: :py:data:`GENE_ANNOTATION_COLUMNS` of ``gene``."""
        return self.get_table(
            "gene", usecols=list(GENE_ANNOTATION_COLUMNS), save_after_calculation=self.store_raw_always
        )

    def _check_selection(self, res: pd.DataFrame, selection: list[str]) -> None:
        """Refuse a selection of the externals YAML that names databases the release's tables do not have.

        Raises:
            ExternalsError: If an enabled database has no cross references at the release.
        """
        available_databases = set(res["name_db"].unique())
        missing_databases = sorted(d for d in selection if d not in available_databases)
        if missing_databases:
            raise ExternalsError(
                f"The externals YAML enables {missing_databases} for {self.organism} {self.form}, assembly "
                f"{self.assembly}, release {self.release}, but Ensembl's tables of that release have "
                f"no such cross references. Correct the YAML "
                f"({self.external_inst.file_name_modified_yaml(location='configured')!r})."
            )

    def create_database_content(self) -> pd.DataFrame:
        """Retrieve external-database metadata for every assembly, release, and form.

        The helper iterates over every genome assembly configured for the current organism in
        :py:attr:`idtrack.organisms.ORGANISM_TABLE`, every available Ensembl release for each assembly,
        and every identifier *form* supported by the package, downloading the ``external_database`` table for
        each combination.  The resulting frames are concatenated, enriched with ``assembly``, ``release``,
        ``form``, and ``organism`` columns, and returned to the caller.

        Returns:
            pandas.DataFrame: External-database relationships augmented with assembly, release, form, and organism
                columns.
        """
        frames: list[pd.DataFrame] = []
        assemblies_processed: list[int] = []
        assemblies_without_release: list[int] = []

        for k in organisms.assemblies_by_priority(self.organism):
            # An assembly is left out only when its complete index has no release in the window; a failure to build
            # the index raises (a template made without an assembly would silently lack its databases).
            core_index = self._get_core_db_index(organism=self.organism, assembly=int(k), **self._core_index_kwargs)
            if not any(self.first_release <= int(r) <= self.snapshot_release for r in core_index["releases"]):
                assemblies_without_release.append(int(k))
                self.log.warning(
                    f"Assembly {k} of {self.organism} has no Ensembl release in the window "
                    f"{self.first_release}..{self.snapshot_release}; it has no entries in the template YAML."
                )
                continue
            self.log.info(f"Processing assembly {k} for {self.organism}...")
            dm_assembly = self.change_assembly(k, last_possible_release=True)
            assemblies_processed.append(int(k))
            for j in dm_assembly.available_form_of_interests:  # For all assemblies possible.
                for i in dm_assembly.available_releases:
                    self.log.info(
                        f"Database content is being created for "
                        f"`{self.organism}`, assembly `{k}`, form `{j}`, ensembl release `{i}`"
                    )
                    df_temp = dm_assembly.change_release(i).change_form(j).get_db("external_database")
                    frames.append(df_temp.assign(assembly=k, release=i, form=j))

        self.log.info(
            f"Assembly processing complete for {self.organism}: processed {sorted(assemblies_processed)}, "
            f"without a release in the window: {sorted(assemblies_without_release) or 'none'}."
        )
        df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        df["organism"] = self.organism
        return df.reset_index(drop=True)

    def create_release_id(self) -> pd.DataFrame:
        """Return deduplicated stable-identifier/version pairs for the current form and release.

        Raw identifiers are fetched via :py:meth:`DatabaseManager.get_db`, trimmed to the canonical columns, and
        sanity-checked.  Two
        integrity rules are enforced: (1) the delimiter
        :py:data:`idtrack.ids.ID_VERSION_DELIMITER` must **not** appear inside any stable identifier, and
        (2) every stable identifier must be unique after deduplication.  Violations raise
        :py:class:`idtrack.errors.TableFormatError`.

        Returns:
            pandas.DataFrame: Two-column dataframe ``[{form}_stable_id, {form}_version]`` with duplicates removed.

        Raises:
            ValueError: If the delimiter is present inside any stable identifier or if identifiers are not
                unique after deduplication.
        """
        dbm_the_ids = self.get_db(f"idsraw_{self.form}")
        dbm_the_ids = dbm_the_ids[self._identifiers]

        if not np.all(dbm_the_ids[f"{self.form}_stable_id"].str.find(ids.ID_VERSION_DELIMITER) == -1):
            raise TableFormatError("The delimiter is in Ensembl IDs.")

        dbm_the_ids = dbm_the_ids.drop_duplicates(keep="first")
        if not dbm_the_ids[f"{self.form}_stable_id"].is_unique:
            raise TableFormatError("The stable IDs are not unique")

        dbm_the_ids = dbm_the_ids.reset_index(drop=True)
        dbm_the_ids = dbm_the_ids.drop_duplicates()
        return dbm_the_ids

    def create_gene_region(self) -> pd.DataFrame:
        """Tell for every gene of the release where it lies, and whether that is on the reference sequence.

        Ensembl annotates genes on the chromosomes and scaffolds of an assembly (its reference sequence) and also on
        sequence regions beside it: alternative loci (haplotypes), assembly patches and LRG records. A gene on such a
        region is a copy of, or an alternative to, a gene of the reference sequence and has a stable id of its own.
        The core database marks these regions with the attribute :py:data:`NON_REFERENCE_ATTRIBUTE`
        (``seq_region_attrib`` joined to ``attrib_type``), in every release and for every kind of region; the table
        ``assembly_exception`` lists only the patches and haplotypes among them (not the LRG records), so it does not
        say which genes lie outside the reference sequence; it says which region of the reference a patch or a
        haplotype stands for (:py:meth:`create_assembly_exceptions`).

        The place of a gene is its sequence region (by name), its start and end (1-based, inclusive, as Ensembl
        counts) and its strand on the assembly of the release, and what the region is (its kind, one of
        :py:data:`idtrack.graph.genes.PLACE_KINDS`): the mitochondrial genome where the region carries
        :py:data:`MITOCHONDRIAL_ATTRIBUTE`, else a chromosome where its coordinate system is
        :py:data:`CHROMOSOME_COORD_SYSTEM` or the region carries :py:data:`KARYOTYPE_ATTRIBUTE`, else a scaffold (or
        another sequence below a chromosome).

        The table is a function of five raw tables of the release (``gene`` with its sequence region and place,
        ``seq_region``, ``coord_system``, ``seq_region_attrib`` and ``attrib_type``) and of the gene ids
        (``idsraw_gene``).

        Returns:
            pandas.DataFrame: One row per gene of the release: ``gene_stable_id``, ``gene_version`` (as in the ``ids``
            table of the gene form), ``non_reference`` (``True`` for a gene on a region outside the reference
            sequence; a release whose core database defines no such attribute or marks no region has ``False``
            throughout), ``seq_region`` (the region's name), ``start``, ``end``, ``strand`` and ``kind``.

        Raises:
            ValueError: If a gene of the id table has no sequence region in the ``gene`` table.
        """
        gene_ids = self.get_db(f"idsraw_{ids.BACKBONE_FORM}")
        regions = self._gene_annotation()
        names, beside, kind_of = self._sequence_regions()
        region_of = dict(zip(regions["gene_id"].astype(int), regions["seq_region_id"]))
        gene_regions = gene_ids["gene_id"].astype(int).map(region_of)
        missing = gene_regions.isna().to_numpy()
        if missing.any():
            raise TableFormatError(
                f"{int(missing.sum())} genes of {self.organism} release {self.release} have no sequence region "
                f"in the `gene` table, e.g. {gene_ids['gene_stable_id'][missing].iloc[0]!r}."
            )
        result = gene_ids[["gene_stable_id", "gene_version"]].copy()
        result["non_reference"] = gene_regions.astype(np.int64).isin(beside).to_numpy(dtype=bool)
        name_of = dict(zip(names["seq_region_id"].astype(int), names["name"].astype(str)))
        result["seq_region"] = gene_regions.astype(np.int64).map(name_of).to_numpy()
        if result["seq_region"].isna().any():
            raise TableFormatError(
                f"Genes of {self.organism} release {self.release} lie on sequence regions that the `seq_region` table "
                f"does not have, e.g. {result['gene_stable_id'][result['seq_region'].isna()].iloc[0]!r}."
            )
        genes = gene_ids["gene_id"].astype(int)
        for column, source in (
            ("start", "seq_region_start"),
            ("end", "seq_region_end"),
            ("strand", "seq_region_strand"),
        ):
            result[column] = genes.map(
                dict(zip(regions["gene_id"].astype(int), regions[source].astype(int)))
            ).to_numpy()
        result["kind"] = [kind_of(region) for region in gene_regions.astype(np.int64).tolist()]
        return result.drop_duplicates(ignore_index=True)

    def _sequence_regions(self) -> tuple[pd.DataFrame, set[int], Callable[[int], str]]:
        """Return the release's sequence regions, those outside the reference sequence, and what each region is.

        Read from ``seq_region``, ``coord_system``, ``seq_region_attrib`` and ``attrib_type``: a region outside the
        reference sequence carries :py:data:`NON_REFERENCE_ATTRIBUTE`; its kind (one of
        :py:data:`idtrack.graph.genes.PLACE_KINDS`) is the mitochondrial genome where it carries
        :py:data:`MITOCHONDRIAL_ATTRIBUTE`, else a chromosome where its coordinate system is
        :py:data:`CHROMOSOME_COORD_SYSTEM` or it carries :py:data:`KARYOTYPE_ATTRIBUTE`, else a scaffold.

        Returns:
            tuple: ``seq_region`` (``seq_region_id``, ``name``, ``coord_system_id``), the ids of the regions outside
            the reference sequence, and the kind of a region by its id.
        """
        raw = {"save_after_calculation": self.store_raw_always}
        names = self.get_table("seq_region", usecols=["seq_region_id", "name", "coord_system_id"], **raw)
        systems = self.get_table("coord_system", usecols=["coord_system_id", "name"], **raw)
        attributes = self.get_table("attrib_type", usecols=["attrib_type_id", "code"], **raw)
        marks = self.get_table("seq_region_attrib", usecols=["seq_region_id", "attrib_type_id"], **raw)

        def marked(code: str) -> set[int]:
            codes = attributes.loc[attributes["code"] == code, "attrib_type_id"]
            return {int(i) for i in marks.loc[marks["attrib_type_id"].isin(codes), "seq_region_id"].dropna().unique()}

        beside = marked(NON_REFERENCE_ATTRIBUTE)
        mitochondrial = marked(MITOCHONDRIAL_ATTRIBUTE)
        chromosome_systems = {
            int(i) for i in systems.loc[systems["name"] == CHROMOSOME_COORD_SYSTEM, "coord_system_id"].dropna()
        }
        chromosomes = marked(KARYOTYPE_ATTRIBUTE) | {
            int(region)
            for region, system in zip(names["seq_region_id"].tolist(), names["coord_system_id"].tolist())
            if not pd.isna(system) and int(system) in chromosome_systems
        }

        def kind_of(region: int) -> str:
            return MITOCHONDRIAL if region in mitochondrial else CHROMOSOME if region in chromosomes else SCAFFOLD

        return names, beside, kind_of

    def create_assembly_exceptions(self) -> pd.DataFrame:
        """Return Ensembl's assembly exceptions of the release: which region of the reference each other sequence takes.

        An assembly patch, a haplotype (an alternative locus) or a pseudoautosomal region is, in Ensembl's core
        database, a sequence region of its own whose bases stand for a region of the reference sequence: the table
        ``assembly_exception`` gives, per such region, its bases (``seq_region_start`` to ``seq_region_end``), the
        region of the reference they stand for (``exc_seq_region_id`` from ``exc_seq_region_start`` to
        ``exc_seq_region_end``) and the type of the exception (``exc_type``: ``PATCH_FIX``, ``PATCH_NOVEL``, ``HAP``,
        ``PAR``). A gene on a patch or a haplotype is a copy of, or an alternative to, a gene of that reference region
        (:py:func:`idtrack.graph.genes.relation`); the Y copy of a gene of a pseudoautosomal region lies at the place of
        its X gene (:py:attr:`idtrack.TheGraph.repeated_regions`). An LRG record has no exception.

        The table is a function of five raw tables of the release (``assembly_exception``, ``seq_region``,
        ``coord_system``, ``seq_region_attrib`` and ``attrib_type``).

        Returns:
            pandas.DataFrame: One row per exception: ``seq_region`` (the name of the region), ``start``, ``end``,
            ``type``, ``ref_region`` (the name of the reference region), ``ref_start``, ``ref_end``, ``ref_kind``
            (its kind, one of :py:data:`idtrack.graph.genes.PLACE_KINDS`), sorted.
        """
        raw = {"save_after_calculation": self.store_raw_always}
        columns = ASSEMBLY_EXCEPTION_COLUMNS
        rows = self.get_table("assembly_exception", usecols=list(columns), **raw)
        names, _, kind_of = self._sequence_regions()
        name_of = dict(zip(names["seq_region_id"].astype(np.int64).tolist(), names["name"].astype(str).tolist()))
        out = []
        for region, start, end, kind, ref, ref_start, ref_end in zip(*(rows[c].tolist() for c in columns)):
            if any(pd.isna(v) for v in (region, start, end, ref, ref_start, ref_end)):
                continue
            mine, theirs = name_of.get(int(region)), name_of.get(int(ref))
            if mine is None or theirs is None:
                raise TableFormatError(
                    f"An assembly exception of {self.organism} release {self.release} names a sequence region that "
                    f"the `seq_region` table does not have ({int(region)} or {int(ref)})."
                )
            out.append((mine, int(start), int(end), str(kind), theirs, int(ref_start), int(ref_end), kind_of(int(ref))))
        frame = pd.DataFrame(
            sorted(set(out)),
            columns=["seq_region", "start", "end", "type", "ref_region", "ref_start", "ref_end", "ref_kind"],
        )
        return frame.astype({"start": np.int64, "end": np.int64, "ref_start": np.int64, "ref_end": np.int64})

    def assembly_chain(self, earlier: int, later: int) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        """Return Ensembl's mapping of one assembly of the organism onto another, where Ensembl publishes one.

        Ensembl's chain file of the pair (``<name>_to_<name>.chain.gz`` in the ``assembly_chain`` folder of a release,
        :py:mod:`idtrack.build.assembly_chain`) is looked for in the folders of the window's releases, newest first,
        downloaded once into the local repository and kept there (raw downloads are never removed).

        Args:
            earlier: The assembly key of the assembly mapped.
            later: The assembly key of the assembly it is mapped onto.

        Returns:
            tuple | None: The aligned blocks (:py:data:`idtrack.build.assembly_chain.BLOCK_COLUMNS`) and what a graph's
            record says of the file (``from``, ``to``, ``release``, ``sha256``, ``file``); ``None`` where Ensembl
            publishes no such file in those folders.
        """
        first, second = organisms.assembly_name(self.organism, int(earlier)), organisms.assembly_name(
            self.organism, int(later)
        )
        if not first or not second:
            return None
        names = (first, second)
        last = int(self.release) if np.isinf(self.snapshot_release) else int(self.snapshot_release)
        releases = range(int(self.first_release), last + 1)
        path, release = assembly_chain.fetch_chain(self.local_repository, self.organism, names[0], names[1], releases)
        if path is None or release is None:
            return None
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        info = {
            "file": assembly_chain.chain_file_name(*names),
            "from": int(earlier),
            "to": int(later),
            "release": int(release),
            "sha256": digest,
        }
        return assembly_chain.read_chain(path), info

    def create_external_all(self, all_external: bool = False) -> pd.DataFrame:
        """Download and collate cross-reference mappings from every supported genome assembly.

        The manager cycles through every genome assembly recognised for the current organism (ordered by
        :py:data:`idtrack.organisms.ORGANISM_TABLE`), fetches either the filtered
        *external_relevant* mapping table (when ``all_external=False``) or the full *external* mapping table
        (when ``all_external=True``) for each via :py:meth:`~DatabaseManager.get_db`, labels every row with
        its source assembly, and concatenates the tables, keeping one copy of every unique
        ``(release, graph_id, id_db, name_db, ensembl_identity, xref_identity, assembly)`` combination (duplicates
        are resolved within each assembly only). It bypasses the :py:meth:`~DatabaseManager.get_db` caching layer
        and never writes the result to the local repository.

        Args:
            all_external (bool): If ``False`` (default), restrict results to databases enabled in the externals
                YAML configuration (``external_relevant``). If ``True``, include all external databases of Ensembl
                (``external``).

        Returns:
            pandas.DataFrame: The cross-reference table with the columns ``release``, ``graph_id``, ``id_db``,
            ``name_db``, ``ensembl_identity``, ``xref_identity``, ``is_label`` and ``assembly``.
        """
        ass = set(self.external_inst.give_list_for_case(give_type="assembly"))
        table_key = "external" if all_external else "external_relevant"

        cfg = organisms.ORGANISM_TABLE[self.organism]
        if not all_external:
            # The genes' display names are held whatever the YAML selects (:py:meth:`_with_display_names`), and the
            # databases that continue a selected one at releases at which the YAML lists nothing for a form are read
            # with it (:py:meth:`database_filing`): every assembly with a core database at the release is read.
            ass |= set(cfg)
        ass_sorted = sorted({int(a) for a in ass if int(a) in cfg}, key=lambda a: cfg[a]["Priority"])
        frames: list[pd.DataFrame] = []
        for i in ass_sorted:  # sort according to per-organism priority (1 = newest)
            # An assembly is passed over only when its complete index has no core database at this release; a failure
            # to build the index raises (leaving the assembly out would silently drop its external edges).
            core_index = self._get_core_db_index(organism=self.organism, assembly=i, **self._core_index_kwargs)
            if int(self.release) not in core_index["port_for_release"]:
                self.log.info(
                    "create_external_all: %s assembly %s has no core database at release %s.",
                    self.organism,
                    i,
                    self.release,
                )
                continue
            df_temp = self.change_assembly(i).get_db(table_key)
            frames.append(df_temp.assign(assembly=i))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Rows equal in every column but the identity scores are one row, within an assembly.
        compare_columns = [i for i in df.columns if i != "assembly" and not i.endswith("_identity")] + ["assembly"]
        return df.drop_duplicates(keep="first", ignore_index=True, subset=compare_columns)

    def get_db(
        self,
        table_key: str,
        create_even_if_exist: bool = False,
        save_after_calculation: bool = True,
    ) -> pd.DataFrame:
        """Return a derived table, from the repository or computed.

        Every stored table is kept in a file of its own whose name carries everything the table depends on, so a
        table computed for one window or externals selection is never served for another:

        * ``external`` (all cross references of the form), ``external_database`` (row counts per database): keyed by
          release and form.
        * ``external_relevant``: the rows of the databases the externals YAML enables for this form, assembly and
          release; keyed by release, form and the sorted list of those databases (its digest in the file name, the
          list itself in the file, checked on read). ``external_relevant-database`` counts its rows (not stored).
        * ``idhistory`` and ``idhistory_narrow``: Ensembl's id history of the release, stored for every mapping
          session the release records (keyed by release and form) and restricted on reading to the sessions whose
          ``old_release`` lies in the manager's window.
        * ``relationcurrent``: gene, transcript and translation node names per transcript; keyed by release.
        * Not stored, because they are cheap functions of stored tables: ``idsraw_<form>`` (raw ids of *form*,
          :py:meth:`create_ids`), ``ids`` (ids and versions of the manager's form, :py:meth:`create_release_id`),
          ``generegion`` (which genes of the release lie outside the reference sequence,
          :py:meth:`create_gene_region`), ``assemblyexception`` (which region of the reference each patch, haplotype
          and pseudoautosomal region stands for, :py:meth:`create_assembly_exceptions`). The content of Ensembl's
          external databases and the list of the available ones are methods of their own
          (:py:meth:`create_database_content`, :py:meth:`create_available_databases`).

        Args:
            table_key (str): One of the indicators above.
            create_even_if_exist (bool): Compute a stored table again and replace the stored copy.
            save_after_calculation (bool): Store a computed table (ignored for the tables that are not stored).

        Returns:
            pandas.DataFrame: The requested table.

        Raises:
            ValueError: If *table_key* is not one of the indicators above.
        """
        split_ind = table_key.split("_")
        main_ind = split_ind[0]
        param1_ind = split_ind[1] if len(split_ind) > 1 else None
        if len(split_ind) > 2:
            raise InvalidArgumentError(f"`table_key` {table_key!r} has too many underscore-separated parts (max 2).")

        def cached(name: str, compute: Any, **params: Any) -> pd.DataFrame:
            return self._cached(
                self.table_path(name),
                self._table_params(name, **params),
                compute,
                create_even_if_exist,
                save_after_calculation,
            )

        if main_ind == "idsraw":
            if param1_ind not in self.available_form_of_interests:
                raise InvalidArgumentError(
                    f"'idsraw' should be used together with one "
                    f"of followings: {','.join(self.available_form_of_interests)}."
                )
            return self.create_ids(form=param1_ind)

        if main_ind == "ids" and param1_ind is None:
            return self.create_release_id()

        if main_ind == "generegion" and param1_ind is None:
            return self.create_gene_region()

        if main_ind == "assemblyexception" and param1_ind is None:
            return self.create_assembly_exceptions()

        if main_ind == "external" and param1_ind in (None, "database"):
            name = f"external{'-database' if param1_ind else ''}_{self.form}"
            return cached(
                name,
                lambda: self.create_external_db(filter_mode=param1_ind or "all"),
                rules=EXTERNAL_RULES_VERSION,
                labels=self._label_names_digest(),
            )

        if main_ind == "external" and param1_ind in ("relevant", "relevant-database"):
            selection = self._selection()
            digest = hashlib.sha256(json.dumps(selection).encode("utf-8")).hexdigest()[:16]
            name = f"external-relevant_{self.form}__dbs={digest}"
            rows = cached(
                name,
                lambda: self.create_external_db(filter_mode="relevant"),
                databases=selection,
                rules=EXTERNAL_RULES_VERSION,
                labels=self._label_names_digest(),
                filing=self.database_filing().digest(),
            )
            return rows if param1_ind == "relevant" else self._count_databases(rows)

        if main_ind == "idhistory" and param1_ind in (None, "narrow"):
            name = f"idhistory{'-narrow' if param1_ind else ''}_{self.form}"
            history = cached(name, lambda: self.create_id_history(narrow=param1_ind == "narrow"))
            in_window = (history["old_release"] >= self.first_release) & (
                history["old_release"] <= self.snapshot_release
            )
            return history[in_window].reset_index(drop=True)

        if main_ind == "relationcurrent" and param1_ind is None:
            return cached("relationcurrent", self.create_relation_current)

        raise InvalidArgumentError(f"Unexpected entry for `table_key`: {table_key!r}.")

    @staticmethod
    def _count_databases(rows: pd.DataFrame) -> pd.DataFrame:
        """Return ``name_db``/``count`` (rows per database), most rows first, ties by name."""
        counts = rows["name_db"].value_counts(sort=False)
        out = pd.DataFrame({"name_db": counts.index.astype(object), "count": counts.to_numpy(dtype=np.int64)})
        out = out.sort_values(["count", "name_db"], ascending=[False, True], kind="stable", ignore_index=True)
        out["name_db"] = out["name_db"].astype(pd.Series(["a"]).dtype)
        return out

    def id_ver_from_df(self, table: pd.DataFrame) -> list[str]:
        """Assemble fully qualified node names from a *stable-ID / version* DataFrame.

        This convenience routine converts a two-column frame—usually produced by
        :py:meth:`DatabaseManager.get_db` with the *ids* form—into the canonical node
        labels used throughout ID-track graphs (e.g. ``ENSG00000000001.1``).  It first
        validates that the input columns match :py:data:`self._identifiers`
        (typically ``["gene_stable_id", "gene_version"]`` or analogous for the
        current ``form``), then joins them with :py:func:`idtrack.ids.join_columns`.

        Args:
            table (pandas.DataFrame): Two-column frame containing the stable
                identifiers and their Ensembl version numbers.  The column order and
                names **must** exactly match ``self._identifiers``; otherwise an
                exception is raised.

        Returns:
            list[str]: Ordered list where each element is ``"<ID>.<version>"``; for a row
                without version (the empty id of a transcript that has no translation)
                the id itself.

        Raises:
            ValueError: If ``table`` does not contain the expected column
                names stored in :py:data:`self._identifiers`.
        """
        if list(table.columns) != list(self._identifiers):
            raise TableFormatError(
                f"Column names of `table` are not as expected. " f"{list(table.columns)} vs {self._identifiers}"
            )
        return ids.join_columns(table.iloc[:, 0].to_numpy(dtype=object), table.iloc[:, 1].to_numpy())

    def _require_versions(self, df: pd.DataFrame, version_column: str, form: str) -> pd.DataFrame:
        """Return *df* with the versions of its ids as integers; refuse ids that carry no version.

        idtrack follows ``<id>.<version>`` through the releases: the graph's nodes, the history and the tables that
        feed them are named that way. Ensembl gives the ids of some organisms no versions (*S. cerevisiae*:
        ``YAL001C``), in all or in some releases. Such an organism is not supported, and nothing is made up for it
        (no synthetic version, no table without versions): the id table of the release is refused here, and with it
        every table that is derived from it.

        Args:
            df: A table whose rows all have an id.
            version_column: Its column of versions.
            form: The form of the ids (``"gene"``, ``"transcript"``, ``"translation"``), for the message.

        Returns:
            pandas.DataFrame: *df* itself, with *version_column* as integers.

        Raises:
            ValueError: If the column holds no version, or a version for some of the ids only.
        """
        missing = pd.isna(df[version_column])
        if missing.any():
            found = "no version" if missing.all() else f"no version for {int(missing.sum())} of {len(df)} ids"
            raise BuildError(
                f"Organisms whose Ensembl ids carry no versions are not supported: the {form} ids of {self.organism} "
                f"have {found} at Ensembl release {self.release} (genome assembly {self.assembly}). "
                "idtrack follows `<id>.<version>` through the releases."
            )
        df[version_column] = df[version_column].astype(int)
        return df
