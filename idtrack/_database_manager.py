#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

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
from contextlib import contextmanager
from functools import cached_property
from typing import Any, SupportsIndex

import numpy as np
import pandas as pd
import pymysql

import idtrack._utils_hdf5 as hs
from idtrack._db import DB, EnsemblSchemaError
from idtrack._ensembl_access import EnsemblDownloadError, ensembl_access_mode, https_get, reachability, rest_get_json
from idtrack._external_databases import ExternalDatabases
from idtrack._mysql_dump import read_mysql_dump


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
    * :py:meth:`available_releases` — list releases of the assembly within the ignore window.
    * :py:meth:`change_release` — a new manager for another Ensembl release (likewise :py:meth:`change_form`,
      :py:meth:`change_assembly`, :py:meth:`change_release_auto_assembly`).
    * :py:meth:`get_table` — a raw Ensembl table (stored after the first download); :py:meth:`download_table`
      fetches one without storing it.
    * :py:meth:`get_db` — a derived table (ids, history, relations, cross references).
    * :py:meth:`create_external_all` — the cross references of every assembly the externals YAML enables for the
      release.
    * :py:attr:`organism`, :py:attr:`form`, :py:attr:`ensembl_release`,
      :py:attr:`genome_assembly` — core configuration knobs, surfaced for quick inspection.

    A manager's configuration does not change after construction; the ``change_*`` methods return new managers.
    Use the built-in :py:meth:`__str__` for a concise, human-readable dump of that state.
    """

    # Keyed by (organism, assembly key, DB.organism_table_version) so that `DB.register_organism` invalidates it.
    _core_db_index_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
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
    def _release_window(organism: str, genome_assembly: int) -> tuple[int, int | None]:
        """Return the releases an assembly may have: its ``ReleaseRange`` clipped to idtrack's floor (48).

        Both discovery paths (the MySQL probe and the HTTPS scan) accept only releases inside this window, so they
        agree on what an assembly is even where two assemblies share a core-database suffix.

        Args:
            organism: Ensembl species name.
            genome_assembly: Assembly key of the organism table.

        Returns:
            tuple[int, Optional[int]]: ``(first, last)``; ``last`` is ``None`` for an open-ended assembly.
        """
        first, last = DB.release_range(organism, int(genome_assembly))
        floor = int(min(DB.mysql_port_min_release.values()))
        return max(int(first), floor), last

    @classmethod
    def _ensembl_latest_release(cls) -> int:
        """Return Ensembl's latest release, asked from REST once per :py:attr:`idtrack.DB.rest_species_cache_seconds`.

        The answer expires (one day by default) so that a long-running process learns of a new release. If Ensembl
        REST does not answer (after its retries), :py:class:`idtrack.EnsemblRestError` propagates.

        Returns:
            int: The latest release.

        Raises:
            ValueError: If the answer is not the expected payload.
        """
        cached = cls._ensembl_latest_release_cache
        if cached is not None and time.monotonic() - cached[1] < float(DB.rest_species_cache_seconds):
            return int(cached[0])

        url = f"{DB.rest_server_api}/info/data"
        payload = rest_get_json("/info/data", what="the latest Ensembl release")
        releases = payload.get("releases") if isinstance(payload, dict) else None
        if not isinstance(releases, list) or not releases:
            raise ValueError(f"Unexpected Ensembl REST payload at {url!r}: expected non-empty 'releases' list.")

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
    def _ftp_mysql_root_candidates(cls, *, organism: str, genome_assembly: int, release: int) -> tuple[str, ...]:
        """Return candidate HTTPS roots to search for an Ensembl core DB directory."""
        base = f"https://{DB.ensembl_ftp_base}/pub/release-{int(release)}/mysql/"
        # GRCh37 became a separate archive once GRCh38 became the primary assembly (Ensembl release 76).
        if organism == "homo_sapiens" and int(genome_assembly) == 37 and int(release) >= 76:
            grch37 = f"https://{DB.ensembl_ftp_base}/pub/grch37/release-{int(release)}/mysql/"
            return (grch37, base)
        return (base,)

    @classmethod
    def _ftp_find_core_db_dir(
        cls, *, organism: str, genome_assembly: int, release: int
    ) -> tuple[str | None, str | None]:
        """Locate the core DB directory name (may include patch letters) and its HTTPS directory URL.

        The exact name is probed first; if Ensembl answers that it does not exist (HTTP 404), the release's
        directory is listed to find a name with patch letters (e.g. ``homo_sapiens_core_54_36p``). If Ensembl does
        not answer (after the retries of :py:func:`idtrack._ensembl_access.https_get`), its
        :py:class:`idtrack.EnsemblDownloadError` propagates: "unknown" is never taken for "absent".

        Args:
            organism: Ensembl species name.
            genome_assembly: Assembly key of the organism table.
            release: Ensembl release.

        Returns:
            tuple[Optional[str], Optional[str]]: ``(name, url)``, or ``(None, None)`` when Ensembl answered that the
            release has no such database.
        """
        release = int(release)
        genome_assembly = int(genome_assembly)
        expected = f"{organism}_core_{release}_{DB.db_suffix(organism, genome_assembly)}"
        pattern = re.compile(rf"^{re.escape(expected)}[a-z]*$")

        for root in cls._ftp_mysql_root_candidates(organism=organism, genome_assembly=genome_assembly, release=release):
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
    def _get_core_db_index_from_ftp(cls, *, organism: str, genome_assembly: int, ports: list[int]) -> dict[str, Any]:
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
            genome_assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly (releases are assigned to them).

        Returns:
            dict[str, Any]: The index (see :py:meth:`_get_core_db_index`), with ``scanned_up_to``.
        """
        log = logging.getLogger(__name__)
        latest_release = cls._ensembl_latest_release()
        min_release, range_max = cls._release_window(organism, int(genome_assembly))
        max_release = min(int(range_max) if range_max is not None else int(latest_release), int(latest_release))
        log.info(
            f"Scanning the HTTPS dumps of {organism} assembly {genome_assembly} "
            f"(core suffix {DB.db_suffix(organism, int(genome_assembly))!r}) for releases {min_release}-{max_release}."
        )

        releases: set[int] = set()
        db_for_release: dict[int, str] = {}
        db_dir_url_by_release: dict[int, str] = {}
        for rel in range(min_release, max_release + 1):
            db_name, db_dir_url = cls._ftp_find_core_db_dir(
                organism=organism, genome_assembly=genome_assembly, release=int(rel)
            )
            if db_name and db_dir_url:
                releases.add(int(rel))
                db_for_release[int(rel)] = str(db_name)
                db_dir_url_by_release[int(rel)] = str(db_dir_url)

        releases_union = sorted(releases)
        if not releases_union:
            log.warning(
                f"Ensembl's HTTPS dumps have no core database of {organism!r} assembly {int(genome_assembly)} "
                f"in releases {min_release}..{max_release}."
            )

        # Assign each release to the first configured port that could plausibly host it (based on min-release rules).
        releases_by_port: dict[int, set[int]] = {int(p): set() for p in ports}
        db_by_port_release: dict[int, dict[int, str]] = {int(p): {} for p in ports}
        port_for_release: dict[int, int] = {}
        for rel in releases_union:
            chosen: int | None = None
            for p in ports:
                min_rel = DB.mysql_port_min_release.get(int(p))
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
            "genome_assembly": int(genome_assembly),
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
    def _https_index_path(local_repository: str, organism: str, genome_assembly: int) -> str:
        """Return the file in *local_repository* that keeps the HTTPS-dump index of one organism and assembly."""
        return os.path.join(local_repository, f"ensembl_https_index_{organism}_assembly-{int(genome_assembly)}.json")

    @classmethod
    def _index_covers(
        cls, index: dict[str, Any], organism: str, genome_assembly: int, needed_up_to: int | None
    ) -> bool:
        """Return whether a core-DB index answers for every release up to *needed_up_to*.

        An index answers when its scan reached *needed_up_to* (``None``: the latest Ensembl release), clipped to the
        end of the assembly's release range: releases that existed at the scan cannot disappear, and releases after
        it are not needed. Otherwise only an index scanned from the network in this process, less than
        :py:attr:`idtrack.DB.rest_species_cache_seconds` ago, answers (it reached the latest release then; a
        long-running process scans again to learn of a new one).

        Args:
            index: A core-DB index (see :py:meth:`_get_core_db_index`).
            organism: Ensembl species name.
            genome_assembly: Assembly key of the organism table.
            needed_up_to: Last release needed; ``None`` for the latest Ensembl release.

        Returns:
            bool: ``True`` if *index* can be used without scanning again.
        """
        _, range_max = cls._release_window(organism, int(genome_assembly))
        if needed_up_to is None:
            need = range_max
        else:
            need = int(needed_up_to) if range_max is None else min(int(needed_up_to), int(range_max))
        scanned_up_to = index.get("scanned_up_to")
        if need is not None and scanned_up_to is not None and int(scanned_up_to) >= int(need):
            return True
        scanned_at = index.get("scanned_live_at")
        return scanned_at is not None and time.monotonic() - float(scanned_at) < float(DB.rest_species_cache_seconds)

    @classmethod
    def _https_index_signature(cls, organism: str, genome_assembly: int, ports: list[int]) -> dict[str, Any]:
        """Return the organism-table fields an HTTPS-dump index depends on (a saved index must match them).

        Args:
            organism: Ensembl species name.
            genome_assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly (releases are assigned to them).

        Returns:
            dict[str, Any]: Organism, assembly, core-database suffix, release window and ports.
        """
        first, last = cls._release_window(organism, int(genome_assembly))
        return {
            "organism": organism,
            "genome_assembly": int(genome_assembly),
            "db_suffix": DB.db_suffix(organism, int(genome_assembly)),
            "release_window": [int(first), None if last is None else int(last)],
            "ports": [int(p) for p in ports],
        }

    @classmethod
    def _read_https_index(
        cls, path: str, organism: str, genome_assembly: int, ports: list[int]
    ) -> dict[str, Any] | None:
        """Read an index saved by :py:meth:`_write_https_index`.

        Args:
            path: The saved file.
            organism: Ensembl species name.
            genome_assembly: Assembly key of the organism table.
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
            organism, genome_assembly, ports
        ):
            return None
        try:
            return {
                "organism": organism,
                "genome_assembly": int(genome_assembly),
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
        tmp_path = f"{path}.tmp-{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
            os.replace(tmp_path, path)
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not save the HTTPS-dump index %r: %s", path, exc)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def _https_index(
        cls,
        *,
        organism: str,
        genome_assembly: int,
        ports: list[int],
        local_repository: str | None,
        needed_up_to: int | None,
    ) -> dict[str, Any]:
        """Return the HTTPS-dump index, from *local_repository* when a saved one covers the need, else scanned.

        The scan asks Ensembl REST for the latest release and probes the dump directory of every release, so a saved
        index is what lets a manager whose window ends at a known release (every graph built or loaded through
        :py:meth:`idtrack.API.build_graph`) work without the network once its tables and graph are on disk.

        Args:
            organism: Ensembl species name.
            genome_assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly.
            local_repository: Folder where the index is saved; ``None`` neither reads nor saves one.
            needed_up_to: Last release needed; ``None`` for the latest Ensembl release.

        Returns:
            dict[str, Any]: The HTTPS-dump index; ``scanned_live_at`` (``time.monotonic()``) is set when it was scanned
            in this call.
        """
        path = cls._https_index_path(local_repository, organism, genome_assembly) if local_repository else None
        if path is not None:
            saved = cls._read_https_index(path, organism, genome_assembly, ports)
            if saved is not None and cls._index_covers(saved, organism, genome_assembly, needed_up_to):
                logging.getLogger(__name__).info(
                    "Ensembl releases of %s assembly %s read from %r (scanned up to release %s).",
                    organism,
                    genome_assembly,
                    path,
                    saved["scanned_up_to"],
                )
                return saved
        index = cls._get_core_db_index_from_ftp(organism=organism, genome_assembly=int(genome_assembly), ports=ports)
        if path is not None:
            cls._write_https_index(path, index, ports)
        return {**index, "scanned_live_at": time.monotonic()}

    @classmethod
    def _probe_mysql_core_schemas_by_port(
        cls, *, organism: str, genome_assembly: int, ports: list[int]
    ) -> tuple[dict[int, set[int]], dict[int, dict[int, str]], dict[int, Exception]]:
        """Return per-port core DB releases and schema names from the live Ensembl MySQL service.

        Only names of the assembly's core databases (``DbSuffix``, optional patch letters) whose release lies in the
        assembly's release window (:py:meth:`_release_window`) are kept, exactly as on the HTTPS path. Ports that
        :py:attr:`idtrack.DB.mysql_port_assembly` reserves for another assembly are never queried. Ports that the
        process-wide :py:data:`idtrack._ensembl_access.reachability` reports unusable (blocked, or ``"https"``
        access mode) are skipped without a connection attempt.

        Args:
            organism: Ensembl species name.
            genome_assembly: Assembly key of the organism table.
            ports: Configured MySQL ports of the assembly.

        Returns:
            tuple: ``(releases_by_port, db_by_port_release, connection_errors)``.
        """
        # Some Ensembl releases encode patch builds as a trailing letter, e.g. `homo_sapiens_core_56_37a`.
        # Treat these as part of the same assembly for configuration and graph-level purposes.
        pattern = DB.core_db_name_pattern(organism, int(genome_assembly))
        first_release, last_release = cls._release_window(organism, int(genome_assembly))
        suffix = DB.db_suffix(organism, int(genome_assembly))
        # Use an information_schema filter to avoid `SHOW DATABASES` timeouts on busy ports.
        escaped_organism = organism.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = f"{escaped_organism}\\_core\\_%\\_{suffix}%"

        # Keep keys for all configured ports, even if some are unreachable.
        releases_by_port: dict[int, set[int]] = {int(p): set() for p in ports}
        db_by_port_release: dict[int, dict[int, str]] = {int(p): {} for p in ports}
        connection_errors: dict[int, Exception] = {}

        for port in ports:
            owner = DB.mysql_port_assembly.get(int(port))
            if owner is not None and owner != (organism, int(genome_assembly)):
                # A port reserved for another assembly (3337, the GRCh37 archive) holds frozen copies of older
                # assemblies under later release numbers (e.g. MMUL_1 as `..._98_10`), which the suffix and release
                # range would attribute to the wrong assembly. The organism table cannot list such a port for another
                # assembly; this guards tables assigned without validation.
                continue
            # Avoid long hangs when outbound MySQL ports are blocked (common on managed networks): the port is
            # probed at most once per process.
            if not reachability.mysql_port_usable(int(port)):
                connection_errors[int(port)] = ConnectionError(
                    f"MySQL {DB.mysql_host}:{int(port)} is not used (unreachable, or Ensembl access mode 'https')"
                )
                continue

            results_query = None
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    with pymysql.connect(
                        host=DB.mysql_host,
                        user=DB.myqsl_user,
                        password=DB.mysql_togo,
                        port=int(port),
                        connect_timeout=max(DB.connection_timeout, 30),
                        read_timeout=max(DB.reading_timeout, 60),
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
                    reachability.mark_unreachable(DB.mysql_host, int(port), reason=str(last_exc))
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
    def _refresh_core_db_index_mysql(
        cls, index: dict[str, Any], *, organism: str, genome_assembly: int
    ) -> dict[str, Any]:
        """Refresh the MySQL-derived portion of a cached core-index in place."""
        ports = list(index.get("ports") or DB.assembly_mysqlport_priority[organism][int(genome_assembly)]["Ports"])
        ports = [int(p) for p in ports]
        releases_by_port, db_by_port_release, connection_errors = cls._probe_mysql_core_schemas_by_port(
            organism=organism,
            genome_assembly=int(genome_assembly),
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
        genome_assembly: int,
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
            genome_assembly: Assembly key of the organism table (e.g. ``38`` for GRCh38).
            local_repository: Folder of the caller's cached tables, where the HTTPS index is saved; ``None`` keeps it
                in memory only.
            needed_up_to: Last release the caller needs (its ``ignore_after``); ``None`` means the latest Ensembl
                release.

        Returns:
            dict[str, Any]: Mapping describing reachable releases/ports for this organism/assembly.

        If the dump scan cannot be completed (a request stays unanswered), its
        :py:class:`idtrack.EnsemblDownloadError` propagates; no partial index is ever returned or
        cached. An unknown organism or assembly raises the organism table's error.
        """
        key = (organism, int(genome_assembly), int(DB.organism_table_version))
        cached = cls._core_db_index_cache.get(key)
        if cached is not None and not cls._index_covers(cached, organism, int(genome_assembly), needed_up_to):
            cached = None
        if cached is not None:
            errors = cached.get("mysql_connection_errors") or {}
            if errors:
                # If a previously failing port is usable now (only possible after the reachability memory was
                # reset, e.g. by `ConnectionBridge`), refresh the MySQL-derived portion so downstream callers don't
                # get stuck with stale partial results. Asking the memory costs nothing otherwise.
                if any(reachability.mysql_port_usable(int(port)) for port in errors):
                    refreshed = cls._refresh_core_db_index_mysql(
                        cached, organism=organism, genome_assembly=int(genome_assembly)
                    )
                    cls._core_db_index_cache[key] = refreshed
                    return refreshed
            return cached

        ports = list(DB._assembly_entry(organism, int(genome_assembly))["Ports"])
        releases_by_port, db_by_port_release, connection_errors = cls._probe_mysql_core_schemas_by_port(
            organism=organism,
            genome_assembly=int(genome_assembly),
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
            genome_assembly=int(genome_assembly),
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
            "genome_assembly": int(genome_assembly),
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
        ensembl_release: int | None = None,
        ignore_before: int | None = None,
        ignore_after: int | float | None = None,
        store_raw_always: bool = True,
        genome_assembly: SupportsIndex | float | str | None = None,
    ):
        """Initialize a :py:class:`DatabaseManager` for a specific organism, release, and assembly.

        Args:
            organism (str): Canonical species name in Ensembl schema (e.g. ``"homo_sapiens"``); one of
                :py:attr:`idtrack.DB.supported_organisms`, otherwise :py:class:`NotImplementedError` is raised.
            form (str): Biological entity level of interest—one of ``"gene"``, ``"transcript"``,
                ``"translation"``, …—governing which stable-ID columns will be expected downstream.
            local_repository (str): Directory of the cached tables (:py:attr:`tables_directory`, one HDF5 file per
                table), the saved index of Ensembl's dumps and the graphs. It must exist and be readable; it must be
                writable only when a table has to be computed and stored (a read-only repository serves what it
                holds).
            ensembl_release (Optional[int]): Target Ensembl release number. If ``None`` the most
                recent release available for *genome_assembly* is selected automatically.
            ignore_before (Optional[int]): Earliest release to include when building cross-release ID
                histories. Defaults to the first release of the organism in the organism table
                (:py:meth:`idtrack.DB.first_release`, over all its assemblies), the same default as
                :py:class:`idtrack.API` uses.
            ignore_after (Optional[int | float]): Latest release to include when building histories.
                ``np.inf`` (the default) disables the upper bound and includes all newer releases.
            store_raw_always (bool): When ``True`` (default) raw Ensembl tables are stored in the repository after
                their download; when ``False`` they are not, and every derived table that needs them downloads them
                again.
            genome_assembly (Optional[int]): Assembly key of the organism table
                (:py:attr:`idtrack.DB.assembly_mysqlport_priority`), as an integer of any integer type, an integral
                float or a string of digits (:py:meth:`idtrack.DB.assembly_key`). This selects the **primary** assembly used for
                data access (e.g. ``38`` = human GRCh38, ``39`` = mouse GRCm39, ``111`` = pig Sscrofa11.1,
                ``10`` = rhesus Mmul_10). The key usually equals the code in core schema names
                (``<organism>_core_<release>_<code>``); see the table's ``DbSuffix``. If omitted, the highest-priority
                assembly for *organism* is used. If *ensembl_release* is provided, the selection is restricted to
                assemblies that actually contain that release; otherwise to assemblies with a release inside the
                ``ignore_before``/``ignore_after`` window.

        Raises:
            ValueError: If *form* is not in the supported list, *local_repository* fails basic
                path/read/write checks, or the Ensembl access mode (``IDTRACK_ENSEMBL_ACCESS`` /
                :py:attr:`idtrack.DB.ensembl_access`) is invalid.
            RuntimeError: If internal port/release configuration is inconsistent.
            NotImplementedError: If *organism* is not in the organism table (:py:attr:`idtrack.DB.supported_organisms`).
        """
        if organism not in DB.supported_organisms:
            raise NotImplementedError(
                f"Organism {organism!r} is not configured. Supported organisms: {DB.supported_organisms}. "
                "To add an organism, register it with `idtrack.DB.register_organism` (or add it to "
                "`DB.assembly_mysqlport_priority` in idtrack/_db.py) and provide its externals YAML file "
                f"`{organism}_externals_modified.yml`; see 'Adding organism support' in the contributing guide."
            )

        # Validate the Ensembl access mode first, so that an invalid `IDTRACK_ENSEMBL_ACCESS` / `DB.ensembl_access`
        # raises its own error here instead of surfacing later inside an assembly probe.
        ensembl_access_mode()

        if not os.path.isdir(local_repository):
            raise ValueError(f"`local_repository` must be an existing directory: {local_repository!r}")
        # Write access is needed only when a table is to be stored (see `_store`): a repository on a read-only volume
        # serves the tables and graphs it holds.
        if not os.access(local_repository, os.R_OK):
            raise ValueError(f"`local_repository` must be readable: {local_repository!r}")

        if form not in DB.forms_in_order:
            raise ValueError(f"Form {form!r} is not supported. Supported forms: {DB.forms_in_order}")

        ignore_before_effective: int | None
        if ignore_before is None:
            ignore_before_effective = None
        else:
            try:
                ignore_before_float = float(ignore_before)
                ignore_before_int = int(ignore_before_float)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("ignore_before must be an integer Ensembl release number.") from exc
            if ignore_before_float != ignore_before_int:
                raise ValueError("ignore_before must be an integer Ensembl release number.")
            ignore_before_effective = ignore_before_int

        if ignore_after is None:
            ignore_after_effective: int | float = np.inf
        else:
            try:
                ignore_after_float = float(ignore_after)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("ignore_after must be an integer Ensembl release number or np.inf.") from exc
            if np.isinf(ignore_after_float):
                ignore_after_effective = np.inf
            else:
                ignore_after_int = int(ignore_after_float)
                if ignore_after_float != ignore_after_int:
                    raise ValueError("ignore_after must be an integer Ensembl release number or np.inf.")
                ignore_after_effective = ignore_after_int

        if ignore_before_effective is not None and not np.isinf(ignore_after_effective):
            if int(ignore_after_effective) < int(ignore_before_effective):
                raise ValueError(
                    f"ignore_after must be >= ignore_before (got ignore_before={ignore_before_effective}, "
                    f"ignore_after={ignore_after_effective})."
                )

        configured_assemblies = DB.assembly_mysqlport_priority[organism]
        # The window's end decides whether a saved HTTPS-dump index can answer without the network.
        index_kwargs: dict[str, Any] = {
            "local_repository": local_repository,
            "needed_up_to": None if np.isinf(ignore_after_effective) else int(ignore_after_effective),
        }
        if ensembl_release is not None and float(ensembl_release) != int(ensembl_release):
            raise ValueError("Floating-point Ensembl releases (e.g. 18.2) are not supported; pass an integer release.")

        if genome_assembly is None:
            # Pick the highest-priority assembly that has the requested release (or, without one, a release inside the
            # ignore window; e.g. a mouse snapshot at release 100 is GRCm38, because GRCm39 starts at 103). An
            # assembly is passed over only when its complete index lacks the release; an index that cannot be built
            # raises, because taking the next assembly would silently give another primary assembly (one failed
            # request would make human GRCh37).
            requested_release = int(ensembl_release) if ensembl_release is not None else None

            def _has_release_in_window(index: dict[str, Any]) -> bool:
                return any(
                    (ignore_before_effective is None or int(r) >= int(ignore_before_effective))
                    and (np.isinf(ignore_after_effective) or int(r) <= int(ignore_after_effective))
                    for r in index.get("releases", [])
                )

            available_summary: dict[int, tuple[int | None, int | None]] = {}
            for asm in DB.assemblies_by_priority(organism):
                index = self._get_core_db_index(organism=organism, genome_assembly=int(asm), **index_kwargs)
                releases = index.get("releases") or []
                available_summary[int(asm)] = (min(releases), max(releases)) if releases else (None, None)
                if (requested_release is None and _has_release_in_window(index)) or (
                    requested_release is not None and requested_release in index["port_for_release"]
                ):
                    genome_assembly = int(asm)
                    break

            if genome_assembly is None:
                raise ValueError(
                    f"No assembly of {organism!r} has a core database "
                    + (
                        f"at release {requested_release}"
                        if requested_release is not None
                        else f"in the window {ignore_before_effective}..{ignore_after_effective}"
                    )
                    + f". Available (min, max) releases by assembly: {available_summary}."
                )
        else:
            genome_assembly = DB.assembly_key(genome_assembly)
            if genome_assembly not in configured_assemblies:
                raise ValueError(
                    f"Genome assembly {genome_assembly} is not configured for organism {organism!r}. "
                    f"Supported assemblies: {sorted(configured_assemblies)}"
                )

        # MYSQL Settings
        self.genome_assembly = genome_assembly
        core_index = self._get_core_db_index(organism=organism, genome_assembly=self.genome_assembly, **index_kwargs)

        if ensembl_release is None:
            if not core_index["releases"]:
                raise ValueError(
                    f"No Ensembl core databases found for {organism!r} on assembly {self.genome_assembly} "
                    f"(tried MySQL ports {core_index['ports']})."
                )
            if not np.isinf(ignore_after_effective):
                eligible = [r for r in core_index["releases"] if r <= int(ignore_after_effective)]
                if not eligible:
                    raise ValueError(
                        f"No Ensembl core database found for {organism!r} on assembly {self.genome_assembly} "
                        f"at or before ignore_after={ignore_after_effective!r}. "
                        f"Available releases: {core_index['releases']}"
                    )
                ensembl_release = eligible[-1]
            else:
                ensembl_release = core_index["releases"][-1]

        ensembl_release = int(ensembl_release)
        if ensembl_release not in core_index["port_for_release"]:
            raise ValueError(
                f"No Ensembl core database found for {organism!r} release {ensembl_release} on assembly {self.genome_assembly}. "
                f"Tried MySQL ports {core_index['ports']}. Available releases: {core_index['releases']}"
            )
        selected_port = int(core_index["port_for_release"][ensembl_release])

        self.mysql_settings: dict[str, Any] = {
            "host": DB.mysql_host,
            "user": DB.myqsl_user,
            "password": DB.mysql_togo,
            "port": selected_port,
            "connect_timeout": max(DB.connection_timeout, 30),
            "read_timeout": max(DB.reading_timeout, 60),
        }

        # The logger for informing the user about the progress.
        self.log = logging.getLogger(__name__)

        # Instance attributes
        self.local_repository = local_repository
        self.organism = organism
        self.form = form
        self.store_raw_always = store_raw_always
        if ignore_before_effective is None:
            ignore_before_effective = DB.first_release(organism)
        self.ignore_before = int(ignore_before_effective)
        # If ignore_after is not specified, then set it to infinite.
        self.ignore_after: int | float = ignore_after_effective

        # Protected attributes
        self.available_form_of_interests = copy.deepcopy(DB.forms_in_order)  # Warning: the order is important.
        self._column_sep = "_COL_"
        self._identifiers = [f"{self.form}_stable_id", f"{self.form}_version"]

        self.ensembl_release = int(ensembl_release)
        # _ = self.external_inst.load_modified_yaml()

        if not np.isinf(self.ignore_after) and self.ignore_after < self.ignore_before:
            raise ValueError(
                f"ignore_after must be >= ignore_before (got ignore_before={self.ignore_before}, ignore_after={self.ignore_after})."
            )

        releases_in_window = sorted(r for r in core_index["releases"] if self.ignore_after >= r >= self.ignore_before)
        if not releases_in_window:
            msg = (
                "No Ensembl releases remain after applying "
                f"ignore_before={self.ignore_before} and ignore_after={self.ignore_after} "
                f"for {self.organism!r} assembly {int(self.genome_assembly)}. "
                f"Available releases: {core_index['releases']}"
            )
            raise ValueError(msg)
        if self.ensembl_release not in releases_in_window:
            raise ValueError(
                f"ensembl_release={self.ensembl_release} is outside the ignore window "
                f"[{self.ignore_before}, {self.ignore_after}] for {self.organism!r} assembly {int(self.genome_assembly)}. "
                f"Releases within window: {releases_in_window}"
            )

        # Only enforce port-level minimum-release rules when we expect to query the live MySQL service.
        # When a release is accessed via the HTTPS/FTP dumps, the configured MySQL port is only a
        # bookkeeping detail and may not support the historical release directly.
        mysql_releases = set(core_index.get("releases_on_mysql", []) or [])
        if mysql_releases and int(self.ensembl_release) in mysql_releases:
            min_port_release = DB.mysql_port_min_release.get(int(self.mysql_settings["port"]))
            if min_port_release is not None and self.ensembl_release < int(min_port_release):
                raise RuntimeError(
                    f"Internal configuration error: selected port {int(self.mysql_settings['port'])} has min release "
                    f"{int(min_port_release)} but release {int(self.ensembl_release)} was selected."
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
            f"    Ensembl Release: {self.ensembl_release}{os.linesep}"
            f"    Genome Assembly: {self.genome_assembly}{os.linesep}"
            f"    Ignore Before: {self.ignore_before}{os.linesep}"
            f"    Ignore After: {self.ignore_after}{os.linesep}"
            f"    Local Repository: {self.local_repository}{os.linesep}"
            f"    Store Raw Always: {self.store_raw_always}{os.linesep}"
        )

    @property
    def _core_index_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of :py:meth:`_get_core_db_index` for this manager (its repository and window end)."""
        ignore_after = getattr(self, "ignore_after", np.inf)
        return {
            "local_repository": getattr(self, "local_repository", None),
            "needed_up_to": None if np.isinf(ignore_after) else int(ignore_after),
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
    def available_releases_all_assemblies(self) -> list[int]:
        """Return all Ensembl releases that have a core database of at least one configured assembly.

        For clean-handoff species (e.g. mouse: 37 → 38 → 39), no single assembly spans the full release
        history. Graph construction and YAML template generation therefore need a release catalogue that is the
        union over *all* assemblies configured for :py:attr:`organism` in
        :py:data:`idtrack.DB.assembly_mysqlport_priority`.

        The list is filtered by the manager's ``ignore_before`` / ``ignore_after`` window and cached for the
        lifetime of this :py:class:`DatabaseManager` instance. Every assembly's index must be complete: if one cannot
        be built (Ensembl does not answer), the error is raised instead of leaving that assembly's releases out.

        Returns:
            list[int]: Sorted release numbers available for at least one configured assembly.
        """
        releases: set[int] = set()
        for asm in DB.assemblies_by_priority(self.organism):
            core_index = self._get_core_db_index(
                organism=self.organism, genome_assembly=int(asm), **self._core_index_kwargs
            )
            releases.update(int(r) for r in core_index["releases"] if self.ignore_after >= r >= self.ignore_before)
        return sorted(releases)

    def available_releases_versions(self) -> list[int]:
        """Discover valid Ensembl releases for the configured organism and assembly.

        Availability is discovered via the cached, multi-port aware core index built by
        :py:meth:`_get_core_db_index`. The resulting union of releases is filtered against the manager's
        ``ignore_before`` / ``ignore_after`` bounds and returned in ascending order.

        Returns:
            list[int]: Sorted list of release numbers that exist on the mirror and comply with the ignore window.
        """
        # Use the pre-built core-DB index (union across ports) so multi-port assemblies work correctly.
        core_index = self._get_core_db_index(
            organism=self.organism, genome_assembly=self.genome_assembly, **self._core_index_kwargs
        )
        releases_final = sorted(r for r in core_index["releases"] if self.ignore_after >= r >= self.ignore_before)
        return releases_final

    @cached_property
    def mysql_database(self) -> str:
        """Return the canonical Ensembl Core schema name for the current organism, release, and assembly.

        The schema naming convention is deterministic:

        ``{organism}_core_{ensembl_release}_{DbSuffix}[<patch>]``

        where ``DbSuffix`` is the assembly's core-database code (:py:meth:`idtrack.DB.db_suffix`, usually the
        assembly key itself).

        For multi-port assemblies (e.g. *sus_scrofa* assembly ``102``), the *port* is selected in
        :py:meth:`__init__` using :py:meth:`_get_core_db_index`.  Once the release is validated to exist on that
        chosen port, the schema name itself does not require another server-side discovery query.

        Returns:
            str: Schema name like ``"homo_sapiens_core_111_38"``.

        Raises:
            ValueError: If the current release is not available for this (organism, assembly) pair.
        """
        core_index = self._get_core_db_index(
            organism=self.organism, genome_assembly=self.genome_assembly, **self._core_index_kwargs
        )
        if int(self.ensembl_release) not in core_index["port_for_release"]:
            raise ValueError(
                f"No Ensembl core database found for {self.organism!r} release {int(self.ensembl_release)} "
                f"on assembly {int(self.genome_assembly)}. Available releases: {core_index['releases']}"
            )
        return str(core_index["db_for_release"][int(self.ensembl_release)])

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

    def change_release_auto_assembly(self, ensembl_release: int) -> "DatabaseManager":
        """Clone the manager for *ensembl_release* while inferring a compatible genome assembly.

        Unlike :py:meth:`change_release`, this helper allows :py:attr:`genome_assembly` to change when the
        requested release is not present in the current assembly (common for clean-handoff species).
        Assembly inference follows the same priority rules as :py:meth:`__init__` with ``genome_assembly=None``:
        pick the highest-priority configured assembly that contains the requested release.

        Args:
            ensembl_release (int): Desired Ensembl release number.

        Returns:
            DatabaseManager: Fresh manager initialised for *ensembl_release* with an inferred assembly.
        """
        return DatabaseManager(
            organism=self.organism,
            ensembl_release=int(ensembl_release),
            form=self.form,
            local_repository=self.local_repository,
            ignore_before=self.ignore_before,
            ignore_after=self.ignore_after,
            store_raw_always=self.store_raw_always,
            genome_assembly=None,
        )

    def change_assembly(self, genome_assembly: int, last_possible_ensembl_release: bool = False) -> "DatabaseManager":
        """Clone the manager while targeting a new genome assembly (e.g. GRCh38 → GRCh37).

        Genome assemblies are identified by the integer keys of :py:attr:`idtrack.DB.assembly_mysqlport_priority`
        (``38`` for *GRCh38*, ``37`` for *GRCh37*, ``39`` for *GRCm39*, ``111`` for *Sscrofa11.1*, ``1`` for rhesus
        *MMUL_1*, …).  When
        *last_possible_ensembl_release* is ``True`` the
        method automatically picks the most recent Ensembl release that **still** provides MySQL dumps for the
        requested assembly, ensuring compatibility.  All other settings are copied verbatim.

        Args:
            genome_assembly (int): Assembly code configured under
                :py:attr:`DB.assembly_mysqlport_priority` for this manager's organism.
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
        :py:func:`idtrack._ensembl_access.ensembl_access_mode`).

        Returns:
            pandas.DataFrame: One column named ``"available_databases"``, sorted.
        """
        ensembl_access_mode()  # an invalid mode must raise here, not be read as "no databases found"
        self.log.info(f"Available core databases for {self.organism} are being listed.")
        names: set[str] = set()
        for assembly in DB.assemblies_by_priority(self.organism):
            core_index = self._get_core_db_index(
                organism=self.organism, genome_assembly=int(assembly), **self._core_index_kwargs
            )
            names.update(str(v) for v in core_index.get("db_for_release", {}).values() if v)
        return pd.DataFrame(sorted(names), columns=["available_databases"])

    # ------------------------------------------------------------------------------------------------------------
    # Cached tables: one file per table (see idtrack._utils_hdf5); the file name carries every parameter the table
    # depends on, and the file records them again (checked when it is read).

    @property
    def tables_directory(self) -> str:
        """Folder of this organism's and assembly's cached tables: ``<local_repository>/idtrack_tables/<org>_assembly-<key>``."""
        return os.path.join(
            self.local_repository, "idtrack_tables", f"{self.organism}_assembly-{int(self.genome_assembly)}"
        )

    def table_path(self, name: str, ensembl_release: int | None = None) -> str:
        """Return the file of a cached table.

        Args:
            name: The table's key without the release, e.g. ``"mysql_gene__cols=gene_id+stable_id+version"`` or
                ``"idhistory-narrow_gene"``; it names every parameter the table depends on.
            ensembl_release: Release of the table; the manager's release by default.

        Returns:
            str: ``<tables_directory>/ens<release>_<name>.h5``.
        """
        release = int(self.ensembl_release if ensembl_release is None else ensembl_release)
        return os.path.join(self.tables_directory, f"ens{release}_{name}.h5")

    def table_digests(self) -> dict[str, str]:
        """Return ``{file name: content SHA-256}`` of every table stored for this organism and assembly.

        The digests identify the tables' content (:py:func:`idtrack._utils_hdf5.read_table_digest`), so that a graph
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

    def _table_params(self, name: str, ensembl_release: int | None = None, **extra: Any) -> dict[str, Any]:
        """Return the parameters recorded with a cached table (checked again when it is read)."""
        params: dict[str, Any] = {
            "organism": self.organism,
            "genome_assembly": int(self.genome_assembly),
            "table": name,
            "ensembl_release": int(self.ensembl_release if ensembl_release is None else ensembl_release),
        }
        params.update(extra)
        return params

    def _store(self, path: str, df: pd.DataFrame, params: dict[str, Any]) -> None:
        """Write a table to the repository (atomically, see :py:func:`idtrack._utils_hdf5.write_table`).

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
            raise ValueError("Empty 'usecols' parameter, or 'usecols' is not a list.")
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
        core_index = self._get_core_db_index(
            organism=self.organism, genome_assembly=self.genome_assembly, **self._core_index_kwargs
        )
        db_dir = core_index.get("db_dir_url_by_release", {}).get(int(self.ensembl_release))
        if isinstance(db_dir, str) and db_dir:
            return db_dir

        db_name = self.mysql_database
        release = int(self.ensembl_release)
        for root in self._ftp_mysql_root_candidates(
            organism=self.organism, genome_assembly=self.genome_assembly, release=release
        ):
            candidate = f"{root}{db_name}/"
            resp = https_get(candidate, what=f"the dump directory of {db_name}", stream=True)
            if resp is not None:
                resp.close()
                return candidate

        raise FileNotFoundError(
            f"Ensembl has no MySQL dump directory for {self.organism!r} release {release} "
            f"assembly {int(self.genome_assembly)} (expected db {db_name!r})."
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
            rel_target = int(self.ensembl_release)
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
            raise ValueError(f"No CREATE TABLE blocks found in Ensembl schema dump: {sql_url!r}")

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

        max_attempts = max(1, int(DB.rest_attempts))
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
                    sleep_s = float(DB.rest_retry_wait) * (2**attempt)
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
            f"Raw table for `{table_key}` on ensembl release `{self.ensembl_release}` "
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
                            raise ValueError(f"Table {table_key!r} column {col!r} holds undecodable bytes.") from exc
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
        :py:mod:`idtrack._ensembl_access`: the port answered its probe and the access mode is not ``"https"``), the
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
                *usecols* (:py:class:`idtrack._db.EnsemblSchemaError`).
            ValueError: If the query returns text that cannot be decoded.

        Notes:
            If the HTTPS dumps cannot be read, :py:class:`idtrack.EnsemblDownloadError` propagates.
        """
        if usecols == []:
            usecols = None

        def _quote_identifier(identifier: str) -> str:
            if not isinstance(identifier, str):
                raise ValueError(f"Unsafe SQL identifier type: {type(identifier)!r}")
            if not re.fullmatch(r"[0-9A-Za-z_]+", identifier):
                raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
            return f"`{identifier}`"

        core_index = self._get_core_db_index(
            organism=self.organism, genome_assembly=self.genome_assembly, **self._core_index_kwargs
        )
        mysql_releases = set(core_index.get("releases_on_mysql", []) or [])
        # If the requested release is not present on the live MySQL service, go straight to HTTPS dumps.
        if int(self.ensembl_release) not in mysql_releases:
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
                    raise ValueError(f"Table {table_key!r} has no columns.")
                if any(self._column_sep in c for c, _ in typed):
                    raise ValueError(f"Column names of {table_key!r} contain the separator {self._column_sep!r}.")

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
        self.log.info(
            f"Raw table for `{table_key}` on ensembl release `{self.ensembl_release}` was downloaded{info_usecols}"
        )
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
                :py:data:`idtrack.DB.forms_in_order` (``"gene"``, ``"transcript"``, or
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
                self.ensembl_release,
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
            raise ValueError(f"Expected 6 columns {sorted(cols)}, got {len(df.columns)} columns: {list(df.columns)}.")

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
                :py:data:`idtrack.DB.id_ver_delimiter` is found inside any ``*_stable_id`` field, indicating
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
        if not np.all(sm["new_stable_id"].str.find(DB.id_ver_delimiter) == -1):
            raise ValueError(f"new_stable_id column contains delimiter {DB.id_ver_delimiter!r} which is not allowed.")
        if not np.all(sm["old_stable_id"].str.find(DB.id_ver_delimiter) == -1):
            raise ValueError(f"old_stable_id column contains delimiter {DB.id_ver_delimiter!r} which is not allowed.")

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
                    ``["name_db", "count"]`` giving the number of rows of each external database (most rows first,
                    ties by name). ``count`` is an ``int64``.

        Raises:
            ValueError: If *filter_mode* is not one of the accepted literals **or** if the YAML allow-list claims a
                database that is absent from the retrieved mappings—indicating the configuration and MySQL data are
                out of sync.
            EnsemblSchemaError: If ``identity_xref`` lacks columns other than its identity columns.

        Notes:
            *Synonym handling* - any synonym brought in from ``external_synonym`` is prefixed with
            :py:data:`DB.synonym_id_nodes_prefix`, and its ``name_db`` is likewise prefixed so that synonym nodes remain
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
            raise ValueError(
                f"filter_mode must be 'all', 'relevant', 'database', or 'relevant-database', got {filter_mode!r}."
            )
        relevant = filter_mode in ("relevant", "relevant-database")
        prefix = DB.synonym_id_nodes_prefix
        identities = ["ensembl_identity", "xref_identity"]
        db_id, db_name, id_graph = "id_db", "name_db", "graph_id"

        # Tables loaded via HTTPS/FTP dumps can contain join-key columns as strings. Some older releases also include
        # malformed/continuation lines that shift text into numeric ID columns. Pandas refuses merges when dtypes
        # disagree (e.g. int64 vs object), so normalise here.
        def _coerce_numeric_key(df: pd.DataFrame, col: str, table_name: str) -> pd.DataFrame:
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

        # External databases first: with `relevant`, everything below is restricted to the enabled databases before
        # any join (the result is the same; the join of all cross references is never built).
        m = {"save_after_calculation": self.store_raw_always}
        ed = self.get_table("external_db", usecols=["external_db_id", "db_name", "db_display_name"], **m)
        ed = _coerce_numeric_key(ed, "external_db_id", "external_db")
        isin_list: list[str] = []
        if relevant:
            isin_list = list(self.external_inst.give_list_for_case(give_type="db"))
            # A database is named by `db_name` (accession rows) or `db_display_name` (label rows), with the synonym
            # prefix for synonym rows; keep the external_db rows that can produce one of the enabled names.
            bare = {n[len(prefix) :] if n.startswith(prefix) else n for n in isin_list}
            ed = ed[ed["db_name"].isin(bare) | ed["db_display_name"].isin(bare)]

        x = self.get_table("xref", usecols=["xref_id", "external_db_id", "dbprimary_acc", "display_label"], **m)
        x = _coerce_numeric_key(x, "external_db_id", "xref")
        if relevant:
            x = x[x["external_db_id"].isin(ed["external_db_id"])]
        es = self.get_table("external_synonym", usecols=["xref_id", "synonym"], **m)
        if relevant:
            es = es[es["xref_id"].isin(x["xref_id"])]
        ox = self.get_table(
            "object_xref", usecols=["ensembl_id", "ensembl_object_type", "xref_id", "object_xref_id"], **m
        )
        ox = ox[ox["ensembl_object_type"].str.lower() == self.form.lower()]
        if relevant:
            ox = ox[ox["xref_id"].isin(x["xref_id"])]
        # Handle column name changes in identity_xref table across Ensembl releases.
        # Old releases (e.g., release 48) use: target_identity, query_identity
        # New releases use: ensembl_identity, xref_identity
        try:
            ix = self.get_table("identity_xref", usecols=["ensembl_identity", "xref_identity", "object_xref_id"], **m)
        except EnsemblSchemaError as e:
            if not ({"ensembl_identity", "xref_identity"} & set(e.missing_columns)):
                raise
            self.log.info(
                "Using legacy column names (target_identity, query_identity) for identity_xref table "
                f"in Ensembl release {self.ensembl_release}."
            )
            try:
                ix = self.get_table(
                    "identity_xref", usecols=["target_identity", "query_identity", "object_xref_id"], **m
                )
                ix = ix.rename(columns={"target_identity": "ensembl_identity", "query_identity": "xref_identity"})
            except EnsemblSchemaError:
                self.log.warning(
                    f"identity_xref table missing identity columns for Ensembl release {self.ensembl_release}. "
                    "Using NaN values for ensembl_identity and xref_identity."
                )
                ix = self.get_table("identity_xref", usecols=["object_xref_id"], **m)
                ix["ensembl_identity"] = np.nan
                ix["xref_identity"] = np.nan
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
        del es

        a = self.get_db(f"idsraw_{self.form}")
        comb = pd.merge(
            a,
            ox,
            how="left",
            left_on=f"{self.form}_id",
            right_on="ensembl_id",
            validate="one_to_many",
        )
        del ox
        comb = comb.merge(x, how="left", on="xref_id", validate="many_to_many")
        del x
        comb = comb.merge(ed, how="left", on="external_db_id", validate="many_to_one")
        comb = comb.merge(ix, how="left", on="object_xref_id", validate="many_to_one")
        del ix

        # Node name "ID.Version" once per id of the form, then attached by the internal key.
        names = pd.Series(self.id_ver_from_df(a[self._identifiers].reset_index(drop=True)), index=a[f"{self.form}_id"])
        graph_ids = comb[f"{self.form}_id"].map(names)
        del a

        # One row per (Ensembl id, external id, database): labels with the display name, accessions with the name.
        parts = []
        for id_col, name_col in (("display_label", "db_display_name"), ("dbprimary_acc", "db_name")):
            part = pd.DataFrame(
                {
                    id_graph: graph_ids.to_numpy(),
                    db_id: comb[id_col].to_numpy(),
                    db_name: comb[name_col].to_numpy(),
                    identities[0]: comb[identities[0]].to_numpy(),
                    identities[1]: comb[identities[1]].to_numpy(),
                }
            )
            parts.append(part)
        del comb, graph_ids
        res = pd.concat(parts, axis=0, ignore_index=True)
        del parts
        bare_id = res[db_id].where(
            ~res[db_id].astype(object).str.startswith(prefix, na=False), res[db_id].str[len(prefix) :]
        )
        res = res[~(res[id_graph].isna() | res[db_name].isna() | res[db_id].isna()) & (bare_id != "")]

        # A total order (all columns, stable), so that the table does not depend on the order of the input rows; the
        # identities are compared as float64 (pandas cannot sort by float16 columns).
        res = res.astype({identities[0]: np.float64, identities[1]: np.float64})
        res = res.sort_values(by=[db_name, id_graph, db_id, *identities], kind="stable", na_position="last")
        # Convert some columns type for convenience.
        res = res.astype({identities[0]: np.float16, identities[1]: np.float16})
        # Drop duplicates if exists. Note that it is not trivial, there are many duplicated lines after adding
        # these columns as rows. Because, for some of them, comb_X_columns are actually the same.
        res = res.drop_duplicates(ignore_index=True)
        res.insert(0, "release", np.full(len(res), self.ensembl_release, dtype=np.uint8))

        # Change the synonym IDs' database name
        is_synonym = res[db_id].astype(object).str.startswith(prefix).to_numpy(dtype=bool)
        res[db_name] = np.where(is_synonym, prefix, "").astype(object) + res[db_name].astype(object)
        res[db_name] = res[db_name].astype(pd.Series(["a"]).dtype)
        # Unless you specifically look at synonyms, they will not mean the same thing as the counterparts.
        # They will be used as the bridging point in the pathfinder algorithm only.

        if relevant:
            available_databases = set(res[db_name].unique())
            missing_databases = sorted(d for d in isin_list if d not in available_databases)
            if missing_databases:
                raise ValueError(
                    f"The externals YAML enables {missing_databases} for {self.organism} {self.form}, assembly "
                    f"{self.genome_assembly}, release {self.ensembl_release}, but Ensembl's tables of that release have "
                    f"no such cross references. Correct the YAML "
                    f"({self.external_inst.file_name_modified_yaml(mode='configured')!r})."
                )
            res = res[res[db_name].isin(isin_list)]

        res = res.reset_index(drop=True)
        if filter_mode in ("all", "relevant"):
            return res
        return self._count_databases(res)

    def create_database_content(self, just_download: bool = False) -> pd.DataFrame:
        """Retrieve and optionally cache external-database metadata for every assembly, release, and form.

        The helper iterates over every genome assembly configured for the current organism in
        :py:attr:`idtrack.DB.assembly_mysqlport_priority`, every available Ensembl release for each assembly,
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
        frames: list[pd.DataFrame] = []
        assemblies_processed: list[int] = []
        assemblies_without_release: list[int] = []

        for k in DB.assemblies_by_priority(self.organism):
            # An assembly is left out only when its complete index has no release in the window; a failure to build
            # the index raises (a template made without an assembly would silently lack its databases).
            core_index = self._get_core_db_index(
                organism=self.organism, genome_assembly=int(k), **self._core_index_kwargs
            )
            if not any(self.ignore_before <= int(r) <= self.ignore_after for r in core_index["releases"]):
                assemblies_without_release.append(int(k))
                self.log.warning(
                    f"Assembly {k} of {self.organism} has no Ensembl release in the window "
                    f"{self.ignore_before}..{self.ignore_after}; it has no entries in the template YAML."
                )
                continue
            self.log.info(f"Processing assembly {k} for {self.organism}...")
            dm_assembly = self.change_assembly(k, last_possible_ensembl_release=True)
            assemblies_processed.append(int(k))
            for j in dm_assembly.available_form_of_interests:  # For all assemblies possible.
                for i in dm_assembly.available_releases:
                    self.log.info(
                        f"Database content is being created for "
                        f"`{self.organism}`, assembly `{k}`, form `{j}`, ensembl release `{i}`"
                    )
                    df_temp = dm_assembly.change_release(i).change_form(j).get_db("external_database")
                    if not just_download:
                        df_temp = df_temp.assign(assembly=k, release=i, form=j)
                        frames.append(df_temp)

        self.log.info(
            f"Assembly processing complete for {self.organism}: processed {sorted(assemblies_processed)}, "
            f"without a release in the window: {sorted(assemblies_without_release) or 'none'}."
        )
        df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

        if not just_download:
            df["organism"] = self.organism
            df = df.reset_index(drop=True)
            return df
        else:
            return df

    def create_release_id(self) -> pd.DataFrame:
        """Return deduplicated stable-identifier/version pairs for the current form and release.

        Raw identifiers are fetched via :py:meth:`DatabaseManager.get_db`, trimmed to the canonical columns, and
        sanity-checked.  Two
        integrity rules are enforced: (1) the delimiter
        :py:data:`idtrack.DB.id_ver_delimiter` must **not** appear inside any stable identifier, and
        (2) every stable identifier must be unique after deduplication.  Violations raise
        :py:class:`ValueError`.

        Returns:
            pandas.DataFrame: Two-column dataframe ``[{form}_stable_id, {form}_version]`` with duplicates removed.

        Raises:
            ValueError: If the delimiter is present inside any stable identifier or if identifiers are not
                unique after deduplication.
        """
        dbm_the_ids = self.get_db(f"idsraw_{self.form}")
        dbm_the_ids = dbm_the_ids[self._identifiers]

        if not np.all(dbm_the_ids[f"{self.form}_stable_id"].str.find(DB.id_ver_delimiter) == -1):
            raise ValueError("The delimiter is in Ensembl IDs.")

        dbm_the_ids = dbm_the_ids.drop_duplicates(keep="first")
        if not dbm_the_ids[f"{self.form}_stable_id"].is_unique:
            raise ValueError("The stable IDs are not unique")

        dbm_the_ids = dbm_the_ids.reset_index(drop=True)
        dbm_the_ids = dbm_the_ids.drop_duplicates()
        return dbm_the_ids

    def create_gene_region(self) -> pd.DataFrame:
        """Tell for every gene of the release whether it lies on the reference sequence of the assembly.

        Ensembl annotates genes on the chromosomes and scaffolds of an assembly (its reference sequence) and also on
        sequence regions beside it: alternative loci (haplotypes), assembly patches and LRG records. A gene on such a
        region is a copy of, or an alternative to, a gene of the reference sequence and has a stable id of its own.
        The core database marks these regions with the attribute :py:attr:`idtrack.DB.non_reference_attribute`
        (``seq_region_attrib`` joined to ``attrib_type``), in every release and for every kind of region; the table
        ``assembly_exception`` lists only some of them and is empty in recent human releases, so it is not read.

        The table is a function of three raw tables of the release (``gene`` with its sequence region,
        ``seq_region_attrib`` and ``attrib_type``) and of the gene ids (``idsraw_gene``).

        Returns:
            pandas.DataFrame: One row per gene of the release: ``gene_stable_id``, ``gene_version`` (as in the ``ids``
            table of the gene form) and ``non_reference`` (``True`` for a gene on a region outside the reference
            sequence). A release whose core database defines no such attribute or marks no region has ``False``
            throughout.

        Raises:
            ValueError: If a gene of the id table has no sequence region in the ``gene`` table.
        """
        ids = self.get_db(f"idsraw_{DB.backbone_form}")
        raw = {"save_after_calculation": self.store_raw_always}
        regions = self.get_table("gene", usecols=["gene_id", "seq_region_id"], **raw)
        attributes = self.get_table("attrib_type", usecols=["attrib_type_id", "code"], **raw)
        marks = self.get_table("seq_region_attrib", usecols=["seq_region_id", "attrib_type_id"], **raw)

        codes = attributes.loc[attributes["code"] == DB.non_reference_attribute, "attrib_type_id"]
        beside = {int(i) for i in marks.loc[marks["attrib_type_id"].isin(codes), "seq_region_id"].dropna().unique()}
        region_of = dict(zip(regions["gene_id"].astype(int), regions["seq_region_id"]))
        gene_regions = ids["gene_id"].astype(int).map(region_of)
        missing = gene_regions.isna().to_numpy()
        if missing.any():
            raise ValueError(
                f"{int(missing.sum())} genes of {self.organism} release {self.ensembl_release} have no sequence region "
                f"in the `gene` table, e.g. {ids['gene_stable_id'][missing].iloc[0]!r}."
            )
        result = ids[["gene_stable_id", "gene_version"]].copy()
        result["non_reference"] = gene_regions.astype(np.int64).isin(beside).to_numpy(dtype=bool)
        return result.drop_duplicates(ignore_index=True)

    def check_if_change_assembly_works(self, target_assembly: int) -> bool:
        """Tell whether this manager can be cloned to operate on *target_assembly* at its release.

        Calls :py:meth:`DatabaseManager.change_assembly` and turns a :py:class:`ValueError` (the assembly is not
        configured, or has no core database at this release) into ``False``, so that loops over releases and
        assemblies can pass over the combinations that do not exist. Any other error propagates.

        Args:
            target_assembly (int): Genome-assembly code to test (a key of
                :py:attr:`idtrack.DB.assembly_mysqlport_priority` for the manager's organism).

        Returns:
            bool: ``True`` if the assembly switch succeeds without raising :py:class:`ValueError`; ``False`` otherwise.
        """
        try:
            self.change_assembly(target_assembly)
            return True
        except ValueError:
            return False

    def create_external_all(
        self, return_mode: str, narrow_external: bool = True
    ) -> pd.DataFrame | pd.core.groupby.generic.DataFrameGroupBy:
        """Download and collate cross-reference mappings from every supported genome assembly.

        The manager cycles through every genome assembly recognised for the current organism (ordered by
        :py:data:`idtrack.DB.assembly_mysqlport_priority`), fetches either the filtered
        *external_relevant* mapping table (when ``narrow_external=True``) or the full *external* mapping table
        (when ``narrow_external=False``) for each via :py:meth:`~DatabaseManager.get_db`, labels every row with
        its source assembly, and finally concatenates the tables. Because this helper is intended for
        **ad-hoc inspection only**, it bypasses the :py:meth:`~DatabaseManager.get_db` caching layer and
        therefore **never writes** the result to the local repository.

        Args:
            return_mode (str): Strategy for handling rows that appear in more than one assembly.
            narrow_external (bool): If ``True`` (default), restrict results to databases enabled in the external
                YAML configuration (``external_relevant``). If ``False``, include all external databases provided by
                the Ensembl MySQL server (``external``).

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
        df_indicator = "external_relevant" if narrow_external else "external"

        cfg = DB.assembly_mysqlport_priority[self.organism]
        ass_sorted = sorted({int(a) for a in ass if int(a) in cfg}, key=lambda a: cfg[a]["Priority"])
        frames: list[pd.DataFrame] = []
        for i in ass_sorted:  # sort according to per-organism priority (1 = newest)
            # An assembly is passed over only when its complete index has no core database at this release; a failure
            # to build the index raises (leaving the assembly out would silently drop its external edges).
            core_index = self._get_core_db_index(organism=self.organism, genome_assembly=i, **self._core_index_kwargs)
            if int(self.ensembl_release) not in core_index["port_for_release"]:
                self.log.info(
                    "create_external_all: %s assembly %s has no core database at release %s.",
                    self.organism,
                    i,
                    self.ensembl_release,
                )
                continue
            df_temp = self.change_assembly(i).get_db(df_indicator)
            frames.append(df_temp.assign(assembly=i))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        compare_columns = [
            i for i in df.columns if i != "assembly" and not i.endswith("_identity")
        ]  # 'ensembl_identity', 'xref_identity'
        compare_columns_2 = compare_columns + ["assembly"]

        if return_mode == "all":
            # Drop duplicate rows that have all the columns the same with another row in the dataframe.
            # This also looks for 'assembly' columns so it is possible to say assemlies are evaluated separately.
            df = df.drop_duplicates(keep="first", ignore_index=True, subset=compare_columns_2)
            return df

        elif return_mode == "unique":
            # Unlike above, this does not also look for 'assembly' columns, so an entry found in the higher priority
            # assembly will be kept but the others will be removed.
            df = df.drop_duplicates(keep="first", ignore_index=True, subset=compare_columns)
            # Note that: after transition to new assembly. ensembl does not assign new versions etc to the older
            # keep the most priority one.
            return df

        elif return_mode == "duplicated":
            df = df[df.duplicated(subset=compare_columns, keep=False)]
            dfg = df.groupby(by=compare_columns)
            return dfg

        else:
            raise ValueError(f"Undefined parameter for 'return_mode': {return_mode}.")

    def get_db(
        self,
        df_indicator: str,
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
          :py:meth:`create_gene_region`), ``externalcontent`` (:py:meth:`create_database_content`) and
          ``availabledatabases`` (:py:meth:`create_available_databases`).

        Args:
            df_indicator (str): One of the indicators above.
            create_even_if_exist (bool): Compute a stored table again and replace the stored copy.
            save_after_calculation (bool): Store a computed table (ignored for the tables that are not stored).

        Returns:
            pandas.DataFrame: The requested table.

        Raises:
            ValueError: If *df_indicator* is not one of the indicators above.
        """
        split_ind = df_indicator.split("_")
        main_ind = split_ind[0]
        param1_ind = split_ind[1] if len(split_ind) > 1 else None
        if len(split_ind) > 2:
            raise ValueError(f"df_indicator {df_indicator!r} has too many underscore-separated parts (max 2).")

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
                raise ValueError(
                    f"'idsraw' should be used together with one "
                    f"of followings: {','.join(self.available_form_of_interests)}."
                )
            return self.create_ids(form=param1_ind)

        if main_ind == "ids" and param1_ind is None:
            return self.create_release_id()

        if main_ind == "generegion" and param1_ind is None:
            return self.create_gene_region()

        if main_ind == "external" and param1_ind in (None, "database"):
            name = f"external{'-database' if param1_ind else ''}_{self.form}"
            return cached(name, lambda: self.create_external_db(filter_mode=param1_ind or "all"))

        if main_ind == "external" and param1_ind in ("relevant", "relevant-database"):
            selection = sorted(self.external_inst.give_list_for_case(give_type="db"))
            digest = hashlib.sha256(json.dumps(selection).encode("utf-8")).hexdigest()[:16]
            name = f"external-relevant_{self.form}__dbs={digest}"
            rows = cached(name, lambda: self.create_external_db(filter_mode="relevant"), databases=selection)
            return rows if param1_ind == "relevant" else self._count_databases(rows)

        if main_ind == "idhistory" and param1_ind in (None, "narrow"):
            name = f"idhistory{'-narrow' if param1_ind else ''}_{self.form}"
            history = cached(name, lambda: self.create_id_history(narrow=param1_ind == "narrow"))
            in_window = (history["old_release"] >= self.ignore_before) & (history["old_release"] <= self.ignore_after)
            return history[in_window].reset_index(drop=True)

        if main_ind == "relationcurrent" and param1_ind is None:
            return cached("relationcurrent", self.create_relation_current)

        if main_ind == "externalcontent" and param1_ind is None:
            return self.create_database_content()

        if main_ind == "availabledatabases" and param1_ind is None:
            return self.create_available_databases()

        raise ValueError(f"Unexpected entry for 'df_indicator': {df_indicator!r}.")

    @staticmethod
    def _count_databases(rows: pd.DataFrame) -> pd.DataFrame:
        """Return ``name_db``/``count`` (rows per database), most rows first, ties by name."""
        counts = rows["name_db"].value_counts(sort=False)
        out = pd.DataFrame({"name_db": counts.index.astype(object), "count": counts.to_numpy(dtype=np.int64)})
        out = out.sort_values(["count", "name_db"], ascending=[False, True], kind="stable", ignore_index=True)
        out["name_db"] = out["name_db"].astype(pd.Series(["a"]).dtype)
        return out

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
            list[str]: Ordered list where each element is ``"<ID>.<version>"``; for a row
                without version (the empty id of a transcript that has no translation)
                the id itself.

        Raises:
            ValueError: If ``dbm_the_ids`` does not contain the expected column
                names stored in :py:data:`self._identifiers`.
        """
        if list(dbm_the_ids.columns) != list(self._identifiers):
            raise ValueError(
                f"Column names of the 'dbm_the_ids' is not as expected. "
                f"{list(dbm_the_ids.columns)} vs {self._identifiers}"
            )
        ids = dbm_the_ids.iloc[:, 0]
        versions = dbm_the_ids.iloc[:, 1]
        if not pd.api.types.is_numeric_dtype(versions):
            # Text versions (e.g. the `Retired`/`Void` sentinels of DB.alternative_versions): element by element.
            return [
                DatabaseManager.node_name_maker(DatabaseManager.node_dict_maker(i, j))
                for i, j in zip(ids.to_numpy(dtype=object), versions.to_numpy(dtype=object))
            ]
        names = pd.Series(ids.to_numpy(dtype=object, copy=True), dtype=object)  # never write into the caller's frame
        has = versions.notna().to_numpy()
        if has.any():
            numbers = versions.to_numpy()[has]
            whole = numbers.astype(np.int64)
            if np.any(whole != numbers):
                bad = int(np.flatnonzero(whole != numbers)[0])
                raise ValueError(f"Version is floating: {(names[has].iloc[bad], numbers[bad])}")
            names[has] = names[has] + DB.id_ver_delimiter + pd.Series(whole, dtype=np.int64).astype(str).to_numpy()
        return names.tolist()

    @staticmethod
    def node_name_maker(node_dict: dict[str, Any]) -> str:
        """Concatenate *ID* and *Version* into a single node label.

        Given the miniature dictionary returned by
        :py:meth:`DatabaseManager.node_dict_maker`, this helper builds the string
        representation that uniquely identifies a biological entity within the graph
        layer.  It appends the version (a number, or one of the markers of
        :py:data:`idtrack.DB.alternative_versions`) to the stable ID using
        :py:data:`idtrack.DB.id_ver_delimiter` (``"."`` by default).  An entry
        without version is an absent identifier (the empty id of a transcript that
        has no translation) and is returned as it is.

        Args:
            node_dict (dict[str, Any]): Mapping with exactly two keys, ``"ID"`` and
                ``"Version"``, as produced by
                :py:meth:`DatabaseManager.node_dict_maker`.

        Returns:
            str: ``"<ID>.<version>"``, or ``"<ID>"`` for an entry without version.
        """
        version = node_dict["Version"]
        if version is not None and not (pd.api.types.is_scalar(version) and pd.isna(version)) and version != "":
            return node_dict["ID"] + DB.id_ver_delimiter + str(version)
        else:
            return node_dict["ID"]

    @staticmethod
    def node_dict_maker(id_entry: str, version_entry: Any) -> dict[str, Any]:
        """Return a normalized *ID/Version* dictionary from raw column values.

        This helper creates the canonical structure consumed by
        :py:meth:`DatabaseManager.node_name_maker` and higher-level graph utilities,
        ensuring that version numbers are strictly integers whenever possible.  It
        also recognises special placeholders defined in
        :py:data:`idtrack.DB.alternative_versions` (e.g. ``"Retired"`` or
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
        if (
            version_entry is not None
            and not pd.isna(version_entry)
            and version_entry != ""
            and version_entry not in DB.alternative_versions
        ):
            if int(version_entry) != float(version_entry):
                raise ValueError(f"Version is floating: {(id_entry, version_entry)}")
            else:
                version_entry = int(version_entry)
        return {"ID": id_entry, "Version": version_entry}

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
            raise ValueError(
                f"Organisms whose Ensembl ids carry no versions are not supported: the {form} ids of {self.organism} "
                f"have {found} at Ensembl release {self.ensembl_release} (genome assembly {self.genome_assembly}). "
                "idtrack follows `<id>.<version>` through the releases."
            )
        df[version_column] = df[version_column].astype(int)
        return df
