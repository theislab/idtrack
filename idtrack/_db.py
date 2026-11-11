#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import h5py


class DB:
    """Store constants shared across IDTrack modules for Ensembl data access and graph construction.

    This class centralizes every constant that multiple components (e.g. :py:class:`idtrack.graph.GraphMaker`,
    :py:class:`idtrack.pathfinder.PathFinder`) rely on when talking to the Ensembl FTP mirror, REST API,
    and public MySQL instances. Housing the values in one namespace prevents circular imports, ensures a single
    source of truth, and simplifies testing. `DB` is never instantiated; import the class and reference its
    attributes directly. The organism table is the only part meant to change at run time, and only through
    :py:meth:`DB.register_organism`.

    Attributes:
        id_ver_delimiter (str): Character separating a stable identifier from its version suffix.
        first_version (int): Default version assumed when an ID lacks an explicit version component.

        connection_timeout (int): TCP connect timeout in seconds used by both FTP and REST clients.
        reading_timeout  (int): Socket read timeout in seconds applied to FTP and REST operations.

        ensembl_ftp_base (str): Hostname of the Ensembl public FTP mirror.
        rest_server_api (str): Root URL of the Ensembl REST API.
        rest_server_ext (str): Resource path appended to :py:data:`rest_server_api` to query species metadata.

        mysql_host (str): Hostname of the Ensembl public MySQL server.
        myqsl_user (str): Username for anonymous MySQL access.
        mysql_togo (str): Placeholder string kept for backward compatibility when assembling connection URLs.

        mysql_port_min_release (dict[int, int]): Minimum Ensembl release supported by each public MySQL port.
        mysql_probe_timeout (float): TCP connect timeout in seconds used the first time a MySQL port is probed in a
            process (see :py:mod:`idtrack._ensembl_access`).
        ensembl_access (Optional[str]): How Ensembl is reached. ``"auto"`` uses the public MySQL service where its
            ports answer and the HTTPS dumps otherwise; ``"https"`` never touches MySQL. ``None`` (the default) reads
            the environment variable named by :py:attr:`ensembl_access_env_var` and falls back to ``"auto"``.
        ensembl_access_env_var (str): Name of the environment variable read when :py:attr:`ensembl_access` is
            ``None`` (``IDTRACK_ENSEMBL_ACCESS``).
        ensembl_access_modes (tuple[str, ...]): Accepted access modes.

        assembly_mysqlport_priority (dict[str, dict[int, dict[str, Any]]]): The one table of organisms,
            ``{organism -> {assembly key -> {...}}}``. Each assembly entry holds ``Ports``, ``Priority``,
            ``ReleaseRange`` and, optionally, ``Name`` and ``DbSuffix`` (see the comment block of the table).
            Change it only through :py:meth:`register_organism`, which validates it and recomputes every
            derived attribute below.
        symbol_databases (dict[str, tuple[str, ...]]): Ordered gene-symbol databases per organism (names as in the
            externals YAML). The first one is the default final database of
            :py:class:`idtrack.HarmonizeFeatures`.
        organism_table_version (int): Incremented whenever :py:meth:`register_organism` changes the table, so that
            caches derived from it (e.g. the core-database index) can tell they are stale.

        supported_organisms (tuple[str, ...]): Sorted organism names of the table (derived).
        all_assemblies (set[int]): Union of every assembly key across supported organisms (derived).

        synonym_id_nodes_prefix (str): Prefix inserted before node identifiers that represent synonym edges.
        no_old_node_id (str): Sentinel used when a historical ID is retired.
        no_new_node_id (str): Sentinel used when no future successor exists.
        alternative_versions (set[str]): Two sentinels—:py:data:`no_new_node_id` and :py:data:`no_old_node_id`.

        hyperconnecting_threshold (int): Maximum allowable out-degree before a node is considered
            hyper-connected and ignored by breadth-first expansions.

        node_type_str (str): Edge/Node attribute key holding the node type value.
        nts_external (str): Canonical node type assigned to entities originating outside Ensembl.
        forms_in_order (list[str]): Stable ordering of Ensembl entity forms (``gene``, ``transcript``,
            ``translation``). Order matters when inferring parent/child relationships.
        backbone_form (str): Form selected as backbone for graph traversals (always ``gene``).

        nts_ensembl (dict[str, str]): Map each canonical form to its namespaced node type
            (``gene`` → ``ensembl_gene``, etc.).
        nts_ensembl_reverse (dict[str, str]): Reverse mapping of :py:data:`nts_ensembl`.
        nts_assembly (dict[int, dict[str, str]]): Assembly key → form → assembly-specific node type (derived).
        nts_assembly_reverse (dict[str, str]): Assembly-specific node type → :py:data:`nts_ensembl` value (derived).
        nts_base_ensembl (dict[str, str]): Reduced node type names stripped of assembly suffixes.
        nts_base_ensembl_reverse (dict[str, str]): Reverse mapping of :py:data:`nts_base_ensembl`.

        nts_bidirectional_synonymous_search (set[str]): Node types for which synonym searches are
            performed bidirectionally.
        nts_assembly_gene (set[str]): Every assembly-specific node type that represents a gene (derived).

        connection_dict (str): Edge attribute key whose value stores connection metadata dictionaries.
        conn_dict_str_ensembl_base (str): Constant placed under :py:data:`connection_dict` when the edge
            points to an Ensembl data source.

        external_search_settings (dict[str, Any]): Default limits for outward traversal into external
            databases. Keys are ``jump_limit``, ``synonymous_max_depth``, and ``nts_backbone``.

        placeholder_na (str): Sentinel stored in HDF5 datasets where a true NA/None value is not
            permitted or would break downstream type expectations.
        UTF8 (str): The literal string ``"utf-8"``—a canonical spelling of the UTF-8 encoding name
            used when writing variable-length strings to HDF5 files.
        UTF8_STR (h5py.Datatype): Pre-configured variable-length UTF-8 string dtype created via
            :py:func:`h5py.string_dtype`.  Pass this value when creating HDF5 datasets that should
            hold arbitrary Unicode text to avoid hard-coding datatypes throughout the codebase.
    """

    # Delimiter separating the ID and the version of an ensembl identifier.
    id_ver_delimiter = "."
    first_version: int = 1  # just and arbitrary assumption

    # FTP and REST API Timeouts. It’s a good practice to set connect timeouts to slightly larger than a multiple of 3,
    # which is the default TCP packet retransmission window.
    connection_timeout = 12
    reading_timeout = 12

    # FTP server from which the above databases will be downloaded.
    ensembl_ftp_base = "ftp.ensembl.org"

    # Ensembl REST API to fetch information of organisms and latest releases.
    rest_server_api = "https://rest.ensembl.org"
    rest_server_ext = "/info/species?"

    # MYSQL
    mysql_host = "ensembldb.ensembl.org"
    myqsl_user = "anonymous"
    mysql_togo = ""
    mysql_port_min_release: dict[int, int] = {
        # https://www.ensembl.org/info/data/mysql.html
        3306: 48,
        5306: 48,  # mirror of 3306 (redundancy / load balancing; should serve equivalent core schemas)
        3337: 79,
    }
    # Connect timeout for the first (and only) TCP probe of a MySQL port in a process.
    mysql_probe_timeout: float = 5.0

    # Access mode: "auto" (MySQL where reachable, HTTPS dumps otherwise) or "https" (never touch MySQL).
    # `None` means: read the environment variable `IDTRACK_ENSEMBL_ACCESS`, else "auto".
    ensembl_access: Optional[str] = None
    ensembl_access_env_var = "IDTRACK_ENSEMBL_ACCESS"
    ensembl_access_modes = ("auto", "https")

    assembly_mysqlport_priority: dict[str, dict[int, dict[str, Any]]] = {
        # The one table of organisms. Change it with `DB.register_organism`, which validates it and recomputes
        # everything derived from it (`supported_organisms`, `all_assemblies`, the `nts_assembly*` node types).
        #
        # Keys and fields
        # ---------------
        # - The key of an assembly entry is an integer that is unique within the organism. It is what the rest of
        #   idtrack sees: node types (`assembly_<key>_ensembl_gene`), cache files (`<organism>_assembly-<key>.h5`),
        #   the assembly keys of the externals YAML files and the public `genome_assembly` argument.
        # - Core databases are named `<organism>_core_<release>_<DbSuffix>[patch letter]` (e.g. `..._37a`); patch
        #   letters belong to the same assembly. `DbSuffix` defaults to the key, and the key equals Ensembl's code
        #   wherever that is unambiguous. The code is species-specific (`38` is GRCh38 for human, GRCm38 for mouse).
        # - Where Ensembl reuses one code for two assemblies of an organism, the earlier assembly gets a key of its
        #   own and its `DbSuffix` names the shared code; `ReleaseRange` then tells the two apart. Rhesus macaque is
        #   the example: `macaca_mulatta_core_<release>_10` is MMUL_1 up to release 85 and Mmul_10 from release
        #   98, so Mmul_10 keeps key 10 and MMUL_1 has key 1 with `DbSuffix: "10"`. Entries that share a suffix
        #   must have non-overlapping release ranges.
        # - "ReleaseRange" is the [first, last] Ensembl release of the assembly; None as last means "current latest
        #   and beyond". Both discovery paths (MySQL and HTTPS dumps) only accept releases inside it. Releases before
        #   48 are out of reach regardless (`mysql_port_min_release`).
        # - "Ports" are the MySQL ports to try in preference order; DatabaseManager picks the first port that
        #   contains the requested release. "Priority" orders the assemblies of an organism (1 = preferred, newest).
        # - "Name" is the assembly name as Ensembl gives it (`assembly.default` in the core database's `meta`
        #   table); it is informative only.
        "homo_sapiens": {
            # GRCh38 (Ensembl release 76+; primary assembly since release 76)
            38: {"Name": "GRCh38", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [76, None]},
            # GRCh37 (Ensembl release 55+; archive since release 76)
            # - MySQL port 3337 hosts GRCh37 **release 79 onwards** (archive service).
            # - Some GRCh37 schemas can also appear on the main ports (3306/5306) for select releases; keep them
            #   as fallbacks for environments where 3337 is unreachable.
            # - Releases 55–78 are accessed via the HTTPS/FTP MySQL dumps (DatabaseManager FTP fallback).
            37: {"Name": "GRCh37", "Ports": [3337, 3306, 5306], "Priority": 2, "ReleaseRange": [55, None]},
            # NCBI36 (historic; clean handoff to GRCh37 at Ensembl release 55)
            # - Ensembl FTP contains releases 48–54 under directory names like
            #   `homo_sapiens_core_<release>_36<patch>` (e.g. `homo_sapiens_core_54_36p`).
            # - Not expected to be available on the live public MySQL service anymore; accessed via HTTPS/FTP dumps.
            36: {"Name": "NCBI36", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 54]},
        },
        "mus_musculus": {
            # GRCm39 (Ensembl release 103+; clean handoff from GRCm38 at release 103)
            39: {"Name": "GRCm39", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [103, None]},
            # GRCm38 (Ensembl release 68–102)
            # - Port 3337 also hosts a subset of GRCm38 releases (archive mirror); keep it as a fallback.
            38: {"Name": "GRCm38", "Ports": [3306, 5306, 3337], "Priority": 2, "ReleaseRange": [68, 102]},
            # NCBIM37 (Ensembl release 48–67; clean handoff to GRCm38 at release 68)
            # - Early releases can carry a patch-letter suffix (e.g. `..._37a`) and are handled by DatabaseManager.
            37: {"Name": "NCBIM37", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 67]},
        },
        "sus_scrofa": {
            # Sscrofa11.1 (Ensembl release 90+; clean handoff from Sscrofa10.2 at release 90)
            111: {"Name": "Sscrofa11.1", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [90, None]},
            # Sscrofa10.2 (archive) — Ensembl release 67–89
            # - Port 3337 hosts later archive releases (79–99); keep it as a fallback when present.
            102: {"Name": "Sscrofa10.2", "Ports": [3306, 5306, 3337], "Priority": 2, "ReleaseRange": [67, 89]},
            # Sscrofa9 (historic; clean handoff to Sscrofa10.2 at release 67)
            # - Ensembl FTP contains releases 56–66 under directory names like
            #   `sus_scrofa_core_<release>_9<patch>` (e.g. `sus_scrofa_core_60_9d`).
            # - Not expected to be available on the live public MySQL service anymore; accessed via HTTPS/FTP dumps.
            9: {"Name": "Sscrofa9", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [56, 66]},
        },
        "macaca_mulatta": {
            # Rhesus macaque. Clean handoffs: MMUL_1 up to 85, Mmul_8.0.1 86–97, Mmul_10 from 98 (one assembly per
            # release in the dump listings). Ports 3306/5306 serve releases 48+; 3337 hosts only the GRCh37 archive.
            # Mmul_10 (Ensembl release 98+)
            10: {"Name": "Mmul_10", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [98, None]},
            # Mmul_8.0.1 (Ensembl release 86–97)
            801: {"Name": "Mmul_8.0.1", "Ports": [3306, 5306], "Priority": 2, "ReleaseRange": [86, 97]},
            # MMUL_1 (Ensembl release 47–85; 48 is idtrack's floor). Its core databases share the code `10` with
            # Mmul_10 (`macaca_mulatta_core_63_10`; `..._48_10g` to `..._62_10q` carry patch letters), so it has
            # its own key 1 and names the shared code in `DbSuffix`.
            1: {"Name": "MMUL_1", "DbSuffix": "10", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 85]},
        },
        "macaca_fascicularis": {
            # Cynomolgus (crab-eating) macaque. Clean handoff from 5.0 to 6.0 at release 103.
            # Macaca_fascicularis_6.0 (Ensembl release 103+)
            6: {"Name": "Macaca_fascicularis_6.0", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [103, None]},
            # Macaca_fascicularis_5.0 (Ensembl release 91–102)
            5: {"Name": "Macaca_fascicularis_5.0", "Ports": [3306, 5306], "Priority": 2, "ReleaseRange": [91, 102]},
        },
    }

    # Ordered gene-symbol databases per organism (names as in the externals YAML files). The first one is the
    # default final database of `HarmonizeFeatures`. Declared next to the table, validated with it, and set
    # together with it by `register_organism`.
    symbol_databases: dict[str, tuple[str, ...]] = {
        "homo_sapiens": ("HGNC Symbol",),
        "mus_musculus": ("MGI Symbol",),
        # Pig and the macaques carry VGNC symbols on their current assemblies and HGNC symbols on earlier ones.
        "sus_scrofa": ("VGNC Symbol", "HGNC Symbol"),
        "macaca_mulatta": ("VGNC Symbol", "HGNC Symbol"),
        "macaca_fascicularis": ("VGNC Symbol", "HGNC Symbol"),
    }

    # Fields an assembly entry may carry, and which of them are required.
    assembly_entry_fields = ("Name", "DbSuffix", "Ports", "Priority", "ReleaseRange")
    assembly_entry_required_fields = ("Ports", "Priority", "ReleaseRange")

    organism_table_version: int = 0

    # Derived from the table by `_rebuild_organism_tables` (called at import and by `register_organism`).
    supported_organisms: tuple[str, ...] = ()
    all_assemblies: set[int] = set()

    # Protected Non-int Version Strings/Thresholds
    synonym_id_nodes_prefix = "synonym_id::"
    no_old_node_id = "Void"
    no_new_node_id = "Retired"
    alternative_versions = {no_new_node_id, no_old_node_id}
    hyperconnecting_threshold = 20

    # Node Types
    node_type_str = "node_type"
    nts_external = "external"
    forms_in_order = ["gene", "transcript", "translation"]  # Warning: the order is important here.
    backbone_form = "gene"  # No other choice possible.

    nts_ensembl: dict[str, str] = {}  # ensembl_gene
    for _form in forms_in_order:
        nts_ensembl[_form] = f"ensembl_{_form}"
    nts_ensembl_reverse: dict[str, str] = {}
    for _form, _nts in nts_ensembl.items():
        nts_ensembl_reverse[_nts] = _form

    # Derived from the organism table (`_rebuild_organism_tables`): `assembly_<key>_ensembl_<form>`.
    nts_assembly: dict[int, dict[str, str]] = {}
    nts_assembly_reverse: dict[str, str] = {}
    nts_assembly_gene: set[str] = set()

    nts_base_ensembl: dict[str, str] = {}
    for _form in forms_in_order:
        nts_base_ensembl[_form] = f"base_ensembl_{_form}"
    nts_base_ensembl_reverse: dict[str, str] = {}
    for _form, _nts in nts_base_ensembl.items():
        nts_base_ensembl_reverse[_nts] = _form
    nts_bidirectional_synonymous_search = {nts_external, nts_base_ensembl[backbone_form]}

    # Edge/Note Attributes:
    connection_dict: str = "connection"
    conn_dict_str_ensembl_base = "ensembl_base"  # as a database in connection_dict

    # PathFinder Settings
    external_search_settings: dict[str, Any] = {
        "jump_limit": 2,
        "synonymous_max_depth": 2,
        "nts_backbone": nts_ensembl[backbone_form],
    }

    # HDF5 Settings
    placeholder_na = "_PLACEHOLDER_NA_"
    UTF8 = "utf-8"
    UTF8_STR = h5py.string_dtype(encoding=UTF8)

    # ------------------------------------------------------------------------------------------------------------
    # Organism table: validation, derived attributes, run-time registration and accessors
    # ------------------------------------------------------------------------------------------------------------

    @classmethod
    def _validate_organism_table(
        cls, table: Mapping[str, Mapping[int, Mapping[str, Any]]], symbol_databases: Mapping[str, Sequence[str]]
    ) -> None:
        """Check the organism table and the symbol-database table for consistency.

        Rules: organism names are lower-case Ensembl production names; every organism has at least one assembly;
        assembly keys are positive integers; each entry carries ``Ports`` (known MySQL ports, see
        :py:attr:`mysql_port_min_release`), ``Priority`` and ``ReleaseRange``, plus optionally ``Name`` and
        ``DbSuffix`` and nothing else; priorities are unique per organism and start at 1; release ranges are
        ``[first, last]`` with ``last`` ``None`` or not smaller than ``first``; entries of one organism that share a
        ``DbSuffix`` have non-overlapping release ranges; the symbol-database table names exactly the organisms of
        the organism table.

        Args:
            table: Candidate value of :py:attr:`assembly_mysqlport_priority`.
            symbol_databases: Candidate value of :py:attr:`symbol_databases`.

        Raises:
            ValueError: If any rule is violated; the message names the organism, the assembly and the rule.
        """
        if not isinstance(table, Mapping) or not table:
            raise ValueError("The organism table must be a non-empty mapping {organism: {assembly key: {...}}}.")

        for organism, assemblies in table.items():
            if not isinstance(organism, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", organism):
                raise ValueError(
                    f"Organism name {organism!r} must be an Ensembl production name (lower case, e.g. 'homo_sapiens')."
                )
            if not isinstance(assemblies, Mapping) or not assemblies:
                raise ValueError(f"Organism {organism!r} must declare at least one assembly.")

            priorities: list[int] = []
            windows_by_suffix: dict[str, list[tuple[int, float, int]]] = {}
            for key, entry in assemblies.items():
                where = f"{organism!r} assembly {key!r}"
                if isinstance(key, bool) or not isinstance(key, int) or key <= 0:
                    raise ValueError(f"{where}: assembly keys must be positive integers.")
                if not isinstance(entry, Mapping):
                    raise ValueError(f"{where}: the entry must be a mapping.")
                unknown = sorted(set(entry) - set(cls.assembly_entry_fields))
                if unknown:
                    raise ValueError(f"{where}: unknown fields {unknown}; allowed: {list(cls.assembly_entry_fields)}.")
                missing = [f for f in cls.assembly_entry_required_fields if f not in entry]
                if missing:
                    raise ValueError(f"{where}: missing required fields {missing}.")

                ports = entry["Ports"]
                if (
                    not isinstance(ports, (list, tuple))
                    or not ports
                    or any(isinstance(p, bool) or not isinstance(p, int) for p in ports)
                    or len(set(ports)) != len(ports)
                ):
                    raise ValueError(f"{where}: 'Ports' must be a non-empty list of distinct integers.")
                unknown_ports = sorted(set(ports) - set(cls.mysql_port_min_release))
                if unknown_ports:
                    raise ValueError(
                        f"{where}: unknown MySQL ports {unknown_ports}; known: {sorted(cls.mysql_port_min_release)}."
                    )

                priority = entry["Priority"]
                if isinstance(priority, bool) or not isinstance(priority, int):
                    raise ValueError(f"{where}: 'Priority' must be an integer.")
                priorities.append(priority)

                window = entry["ReleaseRange"]
                if not isinstance(window, (list, tuple)) or len(window) != 2:
                    raise ValueError(f"{where}: 'ReleaseRange' must be [first, last] (last may be None).")
                first, last = window
                if isinstance(first, bool) or not isinstance(first, int) or first < 1:
                    raise ValueError(f"{where}: the first release of 'ReleaseRange' must be a positive integer.")
                if last is not None and (isinstance(last, bool) or not isinstance(last, int) or last < first):
                    raise ValueError(f"{where}: the last release of 'ReleaseRange' must be None or >= {first}.")

                name = entry.get("Name")
                if name is not None and (not isinstance(name, str) or not name.strip()):
                    raise ValueError(f"{where}: 'Name' must be a non-empty string.")

                suffix = entry.get("DbSuffix", str(key))
                if not isinstance(suffix, str) or not re.fullmatch(r"[0-9]+", suffix):
                    raise ValueError(f"{where}: 'DbSuffix' must be a string of digits (e.g. '10').")
                windows_by_suffix.setdefault(suffix, []).append(
                    (first, float("inf") if last is None else float(last), key)
                )

            if len(set(priorities)) != len(priorities) or min(priorities) != 1:
                raise ValueError(
                    f"Organism {organism!r}: priorities must be unique and start at 1 (got {sorted(priorities)})."
                )

            for suffix, windows in windows_by_suffix.items():
                windows.sort()
                for (_first_a, last_a, key_a), (first_b, _last_b, key_b) in zip(windows, windows[1:]):
                    if first_b <= last_a:
                        raise ValueError(
                            f"Organism {organism!r}: assemblies {key_a} and {key_b} share the core-database suffix "
                            f"{suffix!r} and their release ranges overlap; give them disjoint 'ReleaseRange's."
                        )

        if not isinstance(symbol_databases, Mapping):
            raise ValueError("The symbol-database table must be a mapping {organism: (database, ...)}.")
        if set(symbol_databases) != set(table):
            raise ValueError(
                "The symbol-database table must name exactly the organisms of the organism table "
                f"(missing: {sorted(set(table) - set(symbol_databases))}, "
                f"extra: {sorted(set(symbol_databases) - set(table))})."
            )
        for organism, databases in symbol_databases.items():
            if isinstance(databases, str) or not isinstance(databases, Sequence):
                raise ValueError(f"Organism {organism!r}: symbol databases must be a sequence of names.")
            if any(not isinstance(d, str) or not d for d in databases) or len(set(databases)) != len(databases):
                raise ValueError(f"Organism {organism!r}: symbol databases must be distinct non-empty strings.")

    @classmethod
    def _rebuild_organism_tables(cls) -> None:
        """Validate the organism table and recompute every attribute derived from it.

        Called once at import and by :py:meth:`register_organism`. The derived dictionaries and sets are updated in
        place, and :py:meth:`register_organism` changes the table and :py:attr:`symbol_databases` in place too, so
        references to any of them taken earlier stay current. (Replacing :py:attr:`assembly_mysqlport_priority` by
        assignment, as some tests do, is not seen by such references.)
        """
        cls._validate_organism_table(cls.assembly_mysqlport_priority, cls.symbol_databases)

        cls.supported_organisms = tuple(sorted(cls.assembly_mysqlport_priority))

        assemblies = {int(key) for entries in cls.assembly_mysqlport_priority.values() for key in entries}
        cls.all_assemblies.clear()
        cls.all_assemblies.update(assemblies)

        cls.nts_assembly.clear()
        for key in sorted(assemblies):
            cls.nts_assembly[key] = {form: f"assembly_{key}_ensembl_{form}" for form in cls.forms_in_order}

        cls.nts_assembly_reverse.clear()
        for forms in cls.nts_assembly.values():
            for form, node_type in forms.items():
                cls.nts_assembly_reverse[node_type] = cls.nts_ensembl[form]

        cls.nts_assembly_gene.clear()
        cls.nts_assembly_gene.update(forms["gene"] for forms in cls.nts_assembly.values())

    @classmethod
    def register_organism(
        cls,
        name: str,
        assemblies: Mapping[int, Mapping[str, Any]],
        symbol_databases: Optional[Sequence[str]] = None,
    ) -> None:
        """Add an organism to the organism table, or replace its entry, and recompute the derived attributes.

        This is the supported way to try an organism that idtrack does not declare (or to change a declared one)
        within a Python session; to make it permanent, add the same entry to :py:attr:`assembly_mysqlport_priority`
        in ``idtrack/_db.py``. The whole table is validated before anything changes (see
        :py:meth:`_validate_organism_table`), so a rejected entry leaves the table as it was. The table,
        :py:attr:`symbol_databases` and the derived attributes are updated in place, so references to them taken
        earlier stay current.

        Example:
            >>> DB.register_organism(
            ...     "danio_rerio",
            ...     {11: {"Name": "GRCz11", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [92, None]}},
            ...     symbol_databases=["ZFIN"],
            ... )

        Args:
            name: Ensembl production name of the organism (e.g. ``"danio_rerio"``).
            assemblies: ``{assembly key: entry}`` with the fields described in the comment block of
                :py:attr:`assembly_mysqlport_priority`.
            symbol_databases: Ordered gene-symbol databases of the organism (names as in its externals YAML). Required
                for a new organism (an empty list if it has none, in which case :py:class:`idtrack.HarmonizeFeatures`
                needs an explicit ``final_database``). When an organism already in the table is replaced and this is
                omitted (``None``), its current symbol databases are kept.

        Raises:
            ValueError: If *symbol_databases* is omitted for a new organism, or if the resulting table violates a
                rule of :py:meth:`_validate_organism_table`.
        """
        if symbol_databases is None:
            if name not in cls.symbol_databases:
                raise ValueError(
                    f"Organism {name!r} is new: pass `symbol_databases` (its ordered gene-symbol databases, e.g. "
                    "['VGNC Symbol', 'HGNC Symbol'], or an empty list if it has none)."
                )
            symbol_databases = cls.symbol_databases[name]
        table = copy.deepcopy(cls.assembly_mysqlport_priority)
        table[name] = copy.deepcopy(dict(assemblies))
        symbols = dict(cls.symbol_databases)
        symbols[name] = tuple(symbol_databases)
        cls._validate_organism_table(table, symbols)

        cls.assembly_mysqlport_priority[name] = table[name]
        cls.symbol_databases[name] = symbols[name]
        cls.organism_table_version += 1
        cls._rebuild_organism_tables()

    @classmethod
    def _assembly_entry(cls, organism: str, assembly: int) -> Mapping[str, Any]:
        """Return the table entry of ``(organism, assembly)`` or raise a ``ValueError`` naming what is configured."""
        if organism not in cls.assembly_mysqlport_priority:
            raise ValueError(
                f"Organism {organism!r} is not configured. Supported organisms: {cls.supported_organisms}. "
                "Add it with `idtrack.DB.register_organism`."
            )
        entries = cls.assembly_mysqlport_priority[organism]
        if isinstance(assembly, bool) or int(assembly) not in entries:
            raise ValueError(
                f"Genome assembly {assembly!r} is not configured for organism {organism!r}. "
                f"Supported assemblies: {sorted(entries)}"
            )
        return entries[int(assembly)]

    @classmethod
    def db_suffix(cls, organism: str, assembly: int) -> str:
        """Return the assembly code used in core database names (``DbSuffix``, defaulting to the key)."""
        return str(cls._assembly_entry(organism, assembly).get("DbSuffix", str(int(assembly))))

    @classmethod
    def release_range(cls, organism: str, assembly: int) -> tuple[int, Optional[int]]:
        """Return the configured ``(first, last)`` release of an assembly; ``last`` is ``None`` when open-ended."""
        first, last = cls._assembly_entry(organism, assembly)["ReleaseRange"]
        return int(first), (None if last is None else int(last))

    @classmethod
    def assembly_name(cls, organism: str, assembly: int) -> Optional[str]:
        """Return Ensembl's name of an assembly (e.g. ``"Mmul_10"``), or ``None`` if the table does not give one."""
        return cls._assembly_entry(organism, assembly).get("Name")

    @classmethod
    def assemblies_by_priority(cls, organism: str) -> list[int]:
        """Return the assembly keys of an organism ordered by ``Priority`` (preferred first)."""
        if organism not in cls.assembly_mysqlport_priority:
            raise ValueError(
                f"Organism {organism!r} is not configured. Supported organisms: {cls.supported_organisms}. "
                "Add it with `idtrack.DB.register_organism`."
            )
        entries = cls.assembly_mysqlport_priority[organism]
        return sorted((int(k) for k in entries), key=lambda k: int(entries[k]["Priority"]))

    @classmethod
    def core_db_name_pattern(cls, organism: str, assembly: int) -> "re.Pattern[str]":
        """Return the regular expression of the assembly's core database names.

        The pattern is ``^<organism>_core_(?P<release>[0-9]+)_<DbSuffix>(?P<patch>[a-z]*)$``. It does not apply the
        release range; use :py:meth:`assembly_for_core_db` to map a name to its assembly key.

        Args:
            organism: Ensembl production name.
            assembly: Assembly key of the organism.

        Returns:
            re.Pattern[str]: Compiled pattern with the groups ``release`` and ``patch``.
        """
        suffix = re.escape(cls.db_suffix(organism, assembly))
        return re.compile(rf"^{re.escape(organism)}_core_(?P<release>[0-9]+)_{suffix}(?P<patch>[a-z]*)$")

    @classmethod
    def assembly_for_core_db(cls, organism: str, db_suffix: str, release: int) -> Optional[int]:
        """Return the assembly key whose ``DbSuffix`` and ``ReleaseRange`` match a core database, else ``None``.

        Args:
            organism: Ensembl production name.
            db_suffix: Assembly code of the core database name without patch letters (e.g. ``"10"``).
            release: Ensembl release of the core database.

        Returns:
            Optional[int]: The assembly key, or ``None`` if no configured assembly of the organism matches.
        """
        for key in cls.assembly_mysqlport_priority.get(organism, {}):
            first, last = cls.release_range(organism, key)
            if (
                cls.db_suffix(organism, key) == str(db_suffix)
                and first <= int(release)
                and (last is None or int(release) <= last)
            ):
                return int(key)
        return None


DB._rebuild_organism_tables()


class EmptyConversionMetricsError(ValueError):
    """Raised when conversion_metrics is empty and no alternative conversion is possible."""

    pass


MISSING_VALUES = [
    "na",
    "NA",
    "Na",
    "n/a",
    "N/A",
    "n.a.",
    "N.A.",
    "na.",
    "Na.",
    "NA.",
    "n-a",
    "N-A",
    "Na-",
    "NA-",
    "n_a",
    "N_A",
    "Na_",
    "NA_",
    "none",
    "None",
    "NONE",
    "non",
    "Non",
    "NON",
    "void",
    "Void",
    "VOID",
    "blank",
    "Blank",
    "BLANK",
    "omitted",
    "Omitted",
    "OMITTED",
    "NaN",
    "nan",
    "Nan",
    "NAN",
    "-",
    "_",
    "--",
    "__",
    "---",
    "___",
    "null",
    "Null",
    "NULL",
    "nil",
    "Nil",
    "NIL",
    "undefined",
    "Undefined",
    "UnDefined",
    "UNDEFINED",
    "unassigned",
    "Unassigned",
    "UnAssigned",
    "UNASSIGNED",
    "unspecified",
    "Unspecified",
    "UNSPECIFIED",
    "",
    " ",
    "  ",
    "   ",
    "    ",
    ".",
    "..",
    "...",
    "....",
    "?",
    "??",
    "???",
    "????",
    "missing",
    "Missing",
    "MISSING",
    "absent",
    "Absent",
    "ABSENT",
    "EMPTY",
    "empty",
    "Empty",
    "not available",
    "Not Available",
    "NOT AVAILABLE",
    "notavailable",
    "NotAvailable",
    "NOTAVAILABLE",
    "unknown",
    "Unknown",
    "UnKnown",
    "UNKNOWN",
    "not known",
    "Not Known",
    "NOT KNOWN",
    "notknown",
    "NotKnown",
    "NOTKNOWN",
    "undisclosed",
    "Undisclosed",
    "UNDISCLOSED",
    "not disclosed",
    "Not Disclosed",
    "NOT DISCLOSED",
    "notdisclosed",
    "NotDisclosed",
    "NOTDISCLOSED",
    "not applicable",
    "Not Applicable",
    "NOT APPLICABLE",
    "notapplicable",
    "NotApplicable",
    "NOTAPPLICABLE",
    "to be filled",
    "To Be Filled",
    "TO BE FILLED",
    "to be determined",
    "To Be Determined",
    "TO BE DETERMINED",
    "tbd",
    "TBD",
    "tba",
    "TBA",
    "n.k.",
    "N.K.",
    "n.k",
    "N.K",
    "n/k",
    "N/K",
    "n/k/a",
    "N/K/A",
    "n.d.",
    "N.D.",
    "n.d",
    "N.D",
    "n/d",
    "N/D",
    "n/d/a",
    "N/D/A",
    "n.a.p.",
    "N.A.P.",
    "n.a.p",
    "N.A.P",
    "n/a/p",
    "N/A/P",
    "n/a/p/a",
    "N/A/P/A",
    "xx",
    "XX",
    "x",
    "X",
    "xxx",
    "XXX",
    "xxxx",
    "XXXX",
    "N/M",
    "n/m",
    "N.M.",
    "n.m.",
    "n.m",
    "N.M",
    "N/C",
    "n/c",
    "N.C.",
    "n.c.",
    "n.c",
    "N.C",
    "n.s.",
    "N.S.",
    "n.s",
    "N.S",
    "n/s",
    "N/S",
    "n/s/a",
    "N/S/A",
]
