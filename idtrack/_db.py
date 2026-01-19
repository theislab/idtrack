#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


from typing import Any

import h5py


class DB:
    """Store constants shared across IDTrack modules for Ensembl data access and graph construction.

    This class centralizes every constant that multiple components (e.g. :py:class:`idtrack.graph.GraphMaker`,
    :py:class:`idtrack.pathfinder.PathFinder`) rely on when talking to the Ensembl FTP mirror, REST API,
    and public MySQL instances. Housing the values in one immutable namespace prevents circular imports,
    ensures a single source of truth, and simplifies testing. `DB` is never instantiated; import the class
    and reference its attributes directly.

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

        assembly_mysqlport_priority (dict[str, dict[int, dict[str, Any]]]): Organism-aware mapping
            ``{organism -> {assembly -> {...}}}`` defining:
            - `Ports`: ordered list of MySQL ports to try for that assembly
            - `Priority`: assembly priority within the organism (1 = newest / preferred)

        mysql_port_min_release (dict[int, int]): Minimum Ensembl release supported by each public MySQL port.

        all_assemblies (set[int]): Union of every configured assembly code across supported organisms.

        main_assembly (int): Backward-compatibility default assembly (human GRCh38 = 38).

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
        nts_assembly (dict[str, dict[str, str]]): Form-to-assembly-specific node type map.
        nts_assembly_reverse (dict[str, dict[str, str]]): Reverse mapping of :py:data:`nts_assembly`.
        nts_base_ensembl (dict[str, str]): Reduced node type names stripped of assembly suffixes.
        nts_base_ensembl_reverse (dict[str, str]): Reverse mapping of :py:data:`nts_base_ensembl`.

        nts_bidirectional_synonymous_search (set[str]): Node types for which synonym searches are
            performed bidirectionally.
        nts_assembly_gene (set[str]): Every node type that represents a gene, regardless of assembly.

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

    assembly_mysqlport_priority: dict[str, dict[int, dict[str, Any]]] = {
        # Notes
        # -----
        # - The `<assembly>` suffix in `<organism>_core_<release>_<assembly>` is a numeric code (sometimes with a
        #   trailing patch letter like `37a`) and therefore species-specific (e.g. `38` is GRCh38 for human but
        #   GRCm38 for mouse).
        # - Some (organism, assembly) pairs are hosted on multiple ports depending on release; we list ports in
        #   preference order and let DatabaseManager pick the first port that contains the requested release.
        # - "ReleaseRange" specifies the expected range of releases for this assembly [min, max].
        #   Use None for max to indicate "current latest and beyond".
        "homo_sapiens": {
            # GRCh38 (Ensembl release 76+; primary assembly since release 76)
            38: {"Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [76, None]},
            # GRCh37 (Ensembl release 55+; archive since release 76)
            # - MySQL port 3337 hosts GRCh37 **release 79 onwards** (archive service).
            # - Some GRCh37 schemas can also appear on the main ports (3306/5306) for select releases; keep them
            #   as fallbacks for environments where 3337 is unreachable.
            # - Releases 55–78 are accessed via the HTTPS/FTP MySQL dumps (DatabaseManager FTP fallback).
            37: {"Ports": [3337, 3306, 5306], "Priority": 2, "ReleaseRange": [55, None]},
            # NCBI36 (historic; clean handoff to GRCh37 at Ensembl release 55)
            # - Ensembl FTP contains releases 48–54 under directory names like
            #   `homo_sapiens_core_<release>_36<patch>` (e.g. `homo_sapiens_core_54_36p`).
            # - Not expected to be available on the live public MySQL service anymore; accessed via HTTPS/FTP dumps.
            36: {"Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 54]},
        },
        "mus_musculus": {
            # GRCm39 (Ensembl release 103+; clean handoff from GRCm38 at release 103)
            39: {"Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [103, None]},
            # GRCm38 (Ensembl release 68–102)
            # - Port 3337 also hosts a subset of GRCm38 releases (archive mirror); keep it as a fallback.
            38: {"Ports": [3306, 5306, 3337], "Priority": 2, "ReleaseRange": [68, 102]},
            # GRCm37 / NCBIm37 (Ensembl release 48–67; clean handoff to GRCm38 at release 68)
            # - Early releases can carry a patch-letter suffix (e.g. `..._37a`) and are handled by DatabaseManager.
            37: {"Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 67]},
        },
        "sus_scrofa": {
            # Sscrofa11.1 (Ensembl release 90+; clean handoff from Sscrofa10.2 at release 90)
            111: {"Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [90, None]},
            # Sscrofa10.2 (archive) — Ensembl release 67–89
            # - Port 3337 hosts later archive releases (79–99); keep it as a fallback when present.
            102: {"Ports": [3306, 5306, 3337], "Priority": 2, "ReleaseRange": [67, 89]},
            # Sscrofa9.2 (historic; clean handoff to Sscrofa10.2 at release 67)
            # - Ensembl FTP contains releases 56–66 under directory names like
            #   `sus_scrofa_core_<release>_9<patch>` (e.g. `sus_scrofa_core_60_9d`).
            # - Not expected to be available on the live public MySQL service anymore; accessed via HTTPS/FTP dumps.
            9: {"Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [56, 66]},
        },
    }

    supported_organisms = tuple(sorted(assembly_mysqlport_priority.keys()))

    # NOTE: Do not use comprehensions that reference other class attributes inside the class body.
    # In Python 3, comprehensions execute in their own scope and cannot "see" class-local names.
    all_assemblies: set[int] = set()
    for _org, _assemblies in assembly_mysqlport_priority.items():
        for _asm in _assemblies:
            all_assemblies.add(int(_asm))

    # Backward-compatibility default for older code/tests.
    main_assembly = 38

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

    nts_assembly: dict[int, dict[str, str]] = {}  # assembly_37_ensembl_gene
    for _asm in all_assemblies:
        nts_assembly[_asm] = {}
        for _form in ["gene", "transcript", "translation"]:
            nts_assembly[_asm][_form] = f"assembly_{_asm}_ensembl_{_form}"

    nts_assembly_reverse = dict()
    for i in nts_assembly:
        for j in nts_assembly[i]:
            nts_assembly_reverse[nts_assembly[i][j]] = nts_ensembl[j]

    nts_base_ensembl: dict[str, str] = {}
    for _form in forms_in_order:
        nts_base_ensembl[_form] = f"base_ensembl_{_form}"
    nts_base_ensembl_reverse: dict[str, str] = {}
    for _form, _nts in nts_base_ensembl.items():
        nts_base_ensembl_reverse[_nts] = _form
    nts_bidirectional_synonymous_search = {nts_external, nts_base_ensembl[backbone_form]}

    # Only gene assembly genes
    nts_assembly_gene = set()
    for ntas1 in nts_assembly:
        nts_assembly_gene.add(nts_assembly[ntas1]["gene"])

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
