"""The organisms idtrack knows: their genome assemblies, release ranges and order, symbol databases and short names.

The one table of organisms is :py:data:`ORGANISM_TABLE` (``{organism: {assembly key: entry}}``); change it only with
:py:func:`register_organism`, which validates the whole table first and recomputes what is derived from it. The
functions of this module answer questions about it (the first release of an organism, its assemblies by priority,
the assembly that serves a release) without the network, and :py:func:`resolve_organism` turns an organism's name or
short name into the name of the table.

This module imports only :py:mod:`idtrack.errors`, :py:mod:`idtrack.ids` and :py:mod:`idtrack.settings`.
"""

from __future__ import annotations

import copy
import numbers
import re
from collections.abc import Mapping, Sequence
from typing import Any

from idtrack import ids, settings
from idtrack.errors import InvalidArgumentError, UnknownAssemblyError, UnknownOrganismError, UnknownReleaseError

#: The table of the organisms of idtrack, ``{organism: {assembly key: entry}}``, each entry with the assembly's
#: ``Name``, MySQL ``Ports``, ``Priority`` and ``ReleaseRange`` (and ``DbSuffix`` where Ensembl reuses a code).
ORGANISM_TABLE: dict[str, dict[int, dict[str, Any]]] = {
    # The one table of organisms. Change it with `register_organism`, which validates it and recomputes everything
    # derived from it (`supported_organisms()`, the `assembly_<key>_ensembl_<form>` node types).
    #
    # Keys and fields
    # ---------------
    # - The key of an assembly entry is an integer that is unique within the organism. It is what the rest of
    #   idtrack sees: node types (`assembly_<key>_ensembl_gene`), table folders
    #   (`idtrack_tables/<organism>_assembly-<key>/`), graph file names (`graph_<organism>_assembly<key>_...`), the
    #   assembly keys of the externals YAML files and the `assembly` argument of `DatabaseManager`.
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
    #   48 are out of reach regardless (`idtrack.settings.ENSEMBL.mysql_port_min_release`).
    # - "Ports" are the MySQL ports to try in preference order; DatabaseManager picks the first port that
    #   contains the requested release. A port of `idtrack.settings.ENSEMBL.mysql_port_assembly` may be listed only by its
    #   assembly.
    #   "Priority" orders the assemblies of an organism (1 = preferred, newest).
    # - "Name" is the assembly name as Ensembl gives it (`assembly.default` in the core database's `meta`
    #   table). idtrack does not read it at run time; the live MySQL tests check it against every core database
    #   the table attributes to the assembly.
    "homo_sapiens": {
        # GRCh38 (Ensembl release 76+; primary assembly since release 76)
        38: {"Name": "GRCh38", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [76, None]},
        # GRCh37 (Ensembl release 55+; archive since release 76): the one parallel assembly Ensembl maintains,
        # on its archive port 3337 (`idtrack.settings.ENSEMBL.mysql_port_assembly`).
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
        38: {"Name": "GRCm38", "Ports": [3306, 5306], "Priority": 2, "ReleaseRange": [68, 102]},
        # NCBIM37 (Ensembl release 48–67; clean handoff to GRCm38 at release 68)
        # - Early releases can carry a patch-letter suffix (e.g. `..._37a`) and are handled by DatabaseManager.
        37: {"Name": "NCBIM37", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 67]},
    },
    "sus_scrofa": {
        # Sscrofa11.1 (Ensembl release 90+; clean handoff from Sscrofa10.2 at release 90)
        111: {"Name": "Sscrofa11.1", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [90, None]},
        # Sscrofa10.2 (Ensembl release 67–89). Core databases named `sus_scrofa_core_90_102` to `..._99_102`
        # exist only on the archive port 3337: frozen copies, not annotation of those releases, and excluded on
        # purpose (see `idtrack.settings.ENSEMBL.mysql_port_assembly`).
        102: {"Name": "Sscrofa10.2", "Ports": [3306, 5306], "Priority": 2, "ReleaseRange": [67, 89]},
        # Sscrofa9 (historic; clean handoff to Sscrofa10.2 at release 67)
        # - Ensembl FTP contains releases 56–66 under directory names like
        #   `sus_scrofa_core_<release>_9<patch>` (e.g. `sus_scrofa_core_60_9d`).
        # - Not expected to be available on the live public MySQL service anymore; accessed via HTTPS/FTP dumps.
        9: {"Name": "Sscrofa9", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [56, 66]},
    },
    "macaca_mulatta": {
        # Rhesus macaque. Clean handoffs: MMUL_1 up to 85, Mmul_8.0.1 86–97, Mmul_10 from 98 (one assembly per
        # release in the dump listings). Ports 3306/5306 serve releases 48+ (3337: see
        # `idtrack.settings.ENSEMBL.mysql_port_assembly`).
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


#: Ordered gene-symbol databases per organism (names as in the externals YAML files). The first one is the default
#: final database of :py:class:`idtrack.harmonize.HarmonizeFeatures`. Set together with the table by
#: :py:func:`register_organism`.
SYMBOL_DATABASES: dict[str, tuple[str, ...]] = {
    "homo_sapiens": ("HGNC Symbol",),
    "mus_musculus": ("MGI Symbol",),
    # Pig and the macaques carry VGNC symbols on their current assemblies and HGNC symbols on earlier ones.
    "sus_scrofa": ("VGNC Symbol", "HGNC Symbol"),
    "macaca_mulatta": ("VGNC Symbol", "HGNC Symbol"),
    "macaca_fascicularis": ("VGNC Symbol", "HGNC Symbol"),
}

#: Ensembl's name of NCBI Gene's database (``external_db.db_name``, the same at every release). Its accessions are the
#: NCBI gene numbers, its label rows NCBI's official symbols, which a graph names by one name at every release (the name
#: Ensembl's tables give them: :py:attr:`idtrack.TheGraph.label_names`); NCBI's symbols are the symbols of a name
#: authority beside :py:data:`SYMBOL_DATABASES` (:py:attr:`idtrack.TheGraph.authority_databases`).
NCBI_GENE_DATABASE = "EntrezGene"

#: The pseudoautosomal regions of an assembly, by Ensembl's name of the assembly: pairs of a region of Y and the region
#: of X that it repeats, ``((region, first base, last base), (region, first base, last base))`` (the Genome Reference
#: Consortium's PAR1 and PAR2 of GRCh38). From release 110 Ensembl gives the Y copies of the genes of these regions ids
#: of their own. Ensembl records the same regions in its core databases as assembly exceptions of
#: the type ``PAR`` (table ``assembly_exception``), which a built graph holds since the builder reads the exceptions
#: for the copies on patches and haplotypes (:py:attr:`idtrack.TheGraph.assembly_exceptions`; GRCh38's two ``PAR``
#: rows equal this table); the rule of the Y copies still reads this table, which also serves graphs made in memory.
#: It holds only the assembly whose Y copies have ids of their own.
PSEUDOAUTOSOMAL_REGIONS: dict[str, tuple[tuple[tuple[str, int, int], tuple[str, int, int]], ...]] = {
    "GRCh38": (
        (("Y", 10_001, 2_781_479), ("X", 10_001, 2_781_479)),
        (("Y", 56_887_903, 57_217_415), ("X", 155_701_383, 156_030_895)),
    ),
}


#: Short names of the organisms. A declared alias names its organism wherever an organism is asked for, without the
#: network, and wins over what Ensembl lists under the same word (``"pig"`` is the common name of every pig breed
#: there). Lower case, unique across organisms.
ALIASES: dict[str, tuple[str, ...]] = {
    "homo_sapiens": ("human",),
    "mus_musculus": ("mouse",),
    "sus_scrofa": ("pig",),
    "macaca_mulatta": ("rhesus",),
    "macaca_fascicularis": ("cynomolgus",),
}

#: The label under which a graph reports the Ensembl ids of each form on each assembly key,
#: ``{assembly: {form: "assembly_<key>_ensembl_<form>"}}``: the source of an Ensembl id on an assembly in
#: :py:meth:`idtrack.TheGraph.trios` and :py:meth:`idtrack.TheGraph.combined`, in the inference of the source of a
#: list, and in the self-checks (derived from the table; updated in place by :py:func:`register_organism`).
ASSEMBLY_LABELS: dict[int, dict[str, str]] = {}

_ENTRY_FIELDS = ("Name", "DbSuffix", "Ports", "Priority", "ReleaseRange")
_REQUIRED_FIELDS = ("Ports", "Priority", "ReleaseRange")
_state: dict[str, Any] = {"version": 0, "supported": ()}


def supported_organisms() -> tuple[str, ...]:
    """Return the organisms of the table, sorted.

    Returns:
        tuple[str, ...]: Ensembl production names (``homo_sapiens``, ...).
    """
    return _state["supported"]


def table_version() -> int:
    """Return a number that changes whenever :py:func:`register_organism` changes the table.

    Caches derived from the table (the index of Ensembl's core databases) are keyed by it.

    Returns:
        int: The version of the table in this process.
    """
    return _state["version"]


def _invalid(message: str) -> InvalidArgumentError:
    return InvalidArgumentError(message, argument="organism table")


def _validate(
    table: Mapping[Any, Any],
    symbol_databases: Mapping[Any, Any],
    aliases: Mapping[Any, Any] | None = None,
) -> None:
    """Check the organism table and the symbol-database table for consistency.

    Rules: organism names are lower-case Ensembl production names; every organism has at least one assembly;
    assembly keys are positive integers; each entry carries ``Ports`` (known MySQL ports,
    :py:attr:`idtrack.settings.EnsemblSettings.mysql_port_min_release`, and none that
    :py:attr:`idtrack.settings.EnsemblSettings.mysql_port_assembly` reserves for another assembly), ``Priority`` and
    ``ReleaseRange``, plus optionally ``Name`` and ``DbSuffix`` and nothing else; priorities are unique per organism and
    start at 1; release ranges are ``[first, last]`` with ``last`` ``None`` or not smaller than ``first``; entries of one
    organism that share a ``DbSuffix`` have non-overlapping release ranges; the symbol-database table names exactly
    the organisms of the organism table; aliases belong to organisms of the table, are lower-case words that are no
    organism name, and no alias is declared for two organisms.

    Args:
        table: Candidate value of :py:data:`ORGANISM_TABLE`.
        symbol_databases: Candidate value of :py:data:`SYMBOL_DATABASES`.
        aliases: Candidate value of :py:data:`ALIASES`; ``None`` for a table without aliases.

    Raises:
        InvalidArgumentError: If any rule is violated; the message names the organism, the assembly and the rule.
    """
    port_min_release = settings.ENSEMBL.mysql_port_min_release
    port_assembly = settings.ENSEMBL.mysql_port_assembly
    if not isinstance(table, Mapping) or not table:
        raise _invalid("The organism table must be a non-empty mapping {organism: {assembly key: {...}}}.")

    for organism, assemblies in table.items():
        if not isinstance(organism, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", organism):
            raise _invalid(
                f"Organism name {organism!r} must be an Ensembl production name (lower case, e.g. 'homo_sapiens')."
            )
        if not isinstance(assemblies, Mapping) or not assemblies:
            raise _invalid(f"Organism {organism!r} must declare at least one assembly.")

        priorities: list[int] = []
        windows_by_suffix: dict[str, list[tuple[int, float, int]]] = {}
        for key, entry in assemblies.items():
            where = f"{organism!r} assembly {key!r}"
            if isinstance(key, bool) or not isinstance(key, int) or key <= 0:
                raise _invalid(f"{where}: assembly keys must be positive integers.")
            if not isinstance(entry, Mapping):
                raise _invalid(f"{where}: the entry must be a mapping.")
            unknown = sorted(set(entry) - set(_ENTRY_FIELDS))
            if unknown:
                raise _invalid(f"{where}: unknown fields {unknown}; allowed: {list(_ENTRY_FIELDS)}.")
            missing = [f for f in _REQUIRED_FIELDS if f not in entry]
            if missing:
                raise _invalid(f"{where}: missing required fields {missing}.")

            ports = entry["Ports"]
            if (
                not isinstance(ports, (list, tuple))
                or not ports
                or any(isinstance(p, bool) or not isinstance(p, int) for p in ports)
                or len(set(ports)) != len(ports)
            ):
                raise _invalid(f"{where}: 'Ports' must be a non-empty list of distinct integers.")
            unknown_ports = sorted(set(ports) - set(port_min_release))
            if unknown_ports:
                raise _invalid(f"{where}: unknown MySQL ports {unknown_ports}; known: {sorted(port_min_release)}.")
            foreign_ports = sorted(p for p in ports if port_assembly.get(p, (organism, key)) != (organism, key))
            if foreign_ports:
                raise _invalid(
                    f"{where}: MySQL ports {foreign_ports} serve only {[port_assembly[p] for p in foreign_ports]} "
                    "(see `idtrack.settings.ENSEMBL.mysql_port_assembly`)."
                )

            priority = entry["Priority"]
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise _invalid(f"{where}: 'Priority' must be an integer.")
            priorities.append(priority)

            window = entry["ReleaseRange"]
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise _invalid(f"{where}: 'ReleaseRange' must be [first, last] (last may be None).")
            first, last = window
            if isinstance(first, bool) or not isinstance(first, int) or first < 1:
                raise _invalid(f"{where}: the first release of 'ReleaseRange' must be a positive integer.")
            if last is not None and (isinstance(last, bool) or not isinstance(last, int) or last < first):
                raise _invalid(f"{where}: the last release of 'ReleaseRange' must be None or >= {first}.")

            name = entry.get("Name")
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise _invalid(f"{where}: 'Name' must be a non-empty string.")

            suffix = entry.get("DbSuffix", str(key))
            if not isinstance(suffix, str) or not re.fullmatch(r"[0-9]+", suffix):
                raise _invalid(f"{where}: 'DbSuffix' must be a string of digits (e.g. '10').")
            windows_by_suffix.setdefault(suffix, []).append((first, float("inf") if last is None else float(last), key))

        if len(set(priorities)) != len(priorities) or min(priorities) != 1:
            raise _invalid(
                f"Organism {organism!r}: priorities must be unique and start at 1 (got {sorted(priorities)})."
            )

        for suffix, windows in windows_by_suffix.items():
            windows.sort()
            for (_first_a, last_a, key_a), (first_b, _last_b, key_b) in zip(windows, windows[1:]):
                if first_b <= last_a:
                    raise _invalid(
                        f"Organism {organism!r}: assemblies {key_a} and {key_b} share the core-database suffix "
                        f"{suffix!r} and their release ranges overlap; give them disjoint 'ReleaseRange's."
                    )

    if not isinstance(symbol_databases, Mapping):
        raise _invalid("The symbol-database table must be a mapping {organism: (database, ...)}.")
    if set(symbol_databases) != set(table):
        raise _invalid(
            "The symbol-database table must name exactly the organisms of the organism table "
            f"(missing: {sorted(set(table) - set(symbol_databases))}, "
            f"extra: {sorted(set(symbol_databases) - set(table))})."
        )
    for organism, databases in symbol_databases.items():
        if isinstance(databases, str) or not isinstance(databases, Sequence):
            raise _invalid(f"Organism {organism!r}: symbol databases must be a sequence of names.")
        if any(not isinstance(d, str) or not d for d in databases) or len(set(databases)) != len(databases):
            raise _invalid(f"Organism {organism!r}: symbol databases must be distinct non-empty strings.")

    if aliases is None:
        return
    if not isinstance(aliases, Mapping) or not set(aliases) <= set(table):
        raise _invalid("The alias table must be a mapping {organism of the organism table: (alias, ...)}.")
    owner_of: dict[str, str] = {}
    for organism, names in aliases.items():
        if isinstance(names, str) or not isinstance(names, Sequence):
            raise _invalid(f"Organism {organism!r}: aliases must be a sequence of names.")
        for alias in names:
            if not isinstance(alias, str) or not alias or alias != alias.strip().lower() or alias in table:
                raise _invalid(
                    f"Organism {organism!r}: alias {alias!r} must be a non-empty lower-case name that is not "
                    "an organism name."
                )
            if owner_of.setdefault(alias, organism) != organism or list(names).count(alias) > 1:
                raise _invalid(f"Alias {alias!r} is declared twice ({owner_of[alias]!r}, {organism!r}).")


def _rebuild() -> None:
    """Validate the organism table and recompute everything derived from it (at import and on registration).

    The derived dictionaries are updated in place, and :py:func:`register_organism` changes the table and
    :py:data:`SYMBOL_DATABASES` in place too, so references to any of them taken earlier stay current.
    """
    _validate(ORGANISM_TABLE, SYMBOL_DATABASES, ALIASES)
    _state["supported"] = tuple(sorted(ORGANISM_TABLE))
    assemblies = sorted({int(key) for entries in ORGANISM_TABLE.values() for key in entries})
    ASSEMBLY_LABELS.clear()
    for key in assemblies:
        ASSEMBLY_LABELS[key] = {form: f"assembly_{key}_ensembl_{form}" for form in ids.FORMS}


def register_organism(
    name: str,
    assemblies: Mapping[int, Mapping[str, Any]],
    symbol_databases: Sequence[str] | None = None,
    aliases: Sequence[str] | None = None,
) -> None:
    """Add an organism to the organism table, or replace its entry, and recompute what is derived from the table.

    This is the supported way to try an organism that idtrack does not declare (or to change a declared one) within a
    Python session; to make it permanent, add the same entry to :py:data:`ORGANISM_TABLE` in
    ``idtrack/organisms.py``. The whole table is validated before anything changes, so a rejected entry leaves the
    table as it was. The table, :py:data:`SYMBOL_DATABASES` and the derived values are updated in place, so references
    to them taken earlier stay current.

    Example:
        >>> from idtrack import organisms
        >>> organisms.register_organism(
        ...     "danio_rerio",
        ...     {11: {"Name": "GRCz11", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [92, None]}},
        ...     symbol_databases=["ZFIN"],
        ... )

    Args:
        name: Ensembl production name of the organism (e.g. ``"danio_rerio"``).
        assemblies: ``{assembly key: entry}`` with the fields described in the comment block of
            :py:data:`ORGANISM_TABLE`.
        symbol_databases: Ordered gene-symbol databases of the organism (names as in its externals YAML). Required
            for a new organism (an empty list if it has none, in which case
            :py:class:`idtrack.harmonize.HarmonizeFeatures` needs an explicit ``final_database``). When an organism
            already in the table is replaced and this is omitted (``None``), its current symbol databases are kept.
        aliases: Short names of the organism (lower case, e.g. ``["zebrafish"]``), which then resolve to it wherever
            an organism is asked for. ``None`` keeps the aliases of an organism that is replaced and gives a new
            organism none.

    Raises:
        InvalidArgumentError: If *symbol_databases* is omitted for a new organism, or if the resulting table violates
            a rule of the table.
    """
    if symbol_databases is None:
        if name not in SYMBOL_DATABASES:
            raise _invalid(
                f"Organism {name!r} is new: pass `symbol_databases` (its ordered gene-symbol databases, e.g. "
                "['VGNC Symbol', 'HGNC Symbol'], or an empty list if it has none)."
            )
        symbol_databases = SYMBOL_DATABASES[name]
    # The validator only reads, so the candidate table may hold the caller's mappings as they are; the stored entry
    # is a copy made of plain dictionaries, whatever mapping types the caller passed.
    table: dict[str, Mapping[int, Mapping[str, Any]]] = dict(ORGANISM_TABLE)
    table[name] = assemblies
    symbols = dict(SYMBOL_DATABASES)
    symbols[name] = tuple(symbol_databases)
    declared: dict[str, Sequence[str]] = dict(ALIASES)
    if aliases is not None:
        declared[name] = aliases
    _validate(table, symbols, declared)

    ORGANISM_TABLE[name] = {int(key): copy.deepcopy(dict(entry)) for key, entry in assemblies.items()}
    SYMBOL_DATABASES[name] = symbols[name]
    if name in declared:
        ALIASES[name] = tuple(declared[name])
    _state["version"] += 1
    _rebuild()


def check_organism(organism: Any) -> str:
    """Return *organism* if it is an organism of the table, else raise naming the organisms there are.

    Args:
        organism: The value given.

    Returns:
        str: The organism.

    Raises:
        UnknownOrganismError: If it is not an organism of the table.
    """
    if not isinstance(organism, str) or organism not in ORGANISM_TABLE:
        raise UnknownOrganismError(
            f"Organism {organism!r} is not configured. Supported organisms: {supported_organisms()}. "
            "Add it with `idtrack.organisms.register_organism`.",
            argument="organism",
            value=organism,
            allowed=supported_organisms(),
        )
    return organism


def resolve_organism(name: str) -> str:
    """Return the organism of the table that a name means, without the network.

    Accepted: the Ensembl production name (``homo_sapiens``), the same with spaces or capitals (``Homo sapiens``),
    and a declared short name (``human``, :py:data:`ALIASES`). Ensembl's other names (taxon ids, common names it
    lists, assembly names) need Ensembl's species list: :py:meth:`idtrack.API.ensembl_organism`.

    Args:
        name: The name.

    Returns:
        str: The organism's production name.

    Raises:
        UnknownOrganismError: If the name means no organism of the table.
    """
    if isinstance(name, str):
        key = re.sub(r"[\s_]+", "_", name.strip().lower())
        if key in ORGANISM_TABLE:
            return key
        for organism, names in ALIASES.items():
            if name.strip().lower() in names:
                return organism
    raise UnknownOrganismError(
        f"{name!r} names no organism of idtrack's table ({', '.join(supported_organisms())}, or a short name: "
        f"{', '.join(a for names in ALIASES.values() for a in names)}).",
        argument="organism",
        value=name,
        allowed=supported_organisms(),
    )


def assembly_key(value: Any) -> int:
    """Return *value* as an assembly key of the organism table, refusing values that only look like one.

    Accepted: an integer (also a NumPy integer), an integral float (``38.0``) or a string of digits (as read from
    YAML or JSON). Booleans and fractional numbers are refused: ``int(True)`` and ``int(38.9)`` would silently name
    another assembly (``1`` is rhesus MMUL_1). This is the one rule for every place that takes an assembly from a
    caller.

    Args:
        value: The value given by the caller.

    Returns:
        int: The assembly key.

    Raises:
        InvalidArgumentError: For any other value.
    """
    # NumPy booleans are neither `bool` nor registered numbers, so they reach the last line like any other type.
    if not isinstance(value, bool):
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real) and float(value).is_integer():
            return int(float(value))
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    raise InvalidArgumentError(
        f"Genome assembly must be an integer key of the organism table, got {value!r}.",
        argument="assembly",
        value=value,
    )


def _entry(organism: str, assembly: Any) -> Mapping[str, Any]:
    """Return the table entry of ``(organism, assembly)`` or raise naming what is configured."""
    entries = ORGANISM_TABLE[check_organism(organism)]
    key = assembly_key(assembly)
    if key not in entries:
        raise UnknownAssemblyError(
            f"Genome assembly {assembly!r} is not configured for organism {organism!r}. "
            f"Supported assemblies: {sorted(entries)}",
            argument="assembly",
            value=assembly,
            allowed=sorted(entries),
        )
    return entries[key]


def db_suffix(organism: str, assembly: Any) -> str:
    """Return the assembly code used in core database names (``DbSuffix``, defaulting to the key).

    Args:
        organism: Ensembl production name.
        assembly: Assembly key.

    Returns:
        str: The code.
    """
    return str(_entry(organism, assembly).get("DbSuffix", str(assembly_key(assembly))))


def release_range(organism: str, assembly: Any) -> tuple[int, int | None]:
    """Return the configured ``(first, last)`` release of an assembly; ``last`` is ``None`` when open-ended.

    Args:
        organism: Ensembl production name.
        assembly: Assembly key.

    Returns:
        tuple[int, int | None]: The release range.
    """
    first, last = _entry(organism, assembly)["ReleaseRange"]
    return int(first), (None if last is None else int(last))


def assemblies_by_priority(organism: str) -> list[int]:
    """Return the assembly keys of an organism ordered by ``Priority`` (preferred first).

    Args:
        organism: Ensembl production name.

    Returns:
        list[int]: The assembly keys.
    """
    entries = ORGANISM_TABLE[check_organism(organism)]
    return sorted((int(k) for k in entries), key=lambda k: int(entries[k]["Priority"]))


def first_release(organism: str) -> int:
    """Return the first Ensembl release of an organism in the organism table (over all its assemblies).

    This is the default start of every release window (``first_release`` of
    :py:class:`idtrack.build.DatabaseManager` and of the graphs :py:meth:`idtrack.API.build` builds): a window that
    starts there covers the whole history that idtrack can read for the organism.

    Args:
        organism: Ensembl production name.

    Returns:
        int: The smallest first release of the ``ReleaseRange`` of the organism's assemblies.
    """
    return min(release_range(organism, assembly)[0] for assembly in assemblies_by_priority(organism))


def primary_assembly(organism: str, snapshot_release: int) -> int:
    """Return the preferred assembly of an organism that has a release: the primary assembly of a graph.

    Args:
        organism: Ensembl production name.
        snapshot_release: The last release of the graph.

    Returns:
        int: The assembly key.

    Raises:
        UnknownReleaseError: If no assembly of the organism has the release.
    """
    for assembly in assemblies_by_priority(organism):
        first, last = release_range(organism, assembly)
        if first <= snapshot_release and (last is None or snapshot_release <= last):
            return int(assembly)
    raise UnknownReleaseError(
        f"No assembly of {organism} has release {snapshot_release} in the organism table.",
        argument="snapshot_release",
        value=snapshot_release,
    )


def assembly_name(organism: str, assembly: Any) -> str | None:
    """Return Ensembl's name of an assembly (e.g. ``"Mmul_10"``), or ``None`` if the table does not give one.

    Args:
        organism: Ensembl production name.
        assembly: Assembly key.

    Returns:
        str | None: The name.
    """
    return _entry(organism, assembly).get("Name")


def pseudoautosomal_repeats(organism: str | None) -> dict[int, tuple[tuple[str, int, int, str, int], ...]]:
    """Return, per assembly key of an organism, its pseudoautosomal regions of Y with the region of X each repeats.

    Read from :py:data:`PSEUDOAUTOSOMAL_REGIONS` by the assembly's name; the one rule of a gene reads a place in such a
    region of Y on the region of X it repeats (:py:mod:`idtrack.graph.genes`).

    Args:
        organism: Ensembl production name (``None`` or an organism outside the table: none).

    Returns:
        dict[int, tuple[tuple[str, int, int, str, int], ...]]: ``{assembly: ((region of Y, first base, last base,
        region of X, its first base), ...)}`` for the assemblies that have such regions.
    """
    found: dict[int, tuple[tuple[str, int, int, str, int], ...]] = {}
    for assembly, entry in ORGANISM_TABLE.get(organism or "", {}).items():
        pairs = PSEUDOAUTOSOMAL_REGIONS.get(entry.get("Name") or "", ())
        if pairs:
            found[int(assembly)] = tuple((y[0], y[1], y[2], x[0], x[1]) for y, x in pairs)
    return found


def core_db_name_pattern(organism: str, assembly: Any) -> re.Pattern[str]:
    """Return the regular expression of the assembly's core database names.

    The pattern is ``^<organism>_core_(?P<release>[0-9]+)_<DbSuffix>(?P<patch>[a-z]*)$``. It does not apply the
    release range.

    Args:
        organism: Ensembl production name.
        assembly: Assembly key of the organism.

    Returns:
        re.Pattern[str]: Compiled pattern with the groups ``release`` and ``patch``.
    """
    suffix = re.escape(db_suffix(organism, assembly))
    return re.compile(rf"^{re.escape(organism)}_core_(?P<release>[0-9]+)_{suffix}(?P<patch>[a-z]*)$")


_rebuild()

__all__ = [
    "ALIASES",
    "ASSEMBLY_LABELS",
    "NCBI_GENE_DATABASE",
    "PSEUDOAUTOSOMAL_REGIONS",
    "ORGANISM_TABLE",
    "SYMBOL_DATABASES",
    "assemblies_by_priority",
    "assembly_key",
    "assembly_name",
    "check_organism",
    "core_db_name_pattern",
    "pseudoautosomal_repeats",
    "db_suffix",
    "first_release",
    "primary_assembly",
    "register_organism",
    "release_range",
    "resolve_organism",
    "supported_organisms",
    "table_version",
]
