"""Graph files: their names, the identity a file records, and finding the files of a folder.

A graph file's name tells graphs apart that can sit in one folder:
``graph_<organism>_assembly<primary assembly>_min<first>_max<last>[_narrow][_forms<a-b>][_extall].npz``. What a name
does not say (the selection of external databases, the organism table, the format version) is in the manifest inside
the file (:py:meth:`idtrack.graph.TheGraph.load` checks it). Nothing here needs the network, a table or PyYAML: the
selection of external databases is given as parsed content.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any

from idtrack import organisms

GRAPH_FILE_NAME = re.compile(
    r"^graph_(?P<organism>.+)_assembly(?P<assembly>-?[0-9]+)(?:_min(?P<first_release>[0-9]+))?"
    r"(?:_max(?P<snapshot_release>[0-9]+))?(?P<narrow>_narrow)?(?:_forms(?P<forms>[a-z-]+))?(?P<extall>_extall)?\.npz$"
)


def _limit(value: int | float | None) -> int | None:
    return None if value is None or (isinstance(value, float) and math.isinf(value)) else int(value)


def graph_file_path(
    local_repository: str,
    organism: str,
    assembly: int,
    first_release: int | float | None,
    snapshot_release: int | float | None,
    narrow: bool = True,
    forms: list[str] | None = None,
    narrow_external: bool = True,
) -> str:
    """Return the path of a graph file; needs no manager and no network.

    Args:
        local_repository: Folder of the graph files.
        organism: Organism name, e.g. ``homo_sapiens``.
        assembly: Primary assembly of the graph (key of the organism table).
        first_release: First release of the window (infinity or ``None`` for no limit).
        snapshot_release: Last release of the window (infinity or ``None`` for no limit).
        narrow: See :py:meth:`idtrack.build.GraphMaker.construct_graph`.
        forms: The forms if they are not all forms; ``None`` for all.
        narrow_external: See :py:meth:`idtrack.build.GraphMaker.construct_graph`.

    Returns:
        str: The path.
    """
    first, last = _limit(first_release), _limit(snapshot_release)
    min_ext = "" if first is None else f"_min{first}"
    max_ext = "" if last is None else f"_max{last}"
    narrow_ext = "_narrow" if narrow else ""
    forms_ext = f"_forms{'-'.join(forms)}" if forms else ""
    external_ext = "" if narrow_external else "_extall"
    name = f"graph_{organism}_assembly{int(assembly)}{min_ext}{max_ext}{narrow_ext}{forms_ext}{external_ext}.npz"
    return os.path.join(local_repository, name)


def list_graph_files(local_repository: str, organism: str | None = None) -> list[dict[str, Any]]:
    """List the graph files of a folder with what their names say.

    Args:
        local_repository: Folder to look in.
        organism: Only graphs of this organism; ``None`` for all.

    Returns:
        list[dict]: One entry per file, sorted by name: ``path``, ``organism``, ``assembly``,
        ``first_release``, ``snapshot_release`` (``None`` where the name has no limit), ``narrow``, ``forms`` (``None``
        for all forms) and ``narrow_external``.
    """
    found: list[dict[str, Any]] = []
    if not os.path.isdir(local_repository):
        return found
    for name in sorted(os.listdir(local_repository)):
        match = GRAPH_FILE_NAME.match(name)
        if not match or (organism is not None and match.group("organism") != organism):
            continue
        found.append(
            {
                "path": os.path.join(local_repository, name),
                "organism": match.group("organism"),
                "assembly": int(match.group("assembly")),
                "first_release": None if match.group("first_release") is None else int(match.group("first_release")),
                "snapshot_release": (
                    None if match.group("snapshot_release") is None else int(match.group("snapshot_release"))
                ),
                "narrow": match.group("narrow") is not None,
                "forms": None if match.group("forms") is None else match.group("forms").split("-"),
                "narrow_external": match.group("extall") is None,
            }
        )
    return found


def selection_fingerprint(content: dict | None, organism: str, forms: list[str], first: Any, last: Any) -> str:
    """Return a digest of the selection of external databases that takes effect in a graph.

    Of a parsed externals YAML this is, per form, database and assembly, the included releases that lie in the
    window of the graph and in the release range the organism table gives the assembly. An entry that can never
    produce an edge (a database that is not included, a release outside the window, an assembly at releases it does
    not have, an assembly the organism does not have) is not part of the identity of a graph: tidying such entries in
    a YAML file leaves the graphs built with it readable.

    Args:
        content: The parsed YAML, or ``None``.
        organism: Organism of the graph.
        forms: The forms of the graph.
        first: First release of the window (may be infinite).
        last: Last release of the window (may be infinite).

    Returns:
        str: The digest.
    """
    table = organisms.ORGANISM_TABLE[organism]
    selected = (content or {}).get(organism) or {}
    selection = []
    for form in sorted(forms):
        for database in sorted(selected.get(form) or {}):
            per_assembly = ((selected[form][database] or {}).get("Assembly")) or {}
            for assembly in sorted(per_assembly, key=lambda a: int(a)):
                item = per_assembly[assembly]
                if not item.get("Include") or int(assembly) not in table:
                    continue
                lowest, highest = table[int(assembly)]["ReleaseRange"]
                releases = sorted(
                    r
                    for r in {int(x) for x in str(item["Ensembl release"]).split(",")}
                    if first <= r <= last and lowest <= r and (highest is None or r <= highest)
                )
                if releases:
                    selection.append([form, str(database), int(assembly), releases])
    return hashlib.sha256(json.dumps(selection, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def table_fingerprint(organism: str) -> str:
    """Return a digest of an organism's entry in the organism table (what shapes a graph).

    Args:
        organism: The organism.

    Returns:
        str: The digest of the assemblies' keys, release ranges, order and database suffixes.
    """
    table = organisms.ORGANISM_TABLE[organism]
    content = [
        [int(a), list(table[a]["ReleaseRange"]), int(table[a]["Priority"]), str(table[a].get("DbSuffix", a))]
        for a in sorted(table)
    ]
    return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()[:20]


__all__ = ["GRAPH_FILE_NAME", "graph_file_path", "list_graph_files", "selection_fingerprint", "table_fingerprint"]
