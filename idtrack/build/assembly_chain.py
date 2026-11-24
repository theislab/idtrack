#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Ensembl's mappings between two assemblies of an organism: its chain files, fetched, kept and read.

Where an organism changed assembly, Ensembl publishes how the bases of one assembly lie on the other in the
``assembly_chain`` folder of a release (``https://ftp.ensembl.org/pub/release-<r>/assembly_chain/<organism>/``), one
file per ordered pair of assemblies, ``<from>_to_<to>.chain.gz``, named by the assemblies' names (the organism table's
``Name``). A file is a set of UCSC chains: aligned blocks of bases of the first assembly (the target of the chain
format) and where they lie on the second (the query), on its forward or reverse strand. Ensembl publishes such files
for some organisms only, and keeps them only in some releases' folders (the organism's newest file of a pair is read:
:py:func:`fetch_chain`).

A gene's place on an earlier assembly is carried onto a later one by its aligned bases (:py:meth:`AssemblyChain.carry`):
on every sequence region of the later assembly that holds some of them, the place spans from the first to the last base
carried there, on the strand the chain gives. The places carried are kept with the graph
(:py:meth:`idtrack.TheGraph.carried_places`) and compared with the places of the later assembly by overlap, as places
on one assembly are (:py:func:`idtrack.graph.genes.relation`).

The downloaded files are kept in the local repository (``<local repository>/assembly_chain/<organism>/``) beside an
index of the releases whose folder was listed and what it held, so that a later build reads them without the network;
nothing here deletes a downloaded file.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from idtrack import settings
from idtrack._atomic import replacing, write_text
from idtrack.build.ensembl_access import https_get

#: The folder of the local repository that holds Ensembl's chain files, per organism.
CHAIN_FOLDER = "assembly_chain"

_log = logging.getLogger(__name__)


def chain_file_name(from_name: str, to_name: str) -> str:
    """Return Ensembl's name of the chain file that maps one assembly onto another (``NCBIM37_to_GRCm38.chain.gz``)."""
    return f"{from_name}_to_{to_name}.chain.gz"


#: The columns of a table of aligned blocks (:py:func:`parse_chain`): the region of the first assembly, the block's
#: start (0-based) and end (exclusive) there, the region of the second assembly, the block's start there on its forward
#: strand (0-based) and the strand (``1`` or ``-1``).
BLOCK_COLUMNS = ["region", "start", "end", "onto", "onto_start", "strand"]


def parse_chain(lines: Iterable[str]) -> pd.DataFrame:
    """Return the aligned blocks of a chain file (the UCSC chain format: a ``chain`` header, then ``size dt dq`` lines).

    Args:
        lines: The lines of the file.

    Returns:
        pandas.DataFrame: One row per aligned block (:py:data:`BLOCK_COLUMNS`), regions named as Ensembl names them.
    """
    rows: list[tuple[str, int, int, str, int, int]] = []
    t_name = q_name = ""
    t_pos = q_pos = q_size = 0
    q_minus = False
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "chain":
            t_name, t_pos = _region(fields[2]), int(fields[5])
            q_name, q_size, q_minus, q_pos = _region(fields[7]), int(fields[8]), fields[9] == "-", int(fields[10])
            continue
        size = int(fields[0])
        start = q_size - (q_pos + size) if q_minus else q_pos
        rows.append((t_name, t_pos, t_pos + size, q_name, start, -1 if q_minus else 1))
        if len(fields) >= 3:
            t_pos += size + int(fields[1])
            q_pos += size + int(fields[2])
    return pd.DataFrame(rows, columns=BLOCK_COLUMNS)


class AssemblyChain:
    """The aligned blocks of one mapping of an assembly onto another, by region of the first, for carrying places."""

    def __init__(self, blocks: pd.DataFrame):
        """Index a table of aligned blocks (:py:data:`BLOCK_COLUMNS`, :py:func:`parse_chain`)."""
        self.regions: list[str] = sorted(set(blocks["onto"].astype(str).tolist()))
        code = {name: i for i, name in enumerate(self.regions)}
        self.blocks: dict[str, tuple[np.ndarray, ...]] = {}
        for region, group in blocks.groupby(blocks["region"].astype(str), sort=True):
            table = np.column_stack(
                [
                    group["start"].to_numpy(dtype=np.int64),
                    group["end"].to_numpy(dtype=np.int64),
                    np.asarray([code[str(name)] for name in group["onto"].tolist()], dtype=np.int64),
                    group["onto_start"].to_numpy(dtype=np.int64),
                    group["strand"].to_numpy(dtype=np.int64),
                ]
            )
            table = table[np.lexsort(table.T[::-1])]
            self.blocks[str(region)] = (
                table[:, 0],
                table[:, 1],
                np.maximum.accumulate(table[:, 1]),
                table[:, 2],
                table[:, 3],
                table[:, 4],
            )

    def carry(self, region: str, start: int, end: int, strand: int) -> list[tuple[str, int, int, int]]:
        """Return where a place of the first assembly lies on the second, by its aligned bases.

        Args:
            region: The sequence region.
            start: The first base (1-based, inclusive, as Ensembl counts).
            end: The last base (inclusive).
            strand: ``1`` or ``-1``.

        Returns:
            list[tuple[str, int, int, int]]: ``(region, first base, last base, strand)`` per region and strand of the
            second assembly that holds aligned bases of the place, from the first to the last of them; empty when no
            base of the place is aligned.
        """
        table = self.blocks.get(_region(region))
        if table is None:
            return []
        starts, ends, reach, target, there, sign = table
        low, high = int(start) - 1, int(end)
        first = int(np.searchsorted(reach, low, side="right"))
        last = int(np.searchsorted(starts, high, side="left"))
        spans: dict[tuple[int, int], list[int]] = {}
        for j in range(first, last):
            a, b = max(int(starts[j]), low), min(int(ends[j]), high)
            if a >= b:
                continue
            offset = int(there[j])
            if int(sign[j]) == 1:
                lo, hi = offset + (a - int(starts[j])), offset + (b - int(starts[j]))
            else:  # the block runs backwards on the forward strand of the second assembly
                lo, hi = offset + (int(ends[j]) - b), offset + (int(ends[j]) - a)
            key = (int(target[j]), int(sign[j]) * int(strand))
            span = spans.get(key)
            if span is None:
                spans[key] = [lo, hi]
            else:
                span[0], span[1] = min(span[0], lo), max(span[1], hi)
        return sorted((self.regions[t], lo + 1, hi, s) for (t, s), (lo, hi) in spans.items())


def _region(name: str) -> str:
    """Return Ensembl's name of a sequence region as a chain file writes it (``chr1`` is ``1``, ``chrM`` is ``MT``)."""
    name = str(name)
    if name.startswith("chr"):
        name = name[3:]
    return "MT" if name == "M" else name


def _index_path(local_repository: str, organism: str) -> str:
    return os.path.join(local_repository, CHAIN_FOLDER, organism, "index.json")


def _read_index(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            found = json.load(handle)
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}


def _listing(organism: str, release: int) -> list[str] | None:
    """Return the chain files of an organism in a release's folder (``None``: the release has no such folder)."""
    from idtrack.build.database_manager import DatabaseManager

    url = f"https://{settings.ENSEMBL.https_host}/pub/release-{int(release)}/{CHAIN_FOLDER}/{organism}/"
    response = https_get(url, what=f"the assembly chain folder of {organism} at release {release}")
    if response is None:
        return None
    return sorted(
        name for name in DatabaseManager._parse_apache_dir_listing_files(response.text) if name.endswith(".chain.gz")
    )


def fetch_chain(
    local_repository: str, organism: str, from_name: str, to_name: str, releases: Sequence[int]
) -> tuple[str | None, int | None]:
    """Return the local copy of Ensembl's chain file that maps one assembly onto another, fetching it where needed.

    The folders of the given releases are listed newest first (each listing is kept in the repository's index, so a
    release is listed once) until one holds the file; that file is downloaded once and kept. Ensembl publishes no file
    for most pairs of assemblies (and none at all for most organisms): then nothing is fetched.

    Args:
        local_repository: The local repository (the file goes into ``assembly_chain/<organism>/``).
        organism: The organism (Ensembl's production name).
        from_name: The name of the assembly mapped (the organism table's ``Name``).
        to_name: The name of the assembly mapped onto.
        releases: The releases whose folders may hold the file (those of the graph's window).

    Returns:
        tuple[str | None, int | None]: The file and the release whose folder it was read from; ``(None, None)`` when
        none of the releases' folders holds it.
    """
    folder = os.path.join(local_repository, CHAIN_FOLDER, organism)
    name = chain_file_name(from_name, to_name)
    path = os.path.join(folder, name)
    index_path = _index_path(local_repository, organism)
    index = _read_index(index_path)
    listed: dict[str, list[str] | None] = index.setdefault("listings", {})
    found: dict[str, int] = index.setdefault("files", {})
    if name in found and os.path.isfile(path):
        return path, int(found[name])
    changed = False
    try:
        for release in sorted({int(r) for r in releases}, reverse=True):
            key = str(release)
            if key not in listed:
                listed[key] = _listing(organism, release)
                changed = True
            if name in (listed[key] or ()):
                url = f"https://{settings.ENSEMBL.https_host}/pub/release-{release}/{CHAIN_FOLDER}/{organism}/{name}"
                response = https_get(url, what=f"the assembly chain {name} of {organism}")
                if response is None:  # listed but not served: read as absent from this folder
                    continue
                os.makedirs(folder, exist_ok=True)
                with replacing(path) as temporary, open(temporary, "wb") as handle:
                    handle.write(response.content)
                found[name] = release
                changed = True
                _log.info("Ensembl's assembly chain %s of %s read from release %d.", name, organism, release)
                return path, release
        return None, None
    finally:
        if changed:
            os.makedirs(folder, exist_ok=True)
            write_text(index_path, json.dumps(index, indent=1, sort_keys=True))


def read_chain(path: str) -> pd.DataFrame:
    """Return the aligned blocks of a kept chain file (gzip-compressed, as Ensembl publishes it: :py:func:`parse_chain`)."""
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return parse_chain(handle)


__all__ = [
    "BLOCK_COLUMNS",
    "CHAIN_FOLDER",
    "AssemblyChain",
    "chain_file_name",
    "fetch_chain",
    "parse_chain",
    "read_chain",
]
