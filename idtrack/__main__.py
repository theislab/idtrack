#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""The command line of idtrack: ``idtrack <command>`` (or ``python -m idtrack <command>``).

Four commands, each a thin layer over the API:

* ``idtrack build``: build the graph of an organism and a snapshot release (:py:meth:`idtrack.API.build`);
* ``idtrack verify``: check a graph file: its format and record, its content against the digest it records and,
  with the tables it was built from, the self-tests of :py:class:`idtrack.checks.GraphChecks`;
* ``idtrack info``: what a graph file holds (:py:meth:`idtrack.API.describe`);
* ``idtrack convert``: convert identifiers with a graph file (:py:meth:`idtrack.API.convert_many`).

Without a command it prints its help. An error of idtrack (:py:class:`idtrack.errors.IdtrackError`) is printed as
``idtrack: error: <message>`` and the exit status is 1; a command line that cannot be parsed exits with status 2.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from typing import Any

from idtrack import __version__

#: The self-tests that ``idtrack verify --repository`` runs: the checks of :py:class:`idtrack.checks.GraphChecks` that
#: compare the whole graph with its tables or with itself. ``is_final_external_conversion_robust`` (it converts the
#: genes of a release to a database) and the conversion runs (``history_...``) are left to Python.
SELF_CHECKS: tuple[str, ...] = (
    "is_id_functions_consistent_ensembl",
    "is_id_functions_consistent_ensembl_2",
    "is_id_functions_consistent_external",
    "is_range_functions_robust",
    "is_base_is_range_correct",
    "is_combined_summary_complete",
    "is_edge_with_same_nts_only_at_backbone_nodes",
    "is_node_consistency_robust",
)

_DESCRIPTION = (
    "idtrack converts gene identifiers (Ensembl ids, symbols and the ids of external databases) across Ensembl "
    "releases, genome assemblies and databases, with a graph of an organism's identifier history built from "
    "Ensembl's tables. Build a graph once (idtrack build; it downloads Ensembl's tables), then convert with its file "
    "(idtrack convert), which needs no network."
)

_EPILOG = f"""\
how to start:
  idtrack build human 115 --repository idtrack_cache
  idtrack info idtrack_cache/graph_homo_sapiens_assembly38_min48_max115.npz
  idtrack convert idtrack_cache/graph_homo_sapiens_assembly38_min48_max115.npz TP53 BRCA1

Each command has its own help (idtrack <command> --help).

idtrack {__version__}
documentation: https://idtrack.readthedocs.io/"""


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command line."""
    parser = argparse.ArgumentParser(
        prog="idtrack",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"idtrack {__version__}",
        help="Show the package version and exit.",
    )
    commands = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    build = commands.add_parser(
        "build",
        help="Build the graph of an organism and a snapshot release from Ensembl's tables.",
        description=(
            "Build the graph of an organism and a snapshot release from Ensembl's tables (downloaded into the "
            "repository; tables already there are not downloaded again), write its file into the repository and "
            "print the file's path. If the repository holds the file of this graph, it is loaded and checked instead "
            "(unless --rebuild). Progress is logged to standard error."
        ),
    )
    build.add_argument(
        "organism", help="The organism: homo_sapiens, mus_musculus, sus_scrofa, ... or human, mouse, pig."
    )
    build.add_argument("snapshot_release", type=int, help="The last Ensembl release of the graph (e.g. 115).")
    build.add_argument(
        "-r",
        "--repository",
        default="idtrack_cache",
        help="The folder of the downloaded tables and the graph files; made if missing (default: idtrack_cache).",
    )
    build.add_argument("--first-release", type=int, help="The first release of the graph (default: the organism's).")
    build.add_argument("--rebuild", action="store_true", help="Build even if the graph file exists, and replace it.")
    build.add_argument("-q", "--quiet", action="store_true", help="Log warnings and errors only.")

    verify = commands.add_parser(
        "verify",
        help="Check a graph file: format, record, content digest and, with its tables, the self-tests.",
        description=(
            "Check a graph file: that this version of idtrack reads its format and that its record is valid for the "
            "organism table of this version, and that its content gives the digest the file records. With "
            "--repository, the self-tests also compare the graph with the tables of that repository (the folder the "
            "graph was built in; a table it lacks is downloaded). The exit status is 0 when everything holds."
        ),
    )
    verify.add_argument("graph", help="The graph file (graph_<organism>_assembly<key>_min<first>_max<last>.npz).")
    verify.add_argument("-r", "--repository", help="The folder of the tables the graph was built from.")

    info = commands.add_parser(
        "info",
        help="Show what a graph file holds.",
        description="Show what a graph file holds: organism, releases, assemblies, databases and what built it.",
    )
    info.add_argument("graph", help="The graph file.")
    info.add_argument("--json", action="store_true", help="Print the description as JSON.")

    convert = commands.add_parser(
        "convert",
        help="Convert identifiers with a graph file.",
        description=(
            "Convert identifiers with a graph file, to the graph's snapshot release unless --to-release says "
            "otherwise, and print one tab-separated line per identifier: the identifier, the status, the targets "
            "(separated by commas; several targets of the best answer are a tie or a split) and the reason. An "
            "identifier that cannot be converted is a line with its status, never an error of the command."
        ),
    )
    convert.add_argument("graph", help="The graph file.")
    convert.add_argument("identifiers", nargs="*", help="The identifiers.")
    convert.add_argument(
        "-f", "--file", help="A file of identifiers, one per line (- for standard input); blank lines are skipped."
    )
    convert.add_argument("--from-release", type=int, help="The release the identifiers are read at.")
    convert.add_argument("--to-release", type=int, help="The target release (default: the graph's snapshot).")
    convert.add_argument(
        "--final-database", help="The database of the targets, e.g. 'HGNC Symbol' (default: Ensembl gene ids)."
    )
    convert.add_argument("--from-database", help="The external database the identifiers are ids of.")
    convert.add_argument("--all", action="store_true", help="Print every candidate's target, not only the best.")
    convert.add_argument("--no-bridges", action="store_true", help="Follow Ensembl's history only, without bridges.")
    convert.add_argument("--json", action="store_true", help="Print the full results as a JSON list.")
    return parser


def _build(args: argparse.Namespace) -> int:
    """Run ``idtrack build``."""
    from idtrack import API, organisms

    organism = organisms.resolve_organism(args.organism)
    API.configure_logger("WARNING" if args.quiet else "INFO")
    os.makedirs(args.repository, exist_ok=True)
    started = time.monotonic()
    api = API.build(
        organism, args.snapshot_release, args.repository, first_release=args.first_release, rebuild=args.rebuild
    )
    path = API.graph_path(organism, args.snapshot_release, args.repository, first_release=args.first_release)
    described = api.describe()
    print(
        f"{path}\n{described.organism}, releases {described.releases[0]} to {described.releases[-1]}, "
        f"{os.path.getsize(path) / 1e6:.1f} MB, {time.monotonic() - started:.0f} s"
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    """Run ``idtrack verify``."""
    from idtrack import API
    from idtrack.build.builder import _content_digest
    from idtrack.checks import GraphChecks

    api = API.load(args.graph)
    record = api.graph.record
    print(f"format and record: ok ({record.organism}, releases {record.first_release} to {record.snapshot_release})")
    computed = _content_digest(api.graph._store)
    holds = computed == record.content_digest
    if holds:
        print(f"content digest: ok ({computed})")
    else:
        print(f"content digest: DIFFERS (recorded {record.content_digest}, computed {computed})")
    if args.repository is not None:
        if not os.path.isdir(args.repository):
            print(f"idtrack: error: the repository {args.repository!r} is not a folder", file=sys.stderr)
            return 1
        manager = API.database_manager(
            record.organism, record.snapshot_release, args.repository, first_release=record.first_release
        )
        checks = GraphChecks(api.graph, manager)
        for name in SELF_CHECKS:
            method = getattr(checks, name)
            started = time.monotonic()
            options = {"progress": False} if "progress" in inspect.signature(method).parameters else {}
            result = method(**options)
            holds = holds and result.holds
            print(f"{name}: {'holds' if result.holds else 'DOES NOT HOLD'} ({time.monotonic() - started:.1f} s)")
            if not result.holds:
                print(json.dumps({k: v for k, v in result.details.items() if k != "traceback"}, default=str))
    print("verified" if holds else "NOT verified")
    return 0 if holds else 1


def _info(args: argparse.Namespace) -> int:
    """Run ``idtrack info``."""
    from idtrack import API

    described = API.load(args.graph).describe()
    if args.json:
        print(json.dumps(described.to_dict(), indent=1, allow_nan=False))
        return 0
    releases = described.releases
    print(f"organism: {described.organism}")
    print(f"releases: {releases[0]} to {releases[-1]} (snapshot {described.snapshot_release})")
    print(f"primary assembly: {described.primary_assembly}")
    for assembly in described.assemblies:
        served = f"serves {assembly.serves[0]} to {assembly.serves[-1]}" if assembly.serves else "serves no release"
        print(
            f"assembly {assembly.assembly} ({assembly.name}): {served}, "
            f"{len(assembly.external_databases)} external databases"
        )
    print(f"external databases: {', '.join(described.external_databases)}")
    print(f"final databases: {', '.join(described.final_databases)}")
    print(f"built by: idtrack {described.package_version} (commit {described.package_commit}), {described.build_date}")
    print(f"content digest: {described.content_digest}")
    return 0


def _identifiers(args: argparse.Namespace) -> list[str]:
    """Return the identifiers of ``idtrack convert``: those of the command line, then those of the file."""
    identifiers = list(args.identifiers)
    if args.file is not None:
        if args.file == "-":
            lines = sys.stdin.read().splitlines()
        else:
            with open(args.file, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        identifiers.extend(line.strip() for line in lines if line.strip())
    return identifiers


def _convert(args: argparse.Namespace) -> int:
    """Run ``idtrack convert``."""
    from idtrack import API

    identifiers = _identifiers(args)
    if not identifiers:
        print("idtrack: error: no identifiers (give them as arguments or with --file)", file=sys.stderr)
        return 1
    api = API.load(args.graph)
    results = api.convert_many(
        identifiers,
        args.from_release,
        args.to_release,
        args.final_database,
        from_database=args.from_database,
        allow_bridges=not args.no_bridges,
    )
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=1, allow_nan=False))
        return 0
    strategy = "all" if args.all else "best"
    print("query\tstatus\ttargets\treason")
    for result in results:
        print(f"{result.query_id}\t{result.status}\t{','.join(result.targets(strategy))}\t{result.reason or ''}")
    return 0


_COMMANDS: dict[str, Any] = {"build": _build, "verify": _verify, "info": _info, "convert": _convert}


def main(argv: list[str] | None = None) -> int:
    """Run the command line (``python -m idtrack``, or the ``idtrack`` console script).

    Args:
        argv: Arguments without the program name; ``None`` means ``sys.argv[1:]``.

    Returns:
        int: The exit status: 0 when the command did its work, 1 when idtrack raised an error or a check does not hold.
        A command line that cannot be parsed makes :py:mod:`argparse` exit with status 2.
    """
    from idtrack.errors import IdtrackError

    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return int(_COMMANDS[args.command](args))
    except (IdtrackError, OSError) as error:
        print(f"idtrack: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
