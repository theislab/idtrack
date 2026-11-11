"""Build tiny, real gene graphs for the end-to-end tests.

The graph holds a handful of genes over a few releases of one assembly. Its tables come from the public Ensembl MySQL
service through targeted queries when that service is usable (see :py:mod:`idtrack._ensembl_access`), and from
Ensembl's HTTPS dumps otherwise; both sources yield the same graph.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pymysql

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
from idtrack._ensembl_access import reachability
from idtrack._the_graph import TheGraph


@dataclass(frozen=True)
class RealGraphSpec:
    """Parameters for building a tiny, real gene graph snapshot.

    Attributes:
        organism: Ensembl species name (e.g. ``"homo_sapiens"``).
        genome_assembly: Genome assembly integer used by idtrack (e.g. 38).
        releases: Ensembl releases to include in the snapshot.
        seed_gene_ids: Optional seed gene stable IDs to include.
        external_databases: Optional external databases to attach to gene nodes.
    """

    organism: str
    genome_assembly: int
    releases: tuple[int, ...]
    seed_gene_ids: tuple[str, ...] = ()
    external_databases: tuple[str, ...] = ()

    @property
    def min_release(self) -> int:
        """Minimum Ensembl release included in the snapshot."""
        return min(self.releases)

    @property
    def max_release(self) -> int:
        """Maximum Ensembl release included in the snapshot."""
        return max(self.releases)


def _mysql_connect(
    *,
    organism: str,
    genome_assembly: int,
    ensembl_release: int,
    database: str | None = None,
) -> pymysql.connections.Connection:
    core_index = DatabaseManager._get_core_db_index(organism=organism, genome_assembly=genome_assembly)
    port = core_index["port_for_release"][int(ensembl_release)]
    which_mysql_server: dict[str, Any] = {
        "host": DB.mysql_host,
        "user": DB.myqsl_user,
        "password": DB.mysql_togo,
        "port": int(port),
    }
    if database is not None:
        which_mysql_server["database"] = database
    return pymysql.connect(**which_mysql_server)


def _fetch_available_databases(conn: pymysql.connections.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SHOW DATABASES")
        rows = cur.fetchall()
    dbs: list[str] = []
    for (name,) in rows:
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        dbs.append(str(name))
    return dbs


def _resolve_core_db_name(*, organism: str, release: int, genome_assembly: int) -> str:
    core_index = DatabaseManager._get_core_db_index(organism=organism, genome_assembly=genome_assembly)
    return str(core_index["db_for_release"][int(release)])


def _discover_gene_id_change(
    *, conn: pymysql.connections.Connection, min_release: int, max_release: int
) -> tuple[str, int, str, int, int, int, float] | None:
    """Return one stable-id change event within [min_release, max_release] (old_id != new_id)."""
    sql = """
        SELECT
            se.old_stable_id,
            se.old_version,
            se.new_stable_id,
            se.new_version,
            se.score,
            ms.old_release,
            ms.new_release
        FROM stable_id_event se
        JOIN mapping_session ms USING (mapping_session_id)
        WHERE se.type = 'gene'
          AND ms.old_release >= %s AND ms.old_release <= %s
          AND ms.new_release >= %s AND ms.new_release <= %s
          AND se.old_stable_id <> se.new_stable_id
        ORDER BY ms.old_release, ms.new_release, se.old_stable_id, se.new_stable_id
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (min_release, max_release, min_release, max_release))
        row = cur.fetchone()
    if not row:
        return None
    old_id, old_ver, new_id, new_ver, score, old_rel, new_rel = row
    if isinstance(old_id, bytes):
        old_id = old_id.decode("utf-8")
    if isinstance(new_id, bytes):
        new_id = new_id.decode("utf-8")
    return (
        str(old_id),
        int(old_ver),
        str(new_id),
        int(new_ver),
        int(old_rel),
        int(new_rel),
        float(score) if score is not None else float("nan"),
    )


def _iter_chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_gene_versions(*, conn: pymysql.connections.Connection, stable_ids: list[str]) -> dict[str, int]:
    """Return `{stable_id: version}` for genes present in the current DB."""
    if not stable_ids:
        return {}
    result: dict[str, int] = {}
    for chunk in _iter_chunks(stable_ids, 200):
        placeholders = ",".join(["%s"] * len(chunk))
        sql = f"SELECT stable_id, version FROM gene WHERE stable_id IN ({placeholders})"  # noqa: S608
        with conn.cursor() as cur:
            cur.execute(sql, chunk)
            rows = cur.fetchall()
        for stable_id, version in rows:
            if isinstance(stable_id, bytes):
                stable_id = stable_id.decode("utf-8")
            result[str(stable_id)] = int(version)
    return result


def _fetch_seed_gene_ids(*, conn: pymysql.connections.Connection, limit: int = 5) -> list[str]:
    """Pick a small, deterministic set of gene stable IDs from the current core DB."""
    if limit <= 0:
        return []
    sql = "SELECT stable_id FROM gene ORDER BY stable_id LIMIT %s"
    with conn.cursor() as cur:
        cur.execute(sql, (int(limit),))
        rows = cur.fetchall()
    seed: list[str] = []
    for (stable_id,) in rows:
        if isinstance(stable_id, bytes):
            stable_id = stable_id.decode("utf-8")
        seed.append(str(stable_id))
    return seed


def _fetch_external_mappings(
    *,
    conn: pymysql.connections.Connection,
    stable_ids: list[str],
    external_databases: tuple[str, ...],
) -> list[tuple[str, int, str, str]]:
    """Return rows as `(stable_id, version, external_db_name, external_id)`."""
    if not stable_ids or not external_databases:
        return []

    wanted_display = tuple(db for db in external_databases if " " in db or db.lower() != db)
    wanted_name = tuple(db for db in external_databases)
    if not wanted_display and not wanted_name:
        return []

    rows_out: list[tuple[str, int, str, str]] = []
    for chunk in _iter_chunks(stable_ids, 100):
        placeholders = ",".join(["%s"] * len(chunk))
        # Pull both db_display_name (display_label) and db_name (dbprimary_acc), mirroring DatabaseManager.create_external_db.
        sql = f"""  # noqa: S608
            SELECT
                g.stable_id,
                g.version,
                x.dbprimary_acc,
                x.display_label,
                ed.db_name,
                ed.db_display_name
            FROM gene g
            JOIN object_xref ox
              ON ox.ensembl_id = g.gene_id AND ox.ensembl_object_type = 'Gene'
            JOIN xref x
              ON x.xref_id = ox.xref_id
            JOIN external_db ed
              ON ed.external_db_id = x.external_db_id
            WHERE g.stable_id IN ({placeholders})
              AND (
                ed.db_display_name IN ({",".join(["%s"] * len(wanted_display))})
                OR ed.db_name IN ({",".join(["%s"] * len(wanted_name))})
              )
        """
        params: list[str] = list(chunk) + list(wanted_display) + list(wanted_name)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        for stable_id, version, dbprimary_acc, display_label, db_name, db_display_name in rows:
            if isinstance(stable_id, bytes):
                stable_id = stable_id.decode("utf-8")
            if isinstance(db_name, bytes):
                db_name = db_name.decode("utf-8")
            if isinstance(db_display_name, bytes):
                db_display_name = db_display_name.decode("utf-8")
            if isinstance(dbprimary_acc, bytes):
                dbprimary_acc = dbprimary_acc.decode("utf-8")
            if isinstance(display_label, bytes):
                display_label = display_label.decode("utf-8")

            stable_id_s = str(stable_id)
            version_i = int(version)

            # db_display_name → display_label
            if db_display_name in external_databases and display_label is not None:
                rows_out.append((stable_id_s, version_i, str(db_display_name), str(display_label)))
            # db_name → dbprimary_acc
            if db_name in external_databases and dbprimary_acc is not None:
                rows_out.append((stable_id_s, version_i, str(db_name), str(dbprimary_acc)))

    return _unique(rows_out)


def uses_https_dumps(spec: RealGraphSpec) -> bool:
    """Return whether :py:func:`build_real_gene_graph` would read *spec* from the HTTPS dumps (MySQL not usable)."""
    core_index = DatabaseManager._get_core_db_index(organism=spec.organism, genome_assembly=spec.genome_assembly)
    port = core_index["port_for_release"][int(spec.max_release)]
    return not reachability.mysql_port_usable(int(port))


@lru_cache(maxsize=64)
def _https_table(
    organism: str, genome_assembly: int, release: int, table: str, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Download one table of one release from the HTTPS dumps (decoded as MySQL would return it)."""
    with tempfile.TemporaryDirectory(prefix="idtrack_real_graph_") as tmpdir:
        dm = DatabaseManager(
            organism=organism,
            form="gene",
            local_repository=tmpdir,
            ensembl_release=int(release),
            ignore_before=int(release),
            ignore_after=int(release),
            genome_assembly=int(genome_assembly),
            store_raw_always=False,
        )
        return dm._download_table_from_ftp(table, usecols=list(columns))


class _HTTPSSource:
    """The queries of the MySQL source, answered from whole tables of the HTTPS dumps."""

    def __init__(self, organism: str, genome_assembly: int) -> None:
        self.organism = organism
        self.genome_assembly = int(genome_assembly)

    def _table(self, release: int, table: str, columns: tuple[str, ...]) -> pd.DataFrame:
        return _https_table(self.organism, self.genome_assembly, int(release), table, columns)

    def gene_id_change(self, release: int, min_release: int, max_release: int):
        events = self._table(
            release,
            "stable_id_event",
            ("old_stable_id", "old_version", "new_stable_id", "new_version", "mapping_session_id", "type", "score"),
        )
        sessions = self._table(release, "mapping_session", ("mapping_session_id", "old_release", "new_release"))
        sessions = sessions.assign(
            old_release=pd.to_numeric(sessions["old_release"], errors="coerce"),
            new_release=pd.to_numeric(sessions["new_release"], errors="coerce"),
        )
        df = events.merge(sessions, on="mapping_session_id", how="inner")
        df = df[
            (df["type"] == "gene")
            & df["old_release"].between(min_release, max_release)
            & df["new_release"].between(min_release, max_release)
            & df["old_stable_id"].notna()
            & df["new_stable_id"].notna()
            & (df["old_stable_id"] != df["new_stable_id"])
        ]
        if df.empty:
            return None
        row = df.sort_values(["old_release", "new_release", "old_stable_id", "new_stable_id"]).iloc[0]
        return (
            str(row["old_stable_id"]),
            int(row["old_version"]),
            str(row["new_stable_id"]),
            int(row["new_version"]),
            int(row["old_release"]),
            int(row["new_release"]),
            float(row["score"]) if pd.notna(row["score"]) else float("nan"),
        )

    def seed_gene_ids(self, release: int, limit: int = 5) -> list[str]:
        genes = self._table(release, "gene", ("gene_id", "stable_id", "version"))
        return sorted(str(s) for s in genes["stable_id"].dropna())[: max(0, int(limit))]

    def gene_versions(self, release: int, stable_ids: list[str]) -> dict[str, int]:
        genes = self._table(release, "gene", ("gene_id", "stable_id", "version"))
        genes = genes[genes["stable_id"].isin(set(stable_ids))]
        return {str(s): int(v) for s, v in zip(genes["stable_id"], genes["version"])}

    def external_mappings(
        self, release: int, stable_ids: list[str], external_databases: tuple[str, ...]
    ) -> list[tuple[str, int, str, str]]:
        if not stable_ids or not external_databases:
            return []
        genes = self._table(release, "gene", ("gene_id", "stable_id", "version"))
        genes = genes[genes["stable_id"].isin(set(stable_ids))]
        links = self._table(release, "object_xref", ("ensembl_id", "ensembl_object_type", "xref_id"))
        links = links[(links["ensembl_object_type"] == "Gene") & links["ensembl_id"].isin(set(genes["gene_id"]))]
        xrefs = self._table(release, "xref", ("xref_id", "external_db_id", "dbprimary_acc", "display_label"))
        dbs = self._table(release, "external_db", ("external_db_id", "db_name", "db_display_name"))
        joined = (
            genes.merge(links, left_on="gene_id", right_on="ensembl_id")
            .merge(xrefs, on="xref_id")
            .merge(dbs, on="external_db_id")
        )
        rows_out: list[tuple[str, int, str, str]] = []
        for row in joined.itertuples(index=False):
            if row.db_display_name in external_databases and pd.notna(row.display_label):
                rows_out.append(
                    (str(row.stable_id), int(row.version), str(row.db_display_name), str(row.display_label))
                )
            if row.db_name in external_databases and pd.notna(row.dbprimary_acc):
                rows_out.append((str(row.stable_id), int(row.version), str(row.db_name), str(row.dbprimary_acc)))
        return _unique(rows_out)


class _MySQLSource:
    """Targeted queries against the public Ensembl MySQL service (one connection per release)."""

    def __init__(self, organism: str, genome_assembly: int) -> None:
        self.organism = organism
        self.genome_assembly = int(genome_assembly)

    def _connect(self, release: int) -> pymysql.connections.Connection:
        db_name = _resolve_core_db_name(organism=self.organism, release=release, genome_assembly=self.genome_assembly)
        return _mysql_connect(
            organism=self.organism, genome_assembly=self.genome_assembly, ensembl_release=release, database=db_name
        )

    def gene_id_change(self, release: int, min_release: int, max_release: int):
        with self._connect(release) as conn:
            return _discover_gene_id_change(conn=conn, min_release=min_release, max_release=max_release)

    def seed_gene_ids(self, release: int, limit: int = 5) -> list[str]:
        with self._connect(release) as conn:
            return _fetch_seed_gene_ids(conn=conn, limit=limit)

    def gene_versions(self, release: int, stable_ids: list[str]) -> dict[str, int]:
        with self._connect(release) as conn:
            return _fetch_gene_versions(conn=conn, stable_ids=stable_ids)

    def external_mappings(
        self, release: int, stable_ids: list[str], external_databases: tuple[str, ...]
    ) -> list[tuple[str, int, str, str]]:
        with self._connect(release) as conn:
            return _fetch_external_mappings(conn=conn, stable_ids=stable_ids, external_databases=external_databases)


def _unique(rows: list[tuple[str, int, str, str]]) -> list[tuple[str, int, str, str]]:
    """Deduplicate while preserving order."""
    seen: set[tuple[str, int, str, str]] = set()
    unique: list[tuple[str, int, str, str]] = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            unique.append(row)
    return unique


def _add_connection_edge(g: TheGraph, n1: str, n2: str, db_name: str, assembly: int, release: int) -> None:
    """Match GraphMaker.add_edge semantics for `DB.connection_dict` edges."""
    if not g.has_edge(n1, n2):
        g.add_edge(n1, n2, **{DB.connection_dict: {db_name: {assembly: {release}}}})
        return

    edge_data = g.get_edge_data(n1, n2)
    if edge_data is None or len(edge_data) != 1:
        raise ValueError(f"Expected exactly one edge between {n1!r} and {n2!r}, got: {edge_data}")
    if db_name not in edge_data[0][DB.connection_dict]:
        edge_data[0][DB.connection_dict][db_name] = {assembly: {release}}
        return
    if assembly not in edge_data[0][DB.connection_dict][db_name]:
        edge_data[0][DB.connection_dict][db_name][assembly] = {release}
        return
    edge_data[0][DB.connection_dict][db_name][assembly].add(release)


def build_real_gene_graph(spec: RealGraphSpec, source: Optional[str] = None) -> tuple[TheGraph, dict[str, str]]:
    """Build a small, real gene-only graph for the selected Ensembl releases of one organism and assembly.

    Args:
        spec: Real-graph build specification (organism, assembly, releases, and seed identifiers).
        source: ``"mysql"`` or ``"https"``; by default MySQL when its port is usable, the HTTPS dumps otherwise.

    Returns:
        tuple[TheGraph, dict[str, str]]: ``(graph, fixtures)`` where fixtures contains stable example identifiers
        discovered during build.

    Raises:
        ValueError: If ``spec.releases`` is empty or *source* is unknown.
    """
    releases = list(spec.releases)
    if not releases:
        raise ValueError("spec.releases cannot be empty")

    if source is None:
        source = "https" if uses_https_dumps(spec) else "mysql"
    if source == "https":
        tables: Any = _HTTPSSource(spec.organism, spec.genome_assembly)
    elif source == "mysql":
        tables = _MySQLSource(spec.organism, spec.genome_assembly)
    else:
        raise ValueError(f"Unknown source {source!r}; use 'mysql' or 'https'.")

    change_event: tuple[str, int, str, int, int, int, float] | None
    change_event = tables.gene_id_change(spec.max_release, spec.min_release, spec.max_release)

    seed_ids = list(spec.seed_gene_ids)
    fixtures: dict[str, str] = {}
    if change_event is not None:
        old_id, _old_ver, new_id, _new_ver, old_rel, new_rel, _score = change_event
        seed_ids.extend([old_id, new_id])
        fixtures.update(
            {
                "changed_old_stable_id": old_id,
                "changed_new_stable_id": new_id,
                "changed_old_release": str(old_rel),
                "changed_new_release": str(new_rel),
            }
        )

    if not seed_ids:
        seed_ids.extend(tables.seed_gene_ids(spec.max_release, limit=5))

    stable_ids = sorted(set(seed_ids))
    fixtures["seed_gene_ids"] = ",".join(stable_ids)

    g = TheGraph(
        name=f"{spec.organism}_{spec.max_release}_gene_small",
        type="gene",
        organism=spec.organism,
        genome_assembly=spec.genome_assembly,
        ensembl_release=spec.max_release,
        confident_for_release=releases,
        version_info="with_version",
        narrow=True,
        narrow_external=True,
        misplaced_external_entry=set(),
    )
    g._attach_included_forms(["gene"])

    # Collect versions and externals per release.
    versions_by_release: dict[int, dict[str, int]] = {}
    for rel in releases:
        versions = tables.gene_versions(rel, stable_ids)
        versions_by_release[rel] = versions

        for stable_id, version in versions.items():
            node_name = f"{stable_id}{DB.id_ver_delimiter}{version}"
            g.add_node(
                node_name,
                **{
                    DB.node_type_str: DB.nts_ensembl["gene"],
                    "ID": stable_id,
                    "Version": int(version),
                },
            )

        # External edges: only for genes present in this release.
        ext_rows = tables.external_mappings(rel, sorted(versions.keys()), spec.external_databases)
        for stable_id, version, db, ext_id in ext_rows:
            gene_node = f"{stable_id}{DB.id_ver_delimiter}{version}"
            if gene_node not in g.nodes:
                continue
            if ext_id not in g.nodes:
                g.add_node(ext_id, **{DB.node_type_str: DB.nts_external})
            _add_connection_edge(g, ext_id, gene_node, db, spec.genome_assembly, rel)

    # Add backbone birth/version-change/retirement edges per stable_id.
    for stable_id in stable_ids:
        present = [(rel, versions_by_release[rel].get(stable_id)) for rel in releases]
        present = [(rel, ver) for rel, ver in present if ver is not None]
        if not present:
            continue

        # Birth edge from `<stable_id>.Void` to first version node.
        first_rel, first_ver = present[0]
        void_node = f"{stable_id}{DB.id_ver_delimiter}{DB.no_old_node_id}"
        first_node = f"{stable_id}{DB.id_ver_delimiter}{first_ver}"
        if void_node not in g.nodes:
            g.add_node(
                void_node,
                **{
                    DB.node_type_str: DB.nts_ensembl["gene"],
                    "ID": stable_id,
                    "Version": DB.no_old_node_id,
                },
            )
        if first_node in g.nodes:
            g.add_edge(void_node, first_node, weight=float("nan"), old_release=first_rel, new_release=first_rel)

        # Version-change edges between consecutive releases.
        for (rel_a, ver_a), (rel_b, ver_b) in zip(present, present[1:]):
            if ver_a == ver_b:
                continue
            n1 = f"{stable_id}{DB.id_ver_delimiter}{ver_a}"
            n2 = f"{stable_id}{DB.id_ver_delimiter}{ver_b}"
            if n1 in g.nodes and n2 in g.nodes:
                g.add_edge(n1, n2, weight=1.0, old_release=rel_a, new_release=rel_b)

        # Retirement edge if the ID disappears before the last release in window.
        last_rel, last_ver = present[-1]
        if last_rel != spec.max_release:
            retired_node = f"{stable_id}{DB.id_ver_delimiter}{DB.no_new_node_id}"
            if retired_node not in g.nodes:
                g.add_node(
                    retired_node,
                    **{
                        DB.node_type_str: DB.nts_ensembl["gene"],
                        "ID": stable_id,
                        "Version": DB.no_new_node_id,
                    },
                )
            last_node = f"{stable_id}{DB.id_ver_delimiter}{last_ver}"
            next_rel = last_rel + 1
            if last_node in g.nodes:
                g.add_edge(last_node, retired_node, weight=float("nan"), old_release=last_rel, new_release=next_rel)

        # Self-loop for the latest-release version node to mark continued activity.
        if present[-1][0] == spec.max_release:
            last_rel, last_ver = present[-1]
            last_node = f"{stable_id}{DB.id_ver_delimiter}{last_ver}"
            earliest_for_this_version = min(rel for rel, ver in present if ver == last_ver)
            if last_node in g.nodes:
                g.add_edge(
                    last_node,
                    last_node,
                    weight=1.0,
                    old_release=earliest_for_this_version,
                    new_release=np.inf,
                )

    # Add one real stable-id change edge if discovered.
    if change_event is not None:
        old_id, old_ver, new_id, new_ver, old_rel, new_rel, score = change_event
        n1 = f"{old_id}{DB.id_ver_delimiter}{old_ver}"
        n2 = f"{new_id}{DB.id_ver_delimiter}{new_ver}"
        if n1 in g.nodes and n2 in g.nodes:
            g.add_edge(n1, n2, weight=score, old_release=old_rel, new_release=new_rel)

    # Add versionless base nodes + edges (release-scoped), mirroring GraphMaker.
    for rel in releases:
        for stable_id, version in versions_by_release.get(rel, {}).items():
            node_name = f"{stable_id}{DB.id_ver_delimiter}{version}"
            if node_name not in g.nodes:
                continue
            if stable_id not in g.nodes:
                g.add_node(stable_id, **{DB.node_type_str: DB.nts_base_ensembl["gene"]})
            _add_connection_edge(g, stable_id, node_name, DB.nts_base_ensembl["gene"], spec.genome_assembly, rel)

    # Populate `available_releases` on connection edges, matching GraphMaker.construct_graph.
    for u, v, k in g.edges:
        edge_data = g.get_edge_data(u, v, k)
        if edge_data and DB.connection_dict in edge_data:
            thed = edge_data[DB.connection_dict]
            g[u][v][k]["available_releases"] = {r for db in thed for asm in thed[db] for r in thed[db][asm]}

    # Convenience fixtures: resolve versioned nodes for the discovered stable-id change.
    if change_event is not None:
        old_id, _old_ver, new_id, _new_ver, old_rel, new_rel, _score = change_event
        old_ver_actual = versions_by_release.get(old_rel, {}).get(old_id)
        new_ver_actual = versions_by_release.get(new_rel, {}).get(new_id)
        if old_ver_actual is not None:
            fixtures["changed_old_node"] = f"{old_id}{DB.id_ver_delimiter}{old_ver_actual}"
        if new_ver_actual is not None:
            fixtures["changed_new_node"] = f"{new_id}{DB.id_ver_delimiter}{new_ver_actual}"

    return g, fixtures


def load_or_build_real_gene_graph(path: Path, spec: RealGraphSpec) -> tuple[TheGraph, dict[str, str]]:
    """Load a cached snapshot from `path`, or build and cache a new one."""
    import pickle  # noqa: S403

    if path.exists():
        try:
            payload = pickle.loads(path.read_bytes())  # noqa: S301
        except Exception:
            # Treat corrupted / incompatible caches as a cache miss.
            # This keeps integration tests robust across dependency/Python upgrades.
            try:
                path.unlink()
            except OSError:
                pass
        else:
            if isinstance(payload, dict) and "graph" in payload:
                graph = payload["graph"]
                fixtures = payload.get("fixtures", {}) or {}
                # Backfill newer fixture keys for older caches.
                if isinstance(graph, TheGraph) and "seed_gene_ids" not in fixtures:
                    stable_ids = sorted(
                        {
                            graph.nodes[n]["ID"]
                            for n in graph.nodes
                            if graph.nodes[n].get(DB.node_type_str) == DB.nts_ensembl["gene"] and "ID" in graph.nodes[n]
                        }
                    )
                    fixtures = dict(fixtures)
                    fixtures["seed_gene_ids"] = ",".join(stable_ids)
                    path.write_bytes(
                        pickle.dumps({"graph": graph, "fixtures": fixtures}, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                return graph, fixtures
            return payload, {}

    graph, fixtures = build_real_gene_graph(spec)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_bytes(pickle.dumps({"graph": graph, "fixtures": fixtures}, protocol=pickle.HIGHEST_PROTOCOL))
    return graph, fixtures
