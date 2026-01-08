from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymysql

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB
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

    # Deduplicate while preserving order
    seen: set[tuple[str, int, str, str]] = set()
    unique: list[tuple[str, int, str, str]] = []
    for row in rows_out:
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


def build_real_gene_graph(spec: RealGraphSpec) -> tuple[TheGraph, dict[str, str]]:
    """Build a small, real homo_sapiens gene-only graph for the selected Ensembl releases.

    Args:
        spec: Real-graph build specification (organism, assembly, releases, and seed identifiers).

    Returns:
        tuple[TheGraph, dict[str, str]]: ``(graph, fixtures)`` where fixtures contains stable example identifiers
        discovered during build.

    Raises:
        ValueError: If ``spec.releases`` is empty.
    """
    releases = list(spec.releases)
    if not releases:
        raise ValueError("spec.releases cannot be empty")

    db_for_max = _resolve_core_db_name(
        organism=spec.organism, release=spec.max_release, genome_assembly=spec.genome_assembly
    )
    change_event: tuple[str, int, str, int, int, int, float] | None
    with _mysql_connect(
        organism=spec.organism,
        genome_assembly=spec.genome_assembly,
        ensembl_release=spec.max_release,
        database=db_for_max,
    ) as conn_max:
        change_event = _discover_gene_id_change(
            conn=conn_max, min_release=spec.min_release, max_release=spec.max_release
        )

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
        with _mysql_connect(
            organism=spec.organism,
            genome_assembly=spec.genome_assembly,
            ensembl_release=spec.max_release,
            database=db_for_max,
        ) as conn_max:
            seed_ids.extend(_fetch_seed_gene_ids(conn=conn_max, limit=5))

    stable_ids = sorted(set(seed_ids))
    fixtures["seed_gene_ids"] = ",".join(stable_ids)
    db_by_release = {
        rel: _resolve_core_db_name(organism=spec.organism, release=rel, genome_assembly=spec.genome_assembly)
        for rel in releases
    }

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
        db_name = db_by_release[rel]
        with _mysql_connect(
            organism=spec.organism,
            genome_assembly=spec.genome_assembly,
            ensembl_release=rel,
            database=db_name,
        ) as conn:
            versions = _fetch_gene_versions(conn=conn, stable_ids=stable_ids)
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
            ext_rows = _fetch_external_mappings(
                conn=conn,
                stable_ids=sorted(versions.keys()),
                external_databases=spec.external_databases,
            )
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
