#!/usr/bin/env python3
"""The cut tables of the tests are what Ensembl serves: one release fetched again over HTTPS and cut again.

The offline tests of ``tests/unit/test_real_*.py`` build graphs from ``tests/data/real/<organism>.tables.json.xz``.
This test downloads the raw tables of one release of each organism from Ensembl's HTTPS dumps through
``DatabaseManager`` (the production download, parser and storage), cuts them with the build script's own rules and
compares the result, table by table and row by row, with what the file holds. It is the end of the chain "real
Ensembl data in, tested answers out" that needs the network, and it runs only when asked (``--runslow``): the
tables of a release are tens to hundreds of megabytes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import requests

from idtrack._database_manager import DatabaseManager
from idtrack._db import DB, EnsemblSchemaError
from tests._real_tables import ORGANISMS, RealTables, tables_path

BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "data" / "real" / "build_real_tables.py"


def build_script():
    """Import ``tests/data/real/build_real_tables.py`` (a script, not a module of a package)."""
    spec = importlib.util.spec_from_file_location("build_real_tables", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensembl_dumps_reachable() -> bool:
    """Tell whether Ensembl's HTTPS dumps answer (the one accepted reason to skip)."""
    try:
        return bool(requests.head(f"https://{DB.ensembl_ftp_base}/pub/", timeout=(10, 30), allow_redirects=True).ok)
    except requests.RequestException:
        return False


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.integration
@pytest.mark.parametrize("organism", [o for o in ORGANISMS if tables_path(o).is_file()])
def test_cut_tables_of_the_first_release_equal_ensembls_dumps_cut_again(organism, tmp_path, monkeypatch):
    """Every table of the organism's first release in the file equals the fresh download, cut to the same genes."""
    if not ensembl_dumps_reachable():
        pytest.skip("Ensembl's HTTPS dumps are not reachable")
    script = build_script()
    stored = RealTables.load(organism)
    assembly, release = min((r, a) for (a, r, name) in stored.tables if name == "gene")[::-1]
    source, scratch = tmp_path / "source", tmp_path / "scratch"
    for folder in (source, scratch):
        folder.mkdir()
        stored.write_indices(folder)  # the releases of every assembly, as a build leaves them: no scan needed

    cutter = script.Cutter(organism, stored.genes, str(source), stored.snapshot)
    cutter.install(monkeypatch)
    manager = DatabaseManager(
        organism=organism,
        form=DB.backbone_form,
        local_repository=str(scratch),
        ensembl_release=release,
        ignore_after=stored.snapshot,
        genome_assembly=assembly,
    )
    compared = 0
    for (a, r, name), entry in sorted(stored.tables.items()):
        if (a, r) != (assembly, release):
            continue
        columns = [column for column, _kind in entry["columns"]]
        if not columns:
            with pytest.raises(EnsemblSchemaError):
                manager.download_table(name)
            continue
        fresh = manager.download_table(name, usecols=columns)
        kept = pd.DataFrame({column: values for column, values in zip(columns, entry["data"])})
        assert list(fresh.columns) == columns
        assert len(fresh) == len(kept), name
        order = [*columns]
        left = fresh.astype(object).where(fresh.notna(), None).sort_values(order, key=lambda c: c.map(repr))
        right = kept.astype(object).where(kept.notna(), None).sort_values(order, key=lambda c: c.map(repr))
        for column in columns:
            assert [repr(_number(v)) for v in left[column]] == [repr(_number(v)) for v in right[column]], (name, column)
        for column in entry["absent"]:
            with pytest.raises(EnsemblSchemaError):
                manager.download_table(name, usecols=[column])
        compared += 1
    assert compared >= 8
    assert not cutter.open_events


def _number(value):
    """Return a cell as the file holds it: an integral float as an ``int``."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
