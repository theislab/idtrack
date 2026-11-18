#!/usr/bin/env python3
"""Externals YAML files are checked against the organism table, read once, and written atomically."""

import glob
import importlib.resources
import logging
import os

import pandas as pd
import pytest
import yaml

import idtrack.build.externals as ext
from idtrack import organisms
from idtrack.build.externals import ExternalDatabases, load_externals_yaml, write_externals_yaml
from idtrack.errors import ExternalsError

SHIPPED = sorted(
    glob.glob(os.path.join(str(importlib.resources.files("idtrack") / "default_config"), "*_externals_*.yml"))
)


def test_the_shipped_files_are_found():
    """The check below runs over every organism's shipped files (an empty list would skip it silently)."""
    assert {os.path.basename(p).split("_externals_")[0] for p in SHIPPED} == set(organisms.supported_organisms())


def _entry(releases: str, include=True, assembly: str = "111") -> dict:
    return {
        "Assembly": {assembly: {"Ensembl release": releases, "Include": include}},
    }


def _write(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)


@pytest.mark.parametrize("path", SHIPPED, ids=os.path.basename)
def test_every_shipped_file_agrees_with_the_organism_table(path):
    """organisms_externals-2: shipped assembly keys and releases lie inside the organism table's windows."""
    assert load_externals_yaml(path, shipped=True)


def test_pig_releases_90_to_99_have_only_sscrofa11(tmp_path):
    """organisms_externals-2: at pig release 95 the YAML names assembly 111 only (102 ends at 89)."""
    e = ExternalDatabases("sus_scrofa", 95, "gene", str(tmp_path), 111)
    assert e.give_list_for_case("assembly") == [111]


def test_a_shipped_file_outside_the_table_is_an_error(tmp_path):
    """organisms_externals-2: a shipped file with a release outside its assembly's window does not load."""
    path = tmp_path / "f.yml"
    _write(path, {"sus_scrofa": {"gene": {"EntrezGene": _entry("88,89,90", assembly="102")}}})
    with pytest.raises(ExternalsError, match="outside its window"):
        load_externals_yaml(str(path), shipped=True)


def test_a_user_file_outside_the_table_is_warned_and_trimmed(tmp_path, caplog):
    """organisms_externals-2: a user file's unknown assembly and out-of-window releases are named and left out."""
    path = tmp_path / "sus_scrofa_externals_modified.yml"
    _write(
        path,
        {
            "sus_scrofa": {
                "gene": {
                    "EntrezGene": _entry("88,89,90", assembly="102"),
                    "HGNC Symbol": _entry("95", assembly="38"),
                }
            }
        },
    )
    with caplog.at_level(logging.WARNING, logger="idtrack.build.externals"):
        data = load_externals_yaml(str(path), shipped=False)
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "[90]" in message and "'38'" in message
    assert data["sus_scrofa"]["gene"]["EntrezGene"]["Assembly"]["102"]["Ensembl release"] == "88,89"
    assert "HGNC Symbol" not in data["sus_scrofa"]["gene"]
    assert ExternalDatabases("sus_scrofa", 89, "gene", str(tmp_path), 102).give_list_for_case("assembly") == [102]


def test_releases_before_the_first_release_of_ensembl_s_databases_are_left_out(tmp_path, caplog):
    """An organism registered with a range from release 40 is read from release 48 on (Ensembl's first database)."""
    from tests.unit.test_organism_table import restored_organism_table

    with restored_organism_table():
        organisms.register_organism(
            "danio_rerio",
            {9: {"Name": "Zv9", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [40, None]}},
            symbol_databases=["ZFIN"],
        )
        path = tmp_path / "danio_rerio_externals_modified.yml"
        _write(path, {"danio_rerio": {"gene": {"ZFIN": _entry("45,47,48,50", assembly="9")}}})
        with caplog.at_level(logging.WARNING, logger="idtrack.build.externals"):
            data = load_externals_yaml(str(path), shipped=False)
        assert data["danio_rerio"]["gene"]["ZFIN"]["Assembly"]["9"]["Ensembl release"] == "48,50"
        assert "[45, 47] lie outside its window 48..None" in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bad", ["false", "no", 0, None, 1])
def test_include_must_be_a_boolean(tmp_path, bad):
    """organisms_externals-9: a quoted "false" (or a number) is an error naming the entry, never "enabled"."""
    path = tmp_path / "f.yml"
    _write(path, {"sus_scrofa": {"gene": {"EntrezGene": _entry("110", include=bad)}}})
    with pytest.raises(ExternalsError, match="sus_scrofa/gene/EntrezGene assembly '111'"):
        load_externals_yaml(str(path), shipped=False)


def test_the_file_is_parsed_once_and_the_fallback_announced_once(tmp_path, monkeypatch, caplog):
    """organisms_externals-8: 100 look-ups parse the file once and log the fallback to the shipped file once."""
    parsed = []
    real = ext.validate_externals_yaml
    monkeypatch.setattr(ext, "validate_externals_yaml", lambda d, p, s: parsed.append(p) or real(d, p, s))
    monkeypatch.setattr(ext, "_yaml_cache", {})
    monkeypatch.setattr(ext, "_fallback_announced", set())
    e = ExternalDatabases("sus_scrofa", 110, "gene", str(tmp_path), 111)
    with caplog.at_level(logging.WARNING, logger="idtrack.build.externals"):
        for _ in range(100):
            e.give_list_for_case("db")
    assert len(parsed) == 1
    assert sum("default file" in r.getMessage() for r in caplog.records) == 1


def test_a_changed_file_is_read_again(tmp_path):
    """The parse cache follows the file: an edit is seen by the next look-up."""
    path = tmp_path / "sus_scrofa_externals_modified.yml"
    _write(path, {"sus_scrofa": {"gene": {"EntrezGene": _entry("110")}}})
    e = ExternalDatabases("sus_scrofa", 110, "gene", str(tmp_path), 111)
    assert e.give_list_for_case("db") == ["EntrezGene"]
    _write(path, {"sus_scrofa": {"gene": {"EntrezGene": _entry("110", include=False), "HGNC Symbol": _entry("110")}}})
    st = os.stat(path)
    os.utime(path, ns=(st.st_mtime_ns + 10**9, st.st_mtime_ns + 10**9))
    assert e.give_list_for_case("db") == ["HGNC Symbol"]


def test_an_interrupted_write_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    """organisms_externals-10: writing goes to a temporary file that replaces the original only when complete."""
    path = str(tmp_path / "f.yml")
    original = {"sus_scrofa": {"gene": {"EntrezGene": _entry("110")}}}
    write_externals_yaml(original, path)
    before = open(path).read()
    real_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        write_externals_yaml({"sus_scrofa": {"gene": {"HGNC Symbol": _entry("111")}}}, path)
    monkeypatch.setattr(os, "replace", real_replace)
    assert open(path).read() == before
    assert os.listdir(tmp_path) == ["f.yml"]


def test_extending_the_curated_file_keeps_a_backup(tmp_path):
    """organisms_externals-10: the replaced curated file is kept as `.bak`."""
    path = tmp_path / "sus_scrofa_externals_modified.yml"
    write_externals_yaml({"sus_scrofa": {"gene": {"EntrezGene": _entry("110")}}}, str(path))
    before = path.read_text()
    content = pd.DataFrame(
        {"organism": ["sus_scrofa"], "form": ["gene"], "name_db": ["EntrezGene"], "assembly": [111], "release": [111]}
    )
    ExternalDatabases("sus_scrofa", 111, "gene", str(tmp_path), 111).extend_modified_yaml(content)
    assert (tmp_path / "sus_scrofa_externals_modified.yml.bak").read_text() == before
    assert "110,111" in path.read_text()


def test_template_merges_assembly_keys_given_as_text_and_as_numbers(tmp_path):
    """organisms_externals-13: assemblies 111 and "111" are one assembly entry of the template."""
    df = pd.DataFrame(
        {
            "organism": ["sus_scrofa"] * 2,
            "form": ["gene"] * 2,
            "name_db": ["EntrezGene"] * 2,
            "assembly": pd.Series([111, "111"], dtype=object),
            "release": [110, 111],
        }
    )
    e = ExternalDatabases("sus_scrofa", 111, "gene", str(tmp_path), 111)
    e.create_template_yaml(df)
    data = yaml.safe_load(open(e.file_name_template_yaml()))
    assert data["sus_scrofa"]["gene"]["EntrezGene"]["Assembly"] == {
        "111": {"Ensembl release": "110,111", "Include": False}
    }


def test_db_list_is_sorted(tmp_path):
    """Deterministic: the enabled databases come back in one order whatever the hash seed."""
    path = tmp_path / "sus_scrofa_externals_modified.yml"
    _write(path, {"sus_scrofa": {"gene": {n: _entry("110") for n in ["Z", "A", "M"]}}})
    assert ExternalDatabases("sus_scrofa", 110, "gene", str(tmp_path), 111).give_list_for_case("db") == ["A", "M", "Z"]
    assert organisms.ORGANISM_TABLE["sus_scrofa"]  # the table is the one the checks use
