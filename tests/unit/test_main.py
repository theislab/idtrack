"""The command line (``idtrack``, ``python -m idtrack``): its help and its four commands on a real graph."""

import json
import logging

import pytest

import idtrack
import idtrack.__main__ as idtrack_main
from idtrack.build.database_manager import DatabaseManager
from tests._real_tables import SNAPSHOT, tables_path

PIG = "sus_scrofa"
needs_pig_tables = pytest.mark.skipif(not tables_path(PIG).is_file(), reason="the cut pig tables are not available")


@pytest.fixture(autouse=True)
def _package_logger_restored():
    """Leave the package logger as each test found it (``idtrack build`` configures it, as a program does)."""
    package = logging.getLogger("idtrack")
    level, handlers = package.level, list(package.handlers)
    yield
    package.setLevel(level)
    package.handlers[:] = handlers


def test_package_exports_resolve() -> None:
    """Every exported name resolves, and the version comes from the installed metadata (or says it is unknown)."""
    import importlib.util

    extra = importlib.util.find_spec("anndata") is not None and importlib.util.find_spec("scipy") is not None
    for name in idtrack.__all__:
        assert getattr(idtrack, name) is not None, name
    assert "HarmonizeFeatures" not in idtrack.__all__  # it needs the extra: by name only
    if extra:
        assert idtrack.HarmonizeFeatures is not None
    else:
        with pytest.raises(ImportError, match="idtrack\\[harmonize\\]"):
            idtrack.HarmonizeFeatures
    assert idtrack.__version__ == "unknown" or idtrack.__version__[0].isdigit()


def test_without_a_command_the_help_says_what_the_program_does_and_how_to_start(capsys) -> None:
    """``idtrack`` alone prints the help: what it does, the four commands and the first commands to run."""
    assert idtrack_main.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: idtrack")
    assert "converts gene identifiers" in out
    for command in ("build", "verify", "info", "convert"):
        assert f"    {command} " in out
    assert "how to start:" in out and "idtrack build human 115" in out and "idtrack convert " in out


def test_version_flag_prints_the_version(capsys) -> None:
    """``--version`` prints the version and exits with status 0."""
    with pytest.raises(SystemExit) as exc_info:
        idtrack_main.main(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"idtrack {idtrack.__version__}\n"


def test_an_unknown_command_exits_with_status_2(capsys) -> None:
    """A command line that cannot be parsed is argparse's error (status 2)."""
    with pytest.raises(SystemExit) as exc_info:
        idtrack_main.main(["somearg"])
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_an_error_of_idtrack_is_printed_with_status_1(tmp_path, capsys) -> None:
    """A missing graph file is idtrack's error: one line on standard error, status 1, no traceback."""
    assert idtrack_main.main(["info", str(tmp_path / "graph.npz")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("idtrack: error: No graph file") and "Traceback" not in captured.err


def test_build_resolves_the_organism_and_makes_the_repository(tmp_path, monkeypatch, capsys) -> None:
    """``idtrack build`` reads a short name offline, makes the folder and calls the API's build with the arguments."""
    calls = []

    class Built:
        def describe(self):
            return type("D", (), {"organism": PIG, "releases": (56, 115)})()

    def build(organism, snapshot_release, local_repository, *, first_release=None, rebuild=False):
        calls.append((organism, snapshot_release, local_repository, first_release, rebuild))
        path = idtrack.API.graph_path(organism, snapshot_release, local_repository, first_release=first_release)
        with open(path, "wb") as handle:
            handle.write(b"x" * 1000)
        return Built()

    monkeypatch.setattr(idtrack.API, "build", staticmethod(build))
    repository = tmp_path / "new" / "repository"
    assert idtrack_main.main(["build", "pig", "115", "-r", str(repository), "--first-release", "60", "-q"]) == 0
    assert calls == [(PIG, 115, str(repository), 60, False)]
    out = capsys.readouterr().out.splitlines()
    assert out[0] == idtrack.API.graph_path(PIG, 115, str(repository), first_release=60)
    assert out[1].startswith("sus_scrofa, releases 56 to 115, 0.0 MB")


def test_build_refuses_an_organism_it_does_not_know(tmp_path, capsys) -> None:
    """An unknown organism is refused before anything is made or downloaded."""
    assert idtrack_main.main(["build", "zebra", "115", "-r", str(tmp_path / "r")]) == 1
    assert "names no organism" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


@pytest.fixture
def pig_graph(real_world) -> str:
    """The file of the graph of pig, built from its cut tables."""
    world = real_world(PIG)
    return idtrack.API.graph_path(PIG, SNAPSHOT, world.repository)


@needs_pig_tables
def test_info_shows_what_the_graph_holds(pig_graph, capsys) -> None:
    """``idtrack info`` prints the description; with ``--json`` it is the API's description as JSON."""
    described = idtrack.API.load(pig_graph).describe()
    assert idtrack_main.main(["info", pig_graph]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"organism: {PIG}\nreleases: {described.releases[0]} to {described.releases[-1]}")
    assert f"content digest: {described.content_digest}" in out
    assert idtrack_main.main(["info", pig_graph, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == described.to_dict()


@needs_pig_tables
def test_convert_answers_as_the_api_does(pig_graph, tmp_path, capsys) -> None:
    """``idtrack convert`` gives, for identifiers of the arguments and of a file, what ``convert_many`` gives."""
    api = idtrack.API.load(pig_graph)
    gene = next(node for node in api.graph.nodes if api.graph.node_type(node) == idtrack.ids.BASE_GENE)
    listing = tmp_path / "ids.txt"
    listing.write_text("  NOT_A_GENE  \n\n")
    assert idtrack_main.main(["convert", pig_graph, gene, "-f", str(listing), "--all"]) == 0
    lines = capsys.readouterr().out.splitlines()
    expected = api.convert_many([gene, "NOT_A_GENE"])
    assert lines[0] == "query\tstatus\ttargets\treason"
    assert lines[1:] == [f"{r.query_id}\t{r.status}\t{','.join(r.targets('all'))}\t{r.reason or ''}" for r in expected]
    assert lines[2] == "NOT_A_GENE\tnot_found\t\tnot_in_graph"
    assert idtrack_main.main(["convert", pig_graph, gene, "--json", "--final-database", "base_ensembl_gene"]) == 0
    (result,) = json.loads(capsys.readouterr().out)
    assert result == api.convert_many([gene], final_database="base_ensembl_gene")[0].to_dict()


@needs_pig_tables
def test_convert_refuses_bad_arguments_and_no_identifiers(pig_graph, capsys) -> None:
    """An unknown database is idtrack's error (status 1); so is a call without identifiers."""
    assert idtrack_main.main(["convert", pig_graph, "TP53", "--final-database", "no such database"]) == 1
    assert "is not a database of this graph" in capsys.readouterr().err
    assert idtrack_main.main(["convert", pig_graph]) == 1
    assert "no identifiers" in capsys.readouterr().err


@needs_pig_tables
def test_verify_recomputes_the_content_digest(pig_graph, monkeypatch, capsys) -> None:
    """The content of a graph file gives the digest it records; a file whose content differs is not verified."""
    assert idtrack_main.main(["verify", pig_graph]) == 0
    out = capsys.readouterr().out
    assert "content digest: ok" in out and out.endswith("verified\n")

    import idtrack.build.builder

    monkeypatch.setattr(idtrack.build.builder, "_content_digest", lambda store: "0" * 64)
    assert idtrack_main.main(["verify", pig_graph]) == 1
    out = capsys.readouterr().out
    assert "content digest: DIFFERS" in out and out.endswith("NOT verified\n")


@needs_pig_tables
def test_verify_with_the_tables_runs_the_self_tests(real_world, pig_graph, monkeypatch, capsys) -> None:
    """With ``--repository`` every self-test runs against the graph's tables, and they all hold on the pig graph."""
    world = real_world(PIG)
    world.tables.install(monkeypatch, world.repository)
    monkeypatch.setattr(DatabaseManager, "_core_db_index_cache", {})
    assert idtrack_main.main(["verify", pig_graph, "--repository", world.repository]) == 0
    out = capsys.readouterr().out
    for name in idtrack_main.SELF_CHECKS:
        assert f"{name}: holds" in out
    assert out.endswith("verified\n")
    assert idtrack_main.main(["verify", pig_graph, "--repository", world.repository + "_missing"]) == 1
    assert "is not a folder" in capsys.readouterr().err
