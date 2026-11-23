#!/usr/bin/env python3
"""The edge of the public API: wrong arguments, results without a target, strict JSON, unknown nodes.

Wrong arguments raise the class the error hierarchy promises, naming the argument; results without a target say why;
results refuse what strict JSON cannot hold; the graph's accessors refuse an unknown node as an input error. The graph
is the hand-made one of :py:mod:`tests.unit.test_explained_path` (no network, no file).
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket

import numpy as np
import pytest

import idtrack
from idtrack import API
from idtrack.errors import (
    ArgumentTypeError,
    IntegrityError,
    InvalidArgumentError,
    UnknownIdentifierError,
)
from idtrack.results import REASONS, ConversionResult
from idtrack.settings import SearchSettings
from tests.unit.test_explained_path import hand_made_graph


@pytest.fixture(scope="module")
def api() -> API:
    """An API over the hand-made graph."""
    return API(hand_made_graph())


def raises(error_class, argument, call, *args, **kwargs):
    """Assert that a call raises *error_class* naming *argument*, and return the error."""
    with pytest.raises(error_class) as caught:
        call(*args, **kwargs)
    assert type(caught.value) is error_class
    assert caught.value.argument == argument
    return caught.value


# ------------------------------------------------------------------------------------------------- wrong arguments


@pytest.mark.wrong_argument_type
@pytest.mark.parametrize("results", ["abc", b"abc", 5, None])
def test_classify_refuses_what_is_not_a_list_of_results(results):
    """``classify`` of a string, bytes, a number or nothing is a type error naming ``results``."""
    raises(ArgumentTypeError, "results", API.classify, results)


@pytest.mark.wrong_argument_type
def test_classify_refuses_a_list_that_holds_other_values(api):
    """A list with a value that is not a result names the positions."""
    error = raises(ArgumentTypeError, "results", API.classify, [api.convert("A.1"), 1, "x"])
    assert error.value == [1, 2]
    with pytest.raises(ArgumentTypeError):
        API.classify(api.convert("A.1"))  # one result, not a list


@pytest.mark.wrong_argument_type
def test_load_checks_its_settings_before_reading_the_file(tmp_path):
    """Settings of the wrong type are refused before the file is looked at (the path does not exist)."""
    path = str(tmp_path / "absent.npz")
    raises(ArgumentTypeError, "graph_settings", API.load, path, graph_settings={"hyperconnective_threshold": 3})
    raises(ArgumentTypeError, "search", API.load, path, search=1)
    raises(ArgumentTypeError, "path", API.load, 5)
    assert not os.path.exists(path)


def test_search_settings_refuse_an_unknown_setting():
    """``SearchSettings.replace`` names the unknown setting and lists the valid ones."""
    error = raises(InvalidArgumentError, "nope", SearchSettings().replace, nope=1)
    assert "budget_states" in error.allowed and "bridge_limit" in error.allowed
    assert SearchSettings().replace(budget_states=10).budget_states == 10


@pytest.mark.wrong_argument_type
def test_a_gene_must_be_a_string(api):
    """``names_of`` and ``gene_to_database`` of a number are a type error naming ``gene``, not an unknown id."""
    raises(ArgumentTypeError, "gene", api.names_of, 5)
    raises(ArgumentTypeError, "gene", api.gene_to_database, 5, "HGNC Symbol")
    raises(ArgumentTypeError, "gene", api.gene_to_database, None, "HGNC Symbol")


@pytest.mark.wrong_argument_type
def test_a_name_must_be_a_string(api):
    """``genes_named`` of a number names ``name``."""
    raises(ArgumentTypeError, "name", api.genes_named, 5)
    raises(ArgumentTypeError, "name", api.graph.genes_named, 5)


@pytest.mark.wrong_argument_type
def test_the_window_of_a_graph_is_checked(tmp_path):
    """A release given as a string, or a window that ends before it begins, is refused naming the argument."""
    repository = str(tmp_path)
    raises(ArgumentTypeError, "first_release", API.graph_path, "homo_sapiens", 115, repository, first_release="56")
    raises(ArgumentTypeError, "snapshot_release", API.graph_path, "homo_sapiens", "115", repository)
    raises(ArgumentTypeError, "snapshot_release", API.graph_path, "homo_sapiens", 115.0, repository)
    error = raises(
        InvalidArgumentError, "first_release", API.graph_path, "homo_sapiens", 100, repository, first_release=101
    )
    assert error.value == 101
    assert API.graph_path("homo_sapiens", 100, repository, first_release=100).endswith("_min100_max100.npz")


@pytest.mark.wrong_argument_type
def test_database_manager_refuses_a_string_release_before_any_network_or_folder(tmp_path, monkeypatch):
    """``API.database_manager`` with the snapshot ``"115"``: a type error naming ``snapshot_release``.

    It is raised at the edge: no socket is opened and the folder is not touched.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    folder = tmp_path / "absent"
    raises(ArgumentTypeError, "snapshot_release", API.database_manager, "homo_sapiens", "115", str(folder))
    raises(
        ArgumentTypeError, "first_release", API.database_manager, "homo_sapiens", 115, str(folder), first_release="90"
    )
    raises(
        InvalidArgumentError, "first_release", API.database_manager, "homo_sapiens", 90, str(folder), first_release=91
    )
    assert not folder.exists()


# ------------------------------------------------------------------------------------ a result without a target


def test_no_conversion_carries_the_engines_reason(api):
    """``E.1`` retires at 4; without bridges nothing leads to release 9: the API reports the engine's reason."""
    result = api.convert("E.1", allow_bridges=False)
    engine = api.engine.convert("E.1", allow_bridges=False)
    assert (result.status, result.candidates) == ("no_conversion", ())
    assert result.reason == engine.reason == "no_path_to_target"
    assert API.reason_text(result) == REASONS["no_path_to_target"]
    # With bridges the same identifier converts, and a result with a target has no reason.
    bridged = api.convert("E.1")
    assert (bridged.status, bridged.reason, API.reason_text(bridged)) == ("converted", None, None)


def test_no_target_says_that_the_genes_have_no_id_in_the_database(api):
    """``C.1`` has no HGNC Symbol: the gene stands in for the missing id, and the result says why."""
    result = api.convert("C.1", final_database="HGNC Symbol")
    assert (result.status, result.no_target, result.final_database) == ("no_target", True, "ensembl_gene")
    assert [(c.gene, c.target) for c in result.candidates] == [("C.1", "C.1")]
    assert result.reason == "no_id_in_final_database"
    assert API.reason_text(result) == REASONS["no_id_in_final_database"]
    named = api.convert("A.1", final_database="HGNC Symbol")
    assert (named.status, named.reason) == ("converted", None)


@pytest.mark.parametrize(
    "unknown,reason", [("NOT_AN_ID", "not_in_graph"), ("A.7", "unknown_version")], ids=["not_in_graph", "version"]
)
def test_not_found_says_why(api, unknown, reason):
    """An identifier the graph does not hold, and an Ensembl id with a version it does not hold."""
    result = api.convert(unknown)
    assert result.reason == reason and API.reason_text(result) == REASONS[reason]


# ------------------------------------------------------------------------------------------------- strict JSON


@pytest.mark.parametrize(
    "change",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": float("-inf")},
        {"bridges": np.int64(1)},
    ],
    ids=["nan", "inf", "-inf", "numpy-int"],
)
def test_to_dict_refuses_what_json_cannot_hold_exactly(api, change):
    """A result holding NaN, an infinity or a NumPy number cannot be written as strict JSON: an integrity error."""
    result = api.convert("A.1")
    broken = dataclasses.replace(result, candidates=(dataclasses.replace(result.candidates[0], **change),))
    with pytest.raises(IntegrityError):
        broken.to_dict()


def test_to_dict_refuses_a_set(api):
    """A set has no order in JSON: refused."""
    result = api.convert("A.1")
    with pytest.raises(IntegrityError):
        dataclasses.replace(result, known_versions={"A.1"}).to_dict()


def test_to_dict_of_a_valid_result_is_strict_json_and_reads_back(api):
    """The counterpart of the refusals: a real result writes with ``allow_nan=False`` and reads back equal."""
    for identifier in ("A.1", "E.1", "OLDSYM", "X.1"):
        result = api.convert(identifier, explain=True, final_database="HGNC Symbol")
        assert ConversionResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result


# ------------------------------------------------------------------------------------- the graph's accessors


@pytest.mark.parametrize(
    "accessor", ["node_type", "is_backbone", "history_edges", "xref_edges", "active_ranges", "non_reference_releases"]
)
def test_an_unknown_node_is_an_input_error(api, accessor):
    """The graph's public accessors refuse a node that is not in the graph as an unknown identifier."""
    error = raises(UnknownIdentifierError, "node", getattr(api.graph, accessor), "NOPE")
    assert error.value == "NOPE"


@pytest.mark.parametrize("accessor", ["base_id", "version"])
def test_a_node_without_version_has_no_base_id(api, accessor):
    """``base_id``/``version`` of an external id: not an Ensembl ``<id>.<version>`` node; of nothing: unknown."""
    raises(InvalidArgumentError, "node", getattr(api.graph, accessor), "SYM")
    raises(UnknownIdentifierError, "node", getattr(api.graph, accessor), "NOPE")
    assert (api.graph.base_id("A.1"), api.graph.version("A.1")) == ("A", "1")


def test_the_engine_is_not_exported_at_the_top():
    """The API is the edge where values are checked; the engine stays in ``idtrack.engine``."""
    assert "Track" not in idtrack.__all__
    with pytest.raises(AttributeError):
        idtrack.Track  # noqa: B018 - the access itself is the test
    from idtrack.engine import Track

    assert isinstance(API(hand_made_graph()).engine, Track)
