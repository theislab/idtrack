"""Unit tests of :py:class:`idtrack.API` that need no graph: construction, logging, organism names, the manager."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from idtrack import errors


class TestAPIWithMockedDependencies:
    """The parts of the API that need no graph."""

    @pytest.mark.wrong_argument_type
    def test_api_takes_a_graph_only(self, temp_dir):
        """An API holds a loaded graph; anything else is refused naming the way to load one."""
        from idtrack.api import API

        with pytest.raises(errors.ArgumentTypeError, match="API.load"):
            API(temp_dir)  # type: ignore[arg-type]

    def test_api_configure_logger(self):
        """configure_logger sets the package logger's level and adds one handler however often it is called."""
        import logging

        from idtrack.api import API

        package = logging.getLogger("idtrack")
        saved = (package.level, list(package.handlers))
        try:
            API.configure_logger(logging.INFO)
            n_handlers = len(package.handlers)
            API.configure_logger(logging.DEBUG)
            assert package.level == logging.DEBUG and len(package.handlers) == n_handlers
        finally:
            package.setLevel(saved[0])
            package.handlers[:] = saved[1]

    def test_api_resolve_organism(self):
        """resolve_organism uses VerifyOrganism (Ensembl's species list)."""
        from idtrack.api import API

        with patch("idtrack.build.species.VerifyOrganism") as mock_verify:
            mock_instance = MagicMock()
            mock_instance.get_formal_name.return_value = "homo_sapiens"
            mock_instance.get_latest_release.return_value = 110
            mock_verify.return_value = mock_instance

            formal_name, latest_release = API.ensembl_organism("human")

            assert formal_name == "homo_sapiens"
            assert latest_release == 110
            mock_verify.assert_called_once_with("human")

    def test_database_manager_defaults_first_release(self, temp_dir):
        """database_manager leaves the default of first_release to the manager (one default for both)."""
        from idtrack.api import API

        with patch("idtrack.build.database_manager.DatabaseManager") as mock_dm:
            mock_dm.return_value = MagicMock()
            API.database_manager("homo_sapiens", 110, temp_dir)

            _args, kwargs = mock_dm.call_args
            assert kwargs["first_release"] is None
            assert kwargs["snapshot_release"] == 110
            assert kwargs["local_repository"] == temp_dir

    def test_database_manager_respects_first_release_override(self, temp_dir):
        """database_manager passes an explicit first release through."""
        from idtrack.api import API

        with patch("idtrack.build.database_manager.DatabaseManager") as mock_dm:
            mock_dm.return_value = MagicMock()
            API.database_manager("homo_sapiens", 110, temp_dir, first_release=90)

            _args, kwargs = mock_dm.call_args
            assert kwargs["first_release"] == 90


class TestAPIErrorHandling:
    """Loading and building fail early and say why."""

    @pytest.mark.wrong_argument_type
    def test_load_of_a_missing_file_is_a_graph_file_error_naming_it(self, temp_dir):
        """API.load never builds: a missing file is a graph-file error that names the file."""
        from idtrack.api import API

        path = os.path.join(temp_dir, "graph_homo_sapiens_assembly38_min48_max115.npz")
        with pytest.raises(errors.GraphFileError, match="No graph file") as raised:
            API.load(path)
        assert raised.value.path == path
        with pytest.raises(errors.ArgumentTypeError):
            API.load(115)  # type: ignore[arg-type]

    def test_load_of_an_unreadable_file_is_a_graph_file_error(self, temp_dir):
        """API.load of a file that is not a graph is a graph-file error, not a mismatch."""
        from idtrack.api import API

        path = os.path.join(temp_dir, "graph_homo_sapiens_assembly38_min48_max115.npz")
        with open(path, "wb") as handle:
            handle.write(b"not a graph")
        with pytest.raises(errors.GraphFileError, match="cannot be read") as raised:
            API.load(path)
        assert not isinstance(raised.value, errors.GraphMismatchError)

    def test_build_into_a_missing_folder_is_refused_before_any_network(self):
        """API.build checks its folder first (no manager, no network)."""
        from idtrack.api import API

        with patch("idtrack.build.database_manager.DatabaseManager") as mock_dm:
            with pytest.raises(errors.InvalidArgumentError, match="existing folder"):
                API.build("homo_sapiens", 115, "/nonexistent/path/that/does/not/exist")
            mock_dm.assert_not_called()
        with pytest.raises(errors.UnknownOrganismError):
            API.build("danio_rerio", 115, "/tmp")  # noqa: S108 - any existing folder; nothing is written

    @pytest.mark.wrong_argument_type
    def test_graph_path_needs_no_network(self, temp_dir):
        """API.graph_path names the file from the organism table alone and checks its arguments."""
        from idtrack.api import API

        assert API.graph_path("sus_scrofa", 115, temp_dir) == os.path.join(
            temp_dir, "graph_sus_scrofa_assembly111_min56_max115.npz"
        )
        with pytest.raises(errors.ArgumentTypeError):
            API.graph_path("sus_scrofa", 115.0, temp_dir)  # type: ignore[arg-type]
