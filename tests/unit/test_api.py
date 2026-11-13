#!/usr/bin/env python3
"""Unit tests for idtrack._api module.

Tests the API class which provides the high-level facade.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestAPIWithMockedDependencies:
    """Test API with mocked dependencies."""

    def test_api_can_be_instantiated_with_mocks(self, temp_dir):
        """Test API instantiation - API only takes local_repository."""
        from idtrack._api import API

        # API __init__ only takes local_repository parameter
        # It doesn't need any external calls during instantiation
        api = API(local_repository=temp_dir)

        assert api is not None
        assert api.local_repository == temp_dir
        assert api.track is None
        assert api.log.name == "idtrack._api"

    def test_api_configure_logger(self, temp_dir):
        """configure_logger sets the package logger's level and adds one handler however often it is called."""
        import logging

        from idtrack._api import API

        api = API(local_repository=temp_dir)
        package = logging.getLogger("idtrack")
        saved = (package.level, list(package.handlers))
        try:
            api.configure_logger(logging.INFO)
            n_handlers = len(package.handlers)
            api.configure_logger(logging.DEBUG)
            assert package.level == logging.DEBUG and len(package.handlers) == n_handlers
        finally:
            package.setLevel(saved[0])
            package.handlers[:] = saved[1]

    def test_api_resolve_organism(self, temp_dir):
        """Test resolve_organism uses VerifyOrganism."""
        from idtrack._api import API

        api = API(local_repository=temp_dir)

        with patch("idtrack._api.VerifyOrganism") as mock_verify:
            mock_instance = MagicMock()
            mock_instance.get_formal_name.return_value = "homo_sapiens"
            mock_instance.get_latest_release.return_value = 110
            mock_verify.return_value = mock_instance

            formal_name, latest_release = api.resolve_organism("human")

            assert formal_name == "homo_sapiens"
            assert latest_release == 110
            mock_verify.assert_called_once_with("human")

    def test_get_database_manager_defaults_ignore_before(self, temp_dir):
        """get_database_manager leaves the default of ignore_before to the manager (one default for both)."""
        from idtrack._api import API

        api = API(local_repository=temp_dir)

        with patch("idtrack._database_manager.DatabaseManager") as mock_dm:
            mock_dm.return_value = MagicMock()
            api.get_database_manager(organism_name="homo_sapiens", snapshot_release=110)

            _args, kwargs = mock_dm.call_args
            assert kwargs["ignore_before"] is None
            assert kwargs["ignore_after"] == 110

    def test_get_database_manager_respects_ignore_before_override(self, temp_dir):
        """get_database_manager should pass through an explicit ignore_before."""
        from idtrack._api import API

        api = API(local_repository=temp_dir)

        with patch("idtrack._database_manager.DatabaseManager") as mock_dm:
            mock_dm.return_value = MagicMock()
            api.get_database_manager(organism_name="homo_sapiens", snapshot_release=110, ignore_before=90)

            _args, kwargs = mock_dm.call_args
            assert kwargs["ignore_before"] == 90


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_invalid_repository_handling(self):
        """API should not validate `local_repository` during initialization."""
        from idtrack._api import API

        local_repository = "/nonexistent/path/that/does/not/exist"
        api = API(local_repository=local_repository)
        assert api.local_repository == local_repository
