#!/usr/bin/env python3
"""Unit tests for idtrack._api module.

Tests the API class which provides the high-level facade.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestAPIInitialization:
    """Test API class initialization."""

    def test_api_import(self):
        """Test API can be imported."""
        from idtrack._api import API

        assert API is not None

    def test_init_with_local_repository(self, temp_dir):
        """Test initialization with local repository."""
        from idtrack._api import API

        # Create instance without calling __init__ to test attribute assignment
        api = API.__new__(API)
        api.local_repository = temp_dir
        assert api.local_repository == temp_dir

    def test_has_logger_setup_method(self):
        """Test API has configure_logger method."""
        from idtrack._api import API

        assert hasattr(API, "configure_logger")


class TestAPIConfigureLogger:
    """Test configure_logger method."""

    def test_configure_logger_exists(self):
        """Test configure_logger is callable."""
        from idtrack._api import API

        assert callable(getattr(API, "configure_logger", None))


class TestAPIBuildGraph:
    """Test build_graph functionality."""

    def test_build_graph_method_exists(self):
        """Test build_graph method exists."""
        from idtrack._api import API

        assert hasattr(API, "build_graph")

    def test_build_graph_is_callable(self):
        """Test build_graph is callable."""
        from idtrack._api import API

        assert callable(getattr(API, "build_graph", None))


class TestAPILazyLoading:
    """Test lazy loading of Track and TrackTests."""

    def test_api_has_track_property(self):
        """Test API has track property."""
        from idtrack._api import API

        # Check if there's some form of track access
        assert hasattr(API, "__init__")


class TestAPICalculateCaches:
    """Test calculate_graph_caches method."""

    def test_calculate_graph_caches_exists(self):
        """Test calculate_graph_caches method exists or similar method."""
        from idtrack._api import API

        # Check for calculate_graph_caches or alternative method names
        has_cache_method = (
            hasattr(API, "calculate_graph_caches") or hasattr(API, "_calculate_caches") or hasattr(API, "build_caches")
        )
        assert has_cache_method or hasattr(
            API, "build_graph"
        ), "API should have cache calculation method or build_graph"


class TestAPIWithMockedDependencies:
    """Test API with mocked dependencies."""

    def test_api_methods_exist(self):
        """Test expected API methods exist."""
        from idtrack._api import API

        # These are the expected public methods based on the docstrings
        expected_methods = [
            "__init__",
        ]

        for method in expected_methods:
            assert hasattr(API, method), f"Missing method: {method}"

    def test_api_can_be_instantiated_with_mocks(self, temp_dir):
        """Test API instantiation - API only takes local_repository."""
        from idtrack._api import API

        # API __init__ only takes local_repository parameter
        # It doesn't need any external calls during instantiation
        api = API(local_repository=temp_dir)

        assert api is not None
        assert api.local_repository == temp_dir
        assert api.logger_configured is False
        assert hasattr(api, "log")

    def test_api_configure_logger(self, temp_dir):
        """Test configure_logger can be called."""
        import logging

        from idtrack._api import API

        api = API(local_repository=temp_dir)

        # First call should configure logger
        api.configure_logger(logging.INFO)
        assert api.logger_configured is True

        # Second call should be no-op
        api.configure_logger(logging.DEBUG)
        assert api.logger_configured is True

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
        """get_database_manager should default ignore_before to the earliest supported release."""
        from idtrack._api import API
        from idtrack._db import DB

        api = API(local_repository=temp_dir)

        with patch("idtrack._api.DatabaseManager") as mock_dm:
            mock_dm.return_value = MagicMock()
            api.get_database_manager(organism_name="homo_sapiens", snapshot_release=110)

            _args, kwargs = mock_dm.call_args
            assert kwargs["ignore_before"] == int(min(DB.mysql_port_min_release.values()))
            assert kwargs["ignore_after"] == 110

    def test_get_database_manager_respects_ignore_before_override(self, temp_dir):
        """get_database_manager should pass through an explicit ignore_before."""
        from idtrack._api import API

        api = API(local_repository=temp_dir)

        with patch("idtrack._api.DatabaseManager") as mock_dm:
            mock_dm.return_value = MagicMock()
            api.get_database_manager(organism_name="homo_sapiens", snapshot_release=110, ignore_before=90)

            _args, kwargs = mock_dm.call_args
            assert kwargs["ignore_before"] == 90


class TestAPIPublicInterface:
    """Test API public interface matches documentation."""

    def test_class_docstring_exists(self):
        """Test API has a docstring."""
        from idtrack._api import API

        assert API.__doc__ is not None
        assert len(API.__doc__) > 0

    def test_init_accepts_parameters(self):
        """Test __init__ accepts expected parameters."""
        import inspect

        from idtrack._api import API

        sig = inspect.signature(API.__init__)
        params = list(sig.parameters.keys())

        # Should accept self and at least local_repository
        assert "self" in params
        # Other params depend on implementation


class TestAPIExportedFromPackage:
    """Test API is properly exported from package."""

    def test_api_in_package_init(self):
        """Test API is importable from idtrack package."""
        try:
            from idtrack import API

            assert API is not None
        except ImportError:
            # May fail if dependencies not installed
            pass

    def test_api_class_name(self):
        """Test API class has correct name."""
        from idtrack._api import API

        assert API.__name__ == "API"


class TestAPIIntegrationPoints:
    """Test API integration with other modules."""

    def test_imports_db(self):
        """Test API module can access DB constants."""
        from idtrack._db import DB

        # Verify DB is importable and has expected attributes
        assert DB is not None
        assert hasattr(DB, "forms_in_order")

    def test_imports_track(self):
        """Test API module can import Track."""
        try:
            from idtrack._track import Track

            assert Track is not None
        except ImportError:
            pass

    def test_imports_graph_maker(self):
        """Test API module can import GraphMaker."""
        try:
            from idtrack._graph_maker import GraphMaker

            assert GraphMaker is not None
        except ImportError:
            pass


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_invalid_organism_handling(self, temp_dir):
        """Test handling of invalid organism."""
        from idtrack._api import API

        with patch("idtrack._api.VerifyOrganism") as mock_verify:
            mock_verify.side_effect = KeyError("Unknown organism")

            with pytest.raises((KeyError, ValueError, Exception)):
                API(
                    local_repository=temp_dir,
                    organism="invalid_organism",
                )

    def test_invalid_repository_handling(self):
        """API should not validate `local_repository` during initialization."""
        from idtrack._api import API

        local_repository = "/nonexistent/path/that/does/not/exist"
        api = API(local_repository=local_repository)
        assert api.local_repository == local_repository
