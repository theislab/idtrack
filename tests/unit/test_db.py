#!/usr/bin/env python3
"""Unit tests for idtrack._db module.

Tests the DB constants class and related utilities.
"""

from __future__ import annotations

import pytest


class TestDBConstants:
    """Test the DB constants class."""

    def test_db_class_exists(self):
        """Test that DB class can be imported."""
        from idtrack._db import DB

        assert DB is not None

    def test_id_ver_delimiter(self):
        """Test ID version delimiter is a dot."""
        from idtrack._db import DB

        assert DB.id_ver_delimiter == "."

    def test_first_version_is_integer(self):
        """Test first_version is an integer."""
        from idtrack._db import DB

        assert isinstance(DB.first_version, int)
        assert DB.first_version >= 1

    def test_timeout_values_are_positive(self):
        """Test timeout values are positive integers."""
        from idtrack._db import DB

        assert DB.connection_timeout > 0
        assert DB.reading_timeout > 0

    def test_ensembl_ftp_base_is_valid_host(self):
        """Test FTP base is a valid hostname."""
        from idtrack._db import DB

        assert isinstance(DB.ensembl_ftp_base, str)
        assert "ensembl" in DB.ensembl_ftp_base.lower()

    def test_rest_server_api_is_https(self):
        """Test REST API uses HTTPS."""
        from idtrack._db import DB

        assert DB.rest_server_api.startswith("https://")

    def test_rest_server_ext_is_valid_path(self):
        """Test REST extension is a valid path."""
        from idtrack._db import DB

        assert DB.rest_server_ext.startswith("/")

    def test_mysql_host_is_valid(self):
        """Test MySQL host is valid."""
        from idtrack._db import DB

        assert isinstance(DB.mysql_host, str)
        assert len(DB.mysql_host) > 0

    def test_mysql_user_is_anonymous(self):
        """Test MySQL user is anonymous."""
        from idtrack._db import DB

        assert DB.myqsl_user == "anonymous"


class TestDBAssemblyConfiguration:
    """Test assembly-related configuration."""

    def test_assembly_mysqlport_priority_structure(self):
        """Test assembly MySQL port priority has correct structure."""
        from idtrack._db import DB

        assert isinstance(DB.assembly_mysqlport_priority, dict)
        assert len(DB.assembly_mysqlport_priority) >= 1

        for organism, assemblies in DB.assembly_mysqlport_priority.items():
            assert isinstance(organism, str)
            assert isinstance(assemblies, dict)
            assert assemblies
            for assembly, config in assemblies.items():
                assert isinstance(assembly, int)
                assert "Ports" in config
                assert "Priority" in config
                assert isinstance(config["Ports"], list)
                assert config["Ports"]
                assert all(isinstance(p, int) for p in config["Ports"])
                assert isinstance(config["Priority"], int)

    def test_assembly_38_has_highest_priority(self):
        """Test assembly 38 (GRCh38) has priority 1."""
        from idtrack._db import DB

        assert "homo_sapiens" in DB.assembly_mysqlport_priority
        assert 38 in DB.assembly_mysqlport_priority["homo_sapiens"]
        assert DB.assembly_mysqlport_priority["homo_sapiens"][38]["Priority"] == 1

    def test_assembly_37_exists(self):
        """Test assembly 37 (GRCh37) exists."""
        from idtrack._db import DB

        assert "homo_sapiens" in DB.assembly_mysqlport_priority
        assert 37 in DB.assembly_mysqlport_priority["homo_sapiens"]

    def test_main_assembly_is_highest_priority(self):
        """Test main_assembly is the highest priority assembly."""
        from idtrack._db import DB

        # Main assembly should be the one with priority 1
        assert DB.main_assembly == 38

    def test_assembly_priorities_are_unique_per_organism(self):
        """Test assembly priorities are unique and start at 1 per organism."""
        from idtrack._db import DB

        for _organism, assemblies in DB.assembly_mysqlport_priority.items():
            priorities = sorted(int(cfg["Priority"]) for cfg in assemblies.values())
            assert priorities == sorted(set(priorities))
            assert priorities[0] == 1


class TestDBNodeTypes:
    """Test node type constants."""

    def test_node_type_str_exists(self):
        """Test node_type_str key exists."""
        from idtrack._db import DB

        assert isinstance(DB.node_type_str, str)
        assert DB.node_type_str == "node_type"

    def test_nts_external_is_string(self):
        """Test external node type is a string."""
        from idtrack._db import DB

        assert isinstance(DB.nts_external, str)
        assert DB.nts_external == "external"

    def test_forms_in_order(self):
        """Test forms are in correct order."""
        from idtrack._db import DB

        assert DB.forms_in_order == ["gene", "transcript", "translation"]

    def test_backbone_form_is_gene(self):
        """Test backbone form is gene."""
        from idtrack._db import DB

        assert DB.backbone_form == "gene"

    def test_nts_ensembl_contains_all_forms(self):
        """Test nts_ensembl contains all forms."""
        from idtrack._db import DB

        for form in DB.forms_in_order:
            assert form in DB.nts_ensembl
            assert DB.nts_ensembl[form] == f"ensembl_{form}"

    def test_nts_ensembl_reverse_is_inverse(self):
        """Test nts_ensembl_reverse is the inverse mapping."""
        from idtrack._db import DB

        for form, node_type in DB.nts_ensembl.items():
            assert DB.nts_ensembl_reverse[node_type] == form

    def test_nts_assembly_structure(self):
        """Test nts_assembly has correct structure."""
        from idtrack._db import DB

        for assembly in DB.all_assemblies:
            assert assembly in DB.nts_assembly
            for form in DB.forms_in_order:
                assert form in DB.nts_assembly[assembly]
                expected = f"assembly_{assembly}_ensembl_{form}"
                assert DB.nts_assembly[assembly][form] == expected

    def test_nts_assembly_reverse_is_complete(self):
        """Test nts_assembly_reverse covers all assembly node types."""
        from idtrack._db import DB

        for assembly in DB.nts_assembly:
            for form in DB.nts_assembly[assembly]:
                node_type = DB.nts_assembly[assembly][form]
                assert node_type in DB.nts_assembly_reverse

    def test_nts_base_ensembl_structure(self):
        """Test nts_base_ensembl has correct structure."""
        from idtrack._db import DB

        for form in DB.forms_in_order:
            assert form in DB.nts_base_ensembl
            assert DB.nts_base_ensembl[form] == f"base_ensembl_{form}"

    def test_nts_assembly_gene_contains_gene_types(self):
        """Test nts_assembly_gene contains all assembly-specific gene types."""
        from idtrack._db import DB

        assert isinstance(DB.nts_assembly_gene, set)
        for assembly in DB.all_assemblies:
            expected = f"assembly_{assembly}_ensembl_gene"
            assert expected in DB.nts_assembly_gene


class TestDBSentinels:
    """Test sentinel values."""

    def test_synonym_id_nodes_prefix(self):
        """Test synonym ID nodes prefix."""
        from idtrack._db import DB

        assert isinstance(DB.synonym_id_nodes_prefix, str)
        assert len(DB.synonym_id_nodes_prefix) > 0

    def test_no_old_node_id(self):
        """Test no_old_node_id sentinel."""
        from idtrack._db import DB

        assert DB.no_old_node_id == "Void"

    def test_no_new_node_id(self):
        """Test no_new_node_id sentinel."""
        from idtrack._db import DB

        assert DB.no_new_node_id == "Retired"

    def test_alternative_versions_set(self):
        """Test alternative_versions is a set with both sentinels."""
        from idtrack._db import DB

        assert isinstance(DB.alternative_versions, set)
        assert DB.no_old_node_id in DB.alternative_versions
        assert DB.no_new_node_id in DB.alternative_versions
        assert len(DB.alternative_versions) == 2

    def test_hyperconnecting_threshold(self):
        """Test hyperconnecting threshold is reasonable."""
        from idtrack._db import DB

        assert isinstance(DB.hyperconnecting_threshold, int)
        assert DB.hyperconnecting_threshold > 0
        assert DB.hyperconnecting_threshold == 20


class TestDBEdgeAttributes:
    """Test edge attribute constants."""

    def test_connection_dict_key(self):
        """Test connection dict key."""
        from idtrack._db import DB

        assert DB.connection_dict == "connection"

    def test_conn_dict_str_ensembl_base(self):
        """Test Ensembl base connection string."""
        from idtrack._db import DB

        assert DB.conn_dict_str_ensembl_base == "ensembl_base"


class TestDBExternalSearchSettings:
    """Test external search settings."""

    def test_external_search_settings_structure(self):
        """Test external search settings has required keys."""
        from idtrack._db import DB

        assert isinstance(DB.external_search_settings, dict)
        assert "jump_limit" in DB.external_search_settings
        assert "synonymous_max_depth" in DB.external_search_settings
        assert "nts_backbone" in DB.external_search_settings

    def test_external_search_settings_values(self):
        """Test external search settings have valid values."""
        from idtrack._db import DB

        assert DB.external_search_settings["jump_limit"] > 0
        assert DB.external_search_settings["synonymous_max_depth"] > 0
        assert DB.external_search_settings["nts_backbone"] == DB.nts_ensembl["gene"]


class TestDBHDF5Settings:
    """Test HDF5 settings."""

    def test_placeholder_na(self):
        """Test placeholder NA value."""
        from idtrack._db import DB

        assert isinstance(DB.placeholder_na, str)
        assert len(DB.placeholder_na) > 0

    def test_utf8_constant(self):
        """Test UTF8 encoding constant."""
        from idtrack._db import DB

        assert DB.UTF8 == "utf-8"

    def test_utf8_str_dtype(self):
        """Test UTF8_STR is h5py dtype."""
        import h5py

        from idtrack._db import DB

        assert DB.UTF8_STR == h5py.string_dtype(encoding=DB.UTF8)


class TestEmptyConversionMetricsError:
    """Test the custom exception class."""

    def test_exception_is_value_error(self):
        """Test EmptyConversionMetricsError inherits from ValueError."""
        from idtrack._db import EmptyConversionMetricsError

        assert issubclass(EmptyConversionMetricsError, ValueError)

    def test_exception_can_be_raised(self):
        """Test exception can be raised and caught."""
        from idtrack._db import EmptyConversionMetricsError

        with pytest.raises(EmptyConversionMetricsError):
            raise EmptyConversionMetricsError("test message")

    def test_exception_message(self):
        """Test exception preserves message."""
        from idtrack._db import EmptyConversionMetricsError

        msg = "conversion metrics are empty"
        with pytest.raises(EmptyConversionMetricsError) as exc_info:
            raise EmptyConversionMetricsError(msg)
        assert msg in str(exc_info.value)


class TestMissingValues:
    """Test MISSING_VALUES constant."""

    def test_missing_values_is_list(self):
        """Test MISSING_VALUES is a list."""
        from idtrack._db import MISSING_VALUES

        assert isinstance(MISSING_VALUES, list)

    def test_missing_values_contains_common_nulls(self):
        """Test MISSING_VALUES contains common null representations."""
        from idtrack._db import MISSING_VALUES

        expected = ["NA", "N/A", "None", "null", "NULL", "", "nan", "NaN"]
        for val in expected:
            assert val in MISSING_VALUES, f"Missing expected value: {val}"

    def test_missing_values_contains_empty_strings(self):
        """Test MISSING_VALUES contains various empty strings."""
        from idtrack._db import MISSING_VALUES

        assert "" in MISSING_VALUES
        assert " " in MISSING_VALUES

    def test_missing_values_contains_placeholders(self):
        """Test MISSING_VALUES contains placeholder values."""
        from idtrack._db import MISSING_VALUES

        placeholders = ["missing", "unknown", "undefined", "TBD", "not available"]
        for p in placeholders:
            # Check case-insensitive
            assert any(p.lower() == v.lower() for v in MISSING_VALUES), f"Missing: {p}"


class TestDBImmutability:
    """Test that DB class attributes behave as expected."""

    def test_db_is_not_instantiated(self):
        """Test DB class is used directly, not instantiated."""
        from idtrack._db import DB

        # Access attributes directly from class
        assert hasattr(DB, "id_ver_delimiter")
        assert hasattr(DB, "forms_in_order")

    def test_class_attributes_accessible(self):
        """Test all expected class attributes are accessible."""
        from idtrack._db import DB

        required_attrs = [
            "id_ver_delimiter",
            "first_version",
            "connection_timeout",
            "reading_timeout",
            "ensembl_ftp_base",
            "rest_server_api",
            "mysql_host",
            "assembly_mysqlport_priority",
            "main_assembly",
            "node_type_str",
            "nts_external",
            "forms_in_order",
            "backbone_form",
            "nts_ensembl",
            "connection_dict",
        ]
        for attr in required_attrs:
            assert hasattr(DB, attr), f"Missing attribute: {attr}"
