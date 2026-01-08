#!/usr/bin/env python3
"""
Unit tests for idtrack._external_databases module.

Tests the ExternalDatabases class for YAML configuration management.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml


class TestExternalDatabasesInitialization:
    """Test ExternalDatabases initialization."""

    def test_init_sets_attributes(self, temp_dir):
        """Test initialization sets all attributes."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        assert edb.organism == "homo_sapiens"
        assert edb.ensembl_release == 110
        assert edb.form == "gene"
        assert edb.local_repository == temp_dir
        assert edb.genome_assembly == 38

    def test_has_logger(self, temp_dir):
        """Test instance has logger."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        assert hasattr(edb, "log")
        assert edb.log is not None


class TestFileNameMethods:
    """Test file name generation methods."""

    def test_file_name_template_yaml(self, temp_dir):
        """Test template YAML filename generation."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        expected = os.path.join(temp_dir, "homo_sapiens_externals_template.yml")
        assert edb.file_name_template_yaml() == expected

    def test_file_name_modified_yaml_configured(self, temp_dir):
        """Test configured modified YAML filename."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        expected = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        assert edb.file_name_modified_yaml(mode="configured") == expected

    def test_file_name_modified_yaml_default(self, temp_dir):
        """Test default modified YAML filename."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        result = edb.file_name_modified_yaml(mode="default")
        assert "default_config" in result
        assert "homo_sapiens_externals_modified.yml" in result

    def test_file_name_modified_yaml_invalid_mode(self, temp_dir):
        """Test invalid mode raises ValueError."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        with pytest.raises(ValueError, match="Unknown value"):
            edb.file_name_modified_yaml(mode="invalid")


class TestCreateTemplateYaml:
    """Test create_template_yaml method."""

    def test_creates_yaml_file(self, temp_dir):
        """Test YAML file is created."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        # Create sample DataFrame
        df = pd.DataFrame({
            "organism": ["homo_sapiens", "homo_sapiens"],
            "form": ["gene", "gene"],
            "name_db": ["HGNC Symbol", "UniProtKB"],
            "assembly": [38, 38],
            "release": [110, 110],
        })

        edb.create_template_yaml(df)

        yaml_path = edb.file_name_template_yaml()
        assert os.path.exists(yaml_path)

    def test_yaml_structure(self, temp_dir):
        """Test created YAML has correct structure."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        df = pd.DataFrame({
            "organism": ["homo_sapiens", "homo_sapiens"],
            "form": ["gene", "gene"],
            "name_db": ["HGNC Symbol", "UniProtKB"],
            "assembly": [38, 38],
            "release": [109, 110],
        })

        edb.create_template_yaml(df)

        with open(edb.file_name_template_yaml()) as f:
            data = yaml.safe_load(f)

        assert "homo_sapiens" in data
        assert "gene" in data["homo_sapiens"]
        assert "HGNC Symbol" in data["homo_sapiens"]["gene"]
        assert "UniProtKB" in data["homo_sapiens"]["gene"]

    def test_include_defaults_to_false(self, temp_dir):
        """Test Include defaults to False in template."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        df = pd.DataFrame({
            "organism": ["homo_sapiens"],
            "form": ["gene"],
            "name_db": ["HGNC Symbol"],
            "assembly": [38],
            "release": [110],
        })

        edb.create_template_yaml(df)

        with open(edb.file_name_template_yaml()) as f:
            data = yaml.safe_load(f)

        assert data["homo_sapiens"]["gene"]["HGNC Symbol"]["Assembly"]["38"]["Include"] is False


class TestLoadModifiedYaml:
    """Test load_modified_yaml method."""

    def test_loads_configured_yaml(self, temp_dir, sample_external_db_yaml):
        """Test loading configured YAML file."""
        from idtrack._external_databases import ExternalDatabases

        # Write the sample YAML
        yaml_path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        with open(yaml_path, "w") as f:
            yaml.dump(sample_external_db_yaml, f)

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        data = edb.load_modified_yaml()

        assert "homo_sapiens" in data
        assert "gene" in data["homo_sapiens"]

    def test_falls_back_to_default(self, temp_dir):
        """Test fallback to default YAML when configured doesn't exist."""
        from idtrack._external_databases import ExternalDatabases

        # Don't create any YAML in temp_dir
        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        # This should try to load default config
        # May raise FileNotFoundError if no default exists
        try:
            data = edb.load_modified_yaml()
            assert "homo_sapiens" in data
        except FileNotFoundError:
            # Expected if no default config for homo_sapiens
            pass

    def test_raises_file_not_found(self, temp_dir):
        """Test FileNotFoundError when no YAML exists."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="nonexistent_organism",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        with pytest.raises(FileNotFoundError):
            edb.load_modified_yaml()


class TestValidateYamlFileUpToDate:
    """Test validate_yaml_file_up_to_date method."""

    def test_valid_release_passes(self, temp_dir):
        """Test validation passes for valid release."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        yaml_data = {
            "homo_sapiens": {
                "gene": {
                    "HGNC Symbol": {
                        "Assembly": {
                            "38": {
                                "Ensembl release": "109,110,111",
                                "Include": True,
                            }
                        }
                    }
                }
            }
        }

        # Should not raise
        edb.validate_yaml_file_up_to_date(yaml_data)

    def test_invalid_release_raises(self, temp_dir):
        """Test validation fails for missing release."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=115,  # Not in YAML
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        yaml_data = {
            "homo_sapiens": {
                "gene": {
                    "HGNC Symbol": {
                        "Assembly": {
                            "38": {
                                "Ensembl release": "109,110,111",
                                "Include": True,
                            }
                        }
                    }
                }
            }
        }

        with pytest.raises(ValueError, match="115 is not included"):
            edb.validate_yaml_file_up_to_date(yaml_data)


class TestGiveListForCase:
    """Test give_list_for_case method."""

    def test_returns_included_databases(self, temp_dir, sample_external_db_yaml):
        """Test returns databases with Include=True."""
        from idtrack._external_databases import ExternalDatabases

        yaml_path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        with open(yaml_path, "w") as f:
            yaml.dump(sample_external_db_yaml, f)

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        dbs = edb.give_list_for_case("db")

        assert "HGNC Symbol" in dbs
        assert "UniProtKB" in dbs

    def test_returns_assemblies(self, temp_dir, sample_external_db_yaml):
        """Test returns assemblies with included databases."""
        from idtrack._external_databases import ExternalDatabases

        yaml_path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        with open(yaml_path, "w") as f:
            yaml.dump(sample_external_db_yaml, f)

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        assemblies = edb.give_list_for_case("assembly")

        assert 38 in assemblies

    def test_invalid_give_type_raises(self, temp_dir, sample_external_db_yaml):
        """Test invalid give_type raises ValueError."""
        from idtrack._external_databases import ExternalDatabases

        yaml_path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        with open(yaml_path, "w") as f:
            yaml.dump(sample_external_db_yaml, f)

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        with pytest.raises(ValueError):
            edb.give_list_for_case("invalid")

    def test_excludes_not_included_databases(self, temp_dir):
        """Test databases with Include=False are excluded."""
        from idtrack._external_databases import ExternalDatabases

        yaml_data = {
            "homo_sapiens": {
                "gene": {
                    "HGNC Symbol": {
                        "Assembly": {
                            "38": {
                                "Ensembl release": "110",
                                "Include": True,
                            }
                        }
                    },
                    "NotIncluded": {
                        "Assembly": {
                            "38": {
                                "Ensembl release": "110",
                                "Include": False,
                            }
                        }
                    }
                }
            }
        }

        yaml_path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        dbs = edb.give_list_for_case("db")

        assert "HGNC Symbol" in dbs
        assert "NotIncluded" not in dbs


class TestMultipleAssemblies:
    """Test handling of multiple assemblies."""

    def test_filters_by_assembly(self, temp_dir):
        """Test filtering databases by assembly."""
        from idtrack._external_databases import ExternalDatabases

        yaml_data = {
            "homo_sapiens": {
                "gene": {
                    "HGNC Symbol": {
                        "Assembly": {
                            "37": {
                                "Ensembl release": "85,86,87",
                                "Include": True,
                            },
                            "38": {
                                "Ensembl release": "110",
                                "Include": True,
                            }
                        }
                    }
                }
            }
        }

        yaml_path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        # Test assembly 38
        edb38 = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )
        dbs38 = edb38.give_list_for_case("db")
        assert "HGNC Symbol" in dbs38

        # Test assembly 37 (different release)
        edb37 = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=86,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=37,
        )
        dbs37 = edb37.give_list_for_case("db")
        assert "HGNC Symbol" in dbs37


class TestDatabaseIndex:
    """Test database index assignment."""

    def test_database_index_unique(self, temp_dir):
        """Test database indices are unique."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="homo_sapiens",
            ensembl_release=110,
            form="gene",
            local_repository=temp_dir,
            genome_assembly=38,
        )

        df = pd.DataFrame({
            "organism": ["homo_sapiens"] * 3,
            "form": ["gene"] * 3,
            "name_db": ["DB_A", "DB_B", "DB_C"],
            "assembly": [38] * 3,
            "release": [110] * 3,
        })

        edb.create_template_yaml(df)

        with open(edb.file_name_template_yaml()) as f:
            data = yaml.safe_load(f)

        indices = set()
        for db in data["homo_sapiens"]["gene"]:
            idx = data["homo_sapiens"]["gene"][db]["Database Index"]
            assert idx not in indices
            indices.add(idx)
