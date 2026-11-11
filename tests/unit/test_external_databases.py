#!/usr/bin/env python3
"""Unit tests for idtrack._external_databases module.

Tests the ExternalDatabases class for YAML configuration management.
"""

from __future__ import annotations

import os

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
        df = pd.DataFrame(
            {
                "organism": ["homo_sapiens", "homo_sapiens"],
                "form": ["gene", "gene"],
                "name_db": ["HGNC Symbol", "UniProtKB"],
                "assembly": [38, 38],
                "release": [110, 110],
            }
        )

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

        df = pd.DataFrame(
            {
                "organism": ["homo_sapiens", "homo_sapiens"],
                "form": ["gene", "gene"],
                "name_db": ["HGNC Symbol", "UniProtKB"],
                "assembly": [38, 38],
                "release": [109, 110],
            }
        )

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

        df = pd.DataFrame(
            {
                "organism": ["homo_sapiens"],
                "form": ["gene"],
                "name_db": ["HGNC Symbol"],
                "assembly": [38],
                "release": [110],
            }
        )

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
                    },
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
                            },
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

        df = pd.DataFrame(
            {
                "organism": ["homo_sapiens"] * 3,
                "form": ["gene"] * 3,
                "name_db": ["DB_A", "DB_B", "DB_C"],
                "assembly": [38] * 3,
                "release": [110] * 3,
            }
        )

        edb.create_template_yaml(df)

        with open(edb.file_name_template_yaml()) as f:
            data = yaml.safe_load(f)

        indices = set()
        for db in data["homo_sapiens"]["gene"]:
            idx = data["homo_sapiens"]["gene"][db]["Database Index"]
            assert idx not in indices
            indices.add(idx)


DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "idtrack", "default_config")


class TestMissingYaml:
    """An organism without any externals YAML gets one clear error."""

    def test_message_names_file_and_recipe(self, temp_dir):
        """Message names file and recipe."""
        from idtrack._external_databases import ExternalDatabases

        edb = ExternalDatabases(
            organism="danio_rerio", ensembl_release=110, form="gene", local_repository=temp_dir, genome_assembly=11
        )
        with pytest.raises(FileNotFoundError) as info:
            edb.load_modified_yaml()
        message = str(info.value)
        assert os.path.join(temp_dir, "danio_rerio_externals_modified.yml") in message
        assert "create_template_yaml" in message and "02_prepare_new_external_yaml" in message

    def test_file_without_the_organism_means_no_externals(self, temp_dir):
        """File without the organism means no externals."""
        from idtrack._external_databases import ExternalDatabases

        with open(os.path.join(temp_dir, "danio_rerio_externals_modified.yml"), "w") as handle:
            yaml.safe_dump({"other_organism": {}}, handle)
        edb = ExternalDatabases(
            organism="danio_rerio", ensembl_release=110, form="gene", local_repository=temp_dir, genome_assembly=11
        )
        assert edb.give_list_for_case("db") == []
        assert edb.give_list_for_case("assembly") == []
        assert edb.covered_releases() == set()


class TestShippedFiles:
    """The shipped YAML files, the writer and the allowlists agree."""

    @pytest.mark.parametrize("name", sorted(f for f in os.listdir(DEFAULT_CONFIG_DIR) if f.endswith(".yml")))
    def test_writer_reproduces_shipped_file(self, temp_dir, name):
        """Writer reproduces shipped file."""
        from idtrack._external_databases import write_externals_yaml

        source = os.path.join(DEFAULT_CONFIG_DIR, name)
        with open(source) as handle:
            data = yaml.safe_load(handle)
        target = os.path.join(temp_dir, name)
        write_externals_yaml(data, target)
        with open(source) as a, open(target) as b:
            assert a.read() == b.read()

    @pytest.mark.parametrize(
        "organism",
        sorted(
            f.removesuffix("_externals_template.yml")
            for f in os.listdir(DEFAULT_CONFIG_DIR)
            if f.endswith("_externals_template.yml")
        ),
    )
    def test_modified_file_is_template_plus_allowlist(self, organism):
        """Modified file is template plus allowlist."""
        from idtrack._default_externals import DEFAULT_EXTERNALS, apply_allowlist

        with open(os.path.join(DEFAULT_CONFIG_DIR, f"{organism}_externals_template.yml")) as handle:
            template = yaml.safe_load(handle)
        with open(os.path.join(DEFAULT_CONFIG_DIR, f"{organism}_externals_modified.yml")) as handle:
            modified = yaml.safe_load(handle)
        applied, enabled, _absent = apply_allowlist(template, organism, DEFAULT_EXTERNALS[organism])
        assert enabled > 0
        assert applied == modified

    def test_every_shipped_modified_file_has_a_template_and_allowlist(self):
        """Every shipped modified file has a template and allowlist."""
        from idtrack._db import DB
        from idtrack._default_externals import DEFAULT_EXTERNALS

        modified = {
            f.removesuffix("_externals_modified.yml")
            for f in os.listdir(DEFAULT_CONFIG_DIR)
            if f.endswith("_externals_modified.yml")
        }
        templates = {
            f.removesuffix("_externals_template.yml")
            for f in os.listdir(DEFAULT_CONFIG_DIR)
            if f.endswith("_externals_template.yml")
        }
        assert modified == templates
        assert modified <= set(DEFAULT_EXTERNALS) == set(DB.supported_organisms)

    def test_allowlists_use_configured_assemblies(self):
        """Allowlists use configured assemblies."""
        from idtrack._db import DB
        from idtrack._default_externals import DEFAULT_EXTERNALS

        for organism, allowlist in DEFAULT_EXTERNALS.items():
            configured = set(DB.assembly_mysqlport_priority[organism])
            for databases in allowlist.values():
                for assemblies in databases.values():
                    assert set(assemblies) == configured, organism

    def test_notebook_allowlists_match_the_package(self):
        """Tutorial 02 shows the allowlists as editable dicts; they must equal the package's lists."""
        import ast
        import json
        import re

        from idtrack._default_externals import DEFAULT_EXTERNALS

        notebook = os.path.join(
            os.path.dirname(__file__), "..", "..", "docs", "_notebooks", "02_prepare_new_external_yaml.ipynb"
        )
        with open(notebook, encoding="utf-8") as handle:
            cells = json.load(handle)["cells"]
        names = {
            "DEFAULT_HS_EXTERNALS": "homo_sapiens",
            "DEFAULT_MM_EXTERNALS": "mus_musculus",
            "DEFAULT_SS_EXTERNALS": "sus_scrofa",
            "DEFAULT_MMUL_EXTERNALS": "macaca_mulatta",
            "DEFAULT_MFAS_EXTERNALS": "macaca_fascicularis",
        }
        found = {}
        for cell in cells:
            source = "".join(cell["source"])
            for match in re.finditer(r"^(DEFAULT_\w+_EXTERNALS) = (\{.*?\n\})$", source, flags=re.S | re.M):
                found[match.group(1)] = ast.literal_eval(match.group(2))
        assert set(found) == set(names)
        for variable, organism in names.items():
            assert found[variable] == DEFAULT_EXTERNALS[organism], variable


def _content(rows):
    return pd.DataFrame(rows, columns=["organism", "form", "name_db", "assembly", "release", "count"])


class TestExtendModifiedYaml:
    """`extend_modified_yaml` adds releases, keeps flags and lists new databases."""

    @pytest.fixture
    def base_yaml(self, temp_dir):
        """Write a small modified YAML (human, releases 114-115) into the local repository."""
        data = {
            "homo_sapiens": {
                "gene": {
                    "HGNC Symbol": {
                        "Assembly": {
                            "37": {"Ensembl release": "114,115", "Include": True},
                            "38": {"Ensembl release": "114,115", "Include": True},
                        },
                        "Database Index": 3,
                        "Potential Synonymous": "",
                    },
                    "WikiGene": {
                        "Assembly": {"38": {"Ensembl release": "114,115", "Include": False}},
                        "Database Index": 7,
                        "Potential Synonymous": "",
                    },
                },
                "transcript": {
                    "RefSeq_mRNA": {
                        "Assembly": {"38": {"Ensembl release": "115", "Include": True}},
                        "Database Index": 5,
                        "Potential Synonymous": "",
                    }
                },
            }
        }
        path = os.path.join(temp_dir, "homo_sapiens_externals_modified.yml")
        with open(path, "w") as handle:
            yaml.safe_dump(data, handle)
        return path

    def _edb(self, temp_dir, release=116):
        from idtrack._external_databases import ExternalDatabases

        return ExternalDatabases(
            organism="homo_sapiens", ensembl_release=release, form="gene", local_repository=temp_dir, genome_assembly=38
        )

    def test_adds_release_keeps_flags_lists_new(self, temp_dir, base_yaml):
        """Adds release keeps flags lists new."""
        edb = self._edb(temp_dir)
        content = _content(
            [
                ("homo_sapiens", "gene", "HGNC Symbol", 38, 116, 10),
                ("homo_sapiens", "gene", "HGNC Symbol", 37, 116, 9),
                ("homo_sapiens", "gene", "WikiGene", 38, 116, 5),
                ("homo_sapiens", "gene", "WikiGene", 37, 116, 5),
                ("homo_sapiens", "gene", "Brand-new DB", 38, 116, 1),
                ("homo_sapiens", "transcript", "RefSeq_mRNA", 38, 116, 4),
                ("homo_sapiens", "transcript", "RefSeq_mRNA", 39, 116, 4),
                ("mus_musculus", "gene", "MGI Symbol", 39, 116, 4),
            ]
        )
        report = edb.extend_modified_yaml(content)
        assert report["path"] == base_yaml
        assert report["releases"] == [116]
        assert report["extended_entries"] == 4
        assert report["new_databases"] == [("gene", "Brand-new DB", 38)]
        assert report["new_assembly_entries"] == [
            ("gene", "WikiGene", 37, False),
            ("transcript", "RefSeq_mRNA", 39, True),
        ]
        # Only a database enabled on all its assemblies is switched on for a new assembly, and it is listed.
        assert report["switched_on"] == [("transcript", "RefSeq_mRNA", 39)]

        with open(base_yaml) as handle:
            y = yaml.safe_load(handle)["homo_sapiens"]
        assert y["gene"]["HGNC Symbol"]["Assembly"]["38"] == {"Ensembl release": "114,115,116", "Include": True}
        assert y["gene"]["HGNC Symbol"]["Assembly"]["37"]["Ensembl release"] == "114,115,116"
        assert y["gene"]["WikiGene"]["Assembly"]["38"] == {"Ensembl release": "114,115,116", "Include": False}
        assert y["gene"]["Brand-new DB"]["Assembly"]["38"] == {"Ensembl release": "116", "Include": False}
        assert y["gene"]["Brand-new DB"]["Database Index"] == 8
        assert y["transcript"]["RefSeq_mRNA"]["Assembly"]["39"]["Include"] is True
        assert list(y["gene"]) == sorted(y["gene"])

        # The extended file now validates for 116, and a re-run adds nothing.
        assert edb.load_modified_yaml()["homo_sapiens"]
        assert edb.covered_releases() == {114, 115, 116}
        assert self._edb(temp_dir).extend_modified_yaml(content)["extended_entries"] == 0

    def test_writes_to_output_path_and_rejects_bad_content(self, temp_dir, base_yaml):
        """Writes to output path and rejects bad content."""
        edb = self._edb(temp_dir)
        out = os.path.join(temp_dir, "elsewhere.yml")
        edb.extend_modified_yaml(_content([("homo_sapiens", "gene", "HGNC Symbol", 38, 116, 1)]), output_path=out)
        assert os.path.exists(out)
        with open(base_yaml) as handle:
            assert "116" not in handle.read()
        with pytest.raises(ValueError, match="no row for organism"):
            edb.extend_modified_yaml(_content([("mus_musculus", "gene", "MGI Symbol", 39, 116, 1)]))
        with pytest.raises(ValueError, match="lacks the columns"):
            edb.extend_modified_yaml(pd.DataFrame({"organism": ["homo_sapiens"]}))

    def test_missing_release_error_mentions_extension(self, temp_dir, base_yaml):
        """Missing release error mentions extension."""
        from idtrack._external_databases import ExternalsReleaseNotCoveredError

        with pytest.raises(ExternalsReleaseNotCoveredError, match="extend_modified_yaml"):
            self._edb(temp_dir, release=116).load_modified_yaml()
