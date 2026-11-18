"""Unit tests of the bottom layer.

The graph's vocabulary (`idtrack.ids`), the organism table's shape (`idtrack.organisms`), the settings
(`idtrack.settings`) and the error hierarchy (`idtrack.errors`).
"""

import dataclasses
import pickle  # noqa: S403 - the tests check that errors survive pickling

import pytest

from idtrack import errors, ids, organisms, settings
from idtrack.build.species import SPECIES_PATH


class TestEnsemblSettings:
    """How the build stack reaches Ensembl."""

    def test_timeout_values_are_positive(self):
        """Both Ensembl timeouts are positive."""
        assert settings.ENSEMBL.connection_timeout > 0
        assert settings.ENSEMBL.reading_timeout > 0

    def test_https_host_is_valid_host(self):
        """The HTTPS host is a string naming an Ensembl server."""
        assert isinstance(settings.ENSEMBL.https_host, str)
        assert "ensembl" in settings.ENSEMBL.https_host.lower()

    def test_rest_server_is_https(self):
        """The REST server is reached over HTTPS."""
        assert settings.ENSEMBL.rest_server.startswith("https://")

    def test_species_path_is_a_path(self):
        """The species path on the HTTPS server is absolute."""
        assert SPECIES_PATH.startswith("/")

    def test_mysql_host_is_valid(self):
        """The MySQL host is a non-empty string."""
        assert isinstance(settings.ENSEMBL.mysql_host, str)
        assert len(settings.ENSEMBL.mysql_host) > 0

    def test_mysql_user_is_anonymous(self):
        """The MySQL user is Ensembl's anonymous account."""
        assert settings.ENSEMBL.mysql_user == "anonymous"

    def test_configure_checks_every_value_before_changing_any(self, monkeypatch):
        """A wrong value leaves every setting as it was; a right one is set."""
        monkeypatch.setattr(settings.ENSEMBL, "reading_timeout", settings.ENSEMBL.reading_timeout)
        monkeypatch.setattr(settings.ENSEMBL, "access", settings.ENSEMBL.access)
        before = dataclasses.asdict(settings.ENSEMBL)
        with pytest.raises(errors.InvalidArgumentError, match="access"):
            settings.configure_ensembl(reading_timeout=99, access="mysql-only")
        with pytest.raises(errors.InvalidArgumentError, match="Unknown Ensembl settings"):
            settings.configure_ensembl(no_such_setting=1)
        with pytest.raises(errors.ArgumentTypeError):
            settings.configure_ensembl(rest_attempts=1.5)
        with pytest.raises(errors.InvalidArgumentError):
            settings.configure_ensembl(reading_timeout=0)
        assert dataclasses.asdict(settings.ENSEMBL) == before
        assert settings.configure_ensembl(reading_timeout=120, access="https") is settings.ENSEMBL
        assert settings.ENSEMBL.reading_timeout == 120 and settings.ENSEMBL.access == "https"


class TestSearchAndGraphSettings:
    """Settings that a call or a loaded graph carries, never the process."""

    def test_search_defaults(self):
        """The search settings' defaults are the documented budgets and jump limit."""
        search = settings.SearchSettings()
        assert (search.budget_states, search.budget_seconds, search.bridge_limit) == (200_000, 30.0, 2)
        assert any(name.startswith("Clone_based") for name in search.non_bridging_databases)

    def test_search_settings_are_checked_and_frozen(self):
        """Wrong search settings are rejected; the settings are frozen and `replace` makes a copy."""
        with pytest.raises(errors.InvalidArgumentError):
            settings.SearchSettings(budget_states=0)
        with pytest.raises(errors.ArgumentTypeError):
            settings.SearchSettings(budget_states=True)
        with pytest.raises(errors.InvalidArgumentError):
            settings.SearchSettings(bridge_limit=-1)
        search = settings.SearchSettings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            search.bridge_limit = 3  # type: ignore[misc]
        assert search.replace(budget_seconds=1.0).budget_seconds == 1.0
        assert search.budget_seconds == 30.0

    def test_graph_settings(self):
        """The graph settings' defaults, their check of cache sizes, and their hash."""
        graph = settings.GraphSettings()
        assert graph.hyperconnective_threshold == 5
        assert set(graph.cache_sizes) == {"adjacency", "node", "edges", "summary", "trios", "names"}
        with pytest.raises(errors.InvalidArgumentError):
            settings.GraphSettings(cache_sizes={"node": 1})
        assert hash(graph) == hash(settings.GraphSettings())


class TestOrganismTable:
    """The shape of the organism table."""

    def test_table_structure(self):
        """Every organism's assemblies carry integer ports and an integer priority."""
        assert isinstance(organisms.ORGANISM_TABLE, dict)
        assert len(organisms.ORGANISM_TABLE) >= 1
        for organism, assemblies in organisms.ORGANISM_TABLE.items():
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
        """GRCh38 is human's priority-1 assembly."""
        assert organisms.ORGANISM_TABLE["homo_sapiens"][38]["Priority"] == 1

    def test_assembly_37_exists(self):
        """GRCh37 is in human's table."""
        assert 37 in organisms.ORGANISM_TABLE["homo_sapiens"]

    def test_highest_priority_assembly_per_organism(self):
        """The priority-1 assembly of each organism is the one `assemblies_by_priority` puts first."""
        expected = {
            "homo_sapiens": 38,
            "mus_musculus": 39,
            "sus_scrofa": 111,
            "macaca_mulatta": 10,
            "macaca_fascicularis": 6,
        }
        for organism, assembly in expected.items():
            assert organisms.assemblies_by_priority(organism)[0] == assembly
            assert organisms.ORGANISM_TABLE[organism][assembly]["Priority"] == 1
            assert organisms.primary_assembly(organism, 115) == assembly

    def test_assembly_priorities_are_unique_per_organism(self):
        """Each organism's priorities are unique and start at 1."""
        for _organism, assemblies in organisms.ORGANISM_TABLE.items():
            priorities = sorted(int(cfg["Priority"]) for cfg in assemblies.values())
            assert priorities == sorted(set(priorities))
            assert priorities[0] == 1

    def test_assembly_node_types(self):
        """Every assembly of the table has a node type per form, named after the assembly and form."""
        for assembly in organisms.ASSEMBLY_NODE_TYPES:
            for form in ids.FORMS:
                assert organisms.ASSEMBLY_NODE_TYPES[assembly][form] == f"assembly_{assembly}_ensembl_{form}"
        held = {int(a) for entries in organisms.ORGANISM_TABLE.values() for a in entries}
        assert set(organisms.ASSEMBLY_NODE_TYPES) == held

    def test_resolve_organism_offline(self):
        """Organism names resolve without the network; unknown or unconfigured organisms raise."""
        assert organisms.resolve_organism("homo_sapiens") == "homo_sapiens"
        assert organisms.resolve_organism(" Homo sapiens ") == "homo_sapiens"
        assert organisms.resolve_organism("PIG") == "sus_scrofa"
        with pytest.raises(errors.UnknownOrganismError, match="names no organism"):
            organisms.resolve_organism("danio_rerio")
        with pytest.raises(errors.UnknownOrganismError, match="not configured"):
            organisms.check_organism("danio_rerio")

    def test_primary_assembly_at_the_boundaries_of_the_ranges(self):
        """The last release of an assembly's range is still that assembly's; the next is the next assembly's."""
        assert (organisms.primary_assembly("mus_musculus", 102), organisms.primary_assembly("mus_musculus", 103)) == (
            38,
            39,
        )
        assert (organisms.primary_assembly("sus_scrofa", 89), organisms.primary_assembly("sus_scrofa", 90)) == (
            102,
            111,
        )
        assert organisms.primary_assembly("macaca_mulatta", 97) == 801
        assert organisms.primary_assembly("sus_scrofa", 66) == 9

    def test_primary_assembly_of_a_release_outside_every_range(self):
        """A release that no assembly covers raises `UnknownReleaseError`."""
        with pytest.raises(errors.UnknownReleaseError, match="No assembly"):
            organisms.primary_assembly("sus_scrofa", 50)


class TestVocabulary:
    """Node types, sentinels and the `<id>.<version>` rule."""

    def test_node_type_key(self):
        """The node attribute that holds the node type is `node_type`."""
        assert ids.NODE_TYPE_KEY == "node_type"

    def test_external(self):
        """The node type of external ids is `external`."""
        assert ids.EXTERNAL == "external"

    def test_forms_in_order(self):
        """The Ensembl forms are gene, transcript and translation, in that order."""
        assert list(ids.FORMS) == ["gene", "transcript", "translation"]

    def test_backbone_form_is_gene(self):
        """The backbone form is gene, with its node types."""
        assert ids.BACKBONE_FORM == "gene"
        assert ids.GENE == "ensembl_gene" and ids.BASE_GENE == "base_ensembl_gene"

    def test_ensembl_types(self):
        """The Ensembl node types of each form and their reverse lookup."""
        for form in ids.FORMS:
            assert ids.ENSEMBL_TYPE[form] == f"ensembl_{form}"
            assert ids.BASE_ENSEMBL_TYPE[form] == f"base_ensembl_{form}"
        for form, node_type in ids.ENSEMBL_TYPE.items():
            assert ids.FORM_OF_ENSEMBL_TYPE[node_type] == form

    def test_vocabulary_cannot_be_changed(self):
        """The vocabulary's mappings are read-only."""
        with pytest.raises(TypeError):
            ids.ENSEMBL_TYPE["gene"] = "x"  # type: ignore[index]

    def test_sentinels(self):
        """The synonym prefix and the sentinel versions."""
        assert ids.SYNONYM_PREFIX == "synonym_id::"
        assert ids.VOID == "Void" and ids.RETIRED == "Retired"
        assert ids.SENTINEL_VERSIONS == frozenset({"Void", "Retired"})

    def test_connection_key(self):
        """The edge attribute that holds the connections is `connection`."""
        assert ids.CONNECTION_KEY == "connection"

    def test_id_version_delimiter(self):
        """The delimiter between a stable id and its version is a dot."""
        assert ids.ID_VERSION_DELIMITER == "."

    def test_join_columns_writes_what_join_writes(self):
        """Columns of ids and versions join as `join` joins one pair; a fractional version is refused."""
        import numpy as np

        stable = ["ENSG1", "ENSG2", "ENST3"]
        numeric = np.array([3.0, 12.0, np.nan])
        assert ids.join_columns(stable, numeric) == [ids.join(i, v) for i, v in zip(stable, numeric)]
        assert ids.join_columns(stable, numeric) == ["ENSG1.3", "ENSG2.12", "ENST3"]
        text = np.array(["3", "Retired", ""], dtype=object)
        assert ids.join_columns(stable, text) == ["ENSG1.3", "ENSG2.Retired", "ENST3"]
        with pytest.raises(errors.TableFormatError, match="floating"):
            ids.join_columns(stable, np.array([1.0, 2.5, 3.0]))
        assert ids.versions_prefix("ENSG1") == "ENSG1." and ids.is_versioned_name("ENSG1.3")
        assert not ids.is_versioned_name("ENSG1") and not ids.is_versioned_name("A.B.1")

    def test_join_and_split(self):
        """`join` and `split` of `<id>.<version>` names, and their errors."""
        assert ids.join("ENSG1", 3) == "ENSG1.3"
        assert ids.join("ENSG1", 3.0) == "ENSG1.3"
        assert ids.join("ENSG1", "3") == "ENSG1.3"
        assert ids.join("ENSG1", "Retired") == "ENSG1.Retired"
        for missing in (None, float("nan"), ""):
            assert ids.join("ENST1", missing) == "ENST1"
        with pytest.raises(errors.TableFormatError, match="floating"):  # a version read from a malformed table
            ids.join("ENSG1", 1.5)
        assert ids.split("ENSG1.3") == ("ENSG1", "3")
        with pytest.raises(errors.IntegrityError, match="delimiters"):
            ids.split("ENSG1")
        with pytest.raises(errors.IntegrityError, match="delimiters"):
            ids.split("A.B.1")

    def test_stable_id_of(self):
        """`stable_id_of` strips the synonym prefix and a numeric version only."""
        assert ids.stable_id_of("synonym_id::TP53") == "TP53"
        assert ids.stable_id_of("ENSG00000141510.18") == "ENSG00000141510"
        assert ids.stable_id_of("LRG_321.1") == "LRG_321"
        assert ids.stable_id_of("SLA.7") == "SLA.7"
        # only a numeric version is a version: the sentinel nodes and other suffixes stay
        assert ids.stable_id_of("ENSG00000141510.Void") == "ENSG00000141510.Void"
        assert ids.stable_id_of("ensg00000141510.x1") == "ensg00000141510.x1"


class TestErrors:
    """One hierarchy; each input error is also the built-in error it replaces; errors survive pickling."""

    def test_hierarchy(self):
        """Every error is an `IdtrackError`; input errors are also `ValueError` or `TypeError`."""
        for cls in (
            errors.InputError,
            errors.GraphFileError,
            errors.SearchBudgetExceededError,
            errors.BuildError,
            errors.IntegrityError,
        ):
            assert issubclass(cls, errors.IdtrackError)
        for cls in (
            errors.InvalidArgumentError,
            errors.MissingValueError,
            errors.UnknownOrganismError,
            errors.UnknownReleaseError,
            errors.UnknownAssemblyError,
            errors.UnknownDatabaseError,
            errors.UnknownIdentifierError,
            errors.InactiveAtReleaseError,
            errors.UnknownVersionError,
        ):
            assert issubclass(cls, errors.InputError) and issubclass(cls, ValueError)
        assert issubclass(errors.ArgumentTypeError, errors.InputError) and issubclass(
            errors.ArgumentTypeError, TypeError
        )
        assert issubclass(errors.GraphMismatchError, errors.GraphFileError)
        assert issubclass(errors.EnsemblRestError, errors.BuildError) and issubclass(
            errors.EnsemblRestError, ConnectionError
        )
        assert not issubclass(errors.IntegrityError, errors.InputError)

    def test_errors_keep_their_attributes_through_pickle(self):
        """Errors survive pickling with their message and attributes."""
        made = [
            errors.UnknownDatabaseError("bad", argument="final_database", value="X", allowed=["A", "B"]),
            errors.InactiveAtReleaseError("G.1", 70, [[80, 90]]),
            errors.UnknownVersionError("G.9", ("G.1", "G.2")),
            errors.GraphFileError("no file", "/x.npz"),
            errors.SearchBudgetExceededError("G.1", 10, 1.0, 5, 2.0),
            errors.EnsemblSchemaError("no table", "gene", ("a",)),
        ]
        for error in made:
            back = pickle.loads(pickle.dumps(error))  # noqa: S301 - loads only what the test dumped
            assert type(back) is type(error) and str(back) == str(error)
            assert vars(back) == vars(error)
        assert made[0].allowed == ("A", "B") and made[0].argument == "final_database"
        assert "70" in str(made[1]) and "80-90" in str(made[1])
