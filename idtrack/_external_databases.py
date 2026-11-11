#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Where the documented recipe for making an externals YAML lives (shown in error messages).
_EXTERNALS_RECIPE = (
    "the tutorial 'External database configuration' (docs/_notebooks/02_prepare_new_external_yaml.ipynb) and "
    "'Adding organism support' in the contributing guide"
)


class ExternalsReleaseNotCoveredError(ValueError):
    """Raised when an externals YAML file has no entry for the requested Ensembl release."""


def _yaml_scalar(value: Any) -> str:
    """Format a scalar the way the shipped externals YAML files do (plain when unambiguous, else double-quoted)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    if text and text == text.strip() and "\n" not in text:
        try:
            if yaml.safe_load(f"k: {text}") == {"k": text}:
                return text
        except yaml.YAMLError:
            pass
    return json.dumps(text, ensure_ascii=False)


def _yaml_key(key: Any) -> str:
    """Format a mapping key like :py:func:`_yaml_scalar`, checking that it reads back as the same key."""
    text = str(key)
    if text and text == text.strip() and "\n" not in text:
        try:
            if yaml.safe_load(f"{text}: 0") == {text: 0}:
                return text
        except yaml.YAMLError:
            pass
    return json.dumps(text, ensure_ascii=False)


def write_externals_yaml(data: dict, path: str) -> None:
    """Write an externals YAML file in the layout of the files shipped in ``idtrack/default_config``.

    The layout is block style with four-space indentation, the keys of every mapping sorted (as
    :py:func:`yaml.dump` sorts them), assembly keys and other strings that would read back as another type
    double-quoted (``"38"``, ``""``), and release lists plain (``Ensembl release: 90,91,92``). Loading the result
    with :py:func:`yaml.safe_load` gives back *data*.

    Args:
        data: Parsed externals YAML, ``{organism: {form: {database: entry}}}``, each entry holding ``Assembly``,
            ``Database Index`` and ``Potential Synonymous``.
        path: Target file path (overwritten).

    Raises:
        ValueError: If the written text would not read back as *data* (a guard on the formatter itself).
    """
    lines: list[str] = []

    def emit(node: Any, depth: int) -> None:
        indent = "    " * depth
        for key, value in sorted(node.items(), key=lambda kv: str(kv[0])):
            if isinstance(value, dict):
                lines.append(f"{indent}{_yaml_key(key)}:")
                emit(value, depth + 1)
            else:
                lines.append(f"{indent}{_yaml_key(key)}: {_yaml_scalar(value)}")

    emit(data, 0)
    text = "\n".join(lines) + "\n"
    if yaml.safe_load(text) != data:  # pragma: no cover - guards the formatter itself
        raise ValueError(f"Internal error: the externals YAML written to {path!r} does not read back unchanged.")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


class ExternalDatabases:
    """Manage third-party metadata for Ensembl entities through YAML side-car files.

    This helper encapsulates everything related to the *external* (i.e. non-Ensembl) databases that
    can be linked to a given organism, genome assembly, release, and biological form
    (*gene* / *transcript* / *translation*).  Examples of such resources include ArrayExpress,
    RefSeq, Uniprot, HGNC, and dozens of smaller annotation providers.  Rather than hard-coding
    those relationships, the wider *ID-Track* toolkit stores them in a human-readable YAML file that
    lives next to the local data cache managed by :py:class:`_database_manager.DatabaseManager`.

    The YAML workflow is:

    1. :py:meth:`create_template_yaml` enumerates every known combination and writes a *template*
       where each entry is marked ``Include: false``.
    2. A user (or an automated post-processing step) reviews the template, toggling ``Include`` to
       ``true`` for the resources they need.
    3. The modified YAML is saved under *local_repository*; subsequent calls to
       :py:meth:`load_modified_yaml` return it as a plain :py:class:`dict` for downstream logic.
    4. :py:meth:`validate_yaml_file_up_to_date` warns if the user file lags behind a newer template
       (e.g. because a later Ensembl release introduced extra tables).
    5. Utility helpers such as :py:meth:`give_list_for_case` expose convenient filtered views—e.g.
       *all databases that should be downloaded for the current form*, or *all releases supported by
       assembly 38*.

    In short, *ExternalDatabases* provides a single, version-controlled “contract” describing which
    third-party tables belong in an ID-track run, while granting users explicit opt-in control over
    optional resources.
    """

    def __init__(
        self,
        organism: str,
        ensembl_release: int,
        form: str,
        local_repository: str,
        genome_assembly: int,
    ):
        """Instantiate a YAML controller tied to a specific organism, release, and assembly.

        The constructor mirrors the core configuration of :py:class:`_database_manager.DatabaseManager`
        so that both objects operate on the exact same coordinate system.  No I/O is performed at
        construction time; paths are merely recorded, and loggers are configured.  Heavy-weight
        actions—such as scanning the cache for existing YAMLs or writing new ones—happen lazily when
        the corresponding methods are called.

        Args:
            organism (str): Canonical Ensembl species identifier in *snake_case*
                (e.g. ``"homo_sapiens"``), exactly as in :py:attr:`idtrack.DB.supported_organisms`.
            ensembl_release (int): Target Ensembl release number (e.g. ``110``). Must correspond to a
                release that actually exists for *organism* and *genome_assembly*.
            form (str): Entity level—``"gene"``, ``"transcript"``, or ``"translation"``.  Any other
                value raises :py:class:`ValueError` in higher-level validation.
            local_repository (str): Writable directory where YAML files and downloads are cached.
                The directory need not pre-exist; if missing, most public methods will attempt to create it.
            genome_assembly (int): Assembly key of :py:attr:`idtrack.DB.assembly_mysqlport_priority`
                (e.g. ``38`` = human GRCh38, ``37`` = human GRCh37, ``39`` = mouse GRCm39, ``111`` = pig Sscrofa11.1,
                ``10`` = rhesus Mmul_10). The YAML's assembly keys are these keys. Used to disambiguate multiple
                assemblies available for the same *organism*/*release* pair.
        """
        # The logger for informing the user about the progress.
        self.log = logging.getLogger("external_databases")

        # Instance attributes
        self.local_repository = local_repository
        self.organism = organism
        self.form = form
        self.ensembl_release = ensembl_release
        self.genome_assembly = genome_assembly

    def create_template_yaml(self, df: pd.DataFrame):
        """Generate a template YAML enumerating external-database options.

        This helper scans *df*—typically the dataframe returned by
        :py:meth:`idtrack._database_manager.DatabaseManager.create_database_content`—and
        writes a scaffold configuration file to
        :py:meth:`~ExternalDatabases.file_name_template_yaml`.
        The file lists every *organism* → *form* → database combination observed in
        *df*, grouped by genome assembly and Ensembl release.  For each entry the
        template records whether the database should be *included* when building an
        ID-history graph, its integer **Database Index**, and an empty
        **Potential Synonymous** placeholder that future versions may use to flag
        overlapping resources.

        Users are **expected to edit** the generated file—changing ``Include`` from
        ``false`` to ``true`` where appropriate—and **rename** it by appending
        ``_modified`` to the filename before the package will load it.  A warning to
        that effect is emitted via :py:meth:`logging.Logger.warning`.

        The resulting YAML resembles the structure below (truncated for brevity):

        .. code-block:: yaml

            homo_sapiens:
                gene:
                    ArrayExpress:
                        Assembly:
                            "37":
                                Ensembl release: 79,80,81,82,83,84,85,86,87,88,89
                                Include: false
                            "38":
                                Ensembl release: 79,80,81,82,83,84,85,86,87,88,89
                                Include: false
                        Database Index: 0
                        Potential Synonymous: ""
                    Clone-based (Ensembl):
                        Assembly:
                            "37":
                                Ensembl release: 79,80,81,82,83,84,85
                                Include: false
                            "38":
                                Ensembl release: 79,80,81,82,83,84,85
                                Include: false
                        Database Index: 5
                        Potential Synonymous: ""

        **Editing guidelines**

        * Set ``Include`` to *true* for every assembly of the databases you need.
        * Save the edited file with ``_modified`` appended to the base name so that
          downstream routines load the customised version.

        Args:
            df (pandas.DataFrame): Dataframe containing at least the columns
                ``["organism", "form", "name_db", "assembly", "release"]``.  It
                should be produced by
                :py:meth:`idtrack._database_manager.DatabaseManager.create_database_content`
                so that the expected schema is guaranteed.

        Raises:
            ValueError: If *df* contains duplicate *assembly* entries for the same
                organism/form/database triple, causing an internal consistency check
                to fail.

        Notes:
            The **Potential Synonymous** is now all empty. In the following versions, it is aimed to integrate
            a feature that prevent to heve synonymous databases in the list. Likewise, `Database Index` has now no
            use case, in the program.
            It is important to follow the final warning raised by the method. ''Please edit the file based on
            requested external databases and add '_modified' to the file name.''. The editing should be done by
            converting `Include` sections from `false` to `true`. It is recommended to make the change for each
            assembly for a given database.
        """

        def list_to_str(iterable):
            return ",".join(map(str, iterable))

        r: dict[str, dict] = dict()
        database_id = {item: i for i, item in enumerate(sorted(np.unique(df["name_db"])))}
        for a1 in sorted(np.unique(df["organism"])):
            df_a1 = df[df["organism"] == a1]
            for a2 in sorted(np.unique(df_a1["form"])):
                df_a2 = df_a1[df_a1["form"] == a2]
                for a3 in sorted(np.unique(df_a2["name_db"])):
                    df_a3 = df_a2[df_a2["name_db"] == a3]
                    for a4 in sorted(np.unique(df_a3["assembly"])):
                        df_a4 = df_a3[df_a3["assembly"] == a4]
                        a4_str = str(a4)

                        if a1 not in r:
                            r[a1] = dict()
                        if a2 not in r[a1]:
                            r[a1][a2] = dict()
                        if a3 not in r[a1][a2]:
                            r[a1][a2][a3] = {
                                "Database Index": database_id[a3],
                                "Potential Synonymous": "",
                            }
                            r[a1][a2][a3]["Assembly"] = dict()
                        if a4_str not in r[a1][a2][a3]["Assembly"]:
                            r[a1][a2][a3]["Assembly"][a4_str] = {
                                "Ensembl release": list_to_str(sorted(np.unique(df_a4["release"]))),
                                "Include": False,
                            }
                        else:
                            raise ValueError(
                                f"Unexpected assembly format: expected int, got {type(a4).__name__} for assembly {a4!r} "
                                f"in database {a1}/{a2}/{a3}."
                            )

        write_externals_yaml(r, self.file_name_template_yaml())

        self.log.warning(
            f"File created on {self.file_name_template_yaml()}\n"
            f"Please edit the file based on requested external databases "
            f"and add '_modified' to the file name. See package documentation for further detail."
        )

    def file_name_template_yaml(self) -> str:
        """Return absolute path to the *template* YAML configuration file.

        A helper that deterministically builds the filename used by
        :py:meth:`ExternalDatabases.create_template_yaml` when it first scaffolds the
        external-database configuration for *organism*.  Centralising the logic here
        keeps every component of *idtrack* that may need the path (tests, CLI tools,
        future maintenance scripts) in perfect sync with a single implementation.  The
        method performs **no** I/O; it merely concatenates
        :py:attr:`ExternalDatabases.local_repository` and the conventional filename
        pattern ``"<organism>_externals_template.yml"`` so callers can decide whether
        to create, read, or overwrite the file.

        Returns:
            str: Absolute path of ``<organism>_externals_template.yml`` located inside
            :py:attr:`ExternalDatabases.local_repository`.
        """
        return os.path.join(self.local_repository, f"{self.organism}_externals_template.yml")

    def file_name_modified_yaml(self, mode: str) -> str:
        """Resolve the path to a *modified* YAML file customised by the user or shipped with the package.

        The method supports two *modes* that map to different storage locations:

        - ``"configured"`` - the user-edited file living in :py:attr:`ExternalDatabases.local_repository`.
        - ``"default"`` - the read-only fallback bundled under ``<package_root>/default_config`` for quick starts
          and unit tests.

        By funnelling every lookup through this routine, higher-level helpers such as
        :py:meth:`ExternalDatabases.load_modified_yaml` remain agnostic about the
        underlying directory structure and can focus on validation and parsing
        instead.

        Args:
            mode (str): Either ``"configured"`` or ``"default"`` selecting the
                corresponding search location.

        Returns:
            str: Absolute path of the requested YAML file.

        Raises:
            ValueError: If *mode* is not one of the recognised values.
        """
        fb = f"{self.organism}_externals_modified.yml"
        if mode == "configured":
            return os.path.join(self.local_repository, fb)

        elif mode == "default":
            return os.path.join(os.path.dirname(__file__), "default_config", fb)

        else:
            raise ValueError(f"Unknown value for 'mode': {mode}.")

    def _modified_yaml_path(self) -> str:
        """Return the path of the modified YAML to use: the local repository's copy, else the shipped default.

        Returns:
            str: Path of the file to read.

        Raises:
            FileNotFoundError: If neither exists; the message names the file to create and how to make it.
        """
        configured = self.file_name_modified_yaml(mode="configured")
        if os.access(configured, os.R_OK):
            return configured
        self.log.warning(f"External database config is not found in provided temp directory: `{configured}`.")

        default = self.file_name_modified_yaml(mode="default")
        if os.access(default, os.R_OK):
            self.log.warning(f"The package uses the default config file for {self.organism}.")
            return default

        raise FileNotFoundError(
            f"No externals YAML file for organism {self.organism!r}: neither `{configured}` (local repository) nor "
            f"`{default}` (shipped with idtrack) exists. Create `{configured}`: make a template with "
            "`DatabaseManager.create_database_content()` and `ExternalDatabases.create_template_yaml()`, set "
            "`Include: true` for the databases to use (e.g. with `idtrack._default_externals.apply_allowlist`) and "
            f"save it under that name. The documented recipe is {_EXTERNALS_RECIPE}."
        )

    def read_modified_yaml(self) -> dict:
        """Read the modified YAML (local repository first, then the shipped default) without release checks.

        Returns:
            dict: Parsed YAML content keyed by ``{organism → form → database → Assembly → {...}}``.

        Raises:
            FileNotFoundError: If neither the *configured* nor the *default* YAML file exists (see
                :py:meth:`load_modified_yaml`).
        """
        with open(self._modified_yaml_path()) as yaml_file:
            return yaml.safe_load(yaml_file) or {}

    def load_modified_yaml(self) -> dict:
        """Load the user-edited or default YAML configuration and verify release compatibility.

        This convenience wrapper searches for the *configured* YAML file first; if it
        does not exist or lacks read permissions a warning is logged and the
        *default* YAML file shipped with the package is tried instead.  Failure to
        locate **either** file aborts with a :py:class:`FileNotFoundError` that names the
        file to create and the documented way to create it.  After loading, the method
        delegates to :py:meth:`ExternalDatabases.validate_yaml_file_up_to_date` to ensure
        that the currently requested Ensembl release is represented in the configuration.

        Returns:
            dict: Parsed YAML content keyed by ``{organism → form → database → Assembly → {...}}``.

        Raises:
            FileNotFoundError: If neither the *configured* nor the *default* YAML file can be accessed.
        """
        y = self.read_modified_yaml()
        self.validate_yaml_file_up_to_date(y)
        return y

    @staticmethod
    def _releases_of(read_yaml_file: dict, organism: str | None = None) -> set[int]:
        """Return every release listed in a parsed externals YAML (for one organism, or for all)."""
        organisms = [organism] if organism is not None else list(read_yaml_file or {})
        return {
            int(e)
            for org in organisms
            for _, j2 in ((read_yaml_file or {}).get(org) or {}).items()
            for _, j3 in (j2 or {}).items()
            for _, j4 in j3["Assembly"].items()
            for e in str(j4["Ensembl release"]).split(",")
        }

    def covered_releases(self) -> set[int]:
        """Return the Ensembl releases that the organism's externals YAML lists (any form, database or assembly).

        Returns:
            set[int]: Releases with at least one entry; empty if the file has no entry for the organism.

        Raises:
            FileNotFoundError: If no externals YAML file exists for the organism (see :py:meth:`load_modified_yaml`).
        """
        return self._releases_of(self.read_modified_yaml(), self.organism)

    def validate_yaml_file_up_to_date(self, read_yaml_file):
        """Assert that the YAML configuration lists the active Ensembl release.

        The external-database mapping evolves with each Ensembl release.  This helper
        extracts the set of releases encoded in *read_yaml_file*—no matter how deeply
        nested—and verifies that :py:attr:`ExternalDatabases.ensembl_release` is
        present.  Triggering an exception here prevents downstream graph-construction
        logic from silently operating on incomplete or outdated metadata, prompting
        users to extend the YAML file (:py:meth:`extend_modified_yaml`) before proceeding.

        A file that exists but has no entry for the organism or form is accepted: the graph is then built without
        external databases for it.

        Args:
            read_yaml_file (dict): Dictionary produced by
                :py:meth:`ExternalDatabases.load_modified_yaml` containing the loaded YAML structure.

        Raises:
            ExternalsReleaseNotCoveredError: If the current Ensembl release is absent from the YAML configuration
                (a :py:class:`ValueError`).
        """
        # Allow "no external databases" runs (a file without an entry for this organism or form).
        if not read_yaml_file or self.organism not in read_yaml_file or self.form not in read_yaml_file[self.organism]:
            return

        if not read_yaml_file[self.organism][self.form]:
            return

        ensembl_releases = self._releases_of(read_yaml_file, self.organism)
        if self.ensembl_release not in ensembl_releases:
            raise ExternalsReleaseNotCoveredError(
                f"The Ensembl release {self.ensembl_release} is not included in any entry of the YAML config file.\n"
                f"Please extend the configuration to this release (`ExternalDatabases.extend_modified_yaml`), or "
                f"create the graph/track object using a supported release (e.g., {max(ensembl_releases)}).\n"
                "You may also report this issue on GitHub so the default configuration can be updated."
            )

    def extend_modified_yaml(
        self, content: pd.DataFrame, output_path: str | None = None, base_path: str | None = None
    ) -> dict[str, Any]:
        """Extend the organism's modified externals YAML to the releases of a database-content table.

        This is the way to bring an existing ``<organism>_externals_modified.yml`` up to new Ensembl releases
        without regenerating and re-curating it. *content* is the output of
        :py:meth:`idtrack._database_manager.DatabaseManager.create_database_content` for the new releases (build the
        manager with ``ignore_before`` and ``ignore_after`` set to them). For every ``(form, database, assembly)``
        that *content* holds:

        * an existing entry gets the new releases added to its ``Ensembl release`` list, and keeps its ``Include``
          flag;
        * a database that the file does not have yet is added with ``Include: false`` and reported under
          ``"new_databases"``, so that a person can decide whether to enable it;
        * a known database that appears with a new assembly gets an entry for that assembly. Its ``Include`` flag is
          ``true`` only if the database is ``Include: true`` on every assembly it already has (the database is then
          enabled as a whole, as in the shipped files); otherwise it is ``false``. All such entries are reported
          under ``"new_assembly_entries"``, and the ones switched on, which no person has reviewed, are listed again
          under ``"switched_on"`` and in a warning.

        The base file is *base_path* if given, else the local repository's copy if present, else the shipped
        default. The result is written to *output_path* (default: ``<local_repository>/<organism>_externals_modified.yml``),
        in the layout of the shipped files. Passing a template as both *base_path* and *output_path* extends the
        template the same way (all its flags are ``false``).

        Args:
            content: Frame with at least the columns ``organism``, ``form``, ``name_db``, ``assembly`` and
                ``release``; rows of other organisms are ignored.
            output_path: Where to write the extended file.
            base_path: File to extend instead of the modified YAML that :py:meth:`load_modified_yaml` would read.

        Returns:
            dict[str, Any]: Report with keys ``"path"`` (file written), ``"releases"`` (releases of *content*),
            ``"extended_entries"`` (number of existing entries that gained releases), ``"new_databases"`` (list of
            ``(form, database, assembly)``), ``"new_assembly_entries"`` (list of
            ``(form, database, assembly, include)``) and ``"switched_on"`` (list of ``(form, database, assembly)``
            of the new assembly entries added with ``Include: true``).

        Raises:
            ValueError: If *content* lacks a required column or has no row for the organism.
        """
        required = {"organism", "form", "name_db", "assembly", "release"}
        missing_cols = sorted(required - set(content.columns))
        if missing_cols:
            raise ValueError(
                f"`content` lacks the columns {missing_cols} (expected the output of create_database_content)."
            )
        rows = content[content["organism"] == self.organism]
        if rows.empty:
            raise ValueError(f"`content` has no row for organism {self.organism!r}.")

        if base_path is not None:
            with open(base_path) as yaml_file:
                y = yaml.safe_load(yaml_file) or {}
        else:
            y = self.read_modified_yaml()
        org = y.setdefault(self.organism, {})
        indexes = [int(entry.get("Database Index", -1)) for dbs in org.values() for entry in (dbs or {}).values()]
        next_index = (max(indexes) + 1) if indexes else 0

        extended = 0
        new_databases: list[tuple[str, str, int]] = []
        new_assembly_entries: list[tuple[str, str, int, bool]] = []
        grouped = rows.groupby(["form", "name_db", "assembly"], sort=True)["release"]
        new_db_entries: dict[tuple[str, str], dict] = {}
        for (form, db_name, assembly), releases in grouped:
            form, db_name, asm_key = str(form), str(db_name), str(int(assembly))
            rels = {int(r) for r in releases}
            if org.get(form) is None:
                org[form] = {}
            dbs = org[form]
            if db_name not in dbs:
                entry = new_db_entries.setdefault((form, db_name), {"Assembly": {}, "Potential Synonymous": ""})
                entry["Assembly"][asm_key] = {"Ensembl release": ",".join(map(str, sorted(rels))), "Include": False}
                new_databases.append((form, db_name, int(asm_key)))
                continue
            assemblies = dbs[db_name]["Assembly"]
            if asm_key not in assemblies:
                flags = {bool(v["Include"]) for v in assemblies.values()}
                include = flags == {True}
                assemblies[asm_key] = {"Ensembl release": ",".join(map(str, sorted(rels))), "Include": include}
                new_assembly_entries.append((form, db_name, int(asm_key), include))
                continue
            current = {int(r) for r in str(assemblies[asm_key]["Ensembl release"]).split(",")}
            if not rels <= current:
                assemblies[asm_key]["Ensembl release"] = ",".join(map(str, sorted(current | rels)))
                extended += 1

        for (form, db_name), entry in sorted(new_db_entries.items()):
            org[form][db_name] = {
                "Assembly": entry["Assembly"],
                "Database Index": next_index,
                "Potential Synonymous": "",
            }
            next_index += 1

        path = output_path or self.file_name_modified_yaml(mode="configured")
        write_externals_yaml(y, path)

        report: dict[str, Any] = {
            "path": path,
            "releases": sorted({int(r) for r in rows["release"]}),
            "extended_entries": extended,
            "new_databases": new_databases,
            "new_assembly_entries": new_assembly_entries,
            "switched_on": [(form, db_name, asm) for form, db_name, asm, include in new_assembly_entries if include],
        }
        self.log.info(
            f"Externals YAML for {self.organism} extended to releases {report['releases']}: {extended} entries gained "
            f"releases; written to `{path}`."
        )
        if new_databases:
            self.log.warning(
                f"{len(new_databases)} new (form, database, assembly) entries were added with `Include: false`; "
                f"review them and enable the ones to use: {new_databases}"
            )
        if new_assembly_entries:
            self.log.warning(f"Known databases appeared with new assemblies: {new_assembly_entries}")
        if report["switched_on"]:
            self.log.warning(
                "These new assembly entries were added with `Include: true` because their database is enabled on all "
                f"its other assemblies; review them: {report['switched_on']}"
            )
        return report

    def give_list_for_case(self, give_type: str) -> list:
        """Return database names or assembly codes extracted from the external-DB YAML file.

        The helper provides a lightweight way for higher-level components (e.g.
        :py:class:`~idtrack._database_manager.DatabaseManager`) to discover which
        external resources—or which genome assemblies—are currently *eligible*
        according to the user-editable YAML configuration created by
        :py:meth:`ExternalDatabases.create_template_yaml`.  Instead of forcing the
        caller to parse the YAML structure manually, the method filters the entries
        for the manager's *organism*, *form*, *Ensembl release* and
        *genome assembly* and returns the requested slice.

        Args:
            give_type (str): Kind of list to return.  Accepted values are

                - ``"db"``
                  external-database **names** (``str``) whose
                  ``Include`` flag is ``true`` for the current organism, form,
                  assembly and Ensembl release.

                - ``"assembly"``
                  genome-assembly **codes** (``int``) for which the YAML enables
                  *at least one* external database (``Include: true``) at the
                  current Ensembl release.

        Returns:
            list[str] | list[int]:
                - When *give_type* is ``"db"``, a list of database names.
                - When *give_type* is ``"assembly"``, a list of assembly codes.

        Raises:
            ValueError: If *give_type* is not ``"db"`` nor ``"assembly"`` or if an
                unexpected internal inconsistency is encountered while traversing
                the YAML structure.
        """
        if give_type not in ("db", "assembly"):
            raise ValueError(f"give_type must be 'db' or 'assembly', got {give_type!r}.")
        the_dict_loaded = self.load_modified_yaml()
        # A file without an entry for this organism or form means "no external databases" (see
        # `validate_yaml_file_up_to_date`).
        the_dict = (the_dict_loaded.get(self.organism) or {}).get(self.form) or {}

        result = set()
        for db_name in the_dict:
            for asm in the_dict[db_name]["Assembly"]:
                item = the_dict[db_name]["Assembly"][asm]
                releases = {int(r) for r in str(item["Ensembl release"]).split(",")}
                if self.ensembl_release not in releases or not item["Include"]:
                    continue
                if give_type == "db":
                    # Only add db_name if assembly matches
                    if int(asm) == self.genome_assembly:
                        result.add(db_name)
                else:
                    result.add(int(asm))

        return list(result)
