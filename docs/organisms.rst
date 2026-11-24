.. _organisms:

=========
Organisms
=========

Which organisms idtrack supports, how their genome assemblies are named, how an organism is named in a call, and how
another organism is added.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


Supported organisms
-------------------

idtrack supports the following organisms. Each is declared once, in the organism table
:py:data:`idtrack.organisms.ORGANISM_TABLE`, and ships a default external-database configuration
(``idtrack/default_config/<organism>_externals_modified.yml``).

.. list-table::
    :header-rows: 1
    :widths: 18 22 40 20

    * - Organism
      - Identifier
      - Assemblies (key: name, Ensembl releases)
      - Symbol databases
    * - Human
      - ``"homo_sapiens"``, ``"human"`` or ``9606``
      - 38: GRCh38, 76+; 37: GRCh37, 55+ (maintained beside GRCh38 from 76); 36: NCBI36, 48-54
      - HGNC Symbol
    * - Mouse
      - ``"mus_musculus"``, ``"mouse"`` or ``10090``
      - 39: GRCm39, 103+; 38: GRCm38, 68-102; 37: NCBIM37, 48-67
      - MGI Symbol
    * - Pig
      - ``"sus_scrofa"``, ``"pig"`` or ``9823``
      - 111: Sscrofa11.1, 90+; 102: Sscrofa10.2, 67-89; 9: Sscrofa9, 56-66
      - VGNC Symbol, HGNC Symbol
    * - Rhesus macaque
      - ``"macaca_mulatta"``, ``"rhesus"`` or ``9544``
      - 10: Mmul_10, 98+; 801: Mmul_8.0.1, 86-97; 1: MMUL_1, 48-85
      - VGNC Symbol, HGNC Symbol
    * - Cynomolgus macaque
      - ``"macaca_fascicularis"``, ``"cynomolgus"`` or ``9541``
      - 6: Macaca_fascicularis_6.0, 103+; 5: Macaca_fascicularis_5.0, 91-102
      - VGNC Symbol, HGNC Symbol

The **key** of an assembly is what idtrack uses everywhere (the ``assembly`` argument of
:py:class:`idtrack.build.DatabaseManager`, labels such as ``assembly_801_ensembl_gene``, the table folders
``idtrack_tables/<organism>_assembly-<key>/``, the graph file names and the assembly keys of the externals YAML). It is
Ensembl's code from the core database name ``<organism>_core_<release>_<code>`` wherever that is unambiguous. Rhesus is
the exception: Ensembl names both MMUL_1 (releases up to 85) and Mmul_10 (from 98)
``macaca_mulatta_core_<release>_10``, so Mmul_10 keeps key ``10`` and MMUL_1 has the key ``1``; idtrack tells them
apart by release. Releases before 48 are not available in idtrack. Except for human (GRCh37 and GRCh38 overlap),
every organism has one assembly per release, and idtrack links the assemblies through Ensembl's own history. A graph
converts on the assembly that Ensembl maintained at each release (human: GRCh38 from 76); GRCh37 ids after release
75 are in the human graph as ids to convert from.

Organism names are matched case-insensitively. Ensembl species names (also written with spaces, e.g.
``"Macaca mulatta"``), taxon IDs, the assembly names of the table (``"GRCm38"``) and the short names in the table
above resolve; a short name declared for an organism wins over Ensembl's other species of that common name (``"pig"``
is ``sus_scrofa``, not one of the pig breeds that Ensembl also lists). Another common name that several Ensembl
species share raises an error listing the candidates.

The symbol databases (:py:data:`idtrack.organisms.SYMBOL_DATABASES`) are the default final database of
:py:class:`idtrack.harmonize.HarmonizeFeatures` (the first one) and the ones to try for gene symbols. In current Ensembl, macaque genes carry VGNC symbols on the newest assemblies
and HGNC symbols on earlier ones, and cynomolgus 6.0 has few symbols of either kind.

idtrack reads Ensembl ids with their versions: an organism whose Ensembl ids carry no versions is not supported (a
graph build refuses it with an error that says so). The five organisms above all have versions.


.. _adding-an-organism:

Adding an organism
------------------

An organism is declared in one place, the organism table ``ORGANISM_TABLE`` in ``idtrack/organisms.py``
(:py:data:`idtrack.organisms.ORGANISM_TABLE`); everything else (supported organisms, assembly labels, assembly priorities) is derived
from it. Rhesus and cynomolgus macaque were added this way. The recipe, for a contributor with a clone of the repository
(:doc:`contributing`):

1. **Establish the assemblies.** List the organism's core databases in Ensembl's dump listings
   (``https://ftp.ensembl.org/pub/release-<n>/mysql/``): each is named
   ``<organism>_core_<release>_<code>[patch letter]``. Read each assembly's name from the ``meta`` table
   (``assembly.default``) and note the first and last release of each code. Check that every release has one
   assembly (a clean handoff) or note where two overlap (as human GRCh37 and GRCh38 do).

2. **Write the table entry.** One entry per assembly, keyed by an integer that is unique within the organism:

   .. code-block:: python

       "macaca_mulatta": {
           10: {"Name": "Mmul_10", "Ports": [3306, 5306], "Priority": 1, "ReleaseRange": [98, None]},
           801: {"Name": "Mmul_8.0.1", "Ports": [3306, 5306], "Priority": 2, "ReleaseRange": [86, 97]},
           1: {"Name": "MMUL_1", "DbSuffix": "10", "Ports": [3306, 5306], "Priority": 3, "ReleaseRange": [48, 85]},
       },

   The key equals Ensembl's code where that is unambiguous. Where Ensembl reuses a code for two assemblies of
   the organism (rhesus ``_10`` is MMUL_1 up to 85 and Mmul_10 from 98), the earlier assembly gets its own key
   and names the shared code in ``DbSuffix``; their ``ReleaseRange`` values must not overlap. ``Ports`` are the
   public MySQL ports in preference order (3306 and 5306 serve release 48 onwards; 3337 is the GRCh37 archive
   and may be listed for human GRCh37 only, see :py:attr:`idtrack.settings.EnsemblSettings.mysql_port_assembly`); ``Priority`` 1 is the preferred, newest
   assembly. Add the organism's symbol databases to ``SYMBOL_DATABASES`` (e.g. ``("VGNC Symbol", "HGNC
   Symbol")``) and its short names to ``ALIASES`` (lower case, unique across organisms, e.g.
   ``("rhesus",)``). The table is validated at import. IDTrack reads Ensembl ids with versions: an organism whose ids
   carry no versions is not supported.

3. **Try it without editing the package.** ``idtrack.organisms.register_organism(name, assemblies,
   symbol_databases, aliases)`` adds the same entry for the running Python session, validates it, and recomputes
   everything derived from the table.

4. **Make the externals file.** Generate the template over every release and form
   (``DatabaseManager.create_database_content()`` and ``ExternalDatabases.create_template_yaml()`` of
   :py:mod:`idtrack.build`; see :doc:`databases` and :doc:`_notebooks/06_prepare_new_external_yaml`) and put it in
   ``idtrack/default_config/`` as ``<organism>_externals_template.yml``. Add the organism's allowlist to
   ``DEFAULT_EXTERNALS`` in ``idtrack/build/default_externals.py`` (and to tutorial 6, :doc:`_notebooks/06_prepare_new_external_yaml`), then write the shipped
   selection, ``<organism>_externals_modified.yml``, from the template, from the root of the repository:

   .. code-block:: python

       import yaml

       from idtrack.build.default_externals import DEFAULT_EXTERNALS, apply_allowlist
       from idtrack.build.externals import write_externals_yaml

       organism = "macaca_mulatta"
       folder = "idtrack/default_config"
       with open(f"{folder}/{organism}_externals_template.yml") as handle:
           template = yaml.safe_load(handle)
       modified, enabled, absent = apply_allowlist(template, organism, DEFAULT_EXTERNALS[organism])
       write_externals_yaml(modified, f"{folder}/{organism}_externals_modified.yml")

   The same code after a change of an allowlist writes the files again; the unit tests check that every shipped
   modified file is what it writes, byte for byte. Both files are shipped: the template is the list of every
   database a user can select from (tutorial 7, :doc:`_notebooks/07_advanced_topics`, builds other selections from it). When Ensembl publishes a new release,
   extend the shipped files with ``ExternalDatabases.extend_modified_yaml`` instead of regenerating them.

5. **Teach the tests.** Add a row per assembly to ``EXPECTED_RANGES`` in
   ``tests/integration/test_ensembl_ftp_fallback.py`` and the organism's defaults (assembly, release window, seed
   genes) to ``REAL_GRAPH_DEFAULTS`` in ``tests/conftest.py``. The unit tests check the table, the allowlist and
   the shipped files; the live MySQL tests check ports and releases where MySQL is reachable.

6. **Document it.** Add the organism to the table of this page and to the tutorials that list organisms.

7. **Validate and submit.** Build a small graph across each assembly handoff, convert known identifiers, and
   open a pull request with the table entry, the externals files, the tests and the docs.
