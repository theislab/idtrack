=========
Reference
=========

Every public class, function and constant of idtrack, one page per module. Names without a leading underscore are
public. The names a user needs most are importable from ``idtrack`` itself (``from idtrack import API``); they are
documented under the module that holds them.

.. automodule:: idtrack


The public API at a glance
--------------------------

**Loading, building and converting** (:doc:`api`): :py:class:`idtrack.API`.

.. list-table::
    :header-rows: 1
    :widths: 36 64

    * - Method
      - Does
    * - :py:meth:`~idtrack.API.build`, :py:meth:`~idtrack.API.load`, :py:meth:`~idtrack.API.graph_path`
      - Build a graph (or load its file from the local repository); load a graph file; the path of a graph's file.
    * - :py:meth:`~idtrack.API.convert`, :py:meth:`~idtrack.API.convert_many`
      - Convert one identifier, or a batch.
    * - :py:meth:`~idtrack.API.classify`, :py:meth:`~idtrack.API.classification_summary`,
        :py:meth:`~idtrack.API.reason_text`
      - Bin a batch's results (1→0, 1→1, 1→n), count them, and give the sentence of a result's reason.
    * - :py:meth:`~idtrack.API.source_inference`, :py:meth:`~idtrack.API.infer_source`
      - Say which release, assembly or database a list comes from.
    * - :py:meth:`~idtrack.API.genes_named`, :py:meth:`~idtrack.API.names_of`, :py:meth:`~idtrack.API.gene_to_database`
      - Which genes a name names; the names of a gene; a gene's ids in a database.
    * - :py:meth:`~idtrack.API.describe`, :py:meth:`~idtrack.API.releases`, :py:meth:`~idtrack.API.external_databases`,
        :py:meth:`~idtrack.API.final_databases`, :py:meth:`~idtrack.API.databases_by_assembly`,
        :py:meth:`~idtrack.API.database_forms`
      - What a graph holds.
    * - :py:meth:`~idtrack.API.ensembl_organism`, :py:meth:`~idtrack.API.database_manager`,
        :py:meth:`~idtrack.API.configure_logger`
      - Ensembl's name and newest release of an organism (network); the manager of a graph's tables; logging.

**Results** (:doc:`results`): :py:class:`idtrack.results.ConversionResult` with its
:py:class:`~idtrack.results.Candidate` objects, :py:class:`~idtrack.results.OtherReading`,
:py:class:`~idtrack.results.ErrorInfo` and, with ``explain=True``, :py:class:`~idtrack.results.ExplainedPath`
(:py:class:`~idtrack.results.PathNode`, :py:class:`~idtrack.results.PathStep`,
:py:class:`~idtrack.results.BridgeCheck`); :py:class:`~idtrack.results.NameMatch`,
:py:class:`~idtrack.results.SourceInference`, :py:class:`~idtrack.results.GraphDescription` and
:py:class:`~idtrack.results.AssemblyDescription`. Their vocabularies: :py:data:`~idtrack.results.STATUSES`,
:py:data:`~idtrack.results.REASONS`, :py:data:`~idtrack.results.CRITERIA`, :py:data:`~idtrack.results.MATCH_STEPS`,
:py:data:`~idtrack.results.MATCH_PREFIXES`, :py:data:`~idtrack.results.READING_REASONS`,
:py:data:`~idtrack.results.STEP_REASONS`, :py:data:`~idtrack.results.BRIDGE_RULES` (explained in :doc:`../concepts`).

**Errors** (:doc:`errors`): every error is an :py:class:`idtrack.errors.IdtrackError`; a caller's mistake is an
:py:class:`~idtrack.errors.InputError` that names the argument and the valid values.

**Settings** (:doc:`settings`): :py:class:`idtrack.settings.SearchSettings` (budget and bridges of a search),
:py:class:`~idtrack.settings.GraphSettings` (a loaded graph's caches), :py:func:`~idtrack.settings.configure_ensembl`
(how Ensembl is reached).

**Harmonization** (:doc:`harmonize`): :py:class:`idtrack.harmonize.HarmonizeFeatures` and its pure core.

**The command line**: :doc:`../cli`.


Modules
-------

.. toctree::
   :maxdepth: 1

   api
   results
   errors
   settings
   organisms
   reading
   harmonize
   ids
   graph
   engine
   checks
   build
