.. _harmonization:

=====================
Harmonizing Datasets
=====================

:py:class:`idtrack.harmonize.HarmonizeFeatures` converts the features (genes) of several ``.h5ad`` datasets to one
Ensembl release and merges the datasets into one AnnData, without dropping or rewriting anything silently. It needs
the ``harmonize`` extra (anndata and scipy: ``pip install "idtrack[harmonize]"``, :doc:`installation`). Its pure core
(:py:func:`idtrack.harmonize.convert_features`, :py:func:`idtrack.harmonize.feature_table`,
:py:func:`idtrack.harmonize.output_plan`) works on a list of names without files and needs pandas only.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


A small example
---------------

Two toy datasets, one with symbols and an Ensembl id, one with a name that is no gene:

.. code-block:: python

    import anndata as ad
    import numpy as np
    import pandas as pd

    from idtrack.harmonize import HarmonizeFeatures

    for name, genes in {"study_a": ["TP53", "BRCA1", "ENSG00000146648"], "study_b": ["TP53", "KRAS", "NOT_A_GENE"]}.items():
        X = np.arange(10 * len(genes), dtype=np.float32).reshape(10, len(genes))
        cells = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
        ad.AnnData(X=X, obs=cells, var=pd.DataFrame(index=genes)).write_h5ad(f"{name}.h5ad")

    harmonizer = HarmonizeFeatures(
        project_name="my_project",
        data_h5ad_dict={"study_a": "study_a.h5ad", "study_b": "study_b.h5ad"},
        project_local_repository="./harmonization",   # the result cache, one JSON file per dataset
        idtrack_local_repository="./idtrack_cache",   # the local repository that holds the graph
        to_release=115,
        organism="homo_sapiens",
        snapshot_release=115,
        final_database="HGNC Symbol",
        verbosity=0,
    )
    report = harmonizer.conversion_report()        # one row per input feature
    print(report[["dataset", "query_id", "status", "ensembl_gene", "final_id"]].to_string())

    adata = harmonizer.unify_multiple_anndatas(features="union")   # or features="intersect"
    print(adata.var[["converted_id", "idtrack_status_study_a", "idtrack_status_study_b"]].to_string())

.. code-block:: text

       dataset         query_id        status        ensembl_gene final_id
    0  study_a             TP53     converted  ENSG00000141510.20     TP53
    1  study_a            BRCA1     converted  ENSG00000012048.27    BRCA1
    2  study_a  ENSG00000146648     converted  ENSG00000146648.22     EGFR
    3  study_b             TP53     converted  ENSG00000141510.20     TP53
    4  study_b             KRAS     converted  ENSG00000133703.15     KRAS
    5  study_b       NOT_A_GENE  not_in_graph                 NaN      NaN
                       converted_id idtrack_status_study_a    idtrack_status_study_b
    ENSG00000012048.27        BRCA1              converted                       NaN
    ENSG00000133703.15         KRAS                    NaN                 converted
    ENSG00000141510.20         TP53              converted                 converted
    ENSG00000146648.22         EGFR              converted                       NaN
    NOT_A_GENE                  NaN                    NaN  unconverted_not_in_graph

The constructor converts every feature of every dataset (the long part: :doc:`performance`) and caches the results;
the methods after it read them.


The rules of the output
-----------------------

* **One converted gene is one feature,** named by its Ensembl gene id (with version) at the target release. Its name
  in the final database (default: the organism's first symbol database, :py:data:`idtrack.organisms.SYMBOL_DATABASES`)
  is in the column ``converted_id`` of the merged AnnData (the argument ``converted_id_column`` names it) and in the
  column named after the database (``HGNC Symbol``) of each harmonised dataset
  (:py:meth:`idtrack.harmonize.HarmonizeFeatures.feature_harmonizer`); where a gene has several, the others are in
  ``<final database>_alternatives``. A gene without a name there has a missing value in that column and the status
  ``converted_no_final_id``: the final-database column never holds an Ensembl id. Which gene a feature is, is
  decided as Ensembl genes, so a feature whose gene has no name in the final database is that gene, never another
  gene that has one (:ref:`concept-ranking`).
* **Nothing is dropped silently.** Every output feature has ``var["idtrack_status"]`` (per dataset in the merged
  object, ``idtrack_status_<dataset>``; the values of :py:data:`idtrack.harmonize.STATUSES`, below) and
  ``var["idtrack_reason"]``. Unconverted features keep their input names; ``keep_unconverted=False`` (of
  :py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas` and
  :py:meth:`idtrack.harmonize.HarmonizeFeatures.feature_harmonizer`) leaves them out. An unconverted name equal to a
  converted gene id is renamed ``unconverted:<name>``, so converted and unconverted measurements never share a row.
* **Several input features of one dataset on one gene** (an Ensembl merge, or two aliases; ``merge=`` of the same two
  methods): ``merge="sum"`` adds them up (right for counts; status ``merged``); ``merge="flag"`` (default, for
  normalised data, where a sum has no meaning) keeps the feature that names the gene itself as the gene's feature
  and flags the others (``unconverted_merge_partner``).
* **Each dataset is converted on its own:** a dataset is harmonised the same alone and together with others. A
  feature that is a gene id existing at the target release is that gene.
  ``from_release_by_dataset={"study_a": 100}`` reads a dataset's features at the release it was annotated with (a
  feature the graph does not hold there is read where it does, and ``idtrack_reason`` says so);
  ``from_database_by_dataset={"study_a": "HGNC Symbol"}`` reads them as ids of that database (a symbol that another
  database gives to another gene is then that database's gene, not a tie). Nothing is inferred on your behalf:
  without them, a symbol that two databases give to two genes is ``unconverted_ambiguous`` when both genes can be
  named in the final database; when only one of them has a name there, the feature is converted to that gene (a tie
  is judged among the genes that can be named in the final database: :ref:`concept-ties`).
* **Names made unique by single-cell tools** (``SRF.1`` beside ``SRF`` in one dataset) are copies of the name they
  repeat (:ref:`reading-rules`); ``idtrack_reason`` names the feature copied.
* ``api=`` passes an :py:class:`idtrack.API` whose graph is already loaded; without it, the graph is loaded only
  when some dataset is not in the result cache.


Statuses
--------

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - ``var["idtrack_status"]``
      - The feature
    * - ``converted``
      - is one gene at the target release, named in the final database.
    * - ``converted_no_final_id``
      - is one gene that has no id in the final database (``converted_id`` is missing).
    * - ``merged``, ``merged_no_final_id``
      - is the sum of several input features of one gene (``merge="sum"``).
    * - ``unconverted_merge_partner``
      - shares its gene with another input feature that names the gene itself (``merge="flag"``).
    * - ``unconverted_ambiguous``
      - stands for several genes (a tie or a split), each with a name in the final database.
    * - ``unconverted_no_conversion``
      - is in the graph but has no gene at the target release (a retired id).
    * - ``unconverted_not_in_graph``
      - is no identifier of the graph (or an id with a version the graph does not hold).
    * - ``unconverted_missing_value``
      - is a spelling of a missing value (``nan``, ``NA``, ...), which is never a gene.
    * - ``unconverted_error``
      - raised an error that concerns it alone (a search over its budget).

``conversion_report()`` gives the same per input feature in shorter words (``converted``, ``converted_no_final_id``,
``ambiguous``, ``no_conversion``, ``not_in_graph``, ``missing_value``, ``error``), with the reason, the gene and the
final-database id; :ref:`concept-statuses` maps both to the statuses of a conversion.


Merging the datasets
--------------------

:py:meth:`idtrack.harmonize.HarmonizeFeatures.unify_multiple_anndatas` merges the harmonised datasets without losing or
rewriting anything by default: ``features="union"`` keeps every feature of any dataset (zeros in ``X`` and in every
layer where a dataset lacks it; ``"intersect"`` keeps the features of every dataset); ``obs`` keeps every column of
any dataset (``obs_join="outer"``; a dataset without a column has missing values there; ``obs_join="inner"`` keeps
only the common columns, and ``obs_columns_to_keep`` selects among them); every ``var`` column is kept (one column
where the datasets agree on each feature, else one per dataset, ``<column>_<dataset>``); layers, ``obsm``, ``obsp`` and
``uns`` entries that every dataset has are carried over, and ``X`` keeps its dtype. ``raw`` is dropped, because its
features are not harmonised: keep raw counts in a layer. The column ``handle_anndata`` of ``obs`` names each cell's
dataset.


The result cache
----------------

The conversions of each dataset are stored in ``project_local_repository`` as one JSON file per dataset. A stored file
is used only if it was written under the same rules of conversion (:py:data:`idtrack.settings.RULES_VERSION`, raised
with every change of an answer) for the same graph (its content digest) and the same feature names, organism, final
database, target and source release; anything else is converted again and the file replaced. Reading the cache back
takes about as long as converting again (:ref:`known-costs`). For a complete workflow on real datasets, see
:doc:`_notebooks/04_tutorial_harmonization`.
