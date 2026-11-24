==========
Quickstart
==========

From a list of gene identifiers to their genes at one Ensembl release, in a few minutes once a graph is on disk.
Install idtrack first (``pip install idtrack``; :doc:`installation`). What an answer means is on :doc:`concepts`; when
it can be wrong, on :doc:`limitations`.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


1. Build or load a graph
------------------------

idtrack answers from a **graph** of one organism's identifier history, built once from Ensembl's tables and then
loaded from its file (:ref:`concept-graph`). The graph and the tables it is built from live in a folder you choose,
the **local repository**:

.. code-block:: python

    import os

    from idtrack import API

    os.makedirs("./idtrack_cache", exist_ok=True)
    # The first time: downloads Ensembl's tables and builds the graph (minutes for cynomolgus, hours for human).
    # Afterwards: loads the graph file from the folder in seconds, without network.
    api = API.build("homo_sapiens", 115, "./idtrack_cache")

``115`` is the **snapshot release**, the last Ensembl release of the graph. Pin it in your scripts: the same graph
gives the same answers in every run, while a script that builds at Ensembl's newest release
(:py:meth:`idtrack.API.ensembl_organism` gives it) builds a new graph, with possibly other answers, after every Ensembl
release. A graph of release 115 converts to and from every release from the organism's first one (human 48) to 115.
How long a build takes and how much memory it needs, per organism: :doc:`performance`.

A graph file on disk loads alone, with no table and no network (a missing file is an
:py:class:`idtrack.errors.GraphFileError`):

.. code-block:: python

    api = API.load(API.graph_path("homo_sapiens", 115, "./idtrack_cache"))
    print(api.describe().organism, api.releases()[0], "to", api.releases()[-1])

.. code-block:: text

    homo_sapiens 48 to 115

:py:meth:`idtrack.API.build` loads the file when it is in the folder, and builds only when it is not; a file under the
expected name that holds another graph (built with another externals configuration, organism table or graph format)
raises :py:class:`idtrack.errors.GraphMismatchError` rather than answering from it, and
``API.build(..., rebuild=True)`` builds the graph again and replaces the file.


2. Convert an identifier
------------------------

.. code-block:: python

    result = api.convert("TP53")       # to the snapshot release
    print(result.status, result.targets())

.. code-block:: text

    converted ('ENSG00000141510.20',)

The target is the Ensembl gene id with its version at the target release. ``result.targets()`` is the best answer;
it holds several genes when they are equal in all evidence (``status == "tie"``: idtrack returns a tie rather than a
guess). A result without a target says why:

.. code-block:: python

    for query in ["NOT_A_GENE", "ENSG00000141510.3", "SNORD116"]:
        result = api.convert(query)
        print(query, result.status, result.reason, result.targets())

.. code-block:: text

    NOT_A_GENE not_found not_in_graph ()
    ENSG00000141510.3 unknown_version unknown_version ()
    SNORD116 tie None ('ENSG00000202498.1', 'ENSG00000212553.1', 'ENSG00000252985.1')

``not_found`` (the string is no identifier of the graph), ``unknown_version`` (a version that does not exist;
``result.known_versions`` lists those that do), ``no_conversion`` (in the graph, but no gene descends from it at the
target release: a retired id), ``no_target`` (the gene has no id in the database asked for) and ``tie``: every
status and reason, with what to do, is in :ref:`concept-statuses`.


3. Another release, another database
------------------------------------

An identifier from a table made with an older release converts to the snapshot release, through Ensembl's history of
retired, merged and split ids; ``to_release`` converts to any release of the graph, ``final_database`` to another
database:

.. code-block:: python

    print(api.convert("ENSG00000141510", from_release=100).targets())
    print(api.convert("ENSG00000141510", from_release=100, to_release=75).targets())
    print(api.convert("ENSG00000141510", final_database="HGNC Symbol").targets())
    print(api.convert("TP53", final_database="base_ensembl_gene").targets())

.. code-block:: text

    ('ENSG00000141510.20',)
    ('ENSG00000141510.11',)
    ('TP53',)
    ('ENSG00000141510',)

Give the release your table was made with: without ``from_release`` a symbol is read as what it names at the target
release (or, where it names nothing there, at its latest release), not as what it named years ago
(:ref:`reading-starting-release`). ``api.final_databases()`` lists the databases you can convert to;
:ref:`database-names` says which name to pass.


4. A batch
----------

.. code-block:: python

    genes = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC", "NOT_A_GENE"]
    results = api.convert_many(genes)          # progress=True shows a progress bar
    for gene, result in zip(genes, results):
        print(f"{gene:<10} {result.status:<10} {', '.join(result.targets())}")
    print(API.classification_summary(API.classify(results)))

.. code-block:: text

    TP53       converted  ENSG00000141510.20
    BRCA1      converted  ENSG00000012048.27
    EGFR       converted  ENSG00000146648.22
    KRAS       converted  ENSG00000133703.15
    MYC        converted  ENSG00000136997.22
    NOT_A_GENE not_found

    IDTrack conversion summary:
      Total processed: 6
      1→0: 1 (16.7%)
      Read at another release: 0 (0.0%)
      1→1: 5 (83.3%)
        Changed only: 5 (100.0%)
        Alternative targets: 0 (0.0%)
        Rest: 0 (0.0%)
      1→n: 0 (0.0%)
        Changed only: 0 (0.0%)
        Alternative targets: 0 (0.0%)
      Statuses:
        converted: 5
        not_found: 1

The arguments are checked once, before any conversion, and each distinct identifier is converted once. An error of
one item (an empty string, a missing value such as ``nan``, a search past its budget) does not stop the batch: that
item has ``status == "error"`` and the error in ``result.error``. An item that repeats another item of the batch with
a suffix of a single-cell tool (``SRF.1`` beside ``SRF``) is that item's copy (:ref:`reading-rules`). The bins of the
summary are explained in :ref:`concept-statuses`.


5. Another organism
-------------------

One :py:class:`idtrack.API` holds one graph; use one object per organism:

.. code-block:: python

    mouse = API.build("mus_musculus", 115, "./idtrack_cache")
    for gene, result in zip(["Trp53", "Brca1", "Egfr"], mouse.convert_many(["Trp53", "Brca1", "Egfr"])):
        print(gene, result.targets())

.. code-block:: text

    Trp53 ('ENSMUSG00000059552.14',)
    Brca1 ('ENSMUSG00000017146.13',)
    Egfr ('ENSMUSG00000020122.17',)

Human, mouse, pig, rhesus and cynomolgus macaque are supported (:doc:`organisms`).


6. Why this answer
------------------

``explain=True`` adds the path behind the answer: how the string was read, each step of Ensembl's history, each
bridge considered (:ref:`concept-explained-path`):

.. code-block:: python

    result = api.convert("TP53", explain=True)
    print("Read as:", result.graph_id, "| match:", result.match, "| at release", result.from_release)
    print("Copies set apart:", list(result.alternative_locus_genes))
    best = result.view("best")[0]
    for index in best.path + best.final_path:
        step = result.path.steps[index]
        print(step.kind, step.source, "->", step.target, "| release", step.release, "|", ", ".join(step.databases))

.. code-block:: text

    Read as: TP53 | match: exact | at release 115
    Copies set apart: ['LRG_321.1']
    entry TP53 -> ENSG00000141510.20 | release 115 | HGNC Symbol, NCBI gene (formerly Entrezgene), UniProtKB Gene Name

Which genes a name names, in which database, at which release, is :py:meth:`idtrack.API.genes_named`;
:py:meth:`idtrack.API.names_of` gives the names of a gene:

.. code-block:: python

    spans = {}
    for m in api.genes_named("TP53"):  # one NameMatch per name, gene version and database, with its releases
        spans.setdefault((m.database, m.synonym, m.gene.split(".")[0]), []).extend(m.releases)
    for (database, synonym, gene), releases in sorted(spans.items()):
        print(f"{database:<32} {'synonym' if synonym else 'name':<8} {gene} {min(releases)} to {max(releases)}")

.. code-block:: text

    HGNC Symbol                      name     ENSG00000141510 48 to 115
    NCBI gene (formerly Entrezgene)  name     ENSG00000141510 63 to 115
    NCBI gene (formerly Entrezgene)  synonym  ENSG00000067369 84 to 87
    UniProtKB Gene Name              name     ENSG00000141510 63 to 115
    UniProtKB Gene Name              synonym  ENSG00000141510 63 to 77
    Vega gene                        name     ENSG00000141510 76 to 88

``TP53`` names ENSG00000141510 in HGNC, NCBI and UniProt, and NCBI recorded it as a synonym of another gene,
ENSG00000067369 (TP53BP1), at releases 84 to 87; ``fold_case=True`` ignores letter case.


7. Without Python
-----------------

.. code-block:: console

    $ idtrack build human 115 --repository idtrack_cache
    $ idtrack convert idtrack_cache/graph_homo_sapiens_assembly38_min48_max115.npz TP53 BRCA1 --from-release 100

:doc:`cli` describes every command and option.


Where next
----------

.. list-table::
    :widths: 32 68

    * - :doc:`concepts`
      - What an answer means: history, bridges, ranking, ties, statuses and reasons.
    * - :doc:`reading`
      - Why a string was read as it was: the starting release, cleaning, matching, versions.
    * - :doc:`limitations`
      - When an answer can be wrong or missing, and how often.
    * - :doc:`harmonization`
      - Converting and merging ``.h5ad`` datasets.
    * - :doc:`tutorials`
      - The tutorials, from installation to production patterns.
