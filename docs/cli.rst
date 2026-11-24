.. _cli:

============
Command Line
============

``idtrack`` (or ``python -m idtrack``) builds, checks, describes and converts with graph files, without writing
Python. Each command is a thin layer over the API, and gives the same answers. ``idtrack --help`` and
``idtrack <command> --help`` print the options below; ``idtrack --version`` the version.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


A first session
---------------

.. code-block:: console

    $ export IDTRACK_ENSEMBL_ACCESS=https          # where Ensembl's MySQL ports are blocked
    $ idtrack build cynomolgus 115 --repository idtrack_cache
    $ idtrack info idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz
    $ idtrack convert idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz TP53 BRCA1 MT-CO1 NOT_A_GENE

The last command prints:

.. code-block:: text

    query	status	targets	reason
    TP53	converted	ENSMFAG00000038506.2
    BRCA1	converted	ENSMFAG00000043956.2
    MT-CO1	converted	ENSMFAG00000060073.1
    NOT_A_GENE	not_found		not_in_graph

The cynomolgus graph is the smallest (12 to 14 minutes from nothing); human takes hours (:doc:`performance`).


``idtrack build``
-----------------

.. code-block:: text

    idtrack build [-r REPOSITORY] [--first-release FIRST_RELEASE] [--rebuild] [-q] organism snapshot_release

Builds the graph of an organism and a snapshot release from Ensembl's tables (:py:meth:`idtrack.API.build`), writes its
file into the repository, and prints the file's path and a line of what it holds. Tables already in the repository
are not downloaded again; if the repository holds the file of this graph, it is loaded and checked instead.

.. list-table::
    :widths: 32 68

    * - ``organism``
      - ``homo_sapiens``, ``mus_musculus``, ``sus_scrofa``, ``macaca_mulatta``, ``macaca_fascicularis``, or the short
        names ``human``, ``mouse``, ``pig``, ``rhesus``, ``cynomolgus`` (:doc:`organisms`).
    * - ``snapshot_release``
      - The last Ensembl release of the graph (e.g. ``115``).
    * - ``-r``, ``--repository``
      - The local repository: the folder of downloaded tables and graph files; made if missing (default
        ``idtrack_cache``).
    * - ``--first-release``
      - The first release of the graph (default: the organism's first release in idtrack).
    * - ``--rebuild``
      - Build even if the graph file exists, and replace it.
    * - ``-q``, ``--quiet``
      - Log warnings and errors only (progress is logged to standard error).


``idtrack verify``
------------------

.. code-block:: text

    idtrack verify [-r REPOSITORY] graph

Checks a graph file: that this version reads its format, that its record is valid for this version's organism table,
and that its content gives the digest the file records (seconds). With ``--repository`` (the folder it was built in;
a table it lacks is downloaded), the self-tests of :py:class:`idtrack.checks.GraphChecks` also compare the graph with
its tables (cynomolgus under a minute, human about 8 minutes). It prints one line per check and ``verified`` or ``NOT
verified``; the exit status is 0 when everything holds and 1 otherwise.


``idtrack info``
----------------

.. code-block:: text

    idtrack info [--json] graph

Shows what a graph file holds (:py:meth:`idtrack.API.describe`): organism, releases, primary assembly, the assemblies
with the releases they serve, the external and final databases, the idtrack version and commit that built it, and its
content digest. ``--json`` prints the description as JSON.


``idtrack convert``
-------------------

.. code-block:: text

    idtrack convert [-f FILE] [--from-release FROM_RELEASE] [--to-release TO_RELEASE]
                    [--final-database FINAL_DATABASE] [--from-database FROM_DATABASE]
                    [--all] [--no-bridges] [--json] graph [identifiers ...]

Converts identifiers with a graph file (:py:meth:`idtrack.API.convert_many`; no network), to the graph's snapshot
release unless ``--to-release`` says otherwise.

.. list-table::
    :widths: 32 68

    * - ``identifiers``
      - The identifiers, on the command line.
    * - ``-f``, ``--file``
      - A file of identifiers, one per line (``-`` for standard input); blank lines are skipped.
    * - ``--from-release``
      - The release the identifiers are read at (``from_release``; default: :ref:`reading-starting-release`).
    * - ``--to-release``
      - The target release (default: the graph's snapshot release).
    * - ``--final-database``
      - The database of the targets, e.g. ``"HGNC Symbol"`` (default: Ensembl gene ids; :ref:`database-names`).
    * - ``--from-database``
      - The external database the identifiers are ids of.
    * - ``--all``
      - Print every candidate's target, not only the best answer's (``view("all")``).
    * - ``--no-bridges``
      - Follow Ensembl's history only (``allow_bridges=False``).
    * - ``--json``
      - Print the full results as a JSON list (``ConversionResult.to_dict()`` of each).

**Output.** A header line, then one tab-separated line per identifier: the identifier, its status, its targets
(separated by commas: several targets of the best answer are a tie, of several genes or of several ids of one gene:
:ref:`concept-ties`)
and its reason (:ref:`concept-statuses`). An identifier that cannot be converted is a line with its status, never an
error of the command. To another database (the genes of cynomolgus's newest assembly carry few VGNC or HGNC
symbols, so ``--final-database "VGNC Symbol"`` answers these ``no_target``; UniProt's gene names they do carry):

.. code-block:: console

    $ idtrack convert idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz --final-database "UniProtKB Gene Name" TP53 BRCA1 MT-CO1

.. code-block:: text

    query	status	targets	reason
    TP53	converted	TP53
    BRCA1	converted	BRCA1
    MT-CO1	converted	COX1


Exit status
-----------

0 when the command did its work (``convert``: whatever the statuses of the identifiers); 1 when idtrack raised an
error (printed as one line, ``idtrack: error: <message>``, on standard error) or, for ``verify``, a check does not
hold; 2 when the command line cannot be parsed. ``idtrack`` without a command prints its help.
