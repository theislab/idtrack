.. _comparison:

=======================
idtrack and Other Tools
=======================

Which tool answers which question about gene identifiers, and how often it is right, measured: the benchmark
repository `idtrack-reproducibility <https://github.com/theislab/idtrack-reproducibility>`_ asked the tools people use
for these questions the same lines idtrack answers, judged every answer by the same truth and the same scorer, and
read a sample of every disagreement with idtrack (its family G, ``docs/families/g.md``). Every number below is copied
from its published tables at commit ``49f1e31`` (``results/g/main/all/``) and its table of known limitations at commit
``72eeb8a``; idtrack's answers are this version's, on its graphs of Ensembl 115.

.. contents::
    :local:
    :depth: 1
    :backlinks: none


The questions and the lines
---------------------------

Five tasks, each a kind of question a user asks of a list of identifiers:

* **a**: an Ensembl id of an earlier release to its gene at release 115;
* **b**: an Ensembl id at its own release;
* **c**: a symbol or name, at the release it was read at or without a release;
* **d**: a previous symbol, an alias or a synonym;
* **e**: an external identifier: an NCBI gene id, a RefSeq or a UniProt accession.

The lines are seeded samples of the benchmarks' families of idtrack (their published lines, with the truth they hold):
per organism and family up to 10,000 lines asked of each web service and up to 100,000 of each local tool. A tool is
asked a line only where it can pose it (its organisms, tasks and releases); every other line is ``not_supported`` and
stays out of its rate. A tool that answers at another release than 115 is judged only on the lines whose truth is the
same at its release as at 115 (the rule of equal truth: at most 4.3 % of a tool's posed lines are left out). A
tool is asked an identifier without its version (``.<n>``), as none of them documents versioned input; idtrack is
asked the line as it stands.

A **rate** is right / (right + wrong + error), the lines the truth cannot decide set apart, under the view ``all``: the
answer must be exactly the truth's genes (compared without versions). idtrack's rate is on the same lines as the
tool's.


The tools
---------

Each tool was asked once, on 2026-11-24 and 25 (the local packages at the versions named), as its own documentation
says, with its best settings for the question.

.. list-table::
    :header-rows: 1
    :widths: 24 30 16 16 14

    * - Tool
      - Asked
      - Answers at release
      - Organisms
      - Tasks
    * - Ensembl REST
      - ``archive/id``, ``xrefs/symbol``, ``lookup/symbol``
      - 116 (its only one)
      - all five
      - a, b, c, d, e
    * - BioMart
      - martservice queries of the archive of the line's release
      - 105 to 116
      - all five
      - b, c, d, e
    * - g:Profiler g:Convert
      - ``e114_eg62_p19`` and its archives e94 to e113
      - 114 (archives at theirs)
      - all five
      - b, c, d, e
    * - MyGene.info
      - build 20261117
      - 116
      - all five
      - b, c, d, e
    * - HGNC REST
      - symbol, previous symbol, alias (the multi-symbol checker's order); NCBI, RefSeq and UniProt ids
      - current (judged at 116)
      - human
      - c, d, e
    * - UniProt ID mapping
      - release 2026_04
      - judged at 116
      - all five
      - e
    * - BridgeDb web service
      - Ensembl 111 data
      - 111
      - pig, mouse, human, rhesus
      - b, c (human), e
    * - gget 0.30.8 ``info``
      - Ensembl REST
      - 116
      - all five
      - b
    * - pyensembl 2.38.0
      - Ensembl's GTF of the release
      - 115 and each line's release
      - all five
      - b, c
    * - EnsDb 113 (ensembldb 2.34.0)
      - AnnotationHub's EnsDb of Ensembl 113
      - 113
      - all five
      - b, c, e
    * - org.*.eg.db 3.22.0
      - ``mapIds`` (SYMBOL, ALIAS, ENTREZID, REFSEQ, UNIPROT to ENSEMBL)
      - judged at 115
      - human, mouse, rhesus
      - c, d, e
    * - NCBI's files
      - ``gene_info``, ``gene_history``, ``gene2ensembl``, ``gene2refseq``
      - judged at 116
      - all five
      - c, d, e
    * - limma 3.68.5
      - ``alias2SymbolTable``, then org.*.eg.db
      - judged at 115
      - human, mouse
      - c, d
    * - HGNChelper 0.8.16
      - ``checkGeneSymbols`` (its map of 2026-11-01), then org.*.eg.db
      - judged at 115
      - human, mouse
      - c, d
    * - Seurat 5.5.1
      - ``UpdateSymbolList`` (HGNC), then org.Hs.eg.db
      - judged at 116
      - human
      - c, d

Not run: Ensembl's ID History Converter (its web page answers only a browser that runs JavaScript, and its Perl route
needs Ensembl's MySQL servers), BED (a Neo4j server), bioDBnet (its documentation could not be read); pybiomart and
biomaRt send BioMart's own query and are scored as BioMart.


The rates
---------

Each cell is the tool's rate / idtrack's rate on the same lines, in per cent, with the number of lines (a dash where
the tool does not pose the task for the organism).

Only Ensembl REST (``archive/id``) and idtrack pose task a, 44 to 50 % of the lines asked of the web services. On
Ensembl ids at their own release (task b) most tools and idtrack are right on every line. On symbols (c), previous
symbols and aliases (d) and external identifiers (e) no tool's rate reaches idtrack's on the same lines in any organism.

**a: an old Ensembl id to its gene at 115**

.. list-table::
    :header-rows: 1
    :widths: 30 14 14 14 14 14

    * - Tool
      - Pig
      - Mouse
      - Human
      - Rhesus
      - Cynomolgus
    * - Ensembl REST
      - 76 / 98 (24,171)
      - 92 / 99 (23,259)
      - 94 / 98 (24,503)
      - 82 / 100 (18,494)
      - 100 / 100 (17,772)

**b: an Ensembl id at its own release**

.. list-table::
    :header-rows: 1
    :widths: 30 14 14 14 14 14

    * - Tool
      - Pig
      - Mouse
      - Human
      - Rhesus
      - Cynomolgus
    * - Ensembl REST
      - 100 / 100 (1,173)
      - 100 / 100 (1,721)
      - 100 / 100 (1,478)
      - 100 / 100 (1,506)
      - 100 / 100 (2,228)
    * - BioMart
      - 100 / 100 (1,173)
      - 100 / 100 (1,721)
      - 100 / 100 (1,478)
      - 100 / 100 (1,506)
      - 100 / 100 (2,228)
    * - g:Profiler
      - 100 / 100 (1,173)
      - 100 / 100 (1,721)
      - 90 / 100 (1,478)
      - 100 / 100 (1,506)
      - 100 / 100 (2,228)
    * - MyGene.info
      - 98 / 100 (1,173)
      - 100 / 100 (1,721)
      - 100 / 100 (1,478)
      - 99 / 100 (1,506)
      - 100 / 100 (2,228)
    * - BridgeDb
      - 93 / 100 (1,172)
      - 100 / 100 (1,499)
      - 91 / 100 (1,285)
      - 99 / 100 (1,506)
      - –
    * - gget
      - 100 / 100 (1,173)
      - 100 / 100 (1,721)
      - 100 / 100 (1,478)
      - 100 / 100 (1,506)
      - 100 / 100 (2,228)
    * - pyensembl
      - 100 / 100 (8,595)
      - 100 / 100 (13,223)
      - 100 / 100 (13,934)
      - 100 / 100 (9,863)
      - 100 / 100 (13,124)
    * - EnsDb 113
      - 100 / 100 (8,595)
      - 100 / 100 (13,216)
      - 100 / 100 (13,921)
      - 100 / 100 (9,863)
      - 100 / 100 (13,124)

**c: a symbol or name**

.. list-table::
    :header-rows: 1
    :widths: 30 14 14 14 14 14

    * - Tool
      - Pig
      - Mouse
      - Human
      - Rhesus
      - Cynomolgus
    * - Ensembl REST
      - 93 / 98 (9,783)
      - 90 / 98 (9,677)
      - 64 / 98 (8,847)
      - 92 / 100 (5,054)
      - 93 / 100 (6,320)
    * - Ensembl REST ``lookup/symbol``
      - 90 / 98 (9,783)
      - 83 / 98 (9,677)
      - 67 / 98 (8,847)
      - 86 / 100 (5,054)
      - 81 / 100 (6,320)
    * - BioMart
      - 91 / 99 (10,124)
      - 87 / 98 (9,876)
      - 68 / 98 (8,952)
      - 87 / 100 (5,412)
      - 88 / 100 (6,677)
    * - g:Profiler
      - 92 / 99 (10,274)
      - 87 / 98 (9,848)
      - 69 / 98 (9,365)
      - 88 / 100 (5,659)
      - 92 / 100 (7,772)
    * - MyGene.info
      - 90 / 98 (9,811)
      - 86 / 98 (9,715)
      - 69 / 98 (8,891)
      - 83 / 100 (5,081)
      - 80 / 100 (6,342)
    * - HGNC
      - –
      - –
      - 71 / 98 (8,847)
      - –
      - –
    * - BridgeDb
      - –
      - –
      - 70 / 98 (8,607)
      - –
      - –
    * - pyensembl
      - 90 / 100 (92,841)
      - 78 / 98 (102,278)
      - 68 / 98 (103,226)
      - 86 / 100 (81,506)
      - 95 / 100 (83,390)
    * - EnsDb 113
      - 89 / 100 (80,068)
      - 74 / 98 (84,766)
      - 62 / 97 (86,952)
      - 82 / 100 (68,852)
      - 94 / 100 (73,354)
    * - org.*.eg.db
      - –
      - 60 / 99 (88,435)
      - 59 / 97 (87,412)
      - 74 / 100 (68,983)
      - –
    * - NCBI's files
      - 23 / 100 (78,091)
      - 60 / 99 (88,164)
      - 63 / 97 (87,141)
      - 74 / 100 (64,161)
      - 67 / 100 (71,826)
    * - limma
      - –
      - 64 / 99 (88,435)
      - 62 / 97 (87,412)
      - –
      - –
    * - HGNChelper
      - –
      - 72 / 99 (88,435)
      - 61 / 97 (87,412)
      - –
      - –
    * - Seurat
      - –
      - –
      - 59 / 98 (8,891)
      - –
      - –

**d: a previous symbol, an alias or a synonym**

.. list-table::
    :header-rows: 1
    :widths: 30 14 14 14 14 14

    * - Tool
      - Pig
      - Mouse
      - Human
      - Rhesus
      - Cynomolgus
    * - Ensembl REST
      - 79 / 100 (2,374)
      - 90 / 100 (3,116)
      - 85 / 99 (3,011)
      - 82 / 100 (2,438)
      - 45 / 99 (1,323)
    * - Ensembl REST ``lookup/symbol``
      - 10 / 100 (2,374)
      - 15 / 100 (3,116)
      - 6 / 99 (3,011)
      - 12 / 100 (2,438)
      - 32 / 99 (1,323)
    * - BioMart
      - 73 / 100 (2,951)
      - 90 / 100 (3,383)
      - 91 / 99 (3,041)
      - 76 / 100 (2,478)
      - 35 / 99 (1,350)
    * - g:Profiler
      - 17 / 100 (2,964)
      - 15 / 100 (3,654)
      - 8 / 100 (3,530)
      - 16 / 100 (3,192)
      - 27 / 99 (1,618)
    * - MyGene.info
      - 92 / 100 (2,385)
      - 80 / 100 (3,130)
      - 73 / 99 (3,024)
      - 90 / 100 (2,442)
      - 72 / 99 (1,325)
    * - HGNC
      - –
      - –
      - 92 / 99 (3,011)
      - –
      - –
    * - org.*.eg.db
      - –
      - 80 / 100 (35,349)
      - 68 / 100 (33,844)
      - –
      - –
    * - NCBI's files
      - 10 / 99 (10,130)
      - 82 / 100 (35,287)
      - 71 / 100 (33,715)
      - 78 / 99 (6,588)
      - 49 / 99 (3,434)
    * - limma
      - –
      - 80 / 100 (35,349)
      - 67 / 100 (33,844)
      - –
      - –
    * - HGNChelper
      - –
      - 32 / 100 (35,349)
      - 82 / 100 (33,844)
      - –
      - –
    * - Seurat
      - –
      - –
      - 38 / 99 (3,025)
      - –
      - –

**e: an external identifier**

.. list-table::
    :header-rows: 1
    :widths: 30 14 14 14 14 14

    * - Tool
      - Pig
      - Mouse
      - Human
      - Rhesus
      - Cynomolgus
    * - Ensembl REST
      - 68 / 100 (8,200)
      - 69 / 100 (8,432)
      - 60 / 100 (7,779)
      - 69 / 100 (8,298)
      - 69 / 100 (7,389)
    * - BioMart
      - 70 / 100 (8,908)
      - 70 / 100 (8,879)
      - 66 / 100 (8,045)
      - 70 / 100 (8,444)
      - 72 / 100 (8,476)
    * - g:Profiler
      - 73 / 100 (9,482)
      - 73 / 100 (9,642)
      - 69 / 100 (8,941)
      - 73 / 100 (9,588)
      - 74 / 100 (9,084)
    * - MyGene.info
      - 51 / 100 (8,200)
      - 69 / 100 (8,432)
      - 63 / 100 (7,779)
      - 53 / 100 (8,298)
      - 55 / 100 (7,389)
    * - HGNC
      - –
      - –
      - 40 / 100 (7,779)
      - –
      - –
    * - UniProt
      - 65 / 100 (3,831)
      - 48 / 100 (4,226)
      - 74 / 100 (4,314)
      - 78 / 100 (3,641)
      - 75 / 100 (3,812)
    * - BridgeDb
      - 71 / 100 (7,796)
      - 67 / 100 (7,936)
      - 68 / 100 (7,626)
      - 68 / 100 (8,294)
      - –
    * - EnsDb 113
      - 81 / 100 (11,215)
      - 60 / 99 (13,580)
      - 58 / 99 (13,342)
      - 77 / 100 (11,994)
      - 76 / 100 (10,442)
    * - org.*.eg.db
      - –
      - 68 / 100 (83,657)
      - 57 / 100 (77,681)
      - 73 / 100 (82,129)
      - –
    * - NCBI's files
      - 40 / 100 (43,657)
      - 67 / 100 (40,655)
      - 63 / 100 (32,748)
      - 64 / 100 (44,196)
      - 57 / 100 (37,098)


What the disagreements are
--------------------------

The rates are no ranking of the tools on what each is built for. Of every disagreement with idtrack, a seeded sample
of at most 200 lines per tool, task and organism was read, each from the stored responses and Ensembl's tables of the
release the tool answers at; a line where the tool is wrong falls in one of four groups:

* **not a question the tool claims**: the line asks a name, id or accession as an earlier release read it, and the tool
  reads today's tables (it claims no history); a synonym asked of a lookup that reads display names only; no answer
  where Ensembl's tables of the tool's release do not name the gene either;
* **right under another reading**: the tool names the gene its own authority gives (HGNC, MGI, NCBI, UniProt), a copy
  on a patch, a part of the truth's genes, a successor that Ensembl's history allows on a branch, another gene that the
  string also names at the tool's release, or an id that is no gene of the tool's release;
* **truly wrong**: another current gene by none of these readings, or no answer where Ensembl's tables of the tool's
  release name the truth's gene;
* **the service failed**: an error answer.

For Ensembl REST on tasks c, d and e, for BioMart, HGNC, UniProt, EnsDb and pyensembl, for g:Profiler and BridgeDb on
tasks c and e, for Seurat on c, and for the external identifiers (task e) of every tool, the largest group of the wrong
lines is questions the tool does not claim to answer (43 to 100 %), and Ensembl REST's ``archive/id`` is right under
another reading on 99 % of its wrong lines; the lines truly wrong are the largest group for limma, HGNChelper,
org.*.eg.db and NCBI's files on tasks c and d, for Seurat, MyGene.info and g:Profiler on d, for MyGene.info on c, and on
task b, where g:Profiler, BridgeDb and MyGene.info lack an Ensembl gene in their own database. The shares are weighted
back from the sample to all of a tool's disagreements of that task, every organism together:

.. list-table::
    :header-rows: 1
    :widths: 26 6 10 14 14 10 10

    * - Tool
      - Task
      - Wrong lines read
      - Not a question the tool claims (%)
      - Right under another reading (%)
      - Truly wrong (%)
      - The service failed (%)
    * - Ensembl REST
      - a
      - 207
      - 0
      - 99
      - 1
      - 0
    * - Ensembl REST
      - c
      - 433
      - 53
      - 42
      - 0
      - 5
    * - Ensembl REST
      - d
      - 746
      - 86
      - 14
      - 0
      - 0
    * - Ensembl REST
      - e
      - 529
      - 82
      - 17
      - 2
      - 0
    * - Ensembl REST ``lookup/symbol``
      - c
      - 313
      - 44
      - 43
      - 13
      - 0
    * - Ensembl REST ``lookup/symbol``
      - d
      - 853
      - 95
      - 0
      - 5
      - 0
    * - BioMart
      - c
      - 311
      - 66
      - 18
      - 16
      - 0
    * - BioMart
      - d
      - 761
      - 74
      - 6
      - 21
      - 0
    * - BioMart
      - e
      - 578
      - 85
      - 15
      - 0
      - 0
    * - g:Profiler
      - b
      - 142
      - 0
      - 0
      - 100
      - 0
    * - g:Profiler
      - c
      - 363
      - 60
      - 23
      - 17
      - 0
    * - g:Profiler
      - d
      - 857
      - 16
      - 0
      - 84
      - 0
    * - g:Profiler
      - e
      - 573
      - 85
      - 14
      - 1
      - 0
    * - MyGene.info
      - b
      - 42
      - 0
      - 0
      - 100
      - 0
    * - MyGene.info
      - c
      - 330
      - 33
      - 25
      - 41
      - 0
    * - MyGene.info
      - d
      - 810
      - 19
      - 24
      - 58
      - 0
    * - MyGene.info
      - e
      - 513
      - 71
      - 14
      - 15
      - 0
    * - HGNC
      - c
      - 46
      - 72
      - 28
      - 0
      - 0
    * - HGNC
      - d
      - 177
      - 59
      - 38
      - 3
      - 0
    * - HGNC
      - e
      - 148
      - 67
      - 1
      - 32
      - 0
    * - UniProt
      - e
      - 771
      - 90
      - 3
      - 7
      - 0
    * - BridgeDb
      - b
      - 215
      - 0
      - 0
      - 100
      - 0
    * - BridgeDb
      - c
      - 45
      - 73
      - 22
      - 4
      - 0
    * - BridgeDb
      - e
      - 463
      - 84
      - 13
      - 3
      - 0
    * - pyensembl
      - c
      - 197
      - 46
      - 20
      - 34
      - 0
    * - EnsDb 113
      - c
      - 196
      - 53
      - 19
      - 29
      - 0
    * - EnsDb 113
      - e
      - 430
      - 100
      - 0
      - 0
      - 0
    * - org.*.eg.db
      - c
      - 112
      - 29
      - 12
      - 60
      - 0
    * - org.*.eg.db
      - d
      - 89
      - 7
      - 5
      - 89
      - 0
    * - org.*.eg.db
      - e
      - 170
      - 80
      - 11
      - 9
      - 0
    * - NCBI's files
      - c
      - 226
      - 19
      - 37
      - 44
      - 0
    * - NCBI's files
      - d
      - 421
      - 9
      - 41
      - 50
      - 0
    * - NCBI's files
      - e
      - 354
      - 46
      - 32
      - 22
      - 0
    * - limma
      - c
      - 68
      - 41
      - 7
      - 51
      - 0
    * - limma
      - d
      - 98
      - 10
      - 20
      - 70
      - 0
    * - HGNChelper
      - c
      - 68
      - 38
      - 18
      - 44
      - 0
    * - HGNChelper
      - d
      - 102
      - 12
      - 7
      - 81
      - 0
    * - Seurat
      - c
      - 37
      - 43
      - 24
      - 32
      - 0
    * - Seurat
      - d
      - 134
      - 4
      - 3
      - 93
      - 0

What a tool is not built for, as the reading found it: Ensembl REST's ``archive/id`` names an id's possible
replacements at the release that retired it, one step of Ensembl's history, and does not follow them on (its wrong
lines of task a are almost all such steps); the services and packages read a name or an accession as their one release
reads it, so a line that asks for an earlier reading asks for history they do not claim; ``lookup/symbol`` reads
display names only; g:Profiler, BridgeDb and MyGene.info know an Ensembl gene only where their own database has it;
NCBI's files and MyGene.info give pig's NCBI genes the Ensembl genes of other pig breeds, not of the reference
assembly.


When another tool is the better choice
--------------------------------------

* **The current release.** The web services answer at Ensembl's current release (116 when they were asked) and at
  today's state of the authorities; a graph of idtrack answers up to its snapshot release (115 here) until a graph of a
  later release is built.
* **No graph to build.** A service or an annotation package answers at once; idtrack needs its graph, which takes 8
  minutes (cynomolgus) to about 2.5 hours (human) to build once (:doc:`performance`).
* **Other databases and attributes.** BioMart, MyGene.info, g:Profiler and UniProt's ID mapping reach databases and
  attributes that a graph's selection of databases does not hold (:doc:`databases`); idtrack converts only between the
  databases its graph holds.
* **The authority's own answer.** For today's official human symbol, HGNC is the authority, and NCBI's files are
  NCBI's own assignment of its gene ids; idtrack answers by Ensembl's records, which can differ from the authority's
  file (:ref:`limitation-cross-references`).
* **Ensembl ids of one release.** Asked at their own release (task b), Ensembl REST, BioMart, gget, pyensembl and
  EnsDb are right on every line, as idtrack is.

What idtrack adds is history: an identifier, a name or an accession read as the release that wrote it read it, carried
to the target through Ensembl's own history, with the path that led there and a tie where the evidence cannot decide.


Where idtrack was wrong on these lines
--------------------------------------

Of the sampled lines where idtrack is wrong (1,015: pig 163, mouse 375, human 408, rhesus 52, cynomolgus 17), every one
was read to its cause:

.. list-table::
    :header-rows: 1
    :widths: 62 10 28

    * - Cause
      - Lines
      - Where stated
    * - A bridge where Ensembl's history gives the id no successor, to a gene away from the old place
      - 361
      - :ref:`limitation-bridges`
    * - Ensembl's history carries the gene to another place
      - 130
      - :ref:`limitation-moves`
    * - A name another source gives another gene, a string read as another gene's name or at another release, a value
        of another database, an Rfam family label
      - 213
      - :ref:`limitation-other-sources`
    * - A symbol matched only in another letter case
      - 78
      - :ref:`limitation-case`
    * - A name read at an old release whose genes' history ends, answered through a bridge
      - 74
      - :ref:`limitation-bridges`, :ref:`limitation-assembly-mapping`
    * - A name only copies on patches or haplotypes display, or one in the truth's answer: idtrack answers the gene of
        the reference sequence and lists the copies apart
      - 68
      - :ref:`reading-copies`
    * - Two genes with one string in two letter cases
      - 14
      - :ref:`limitation-spellings`
    * - A name that is an identifier of a database the graph does not hold
      - 11
      - :ref:`limitation-databases`
    * - miRBase's names on miRNA genes only
      - 21
      - :ref:`limitation-mirbase`
    * - A number read in another numbering scheme
      - 6
      - :ref:`limitation-numbers`
    * - An NCBI gene id that NCBI retired and Ensembl keeps as a synonym
      - 3
      - :ref:`limitation-cross-references`
    * - An accession's version read where Ensembl attached it
      - 3
      - :ref:`reading-accession-versions`
    * - Ensembl gave the gene a new id without an event (no successor)
      - 3
      - :ref:`limitation-bridges`
    * - A string that stands for a missing value, refused
      - 1
      - :ref:`reading-cleaning`
    * - The benchmark's truth: UniProt's gene names of releases 62 to 72 not read, a name of Ensembl's Vega or
        Havana records not read, a string read at an earlier release than the target
      - 29
      - not idtrack's

Of the 1,015 lines, 97 are the benchmark's own truth (the last row and 68 lines of copies, where the answer is the
gene of the reference sequence at the copies' place, :ref:`reading-copies`); the benchmark repository's
known-limitations table gives each cause's lines per organism.


The limits of the comparison
----------------------------

* Each service was asked on one date (2026-11-24 and 25); the services change on their own rhythm.
* BioMart's archives of releases 105 and 108 to 112 answered HTTP 500 on those dates: their lines are counted as not
  answered (pig 34, 263 and 247 lines of tasks c, d and e; mouse 40, 275, 306; human 61, 507, 457; rhesus 84, 717,
  558; cynomolgus 363, 351, 458), never as wrong.
* The truth reads Ensembl's tables of 115, or of the tool's release by the rule of equal truth: a tool built on
  another authority (NCBI, HGNC, UniProt) is right by that authority where the reading says so, and still counted
  wrong in its rate.
* NCBI's files and the org.*.eg.db packages name no Ensembl release; they are judged at 116 and 115.
* The reading is a seeded sample of at most 200 lines per tool, task and organism, its shares weighted back to every
  disagreement; a group is what Ensembl's tables show, not a statement of a tool's intent.
* idtrack's graphs are of release 115; no graph of 116 was built.
