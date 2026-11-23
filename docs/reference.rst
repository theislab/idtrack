Reference
=========

The public API of IDTrack, module by module. The most used names are also importable from ``idtrack`` itself
(``idtrack.API``, ``idtrack.ConversionResult``, the errors, ...); they are documented here under the module that
holds them. Modules and names without a leading underscore are public.

.. contents::
    :local:
    :depth: 2
    :backlinks: none


The API
-------

.. automodule:: idtrack.api
    :members:
    :show-inheritance:


Results
-------

.. automodule:: idtrack.results
    :members:
    :show-inheritance:


Errors
------

.. automodule:: idtrack.errors
    :members:
    :show-inheritance:


Settings
--------

.. automodule:: idtrack.settings
    :members:
    :show-inheritance:


Organisms
---------

.. automodule:: idtrack.organisms
    :members:


The graph's vocabulary
----------------------

.. automodule:: idtrack.ids
    :members:


How an identifier is read
-------------------------

.. automodule:: idtrack.reading
    :members:


The graph and its file
----------------------

.. automodule:: idtrack.graph
    :members:
    :inherited-members: MultiDiGraph, DiGraph, Graph
    :show-inheritance:

.. automodule:: idtrack.graph.files
    :members:

.. automodule:: idtrack.graph.genes
    :members:


The conversion engine
---------------------

.. automodule:: idtrack.engine
    :members:
    :show-inheritance:

.. automodule:: idtrack.engine.ranking
    :members:

.. automodule:: idtrack.engine.explain
    :members:


Self-tests of a graph
---------------------

.. automodule:: idtrack.checks
    :members:
    :show-inheritance:


Harmonization
-------------

.. automodule:: idtrack.harmonize
    :members:

.. autoclass:: idtrack.harmonize.HarmonizeFeatures
    :members:


Building graphs
---------------

.. automodule:: idtrack.build

.. autoclass:: idtrack.build.GraphMaker
    :members:

.. autoclass:: idtrack.build.DatabaseManager
    :members:

.. autoclass:: idtrack.build.ExternalDatabases
    :members:

.. autoclass:: idtrack.build.VerifyOrganism
    :members:

.. automodule:: idtrack.build.externals
    :members:
    :exclude-members: ExternalDatabases

.. automodule:: idtrack.build.default_externals
    :members:

.. automodule:: idtrack.build.tables
    :members:

.. automodule:: idtrack.build.ensembl_access
    :members:

.. automodule:: idtrack.build.mysql_dump
    :members:

.. automodule:: idtrack.build.species
    :members:
    :exclude-members: VerifyOrganism

.. automodule:: idtrack.build.database_manager
    :members:
    :exclude-members: DatabaseManager
