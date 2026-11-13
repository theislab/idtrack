Reference
=========

The public names are importable from ``idtrack`` (``idtrack.API``, ``idtrack.HarmonizeFeatures``, ...) and are
documented here under those names. The modules ``idtrack._<name>`` hold them; what is documented under a module path
below (helper functions, constants, the graph's value types) is used by the public classes and may change between
versions.

.. contents::
    :local:
    :depth: 2
    :backlinks: none


Conversion and harmonization
----------------------------

.. autoclass:: idtrack.API
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack.HarmonizeFeatures
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: idtrack._harmonize_features
    :members:
    :exclude-members: HarmonizeFeatures

.. automodule:: idtrack._api
    :members:
    :private-members:
    :exclude-members: API


The conversion engine and the graph
-----------------------------------

.. autoclass:: idtrack.Track
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack.TheGraph
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack.GraphMaker
    :members:
    :private-members:
    :show-inheritance:

.. automodule:: idtrack._track
    :members:
    :private-members:
    :exclude-members: Track, InactiveAtReleaseError, UnknownVersionError, SearchBudgetExceededError

.. automodule:: idtrack._the_graph
    :members:
    :private-members:
    :ignore-module-all:
    :exclude-members: TheGraph, GraphMismatchError

.. automodule:: idtrack._graph_maker
    :members:
    :private-members:
    :exclude-members: GraphMaker, HistoryValidationError, GraphMismatchError


Errors
------

.. autoexception:: idtrack.InactiveAtReleaseError
    :members:
    :show-inheritance:

.. autoexception:: idtrack.UnknownVersionError
    :members:
    :show-inheritance:

.. autoexception:: idtrack.SearchBudgetExceededError
    :members:
    :show-inheritance:

.. autoexception:: idtrack.GraphMismatchError
    :members:
    :show-inheritance:

.. autoexception:: idtrack.HistoryValidationError
    :members:
    :show-inheritance:

.. autoexception:: idtrack.EnsemblRestError
    :members:
    :show-inheritance:

.. autoexception:: idtrack.EnsemblDownloadError
    :members:
    :show-inheritance:


Ensembl tables, organisms and external databases
------------------------------------------------

.. autoclass:: idtrack.DB
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack.DatabaseManager
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack.ExternalDatabases
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack.VerifyOrganism
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack.ConnectionBridge
    :members:
    :private-members:
    :exclude-members: _BridgeState
    :show-inheritance:

.. automodule:: idtrack._db
    :members:
    :private-members:
    :exclude-members: DB, EnsemblRestError

.. automodule:: idtrack._database_manager
    :members:
    :private-members:
    :exclude-members: DatabaseManager

.. automodule:: idtrack._external_databases
    :members:
    :private-members:
    :exclude-members: ExternalDatabases

.. automodule:: idtrack._verify_organism
    :members:
    :private-members:
    :exclude-members: VerifyOrganism

.. automodule:: idtrack._ensembl_access
    :members:
    :private-members:
    :exclude-members: EnsemblDownloadError

.. automodule:: idtrack._mysql_dump
    :members:
    :private-members:

.. automodule:: idtrack._default_externals
    :members:
    :private-members:

.. automodule:: idtrack._utils_hdf5
    :members:


Self-tests
----------

.. autoclass:: idtrack.TrackTests
    :members:
    :private-members:
    :show-inheritance:

.. autoclass:: idtrack._track_tests.CheckResult
    :members:


.. toctree::
    :maxdepth: 1
    :hidden:

    reference/external_mappers
