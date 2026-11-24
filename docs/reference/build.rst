Building graphs (for maintainers)
=================================

The build stack: how a graph is made from Ensembl's tables. A user needs none of it beyond
:py:meth:`idtrack.API.build`; it is documented for maintainers and for those who extend idtrack.

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

.. automodule:: idtrack.build.continuity
    :members:

.. automodule:: idtrack.build.assembly_chain
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
