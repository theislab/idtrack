**idtrack**
===========

|PyPI| |PyPIDownloads| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov|

.. |PyPI| image:: https://img.shields.io/pypi/v/idtrack.svg
   :target: https://pypi.org/project/idtrack/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/idtrack
   :target: https://pypi.org/project/idtrack
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/theislab/idtrack
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/idtrack/latest.svg?label=Read%20the%20Docs
   :target: https://idtrack.readthedocs.io/
   :alt: Read the documentation at https://idtrack.readthedocs.io/
.. |Build| image:: https://github.com/theislab/idtrack/actions/workflows/build_package.yml/badge.svg?branch=main
   :target: https://github.com/theislab/idtrack/actions/workflows/build_package.yml
   :alt: Build Package Status
.. |Tests| image:: https://github.com/theislab/idtrack/actions/workflows/run_tests.yml/badge.svg?branch=main
   :target: https://github.com/theislab/idtrack/actions/workflows/run_tests.yml
   :alt: Tests status
.. |PyPIDownloads| image:: https://pepy.tech/badge/idtrack
   :target: https://pepy.tech/project/idtrack
   :alt: downloads
.. |Codecov| image:: https://codecov.io/gh/theislab/idtrack/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/theislab/idtrack
   :alt: Codecov

.. image:: https://raw.githubusercontent.com/theislab/idtrack/main/docs/_logo/logo.png
    :width: 350
    :alt: Logo


Key Features
------------

* Cross-temporal identifier mapping across Ensembl releases ("time travel" through ID history)
* Cross-database mapping between Ensembl IDs and external namespaces via configurable cross-references
* Reproducible, cache-backed graph snapshots anchored to a chosen release boundary
* Batch conversion utilities and conversion outcome classification (1:0, 1:1, 1:n)
* Optional feature harmonization workflows for multi-dataset integration

Usage
-----

Please see the `Documentation <Documentation_>`_ for details.

.. _PyPI: https://pypi.org/
.. _pip: https://pip.pypa.io/
.. _Documentation: https://idtrack.readthedocs.io/
