=======
Credits
=======

IDTrack is developed and maintained by researchers committed to reproducible
and transparent biological identifier mapping.

.. contents::
    :local:
    :depth: 2
    :backlinks: none


Development Lead
----------------

.. list-table::
    :widths: 30 70
    :class: borderless

    * - **Kemal Inecik**
      - Lead Developer

        `k.inecik@gmail.com <mailto:k.inecik@gmail.com>`_

        `GitHub <https://github.com/kemalInecik>`_


Contributors
------------

We welcome contributions from the community! The following individuals have contributed
to IDTrack through code, documentation, bug reports, or feature suggestions:

Contributors are credited automatically through the project’s Git history. The most up-to-date list is available on GitHub.

See the :doc:`contributing` guide to learn how you can help improve IDTrack.
You can also see the live contributor list on GitHub:
`github.com/theislab/idtrack/graphs/contributors <https://github.com/theislab/idtrack/graphs/contributors>`_.


Acknowledgments
---------------

IDTrack builds upon the work of many open-source projects and biological databases.
We are grateful to the developers and maintainers of these resources.

Data Sources
~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Database
      - Description
    * - `Ensembl <https://www.ensembl.org/>`_
      - Genome annotation, identifier history, and cross-references
    * - `HGNC <https://www.genenames.org/>`_
      - Human Gene Nomenclature Committee - official human gene symbols
    * - `MGI <http://www.informatics.jax.org/>`_
      - Mouse Genome Informatics - mouse gene nomenclature and annotation
    * - `RGD <https://rgd.mcw.edu/>`_
      - Rat Genome Database - rat gene nomenclature and annotation
    * - `UniProt <https://www.uniprot.org/>`_
      - Universal Protein Resource - protein sequence and annotation
    * - `NCBI Gene <https://www.ncbi.nlm.nih.gov/gene>`_
      - National Center for Biotechnology Information - gene information

Software Dependencies
~~~~~~~~~~~~~~~~~~~~~

IDTrack relies on excellent open-source software:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Package
      - Purpose
    * - `NetworkX <https://networkx.org/>`_
      - Graph data structures and algorithms
    * - `pandas <https://pandas.pydata.org/>`_
      - Data manipulation and analysis
    * - `NumPy <https://numpy.org/>`_
      - Numerical computing
    * - `AnnData <https://anndata.readthedocs.io/>`_
      - Annotated data matrices for single-cell analysis
    * - `h5py <https://www.h5py.org/>`_
      - HDF5 file format support
    * - `requests <https://requests.readthedocs.io/>`_
      - HTTP library for database downloads
    * - `Rich <https://rich.readthedocs.io/>`_
      - Beautiful terminal output and progress bars

Documentation Tools
~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Tool
      - Purpose
    * - `Sphinx <https://www.sphinx-doc.org/>`_
      - Documentation generation
    * - `nbsphinx <https://nbsphinx.readthedocs.io/>`_
      - Jupyter notebook integration
    * - `Read the Docs <https://readthedocs.org/>`_
      - Documentation hosting


Institutional Support
---------------------

This project was developed at:

- `Theis Lab <https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab>`_, Helmholtz Center Munich
- Institute of Computational Biology

We thank the lab members and collaborators for their feedback and support during development.


Citing IDTrack
--------------

If you use IDTrack in your research, please cite:

.. code-block:: bibtex

    @software{idtrack,
      author = {Inecik, Kemal},
      title = {IDTrack: Cross-Temporal and Cross-Database Biological Identifier Mapping},
      url = {https://github.com/theislab/idtrack},
      year = {2025}
    }

For academic publications, you may also cite the specific version you used:

.. code-block:: bibtex

    @software{idtrack_version,
      author = {Inecik, Kemal},
      title = {IDTrack},
      version = {0.0.5},
      url = {https://github.com/theislab/idtrack},
      year = {2025}
    }


License
-------

IDTrack is released under the `BSD 3-Clause License <https://opensource.org/licenses/BSD-3-Clause>`_.
For the full license text, see the repository file:
`LICENSE <https://github.com/theislab/idtrack/blob/development/LICENSE>`_.
