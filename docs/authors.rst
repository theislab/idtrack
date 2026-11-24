=======
Credits
=======

idtrack is developed and maintained by researchers committed to reproducible
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

Contributors are credited through the project's Git history; the list is on GitHub:
`github.com/theislab/idtrack/graphs/contributors <https://github.com/theislab/idtrack/graphs/contributors>`_. The
:doc:`contributing` guide says how to contribute.


Acknowledgments
---------------

idtrack builds upon the work of many open-source projects and biological databases.
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
      - HUGO Gene Nomenclature Committee - official human gene symbols
    * - `VGNC <https://vertebrate.genenames.org/>`_
      - Vertebrate Gene Nomenclature Committee - gene symbols of other vertebrates (pig, the macaques)
    * - `MGI <http://www.informatics.jax.org/>`_
      - Mouse Genome Informatics - mouse gene nomenclature and annotation
    * - `UniProt <https://www.uniprot.org/>`_
      - Universal Protein Resource - protein sequence and annotation
    * - `NCBI Gene <https://www.ncbi.nlm.nih.gov/gene>`_ and `RefSeq <https://www.ncbi.nlm.nih.gov/refseq/>`_
      - National Center for Biotechnology Information - gene ids, official symbols, reference sequences
    * - `miRBase <https://www.mirbase.org/>`_ and `Rfam <https://rfam.org/>`_
      - microRNA precursors and RNA families
    * - `CCDS <https://www.ncbi.nlm.nih.gov/projects/CCDS/>`_, Havana and Vega
      - Consensus coding sequences and the manual annotation of Ensembl/GENCODE

Software Dependencies
~~~~~~~~~~~~~~~~~~~~~

idtrack relies on excellent open-source software:

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
    * - `h5py <https://www.h5py.org/>`_
      - HDF5 file format support
    * - `requests <https://requests.readthedocs.io/>`_
      - HTTP library for database downloads
    * - `PyMySQL <https://pymysql.readthedocs.io/>`_
      - Ensembl's public MySQL server
    * - `PyYAML <https://pyyaml.org/>`_
      - The externals configuration files
    * - `tqdm <https://tqdm.github.io/>`_
      - Progress bars
    * - `AnnData <https://anndata.scverse.org/>`_ and `SciPy <https://scipy.org/>`_
      - Annotated data matrices for single-cell analysis (the ``harmonize`` extra)

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


Citing idtrack
--------------

If you use idtrack in your research, please cite:

.. code-block:: bibtex

    @software{idtrack,
      author = {Inecik, Kemal},
      title = {IDTrack: Cross-Temporal and Cross-Database Biological Identifier Mapping},
      url = {https://github.com/theislab/idtrack},
      version = {0.1.2},
      year = {2026}
    }

Cite the version you used: ``version`` is the value of ``idtrack.__version__`` (or ``idtrack --version``).


License
-------

idtrack is released under the `BSD 3-Clause License <https://opensource.org/licenses/BSD-3-Clause>`_.
For the full license text, see the repository file:
`LICENSE <https://github.com/theislab/idtrack/blob/main/LICENSE>`_.
