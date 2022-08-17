.. highlight:: shell

============
Installation
============


Stable release
--------------

To install idtrack, run this command in your terminal:

.. code-block:: console

    $ pip install idtrack

This is the preferred method to install idtrack, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

If you use an Apple Silicon processor, you have to be sure that you have the ``HDF5`` installation already:

.. code-block:: console

    $ conda install hdf5

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for idtrack can be downloaded from the `Github repo`_.
Please note that you require `poetry`_ to be installed.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/theislab/idtrack

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/theislab/idtrack/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ make install


.. _Github repo: https://github.com/theislab/idtrack
.. _tarball: https://github.com/theislab/idtrack/tarball/master
.. _poetry: https://python-poetry.org/
