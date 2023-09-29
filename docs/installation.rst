********************
Getting started
********************

For installing EDAspy from Pypi execute the following command using pip:

.. code-block:: bash

    pip install EDAspy

Build from Source
=================

Prerequisites
-------------

- Python 3.6, 3.7, 3.8 or 3.9.
- Pybnesian, numpy, pandas, pgmpy.

Building
--------

Clone the repository:

.. code-block:: bash

    git clone https://github.com/VicentePerezSoloviev/EDAspy.git
    cd EDAspy
    git checkout v1.1.2  # You can checkout a specific version if you want
    python setup.py install

Testing
=======

The library contains tests that can be executed using `pytest <https://docs.pytest.org/>`_. Install it using
pip:

.. code-block:: bash

    pip install pytest

Run the tests with:

.. code-block:: bash

    pytest
