Installation
============

This page describes how to install SymOp for both regular use, development and documentation.


Requirements
------------

SymOp requires:

- ``Python 3.11`` or newer
- ``pip`` for package installation

The core package depends on:

- ``numpy``
- ``scipy``
- ``matplotlib``

Install from PyPI
-----------------

Install the latest released version from PyPI with:

.. code-block:: bash

   pip install symop

This installs SymOp together with its required runtime dependencies.

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade symop

Install from GitHub
-------------------

To install the latest version directly from the GitHub repository:

.. code-block:: bash

   pip install git+https://github.com/tqsd/symop.git

This is useful if you want the newest development version before a release is published on PyPI.

Install in a Virtual Environment
--------------------------------

It is recommended to install SymOp inside a virtual environment.

Create and activate one with:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

Then install SymOp:

.. code-block:: bash

   pip install symop

On Windows, activate the virtual environment with:

.. code-block:: bash

   .venv\Scripts\activate

Development Installation
------------------------

To work on SymOp locally, clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/tqsd/symop.git
   cd symop
   pip install -e .[dev]

This installs SymOp together with the development dependencies, including:

- ``mypy``
- ``pytest``
- ``pytest-cov``
- ``pre-commit``
- ``ruff``
- ``coverage``
- ``tox``

Documentation Installation
--------------------------

To build the documentation locally, install the documentation dependencies:

.. code-block:: bash

   pip install -e .[docs]

If you also want the development tools at the same time, install both extras:

.. code-block:: bash

   pip install -e .[dev,docs]

Build the Documentation
-----------------------

After installing the documentation dependencies, build the HTML documentation from the repository root:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

The generated documentation will be available in:

.. code-block:: text

   docs/_build/html

Verify the Installation
-----------------------

You can verify that SymOp is installed by importing it in Python:

.. code-block:: bash

   python -c "import symop; print(symop.__name__)"

If the installation succeeds, this should print:

.. code-block:: text

   symop
