Installation
============

The package is available on Test PyPI and can be built from source.

From Test PyPI
--------------

The package is currently available on Test PyPI as ``pyfastcpd``:

.. code-block:: bash

   # Install Armadillo first (required for building C++ extension)
   # macOS:
   brew install armadillo

   # Ubuntu/Debian:
   sudo apt-get install libarmadillo-dev

   # Then install pyfastcpd
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyfastcpd

.. note::
   Test PyPI currently only has source distribution (no pre-built wheels). You must install Armadillo before running pip install.

From Source
-----------

For development or if you want the latest features:

**System Requirements:**

- Python ≥ 3.8
- C++17 compiler
- **Armadillo library** (required for C++ compilation)
- BLAS/LAPACK

**Installation Steps:**

.. code-block:: bash

   # 1. Install Armadillo (required!)
   # macOS:
   brew install armadillo

   # Ubuntu/Debian:
   sudo apt-get install libarmadillo-dev

   # 2. Clone repository
   git clone https://github.com/zhangxiany-tamu/fastcpd_Python.git
   cd fastcpd_Python

   # 3. Install with editable mode
   pip install -e .

   # 4. Optional extras for examples/benchmarks/time series
   pip install -e .[dev,test,benchmark,timeseries]

   # 5. Optional: Install Numba for 7-14x GLM speedup
   pip install numba

**Supported Platforms:**

- Linux (x86_64, aarch64)
- macOS (Intel x86_64, Apple Silicon arm64)
- Windows (experimental, source builds only)

Optional Dependencies
---------------------

For Additional Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Numba for 7-14x speedup on GLM models (binomial, poisson)
   pip install numba

For Time Series Models
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # ARMA models
   pip install statsmodels

   # GARCH models
   pip install arch

   # Or install both with:
   pip install pyfastcpd[timeseries]

For Visualization
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install matplotlib

For Development
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install all development dependencies
   pip install pyfastcpd[dev]

   # This includes:
   # - pytest, pytest-cov (testing)
   # - black, ruff (formatting/linting)
   # - mypy (type checking)

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import fastcpd
   from fastcpd.segmentation import mean
   import numpy as np

   # Check version
   print(f"fastcpd version: {fastcpd.__version__}")

   # Run simple test
   data = np.concatenate([
       np.random.normal(0, 1, 100),
       np.random.normal(5, 1, 100)
   ])
   result = mean(data)
   print(f"Detected change points: {result.cp_set}")

Expected output:

.. code-block:: text

   fastcpd version: 0.18.3
   Detected change points: [100]

Troubleshooting
---------------

C++ Extension Import Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see ``ImportError: C++ extension '_fastcpd_impl' is not available``:

1. Make sure Armadillo is installed:

   .. code-block:: bash

      # macOS
      brew list armadillo

      # Ubuntu/Debian
      dpkg -l | grep armadillo

2. Reinstall the package:

   .. code-block:: bash

      pip install --force-reinstall pyfastcpd

CMake Error: Could NOT find Armadillo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to install the Armadillo library before building:

.. code-block:: bash

   # macOS
   brew install armadillo

   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install libarmadillo-dev

Compilation Errors on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Windows support is experimental. You'll need:

- Visual Studio 2019 or later with C++ support
- Armadillo compiled for Windows (see `Armadillo documentation <http://arma.sourceforge.net/>`_)

For easier installation on Windows, wait for pre-built wheels to be available on PyPI.

Next Steps
----------

- :doc:`quickstart` - Learn basic usage
- :doc:`models` - Explore available models
- :doc:`tutorials` - Follow step-by-step tutorials
