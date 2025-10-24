Comparison with Other Packages
===============================

Quick comparison of fastcpd-python with other change point detection libraries.

Python Packages
---------------

**ruptures** - Comprehensive offline change point detection

- Pure Python (no C++ compilation)
- Multiple algorithms: PELT, Binary Segmentation, Bottom-Up, Window-based
- Mature codebase with large user base
- Extensive documentation

**fastcpd-python** - This package

- GLM support (Binomial, Poisson)
- Time series models (AR, ARMA, VAR, GARCH)
- SeGD algorithm for large datasets
- Requires C++ (Armadillo)

R Packages
----------

**changepoint** - Established R package

- Mature, well-tested
- CRAN distribution
- Classical methods (PELT, BinSeg)

**fastcpd** - Original R implementation

- PELT + SeGD algorithms
- CRAN distribution
- Native R integration

Feature Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15

   * - Feature
     - fastcpd-py
     - ruptures
     - R fastcpd
     - changepoint
   * - PELT
     - Yes
     - Yes
     - Yes
     - Yes
   * - Binary Segmentation
     - No
     - Yes
     - No
     - Yes
   * - SeGD
     - Yes
     - No
     - Yes
     - No
   * - Mean/Variance
     - Yes
     - Yes
     - Yes
     - Yes
   * - GLM
     - Yes
     - No
     - Yes
     - No
   * - Time Series
     - Yes
     - No
     - Yes
     - No
   * - Nonparametric
     - Yes
     - Yes
     - Yes
     - No
   * - Installation
     - pip + C++
     - pip only
     - CRAN
     - CRAN

When to Use Each
----------------

**Use fastcpd-python** for:

- GLM or time series models in Python
- Large datasets (SeGD algorithm)
- Comprehensive evaluation metrics

**Use ruptures** for:

- Pure Python without C++ dependencies
- Binary segmentation or bottom-up algorithms
- Established Python package

**Use R fastcpd** for:

- R ecosystem integration
- CRAN distribution preference

**Use changepoint** for:

- Mature R package
- Classical methods in R

Migration Examples
------------------

From ruptures
~~~~~~~~~~~~~

.. code-block:: python

   # ruptures
   import ruptures as rpt
   algo = rpt.Pelt(model="rbf").fit(signal)
   result = algo.predict(pen=10)

   # fastcpd-python
   from fastcpd.segmentation import rbf
   result = rbf(signal, beta=10.0)
   change_points = result.cp_set

From R fastcpd
~~~~~~~~~~~~~~

.. code-block:: r

   # R
   library(fastcpd)
   result <- fastcpd.mean(data, beta="MBIC")

.. code-block:: python

   # Python
   from fastcpd.segmentation import mean
   result = mean(data, beta="MBIC")

Summary
-------

Choose based on:

1. Programming language (Python vs R)
2. Model requirements (standard vs GLM vs time series)
3. Installation constraints (pure Python vs C++)
4. Algorithm needs (PELT, SeGD, Binary Segmentation)
