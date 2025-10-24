Available Models
================

fastcpd-python supports 14 different model families for change point detection, categorized into parametric and nonparametric methods.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 20 30 20 30

   * - Model Family
     - Description
     - Implementation
     - Key Use Cases
   * - **mean**
     - Mean change detection
     - C++ (fast)
     - Time series monitoring, sensor data
   * - **variance**
     - Variance/covariance change
     - C++ (fast)
     - Volatility detection, quality control
   * - **meanvariance**
     - Combined mean & variance
     - C++ (fast)
     - Comprehensive distributional changes
   * - **binomial**
     - Logistic regression
     - Python + Numba
     - Binary classification, success rates
   * - **poisson**
     - Poisson regression
     - Python + Numba
     - Count data, event rates
   * - **lm**
     - Linear regression
     - Python
     - Continuous outcomes, trend changes
   * - **lasso**
     - L1-penalized regression
     - Python
     - High-dimensional data, sparse changes
   * - **ar**
     - AR(p) autoregressive
     - Python
     - Univariate time series
   * - **var**
     - VAR(p) vector autoregressive
     - Python
     - Multivariate time series
   * - **arma**
     - ARMA(p,q) time series
     - Python (statsmodels)
     - Stationary time series with MA component
   * - **garch**
     - GARCH(p,q) volatility
     - Python (arch)
     - Financial volatility, heteroskedasticity
   * - **rank**
     - Rank-based (nonparametric)
     - Python
     - Distribution-free detection
   * - **rbf**
     - RBF kernel (nonparametric)
     - Python (RFF)
     - Complex distributional changes

Parametric Models
-----------------

Mean and Variance Models
~~~~~~~~~~~~~~~~~~~~~~~~

**Mean Change Detection**

Detects changes in the mean of univariate or multivariate data.

.. code-block:: python

   from fastcpd.segmentation import mean

   # Univariate
   result = mean(data, beta="MBIC")

   # Multivariate (3D)
   data_3d = np.random.randn(500, 3)
   result = mean(data_3d, beta="MBIC")

**Cost Function:**

The minimum negative log-likelihood of the multivariate Gaussian model:

.. math::

   C(x_{s:t})=\frac{1}{2}\sum_{i=s}^{t}(x_{i}-\overline{x}_{s:t})^{\top}\hat{\Sigma}^{-1}(x_{i}-\overline{x}_{s:t})+\frac{(t-s+1)d}{2}\log(2\pi)+\frac{t-s+1}{2}\log(|\hat{\Sigma}|)

where :math:`\overline{x}_{s:t} = (t-s+1)^{-1}\sum_{i=s}^{t}x_{i}` is the segment mean and :math:`\hat{\Sigma}` is the Rice estimator for the covariance matrix.

**Parameters:**

- None required (dimensionality inferred from data)

**Implementation:** C++ core for computational efficiency

---

**Variance Change Detection**

Detects changes in variance/covariance structure.

.. code-block:: python

   from fastcpd.segmentation import variance

   result = variance(data, beta="MBIC")

**Cost Function:**

Detects changes in the covariance matrix, assuming a fixed but unknown mean vector:

.. math::

   C(x_{s:t})=\frac{t-s+1}{2}\left\{d\log(2\pi)+d+\log\left(\left|\frac{1}{t-s+1}\sum_{i=s}^{t}(x_{i}-\overline{x}_{1:T})(x_{i}-\overline{x}_{1:T})^{\top}\right|\right)\right\}

where :math:`\overline{x}_{1:T}` is the mean estimated over the entire data sequence (not just the segment).

---

**Mean and Variance Change Detection**

Detects changes in both mean and variance simultaneously.

.. code-block:: python

   from fastcpd.segmentation import meanvariance

   result = meanvariance(data, beta="MBIC")

**Cost Function:**

Detects changes in both the mean vector and covariance matrix:

.. math::

   C(x_{s:t})=\frac{t-s+1}{2}\left\{d\log(2\pi)+d+\log\left(\left|\frac{1}{t-s+1}\sum_{i=s}^{t}(x_{i}-\overline{x}_{s:t})(x_{i}-\overline{x}_{s:t})^{\top}\right|\right)\right\}

where :math:`\overline{x}_{s:t}` is the mean estimated within the segment.

GLM Models
~~~~~~~~~~

**Logistic Regression (Binomial)**

Detects changes in logistic regression coefficients (binary outcomes).

.. code-block:: python

   from fastcpd.segmentation import logistic_regression

   # Data format: first column = binary response (0/1)
   #               remaining columns = predictors
   data = np.column_stack([y, X])
   result = logistic_regression(data, beta="MBIC")

**Model:**

.. math::

   \log\left(\frac{p}{1-p}\right) = X\beta

**Cost Function:**

For GLM with known dispersion parameter :math:`\phi`:

.. math::

   C(z_{s:t})=\min_{\theta}-\sum_{i=s}^{t}\left\{\frac{y_{i}\gamma_{i}-b(\gamma_{i})}{w^{-1}\phi}+c(y_{i},\phi)\right\}

where :math:`\gamma_i` is the canonical parameter and :math:`b(\cdot)`, :math:`c(\cdot)` are functions specific to the exponential family distribution.

**Dependencies:** scikit-learn (LogisticRegression backend)

**Acceleration:** Optional Numba JIT compilation

**Options:**

.. code-block:: python

   # Control PELT/SeGD interpolation
   result = logistic_regression(data, vanilla_percentage=0.5)

---

**Poisson Regression**

Detects changes in Poisson regression coefficients (count data).

.. code-block:: python

   from fastcpd.segmentation import poisson_regression

   # First column = count response
   # Remaining columns = predictors
   data = np.column_stack([y, X])
   result = poisson_regression(data, beta="MBIC")

**Model:**

.. math::

   \log(\lambda) = X\beta

**Cost Function:**

Uses the GLM cost function for the Poisson distribution (same as logistic regression but with Poisson-specific :math:`b(\cdot)` and :math:`c(\cdot)` functions).

**Dependencies:** scikit-learn (uses PoissonRegressor with LBFGS)

Regression Models
~~~~~~~~~~~~~~~~~

**Linear Regression**

Detects changes in linear regression coefficients.

.. code-block:: python

   from fastcpd.segmentation import linear_regression

   # First column = continuous response
   # Remaining columns = predictors
   data = np.column_stack([y, X])
   result = linear_regression(data, beta="MBIC")

**Model:**

.. math::

   y = X\beta + \epsilon, \quad \epsilon \sim N(0, \sigma^2)

**Cost Function:**

The minimum negative log-likelihood:

.. math::

   C(z_{s:t})=\sum_{i=s}^{t}\left(\frac{1}{2}\log(2\pi\hat{\sigma}^{2})+\frac{(y_{i}-x_{i}^{\top}\hat{\theta})^{2}}{2\hat{\sigma}^{2}}\right)

where :math:`\hat{\theta} = \left(\sum_{i=s}^{t}x_{i}x_{i}^{\top}\right)^{-1}\sum_{i=s}^{t}x_{i}y_{i}` is the least squares estimator and :math:`\hat{\sigma}^{2}` is the generalized Rice estimator (G-Rice).

**Implementation:** Python with sklearn backend

---

**LASSO Regression**

Detects changes in sparse regression coefficients with L1 penalization.

.. code-block:: python

   from fastcpd.segmentation import lasso

   # Manual alpha (regularization strength)
   result = lasso(data, alpha=0.1, beta="MBIC")

   # Cross-validation to select alpha (slower but automatic)
   result = lasso(data, cv=True, beta="MBIC")

**Cost Function:**

The :math:`L_1`-penalized least squares objective:

.. math::

   C(z_{s:t})=\min_{\theta\in\Theta}\frac{1}{2}\sum_{i=s}^{t}(y_{i}-x_{i}^{\top}\theta)^{2}+\lambda_{s:t}\sum_{j=1}^{d}|\theta_{j}|

where the penalty parameter is:

.. math::

   \lambda_{s:t}=\hat{\sigma}\sqrt{2\log(d)/(t-s+1)}

**Use Cases:**

- High-dimensional data (p >> n)
- Sparse coefficient changes
- Feature selection at change points

Time Series Models
~~~~~~~~~~~~~~~~~~

**AR(p) - Autoregressive Model**

Detects changes in AR model coefficients.

.. code-block:: python

   from fastcpd.segmentation import ar

   # AR(2) model
   result = ar(data, p=2, beta="MBIC")

**Model:**

.. math::

   y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t

**Cost Function:**

Identical to standard linear regression cost function, treating AR as a linear model:

.. math::

   C(z_{s:t})=\min_{\theta}\left(\frac{1}{2}\log(2\pi\hat{\sigma}^{2})+\sum_{i=s}^{t}\frac{(x_{i}-x_{i-1:i-p}^{\top}\theta)^{2}}{2\hat{\sigma}^{2}}\right)

where :math:`\hat{\sigma}^{2}` is the generalized Rice estimator.

**Parameters:**

- ``p``: Order of AR model

---

**VAR(p) - Vector Autoregressive Model**

Detects changes in multivariate AR model coefficients.

.. code-block:: python

   from fastcpd.segmentation import var

   # VAR(1) for 3D time series
   data_3d = np.random.randn(500, 3)
   result = var(data_3d, p=1, beta="MBIC")

**Model:**

.. math::

   \mathbf{y}_t = \Phi_1 \mathbf{y}_{t-1} + \cdots + \Phi_p \mathbf{y}_{t-p} + \boldsymbol{\epsilon}_t

where :math:`\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \Sigma)` with :math:`\Sigma` assumed constant.

**Cost Function:**

The minimum negative log-likelihood:

.. math::

   C(z_{s:t})=\min_{A_{1},...,A_{p}}\frac{1}{2}\sum_{i=s}^{t}\left(x_{i}-\sum_{j=1}^{p}A_{j}x_{t-j}\right)^{\top}\hat{\Sigma}^{-1}\left(x_{i}-\sum_{j=1}^{p}A_{j}x_{t-j}\right) +\frac{t-s+1}{2}\{d\log(2\pi)+\log(|\hat{\Sigma}|)\}

where :math:`\hat{\Sigma}` is an extension of the G-Rice estimator for multi-response linear models.

---

**ARMA(p,q) - Autoregressive Moving Average**

Detects changes in ARMA model parameters.

.. code-block:: python

   from fastcpd.segmentation import arma

   # ARMA(1,1) model
   result = arma(data, p=1, q=1, beta="MBIC")

**Dependencies:** ``pip install statsmodels``

**Model:**

.. math::

   y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}

**Cost Function:**

Two approaches are available:

1. **Coefficient changes only (fixed** :math:`\sigma^2` **):**

.. math::

   C(x_{s:t})=\min_{\phi,\psi}\frac{t-s+1}{2}\{\log(2\pi)+\log(\hat{\sigma}^{2})\}+\frac{1}{2\hat{\sigma}^{2}}\sum_{i=s}^{t}(x_{i}-\phi^{\top}x_{i-1:i-p}-\psi^{\top}\epsilon_{i-1:i-q})^{2}

2. **Coefficient and variance changes (for SeGD):**

.. math::

   C(x_{s:t})=\min_{\phi,\psi,\sigma^{2}}\frac{t-s+1}{2}\{\log(2\pi)+\log(\sigma^{2})\}+\frac{1}{2\sigma^{2}}\sum_{i=s}^{t}(x_{i}-\phi^{\top}x_{i-1:i-p}-\psi^{\top}\epsilon_{i-1:i-q})^{2}

**Parameters:**

- ``p``: AR order
- ``q``: MA order

---

**GARCH(p,q) - Generalized Autoregressive Conditional Heteroskedasticity**

Detects changes in volatility model parameters.

.. code-block:: python

   from fastcpd.segmentation import garch

   # GARCH(1,1) model
   result = garch(data, p=1, q=1, beta=2.0)

**Dependencies:** ``pip install arch``

**Model:**

.. math::

   x_{t}=\sigma_{t}\epsilon_{t}

.. math::

   \sigma_t^2 = \omega + \alpha_1 x_{t-1}^2 + \cdots + \alpha_p x_{t-p}^2 + \beta_1 \sigma_{t-1}^2 + \cdots + \beta_q \sigma_{t-q}^2

**Use Cases:**

- Financial volatility
- Regime changes in variance
- Risk modeling

Nonparametric Models
--------------------

**Rank-Based Detection**

Distribution-free change detection using rank statistics.

.. code-block:: python

   from fastcpd.segmentation import rank

   result = rank(data, beta=50.0)

**Features:**

- No distributional assumptions
- Robust to outliers
- Monotonic-invariant

**Cost Function:**

Based on Mann-Whitney U statistic comparing ranks in different segments.

**Use Cases:**

- Unknown or complex distributions
- Outlier-prone data
- Ordinal data

---

**RBF Kernel Detection**

Detects distributional changes using Gaussian kernel methods via Random Fourier Features.

.. code-block:: python

   from fastcpd.segmentation import rbf

   result = rbf(
       data,
       beta=30.0,
       gamma=None,         # Auto: 1/median^2 of pairwise distances
       feature_dim=256,    # RFF dimension
       seed=0
   )

**Features:**

- Detects complex distributional changes
- Multivariate support
- Efficient via Random Fourier Features (RFF)

**Cost Function:**

.. math::

   C(y_{s:t}) = \sum_{i=s}^{t} k(y_i, y_i) - \frac{2}{t-s+1} \sum_{i,j=s}^{t} k(y_i, y_j)

where :math:`k(x,y) = \exp(-\gamma \|x-y\|^2)` is the RBF kernel. This measures variance in the kernel-embedded space.

**Parameters:**

- ``gamma``: RBF bandwidth (default: 1/median² of pairwise distances)
- ``feature_dim``: Random Fourier Feature dimension (default: 256)
- ``seed``: Random seed for reproducibility

Model Selection Guide
---------------------

Choosing the Right Model
~~~~~~~~~~~~~~~~~~~~~~~~

**By Data Type:**

.. list-table::
   :header-rows: 1

   * - Data Type
     - Recommended Models
   * - Continuous, univariate
     - mean, variance, meanvariance
   * - Continuous, multivariate
     - mean, meanvariance, var
   * - Binary (0/1)
     - binomial (logistic regression)
   * - Count (0,1,2,...)
     - poisson
   * - With predictors
     - lm, lasso, binomial, poisson
   * - Time series (stationary)
     - ar, arma
   * - Time series (multivariate)
     - var
   * - Volatility/variance changes
     - garch, variance
   * - Unknown distribution
     - rank, rbf

**By Goal:**

.. list-table::
   :header-rows: 1

   * - Goal
     - Recommended Models
   * - Fast detection on large data
     - mean, variance (C++ implementation)
   * - High accuracy, any speed
     - Choose model matching data generation process
   * - Robust to outliers
     - rank, rbf
   * - High-dimensional predictors
     - lasso
   * - Interpretability
     - lm, ar, mean

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

**Speed (relative to n=1000):**

.. list-table::
   :header-rows: 1

   * - Model
     - Speed Rating
     - Notes
   * - mean, variance, meanvariance
     - Very Fast
     - C++ implementation, fastest
   * - binomial, poisson (with Numba)
     - Fast
     - 7-14x faster with Numba
   * - binomial, poisson (no Numba)
     - Moderate
     - Still competitive
   * - lm, lasso
     - Moderate
     - Optimized sklearn backend
   * - ar, var
     - Moderate
     - Pure Python
   * - arma, garch
     - Slow
     - Uses statsmodels/arch
   * - rank, rbf
     - Moderate
     - Nonparametric methods

**Accuracy:**

All models provide accurate detection when properly configured. Choose the model that matches your data generation process for best results.

Common Parameters
-----------------

All Models Support
~~~~~~~~~~~~~~~~~~

- ``beta``: Penalty for change points ("BIC", "MBIC", "MDL", or numeric)
- ``trim``: Proportion to trim from boundaries (default: 0.02)
- ``min_segment_length``: Minimum distance between change points

GLM Models (binomial, poisson) Also Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``vanilla_percentage``: PELT/SeGD interpolation (0.0 to 1.0 or 'auto')

Example: Comparing Models
-------------------------

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import mean, variance, rank, rbf

   # Generate data
   np.random.seed(42)
   data = np.concatenate([
       np.random.normal(0, 1, 200),
       np.random.normal(3, 1, 200)
   ])

   # Try different models
   models = {
       'mean': mean(data, beta="MBIC"),
       'variance': variance(data, beta="MBIC"),
       'rank': rank(data, beta=50.0),
       'rbf': rbf(data, beta=30.0)
   }

   # Compare results
   for name, result in models.items():
       print(f"{name:12s}: {result.cp_set}")

Next Steps
----------

- :doc:`quickstart` - Try models on example data
- :doc:`evaluation` - Evaluate detection performance
- :doc:`../advanced/algorithms` - Learn about PELT and SeGD algorithms
