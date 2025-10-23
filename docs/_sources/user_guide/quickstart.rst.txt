Quick Start
===========

Basic usage examples for common detection scenarios.

Basic Change Point Detection
-----------------------------

Mean Change Detection
~~~~~~~~~~~~~~~~~~~~~

Detect changes in the mean of a time series:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from fastcpd.segmentation import mean

   # Generate data with mean changes
   np.random.seed(42)
   data = np.concatenate([
       np.random.normal(0, 1, 100),
       np.random.normal(5, 1, 100),
       np.random.normal(2, 1, 100)
   ])

   # Detect change points
   result = mean(data, beta="MBIC")
   print(f"Detected change points: {result.cp_set}")
   # Output: [100 200]

   # Plot results
   plt.figure(figsize=(12, 4))
   plt.plot(data, label='Data', linewidth=1)
   for cp in result.cp_set:
       plt.axvline(cp, color='red', linestyle='--', linewidth=2)
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.title('Mean Change Detection')
   plt.legend()
   plt.show()

Variance Change Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect changes in variance:

.. code-block:: python

   from fastcpd.segmentation import variance

   # Generate data with variance changes
   np.random.seed(42)
   data = np.concatenate([
       np.random.normal(0, 0.5, 100),
       np.random.normal(0, 2.5, 100),
       np.random.normal(0, 0.5, 100)
   ])

   # Detect change points
   result = variance(data, beta="MBIC")
   print(f"Detected change points: {result.cp_set}")

Multivariate Mean Change
~~~~~~~~~~~~~~~~~~~~~~~~~

Detect changes in multivariate data:

.. code-block:: python

   from fastcpd.segmentation import mean

   # Generate 3D data with mean change
   np.random.seed(42)
   data = np.concatenate([
       np.random.multivariate_normal([0, 0, 0], np.eye(3), 150),
       np.random.multivariate_normal([5, 5, 5], np.eye(3), 150)
   ])

   result = mean(data, beta="MBIC")
   print(f"Detected change points: {result.cp_set}")
   print(f"Data shape: {data.shape}")

Time Series Models
------------------

AR Model Change Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect changes in autoregressive model parameters:

.. code-block:: python

   from fastcpd.segmentation import ar

   # Generate AR(1) data with coefficient change
   np.random.seed(100)

   # First segment: AR(1) with φ = 0.8
   data1 = np.zeros(150)
   for i in range(1, 150):
       data1[i] = 0.8 * data1[i-1] + np.random.normal(0, 0.8)

   # Second segment: AR(1) with φ = -0.7
   data2 = np.zeros(150)
   for i in range(1, 150):
       data2[i] = -0.7 * data2[i-1] + np.random.normal(0, 0.8)

   data = np.concatenate([data1, data2])

   # Detect change points
   result = ar(data, p=1, beta="MBIC")
   print(f"Detected change points: {result.cp_set}")

ARMA and GARCH Models
~~~~~~~~~~~~~~~~~~~~~

For ARMA and GARCH models, install required dependencies first:

.. code-block:: bash

   pip install statsmodels arch

Then use them:

.. code-block:: python

   from fastcpd.segmentation import arma, garch

   # ARMA(1,1) model
   result_arma = arma(data, p=1, q=1, beta="MBIC")

   # GARCH(1,1) model
   result_garch = garch(data, p=1, q=1, beta=2.0)

Regression Models
-----------------

Linear Regression
~~~~~~~~~~~~~~~~~

Detect changes in linear regression coefficients:

.. code-block:: python

   from fastcpd.segmentation import linear_regression

   # Simulate linear regression with change
   n = 500
   X = np.random.randn(n, 2)

   # First segment: y = 2*x1 + 3*x2 + noise
   y1 = 2 * X[:n//2, 0] + 3 * X[:n//2, 1] + np.random.randn(n//2)

   # Second segment: y = -1*x1 + 5*x2 + noise
   y2 = -1 * X[n//2:, 0] + 5 * X[n//2:, 1] + np.random.randn(n//2)

   y = np.concatenate([y1, y2])

   # Combine response and predictors (first column = response)
   data = np.column_stack([y, X])

   result = linear_regression(data, beta="MBIC")
   print(f"Detected change points: {result.cp_set}")

Logistic Regression
~~~~~~~~~~~~~~~~~~~

Detect changes in logistic regression parameters:

.. code-block:: python

   from fastcpd.segmentation import logistic_regression

   # Simulate logistic regression with change
   n = 500
   X = np.random.randn(n, 2)

   # First segment: strong positive effect
   prob1 = 1 / (1 + np.exp(-(2*X[:n//2, 0] + 3*X[:n//2, 1])))
   y1 = (np.random.rand(n//2) < prob1).astype(float)

   # Second segment: negative effect
   prob2 = 1 / (1 + np.exp(-(-1*X[n//2:, 0] + 2*X[n//2:, 1])))
   y2 = (np.random.rand(n//2) < prob2).astype(float)

   y = np.concatenate([y1, y2])
   data = np.column_stack([y, X])

   result = logistic_regression(data, beta="MBIC")
   print(f"Detected change points: {result.cp_set}")

LASSO Regression
~~~~~~~~~~~~~~~~

Detect changes in sparse regression:

.. code-block:: python

   from fastcpd.segmentation import lasso

   # Simulate sparse regression with change
   n = 500
   p = 20  # 20 predictors
   X = np.random.randn(n, p)

   # First segment: only first 3 features matter
   y1 = 2*X[:n//2, 0] + 3*X[:n//2, 1] - 1.5*X[:n//2, 2] + np.random.randn(n//2)

   # Second segment: different sparse coefficients
   y2 = -1*X[n//2:, 5] + 2*X[n//2:, 8] + np.random.randn(n//2)

   y = np.concatenate([y1, y2])
   data = np.column_stack([y, X])

   result = lasso(data, alpha=0.1, beta="MBIC")
   print(f"Detected change points: {result.cp_set}")

Nonparametric Methods
---------------------

Rank-Based Detection
~~~~~~~~~~~~~~~~~~~~

Distribution-free change detection:

.. code-block:: python

   from fastcpd.segmentation import rank

   # Works with any distribution
   data = np.concatenate([
       np.random.exponential(1.0, 200),
       np.random.exponential(3.0, 200)
   ])

   result = rank(data, beta=50.0)
   print(f"Detected change points: {result.cp_set}")

RBF Kernel Detection
~~~~~~~~~~~~~~~~~~~~

Detect distributional changes using kernel methods:

.. code-block:: python

   from fastcpd.segmentation import rbf

   result = rbf(data, beta=30.0)
   print(f"Detected change points: {result.cp_set}")

Working with Results
--------------------

The FastcpdResult Object
~~~~~~~~~~~~~~~~~~~~~~~~

All detection functions return a ``FastcpdResult`` object:

.. code-block:: python

   result = mean(data, beta="MBIC")

   # Access detected change points
   print(result.cp_set)          # Final change points
   print(result.raw_cp_set)      # Raw change points (before post-processing)

   # Access additional information
   print(result.cost_values)     # Cost values
   print(result.residuals)       # Model residuals
   print(result.thetas)          # Parameter estimates
   print(result.data)            # Original data
   print(result.family)          # Model family used

Plotting Results
~~~~~~~~~~~~~~~~

Use the built-in plot method:

.. code-block:: python

   result = mean(data, beta="MBIC")
   fig, ax = result.plot()
   plt.show()

Or use the visualization module for more options:

.. code-block:: python

   from fastcpd.visualization import plot_detection

   plot_detection(
       data=data,
       true_cps=[100, 200],
       pred_cps=result.cp_set.tolist(),
       title="Mean Change Detection"
   )

Choosing the Penalty (beta)
----------------------------

The ``beta`` parameter controls the penalty for adding change points. Higher values → fewer change points.

**String Options:**

- ``"MBIC"`` (default): Modified BIC = (p + 2) * log(n) / 2
- ``"BIC"``: Bayesian Information Criterion = p * log(n) / 2
- ``"MDL"``: Minimum Description Length = (p / 2) * log(n)

**Numeric Values:**

.. code-block:: python

   # Use specific penalty value
   result = mean(data, beta=10.0)  # More change points
   result = mean(data, beta=50.0)  # Fewer change points

**Comparison:**

.. code-block:: python

   for beta_val in ["BIC", "MBIC", "MDL", 5.0, 20.0]:
       result = mean(data, beta=beta_val)
       print(f"Beta={beta_val}: {len(result.cp_set)} change points")

Advanced Options
----------------

Minimum Segment Length
~~~~~~~~~~~~~~~~~~~~~~

Enforce minimum distance between change points:

.. code-block:: python

   result = mean(data, beta="MBIC", min_segment_length=30)

Vanilla Percentage (GLM Models Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control PELT/SeGD interpolation:

.. code-block:: python

   # 0.0 = pure SeGD (fast), 1.0 = pure PELT (accurate)
   result = logistic_regression(data, vanilla_percentage=0.5)

   # 'auto' = adaptive based on data size
   result = logistic_regression(data, vanilla_percentage='auto')

Next Steps
----------

- :doc:`models` - Learn about all available models
- :doc:`evaluation` - Evaluate detection performance
- :doc:`visualization` - Create publication-quality plots
- :doc:`tutorials` - Follow detailed tutorials
