Datasets API
============

Dataset generation utilities for benchmarking and testing.

Overview
--------

.. automodule:: fastcpd.datasets
   :members:
   :undoc-members:

Dataset Generators
------------------

Parametric Datasets
~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.datasets.make_mean_change
.. autofunction:: fastcpd.datasets.make_variance_change

Regression Datasets
~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.datasets.make_regression_change
.. autofunction:: fastcpd.datasets.make_glm_change

Time Series Datasets
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.datasets.make_arma_change
.. autofunction:: fastcpd.datasets.make_garch_change

Example Usage
-------------

Basic Dataset Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.datasets import make_mean_change
   import numpy as np

   # Generate dataset with 3 change points
   np.random.seed(42)
   data_dict = make_mean_change(
       n_samples=600,
       n_changepoints=3,
       seed=42
   )

   # Access components
   print(f"Data shape: {data_dict['data'].shape}")
   print(f"Change points: {data_dict['changepoints']}")

GLM Dataset
~~~~~~~~~~~

.. code-block:: python

   from fastcpd.datasets import make_glm_change

   # Logistic regression dataset
   data_dict = make_glm_change(
       n_samples=800,
       n_predictors=5,
       n_changepoints=2,
       family='binomial',
       seed=42
   )

   # Extract response and predictors
   y = data_dict['data'][:, 0]
   X = data_dict['data'][:, 1:]
