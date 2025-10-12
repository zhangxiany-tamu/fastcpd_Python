fastcpd-python Documentation
=============================

**Fast change point detection in Python using PELT and Sequential Gradient Descent (SeGD)**

``fastcpd-python`` is a comprehensive Python package for detecting change points in time series and sequential data. It combines speed, accuracy, and rich evaluation capabilities.

.. note::
   🎯 **Status**: Production ready for mean/variance/GLM models.

Key Features
------------

✅ **Comprehensive Evaluation**
   - 7 evaluation metrics
   - Multi-annotator support with covering metric
   - Rich return values (dicts with detailed breakdowns)

✅ **Rich Dataset Generation**
   - 7 generators with metadata (SNR, R², AUC, etc.)
   - Multi-annotator simulation
   - GLM, GARCH, ARMA support

✅ **Visualization**
   - 6 plotting functions for publication-quality figures
   - Automatic metric overlay
   - Multi-annotator visualization

✅ **Performance**
   - Fast C++ implementation for core models
   - Optional Numba acceleration for GLM models
   - Hybrid PELT/SeGD algorithm

✅ **Multiple Models**
   - Mean, Variance, MeanVariance (C++, fastest)
   - Binomial, Poisson (GLM with SeGD)
   - LASSO, Linear Regression
   - ARMA, GARCH (time series)

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Basic installation
   pip install -e .

   # For 7-14x additional speedup (recommended)
   pip install numba

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from fastcpd import fastcpd

   # Generate data with change points
   data = np.concatenate([
       np.random.normal(0, 1, 300),
       np.random.normal(5, 1, 400)
   ])

   # Detect change points
   result = fastcpd(data, family='mean', beta='MBIC')
   print(result.cp_set)  # [300]

Evaluation & Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.datasets import make_mean_change
   from fastcpd.metrics import evaluate_all
   from fastcpd.visualization import plot_detection

   # Generate data with known change points
   data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

   # Detect
   result = fastcpd(data_dict['data'], family='mean', beta='MBIC')

   # Evaluate
   metrics = evaluate_all(
       data_dict['changepoints'],
       result.cp_set.tolist(),
       n_samples=500,
       margin=10
   )

   # Visualize
   plot_detection(
       data_dict['data'],
       data_dict['changepoints'],
       result.cp_set.tolist(),
       metric_result=metrics
   )

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/models
   user_guide/evaluation
   user_guide/visualization
   user_guide/tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/detection
   api/metrics
   api/datasets
   api/visualization

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   advanced/algorithms
   advanced/performance
   advanced/benchmarks
   advanced/comparison

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/changelog
   development/roadmap

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Community
=========

- **GitHub**: https://github.com/[your-username]/fastcpd-python
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas

Citation
========

If you use fastcpd in your research, please cite:

.. code-block:: bibtex

   @article{zhang2023sequential,
     title={Sequential Gradient Descent and Quasi-Newton's Method for Change-Point Analysis},
     author={Zhang, Xianyang and Dawn, Trisha},
     journal={Proceedings of AISTATS},
     year={2023}
   }

License
=======

This project is licensed under the MIT License - see the LICENSE file for details.
