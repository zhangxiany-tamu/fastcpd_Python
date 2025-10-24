fastcpd
=======

Fast change point detection in Python using PELT and SeGD algorithms.

Overview
--------

``fastcpd`` is a Python library for detecting structural breaks in time series and sequential data. It implements efficient algorithms for identifying points where statistical properties change.

**Key capabilities:**

* Multiple detection algorithms (PELT, SeGD)
* Parametric and nonparametric models
* Comprehensive evaluation metrics
* Built-in dataset generation

Installation
------------

From Test PyPI:

.. code-block:: bash

   # Install Armadillo (required for C++ extension)
   brew install armadillo  # macOS
   # sudo apt-get install libarmadillo-dev  # Linux

   # Install package
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyfastcpd

From source:

.. code-block:: bash

   git clone https://github.com/zhangxiany-tamu/fastcpd_Python.git
   cd fastcpd_Python
   pip install -e .

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import mean

   # Generate data with mean change at position 300
   data = np.concatenate([
       np.random.normal(0, 1, 300),
       np.random.normal(5, 1, 400)
   ])

   # Detect change points
   result = mean(data, beta="MBIC")
   print(result.cp_set)  # [300]

Supported Models
----------------

**Parametric:**

* Mean, variance, mean+variance
* Binomial and Poisson regression
* Linear regression and LASSO
* AR, VAR, ARMA, GARCH time series

**Nonparametric:**

* Rank-based detection
* RBF kernel methods

See :doc:`user_guide/models` for details.

Algorithm Features
------------------

PELT Algorithm
~~~~~~~~~~~~~~

Exact optimization with pruning for linear time complexity (average case).

SeGD Algorithm
~~~~~~~~~~~~~~

Fast gradient-based approximation for large datasets. Configurable via ``vanilla_percentage`` parameter:

.. code-block:: python

   # Pure PELT (exact)
   result = fastcpd(data, family="binomial", vanilla_percentage=1.0)

   # Pure SeGD (fast)
   result = fastcpd(data, family="binomial", vanilla_percentage=0.0)

   # Adaptive
   result = fastcpd(data, family="binomial", vanilla_percentage='auto')

Implementation
~~~~~~~~~~~~~~

* Core models (mean, variance): C++ for speed
* GLM models: Python with optional Numba acceleration
* Time series: Python with statsmodels/arch

Evaluation
----------

Six evaluation metrics included:

* Precision, Recall, F1-Score
* Hausdorff distance
* Covering metric (multi-annotator)
* Annotation error

Example:

.. code-block:: python

   from fastcpd.metrics import evaluate_all

   metrics = evaluate_all(
       true_cps=[100, 200, 300],
       pred_cps=[98, 205, 310],
       n_samples=500,
       margin=10
   )

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/installation
   user_guide/quickstart
   user_guide/models
   user_guide/evaluation
   user_guide/visualization
   user_guide/tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/detection
   api/metrics
   api/datasets
   api/visualization

.. toctree::
   :maxdepth: 1
   :caption: Advanced
   :hidden:

   advanced/algorithms
   advanced/comparison
   advanced/music_segmentation

Links
-----

* GitHub: https://github.com/zhangxiany-tamu/fastcpd_Python
* Issues: https://github.com/zhangxiany-tamu/fastcpd_Python/issues
* Test PyPI: https://test.pypi.org/project/pyfastcpd/

Citation
--------

If you use this software in your research, please cite:

.. code-block:: bibtex

   @article{zhang2023sequential,
     title={Sequential Gradient Descent and Quasi-Newton's Method for Change-Point Analysis},
     author={Zhang, Xianyang and Dawn, Trisha},
     journal={Proceedings of AISTATS},
     year={2023}
   }

License
-------

MIT License. See LICENSE file for details.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
