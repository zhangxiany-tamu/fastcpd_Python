Detection API
=============

Core detection functions and result objects.

fastcpd.fastcpd
---------------

.. autofunction:: fastcpd.fastcpd.fastcpd

FastcpdResult
-------------

.. autoclass:: fastcpd.fastcpd.FastcpdResult
   :members:
   :undoc-members:

fastcpd.segmentation
--------------------

Convenience functions for common detection scenarios.

Parametric Methods
~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.segmentation.mean
.. autofunction:: fastcpd.segmentation.variance
.. autofunction:: fastcpd.segmentation.meanvariance

Regression Methods
~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.segmentation.linear_regression
.. autofunction:: fastcpd.segmentation.logistic_regression
.. autofunction:: fastcpd.segmentation.poisson_regression
.. autofunction:: fastcpd.segmentation.lasso

Time Series Methods
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.segmentation.ar
.. autofunction:: fastcpd.segmentation.var
.. autofunction:: fastcpd.segmentation.arma
.. autofunction:: fastcpd.segmentation.garch

Nonparametric Methods
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.segmentation.rank
.. autofunction:: fastcpd.segmentation.rbf
