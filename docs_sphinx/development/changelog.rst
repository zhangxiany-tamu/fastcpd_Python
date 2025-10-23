Changelog
=========

All notable changes to fastcpd-python are documented here.

Version 0.18.3 (2025-10-16)
---------------------------

Fixed
~~~~~

- Fixed C++ compilation error: Added missing ``#include <string>`` in ``src/ref_family_python.cc``
- Fixed installation error on Test PyPI by updating build dependencies
- Updated ``scikit-build-core`` from >=0.8 to >=0.10.0 for compatibility
- Added explicit ``packaging>=24.0`` requirement

Changed
~~~~~~~

- Build system now requires ``scikit-build-core>=0.10.0``, ``nanobind>=2.0``, and ``packaging>=24.0``

Version 0.18.0 (2025-10-11)
---------------------------

Added
~~~~~

- **GARCH models** (``family="garch"``) for volatility change detection
- **ARMA models** (``family="arma"``) for time series analysis
- **Linear regression** (``family="lm"``) with perfect accuracy
- **LASSO regression** (``family="lasso"``) with L1 penalization
- **Comprehensive evaluation metrics module**:

  - Precision, Recall, F1-Score
  - Hausdorff distance, Covering metric
  - Annotation error, One-to-one correspondence
  - Multi-annotator metrics

- **Dataset generation utilities** with rich metadata
- **Visualization tools** for change point analysis (6 plotting functions)
- Support for ``vanilla_percentage`` parameter (PELT/SeGD interpolation)
- **Nonparametric methods**: rank-based and RBF kernel detection

Changed
~~~~~~~

- Improved GLM accuracy (71% win rate vs R implementation)
- Phase 1 optimizations: 1.4x speedup with adaptive pruning and warm start
- Optional Numba acceleration (7-14x additional speedup for GLM)

Performance
~~~~~~~~~~~

- Mean/Variance models: 2-14x faster than R (C++ implementation)
- GLM models: Competitive performance with Numba installed
- ARMA/GARCH: Pure Python implementation with excellent accuracy

Version 0.1.0 (2025-06-01)
--------------------------

Added
~~~~~

- Initial Python implementation of fastcpd
- Core models: mean, variance, meanvariance
- GLM models: binomial, poisson
- C++ implementation for core models using nanobind
- Basic PELT algorithm with pruning
- SeGD (Sequential Gradient Descent) support
- pytest-based test suite

Changed
~~~~~~~

- Ported from R package to Python
- Adapted C++ code for Python bindings

---

Release Notes
-------------

Version 0.18.0 Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

**New Capabilities:**

- Complete time series support (ARMA/GARCH)
- Full regression suite (linear/LASSO)
- Industry-leading evaluation metrics
- Flexible PELT/SeGD interpolation

**Performance:**

- Up to 14x faster with Numba for GLM
- 2-14x faster for core models vs R
- Suitable for datasets up to n=1000+

**Quality:**

- Comprehensive test suite (76/76 tests passing)
- Extensive documentation and examples
- Production-ready for most use cases

Deprecation Notices
-------------------

None currently.

Upcoming Changes
----------------

See :doc:`roadmap` for planned features.

Migration Guides
----------------

From 0.1.0 to 0.18.0
~~~~~~~~~~~~~~~~~~~~

**Breaking Changes:** None

**New Features:** All backward compatible

**Recommended Updates:**

.. code-block:: bash

   # Update to latest version
   pip install --upgrade pyfastcpd

   # Install Numba for 7-14x GLM speedup
   pip install numba

**API Changes:** None

Contributors
------------

Version 0.18.3
~~~~~~~~~~~~~~

- Xianyang Zhang - Build system fixes

Version 0.18.0
~~~~~~~~~~~~~~

- Xianyang Zhang - Core development
- Contributors - Testing and feedback

Version 0.1.0
~~~~~~~~~~~~~

- Xianyang Zhang - Initial implementation
