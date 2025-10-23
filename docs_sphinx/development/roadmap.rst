Development Roadmap
===================

Planned features and development priorities for fastcpd-python.

Current Status (v0.18.3)
------------------------

**Completed:**

- 14 model families (mean, variance, GLM, regression, time series, nonparametric)
- PELT and SeGD algorithms
- 7 evaluation metrics
- 7 dataset generators
- 6 visualization functions
- C++ optimization for core models
- Numba acceleration for GLM

**In Progress:**

- Documentation website
- Example gallery
- Community building

Immediate Priorities (Next 3 Months)
------------------------------------

1. Documentation Website (Priority: Highest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Status:** In progress

**Deliverables:**

- Sphinx documentation site
- GitHub Pages hosting
- Complete API reference
- Step-by-step tutorials
- Example gallery

**Timeline:** 2-3 weeks

2. Enhanced Examples (Priority: High)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Deliverables:**

- 20+ getting started examples
- Model-specific tutorials
- Real-world data examples
- 10+ Jupyter notebooks

**Timeline:** 3-4 weeks

3. Community Infrastructure (Priority: High)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Deliverables:**

- Issue templates
- GitHub Discussions
- Release process docs
- Contributor recognition

**Timeline:** 1-2 weeks

Short-Term (3-6 Months)
-----------------------

4. Binary Segmentation Algorithm (Priority: High)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Fast approximation for large datasets

**Use Cases:**

- Quick exploratory analysis
- Large datasets (n > 100,000)
- Real-time applications

**API:**

.. code-block:: python

   from fastcpd import binary_segmentation
   result = binary_segmentation(data, n_changepoints=5)

**Timeline:** 2 weeks

5. Bottom-Up Algorithm (Priority: High)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Hierarchical merging for unknown # of change points

**Use Cases:**

- Unknown number of change points
- Hierarchical structure
- Multi-scale analysis

**Timeline:** 2 weeks

6. Online Change Point Detection (Priority: Highest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Real-time detection for streaming data

**Features:**

- Online PELT
- Online SeGD
- Adaptive thresholds
- Alert system

**API:**

.. code-block:: python

   from fastcpd import OnlineDetector

   detector = OnlineDetector(family="mean", beta="MBIC")
   for new_data in stream:
       if detector.update(new_data):
           print(f"Change detected at {detector.last_cp}")

**Use Cases:**

- Real-time monitoring
- IoT sensor data
- Financial trading
- Network anomaly detection

**Timeline:** 8-12 weeks

7. Phase 2 Performance Optimizations (Priority: High)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goals:**

- Better pruning strategies
- Vectorization improvements
- Memory optimization
- 2-3x additional speedup for GLM

**Timeline:** 6 weeks

Medium-Term (6-12 Months)
-------------------------

8. Dynamic Programming Algorithm (Priority: Medium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Exact solution with different constraints

**Timeline:** 3 weeks

9. Custom Cost Functions API (Priority: Medium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- Abstract base class for cost functions
- Built-in costs: L1, L2, RBF, cosine, rank
- User-defined cost support

**API:**

.. code-block:: python

   from fastcpd import CostFunction

   class MyCost(CostFunction):
       def compute(self, data_segment):
           # Your cost implementation
           return cost_value

   result = fastcpd(data, cost=MyCost(), beta=10.0)

**Timeline:** 4 weeks

10. Uncertainty Quantification (Priority: High)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- Confidence intervals via bootstrap
- Bayesian posterior distributions
- Permutation tests
- Cross-validation for model selection

**API:**

.. code-block:: python

   result = mean(data, beta="MBIC", uncertainty=True)
   print(result.cp_intervals)  # Confidence intervals

**Timeline:** 6-8 weeks

11. Kernel Change Detection (Priority: Medium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- Maximum Mean Discrepancy (MMD)
- Additional kernel functions
- Kernel combination methods

**Timeline:** 6 weeks

12. Large Dataset Support (Priority: Medium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- Streaming API for chunked processing
- Out-of-core for datasets larger than RAM
- HDF5/Zarr support
- Progress bars

**Timeline:** 4 weeks

Long-Term (12+ Months)
----------------------

13. Phase 3: C++ Extensions (Priority: High)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goals:**

- GLM in C++ for better performance
- Multi-threading support
- SIMD optimizations

**Timeline:** 12 weeks

14. Multivariate Extensions (Priority: Medium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- Network/graph change detection
- Spatial change detection
- Panel data support

**Timeline:** 12 weeks

15. AutoML Features (Priority: Medium)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- Automatic model selection
- Auto beta tuning
- Hyperparameter search
- Ensemble methods

**API:**

.. code-block:: python

   from fastcpd import AutoDetector

   detector = AutoDetector()
   result = detector.fit(data, true_cps=known_cps)
   # Automatically selects best model and parameters

**Timeline:** 8 weeks

16. GPU Acceleration (Priority: Low)
~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- CuPy/PyTorch backend
- GPU-accelerated cost computation
- For n > 100,000

**Timeline:** 8 weeks (research phase)

17. Advanced Integrations (Priority: Medium)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Features:**

- scikit-learn compatible API
- Native pandas DataFrame support
- Polars support
- Arrow/Parquet formats

**Timeline:** 6 weeks

Feature Requests
----------------

Submit feature requests via `GitHub Issues <https://github.com/zhangxiany-tamu/fastcpd_Python/issues>`_.

Success Metrics
---------------

6 Months
~~~~~~~~

- [ ] 500+ GitHub stars
- [ ] 50+ forks
- [ ] 5+ external contributors
- [ ] 100+ PyPI downloads/month
- [ ] Complete documentation site
- [ ] 6+ detection algorithms

12 Months
~~~~~~~~~

- [ ] 1,000+ GitHub stars
- [ ] 100+ forks
- [ ] 10+ external contributors
- [ ] 1,000+ PyPI downloads/month
- [ ] Online detection capability
- [ ] 10+ cost functions
- [ ] 200+ tests passing

24 Months
~~~~~~~~~

- [ ] 5+ published papers using fastcpd-python
- [ ] 3+ industry users
- [ ] Referenced in courses or textbooks
- [ ] Full algorithm suite (6+ algorithms)
- [ ] AutoML capabilities

Contributing
------------

Want to help implement these features? See :doc:`contributing` for guidelines.

Priority votes and feedback welcome via GitHub Discussions!

Last Updated
------------

This roadmap was last updated on October 2025.

For latest status, see `GitHub Projects <https://github.com/zhangxiany-tamu/fastcpd_Python/projects>`_.
