Evaluation Metrics
==================

fastcpd-python provides 6 comprehensive evaluation metrics for assessing change point detection performance.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Metric
     - Description
     - Range
     - Best Value
   * - **Precision**
     - Fraction of detected CPs that are correct
     - [0, 1]
     - 1.0
   * - **Recall**
     - Fraction of true CPs that are detected
     - [0, 1]
     - 1.0
   * - **F1-Score**
     - Harmonic mean of precision and recall
     - [0, 1]
     - 1.0
   * - **Hausdorff**
     - Maximum distance between true and detected sets
     - [0, ∞)
     - 0.0
   * - **Covering**
     - Multi-annotator agreement metric
     - [0, 1]
     - 1.0
   * - **Annotation Error**
     - Average distance of detections from true CPs
     - [0, ∞)
     - 0.0

Quick Start
-----------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.metrics import evaluate_all
   from fastcpd.segmentation import mean
   from fastcpd.datasets import make_mean_change
   import numpy as np

   # Generate data with known change points
   data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

   # Detect change points
   result = mean(data_dict['data'], beta="MBIC")

   # Evaluate all metrics
   metrics = evaluate_all(
       true_cps=data_dict['changepoints'],
       pred_cps=result.cp_set.tolist(),
       n_samples=500,
       margin=10  # Tolerance window
   )

   # Display results
   print(f"Precision: {metrics['point_metrics']['precision']:.3f}")
   print(f"Recall:    {metrics['point_metrics']['recall']:.3f}")
   print(f"F1-Score:  {metrics['point_metrics']['f1_score']:.3f}")

Example output:

.. code-block:: text

   Precision: 1.000
   Recall:    0.667
   F1-Score:  0.800

Individual Metrics
------------------

Precision and Recall
~~~~~~~~~~~~~~~~~~~~

**Precision**: What fraction of detected change points are correct?

.. math::

   \text{Precision} = \frac{TP}{TP + FP}

**Recall (Sensitivity)**: What fraction of true change points were detected?

.. math::

   \text{Recall} = \frac{TP}{TP + FN}

.. code-block:: python

   from fastcpd.metrics import precision_recall

   true_cps = [100, 200, 300]
   detected_cps = [98, 205, 350]

   pr = precision_recall(true_cps, detected_cps, n_samples=500, margin=10)

   print(f"Precision: {pr['precision']:.3f}")
   print(f"Recall:    {pr['recall']:.3f}")

**Parameters:**

- ``margin``: Tolerance window (default: 10)

  - A detected CP within ``margin`` of a true CP is considered correct
  - Common values: 5-20 depending on application

**Interpretation:**

- **High Precision, Low Recall**: Conservative detection (few false positives, many misses)
- **Low Precision, High Recall**: Liberal detection (many false positives, few misses)
- **High Both**: Excellent detection
- **Low Both**: Poor detection

F1-Score
~~~~~~~~

Harmonic mean of precision and recall, balancing both metrics.

.. math::

   F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}

.. code-block:: python

   from fastcpd.metrics import precision_recall

   pr = precision_recall(true_cps, detected_cps, n_samples=500, margin=10)
   print(f"F1-Score: {pr['f1_score']:.3f}")

**Advantages:**

- Single metric summarizing performance
- Balances precision and recall
- Standard in machine learning

**Disadvantages:**

- Doesn't capture the magnitude of errors
- Sensitive to ``margin`` parameter

Hausdorff Distance
~~~~~~~~~~~~~~~~~~

Maximum distance between true and detected change point sets (in both directions).

.. math::

   d_H(A, B) = \max\left(\max_{a \in A} \min_{b \in B} |a - b|, \max_{b \in B} \min_{a \in A} |a - b|\right)

.. code-block:: python

   from fastcpd.metrics import hausdorff_distance

   hd = hausdorff_distance(true_cps, detected_cps)
   print(f"Hausdorff Distance: {hd['hausdorff']}")

**Interpretation:**

- ``0``: Perfect match
- Larger values: Worse match
- Captures worst-case error
- Useful for applications requiring guarantees

**Use Cases:**

- Medical applications (critical to not miss)
- Safety-critical systems
- Quality control

Annotation Error
~~~~~~~~~~~~~~~~

Mean absolute error between optimally matched true and detected change points.

.. math::

   \text{AE} = \frac{1}{|\text{Matched Pairs}|} \sum_{(t,d) \in \text{Matched Pairs}} |t - d|

where matched pairs are determined by greedy closest-pair matching between true and detected sets.

.. code-block:: python

   from fastcpd.metrics import annotation_error

   ae = annotation_error(true_cps, detected_cps)
   print(f"Annotation Error: {ae['error']:.2f}")

**Interpretation:**

- ``0``: All matched pairs have perfect localization
- Lower is better
- Measures average localization accuracy for matched pairs
- Unmatched change points (when |true| ≠ |detected|) do not contribute to the error

Covering Metric
~~~~~~~~~~~~~~~

Multi-annotator agreement metric. Measures how well detected change points "cover" multiple sets of annotations.

.. math::

   \text{Covering} = \frac{1}{K} \sum_{k=1}^K \frac{|D \cap T_k^{\text{margin}}|}{|T_k|}

where :math:`T_k` is the k-th annotator's change points, and :math:`T_k^{\text{margin}}` is expanded by ``margin``.

.. code-block:: python

   from fastcpd.metrics import covering_metric

   # Multiple annotators
   true_cps_multi = [
       [100, 200, 300],     # Annotator 1
       [105, 195, 305],     # Annotator 2
       [98, 203, 298]       # Annotator 3
   ]

   detected_cps = [102, 201, 299]

   covering = covering_metric(true_cps_multi, detected_cps, margin=10)
   print(f"Covering: {covering:.3f}")

**Use Cases:**

- Datasets with multiple expert annotations
- Ambiguous change point locations
- Robustness assessment

**Advantages:**

- Accounts for annotation uncertainty
- More realistic than single ground truth
- Commonly used in research papers

Evaluate All Metrics at Once
-----------------------------

.. code-block:: python

   from fastcpd.metrics import evaluate_all

   metrics = evaluate_all(
       true_cps=[100, 200, 300],
       pred_cps=[98, 205, 310],
       n_samples=500,
       margin=10
   )

   # Returns dictionary with all metrics
   print("All Metrics:")
   for metric_name, value in metrics.items():
       print(f"  {metric_name:20s}: {value:.3f}")

Example output:

.. code-block:: text

   All Metrics:
     precision           : 1.000
     recall              : 1.000
     f1_score            : 1.000
     hausdorff           : 10.000
     annotation_error    : 6.333
     one_to_one          : 1.000

Metric Return Values
--------------------

Rich Dictionary Returns
~~~~~~~~~~~~~~~~~~~~~~~

Most metrics return a dictionary with detailed breakdown:

.. code-block:: python

   from fastcpd.metrics import precision_recall

   result = precision_recall(true_cps, detected_cps, n_samples=500, margin=10)

   # Dictionary with detailed fields
   print(result)
   # {
   #     'precision': 0.667,
   #     'recall': 0.667,
   #     'f1_score': 0.667,
   #     'true_positives': 2,
   #     'false_positives': 1,
   #     'false_negatives': 1,
   #     'n_true': 3,
   #     'n_detected': 3,
   #     'margin': 10
   # }

**Advantages:**

- Detailed debugging information
- Understand exactly what happened
- Report multiple aspects

Choosing Metrics
----------------

By Application
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Application
     - Recommended Metrics
   * - General research
     - Precision, Recall, F1-Score
   * - Medical/safety-critical
     - Recall (don't miss!), Hausdorff
   * - Quality control
     - Precision (avoid false alarms), F1
   * - Multi-annotator data
     - Covering metric
   * - Exact localization needed
     - Annotation Error, Hausdorff
   * - Algorithm comparison
     - F1-Score, Covering

By Constraint
~~~~~~~~~~~~~

**Minimize False Positives** (e.g., avoid unnecessary interventions):

- Focus on **Precision**
- Use conservative ``beta`` values

**Minimize False Negatives** (e.g., don't miss critical events):

- Focus on **Recall**
- Use liberal ``beta`` values

**Balance Both**:

- Use **F1-Score**
- Tune ``beta`` to maximize F1

Best Practices
--------------

Choosing the Margin
~~~~~~~~~~~~~~~~~~~

The ``margin`` parameter is critical:

.. code-block:: python

   # Too small: penalizes small localization errors
   metrics_strict = evaluate_all(true_cps, detected_cps, n_samples=500, margin=2)

   # Reasonable: 10-20 typical
   metrics_moderate = evaluate_all(true_cps, detected_cps, n_samples=500, margin=10)

   # Too large: accepts poor localization
   metrics_loose = evaluate_all(true_cps, detected_cps, n_samples=500, margin=50)

**Guidelines:**

- **Small data (n<100)**: ``margin=2-5``
- **Medium data (n=100-1000)**: ``margin=5-15``
- **Large data (n>1000)**: ``margin=10-30``
- **Application-specific**: Use domain knowledge

Report Multiple Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

Don't rely on a single metric:

.. code-block:: python

   # Report at least these three
   print(f"Precision: {metrics['point_metrics']['precision']:.3f}")
   print(f"Recall:    {metrics['point_metrics']['recall']:.3f}")
   print(f"F1-Score:  {metrics['point_metrics']['f1_score']:.3f}")

   # Plus application-specific
   print(f"Hausdorff: {metrics['hausdorff']:.1f}")

Cross-Validation for Beta Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.segmentation import mean
   from fastcpd.metrics import precision_recall
   from fastcpd.datasets import make_mean_change

   # Generate data
   data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

   # Try different beta values
   beta_values = [5.0, 10.0, 15.0, 20.0, 30.0, "BIC", "MBIC", "MDL"]
   results = []

   for beta_val in beta_values:
       result = mean(data_dict['data'], beta=beta_val)
       pr = precision_recall(
           data_dict['changepoints'],
           result.cp_set.tolist(),
           n_samples=500,
           margin=10
       )
       f1 = pr['f1_score']
       results.append((beta_val, f1, len(result.cp_set)))
       print(f"Beta={str(beta_val):8s}: F1={f1:.3f}, n_cp={len(result.cp_set)}")

   # Select best beta
   best_beta, best_f1, best_ncp = max(results, key=lambda x: x[1])
   print(f"\nBest: Beta={best_beta}, F1={best_f1:.3f}")

Integration with Datasets
--------------------------

All dataset generators return metadata including ground truth change points:

.. code-block:: python

   from fastcpd.datasets import make_mean_change, make_glm_change

   # Mean change dataset
   data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)
   # Returns: data, changepoints, means, noise_std, SNR, etc.

   # GLM dataset
   data_dict = make_glm_change(
       n_samples=500,
       n_predictors=5,
       n_changepoints=2,
       family='binomial'
   )
   # Returns: data, changepoints, coefficients, AUC, etc.

   # Detect and evaluate
   result = mean(data_dict['data'], beta="MBIC")
   metrics = evaluate_all(
       data_dict['changepoints'],
       result.cp_set.tolist(),
       n_samples=500,
       margin=10
   )

Next Steps
----------

- :doc:`visualization` - Visualize metrics and detection results
- :doc:`../api/metrics` - Complete API reference
- :doc:`../api/datasets` - Dataset generation API
