Metrics API
===========

Evaluation metrics for change point detection.

Overview
--------

.. automodule:: fastcpd.metrics
   :members:
   :undoc-members:

Individual Metrics
------------------

Precision and Recall
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.metrics.precision_recall
.. autofunction:: fastcpd.metrics.f_beta_score

Distance Metrics
~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.metrics.hausdorff_distance
.. autofunction:: fastcpd.metrics.annotation_error

Agreement Metrics
~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.metrics.covering_metric

Segmentation Metrics
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.metrics.adjusted_rand_index

Combined Evaluation
-------------------

.. autofunction:: fastcpd.metrics.evaluate_all

Return Value Structure
----------------------

Most metrics return a dictionary with the following structure:

.. code-block:: python

   {
       'metric_value': float,        # Main metric value
       'true_positives': int,        # Number of true positives
       'false_positives': int,       # Number of false positives
       'false_negatives': int,       # Number of false negatives
       'n_true': int,                # Number of true change points
       'n_detected': int,            # Number of detected change points
       'margin': int,                # Tolerance margin used
       # ... additional fields ...
   }

Example Usage
-------------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.metrics import precision_recall

   true_cps = [100, 200, 300]
   detected_cps = [98, 205, 350]

   # Get precision, recall, and F1
   pr = precision_recall(true_cps, detected_cps, n_samples=500, margin=10)

   print(f"Precision: {pr['precision']:.3f}")
   print(f"Recall:    {pr['recall']:.3f}")
   print(f"F1-Score:  {pr['f1_score']:.3f}")

Comprehensive Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.metrics import evaluate_all

   # Evaluate all metrics at once
   metrics = evaluate_all(
       true_cps=true_cps,
       pred_cps=detected_cps,
       n_samples=500,
       margin=10
   )

   # Access results
   print(f"Precision: {metrics['point_metrics']['precision']:.3f}")
   print(f"Recall: {metrics['point_metrics']['recall']:.3f}")
   print(f"F1-Score: {metrics['point_metrics']['f1_score']:.3f}")
   print(f"Hausdorff: {metrics['distance_metrics']['hausdorff']:.1f}")

Multi-Annotator Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.metrics import covering_metric

   # Multiple expert annotations
   annotations = [
       [100, 200, 300],  # Expert 1
       [105, 195, 305],  # Expert 2
       [98, 203, 298]    # Expert 3
   ]

   detected_cps = [102, 201, 299]

   result = covering_metric(annotations, detected_cps, margin=10)
   print(f"Covering: {result['covering_score']:.3f}")
