Visualization API
=================

Plotting functions for change point detection results.

Overview
--------

.. automodule:: fastcpd.visualization
   :members:
   :undoc-members:

Plotting Functions
------------------

Main Plot Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: fastcpd.visualization.plot_detection
.. autofunction:: fastcpd.visualization.plot_annotators
.. autofunction:: fastcpd.visualization.plot_metric_comparison

Advanced Plots
~~~~~~~~~~~~~~

.. autofunction:: fastcpd.visualization.plot_roc_curve
.. autofunction:: fastcpd.visualization.plot_dataset_characteristics

Example Usage
-------------

Basic Detection Plot
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.visualization import plot_detection
   import matplotlib.pyplot as plt
   import numpy as np

   # Generate simple data
   data = np.concatenate([
       np.random.normal(0, 1, 200),
       np.random.normal(3, 1, 200)
   ])

   # Plot
   fig, ax = plot_detection(
       data=data,
       true_cps=[200],
       pred_cps=[198],
       title="Example Detection"
   )
   plt.show()

With Metrics Overlay
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.metrics import evaluate_all
   from fastcpd.visualization import plot_detection

   # Evaluate metrics
   metrics = evaluate_all(
       true_cps=[200],
       pred_cps=[198],
       n_samples=400,
       margin=10
   )

   # Plot with metrics
   fig, ax = plot_detection(
       data=data,
       true_cps=[200],
       pred_cps=[198],
       metric_result=metrics,
       title="Detection with Metrics"
   )

Multi-Annotator Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.visualization import plot_annotators

   # Multiple expert annotations
   annotators_list = [
       [100, 200, 300],
       [105, 195, 305],
       [98, 203, 298]
   ]

   fig, ax = plot_annotators(
       data=data,
       annotators_list=annotators_list,
       pred_cps=[102, 201, 299],
       title="Multi-Annotator Detection"
   )

Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.visualization import plot_metric_comparison

   # Compare multiple algorithms
   results_dict = {
       'Mean': metrics_mean,
       'Rank': metrics_rank,
       'RBF': metrics_rbf
   }

   fig, axes = plot_metric_comparison(
       results_dict=results_dict,
       metrics=['precision', 'recall', 'f1_score']
   )

Customization Options
----------------------

Figure Size
~~~~~~~~~~~

.. code-block:: python

   fig, ax = plot_detection(
       data=data,
       pred_cps=[100],
       figsize=(14, 6)  # Custom size
   )

Title and Labels
~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, ax = plot_detection(
       data=data,
       pred_cps=[100],
       title="Custom Title"
   )

   # Further customization
   ax.set_xlabel('Time (seconds)')
   ax.set_ylabel('Sensor Reading')

Legend Control
~~~~~~~~~~~~~~

.. code-block:: python

   fig, ax = plot_detection(
       data=data,
       pred_cps=[100],
       show_legend=False  # Hide legend
   )

Return Values
-------------

All plotting functions return matplotlib figure and axes objects:

.. code-block:: python

   fig, ax = plot_detection(data, pred_cps=[100])

   # fig: matplotlib.figure.Figure
   # ax: matplotlib.axes.Axes (or array of axes for subplots)

   # Further customization
   ax.grid(True, alpha=0.3)
   fig.tight_layout()

   # Save
   fig.savefig('my_plot.pdf', dpi=300, bbox_inches='tight')

Saving Figures
--------------

Save in Multiple Formats
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, ax = plot_detection(data, pred_cps=[100])

   # High-resolution PNG
   fig.savefig('plot.png', dpi=300, bbox_inches='tight')

   # Vector graphics (PDF)
   fig.savefig('plot.pdf', bbox_inches='tight')

   # SVG
   fig.savefig('plot.svg', bbox_inches='tight')

Publication Quality
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Set publication style
   plt.rcParams['font.size'] = 12
   plt.rcParams['axes.labelsize'] = 14
   plt.rcParams['figure.dpi'] = 100
   plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts

   fig, ax = plot_detection(
       data=data,
       pred_cps=[100],
       figsize=(10, 5)
   )

   fig.savefig('publication_figure.pdf', dpi=300, bbox_inches='tight')
