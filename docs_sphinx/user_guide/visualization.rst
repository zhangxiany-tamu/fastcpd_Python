Visualization
=============

fastcpd-python provides 5 publication-quality plotting functions for visualizing change point detection results and evaluation metrics.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Function
     - Purpose
     - Key Features
   * - ``plot_detection``
     - Plot data with detected and true change points
     - Automatic metric overlay
   * - ``plot_metric_comparison``
     - Compare metrics across algorithms
     - Side-by-side comparison
   * - ``plot_annotators``
     - Visualize multiple annotations
     - Agreement visualization
   * - ``plot_roc_curve``
     - ROC curve for detection performance
     - Threshold analysis
   * - ``plot_dataset_characteristics``
     - Visualize dataset properties
     - SNR, difficulty analysis

Quick Start
-----------

Basic Detection Plot
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.visualization import plot_detection
   from fastcpd.segmentation import mean
   from fastcpd.datasets import make_mean_change
   import matplotlib.pyplot as plt

   # Generate data
   data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

   # Detect change points
   result = mean(data_dict['data'], beta="MBIC")

   # Visualize
   fig, ax = plot_detection(
       data=data_dict['data'],
       true_cps=data_dict['changepoints'],
       pred_cps=result.cp_set.tolist(),
       title="Mean Change Detection"
   )
   plt.show()

With Metrics Overlay
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.metrics import evaluate_all

   # Evaluate metrics
   metrics = evaluate_all(
       data_dict['changepoints'],
       result.cp_set.tolist(),
       n_samples=500,
       margin=10
   )

   # Plot with metrics
   fig, ax = plot_detection(
       data=data_dict['data'],
       true_cps=data_dict['changepoints'],
       pred_cps=result.cp_set.tolist(),
       metric_result=metrics,  # Automatically displays metrics
       title="Mean Change Detection"
   )

Plotting Functions
------------------

plot_detection
~~~~~~~~~~~~~~

Main plotting function for visualizing detection results.

**Signature:**

.. code-block:: python

   plot_detection(
       data,
       true_cps=None,
       pred_cps=None,
       metric_result=None,
       title=None,
       figsize=(12, 6),
       show_legend=True
   )

**Parameters:**

- ``data``: Array of shape (n,) or (n, d) - data to plot
- ``true_cps``: List of true change point locations (optional)
- ``pred_cps``: List of detected change point locations (optional)
- ``metric_result``: Dictionary from ``evaluate_all()`` (optional)
- ``title``: Plot title
- ``figsize``: Figure size
- ``show_legend``: Whether to show legend

**Returns:**

- ``fig, ax``: Matplotlib figure and axes objects

**Example:**

.. code-block:: python

   import numpy as np
   from fastcpd.visualization import plot_detection

   # Simple data
   data = np.concatenate([
       np.random.normal(0, 1, 200),
       np.random.normal(3, 1, 200),
       np.random.normal(1, 1, 200)
   ])

   fig, ax = plot_detection(
       data=data,
       true_cps=[200, 400],
       pred_cps=[198, 405],
       title="Example Detection"
   )

**Multivariate Data:**

For multivariate data, plots the first 3 dimensions:

.. code-block:: python

   # 5D data
   data_5d = np.random.randn(500, 5)

   # Plots first 3 dimensions automatically
   plot_detection(
       data=data_5d,
       pred_cps=[200, 350]
   )

plot_annotators
~~~~~~~~~~~~~~~

Visualize multiple expert annotations and detection agreement.

.. code-block:: python

   from fastcpd.visualization import plot_annotators

   # Multiple annotators
   annotators_list = [
       [100, 200, 300],  # Expert 1
       [105, 195, 305],  # Expert 2
       [98, 203, 298]    # Expert 3
   ]

   detected_cps = [102, 201, 299]

   fig, ax = plot_annotators(
       data=data,
       annotators_list=annotators_list,
       pred_cps=detected_cps,
       title="Multi-Annotator Detection"
   )

**Features:**

- Shows all annotators with different colors
- Highlights agreement regions
- Overlays detected change points
- Computes covering metric

plot_metric_comparison
~~~~~~~~~~~~~~~~~~~~~~

Compare detection performance across multiple algorithms or parameters.

.. code-block:: python

   from fastcpd.visualization import plot_metric_comparison
   from fastcpd.segmentation import mean, rank, rbf

   # Try multiple algorithms
   algorithms = ['mean', 'rank', 'rbf']
   metrics_list = []

   for algo in algorithms:
       if algo == 'mean':
           result = mean(data, beta="MBIC")
       elif algo == 'rank':
           result = rank(data, beta=50.0)
       elif algo == 'rbf':
           result = rbf(data, beta=30.0)

       metrics = evaluate_all(
           true_cps=[100, 200, 300],
           pred_cps=result.cp_set.tolist(),
           n_samples=500,
           margin=10
       )
       metrics_list.append(metrics)

   # Compare
   fig, axes = plot_metric_comparison(
       metrics_list=metrics_list,
       algorithm_names=algorithms,
       metrics_to_plot=['precision', 'recall', 'f1_score']
   )

**Features:**

- Side-by-side bar charts
- Multiple metrics simultaneously
- Easy algorithm comparison

Customization
-------------

Custom Styling
~~~~~~~~~~~~~~

All functions return ``fig, ax`` for further customization:

.. code-block:: python

   fig, ax = plot_detection(data, pred_cps=[100, 200])

   # Customize
   ax.set_xlabel('Time (seconds)', fontsize=14)
   ax.set_ylabel('Signal Amplitude', fontsize=14)
   ax.set_title('Custom Title', fontsize=16, fontweight='bold')
   ax.grid(True, alpha=0.3, linestyle='--')

   # Adjust legend
   ax.legend(loc='upper right', fontsize=12)

   # Tight layout
   fig.tight_layout()

Custom Colors
~~~~~~~~~~~~~

.. code-block:: python

   fig, ax = plot_detection(data, pred_cps=[100])

   # Get line objects
   lines = ax.get_lines()

   # Customize colors
   lines[0].set_color('darkblue')     # Data
   lines[1].set_color('red')          # Change points
   lines[1].set_linewidth(3)

Saving Figures
~~~~~~~~~~~~~~

.. code-block:: python

   fig, ax = plot_detection(data, pred_cps=[100, 200])

   # Save as PNG
   fig.savefig('detection_result.png', dpi=300, bbox_inches='tight')

   # Save as PDF (vector graphics)
   fig.savefig('detection_result.pdf', bbox_inches='tight')

   # Save as SVG
   fig.savefig('detection_result.svg', bbox_inches='tight')

Publication-Ready Figures
--------------------------

Recommended Settings
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Set publication style
   plt.rcParams['font.size'] = 12
   plt.rcParams['axes.labelsize'] = 14
   plt.rcParams['axes.titlesize'] = 16
   plt.rcParams['xtick.labelsize'] = 12
   plt.rcParams['ytick.labelsize'] = 12
   plt.rcParams['legend.fontsize'] = 12
   plt.rcParams['figure.titlesize'] = 16
   plt.rcParams['lines.linewidth'] = 2

   # Use publication-friendly backend
   plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDFs

   # Create plot
   fig, ax = plot_detection(
       data=data,
       true_cps=[100, 200],
       pred_cps=[98, 205],
       figsize=(10, 5)
   )

   # Save high-quality figure
   fig.savefig('figure1.pdf', dpi=300, bbox_inches='tight')

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from fastcpd.visualization import plot_detection
   from fastcpd.segmentation import mean
   from fastcpd.datasets import make_mean_change
   from fastcpd.metrics import evaluate_all

   # Generate data
   np.random.seed(42)
   data_dict = make_mean_change(
       n_samples=600,
       n_changepoints=3,
       mean_shift=3.0,
       noise_std=1.0,
       seed=42
   )

   # Detect
   result = mean(data_dict['data'], beta="MBIC")

   # Evaluate
   metrics = evaluate_all(
       data_dict['changepoints'],
       result.cp_set.tolist(),
       n_samples=600,
       margin=10
   )

   # Plot
   fig, ax = plot_detection(
       data=data_dict['data'],
       true_cps=data_dict['changepoints'],
       pred_cps=result.cp_set.tolist(),
       metric_result=metrics,
       title='Mean Change Detection with PELT Algorithm',
       figsize=(12, 6)
   )

   # Customize
   ax.set_xlabel('Time Index', fontsize=14)
   ax.set_ylabel('Signal Value', fontsize=14)
   ax.grid(True, alpha=0.3)

   # Save
   fig.savefig('mean_detection_example.pdf', dpi=300, bbox_inches='tight')
   plt.show()

Advanced Visualizations
-----------------------

Heatmap of Detection Across Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from fastcpd.segmentation import mean

   # Parameter grid
   beta_values = np.logspace(0.5, 2, 20)  # 10^0.5 to 10^2
   n_values = len(beta_values)

   # Store results
   detection_matrix = np.zeros((n_values, len(data)))

   for i, beta_val in enumerate(beta_values):
       result = mean(data, beta=beta_val)
       for cp in result.cp_set:
           if 0 <= cp < len(data):
               detection_matrix[i, int(cp)] = 1

   # Plot heatmap
   fig, ax = plt.subplots(figsize=(12, 6))
   im = ax.imshow(detection_matrix, aspect='auto', cmap='YlOrRd',
                  extent=[0, len(data), beta_values[0], beta_values[-1]])
   ax.set_xlabel('Time Index')
   ax.set_ylabel('Beta Value')
   ax.set_title('Change Point Detection Across Beta Values')
   plt.colorbar(im, label='Detection')
   plt.show()

Comparison of Multiple Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.segmentation import mean, variance, rank, rbf

   models = {
       'Mean': mean(data, beta="MBIC"),
       'Variance': variance(data, beta="MBIC"),
       'Rank': rank(data, beta=50.0),
       'RBF': rbf(data, beta=30.0)
   }

   fig, axes = plt.subplots(2, 2, figsize=(14, 10))
   axes = axes.flatten()

   for i, (name, result) in enumerate(models.items()):
       axes[i].plot(data, linewidth=1, alpha=0.7, label='Data')
       for cp in result.cp_set:
           axes[i].axvline(cp, color='red', linestyle='--', linewidth=2)
       axes[i].set_title(f'{name} Model', fontsize=14)
       axes[i].set_xlabel('Time')
       axes[i].set_ylabel('Value')
       axes[i].legend()
       axes[i].grid(True, alpha=0.3)

   fig.tight_layout()
   plt.show()

Integration with Other Tools
-----------------------------

Plotly for Interactive Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import plotly.graph_objects as go

   fig = go.Figure()

   # Add data trace
   fig.add_trace(go.Scatter(
       x=list(range(len(data))),
       y=data,
       mode='lines',
       name='Data'
   ))

   # Add change points
   for cp in result.cp_set:
       fig.add_vline(x=cp, line_dash="dash", line_color="red")

   fig.update_layout(
       title="Interactive Change Point Detection",
       xaxis_title="Time",
       yaxis_title="Value",
       hovermode='x unified'
   )

   fig.show()

Seaborn for Statistical Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import seaborn as sns
   import pandas as pd

   # Create DataFrame with segments
   segments = []
   cps = [0] + result.cp_set.tolist() + [len(data)]
   for i in range(len(cps) - 1):
       segment_data = data[int(cps[i]):int(cps[i+1])]
       segments.extend([i] * len(segment_data))

   df = pd.DataFrame({'value': data, 'segment': segments})

   # Violin plot by segment
   sns.violinplot(data=df, x='segment', y='value')
   plt.title('Distribution by Segment')
   plt.show()

Next Steps
----------

- :doc:`../api/visualization` - Complete visualization API reference
- :doc:`evaluation` - Learn about evaluation metrics
- :doc:`tutorials` - Follow step-by-step tutorials
