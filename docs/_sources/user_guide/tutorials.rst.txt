Tutorials
=========

Step-by-step tutorials for common use cases.

Tutorial 1: First Detection
----------------------------

**Goal**: Detect mean changes in synthetic data and evaluate results.

**Prerequisites**: Basic Python, NumPy

Step 1: Generate Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from fastcpd.datasets import make_mean_change

   # Generate data with 3 change points
   np.random.seed(42)
   data_dict = make_mean_change(
       n_samples=600,
       n_changepoints=3,
       mean_shift=3.0,
       noise_std=1.0,
       seed=42
   )

   # Inspect what we got
   print(f"Data shape: {data_dict['data'].shape}")
   print(f"True change points: {data_dict['changepoints']}")
   print(f"Segment means: {data_dict['means']}")
   print(f"SNR: {data_dict['SNR']:.2f}")

Step 2: Detect Change Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.segmentation import mean

   # Detect using MBIC penalty
   result = mean(data_dict['data'], beta="MBIC")

   print(f"Detected change points: {result.cp_set}")
   print(f"Number detected: {len(result.cp_set)}")

Step 3: Evaluate Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.metrics import evaluate_all

   metrics = evaluate_all(
       true_cps=data_dict['changepoints'],
       pred_cps=result.cp_set.tolist(),
       n_samples=600,
       margin=10
   )

   print(f"Precision: {metrics['point_metrics']['precision']:.3f}")
   print(f"Recall:    {metrics['point_metrics']['recall']:.3f}")
   print(f"F1-Score:  {metrics['point_metrics']['f1_score']:.3f}")

Step 4: Visualize
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fastcpd.visualization import plot_detection
   import matplotlib.pyplot as plt

   fig, ax = plot_detection(
       data=data_dict['data'],
       true_cps=data_dict['changepoints'],
       pred_cps=result.cp_set.tolist(),
       metric_result=metrics,
       title="Tutorial 1: First Detection"
   )
   plt.show()

**Expected Output**: You should see perfect or near-perfect detection (F1 ≥ 0.9).

Tutorial 2: Comparing Algorithms
---------------------------------

**Goal**: Compare different detection methods on the same data.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import mean, rank, rbf
   from fastcpd.datasets import make_mean_change
   from fastcpd.metrics import evaluate_all
   from fastcpd.visualization import plot_metric_comparison

   # Generate data
   np.random.seed(42)
   data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

   # Try different algorithms
   algorithms = {
       'Mean (Parametric)': mean(data_dict['data'], beta="MBIC"),
       'Rank (Nonparametric)': rank(data_dict['data'], beta=50.0),
       'RBF (Nonparametric)': rbf(data_dict['data'], beta=30.0)
   }

   # Evaluate each
   results = {}
   for name, result in algorithms.items():
       metrics = evaluate_all(
           data_dict['changepoints'],
           result.cp_set.tolist(),
           n_samples=500,
           margin=10
       )
       results[name] = metrics
       print(f"\n{name}:")
       print(f"  F1-Score: {metrics['point_metrics']['f1_score']:.3f}")
       print(f"  Detected: {len(result.cp_set)} change points")

   # Plot comparison
   plot_metric_comparison(
       metrics_list=list(results.values()),
       algorithm_names=list(results.keys()),
       metrics_to_plot=['precision', 'recall', 'f1_score']
   )

**Key Insight**: Parametric methods (mean) usually perform best when assumptions are met. Nonparametric methods are more robust to violations.

Tutorial 3: Parameter Tuning
-----------------------------

**Goal**: Find optimal beta penalty via cross-validation.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import mean
   from fastcpd.datasets import make_mean_change
   from fastcpd.metrics import f1_score
   import matplotlib.pyplot as plt

   # Generate data
   np.random.seed(42)
   data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)

   # Try different beta values
   beta_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
   f1_scores = []
   n_detected = []

   for beta_val in beta_values:
       result = mean(data_dict['data'], beta=beta_val)
       f1 = f1_score(
           data_dict['changepoints'],
           result.cp_set.tolist(),
           n_samples=500,
           margin=10
       )['f1_score']
       f1_scores.append(f1)
       n_detected.append(len(result.cp_set))
       print(f"Beta={beta_val:5.1f}: F1={f1:.3f}, n_cp={len(result.cp_set)}")

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   # F1 vs beta
   ax1.plot(beta_values, f1_scores, 'o-', linewidth=2)
   ax1.axhline(y=max(f1_scores), color='r', linestyle='--', alpha=0.5)
   ax1.set_xlabel('Beta Value')
   ax1.set_ylabel('F1-Score')
   ax1.set_title('Performance vs Beta')
   ax1.grid(True, alpha=0.3)

   # Number of CPs vs beta
   ax2.plot(beta_values, n_detected, 's-', linewidth=2, color='green')
   ax2.axhline(y=len(data_dict['changepoints']), color='r',
               linestyle='--', alpha=0.5, label='True # CPs')
   ax2.set_xlabel('Beta Value')
   ax2.set_ylabel('Number of Detected CPs')
   ax2.set_title('Detected Change Points vs Beta')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   # Find optimal
   best_idx = np.argmax(f1_scores)
   print(f"\nOptimal Beta: {beta_values[best_idx]}")
   print(f"Best F1-Score: {f1_scores[best_idx]:.3f}")

**Key Insight**: Higher beta → fewer change points. Tune to balance precision and recall.

Tutorial 4: Real-World Example - Sensor Data
---------------------------------------------

**Goal**: Detect temperature sensor anomalies.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import meanvariance
   from fastcpd.metrics import evaluate_all
   from fastcpd.visualization import plot_detection
   import matplotlib.pyplot as plt

   # Simulate sensor data (hourly temperature readings)
   np.random.seed(123)

   # Normal operation (24 hours)
   normal = np.random.normal(20.0, 0.5, 24)

   # Sensor drift (12 hours)
   drift = np.random.normal(22.0, 0.5, 12)

   # Malfunction (high variance, 12 hours)
   malfunction = np.random.normal(20.0, 3.0, 12)

   # Back to normal (12 hours)
   recovered = np.random.normal(20.0, 0.5, 12)

   # Combine
   temperature_data = np.concatenate([normal, drift, malfunction, recovered])
   true_cps = [24, 36, 48]  # Change point indices

   # Detect using meanvariance (detects both mean and variance changes)
   result = meanvariance(temperature_data, beta="MBIC")

   print(f"True change points (hours): {true_cps}")
   print(f"Detected change points: {result.cp_set}")

   # Evaluate
   metrics = evaluate_all(
       true_cps=true_cps,
       pred_cps=result.cp_set.tolist(),
       n_samples=len(temperature_data),
       margin=2  # 2-hour tolerance
   )

   print(f"F1-Score: {metrics['point_metrics']['f1_score']:.3f}")

   # Plot
   fig, ax = plot_detection(
       data=temperature_data,
       true_cps=true_cps,
       pred_cps=result.cp_set.tolist(),
       metric_result=metrics,
       title="Sensor Anomaly Detection"
   )
   ax.set_xlabel('Time (hours)')
   ax.set_ylabel('Temperature (°C)')
   plt.show()

**Key Insight**: Use ``meanvariance`` when both mean and variance can change.

Tutorial 5: Logistic Regression with Predictors
------------------------------------------------

**Goal**: Detect changes in customer conversion rates based on features.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import logistic_regression
   from fastcpd.datasets import make_glm_change
   from fastcpd.metrics import evaluate_all

   # Generate synthetic customer data
   # Features: age, income, time_on_site, previous_purchases
   np.random.seed(42)

   data_dict = make_glm_change(
       n_samples=800,
       n_predictors=4,
       n_changepoints=2,
       family='binomial',
       coefficient_shift=1.5,
       seed=42
   )

   print(f"True change points: {data_dict['changepoints']}")
   print(f"Segment coefficients:")
   for i, coef in enumerate(data_dict['coefficients']):
       print(f"  Segment {i+1}: {coef}")
   print(f"Dataset AUC: {data_dict['AUC']:.3f}")

   # Detect coefficient changes
   result = logistic_regression(data_dict['data'], beta="MBIC")

   print(f"\nDetected change points: {result.cp_set}")

   # Evaluate
   metrics = evaluate_all(
       data_dict['changepoints'],
       result.cp_set.tolist(),
       n_samples=800,
       margin=15
   )

   print(f"F1-Score: {metrics['point_metrics']['f1_score']:.3f}")

   # Interpretation
   print("\nInterpretation:")
   print("Change points indicate shifts in customer behavior patterns")
   print("Could correspond to: marketing campaigns, product changes, etc.")

**Key Insight**: GLM models detect changes in relationship between predictors and outcome.

Tutorial 6: Time Series with ARMA
----------------------------------

**Goal**: Detect regime changes in economic time series.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import arma
   import matplotlib.pyplot as plt

   # Simulate economic time series with regime change
   np.random.seed(42)
   n = 400

   # Regime 1: ARMA(1,1) with φ=0.7, θ=0.3
   regime1 = np.zeros(200)
   errors1 = np.random.normal(0, 1, 200)
   for t in range(1, 200):
       regime1[t] = 0.7 * regime1[t-1] + errors1[t] + 0.3 * errors1[t-1]

   # Regime 2: ARMA(1,1) with φ=-0.5, θ=0.6
   regime2 = np.zeros(200)
   errors2 = np.random.normal(0, 1, 200)
   for t in range(1, 200):
       regime2[t] = -0.5 * regime2[t-1] + errors2[t] + 0.6 * errors2[t-1]

   # Combine
   time_series = np.concatenate([regime1, regime2])

   # Detect (requires statsmodels)
   try:
       result = arma(time_series, p=1, q=1, beta="MBIC")
       print(f"Detected change point: {result.cp_set}")
       print(f"True change point: [200]")

       # Plot
       plt.figure(figsize=(12, 4))
       plt.plot(time_series, linewidth=1, alpha=0.7)
       plt.axvline(200, color='gray', linestyle=':', linewidth=2,
                   label='True')
       for cp in result.cp_set:
           plt.axvline(cp, color='red', linestyle='--', linewidth=2,
                       label='Detected')
       plt.xlabel('Time')
       plt.ylabel('Value')
       plt.title('ARMA Regime Change Detection')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.show()

   except ImportError:
       print("Please install statsmodels: pip install statsmodels")

**Key Insight**: ARMA models detect changes in autocorrelation structure.

Tutorial 7: High-Dimensional Data with LASSO
---------------------------------------------

**Goal**: Detect sparse coefficient changes in high-dimensional regression.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import lasso
   from fastcpd.metrics import evaluate_all

   # Generate high-dimensional data (20 predictors, only 3 relevant)
   np.random.seed(42)
   n = 600
   p = 20

   X = np.random.randn(n, p)

   # Segment 1: only features 0, 1, 2 matter
   y1 = (2*X[:300, 0] + 3*X[:300, 1] - 1.5*X[:300, 2] +
         np.random.randn(300))

   # Segment 2: different sparse coefficients (features 5, 8, 12)
   y2 = (-1*X[300:, 5] + 2*X[300:, 8] + 1.5*X[300:, 12] +
         np.random.randn(300))

   y = np.concatenate([y1, y2])
   data = np.column_stack([y, X])

   # Detect with LASSO
   result = lasso(data, alpha=0.1, beta="MBIC")

   print(f"True change point: [300]")
   print(f"Detected: {result.cp_set}")

   # Evaluate
   metrics = evaluate_all(
       true_cps=[300],
       pred_cps=result.cp_set.tolist(),
       n_samples=600,
       margin=20
   )

   print(f"F1-Score: {metrics['point_metrics']['f1_score']:.3f}")

   # Identify which features changed
   print("\nActive features by segment:")
   print("Segment 1: [0, 1, 2] (ground truth)")
   print("Segment 2: [5, 8, 12] (ground truth)")

**Key Insight**: LASSO detects changes in sparse coefficient structure.

Tutorial 8: Multi-Annotator Evaluation
---------------------------------------

**Goal**: Handle uncertainty in ground truth annotations.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import mean
   from fastcpd.metrics import covering_metric
   from fastcpd.visualization import plot_multi_annotator
   import matplotlib.pyplot as plt

   # Generate data
   np.random.seed(42)
   data = np.concatenate([
       np.random.normal(0, 1, 200),
       np.random.normal(3, 1, 200),
       np.random.normal(1, 1, 200)
   ])

   # Simulate 3 expert annotations (with disagreement)
   annotations = [
       [200, 400],      # Expert 1
       [195, 405],      # Expert 2
       [205, 398]       # Expert 3
   ]

   # Detect
   result = mean(data, beta="MBIC")

   # Evaluate covering metric
   covering = covering_metric(
       annotations,
       result.cp_set.tolist(),
       margin=10
   )

   print(f"Detected change points: {result.cp_set}")
   print(f"Covering metric: {covering:.3f}")
   print("(Higher is better, measures agreement with multiple annotators)")

   # Visualize
   fig, ax = plot_multi_annotator(
       data=data,
       annotations=annotations,
       pred_cps=result.cp_set.tolist(),
       annotator_names=["Expert 1", "Expert 2", "Expert 3"],
       margin=10
   )
   plt.show()

**Key Insight**: Use covering metric when ground truth is uncertain.

Tutorial 9: Batch Processing Multiple Datasets
-----------------------------------------------

**Goal**: Process multiple time series efficiently.

.. code-block:: python

   import numpy as np
   from fastcpd.segmentation import mean
   from fastcpd.metrics import f1_score
   import pandas as pd

   # Simulate multiple sensors
   n_sensors = 10
   results_summary = []

   for sensor_id in range(n_sensors):
       # Generate data for this sensor
       np.random.seed(sensor_id)
       n_cps = np.random.randint(1, 4)
       cp_locations = sorted(np.random.choice(range(100, 900), n_cps, replace=False))

       # Generate segments
       segments = []
       prev_cp = 0
       for cp in cp_locations + [1000]:
           segment_len = cp - prev_cp
           segment_mean = np.random.normal(0, 5)
           segments.append(np.random.normal(segment_mean, 1, segment_len))
           prev_cp = cp

       data = np.concatenate(segments)

       # Detect
       result = mean(data, beta="MBIC")

       # Evaluate
       f1 = f1_score(
           cp_locations,
           result.cp_set.tolist(),
           n_samples=1000,
           margin=10
       )['f1_score']

       # Store results
       results_summary.append({
           'sensor_id': sensor_id,
           'n_true_cps': n_cps,
           'n_detected_cps': len(result.cp_set),
           'f1_score': f1
       })

   # Summary statistics
   df = pd.DataFrame(results_summary)
   print(df)
   print(f"\nAverage F1-Score: {df['f1_score'].mean():.3f}")
   print(f"Detection rate: {(df['n_detected_cps'] == df['n_true_cps']).mean():.1%}")

**Key Insight**: Batch processing enables systematic evaluation across datasets.

Tutorial 10: Custom Visualization
----------------------------------

**Goal**: Create publication-ready figures.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from fastcpd.segmentation import mean
   from fastcpd.datasets import make_mean_change

   # Set publication style
   plt.rcParams['font.size'] = 12
   plt.rcParams['axes.labelsize'] = 14
   plt.rcParams['axes.titlesize'] = 16
   plt.rcParams['lines.linewidth'] = 2
   plt.rcParams['figure.dpi'] = 100

   # Generate data
   np.random.seed(42)
   data_dict = make_mean_change(n_samples=600, n_changepoints=3, seed=42)

   # Detect
   result = mean(data_dict['data'], beta="MBIC")

   # Create custom plot
   fig, ax = plt.subplots(figsize=(10, 5))

   # Plot data
   ax.plot(data_dict['data'], linewidth=1.5, alpha=0.8,
           color='steelblue', label='Observed Data')

   # Plot true change points
   for cp in data_dict['changepoints']:
       ax.axvline(cp, color='green', linestyle=':', linewidth=2,
                  alpha=0.7, label='Ground Truth' if cp == data_dict['changepoints'][0] else '')

   # Plot detected change points
   for cp in result.cp_set:
       ax.axvline(cp, color='red', linestyle='--', linewidth=2,
                  label='Detected' if cp == result.cp_set[0] else '')

   # Formatting
   ax.set_xlabel('Time Index', fontsize=14)
   ax.set_ylabel('Signal Value', fontsize=14)
   ax.set_title('Change Point Detection Results', fontsize=16, fontweight='bold')
   ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
   ax.legend(loc='upper right', fontsize=12, framealpha=0.95)

   # Add annotations
   ax.text(0.02, 0.98, f'F1-Score: 1.000', transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.5), fontsize=11)

   plt.tight_layout()

   # Save high-quality figure
   fig.savefig('publication_figure.pdf', dpi=300, bbox_inches='tight')
   fig.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
   plt.show()

**Key Insight**: Customize plots for specific publication requirements.

Next Steps
----------

- Explore :doc:`../api/detection` for advanced API options
- Read :doc:`../advanced/algorithms` to understand PELT and SeGD
- See :doc:`../advanced/comparison` for comparisons with other packages
