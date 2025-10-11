"""Visualization utilities for change point detection.

This module provides plotting functions to visualize:
- Change point detection results
- Metric comparisons across algorithms
- Multi-annotator scenarios
- Dataset characteristics
"""

import numpy as np
from typing import Union, List, Dict, Optional, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn(
        "matplotlib not installed. Install with: pip install matplotlib",
        ImportWarning
    )


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_detection(
    data: Union[np.ndarray, List],
    true_cps: Union[List, np.ndarray],
    pred_cps: Union[List, np.ndarray],
    metric_result: Optional[Dict] = None,
    title: str = "Change Point Detection",
    figsize: tuple = (14, 8),
    show_legend: bool = True,
    save_path: Optional[str] = None
) -> tuple:
    """Plot change point detection results.

    Args:
        data: Time series data (1D or 2D array)
        true_cps: True change point indices
        pred_cps: Predicted change point indices
        metric_result: Optional dict from metrics.evaluate_all()
        title: Plot title
        figsize: Figure size (width, height)
        show_legend: Whether to show legend
        save_path: Optional path to save figure

    Returns:
        Tuple of (fig, axes)

    Examples:
        >>> from fastcpd.visualization import plot_detection
        >>> from fastcpd.datasets import make_mean_change
        >>> from fastcpd import fastcpd
        >>> from fastcpd.metrics import evaluate_all
        >>>
        >>> # Generate data
        >>> data_dict = make_mean_change(n_samples=500, n_changepoints=3)
        >>> result = fastcpd(data_dict['data'], family='mean')
        >>>
        >>> # Evaluate and plot
        >>> metrics = evaluate_all(data_dict['changepoints'], result.cp_set.tolist(),
        ...                        n_samples=500, margin=10)
        >>> plot_detection(data_dict['data'], data_dict['changepoints'],
        ...               result.cp_set.tolist(), metrics)
    """
    _check_matplotlib()

    data = np.asarray(data)
    true_cps = np.asarray(true_cps)
    pred_cps = np.asarray(pred_cps)

    # Create figure with optional metrics panel
    if metric_result is not None:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_metrics = fig.add_subplot(gs[1])
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
        ax_metrics = None

    # Plot data
    if data.ndim == 1:
        ax_main.plot(data, 'k-', linewidth=0.8, alpha=0.7, label='Data')
    else:
        # Plot first 3 dimensions
        for i in range(min(3, data.shape[1])):
            ax_main.plot(data[:, i], linewidth=0.8, alpha=0.7, label=f'Dim {i+1}')

    # Plot true change points
    for i, cp in enumerate(true_cps):
        label = 'True CP' if i == 0 else None
        ax_main.axvline(x=cp, color='green', linestyle='--', linewidth=2,
                       alpha=0.8, label=label)

    # Plot predicted change points
    for i, cp in enumerate(pred_cps):
        label = 'Predicted CP' if i == 0 else None
        ax_main.axvline(x=cp, color='red', linestyle=':', linewidth=2,
                       alpha=0.8, label=label)

    ax_main.set_xlabel('Time', fontsize=12)
    ax_main.set_ylabel('Value', fontsize=12)
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    if show_legend:
        ax_main.legend(loc='upper right', fontsize=10)

    # Add metrics panel if provided
    if metric_result is not None and ax_metrics is not None:
        ax_metrics.axis('off')

        # Extract key metrics
        pm = metric_result.get('point_metrics', {})
        dm = metric_result.get('distance_metrics', {})
        sm = metric_result.get('segmentation_metrics', {})

        metrics_text = (
            f"Precision: {pm.get('precision', 0):.3f}  |  "
            f"Recall: {pm.get('recall', 0):.3f}  |  "
            f"F1: {pm.get('f1_score', 0):.3f}  |  "
            f"Hausdorff: {dm.get('hausdorff', np.nan):.1f}  |  "
            f"ARI: {sm.get('adjusted_rand_index', 0):.3f}\n"
            f"TP: {pm.get('true_positives', 0)}  |  "
            f"FP: {pm.get('false_positives', 0)}  |  "
            f"FN: {pm.get('false_negatives', 0)}"
        )

        ax_metrics.text(0.5, 0.5, metrics_text,
                       transform=ax_metrics.transAxes,
                       fontsize=11, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, (ax_main, ax_metrics) if ax_metrics else (ax_main,)


def plot_metric_comparison(
    results_dict: Dict[str, Dict],
    metrics: List[str] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> tuple:
    """Compare metrics across multiple algorithms.

    Args:
        results_dict: Dict mapping algorithm names to metric results
                     {'PELT': metrics_dict, 'SeGD': metrics_dict, ...}
        metrics: List of metrics to compare (default: precision, recall, f1)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Tuple of (fig, ax)

    Examples:
        >>> from fastcpd.metrics import evaluate_all
        >>> from fastcpd.visualization import plot_metric_comparison
        >>>
        >>> # Assume you have multiple algorithm results
        >>> pelt_metrics = evaluate_all(true_cps, pelt_cps, n_samples=500, margin=10)
        >>> segd_metrics = evaluate_all(true_cps, segd_cps, n_samples=500, margin=10)
        >>>
        >>> results = {'PELT': pelt_metrics, 'SeGD': segd_metrics}
        >>> plot_metric_comparison(results)
    """
    _check_matplotlib()

    if metrics is None:
        metrics = ['precision', 'recall', 'f1_score']

    fig, ax = plt.subplots(figsize=figsize)

    algorithms = list(results_dict.keys())
    n_algorithms = len(algorithms)
    n_metrics = len(metrics)

    # Prepare data
    metric_values = {metric: [] for metric in metrics}

    for algo in algorithms:
        result = results_dict[algo]
        pm = result.get('point_metrics', {})

        for metric in metrics:
            value = pm.get(metric, 0)
            metric_values[metric].append(value)

    # Plot grouped bar chart
    x = np.arange(n_algorithms)
    width = 0.8 / n_metrics

    for i, metric in enumerate(metrics):
        offset = (i - n_metrics/2 + 0.5) * width
        ax.bar(x + offset, metric_values[metric], width,
               label=metric.replace('_', ' ').title())

    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_annotators(
    data: Union[np.ndarray, List],
    annotators_list: List[List],
    pred_cps: Union[List, np.ndarray],
    title: str = "Multi-Annotator Change Point Detection",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None
) -> tuple:
    """Visualize multi-annotator scenario.

    Args:
        data: Time series data
        annotators_list: List of lists, each containing one annotator's CPs
        pred_cps: Predicted change points
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Tuple of (fig, ax)

    Examples:
        >>> from fastcpd.datasets import add_annotation_noise
        >>> from fastcpd.visualization import plot_annotators
        >>>
        >>> true_cps = [100, 200, 300]
        >>> annotators = add_annotation_noise(true_cps, n_annotators=5, seed=42)
        >>> pred_cps = [98, 205, 295]
        >>>
        >>> plot_annotators(data, annotators, pred_cps)
    """
    _check_matplotlib()

    data = np.asarray(data)
    pred_cps = np.asarray(pred_cps)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot data
    if data.ndim == 1:
        ax.plot(data, 'k-', linewidth=0.8, alpha=0.5, label='Data')
    else:
        ax.plot(data[:, 0], 'k-', linewidth=0.8, alpha=0.5, label='Data')

    # Plot each annotator's CPs with different colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(annotators_list)))

    for i, (annotator_cps, color) in enumerate(zip(annotators_list, colors)):
        for j, cp in enumerate(annotator_cps):
            label = f'Annotator {i+1}' if j == 0 else None
            ax.axvline(x=cp, color=color, linestyle='--', linewidth=1.5,
                      alpha=0.6, label=label)

    # Plot predictions
    for i, cp in enumerate(pred_cps):
        label = 'Algorithm' if i == 0 else None
        ax.axvline(x=cp, color='red', linestyle='-', linewidth=2.5,
                  alpha=0.8, label=label)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_dataset_characteristics(
    data_dict: Dict[str, Any],
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None
) -> tuple:
    """Visualize dataset characteristics and metadata.

    Args:
        data_dict: Dictionary from datasets module (e.g., make_mean_change)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Tuple of (fig, axes)

    Examples:
        >>> from fastcpd.datasets import make_mean_change
        >>> from fastcpd.visualization import plot_dataset_characteristics
        >>>
        >>> data_dict = make_mean_change(n_samples=500, n_changepoints=3, seed=42)
        >>> plot_dataset_characteristics(data_dict)
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, hspace=0.3, wspace=0.3)

    data = np.asarray(data_dict['data'])
    changepoints = data_dict['changepoints']
    metadata = data_dict.get('metadata', {})

    # 1. Data plot with CPs
    ax1 = fig.add_subplot(gs[0, :])
    if data.ndim == 1:
        ax1.plot(data, 'b-', linewidth=0.8, alpha=0.7)
    else:
        for i in range(min(3, data.shape[1])):
            ax1.plot(data[:, i], linewidth=0.8, alpha=0.7, label=f'Dim {i+1}')

    for cp in changepoints:
        ax1.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax1.set_title('Data with True Change Points', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    if data.ndim > 1:
        ax1.legend()

    # 2. Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('Data Distribution', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Segment lengths
    ax3 = fig.add_subplot(gs[1, 1])
    segment_lengths = metadata.get('segment_lengths', [])
    if segment_lengths:
        ax3.bar(range(len(segment_lengths)), segment_lengths, alpha=0.7, edgecolor='black')
        ax3.set_title('Segment Lengths', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Segment')
        ax3.set_ylabel('Length')
        ax3.grid(True, alpha=0.3, axis='y')

    # 4. Metadata text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    metadata_text = "Dataset Metadata:\n"
    metadata_text += f"  n_samples: {len(data) if data.ndim == 1 else data.shape[0]}\n"
    metadata_text += f"  n_changepoints: {len(changepoints)}\n"

    # Add specific metadata
    if 'snr_db' in metadata:
        metadata_text += f"  SNR: {metadata['snr_db']:.2f} dB\n"
    if 'difficulty' in metadata:
        metadata_text += f"  Difficulty: {metadata['difficulty']:.3f}\n"
    if 'change_type' in metadata:
        metadata_text += f"  Change type: {metadata['change_type']}\n"
    if 'r_squared_per_segment' in metadata:
        r2_vals = metadata['r_squared_per_segment']
        metadata_text += f"  R² per segment: {[f'{r:.3f}' for r in r2_vals]}\n"
    if 'family' in metadata:
        metadata_text += f"  Family: {metadata['family']}\n"

    ax4.text(0.5, 0.5, metadata_text, transform=ax4.transAxes,
            fontsize=11, ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, (ax1, ax2, ax3, ax4)


def plot_roc_curve(
    true_cps: Union[List, np.ndarray],
    pred_cps_list: List[Union[List, np.ndarray]],
    labels: List[str],
    n_samples: int,
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None
) -> tuple:
    """Plot ROC-like curve for change point detection.

    For each threshold (margin), computes precision and recall.

    Args:
        true_cps: True change points
        pred_cps_list: List of predicted CP arrays (one per algorithm)
        labels: Algorithm labels
        n_samples: Total number of samples
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Tuple of (fig, ax)
    """
    _check_matplotlib()

    from fastcpd.metrics import precision_recall

    fig, ax = plt.subplots(figsize=figsize)

    margins = [1, 2, 5, 10, 20, 50, 100]

    for pred_cps, label in zip(pred_cps_list, labels):
        precisions = []
        recalls = []

        for margin in margins:
            result = precision_recall(true_cps, pred_cps, margin=margin)
            precisions.append(result['precision'])
            recalls.append(result['recall'])

        ax.plot(recalls, precisions, 'o-', linewidth=2, label=label)

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve (varying margin)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
