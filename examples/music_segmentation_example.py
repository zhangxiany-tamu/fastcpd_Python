"""
Music Segmentation Example using fastcpd
=========================================

This example demonstrates how to use fastcpd for music segmentation,
detecting structural changes in a music signal using tempogram representation.

Requirements:
    pip install librosa matplotlib pyfastcpd

Author: fastcpd team
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from fastcpd.segmentation import rbf, fastcpd
import os


def fig_ax(figsize=(15, 5), dpi=150):
    """Return a (matplotlib) figure and ax objects with given size."""
    return plt.subplots(figsize=figsize, dpi=dpi)


def main():
    # Create output directory for images
    output_dir = "docs/images/music_segmentation"
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Music Segmentation with fastcpd")
    print("=" * 60)

    # Load the data
    print("\n1. Loading music data...")
    duration = 30  # in seconds
    signal, sampling_rate = librosa.load(
        librosa.ex("nutcracker"),
        duration=duration
    )
    print(f"   Loaded {duration}s of audio at {sampling_rate} Hz")
    print(f"   Signal shape: {signal.shape}")

    # Visualize the sound envelope
    print("\n2. Visualizing sound envelope...")
    fig, ax = fig_ax()
    ax.plot(np.arange(signal.size) / sampling_rate, signal)
    ax.set_xlim(0, signal.size / sampling_rate)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sound Envelope - Dance of the Sugar Plum Fairy")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "music_segmentation_envelope.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    # Compute the tempogram
    print("\n3. Computing tempogram representation...")
    hop_length_tempo = 256
    oenv = librosa.onset.onset_strength(
        y=signal,
        sr=sampling_rate,
        hop_length=hop_length_tempo
    )
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv,
        sr=sampling_rate,
        hop_length=hop_length_tempo,
    )
    print(f"   Tempogram shape: {tempogram.shape}")

    # Display the tempogram
    print("\n4. Visualizing tempogram...")
    fig, ax = fig_ax()
    librosa.display.specshow(
        tempogram,
        ax=ax,
        hop_length=hop_length_tempo,
        sr=sampling_rate,
        x_axis="s",
        y_axis="tempo",
    )
    ax.set_title("Tempogram")
    plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "music_segmentation_tempogram.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    # Detect change points
    print("\n5. Detecting change points with fastcpd...")
    data = tempogram.T  # Transpose to have time as rows

    # Use RBF kernel for nonparametric detection (better for tempogram data)
    # With the corrected cost function, smaller beta values are needed
    result = rbf(data, beta=1.0)

    print(f"   Detected {len(result.cp_set)} change points")
    print(f"   Change point locations (frames): {result.cp_set}")

    # Convert to timestamps
    bkps_times = librosa.frames_to_time(
        result.cp_set,
        sr=sampling_rate,
        hop_length=hop_length_tempo
    )

    print(f"\n6. Change point times:")
    for i, t in enumerate(bkps_times, 1):
        print(f"   Change point {i}: {t:.2f} seconds")

    # Visualize results
    print("\n7. Visualizing segmentation results...")
    fig, ax = fig_ax()
    librosa.display.specshow(
        tempogram,
        ax=ax,
        x_axis="s",
        y_axis="tempo",
        hop_length=hop_length_tempo,
        sr=sampling_rate,
    )

    # Draw vertical lines at change points
    for i, b in enumerate(bkps_times, 1):
        ax.axvline(b, ls="--", color="white", lw=3)

    ax.set_title(f"Tempogram with {len(result.cp_set)} Detected Change Points (fastcpd RBF)")
    plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "music_segmentation_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    # Analyze segments
    print("\n8. Segment analysis:")
    # Convert change point times to sample indices and add boundaries
    bkps_sample_indices = (sampling_rate * bkps_times).astype(int).tolist()
    all_boundaries = [0] + bkps_sample_indices + [len(signal)]

    for i, (start, end) in enumerate(zip(all_boundaries[:-1], all_boundaries[1:]), 1):
        segment = signal[start:end]
        seg_duration = segment.size / sampling_rate
        seg_energy = np.mean(segment ** 2)
        print(f"   Segment {i}:")
        print(f"      Duration: {seg_duration:.2f} seconds")
        print(f"      Energy: {seg_energy:.6f}")

    print("\n" + "=" * 60)
    print("Music segmentation complete!")
    print("=" * 60)

    # Demonstrate different penalty values
    print("\n9. Effect of penalty parameter:")
    for test_beta in [10, 5, 1, 0.5]:
        result_test = rbf(data, beta=test_beta)
        print(f"   Beta={test_beta}: {len(result_test.cp_set)} change points")

    # Compare with parametric mean model
    print("\n10. Comparison with parametric model:")
    result_mean = fastcpd(data, family="mean", beta="MBIC")
    print(f"   RBF kernel (nonparametric): {len(result.cp_set)} change points")
    print(f"   Mean model (parametric): {len(result_mean.cp_set)} change points")
    print(f"   -> RBF is better for complex tempogram data!")


if __name__ == "__main__":
    main()
