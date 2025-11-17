#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Run comprehensive evaluation of link functions and generate comparison plots.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import datadir
from evaluate_links import (
    compare_link_functions,
    compute_predictive_metrics,
    print_comparison_table,
    train_test_split,
)
from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft


def load_and_prepare_data(filename):
    """Load and prepare dataset."""
    print(f"Loading data from: {filename}")
    data = Dataset.from_file(datadir + filename).prepare()
    print(f"  Grid shape: {data.shape}")
    print(f"  Total spikes: {np.sum(data.K):.0f}")
    print(f"  Total time: {np.sum(data.N):.2f}s")
    print(f"  Mean rate: {np.sum(data.K)/np.sum(data.N):.2f} Hz")
    return data


def plot_train_test_split(data, N_train, K_train, test_mask, filename="figures/train_test_split.png"):
    """Visualize train/test split."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Training data
    ax = axes[0]
    rate_train = np.where(N_train > 0, K_train / N_train, np.nan)
    im = data.arena.imshow(rate_train, lw=3, domask=True)
    ax.set_title("Training Data", fontsize=12, fontweight="bold")

    # Test mask
    ax = axes[1]
    test_viz = test_mask.astype(float)
    test_viz[~data.arena.mask] = np.nan
    im = ax.imshow(test_viz, extent=data.arena.extent, cmap="coolwarm")
    ax.plot(*data.arena.perim_m.T, color="white", lw=3)
    ax.set_title("Test Bins (20%)", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Full data
    ax = axes[2]
    rate_full = np.where(data.N > 0, data.K / data.N, np.nan)
    data.arena.imshow(rate_full, lw=3, domask=True)
    ax.set_title("Full Dataset", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved train/test split visualization to: {filename}")
    plt.close()


def plot_comparison_metrics(results, filename="figures/comparison_metrics.png"):
    """Plot comparison of different metrics across link functions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Filter out failed evaluations
    valid_results = {k: v for k, v in results.items() if v is not None}
    link_names = list(valid_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(link_names)))

    # Extract metrics
    metrics = {
        "ELBO": [valid_results[l]["elbo"] for l in link_names],
        "Train LL": [valid_results[l]["train_ll"] for l in link_names],
        "Validation LL": [valid_results[l]["validation_ll"] for l in link_names],
        "MSE": [valid_results[l].get("mse", 0) for l in link_names],
        "R²": [valid_results[l].get("r2", 0) for l in link_names],
        "Deviance": [valid_results[l].get("deviance", 0) for l in link_names],
    }

    # Plot each metric
    for idx, (metric_name, values) in enumerate(metrics.items()):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        bars = ax.bar(range(len(link_names)), values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(link_names)))
        ax.set_xticklabels(link_names, rotation=45, ha="right")
        ax.set_ylabel(metric_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Highlight best
        if metric_name in ["ELBO", "Train LL", "Validation LL", "R²"]:
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor("red")
            bars[best_idx].set_linewidth(2.5)
        elif metric_name in ["MSE", "Deviance"]:
            best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor("red")
            bars[best_idx].set_linewidth(2.5)

        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars, values, strict=False)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.1f}" if abs(val) > 1 else f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle(
        "Link Function Performance Comparison", fontsize=14, fontweight="bold", y=1.00
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved metrics comparison to: {filename}")
    plt.close()


def plot_rate_map_comparison(data, results, filename="figures/rate_map_comparison.png"):
    """Plot rate maps from different link functions."""
    # Filter out failed evaluations
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_links = len(valid_results)
    fig, axes = plt.subplots(1, n_links, figsize=(5 * n_links, 4))

    if n_links == 1:
        axes = [axes]

    for idx, (link_name, result) in enumerate(valid_results.items()):
        ax = axes[idx]
        plt.sca(ax)

        # Get predicted rate map
        rate_map = result["result"].info.r

        # Plot
        data.arena.imshow(rate_map, lw=5, domask=True)
        ax.set_title(
            f"{link_name}\nVal LL: {result['validation_ll']:.1f}",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved rate map comparison to: {filename}")
    plt.close()


def plot_prediction_scatter(data, results, filename="figures/prediction_scatter.png"):
    """Scatter plot of predicted vs observed rates."""
    # Filter out failed evaluations
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_links = len(valid_results)
    fig, axes = plt.subplots(1, n_links, figsize=(5 * n_links, 4))

    if n_links == 1:
        axes = [axes]

    for idx, (link_name, result) in enumerate(valid_results.items()):
        ax = axes[idx]

        # Get test data
        test_mask = result["test_mask"]
        if np.sum(test_mask) == 0:
            continue

        # Predictions
        from nonlinearity import get_link_function

        link = get_link_function(link_name)
        mu_test = result["result"].info.mu[test_mask]
        v_test = result["result"].zv.v[test_mask]
        predicted = link.expected_rate(mu_test, v_test)

        # Observed
        observed = result["K_test"][test_mask] / result["N_test"][test_mask]

        # Scatter
        ax.scatter(observed, predicted, alpha=0.5, s=20, edgecolors="black", linewidth=0.5)

        # Diagonal line
        max_val = max(np.max(observed), np.max(predicted))
        ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect prediction")

        # Labels
        ax.set_xlabel("Observed rate (Hz)", fontsize=11)
        ax.set_ylabel("Predicted rate (Hz)", fontsize=11)
        ax.set_title(
            f"{link_name}\nR² = {result.get('r2', 0):.3f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved prediction scatter to: {filename}")
    plt.close()


def plot_uncertainty_vs_error(data, results, filename="figures/uncertainty_vs_error.png"):
    """Plot how prediction error relates to posterior uncertainty."""
    # Filter out failed evaluations
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_links = len(valid_results)
    fig, axes = plt.subplots(1, n_links, figsize=(5 * n_links, 4))

    if n_links == 1:
        axes = [axes]

    for idx, (link_name, result) in enumerate(valid_results.items()):
        ax = axes[idx]

        # Get test data
        test_mask = result["test_mask"]
        if np.sum(test_mask) == 0:
            continue

        # Predictions and uncertainty
        from nonlinearity import get_link_function

        link = get_link_function(link_name)
        mu_test = result["result"].info.mu[test_mask]
        v_test = result["result"].zv.v[test_mask]
        predicted = link.expected_rate(mu_test, v_test)
        uncertainty = np.sqrt(v_test)

        # Observed and error
        observed = result["K_test"][test_mask] / result["N_test"][test_mask]
        error = np.abs(predicted - observed)

        # Scatter with color by density
        ax.scatter(uncertainty, error, alpha=0.5, s=20, edgecolors="black", linewidth=0.5)

        # Labels
        ax.set_xlabel("Posterior std dev", fontsize=11)
        ax.set_ylabel("Absolute error (Hz)", fontsize=11)
        ax.set_title(f"{link_name}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Correlation
        corr = np.corrcoef(uncertainty, error)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"ρ = {corr:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved uncertainty vs error plot to: {filename}")
    plt.close()


def main():
    """Run full evaluation pipeline."""
    print("=" * 80)
    print("Link Function Evaluation")
    print("=" * 80)

    # Load data
    data = load_and_prepare_data("r2405_051216b_cell1816.mat")

    # Compare multiple link functions
    # Note: ReLU has numerical integration which is slower
    # Squared link currently has numerical stability issues
    link_names = ["exponential", "relu"]

    # Run evaluation
    print("\n" + "=" * 80)
    print("Running evaluation with train/test split (20% held out)...")
    print("=" * 80)

    results = compare_link_functions(
        data, link_names=link_names, test_fraction=0.2, verbose=True, eps=1e-5
    )

    # Print results
    print_comparison_table(results)

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)

    # Get train/test split for visualization
    result = results["exponential"]
    plot_train_test_split(
        data, result["N_train"], result["K_train"], result["test_mask"]
    )
    plot_comparison_metrics(results)
    plot_rate_map_comparison(data, results)
    plot_prediction_scatter(data, results)
    plot_uncertainty_vs_error(data, results)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
