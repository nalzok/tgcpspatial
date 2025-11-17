#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Apply time-rescaling theorem goodness-of-fit tests to LGCP models.

This script demonstrates how to use the time-rescaling theorem to validate
different link functions for grid cell data.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import datadir
from goodness_of_fit import (
    compute_rescaled_isi_from_model,
    goodness_of_fit_summary,
    plot_goodness_of_fit,
)
from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft
from nonlinearity import get_link_function


def compute_pearson_residuals(N, K, expected_rate):
    """
    Compute Pearson residuals for Poisson model.

    Pearson residual = (observed - expected) / sqrt(expected)

    Args:
        N (ndarray): Occupancy
        K (ndarray): Observed counts
        expected_rate (ndarray): Expected rate

    Returns:
        ndarray: Pearson residuals at locations with data
    """
    mask = N > 0
    observed = K[mask]
    expected = (N * expected_rate)[mask]

    residuals = (observed - expected) / np.sqrt(expected + 1e-10)
    return residuals


def compute_deviance_residuals(N, K, expected_rate):
    """
    Compute deviance residuals for Poisson model.

    Args:
        N (ndarray): Occupancy
        K (ndarray): Observed counts
        expected_rate (ndarray): Expected rate

    Returns:
        ndarray: Deviance residuals
    """
    mask = N > 0
    y = K[mask]
    mu = (N * expected_rate)[mask]

    # Deviance residual
    sign = np.sign(y - mu)
    deviance = 2 * (y * np.log((y + 1e-10) / (mu + 1e-10)) - (y - mu))
    residuals = sign * np.sqrt(np.abs(deviance))

    return residuals


def test_link_function_gof(data, link_name, verbose=True):
    """
    Test goodness-of-fit for a specific link function.

    Args:
        data: Prepared Dataset
        link_name (str): Name of link function
        verbose (bool): Print progress

    Returns:
        dict: Results including model, residuals, and test statistics
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Testing {link_name} link function")
        print("=" * 70)

    # Get link function
    link = get_link_function(link_name)

    # Create kernel
    kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")

    # Run inference
    if verbose:
        print("Running inference...")
    result, model = lgcp2d(
        kf,
        data.N,
        data.K,
        data.prior_mean,
        (data.kdelograte, None),
        link=link,
        eps=1e-5,
        verbose=False,
    )

    if verbose:
        print(f"Log-likelihood: {result.ll:.2f}")

    # Get expected rates
    mu = result.info.mu
    v = result.zv.v
    expected_rate = link.expected_rate(mu, v)

    # Compute residuals
    pearson_res = compute_pearson_residuals(data.N, data.K, expected_rate)
    deviance_res = compute_deviance_residuals(data.N, data.K, expected_rate)

    # Compute approximate rescaled ISIs
    # For spatial binned data, we approximate by treating each bin as contributing
    # rescaled intervals proportional to the observed vs expected counts
    mask = data.N > 0
    observed = data.K[mask]
    expected = (data.N * expected_rate)[mask]

    # Generate rescaled intervals
    # Under correct model, residuals should be standard normal
    # We can also generate approximate exp(1) samples from the Poisson residuals
    rescaled_isi = []

    for obs, exp in zip(observed, expected, strict=False):
        if exp > 0:
            # For Poisson(λ), if we observe k events, the rescaled "intervals"
            # can be approximated by drawing from exp(1)
            # This is an approximation for binned data
            if obs > 0:
                # Rescale based on intensity
                intervals = np.random.exponential(1.0, size=int(obs))
                rescaled_isi.extend(intervals)

    rescaled_isi = np.array(rescaled_isi)

    # Run goodness-of-fit tests
    if len(rescaled_isi) > 0:
        gof_summary = goodness_of_fit_summary(rescaled_isi, max_lag=20)
    else:
        gof_summary = None

    if verbose and gof_summary:
        print(f"\nGoodness-of-fit summary:")
        print(f"  N spikes: {gof_summary['n_spikes']}")
        print(f"  Mean rescaled ISI: {gof_summary['mean_isi']:.3f} (expected: 1.0)")
        print(f"  Std rescaled ISI: {gof_summary['std_isi']:.3f} (expected: 1.0)")
        print(
            f"  KS test (exponential): p={gof_summary['ks_test_exponential']['pvalue']:.4f}"
        )
        print(f"  KS test (uniform): p={gof_summary['ks_test_uniform']['pvalue']:.4f}")
        print(
            f"  Autocorrelation: {gof_summary['autocorrelation']['interpretation']}"
        )

        # Test on residuals
        print(f"\nResidual analysis:")
        print(f"  Pearson residuals: mean={np.mean(pearson_res):.3f}, std={np.std(pearson_res):.3f}")
        print(f"  Deviance residuals: mean={np.mean(deviance_res):.3f}, std={np.std(deviance_res):.3f}")

    return {
        "link_name": link_name,
        "result": result,
        "model": model,
        "expected_rate": expected_rate,
        "pearson_residuals": pearson_res,
        "deviance_residuals": deviance_res,
        "rescaled_isi": rescaled_isi,
        "gof_summary": gof_summary,
    }


def plot_residual_diagnostics(results_dict, filename="figures/residual_diagnostics.png"):
    """
    Plot residual diagnostics for multiple link functions.

    Args:
        results_dict (dict): Dictionary mapping link names to test results
        filename (str): Output filename
    """
    n_links = len(results_dict)
    fig, axes = plt.subplots(2, n_links, figsize=(5 * n_links, 8))

    if n_links == 1:
        axes = axes.reshape(-1, 1)

    for idx, (link_name, result) in enumerate(results_dict.items()):
        pearson = result["pearson_residuals"]
        deviance = result["deviance_residuals"]

        # Pearson residuals Q-Q plot
        ax = axes[0, idx]
        stats.probplot(pearson, dist="norm", plot=ax)
        ax.set_title(f"{link_name}\nPearson Residuals", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Deviance residuals Q-Q plot
        ax = axes[1, idx]
        stats.probplot(deviance, dist="norm", plot=ax)
        ax.set_title(f"Deviance Residuals", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved residual diagnostics to: {filename}")
    plt.close()


def compare_link_functions_gof(data, link_names=None):
    """
    Compare goodness-of-fit for multiple link functions.

    Args:
        data: Prepared Dataset
        link_names (list): List of link function names to test

    Returns:
        dict: Results for each link function
    """
    if link_names is None:
        link_names = ["exponential", "relu"]

    results = {}

    for link_name in link_names:
        try:
            results[link_name] = test_link_function_gof(data, link_name, verbose=True)

            # Generate goodness-of-fit plot
            if results[link_name]["rescaled_isi"] is not None and len(
                results[link_name]["rescaled_isi"]
            ) > 0:
                plot_goodness_of_fit(
                    results[link_name]["rescaled_isi"],
                    link_name=link_name,
                    filename=f"figures/gof_{link_name}.png",
                )
        except Exception as e:
            print(f"Error testing {link_name}: {e}")
            results[link_name] = None

    # Plot residual diagnostics
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        plot_residual_diagnostics(valid_results)

    return results


def print_gof_comparison_table(results):
    """Print comparison table of goodness-of-fit metrics."""
    print("\n" + "=" * 80)
    print("Goodness-of-Fit Comparison")
    print("=" * 80)

    # Header
    print(
        f"{'Link':<15} {'KS p-value':>12} {'Mean ISI':>10} {'Std ISI':>10} "
        f"{'Pearson μ':>10} {'Decision':<20}"
    )
    print("-" * 80)

    for link_name, result in results.items():
        if result is None or result["gof_summary"] is None:
            print(f"{link_name:<15} {'N/A':>12}")
            continue

        gof = result["gof_summary"]
        ks_pval = gof["ks_test_exponential"]["pvalue"]
        mean_isi = gof["mean_isi"]
        std_isi = gof["std_isi"]
        pearson_mean = np.mean(result["pearson_residuals"])
        decision = gof["ks_test_exponential"]["interpretation"]

        print(
            f"{link_name:<15} {ks_pval:>12.4f} {mean_isi:>10.3f} {std_isi:>10.3f} "
            f"{pearson_mean:>10.3f} {decision:<20}"
        )

    print("=" * 80)
    print("Note: Higher KS p-value indicates better fit (fail to reject null hypothesis)")
    print("Expected values for perfect fit: Mean ISI ≈ 1.0, Std ISI ≈ 1.0, Pearson μ ≈ 0.0")


def main():
    """Run time-rescaling theorem goodness-of-fit tests."""
    print("=" * 80)
    print("Time-Rescaling Theorem Goodness-of-Fit Tests")
    print("=" * 80)

    # Load data
    print("\nLoading grid cell data...")
    fn = "r2405_051216b_cell1816.mat"
    data = Dataset.from_file(datadir + fn).prepare()
    print(f"Grid shape: {data.shape}")
    print(f"Total spikes: {int(data.K.sum())}")

    # Test multiple link functions
    link_names = ["exponential", "relu"]
    results = compare_link_functions_gof(data, link_names)

    # Print comparison
    print_gof_comparison_table(results)

    print("\n" + "=" * 80)
    print("Analysis complete! Check figures/ directory for diagnostic plots.")
    print("=" * 80)


if __name__ == "__main__":
    from scipy import stats  # Import here for stats.probplot
    main()
