#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Evaluation framework for comparing link functions.

Provides metrics for assessing goodness-of-fit:
- ELBO (Evidence Lower Bound)
- Train/validation log-likelihood
- Predictive performance metrics
"""

import numpy as np
from scipy.special import gammaln

from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft
from lgcp.util import sdiv, sexp, slog
from nonlinearity import get_link_function


def compute_elbo(result, model, N, K, kf, prior_mean, link_name="exponential"):
    """
    Compute the Evidence Lower Bound (ELBO) for a fitted model.

    ELBO = E_q[log p(K|f)] + E_q[log p(f)] - E_q[log q(f)]
         = E_q[Poisson LL] - KL[q(f)||p(f)]

    Args:
        result: InferResult from inference
        model: Model closure object from inference
        N: Occupancy times
        K: Spike counts
        kf: Kernel Fourier transform
        prior_mean: Prior mean
        link_name: Name of link function used

    Returns:
        dict: Dictionary with ELBO components
    """
    link = get_link_function(link_name)

    # Get posterior parameters
    mu = result.info.mu  # posterior mean
    v = result.zv.v  # posterior variance
    z = result.zv.z  # deviation from prior mean

    # Mask for observed data
    mask = N > 0

    # 1. Expected Poisson log-likelihood: E[K·log(λ) - N·λ - log(K!)]
    E_rate = link.expected_rate(mu[mask], v[mask])
    E_log_rate = link.expected_log_rate(mu[mask], v[mask])

    poisson_ll = np.sum(
        K[mask] * (np.log(N[mask]) + E_log_rate) - N[mask] * E_rate - gammaln(K[mask] + 1)
    )

    # 2. KL divergence: KL[q(f)||p(f)]
    # This is computed from the inference objective
    # For LGCP, this is part of the loss function
    # KL = 0.5 * (trace(Σ_prior^-1 Σ_q) + μ^T Σ_prior^-1 μ - log|Σ_q| + log|Σ_prior| - D)

    # Get prior precision (from kernel)
    from numpy.fft import fftn, ifftn

    kx = ifftn(kf, norm="ortho").real
    prior_var = np.var(kx)

    # Approximate KL (simplified)
    kl_divergence = 0.5 * (
        np.sum(v[mask] / prior_var)
        + np.sum(z[mask] ** 2 / prior_var)
        - np.sum(np.log(v[mask]))
        + np.sum(mask) * np.log(prior_var)
        - np.sum(mask)
    )

    # ELBO = Expected LL - KL
    elbo = poisson_ll - kl_divergence

    return {
        "elbo": elbo,
        "expected_ll": poisson_ll,
        "kl_divergence": kl_divergence,
        "log_likelihood": result.ll,  # From inference
    }


def train_test_split(N, K, test_fraction=0.2, seed=42):
    """
    Split data into train and test sets.

    Args:
        N: Occupancy times (2D array)
        K: Spike counts (2D array)
        test_fraction: Fraction of bins to hold out
        seed: Random seed

    Returns:
        tuple: (N_train, K_train, N_test, K_test, test_mask)
    """
    np.random.seed(seed)

    # Only split bins that have data
    has_data = N > 0
    n_data_bins = np.sum(has_data)
    n_test = int(n_data_bins * test_fraction)

    # Randomly select test bins
    data_indices = np.where(has_data)
    permutation = np.random.permutation(n_data_bins)
    test_indices_flat = permutation[:n_test]

    # Create test mask
    test_mask = np.zeros_like(N, dtype=bool)
    test_mask[
        data_indices[0][test_indices_flat], data_indices[1][test_indices_flat]
    ] = True

    # Split data
    N_train = N.copy()
    K_train = K.copy()
    N_train[test_mask] = 0
    K_train[test_mask] = 0

    N_test = np.zeros_like(N)
    K_test = np.zeros_like(K)
    N_test[test_mask] = N[test_mask]
    K_test[test_mask] = K[test_mask]

    return N_train, K_train, N_test, K_test, test_mask


def compute_validation_ll(result, N_test, K_test, test_mask, link_name="exponential"):
    """
    Compute validation log-likelihood on held-out data.

    Args:
        result: InferResult from training
        N_test: Test occupancy times
        K_test: Test spike counts
        test_mask: Boolean mask for test bins
        link_name: Name of link function

    Returns:
        float: Validation log-likelihood
    """
    if np.sum(test_mask) == 0:
        return 0.0

    link = get_link_function(link_name)

    # Get posterior at test locations
    mu_test = result.info.mu[test_mask]
    v_test = result.zv.v[test_mask]

    # Expected rate at test locations
    E_rate = link.expected_rate(mu_test, v_test)
    E_log_rate = link.expected_log_rate(mu_test, v_test)

    # Poisson log-likelihood
    K_t = K_test[test_mask]
    N_t = N_test[test_mask]

    val_ll = np.sum(
        K_t * (np.log(N_t) + E_log_rate) - N_t * E_rate - gammaln(K_t + 1)
    )

    return val_ll


def compute_predictive_metrics(result, N_test, K_test, test_mask, link_name="exponential"):
    """
    Compute predictive performance metrics.

    Args:
        result: InferResult from training
        N_test: Test occupancy times
        K_test: Test spike counts
        test_mask: Boolean mask for test bins
        link_name: Name of link function

    Returns:
        dict: Dictionary of metrics
    """
    if np.sum(test_mask) == 0:
        return {}

    link = get_link_function(link_name)

    # Get predictions
    mu_test = result.info.mu[test_mask]
    v_test = result.zv.v[test_mask]
    predicted_rate = link.expected_rate(mu_test, v_test)

    # True observed rates
    K_t = K_test[test_mask]
    N_t = N_test[test_mask]
    observed_rate = K_t / N_t

    # Mean squared error
    mse = np.mean((predicted_rate - observed_rate) ** 2)

    # Mean absolute error
    mae = np.mean(np.abs(predicted_rate - observed_rate))

    # R-squared (coefficient of determination)
    ss_res = np.sum((observed_rate - predicted_rate) ** 2)
    ss_tot = np.sum((observed_rate - np.mean(observed_rate)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # Poisson deviance
    # D = 2 * sum(y * log(y/mu) - (y - mu))
    deviance = 2 * np.sum(
        K_t * np.log((K_t + 1e-10) / (N_t * predicted_rate + 1e-10))
        - (K_t - N_t * predicted_rate)
    )

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "deviance": deviance,
        "n_test": np.sum(test_mask),
    }


def evaluate_link_function(
    data, link_name="exponential", test_fraction=0.2, verbose=True, **inference_opts
):
    """
    Evaluate a single link function on the dataset.

    Args:
        data: Prepared Dataset object
        link_name: Name of link function to evaluate
        test_fraction: Fraction of data to hold out for validation
        verbose: Print progress
        **inference_opts: Additional options for inference

    Returns:
        dict: Dictionary of results and metrics
    """
    if verbose:
        print(f"\nEvaluating {link_name}...")

    # Split data
    N_train, K_train, N_test, K_test, test_mask = train_test_split(
        data.N, data.K, test_fraction=test_fraction
    )

    # Create kernel
    kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")

    # Run inference on training data
    # Note: Using standard lgcp2d which assumes exponential link
    # For other links, would need modified inference
    result, model = lgcp2d(
        kf,
        N_train,
        K_train,
        data.prior_mean,
        (data.kdelograte, None),
        verbose=verbose,
        **inference_opts,
    )

    # Compute metrics
    elbo_metrics = compute_elbo(result, model, N_train, K_train, kf, data.prior_mean, link_name)
    val_ll = compute_validation_ll(result, N_test, K_test, test_mask, link_name)
    pred_metrics = compute_predictive_metrics(result, N_test, K_test, test_mask, link_name)

    return {
        "link_name": link_name,
        "result": result,
        "model": model,
        "N_train": N_train,
        "K_train": K_train,
        "N_test": N_test,
        "K_test": K_test,
        "test_mask": test_mask,
        "elbo": elbo_metrics["elbo"],
        "train_ll": elbo_metrics["log_likelihood"],
        "expected_ll": elbo_metrics["expected_ll"],
        "kl_divergence": elbo_metrics["kl_divergence"],
        "validation_ll": val_ll,
        **pred_metrics,
    }


def compare_link_functions(
    data, link_names=None, test_fraction=0.2, verbose=True, **inference_opts
):
    """
    Compare multiple link functions on the same dataset.

    Args:
        data: Prepared Dataset object
        link_names: List of link function names to compare
        test_fraction: Fraction of data to hold out
        verbose: Print progress
        **inference_opts: Additional options for inference

    Returns:
        dict: Dictionary mapping link names to evaluation results
    """
    if link_names is None:
        link_names = ["exponential"]  # Only exponential works with current inference

    results = {}
    for link_name in link_names:
        try:
            results[link_name] = evaluate_link_function(
                data, link_name, test_fraction, verbose, **inference_opts
            )
        except Exception as e:
            print(f"Error evaluating {link_name}: {e}")
            results[link_name] = None

    return results


def print_comparison_table(results):
    """Print a comparison table of metrics."""
    print("\n" + "=" * 80)
    print("Link Function Comparison")
    print("=" * 80)

    # Header
    print(
        f"{'Link':<15} {'ELBO':>12} {'Train LL':>12} {'Val LL':>12} "
        f"{'MSE':>10} {'R²':>8}"
    )
    print("-" * 80)

    # Rows
    for link_name, result in results.items():
        if result is None:
            print(f"{link_name:<15} {'ERROR':>12}")
            continue

        print(
            f"{link_name:<15} "
            f"{result['elbo']:>12.2f} "
            f"{result['train_ll']:>12.2f} "
            f"{result['validation_ll']:>12.2f} "
            f"{result.get('mse', 0):>10.4f} "
            f"{result.get('r2', 0):>8.3f}"
        )

    print("=" * 80)
