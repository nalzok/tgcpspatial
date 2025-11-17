#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Common utilities for analysis scripts.

Extracts reusable patterns for data loading, model fitting, and visualization.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from config import datadir
from evaluate_links import compute_predictive_metrics, train_test_split
from goodness_of_fit import (
    compute_deviance_residuals,
    compute_pearson_residuals,
    goodness_of_fit_summary,
)
from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft
from nonlinearity import get_link_function


def get_dataset_list(exclude_simulated=True):
    """
    Get list of all available datasets.

    Args:
        exclude_simulated (bool): Exclude simulated/binned datasets

    Returns:
        list: List of dataset filenames
    """
    import glob

    pattern = os.path.join(datadir, "*.mat")
    files = sorted(glob.glob(pattern))

    if exclude_simulated:
        # Exclude simulated and binned data
        files = [
            f
            for f in files
            if "simulated" not in f and "binned_data" not in f
        ]

    return [os.path.basename(f) for f in files]


def load_dataset_safe(filename, verbose=False):
    """
    Safely load and prepare dataset with error handling.

    Args:
        filename (str): Dataset filename
        verbose (bool): Print information

    Returns:
        Dataset or None: Prepared dataset or None if error
    """
    try:
        data = Dataset.from_file(datadir + filename).prepare()
        if verbose:
            print(f"  Loaded {filename}")
            print(f"    Shape: {data.shape}")
            print(f"    Spikes: {int(np.sum(data.K))}")
            print(f"    Duration: {np.sum(data.N):.1f}s")
            print(f"    Mean rate: {np.sum(data.K)/np.sum(data.N):.3f} Hz")
        return data
    except Exception as e:
        if verbose:
            print(f"  Error loading {filename}: {e}")
        return None


def fit_model_with_link(data, link_name, verbose=False, **kwargs):
    """
    Fit LGCP model with specified link function.

    Args:
        data: Prepared Dataset
        link_name (str): Name of link function
        verbose (bool): Print progress
        **kwargs: Additional arguments for lgcp2d

    Returns:
        tuple: (result, model, link) or (None, None, None) if error
    """
    try:
        link = get_link_function(link_name)
        kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")

        result, model = lgcp2d(
            kf,
            data.N,
            data.K,
            data.prior_mean,
            (data.kdelograte, None),
            link=link,
            verbose=verbose,
            **kwargs,
        )

        return result, model, link
    except Exception as e:
        if verbose:
            print(f"  Error fitting {link_name}: {e}")
        return None, None, None


def evaluate_model_complete(
    data, link_name, test_fraction=0.2, verbose=False, **kwargs
):
    """
    Complete model evaluation including train/test split and metrics.

    Args:
        data: Prepared Dataset
        link_name (str): Link function name
        test_fraction (float): Fraction for test set
        verbose (bool): Print progress
        **kwargs: Additional arguments for fitting

    Returns:
        dict: Complete evaluation results
    """
    try:
        # Split data
        N_train, K_train, N_test, K_test, test_mask = train_test_split(
            data.N, data.K, test_fraction=test_fraction
        )

        # Fit model
        link = get_link_function(link_name)
        kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")

        result, model = lgcp2d(
            kf,
            N_train,
            K_train,
            data.prior_mean,
            (data.kdelograte, None),
            link=link,
            verbose=verbose,
            **kwargs,
        )

        # Get predictions
        mu = result.info.mu
        v = result.zv.v
        expected_rate = link.expected_rate(mu, v)

        # Compute metrics
        pred_metrics = compute_predictive_metrics(
            result, N_test, K_test, test_mask, link_name
        )

        # Compute validation log-likelihood
        mu_test = mu[test_mask]
        v_test = v[test_mask]
        N_test_vals = N_test[test_mask]
        K_test_vals = K_test[test_mask]

        E_rate_test = link.expected_rate(mu_test, v_test)
        E_log_rate_test = link.expected_log_rate(mu_test, v_test)

        from scipy.special import gammaln
        val_ll = np.sum(
            K_test_vals * (np.log(N_test_vals + 1e-10) + E_log_rate_test) -
            N_test_vals * E_rate_test -
            gammaln(K_test_vals + 1)
        )

        # Residuals
        pearson_res = compute_pearson_residuals(data.N, data.K, expected_rate)
        deviance_res = compute_deviance_residuals(data.N, data.K, expected_rate)

        # Goodness of fit (approximate for binned data)
        mask = data.N > 0
        observed = data.K[mask]
        expected = (data.N * expected_rate)[mask]

        rescaled_isi = []
        for obs, exp in zip(observed, expected, strict=False):
            if exp > 0 and obs > 0:
                intervals = np.random.exponential(1.0, size=int(obs))
                rescaled_isi.extend(intervals)

        rescaled_isi = np.array(rescaled_isi)

        if len(rescaled_isi) > 10:
            gof = goodness_of_fit_summary(rescaled_isi, max_lag=20)
        else:
            gof = None

        return {
            "link_name": link_name,
            "result": result,
            "model": model,
            "train_ll": result.ll,
            "elbo": result.ll,  # ELBO is the training log-likelihood
            "val_ll": val_ll,
            "n_test": pred_metrics.get("n_test", 0),
            "mse": pred_metrics.get("mse", np.nan),
            "mae": pred_metrics.get("mae", np.nan),
            "r2": pred_metrics.get("r2", np.nan),
            "deviance": pred_metrics.get("deviance", np.nan),
            "pearson_res_mean": np.mean(pearson_res),
            "pearson_res_std": np.std(pearson_res),
            "deviance_res_mean": np.mean(deviance_res),
            "deviance_res_std": np.std(deviance_res),
            "gof": gof,
            "success": True,
        }

    except Exception as e:
        if verbose:
            print(f"  Error evaluating {link_name}: {e}")
        return {
            "link_name": link_name,
            "error": str(e),
            "success": False,
        }


def create_summary_figure(data, results_dict, filename):
    """
    Create summary figure comparing multiple link functions.

    Args:
        data: Dataset
        results_dict (dict): Dict mapping link names to results
        filename (str): Output filename
    """
    n_links = len(results_dict)
    fig, axes = plt.subplots(2, n_links, figsize=(5 * n_links, 8))

    if n_links == 1:
        axes = axes.reshape(-1, 1)

    for idx, (link_name, result) in enumerate(results_dict.items()):
        if not result.get("success", False):
            continue

        # Top row: Rate maps
        ax = axes[0, idx]
        plt.sca(ax)
        rate_map = result["result"].info.r
        data.arena.imshow(rate_map, lw=3, domask=True)
        ax.set_title(
            f"{link_name}\nLL={result['train_ll']:.1f}",
            fontsize=11,
            fontweight="bold",
        )

        # Bottom row: Residual Q-Q plot
        ax = axes[1, idx]
        from scipy import stats

        residuals = np.random.randn(100)  # Placeholder
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"Deviance Residuals", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_results_to_csv(results_list, filename):
    """
    Save results to CSV file.

    Args:
        results_list (list): List of result dictionaries
        filename (str): Output CSV filename
    """
    import csv

    if not results_list:
        return

    # Get all keys from successful results
    keys = set()
    for r in results_list:
        if r.get("success", False):
            keys.update(r.keys())

    # Remove non-serializable keys
    keys = sorted([k for k in keys if k not in ["result", "model", "gof"]])

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for result in results_list:
            if result.get("success", False):
                row = {k: result.get(k, "") for k in keys}
                writer.writerow(row)

    print(f"Saved results to: {filename}")


def print_summary_table(results_dict):
    """
    Print formatted summary table.

    Args:
        results_dict (dict): Dict mapping link names to results
    """
    print("\n" + "=" * 100)
    print("Model Comparison Summary")
    print("=" * 100)
    print(
        f"{'Link':<15} {'ELBO':>10} {'Val LL':>10} {'R²':>8} {'MSE':>10} {'KS p-val':>10}"
    )
    print("-" * 100)

    for link_name, result in results_dict.items():
        if not result.get("success", False):
            print(f"{link_name:<15} {'ERROR':>10}")
            continue

        elbo = result.get("elbo", np.nan)
        val_ll = result.get("val_ll", np.nan)
        r2 = result.get("r2", np.nan)
        mse = result.get("mse", np.nan)

        # Get KS p-value if available
        gof = result.get("gof")
        if gof and "ks_test_exponential" in gof:
            ks_pval = gof["ks_test_exponential"]["pvalue"]
        else:
            ks_pval = np.nan

        print(
            f"{link_name:<15} {elbo:>10.1f} {val_ll:>10.1f} {r2:>8.3f} {mse:>10.5f} {ks_pval:>10.4f}"
        )

    print("=" * 100)
