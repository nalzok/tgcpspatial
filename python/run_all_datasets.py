#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Comprehensive analysis across all Krupic 2018 datasets.

Tests exponential, ReLU, and squared link functions on each dataset,
computing ELBO, validation log-likelihood, and goodness-of-fit metrics.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from config import datadir
from evaluate_links import compute_predictive_metrics, train_test_split
from goodness_of_fit import goodness_of_fit_summary
from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft
from nonlinearity import get_link_function


def get_dataset_list(exclude_simulated=True):
    """Get list of all available datasets."""
    pattern = os.path.join(datadir, "*.mat")
    files = sorted(glob.glob(pattern))

    if exclude_simulated:
        files = [f for f in files if "simulated" not in f and "binned_data" not in f]

    return [os.path.basename(f) for f in files]


def load_dataset_safe(filename, verbose=False):
    """Safely load and prepare dataset with error handling."""
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


def evaluate_model_complete(data, link_name, test_fraction=0.2, verbose=False, **kwargs):
    """
    Complete model evaluation including train/test split and all metrics.

    Returns dict with:
    - train_ll (ELBO)
    - val_ll (validation log-likelihood)
    - elbo (same as train_ll, for clarity)
    - Other metrics: r2, mse, deviance, etc.
    """
    try:
        # Split data
        N_train, K_train, N_test, K_test, test_mask = train_test_split(
            data.N, data.K, test_fraction=test_fraction
        )

        # Fit model on training data
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

        # Get predictions on test set
        mu = result.info.mu
        v = result.zv.v

        # Compute validation log-likelihood
        mu_test = mu[test_mask]
        v_test = v[test_mask]
        N_test_vals = N_test[test_mask]
        K_test_vals = K_test[test_mask]

        E_rate_test = link.expected_rate(mu_test, v_test)
        E_log_rate_test = link.expected_log_rate(mu_test, v_test)

        # Validation LL computation
        from scipy.special import gammaln
        val_ll = np.sum(
            K_test_vals * (np.log(N_test_vals + 1e-10) + E_log_rate_test) -
            N_test_vals * E_rate_test -
            gammaln(K_test_vals + 1)
        )

        # Compute predictive metrics
        pred_metrics = compute_predictive_metrics(
            result, N_test, K_test, test_mask, link_name
        )

        # Goodness of fit
        E_rate_full = link.expected_rate(mu, v)
        mask = data.N > 0
        observed = data.K[mask]
        expected = (data.N * E_rate_full)[mask]

        # Approximate GOF using binned data
        rescaled_isi = []
        for obs, exp in zip(observed, expected):
            if exp > 0 and obs > 0:
                intervals = np.random.exponential(1.0, size=int(min(obs, 100)))
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
            "gof": gof,
            "success": True,
        }

    except Exception as e:
        if verbose:
            print(f"  Error evaluating {link_name}: {e}")
            import traceback
            traceback.print_exc()
        return {
            "link_name": link_name,
            "error": str(e),
            "success": False,
        }


def analyze_single_dataset(filename, link_names, output_dir="figures"):
    """Run complete analysis on a single dataset."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {filename}")
    print('='*80)

    # Load data
    data = load_dataset_safe(filename, verbose=True)
    if data is None:
        print(f"  Skipping {filename} (failed to load)")
        return None

    # Create dataset-specific output directory
    dataset_name = os.path.splitext(filename)[0]
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Evaluate each link function
    results = {}
    for link_name in link_names:
        print(f"\n  Testing {link_name} link...")
        result = evaluate_model_complete(
            data, link_name, test_fraction=0.2, verbose=False, eps=1e-5
        )
        results[link_name] = result

        if result.get("success", False):
            print(f"    ELBO: {result['elbo']:.2f}")
            print(f"    Val LL: {result['val_ll']:.2f}")
            print(f"    R²: {result['r2']:.3f}")
            print(f"    MSE: {result['mse']:.5f}")

    # Print summary table
    print_summary_table(results)

    # Save rate map comparison
    try:
        plot_rate_maps(data, results, dataset_dir, dataset_name)
    except Exception as e:
        print(f"  Error creating plots: {e}")

    return {"filename": filename, "results": results, "data": data}


def plot_rate_maps(data, results_dict, output_dir, dataset_name):
    """Create rate map comparison plot."""
    valid_results = {k: v for k, v in results_dict.items() if v.get("success", False)}
    if not valid_results:
        return

    n_links = len(valid_results)
    fig, axes = plt.subplots(1, n_links, figsize=(5 * n_links, 4))

    if n_links == 1:
        axes = [axes]

    for idx, (link_name, result) in enumerate(valid_results.items()):
        ax = axes[idx]
        plt.sca(ax)

        rate_map = result["result"].info.r
        data.arena.imshow(rate_map, lw=3, domask=True)
        ax.set_title(
            f"{link_name}\nELBO={result['elbo']:.0f}, Val LL={result['val_ll']:.0f}\nR²={result['r2']:.2f}",
            fontsize=10,
            fontweight="bold",
        )

    plt.suptitle(f"Rate Maps: {dataset_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{dataset_name}_rate_maps.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved rate maps to: {filename}")


def print_summary_table(results_dict):
    """Print formatted summary table."""
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


def save_results_to_csv(results_list, filename):
    """Save results to CSV file."""
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

    print(f"\nSaved results to: {filename}")


def create_cross_dataset_summary(all_results, output_dir="figures"):
    """Create summary comparing results across all datasets."""
    print(f"\n{'='*80}")
    print("Creating cross-dataset summary")
    print('='*80)

    # Collect all results into flat list
    flat_results = []
    for dataset_result in all_results:
        if dataset_result is None:
            continue

        filename = dataset_result["filename"]
        for link_name, result in dataset_result["results"].items():
            if result.get("success", False):
                flat_results.append(
                    {
                        "dataset": filename,
                        "link": link_name,
                        **result,
                    }
                )

    # Save to CSV
    csv_file = os.path.join(output_dir, "all_results.csv")
    save_results_to_csv(flat_results, csv_file)

    # Create summary plots
    create_summary_plots(flat_results, output_dir)

    # Print summary statistics
    print_cross_dataset_statistics(flat_results)


def create_summary_plots(results_list, output_dir):
    """Create cross-dataset summary plots."""
    if not results_list:
        print("  No results to plot")
        return

    links = sorted(set(r["link"] for r in results_list))
    colors = plt.cm.Set2(np.arange(len(links)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. ELBO distribution
    ax = axes[0, 0]
    for idx, link in enumerate(links):
        elbos = [r["elbo"] for r in results_list if r["link"] == link]
        ax.hist(elbos, alpha=0.6, label=link, color=colors[idx], bins=15)
    ax.set_xlabel("ELBO")
    ax.set_ylabel("Count")
    ax.set_title("ELBO Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation LL distribution
    ax = axes[0, 1]
    for idx, link in enumerate(links):
        val_lls = [r["val_ll"] for r in results_list if r["link"] == link]
        ax.hist(val_lls, alpha=0.6, label=link, color=colors[idx], bins=15)
    ax.set_xlabel("Validation Log-Likelihood")
    ax.set_ylabel("Count")
    ax.set_title("Validation LL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. R² distribution
    ax = axes[0, 2]
    for idx, link in enumerate(links):
        r2s = [r["r2"] for r in results_list if r["link"] == link and not np.isnan(r["r2"])]
        ax.hist(r2s, alpha=0.6, label=link, color=colors[idx], bins=15)
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("R² Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. ELBO comparison by link
    ax = axes[1, 0]
    for idx, link in enumerate(links):
        elbos = [r["elbo"] for r in results_list if r["link"] == link]
        ax.boxplot(
            [elbos],
            positions=[idx],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[idx], alpha=0.6),
        )
    ax.set_xticks(range(len(links)))
    ax.set_xticklabels(links, rotation=45, ha="right")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO by Link Function")
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Validation LL comparison by link
    ax = axes[1, 1]
    for idx, link in enumerate(links):
        val_lls = [r["val_ll"] for r in results_list if r["link"] == link]
        ax.boxplot(
            [val_lls],
            positions=[idx],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[idx], alpha=0.6),
        )
    ax.set_xticks(range(len(links)))
    ax.set_xticklabels(links, rotation=45, ha="right")
    ax.set_ylabel("Validation LL")
    ax.set_title("Validation LL by Link Function")
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Dataset count
    ax = axes[1, 2]
    dataset_counts = {}
    for link in links:
        dataset_counts[link] = len([r for r in results_list if r["link"] == link])
    ax.bar(range(len(links)), [dataset_counts[l] for l in links], color=colors, alpha=0.6)
    ax.set_xticks(range(len(links)))
    ax.set_xticklabels(links, rotation=45, ha="right")
    ax.set_ylabel("Number of Datasets")
    ax.set_title("Successful Fits by Link")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    filename = os.path.join(output_dir, "cross_dataset_summary.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross-dataset summary to: {filename}")


def print_cross_dataset_statistics(results_list):
    """Print summary statistics across all datasets."""
    links = sorted(set(r["link"] for r in results_list))

    print("\n" + "=" * 80)
    print("Summary Statistics Across All Datasets")
    print("=" * 80)

    for link in links:
        link_results = [r for r in results_list if r["link"] == link]
        if not link_results:
            continue

        elbos = [r["elbo"] for r in link_results]
        val_lls = [r["val_ll"] for r in link_results]
        r2s = [r["r2"] for r in link_results if not np.isnan(r["r2"])]
        mses = [r["mse"] for r in link_results if not np.isnan(r["mse"])]

        print(f"\n{link}:")
        print(f"  N datasets: {len(link_results)}")
        print(f"  ELBO: {np.mean(elbos):.1f} ± {np.std(elbos):.1f}")
        print(f"  Val LL: {np.mean(val_lls):.1f} ± {np.std(val_lls):.1f}")
        if r2s:
            print(f"  R²: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
        if mses:
            print(f"  MSE: {np.mean(mses):.5f} ± {np.std(mses):.5f}")

    print("=" * 80)


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("Comprehensive Analysis: All Krupic 2018 Datasets")
    print("=" * 80)

    # Get all datasets
    datasets = get_dataset_list(exclude_simulated=True)
    print(f"\nFound {len(datasets)} datasets")

    # Link functions to test
    link_names = ["exponential", "relu", "squared"]
    print(f"Testing link functions: {', '.join(link_names)}")

    # Create output directory
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Analyze each dataset
    all_results = []
    for idx, filename in enumerate(datasets, 1):
        print(f"\n[{idx}/{len(datasets)}] Processing {filename}")
        try:
            result = analyze_single_dataset(filename, link_names, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)

    # Create cross-dataset summary
    create_cross_dataset_summary(all_results, output_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
