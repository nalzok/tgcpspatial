#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Run comprehensive analysis on all datasets in krupic2018/.

This master script:
1. Loads all available datasets
2. Fits models with different link functions
3. Evaluates goodness-of-fit
4. Generates summary statistics and visualizations
5. Saves results for comparison
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_utils import (
    evaluate_model_complete,
    get_dataset_list,
    load_dataset_safe,
    print_summary_table,
    save_results_to_csv,
)


def analyze_single_dataset(filename, link_names, output_dir="results"):
    """
    Run complete analysis on a single dataset.

    Args:
        filename (str): Dataset filename
        link_names (list): List of link function names to test
        output_dir (str): Directory for output files

    Returns:
        dict: Results for each link function
    """
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

    # Print summary for this dataset
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


def create_cross_dataset_summary(all_results, output_dir="results"):
    """
    Create summary comparing results across all datasets.

    Args:
        all_results (list): List of results from all datasets
        output_dir (str): Output directory
    """
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


def create_summary_plots(results_list, output_dir):
    """Create cross-dataset summary plots."""
    if not results_list:
        print("  No results to plot")
        return

    # Extract metrics by link function
    links = sorted(set(r["link"] for r in results_list))
    colors = plt.cm.Set2(np.arange(len(links)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1. Train LL distribution
    ax = axes[0, 0]
    for idx, link in enumerate(links):
        lls = [r["train_ll"] for r in results_list if r["link"] == link]
        ax.hist(lls, alpha=0.6, label=link, color=colors[idx], bins=15)
    ax.set_xlabel("Train Log-Likelihood")
    ax.set_ylabel("Count")
    ax.set_title("Train LL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. R² distribution
    ax = axes[0, 1]
    for idx, link in enumerate(links):
        r2s = [r["r2"] for r in results_list if r["link"] == link and not np.isnan(r["r2"])]
        ax.hist(r2s, alpha=0.6, label=link, color=colors[idx], bins=15)
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("R² Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. MSE distribution
    ax = axes[0, 2]
    for idx, link in enumerate(links):
        mses = [r["mse"] for r in results_list if r["link"] == link and not np.isnan(r["mse"])]
        ax.hist(mses, alpha=0.6, label=link, color=colors[idx], bins=15)
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("MSE Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Comparison: Train LL by link
    ax = axes[1, 0]
    for idx, link in enumerate(links):
        lls = [r["train_ll"] for r in results_list if r["link"] == link]
        ax.boxplot(
            [lls],
            positions=[idx],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[idx], alpha=0.6),
        )
    ax.set_xticks(range(len(links)))
    ax.set_xticklabels(links, rotation=45, ha="right")
    ax.set_ylabel("Train Log-Likelihood")
    ax.set_title("Train LL by Link Function")
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Comparison: R² by link
    ax = axes[1, 1]
    for idx, link in enumerate(links):
        r2s = [r["r2"] for r in results_list if r["link"] == link and not np.isnan(r["r2"])]
        if r2s:
            ax.boxplot(
                [r2s],
                positions=[idx],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=colors[idx], alpha=0.6),
            )
    ax.set_xticks(range(len(links)))
    ax.set_xticklabels(links, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("R² by Link Function")
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

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics Across All Datasets")
    print("=" * 80)
    for link in links:
        link_results = [r for r in results_list if r["link"] == link]
        if not link_results:
            continue

        lls = [r["train_ll"] for r in link_results]
        r2s = [r["r2"] for r in link_results if not np.isnan(r["r2"])]
        mses = [r["mse"] for r in link_results if not np.isnan(r["mse"])]

        print(f"\n{link}:")
        print(f"  N datasets: {len(link_results)}")
        print(f"  Train LL: {np.mean(lls):.1f} ± {np.std(lls):.1f}")
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
    output_dir = "../figures"
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
