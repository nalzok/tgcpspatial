#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Sample synthetic spike trains from fitted Cox process models.

This script demonstrates forward simulation from LGCP models:
1. Sample from the posterior GP
2. Transform through different link functions
3. Generate Poisson spike trains
4. Compare synthetic vs real data
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import datadir
from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft
from nonlinearity import get_link_function


def sample_from_posterior_gp(result, model, n_samples=1, seed=None):
    """
    Sample latent GP values from the posterior distribution.

    Args:
        result: InferResult from lgcp2d
        model: Model closure from lgcp2d
        n_samples (int): Number of samples to draw
        seed (int): Random seed

    Returns:
        ndarray: Samples of shape (H, W, n_samples)
    """
    if seed is not None:
        np.random.seed(seed)

    # Use the sample function from the model
    samples = model.sample(n_samples)

    return samples


def generate_spike_train_from_rate(rate_map, occupancy, link=None, seed=None):
    """
    Generate synthetic spike train from a rate map using Poisson process.

    Args:
        rate_map (ndarray): 2D array of firing rates (Hz)
        occupancy (ndarray): 2D array of time spent at each location (s)
        link: LinkFunction instance (for reference, not used in sampling)
        seed (int): Random seed

    Returns:
        ndarray: Synthetic spike counts per bin
    """
    if seed is not None:
        np.random.seed(seed)

    # Expected counts at each location
    expected_counts = rate_map * occupancy

    # Sample from Poisson distribution
    spike_counts = np.random.poisson(expected_counts)

    return spike_counts


def compare_real_vs_synthetic(
    data, link_name="exponential", n_synthetic=5, seed=42
):
    """
    Compare real data against synthetic samples from fitted model.

    Args:
        data: Prepared Dataset
        link_name (str): Name of link function
        n_synthetic (int): Number of synthetic samples to generate
        seed (int): Random seed

    Returns:
        dict: Results including real and synthetic data
    """
    print(f"\n{'='*70}")
    print(f"Sampling from {link_name} link function")
    print('='*70)

    # Get link function
    link = get_link_function(link_name)

    # Create kernel and run inference
    kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")

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
    print(f"Log-likelihood: {result.ll:.2f}")

    # Get fitted rate map (posterior mean)
    fitted_rate = result.info.r

    # Sample from posterior GP
    print(f"Generating {n_synthetic} synthetic samples...")
    gp_samples = sample_from_posterior_gp(result, model, n_synthetic, seed=seed)

    # Transform GP samples through link function to get rate maps
    synthetic_rates = []
    synthetic_spikes = []

    for i in range(n_synthetic):
        # Get latent GP sample
        latent_sample = gp_samples[:, :, i]

        # Transform through link function
        # For the link function, we need to handle it carefully
        # The samples from model.sample() are already in latent space
        # We need to apply the link to get rates
        rate_sample = link.apply(latent_sample)
        synthetic_rates.append(rate_sample)

        # Generate spike counts
        spike_sample = generate_spike_train_from_rate(
            rate_sample, data.N, link, seed=seed + i
        )
        synthetic_spikes.append(spike_sample)

    return {
        "link_name": link_name,
        "result": result,
        "fitted_rate": fitted_rate,
        "real_spikes": data.K,
        "occupancy": data.N,
        "arena": data.arena,
        "synthetic_rates": synthetic_rates,
        "synthetic_spikes": synthetic_spikes,
        "gp_samples": gp_samples,
    }


def plot_real_vs_synthetic_comparison(
    results, filename="figures/real_vs_synthetic.png"
):
    """
    Plot comparison of real data vs synthetic samples.

    Args:
        results (dict): Results from compare_real_vs_synthetic
        filename (str): Output filename
    """
    link_name = results["link_name"]
    arena = results["arena"]
    real_spikes = results["real_spikes"]
    occupancy = results["occupancy"]
    fitted_rate = results["fitted_rate"]
    synthetic_spikes = results["synthetic_spikes"]
    synthetic_rates = results["synthetic_rates"]

    n_synthetic = len(synthetic_spikes)

    # Create figure: 3 rows x (2 + n_synthetic) columns
    # Row 1: Real spike map, fitted rate map, synthetic rates
    # Row 2: Real rate (empirical), fitted rate, synthetic empirical rates
    # Row 3: Residuals
    fig = plt.figure(figsize=(4 * (2 + n_synthetic), 10))

    # Row 1: Spike counts
    # Real spike count
    ax = plt.subplot(3, 2 + n_synthetic, 1)
    plt.sca(ax)
    arena.imshow(real_spikes, lw=3, domask=True)
    ax.set_title("Real Spike Counts", fontsize=11, fontweight="bold")

    # Fitted rate map
    ax = plt.subplot(3, 2 + n_synthetic, 2)
    plt.sca(ax)
    arena.imshow(fitted_rate, lw=3, domask=True)
    ax.set_title("Fitted Rate Map", fontsize=11, fontweight="bold")

    # Synthetic spike counts
    for i in range(n_synthetic):
        ax = plt.subplot(3, 2 + n_synthetic, 3 + i)
        plt.sca(ax)
        arena.imshow(synthetic_spikes[i], lw=3, domask=True)
        ax.set_title(f"Synthetic Spikes {i+1}", fontsize=11, fontweight="bold")

    # Row 2: Empirical rates
    # Real empirical rate
    ax = plt.subplot(3, 2 + n_synthetic, 2 + n_synthetic + 1)
    plt.sca(ax)
    real_rate = np.where(occupancy > 0, real_spikes / occupancy, np.nan)
    arena.imshow(real_rate, lw=3, domask=True)
    ax.set_title("Real Empirical Rate", fontsize=11, fontweight="bold")

    # Fitted rate (repeated for comparison)
    ax = plt.subplot(3, 2 + n_synthetic, 2 + n_synthetic + 2)
    plt.sca(ax)
    arena.imshow(fitted_rate, lw=3, domask=True)
    ax.set_title("Model Rate", fontsize=11, fontweight="bold")

    # Synthetic empirical rates
    for i in range(n_synthetic):
        ax = plt.subplot(3, 2 + n_synthetic, 2 + n_synthetic + 3 + i)
        plt.sca(ax)
        synth_rate = np.where(
            occupancy > 0, synthetic_spikes[i] / occupancy, np.nan
        )
        arena.imshow(synth_rate, lw=3, domask=True)
        ax.set_title(f"Synthetic Rate {i+1}", fontsize=11, fontweight="bold")

    # Row 3: Residuals (real - fitted)
    ax = plt.subplot(3, 2 + n_synthetic, 2 * (2 + n_synthetic) + 1)
    plt.sca(ax)
    residual = real_spikes - fitted_rate * occupancy
    vmax = np.nanmax(np.abs(residual))
    im = ax.imshow(
        residual,
        extent=arena.extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.plot(*arena.perim_m.T, color="black", lw=3)
    ax.set_title("Real Residual", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax)

    # Model posterior variance
    ax = plt.subplot(3, 2 + n_synthetic, 2 * (2 + n_synthetic) + 2)
    plt.sca(ax)
    variance = results["result"].zv.v
    arena.imshow(variance, lw=3, domask=True)
    ax.set_title("Posterior Variance", fontsize=11, fontweight="bold")

    # Synthetic residuals
    for i in range(n_synthetic):
        ax = plt.subplot(3, 2 + n_synthetic, 2 * (2 + n_synthetic) + 3 + i)
        plt.sca(ax)
        synth_residual = synthetic_spikes[i] - fitted_rate * occupancy
        vmax = np.nanmax(np.abs(synth_residual))
        im = ax.imshow(
            synth_residual,
            extent=arena.extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.plot(*arena.perim_m.T, color="black", lw=3)
        ax.set_title(f"Synthetic Residual {i+1}", fontsize=11, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

    plt.suptitle(
        f"Real vs Synthetic Data: {link_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to: {filename}")
    plt.close()


def plot_summary_statistics(
    results_dict, filename="figures/summary_statistics.png"
):
    """
    Plot summary statistics comparing real vs synthetic data.

    Args:
        results_dict (dict): Dict mapping link names to results
        filename (str): Output filename
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    colors = plt.cm.Set2(np.arange(len(results_dict)))

    for idx, (link_name, results) in enumerate(results_dict.items()):
        real_spikes = results["real_spikes"]
        occupancy = results["occupancy"]
        mask = occupancy > 0

        real_counts = real_spikes[mask]
        real_rates = real_spikes[mask] / occupancy[mask]

        # Collect synthetic statistics
        synthetic_counts_list = []
        synthetic_rates_list = []

        for synth_spikes in results["synthetic_spikes"]:
            synthetic_counts_list.append(synth_spikes[mask])
            synthetic_rates_list.append(synth_spikes[mask] / occupancy[mask])

        color = colors[idx]

        # 1. Spike count distribution
        ax = axes[0, 0]
        ax.hist(
            real_counts,
            bins=30,
            alpha=0.5,
            label=f"{link_name} (real)",
            color=color,
            density=True,
        )
        # Average of synthetic
        all_synth_counts = np.concatenate(synthetic_counts_list)
        ax.hist(
            all_synth_counts,
            bins=30,
            alpha=0.3,
            linestyle="--",
            color=color,
            density=True,
            histtype="step",
            linewidth=2,
        )
        ax.set_xlabel("Spike Count", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Spike Count Distribution", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        # 2. Rate distribution
        ax = axes[0, 1]
        ax.hist(
            real_rates,
            bins=30,
            alpha=0.5,
            label=f"{link_name} (real)",
            color=color,
            density=True,
        )
        all_synth_rates = np.concatenate(synthetic_rates_list)
        ax.hist(
            all_synth_rates,
            bins=30,
            alpha=0.3,
            linestyle="--",
            color=color,
            density=True,
            histtype="step",
            linewidth=2,
        )
        ax.set_xlabel("Firing Rate (Hz)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Rate Distribution", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        # 3. Mean-variance relationship
        ax = axes[0, 2]
        # Real data
        ax.scatter(
            real_counts.mean(),
            real_counts.var(),
            s=100,
            marker="o",
            color=color,
            label=f"{link_name} (real)",
            edgecolors="black",
            linewidth=2,
        )
        # Synthetic data
        synth_means = [s.mean() for s in synthetic_counts_list]
        synth_vars = [s.var() for s in synthetic_counts_list]
        ax.scatter(
            synth_means,
            synth_vars,
            s=50,
            marker="x",
            color=color,
            alpha=0.7,
        )
        ax.set_xlabel("Mean Count", fontsize=10)
        ax.set_ylabel("Variance", fontsize=10)
        ax.set_title("Mean-Variance Relationship", fontsize=11, fontweight="bold")
        # Poisson line
        max_mean = max(real_counts.mean(), max(synth_means))
        x_line = np.linspace(0, max_mean * 1.2, 100)
        ax.plot(x_line, x_line, "k--", alpha=0.3, label="Poisson (var=mean)")
        ax.legend(fontsize=8)

        # 4. Spatial autocorrelation (simplified)
        ax = axes[1, 0]
        from scipy.ndimage import correlate

        # Compute autocorrelation for real data
        real_centered = real_spikes - np.mean(real_spikes[mask])
        real_autocorr = correlate(real_centered, real_centered, mode="constant")
        # Take central line
        center = real_autocorr.shape[0] // 2
        ax.plot(
            real_autocorr[center, :],
            label=f"{link_name} (real)",
            color=color,
            linewidth=2,
        )
        # Average of synthetic
        for i, synth_spikes in enumerate(results["synthetic_spikes"][:1]):
            synth_centered = synth_spikes - np.mean(synth_spikes[mask])
            synth_autocorr = correlate(
                synth_centered, synth_centered, mode="constant"
            )
            ax.plot(
                synth_autocorr[center, :],
                "--",
                alpha=0.5,
                color=color,
                linewidth=1,
            )
        ax.set_xlabel("Lag (bins)", fontsize=10)
        ax.set_ylabel("Autocorrelation", fontsize=10)
        ax.set_title("Spatial Autocorrelation", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        # 5. Q-Q plot of counts
        ax = axes[1, 1]
        from scipy import stats

        # Compare real vs synthetic quantiles
        avg_synth_counts = np.mean(
            [s for s in synthetic_counts_list], axis=0
        )
        ax.scatter(
            np.sort(real_counts),
            np.sort(avg_synth_counts),
            alpha=0.5,
            s=20,
            color=color,
            label=link_name,
        )
        max_val = max(np.max(real_counts), np.max(avg_synth_counts))
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
        ax.set_xlabel("Real Data Quantiles", fontsize=10)
        ax.set_ylabel("Synthetic Data Quantiles", fontsize=10)
        ax.set_title("Q-Q Plot (Counts)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        # 6. Total spike count comparison
        ax = axes[1, 2]
        real_total = np.sum(real_spikes)
        synth_totals = [np.sum(s) for s in results["synthetic_spikes"]]
        ax.bar(
            idx * 0.8,
            real_total,
            width=0.3,
            label=f"{link_name} (real)",
            color=color,
            alpha=0.7,
        )
        ax.scatter(
            [idx * 0.8] * len(synth_totals),
            synth_totals,
            color=color,
            s=50,
            marker="x",
            label=f"{link_name} (synthetic)",
        )
        ax.set_ylabel("Total Spike Count", fontsize=10)
        ax.set_title("Total Spike Count", fontsize=11, fontweight="bold")
        ax.set_xticks([i * 0.8 for i in range(len(results_dict))])
        ax.set_xticklabels(list(results_dict.keys()), rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved summary statistics to: {filename}")
    plt.close()


def main():
    """Run synthetic data generation and comparison."""
    print("=" * 80)
    print("Cox Process Sampling: Real vs Synthetic Data")
    print("=" * 80)

    # Load real data
    print("\nLoading grid cell data...")
    fn = "r2405_051216b_cell1816.mat"
    data = Dataset.from_file(datadir + fn).prepare()
    print(f"Grid shape: {data.shape}")
    print(f"Total spikes: {int(data.K.sum())}")

    # Test different link functions
    link_names = ["exponential", "relu", "squared"]
    n_synthetic = 3

    results_dict = {}

    for link_name in link_names:
        try:
            results = compare_real_vs_synthetic(
                data, link_name, n_synthetic=n_synthetic, seed=42
            )
            results_dict[link_name] = results

            # Plot individual comparison
            plot_real_vs_synthetic_comparison(
                results, filename=f"figures/sampling_{link_name}.png"
            )
        except Exception as e:
            print(f"Error with {link_name}: {e}")
            import traceback

            traceback.print_exc()

    # Plot summary statistics across all link functions
    if results_dict:
        plot_summary_statistics(results_dict)

    print("\n" + "=" * 80)
    print("Synthetic data generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
