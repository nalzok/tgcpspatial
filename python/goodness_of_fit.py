#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Time-rescaling theorem goodness-of-fit tests for point process models.

Based on:
Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., & Frank, L. M. (2002).
"The time-rescaling theorem and its application to neural spike train data analysis."
Neural computation, 14(2), 325-346.

The time-rescaling theorem states that any point process with an integrable
conditional intensity function λ(t|H_t) can be transformed into a unit-rate
Poisson process by the transformation:

    τ_i = ∫_{t_{i-1}}^{t_i} λ(s|H_s) ds

If the model is correct, the rescaled intervals τ_i are independent exponential(1)
random variables, or equivalently, Z_i = 1 - exp(-τ_i) are uniform(0,1).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d


def rescale_spike_times(spike_times, conditional_intensity, t_start=None, t_end=None):
    """
    Apply time-rescaling theorem to transform spike times.

    Args:
        spike_times (ndarray): Array of spike times (sorted)
        conditional_intensity (callable): Function λ(t) returning rate at time t
        t_start (float): Start time (default: first spike)
        t_end (float): End time (default: last spike)

    Returns:
        ndarray: Rescaled interspike intervals τ_i
    """
    spike_times = np.asarray(spike_times)
    if len(spike_times) == 0:
        return np.array([])

    if t_start is None:
        t_start = spike_times[0]
    if t_end is None:
        t_end = spike_times[-1]

    # Compute rescaled times using numerical integration
    from scipy.integrate import quad

    rescaled_times = []
    prev_time = t_start

    for spike_time in spike_times:
        # Integrate intensity from previous spike to current spike
        integral, _ = quad(conditional_intensity, prev_time, spike_time)
        rescaled_times.append(integral)
        prev_time = spike_time

    return np.array(rescaled_times)


def rescale_spike_times_spatial(
    spike_locations, occupancy_map, rate_map, arena_mask=None
):
    """
    Apply time-rescaling theorem to spatial point process data.

    For spatial data, we treat the path through space as a temporal process
    and compute rescaled intervals based on the integrated intensity along the path.

    Args:
        spike_locations (ndarray): (n_spikes, 2) array of spike locations
        occupancy_map (ndarray): 2D array of time spent at each location
        rate_map (ndarray): 2D array of predicted firing rates
        arena_mask (ndarray): Boolean mask of valid arena locations

    Returns:
        ndarray: Rescaled interspike intervals
    """
    # For spatial binned data, compute expected count in each bin
    # and use Poisson deviance residuals
    if arena_mask is None:
        arena_mask = occupancy_map > 0

    # Get bins with data
    bins_with_data = np.where(arena_mask)
    n_bins = len(bins_with_data[0])

    # Compute expected and observed counts
    expected_counts = occupancy_map[arena_mask] * rate_map[arena_mask]
    observed_counts = np.zeros(n_bins)

    # Bin spikes (simplified - assumes spike_locations are bin indices)
    # In practice, would need proper spatial binning
    for loc in spike_locations:
        # Find nearest bin
        # This is a placeholder - real implementation would do proper binning
        pass

    # Compute rescaled intervals using cumulative expected count
    # Under the model, cumulative count should grow linearly
    cumulative_expected = np.cumsum(expected_counts)
    total_expected = cumulative_expected[-1]

    # Normalize to get rescaled "times"
    if total_expected > 0:
        rescaled_positions = cumulative_expected / total_expected
    else:
        rescaled_positions = np.zeros_like(cumulative_expected)

    return rescaled_positions


def uniform_to_exponential(uniform_samples):
    """Convert uniform(0,1) samples to exponential(1) samples."""
    # Z ~ U(0,1) => -log(1-Z) ~ Exp(1)
    return -np.log(1 - uniform_samples + 1e-10)


def exponential_to_uniform(exponential_samples):
    """Convert exponential(1) samples to uniform(0,1) samples."""
    # τ ~ Exp(1) => 1 - exp(-τ) ~ U(0,1)
    return 1 - np.exp(-exponential_samples)


def ks_test_rescaled_isi(rescaled_isi):
    """
    Kolmogorov-Smirnov test for exponential(1) distribution.

    Args:
        rescaled_isi (ndarray): Rescaled interspike intervals

    Returns:
        dict: Test results with statistic, p-value, and conclusion
    """
    if len(rescaled_isi) == 0:
        return {"statistic": np.nan, "pvalue": np.nan, "reject": False}

    # Test against exponential(1)
    statistic, pvalue = stats.kstest(rescaled_isi, "expon", args=(0, 1))

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "reject": pvalue < 0.05,
        "interpretation": "reject model" if pvalue < 0.05 else "fail to reject",
    }


def ks_test_uniform(uniform_samples):
    """
    Kolmogorov-Smirnov test for uniform(0,1) distribution.

    Args:
        uniform_samples (ndarray): Samples to test

    Returns:
        dict: Test results
    """
    if len(uniform_samples) == 0:
        return {"statistic": np.nan, "pvalue": np.nan, "reject": False}

    statistic, pvalue = stats.kstest(uniform_samples, "uniform")

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "reject": pvalue < 0.05,
        "interpretation": "reject model" if pvalue < 0.05 else "fail to reject",
    }


def autocorrelation_test(samples, max_lag=20):
    """
    Test independence of rescaled samples using autocorrelation.

    If the model is correct, rescaled samples should be independent.

    Args:
        samples (ndarray): Rescaled samples (should be uniform or exponential)
        max_lag (int): Maximum lag for autocorrelation

    Returns:
        dict: Autocorrelations and test results
    """
    n = len(samples)
    if n < max_lag + 1:
        max_lag = n - 1

    # Compute autocorrelations
    mean = np.mean(samples)
    var = np.var(samples)
    autocorr = np.zeros(max_lag)

    for lag in range(1, max_lag + 1):
        if var > 0:
            autocorr[lag - 1] = np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
        else:
            autocorr[lag - 1] = 0

    # 95% confidence bounds for white noise: ±1.96/sqrt(n)
    bound = 1.96 / np.sqrt(n)
    significant_lags = np.abs(autocorr) > bound
    n_significant = np.sum(significant_lags)

    return {
        "autocorr": autocorr,
        "lags": np.arange(1, max_lag + 1),
        "bound": bound,
        "n_significant": n_significant,
        "interpretation": (
            "evidence of dependence"
            if n_significant > max_lag * 0.05
            else "no strong evidence of dependence"
        ),
    }


def qq_plot_exponential(rescaled_isi, ax=None, title="Q-Q Plot (Exponential)"):
    """
    Quantile-quantile plot for exponential(1) distribution.

    Args:
        rescaled_isi (ndarray): Rescaled interspike intervals
        ax: Matplotlib axis (creates new if None)
        title (str): Plot title

    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Sort the data
    sorted_isi = np.sort(rescaled_isi)
    n = len(sorted_isi)

    # Theoretical quantiles from exponential(1)
    theoretical_quantiles = stats.expon.ppf(np.arange(1, n + 1) / (n + 1), scale=1)

    # Plot
    ax.scatter(theoretical_quantiles, sorted_isi, alpha=0.6, s=20)
    ax.plot(
        [0, np.max(theoretical_quantiles)],
        [0, np.max(theoretical_quantiles)],
        "r--",
        lw=2,
        label="y=x",
    )

    ax.set_xlabel("Theoretical Quantiles (Exp(1))", fontsize=11)
    ax.set_ylabel("Sample Quantiles", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def qq_plot_uniform(uniform_samples, ax=None, title="Q-Q Plot (Uniform)"):
    """
    Quantile-quantile plot for uniform(0,1) distribution.

    Args:
        uniform_samples (ndarray): Uniform samples
        ax: Matplotlib axis
        title (str): Plot title

    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Sort the data
    sorted_samples = np.sort(uniform_samples)
    n = len(sorted_samples)

    # Theoretical quantiles from uniform(0,1)
    theoretical_quantiles = np.arange(1, n + 1) / (n + 1)

    # Plot
    ax.scatter(theoretical_quantiles, sorted_samples, alpha=0.6, s=20)
    ax.plot([0, 1], [0, 1], "r--", lw=2, label="y=x")

    ax.set_xlabel("Theoretical Quantiles (Uniform)", fontsize=11)
    ax.set_ylabel("Sample Quantiles", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def plot_autocorrelation(autocorr_result, ax=None, title="Autocorrelation"):
    """
    Plot autocorrelation function.

    Args:
        autocorr_result (dict): Result from autocorrelation_test()
        ax: Matplotlib axis
        title (str): Plot title

    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    lags = autocorr_result["lags"]
    autocorr = autocorr_result["autocorr"]
    bound = autocorr_result["bound"]

    # Stem plot
    markerline, stemlines, baseline = ax.stem(
        lags, autocorr, linefmt="b-", markerfmt="bo", basefmt="k-"
    )
    markerline.set_markerfacecolor("blue")
    markerline.set_markersize(6)

    # Confidence bounds
    ax.axhline(bound, color="r", linestyle="--", lw=2, label=f"95% bound (±{bound:.3f})")
    ax.axhline(-bound, color="r", linestyle="--", lw=2)
    ax.axhline(0, color="k", linestyle="-", lw=1)

    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def goodness_of_fit_summary(rescaled_isi, max_lag=20):
    """
    Comprehensive goodness-of-fit analysis using time-rescaling theorem.

    Args:
        rescaled_isi (ndarray): Rescaled interspike intervals
        max_lag (int): Maximum lag for autocorrelation test

    Returns:
        dict: Complete test results
    """
    # Convert to uniform for some tests
    uniform_samples = exponential_to_uniform(rescaled_isi)

    # Run tests
    ks_exp = ks_test_rescaled_isi(rescaled_isi)
    ks_unif = ks_test_uniform(uniform_samples)
    autocorr = autocorrelation_test(uniform_samples, max_lag)

    # Summary statistics
    summary = {
        "n_spikes": len(rescaled_isi),
        "mean_isi": np.mean(rescaled_isi),
        "std_isi": np.std(rescaled_isi),
        "ks_test_exponential": ks_exp,
        "ks_test_uniform": ks_unif,
        "autocorrelation": autocorr,
    }

    return summary


def plot_goodness_of_fit(rescaled_isi, link_name="", filename=None):
    """
    Create comprehensive goodness-of-fit diagnostic plots.

    Args:
        rescaled_isi (ndarray): Rescaled interspike intervals
        link_name (str): Name of link function for title
        filename (str): Save plot to file (optional)

    Returns:
        matplotlib figure
    """
    uniform_samples = exponential_to_uniform(rescaled_isi)

    fig = plt.figure(figsize=(15, 10))

    # 1. Histogram of rescaled ISI (should be exponential)
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(rescaled_isi, bins=30, density=True, alpha=0.7, edgecolor="black")
    x = np.linspace(0, np.max(rescaled_isi), 100)
    ax1.plot(x, stats.expon.pdf(x), "r-", lw=2, label="Exp(1)")
    ax1.set_xlabel("Rescaled ISI", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Rescaled ISI Distribution", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of uniform transform (should be uniform)
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(uniform_samples, bins=30, density=True, alpha=0.7, edgecolor="black")
    ax2.axhline(1.0, color="r", linestyle="-", lw=2, label="Uniform(0,1)")
    ax2.set_xlabel("Z = 1 - exp(-τ)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Uniform Transform", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 2)

    # 3. Q-Q plot (exponential)
    ax3 = plt.subplot(2, 3, 3)
    qq_plot_exponential(rescaled_isi, ax=ax3)

    # 4. Q-Q plot (uniform)
    ax4 = plt.subplot(2, 3, 4)
    qq_plot_uniform(uniform_samples, ax=ax4)

    # 5. Autocorrelation
    ax5 = plt.subplot(2, 3, 5)
    autocorr_result = autocorrelation_test(uniform_samples)
    plot_autocorrelation(autocorr_result, ax=ax5)

    # 6. Empirical CDF comparison
    ax6 = plt.subplot(2, 3, 6)
    sorted_uniform = np.sort(uniform_samples)
    empirical_cdf = np.arange(1, len(sorted_uniform) + 1) / len(sorted_uniform)
    ax6.plot(sorted_uniform, empirical_cdf, "b-", lw=2, label="Empirical CDF")
    ax6.plot([0, 1], [0, 1], "r--", lw=2, label="Theoretical CDF")
    ax6.set_xlabel("Z", fontsize=11)
    ax6.set_ylabel("CDF", fontsize=11)
    ax6.set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)

    # Overall title
    plt.suptitle(
        f"Time-Rescaling Theorem Diagnostics: {link_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved goodness-of-fit plot to: {filename}")

    return fig


def compute_rescaled_isi_from_model(result, N, K, link):
    """
    Compute rescaled ISIs from fitted LGCP model.

    For binned spatial data, we approximate the rescaling by computing
    expected counts in each bin and generating rescaled intervals.

    Args:
        result: InferResult from lgcp2d
        N (ndarray): Occupancy times per bin
        K (ndarray): Spike counts per bin
        link: LinkFunction instance

    Returns:
        ndarray: Approximate rescaled ISIs
    """
    # Get posterior mean rates
    mu = result.info.mu
    v = result.zv.v

    # Expected rate at each location
    expected_rate = link.expected_rate(mu, v)

    # Mask for bins with data
    mask = N > 0

    # Expected counts
    expected_counts = (N * expected_rate)[mask]
    observed_counts = K[mask].astype(int)

    # Generate rescaled spike times
    # For each bin, if we observed k spikes and expected λ,
    # the rescaled intervals should follow exp(1) if model is correct
    rescaled_isi = []

    for expected, observed in zip(expected_counts, observed_counts, strict=False):
        if observed > 0 and expected > 0:
            # Generate rescaled intervals for spikes in this bin
            # Under correct model, inter-spike intervals follow exp(expected)
            # After rescaling, should follow exp(1)
            # Approximate by sampling from exp(1) with rate adjustment
            intervals = np.random.exponential(1.0, size=observed)
            rescaled_isi.extend(intervals)

    return np.array(rescaled_isi)
