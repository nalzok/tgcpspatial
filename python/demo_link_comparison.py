#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Visual comparison of different link functions.
Demonstrates how different g(f) transform latent GP values to rates.
"""

import matplotlib.pyplot as plt
import numpy as np

from nonlinearity import (
    ExponentialLink,
    IdentityLink,
    ReLULink,
    SoftplusLink,
    SquaredLink,
)


def compare_link_functions():
    """Compare how different link functions map latent values to rates."""
    f_range = np.linspace(-3, 3, 200)

    links = [
        ExponentialLink(),
        SquaredLink(),
        ReLULink(),
        SoftplusLink(),
        IdentityLink(),
    ]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    plt.figure(figsize=(12, 4))

    # Plot 1: Direct transformation g(f)
    plt.subplot(1, 3, 1)
    for link, color in zip(links, colors, strict=False):
        rates = link.apply(f_range)
        plt.plot(
            f_range,
            rates,
            label=link.__class__.__name__.replace("Link", ""),
            linewidth=2,
            color=color,
        )
    plt.xlabel("Latent value f", fontsize=11)
    plt.ylabel("Rate λ = g(f)", fontsize=11)
    plt.title("Link Functions: λ = g(f)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.5, 10)

    # Plot 2: Expected rates E[g(f)] with uncertainty
    plt.subplot(1, 3, 2)
    mu_range = np.linspace(-2, 2, 50)
    v_fixed = 0.5  # fixed variance

    for link, color in zip(links, colors, strict=False):
        if link.__class__.__name__ == "IdentityLink":
            continue  # Skip identity for this plot
        expected_rates = link.expected_rate(
            mu_range, np.full_like(mu_range, v_fixed)
        )
        plt.plot(
            mu_range,
            expected_rates,
            label=link.__class__.__name__.replace("Link", ""),
            linewidth=2,
            color=color,
        )

    plt.xlabel("Mean μ", fontsize=11)
    plt.ylabel("Expected rate E[g(f)]", fontsize=11)
    plt.title(
        f"Expected Rates (v={v_fixed})", fontsize=12, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Effect of uncertainty on expected rate
    plt.subplot(1, 3, 3)
    mu_fixed = 0.0  # fixed mean at zero
    v_range = np.linspace(0, 2, 50)

    for link, color in zip(links, colors, strict=False):
        if link.__class__.__name__ == "IdentityLink":
            continue
        expected_rates = link.expected_rate(
            np.full_like(v_range, mu_fixed), v_range
        )
        plt.plot(
            v_range,
            expected_rates,
            label=link.__class__.__name__.replace("Link", ""),
            linewidth=2,
            color=color,
        )

    plt.xlabel("Variance v", fontsize=11)
    plt.ylabel("Expected rate E[g(f)]", fontsize=11)
    plt.title(f"Effect of Uncertainty (μ={mu_fixed})", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("link_function_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved comparison plot to: link_function_comparison.png")


def compare_gradients():
    """Compare gradients of expected rates."""
    mu_range = np.linspace(-2, 2, 100)
    v_fixed = 0.3

    links = [
        ExponentialLink(),
        SquaredLink(),
        ReLULink(),
        SoftplusLink(),
    ]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    plt.figure(figsize=(12, 4))

    # Plot 1: Gradient w.r.t. mean
    plt.subplot(1, 2, 1)
    for link, color in zip(links, colors, strict=False):
        grads = link.expected_rate_gradient_mu(
            mu_range, np.full_like(mu_range, v_fixed)
        )
        plt.plot(
            mu_range,
            grads,
            label=link.__class__.__name__.replace("Link", ""),
            linewidth=2,
            color=color,
        )

    plt.xlabel("Mean μ", fontsize=11)
    plt.ylabel("∂/∂μ E[g(f)]", fontsize=11)
    plt.title(
        "Gradient w.r.t. Mean", fontsize=12, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Gradient w.r.t. variance
    plt.subplot(1, 2, 2)
    v_range = np.linspace(0.01, 2, 100)
    mu_fixed = 0.5

    for link, color in zip(links, colors, strict=False):
        grads = link.expected_rate_gradient_v(
            np.full_like(v_range, mu_fixed), v_range
        )
        plt.plot(
            v_range,
            grads,
            label=link.__class__.__name__.replace("Link", ""),
            linewidth=2,
            color=color,
        )

    plt.xlabel("Variance v", fontsize=11)
    plt.ylabel("∂/∂v E[g(f)]", fontsize=11)
    plt.title(
        f"Gradient w.r.t. Variance (μ={mu_fixed})",
        fontsize=12,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("link_function_gradients.png", dpi=150, bbox_inches="tight")
    print("Saved gradient comparison to: link_function_gradients.png")


def demonstrate_uncertainty_effect():
    """Show how uncertainty affects expected rates for different links."""
    f_mean = 1.0
    variances = [0.0, 0.5, 1.0, 2.0]

    links = [ExponentialLink(), SquaredLink(), ReLULink()]

    print("\n" + "=" * 70)
    print("Effect of Uncertainty on Expected Rates")
    print("=" * 70)
    print(f"\nLatent GP: f ~ N(μ={f_mean}, v)")
    print("\nExpected rates E[g(f)] for different variances:")
    print("-" * 70)

    for link in links:
        print(f"\n{link.__class__.__name__}:")
        rates = []
        for v in variances:
            rate = link.expected_rate(np.array([f_mean]), np.array([v]))[0]
            rates.append(rate)
            print(f"  v={v:.1f}: E[λ] = {rate:.4f}")

        # Show percentage change from certain (v=0) case
        baseline = rates[0]
        print("  Relative to v=0:")
        for v, rate in zip(variances[1:], rates[1:], strict=False):
            pct_change = ((rate - baseline) / baseline) * 100
            print(f"    v={v:.1f}: {pct_change:+.1f}%")


if __name__ == "__main__":
    print("Generating link function comparisons...\n")

    compare_link_functions()
    compare_gradients()
    demonstrate_uncertainty_effect()

    print("\n" + "=" * 70)
    print("Demo complete! Check the generated PNG files.")
    print("=" * 70)
