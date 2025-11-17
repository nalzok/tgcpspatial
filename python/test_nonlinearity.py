#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Test and demonstrate link functions for GP->rate mapping.
"""

import numpy as np

from nonlinearity import (
    ExponentialLink,
    IdentityLink,
    ReLULink,
    SoftplusLink,
    SquaredLink,
    get_link_function,
)


def test_basic_functionality():
    """Test that all link functions can compute basic operations."""
    print("=" * 70)
    print("Testing basic functionality of link functions")
    print("=" * 70)

    # Test inputs
    f = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    mu = np.array([0.0, 1.0, -1.0])
    v = np.array([0.1, 0.5, 1.0])

    links = [
        ExponentialLink(),
        SquaredLink(),
        ReLULink(),
        SoftplusLink(),
        IdentityLink(),
    ]

    for link in links:
        print(f"\n{link.__class__.__name__}:")
        print(f"  apply(f):              {link.apply(f)}")
        print(f"  expected_rate(μ, v):   {link.expected_rate(mu, v)}")
        print(
            f"  grad_μ E[g(f)]:        {link.expected_rate_gradient_mu(mu, v)}"
        )
        print(
            f"  grad_v E[g(f)]:        {link.expected_rate_gradient_v(mu, v)}"
        )


def test_exponential_link():
    """Test exponential link has correct analytical formulas."""
    print("\n" + "=" * 70)
    print("Testing ExponentialLink analytical formulas")
    print("=" * 70)

    link = ExponentialLink()
    mu = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 0.5, 1.0])

    # E[exp(f)] = exp(μ + v/2)
    expected = np.exp(mu + v / 2)
    computed = link.expected_rate(mu, v)
    print(f"\nE[exp(f)] where f ~ N(μ, v):")
    print(f"  Expected: {expected}")
    print(f"  Computed: {computed}")
    print(f"  Match: {np.allclose(expected, computed)}")

    # E[log(exp(f))] = μ
    expected = mu
    computed = link.expected_log_rate(mu, v)
    print(f"\nE[log(exp(f))] = E[f] = μ:")
    print(f"  Expected: {expected}")
    print(f"  Computed: {computed}")
    print(f"  Match: {np.allclose(expected, computed)}")


def test_squared_link():
    """Test squared link has correct formulas."""
    print("\n" + "=" * 70)
    print("Testing SquaredLink analytical formulas")
    print("=" * 70)

    link = SquaredLink()
    mu = np.array([0.0, 1.0, 2.0])
    v = np.array([1.0, 0.5, 0.1])

    # E[f²] = μ² + v
    expected = mu**2 + v
    computed = link.expected_rate(mu, v)
    print(f"\nE[f²] where f ~ N(μ, v):")
    print(f"  Expected: {expected}")
    print(f"  Computed: {computed}")
    print(f"  Match: {np.allclose(expected, computed)}")

    # ∂/∂μ E[f²] = 2μ
    expected_grad = 2 * mu
    computed_grad = link.expected_rate_gradient_mu(mu, v)
    print(f"\n∂/∂μ E[f²] = 2μ:")
    print(f"  Expected: {expected_grad}")
    print(f"  Computed: {computed_grad}")
    print(f"  Match: {np.allclose(expected_grad, computed_grad)}")


def test_relu_link():
    """Test ReLU link formulas."""
    print("\n" + "=" * 70)
    print("Testing ReLULink formulas")
    print("=" * 70)

    link = ReLULink()

    # Test case 1: μ > 0 (mostly positive)
    mu1 = np.array([2.0])
    v1 = np.array([0.5])
    rate1 = link.expected_rate(mu1, v1)
    print(f"\nE[max(f,0)] where f ~ N(2.0, 0.5):")
    print(f"  Result: {rate1[0]:.4f}")
    print(f"  (Should be close to 2.0)")

    # Test case 2: μ < 0 (mostly negative)
    mu2 = np.array([-2.0])
    v2 = np.array([0.5])
    rate2 = link.expected_rate(mu2, v2)
    print(f"\nE[max(f,0)] where f ~ N(-2.0, 0.5):")
    print(f"  Result: {rate2[0]:.4f}")
    print(f"  (Should be close to 0.0)")

    # Test case 3: μ = 0
    mu3 = np.array([0.0])
    v3 = np.array([1.0])
    rate3 = link.expected_rate(mu3, v3)
    print(f"\nE[max(f,0)] where f ~ N(0.0, 1.0):")
    print(f"  Result: {rate3[0]:.4f}")
    # For N(0,1), E[max(f,0)] = 1/√(2π) ≈ 0.3989
    print(f"  Expected: ~0.3989")


def test_registry():
    """Test link function registry."""
    print("\n" + "=" * 70)
    print("Testing link function registry")
    print("=" * 70)

    names = ["exp", "squared", "relu", "softplus", "identity"]
    for name in names:
        link = get_link_function(name)
        print(f"\nget_link_function('{name}'): {link}")
        f = np.array([1.0])
        print(f"  apply(1.0) = {link.apply(f)[0]:.4f}")


def test_log_likelihood_term():
    """Test expected log-likelihood computation."""
    print("\n" + "=" * 70)
    print("Testing expected log-likelihood term")
    print("=" * 70)

    # Simple test case
    K = np.array([5.0, 10.0, 3.0])  # spike counts
    N = np.array([1.0, 2.0, 1.0])  # occupancy
    mu = np.array([1.0, 2.0, 0.5])
    v = np.array([0.1, 0.2, 0.1])

    link = ExponentialLink()
    ll_term = link.expected_log_likelihood_term(K, N, mu, v)

    print(f"\nExpected log-likelihood E[K·log(λ) - N·λ]:")
    print(f"  K = {K}")
    print(f"  N = {N}")
    print(f"  μ = {mu}")
    print(f"  v = {v}")
    print(f"  Result: {ll_term}")
    print(f"  (Higher is better)")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    test_basic_functionality()
    test_exponential_link()
    test_squared_link()
    test_relu_link()
    test_registry()
    test_log_likelihood_term()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
