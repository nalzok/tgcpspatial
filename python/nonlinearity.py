#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Link functions for mapping latent Gaussian process to firing rates.

Each link function g maps latent values f to rates λ = g(f).
For variational inference with f ~ N(μ, v), we need to compute:
- E[g(f)]: expected rate
- E[log(g(f))]: expected log-rate
- Gradients w.r.t. μ and v for optimization
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import norm


class LinkFunction(ABC):
    """Abstract base class for link functions g: f -> λ."""

    @abstractmethod
    def apply(self, f):
        """
        Compute rate λ = g(f).

        Args:
            f (ndarray): Latent function values

        Returns:
            ndarray: Rates λ = g(f)
        """
        pass

    @abstractmethod
    def log_apply(self, f):
        """
        Compute log-rate log(λ) = log(g(f)).

        Args:
            f (ndarray): Latent function values

        Returns:
            ndarray: Log-rates log(g(f))
        """
        pass

    @abstractmethod
    def expected_rate(self, mu, v):
        """
        Compute expected rate E[g(f)] where f ~ N(μ, v).

        Args:
            mu (ndarray): Mean of latent GP
            v (ndarray): Variance of latent GP

        Returns:
            ndarray: Expected rates E[g(f)]
        """
        pass

    @abstractmethod
    def expected_log_rate(self, mu, v):
        """
        Compute expected log-rate E[log(g(f))] where f ~ N(μ, v).

        Args:
            mu (ndarray): Mean of latent GP
            v (ndarray): Variance of latent GP

        Returns:
            ndarray: Expected log-rates E[log(g(f))]
        """
        pass

    @abstractmethod
    def expected_rate_gradient_mu(self, mu, v):
        """
        Compute ∂/∂μ E[g(f)] where f ~ N(μ, v).

        Args:
            mu (ndarray): Mean of latent GP
            v (ndarray): Variance of latent GP

        Returns:
            ndarray: Gradient of expected rate w.r.t. mean
        """
        pass

    @abstractmethod
    def expected_rate_gradient_v(self, mu, v):
        """
        Compute ∂/∂v E[g(f)] where f ~ N(μ, v).

        Args:
            mu (ndarray): Mean of latent GP
            v (ndarray): Variance of latent GP

        Returns:
            ndarray: Gradient of expected rate w.r.t. variance
        """
        pass

    def expected_log_likelihood_term(self, K, N, mu, v):
        """
        Compute E[K·log(g(f)) - N·g(f)] where f ~ N(μ, v).

        This is the expected Poisson log-likelihood term (up to constants).

        Args:
            K (ndarray): Spike counts
            N (ndarray): Occupancy times
            mu (ndarray): Mean of latent GP
            v (ndarray): Variance of latent GP

        Returns:
            ndarray: Expected log-likelihood contribution per bin
        """
        E_rate = self.expected_rate(mu, v)
        E_log_rate = self.expected_log_rate(mu, v)
        return K * E_log_rate - N * E_rate

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExponentialLink(LinkFunction):
    """
    Exponential link: λ = exp(f).

    This is the standard link for Log-Gaussian Cox Processes.
    For f ~ N(μ, v): E[exp(f)] = exp(μ + v/2)
    """

    def apply(self, f):
        return np.exp(f)

    def log_apply(self, f):
        return f

    def expected_rate(self, mu, v):
        return np.exp(mu + v / 2)

    def expected_log_rate(self, mu, v):
        return mu

    def expected_rate_gradient_mu(self, mu, v):
        return np.exp(mu + v / 2)

    def expected_rate_gradient_v(self, mu, v):
        return 0.5 * np.exp(mu + v / 2)


class SquaredLink(LinkFunction):
    """
    Squared link: λ = f².

    For f ~ N(μ, v): E[f²] = μ² + v
    Note: This can produce zero rates when f=0.
    """

    def apply(self, f):
        return f**2

    def log_apply(self, f):
        return 2 * np.log(np.abs(f))

    def expected_rate(self, mu, v):
        return mu**2 + v

    def expected_log_rate(self, mu, v):
        """
        E[log(f²)] = E[2·log|f|] (computed numerically).

        This requires numerical integration for the truncated normal.
        """
        mu_flat = np.ravel(mu)
        v_flat = np.ravel(v)
        result = np.zeros_like(mu_flat)

        for i in range(len(mu_flat)):
            m, s2 = mu_flat[i], v_flat[i]
            s = np.sqrt(s2)

            def integrand(f):
                return 2 * np.log(np.abs(f) + 1e-10) * norm.pdf(f, m, s)

            # Integrate over ±5 standard deviations
            result[i], _ = quad(integrand, m - 5 * s, m + 5 * s)

        return result.reshape(mu.shape)

    def expected_rate_gradient_mu(self, mu, v):
        return 2 * mu

    def expected_rate_gradient_v(self, mu, v):
        return np.ones_like(v)


class ReLULink(LinkFunction):
    """
    Rectified linear link: λ = max(f, 0).

    For f ~ N(μ, v):
    E[max(f,0)] = μ·Φ(μ/σ) + σ·φ(μ/σ)
    where Φ is CDF and φ is PDF of standard normal.
    """

    def apply(self, f):
        return np.maximum(f, 0)

    def log_apply(self, f):
        return np.log(np.maximum(f, 1e-10))

    def expected_rate(self, mu, v):
        sigma = np.sqrt(v)
        # Standardized value
        alpha = mu / (sigma + 1e-10)
        # E[max(f,0)] = μ·Φ(α) + σ·φ(α)
        return mu * norm.cdf(alpha) + sigma * norm.pdf(alpha)

    def expected_log_rate(self, mu, v):
        """
        E[log(max(f,0))] computed numerically.

        Note: This is tricky because log(0) is undefined.
        We use a small epsilon for numerical stability.
        """
        mu_flat = np.ravel(mu)
        v_flat = np.ravel(v)
        result = np.zeros_like(mu_flat)

        for i in range(len(mu_flat)):
            m, s2 = mu_flat[i], v_flat[i]
            s = np.sqrt(s2)

            def integrand(f):
                if f <= 0:
                    return 0.0
                return np.log(f + 1e-10) * norm.pdf(f, m, s)

            # Integrate from 0 to +5σ
            result[i], _ = quad(integrand, 0, m + 5 * s)

        return result.reshape(mu.shape)

    def expected_rate_gradient_mu(self, mu, v):
        sigma = np.sqrt(v)
        alpha = mu / (sigma + 1e-10)
        return norm.cdf(alpha)

    def expected_rate_gradient_v(self, mu, v):
        sigma = np.sqrt(v)
        alpha = mu / (sigma + 1e-10)
        # ∂/∂v E[max(f,0)] = ∂/∂σ E[max(f,0)] · ∂σ/∂v
        # ∂/∂σ E[max(f,0)] = φ(α) - μ·φ(α)·α/σ
        # ∂σ/∂v = 1/(2σ)
        grad_sigma = norm.pdf(alpha) * (1 - alpha * mu / (sigma + 1e-10))
        return grad_sigma / (2 * sigma + 1e-10)


class SoftplusLink(LinkFunction):
    """
    Softplus link: λ = log(1 + exp(f)) = softplus(f).

    This is a smooth approximation to ReLU.
    For large |f|, softplus(f) ≈ max(f, 0).
    """

    def apply(self, f):
        # Numerically stable softplus
        return np.log1p(np.exp(-np.abs(f))) + np.maximum(f, 0)

    def log_apply(self, f):
        return np.log(self.apply(f))

    def expected_rate(self, mu, v):
        """
        E[softplus(f)] computed using Gauss-Hermite quadrature.
        """
        mu_flat = np.ravel(mu)
        v_flat = np.ravel(v)
        result = np.zeros_like(mu_flat)

        for i in range(len(mu_flat)):
            m, s2 = mu_flat[i], v_flat[i]
            s = np.sqrt(s2)

            def integrand(f):
                return self.apply(f) * norm.pdf(f, m, s)

            result[i], _ = quad(integrand, m - 5 * s, m + 5 * s)

        return result.reshape(mu.shape)

    def expected_log_rate(self, mu, v):
        """E[log(softplus(f))] computed numerically."""
        mu_flat = np.ravel(mu)
        v_flat = np.ravel(v)
        result = np.zeros_like(mu_flat)

        for i in range(len(mu_flat)):
            m, s2 = mu_flat[i], v_flat[i]
            s = np.sqrt(s2)

            def integrand(f):
                return self.log_apply(f) * norm.pdf(f, m, s)

            result[i], _ = quad(integrand, m - 5 * s, m + 5 * s)

        return result.reshape(mu.shape)

    def expected_rate_gradient_mu(self, mu, v):
        """∂/∂μ E[softplus(f)] computed numerically."""
        mu_flat = np.ravel(mu)
        v_flat = np.ravel(v)
        result = np.zeros_like(mu_flat)

        for i in range(len(mu_flat)):
            m, s2 = mu_flat[i], v_flat[i]
            s = np.sqrt(s2)

            # E[sigmoid(f)] where sigmoid is derivative of softplus
            def integrand(f):
                sigmoid = 1 / (1 + np.exp(-f))
                return sigmoid * norm.pdf(f, m, s)

            result[i], _ = quad(integrand, m - 5 * s, m + 5 * s)

        return result.reshape(mu.shape)

    def expected_rate_gradient_v(self, mu, v):
        """∂/∂v E[softplus(f)] computed numerically."""
        mu_flat = np.ravel(mu)
        v_flat = np.ravel(v)
        result = np.zeros_like(mu_flat)

        for i in range(len(mu_flat)):
            m, s2 = mu_flat[i], v_flat[i]
            s = np.sqrt(s2)

            # E[sigmoid(f)·(f-μ)] / (2σ)
            def integrand(f):
                sigmoid = 1 / (1 + np.exp(-f))
                return sigmoid * (f - m) * norm.pdf(f, m, s)

            integral, _ = quad(integrand, m - 5 * s, m + 5 * s)
            result[i] = integral / (2 * s)

        return result.reshape(mu.shape)


class IdentityLink(LinkFunction):
    """
    Identity link: λ = f.

    This is a linear model (not suitable for rates that must be positive).
    For f ~ N(μ, v): E[f] = μ
    """

    def apply(self, f):
        return f

    def log_apply(self, f):
        return np.log(np.maximum(f, 1e-10))

    def expected_rate(self, mu, v):
        return mu

    def expected_log_rate(self, mu, v):
        """E[log(f)] for f ~ N(μ, v) - computed numerically."""
        mu_flat = np.ravel(mu)
        v_flat = np.ravel(v)
        result = np.zeros_like(mu_flat)

        for i in range(len(mu_flat)):
            m, s2 = mu_flat[i], v_flat[i]
            s = np.sqrt(s2)

            def integrand(f):
                return np.log(np.maximum(f, 1e-10)) * norm.pdf(f, m, s)

            result[i], _ = quad(integrand, m - 5 * s, m + 5 * s)

        return result.reshape(mu.shape)

    def expected_rate_gradient_mu(self, mu, v):
        return np.ones_like(mu)

    def expected_rate_gradient_v(self, mu, v):
        return np.zeros_like(v)


# Registry of available link functions
LINK_FUNCTIONS = {
    "exponential": ExponentialLink,
    "exp": ExponentialLink,
    "squared": SquaredLink,
    "square": SquaredLink,
    "relu": ReLULink,
    "softplus": SoftplusLink,
    "identity": IdentityLink,
    "linear": IdentityLink,
}


def get_link_function(name):
    """
    Get a link function by name.

    Args:
        name (str): Name of link function (case-insensitive)

    Returns:
        LinkFunction: Instance of the requested link function

    Example:
        >>> link = get_link_function('relu')
        >>> link.apply(np.array([1.0, -0.5, 2.0]))
        array([1. , 0. , 2. ])
    """
    name_lower = name.lower()
    if name_lower not in LINK_FUNCTIONS:
        available = ", ".join(LINK_FUNCTIONS.keys())
        raise ValueError(f"Unknown link function '{name}'. Available: {available}")
    return LINK_FUNCTIONS[name_lower]()
