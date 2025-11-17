#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Example showing how to integrate link functions with the existing LGCP inference.

This demonstrates how to modify the inference code to support different
link functions beyond the standard exponential.
"""

import numpy as np

from lgcp.data import Dataset
from lgcp.kern import kernelft
from nonlinearity import ExponentialLink, ReLULink, SquaredLink, get_link_function


def demonstrate_link_usage():
    """Show basic usage of link functions."""
    print("=" * 70)
    print("Basic Link Function Usage")
    print("=" * 70)

    # Create some example data
    mu = np.random.randn(10, 10)  # Posterior mean (log-space)
    v = np.abs(np.random.randn(10, 10)) * 0.1  # Posterior variance

    # Try different link functions
    links = [
        get_link_function("exponential"),
        get_link_function("squared"),
        get_link_function("relu"),
    ]

    for link in links:
        print(f"\n{link.__class__.__name__}:")

        # Compute expected rate
        expected_rate = link.expected_rate(mu, v)
        print(f"  Mean expected rate: {np.mean(expected_rate):.4f}")

        # Compute gradients for optimization
        grad_mu = link.expected_rate_gradient_mu(mu, v)
        grad_v = link.expected_rate_gradient_v(mu, v)
        print(f"  Mean gradient (μ): {np.mean(grad_mu):.4f}")
        print(f"  Mean gradient (v): {np.mean(grad_v):.4f}")


def modified_inference_pseudocode():
    """
    Pseudocode showing how to modify lgcpnd() to support link functions.

    This is NOT runnable code, but shows the conceptual changes needed.
    """
    print("\n" + "=" * 70)
    print("Integration Pattern for lgcpnd()")
    print("=" * 70)

    pseudocode = '''
def lgcpnd_with_link(kf, N, K, z0f, zh0, vh0, link_name='exponential', **opts):
    """
    Modified version of lgcpnd that supports different link functions.

    New parameter:
        link_name (str): Name of link function ('exponential', 'relu', 'squared', etc.)
    """
    # Get the link function
    link = get_link_function(link_name)

    # ... existing setup code ...

    def _nr(uh, vh):
        """Compute expected rate using the selected link function."""
        z = Ft(uh) + z0  # posterior mean log-rate
        # OLD: return nm * sexp(z + vh * 0.5)
        # NEW: use link function
        return nm * link.expected_rate(z, vh)

    def loss(uh, vh):
        """Loss function with link-specific expected rate."""
        z = Ft(uh) + z0

        # Expected rate using link function
        E_rate = link.expected_rate(z, vh)
        nyr = nm @ (E_rate - ym * link.expected_log_rate(z, vh))

        # ... rest of loss computation ...
        uΛu = ssum(uh**2 * Λh)
        C = _C(uh, vh)
        trΛΣ = ssum(C**2 * Λh)
        ldΣq = ssum(slog(np.diag(C)))
        return ll0 + nyr + 0.5 * (uΛu + trΛΣ) - ldΣq

    def meanupdate(uh, vh):
        """Mean update using link function gradients."""
        z = Ft(uh) + z0
        E_rate = link.expected_rate(z, vh)
        grad_rate = link.expected_rate_gradient_mu(z, vh)

        nr = nm * E_rate
        J = Λh * uh + Fo(nr - nym)

        def Hu(u):
            return Λh * u + Fo(nm * grad_rate * Ft(u))

        Hv = LinearOperator((R, R), Hu, Hu, dtype=np.float32)
        return -np.float32(minres(Hv, J, rtol=mintol, M=M)[0])

    def varupdate(uh, vh):
        """Variance update using link function gradients."""
        # This is more complex and depends on the specific link function
        # For exponential: grad_v = 0.5 * E[exp(f)]
        # For others: use link.expected_rate_gradient_v()
        return np.sum((Fm.T @ _C(uh, vh)) ** 2, 1, "f") - vh

    # Run coordinate descent as before
    uh, vh = coordinate_descent(uh, vh, meanupdate, varupdate, **opts)
    z, r, v, μ = unpack(uh, vh)

    return InferResult(InferState(z, v), -loss(uh, vh), InferInfo(r, uh, vh, μ))
    '''

    print(pseudocode)


def usage_example():
    """Show how to use the modified inference in practice."""
    print("\n" + "=" * 70)
    print("Usage Example")
    print("=" * 70)

    example = '''
# Load data as usual
from config import datadir
from lgcp.data import Dataset
from lgcp.kern import kernelft

data = Dataset.from_file(datadir + "r2405_051216b_cell1816.mat").prepare()

# Create kernel as usual
kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")

# Run inference with DIFFERENT link functions:

# Standard exponential link (default)
result_exp, model_exp = lgcp2d_with_link(
    kf, data.N, data.K, data.prior_mean,
    (data.kdelograte, None),
    link_name='exponential',
    eps=1e-5, verbose=True
)

# Try squared link
result_sq, model_sq = lgcp2d_with_link(
    kf, data.N, data.K, data.prior_mean,
    (data.kdelograte, None),
    link_name='squared',
    eps=1e-5, verbose=True
)

# Try ReLU link
result_relu, model_relu = lgcp2d_with_link(
    kf, data.N, data.K, data.prior_mean,
    (data.kdelograte, None),
    link_name='relu',
    eps=1e-5, verbose=True
)

# Compare results
print(f"Log-likelihood (exponential): {result_exp.ll:.2f}")
print(f"Log-likelihood (squared):     {result_sq.ll:.2f}")
print(f"Log-likelihood (relu):        {result_relu.ll:.2f}")

# Visualize different rate maps
arena = data.arena
arena.imshow(result_exp.info.r, lw=8)  # Exponential
arena.imshow(result_sq.info.r, lw=8)   # Squared
arena.imshow(result_relu.info.r, lw=8) # ReLU
    '''

    print(example)


def key_considerations():
    """Important points to consider when using different link functions."""
    print("\n" + "=" * 70)
    print("Key Considerations")
    print("=" * 70)

    considerations = """
1. CHOICE OF LINK FUNCTION:
   - Exponential: Standard choice, always positive, well-studied
   - Squared: Symmetric, can emphasize high/low firing equally
   - ReLU: Sparse, only positive latent values contribute
   - Softplus: Smooth alternative to ReLU

2. INTERPRETATION DIFFERENCES:
   - With exp link: latent f is in log-space, multiplicative effects
   - With squared link: latent f has symmetric effects
   - With ReLU: latent f has threshold behavior

3. NUMERICAL STABILITY:
   - Some links require numerical integration (slower)
   - Exponential and Squared have analytical expectations (faster)
   - Consider computational cost vs. model flexibility

4. INITIALIZATION:
   - Initial guess (zh0, vh0) should be appropriate for the link
   - For exp: initialize in log-space
   - For squared/relu: different scales may be needed

5. PRIOR SPECIFICATION:
   - Prior mean (prior_mean) interpretation changes with link
   - May need to adjust kernel amplitude (V) for different links

6. MODEL COMPARISON:
   - Compare log-likelihoods across links
   - Check if convergence is stable
   - Validate on held-out data

7. BIOLOGICAL INTERPRETATION:
   - Exponential: standard for log-linear models
   - ReLU: similar to neural network activations
   - Different links → different tuning curve shapes
    """

    print(considerations)


if __name__ == "__main__":
    demonstrate_link_usage()
    modified_inference_pseudocode()
    usage_example()
    key_considerations()

    print("\n" + "=" * 70)
    print("Integration guide complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Modify lgcp/infer.py to add link_name parameter")
    print("2. Replace sexp() calls with link.expected_rate()")
    print("3. Update gradient computations for mean/variance updates")
    print("4. Test with simple synthetic data first")
    print("5. Compare performance across different link functions")
    print("=" * 70)
