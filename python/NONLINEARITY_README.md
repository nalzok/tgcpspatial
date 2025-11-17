# Link Functions for Gaussian Process to Rate Mapping

This module provides a modular framework for different nonlinear link functions that map latent Gaussian process values to firing rates in Cox process models.

## Overview

In a Log-Gaussian Cox Process (LGCP), the standard assumption is that the rate function follows:
```
λ(x) = exp(f(x))
```
where `f(x)` is a Gaussian process.

This module generalizes this to arbitrary link functions:
```
λ(x) = g(f(x))
```

## Available Link Functions

### 1. ExponentialLink (Standard LGCP)
- **Function:** `λ = exp(f)`
- **Properties:** Always positive, standard for LGCP
- **Expected rate:** `E[exp(f)] = exp(μ + v/2)` for `f ~ N(μ, v)`
- **Use case:** Standard choice, well-studied theory

### 2. SquaredLink
- **Function:** `λ = f²`
- **Properties:** Always non-negative, symmetric around zero
- **Expected rate:** `E[f²] = μ² + v`
- **Use case:** When you want symmetric response to positive/negative latent values

### 3. ReLULink
- **Function:** `λ = max(f, 0)`
- **Properties:** Piecewise linear, zero for negative f
- **Expected rate:** `E[max(f,0)] = μ·Φ(μ/σ) + σ·φ(μ/σ)`
- **Use case:** Sparse representations, neural network-inspired

### 4. SoftplusLink
- **Function:** `λ = log(1 + exp(f))`
- **Properties:** Smooth approximation to ReLU, always positive
- **Use case:** Smooth version of ReLU with better gradients

### 5. IdentityLink
- **Function:** `λ = f`
- **Properties:** Linear, can be negative (not suitable for rates!)
- **Expected rate:** `E[f] = μ`
- **Use case:** Debugging, linear models

## Interface Design

All link functions inherit from the abstract `LinkFunction` base class and implement:

### Core Methods

1. **`apply(f)`** - Compute rate `λ = g(f)`
   ```python
   link = ReLULink()
   rates = link.apply(latent_values)
   ```

2. **`expected_rate(mu, v)`** - Compute `E[g(f)]` where `f ~ N(μ, v)`
   ```python
   expected_rates = link.expected_rate(mu, variance)
   ```

3. **`expected_log_rate(mu, v)`** - Compute `E[log(g(f))]`
   ```python
   expected_log_rates = link.expected_log_rate(mu, variance)
   ```

4. **`expected_rate_gradient_mu(mu, v)`** - Compute `∂/∂μ E[g(f)]`
   ```python
   grad_mu = link.expected_rate_gradient_mu(mu, variance)
   ```

5. **`expected_rate_gradient_v(mu, v)`** - Compute `∂/∂v E[g(f)]`
   ```python
   grad_v = link.expected_rate_gradient_v(mu, variance)
   ```

6. **`expected_log_likelihood_term(K, N, mu, v)`** - Compute `E[K·log(g(f)) - N·g(f)]`
   ```python
   ll_term = link.expected_log_likelihood_term(spike_counts, occupancy, mu, variance)
   ```

## Usage Examples

### Basic Usage

```python
from nonlinearity import get_link_function
import numpy as np

# Get a link function by name
link = get_link_function('relu')

# Apply to latent values
f = np.array([-1.0, 0.0, 1.0, 2.0])
rates = link.apply(f)
# Output: [0., 0., 1., 2.]

# Compute expected rates for uncertain latent values
mu = np.array([0.5, 1.0, 1.5])
variance = np.array([0.1, 0.2, 0.3])
expected_rates = link.expected_rate(mu, variance)
```

### For Variational Inference

```python
# In your variational inference loop:
link = ExponentialLink()

# Compute expected rate for loss function
E_rate = link.expected_rate(posterior_mean, posterior_variance)

# Compute gradients for optimization
grad_mu = link.expected_rate_gradient_mu(posterior_mean, posterior_variance)
grad_v = link.expected_rate_gradient_v(posterior_mean, posterior_variance)

# Compute log-likelihood contribution
ll_contribution = link.expected_log_likelihood_term(
    spike_counts, occupancy, posterior_mean, posterior_variance
)
```

### Comparing Link Functions

```python
from nonlinearity import ExponentialLink, ReLULink, SquaredLink
import matplotlib.pyplot as plt

# Test different link functions
f_range = np.linspace(-3, 3, 100)
links = [ExponentialLink(), ReLULink(), SquaredLink()]

for link in links:
    rates = link.apply(f_range)
    plt.plot(f_range, rates, label=link.__class__.__name__)

plt.xlabel('Latent value f')
plt.ylabel('Rate λ = g(f)')
plt.legend()
plt.show()
```

## Integration with Existing LGCP Code

The link functions are designed to integrate with the variational inference in `lgcp/infer.py`.

### Current LGCP Code Pattern
```python
# In lgcpnd function (line 167):
nr = nm * sexp(Ft(uh) + z0 + vh * 0.5)  # Expected rate
```

### With Link Functions
```python
from nonlinearity import get_link_function

def lgcpnd_with_link(kf, N, K, z0f, zh0, vh0, link_name='exponential', **opts):
    link = get_link_function(link_name)

    # Replace sexp(z + v/2) with link.expected_rate(z, v)
    def _nr(uh, vh):
        z = Ft(uh) + z0  # posterior mean
        return nm * link.expected_rate(z, vh)

    # Use link.expected_rate_gradient_mu() for mean updates
    # Use link.expected_rate_gradient_v() for variance updates
    ...
```

## Mathematical Details

### Variational Inference Requirements

For variational inference in Cox processes, we need to compute expectations over the variational posterior `q(f) = N(μ, v)`:

1. **Expected rate:** `E_q[λ] = E_q[g(f)]`
   - Needed for the Poisson likelihood term

2. **Expected log-rate:** `E_q[log λ] = E_q[log g(f)]`
   - Needed for the spike count log-likelihood

3. **Gradients:** `∂/∂μ E_q[g(f)]` and `∂/∂v E_q[g(f)]`
   - Needed for coordinate descent optimization

### Analytical vs Numerical Integration

- **ExponentialLink:** All expectations have closed form
- **SquaredLink:** Expected rate is closed form, log-rate is numerical
- **ReLULink:** Expected rate is closed form (truncated normal), log-rate is numerical
- **SoftplusLink:** All expectations computed numerically (Gauss-Hermite quadrature)

Numerical integration uses `scipy.integrate.quad` with integration bounds at ±5σ from the mean.

## Design Principles

1. **Modularity:** Easy to add new link functions by subclassing `LinkFunction`
2. **Composability:** Link functions work with numpy arrays of any shape
3. **Consistency:** All link functions follow the same interface
4. **Testability:** Each link function includes analytical tests where available
5. **Performance:** Analytical formulas used when available, efficient numerical integration otherwise

## Adding New Link Functions

To add a new link function:

```python
class MyLink(LinkFunction):
    def apply(self, f):
        return my_function(f)

    def expected_rate(self, mu, v):
        # Compute E[my_function(f)] where f ~ N(mu, v)
        # Use analytical formula if available, else numerical integration
        ...

    # Implement other required methods
    ...

# Add to registry
LINK_FUNCTIONS['mylink'] = MyLink
```

## Testing

Run the test suite:
```bash
uv run python test_nonlinearity.py
```

This tests:
- Basic functionality of all link functions
- Analytical formulas against known results
- Gradient computations
- Registry lookup
- Log-likelihood terms

## Future Extensions

Possible additions:
- **Inverse link functions:** For transforming data
- **Parametric link families:** e.g., Box-Cox transformations
- **Compositional links:** Combining multiple transformations
- **Automatic differentiation:** For gradient computation
- **GPU acceleration:** For large-scale inference
