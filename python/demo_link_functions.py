#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Simple demonstration of using different link functions with LGCP inference.

This script shows how to:
1. Load grid cell data
2. Run inference with different link functions
3. Compare the results
"""

from config import datadir
from lgcp.data import Dataset
from lgcp.infer import lgcp2d
from lgcp.kern import kernelft
from nonlinearity import ExponentialLink, ReLULink, get_link_function

# Load and prepare data
print("Loading grid cell data...")
fn = "r2405_051216b_cell1816.mat"
data = Dataset.from_file(datadir + fn).prepare()
print(f"Grid shape: {data.shape}")
print(f"Total spikes: {int(data.K.sum())}")

# Create kernel
kf = kernelft(data.shape, data.P, data.V, angle=data.angle, style="grid")

print("\n" + "=" * 60)
print("Running inference with EXPONENTIAL link (standard LGCP)")
print("=" * 60)
link_exp = ExponentialLink()
result_exp, model_exp = lgcp2d(
    kf,
    data.N,
    data.K,
    data.prior_mean,
    (data.kdelograte, None),
    link=link_exp,
    eps=1e-5,
    verbose=True,
)
print(f"Log-likelihood: {result_exp.ll:.2f}")

print("\n" + "=" * 60)
print("Running inference with RELU link")
print("=" * 60)
link_relu = ReLULink()
result_relu, model_relu = lgcp2d(
    kf,
    data.N,
    data.K,
    data.prior_mean,
    (data.kdelograte, None),
    link=link_relu,
    eps=1e-5,
    verbose=True,
)
print(f"Log-likelihood: {result_relu.ll:.2f}")

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"Exponential link: LL = {result_exp.ll:.2f}")
print(f"ReLU link:        LL = {result_relu.ll:.2f}")
print(
    f"Difference:       ΔLL = {result_relu.ll - result_exp.ll:.2f} "
    f"({'better' if result_relu.ll > result_exp.ll else 'worse'})"
)

print("\nYou can also use the registry:")
print("  link = get_link_function('exponential')")
print("  link = get_link_function('relu')")
print("  link = get_link_function('softplus')")
