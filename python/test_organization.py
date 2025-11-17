#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""Quick test of the reorganized code."""

from analysis_utils import get_dataset_list, load_dataset_safe

print("Testing reorganized codebase...")
print("="*60)

# Test dataset listing
datasets = get_dataset_list(exclude_simulated=True)
print(f"\nFound {len(datasets)} datasets")
print(f"First 5: {datasets[:5]}")

# Test loading one dataset
print(f"\nTesting data loading...")
data = load_dataset_safe(datasets[0], verbose=True)

if data is not None:
    print("✓ Data loading works")
else:
    print("✗ Data loading failed")

print("\n" + "="*60)
print("Organization test complete!")
