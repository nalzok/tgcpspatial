# Analysis Scripts

This directory contains analysis scripts for evaluating LGCP models with different link functions.

## Scripts

### Main Analysis
- **`run_all_datasets.py`** - Master script that runs complete analysis on all datasets
  - Fits models with multiple link functions
  - Evaluates goodness-of-fit
  - Generates summary statistics and visualizations
  - Saves results to CSV
  - **Run this first** to get comprehensive results

### Individual Analyses
- **`run_evaluation.py`** - Evaluation framework with train/test split
  - Computes ELBO, validation log-likelihood
  - Calculates predictive metrics (MSE, R², deviance)
  - Generates comparison plots

- **`test_time_rescaling.py`** - Time-rescaling theorem goodness-of-fit tests
  - Implements Brown et al. (2002) methodology
  - Tests if rescaled ISIs follow Exp(1)
  - Generates 6-panel diagnostic plots

- **`sample_synthetic_data.py`** - Forward sampling from fitted models
  - Samples from posterior GP
  - Generates synthetic spike trains
  - Compares real vs synthetic data

## Usage

### Run on All Datasets
```bash
cd python/analysis
uv run python run_all_datasets.py
```

This will:
1. Process all 15 datasets in krupic2018/
2. Fit exponential and ReLU link functions
3. Save results to `results/` directory
4. Create summary plots and CSV files

### Run Individual Analyses
```bash
# Evaluation framework
uv run python run_evaluation.py

# Goodness-of-fit tests
uv run python test_time_rescaling.py

# Synthetic data sampling
uv run python sample_synthetic_data.py
```

## Output Structure

```
results/
├── all_results.csv                    # All results in CSV format
├── cross_dataset_summary.png          # Summary plots across datasets
├── r2288_180515b_tet2_cell2_GC/       # Per-dataset results
│   └── r2288_180515b_tet2_cell2_GC_rate_maps.png
├── r2289_250515b_tet2_cell2_GC/
│   └── ...
└── ...
```

## Core Modules Used

These analysis scripts use the following core modules from the parent directory:
- `nonlinearity.py` - Link function framework
- `evaluate_links.py` - Evaluation metrics
- `goodness_of_fit.py` - Time-rescaling theorem
- `analysis_utils.py` - Common utilities
- `lgcp/` - LGCP inference engine
- `config.py` - Data paths and configuration
