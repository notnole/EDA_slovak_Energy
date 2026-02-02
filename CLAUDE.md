# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Repository Overview

**Name clarification**: "ipesoft_eda_data" - EDA refers to the **Ipesoft EDA database** (SCADA/energy data system), NOT Exploratory Data Analysis.

## Project Structure Pattern

This repository follows a **hierarchical analysis folder structure**. Each workstream is a top-level folder containing analysis categories, which contain specific analyses.

### Folder Structure Convention

```
WorkstreamFolder/
  category/                    # Analysis category (e.g., label/, features/, models/)
    specific_analysis/         # Individual analysis folder
      summary.md               # REQUIRED: Findings and conclusions
      *.png                    # REQUIRED: Visualization(s)
      data/                    # OPTIONAL: Intermediate data files
      scripts/                 # OPTIONAL: Analysis scripts (for models)
```

### Example: ImbalanceNowcasting/

```
ImbalanceNowcasting/
  label/                       # Label (target variable) analysis
    basic_stats/               # Basic statistics
      summary.md
      01_time_series_overview.png
      02_daily_seasonality.png
    decomposition/             # Time series decomposition
      summary.md
      01_stl_decomposition.png
      data/
    year_comparison/           # Year-over-year comparison
      summary.md
      *.png
  features/                    # Feature analysis
    correlation/               # Feature-label correlations
      summary.md
      01_feature_scatter_grid.png
      02_correlation_matrix.png
      data/
    seasonality/               # Seasonal patterns
    lag_correlation/           # Lag analysis
    decomposition/             # STL decomposition
  models/                      # Model development
    baseline/                  # Deterministic baseline
      summary.md
      scripts/
      data/
    lightgbm/                  # ML models
      summary.md
      scripts/
      plots/
      outputs/
      saved_models/
  report/                      # Final reports
    00_executive_summary.md
    figures/
```

## Code Style Rules

### CRITICAL: No Unicode in Code

**NEVER use emoji or special unicode characters in print statements or code.**

```python
# WRONG - causes encoding errors
print("Loading data...")
print("Done!")

# CORRECT - use ASCII only
print("[*] Loading data...")
print("[+] Done!")
print("--- Processing complete ---")
```

Common prefixes to use:
- `[*]` - Status/info
- `[+]` - Success
- `[-]` - Warning
- `[!]` - Error
- `---` - Section dividers

### Other Conventions

- Scripts run individually, no test suite
- European decimal format in source files (comma separator)
- Date formats: Slovak (D.M.YYYY) and English (M/D/YYYY)
- Outlier threshold: |value| > 300 MW removed as data errors

## Workstreams

### Workstream 1: System Imbalance Nowcasting
**Location**: `ImbalanceNowcasting/`, `data/`

Nowcasting models for Slovak electricity system imbalance. Predicts 15-minute settlement period imbalance from 3-minute SCADA measurements.

**Domain:**
- Label: System Imbalance (MWh) - OKTE publishes per 15-min settlement period
- Features: 3-minute SCADA (regulation, load, production, export/import) from Ipesoft EDA
- Key insight: Imbalance = -0.25 * mean(regulation) approximately

**Lead Times** (minutes until settlement period ends):
- Lead 12: 1 observation (hardest, baseline MAE ~6.6 MWh)
- Lead 9: 2 observations
- Lead 6: 3 observations
- Lead 3: 4 observations
- Lead 0: 5 observations (baseline MAE ~2.2 MWh)

### Workstream 2: Load Prediction Analysis
**Location**: `LoadAnalysis/`, `features/DamasLoad/`

Day-ahead load forecasting for Slovakia grid. DAMAS forecast baseline (MAE ~68 MW, MAPE ~2.3%).

## Data Pipeline

```
RawData/                 -> Source SCADA CSVs
OKTE_Imbalnce/           -> Source imbalance CSVs
        |
        v
data/features/           -> Cleaned 3-min series
data/master/             -> Merged master dataset
        |
        v
ImbalanceNowcasting/     -> Analysis and models
```

**Timing convention**: Feature timestamps = period START (observation available at START + 3min).

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run any analysis script
python ImbalanceNowcasting/models/lightgbm/scripts/train_lightgbm.py
```
