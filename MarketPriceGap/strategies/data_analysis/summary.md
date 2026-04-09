# Data Analysis - Initial Signal Exploration

## Purpose
Initial data inventory and correlation analysis across all available signals for the DA-Imbalance spread trading strategies.

## Key Findings
- 11 data sources identified with precise timing maps (D-1, live, D+1 at 10:00)
- DA price shape persistence: rho=0.76 (yesterday's hourly pattern repeats)
- Feature correlation matrix computed for all hourly and daily signals

## Scripts
- `analyze_all_signals.py` - Full data inventory, timing, correlations

## Data Files
- `data_inventory.csv` - Available data sources and timing
- `timing_map.csv` - When each signal becomes available
- `daily_feature_matrix.csv` - Daily aggregated features
- `master_with_spreads.csv` - Merged dataset with spread columns
- `spread_*.csv` - Hourly, DOW, monthly spread patterns and autocorrelation
- `feature_correlations.csv`, `hourly_feature_correlations.csv` - Correlation matrices
