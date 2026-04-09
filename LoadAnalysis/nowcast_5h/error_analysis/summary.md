# Nowcast 5h Error Analysis

## Overview

Detailed error breakdown of the two-stage load forecast correction model across all 5 horizons (H+1 to H+5), evaluated on the holdout period.

## Performance by Horizon

| Horizon | MAE (MW) | RMSE (MW) | Bias (MW) | Large Error % |
|---------|----------|-----------|-----------|---------------|
| H+1 | 43.3 | 58.3 | -18.4 | 12.5% |
| H+2 | 48.9 | 65.4 | -15.8 | 12.0% |
| H+3 | 50.9 | 66.5 | -13.6 | 12.1% |
| H+4 | 57.3 | 74.5 | -22.8 | 11.8% |
| H+5 | 56.2 | 73.5 | -13.3 | 12.0% |

DAMAS baseline: ~83.4 MW MAE.

## Worst Hours (H+1)

Morning ramp hours have the highest errors:
- H09: 90.2 MW, H10: 89.6 MW, H08: 73.8 MW
- Night hours (H01-H03): 22-25 MW (best)

## Day-of-Week Pattern
- Monday (dow=0) consistently worst across all horizons (~10% higher MAE than average)
- Friday-Saturday (dow=4-5) slightly better

## Files

| File | Description |
|------|-------------|
| `error_summary.json` | Full MAE/RMSE/bias by horizon, hour, and day-of-week |
| `01_mae_by_horizon.png` | Bar chart: MAE by horizon |
| `02_error_distributions.png` | Error histograms per horizon |
| `03_mae_by_hour_heatmap.png` | Heatmap: MAE by hour x horizon |
| `04_actual_vs_predicted.png` | Scatter: actual vs predicted |
| `05_error_timeseries_last30d.png` | Last 30 days error time series |
| `06_cumulative_error_distribution.png` | CDF of absolute errors |
| `h{1-5}_predictions.csv` | Raw predictions per horizon |
