# Baseline Forecast (DAMAS) Error Analysis

## Overview
Analysis of DAMAS day-ahead load forecast performance.

## Overall Metrics

| Metric | Value |
|--------|-------|
| MAE | 68.2 MW |
| RMSE | 88.1 MW |
| MAPE | 2.34% |
| Bias | -6.3 MW (slight over-forecast) |

## Error by Hour

| Hours | MAE (MW) | MAPE (%) | Notes |
|-------|----------|----------|-------|
| Night (2-4) | 48-50 | 2.0% | Best performance |
| Peak (12-14) | **83-86** | **2.7-2.8%** | Worst performance |

## MAE vs MAPE Comparison

| Metric | Range | Ratio |
|--------|-------|-------|
| MAE | 48 - 86 MW | 1.78x |
| MAPE | 1.97 - 2.78% | 1.41x |

**Key insight**: MAPE is more uniform across hours. Peak hour errors are proportionally higher, not just absolutely higher.

## Systematic Bias by Hour

| Hours | Mean Error | Direction |
|-------|------------|-----------|
| Morning (6-9) | -28 to -39 MW | **Over-forecast** |
| Evening (21-24) | +22 to +27 MW | **Under-forecast** |

DAMAS systematically over-forecasts mornings, under-forecasts evenings.

## Plots

- `06_forecast_errors.png` - Error analysis
- `06b_error_heatmap.png` - Error heatmap by hour/dow
- `08_anomalies.png` - Large error detection
- `10_mae_vs_mape_by_hour.png` - MAE vs MAPE comparison
