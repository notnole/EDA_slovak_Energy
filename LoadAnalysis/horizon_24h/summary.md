# 24-Hour Horizon Analysis

## Overview
Analysis of prediction at 24-hour ahead horizon (day-ahead forecasting).

## Challenge
At 24h horizon, we cannot use lag-1h error (which has 0.86 correlation).
We only have:
- Yesterday's complete errors
- Tomorrow's forecast
- 3-minute data features

## Feature Correlations with 24h-Ahead Error

| Feature | Correlation |
|---------|-------------|
| error_lag1 | +0.17 |
| error_lag24 | +0.10 |
| load_trend_lag24 | -0.12 |
| error_same_hour_7d | +0.08 |

Much weaker correlations than lag-1h!

## Model Results (24h Horizon)

| Model | MAE | vs Baseline |
|-------|-----|-------------|
| Naive (predict 0) | 63.4 MW | - |
| Persistence (lag-24) | 87.5 MW | **-38% worse** |
| Same Hour Yesterday | 87.6 MW | **-38% worse** |
| GB Model | 56.5 MW | **+11%** |

**Key insight**: Simple persistence **hurts** at 24h horizon. Need ML to capture complex patterns.

## 3-Minute Data Value

| Feature | Correlation with 24h Error |
|---------|---------------------------|
| **load_trend_lag24** | **-0.123** (best 3-min feature) |
| load_std_lag24 | 0.005 (no value) |
| load_cv_lag24 | -0.006 (no value) |
| load_range_lag24 | 0.002 (no value) |

Only within-hour **trend** has predictive value. Volatility metrics don't help.

## Plots

- `19_24h_horizon_correlation.png` - Feature correlations
- `20_24h_horizon_models.png` - Model comparison
- `21_3min_value_analysis.png` - 3-minute data analysis
