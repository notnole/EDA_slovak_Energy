# Residual Analysis

## Overview
Analysis of forecast skill after removing 2024 seasonal patterns.

## Setup
- Learn seasonal pattern from 2024 (by day_of_week, hour)
- Apply to 2025 data
- Check if baseline captures residuals

## Results

| Metric | Raw Data | Residuals | Change |
|--------|----------|-----------|--------|
| MAE | 66.7 MW | 67.3 MW | +0.9% |
| R-squared | - | **0.931** | - |

## Interpretation

1. **High R-squared (0.93)**: Baseline tracks residual **direction** well
2. **Same MAE**: But doesn't reduce **magnitude** of errors
3. The baseline captures the seasonal pattern AND follows residual trends, but the actual error magnitude remains ~67 MW

## Implication

The baseline is already accounting for seasonal patterns effectively. Improvements must come from:
- Error persistence (today's errors predict tomorrow's)
- Unusual conditions (weather, events)
- Price signals

## Plots

- `11_residual_forecast_skill.png` - Residual analysis
- `12_residual_skill_by_hour.png` - Skill by hour
