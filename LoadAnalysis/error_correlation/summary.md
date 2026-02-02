# Error Correlation Analysis

## Overview
Analysis of DAMAS forecast error patterns and predictability.

## Error Autocorrelation

| Lag | ACF | Interpretation |
|-----|-----|----------------|
| **1h** | **0.864** | Very strong persistence! |
| 2h | 0.731 | Strong |
| 6h | 0.322 | Moderate |
| 24h | 0.222 | Weak |
| 168h | 0.054 | Very weak |

**Key insight**: Errors persist strongly hour-to-hour (0.86) but NOT day-to-day (0.22).

## Error Run Statistics

| Metric | Value |
|--------|-------|
| Mean error run length | **6.4 hours** |
| Max error run length | 81 hours |
| Under-forecast frequency | 47.2% |
| Over-forecast frequency | 52.6% |

## Same-Hour Error Correlation (Day-to-Day)

| Hours | Correlation | Notes |
|-------|-------------|-------|
| Evening (19-23) | **0.32-0.42** | Most predictable errors |
| Peak (9-14) | **0.07-0.13** | Most chaotic errors |

## Lagged Error Predictability

| Model | R-squared |
|-------|-----------|
| Lag-1h error | 74.7% |
| Lag-1h + Lag-24h | 76.5% |
| RF with all lags | 78.1% |

59% potential improvement in predicting errors using lagged features.

## Plots

- `13_error_autocorrelation.png` - Error ACF
- `14_same_hour_error_correlation.png` - Same-hour patterns
- `15_hour_to_hour_error_correlation.png` - Hour correlation matrix
- `16_lagged_error_predictability.png` - Lagged error analysis
- `17_conditional_error_patterns.png` - Conditional errors
- `18_error_prediction_potential.png` - Error prediction models
