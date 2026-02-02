# Autocorrelation Analysis

## Overview
Analysis of load autocorrelation structure and stationarity.

## Load Autocorrelation

| Lag | ACF | Interpretation |
|-----|-----|----------------|
| 1h | 0.963 | Very high persistence |
| 24h | 0.866 | Strong daily pattern |
| 48h | 0.739 | 2-day pattern |
| 168h | 0.876 | Strong weekly pattern |

## Autocorrelation by Hour (Same Hour Day-to-Day)

| Hours | ACF (lag-1 day) | Characteristic |
|-------|-----------------|----------------|
| Night (2-5) | **0.87-0.89** | Most predictable |
| Morning ramp (7-8) | **0.68-0.71** | Most chaotic |
| Peak (9-14) | 0.73-0.82 | Moderate |
| Evening (19-23) | 0.79-0.82 | Moderate |

**Key insight**: Morning ramp hours (7-8) are hardest to predict.

## Stationarity Tests

| Test | Result | Conclusion |
|------|--------|------------|
| ADF (raw) | p < 0.001 | Stationary |
| KPSS (raw) | p < 0.01 | Non-stationary |
| ADF (24h diff) | p < 0.001 | Stationary |
| KPSS (24h diff) | p > 0.05 | Stationary |

24-hour differenced series is fully stationary - good for SARIMA.

## Plots

- `04_autocorrelation.png` - ACF/PACF analysis
- `05_stationarity.png` - Rolling statistics
- `09_autocorrelation_by_hour.png` - ACF by hour of day
