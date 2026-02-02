# Price Autocorrelation

## Overview
Autocorrelation analysis of day-ahead electricity prices.

## ACF Results

| Lag | ACF | Interpretation |
|-----|-----|----------------|
| 1h | ~0.85 | Strong persistence |
| 24h | ~0.60 | Daily pattern |
| 168h | ~0.50 | Weekly pattern |

## Key Patterns
- Strong hourly persistence (similar to load)
- Clear 24-hour (daily) cycle
- Clear 168-hour (weekly) cycle

## PACF Findings
- Significant partial autocorrelation at lags 1, 24
- Suggests AR(1) and AR(24) components

## Comparison to Load
Price autocorrelation structure similar to load, both showing daily and weekly seasonality.

## Plots
- `06_price_autocorrelation.png` - ACF and PACF plots
