# Spread Target Analysis

The spread target is `imb_settlement_price - exec_mid` (hourly-smoothed), representing the profit opportunity from predicting the direction of imbalance settlement price relative to the IDM mid price at T-120min.

## Distribution

- **Mean**: -34.87 EUR/MWh (negative bias -- settlement price tends to be below IDM mid)
- **Median**: close to zero, indicating the mean is driven by extreme negative events
- **Std**: 260.91 EUR/MWh -- extremely high variance
- **Skewness**: -8.40 (extreme left skew -- large negative outliers dominate)
- **Kurtosis**: 221.77 (massive fat tails)
- **P5/P95**: -154.14 / +87.76 EUR/MWh (asymmetric: negative tail is ~2x the positive tail)
- **Frac |spread| > 10**: ~80-90% of periods (spreads are typically large in absolute terms)

### Year-over-year trend:
| Year | Mean Spread | Mean |Spread| | Std | Frac >10 |
|------|-----------|-------------|------|----------|
| 2024 | -47.43 | 107.61 | 345.01 | 89.7% |
| 2025 | -30.72 | 55.92 | 165.20 | 82.7% |
| 2026 | -8.38 | 41.80 | 178.22 | 75.1% |

Clear compression: mean |spread| dropped from 108 (2024) to 56 (2025) to 42 (2026). The negative bias also attenuated from -47 to -8 EUR/MWh, suggesting the settlement price has moved closer to IDM mid over time.

## Seasonality

### Hourly pattern
The widest spreads occur during **evening peak hours 17-21**, with Hour 19 at 146 EUR/MWh mean |spread|. The narrowest spreads are during **off-peak hours 2-5** (~43-52 EUR/MWh). This pattern holds across all years but is most pronounced in 2024.

### Weekly pattern
Weekend spreads are slightly narrower than weekday spreads. The pattern is not dramatic.

### Monthly pattern
Winter months (Nov-Feb) tend to have wider spreads than summer months, consistent with higher settlement price volatility during heating season and volatile renewable output.

## Autocorrelation

- **ACF lag 1h**: 0.217 -- moderate short-term persistence (one hour predicts the next)
- **ACF lag 4h**: 0.056 -- persistence drops quickly
- **ACF lag 24h**: 0.071 -- mild daily pattern

The spread is only weakly autocorrelated, meaning past spread values have limited predictive power for future spreads. The rolling 30-day autocorrelation shows the lag-1 persistence varies considerably over time but generally stays in the 0.1-0.3 range.

## Regime Change

The data shows a **clear structural compression** of spreads over time:
- 2024 had very wide spreads (mean |spread| ~108), especially in the early months when Slovakia was still adjusting to the 15-min settlement period regime
- 2025 saw roughly 50% compression (mean |spread| ~56)
- 2026 shows further compression (mean |spread| ~42)

This compression is **both seasonal and structural**:
- **Structural component**: the year-over-year overlay shows 2026 running below 2025 at matching calendar days
- **Seasonal component**: summer months consistently have narrower spreads than winter

The rolling 30-day mean |spread| shows the trend clearly, with only occasional spikes above the declining trend line.
