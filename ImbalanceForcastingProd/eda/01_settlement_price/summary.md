# Settlement Price Deep Dive

## Distribution and Extremes

- **Overall mean**: 89.52 EUR/MWh, **std**: variable by year
- Settlement prices are **overwhelmingly positive** but with a significant negative tail

### Year-over-year:
| Year | Mean | Std | Frac Negative | Frac >200 |
|------|------|-----|--------------|-----------|
| 2024 | 89.95 | 127.11 | 42.6% | 11.7% |
| 2025 | 83.77 | 123.68 | 7.5% | 4.0% |
| 2026 | 109.38 | 282.81 | 3.9% | 5.2% |

Key observation: 2024 had an exceptionally high fraction of negative settlement prices (42.6%), which has dropped to just 3.9% in 2026. This is the primary driver of the negative spread bias compression. The 2026 std of 282.81 is elevated due to extreme positive outliers despite having fewer extremes overall.

## Hourly Patterns

- **Peak volatility hours**: 17-21 (evening ramp), consistent across all years
- **Extreme price events** (|price| > 100) are concentrated in hours 8-20
- 2024 shows distinctly different hourly patterns vs 2025/2026, reflecting the early market adjustment
- Settlement prices tend to be highest during peak demand hours (8-20) and lowest overnight

## Monthly Evolution

- Settlement volatility shows clear **seasonality**: winter months (Nov-Feb) have higher std
- The P10/P50/P90 bands show occasional months with very wide distributions (extreme spikes)
- Year-over-year comparison shows 2025 and 2026 have converged to similar volatility patterns, while 2024 was structurally different
- Monthly mean settlement prices broadly track DA prices but with significant deviations that create the trading opportunity
