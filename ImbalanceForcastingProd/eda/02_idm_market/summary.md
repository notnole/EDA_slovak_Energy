# IDM Execution Environment

## IDM Mid vs Settlement Price

The IDM mid price at T-120min is a noisy but improving predictor of the settlement price.

### Correlation by year:
| Year | Pearson r | R-squared |
|------|-----------|-----------|
| 2024 | 0.375 | 0.141 |
| 2025 | 0.422 | 0.178 |
| 2026 | 0.494 | 0.244 |

The R-squared has nearly doubled from 2024 to 2026, meaning the IDM market is becoming a **better predictor** of settlement outcomes. This is consistent with market maturation -- as participants learn the 15-min settlement system, IDM prices converge faster to fundamentals.

The scatter plot shows significant dispersion around the diagonal, confirming substantial residual spread even at T-120min. This residual spread is the trading opportunity.

## Bid-Ask Spread

- **Median BA spread at T-120min**: 2.09 EUR/MWh (tight for most periods)
- **Mean BA spread**: 28.10 EUR/MWh (heavily skewed by illiquid periods)
- Liquidity has **improved dramatically** over time -- the fraction of periods with BA < 5 has increased month-over-month
- **Hourly pattern**: overnight hours (0-6) tend to have wider BA spreads, peak hours are tighter
- Execution costs are a material fraction of the spread target (median BA of ~2 vs mean |spread| of ~42-108)

## Convergence Path

The IDM mid price converges toward the settlement price as delivery approaches:

### Mean |Settlement - IDM Mid| by lead and year:
| Lead | 2024 | 2025 | 2026 |
|------|------|------|------|
| 120min | 134.31 | 64.49 | 47.71 |
| 105min | 129.07 | 63.12 | 44.90 |
| 90min | 122.45 | 59.96 | 42.32 |
| 75min | 118.86 | 59.52 | 42.54 |
| 65min | 109.36 | 57.69 | 43.13 |

Key findings:
- The convergence path has **flattened significantly** in 2026 -- the improvement from T-120 to T-65 is only ~10% (47.71 -> 43.13), compared to ~18% in 2024
- In absolute terms, the 2026 mean error at T-120 (47.71) is already lower than 2024's error at T-65 (109.36)
- The **median** convergence shows even tighter values (24.42 at T-120 in 2026), confirming that the mean is driven by tail events
- This suggests that trading at T-120min in 2026 is not significantly worse than waiting -- the information gain from later OB snapshots is modest
