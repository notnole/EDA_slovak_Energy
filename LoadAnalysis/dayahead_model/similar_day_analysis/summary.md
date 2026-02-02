# Similar Day Analysis for Day-Ahead Forecasting

## Hypothesis
If DAMAS uses a statistical model, it likely makes similar errors under similar conditions.
Finding historical "analog" days could help predict tomorrow's errors.

## Key Finding: Similar Days Have DIFFERENT Errors

**The hypothesis is REJECTED.** Even near-identical days have uncorrelated errors.

## Similarity Metrics Tested

| Metric | Correlation with Error Similarity |
|--------|-----------------------------------|
| Forecast Profile | 0.0176 (very weak) |
| Price Profile | 0.0287 (very weak) |
| Day of Week Match | 0.0042 (negligible) |
| Month Similarity | 0.0409 (weak) |
| **Combined** | **0.0342** (weak) |

All correlations are statistically significant (large sample) but **practically meaningless**.

## Analog Day Prediction Results

| Method | MAE | vs Baseline |
|--------|-----|-------------|
| Baseline (zero) | 67.1 MW | - |
| Analog (simple avg) | 73.6 MW | **-9.6%** (WORSE!) |
| Analog (weighted) | 73.5 MW | **-9.6%** (WORSE!) |

**Analog prediction performs WORSE than assuming zero error!**

## Very Similar Day Pairs Analysis

Found 1,608 pairs with same DoW and forecast correlation > 0.99:
- Error correlation: mean = **0.094** (essentially random)
- Error MAE between pairs: **93.9 MW** (high!)

Even with **identical forecasts**, errors are uncorrelated.

## Error Decomposition

| Component | Variance | % of Total |
|-----------|----------|------------|
| Total Error | 7,468 | 100% |
| Systematic (dow+hour+month) | 1,347 | **18%** |
| Random (unpredictable) | 6,121 | **82%** |

**82% of error variance is unpredictable noise.**

## Key Insight: Hourly Directional Bias

While similar days don't predict error magnitude, DAMAS has consistent directional biases:

| Time Period | % Positive Errors | Interpretation |
|-------------|-------------------|----------------|
| Hour 5-6 | 31-34% | Under-forecasts morning ramp |
| Hour 20-21 | 62-67% | Over-forecasts evening peak |
| Sunday overall | 39.9% | Systematic under-forecast |
| Monday morning | 34-38% | Under-forecasts start of week |

These biases ARE captured by our improved model with hour-specific features.

## Why Doesn't Similar Day Work?

DAMAS errors are driven by **unpredictable factors** not in our data:
1. **Weather forecast errors** - temperature deviations from forecast
2. **Demand shocks** - industrial events, COVID effects, etc.
3. **Grid events** - unplanned outages, transmission constraints
4. **Random variation** - inherent unpredictability

DAMAS already does a good job capturing calendar patterns. The remaining errors are noise.

## Temporal Persistence

Same-hour lag correlations are weak:
- Lag-1day: r ~ 0.06-0.11 (very weak)
- Lag-7day: r ~ 0.02-0.07 (negligible)

Yesterday's error doesn't predict today's error well.

## Conclusion

1. **Similar day approach: NOT useful** for day-ahead forecasting
2. **Hour-specific bias: Captured** by improved model (+4% improvement)
3. **+4% is likely near the ceiling** for calendar-based features
4. **To go further**: Need weather forecasts, real-time grid data, or accept the noise

## Plots Generated
- `01_similar_day_analysis.png` - Similarity correlation analysis
- `02_deep_error_analysis.png` - Error decomposition and patterns
