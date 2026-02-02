# Day-Ahead Model Development

## Overview
Development and optimization of the day-ahead load forecast error correction model.

## Setup
At 23:00 on day D, predicting all 24 hours of day D+1 using:
- Today's (D) complete 24 hours of forecast errors
- Tomorrow's (D+1) DAMAS forecast
- 2024 seasonal pattern

## Model Evolution

| Version | Features | MAE | vs Baseline |
|---------|----------|-----|-------------|
| v1 (simple) | Seasonal only | 65.2 MW | -2.8% |
| v2 (today's errors) | + today's complete errors | 60.9 MW | +4.0% |
| Enhanced | + momentum, price, week | 64.8 MW | -0.8% |
| Final (optimized) | Core features only | 62.4 MW | +2.9% |
| **Improved (hour-specific)** | Hour-by-hour features | **60.9 MW** | **+4.0%** |

## Diagnosis: Why Original Only +2.9%?

**Root cause**: Original model used `d1_load_vs_seasonal` as a DAILY AVERAGE, losing 24x granularity.

**The fix**: Use hour-specific features (`d1_same_hour_vs_seasonal`, `d1_same_hour_error`)

See `diagnosis.md` for full analysis.

## Why Not 10% Improvement?

Error persistence decays with time:
- T+1h: r=0.86 → +53% improvement
- T+5h: r=0.35 → +12% improvement
- T+24h: r≈0.15 → +4% improvement (ceiling)

**Conclusion**: DAMAS day-ahead is already good. Short-horizon nowcasting (1-5h) offers much higher returns.

## Key Finding: Simpler is Better

Adding more features **hurt** performance:

| Feature Group | Impact |
|--------------|--------|
| Core features | +2.9% |
| + Momentum | -3.8% |
| + Price | -3.1% |
| + Week stats | -1.4% |

## Final Model Features (by importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | d1_evening_error | 0.186 |
| 2 | d1_load_vs_seasonal | 0.184 |
| 3 | forecast_vs_seasonal | 0.166 |
| 4 | seasonal_error | 0.105 |
| 5 | d1_same_hour_error | 0.074 |

## Cross-Validation

| Fold | Improvement |
|------|-------------|
| 1 | +6.5% |
| 2 | +1.5% |
| 3 | +5.8% |
| 4 | +5.4% |
| 5 | -2.0% |
| **Mean** | **+3.4% +/- 3.2%** |

## What Didn't Help

- **Price features**: DAMAS already uses this info
- **Momentum/trend**: Adds noise, overfits
- **Week-long statistics**: Too much noise
- **Lagged errors (d2-d7)**: Day-to-day correlation weak

## Plots

- `22_simple_dayahead_model.png` - v1 results
- `23_dayahead_model_v2.png` - v2 results (best version)
- `24_price_features_analysis.png` - Price feature analysis
- `25_enhanced_model.png` - Enhanced model (overfit)
- `26_optimized_model.png` - Incremental testing
- `27_final_model.png` - Final optimized model

## Scripts

- `simple_dayahead_model.py` - v1 implementation
- `dayahead_model_v2.py` - v2 with today's complete errors
- `price_features_analysis.py` - Price feature evaluation
- `enhanced_features_model.py` - Many features (overfit)
- `optimized_model.py` - Incremental feature testing
- `final_dayahead_model.py` - Clean final implementation
