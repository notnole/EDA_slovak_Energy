# Lag Correlation Analysis Summary

## Purpose
Analyze if lagged regulation values add predictive power beyond current regulation.

## Key Results

### 1. Regulation Lag Correlation

| Lag | Minutes | Correlation |
|-----|---------|-------------|
| 0 | 0 | -0.688 |
| 1 | 15 | -0.609 |
| 2 | 30 | -0.577 |
| 3 | 45 | -0.542 |
| 4 | 60 | -0.440 |
| 8 | 120 | -0.312 |
| 12 | 180 | -0.245 |

**Correlation decay**: ~11% per 15-minute lag initially, slowing down over time.

### 2. Imbalance Autocorrelation
- Previous period's imbalance correlates with current: r ~ 0.19
- Estimated imbalance (-reg × 0.25) works similarly
- Note: Actual imbalance only available 1 day later in production

### 3. Incremental Value of Lags (R²)

| Model | R² |
|-------|-----|
| reg_mean only | 0.474 |
| + lag1 | 0.498 |
| + lag1-2 | 0.507 |
| + lag1-4 | 0.519 |
| + imb_est_lag1 | 0.505 |
| + imb_est_lag1-2 | 0.512 |

**Finding**: Adding lag1 improves R² by +2.4%. Additional lags provide diminishing returns.

## Conclusions

1. **Lag 1 is useful**: Previous regulation period adds ~2.5% R²
2. **Diminishing returns**: Lags beyond 2-4 periods add little
3. **Estimated imbalance works**: Can use -reg×0.25 as proxy for previous imbalance
4. **Correlation decays slowly**: Even 3-hour-old regulation still has r = -0.25

## Model Implications

- Include `reg_mean_lag1` as feature (adds ~2.5% R²)
- Consider `imb_estimated_lag1` = -reg_mean_lag1 × 0.25
- Beyond lag 2-4, diminishing returns
