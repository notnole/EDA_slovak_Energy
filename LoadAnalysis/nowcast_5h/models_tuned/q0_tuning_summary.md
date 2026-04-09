# Q0 Tuning Results
Generated: 2026-02-02 15:07
Trials per model: 160

## Summary

| Horizon | Baseline | Tuned | Improvement | Features |
|---------|----------|-------|-------------|----------|
| H+1 | 41.36 MW | 38.03 MW | 3.32 MW (8.0%) | 47 |
| H+2 | 50.79 MW | 48.32 MW | 2.47 MW (4.9%) | 38 |
| H+3 | 57.54 MW | 54.89 MW | 2.65 MW (4.6%) | 42 |
| H+4 | 61.77 MW | 59.38 MW | 2.39 MW (3.9%) | 46 |
| H+5 | 64.50 MW | 62.36 MW | 2.14 MW (3.3%) | 41 |

**Average improvement: 2.60 MW**

## Top Features by Horizon

### H+1
- forecast_diff_1h: 839
- error_lag1: 513
- load_trend_lag1: 465
- forecast_load: 412
- hour_sin: 368

### H+2
- forecast_diff_1h: 2169
- hour: 2096
- load_trend_lag1: 1366
- error_lag22: 1195
- error_lag1: 1189

### H+3
- forecast_diff_1h: 652
- error_lag1: 539
- error_lag21: 507
- hour: 503
- error_roll_std_12h: 453

### H+4
- error_lag1: 650
- forecast_diff_1h: 586
- forecast_diff_24h: 544
- error_roll_std_12h: 505
- error_lag20: 488

### H+5
- hour: 866
- forecast_diff_24h: 781
- error_lag19: 634
- error_same_hour_2d: 596
- forecast_diff_1h: 587

## Key Parameter Changes vs Baseline

**H+1:** n_estimators: 300 -> 509, learning_rate: 0.03 -> 0.064, max_depth: 8 -> 5
**H+2:** n_estimators: 300 -> 454, learning_rate: 0.03 -> 0.019, max_depth: 8 -> 11
**H+3:** n_estimators: 300 -> 204, learning_rate: 0.03 -> 0.027, max_error_lag: 8 -> 21
**H+4:** n_estimators: 300 -> 268, learning_rate: 0.03 -> 0.024, max_depth: 8 -> 6
**H+5:** n_estimators: 300 -> 525, learning_rate: 0.03 -> 0.024, max_depth: 8 -> 6

## Files Generated
- `q0_h{1-5}_tuned.joblib` - Tuned models
- `q0_tuning_results.json` - Full results with convergence
- `q0_best_params.json` - Best parameters
- `q0_feature_lists.json` - Feature lists per horizon