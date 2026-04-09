# Q0 Load Nowcasting - Production Models

## Overview

5-hour ahead load error prediction models for Slovak electricity grid.
Predicts the error between DAMAS day-ahead forecast and actual load.

**Target**: `error = actual_load - forecast_load` (MW)

## Model Performance

| Horizon | MAE (MW) | vs Baseline | Features |
|---------|----------|-------------|----------|
| H+1 | 38.03 | -3.32 MW (8.0%) | 47 |
| H+2 | 48.32 | -2.47 MW (4.9%) | 38 |
| H+3 | 54.89 | -2.65 MW (4.6%) | 42 |
| H+4 | 59.38 | -2.39 MW (3.9%) | 46 |
| H+5 | 62.36 | -2.14 MW (3.3%) | 41 |

**Average improvement: 2.60 MW over baseline**

## Baseline Comparison

| Source | Avg MAE (H1-H5) | Notes |
|--------|-----------------|-------|
| Default params | 55.07 MW | LightGBM with basic features |
| **Tuned models** | **52.60 MW** | 160-trial Optuna optimization |
| Two-stage (failed) | 54.11 MW | Data leakage when tested properly |

## Training Details

- **Algorithm**: LightGBM (gradient boosting)
- **Optimization**: Optuna TPE sampler, 160 trials per horizon
- **Cross-validation**: 4-fold walk-forward
  - Fold 1: Train 2024-Q1-Q3, Val Q4_2024
  - Fold 2: Train 2024, Val H1_2025
  - Fold 3: Train 2024 + H1_2025, Val H2_2025
  - Fold 4: Train 2024 + 2025, Val Jan_2026

## Key Hyperparameters (per horizon)

| Param | H+1 | H+2 | H+3 | H+4 | H+5 |
|-------|-----|-----|-----|-----|-----|
| n_estimators | 509 | 454 | 204 | 268 | 525 |
| learning_rate | 0.064 | 0.019 | 0.027 | 0.024 | 0.024 |
| max_depth | 5 | 11 | 8 | 6 | 6 |
| max_error_lag | 24 | 22 | 21 | 24 | 22 |
| finetune_weight | 2.78 | 1.43 | 2.19 | 4.45 | 5.46 |
| finetune_days | 88 | 128 | 150 | 70 | 52 |

## Top Features by Importance

### H+1
1. forecast_diff_1h (839)
2. error_lag1 (513)
3. load_trend_lag1 (465)
4. forecast_load (412)
5. hour_sin (368)

### H+2
1. forecast_diff_1h (2169)
2. hour (2096)
3. load_trend_lag1 (1366)
4. error_lag22 (1195)
5. error_lag1 (1189)

### H+3
1. forecast_diff_1h (652)
2. error_lag1 (539)
3. error_lag21 (507)
4. hour (503)
5. error_roll_std_12h (453)

### H+4
1. error_lag1 (650)
2. forecast_diff_1h (586)
3. forecast_diff_24h (544)
4. error_roll_std_12h (505)
5. error_lag20 (488)

### H+5
1. hour (866)
2. forecast_diff_24h (781)
3. error_lag19 (634)
4. error_same_hour_2d (596)
5. forecast_diff_1h (587)

## Files

```
models_tuned/
  q0_h{1-5}_tuned.joblib      # Production models
  q0_best_params.json          # Hyperparameters
  q0_feature_lists.json        # Feature lists per horizon
  q0_h{1-5}_trial_history.csv  # Optuna trial history
  q0_tuning_results.json       # Full results with convergence
  q0_tuning_summary.md         # Original tuning summary
```

## Usage

```python
import joblib
import pandas as pd

# Load model bundle
bundle = joblib.load('models_tuned/q0_h1_tuned.joblib')
model = bundle['model']
features = bundle['features']
params = bundle['params']

# Make prediction
X = df[features]
prediction = model.predict(X)
```

## Two-Stage Model Investigation

A two-stage residual correction model was investigated but found to provide
no benefit when trained properly:

- Stage 1 predicts the error
- Stage 2 corrects Stage 1's residuals

**Result**: When Stage 2 was trained on out-of-sample Stage 1 predictions
(to prevent data leakage), it made predictions **worse** by an average of
1.24 MW. The apparent improvement in earlier tests was due to data leakage.

See `two_stage_experiment_failed/q0_two_stage_summary.md` for details.

## Recommendations

1. **Use single-stage models** (this folder) for production
2. **Retrain periodically** as new data becomes available
3. **Monitor for drift** - especially after major grid changes
