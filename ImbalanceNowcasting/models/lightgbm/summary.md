# LightGBM Nowcasting Model Summary

## Model Overview

5 separate LightGBM models trained for each lead time (12, 9, 6, 3, 0 minutes).

## Performance vs Baseline

| Lead Time | Baseline MAE | LightGBM MAE | MAE Improvement | Baseline R² | LightGBM R² | R² Improvement |
|-----------|--------------|--------------|-----------------|-------------|-------------|----------------|
| 12 min    | 6.59 MWh     | **4.68 MWh** | **-29.1%**      | 0.307       | **0.669**   | +117.9%        |
| 9 min     | 4.74 MWh     | **3.70 MWh** | **-21.8%**      | 0.631       | **0.787**   | +24.7%         |
| 6 min     | 3.41 MWh     | **2.87 MWh** | **-16.0%**      | 0.781       | **0.869**   | +11.3%         |
| 3 min     | 2.52 MWh     | **2.22 MWh** | **-11.9%**      | 0.842       | **0.912**   | +8.3%          |
| 0 min     | 2.15 MWh     | **1.71 MWh** | **-20.6%**      | 0.860       | **0.911**   | +5.9%          |

**Key Findings**:
- **12-29% MAE reduction** across all lead times
- **Largest R² gains at longer lead times** (where baseline is weakest)
- Lead 12: R² improved from 0.307 to 0.669 (+118%)

## Directional Accuracy (Sign Prediction)

| Lead Time | Accuracy |
|-----------|----------|
| 12 min    | 76.5%    |
| 9 min     | 81.8%    |
| 6 min     | 86.4%    |
| 3 min     | 89.8%    |
| 0 min     | **93.4%** |

## MAE by Imbalance Magnitude (Lead 12 min)

| Magnitude | MAE (MWh) | n samples |
|-----------|-----------|-----------|
| Small (<2 MWh) | 2.96 | 2,162 |
| Medium (2-5 MWh) | 3.49 | 2,640 |
| Large (5-10 MWh) | 4.65 | 2,382 |
| Extreme (>10 MWh) | 7.15 | 2,780 |

**Key Finding**: Large imbalances still harder to predict, but improvement consistent across all magnitudes.

## MAE by Hour Group (Lead 12 min)

| Hour Group | MAE (MWh) |
|------------|-----------|
| Night (0-5h) | 3.77 |
| Morning (6-10h) | 5.17 |
| Peak (11-14h) | 5.25 |
| Afternoon (15-20h) | 4.89 |
| Evening (21-23h) | 4.42 |

**Key Finding**: Night hours are easiest to predict; peak hours most challenging.

## Feature Importance (Lead 12 model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | baseline_pred | 13,909,043 |
| 2 | reg_cumulative_mean | 12,744,421 |
| 3 | imb_proxy_rolling_mean4 | 7,887,697 |
| 4 | imb_proxy_lag1 | 2,799,783 |
| 5 | hour_sin | 1,574,733 |
| 6 | hour_cos | 1,412,640 |
| 7 | load_deviation | 1,306,573 |
| 8 | imb_proxy_lag2 | 901,820 |
| 9 | imb_proxy_rolling_std4 | 752,264 |
| 10 | is_weekend | 182,600 |

**Key Insights**:
1. **baseline_pred** (the deterministic baseline) is the most important feature
2. **reg_cumulative_mean** adds significant value on top of baseline
3. **Lag features** (rolling mean, lag1, lag2) are highly valuable
4. **Hour encoding** and **load_deviation** contribute meaningfully
5. **reg_std/range/trend** have zero importance at lead 12 (only 1 observation)

## Features Used

### Primary
- `reg_cumulative_mean`: Mean of available regulation observations
- `baseline_pred`: Output of deterministic baseline model

### Lag Features
- `imb_proxy_lag1`: Previous period's estimated imbalance (-0.25 × reg mean)
- `imb_proxy_lag2`: Two periods ago
- `imb_proxy_rolling_mean4`: Rolling mean of last 4 periods
- `imb_proxy_rolling_std4`: Rolling std of last 4 periods

### Time Features
- `hour_sin`, `hour_cos`: Cyclical hour encoding
- `is_weekend`: Weekend indicator

### Other
- `load_deviation`: Load minus expected load at same time-of-day
- `reg_std`, `reg_range`, `reg_trend`: Within-period statistics (only useful at shorter leads)

## Model Parameters

```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'min_data_in_leaf': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}
```

Early stopping with patience=50 rounds.

## Data Split

- **Train**: 2024-01-01 to 2025-09-30 (~58K samples per lead time)
- **Test**: 2025-10-01 to 2026-01-24 (~10K samples per lead time)

## Files

- `train_lightgbm.py`: Training script
- `visualize_results.py`: Visualization script
- `lightgbm_results.csv`: Detailed results
- `feature_importance.csv`: Feature importance scores
- `lightgbm_models.pkl`: Trained models (pickle)
- `lightgbm_vs_baseline.png`: Performance comparison chart
- `lightgbm_detailed.png`: Detailed analysis charts

## Next Steps

1. **Hyperparameter tuning** with Optuna for further gains
2. **Add XGBoost/CatBoost** for ensemble
3. **Conformalized Quantile Regression** for prediction intervals
4. **Conditional blending** by hour/magnitude
