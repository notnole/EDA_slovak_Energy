# 5-Hour Nowcasting Model - All Possibilities Explored

## Best Model Performance

### Two-Stage Residual Correction Model (CORRECTED - No Data Leakage)
| Horizon | Baseline MAE | Stage 1 MAE | Stage 2 MAE | Total Improvement | Stage 2 Gain |
|---------|--------------|-------------|-------------|-------------------|--------------|
| **H+1** | 67.0 MW | 41.6 MW | **34.0 MW** | **+49.2%** | +18.3% |
| H+2 | 67.0 MW | 50.3 MW | 46.1 MW | +31.3% | +8.3% |
| H+3 | 67.1 MW | 56.0 MW | 53.4 MW | +20.4% | +4.6% |
| H+4 | 67.1 MW | 60.0 MW | 58.3 MW | +13.1% | +2.8% |
| H+5 | 67.2 MW | 62.4 MW | 61.7 MW | +8.0% | +1.1% |

**Key Result**: H+1 achieves excellent +49.2% improvement. Performance properly degrades with horizon (as expected).

### CRITICAL BUG FIXED: Data Leakage in Stage 2
**Problem Found**: Original Stage 2 used `residual.shift(1)` for all horizons. For H+5, this meant using error[t+4] to predict error[t+5] - only 1 hour effective lead time!

**Fix Applied**: Stage 2 features now shift by `horizon + lag - 1` to ensure no future information leaks.

**Before Fix (LEAKED)**: Flat 34-36 MW across all horizons (suspicious!)
**After Fix (CORRECT)**: Proper decay from 34 MW (H+1) to 62 MW (H+5)

### MAPE Comparison (% of Actual Load)
| Horizon | DAMAS MAPE | Our MAPE | Reduction | Improvement |
|---------|------------|----------|-----------|-------------|
| **H+1** | 2.30% | **1.17%** | **-1.14pp** | **+49.4%** |
| H+2 | 2.30% | 1.58% | -0.72pp | +31.4% |
| H+3 | 2.30% | 1.83% | -0.47pp | +20.5% |
| H+4 | 2.31% | 2.00% | -0.30pp | +13.2% |
| H+5 | 2.31% | 2.12% | -0.19pp | +8.2% |

### Stage 1 Only (Standard Model)
| Horizon | Baseline MAE | Best Model MAE | Improvement | Direction Accuracy |
|---------|--------------|----------------|-------------|-------------------|
| **H+1** | 67.0 MW | 41.6 MW | +37.9% | 78.6% |
| H+2 | 67.0 MW | 50.3 MW | +25.0% | 73.1% |
| H+3 | 67.1 MW | 56.0 MW | +16.5% | 69.1% |
| H+4 | 67.1 MW | 60.0 MW | +10.6% | 65.9% |
| H+5 | 67.2 MW | 62.4 MW | +7.0% | 63.3% |

## Two-Stage Model Architecture

### How It Works
1. **Stage 1**: Predict DAMAS error using standard features (error lags, rolling stats, 3-min features, regulation)
2. **Compute Residual**: Actual error - Stage 1 prediction (OUR residuals, not DAMAS errors)
3. **Stage 2**: Correct using OUR residual features (lags, rolling mean/std) **properly shifted by horizon**

### Why This Works
- Stage 1 residuals have strong autocorrelation (r=0.57 at H+1)
- Using OUR residuals (not DAMAS errors) gives 2x better gain than DAMAS errors
- Stage 2 exploits short-term patterns in residuals
- **CRITICAL**: Stage 2 features must be shifted by horizon h to avoid data leakage!

### Stage 2 Feature Shifting (Corrected)
For horizon h prediction, residual features are shifted by `h + lag - 1`:
- H+1: residual_lag1 = residual.shift(1) ✓
- H+5: residual_lag1 = residual.shift(5) ✓ (not shift(1) which would leak!)

### Stage 2 Features
| Feature | Description | Shift |
|---------|-------------|-------|
| residual_lag1 | Most recent usable residual | horizon |
| residual_lag2-3 | Earlier residuals | horizon+1, horizon+2 |
| residual_roll_mean_3h/6h | Rolling mean of residuals | horizon |
| residual_roll_std_3h/6h | Rolling volatility of residuals | horizon |
| residual_trend_3h | Residual momentum | derived |

## Data Sources Tested

### Available Data
| Source | Records | Coverage | Value |
|--------|---------|----------|-------|
| DAMAS Load (hourly) | 17,548 | 100% | Core |
| 3-min Load | 348,217 | 97.6% | **High** |
| 3-min Regulation | 356,031 | 98.7% | **High** |
| 3-min Production | 47,382 | 43.4% | Low |
| 3-min Export/Import | 37,938 | 43.4% | Low |
| DA Prices | 24,843 | 100% | Marginal |

## What We Tried (That Didn't Help Further)

| Approach | Result | Why |
|----------|--------|-----|
| Stage 3 (residual of residual) | +0.2% | Residuals already noise (autocorr=0.025) |
| Systematic bias features | -6.9% | Collinear with existing features |
| Hour-specific bias correction | -0.5% | Bias patterns not stable across years |
| DOW-hour bias correction | -1.2% | Same reason |
| Error magnitude prediction | 0% | Helps predict size but not direction |
| Sign prediction + regression | 0% | Already captured by regression |
| Calendar features (holidays, DST) | -0.6% | Not enough signal |
| Adaptive dampening | +0.3% | Already captured by Stage 2 |
| Load-regulation interactions | +1.2% | Marginal |

## Final Residual Analysis

After Stage 2:
- **Autocorrelation**: 0.186 at lag 1h, 0.025 after Stage 3
- **Distribution**: Near Gaussian (skewness=0.1, kurtosis=1.8)
- **Worst hours**: 23 (47.4 MW), 8 (45.4 MW), 10 (42.3 MW)
- **Best hours**: 2 (21.0 MW), 3 (23.1 MW), 1 (25.3 MW)

## Interesting Findings

### DAMAS Systematic Failures
- 63 days with |mean error| > 100 MW
- Worst: Feb 13, 2025 with 351 MW under-forecast
- Calendar events (DST, holidays) increase errors by 10-18%
- But calendar features don't help model (patterns not stable)

### Error Magnitude is Predictable
- Absolute error autocorrelation: 0.752 at lag 1h
- Model R² = 0.373 for magnitude prediction
- But can't use this to improve signed predictions

### Sign Prediction
- Model achieves 79.2% accuracy, AUC=0.873
- Regression already captures this (78.6% direction accuracy)
- Worst at hour 23 (62.4%), best at hour 11 (84.1%)

## Optimal Feature Set (Stage 1)

### Tier 1: Core Error (8)
- error_lag1 through error_lag8

### Tier 2: Rolling Statistics (8)
- error_roll_mean/std for 3h, 6h, 12h, 24h windows

### Tier 3: Error Trends (5)
- error_trend_3h, error_trend_6h
- error_accel, error_momentum

### Tier 4: 3-Minute Features (3)
- load_volatility_lag1, load_volatility_lag2
- load_trend_lag1

### Tier 5: Regulation Features (5)
- reg_mean_lag1, reg_mean_lag2, reg_mean_lag3
- reg_std_lag1

### Tier 6: Seasonal & Time (8)
- seasonal_error
- hour, hour_sin, hour_cos
- dow, is_weekend

## Hyperparameters

```python
# Stage 1
lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=50,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
)

# Stage 2
lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=30,
    min_child_samples=20,
)
```

## Folder Structure

```
nowcast_5h/
├── README.md                          # This file
├── scripts/
│   ├── two_stage_model.py            # BEST MODEL - Two-stage residual correction
│   └── mape_comparison.py            # MAPE evaluation script
├── models/
│   ├── stage1_h1-h5.joblib           # Stage 1 models (5 horizons)
│   └── stage2_h1-h5.joblib           # Stage 2 models (5 horizons)
├── analysis/
│   ├── residual_analysis/            # Residual deep analysis
│   ├── bias_features/                # Systematic bias tests (didn't help)
│   ├── feature_exploration/          # Feature EDA and selection
│   └── tricky_days/                  # Day-by-day comparison plots
├── plots/                            # Overview performance plots
└── archive/                          # Superseded scripts and old models
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/two_stage_model.py` | **BEST MODEL** - Production two-stage model |
| `scripts/mape_comparison.py` | MAPE evaluation vs DAMAS |
| `models/stage*_h*.joblib` | Trained LightGBM models |
| `analysis/*/summary.md` | Analysis summaries per topic |

## Improvement Journey (H+1)

| Stage | H+1 MAE | Improvement | Notes |
|-------|---------|-------------|-------|
| DAMAS Baseline | 67.0 MW | - | No correction |
| + Error lags | 43.0 MW | +35.8% | Core signal |
| + Rolling stats | 42.4 MW | +36.7% | Stability |
| + 3-min features | 41.8 MW | +37.6% | Volatility |
| + Regulation | 41.6 MW | +37.9% | External signal |
| **+ Stage 2 (residual correction)** | **34.0 MW** | **+49.2%** | Key breakthrough |

## Conclusions (After Data Leakage Fix)

1. **Two-stage residual correction works for H+1**: +18.3% gain over Stage 1 alone
2. **Use OUR residuals, not DAMAS errors**: 2x better than using DAMAS errors directly
3. **Performance degrades with horizon (as expected)**:
   - H+1: 34.0 MW (+49.2%) - Excellent
   - H+3: 53.4 MW (+20.4%) - Good
   - H+5: 61.7 MW (+8.0%) - Marginal
4. **H+1 is the valuable horizon**: Best to focus prediction efforts here
5. **Stage 3 adds nothing**: Final residuals are essentially Gaussian noise
6. **Calendar/bias features don't help**: Patterns not stable across years
7. **~50% improvement at H+1 is the ceiling**: All additional signal sources exhausted

## Data Leakage Lesson Learned

**Red Flag**: If multi-horizon forecasting shows "flat" performance across horizons, ALWAYS check for data leakage in feature construction. Proper time series forecasting should show degradation with increasing horizon.

## Remaining Unexploited Signal

After extensive exploration, the only potential remaining signal sources are:
1. **External data (weather)**: Not available in current dataset
2. **Real-time DAMAS updates**: If DAMAS revises forecasts intraday
3. **Grid events**: Outages, maintenance schedules (not available)

The two-stage model with +49.2% improvement at H+1 represents the practical ceiling for this dataset.

**Note**: Longer horizons (H+3 to H+5) have diminishing returns because:
- Stage 1 features (error lags) become weaker predictors with distance
- Stage 2 cannot use recent residuals (must shift by horizon to avoid leakage)
- The predictable signal decays naturally with time
