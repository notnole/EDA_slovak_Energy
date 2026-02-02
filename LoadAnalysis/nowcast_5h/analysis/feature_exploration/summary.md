# Feature Exploration

## Purpose
Explore all available data sources and features for predicting DAMAS errors.

## Data Sources Evaluated

| Source | Records | Coverage | Value Added |
|--------|---------|----------|-------------|
| DAMAS Load (hourly) | 17,548 | 100% | Core (baseline) |
| 3-min Load | 348,217 | 97.6% | **High** (volatility signals) |
| 3-min Regulation | 356,031 | 98.7% | **High** (external signal) |
| 3-min Production | 47,382 | 43.4% | Low (too sparse) |
| 3-min Export/Import | 37,938 | 43.4% | Low (too sparse) |
| DA Prices | 24,843 | 100% | Marginal |

## Optimal Feature Set

### Tier 1: Error Lags (8 features)
- error_lag1 through error_lag8
- **Importance**: ~60% of total signal

### Tier 2: Rolling Statistics (8 features)
- error_roll_mean/std for 3h, 6h, 12h, 24h windows
- **Importance**: ~15% of total signal

### Tier 3: 3-Minute Features (3 features)
- load_volatility_lag1, load_trend_lag1
- **Importance**: ~10% of total signal

### Tier 4: Regulation (4 features)
- reg_mean_lag1-3, reg_std_lag1
- **Importance**: ~5% of total signal

### Tier 5: Time/Seasonal (6 features)
- hour, dow, is_weekend, hour_sin/cos, seasonal_error
- **Importance**: ~10% of total signal

## Files
- `explore_all_features.py` - Comprehensive feature exploration
- `hour_specific_test.py` - Hour-specific model tests
- `test_error_trends.py` - Error trend analysis
