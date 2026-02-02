# Label Analysis Summary

## Overview
Comprehensive analysis of the 15-minute System Imbalance (MWh) data to understand the prediction target for the nowcasting model.

**Data**: 72,375 observations from 2024-01-01 to 2026-01-24

## Analysis Components

### 1. Basic Stats (`basic_stats/`)
**Purpose**: Understand daily, weekly, and hourly seasonality patterns

**Key findings**:
- Daily pattern: Higher volatility 7-10 AM and 5-8 PM (demand ramps)
- Weekly pattern: Lower on weekends (less industrial load)
- Monthly pattern: Winter months have larger swings
- Distribution: Mean near zero (1.6 MWh), std = 12.7 MWh

### 2. Year Comparison (`year_comparison/`)
**Purpose**: Determine if 2024 data can be used for training (structural breaks?)

**Key findings**:
- 2024 vs 2025-2026: No significant structural break
- ACF correlation = 0.998 between periods
- Volatility clustering patterns identical
- **Conclusion**: Safe to use all data for training

### 3. Decomposition (`decomposition/`)
**Purpose**: Quantify predictable vs unpredictable variance

**Key findings**:
- STL: Daily pattern (96 periods) explains significant variance
- MSTL: Added weekly pattern (672 periods)
- **Predictable (trend + seasonal)**: ~43%
- **Residual (needs ML)**: ~57%
- Spectral peaks at 96 (daily) and 672 (weekly)

### 4. Advanced Analysis (`advanced/`)
**Purpose**: Deep statistical properties for feature engineering

**Key findings**:
- **Autocorrelation**: Lag 1 = 0.76, diminishes after lag 4
- **Persistence**: 78% probability same sign continues
- **ARCH effects**: Strong volatility clustering
- **Stationarity**: Series is stationary (no differencing)
- **Extreme events**: Cluster in runs of ~2.5 periods

## Consolidated Feature Recommendations

| Feature | Source Analysis | Priority |
|---------|-----------------|----------|
| `hour_of_day` | Basic stats | HIGH |
| `day_of_week` | Basic stats | HIGH |
| `lag_1` to `lag_4` | Advanced | HIGH |
| `rolling_std_4` | Advanced + Decomposition | HIGH |
| `previous_sign` | Advanced (persistence) | MEDIUM |
| `month` | Basic stats | LOW |
| `is_weekend` | Basic stats | LOW |

## Model Architecture Guidance

1. **All data usable**: No structural break between 2024 and 2025-2026
2. **No differencing**: Series is stationary
3. **Target baseline**: Residual std ~8 MWh (after decomposition)
4. **Confidence intervals**: Should vary with recent volatility (ARCH)
5. **Key predictors**: Time features + lag features + rolling volatility

## Next Steps

1. **Feature Analysis**: Explore 3-minute input data
2. **Within Quarter-Hour Dynamics**: How does imbalance evolve within each 15-min period?
3. **Feature Engineering**: Create candidate features from 3-min data
4. **Baseline Model**: Build simple model with recommended features
5. **Evaluation**: Target RMSE < 8 MWh
