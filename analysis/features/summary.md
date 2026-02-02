# Feature Analysis Summary

## Overview
Comprehensive analysis of the 3-minute feature data to identify predictive signals for the imbalance nowcasting model.

## Available Features

| Feature | Coverage | Rows | Description |
|---------|----------|------|-------------|
| Regulation (MW) | 2024-2026 | 356K | Real-time balancing signal |
| Load (MW) | 2024-2026 | 348K | System demand |
| Production (MW) | Oct 2025+ | 47K | System generation |
| Export/Import (MW) | Oct 2025+ | 38K | Cross-border flows |

## Analysis Components

### 1. Time Series (`timeseries/`)
Basic visualization of all features over time.
- Regulation: High-frequency oscillation around zero
- Load: Strong daily/seasonal patterns
- Production/Export: Limited coverage (Oct 2025+)

### 2. Seasonality (`seasonality/`)
Daily, weekly, monthly patterns in features.
- Load: Very predictable (daily pattern 2400→3600 MW)
- Regulation: Mean near zero, variance peaks midday
- Weekend effects significant

### 3. Decomposition (`decomposition/`)
STL decomposition and ToD deviation analysis.
- **Load**: 96% predictable (trend+seasonal), only 4% residual
- **Regulation**: 44% residual - much noisier
- ToD deviation captures the "surprise" vs expected

### 4. Correlation (`correlation/`)
Feature-label correlation analysis.

| Feature | Correlation | Rank |
|---------|-------------|------|
| Regulation Deviation | -0.672 | 1 |
| Regulation Raw | -0.667 | 2 |
| Load Deviation | -0.116 | 3 |
| Load Raw | -0.101 | 4 |
| Production | ~0 | - |
| Export/Import | ~0 | - |

### 5. Trend Extrapolation (`trend_extrapolation/`)
Test if learned patterns generalize from 2024 to 2025-2026.
- **Result**: ToD deviation extrapolates well (+24% improvement on test)
- Safe to use as feature in production

### 6. Quarter-Hour Dynamics (`quarter_hour_dynamics/`)
How prediction improves within 15-min settlement period.
- Minute 0: r = -0.685 (R² = 47%)
- Minute 12: r = -0.895 (R² = 79%)
- Cumulative mean is best single feature

### 7. Conditional Analysis (`conditional/`)
When does regulation-imbalance correlation vary?
- **Stable**: Weekend, load level, load ramp, regulation sign
- **Variable**: Hour (best 11-14, worst 21-23), month, imbalance magnitude
- **Critical**: Small imbalances poorly predicted (r = -0.13), large well predicted (r = -0.78)

### 8. Lag Correlation (`lag_correlation/`)
Do lagged values add predictive power?
- Regulation lag1 (15 min ago): r = -0.61 (decays from -0.69)
- Adding lag1 improves R² by +2.4%
- Estimated imbalance from regulation works as proxy

### 9. Residual Analysis (`residual_analysis/`)
What remains after accounting for regulation?
- Residual std = 8.57 MWh
- Residual autocorrelation (lag1): r = 0.19
- **Key**: Adding residual_lag1 improves R² by +2.5%
- Load deviation adds only +0.3%
- Hour adds ~0% in linear model

## Key Conclusions

### Feature Ranking
1. **Regulation** - Dominant predictor (r = -0.67, R² = 45%)
2. **Regulation Deviation** - Slightly better than raw
3. **Load Deviation** - Weak but independent signal (r = -0.12)
4. **Load Raw** - Less useful than deviation
5. **Production/Export** - Not useful, can drop

### Recommended Features for Model
| Feature | Type | R² Contribution | Rationale |
|---------|------|-----------------|-----------|
| `regulation_mw` | Raw | 47.3% | Primary predictor (r = -0.67) |
| `residual_lag1` | Derived | +2.5% | Captures temporal autocorrelation |
| `load_deviation` | Derived | +0.3% | Small independent signal |
| `hour_of_day` | Time | ~0% | Conditional analysis shows variation, but linear model doesn't capture |

**Expected baseline R² ~ 0.50** with regulation + lag1

### Features NOT Recommended
- `regulation_deviation` - trend too noisy to extrapolate (44% residual)
- `load_mw` raw - deviation is better
- `production_mw` - no correlation, limited coverage
- `export_import_mw` - no correlation, limited coverage
- `is_weekend` - correlation stable across weekend/weekday
- `hour` as linear feature - doesn't help in linear model

## Expected Model Performance

With just regulation as feature:
- R² ≈ 0.45
- RMSE ≈ 9.4 MWh

Remaining ~55% variance needs:
- Time features (HoD, DoW)
- Lag features (previous imbalance values)
- Within quarter-hour dynamics
- Rolling statistics

## Next Steps

1. **Within Quarter-Hour Analysis**: How do features evolve during 15-min settlement?
2. **Lag Analysis**: Time-lagged correlations
3. **Feature Engineering**: Create candidate features
4. **Baseline Model**: Build and evaluate
