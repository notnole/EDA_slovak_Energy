# Advanced Label Analysis Summary

## Purpose
Deep statistical analysis of the imbalance time series to understand its properties and identify optimal features for the nowcasting model.

## Key Findings

### 1. Autocorrelation Decay
- **Strong short-term memory**: ACF at lag 1 = 0.76, lag 4 = 0.60
- **50% of predictive info** captured in first 7 lags (1.75 hours)
- **90% of info** in first 94 lags (~24 hours)
- **PACF shows lag 1 dominant** (0.76), lag 2 much smaller (0.17)

**Model implication**: Use lag 1-4 features (1 hour history). Beyond lag 4, diminishing returns.

### 2. Extreme Events Analysis
- **Negative extreme threshold (P1)**: -33 MWh
- **Positive extreme threshold (P99)**: +35 MWh
- **Extreme events cluster**: Mean run length = 2.5 periods (37 min)
- **Longest extreme run**: 12 hours

**Model implication**: When entering extreme territory, momentum is likely. Consider binary "extreme_flag" feature.

### 3. Persistence / Runs
- **P(positive | positive)** = 78%
- **P(negative | negative)** = 72.5%
- **Mean positive run**: 4.5 periods (68 min)
- **Mean negative run**: 3.6 periods (55 min)

**Model implication**: Sign of previous period is highly predictive. Include "previous_sign" as binary feature.

### 4. Conditional Distributions
- **Mean reversion**: Slope = 0.76 < 1 (not unit root)
- **Conditional volatility varies 2.4x** between calm and volatile periods

**Model implication**: Mean reversion supports using lagged values. Prediction intervals should adapt to recent volatility.

### 5. Stationarity Tests
- **ADF test**: Stationary (p < 0.001)
- **KPSS test**: Non-stationary trend (p = 0.01)
- **Conclusion**: Level stationary with possible deterministic trend

**Model implication**: No differencing needed. Series can be modeled directly without transformation.

### 6. ARCH Effects
- **Strong volatility clustering** confirmed (LM test p < 0.001)
- **Squared residuals ACF at lag 1**: 0.55
- **96 lags significant** in squared residuals

**Model implication**: Rolling standard deviation is a valuable feature. Consider GARCH-type models for prediction intervals.

## Feature Recommendations

| Feature | Rationale | Priority |
|---------|-----------|----------|
| `lag_1` to `lag_4` | Captures 60% of autocorrelation | HIGH |
| `previous_sign` | 75% persistence probability | HIGH |
| `rolling_std_4` | ARCH effects present | HIGH |
| `extreme_flag` | Extremes cluster (2.5 period runs) | MEDIUM |
| `run_length` | How long current sign persisted | MEDIUM |

## Model Architecture Implications

1. **Autoregressive base**: Strong lag-1 dependence supports AR-type models
2. **Feature-based enhancement**: Time features (HoD, DoW) + lag features should capture most predictable variance
3. **Uncertainty quantification**: Volatility clustering means confidence intervals should widen during high-volatility periods
4. **No differencing**: Series is stationary, use raw values

## Connection to Previous Analyses

- **Basic stats**: Daily/weekly patterns → HoD, DoW features
- **Year comparison**: No structural break → use all data for training
- **Decomposition**: 43% predictable from trend+seasonal → confirms feature value
- **Advanced (this)**: Strong autocorrelation → lag features; ARCH → rolling volatility

## Recommended Next Steps

1. Proceed to **feature analysis** (3-minute data exploration)
2. Within quarter-hour dynamics (deferred from label analysis)
3. Build baseline model with: lag_1-4, HoD, DoW, rolling_std
4. Target: RMSE < 8 MWh (residual std from decomposition)
