# Residual Analysis Summary

## Purpose
After accounting for regulation, what patterns remain in the residuals? Is there exploitable structure?

## Simple Model
```
imbalance = -0.186 × regulation + 0.369
R² = 0.473
```

## Residual Characteristics

| Metric | Value |
|--------|-------|
| Mean | 0.00 MWh |
| Std | 8.57 MWh |
| IQR | [-4.1, 4.1] MWh |
| Range | [-209, 83] MWh |
| Skewness | -0.56 (slight left skew) |
| Kurtosis | 16.2 (heavy tails) |

**Not normally distributed** - heavy tails mean occasional large errors.

## Residual Patterns

### By Hour (significant deviations)
| Hour | Mean Residual |
|------|---------------|
| 7 | +1.21 MWh |
| 13 | +1.38 MWh |
| 22 | -1.12 MWh |

Model slightly underpredicts at hours 7 and 13, overpredicts at hour 22.

### By Day of Week
No significant pattern (all within ±0.4 MWh).

## Residual Correlations

| Feature | Correlation |
|---------|-------------|
| **residual_lag1** | **+0.192** |
| load_deviation | -0.054 |
| hour_sin | +0.006 |
| hour_cos | -0.026 |
| residual_lag2 | +0.005 |

**Key finding**: Residuals show temporal autocorrelation (r = 0.19 at lag 1).

## Can Other Features Help?

| Model | R² | MAE | Improvement |
|-------|-----|-----|-------------|
| regulation only | 0.473 | 5.82 | baseline |
| + load_deviation | 0.476 | 5.83 | +0.3% |
| + hour (sin/cos) | 0.473 | 5.82 | ~0% |
| + load_dev + hour | 0.476 | 5.83 | +0.4% |
| **+ residual_lag1** | **0.498** | **5.79** | **+2.5%** |
| full model | 0.501 | 5.79 | +2.8% |

## Conclusions

1. **Residual lag1 is the most valuable addition** (+2.5% R²)
   - Residuals are autocorrelated - errors persist
   - In production: use estimated residual from previous period

2. **Load deviation adds little** (+0.3% R²)
   - Statistically significant but practically small

3. **Hour adds nothing** (~0%)
   - Surprising given conditional analysis showed hour variation
   - Linear model may not capture the pattern

4. **Heavy-tailed errors**
   - Kurtosis = 16 means occasional large errors
   - Model will sometimes be very wrong

## Model Implications

### Recommended features:
1. `regulation_mw` - primary predictor
2. `residual_lag1` or `imb_estimated_lag1` - captures temporal autocorrelation
3. `load_deviation` - small but independent signal

### Not recommended:
- Hour as linear feature (no improvement)
- Lags beyond 1-2 (diminishing returns)

### Expected performance:
- R² ~ 0.50 with regulation + lag1
- MAE ~ 5.8 MWh
- Occasional large errors due to heavy tails
