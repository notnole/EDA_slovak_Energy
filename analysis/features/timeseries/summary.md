# Feature Time Series Analysis Summary

## Purpose
Visualize the raw time series of all 4 features to understand their characteristics and data coverage.

## Data Coverage

| Feature | Rows | Date Range | Coverage |
|---------|------|------------|----------|
| Regulation | 356K | 2024-01 to 2026-01 | Full |
| Load | 348K | 2024-01 to 2026-01 | Full |
| Production | 47K | 2025-10 to 2026-01 | ~3.5 months |
| Export/Import | 38K | 2025-10 to 2026-01 | ~3.5 months |

## Key Observations

### Regulation (MW)
- Mean: ~-7 MW (slight negative bias)
- Std: ~49 MW
- High-frequency oscillation around zero
- This is the real-time balancing signal (inverse of imbalance)

### Load (MW)
- Mean: ~2950 MW
- Range: 1860-4580 MW
- Strong daily pattern (low at night, high midday)
- Clear seasonal pattern (higher in winter)

### Production (MW)
- Mean: ~3700 MW (only Oct 2025+ data)
- Follows load pattern but ~200-400 MW higher
- The excess is exported

### Export/Import (MW)
- Mean: ~363 MW (net exporter)
- Range: -650 to 1270 MW
- Inverse relationship with domestic demand

## Model Implications

1. **Regulation and Load have full coverage** - can use for all training data
2. **Production and Export/Import limited** - only useful for recent period
3. **Strong seasonality in Load** - decomposition will help
4. **Regulation is noisy** - less decomposable, but directly related to imbalance
