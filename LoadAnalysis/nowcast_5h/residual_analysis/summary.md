# Residual Autocorrelation Analysis

## Key Findings

1. **Model residuals are NOT white noise** - they contain strong autocorrelation at lag 1 (ACF = 0.54-0.80).

2. **AR correction provides significant improvement** when done correctly with shift = H (horizon).

3. **Important timing**: At time T (end of hour), we know error for that hour. For H+k model, residual at row T-k uses actual at (T-k)+k = T, which IS known.

## CORRECT Results (Shift = Horizon, No Leakage)

| Horizon | Baseline | AR-Corrected | Improvement |
|---------|----------|--------------|-------------|
| H+1 | 40.4 MW | 32.4 MW | **19.8%** |
| H+2 | 47.1 MW | 41.7 MW | **11.3%** |
| H+3 | 45.7 MW | 41.7 MW | **8.9%** |
| H+4 | 52.0 MW | 48.8 MW | **6.1%** |
| H+5 | 52.0 MW | 50.5 MW | **2.8%** |

**Pattern**: Improvement decreases with horizon because we access increasingly lagged residuals where autocorrelation is weaker.

## Evidence

### ACF Analysis

| Horizon | ACF(1) | ACF(2) | ACF(24) | Interpretation |
|---------|--------|--------|---------|----------------|
| H+1 | 0.54 | - | 0.34 | Moderate persistence |
| H+2 | 0.70 | 0.36 | 0.17 | High persistence |
| H+3 | 0.76 | 0.52 | 0.08 | Very high persistence |
| H+4 | 0.79 | 0.58 | 0.07 | Very high persistence |
| H+5 | 0.80 | 0.61 | 0.05 | Very high persistence |

**Key pattern**: If the model over-predicted by X MW last hour, it will likely over-predict by ~0.7X MW this hour.

### Ljung-Box White Noise Test

All horizons strongly reject the white noise hypothesis (p < 0.001).

## How the Shift Works

### Timing Convention
- Row timestamp T = end of hour (T-1 to T)
- At time T, we know error for hour just ended
- For H+k model, actual_error at row T = error at T+k

### Correct Shift = H (horizon)

At time T, for H+k model:
- Residual at row T-k uses actual_error at (T-k)+k = T
- Error at T is KNOWN (hour just ended)
- So shift = k is correct!

| Horizon | Shift | Uses Residual At | Which Requires Actual At |
|---------|-------|------------------|--------------------------|
| H+1 | 1 | T-1 | T (known) |
| H+2 | 2 | T-2 | T (known) |
| H+3 | 3 | T-3 | T (known) |
| H+4 | 4 | T-4 | T (known) |
| H+5 | 5 | T-5 | T (known) |

### Why Improvement Decreases with Horizon

| Horizon | Usable Lag | ACF at that Lag | AR Improvement |
|---------|------------|-----------------|----------------|
| H+1 | 1 | ~0.54 | 19.8% |
| H+2 | 2 | ~0.36 | 11.3% |
| H+3 | 3 | ~0.30 | 8.9% |
| H+4 | 4 | ~0.21 | 6.1% |
| H+5 | 5 | ~0.10 | 2.8% |

H+1 can use lag-1 residual (high autocorrelation), H+5 must use lag-5 (weak autocorrelation).

## Previous Errors in Analysis

### Error 1: Original AR Analysis (shift = 1 for all)
```python
# WRONG: Used lag-1 residual for ALL horizons
past_resids = test_resid[i-p:i]
```
This gave 25-45% "improvement" but was leakage - H+5 was using residual that requires knowing T+4 actual.

### Error 2: Overcorrected (shift = H+1)
```python
# WRONG: Too conservative
past_resids = [test_resid[i - (horizon + 1) - k] for k in range(p)]
```
This gave only 0-3% improvement because it skipped one extra row unnecessarily.

### Correct Implementation (shift = H)
```python
# CORRECT: Shift by exactly the horizon
past_resids = [test_resid[i - horizon - k] for k in range(p)]
```
At time T, residual at T-H uses actual at (T-H)+H = T, which IS known.

## Recommendation

**Implement AR correction for production**, especially for shorter horizons:

| Horizon | Improvement | Recommendation |
|---------|-------------|----------------|
| H+1 | 19.8% | **Highly recommended** |
| H+2 | 11.3% | **Recommended** |
| H+3 | 8.9% | Recommended |
| H+4 | 6.1% | Optional |
| H+5 | 2.8% | Marginal benefit |

AR(2) captures most benefit with minimal complexity.

## Files

- `01_acf_pacf_all_horizons.png` - ACF/PACF plots showing autocorrelation structure
- `02_correction_comparison.png` - Bar chart comparing correction methods
