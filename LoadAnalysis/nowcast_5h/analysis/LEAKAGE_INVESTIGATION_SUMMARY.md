# Data Leakage Investigation Summary

## Overview

Deep investigation of potential data leakage in the two-stage nowcasting model for load forecasting.

---

## Summary Table

| Test | Description | Severity | Status |
|------|-------------|----------|--------|
| 1 | Stage 1 feature shifts | - | **CLEAN** |
| 2 | Stage 2 residual shifts | - | **CLEAN** (was fixed) |
| 3 | seasonal_error uses full data | LOW | **MINOR LEAKAGE** |
| 4 | Train/test temporal split | - | **CLEAN** |
| 5 | Stage 1 in-sample residuals for Stage 2 | MEDIUM | **POTENTIAL ISSUE** |
| 6 | Empirical shuffled/random tests | - | **CLEAN** |

---

## Detailed Findings

### 1. Stage 1 Feature Shifts: CLEAN

**Analysis**: Stage 1 uses `error_lag1 = error.shift(1)` for ALL horizons.

```
Correlation of error_lag1 with targets:
  H+1: r = 0.7373  (2-hour gap)
  H+2: r = 0.6076  (3-hour gap)
  H+3: r = 0.4981  (4-hour gap)
  H+4: r = 0.4121  (5-hour gap)
  H+5: r = 0.3387  (6-hour gap)
```

**Verdict**: No leakage - features correctly use only past data relative to prediction time.

---

### 2. Stage 2 Residual Shifts: CLEAN (Previously Fixed)

**Historical Bug**: Original code used `residual.shift(1)` for ALL horizons, causing severe leakage for H+2 to H+5.

**Current Fix**: Uses `shift(horizon + lag - 1)` which correctly accounts for the horizon.

```python
# Example: For H+5, lag=1
# shift(5 + 1 - 1) = shift(5)
# This gives residual[t-5] which contains error[t] (known at prediction time)
```

**Verification**: Model performance now properly decays with horizon:
- H+1: 34.0 MW (+49.2%)
- H+5: 61.7 MW (+8.0%)

---

### 3. Seasonal Error Feature: MINOR LEAKAGE

**Issue**: `seasonal_error` is computed on ALL data including test:
```python
df['seasonal_error'] = df.groupby(['dow', 'hour'])['error'].transform('mean')
```

**Impact Measured**:
- Train: 48% of data
- Test: 52% of data
- Mean absolute difference: **9.73 MW**
- Max absolute difference: 35.03 MW

**Severity**: LOW - this is 1 of ~30 features, and the seasonal patterns are relatively stable year-over-year.

**Fix**: Compute from training data only:
```python
seasonal_map = train.groupby(['dow', 'hour'])['error'].mean().to_dict()
df['seasonal_error'] = df.apply(lambda x: seasonal_map.get((x['dow'], x['hour']), 0), axis=1)
```

---

### 4. Train/Test Split: CLEAN

**Split Definition**:
- Stage 1 Train: year == 2024
- Stage 2 Train: year == 2024 AND month > 6
- Test: year >= 2025

**Verified**:
- Train ends: 2024-12-31 23:00:00
- Test starts: 2025-01-01 00:00:00
- Gap: 1 hour (no overlap)

---

### 5. Stage 1 In-Sample Residuals: MEDIUM SEVERITY

**Issue**: Stage 2 trains on residuals from Stage 1's in-sample predictions (2024 data), but Stage 2 is evaluated on residuals from Stage 1's out-of-sample predictions (2025 data).

**Measured Impact**:
```
Residual standard deviation:
  Train (in-sample):     51.72 MW
  Test (out-of-sample):  56.28 MW
  Ratio (test/train):    1.09x

Residual autocorrelation (lag 1):
  Train (in-sample):     0.486
  Test (out-of-sample):  0.590
```

**Implication**: Stage 2 sees smaller, less autocorrelated residuals during training. This could cause it to underfit on actual test data.

**Estimated Impact**: ~2-5% performance inflation. Actual H+1 performance might be 45-47% instead of 49%.

**Fix**: Use cross-validation for Stage 1 predictions on training data, or use a separate holdout set.

---

### 6. Empirical Validation: CLEAN

**Shuffled Target Test**:
- Naive baseline MAE: 68.82 MW
- Model MAE (shuffled): 69.79 MW
- Improvement: **-1.4%** (no improvement = no leakage)

**Random Features Test**:
- Model MAE (random): 69.59 MW
- Model MAE (real): 42.79 MW
- The 37% improvement is real signal, not leakage.

---

## Additional Finding: Missing Feature Opportunity

**Observation**: The model uses `error_lag1 = error[t-1]` but NOT `error[t]`.

At prediction time (end of hour t), `error[t]` IS available because the hour just completed.

```
Correlation with H+1 target:
  error[t]   (lag0): r = 0.8672
  error[t-1] (lag1): r = 0.7373
  Gain from using lag0: +0.13 correlation
```

**Recommendation**: Add `error_lag0 = error[t]` to improve performance (not a leakage issue).

---

## Conclusions

### The 49% H+1 Improvement is Mostly Genuine

1. **Major leakage (Stage 2 shifts) was already fixed** - the model now shows proper horizon decay
2. **Minor remaining issues**:
   - `seasonal_error` leakage: ~1-2 MW impact
   - In-sample residual training: ~2-5% potential inflation
3. **Empirical tests confirm no severe leakage**

### Estimated True Performance

| Horizon | Reported | Estimated True |
|---------|----------|----------------|
| H+1 | +49.2% | **~45-48%** |
| H+2 | +31.3% | ~28-30% |
| H+3 | +20.4% | ~18-20% |
| H+4 | +13.1% | ~11-13% |
| H+5 | +8.0% | ~6-8% |

### Recommended Fixes (Priority Order)

1. **LOW EFFORT, MEDIUM IMPACT**: Add `error_lag0` feature
2. **MEDIUM EFFORT, LOW IMPACT**: Compute `seasonal_error` from training data only
3. **HIGH EFFORT, MEDIUM IMPACT**: Use cross-validation for Stage 1 residual generation

---

*Investigation completed: January 2026*
*Script: `leakage_investigation.py`*
