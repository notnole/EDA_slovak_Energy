# Day-Ahead Model Diagnosis: Why Only +2.9% Improvement?

## The Problem
Day-ahead model achieved only **+2.9% MAE improvement** despite using similar feature concepts as the 5-hour model which achieved **+53.3% improvement** at T+1.

## Key Finding: Loss of Granularity

### Day-Ahead Model (wrong approach)
```python
# d1_load_vs_seasonal: SINGLE VALUE for entire day
today_load_vs_seasonal = today['actual_load_mw'].mean() - today_seasonal_load
```
This collapses **24 hourly deviations into 1 number**, losing critical granularity.

### 5-Hour Model (correct approach)
```python
# actual_vs_seasonal: HOUR-BY-HOUR deviation
df['actual_vs_seasonal'] = df['actual_load_mw'] - df['seasonal_load']
```
This preserves hourly patterns which the 5h model shows are crucial (top-3 feature).

## Comparison of Feature Engineering

| Feature Concept | Day-Ahead Model | 5-Hour Model | Issue |
|-----------------|-----------------|--------------|-------|
| Load vs Seasonal | Daily average | Hour-by-hour | Lost granularity |
| Same-hour error | Single value (d1_same_hour_error) | Lag-based (error_lag1..5) | Limited context |
| Seasonal error | Static lookup | Dynamic lookup per horizon | Less flexible |
| Error persistence | Not used | Rolling mean/std/min/max | Missing key signal |

## Why This Matters

The 5-hour model's feature importance shows:
```
T+1 Top Features:
1. forecast_vs_seasonal_h1: 0.251  ← Day-ahead HAS this
2. actual_vs_seasonal:      0.195  ← Day-ahead DOESN'T have this properly
3. seasonal_error_h1:       0.108  ← Day-ahead has simplified version
```

**The `actual_vs_seasonal` feature is rank #2**, and the day-ahead model's `d1_load_vs_seasonal` is a poor approximation because:
1. It's a daily average (loses 24x granularity)
2. It doesn't account for hour-specific deviations
3. Pattern: "Today hour 10 was +100 MW above seasonal" → useful for predicting tomorrow hour 10
   But: "Today's average was +50 MW" → too aggregated to help individual hours

## The Fix: Use Hour-by-Hour Yesterday's Deviation

For day-ahead prediction of hour H on day D+1, we should use:
```python
# For each target hour, get yesterday's same-hour deviation
d1_hour_H_vs_seasonal = yesterday_hour_H_actual - seasonal_load_for_hour_H
```

This gives 24 features (one per hour) instead of 1 aggregated feature.

## Additional Issues

### 1. Missing Error Persistence Features
Day-ahead model lacks:
- `error_rolling_mean`: Error persistence is strong (r=0.86 at lag-1h)
- `error_rolling_std`: Volatility context
- `error_trend`: Direction of recent errors

### 2. Model Configuration
- Day-ahead: GradientBoosting, 100 estimators, max_depth=4
- 5-hour: LightGBM, 200 estimators, max_depth=6

LightGBM typically performs better for tabular data, and the hyperparameters are more aggressive.

### 3. Test Period Difference
- Day-ahead: Split at 2025-07-01, tests on ~6 months
- 5-hour: Split at year 2025, tests on full 2025

## Recommendations for Improved Day-Ahead Model

### Feature Engineering Changes
1. **Hour-specific `actual_vs_seasonal`**: Use yesterday's hour H deviation for predicting hour H
2. **Error lag features**: Include yesterday's errors by hour (24 features)
3. **Rolling statistics**: Yesterday's error rolling mean/std/trend by time of day

### Model Changes
1. Switch to LightGBM
2. Increase estimators to 200+
3. Increase max_depth to 6

### Expected Improvement
Based on the 5-hour model's performance decay from T+1 (+53%) to T+5 (+12%), and extrapolating:
- T+24 (day-ahead): Reasonable to expect **+8-12% improvement** if properly engineered
- Current +2.9% is leaving significant value on the table

## Actual Results After Fix

| Model | MAE | Improvement | CV Mean |
|-------|-----|-------------|---------|
| Original | 62.4 MW | +2.9% | +3.4% |
| Improved | 60.9 MW | +4.0% | +5.5% |

**Cross-validation variability**: Folds ranged from +1.7% to +11.1%, showing signal exists but is noisy.

## Why Only +4% Instead of +10%?

The fundamental limitation is **error persistence decay over time**:

| Lead Time | Error Correlation | Model Improvement |
|-----------|-------------------|-------------------|
| T+1h | r = 0.86 | +53.3% |
| T+5h | r = 0.35 | +12.0% |
| T+24h | r ≈ 0.15 | +4.0% |

At 24-hour lead time, yesterday's errors have much weaker correlation with tomorrow's errors compared to the 5-hour window where persistence is strong.

## What the 5-Hour Model Has That Day-Ahead Can't Use

1. **`actual_vs_seasonal` (current state)**: At T+1, you know the current hour's deviation. At day-ahead, you only know yesterday.

2. **Strong lag features**: `error_lag1` (r=0.86) is extremely predictive at short horizons but unusable at 24h.

3. **Rolling statistics of recent errors**: 5-hour window captures the current error "regime", but 24h ago is a different regime.

## Conclusion

1. **The fix helped**: +4.0% is better than +2.9%
2. **The original diagnosis was correct**: Hour-specific features > daily aggregates
3. **But 10% is unrealistic at 24h**: Error persistence fundamentally limits day-ahead improvement
4. **The real value is in shorter horizons**: 5-hour nowcasting achieves +53% to +12% where signal is strong

The DAMAS day-ahead forecast is already quite good (it incorporates weather, calendar, etc.). The remaining errors at 24h horizon are largely unpredictable noise, not systematic patterns we can learn.

## Practical Recommendation

Focus efforts on **short-horizon nowcasting** (1-5h) where improvements are substantial (+12-53%), rather than day-ahead where ceiling is ~5-10%.
