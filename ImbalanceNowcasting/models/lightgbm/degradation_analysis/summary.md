# Model Degradation Analysis

**Experiment**: Train on 2024 only, test on full 2025 to detect performance drift.

## Key Finding

**Seasonal pattern, not monotonic drift**: Performance degrades in winter months (Jan-Feb) but recovers in summer/fall. This suggests seasonal retraining rather than continuous drift.

## Training Setup

- **Training period**: 2024-01-01 to 2024-12-31 (34,344 samples)
- **Test period**: 2025-01-01 to 2026-01-24 (36,925 samples, 384 days)
- **Model**: LightGBM v7 (Lead 12 min - hardest case)
- **Training baseline MAE**: 4.43 MWh

## Monthly Performance

| Month | MAE (MWh) | vs Baseline | Status |
|-------|-----------|-------------|--------|
| 2025-01 | 5.37 | +21.3% | RETRAIN |
| 2025-02 | 5.74 | +29.6% | RETRAIN |
| 2025-03 | 5.12 | +15.7% | MONITOR |
| 2025-04 | 5.68 | +28.3% | RETRAIN |
| 2025-05 | 4.69 | +6.0% | OK |
| 2025-06 | 5.13 | +16.0% | MONITOR |
| 2025-07 | 5.10 | +15.1% | MONITOR |
| 2025-08 | 4.54 | +2.5% | OK |
| 2025-09 | 5.03 | +13.6% | MONITOR |
| 2025-10 | 4.90 | +10.6% | MONITOR |
| 2025-11 | 4.28 | **-3.3%** | OK |
| 2025-12 | 4.17 | **-5.8%** | OK |
| 2026-01 | 5.85 | +32.1% | RETRAIN |

## Key Observations

### 1. Winter Problem
- Jan-Feb and Jan 2026 show 20-32% degradation
- Likely due to: heating load volatility, holiday patterns, weather extremes
- 2024 training data doesn't capture 2025 winter patterns well

### 2. Summer/Fall Recovery
- May, Aug, Nov, Dec perform at or below baseline
- Nov-Dec 2025 actually **better** than training performance
- Suggests annual cycle in imbalance patterns

### 3. Not Monotonic Drift
- If this were concept drift, we'd expect continuous degradation
- Instead: seasonal oscillation around baseline
- Model captures fundamental relationships that persist

## Retrain Triggers

### Threshold Analysis
| Metric | Count | Percentage |
|--------|-------|------------|
| Days above +10% | 228/384 | 59.4% |
| Days above +20% | 125/384 | 32.6% |
| First day above +20% | 2025-01-03 | Day 2 |

### Recommendations

1. **Seasonal Retraining**: Update model in December for winter, April for spring
2. **CUSUM Monitoring**: Retrain when CUSUM slope stays positive for 2+ weeks
3. **Rolling Window**: Consider 12-month rolling training to capture recent patterns
4. **Minimum Retrain Frequency**: Quarterly (every 3 months)

## Visualizations

| File | Description |
|------|-------------|
| `01_daily_mae_over_time.png` | Daily MAE with 7/30-day rolling averages and thresholds |
| `02_cusum_drift_detection.png` | CUSUM chart showing cumulative drift from baseline |
| `03_monthly_retrain_recommendation.png` | Monthly bars with OK/MONITOR/RETRAIN status |

## Conclusion

The model trained on 2024 data remains viable for 2025 operations, but shows **seasonal sensitivity**:
- Winter months need attention (20-30% worse)
- Summer/fall performance is stable or even improved
- Quarterly retraining with emphasis on pre-winter updates is recommended

**Optimal retraining schedule**:
- December (include full year data for winter patterns)
- March (include winter data for spring transition)
- Optionally: September (include summer data)

---

## Retrain Simulation Experiment

**Question**: Does retraining help, or is the volatility inherent to the system?

### Setup
- **Static model**: Trained on 2024 only, never retrained
- **Adaptive model**: Retrain whenever 7-day rolling MAE exceeds +20% of baseline
- **Minimum gap**: 14 days between retrains

### Results

| Metric | Static | Adaptive | Difference |
|--------|--------|----------|------------|
| MAE | 5.055 MWh | 4.920 MWh | **+2.7%** |
| Retrains | 0 | 10 | - |

### Monthly Comparison

| Month | Static | Adaptive | Improvement |
|-------|--------|----------|-------------|
| 2025-01 | 5.35 | 5.33 | +0.5% |
| 2025-02 | 5.74 | 5.60 | +2.4% |
| 2025-03 | 5.17 | 5.10 | +1.2% |
| 2025-04 | 5.67 | 5.66 | +0.2% |
| 2025-05 | 4.66 | 4.65 | +0.3% |
| 2025-06 | 5.12 | 5.01 | +2.2% |
| 2025-07 | 5.11 | 5.00 | +2.0% |
| 2025-08 | 4.51 | 4.43 | +1.9% |
| 2025-09 | 5.00 | 4.88 | +2.3% |
| 2025-10 | 4.95 | 4.83 | +2.3% |
| 2025-11 | 4.27 | 4.08 | +4.4% |
| 2025-12 | 4.18 | 4.03 | +3.7% |
| 2026-01 | 6.14 | 5.37 | **+12.6%** |

### Retrain Dates
10 retrains triggered:
1. 2025-01-19
2. 2025-02-02
3. 2025-02-16
4. 2025-03-07
5. 2025-03-23
6. 2025-04-06
7. 2025-04-20
8. 2025-06-09
9. 2025-07-07
10. 2026-01-05

### Key Finding

## CONCLUSION: VOLATILITY IS INHERENT

**Retraining provides only 2.7% improvement despite 10 retrains.** This means:

1. **The model captures the learnable patterns** - adding more recent data doesn't significantly help
2. **Residual error is driven by unobserved factors** - weather, grid events, demand shocks
3. **The same difficult periods remain difficult** - Jan-Apr shows high error even after retraining
4. **Only Jan 2026 shows meaningful benefit** (+12.6%) - fresh winter data helps for winter prediction

### Implications

1. **Don't over-invest in retraining infrastructure** - quarterly updates are sufficient
2. **Focus on feature engineering** - add weather, calendar events, grid status
3. **Consider ensemble methods** - multiple models may capture different patterns
4. **Accept irreducible error** - ~4.5-5.0 MWh MAE may be the floor for Lead 12

### Visualizations

| File | Description |
|------|-------------|
| `04_retrain_simulation.png` | Static vs Adaptive comparison with retrain markers |
| `retrain_simulation_results.csv` | Day-by-day predictions for both strategies |
