# Baseline Model Analysis Summary

## Model Definition
From `models/BaseLine.md`:
```
Prediction = -0.25 * weighted_avg(regulation)
```

Where 0.25 converts MW to MWh for a 15-minute block.

**Important**: Label datetime represents the START of the settlement period.

## Performance by Lead Time

| Lead Time | MAE (MWh) | RMSE (MWh) | R² | Bias |
|-----------|-----------|------------|-----|------|
| 15 min | 6.89 | 9.74 | 0.309 | +0.17 |
| 12 min | 6.56 | 9.42 | 0.355 | -0.36 |
| 9 min | 6.32 | 9.33 | 0.365 | -0.70 |
| 6 min | 4.63 | 7.00 | 0.641 | -0.14 |
| 3 min | 3.59 | 5.68 | 0.762 | -0.03 |
| 0 min | 3.26 | 5.35 | 0.788 | -0.08 |

## Comparison to Spec

| Metric | Spec Expected | Actual |
|--------|---------------|--------|
| MAE at 3 min | 1.0-1.4 MWh | 3.59 MWh |
| MAE at 12 min | >4.0 MWh | 6.56 MWh |

The actual MAE is higher than spec, likely due to:
1. Fixed coefficient 0.25 vs optimal 0.185 (26% difference)
2. Different time period or data characteristics

## Key Characteristics

### Lead Time Effects
- **0-3 min**: Near-measurement (R² = 0.76-0.79)
- **6 min**: Transition zone (R² = 0.64)
- **9-15 min**: Heuristic prediction (R² = 0.31-0.37)

### Error Patterns by Imbalance Size (Lead 12 min)
| Size | Mean Bias | Std Error |
|------|-----------|-----------|
| Small | 0.66 | 7.07 |
| Large | 0.41 | 13.36 |

Large imbalances have higher variance - harder to predict precisely.

### MAE Improvement Over Time
- Lead 15 → 0 min: 53% reduction (6.89 → 3.26)
- Lead 12 → 3 min: 45% reduction (6.56 → 3.59)

## Targets for ML Models

To beat the baseline:
- **At 12 min lead**: Beat MAE = 6.56 MWh, R² = 0.355
- **At 15 min lead**: Beat MAE = 6.89 MWh, R² = 0.309

Potential improvements:
1. Use fitted coefficient (-0.185) instead of fixed (-0.25)
2. Add temporal features (hour, weekday)
3. Add lag features (previous imbalance/regulation)
4. Non-linear models for better handling of extreme values

## Files Generated

### Data
- `data/baseline_performance.csv`
- `data/error_lead12_by_hour.csv`
- `data/error_lead12_by_imbalance.csv`
- `data/error_lead3_by_hour.csv`
- `data/error_lead3_by_imbalance.csv`

### Visualizations
- `viz_prediction_trajectory.png` - How predictions evolve for 6 specific cases
- `viz_error_heatmap.png` - MAE by hour × lead time
- `viz_convergence_funnel.png` - Error percentiles narrowing with decreasing lead time
- `viz_daily_pattern.png` - Full day view of predictions vs actual
- `viz_error_by_magnitude.png` - Error scaling with imbalance size
