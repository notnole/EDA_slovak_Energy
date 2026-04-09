# Combined Load + Solar Surprise vs Imbalance (Summer)

## Concept

**Net Surprise = Solar Surprise - Load Surprise**

- Solar surprise (actual - forecast solar): supply-side error
- Load surprise (actual - forecast load): demand-side error
- Net surprise: overall energy balance surprise (positive = system has more than expected)

## Data

- **Period**: Summer (Jun-Aug) 2024, 2025, daylight hours only
- **Observations**: 3,826
- **Sources**: `data/clean/solar/solar_hourly.csv`, `features/DamasLoad/load_data.csv`, `data/master/master_imbalance_data.csv`

## Key Results

### Variance Explained (R-squared)

| Signal | r | R-squared | Improvement |
|--------|---|-----------|-------------|
| Load surprise | -0.3416 | 0.1167 |  |
| Solar surprise | +0.1642 | 0.0270 | -77% vs load alone |
| Net surprise | +0.3718 | 0.1382 | +18% vs load alone |
| Combined (regression) | +0.3718 | 0.1383 | +19% vs load alone |

### Partial Correlations

| Signal | Partial r | Interpretation |
|--------|-----------|----------------|
| Solar | controlling for Load | +0.1568 | Solar adds info beyond load |
| Load | controlling for Solar | -0.3387 | Load adds info beyond solar |

### Cross-correlation

Load surprise vs Solar surprise: r = -0.051
(Weakly correlated - mostly independent signals)

### Regression Coefficients

| Predictor | Coefficient | Interpretation |
|-----------|-------------|----------------|
| Load surprise | -0.038426 | +1 MW load surprise -> -0.0384 MWh imbalance |
| Solar surprise | +0.040055 | +1 MW solar surprise -> +0.0401 MWh imbalance |
| Intercept | +1.2839 | Baseline imbalance bias |

### Direction Prediction (Net Surprise)

| Net Surprise Bin | N | % System LONG | Mean Imbalance (MWh) |
|-----------------|---|---------------|----------------------|
| <-150 | 128 | 19% | -7.34 |
| -150:-75 | 424 | 41% | -2.13 |
| -75:-25 | 627 | 46% | -0.34 |
| -25:0 | 458 | 55% | +0.78 |
| 0:25 | 448 | 56% | +1.40 |
| 25:75 | 760 | 65% | +2.84 |
| 75:150 | 660 | 77% | +5.47 |
| >150 | 321 | 83% | +9.62 |

## Conclusion

1. **Combined model explains 13.8% of summer imbalance variance**
   vs 11.7% for load surprise alone.

2. Solar surprise adds **independent information** (partial r = +0.157)
   - The two signals are largely independent
     (cross-correlation r = -0.051)

3. **Physical interpretation**: Load surprise direction is *opposite* to imbalance
   (higher demand -> system SHORT), while solar surprise is *same direction*
   (more sun -> system LONG). Net surprise captures both effects.

## Files

- `01_combined_surprise.png` - 6-panel comparison dashboard
- `data/signal_comparison.csv`
- `data/direction_net_surprise.csv`
