# Slovakia Grid Load Analysis - Complete Findings

## Overview

**Data**: Slovakia electricity grid load (2024-2025)
- Hourly data: 17,548 records
- 3-minute data: 348,217 records
- Date range: 2024-01-01 to 2025-12-31

**Target**: Actual load (`actual_load_mw`)
**Baseline to beat**: DAMAS day-ahead forecast (`forecast_load_mw`)

---

## 1. Load Characteristics

### Basic Statistics
| Metric | Value |
|--------|-------|
| Mean load | 2,936 MW |
| Std deviation | 455 MW |
| Minimum | 1,919 MW |
| Maximum | 4,170 MW |
| Load factor | 70.4% |

### Percentiles
| Percentile | Load (MW) |
|------------|-----------|
| P5 (base) | 2,234 |
| P25 | 2,591 |
| P50 (median) | 2,921 |
| P75 | 3,238 |
| P95 (peak) | 3,768 |

---

## 2. Variance Decomposition (STL)

| Component | Variance Explained |
|-----------|-------------------|
| **Seasonal** | **57.0%** |
| **Trend** | **35.8%** |
| Residual | 7.0% |

**Key insight**: Seasonality dominates - capturing daily/weekly patterns is crucial.

---

## 3. Temporal Patterns

### Hourly Profile
| Day Type | Peak Hour | Peak Load | Off-Peak Hour | Off-Peak Load |
|----------|-----------|-----------|---------------|---------------|
| Weekday | Hour 10 | 3,333 MW | Hour 4 | 2,462 MW |
| Weekend | Hour 12 | 2,995 MW | Hour 4 | 2,341 MW |

### Weekday vs Weekend
| Metric | Weekday | Weekend | Difference |
|--------|---------|---------|------------|
| Mean load | 3,030 MW | 2,698 MW | **332 MW** |

**Key insight**: Weekday load is 12% higher than weekend. Must include `is_weekend` as feature.

### Monthly Pattern
- **Highest**: Winter months (Dec-Feb) ~3,200 MW
- **Lowest**: Summer months (Jun-Aug) ~2,700 MW
- Seasonal amplitude: ~500 MW

---

## 4. Autocorrelation Analysis

### Load Autocorrelation
| Lag | ACF | Interpretation |
|-----|-----|----------------|
| 1h | 0.963 | Very high - persistence |
| 24h | 0.866 | Strong daily pattern |
| 48h | 0.739 | 2-day pattern |
| 168h | 0.876 | Strong weekly pattern |

### Autocorrelation by Hour (Same Hour Day-to-Day)
| Hours | ACF (lag-1 day) | Characteristic |
|-------|-----------------|----------------|
| Night (2-5) | **0.87-0.89** | Most predictable |
| Morning ramp (7-8) | **0.68-0.71** | Most chaotic |
| Peak (9-14) | 0.73-0.82 | Moderate |
| Evening (19-23) | 0.79-0.82 | Moderate |

**Key insight**: Morning ramp hours (7-8) are hardest to predict.

---

## 5. Stationarity Tests

| Test | Result | Conclusion |
|------|--------|------------|
| ADF (raw) | p < 0.001 | Stationary |
| KPSS (raw) | p < 0.01 | Non-stationary |
| ADF (24h diff) | p < 0.001 | Stationary |
| KPSS (24h diff) | p > 0.05 | Stationary |

**Key insight**: 24h differenced series is stationary - good for SARIMA modeling.

---

## 6. Baseline Forecast (DAMAS) Performance

### Overall Metrics
| Metric | Value |
|--------|-------|
| MAE | 68.2 MW |
| RMSE | 88.1 MW |
| MAPE | 2.34% |
| Bias | -6.3 MW (slight over-forecast) |

### Error by Hour
| Hours | MAE (MW) | MAPE (%) | Notes |
|-------|----------|----------|-------|
| Night (2-4) | 48-50 | 2.0% | Best performance |
| Peak (12-14) | **83-86** | **2.7-2.8%** | Worst performance |

### MAE vs MAPE Comparison
| Metric | Range | Ratio |
|--------|-------|-------|
| MAE | 48 - 86 MW | 1.78x |
| MAPE | 1.97 - 2.78% | 1.41x |

**Key insight**: MAPE is more uniform across hours. Peak hour errors are proportionally higher, not just absolutely higher.

---

## 7. Error Correlation Analysis

### Error Autocorrelation
| Lag | ACF | Interpretation |
|-----|-----|----------------|
| **1h** | **0.864** | Very strong persistence! |
| 2h | 0.731 | Strong |
| 6h | 0.322 | Moderate |
| 24h | 0.222 | Weak |
| 168h | 0.054 | Very weak |

**Key insight**: Errors persist strongly hour-to-hour (0.86) but NOT day-to-day (0.22).

### Error Run Statistics
- Mean error run length: **6.4 hours**
- Max error run length: **81 hours**
- Under-forecast frequency: 47.2%
- Over-forecast frequency: 52.6%

### Same-Hour Error Correlation (Day-to-Day)
| Hours | Correlation | Notes |
|-------|-------------|-------|
| Evening (19-23) | **0.32-0.42** | Most predictable errors |
| Peak (9-14) | **0.07-0.13** | Most chaotic errors |

### Systematic Bias by Hour
| Hours | Mean Error | Direction |
|-------|------------|-----------|
| Morning (6-9) | -28 to -39 MW | **Over-forecast** |
| Evening (21-24) | +22 to +27 MW | **Under-forecast** |

---

## 8. Residual Analysis (2024 Seasonal Removal)

### Setup
- Learn seasonal pattern from 2024 (by day_of_week, hour)
- Apply to 2025 data
- Check if baseline captures residuals

### Results
| Metric | Raw Data | Residuals | Change |
|--------|----------|-----------|--------|
| MAE | 66.7 MW | 67.3 MW | +0.9% |
| R² | - | **0.931** | - |

**Key insight**:
- High R² (0.93): Baseline tracks residual **direction** well
- Same MAE: But doesn't reduce **magnitude** of errors
- The baseline captures the seasonal pattern AND follows residual trends, but the actual error magnitude remains ~67 MW

---

## 9. 3-Minute Data Analysis

### Comparison with Hourly Data
| Metric | Value |
|--------|-------|
| Correlation (3min mean vs hourly) | 0.9997 |
| Mean absolute difference | 5.4 MW |

### Useful 3-Min Features
| Feature | Correlation with 24h Error |
|---------|---------------------------|
| **load_trend_lag24** | **-0.123** (best 3-min feature) |
| load_std_lag24 | 0.005 (no value) |
| load_cv_lag24 | -0.006 (no value) |
| load_range_lag24 | 0.002 (no value) |

**Key insight**: Within-hour **trend** (load increasing/decreasing) has predictive value. Volatility metrics (std, cv, range) do not.

---

## 10. 24-Hour Horizon Analysis

### Correlation with 24h-Ahead Error
| Feature | Correlation |
|---------|-------------|
| error_lag1 | +0.17 |
| error_lag24 | +0.10 |
| load_trend_lag24 | -0.12 |
| error_same_hour_7d | +0.08 |

### Model Results (24h Horizon)
| Model | MAE | vs Baseline |
|-------|-----|-------------|
| Naive (predict 0) | 63.4 MW | - |
| Persistence (lag-24) | 87.5 MW | **-38% worse** |
| Same Hour Yesterday | 87.6 MW | **-38% worse** |
| GB Model | 56.5 MW | **+11%** |

**Key insight**: Simple persistence **hurts** at 24h horizon. Need ML to capture complex patterns.

---

## 11. Day-Ahead Model (Final)

### Setup
At 23:00 on day D, predict all 24 hours of day D+1 using:
- Today's (D) complete 24 hours of forecast errors
- Tomorrow's (D+1) DAMAS forecast
- 2024 seasonal pattern

### Results
| Model | MAE (MW) | vs Baseline |
|-------|----------|-------------|
| Baseline (DAMAS) | 63.4 | - |
| + Seasonal bias only | 65.2 | -2.8% |
| + Today same hour error only | 81.9 | -29.1% |
| **+ GB Model (all features)** | **60.9** | **+4.0%** |

### Top Features
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **today_vs_seasonal** | 0.141 |
| 2 | **forecast_vs_seasonal** | 0.136 |
| 3 | **today_evening_error** | 0.116 |
| 4 | seasonal_error | 0.093 |
| 5 | today_same_hour_error | 0.077 |
| 6 | today_load_trend | 0.074 |

### Improvement by Hour
| Hours | Improvement |
|-------|-------------|
| **Night (1-5)** | **+10% to +17%** |
| Morning (6-8) | +1% to +5% |
| Peak (9-15) | ~0% (hardest) |
| Afternoon (16-20) | +0% to +3% |
| **Evening (21-24)** | **+6% to +12%** |

---

## 12. Key Takeaways

### What We Learned

1. **Seasonality dominates** (57% of variance)
   - Daily pattern (24h cycle)
   - Weekly pattern (168h cycle)
   - Monthly/seasonal variation

2. **Weekday vs Weekend matters**
   - 332 MW difference in mean load
   - Different hourly profiles
   - Must include as feature

3. **Error patterns**
   - Errors persist hour-to-hour (ACF=0.86)
   - But NOT day-to-day (ACF=0.22)
   - Mean error run: 6.4 hours

4. **Peak hours are hardest**
   - Lowest autocorrelation
   - Highest absolute errors
   - Model shows ~0% improvement

5. **3-min data value is limited**
   - Only `load_trend` useful
   - Volatility features (std, cv, range) have no value

6. **Comparing to seasonal is valuable**
   - `today_vs_seasonal` is top feature
   - `forecast_vs_seasonal` is #2 feature

### Recommendations for Modeling

1. **Use baseline forecast as feature** - it's strong!

2. **Add error correction** using:
   - Today's complete errors (all 24 hours)
   - Comparison to 2024 seasonal pattern
   - Time-of-day period errors (morning, evening, etc.)

3. **Feature engineering priorities**:
   - `actual_vs_seasonal`: deviation from learned pattern
   - `forecast_vs_seasonal`: how unusual is the forecast?
   - Today's errors by period (morning, afternoon, evening, night)
   - Load trend from 3-min data

4. **Expected improvements**:
   - Without weather/prices: **~4%** (achieved)
   - With weather data: **+10-15%** additional
   - With day-ahead prices: **+5-10%** additional
   - **Total potential: 20-30% improvement**

---

## 13. Files Generated

### Data Files
- `features/DamasLoad/load_data.parquet` - Combined hourly data
- `features/DamasLoad/load_data.csv` - CSV version

### Analysis Scripts
- `LoadAnalysis/load_analysis.py` - Main analysis
- `LoadAnalysis/residual_analysis.py` - Residual & hypothesis testing
- `LoadAnalysis/error_correlation_analysis.py` - Error patterns
- `LoadAnalysis/horizon_24h_analysis.py` - 24h horizon study
- `LoadAnalysis/dayahead_model_v2.py` - Final day-ahead model

### Plots (in LoadAnalysis/plots/)
1. `01_stl_decomposition.png` - Time series decomposition
2. `01b_decomposition_weekday_weekend.png` - Weekday/weekend comparison
3. `02_temporal_patterns.png` - Hourly, daily, weekly, monthly patterns
4. `02b_load_heatmap.png` - Load heatmap (hour x day)
5. `03_distributions.png` - Load distributions
6. `04_autocorrelation.png` - ACF/PACF analysis
7. `05_stationarity.png` - Rolling statistics
8. `06_forecast_errors.png` - Baseline error analysis
9. `06b_error_heatmap.png` - Error heatmap
10. `07_load_duration_curve.png` - Load duration curve
11. `08_anomalies.png` - Anomaly detection
12. `09_autocorrelation_by_hour.png` - ACF by hour
13. `10_mae_vs_mape_by_hour.png` - MAE vs MAPE comparison
14. `11_residual_forecast_skill.png` - Residual analysis
15. `12_residual_skill_by_hour.png` - Residual skill by hour
16. `13_error_autocorrelation.png` - Error ACF
17. `14_same_hour_error_correlation.png` - Same-hour error patterns
18. `15_hour_to_hour_error_correlation.png` - Hour correlation matrix
19. `16_lagged_error_predictability.png` - Lagged error analysis
20. `17_conditional_error_patterns.png` - Conditional errors
21. `18_error_prediction_potential.png` - Error prediction models
22. `19_24h_horizon_correlation.png` - 24h feature correlations
23. `20_24h_horizon_models.png` - 24h model comparison
24. `21_3min_value_analysis.png` - 3-min data value
25. `22_simple_dayahead_model.png` - Simple model results
26. `23_dayahead_model_v2.png` - Final model results

---

## 14. Next Steps

1. **Add weather data**
   - Temperature (most important)
   - Cloud cover / solar radiation
   - Wind speed
   - Humidity

2. **Add day-ahead prices**
   - Correlated with demand expectations
   - Market signals

3. **Add calendar features**
   - Public holidays
   - School holidays
   - Special events

4. **Model improvements**
   - Try LSTM/GRU for sequence modeling
   - Ensemble methods
   - Probabilistic forecasting (quantiles)

---

*Analysis completed: January 2026*
*Baseline MAE: 63.4 MW | Current best: 60.9 MW (+4.0%)*
