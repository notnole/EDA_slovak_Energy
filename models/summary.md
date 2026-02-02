# Load Prediction Models

## Overview
Production-ready models for Slovakia grid load prediction.

## Available Models

### 1. Baseline: DAMAS Forecast
**File:** `baseline_damas.py`

Simple baseline that uses DAMAS day-ahead forecast unchanged.

| Metric | Value |
|--------|-------|
| MAE | 64.2 MW |
| MAPE | 2.34% |
| RMSE | 88.1 MW |

```python
from baseline_damas import DamasBaseline

model = DamasBaseline()
predictions = model.predict(forecast_load_mw)
```

### 2. Day-Ahead Error Corrector
**File:** `dayahead_corrector.py`

Gradient boosting model that corrects DAMAS forecast errors.

| Metric | Value |
|--------|-------|
| MAE | 62.4 MW |
| Improvement | +2.9% vs baseline |
| Cross-validation | +3.4% +/- 3.2% |

```python
from dayahead_corrector import DayAheadCorrector

# Train
model = DayAheadCorrector()
model.fit(load_data)

# Predict tomorrow (run at 23:00 on day D)
corrected = model.predict_from_dataframe(df_today, df_tomorrow)

# Save/load
model.save('trained_model.joblib')
model = DayAheadCorrector.load('trained_model.joblib')
```

## Model Comparison

| Model | MAE (MW) | Improvement |
|-------|----------|-------------|
| Baseline (DAMAS) | 64.2 | - |
| **Day-Ahead Corrector** | **62.4** | **+2.9%** |

## Operational Setup

The Day-Ahead Corrector runs at **23:00 on day D** to predict day D+1:

**Inputs required:**
1. Today's (D) complete 24 hours:
   - Actual load (MW)
   - DAMAS forecast (MW)
2. Tomorrow's (D+1) DAMAS forecast
3. Seasonal patterns (trained from 2024)

**Output:**
- Corrected load predictions for all 24 hours of D+1

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | d1_evening_error | 0.186 |
| 2 | d1_load_vs_seasonal | 0.184 |
| 3 | forecast_vs_seasonal | 0.166 |
| 4 | seasonal_error | 0.105 |
| 5 | d1_same_hour_error | 0.074 |

## What Didn't Help

These features were tested but **decreased** performance:
- Price features (-3.1%)
- Momentum/trend features (-3.8%)
- Week-long statistics (-1.4%)
- Lagged errors from days 2-7

## Data Requirements

Models expect data from:
- `features/DamasLoad/load_data.parquet`

Required columns:
- `datetime`, `actual_load_mw`, `forecast_load_mw`
- `hour`, `day_of_week`, `is_weekend`, `year`

## Files

| File | Description |
|------|-------------|
| `baseline_damas.py` | Baseline model class |
| `dayahead_corrector.py` | Error correction model |
| `trained_dayahead_corrector.joblib` | Pre-trained model (after running train) |
| `summary.md` | This documentation |
