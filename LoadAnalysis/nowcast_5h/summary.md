# Two-Stage Load Forecast Error Correction Model

## Overview

This model corrects DAMAS load forecast errors for the Slovak energy grid, providing nowcast predictions for horizons H+1 to H+5 hours. It operates at 4 quarter-hour intervals (Q0=xx:00, Q1=xx:15, Q2=xx:30, Q3=xx:45), using partial real-time load data to improve accuracy as more information becomes available within each hour.

## Model Architecture

**Two-Stage Residual Correction:**

1. **Stage 1** - Predicts the DAMAS forecast error using:
   - Error lags (1-8 hours)
   - Rolling statistics (3h, 6h, 12h, 24h means and std)
   - Error trends and momentum
   - Time features (hour, day of week, cyclical encoding)
   - Load volatility from 3-minute data
   - Quarter-specific: extrapolated H+1 error from partial load data

2. **Stage 2** - Corrects Stage 1 residuals using:
   - Lagged Stage 1 residuals
   - Residual rolling statistics
   - Residual trends

**Key Design Decisions:**
- Out-of-fold predictions for Stage 2 training (prevents data leakage)
- Seasonal bias correction for load extrapolation
- Sample weighting with recency bias
- LightGBM with Optuna-tuned hyperparameters

## Performance

### Holdout Results (January 2026)

| Quarter | H+1 | H+2 | H+3 | H+4 | H+5 |
|---------|-----|-----|-----|-----|-----|
| Q0 (xx:00) | 32.8 | 46.8 | 54.3 | 61.6 | 65.7 |
| Q1 (xx:15) | 20.8 | 38.6 | 53.3 | 61.5 | 64.9 |
| Q2 (xx:30) | 15.7 | 35.4 | 51.1 | 61.0 | 63.4 |
| Q3 (xx:45) | 8.7 | 31.8 | 49.9 | 59.1 | 62.3 |

**DAMAS Baseline: 83.4 MW MAE**

### Improvement Summary

- **Q3 H+1**: 89% improvement over DAMAS (8.7 vs 83.4 MW)
- **Q0 H+1**: 61% improvement over DAMAS (32.8 vs 83.4 MW)
- Later horizons (H+4, H+5) show diminishing but still significant improvements

## Directory Structure

```
nowcast_5h/
├── deployment/              # Production artifacts
│   ├── inference.py         # LoadForecastPredictor class
│   └── models/              # 40 trained models + configs
│       ├── s1_q{0-3}_h{1-5}.joblib
│       ├── s2_q{0-3}_h{1-5}.joblib
│       ├── feature_configs.json
│       ├── seasonal_bias.json
│       └── training_metadata.json
├── tuning/                  # Hyperparameter optimization
│   ├── optuna_stage1.py
│   ├── optuna_stage2.py
│   ├── h1/ to h5/           # Best params per horizon
│   └── tuning_summary.json
├── train_two_stage_model.py # Reference training implementation
└── summary.md               # This file
```

## Usage

### Production Inference

```python
from deployment.inference import LoadForecastPredictor

predictor = LoadForecastPredictor('deployment/models/')

predictions = predictor.predict(
    quarter=2,                    # 0=xx:00, 1=xx:15, 2=xx:30, 3=xx:45
    error_history=[...],          # Last 48+ hourly errors [t-1, t-2, ...]
    forecast_history=[...],       # Last 48+ hourly DAMAS forecasts
    partial_3min_load=[...],      # 3-min load readings for H+1 hour
    hour=14,                      # Current hour (0-23)
    dow=2,                        # Day of week (0=Monday)
)

# Returns: {'h1': {'s1_pred', 's2_pred', 'final_pred'}, 'h2': {...}, ...}
```

### Get Corrected Load Forecast

```python
load_predictions = predictor.predict_load(
    quarter=2,
    error_history=error_history,
    forecast_history=forecast_history,
    partial_3min_load=partial_3min_load,
    hour=14,
    dow=2,
    damas_forecasts=[3520, 3540, 3560, 3580, 3590],  # H+1 to H+5
)

# Returns: {'h1': 3532.8, 'h2': 3548.5, ...}  # Corrected MW values
```

## Training Data

- **Period**: 2024-01-01 to 2026-01-31
- **Source**: DAMAS hourly forecasts and actuals, 3-minute load data
- **Validation**: January 2026 holdout (not used in final production training)

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- lightgbm >= 3.3.0
- scikit-learn >= 1.0.0
- joblib (for model serialization)
