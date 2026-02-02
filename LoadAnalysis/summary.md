# Load Analysis Summary

## Overview
Comprehensive analysis of Slovakia grid load data and DAMAS day-ahead forecast performance, with development of an improved prediction model.

## Data Sources
- **Load Data**: `features/DamasLoad/load_data.parquet` (24,843 hourly records, 2024-2026)
- **Price Data**: `features/DamasPrices/da_prices.parquet` (day-ahead market prices)
- **Seasonal Baseline**: 2024 patterns by day_of_week and hour

## Key Findings

### 1. DAMAS Baseline Performance
- **MAE**: 68.2 MW (2.34% MAPE)
- **Systematic bias**: Over-forecasts mornings (-28 to -39 MW), under-forecasts evenings (+22 to +27 MW)
- **Peak hours worst**: 83-86 MW MAE vs 48-50 MW at night

### 2. Error Predictability
- **Hour-to-hour correlation**: 0.86 (very strong persistence)
- **Day-to-day correlation**: 0.22 (weak - errors don't persist across days)
- **Mean error run length**: 6.4 hours
- **Evening errors most predictable**: 0.32-0.42 same-hour correlation

### 3. Model Development Result
| Version | MAE | vs Baseline |
|---------|-----|-------------|
| Baseline (DAMAS) | 64.2 MW | - |
| **Final Model** | **62.4 MW** | **+2.9%** |

Cross-validation: **+3.4% +/- 3.2%** improvement

### 4. Critical Insight: Simpler is Better
Adding more features **hurt** performance:
- Core features only: +2.9%
- + Momentum features: -3.8%
- + Price features: -3.1%
- + Week statistics: -1.4%

## Analysis Subfolders

| Folder | Description | Key Plots |
|--------|-------------|-----------|
| [decomposition/](decomposition/summary.md) | STL decomposition of load time series | 01_stl_decomposition.png |
| [prophet_decomposition/](prophet_decomposition/summary.md) | **Prophet decomposition with holidays** | 04_holiday_effects.png |
| [temporal_patterns/](temporal_patterns/summary.md) | Hourly, daily, weekly patterns | 02_temporal_patterns.png |
| [autocorrelation/](autocorrelation/summary.md) | ACF/PACF and stationarity tests | 04_autocorrelation.png |
| [baseline_errors/](baseline_errors/summary.md) | DAMAS forecast error analysis | 06_forecast_errors.png |
| [residual_analysis/](residual_analysis/summary.md) | Seasonal residual skill | 11_residual_forecast_skill.png |
| [error_correlation/](error_correlation/summary.md) | Error persistence patterns | 13_error_autocorrelation.png |
| [horizon_24h/](horizon_24h/summary.md) | 24-hour ahead prediction | 20_24h_horizon_models.png |
| [dayahead_model/](dayahead_model/summary.md) | Model development and optimization | 27_final_model.png |

### Prophet Decomposition Key Finding
Holidays cause **massive load drops** (200-715 MW, or 7-24% of average):
- Easter Monday: -715 MW (largest)
- Christmas Day: -579 MW
- New Year's: -556 MW

## Final Model Features (by importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | d1_evening_error | 0.186 |
| 2 | d1_load_vs_seasonal | 0.184 |
| 3 | forecast_vs_seasonal | 0.166 |
| 4 | seasonal_error | 0.105 |
| 5 | d1_same_hour_error | 0.074 |

## What Didn't Help

- **Price features**: DAMAS already incorporates market price information
- **Momentum/trend features**: Add noise, cause overfitting
- **Week-long statistics**: Too much noise, weak day-to-day correlation
- **Lagged errors (d2-d7)**: Day-to-day error correlation too weak (0.22)

## Production Implementation

The final model runs at **23:00 on day D** to predict all 24 hours of day D+1:
1. Uses today's (D) complete 24 hours of forecast errors
2. Uses tomorrow's (D+1) DAMAS forecast
3. Applies 2024 seasonal patterns
4. Core features only (11 features, no price/momentum)

See `dayahead_model/final_dayahead_model.py` for clean implementation.

## Related Analysis
- Price Analysis: `features/DamasPrices/`
- Raw Data: `RawData/Damas/`
