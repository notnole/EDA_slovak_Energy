# Temperature Forecast Analysis: GFS vs Actual

## Data Source
- **Raw file**: `RawData/Tampretures.csv` (Ipesoft EDA export, European decimal format)
- **Cleaned**: `data/clean/weather/temperature_15min.csv`
- **Merged dataset**: `LoadAnalysis/temperature_analysis/data/temp_load_price_merged.csv`

## Period
2025-09-14 to 2026-01-31 (autumn-winter, 3,363 hourly observations)

## GFS Temperature Forecast Quality

| Metric | Value |
|--------|-------|
| Mean bias | -0.54 C (systematic cold bias) |
| MAE | 1.10 C |
| RMSE | 1.33 C |
| Median error | -0.66 C |

- The GFS model **consistently underestimates** temperature (forecasts colder than actual)
- Bias is worst in early morning hours (00:00-06:00), approaching zero during afternoon
- Error distribution is left-skewed, with occasional large cold-bias events exceeding -4 C

## Temperature Effect on Load

| Metric | Value |
|--------|-------|
| Correlation (r) | -0.401 |
| Sensitivity | -31.5 MW per 1 C |
| Cold (<0 C) mean load | 3,456 MW |
| Warm (>10 C) mean load | 2,946 MW |
| Cold-warm delta | +511 MW |

- Strong negative relationship: colder temperatures drive higher load (electric heating)
- The relationship is approximately linear in the 0-15 C range
- Below -6 C, load saturates around 3,600-3,800 MW (heating capacity limits)
- Rolling correlation strengthens from -0.4 in autumn to -0.8 in deep winter (Nov-Jan)

## Temperature Effect on DA Price

| Metric | Value |
|--------|-------|
| Correlation (r) | -0.418 |
| Sensitivity | -2.4 EUR/MWh per 1 C |
| Cold (<0 C) mean price | 129.5 EUR/MWh |
| Warm (>10 C) mean price | 91.7 EUR/MWh |
| Cold-warm delta | +37.8 EUR/MWh |

- Temperature-price relationship is mediated through load (temp -> load -> price)
- Cold spells (blue shading in Plot 4) consistently coincide with price spikes
- Largest price impacts during late-Nov and Jan cold events

## Temperature Error vs Load Forecast Error

| Metric | Value |
|--------|-------|
| Correlation (r) | 0.097 |
| Sensitivity | 7.0 MW per 1 C temp error |

- The direct temp-error to load-error link is **weak** (r = 0.097)
- This is expected: DAMAS load forecasts use their own NWP inputs, not necessarily the same GFS run
- The rolling correlation fluctuates between -0.2 and +0.5, suggesting the relationship is episodic rather than systematic
- Temperature forecast error explains only ~1% of load forecast error variance

## Key Insights for Workflows

1. **For Load Forecasting**: Temperature is a strong predictor of load level (r = -0.40), but the GFS cold bias means the load model may systematically overestimate heating demand if using raw GFS. A bias correction of +0.54 C would improve predictions.

2. **For Price Forecasting**: Temperature explains ~17% of DA price variance. Cold spell detection (below 0 C) is a useful binary feature for price spike prediction.

3. **For Imbalance Nowcasting**: The weak temp-error to load-error link suggests temperature forecast error is NOT a major driver of real-time imbalance. Other factors (solar ramps, wind variability, demand response) dominate at the 15-minute scale.

## Files

| File | Description |
|------|-------------|
| `01_temperature_timeseries.png` | Daily temp forecast vs actual + error + load |
| `02_temperature_scatter_analysis.png` | Scatter: temp vs load, temp vs price, temp error vs load error |
| `03_forecast_error_and_sensitivity.png` | Error distribution, hourly profile, load/price sensitivity curves |
| `04_rolling_correlation_cold_spells.png` | Rolling correlations and cold spell price impact |
| `data/temp_load_price_merged.csv` | Hourly merged dataset for further analysis |
| `scripts/run_analysis.py` | Reproducible analysis script |
