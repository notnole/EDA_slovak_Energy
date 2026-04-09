# Weather-Heat Load Correlation Analysis

## Data

- **Weather**: Hourly actuals from Open-Meteo for Bardejov (2024-01 to 2026-03)
- **Heat load**: Hourly mean power in MW (resampled from mixed-resolution kW)
- **Overlap**: 16,366 hourly records, 682 daily records (2024-01-01 to 2025-12-31)

## Key Findings

### Heating curve

Temperature is the dominant predictor of heat load. Daily mean regression:
- **Slope**: -0.74 MW/degC (or -17.8 MWh/day per degC)
- **r = -0.96** on daily means (R^2 = 0.92)
- **Threshold**: ~15 degC separates heating and non-heating regimes

Reference points (from linear fit):
| Temperature | Daily mean MW | Daily MWh |
|------------|--------------|-----------|
| -10 degC   | 22.0 MW      | ~527 MWh  |
| -5 degC    | 18.3 MW      | ~438 MWh  |
| 0 degC     | 14.6 MW      | ~350 MWh  |
| 5 degC     | 10.9 MW      | ~261 MWh  |
| 10 degC    | 7.2 MW       | ~172 MWh  |
| 15 degC    | 3.5 MW       | ~84 MWh   |
| >15 degC   | ~2 MW        | ~50 MWh   |

### Correlation matrix (all features)

| Feature         | Hourly r with heat load |
|----------------|------------------------|
| Temperature     | -0.81                  |
| Apparent Temp   | -0.81                  |
| Snow Depth      | +0.48                  |
| Solar Radiation | -0.27                  |
| Cloud Cover     | +0.24                  |
| Humidity        | +0.23                  |
| Wind Speed      | -0.00                  |

### Range restriction effect (important statistical note)

The overall correlation (r=-0.81 hourly, r=-0.96 daily) is inflated by **range restriction**. When data is split into seasons, within-season correlations are substantially lower:

| Segment        | Day r   | Night r |
|---------------|---------|---------|
| Winter (DJF)  | -0.81   | -0.76   |
| Shoulder (MA/ON) | -0.68 | -0.63  |
| Summer (MJJAS)| -0.14   | -0.10   |

The combined correlation captures the obvious "winter is cold = high heat, summer is warm = low heat" signal. The within-season correlations are what actually matters for forecasting: given that it's already winter, how much does each degree matter?

This means a forecasting model needs **separate treatment by season** - temperature alone is nearly useless in summer (demand is flat hot water baseload).

### Hourly sensitivity variation

Temperature sensitivity varies by hour (heating season only, T <= 15 degC):
- Strongest at **4-5 AM**: ~-0.85 MW/degC (buildings cooling overnight, most sensitive to outdoor temp)
- Weakest at **midday**: ~-0.55 MW/degC (solar gains, occupancy, internal heat)
- Correlation stable around r = -0.85 across all hours

### Day vs Night feature differences

- **Solar radiation**: Meaningful during day (-0.46) but zero at night (as expected)
- **Wind**: Weak in both regimes
- **Snow depth**: Proxy for sustained cold spells, stronger at night (+0.53)

### Apparent vs actual temperature

Both give identical r = -0.96 on daily means. Wind chill correction adds no predictive value - actual temperature is sufficient.

### Residuals after temperature

After removing the temperature signal from daily heating-season data:
- Residuals are small (typically +/- 2 MW)
- Slight weekday/weekend effect: Sat-Sun residuals shift ~0.5 MW negative
- Day-of-week explains very little additional variance

## Plots

1. `01_heating_curve.png` - Hourly scatter + daily means + piecewise linear fit
2. `02_heating_curve_monthly.png` - Daily means colored by month
3. `02b_daily_temp_vs_MWh.png` - Time series: daily MWh and temperature (inverted axis)
4. `03_correlation_matrix_day_night.png` - Feature correlation: day vs night
5. `03b_correlation_matrix_season_daynight.png` - Feature correlation: 3 seasons x day/night
6. `04_temp_sensitivity_by_hour.png` - Slope and r by hour of day
7. `05_residuals.png` - Residuals after temperature fit + day-of-week breakdown
8. `06_apparent_vs_actual_temp.png` - Wind chill comparison

## Scripts

- `scripts/plot_heating_curves.py` - Heating curve plots (01, 02, 02b, 06)
- `scripts/plot_correlation_matrices.py` - Correlation matrices (03, 03b)
- `scripts/plot_sensitivity_residuals.py` - Sensitivity and residual analysis (04, 05)
