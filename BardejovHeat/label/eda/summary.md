# EDA - Bardejov Heat Load

## Data

- **Source**: `data/Bardejov/heat_load_timeseries.csv`
- Resampled to hourly mean MW, excluding the 2025 Mar-Apr shutdown (zeros distort stats)
- 42,671 hourly readings used

## Key Findings

### Seasonal pattern

- **Peak winter**: 20-25 MW (Jan-Feb), daily energy ~400-500 MWh
- **Shoulder season**: 5-15 MW (Oct-Nov, Mar-Apr)
- **Summer baseload**: 1-4 MW (Jun-Aug), ~50 MWh/day (hot water only)
- Heating season runs roughly Oct through Mar

### Daily pattern

- Sharp morning ramp-up at 5-6 AM
- Daytime plateau with slight dip around midday
- Evening decline starting ~20:00
- Overnight minimum ~4-5 MW (winter) or near-zero (summer)

### Weekday vs weekend

- Heating season (Oct-Mar): weekday mean ~1 MW higher than weekend
- Morning ramp-up is sharper on weekdays (likely commercial/industrial demand)
- Pattern is consistent but the difference is small vs temperature effect

### Load duration curve

Key percentiles of hourly power:
- 10% exceeded: 16.3 MW
- 25% exceeded: 13.0 MW
- 50% exceeded: 7.4 MW
- 75% exceeded: 2.9 MW
- 90% exceeded: 1.7 MW

### Annual energy

- Typical full year: ~45-50 GWh
- Strong year-over-year consistency in winter months
- 2025 lower due to maintenance shutdown

### Weekly zoom patterns

Winter weeks show consistent daily cycling (5 MW overnight to 20+ MW peak). Day-to-day variation in peak is driven by weather, not day-of-week. Summer weeks show the same daily shape at much lower amplitude (0.2-3.5 MW).

## Plots

1. `01_monthly_boxplots.png` - Monthly distribution (whiskers = 5th/95th percentile)
2. `02_load_duration_curve.png` - Load duration curve with key percentiles
3. `03_heatmap_hour_month.png` - Average MW by hour and month
4. `04_weekday_vs_weekend.png` - Weekday vs weekend profiles (heating season)
5. `05_monthly_energy_totals.png` - Monthly energy totals by year
6. `06_cumulative_energy.png` - Cumulative annual energy curves
7. `07_winter_week.png` - One-week zoom: winter (Jan 2024)
8. `08_summer_week.png` - One-week zoom: summer (Jul 2024)
9. `winter_weeks/` - 5 representative winter weeks across 3 winters
