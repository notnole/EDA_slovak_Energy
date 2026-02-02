# Label Analysis Summary

## What This Analysis Contains

Basic statistical analysis and seasonality patterns of the 15-minute system imbalance (the target variable for nowcasting).

**Data**: 72,375 observations from 2024-01-01 to 2026-01-24 (754 days)

---

## Key Findings

### 1. Distribution Characteristics

| Metric | Value |
|--------|-------|
| Mean | +1.61 MWh |
| Median | +0.91 MWh |
| Std Dev | 11.96 MWh |
| Min | -232.0 MWh |
| Max | +90.1 MWh |
| Skewness | -0.43 (slight left skew) |

**Sign distribution**: 55.5% positive (surplus) vs 44.5% negative (deficit)

The system tends toward slight over-generation, but both directions are common.

### 2. Daily (Intraday) Patterns

- **Peak**: 13:00-14:00 (mean ~3.5 MWh)
- **Trough**: 20:00 (mean ~-0.2 MWh)
- **Most volatile**: 11:00-12:00 (std ~16.5 MWh)
- **Daily amplitude**: ~3.75 MWh swing in mean

Pattern: Higher surplus during midday (solar production peak?), deficit tendency in evening.

### 3. Weekly Patterns

- **Highest**: Sunday (mean 3.35 MWh)
- **Lowest**: Wednesday (mean 0.08 MWh)
- **Weekend effect**: +1.46 MWh higher on weekends

Pattern: Harder to balance on weekends (lower demand, less flexible generation).

### 4. Monthly Patterns

- **Peak**: April (mean 2.89 MWh)
- **Trough**: October (mean 0.25 MWh)
- **Most volatile**: November (std 15.02 MWh)

Pattern: Transition months (spring/autumn) show more imbalance, likely due to changing renewable patterns.

### 5. Data Quality Notes

- Removed 1 outlier: -1732 MWh on 2025-12-01 (data error)
- DST days have 100 settlement periods instead of 96 (handled by capping hour at 23)
- Two naming conventions in source files (English/Slovak) - normalized to English

---

## Implications for the Model

### Feature Engineering

| Finding | Recommended Feature |
|---------|---------------------|
| Strong intraday pattern | `hour_of_day` (0-23) or `settlement_period` (1-96) |
| Weekly pattern | `day_of_week` (0-6) or `is_weekend` binary |
| Weekend effect | `is_weekend` may capture +1.46 MWh shift |
| Monthly variation | `month` (1-12) - lower priority, may overfit |
| High volatility midday | Interaction: `hour * rolling_std` |

### Target Variable Handling

1. **No transformation needed** - Distribution is roughly symmetric (skew = -0.43)
2. **Outlier filtering applied** - Values beyond ±300 MWh excluded as data errors
3. **Sign is important** - Model should predict both positive and negative values

### Expected Challenges

1. **High variance** (std = 12 MWh) means predictions will have inherent uncertainty
2. **Evening transition** (19:00-21:00) shows sign changes - harder to predict direction
3. **Midday volatility** - Higher uncertainty during peak hours

### Baseline Performance Expectations

For a naive model (predict mean by hour):
- Expected RMSE: ~10-11 MWh
- Any useful model should significantly beat this

---

## Files in This Folder

| File | Description |
|------|-------------|
| `01_time_series_overview.png` | Full series, daily aggregates, distribution |
| `02_daily_seasonality.png` | Intraday patterns by hour and settlement period |
| `03_weekly_seasonality.png` | Day-of-week patterns, weekday vs weekend |
| `04_monthly_seasonality.png` | Monthly patterns and volatility |
| `05_yearly_seasonality.png` | Year-over-year comparison |
| `label_statistics_report.txt` | Detailed numerical statistics |
| `plot_descriptions.txt` | Text descriptions of each plot |
