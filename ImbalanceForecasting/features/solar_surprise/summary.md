# Solar Surprise Effect on Imbalance Direction (Summer)

## Definition

**Solar Surprise = Actual Solar Production - DA Forecast Solar Production (MW)**

- Positive surprise: more sun than expected -> excess generation
- Negative surprise: less sun than expected -> generation shortfall

## Data

- **Solar source**: `data/clean/solar/solar_hourly.csv` (DAMAS actual + DA forecast)
- **Imbalance source**: `data/master/master_imbalance_data.csv` (aggregated to hourly)
- **Filter**: Summer months only (June, July, August), daylight hours
- **Period**: 2024-06-01 to 2025-08-31
- **Years**: 2024, 2025
- **Summer daylight observations**: 3,826

## Key Findings

### Overall Correlation

| Group | N | Correlation (r) | R-squared | Surprise Std (MW) |
|-------|---|-----------------|-----------|-------------------|
| All summer daylight | 3,826 | +0.1642 | 0.0270 | 36.1 |
| Summer 2024 | 1,732 | +0.1281 | 0.0164 | 40.0 |
| Summer 2025 | 2,094 | +0.2385 | 0.0569 | 32.0 |
| June | 1,169 | +0.2039 | 0.0416 | 43.7 |
| July | 1,247 | +0.1766 | 0.0312 | 33.6 |
| August | 1,410 | +0.1080 | 0.0117 | 28.7 |

### Hourly Correlation Profile

| Hour | N | Correlation | Mean Solar (MW) | Surprise Std (MW) |
|------|---|-------------|-----------------|-------------------|
| 05:00 | 183.0 | -0.0567 | 4 | 6.2 |
| 06:00 | 183.0 | +0.1309 | 23 | 13.4 |
| 07:00 | 183.0 | +0.2509 | 73 | 24.6 |
| 08:00 | 183.0 | +0.2569 | 142 | 36.8 |
| 09:00 | 183.0 | +0.3067 | 205 | 46.2 |
| 10:00 | 183.0 | +0.1048 | 250 | 49.8 |
| 11:00 | 183.0 | +0.1020 | 276 | 50.3 |
| 12:00 | 183.0 | +0.2066 | 283 | 53.9 |
| 13:00 | 183.0 | +0.2579 | 274 | 57.1 |
| 14:00 | 183.0 | +0.1585 | 252 | 56.2 |
| 15:00 | 183.0 | +0.1753 | 214 | 54.0 |
| 16:00 | 183.0 | +0.1743 | 163 | 46.3 |
| 17:00 | 183.0 | +0.0987 | 103 | 33.8 |
| 18:00 | 183.0 | +0.0982 | 44 | 16.4 |
| 19:00 | 183.0 | +0.1062 | 12 | 7.7 |
| 20:00 | 183.0 | -0.0080 | 2 | 1.2 |

### By Solar Surprise Magnitude

| Magnitude | N | Correlation | Mean Imbalance (MWh) |
|-----------|---|-------------|----------------------|
| <10 MW | 2,030 | +0.0388 | +1.46 |
| 10-30 MW | 905 | +0.1226 | +2.73 |
| 30-60 MW | 508 | +0.2195 | +2.15 |
| 60-100 MW | 227 | +0.3746 | +2.02 |
| >100 MW | 141 | +0.1424 | +6.22 |

### Direction Prediction

| Solar Surprise Bin | N | % System LONG | Mean Imbalance (MWh) |
|-------------------|---|---------------|----------------------|
| <-100 | 24 | 50% | +1.54 |
| -100:-50 | 88 | 26% | -4.51 |
| -50:-20 | 367 | 52% | +0.48 |
| -20:0 | 853 | 54% | +1.31 |
| 0:20 | 1,725 | 61% | +1.74 |
| 20:50 | 424 | 69% | +4.83 |
| 50:100 | 228 | 63% | +4.26 |
| >100 | 117 | 72% | +7.18 |

## Physical Interpretation

1. **Expected mechanism**: Solar over-production (positive surprise) should push
   the system LONG (positive imbalance) as more unscheduled generation enters.
   Conversely, solar shortfall should push SHORT.

2. **Correlation strength**: r = +0.164 explains
   2.7% of imbalance variance in summer daylight hours.

3. **Comparison with load surprise**: Load surprise has r = -0.30 with imbalance
   (year-round). Solar surprise is complementary - active only in summer daylight.

## Files

- `01_solar_surprise_analysis.png` - Main 6-panel dashboard
- `02_timeseries_sample.png` - Sample time series per year
- `data/correlation_summary.csv`
- `data/hourly_correlation.csv`
- `data/magnitude_correlation.csv`
- `data/direction_analysis.csv`
