# Prophet Decomposition Analysis

## Overview
Advanced time series decomposition using Facebook Prophet, chosen over STL because it handles **holidays natively** - crucial for Slovakia electricity load.

## Method
```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    holidays=slovak_holidays,  # 27 holidays loaded
    holidays_prior_scale=10.0  # Strong holiday effects
)
```

## Variance Decomposition

| Component | Variance Explained | Interpretation |
|-----------|-------------------|----------------|
| **Yearly** | **50.0%** | Winter high, summer low |
| **Daily** | **39.2%** | Morning ramp, midday peak |
| **Weekly** | **12.4%** | Weekday vs weekend |
| Residual | 11.3% | What's left to model |
| Trend | 3.5% | Long-term changes |

### Comparison with STL (period=168)
| Method | Seasonal | Trend | Residual |
|--------|----------|-------|----------|
| STL | 57.0% | 35.8% | 7.0% |
| Prophet (Weekly+Daily) | 51.6% | 3.5% | 11.3% |

**Key insight**: Prophet separates yearly seasonality (50%) which STL lumped into "trend". The true trend is only 3.5%.

## Holiday Effects (Major Finding!)

| Holiday | Load Impact (MW) | % of Avg Load |
|---------|------------------|---------------|
| **Easter Monday** | **-714.6** | **-24.3%** |
| **Christmas Day** | **-578.8** | **-19.7%** |
| **New Year's Day** | **-555.9** | **-18.9%** |
| Second Day of Christmas | -468.6 | -16.0% |
| Christmas Eve | -446.4 | -15.2% |
| Labor Day | -398.5 | -13.6% |
| Good Friday | -381.1 | -13.0% |
| All Saints' Day | -365.4 | -12.4% |
| Slovak National Uprising | -273.2 | -9.3% |
| Epiphany | -270.0 | -9.2% |
| Cyril and Methodius Day | -269.9 | -9.2% |
| Victory Day | -248.9 | -8.5% |
| Our Lady of Sorrows | -197.6 | -6.7% |

**Holidays reduce load by 200-715 MW** - this is 7-24% of average load!

## Implications for Modeling

1. **Add holiday features** to any load or imbalance model:
   - `is_holiday` (binary)
   - `holiday_type` (categorical)
   - `days_to_holiday` / `days_from_holiday` (bridge effects)

2. **Extended holiday periods** (Christmas week, Easter week) need special treatment

3. **DAMAS errors on holidays** likely follow different patterns - worth investigating

## Generated Files

| File | Description |
|------|-------------|
| `01_prophet_decomposition.png` | Full decomposition (observed, trend, weekly, daily, holidays) |
| `02_prophet_components.png` | Prophet's built-in component plots |
| `03_variance_decomposition.png` | Variance breakdown bar chart |
| `04_holiday_effects.png` | Holiday impact ranking |
| `05_seasonality_patterns.png` | Weekly and daily patterns |
| `06_trend_changepoints.png` | Trend with detected changepoints |
| `07_residual_analysis.png` | Residual diagnostics |
| `prophet_components.parquet` | All components as DataFrame |

## Scripts
- `prophet_decomposition.py` - Main analysis script with Slovak holidays

## Next Steps
1. Analyze DAMAS errors specifically on holidays
2. Add holiday features to day-ahead error correction model
3. Consider bridge day effects (day between holiday and weekend)
