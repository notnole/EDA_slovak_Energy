# Day-Ahead Market Decision Matrix

## Objective

Build decision signals from available forecasts (load, solar, temperature) and recent market history for day-ahead trading. Three targets:
1. **DA vs IDM spread direction**: Will DA be more or less expensive than IDM?
2. **Peak timing**: When will the highest price occur?
3. **Volatility**: Will it be a volatile day?

DA price forecasts were excluded -- previously shown to be worse than a 7-day naive baseline.

## Strict Causality

All features are **DA-available only** -- information known before DA auction gate closure (~10:00 D-1):

| Category | What we have | What we DON'T have |
|----------|-------------|---------------------|
| **Forecasts for D+1** | Load shape (DAMAS), solar shape (TSO), temperature (GFS) | -- |
| **Yesterday's realized** | Prices (DA, IDM, spread), production mix, flows, load, volatility | Same-day production, flows, actual load |
| **Rolling averages** | 7-day lagged means of prices, load, volatility | -- |
| **Calendar** | Day of week, weekend flag, month | -- |

Same-day production (nuclear, gas, hydro), flows (CZ/PL/HU), and actual load/solar/temperature are **dropped after lag computation** -- they are only used to compute yesterday's values and 7-day rolling averages.

## Data

Six data sources merged to one hourly master (25,010 rows, 40 columns):

| Source | Period |
|--------|--------|
| DA + IDM prices + spreads | Jan 2024 - Feb 2026 |
| Load actual + forecast (DAMAS) | Jan 2024 - Jan 2026 |
| Production by type (14 series, EDA format) | Jan 2024 - Feb 2026 |
| Temperature forecast + actual (GFS) | Sep 2025 - Feb 2026 |
| Cross-border flows CZ/PL/HU | Jan 2024 - Jan 2026 |

**Full overlap period**: Sep 15, 2025 to Jan 25, 2026 (9,827 hours / 139 days).
40 DA-available features + 15 targets = 55 columns at daily level.

## Key Findings

### 1. DA-IDM Spread -- The Actionable Signal

**Rule accuracy: 72.2% (baseline 54.1%, +18% skill)**

The spread is strongly predictable from two features:

| Condition | Mean Spread | Direction | Rule Accuracy | N |
|-----------|-------------|-----------|---------------|---|
| **Weekend** | **+28.8 EUR/MWh** | DA > IDM | **89%** | 38 |
| Weekday + High Load Dev | -15.7 EUR/MWh | DA < IDM | 66% | 65 |
| Weekday + Low Load Dev | +1.7 EUR/MWh | Weak signal | 63% | 30 |

**Interpretation**: On weekends, the DA auction systematically overprices compared to IDM by ~29 EUR/MWh on average. This is the strongest and most consistent signal in the dataset. On high-load weekdays, the pattern reverses -- DA tends to be cheaper than IDM by ~16 EUR.

**Top predictors for spread** (Pearson r):
- `day_of_week` (r = +0.49) -- weekend effect
- `load_fcst_vs_7d` (r = -0.46) -- higher-than-normal load pushes DA down vs IDM
- `is_weekend` (r = +0.43)
- `spread_yesterday` (r = +0.41) -- persistence
- `gas_yesterday` (r = -0.31) -- more gas generation yesterday correlates with DA < IDM

### 2. Peak Price Timing -- Nuclear is the Key

**Correlation: yesterday's nuclear output vs peak hour = r = -0.87**

This is the single strongest relationship in the dataset. When yesterday's nuclear output drops below ~2,350 MW (vs normal ~2,450 MW), the peak shifts from morning (H4-H7) to evening (H17-H19). Nuclear maintenance schedules are known ahead, so yesterday's nuclear is a strong proxy for today's.

| Yesterday Nuclear Output | Mean Peak Hour | Peak in Evening | N |
|--------------------------|----------------|-----------------|---|
| < median - 100 MW | ~19:00 | 84% | 19 |
| > median | ~4:00 | 2% | 120 |

However, the threshold-based rule only achieves 71% accuracy -- below the 86% baseline of always predicting "morning peak." Evening peaks are rare (14% of days) and the rule over-predicts them. The nuclear-timing relationship is real but needs more data or a continuous model to exploit.

**Other timing drivers** (DA-available, lagged):
- `load_actual_7d` (r = -0.58) -- lower recent load trend implies later peak
- `da_price_7d` (r = -0.45) -- lower recent prices correlate with later peak

### 3. Volatility -- Hard to Predict

**Rule accuracy: 66.9% (baseline 86.8%, negative skill)**

High volatility days (std > 40 EUR/MWh) are rare (13% of days) and poorly predicted by simple rules. Yesterday's volatility is the best predictor (r = 0.45) but with only 21% precision, the rule generates too many false positives.

**Top predictors** (all DA-available, lagged):
- `da_range_yesterday` (r = +0.49)
- `da_volatility_yesterday` (r = +0.45)
- `da_max_yesterday` (r = +0.40)
- `net_import_yesterday` (r = +0.36) -- net import position as volatility driver

Extreme price events (>200 or <10 EUR/MWh, 33% of days) are slightly better predicted from yesterday's volatility metrics (72.8% vs 67.6% baseline).

### 4. DA Price Level Predictors

While we're not forecasting DA price directly, the top correlates are informative:

| Feature | r with DA Price |
|---------|----------------|
| `idm_vwap_7d` | +0.61 |
| `load_actual_7d` | +0.58 |
| `da_price_yesterday` | +0.54 |
| `load_fcst_offpeak` | +0.53 |
| `da_price_7d` | +0.51 |
| `temp_fcst_mean` | -0.50 |

Price level is mostly persistence + load-driven. The 7-day IDM VWAP is the single best predictor (r = 0.61), confirming that recent market conditions dominate over forecast-based features.

### 5. Demand Pressure Composite

A composite index (z-scored load deviation + temperature deviation + solar deviation) correlates with DA price at r ~ 0.35-0.40. This is modest but consistent: days where load is above normal, temperature is below normal, AND solar is below normal tend to have higher DA prices. The index adds marginal value over individual features.

## Decision Matrix

```
+---------------------------------------------------------------+
|  CONDITION               |  SPREAD   |  PEAK     |  VOLATILITY|
+---------------------------------------------------------------+
|  WEEKEND                 |  DA > IDM |  Morning  |  Low       |
|    (any load/temp)       |  +25 EUR  |  H7-H9    |            |
+---------------------------------------------------------------+
|  WEEKDAY + HIGH LOAD     |  DA < IDM |  Morning  |  Check vol |
|    (load dev > 85 MW)    |  -15 EUR  |  H6-H8    |  history   |
+---------------------------------------------------------------+
|  WEEKDAY + LOW LOAD      |  DA > IDM |  Morning  |  Low       |
|    (load dev < 85 MW)    |  +10 EUR  |  H7-H9    |            |
+---------------------------------------------------------------+
|  LOW NUCLEAR YESTERDAY   |  --       |  Evening  |  High      |
|    (< 2361 MW)           |           |  H17-H19  |            |
+---------------------------------------------------------------+
|  HIGH VOL YESTERDAY      |  --       |  --       |  Elevated  |
|    (std > 41.8 EUR)      |           |           |            |
+---------------------------------------------------------------+
```

All conditions use only DA-available information (forecasts + yesterday's realized values).

## Limitations

1. **Small sample**: 139 days is marginal for rule extraction. All results should be considered preliminary and validated on new data.
2. **No out-of-sample test**: With only 139 days, a proper train/test split would leave too few days in each set. These are in-sample statistics.
3. **Spread signal may not persist**: The weekend DA-IDM spread pattern could change if market participants adapt.
4. **Nuclear as proxy**: Yesterday's nuclear output is used as a proxy for today's (maintenance schedules are persistent). Direct nuclear schedule data would be better.
5. **Solar forecast limited**: Only 5.5 months of solar forecast data (Aug 2025+). Winter months have minimal solar impact.

## Files

| File | Description |
|------|-------------|
| `data/da_master_hourly.csv` | Merged hourly master (25,010 rows x 40 cols) |
| `data/da_daily_features.csv` | Daily feature matrix for overlap period (139 days x 55 cols) |
| `data/da_daily_features_full.csv` | Full daily features (765 days, NaN where signals missing) |
| `plots/01_correlation_heatmap.png` | Feature-target correlation matrix |
| `plots/02_deviation_analysis.png` | Forecast deviations vs DA price outcomes |
| `plots/03_regime_analysis.png` | Yesterday's supply mix, temperature, weekday/weekend regimes |
| `plots/04_spread_drivers.png` | DA-IDM spread by temperature, load, day-of-week |
| `plots/05_decision_rules.png` | Backtest results and decision rule summary |
| `scripts/01_assemble_master.py` | Data assembly (6 sources, EDA parsing) |
| `scripts/02_feature_engineering.py` | 55 columns (40 DA-available features + 15 targets) |
| `scripts/03_analysis.py` | Correlation, deviation, regime, spread analysis |
| `scripts/04_decision_rules.py` | Rule extraction and backtest |
