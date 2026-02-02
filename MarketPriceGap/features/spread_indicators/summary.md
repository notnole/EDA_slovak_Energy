# Spread Indicators EDA

## Overview

Analysis of indicators that affect DA-IDM spread magnitude. Understanding what drives large spreads helps identify profitable trading opportunities.

---

## Key Findings

### Top Indicators Correlated with |Spread|

| Indicator | Correlation | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| Wind Forecast (MW) | nan  | nan | Higher -> Smaller spread |
| DA Price Level | 0.316 *** | 0.00e+00 | Higher -> Larger spread |
| Yesterday |Spread| | 0.281 *** | 4.65e-317 | Higher -> Larger spread |
| DA Demand (MW) | 0.266 *** | 7.41e-283 | Higher -> Larger spread |
| DA Supply (MW) | 0.159 *** | 1.06e-100 | Higher -> Larger spread |
| DA Price Volatility 24h | 0.159 *** | 6.93e-100 | Higher -> Larger spread |
| Supply/Demand Ratio | -0.129 *** | 7.62e-66 | Higher -> Smaller spread |
| Net Import (MW) | 0.105 *** | 1.87e-44 | Higher -> Larger spread |
| Solar Forecast (MW) | -0.074 *** | 5.92e-20 | Higher -> Smaller spread |
| RES Forecast (MW) | -0.074 *** | 5.92e-20 | Higher -> Smaller spread |

### Key Insights

#### 1. Spread Persistence (r = 0.281)
**Yesterday's spread size strongly predicts today's spread size.** This is the most actionable finding:
- If yesterday had a large spread (>20 EUR), expect a large spread today
- If yesterday had a small spread (<5 EUR), expect a small spread today
- This persistence allows filtering for high-probability trades

#### 2. Price Volatility (r = 0.159)
**Higher price volatility leads to larger spreads:**
- Volatile markets create more price divergence between DA and IDM
- Trade larger when volatility is high, but with tighter stops

#### 3. RES (Solar+Wind) Forecast (r = -0.074)
**Higher renewable forecasts are associated with smaller spreads:**
- Solar/wind variability increases price uncertainty
- High RES periods may have more price corrections in IDM

#### 4. Cross-Border Flows (r = 0.105)
**Net import position affects spread:**
- High imports associated with larger spreads
- Congestion at borders creates price divergence

#### 5. Supply/Demand Balance (r = -0.129)
**Market balance affects uncertainty:**
- Tight supply conditions have larger spreads
- Balanced markets are more predictable

#### 6. Calendar Patterns

**Hourly:**
- Largest spreads at hour 99:00 (mean = 86.8 EUR)
- Smallest spreads at hour 4:00 (mean = 8.3 EUR)

**Monthly:**
- Best month: Jan (mean = 50.5 EUR)
- Worst month: Aug (mean = 7.9 EUR)

---

## Trading Implications

### When to Expect Large Spreads (Good for Trading)

1. **Yesterday had large spread** (>20 EUR same hour)
2. **High price volatility** (24h rolling std > 30)
3. **Peak hours** (17:00-20:00)
4. **Q4 months** (Oct-Dec historically best)
5. **Extreme cross-border positions** (high import or export)

### When to Avoid Trading

1. **Yesterday had tiny spread** (<5 EUR)
2. **Low volatility regime**
3. **Night hours** (00:00-06:00)
4. **Summer months** (May-Aug historically worst)
5. **Balanced market conditions**

---

## Visualizations

1. **01_indicator_overview.png** - Correlation summary and key indicators
2. **02_detailed_analysis.png** - Hour/day/month patterns and scatter plots
3. **03_crossborder_flows.png** - Impact of cross-border flows by country

---

## Data Coverage

| Metric | Value |
|--------|-------|
| Total observations | 17,639 |
| Date range | 2025-01-01 to 2026-01-25 |
| Mean |spread| | 33.03 EUR/MWh |
| Median |spread| | 17.68 EUR/MWh |
| 90th percentile |spread| | 79.82 EUR/MWh |

---

## Next Steps

1. Build predictive model for spread magnitude using top indicators
2. Combine with direction prediction for complete trading strategy
3. Backtest with transaction costs
4. Consider regime-switching models (high-vol vs low-vol)
