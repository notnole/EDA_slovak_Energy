# Market Price Gap Analysis - Basic Statistics

## Overview

This analysis examines the price relationships between three Slovak electricity markets:
1. **Day-Ahead (DA)** - Hourly auction prices cleared day before delivery
2. **Intraday Market (IDM)** - Continuous trading up to delivery
3. **Imbalance** - Settlement prices for system deviations

**Data Period**: 2025-01-01 to 2025-12-31
**Records**: 8,722 hourly observations with all three prices

---

## Key Findings

### Price Levels (EUR/MWh)

| Market | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| DA Price | 104.27 | 103.91 | 55.07 | -202.70 | 563.13 |
| IDM VWAP | 102.41 | 100.35 | 57.03 | -435.14 | 721.37 |
| Imbalance | 83.96 | 81.56 | 84.02 | -2338.50 | 1697.03 |

**Observation**: IDM prices are slightly higher than DA on average (+-1.9 EUR/MWh),
while imbalance prices are lower on average (-20.3 EUR/MWh vs DA).
However, imbalance prices have significantly higher volatility (std = 84.0 vs 55.1 for DA).

### Price Spreads (EUR/MWh)

| Spread | Mean | Median | Std Dev | % Positive |
|--------|------|--------|---------|------------|
| DA - IDM | 1.85 | 1.44 | 38.64 | 54.5% |
| IDM - Imbalance | 18.46 | 18.70 | 78.58 | 69.1% |
| DA - Imbalance | 20.31 | 20.35 | 85.61 | 67.8% |

**Key Insight**: The IDM-Imbalance spread averages +18.5 EUR/MWh, meaning participants
buying in IDM typically pay more than the imbalance settlement price. This spread represents the "insurance premium"
for avoiding imbalance risk.

### Correlation with System Imbalance

| Variable | Correlation with System Imbalance (MWh) |
|----------|----------------------------------------|
| DA-IDM Spread | 0.154 |
| IDM-Imbalance Spread | 0.232 |
| DA-Imbalance Spread | 0.283 |

**Key Finding**: There is a negative correlation between system imbalance and the IDM-Imbalance spread
(r = 0.232). When the system is long (positive imbalance),
imbalance prices tend to be lower, increasing the spread. When the system is short (negative imbalance),
imbalance prices spike, reducing or inverting the spread.

---

## Temporal Patterns

### Hourly Profile
- Morning ramp (6-9h): Increased price volatility and spreads
- Midday solar hours (10-14h): Lower prices and compressed spreads
- Evening peak (17-20h): Highest prices and spread variability

### Weekday vs Weekend
- Weekend prices are lower across all markets
- Spreads are slightly smaller on weekends due to lower demand uncertainty

---

## Visualizations

1. **01_price_comparison.png** - Time series of all three market prices
2. **02_price_spreads.png** - Distribution and time evolution of spreads
3. **03_hourly_patterns.png** - Hourly and weekly patterns
4. **04_spread_vs_imbalance.png** - Relationship between spreads and system state

---

## Next Steps

1. **Feature Analysis**: Examine what drives the spreads (load forecast error, renewable generation, etc.)
2. **Predictability**: Can spreads be forecasted from available signals?
3. **Trading Implications**: Identify optimal market for different scenarios
