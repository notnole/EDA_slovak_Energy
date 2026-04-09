# Hub-to-Hub Cross-Border Capacity vs IDM Price Analysis

## Overview

Analysis of how cross-border interconnection capacity affects Slovak IDM prices.
- **Period**: 2025-11-26 to 2026-02-27 (94 days)
- **Observations**: 2214 hourly data points with valid prices
- **IDM Price**: mean 142.2, median 124.8 EUR/MWh
- **Cross-border areas**: CEPS (CZ), APG (AT), MAVIR (HU), PSE (PL), DE (50HzT+TTG avg)

## Key Findings

### Correlations (Import Capacity vs IDM Price)

| Area | Pearson r | Spearman r |
|------|-----------|------------|
| CEPS (CZ) | -0.055 | -0.229 |
| APG (AT) | -0.100 | -0.235 |
| MAVIR (HU) | -0.029 | -0.061 |
| PSE (PL) | -0.109 | -0.254 |
| DE (avg) | +0.012 | -0.100 |
| **Total Import** | **-0.075** | **-0.206** |

### Congestion Premium

| Border | Open Median | Congested Median | Premium | % Congested |
|--------|-------------|-----------------|---------|-------------|
| CEPS (CZ) | 114.0 | 143.2 | +29.2 EUR/MWh | 32% |
| APG (AT) | 117.7 | 145.1 | +27.4 EUR/MWh | 19% |
| MAVIR (HU) | 124.1 | 130.9 | +6.9 EUR/MWh | 7% |
| PSE (PL) | 115.0 | 141.0 | +26.0 EUR/MWh | 33% |
| DE (avg) | 113.4 | 129.0 | +15.6 EUR/MWh | 67% |

### Price by Number of Congested Borders

| # Congested | Median Price | Mean Price | Count |
|-------------|-------------|------------|-------|
| 0 | 107.2 | 118.1 | 545 |
| 1 | 114.2 | 133.8 | 734 |
| 2 | 128.9 | 153.4 | 402 |
| 3 | 153.8 | 169.5 | 315 |
| 4 | 160.7 | 189.1 | 104 |
| 5 | 132.4 | 155.0 | 114 |

### OLS Regression (price ~ h2h features + hour dummies)

- **R-squared**: 0.063 (h2h capacity explains ~6% of IDM price variance beyond hour-of-day)
- Most significant feature: **congested_in_count** (coef=+11.4, p<0.001) - each additional congested border adds ~11.4 EUR/MWh
- **PSE import capacity** significant (coef=-0.024, p=0.004) - 100 MW more PSE import cap reduces price by ~2.4 EUR/MWh
- **DE import capacity** positive coefficient (+0.034, p<0.001) - counterintuitive, likely confounded with time-of-day

### Interpretation

1. **The relationship is nonlinear** (threshold effect): Spearman correlations are 2-4x stronger than Pearson, confirming that congestion acts as a binary switch rather than a continuous gradient. When a border goes from "some capacity" to "zero capacity", the price impact is abrupt.

2. **PSE (Poland) and APG (Austria) borders matter most** for SK IDM prices (strongest rank correlations). MAVIR (Hungary) has minimal impact despite being a direct neighbor - likely because HU-SK congestion is rare (only 7% of hours).

3. **Cumulative congestion effect is strong**: Moving from 0 to 4 congested borders raises median price by +53.5 EUR/MWh (107.2 -> 160.7). This is a large, tradeable signal.

4. **DE (indirect) border congestion is the norm** (67% of hours) and provides a baseline +15.6 EUR/MWh premium. It's less useful as a signal because it's almost always congested.

5. **Low R-squared is expected**: Cross-border capacity is just one of many price drivers (fundamentals, weather, generation mix, DA-IDM arbitrage). The congestion premium is real and significant but coexists with many other factors.

6. **The hourly profile plot (05) reveals anti-correlation**: Price peaks (morning 6-8h, evening 16-18h) coincide with import capacity troughs, consistent with Slovakia importing during peak demand hours and running into capacity constraints.

## Plots

1. `01_correlation_matrix.png` - Full correlation heatmap
2. `02_price_vs_capacity.png` - Per-area scatter with regression
3. `03_congestion_impact.png` - Box plots: congested vs open borders
4. `04_timeseries_sample.png` - Weekly overlay of price + capacity
5. `05_total_capacity_effect.png` - Aggregate capacity effects + hourly profile
---

## H2H Capacity vs System Imbalance

**Overlap period**: 2025-11-26 to 2026-01-24 (60 days, 1424 hourly obs)
- Mean imbalance: -1.4 MWh/h
- Deficit hours: 50.4%

### Correlations (Import Capacity vs Imbalance Volume)

| Area | Pearson r | Spearman r |
|------|-----------|------------|
| CEPS (CZ) | -0.104 | -0.039 |
| APG (AT) | +0.002 | +0.063 |
| MAVIR (HU) | -0.026 | -0.044 |
| PSE (PL) | +0.040 | +0.098 |
| DE (avg) | -0.145 | -0.006 |
| **Total Import** | **-0.066** | **+0.002** |

### Deficit Rate by Congestion

| Border | Deficit % Open | Deficit % Congested | |Imb| Open | |Imb| Congested |
|--------|---------------|--------------------|-----------|--------------------|
| CEPS (CZ) | 51.1% | 48.6% | 21.6 | 18.0 |
| APG (AT) | 49.8% | 54.4% | 20.9 | 17.9 |
| MAVIR (HU) | 50.8% | 37.7% | 20.5 | 18.7 |
| PSE (PL) | 48.0% | 55.5% | 19.9 | 21.1 |
| DE (avg) | 49.7% | 50.7% | 21.9 | 19.8 |

### |Imbalance| by Number of Congested Borders

| # Congested | Mean Imb | |Mean| | |Median| | |P90| | Deficit % | n |
|-------------|----------|--------|---------|-------|-----------|---|
| 0 | +4.1 | 30.8 | 20.8 | 81.0 | 45.2% | 325 |
| 1 | -3.6 | 31.4 | 22.3 | 67.9 | 51.6% | 516 |
| 2 | +0.7 | 27.6 | 18.0 | 61.2 | 53.4% | 298 |
| 3 | -7.9 | 33.1 | 21.5 | 79.6 | 53.3% | 199 |
| 4 | -7.8 | 30.8 | 18.5 | 81.6 | 51.8% | 56 |
| 5 | +11.2 | 21.0 | 13.1 | 60.1 | 33.3% | 30 |

### Imbalance Plots

6. `06_h2h_imbalance_correlation.png` - Correlation matrix: capacity vs imbalance
7. `07_h2h_imbalance_scatter.png` - Per-area scatter: import cap vs imbalance
8. `08_h2h_imbalance_congestion.png` - Box plots: imbalance by congestion state
9. `09_h2h_imbalance_timeseries.png` - Weekly overlay: imbalance + capacity + congestion count
10. `10_h2h_imbalance_deficit.png` - Deficit probability and magnitude analysis