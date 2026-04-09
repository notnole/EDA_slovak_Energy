# Imbalance Forecasting

## Overview

Explores whether **load forecast surprise** (Actual Load - DAMAS Forecast) can predict future imbalance direction, with focus on IDM trading applications.

**Distinct from ImbalanceNowcasting**: Nowcasting uses real-time 3-min SCADA data to predict the current settlement period. Forecasting here uses hourly load surprise to predict imbalance 1-3 hours ahead.

## Key Findings

### Load Surprise -> Imbalance Link
- Overall correlation: r = -0.30 (negative: higher-than-expected load -> system SHORT)
- Day hours stronger (r = -0.31) than night (r = -0.25)
- Strongest at hour 11 (r = -0.39), weakest at hour 19 (r = -0.19)
- Effect strengthens with magnitude: |surprise| > 200 MW gives r = -0.58

### Direction Prediction
At 70% confidence thresholds:
- Load surprise < -125 MW -> 70%+ chance of POSITIVE imbalance (system long)
- Load surprise > +150 MW -> 70%+ chance of NEGATIVE imbalance (system short)
- Between -75 and +75 MW -> low confidence, don't trade

### QH Position Effect
- QH1-2 (first half of hour) more predictable than QH3-4
- TSO activates balancing reserves during the hour, diluting the signal

### IDM Arbitrage (Critical Finding)
**There is a structural IDM-Imbalance spread of +20 EUR/MWh** (Sep-Dec 2025):
- "Always sell on IDM, settle at imbalance" earns 82,374 EUR on 4,129 trades
- Prediction-based filtering REDUCES profits by excluding good trades
- The spread exists regardless of prediction direction
- Simple strategy beats any prediction-based approach

## Sub-Analyses

| Folder | Description | Key Finding |
|--------|-------------|-------------|
| [features/load_surprise/](features/load_surprise/summary.md) | Complete analysis chain | r = -0.30 load surprise vs imbalance |
| features/load_surprise/basic_analysis/ | Correlation by hour, magnitude | Peak at H11, |surprise|>200 MW |
| features/load_surprise/direction_confidence/ | Threshold calibration | 70% confidence at +/-125 MW |
| features/load_surprise/qh_position/ | QH1-4 comparison | First-half more predictable |
| features/load_surprise/predicted_surprise/ | Using nowcast H+1/2/3 predictions | Useful but weaker than realized |
| features/load_surprise/price_impact/ | Price asymmetry by direction | Deficit hours have higher prices |
| [features/load_surprise/idm_trading/](features/load_surprise/idm_trading/summary.md) | IDM arbitrage strategies | Always-sell beats prediction |

## Implication

The load surprise signal is real but small (r = -0.30, ~9% variance explained). For trading purposes, the structural IDM-Imbalance spread dominates any prediction-based edge. Focus should be on exploiting the spread directly rather than improving direction prediction.
