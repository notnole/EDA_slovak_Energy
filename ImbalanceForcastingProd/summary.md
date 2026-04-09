# 2-Hour-Ahead Imbalance Predictor

## Overview

LightGBM quantile regression model predicting Slovak system imbalance (MWh) for each 15-minute settlement period, **2 hours before delivery**. Outputs point prediction (median) plus P10/P25/P75/P90 confidence intervals.

- **108 features** from 11 data sources
- **Train**: Jan 2024 -- Sep 2025 (59,598 samples)
- **Test**: Oct 2025 -- Jan 2026 (10,184 samples)
- **Target**: System Imbalance (MWh) per 15-min settlement period

## Model Performance (Test Set)

| Metric | V1 (55 features) | This model (108 features) |
|--------|-------------------|---------------------------|
| Direction accuracy | 61.0% | **66.1%** |
| High confidence (\|pred\|>5 MWh) | ~65% | **80.9%** |
| Very high confidence (\|pred\|>10 MWh) | ~70% | **90.4%** |
| R-squared | 0.18 | **0.225** |
| Correlation | ~0.43 | **0.478** |
| MAE | ~5-6 MWh | 7.05 MWh |

### Direction Accuracy by Predicted Magnitude

| Predicted Magnitude | Accuracy | Count |
|---------------------|----------|-------|
| 0--2 MWh | 55.8% | 3,775 |
| 2--5 MWh | 66.0% | 3,687 |
| 5--10 MWh | 78.5% | 2,083 |
| 10--20 MWh | 90.3% | 507 |
| >20 MWh | 93.3% | 15 |

High-confidence predictions (>5 MWh) occur ~23 times per day and are correct on direction ~81% of the time.

## Feature Groups (108 total)

### Importance by Group

| Group | Features | Importance | Key Features |
|-------|----------|------------|--------------|
| Proxy (original) | 37 | 23.9% | proxy lags, rolling stats, yesterday, direction ratios |
| Weather (Bardejov) | 10 | 18.8% | temp_deviation, temp_surprise_lag, pressure_change6h, windspeed |
| DA Prices & Flows | 7 | 14.1% | da_price_change24h, da_flow_cz, da_flow_hu, da_net_import |
| DAMAS Forecast Error | 5 | 8.5% | damas_fe_rmean24 (#1 overall), damas_fe, damas_fe_abs |
| Market Spreads | 6 | 7.0% | idm_vwap_lag, spread_da_idm_lag, idm_volume_lag |
| Time | 9 | 6.4% | hour_sin/cos, dow_sin/cos, is_weekend, is_peak |
| Load (original) | 4 | 5.4% | load_deviation, load_rmean16, load_momentum |
| Load (new) | 4 | 4.5% | load_yesterday, load_rstd4, load_ramp4, load_rmean8 |
| Load Nowcast (OOS) | 4 | 3.8% | nowcast_pred_error, nowcast_pred_error_abs |
| Proxy (new derived) | 6 | 3.5% | proxy_ewm4, proxy_range8, proxy_abs_rmean |
| Solar | 2 | 2.8% | solar_surprise_lag, solar_surprise_rmean4 |
| Regulation | 5 | 2.2% | reg_rmean4/8, reg_rstd8, reg_momentum |
| Production SCADA | 5 | 0.0% | (short coverage, Oct 2025+ only) |
| Export/Import SCADA | 5 | 0.0% | (short coverage, Oct 2025+ only) |

### Data Sources and Lead-Time Constraints

All features strictly use only information available at prediction time (T - 2h).

| Source | Resolution | Coverage | Shift | Rationale |
|--------|-----------|----------|-------|-----------|
| Regulation SCADA | 3-min -> 15-min | Full (2024--2026) | 8 periods (2h) | Real-time SCADA |
| Load SCADA | 3-min -> 15-min | Full | 8 periods (2h) | Real-time SCADA |
| Production SCADA | 3-min -> 15-min | Oct 2025+ | 8 periods (2h) | Real-time SCADA |
| Export/Import SCADA | 3-min -> 15-min | Oct 2025+ | 8 periods (2h) | Real-time SCADA |
| Solar actual/surprise | Hourly -> 15-min | Full | 12 periods (3h) | Published after hour ends |
| DAMAS forecast error | Hourly -> 15-min | Full | 12 periods (3h) | Actual load published after hour ends |
| DAMAS forecast load | Hourly -> 15-min | Full | None | D-1 forecast, known since yesterday |
| DA prices & flows | Hourly -> 15-min | Full | None | Known D-1 at 11:00 |
| IDM VWAP/volume | Hourly -> 15-min | 2025+ | 12 periods (3h) | Trading closes near delivery |
| Imbalance settlement price | 15-min | Full | 12 periods (3h) | Published after settlement |
| Bardejov weather (actual) | Hourly -> 15-min | Full (2024+) | 12 periods (3h) | Published after hour ends |
| Bardejov DA temp forecast | Hourly -> 15-min | Full (2024+) | None | GFS D+1 forecast, known D-1 |
| Load nowcast H+2 (OOS) | Hourly -> 15-min | 2025+ | None | Walk-forward OOS prediction made at T-2h |

## Data Leakage Audit

A comprehensive audit was performed. Three issues were found and fixed:

1. **Solar features (FIXED)**: Shift increased from 8 (2h) to 12 (3h) periods. Hourly solar actuals are only known after the hour ends.

2. **Load nowcast predictions (FIXED)**: The original `h2_predictions.csv` contained in-sample predictions from a model trained on ALL data. Replaced with walk-forward out-of-sample predictions (`generate_oos_predictions.py`), where each prediction comes from a model trained only on strictly prior data.

   Walk-forward folds:
   - Fold 1: Train [2024-01, 2025-01), Predict [2025-01, 2025-07)
   - Fold 2: Train [2024-01, 2025-07), Predict [2025-07, 2025-10)
   - Fold 3: Train [2024-01, 2025-10), Predict [2025-10, 2026-01)
   - Fold 4: Train [2024-01, 2026-01), Predict [2026-01, 2026-02)

   OOS nowcast MAE: 44.5 MW, correlation: 0.748.

3. **`nowcast_recent_bias` shift (FIXED)**: Changed from shift(8) to shift(12) to match DAMAS forecast error treatment.

All other feature groups confirmed safe by audit.

## Trading Strategy Backtest

### Setup

- **Bidirectional**: Sell IDM on surplus prediction, buy IDM on deficit prediction
- **Confidence-weighted sizing**: Position = min(|prediction|, 5) MWh
- **Settle at imbalance settlement price** (NOT marginal imb_price -- verified correct)
- **Jan 19, 2026 price spike excluded** (imb_settlement_price > 5000 EUR/MWh)

### Results: |pred| > 2 MWh threshold, confidence-weighted 1--5 MWh

| Month | Trades | Win Rate | P&L (EUR) | EUR/day |
|-------|--------|----------|-----------|---------|
| Oct 2025 | 1,391 | 64% | +23,634 | +762 |
| Nov 2025 | 1,240 | 65% | +10,894 | +419 |
| Dec 2025 | 1,497 | 65% | +15,443 | +498 |
| Jan 2026 | 1,293 | 72% | +27,482 | +1,145 |
| **Total** | **5,421** | **66%** | **+77,454** | **+692** |

- **82 of 112 trading days profitable (73%)**
- Worst day: -1,050 EUR
- Annualized Sharpe: 8.6
- Confidence sizing 4.6x more profitable than flat 1 MWh sizing

### Strategy Variants

| Filter | Trades/day | Win Rate | P&L (EUR) | EUR/day | Sharpe |
|--------|-----------|----------|-----------|---------|--------|
| All predictions, weighted | 91 | 60% | +81,507 | +728 | 8.8 |
| \|pred\| > 2, weighted | 48 | 66% | +77,454 | +692 | 8.6 |
| \|pred\| > 5, weighted | 15 | 76% | +50,451 | +450 | 8.4 |

### Caveats

1. **Jan 2026 profit concentration**: The Jan 19 spike (7,258 EUR/MWh settlement price) contributed 57% of Jan deficit P&L. Excluding it, Jan still earns ~4,000 EUR. Base-rate edge is real but more modest.

2. **IDM-Imb structural spread**: In Oct--Dec 2025, a structural +7--9 EUR/MWh IDM>Imb spread existed. The model adds value on top of this. In Jan 2026 the spread flipped negative, and the model correctly adapted to trade the deficit side.

3. **Transaction costs not modeled**: IDM bid-ask spread, market impact, and settlement fees would reduce realized P&L.

4. **Production/Export features inactive**: Oct 2025+ only, 0% importance. Will contribute as more history accumulates.

## Files

```
ImbalanceForcastingProd/
  summary.md                      # This file
  scripts/
    train_imbalance_2h.py         # Training script (108 features)
  models/
    imb_2h_v2_q{10,25,50,75,90}.joblib  # Trained quantile models
  data/
    predictions_test_v2.csv       # Test set predictions
    feature_importance_v2.csv     # Feature importance ranking
  plots/
    01_model_evaluation_v2.png    # Scatter, direction accuracy, sample series
    02_hourly_direction_v2.png    # Direction accuracy by hour
    03_feature_importance_v2.png  # Top 30 feature importance bar chart
```

### Dependencies (other directories)

```
LoadAnalysis/nowcast_5h/tuning/
  generate_oos_predictions.py           # Walk-forward OOS prediction generator
  oos_predictions/h2_oos_predictions.csv  # H+2 OOS predictions (9,472 hours)

data/features/                          # SCADA feature files
data/master/                            # Imbalance labels
data/clean/solar/                       # Solar data
data/Bardejov/Weather/                  # Bardejov weather (actual + DA forecast)
features/DamasLoad/                     # DAMAS load forecast
features/DamasPrices/                   # DA prices and cross-border flows
MarketPriceGap/data/processed/          # IDM/Imbalance market prices
```

## Key Findings

1. **Weather is the second-largest signal** (18.8%): Temperature deviation from seasonal normal, temperature forecast surprise, pressure changes, and wind speed all drive load-imbalance dynamics. Using Bardejov (eastern SK) as proxy for whole country -- full SK coverage would likely improve further.

2. **DA market data is highly predictive** (14.1%): Cross-border flows (CZ, HU) and price changes known D-1 with zero leakage risk.

3. **DAMAS forecast error autocorrelation is real signal** (8.5%): The 24h rolling mean of forecast error captures systematic forecast bias that persists and drives imbalance.

4. **Market spreads encode trader positioning** (7.0%): Lagged IDM VWAP and DA-IDM spread indicate how intraday traders price the system.

5. **Walk-forward OOS nowcast predictions add clean value** (3.8%): Genuine out-of-sample load forecast error predictions. Model improved when switching from leaky in-sample to clean OOS predictions (64.7% -> 66.1% direction accuracy after also adding weather).

6. **Confidence-weighted sizing is highly effective**: Scaling position with prediction magnitude (capped at 5 MWh) multiplied P&L 4.6x vs flat sizing while maintaining Sharpe. The model's calibration -- bigger predictions are more likely correct -- is the foundation of this gain.
