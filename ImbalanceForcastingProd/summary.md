# Imbalance Forecasting — Production System

## Overview

Trading system for the Slovak electricity market that predicts the IDM-to-settlement price spread and trades on the intraday (IDM) market 2 hours before delivery, settling at the imbalance settlement price.

**Core insight**: Instead of predicting system imbalance (MWh) and hoping it translates to profit, we predict the actual **price spread** between where we can trade (IDM) and where we'll settle (OKTE imbalance settlement price). This directly predicts P&L.

## Architecture

### 3-Stage Stacked Model

```
Stage 1: Load Nowcast (H+2)
    Input: 3-min SCADA (load, regulation), DAMAS load forecast
    Output: OOF predicted load forecast error
    Script: LoadAnalysis/nowcast_5h/tuning/generate_oos_predictions.py

Stage 2: Imbalance Model (108 features, LightGBM MAE)
    Input: SCADA, weather, DA prices, market spreads, Stage 1 OOF
    Output: OOF imbalance direction + magnitude prediction
    Script: scripts/training/train_multi_lead.py

Stage 3: Spread Model (116 features, LightGBM MAE)
    Input: All Stage 2 features + Stage 2 OOF imbalance prediction
    Target: imb_settlement_price - IDM_mid_at_T_minus_120min
    Output: Predicted spread = the trading signal
    Script: scripts/training/train_stacked_model.py
```

Each stage produces walk-forward out-of-fold predictions to avoid leakage. Stage N's OOF from prior folds becomes a feature for Stage N+1's training.

### Execution

- **Timing**: Place orders at T-120min (2h before delivery)
- **Products**: 15-min QH IDM products on OKTE/XBID
- **Gate closure**: T-60min — spreads explode after this, must execute before
- **Spread filter**: Only trade when bid-ask spread <= 10 EUR/MWh

## Performance (Feb-Mar 2026, True Out-of-Sample)

All results use real bid/ask execution prices from the DB_EMS order book.

### Model Comparison

| Model | EUR/day | Sharpe | Max Drawdown | Worst Day | Prof Days |
|-------|---------|--------|-------------|-----------|-----------|
| Imbalance only | +150 | 7.1 | -925 | -755 | 67% |
| Spread only (standalone) | +486 | 12.8 | -560 | -560 | 84% |
| **Stacked (3-stage)** | **+449** | **15.5** | **-466** | **-466** | **84%** |
| Spread + imb filter | +339 | 13.1 | -481 | -481 | 80% |

### Stacked Model Risk Profile

- Max drawdown: **-466 EUR** (1.0 days of profit)
- Max consecutive losing days: **1**
- Losing weeks: **0 out of 9**
- Worst week: **+619 EUR** (still positive)
- Recovery from max drawdown: **4 days**

### Monthly Consistency

| Month | EUR/day | Win Rate |
|-------|---------|----------|
| Feb 2026 | +372 | 60% |
| Mar 2026 | +525 | 62% |

## Features (113 base + 3 stacking = 116 total)

### Feature Groups by Importance

| Group | Features | Importance | Key Signals |
|-------|----------|------------|-------------|
| Proxy (regulation) | 43 | 27% | Regulation lags, momentum, rolling stats, direction ratios |
| Weather (5-city) | 10 | 19% | temp deviation, forecast surprise, pressure, wind, radiation |
| DA Prices & Flows | 7 | 14% | Cross-border flows (CZ, HU), price changes, demand/supply |
| DAMAS Forecast Error | 5 | 9% | 24h rolling mean of load forecast error (top individual feature) |
| Market Spreads | 6 | 7% | Lagged IDM VWAP, DA-IDM spread, IDM volume |
| Time | 9 | 6% | Hour, day-of-week, weekend, peak indicator |
| Load | 8 | 10% | Load deviation, momentum, rolling stats, yesterday |
| Load Nowcast (Stage 1) | 4 | 4% | Walk-forward OOF H+2 load error predictions |
| Solar | 2 | 3% | Solar surprise (actual - DA forecast) |
| Stacking (Stage 2 OOF) | 3 | ~1% | Imbalance prediction, |pred|, direction |
| Production/Export SCADA | 10 | 0% | Short coverage (Oct 2025+), will improve |

### Data Sources and Lead-Time Constraints

| Source | Shift | Rationale |
|--------|-------|-----------|
| SCADA (regulation, load, prod, export) | lead+1 periods | 15-min period completes at T+15min |
| Solar actual, DAMAS actual, IDM/Imb prices | lead+4 periods | Hourly, published after hour ends |
| DA prices, DAMAS forecast, DA temp forecast | None | Known D-1 |
| Weather actuals (Bardejov 5-city) | lead+4 periods | Hourly, published after hour ends |

### Leakage Audit

Comprehensive audit performed. Three issues found and fixed:
1. Solar shift: 8 -> 12 periods (hourly publication delay)
2. Load nowcast: Replaced in-sample with walk-forward OOF predictions
3. SCADA 15-min: shift(lead) -> shift(lead+1) (period not complete at start)

All OB features confirmed clean — no correlation with execution prices.

## Execution Methods Tested

| Method | Description | Result |
|--------|-------------|--------|
| VWAP | Assume execution at volume-weighted average price | +825/day (unrealistic) |
| Market taker (bid/ask) | Hit best bid/ask at T-120min | +486/day with spread filter |
| Market maker (limit) | Place limit at top of book, wait for fill | +117/day, 71% fill rate |

The VWAP assumption overstates P&L by ~70%. Bid/ask execution with a spread filter is the realistic baseline.

## Calibration

P&L-based live calibration adjusts independently for surplus and deficit:
- **Prediction threshold**: Raised when average P&L/trade goes negative, lowered when profitable
- **Position sizing**: Scaled 0.3x-1.5x based on recent profitability
- **Morning reset**: 7-day lookback sets initial thresholds
- **Mid-day recalibration**: Every 15 trades per side, based on rolling P&L

## Project Structure

```
ImbalanceForcastingProd/
    summary.md                          # This file
    data_refresh.md                     # Data source checklist

    scripts/
        training/
            train_imbalance_2h.py       # Single-lead (Lead 8) imbalance model
            train_multi_lead.py         # Multi-lead (4-8) with shared feature engineering
            train_stacked_model.py      # 3-stage stacked: load -> imbalance -> spread

        backtests/
            backtest_realistic.py       # Bid/ask execution vs VWAP comparison
            backtest_limit_orders.py    # Limit order fill simulation from DB_EMS
            backtest_production.py      # Weekly retrain + P&L calibration
            backtest_dual_model.py      # Imbalance + spread ensemble strategies

        data_extraction/
            extract_orderbook_features.py    # 60-min OB features (legacy)
            extract_orderbook_qh.py          # QH OB features at multiple lead times
            extract_intraday_ob_features.py  # Intraday pressure + IDM-DA spread

        analysis/
            test_rmse_quantile.py       # RMSE vs MAE, quantile bands, asymmetric thresholds
            test_spread_prediction.py   # Spread prediction model discovery
            check_spread_leakage.py     # Spread model leakage audit
            plot_backtest.py            # Trading dashboard visualization

    data/
        features/                       # Extracted feature files
            orderbook_features.csv      # 60-min OB (legacy)
            orderbook_qh_features.csv   # QH OB at 5 lead times (330k rows)
            intraday_ob_features.csv    # IDM-DA spread + market pressure (65k rows)
            feature_importance_v2.csv   # Model feature rankings

        predictions/                    # Model output predictions
            predictions_lead{4-8}.csv   # Multi-lead imbalance predictions
            predictions_test_v2.csv     # Single-lead test predictions
            stacked_test_predictions.csv # 3-stage stacked predictions

        backtests/                      # Trade-level backtest results
            backtest_realistic.csv      # Bid/ask execution trades
            backtest_limit_orders.csv   # Limit order fill simulation
            backtest_production_*.csv   # Production simulation results

    models/
        imbalance/                      # Trained LightGBM models
            imb_2h_v2_q{10,25,50,75,90}.joblib  # Single-lead quantile models
            imb_lead{4-8}_q{10,50,90}.joblib     # Multi-lead models

    plots/
        model_evaluation/               # Prediction accuracy plots
        trading/                        # Trading P&L dashboard
```

## Dependencies

### Data Sources (refreshed through April 2026)

| Source | Location | Script |
|--------|----------|--------|
| SCADA (4 signals) | `data/features/*_3min.csv` | `data/features/clean_features.py` |
| Imbalance labels | `data/master/master_imbalance_data.csv` | `data/master/create_master_imbalance.py` |
| DAMAS load | `features/DamasLoad/load_data.csv` | `features/DamasLoad/process_load_data.py` |
| DA prices | `features/DamasPrices/data/da_prices.csv` | `features/DamasPrices/process_da_prices.py` |
| Solar | `data/clean/solar/solar_hourly.csv` | `data/clean/solar/clean_solar_data.py` |
| Market prices | `MarketPriceGap/data/processed/hourly_market_prices.csv` | `MarketPriceGap/scripts/load_market_prices.py` |
| Weather (5-city) | `data/Bardejov/Weather/slovakia_multi_city_weather.csv` | Open-Meteo API |
| Load nowcast OOF | `LoadAnalysis/nowcast_5h/tuning/oos_predictions/` | `generate_oos_predictions.py` |
| Order book | `DB_EMS` (localhost:5432) | `scripts/data_extraction/*.py` |

### External Services

| Service | Purpose | Tier |
|---------|---------|------|
| Open-Meteo API | Weather actuals + DA forecast (5 cities) | Free (10k calls/day) |
| DB_EMS (PostgreSQL) | IDM order book bid/ask (99M rows) | Local |
| beam-solar (PostgreSQL) | Live SCADA + predictions | Local |

## Key Findings

1. **Predict the spread, not the imbalance.** The IDM-to-settlement spread is the actual P&L signal. The spread model outperforms the imbalance model 3x (+486 vs +150/day) because it directly optimizes for what makes money.

2. **3-stage stacking improves risk, not return.** Adding imbalance OOF as a feature to the spread model gives the best Sharpe (15.5) and smallest drawdown (-466 EUR), at a modest P&L cost (-37/day vs standalone).

3. **Execution costs matter enormously.** VWAP backtests overstate P&L by 70%. Bid/ask execution with a spread <= 10 filter is the realistic baseline. Execute at T-120min where spreads are liquid (median 2.1 EUR/MWh).

4. **Weather is the second most important signal (19%).** The Bardejov-only weather (0.7% importance) became 19% with 5-city national coverage. Temperature deviation, forecast surprise, and pressure changes all drive load-imbalance dynamics.

5. **Weekly retraining helps the imbalance model but the spread model is more stable.** The spread model's performance is consistent across months with or without retraining. Live calibration of thresholds and position sizing adds more value than frequent retraining.

6. **The IDM order book is valuable for execution but not prediction.** OB features as model inputs hurt production P&L because they correlate with spread costs. Use the order book for execution decisions (spread filter, limit orders) only.
