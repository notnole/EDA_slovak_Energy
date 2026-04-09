# Data Refresh Checklist (through April 2026)

Target: extend all data sources to **April 9, 2026** for out-of-sample testing of the multi-lead imbalance predictor on Feb--Apr 2026.

## Priority 1: Core SCADA + Labels

- [x] **Regulation SCADA** — `data/features/regulation_3min.csv`
  - Done: 390,351 rows, 2024-01-01 to 2026-04-08
  - Script: `data/features/clean_features.py` (reads 3MIN_REG.csv + Reg3Min26.csv)

- [x] **Load SCADA** — `data/features/load_3min.csv`
  - Done: 382,573 rows, 2024-01-01 to 2026-04-08
  - Script: `data/features/clean_features.py` (reads 3MIN_Load.csv + Load3Min26.csv)

- [x] **Production SCADA** — `data/features/production_3min.csv`
  - Done: 80,396 rows, 2025-10-08 to 2026-04-08
  - Script: `data/features/clean_features.py` (reads 3MIN_Prod.csv + Prod3Min26.csv)

- [x] **Export/Import SCADA** — `data/features/export_import_3min.csv`
  - Done: 98,078 rows, 2025-10-08 to 2026-04-08
  - Script: `data/features/clean_features.py` (reads 3MIN_ACK_REAL_BALNCE.csv + ackRealBalance3Min26.csv)

- [x] **Imbalance Labels (OKTE)** — `data/master/master_imbalance_data.csv`
  - Done: 78,419 rows, 2024-01-01 to 2026-03-29
  - Script: `data/master/create_master_imbalance.py` (globs all OKTE CSVs)
  - Note: Missing last ~10 days (Mar 30 - Apr 9) — OKTE files only go to Mar 29

## Priority 2: Hourly Features

- [x] **DAMAS Load (forecast + actual)** — `features/DamasLoad/load_data.csv`
  - Done: 22,155 rows, 2024-01-01 to 2026-04-09
  - Script: `features/DamasLoad/process_load_data.py` (globs all Zatazenie xlsx)

- [x] **DA Prices & Cross-Border Flows** — `features/DamasPrices/data/da_prices.csv`
  - Done: 27,218 rows, 2024-01-01 to 2026-04-09
  - Script: `features/DamasPrices/process_da_prices.py` (+ DA_market/Total_results_DAM)

- [x] **Solar (actual + DA forecast)** — `data/clean/solar/solar_hourly.csv`
  - Done: 19,917 rows, 2024-01-01 to 2026-04-09
  - Script: `data/clean/solar/clean_solar_data.py` (globs all Vyroba xlsx)
  - Note: Solar forecast only has 2,371 matched rows (new forecast file only covers recent period)

## Priority 3: Market Prices

- [x] **Hourly Market Prices (DA/IDM/Imbalance)** — `MarketPriceGap/data/processed/hourly_market_prices.csv`
  - Done: 19,990 hourly + 79,960 QH rows, through 2026-04-12
  - Script: `MarketPriceGap/scripts/load_market_prices.py` (globs all sources)

## Priority 4: Weather

- [ ] **Bardejov Weather Actual** — `data/Bardejov/Weather/bardejov_weather_actual.csv`
  - Current end: 2026-03-31 (8 days short)
  - Source: Open-Meteo API — not critical, model will use NaN for missing days

- [ ] **Bardejov DA Temp Forecast** — `data/Bardejov/Weather/bardejov_da_forecasts.csv`
  - Current end: 2026-04-03 (5 days short)
  - Source: Open-Meteo API — not critical

## Priority 5: Derived

- [ ] **Load Nowcast OOS Predictions** — `LoadAnalysis/nowcast_5h/tuning/oos_predictions/h2_oos_predictions.csv`
  - Current end: 2026-01-31
  - Action: Re-run `LoadAnalysis/nowcast_5h/tuning/generate_oos_predictions.py` with extended folds
  - Depends on: Regulation SCADA, Load SCADA, DAMAS Load (all done)

## After All Data Refreshed

- [ ] Re-run `ImbalanceForcastingProd/scripts/train_multi_lead.py` with extended test period
- [ ] Backtest cascade strategy on Feb--Apr 2026 (true out-of-sample, never seen by model)
