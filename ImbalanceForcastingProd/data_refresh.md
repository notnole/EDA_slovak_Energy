# Data Refresh Checklist (through April 2026)

Target: extend all data sources to **April 9, 2026** for out-of-sample testing of the multi-lead imbalance predictor on Feb--Apr 2026.

## Priority 1: Core SCADA + Labels (73--74 days missing)

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

- [in individual files] **Imbalance Labels (OKTE)** — `data/master/master_imbalance_data.csv`
  - Current end: 2026-01-24
  - Missing: 2026-01-24 to 2026-04-09 (74 days)
  - Source: OKTE monthly SystemImbalance/OdchylkaSustavy CSVs -> `OKTE_Imbalnce/`
  - Note: 2026 files have English headers (pipeline handles both)

## Priority 2: Hourly Features (67--70 days missing)

- [ii] **DAMAS Load (forecast + actual)** — `features/DamasLoad/load_data.csv`
  - Current end: 2026-01-31
  - Missing: 2026-01-31 to 2026-04-09 (67 days)
  - Source: DAMAS CSV exports (`RawData/Damas/`)

- [i] **DA Prices & Cross-Border Flows** — `features/DamasPrices/data/da_prices.csv`
  - Current end: 2026-01-28
  - Missing: 2026-01-28 to 2026-04-09 (70 days)
  - Source: DA market results (`RawData/DA_market/`)

- [this is in the general producion folder in damas] **Solar (actual + DA forecast)** — `data/clean/solar/solar_hourly.csv`
  - Current end: 2026-01-31
  - Missing: 2026-01-31 to 2026-04-09 (67 days)
  - Source: DAMAS production per type / solar generation data

## Priority 3: Market Prices (33 days missing)

- [these were alredy requested] **Hourly Market Prices (DA/IDM/Imbalance)** — `MarketPriceGap/data/processed/hourly_market_prices.csv`
  - Current end: 2026-03-06
  - Missing: 2026-03-06 to 2026-04-09 (33 days)
  - Source: IDM data (`RawData/IDM_MarketData/`), DA market, OKTE settlement prices

## Priority 4: Weather (5--8 days missing)

- [cant be botherd] **Bardejov Weather Actual** — `data/Bardejov/Weather/bardejov_weather_actual.csv`
  - Current end: 2026-03-31
  - Missing: 2026-03-31 to 2026-04-09 (8 days)
  - Source: Open-Meteo API (historical weather for Bardejov)

- [cant be botherd ] **Bardejov DA Temp Forecast** — `data/Bardejov/Weather/bardejov_da_forecasts.csv`
  - Current end: 2026-04-03
  - Missing: 2026-04-03 to 2026-04-09 (5 days)
  - Source: Open-Meteo API (GFS seamless D+1 forecast)

## Priority 5: Derived (regenerate after above are updated)

- [ lets generet it properly OOF] **Load Nowcast OOS Predictions** — `LoadAnalysis/nowcast_5h/tuning/oos_predictions/h2_oos_predictions.csv`
  - Current end: 2026-01-31
  - Missing: 2026-01-31 to 2026-04-09 (67 days)
  - Action: Re-run `LoadAnalysis/nowcast_5h/tuning/generate_oos_predictions.py` after SCADA + DAMAS updated
  - Depends on: Regulation SCADA, Load SCADA, DAMAS Load

## After All Data Refreshed

- [ ] Re-run `ImbalanceForcastingProd/scripts/train_multi_lead.py` with extended test period
- [ ] Backtest cascade strategy on Feb--Apr 2026 (true out-of-sample, never seen by model)
