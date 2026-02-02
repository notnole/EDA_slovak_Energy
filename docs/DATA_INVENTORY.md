# Data Inventory - Slovak Electricity Market

This document provides a comprehensive overview of all data sources available in this repository, including their origins, timing, and intended use cases.

---

## Quick Reference: Data Availability Timeline

| Time Before Delivery | Available Data |
|---------------------|----------------|
| **D-1 (Day-Ahead)** | DA market results, DAMAS load/production forecasts, Temperature forecasts |
| **H-1 (Intraday)** | IDM 15-min trading data, Cross-border flow plans |
| **Real-time** | 3-min SCADA: Load, Production, Regulation, Balance |
| **T+15 min** | Settlement period imbalance (from OKTE) |

---

## 1. SCADA Data (Private Database - Ipesoft EDA)

**Source**: Internal Ipesoft EDA database (streaming data)
**Location**: `RawData/`
**Update frequency**: Every 1-3 minutes
**Availability**: Real-time (with ~3 min delay for publication)

### Files

| Original Name | Description | Resolution | Start Date | Tag Name |
|---------------|-------------|------------|------------|----------|
| `3MIN_REG.csv` | Regulation energy (MW) | ~1 min | Jan 2024 | `SEPS - 3 minutova regulacna elektrina` |
| `3MIN_Load.csv` | Real system load (MW) | 3 min | Jan 2024 | `PUB_3M.REAL_SYSTEM_LOAD` |
| `3MIN_Prod.csv` | Real system production (MW) | ~1 min | Oct 2025 | `PUB_3M.REAL_SYSTEM_PRODUCTION` |
| `3MIN_Export.csv` | Real balance (MW) - *misnamed* | ~1 min | Oct 2025 | `PUB_3M.REAL_BALANCE` |
| `3MIN_ACK_REAL_BALNCE.csv` | Acknowledged real balance (MW) | ~1 min | Oct 2025 | `PUB_3M.ACKNOWLEDGED_REAL_BALANCE` |
| `ProductionPerType.csv` | Production by fuel type (MW) | 1 hour | Jan 2024 | Multiple tags |
| `Claudes.csv` | Cloud cover (%) - *misnamed* | 15 min | Jan 2025 | `SK.A.Cloud Actual` + forecasts |
| `Tampreture.csv` | Temperature (C) - *typo* | 15 min | Apr 2025 | GFS/Icon forecasts + actuals |

### Data Format (SCADA files)
```
Timestamp,000,Value,
DD/MM/YYYY HH:MM:SS,000,<value>,
```
- European date format (D/M/YYYY)
- Comma as decimal separator (e.g., `12,788` = 12.788)
- Values in MW (power) or MWh (energy)

### ProductionPerType Columns
| Tag | Description |
|-----|-------------|
| `SK.A.Biomas` | Biomass production |
| `Sk.A.FosilOil` | Fossil oil production |
| `Sk.A.HardCoal` | Hard coal production |
| `Sk.A.HydroPump` | Pumped hydro generation |
| `Sk.A.HydroPumpConsumption` | Pumped hydro consumption |
| `Sk.A.HydroReservoir` | Reservoir hydro |
| `Sk.A.HydroRunRiver` | Run-of-river hydro |
| `Sk.A.Lignite` | Lignite production |
| `Sk.A.NatGas` | Natural gas production |
| `SK.A.Nuclear` | Nuclear production |
| `Sk.A.Other` | Other sources |
| `Sk.A.Other.Renewable` | Other renewables |
| `Sk.A.Solar` | Solar actual |
| `Sk.F.Solar` | Solar forecast |

---

## 2. Day-Ahead Market Data (DAMAS - SEPS)

**Source**: DAMAS platform (https://dae.sepsas.sk)
**Location**: `RawData/Damas/` and `RawData/DA_market/`
**Update frequency**: Daily after auction (around 12:00 CET D-1)
**Availability**: D-1 (day before delivery)

### CSV Files - Auction Results

| File Pattern | Description | Resolution | Coverage |
|--------------|-------------|------------|----------|
| `Celkove_vysledky_DT_*.csv` | Complete DA auction results | 1 hour | 2024-2026 |

**Columns (Slovak -> English)**:
| Slovak | English | Unit |
|--------|---------|------|
| Obchodny den | Trading day | date |
| Cislo periody | Period number | 1-24 |
| Perioda | Period | HH:MM-HH:MM |
| Perioda (min) | Period duration | minutes |
| Cena SK (EUR/MWh) | Slovakia price | EUR/MWh |
| Dopyt uspesny (MW) | Successful demand | MW |
| Ponuka uspesna (MW) | Successful supply | MW |
| Tok CZ -> SK (MW) | Flow CZ to SK | MW |
| Tok SK -> CZ (MW) | Flow SK to CZ | MW |
| Tok PL -> SK (MW) | Flow PL to SK | MW |
| Tok SK -> PL (MW) | Flow SK to PL | MW |
| Tok HU -> SK (MW) | Flow HU to SK | MW |
| Tok SK -> HU (MW) | Flow SK to HU | MW |
| Stav zverejnenia | Publication status | text |

**Format**: Semicolon-delimited, European decimals

### XLSX Files - Forecasts and Actuals

| File Pattern (Slovak) | English Name | Description | Availability |
|----------------------|--------------|-------------|--------------|
| `Zatazenie ES SR - Denna predikcia` | Load Day-Ahead Forecast | System load forecast | D-1 |
| `Zatazenie ES SR - Skutocnost` | Load Actuals | Actual system load | D+1 |
| `Vyroba ES SR - Denna predikcia` | Production Day-Ahead Forecast | System production forecast | D-1 |
| `Vyroba ES SR - Skutocnost` | Production Actuals | Actual production | D+1 |
| `Vyroba FVE a Veternych - Predikcia` | Solar & Wind Forecast | RES forecast | D-1 |
| `Plany cezhranicnych vymen` | Cross-Border Flow Plans | Scheduled flows | D-1 |
| `Implicitna aukcia - vysledky` | Implicit Auction Results | DA coupling results | D-1 |

**Note**: DA_market folder contains duplicates of Damas CSV files - can be removed.

---

## 3. Intraday Market Data (OKTE)

**Source**: OKTE (Slovak market operator)
**Location**: `RawData/IDM_MarketData/`
**Update frequency**: Continuous (after each trading window closes)
**Availability**: H-1 to H-0 (up to 5 min before delivery)

### Files

| Folder Pattern | Files | Coverage |
|----------------|-------|----------|
| `IDM_total_results_YYYY-MM-DD_YYYY-MM-DD/` | `15 min.csv`, `60 min.csv` | 2-month chunks |

**Available Periods**: Jan 2025 - Jan 2026

**Columns**:
| Column | Description | Unit |
|--------|-------------|------|
| Delivery day | Delivery date | D.M.YYYY |
| Period number | Period within day | 1-96 (15min) or 1-24 (60min) |
| Period | Time range | HH:MM-HH:MM |
| Buy Orders (MW) | Total buy orders | MW |
| Sell Orders (MW) | Total sell orders | MW |
| Buy trades (MW) | Executed buy trades | MW |
| Sell trades (MW) | Executed sell trades | MW |
| Traded Quantity Difference (MW) | Net position change | MW |
| Weighted average price (EUR/MWh) | VWAP of trades | EUR/MWh |
| Total Traded Quantity (MW) | Total volume | MW |

**Format**: Semicolon-delimited, English headers, decimal points

---

## 4. System Imbalance Data (OKTE)

**Source**: OKTE settlement data
**Location**: `OKTE_Imbalance/` (if exists) or via API
**Update frequency**: T+15 minutes after settlement period
**Availability**: 15 minutes after each settlement period ends

**Key relationship**: `Imbalance ~ -0.25 * mean(Regulation)`

---

## 5. Data Timing for Trading Strategies

### Before Day-Ahead Auction (D-1, before 12:00)
- Weather forecasts (temperature, cloud cover)
- Historical patterns
- ENTSO-E flow schedules

### After Day-Ahead Auction (D-1, after 12:00)
- DA prices (Slovakia and neighbors)
- DA volumes (demand/supply matched)
- Cross-border flow plans
- Load and production forecasts from DAMAS

### Intraday (D-0)
- IDM trading volumes and prices (15-min resolution)
- Updated cross-border flows
- RES forecast updates

### Real-Time (D-0, continuous)
- 3-min SCADA: Load, Production, Regulation
- Real-time balance position
- Frequency and ACE data

### Post-Settlement (T+15 min)
- Official imbalance (MWh per 15-min settlement period)
- Settlement prices

---

## 6. File Naming Issues (To Be Fixed)

| Current Name | Issue | Suggested Name |
|--------------|-------|----------------|
| `Claudes.csv` | Misleading name | `cloud_cover_15min.csv` |
| `Tampreture.csv` | Typo | `temperature_15min.csv` |
| `3MIN_Export.csv` | Wrong name (contains balance, not export) | `real_balance_1min.csv` |
| `3MIN_ACK_REAL_BALNCE.csv` | Typo | `ack_real_balance_1min.csv` |

---

## 7. Data Quality Notes

### Known Issues
1. **ProductionPerType.csv**: Wide format with repeated timestamps per column - needs reshaping
2. **3MIN_REG.csv**: Timestamps not exactly on 3-min boundaries (1-minute data)
3. **Temperature/Cloud data**: Contains `(invalid)` values for some forecast columns
4. **Decimal format**: European (comma) in SCADA, varies in market data

### Recommended Preprocessing
1. Parse European date formats (`DD/MM/YYYY`)
2. Convert comma decimals to dots
3. Handle `(invalid)` values as NaN
4. Align 1-minute data to 3-minute boundaries
5. Reshape wide production data to long format

---

## 8. Folder Structure

```
RawData/
  |-- 3MIN_*.csv              # SCADA streaming data (Ipesoft EDA)
  |-- ProductionPerType.csv   # Production by fuel type
  |-- Claudes.csv             # Cloud cover (misnamed)
  |-- Tampreture.csv          # Temperature (typo)
  |
  |-- Damas/                  # DAMAS downloads
  |   |-- *.csv               # DA auction results
  |   |-- *.xlsx              # Forecasts and actuals
  |   |-- SEPS_DAE_API_REFERENCE.md
  |
  |-- DA_market/              # DUPLICATE of Damas CSVs - remove
  |   |-- *.csv
  |
  |-- IDM_MarketData/         # Intraday market data
      |-- IDM_total_results_*/
          |-- 15 min.csv
          |-- 60 min.csv
```

---

## 9. API Access

### DAMAS (SEPS)
- See `RawData/Damas/SEPS_DAE_API_REFERENCE.md`
- Requires session token (X-XSRF-TOKEN)
- POST requests to LoadData endpoint
- Alternative: Playwright scraping

### OKTE (for imbalance data)
- Public API available
- No authentication required for basic data

---

## 10. Use Cases

### Imbalance Nowcasting
**Required data**: 3MIN_REG, 3MIN_Load, 3MIN_Prod
**Lead times**: 0-12 minutes before settlement period ends
**Target**: Predict final 15-min imbalance

### Load Forecasting
**Required data**: DAMAS load forecast (baseline), Temperature, ProductionPerType
**Horizon**: Day-ahead (24-48 hours)
**Target**: Improve on DAMAS baseline (MAE ~68 MW)

### Price Forecasting
**Required data**: DA prices, IDM volumes, Cross-border flows
**Horizon**: Day-ahead and intraday
**Target**: Predict DA and IDM prices

---

*Last updated: 2026-02-01*
