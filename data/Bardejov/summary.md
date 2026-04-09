# Bardejov District Heating - Data Summary

## Overview

Heat load data for Bardejov district heating system (~30,000 inhabitants), Slovakia.
Data sourced from Ipesoft EDA SCADA system via the CHP plant operator.

## Files

### heat_load_timeseries.csv
- **Period**: 2021-01-01 to 2025-12-31
- **Rows**: ~154,000
- **Columns**: `datetime`, `heat_load_kW`
- **Resolution**: Mixed - hourly (Jan-Apr) and 15-minute (May-Dec) within each year
- **Values**: Power in kW (already corrected for the 2025 kWh issue, see below)
- **Source column**: `IIN_LEISTUNG-NETZ` (Siet / network heat output)

### Monthly XLSX files (Kalkulacie TeHo BJ 2025 01-12.xlsx)
- Raw source data from the CHP plant operator
- Each file covers one calendar month, contains sheets for years 2021-2025
- **DHS sheets**: Raw SCADA data with three output channels:
  - Cols A-E: Hourly data (Date, Time, Siet, Maric, EE)
  - Cols H-L: 15-minute data (same columns)
- **Channels**:
  - **Siet (IIN_LEISTUNG-NETZ)**: Network heat output - main district heating
  - **Maric (IIN_LEISTUNG-REGELKUEHLKREIS)**: Cooling circuit / heat dump
  - **EE (IIAN_DTU_GEN_WIRKLEISTUNG)**: CHP electrical generation (kW)

### Historia vyroby TEHO 2.xlsx
- Historical production data (not yet explored)

### Priprava prevadzky TEHO_3.xlsx
- Operational preparation data (not yet explored)

### Weather/
- **bardejov_weather_actual.csv**: Hourly actuals, 2024-01-01 to 2026-03-31 (~19,700 rows)
  - Columns: temperature_2m, relative_humidity_2m, dewpoint_2m, apparent_temperature, windspeed_10m, shortwave_radiation, direct_normal_irradiance, diffuse_radiation, cloudcover, surface_pressure, precipitation, snowfall, snow_depth, soil_temperature_0_to_7cm
- **bardejov_weather_forecasts.csv**: Forecast data, same period
  - Contains three forecast model columns: best_match_*, ifs_*, gfs_*
  - Same weather variables as actuals for each model

### DAmarket/
- Day-ahead electricity market results for Slovakia
- Files covering 2024, 2025, and early 2026

## Known Data Issues (corrected in CSV)

### 1. Mixed time resolution
Jan-Apr is hourly (24 readings/day), May-Dec is 15-minute (96 readings/day). Must resample to consistent resolution before aggregation.

### 2. kW vs kWh unit change in 2025
2021-2024 values are power in kW. 2025 15-min readings (May-Dec) switched to kWh per 15-min interval - required *4 correction. Evidence: 2025 July mean was 580 vs 2024 July mean of 2,048; after correction: 2,321. **Already corrected in the CSV.**

### 3. Outliers removed
- 2025-06-06 07:00: 35,131,420 kW (data error)
- 2025-10-16 16:30: 66,032,900 kW (data error)
- 2025-01-01 00:00: 39,616 kW spike (vs normal ~10,000)

## Operational Events

- **Christmas 2021 shutdown** (Dec 27-31): Planned CHP shutdown, heat drops to ~2,500 kW
- **Dec 13, 2023 outage**: Unplanned ~18h shutdown, rapid restart at 6pm
- **Peak power**: ~25 MW (winter), summer baseload ~2-5 MW (hot water only)
