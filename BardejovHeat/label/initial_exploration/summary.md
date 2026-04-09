# Initial Exploration - Bardejov Heat Load

## Data Source

- **File**: `data/Bardejov/heat_load_timeseries.csv`
- **Raw source**: Monthly XLSX files (`Kalkulácie TeHo BJ 2025 01-12.xlsx`), DHS sheets
- **Column**: `IIN_LEISTUNG-NETZ` (Siet / network heat output)
- **Period**: 2021-01-01 to 2025-12-31 (153,941 rows)
- **Peak power**: ~25 MW (winter), summer baseload ~2-5 MW (hot water)

## Data Issues Discovered

### 1. Mixed time resolution

The data alternates between hourly and 15-minute resolution within each year:

| Period   | Resolution | Readings/day |
|----------|-----------|-------------|
| Jan-Apr  | Hourly    | 24          |
| May-Dec  | 15-min    | 96          |

This affects any aggregation (daily sums, etc.) - must resample to consistent resolution first.

### 2. kW vs kWh unit change in 2025

- **2021-2024**: All values are **power in kW** regardless of resolution (confirmed by column name `LEISTUNG` = power, and same magnitude for hourly and 15-min readings).
- **2025 15-min readings (May-Dec)**: Values switched to **kWh per 15-min interval** (energy). Must multiply by 4 to convert back to kW.
- **Evidence**: 2025 July mean was 580 vs 2024 July mean of 2,048. After *4 correction: 2,321 - consistent with prior years.
- **CSV has been corrected**: The current `heat_load_timeseries.csv` already has all values in kW.

### 3. Outlier data errors

Two extreme outlier readings in the raw data (prior to CSV correction):
- 2025-06-06 07:00: 35,131,420 (should be ~2,000)
- 2025-10-16 16:30: 66,032,900 (should be ~3,000)
- One 40 MW spike at 2025-01-01 00:00 (39,616 kW vs normal ~10,000)

All removed/corrected in the current CSV.

### 4. Multiple heat output channels

The XLSX files contain three separate output columns:
- **Siet (IIN_LEISTUNG-NETZ)**: Network heat - main district heating output
- **Maric (IIN_LEISTUNG-REGELKUEHLKREIS)**: Cooling circuit / heat dump
- **EE (IIAN_DTU_GEN_WIRKLEISTUNG)**: CHP electrical generation (kW)

The CSV uses only Siet. In some periods (especially 2025), significant heat goes through Maric. For total heat demand analysis, Siet + Maric may be needed.

## Operational Quirks

### Christmas shutdown (Dec 27-31, 2021)

- CHP plant shut down for Christmas holiday (EE = 0 entire period)
- Heat output drops from ~15,000 kW to ~2,500 kW (residual gas boilers / frost protection)
- Gradual ramp-down from Dec 26 afternoon, snap recovery Jan 1
- **Not a data issue** - real planned operational shutdown

### Single-day outage (Dec 13, 2023)

- Abrupt shutdown overnight Dec 12-13: from 15,000 kW to near-zero (~50 kW) by 1am
- Plant off for ~18 hours (entire daytime Dec 13)
- Rapid restart at 6pm, back to 14,000 kW by 9pm
- Mid-week (Wednesday) during cold period - likely **unplanned outage / emergency maintenance**

### Extended shutdown (Mar 13 - Apr 30, 2025)

- Complete zero output for ~7 weeks
- Normal winter operation (360-420 MWh/day) through Mar 12, then abrupt cutoff
- Restarts May 1 at reduced levels
- Likely major planned maintenance or equipment replacement

## Plots

1. `01_heat_load_full_timeseries.png` - Hourly power (MW), full 5-year series
2. `02_daily_heat_load.png` - Daily total energy (MWh/day)
3. `03_year_over_year.png` - Year-over-year seasonal comparison
