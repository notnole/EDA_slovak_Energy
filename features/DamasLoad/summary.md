# DAMAS Load Data

## Overview

Processing script and cleaned data for Slovak grid hourly load (DAMAS day-ahead forecast and actual).

## Files

| File | Description |
|------|-------------|
| `process_load_data.py` | Cleaning script: parses EDA CSV export, European decimals, outputs hourly |
| `load_data.csv` | Cleaned hourly load data (CSV format) |
| `load_data.parquet` | Same data in Parquet format (used by LoadAnalysis and other workstreams) |

## Data

- **Period**: 2024-01-01 to 2026-01-31
- **Resolution**: Hourly
- **Records**: ~24,843
- **Source**: `RawData/Damas/`
- **Key columns**: timestamp, load_actual (MW), load_forecast (MW)
