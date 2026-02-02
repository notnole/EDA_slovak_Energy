# Day-Ahead Prices Analysis Summary

## Overview
Analysis of Slovakia day-ahead electricity market prices from OKTE.

## Data Source
- **Raw files**: `RawData/Damas/Celkove_vysledky_DT_*.csv` (2024-2026)
- **Processed**: `data/da_prices.parquet` (24,843 hourly records)

## Key Statistics

| Metric | Value |
|--------|-------|
| Mean | 101.95 EUR/MWh |
| Std | 55.79 EUR/MWh |
| Min | -202.70 EUR/MWh |
| Max | 850.00 EUR/MWh |
| Negative prices | 2.48% of hours |

## Key Findings

### 1. Price Decomposition
- **44% seasonal** variance (hourly/weekly patterns)
- **33% residual** (much noisier than load's 7%)
- Makes price prediction harder than load prediction

### 2. Temporal Patterns
- Peak hours (9-14): Highest prices
- Night hours (2-4): Lowest prices
- Weekends: ~15% lower than weekdays

### 3. Negative Prices
- Occur 2.48% of time
- 78% happen on weekends
- Most common at midday (solar oversupply)

### 4. Price-Load Correlation
- Overall: 0.36
- Peak hours (hour 14): 0.74
- Night hours: ~0.10

## Analysis Subfolders

| Folder | Description | Key Plot |
|--------|-------------|----------|
| [decomposition/](decomposition/summary.md) | STL decomposition | 01_price_decomposition.png |
| [temporal_patterns/](temporal_patterns/summary.md) | Hourly/daily/weekly patterns | 02_temporal_patterns.png |
| [price_changes/](price_changes/summary.md) | Day-to-day changes | 03_price_changes.png |
| [correlation/](correlation/summary.md) | Price-load relationship | 04_price_load_correlation.png |
| [negative_prices/](negative_prices/summary.md) | Negative price analysis | 05_negative_prices.png |
| [autocorrelation/](autocorrelation/summary.md) | ACF/PACF analysis | 06_price_autocorrelation.png |

## Implication for Load Prediction

**Price features provide limited value for load prediction:**
- DAMAS forecast likely already incorporates price information
- Adding price features to the day-ahead model decreased performance (-3.1%)
- The correlation between price and load forecast error is minimal (~0.05)

## Scripts
- `process_da_prices.py` - Raw data processing
- `price_analysis.py` - Complete analysis and plotting

## Data Dictionary

| Column | Description |
|--------|-------------|
| datetime | Timestamp (hourly) |
| price_eur_mwh | Day-ahead price (EUR/MWh) |
| net_import | Net cross-border flow (MW) |
| price_lag24 | Yesterday same hour price |
| price_change_24h | Price change from yesterday |
| price_change_24h_pct | Percentage change |
| hour, dow, month, year | Temporal features |
| is_weekend | Weekend indicator |
