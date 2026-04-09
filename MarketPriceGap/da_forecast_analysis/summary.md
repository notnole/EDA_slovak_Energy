# DA Price Forecast Analysis

## Data Source
- **Raw file**: `RawData/DA_Price_Forcasts.csv` (Ipesoft EDA export, 5 side-by-side series)
- **Cleaned**: `data/clean/market/da_forecasts/da_price_forecasts_15min.csv`
- **Actual prices**: `MarketPriceGap/data/processed/hourly_market_prices.csv` (corrected hourly, handles mixed 15-min/60-min resolution)

**Note (Feb 2026):** Previous analysis used raw `da_auction_results.csv` with `mode()` resolution detection, which incorrectly mapped 2026 15-minute periods as hourly. This inflated all error metrics. Corrected by using the processed hourly file which properly aggregates 15-min data to hourly.

## Series Inventory

| Series | Description | Valid From | Coverage |
|--------|-------------|------------|----------|
| DE Forecast1 Spot 15 | German spot forecast (benchmark) | 17/09/2025 | 94.0% |
| Forecast M SK spot 15min | SK market forecast | NEVER | 0% (dropped) |
| SK Forecast1 Spot 15 | Slovak forecast model 1 | 11/09/2025 | 96.0% |
| SK Forecast2 Spot 15 | Slovak forecast model 2 | 11/09/2025 | 96.0% |
| SK.F.M.Merged.Spot | SK merged/ensemble forecast | 18/01/2026 | 13.4% |

## Period
2025-09-11 to 2026-01-31 (hourly, ~3,300 observations for SK models)

## Forecast Error Statistics

| Model | Bias | MAE | RMSE | Correlation | N |
|-------|------|-----|------|-------------|---|
| DE Forecast1 | -16.2 EUR/MWh | 22.9 | 36.1 | 0.756 | 3,216 |
| **SK Forecast1** | **-1.0 EUR/MWh** | **18.1** | **30.3** | **0.795** | **3,288** |
| SK Forecast2 | -1.3 EUR/MWh | 18.4 | 30.6 | 0.791 | 3,288 |
| SK Merged | -7.8 EUR/MWh | 17.2 | 27.8 | 0.822 | 336 |

**Best model**: SK Forecast1 -- lowest bias, low MAE, high correlation. SK Merged has the best absolute metrics but only 336 observations (from 18 Jan 2026).

## Key Findings

### 1. Forecast Quality is Moderate (r ~ 0.80)
SK models explain ~63% of hourly price variance (R-squared ~ 0.63). This is reasonable for DA price forecasting. The forecasts track the daily price shape well but miss some magnitude shifts, particularly during volatile periods.

### 2. Hourly Bias Pattern (Smaller Than Previously Reported)
The forecast has a mild hour-of-day bias:
- **Mid-morning (H9-H11)**: Forecast slightly too low by -5 to -7 EUR/MWh
- **Evening (H18-H19)**: Forecast slightly too high by +5 to +6 EUR/MWh

This is a mild "shape exaggeration" -- the forecast slightly overestimates the daily spread. The magnitude is much smaller than previously reported (the earlier analysis showed +/-30 EUR/MWh bias, which was inflated by the data resolution bug).

### 3. DE Forecast as SK Proxy
The DE Forecast1 has a -16.2 EUR/MWh bias (DE prices systematically lower than SK). Despite this, it achieves r=0.756, confirming that DE and SK price shapes are correlated. The SK-DE spread varies over time.

### 4. STL Decomposition of Forecast Error
The forecast error decomposes into:
- **Trend**: Multi-day bias drift, suggesting models react slowly to regime changes
- **Seasonal (24h)**: Mild diurnal cycle confirming the hourly bias pattern
- **Residual**: High-frequency unpredictable component

### 5. Residual Correlation
After removing trend and seasonal components:

**Forecast error residual vs actual price residual: r = -0.640 (R-squared = 41.0%)**

This negative correlation means: when actual prices spike unexpectedly, the forecast misses it (and vice versa). About 41% of the unpredictable price variance flows through as forecast error. This is expected -- the forecast cannot predict genuine surprises.

### 6. Data Quality Correction Impact
The previous analysis (using raw OKTE data with incorrect resolution handling) reported:
- SK Forecast1: MAE=37.7, r=0.353, residual r=-0.844
- These were **severely inflated** by the 2026 data bug (15-min periods treated as hourly)

Corrected values:
- SK Forecast1: MAE=18.1, r=0.795, residual r=-0.640
- **MAE halved, correlation more than doubled**

## Implications

1. **For the BESS indicator**: The DA price forecasts (r=0.80) could be used as an additional feature alongside the current rank-based prediction (load forecast, yesterday's prices, weekly pattern). The forecast captures ~63% of price variance, which is complementary information.

2. **For trading**: The mild hourly bias (+/-6 EUR/MWh) is exploitable through hour-of-day bias correction, but the effect is smaller than previously estimated.

3. **For model improvement**: The residual r=-0.640 means there is still substantial room to improve. Adding real-time features (wind/solar generation, cross-border flows) could capture some of the 41% residual variance.

4. **SK Merged model**: With only 336 observations but the best metrics (r=0.822, MAE=17.2), this ensemble model is promising and should be monitored as more data accumulates.

## Files

| File | Description |
|------|-------------|
| `01_forecast_vs_actual_timeseries.png` | Daily forecast vs actual + error + rolling MAE |
| `02_scatter_and_error_distribution.png` | Scatter plots and error histograms per model |
| `03_stl_decomposition_forecast_error.png` | STL decomposition of SK Forecast1 error |
| `04_residual_and_error_analysis.png` | Hourly bias, price-level bias, residual correlation, rolling quality |
| `data/forecast_vs_actual_merged.csv` | Hourly merged dataset |
| `data/decomposition_residuals.csv` | STL decomposition components |
| `scripts/clean_and_analyze.py` | Reproducible cleaning + analysis script |
