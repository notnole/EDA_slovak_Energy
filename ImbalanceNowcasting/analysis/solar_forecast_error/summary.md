# TSO Solar Forecast Error Analysis

## Data

- **Source**: `RawData/ProductionPerType.csv` — columns `Sk.A.Solar` (actual) and `Sk.F.Solar` (forecast)
- **Period**: 2025-08-19 to 2026-01-31 (~5.5 months, Aug–Jan)
- **Resolution**: Hourly
- **Valid observations**: 3,950 (both actual and forecast non-null)
- **Daylight observations** (actual > 0): 3,690

## Forecast Error Metrics

| Metric | Value | Note |
|--------|-------|------|
| MAE | **10.92 MW** | Average absolute error |
| RMSE | 23.46 MW | Sensitive to large errors |
| MBE | +0.30 MW | Nearly unbiased |
| MAPE | 108% | Inflated by near-zero hours — not meaningful for solar |

The TSO solar forecast is essentially **unbiased** overall (MBE ≈ 0), but individual hours carry significant error concentrated at midday peak.

## Hourly Error Profile

The forecast shows a systematic **morning overforecast / afternoon underforecast** pattern:

| Period | Hours | Bias (MW) | MAE (MW) |
|--------|-------|-----------|----------|
| Early morning | 6–8 | +1 to +7 | 2–19 |
| Peak production | 9–13 | +1 to +5 | 29–33 |
| Afternoon | 14–16 | -2 to -5 | 11–25 |
| Evening | 17–19 | ~0 | 2–6 |

Peak MAE of **~33 MW at 11:00** against average actual production of 138 MW (~24% relative error at peak).

## Monthly Error Profile

| Month | MAE (MW) | MBE (MW) | Mean Actual (MW) |
|-------|----------|----------|------------------|
| Aug | 17.5 | -0.6 | 83.3 |
| Sep | 13.8 | -2.0 | 68.7 |
| Oct | 13.3 | -1.1 | 48.4 |
| Nov | 9.7 | +4.9 | 26.8 |
| Dec | 7.5 | -1.3 | 16.6 |
| Jan | 7.1 | +1.6 | 16.8 |

Errors are larger in summer months (higher production, more cloud variability) and smaller in winter.

## STL Decomposition

### Actual vs Forecast Overlay

STL decomposition (period=24h, robust) applied to actual production, forecast, and the error series.

| Component | Actual | Forecast | Difference |
|-----------|--------|----------|------------|
| **Trend range** | 2–121 MW | 4–118 MW | Similar — trend well captured |
| **Seasonal amplitude** | 357 MW | 384 MW | Forecast overshoots by **+27 MW** |
| **Residual std** | 30.6 MW | 32.1 MW | Similar noise levels |

**Residual correlation: 0.55** — the unpredictable parts of actual and forecast move together, indicating the forecast responds to the same weather drivers.

### Forecast Error Decomposition

| Component | Variance share | Interpretation |
|-----------|---------------|----------------|
| Trend | 3.4% | Minimal systematic drift over time |
| Seasonal (24h) | 21.3% | Repeatable daily error pattern — correctable |
| **Residual** | **87.8%** | Dominant — driven by weather unpredictability |

### Key Observations

1. **Seasonal shape mismatch**: The forecast has a **flatter, wider** daily profile than reality. Actual peaks sharply at 12:00 while the forecast spreads production more evenly across 10:00–17:00. This creates the morning overforecast / afternoon underforecast pattern.

2. **Trend is well matched**: Both series track the seasonal decline from summer to winter closely. The trend difference stays within ±20 MW and averages near zero.

3. **~21% of forecast error is a predictable daily pattern**: A simple hourly bias correction could reduce MAE by an estimated 2–3 MW.

4. **~88% of error is random**: Further improvement requires better cloud/weather nowcasting — not correctable from historical patterns alone.

## Figures

- `01_error_analysis.png` — Actual vs forecast time series, error distribution, MAE by hour, scatter plot
- `02_decomposition_error.png` — STL decomposition of forecast error (trend, seasonal, residual)
- `03_decomposition_overlay.png` — Overlaid STL decomposition of actual and forecast
- `04_decomposition_differences.png` — Daily seasonal pattern comparison, trend difference, residual correlation
