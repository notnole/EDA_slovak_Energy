# Solar forecast vs actual — 30 days

Window: `2026-03-21` → `2026-04-20` (2157 aligned 15-min obs)

Vectors:
- Actual: `Sk.A.Solar`
- Forecast: `Sk.F.Solar`

## Overall metrics

| Metric | Value |
|--------|-------|
| MAE | 21.19 MW |
| RMSE | 42.91 MW |
| Bias (F - A) | -7.30 MW |
| Correlation | 0.924 |
| Mean actual | 79.0 MW |
| Peak actual | 402 MW |

## Daytime only (actual > 50 MW, n=831)

| Metric | Value |
|--------|-------|
| MAE | 49.46 MW |
| Bias | -18.93 MW |
| MAPE | 28.9 % |

## Plots

- `01_timeseries.png` — last 7 days, forecast vs actual
- `02_scatter.png` — scatter with y=x reference
- `03_error_by_hour.png` — MAE and bias by hour of day
- `04_daily_mae.png` — daily MAE and bias over the window
