# Real vs Proxy Imbalance Price -- Comparison EDA

Real:  `Okte.Combine.CenaSysOdchylkySR`  

Proxy: `Okte.Calc.ZuctovaciaCenaOdchylky_Comb`  

Regime from: `Okte.Combine.Odchylka` (split at imb = 0)  

Full window: `2026-01-01` -> `2026-04-22`  

April detail: `2026-04-01` -> `2026-04-22`  


## Comparison metrics

| Regime | N | Corr | MAE | RMSE | Bias (proxy-real) | Med real | Med proxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 10552 | 0.835 | 22.0 | 44.7 | +14.1 | 97.5 | 109.4 |
| surplus | 5120 | 0.865 | 8.1 | 30.4 | -8.1 | 85.4 | 82.1 |
| deficit | 5432 | 0.808 | 35.1 | 54.8 | +35.1 | 111.9 | 142.2 |

## Regime confusion (wrong-direction spread periods)

Surplus: confused = spread > 0 (proxy above real, acting like deficit pricing)  

Deficit: confused = spread < 0 (proxy below real, acting like surplus pricing)  

| Regime | Confused / Total | % periods | % of total abs error | MAE confused | MAE normal |
| --- | --- | --- | --- | --- | --- |
| surplus | 0 / 5120 | 0.0% | 0.0% | 0.0 | 8.1 |
| deficit | 0 / 5432 | 0.0% | 0.0% | 0.0 | 35.1 |

## Plots

- `01_timeseries_2026.png` — daily medians of real vs proxy per regime
- `02_april_detail.png` — 15-min scatter real vs proxy per regime, April
- `03_scatter.png` — proxy vs real scatter (overall + per regime with r and MAE)
- `04_distributions.png` — overlapping histograms real vs proxy per regime
- `05_spread.png` — (proxy-real) spread: daily timeseries + histogram per regime
- `06_hourly.png` — median real vs proxy by hour-of-day per regime
- `07_spread_vs_imbalance.png` — spread vs |imbalance| size (April)
- `08_cumulative_step.png` — cumulative (proxy-real) error over time per regime
- `09_regime_confusion.png` — wrong-direction spread periods per regime
