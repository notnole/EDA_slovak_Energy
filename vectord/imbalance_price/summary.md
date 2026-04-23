# Imbalance Settlement Price EDA

Vector: `Okte.Calc.ZuctovaciaCenaOdchylky_Comb`  

Regime split from: `Okte.Combine.Odchylka` (split at imb = 0)  

Full window: `2026-01-01` -> `2026-04-22`  

April detail: `2026-04-01` -> `2026-04-22`  

N 15-min periods: 10685


## Regime breakdown

| Regime | N | Share |
| --- | --- | --- |
| surplus | 5253 | 49.2% |
| deficit | 5432 | 50.8% |

## Descriptive statistics (EUR/MWh)

| Regime | N | Mean | Median | Std | P5 | P95 | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 10685 | 114.7 | 109.5 | 85.5 | 0.3 | 236.2 | -590.5 | 3654.4 |
| surplus | 5253 | 71.7 | 82.7 | 77.7 | -14.8 | 138.0 | -590.5 | 3654.4 |
| deficit | 5432 | 156.2 | 142.2 | 70.8 | 77.4 | 292.8 | 10.0 | 844.3 |

## Decomposition

Method: Rolling-window (fallback) on daily median price.


## Plots

- `01_timeseries_2026.png` — full-year 15-min scatter coloured by regime, 7-day rolling mean, daily median by regime
- `02_april_detail.png` — April 15-min price + system imbalance on second axis
- `03_price_vs_imbalance.png` — price vs imbalance scatter (April), reveals pricing curve
- `04_distributions.png` — histograms overall, surplus (imb>=0), deficit (imb<0)
- `05_hourly_seasonality.png` — median price by hour-of-day, regime share by hour
- `06_weekday_seasonality.png` — boxplot by day-of-week per regime
- `07_decomposition.png` — daily median price STL decomposition (trend / seasonal / residual)
- `08_regime_boxplot.png` — price boxplot surplus vs deficit
