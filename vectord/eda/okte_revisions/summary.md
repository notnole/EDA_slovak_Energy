# Okte.Odchylka initial vs Okte.Combine.Odchylka (settled)

Window: `2025-04-20` -> `2026-04-20`   N periods: 34996


## Overall

| Metric | Value |
| --- | --- |
| MAE (|settled - initial|) | 0.184 MWh |
| RMSE | 9.579 MWh |
| Bias (settled - initial) | +0.160 MWh |
| Max abs diff | 1725.12 MWh |
| Revised periods | 3.9 % |
| Sign flips (when |either| > 1 MWh) | 0.20 % |
| Mean |settled| (signal scale) | 7.98 MWh |
| **Revision / signal** | **2.3 %** |

## Monthly

| month | n | mae | bias | % revised | % flip | max abs |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-04 | 1020 | 0.300 | +0.300 | 3.9 | 0.00 | 26.00 |
| 2025-05 | 2976 | 0.037 | +0.034 | 2.0 | 0.10 | 25.00 |
| 2025-06 | 2880 | 0.138 | +0.138 | 1.4 | 0.17 | 35.00 |
| 2025-07 | 2976 | 0.079 | -0.007 | 0.4 | 0.00 | 26.00 |
| 2025-08 | 2976 | 0.000 | -0.000 | 0.5 | 0.00 | 0.02 |
| 2025-09 | 2880 | 0.002 | -0.001 | 0.8 | 0.00 | 1.70 |
| 2025-10 | 2976 | 0.038 | -0.018 | 2.3 | 0.03 | 13.02 |
| 2025-11 | 2880 | 0.078 | +0.077 | 1.6 | 0.17 | 60.78 |
| 2025-12 | 2976 | 1.388 | +1.383 | 7.1 | 1.45 | 1725.12 |
| 2026-01 | 2976 | 0.133 | +0.030 | 8.5 | 0.30 | 26.97 |
| 2026-02 | 2688 | 0.008 | +0.008 | 2.6 | 0.00 | 11.46 |
| 2026-03 | 2976 | 0.139 | +0.114 | 4.7 | 0.07 | 139.40 |
| 2026-04 | 1816 | 0.048 | +0.048 | 21.5 | 0.11 | 14.47 |

## 5 largest revisions

| time | initial | settled | diff |
| --- | --- | --- | --- |
| 2025-12-01T18:45:00+00:00 | -1732.07 | -6.95 | +1725.12 |
| 2025-12-01T19:00:00+00:00 | -232.01 | -1.97 | +230.04 |
| 2026-03-12T08:30:00+00:00 | -126.12 | +13.28 | +139.40 |
| 2026-03-30T13:15:00+00:00 | -129.65 | +9.60 | +139.24 |
| 2025-12-01T14:00:00+00:00 | -143.62 | -6.24 | +137.38 |

## Plots

- `01_timeseries_revisions.png` — daily averages
- `02_diff_histogram.png` — revision distribution
- `03_scatter.png` — settled vs initial
- `04_monthly.png` — MAE / bias / revision share by month
- `05_worst_revisions.png` — table of 10 worst
