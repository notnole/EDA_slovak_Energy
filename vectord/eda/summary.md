# Vectord EDA — 30 day snapshot

Window: `2026-03-21T09:00:00+00:00` -> `2026-04-20T09:00:00+00:00`

Vectors probed: 12  (12 ok, 0 empty/error)


## Imbalance

| vector | cadence_label | n | completeness_pct | min | median | mean | p95 | max | unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Okte.Odchylka | 15-min | 2836 | 100 | -129.65 | -0.28 | 0.96 | 27.62 | 63.97 | MWh |
| F.B.Odchylka | n/a | 1 | nan | -3.4 | -3.4 | -3.4 | -3.4 | -3.4 | MWh |


## Price

| vector | cadence_label | n | completeness_pct | min | median | mean | p95 | max | unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Okte.MargCena | 15-min | 2881 | 100 | -154.2 | 111.9 | 101.1 | 183.78 | 305.86 | EUR/MWh |
| SK.F.M.Merged.Spot | 15-min | 2881 | 100 | -36.92 | 100.7 | 92.47 | 164.26 | 189.6 | EUR/MWh |
| PICASSO.MarginalPricess.SEPS_POS.Weighted | 15-min | 2547 | 88.5 | -62.39 | 95.81 | 95.64 | 167.31 | 514.37 | EUR/MWh |
| PICASSO.MarginalPricess.SEPS_NEG.Weighted | 15-min | 2391 | 83 | -457.38 | 80.95 | 73.29 | 129.3 | 294.18 | EUR/MWh |


## Load

| vector | cadence_label | n | completeness_pct | min | median | mean | p95 | max | unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sk.Final.Cons.SEPS | hourly | 720 | 100.1 | 1975 | 2883.5 | 2892.17 | 3576.3 | 3774 | MW |
| SK.F.Cons.M.1 | 15-min | 2881 | 100 | 1938.05 | 2876.92 | 2889.84 | 3559.79 | 3777.82 | MW |


## Gen

| vector | cadence_label | n | completeness_pct | min | median | mean | p95 | max | unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sk.A.Solar | hourly | 720 | 100.1 | 0 | 6.3 | 79.25 | 321.52 | 401.6 | MW |
| SK.A.Nuclear | hourly | 719 | 100.1 | 1779.5 | 1963.3 | 2111.55 | 2458.12 | 2464.4 | MW |
| Sk.A.HydroPump | hourly | 497 | 69.7 | 0 | 28.6 | 109.08 | 393.72 | 555.4 | MW |


## Weather

| vector | cadence_label | n | completeness_pct | min | median | mean | p95 | max | unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sk.T.Actual60 | hourly | 720 | 100.1 | -1.08 | 8.73 | 9.4 | 17.18 | 21.39 | C |


## Plots

Per-vector time-series plots are in `plots/`.


## Observations

- Hourly cadence observed for: Sk.Final.Cons.SEPS, Sk.A.Solar, SK.A.Nuclear, Sk.A.HydroPump, Sk.T.Actual60
- Sub-90% completeness: PICASSO.MarginalPricess.SEPS_POS.Weighted (88%), PICASSO.MarginalPricess.SEPS_NEG.Weighted (83%), Sk.A.HydroPump (70%)
- Irregular spacing (<95% same-gap): PICASSO.MarginalPricess.SEPS_POS.Weighted (94%), PICASSO.MarginalPricess.SEPS_NEG.Weighted (93%), Sk.A.HydroPump (93%)
- **Stale / sparse (n<10):** F.B.Odchylka (n=1, last 2026-02-07T11:06:15+00:00)