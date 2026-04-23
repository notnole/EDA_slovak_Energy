# Vectord Vectors

Documented vectors available in the Vectord / Ipesoft EDA system.
All are time-series with `{time, value}` points. Resolutions are the underlying
signal cadence — vectord just returns whatever is stored.

**Notes**
- Names are **case-sensitive**. Watch for `Sk.*` vs `SK.*` inconsistency (both
  exist — ENTSO-E pipeline mostly uses `Sk.*`, EQ pipelines mix).
- `PICASSO.MarginalPricess` has the typo "Pricess" — that's the real name, not
  a bug to fix.
- Authoritative source: `infra/ansible/templates/lopatovac-config.yaml.j2`
  (pipeline definitions). If you need something not on this list, grep there.
- **Per-asset vectors (batteries, generation blocks) follow a different
  namespace** — see [`localities.md`](localities.md) for the
  `EMS#UNT..#<LOCALITY>#<ATTRIBUTE>` norm and `S.txt` for the attribute
  catalogue.

---

## Real-time balance / ACE (from SEPS DaE)

| Vector | What |
|--------|------|
| `EMS.DaE.PUB_3M.REAL_SYSTEM_LOAD` | System load, 3-min |
| `EMS.DaE.PUB_3M.REAL_SYSTEM_PRODUCTION` | System generation, 3-min |
| `EMS.DaE.PUB_3M.REAL_BALANCE` | ACE real balance, 3-min |
| `EMS.DaE.PUB_3M.ACKNOWLEDGED_REAL_BALANCE` | Acknowledged ACE, 3-min |
| `DaE.OH.RE_WITH_GCC` | ACE with GCC, hourly |
| `DaE.OH.RE_WITH_GCC_SEP_3M_GCC` | ACE with GCC separation, 3-min |
| `DaE.OH.RE_WITH_GCC_SEP_ODCH` | 15-min settlement deviation |

## Imbalance predictions (our models)

| Vector | What |
|--------|------|
| `F.B.Odchylka` | BEAM imbalance prediction (written by lopatovac from `predictions.xgb_normal`) |
| `Okte.Odchylka` | System imbalance from OKTE, 15-min — **initial** publication |
| `Okte.RiadneDenne.Odchylka` | OKTE regular daily revision, 15-min |
| `Okte.Dekadne.Odchylka` | OKTE 10-day revision, 15-min |
| `Okte.Mesacne.Odchylka` | OKTE monthly revision, 15-min |
| `Okte.Konecne.Odchylka` | OKTE final settlement, 15-min |
| `Okte.Combine.Odchylka` | Computed: most-settled available per timestamp (combines the five above in order initial → daily → 10-day → monthly → final). Use this for backtesting against settled truth. |

## Marginal prices — PICASSO (SEPS)

| Vector | What |
|--------|------|
| `PICASSO.MarginalPricess.SEPS_POS.Avg` | Positive marginal price, avg, 15-min |
| `PICASSO.MarginalPricess.SEPS_NEG.Avg` | Negative marginal price, avg, 15-min |
| `PICASSO.MarginalPricess.SEPS_POS.Weighted` | Positive, weighted, 15-min |
| `PICASSO.MarginalPricess.SEPS_NEG.Weighted` | Negative, weighted, 15-min |

## Day-ahead / spot prices

| Vector | What |
|--------|------|
| `Okte.MargCena` | OKTE marginal (clearing) price, hourly |
| `SK.F.M.1.Spot` | SK day-ahead spot forecast (model 1), 15-min |
| `SK.F.M.2.Spot` | SK day-ahead spot forecast (model 2), 15-min |
| `SK.F.M.Merged.Spot` | SK merged spot forecast, 15-min |
| `DE.F.M.1.Spot` | DE day-ahead spot forecast, 15-min |

## Load / consumption forecasts (EQ)

| Vector | What |
|--------|------|
| `SK.F.Cons.M.1` | SK consumption forecast (GFS), 15-min |
| `SK.F.Cons.M.Icon.1` | SK consumption (ICON model), 15-min |
| `SK.F.Cons.ECMF.1` | SK consumption (ECMWF), 15-min |
| `SK.F.Cons.SEPS.1` | SK load forecast from ENTSO-E, hourly |
| `Sk.Final.Cons.SEPS` | SK actual load from ENTSO-E, hourly |

## Generation forecasts

| Vector | What |
|--------|------|
| `Sk.F.Solar` | SK solar generation forecast, hourly (observed cadence) |

## Residual load

| Vector | What |
|--------|------|
| `SK.F.ResL.M.1` | Residual load (GFS), 15-min |
| `SK.F.ResL.M.ISR.1` | Residual load (ISR variant), 15-min |

## Actual generation by source (ENTSO-E)

15-min resolution.

| Vector | What |
|--------|------|
| `Sk.A.Solar` | Solar (**observed hourly** in vectord, despite the 15-min group label) |
| `SK.A.Nuclear` | Nuclear |
| `Sk.A.NatGas` | Natural gas |
| `Sk.A.HardCoal` | Hard coal |
| `Sk.A.Lignite` | Lignite / brown coal |
| `Sk.A.FosilOil` | Fossil oil |
| `Sk.A.Biomas` | Biomass |
| `Sk.A.Other` | Other |
| `Sk.A.Other.Renewable` | Other renewables |
| `Sk.A.HydroReservoir` | Hydro reservoir |
| `Sk.A.HydroRunRiver` | Hydro run-of-river |
| `Sk.A.HydroPump` | Hydro pumped storage (generation) |
| `Sk.A.HydroPumpConsumption` | Hydro pump consumption |
| `Sk.A.HydroPump.CVTG2` … `Sk.A.HydroPump.CVTG6` | Per-unit Čierny Váh pump plants |

## Weather forecast — Bardejov (YR.no)

3-hourly.

| Vector | What |
|--------|------|
| `SK.F.TEHO.Temp` | Air temperature |
| `SK.F.TEHO.Cloud` | Cloud cover fraction |
| `SK.F.TEHO.Humidity` | Relative humidity |
| `SK.F.TEHO.WindSpeed` | Wind speed |
| `SK.F.TEHO.WindDir` | Wind direction |

## Temperature (EQ, national)

| Vector | What |
|--------|------|
| `Sk.T.Actual60` | Actual temperature, hourly |
| `SK.T.GFS.1` | Forecast (GFS), 15-min |
| `SK.T.ECM.1` | Forecast (ECMWF), 15-min |
| `SK.T.Icon.1` | Forecast (ICON), 15-min |

## Cloud cover (EQ)

| Vector | What |
|--------|------|
| `SK.Cloud.EC.Merged` | ECMWF merged cloud forecast, 15-min |
| `SK.Cloud.Icon.Merged` | ICON merged cloud forecast, 15-min |
| `SK.A.Cloud` | Actual cloud cover, 15-min |
