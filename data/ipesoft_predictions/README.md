# Ipesoft Imbalance Predictions

Ipesoft EDA system's own imbalance predictions, exported from signal
`P.DaE.Integrovana_Odchylka_Final_15min_Zn`.

## Source

- Raw file: `IpesoftPredicions.csv` (root of repo)
- Exported from Ipesoft EDA SCADA system
- Signal represents integrated (cumulative) imbalance prediction per 15-min settlement period

## Cleaned file

`ipesoft_imbalance_predictions_202603.csv`

| Column | Description |
|--------|-------------|
| `timestamp` | Observation time (local, Europe/Bratislava, naive) |
| `value_mw` | Original value in MW (instantaneous power) |
| `value_mwh` | Converted to MWh (value_mw / 4) for comparison with OKTE settlement |
| `qh_start` | Start of the 15-min settlement period this observation belongs to |
| `lead_time_min` | Minutes until end of settlement period (12, 9, 6, 3, 0) |

## Notes

- Raw values are in MW. Divide by 4 to get MWh (quarter-hour energy).
- 3-minute observation interval, 5 observations per settlement period.
- Lead time mapping: offset :00 = lead 12, :03 = lead 9, :06 = lead 6, :09 = lead 3, :12 = lead 0.
- This is essentially the same regulation-based baseline formula used in the DB pipeline,
  confirmed by correlation of 0.989 and median ratio of 4.0x (the MW vs MWh factor).
- Coverage: March 2026 (2026-03-01 to 2026-03-31).
