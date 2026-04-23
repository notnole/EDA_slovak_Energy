# BL5 EDA (GBAT_BAT_5)

- Lookback: 30 days
- Generated: 2026-04-23T10:45:40.506415+00:00

## Vector coverage

| attr | unit | n | first | last | min | p05 | median | mean | p95 | max | std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pdod | MW | 2880 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:30:00+00:00 | -1.983 | -1.498 | -0.023 | -0.040 | 1.802 | 2.767 | 0.825 |
| SoC.15M | % | 2881 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:45:00+00:00 | 7.400 | 20.800 | 39.500 | 44.024 | 76.800 | 98.400 | 19.505 |
| PP_Pdb | MW | 2881 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:45:00+00:00 | -1.800 | -1.500 | 0.000 | -0.027 | 1.800 | 2.700 | 0.793 |
| RBO_RE | MWh | 2881 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:45:00+00:00 | -0.195 | -0.028 | -0.002 | -0.003 | 0.006 | 0.241 | 0.032 |
| Adelta | MW | 2880 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:30:00+00:00 | -0.228 | -0.086 | -0.000 | -0.003 | 0.065 | 0.199 | 0.046 |

## Plots

- `plots/01_timeseries_overlay.png` - Pdod vs PP_Pdb, Adelta vs RBO_RE, SoC
- `plots/02_plan_adherence.png` - scatter PP_Pdb vs Pdod (identity = perfect)
- `plots/03_soc_behaviour.png` - SoC histogram + daily trajectories
- `plots/04_adelta_split.png` - Adelta distribution and share explained by RBO_RE
- `plots/05_energy_check.png` - cumulative integral(Pdod) vs SoC change

## Next

Swap `LOCALITY = "GBAT_BAT_5"` for any other locality (e.g. `GBAT_BAT_4`) to reuse this EDA. See `vectord/localities.md` for the naming norm.