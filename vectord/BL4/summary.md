# BL4 EDA (GBAT_BAT_4)

- Lookback: 30 days
- Generated: 2026-04-23T10:49:31.326997+00:00

## Vector coverage

| attr | unit | n | first | last | min | p05 | median | mean | p95 | max | std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pdod | MW | 2880 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:30:00+00:00 | -1.717 | -1.021 | -0.010 | -0.024 | 0.977 | 1.704 | 0.554 |
| SoC.15M | % | 2881 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:45:00+00:00 | 9.800 | 20.700 | 38.800 | 44.256 | 78.100 | 97.500 | 19.744 |
| PP_Pdb | MW | 2881 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:45:00+00:00 | -1.500 | -1.000 | 0.000 | -0.014 | 1.000 | 1.500 | 0.531 |
| RBO_RE | MWh | 2881 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:45:00+00:00 | -0.151 | -0.063 | -0.000 | -0.002 | 0.051 | 0.143 | 0.035 |
| Adelta | MW | 2880 | 2026-03-24T10:45:00+00:00 | 2026-04-23T10:30:00+00:00 | -0.151 | -0.063 | -0.000 | -0.002 | 0.051 | 0.143 | 0.035 |

## Plots

- `plots/01_timeseries_overlay.png` - Pdod vs PP_Pdb, Adelta vs RBO_RE, SoC
- `plots/02_plan_adherence.png` - scatter PP_Pdb vs Pdod (identity = perfect)
- `plots/03_soc_behaviour.png` - SoC histogram + daily trajectories
- `plots/04_adelta_split.png` - Adelta distribution and share explained by RBO_RE
- `plots/05_energy_check.png` - cumulative integral(Pdod) vs SoC change

## Next

Re-run for another block with `python run_eda.py --locality GBAT_BAT_4`. See `vectord/localities.md` for the naming norm.