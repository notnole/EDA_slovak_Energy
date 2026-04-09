# Live Model vs Ipesoft Comparison - March 2026

Evaluation of the deployed LightGBM nowcasting model against Ipesoft EDA's native
imbalance prediction signal (`P.DaE.Integrovana_Odchylka_Final_15min_Zn`).

## Key Finding

The LightGBM model outperforms Ipesoft at all lead times except Lead 0, with the
advantage growing at longer horizons. At Lead 12 (hardest, first observation only),
the model achieves **4.70 MWh MAE vs 6.44 MWh** for Ipesoft — a 27% improvement.

Ipesoft's predictions are effectively the same regulation-based baseline formula
(correlation 0.989, confirmed by MW/4 scaling factor), just served from the SCADA system.

## MAE by Lead Time (MWh)

| Lead | Ipesoft | Baseline | LightGBM | Model vs Ipesoft |
|------|---------|----------|----------|------------------|
| 0 min  | 1.36 | 1.99 | 1.48 | +9% worse |
| 3 min  | 2.34 | 2.35 | 2.16 | -8% better |
| 6 min  | 3.96 | 3.30 | 2.86 | -28% better |
| 9 min  | 4.80 | 4.55 | 3.75 | -22% better |
| 12 min | 6.44 | 6.43 | 4.70 | -27% better |

## QH Directional Accuracy

A QH is marked "wrong" if 3+ out of 5 lead-time predictions have the wrong sign.

- **LightGBM: 91.2%** (224 wrong QHs out of 2,534)
- **Ipesoft: 91.5%** (223 wrong QHs out of 2,618)

The model gets 71.3% of QHs perfectly right (0 mistakes) vs 60.6% for Ipesoft.

## Plots

1. `01_mae_by_lead_time.png` — Grouped bar chart of MAE across all leads
2. `02_error_distribution_lead12.png` — Error density at Lead 12, model is tighter and less biased
3. `03_time_series_lead12.png` — 2-day excerpt (March 10-12) showing actual vs both predictions
4. `04_qh_mistake_distribution.png` — Per-QH directional mistake counts (0-5)
5. `05_scatter_lead12.png` — Predicted vs actual scatter at Lead 12 (r=0.818 vs 0.724)
6. `06_rolling_daily_mae_lead12.png` — Daily MAE stability over the month

## Notes

- Ipesoft signal is in MW (instantaneous power), divided by 4 to convert to MWh for comparison.
- Ipesoft is essentially identical to the regulation baseline (Ipesoft MAE = 6.44 vs Baseline MAE = 6.43 at Lead 12).
- The model's Lead 0 disadvantage vs Ipesoft (1.48 vs 1.36) is small and may reflect
  slight timestamp alignment differences rather than a real accuracy gap.
- Coverage: ~89% of March settlement periods matched (system uptime gaps account for the rest).
- Actuals source: OKTE `SystemImbalance_2026-03-01_2026-03-30.csv`.

## Data

- `data/comparison_stats.csv` — Full per-lead statistics
- Source script: `scripts/plot_comparison.py`
- Cleaned Ipesoft data: `data/ipesoft_predictions/ipesoft_imbalance_predictions_202603.csv`
