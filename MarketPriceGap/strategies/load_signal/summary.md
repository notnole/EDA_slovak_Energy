# Load Prediction Signal Analysis

## Key Findings

- Q1: Load forecast error strongly correlated with spread: r=-0.386, p<10^-28, stable OOS (r=-0.397)
- Quintile spread: Q1 (overforecast -85MW) = +23.7 EUR/MWh, Q5 (underforecast +81MW) = -20.4 EUR/MWh
- Q2: Similar-day deviation (same DOW + season) is noise: r=-0.035 (p=0.34), not significant
- Q3: Adding load signal to temperature filter HURTS performance (Sharpe 1.47 vs 1.67 temp-only)
- Q4: Permutation test (10,000 iterations) confirms load error correlation is real (p<0.0001)

## Statistical Validation

- Bootstrap 95% CI: [-0.452, -0.318]
- Permutation p-value: <0.0001

## Practical Limitation

- Load error is ex-post only (known after delivery), cannot be used as D-1 signal
- Temperature captures the same mechanism and IS available D-1

## Conclusion

Load forecast error explains the DA-imbalance spread but does not improve trading because
it is not observable before delivery. Temperature filtering is the practical substitute.

## Scripts

- `scripts/load_signal_analysis.py` - full 4-question analysis with permutation tests

## Charts

- `load_signal_analysis.png` - correlation, quintile analysis, and combined strategy comparison
