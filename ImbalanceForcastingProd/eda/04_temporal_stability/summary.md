# 04 Temporal Stability

## Feature Importance Drift

### Features Gaining Importance (2026 vs 2024)
- **proxy_yesterday**: +0.0132

### Features Losing Importance
- **load_rmean16**: -0.0123
- **temp_bratislava**: -0.0217
- **reg_rmean8**: -0.0236
- **da_demand**: -0.0269
- **temp_rmean24h**: -0.0311

## Model Signal Shift: 2025 vs 2026

| Feature | 2025 Importance | 2026 Importance | Change |
|---------|----------------|----------------|--------|
| prod_momentum | 0.0000 | 0.0607 | +0.0607 |
| da_price | 0.0982 | 0.0642 | -0.0340 |
| hour_cos | 0.0276 | 0.0594 | +0.0318 |
| prod_rmean8 | 0.0000 | 0.0296 | +0.0296 |
| xborder_momentum | 0.0000 | 0.0259 | +0.0259 |
| da_demand | 0.0492 | 0.0238 | -0.0255 |
| reg_rmean8 | 0.0397 | 0.0220 | -0.0177 |
| proxy_rmin4 | 0.0452 | 0.0283 | -0.0169 |
| da_net_import | 0.0379 | 0.0213 | -0.0165 |
| nowcast_convergence | 0.0023 | 0.0177 | +0.0154 |
| da_price_change24h | 0.0483 | 0.0330 | -0.0153 |
| temp_rmean24h | 0.0442 | 0.0315 | -0.0127 |
| load_rmean16 | 0.0383 | 0.0271 | -0.0113 |
| proxy_dev_from_hour | 0.0252 | 0.0146 | -0.0106 |
| proxy_rmax4 | 0.0213 | 0.0107 | -0.0106 |

## Direction Accuracy vs Spread Magnitude

- Correlation between |spread| and accuracy: r = 0.10 (p = 0.681)
- **Weak relationship**: accuracy does not strongly depend on spread magnitude

## Monthly Performance Summary

| Month | Dir Accuracy | Mean |Spread| | P&L/Day |
|-------|-------------|----------------|---------|
| 2024-07 | 56.0% | 120.6 | 111.1 |
| 2024-08 | 52.9% | 116.6 | 670.5 |
| 2024-09 | 45.6% | 82.9 | 1269.1 |
| 2024-10 | 52.3% | 103.8 | 4008.9 |
| 2024-11 | 74.0% | 128.9 | 4469.4 |
| 2024-12 | 85.1% | 153.3 | 3109.6 |
| 2025-01 | 80.7% | 121.0 | 2018.2 |
| 2025-02 | 85.1% | 125.7 | 3599.5 |
| 2025-03 | 86.4% | 89.1 | 3424.6 |
| 2025-04 | 84.4% | 83.7 | 3251.5 |
| 2025-05 | 83.2% | 68.1 | 3219.4 |
| 2025-06 | 69.3% | 96.6 | 1862.2 |
| 2025-07 | 79.3% | 71.7 | 2670.8 |
| 2025-08 | 85.6% | 72.0 | 3650.5 |
| 2025-09 | 89.7% | 88.1 | 1986.3 |
| 2025-11 | 92.4% | 111.8 | 495.9 |
| 2025-12 | 94.6% | 155.2 | 725.7 |
| 2026-01 | 74.9% | 134.4 | 4651.7 |
| 2026-02 | 78.2% | 93.3 | 3648.0 |
| 2026-03 | 78.0% | 123.0 | 4555.6 |
| 2026-04 | 79.9% | 147.0 | 2291.4 |

**Average direction accuracy**: 76.5%
**Average P&L/day**: 2651.9 EUR/MWh