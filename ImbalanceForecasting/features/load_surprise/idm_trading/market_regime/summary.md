# Market Regime Change

Analysis of how the IDM-Imbalance arbitrage closed in late December 2025.

## Key Finding

The systematic arbitrage disappeared around Jan 1, 2026. Markets converged.

| Period | Avg Spread | Win Rate | Correlation |
|--------|-----------|----------|-------------|
| Sep-Dec 2025 | +20.0 EUR/MWh | 67% | 0.33 |
| Jan 2026 | +0.4 EUR/MWh | 45% | 0.68 |

## Not Seasonal

Compared Jan 2025 vs Jan 2026:
- **Jan 2025**: +35.7 EUR/MWh spread, 70% win rate
- **Jan 2026**: +0.4 EUR/MWh spread, 45% win rate

This is NOT a seasonal pattern - the market genuinely corrected.

## Files

| File | Description |
|------|-------------|
| `check_inefficiency.py` | Checks if arbitrage closed after Dec 15 |
| `check_seasonality.py` | Compares Jan 2025 vs Jan 2026 |
| `11_inefficiency_check.png` | Spread analysis by period |
| `12_seasonality_check.png` | Year-over-year comparison |

## Conclusion

The IDM and Imbalance markets became more tightly coupled (correlation doubled), eliminating the pure arbitrage opportunity. Signal-based trading became essential.
