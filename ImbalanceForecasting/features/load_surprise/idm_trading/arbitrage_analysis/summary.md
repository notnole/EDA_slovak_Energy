# Arbitrage Analysis

Analysis of the IDM-Imbalance price arbitrage opportunity.

## Key Finding

In Sep-Dec 2025, there was a systematic arbitrage: IDM prices were consistently higher than Imbalance Settlement Prices, making "always sell on IDM" profitable.

- **Average spread**: +20 EUR/MWh (Sep-Dec 2025)
- **Win rate**: 67% (selling on IDM beats imbalance settlement)
- **Strategy**: Sell on IDM, settle short position at imbalance price

## Files

| File | Description |
|------|-------------|
| `analyze_idm_strategy.py` | Initial strategy analysis |
| `analyze_idm_arbitrage.py` | Deep dive into arbitrage mechanics |
| `visualize_idm_strategy.py` | Visualization of strategy performance |
| `optimize_threshold.py` | Threshold optimization for signal |
| `09_idm_strategy_sep_dec.png` | Strategy performance Sep-Dec 2025 |
| `10_idm_arbitrage_analysis.png` | Arbitrage analysis visualization |

## Results

The arbitrage existed because IDM and Imbalance prices were loosely coupled (correlation ~0.33). Market participants could profit by systematically selling on IDM.
