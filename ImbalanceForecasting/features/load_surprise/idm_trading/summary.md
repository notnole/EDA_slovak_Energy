# IDM Trading Strategy Analysis

Analysis of using load surprise predictions for IDM (Intraday Market) trading.

## Executive Summary

**Finding**: A structural arbitrage between IDM and Imbalance prices existed in 2025 but closed in January 2026. The load surprise prediction signal became essential for profitability after the arbitrage closed.

| Metric | Sep-Dec 2025 | Jan 2026 |
|--------|--------------|----------|
| Avg Spread (IDM - Imb) | +20.0 EUR/MWh | +0.4 EUR/MWh |
| Always Sell Win Rate | 67% | 45% |
| Signal Win Rate | 71% | 73% |
| Best Strategy | Always Sell | **Signal-Based** |

## Trading Strategy

1. **SELL signal** (pred < -100 MW): Expecting load surplus → low imbalance price → sell on IDM
2. **BUY signal** (pred > +100 MW): Expecting load deficit → high imbalance price → buy on IDM

## Folder Structure

```
idm_trading/
├── arbitrage_analysis/    # Basic arbitrage discovery
├── market_regime/         # Market convergence analysis
├── signal_validation/     # Signal performance validation
├── data_validation/       # Data correctness checks
├── strategy_comparison/   # Final comparison visualizations
└── data/                  # Intermediate data files
```

## Key Results

### 1. Arbitrage Discovery (arbitrage_analysis/)
- Sep-Dec 2025: IDM consistently higher than Imbalance prices
- "Always sell" strategy was profitable with 67% win rate

### 2. Market Regime Change (market_regime/)
- Arbitrage closed around Jan 1, 2026
- NOT seasonal - compared Jan 2025 (+35.7) vs Jan 2026 (+0.4)
- Price correlation increased from 0.33 to 0.68

### 3. Signal Validation (signal_validation/)
- Signal predicts direction with 80% accuracy (SELL) and 65% accuracy (BUY)
- In Jan 2026: Signal made +3,102 while Always Sell made only +812
- Signal is essential when arbitrage doesn't exist

### 4. Data Validation (data_validation/)
- All calculations verified correct
- Extreme prices are real market events
- Market convergence is genuine, not a data artifact

## Conclusion

The load surprise prediction provides valuable trading signals for IDM markets. While a pure arbitrage existed in 2025, the signal's value became clear after markets converged in 2026 - it identifies profitable trading opportunities that the simple "always sell" strategy misses.
