# DA-IDM Filtered Trading Strategy

## Key Discovery

**The predictability of DA-IDM spread direction depends heavily on yesterday's spread SIZE.**

| Yesterday's |Spread| | Accuracy |
|----------------------|----------|
| < 5 EUR (tiny) | ~50% (random) |
| 5-15 EUR (small) | ~54% |
| 15-20 EUR (medium) | ~60% |
| **20-50 EUR (large)** | **~69%** |
| > 50 EUR (huge) | ~76% |

**Insight**: Small spreads are noise. Large spreads indicate momentum that persists.

---

## Optimal Strategy

### The Rule

```
At DA auction (D-1), for each delivery hour H:

1. Check yesterday's spread for hour H: spread_yesterday = DA_price - IDM_price

2. IF |spread_yesterday| < 20 EUR:
   → NO TRADE (signal too weak, ~50% accuracy)

3. IF |spread_yesterday| >= 20 EUR:
   → IF spread_yesterday > 0: SELL on DA, buy back on IDM
   → IF spread_yesterday < 0: BUY on DA, sell on IDM

4. Close position on IDM during delivery day
```

### Performance (2025 Backtest)

| Metric | Value |
|--------|-------|
| **Net Profit** | **57,615 EUR** |
| Total Trades | 2,823 |
| Accuracy | 69.0% |
| Win Rate | 69.0% |
| Avg P&L per Trade | 20.41 EUR |
| Profit Factor | 3.65 |
| Sharpe Ratio | 21.13 |

### Costs Included
- Transaction cost: 0.09 EUR/MWh
- Bid-ask slippage: Random +/- 5 EUR

---

## Comparison: Original vs Filtered Strategy

| Metric | All Trades | Filtered (>=20 EUR) | Improvement |
|--------|------------|---------------------|-------------|
| Trades | 8,722 | 2,823 | -68% |
| Accuracy | 58% | 69% | +11% |
| Net P&L | ~64,000 EUR | 57,615 EUR | -10% |
| Avg/Trade | 7.31 EUR | 20.41 EUR | +179% |

**Trade-off**: 68% fewer trades, but 11% higher accuracy and similar total profit.

---

## Monthly Performance (2025)

| Month | Net P&L | Trades | Accuracy |
|-------|---------|--------|----------|
| 2025-01 | 1,822 EUR | 270 | 63% |
| 2025-02 | 1,601 EUR | 188 | 62% |
| 2025-03 | 802 EUR | 133 | 58% |
| 2025-04 | 808 EUR | 133 | 69% |
| 2025-05 | 151 EUR | 139 | 52% |
| 2025-06 | 751 EUR | 178 | 60% |
| 2025-07 | 85 EUR | 108 | 50% |
| 2025-08 | -83 EUR | 63 | 51% |
| 2025-09 | 1,350 EUR | 161 | 57% |
| 2025-10 | 17,914 EUR | 550 | 68% |
| 2025-11 | 18,948 EUR | 480 | 83% |
| 2025-12 | 13,466 EUR | 420 | 86% |

---

## When the Strategy Works Best

1. **Q4 (Oct-Dec)**: Best performance, 70-86% accuracy
2. **Large spread days**: When yesterday had |spread| > 30 EUR, accuracy reaches 72%+
3. **Night/evening hours**: Hours 18-23 tend to be most predictable

## When to Be Cautious

1. **May-August**: Lower accuracy (50-58%), consider reducing position size
2. **Midday hours (9-11)**: Lower predictability
3. **Very small spreads**: If yesterday's |spread| < 10 EUR, don't trade

---

## Files Generated

- `01_accuracy_by_spread_size.png` - Key insight visualization
- `02_strategy_comparison.png` - Original vs filtered strategy
- `03_optimal_strategy_details.png` - Detailed performance analysis
- `summary.md` - This summary

---

## Implementation Notes

1. **Data needed**: Previous day's DA and IDM prices for each hour
2. **Decision time**: Before DA auction closes (typically ~noon D-1)
3. **Execution**: Place DA order, then close on IDM during delivery day
4. **Position sizing**: Consider scaling with yesterday's spread size (larger spread = higher confidence)
