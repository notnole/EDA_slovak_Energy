# Sell IDM / Buy Imbalance Strategy - Hourly Analysis (CORRECTED)

## Methodology

**This analysis uses the CORRECT methodology:**
- 15-minute resolution data (QH 1 & 2 only)
- Hours 5-21 (matching original analysis)
- Imbalance Settlement Price from master data
- Outlier filter: |imbalance| <= 300 MWh

---

## Executive Summary

| Metric | 2025 | Jan 2026 | Change |
|--------|------|----------|--------|
| Overall Win Rate | 65% | **45%** | **-20 pp** |
| Avg Spread | +19.3 EUR/MWh | **+0.4 EUR/MWh** | **-97%** |
| Profitable Hours | 17/17 | **6/17** | **-11** |

**Key Finding**: The strategy **STOPPED WORKING** in Jan 2026.
- Win rate dropped from 65% to **45%** (below break-even!)
- Average spread collapsed from +19.3 to **+0.4 EUR/MWh**
- **11 out of 17 hours** are now unprofitable

---

## Hour-by-Hour Analysis

### Hours Still Profitable in Jan 2026 (Win Rate >= 50%)

| Hour | 2025 Win Rate | Jan 2026 Win Rate | Change | Jan 2026 Avg Spread |
|------|---------------|-------------------|--------|---------------------|
| 06:00 | 66% | 50% | -16 pp | -8.3 EUR |
| 10:00 | 62% | 52% | -10 pp | 13.7 EUR |
| 12:00 | 52% | 50% | -2 pp | 9.6 EUR |
| 13:00 | 43% | 58% | +16 pp | 9.9 EUR |
| 14:00 | 43% | 52% | +9 pp | -2.2 EUR |
| 20:00 | 85% | 62% | -23 pp | 19.9 EUR |

### Hours NO LONGER Profitable in Jan 2026 (Win Rate < 50%)

| Hour | 2025 Win Rate | Jan 2026 Win Rate | Change | Jan 2026 Avg Spread |
|------|---------------|-------------------|--------|---------------------|
| 05:00 | 65% | 42% | -23 pp | -5.9 EUR |
| 07:00 | 75% | 35% | -40 pp | -30.7 EUR |
| 08:00 | 76% | 42% | -34 pp | -8.4 EUR |
| 09:00 | 69% | 23% | -46 pp | -6.3 EUR |
| 11:00 | 59% | 42% | -17 pp | 4.0 EUR |
| 15:00 | 45% | 35% | -10 pp | -12.1 EUR |
| 16:00 | 53% | 46% | -7 pp | 4.7 EUR |
| 17:00 | 67% | 40% | -28 pp | -13.9 EUR |
| 18:00 | 78% | 46% | -32 pp | 26.7 EUR |
| 19:00 | 80% | 44% | -36 pp | 8.3 EUR |
| 21:00 | 78% | 42% | -36 pp | -2.7 EUR |

---

## Trading Recommendations

### IF Hourly Selection is Possible

1. **ONLY TRADE** during hours: **[6, 10, 12, 13, 14, 20]**
   - These 6 hours still have >= 50% win rate
   - But edge is much weaker than 2025

2. **AVOID** hours: **[5, 7, 8, 9, 11, 15, 16, 17, 18, 19, 21]**
   - These 11 hours are now below break-even

### IF Must Trade All Hours

**RECOMMENDATION: STOP TRADING this strategy.**
- Overall win rate is 45% (below break-even)
- Average spread collapsed to near zero (+0.4 EUR/MWh)
- The market arbitrage has closed

---

## Why Did This Happen?

The original analysis (`ImbalanceForecasting/features/load_surprise/idm_trading/`) found:

1. **Market Convergence**: IDM and Imbalance prices became more correlated
   - Correlation: 0.33 (Sep-Dec 2025) -> 0.68 (Jan 2026)

2. **NOT Seasonal**: Jan 2025 had +35.7 EUR/MWh spread vs Jan 2026 +0.4 EUR/MWh

3. **Signal-Based Trading Needed**: The load surprise prediction signal became essential
   - Always Sell: +812 EUR in Jan 2026
   - Signal-Based: +3,102 EUR in Jan 2026

---

## Visualizations

1. **04_correct_hourly_analysis.png** - Win rate by hour using correct 15-min methodology

---

## Data Notes

- **2025**: Full year, 12,328 observations (hours 5-21)
- **Jan 2026**: 816 observations (24 days, hours 5-21)
- Resolution: 15-minute (QH 1 & 2 only)
