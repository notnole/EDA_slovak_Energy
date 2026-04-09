# Speculative Spread Strategies

## Purpose
Walk-forward ML strategies for 3 electricity spread pairs in the Slovak market.
Uses LightGBM with Half-Kelly position sizing and hourly stop-loss.

## Data Source
- `hourly_market_prices.csv` (corrected Feb 2026: uses `imb_settlement_price` for DA-Imb spreads, QH-to-H aggregation for 2026 DA data)

## Strategies

| Strategy | Spread | P&L (EUR) | Sharpe | Accuracy | Max DD | PF | Months% |
|----------|--------|-----------|--------|----------|--------|-----|---------|
| Baseline (Rules) | DA-IDM | 15,194 | 6.01 | 51.8% | -2,418 | 1.31 | 85% |
| A: DA-IDM (ML) | DA-IDM | -472 | -2.78 | 52.6% | -906 | 0.78 | 50% |
| B: DA-Imb (ML+Risk) | DA-Imb | 10,105 | 13.78 | 58.3% | -803 | 1.83 | 91% |
| B0: Always DA>IMB | DA-Imb | 34,889 | 1.89 | 53.1% | -31,617 | 1.08 | 50% |
| C: IDM-Imb (Morning+ML) | IDM-Imb | 3,728 | 3.66 | 56.4% | -1,738 | 1.37 | 100% |
| C0: Always IDM>IMB | IDM-Imb | 14,624 | 2.14 | 52.9% | -13,401 | 1.12 | 50% |
| Portfolio (A+B+C) | Combined | 4,454 | 7.88 | 56.9% | -564 | 1.61 | 91% |

## Key Findings
- Strategy B (DA-Imbalance) is the best ML strategy: 10,105 EUR, Sharpe 13.78, 91% profitable months
- Always-long DA>IMB benchmark earns more (34,889 EUR) but with 40x larger drawdown (-31,617 vs -803)
- ML adds value by dramatically reducing drawdown, not by increasing total P&L
- Strategy A (DA-IDM ML) fails to beat baseline - DA-IDM spread too competitive for ML
- Strategy C (IDM-Imb) moderate but 100% profitable months

## Year-over-Year (Strategy B)
- 2024: 5,748 EUR (56.3% acc)
- 2025: 3,306 EUR (60.6% acc)
- 2026 (Jan only): 1,052 EUR (59.6% acc)

## Scripts
- `spread_strategies.py` - full ML pipeline with walk-forward validation

## Charts
- `01_cumulative_pnl_comparison.png`
- `02_monthly_pnl_by_strategy.png`
- `03_ml_vs_naive_benchmarks.png`, `03_strategy_comparison_bars.png`
- `04_rolling_accuracy.png`, `04_strategy_comparison_bars.png`
- `05_rolling_accuracy.png`
