# Temperature-Filtered Speculative Strategy

## Strategy Variants

- Base strategy ("sell DA, buy imbalance"): 24,238 EUR, Sharpe 0.56 (Feb 2024 - Jan 2026)
- Skip filter (temp <0C or >25C): 48,847 EUR, Sharpe 1.28 -- skips 91 days averaging -243 EUR/day
- REVERSE filter (trade opposite on extreme temp days): 73,455 EUR, Sharpe 1.69, 3x baseline

## In-Sample / Out-of-Sample Validation

- IS-optimal thresholds (2024): cold <2C, hot >26C
- OOS (2025+): Sharpe 1.76 vs 1.08 baseline -- filter generalizes well

## GFS Forecast Feasibility

- GFS temperature forecast quality: r=0.991, MAE=0.83C
- Filter agreement between GFS and actual: 92.3% -- implementable with D-1 forecast

## Key Results

- Jan 2026: baseline -2,490 EUR -> reverse +2,278 EUR (+4,768 swing)
- Cold reversal robust across thresholds from -3C to +3C; breaks above +3.5C
- URSO reform (Sep 2025): settlement multiplier 1.5->1.1, losses halved (-61->-32 EUR), gains down 27%

## Scripts

- `scripts/backtest_temp_filter.py` - skip-filter backtest
- `scripts/temp_reverse_backtest.py` - reverse-on-extreme-temp backtest

## Charts

- `temp_filter_backtest.png` - cumulative P&L with and without skip filter
- `temp_reverse_strategy.png` - reverse strategy performance and threshold sensitivity
