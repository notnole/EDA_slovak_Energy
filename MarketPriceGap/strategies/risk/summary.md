# Risk Analysis

## Data Source
Reads from `da_indicator/data/` pipeline (DA-IDM speculative + BESS daily P&L).
Note: the da_indicator speculative strategy trades the DA-IDM spread (no imbalance price involved).

## Key Metrics

| Metric | BESS Indicator | Speculative (DA-IDM) |
|--------|---------------|---------------------|
| Total P&L (EUR) | 192,322 | 70,161 |
| Sharpe (ann.) | 29.30 | 6.10 |
| Sortino (ann.) | 170.85 | 11.63 |
| Win Rate | 99.6% | 61.5% |
| Profit Factor | 2,578 | 3.26 |
| Max Drawdown | -57 | -3,601 |
| VaR 95% | 69 | -432 |
| CVaR 95% | 44 | -774 |

## Portfolio
- Daily P&L correlation: -0.068 (good diversification)
- Optimal allocation: BESS 95% / Speculative 5% -> Sharpe 35.40
- Combined capital requirement: 10k EUR (3x max DD)

## Regime Analysis
- BESS performs better in high-vol and extreme-price regimes
- Speculative performs better on weekends (Sharpe 8.66 vs 5.02 weekday) and winter (8.72 vs 1.53 summer)
- Results in `regime_analysis.csv`

## Data
- `risk_metrics.csv` - per-strategy risk metrics
- `allocation_frontier.csv` - efficient frontier allocations
- `regime_analysis.csv` - regime-conditional performance

## Scripts
- `risk_analysis.py`
