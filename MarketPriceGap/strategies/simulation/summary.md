# Portfolio Simulation & Backtest Engine

## Components
- `backtest_engine.py` - reusable backtest framework with degradation costs, transaction costs, and Monte Carlo simulation
- `portfolio_analysis.py` - combined BESS + speculative (DA-IDM) portfolio analysis

## Data Source
Reads from `da_indicator/data/` pipeline (DA-IDM speculative + BESS daily P&L).
Note: the da_indicator speculative strategy trades the DA-IDM spread (no imbalance price involved).

## Key Findings

| Strategy | Total P&L | Annual P&L | Sharpe | Max DD |
|----------|-----------|------------|--------|--------|
| BESS baseline (gross) | 101,516 | 95,073 | 28.58 | -57 |
| BESS baseline (net of degradation) | 16,397 | 15,357 | 4.62 | -5,219 |
| Speculative (gross) | 70,161 | 65,708 | 5.06 | -3,601 |
| Combined portfolio (gross) | 171,677 | 160,782 | 12.19 | -3,308 |
| Combined portfolio (net) | 78,327 | 73,356 | 5.57 | -3,866 |

- Degradation: mid-range scenario (250 EUR/kWh, 5000 cycles) costs 79.7k/year, payback 39.9 years
- Diversification benefit: 19.2% risk reduction from combining BESS + Speculative
- Monte Carlo: 100% probability of positive annual return for combined portfolio

## Results (in results/)
- `strategy_metrics.csv` - per-strategy performance summary
- `portfolio_daily.csv` - daily combined portfolio P&L
- `monte_carlo_results.csv` - Monte Carlo simulation outputs
- `degradation_scenarios.csv` - battery degradation cost scenarios
- `transaction_cost_scenarios.csv` - transaction cost sensitivity
- `quarterly_performance.csv` - quarterly breakdown
- `rolling_*.csv` - rolling Sharpe and P&L for each strategy

## Charts (in results/)
- `portfolio_cumulative_pnl.png` - combined portfolio equity curve
- `rolling_sharpe.png` - rolling Sharpe ratio over time
- `monte_carlo_distributions.png` - Monte Carlo outcome distributions
- `drawdown_analysis.png` - maximum drawdown analysis
- `pnl_correlation.png` - cross-strategy P&L correlation

## Scripts
- `backtest_engine.py` (library)
- `portfolio_analysis.py`
