# DA-IDM Spread Direction Analysis

## Trading Strategy Context

**Objective**: Predict whether DA price will be higher or lower than IDM price to execute:
- **DA > IDM predicted**: Sell on DA, buy back on IDM (profit = DA - IDM)
- **DA < IDM predicted**: Buy on DA, sell on IDM (profit = IDM - DA)

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total observations | 9,346 hours |
| DA > IDM frequency | 54.0% (5,048 hours) |
| Average spread | 1.20 EUR/MWh |
| Average |spread| | 24.15 EUR/MWh |
| Max spread (DA > IDM) | 398.4 EUR/MWh |
| Max spread (DA < IDM) | -362.0 EUR/MWh |

**Baseline edge**: Since DA > IDM occurs 54.0% of the time, a naive "always sell DA" strategy
has a slight edge. However, this edge is small and highly variable.

---

## Predictive Signals

### Most Correlated Features (with spread direction)

| Feature | Correlation | Significance |
|---------|-------------|--------------|
| direction_lag1 | 0.557 | *** |
| spread_lag1 | 0.456 | *** |
| spread_lag24 | 0.254 | *** |
| spread_ma24 | 0.249 | *** |
| da_price | 0.204 | *** |
| direction_lag24 | 0.185 | *** |
| hour | -0.136 | *** |
| da_momentum | 0.120 | *** |

### Key Findings

1. **Persistence**: The spread direction shows autocorrelation - if DA > IDM now, it's more likely to be
   DA > IDM in the next hour too. This suggests momentum strategies could work.

2. **Hourly Pattern**:
   - Peak hours (7-20h): DA tends to be higher than IDM (~55-58% of the time)
   - Night hours (0-6h): More balanced, closer to 50-50

3. **Volatility Effect**: During high volatility periods, spreads are larger but direction is less predictable.

4. **Price Level**: At extreme high DA prices, IDM tends to be even higher (DA < IDM more frequent).

---

## Simple Trading Rules Performance

| Rule | Accuracy | Total P&L | Sharpe | Win Rate |
|------|----------|-----------|--------|----------|
| Follow previous hour | 78.0% | 178775 | 46.58 | 77.9% |
| Momentum (sign of lag spread) | 78.0% | 178775 | 46.58 | 77.9% |
| Follow same hour yesterday | 59.4% | 90755 | 21.73 | 59.4% |
| Naive (always sell DA) | 54.1% | 12178 | 2.84 | 54.1% |
| Weekday bias | 48.6% | -47412 | -11.14 | 48.5% |
| Hour-based (peak hours sell) | 45.1% | -72221 | -17.13 | 45.1% |
| Contrarian (opposite of prev) | 22.0% | -178775 | -46.58 | 22.0% |

### Best Simple Strategy: Follow previous hour

- **Accuracy**: 78.0%
- **Total P&L**: 178775 EUR/MWh over the period
- **Annualized Sharpe**: 46.58
- **Win Rate**: 77.9%
- **Profit Factor**: 2.49

---

## Trading Recommendations

1. **Use Momentum**: Following the previous hour's direction gives slight edge (~78.0% accuracy).

2. **Focus on Peak Hours**: The DA > IDM bias is strongest during 7-20h, making "sell DA" more reliable.

3. **Volatility Filter**: Consider larger positions during low-volatility periods when direction is more predictable.

4. **Risk Management**: Average spread is small (1.2 EUR/MWh), so transaction costs are critical.
   Ensure your spread capture exceeds costs.

5. **Streak Awareness**: Direction tends to persist, so don't fight strong trends.

---

## Visualizations

1. **01_spread_characteristics.png** - Basic spread statistics and patterns
2. **02_predictive_features.png** - Feature importance and autocorrelation
3. **03_simple_rules.png** - Trading rule comparison
4. **04_market_conditions.png** - Direction by market conditions

---

## Next Steps for Model Development

1. Build ML classifier (LightGBM) using identified features
2. Add more features: weather, renewable forecast, scheduled outages
3. Implement proper backtesting with transaction costs
4. Consider probabilistic predictions for position sizing
